import os
import time
# import json
import uuid
import shutil
# from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from ultralytics import YOLO
import glob

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# Global model variables
model_detection = None
model_segmentation = None
model_pose = None

@app.on_event("startup")
def load_models():
    global model_detection, model_segmentation, model_pose
    print("Cargando modelos YOLO...")
    
    # Load detection model (look for best.onnx)
    best_weights = glob.glob('**/weights/best.onnx', recursive=True)
    det_path = best_weights[0] if best_weights else 'notebooks/yolo26n-obb.onnx'
    print(f"Usando modelo de detección: {det_path}")
    
    model_detection = YOLO(det_path, task='obb')
    model_segmentation = YOLO('notebooks/yolo26n-seg.onnx', task='segment')
    model_pose = YOLO('notebooks/yolo26n-pose.onnx', task='pose')
    print("Modelos cargados.")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Redes neuronales listas"}

@app.post("/api/upload_video")
def process_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un video")
        
    os.makedirs("data/videos/original", exist_ok=True)
    video_id = str(uuid.uuid4())[:8]
    video_key = f"upload_{video_id}"
    video_path = f"data/videos/original/{video_key}.mp4"
    
    # Save the uploaded file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"Video guardado en {video_path}, comenzando inferencia...")
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir el video")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Detección robusta de dimensiones reales (respetando rotación de metadata) ---
    # CAP_PROP_FRAME_WIDTH/HEIGHT devuelven las dimensiones del codec raw (sin rotar).
    # En OpenCV 4.x, CAP_PROP_ORIENTATION_AUTO=1 por defecto: cap.read() auto-rota
    # los frames, pero las propiedades WIDTH/HEIGHT siguen reportando las raw.
    # Para videos de celular con rotación, esto causa mismatch entre las coordenadas
    # de inferencia (calculadas sobre frames rotados) y las dimensiones del JSON.
    # Solución: leer un frame de prueba para obtener las dimensiones reales.
    codec_w, codec_h = width, height
    orient_meta = 0
    if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):
        orient_meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    
    ret_test, test_frame = cap.read()
    if ret_test:
        actual_h, actual_w = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rebobinar al inicio
        if actual_w != codec_w or actual_h != codec_h:
            print(f"Rotacion detectada: codec={codec_w}x{codec_h}, "
                  f"real={actual_w}x{actual_h}, orient_meta={orient_meta} grados")
            width, height = actual_w, actual_h
    else:
        print(f"Dimensiones: {width}x{height} (sin rotacion)")

    
    # Remove the 15-second limit now that frame skipping makes it fast enough
    limit_frames = total_frames
    
    json_data = {
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": limit_frames,
            "custom_classes_conf_avg": 0,
            "inference_time_ms_avg": 0
        },
        "frames": []
    }
    
    conf_sum = 0
    conf_count = 0
    inference_time_sum = 0
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < limit_frames:
        # Capture timestamp BEFORE read — POS_MSEC gives the position of the
        # frame about to be decoded, which is what we need for VFR sync.
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Frame Skipping: Procesar 1 de cada 3 frames (efectivamente ~10 FPS)
        if frame_idx % 3 != 0:
            frame_idx += 1
            continue
            
        t0 = time.time()
        frame_data = {"timestamp_ms": timestamp_ms, "detections": [], "poses": [], "segmentations": []}
        
        # Pose
        res_pose = model_pose.predict(frame, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
        if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
            xy = res_pose.keypoints.xy.cpu().numpy()
            conf = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else np.zeros_like(xy[:, :, 0])
            for i in range(len(xy)):
                keypoints_list = [{"x": float(kp[0]), "y": float(kp[1]), "confidence": float(c)} for kp, c in zip(xy[i], conf[i])]
                frame_data["poses"].append({"keypoints": keypoints_list})
                
        # Segmentation
        classes_seg = [i for i in range(80) if i != 0]
        res_seg = model_segmentation.predict(frame, conf=0.30, iou=0.45, imgsz=640, classes=classes_seg, verbose=False)[0]
        if res_seg.masks is not None and res_seg.boxes is not None:
            for i, mask in enumerate(res_seg.masks.xy):
                if len(mask) < 3: continue
                cls_id = int(res_seg.boxes.cls[i].item())
                confidence = float(res_seg.boxes.conf[i].item())
                frame_data["segmentations"].append({
                    "class_name": res_seg.names[cls_id],
                    "confidence": confidence,
                    "polygon": [[float(p[0]), float(p[1])] for p in mask]
                })
                
        # Detection
        res_det = model_detection.predict(frame, conf=0.19, iou=0.25, imgsz=640, verbose=False)[0]
        detections = res_det.obb if (hasattr(res_det, 'obb') and res_det.obb is not None and len(res_det.obb) > 0) else res_det.boxes
        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                cls_id = int(detections.cls[i].item())
                confidence = float(detections.conf[i].item())
                cls_name = res_det.names[cls_id].upper().strip()
                thresh = {'TT TABLE': 0.80, 'TT NET': 0.50, 'TT RACKET': 0.35}.get(cls_name, 0.25)
                
                if confidence >= thresh:
                    if hasattr(detections, 'xyxyxyxy') and detections.xyxyxyxy is not None:
                        pts = detections.xyxyxyxy[i].cpu().numpy()
                        x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
                        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
                    else:
                        x1, y1, x2, y2 = detections.xyxy[i].cpu().numpy()
                        
                    frame_data["detections"].append({
                        "class_name": res_det.names[cls_id],
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                    conf_sum += confidence
                    conf_count += 1
                    
        json_data["frames"].append(frame_data)
        
        t1 = time.time()
        inference_time_sum += (t1 - t0) * 1000
        frame_idx += 1
        
        # Log progress every 10 frames or on the first frame
        if frame_idx == 1 or frame_idx % 10 == 0 or frame_idx == limit_frames:
            print(f"Procesando frame {frame_idx}/{limit_frames}...")
            
    cap.release()
    
    json_data["video_info"]["total_frames"] = frame_idx
    if conf_count > 0:
        json_data["video_info"]["custom_classes_conf_avg"] = conf_sum / conf_count
    if frame_idx > 0:
        json_data["video_info"]["inference_time_ms_avg"] = inference_time_sum / frame_idx
        
    print("Inferencia completada.")
    
    # Eliminar el archivo temporal del video
    try:
        os.remove(video_path)
    except Exception as e:
        print(f"Error al eliminar video temporal: {e}")
    
    return {
        "status": "completed",
        "video_key": video_key,
        "data": json_data
    }

# Serve the 'videos' folder to be able to stream video and json
app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")

# Serve static root files (like index.html, index.css)
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8555, reload=False)