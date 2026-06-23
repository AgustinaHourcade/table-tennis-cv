import os
import time
import uuid
import shutil
import asyncio
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import cv2
from ultralytics import YOLO
import glob

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.config import CLASS_CONF_THRESHOLDS, CONF_SEG, CONF_POSE, MODEL_IMGSZ, MODEL_SEG_IMGSZ

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando aplicación. Cargando modelos...")
    load_models()
    yield
    print("Apagando aplicación. Liberando recursos...")
    # Limpieza de recursos si es necesario

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model_detection = None
model_segmentation = None
model_pose = None

def load_models():
    global model_detection, model_segmentation, model_pose
    print("Cargando modelos YOLO...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load detection model using absolute path
    best_weights = glob.glob(os.path.join(base_dir, 'models', 'weights', 'best.onnx'))
    if best_weights:
        det_path = best_weights[0]
    else:
        # Fallback explícito si glob falla
        det_path = os.path.join(base_dir, 'models', 'weights', 'yolo26m-obb.onnx')
    
    print(f"Usando modelo de detección: {det_path}")
    
    model_detection = YOLO(det_path, task='obb')
    model_segmentation = YOLO(os.path.join(base_dir, 'models', 'weights', 'yolo26n-seg.onnx'), task='segment')
    model_pose = YOLO(os.path.join(base_dir, 'models', 'weights', 'yolo26n-pose.onnx'), task='pose')
    print("Modelos cargados. Iniciando warm-up...")
    dummy_frame = np.zeros((MODEL_IMGSZ, MODEL_IMGSZ, 3), dtype=np.uint8)
    model_detection.predict(dummy_frame, imgsz=MODEL_IMGSZ, verbose=False)
    model_segmentation.predict(dummy_frame, imgsz=MODEL_SEG_IMGSZ, verbose=False)
    model_pose.predict(dummy_frame, imgsz=MODEL_IMGSZ, verbose=False)
    print("Warm-up completado.")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Redes neuronales listas"}

@app.post("/api/upload_video")
async def process_video(file: UploadFile = File(...)):
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
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, process_video_sync, video_path, video_key)
    return result

def process_video_sync(video_path: str, video_key: str):
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir el video")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    processed_count = 0
    
    SEG_SKIP = 3  # Segmentación cada 3 frames procesados
    last_seg_result = None
    
    while cap.isOpened() and frame_idx < limit_frames:
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Frame Skipping: Procesar 1 de cada 3 frames 
        if frame_idx % 3 != 0:
            frame_idx += 1
            continue
            
        t0 = time.time()
        frame_data = {"timestamp_ms": timestamp_ms, "detections": [], "poses": [], "segmentations": []}
        
        # Pose
        res_pose = model_pose.predict(frame, conf=CONF_POSE, iou=0.45, imgsz=MODEL_IMGSZ, verbose=False)[0]
        if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
            xy = res_pose.keypoints.xy.cpu().numpy()
            conf_vals = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else np.zeros_like(xy[:, :, 0])
            for i in range(len(xy)):
                keypoints_list = [{"x": float(kp[0]), "y": float(kp[1]), "confidence": float(c)} for kp, c in zip(xy[i], conf_vals[i])]
                frame_data["poses"].append({"keypoints": keypoints_list})
                
        # Segmentation (con skip de frames)
        classes_seg = [i for i in range(80) if i != 0]
        if processed_count % SEG_SKIP == 0 or last_seg_result is None:
            res_seg = model_segmentation.predict(frame, conf=CONF_SEG, iou=0.45, imgsz=MODEL_SEG_IMGSZ, classes=classes_seg, verbose=False)[0]
            last_seg_result = res_seg
        else:
            res_seg = last_seg_result
            
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
        res_det = model_detection.predict(frame, conf=0.12, iou=0.25, imgsz=MODEL_IMGSZ, verbose=False)[0]
        detections = res_det.obb if (hasattr(res_det, 'obb') and res_det.obb is not None and len(res_det.obb) > 0) else res_det.boxes
        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                cls_id = int(detections.cls[i].item())
                confidence = float(detections.conf[i].item())
                cls_name = res_det.names[cls_id].upper().strip()
                thresh = CLASS_CONF_THRESHOLDS.get(cls_name, 0.25)
                
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
        processed_count += 1
        
        # Log progress
        if processed_count == 1 or processed_count % 10 == 0 or frame_idx == limit_frames:
            print(f"Procesando frame {frame_idx}/{limit_frames} (Procesados: {processed_count})...")
            
    cap.release()
    
    json_data["video_info"]["total_frames"] = frame_idx
    if conf_count > 0:
        json_data["video_info"]["custom_classes_conf_avg"] = conf_sum / conf_count
    if processed_count > 0:
        json_data["video_info"]["inference_time_ms_avg"] = inference_time_sum / processed_count
        
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