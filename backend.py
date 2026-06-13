import os
import time
import json
import uuid
import shutil
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    
    # Load detection model (look for best.pt)
    best_weights = glob.glob('**/weights/best.pt', recursive=True)
    det_path = best_weights[0] if best_weights else 'yolo26n-obb.pt'
    print(f"Usando modelo de detección: {det_path}")
    
    model_detection = YOLO(det_path)
    model_segmentation = YOLO('yolo26n-seg.pt')
    model_pose = YOLO('yolo26n-pose.pt')
    print("Modelos cargados.")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Redes neuronales listas"}

@app.post("/api/upload_video")
def process_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un video")
        
    os.makedirs("videos", exist_ok=True)
    video_id = str(uuid.uuid4())[:8]
    video_key = f"upload_{video_id}"
    video_path = f"videos/{video_key}.mp4"
    json_path = f"videos/{video_key}_data.json"
    
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
    
    # Limit processing to 3 seconds to avoid Hugging Face timeout at 30 FPS
    limit_frames = min(int(3 * fps), total_frames)
    
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
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        frame_data = {"detections": [], "poses": [], "segmentations": []}
        
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
            
    cap.release()
    
    json_data["video_info"]["total_frames"] = frame_idx
    if conf_count > 0:
        json_data["video_info"]["custom_classes_conf_avg"] = conf_sum / conf_count
    if frame_idx > 0:
        json_data["video_info"]["inference_time_ms_avg"] = inference_time_sum / frame_idx
        
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
        
    print(f"Inferencia completada. JSON guardado en {json_path}")
    
    return {
        "video_key": video_key,
        "video_url": f"/videos/{video_key}.mp4",
        "json_url": f"/videos/{video_key}_data.json"
    }

# Serve the 'videos' folder to be able to stream video and json
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# Serve static root files (like index.html, index.css)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8555, reload=False)
