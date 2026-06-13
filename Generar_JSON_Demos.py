import os
import glob
import time
import json
import numpy as np
import cv2
from ultralytics import YOLO

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def process_video_to_json(video_path, output_json, models):
    model_detection, model_segmentation, model_pose = models
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will process max 20 seconds to keep files reasonable
    limit_frames = min(int(20 * fps), total_frames)
    
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
    
    print(f"Processing {video_path}...")
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < limit_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        
        frame_data = {
            "detections": [],
            "poses": [],
            "segmentations": []
        }
        
        # 1. Pose estimation
        res_pose = model_pose.predict(frame, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
        if res_pose.keypoints is not None and len(res_pose.keypoints) > 0:
            xy = res_pose.keypoints.xy.cpu().numpy()
            conf = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints.conf is not None else np.zeros_like(xy[:, :, 0])
            for i in range(len(xy)):
                keypoints_list = []
                for j in range(len(xy[i])):
                    keypoints_list.append({
                        "x": float(xy[i][j][0]),
                        "y": float(xy[i][j][1]),
                        "confidence": float(conf[i][j])
                    })
                frame_data["poses"].append({"keypoints": keypoints_list})
                
        # 2. Segmentation (excluding persons = 0)
        classes_seg = [i for i in range(80) if i != 0]
        res_seg = model_segmentation.predict(frame, conf=0.30, iou=0.45, imgsz=640, classes=classes_seg, verbose=False)[0]
        if res_seg.masks is not None and res_seg.boxes is not None:
            masks_xy = res_seg.masks.xy
            for i, mask in enumerate(masks_xy):
                if len(mask) < 3: continue
                cls_id = int(res_seg.boxes.cls[i].item())
                confidence = float(res_seg.boxes.conf[i].item())
                cls_name = res_seg.names[cls_id]
                polygon = [[float(p[0]), float(p[1])] for p in mask]
                frame_data["segmentations"].append({
                    "class_name": cls_name,
                    "confidence": confidence,
                    "polygon": polygon
                })
                
        # 3. Detection
        res_det = model_detection.predict(frame, conf=0.19, iou=0.25, imgsz=640, verbose=False)[0]
        detections = res_det.obb if (hasattr(res_det, 'obb') and res_det.obb is not None and len(res_det.obb) > 0) else res_det.boxes
        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                cls_id = int(detections.cls[i].item())
                confidence = float(detections.conf[i].item())
                cls_name = res_det.names[cls_id]
                
                # Filter by confidence logic from notebook
                cls_name_upper = cls_name.upper().strip()
                thresh = {'TT TABLE': 0.80, 'TT NET': 0.50, 'TT RACKET': 0.35}.get(cls_name_upper, 0.25)
                if confidence >= thresh:
                    # bounding box logic (extracting axis aligned box from OBB if present)
                    if hasattr(detections, 'xyxyxyxy') and detections.xyxyxyxy is not None:
                        # OBB: convert 4 points to bounding box
                        pts = detections.xyxyxyxy[i].cpu().numpy()
                        x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
                        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
                    else:
                        x1, y1, x2, y2 = detections.xyxy[i].cpu().numpy()
                        
                    frame_data["detections"].append({
                        "class_name": cls_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                    conf_sum += confidence
                    conf_count += 1
                    
        json_data["frames"].append(frame_data)
        
        t1 = time.time()
        inference_time_sum += (t1 - t0) * 1000
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{limit_frames} frames")
            
    cap.release()
    
    # Calculate averages
    json_data["video_info"]["total_frames"] = frame_idx
    if conf_count > 0:
        json_data["video_info"]["custom_classes_conf_avg"] = conf_sum / conf_count
    if frame_idx > 0:
        json_data["video_info"]["inference_time_ms_avg"] = inference_time_sum / frame_idx
        
    with open(output_json, 'w') as f:
        json.dump(json_data, f)
    print(f"Saved JSON to {output_json}")


def main():
    print("Loading models...")
    # Load detection model (look for best.pt)
    best_weights = glob.glob('**/weights/best.pt', recursive=True)
    det_path = best_weights[0] if best_weights else 'yolo26n-obb.pt'
    print(f"Using detection model: {det_path}")
    
    model_det = YOLO(det_path)
    model_seg = YOLO('yolo26n-seg.pt')
    model_pose = YOLO('yolo26n-pose.pt')
    
    models = (model_det, model_seg, model_pose)
    
    # Process videos in videos/ folder
    videos = glob.glob('videos/*.mp4')
    if not videos:
        print("No .mp4 files found in videos/ directory.")
        return
        
    for vid in videos:
        # e.g., videos/demo_01.mp4 -> videos/demo_01_data.json
        basename = os.path.splitext(os.path.basename(vid))[0]
        out_json = os.path.join('videos', f"{basename}_data.json")
        process_video_to_json(vid, out_json, models)
        
if __name__ == '__main__':
    main()
