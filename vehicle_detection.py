import os
import time
import json
from datetime import datetime
import argparse
import cv2
from ultralytics import YOLO
import numpy as np

# ---------- CONFIG ----------
MODEL_PATH = "runs/detect/train/weights/best.pt"  # path to your trained model
CONF_THRESH = 0.35
MAX_DETECTIONS = 30

# ROI: fraction from top (y) for "approaching" area; e.g., bottom 25% -> roi_top_frac = 0.75
ROI_TOP_FRAC = 0.75

# filenames
TRAFFIC_DATA_FILE = "traffic_data.json"
OUT_FRAMES_DIR = "out_frames"

# class names (must match training order)
CLASS_NAMES = ["ambulance", "firetruck"]

# classes considered "vehicles" for simple counting (you can extend)
COUNT_CLASSES = CLASS_NAMES + ["car", "truck", "bus", "motorbike"]  # additional names if model has them
# Note: If your model only has 2 classes, COUNT_CLASSES reduces to those two.

# --------------------------------

def ensure_out_dir():
    os.makedirs(OUT_FRAMES_DIR, exist_ok=True)

def det_in_roi(bbox, frame_shape, roi_top_frac=ROI_TOP_FRAC):
    """
    bbox: [x1, y1, x2, y2] in pixel coords
    frame_shape: (h, w, channels)
    roi_top_frac: fraction (0..1) of height where ROI starts (e.g., 0.75 -> bottom 25% is ROI)
    Returns True if bbox center y is inside the ROI.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    roi_top = int(h * roi_top_frac)
    return cy >= roi_top

def write_traffic_data(intersection_name, vehicle_count, emergency_flag):
    now = datetime.utcnow().isoformat() + "Z"
    data = {
        intersection_name: {
            "vehicles": int(vehicle_count),
            "emergency": bool(emergency_flag),
            "last_seen": now
        }
    }
    with open(TRAFFIC_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def draw_detections(frame, dets):
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        label = d["label"]
        conf = d["conf"]
        color = (0,0,255) if label in CLASS_NAMES else (0,255,0)
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main(args):
    # load model
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # Open capture
    if args.video is None:
        cap = cv2.VideoCapture(0)  # webcam
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("ERROR: cannot open video source")
        return

    ensure_out_dir()
    frame_idx = 0
    last_write_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            h, w = frame.shape[:2]

            # Run model on frame (Ultralytics can accept numpy arrays)
            results = model.predict(source=frame, conf=CONF_THRESH, imgsz=640, max_det=MAX_DETECTIONS, verbose=False)
            r = results[0]

            detections = []
            vehicle_count = 0
            emergency_flag = False

            # boxes: r.boxes.xyxy, r.boxes.cls, r.boxes.conf
            boxes = []
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_arr = r.boxes.cls.cpu().numpy()
                conf_arr = r.boxes.conf.cpu().numpy()

                for box, clsid, conf in zip(xyxy, cls_arr, conf_arr):
                    x1, y1, x2, y2 = box.astype(int).tolist()
                    clsid = int(clsid)
                    # Get label (safe mapping)
                    if clsid < len(CLASS_NAMES):
                        label = CLASS_NAMES[clsid]
                    else:
                        label = str(clsid)
                    detections.append({"label": label, "conf": float(conf), "bbox": [x1,y1,x2,y2]})

                    # Count as vehicle if label in COUNT_CLASSES (note: may need to add names)
                    if label in COUNT_CLASSES:
                        vehicle_count += 1

                    # Emergency logic: if label is ambulance/firetruck AND in ROI -> emergency
                    if label in CLASS_NAMES:
                        if det_in_roi([x1,y1,x2,y2], frame.shape):
                            emergency_flag = True

            # write results json (single intersection named intersection_1)
            write_traffic_data("intersection_1", vehicle_count, emergency_flag)

            # Draw ROI visually
            roi_top = int(h * ROI_TOP_FRAC)
            cv2.rectangle(frame, (0, roi_top), (w-1,h-1), (255,255,0), 2)
            draw_detections(frame, detections)

            # show/save frame at most every 0.2s
            now_t = time.time()
            if now_t - last_write_time > 0.2:
                out_path = os.path.join(OUT_FRAMES_DIR, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                last_write_time = now_t

            if args.display:
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exiting. Last traffic data written to", TRAFFIC_DATA_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to video file (optional). If omitted, uses webcam.")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to YOLO model (.pt)")
    parser.add_argument("--display", action="store_true", help="Show video window")
    args = parser.parse_args()
    # override model if user provided
    if args.model:
        MODEL_PATH = args.model
    main(args)
