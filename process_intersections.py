#!/usr/bin/env python3
"""
process_intersections_local.py

Runs YOLOv8 detection on 4 local intersection images.
Evaluates vehicle count + emergency vehicles (ambulance/firetruck)
and computes green signal times automatically.
"""

import os, json, cv2
from datetime import datetime
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/train/weights/best.pt"   # your trained model
CONF = 0.25
IMG_SIZE = 1280
ROI_TOP_FRAC = 0.75     # bottom 25% considered "approaching"
OUT_DIR = "out_frames_images"
TRAFFIC_JSON = "traffic_data.json"
GREEN_JSON = "green_times.json"

# Emergency classes (must match your model names)
EMERGENCY_CLASSES = ["ambulance", "firetruck"]

# Green signal timing parameters
BASE_GREEN_TIME = 15
PER_VEHICLE_EXTRA = 0.5
MAX_GREEN_TIME = 60
EMERGENCY_GREEN_TIME = 45

# 👇 Your 4 intersection image paths (absolute or relative)
IMAGE_PATHS = [
    "/Users/dathril/Desktop/b2proj/intersection_images/1.jpeg",
    "intersection_images/WhatsApp Image 2025-10-14 at 00.21.45_03076b87.jpg",
    "/Users/dathril/Desktop/b2proj/intersection_images/WhatsApp Image 2025-10-14 at 00.19.26_9f01fe6a.jpg",
    "/Users/dathril/Desktop/b2proj/intersection_images/WhatsApp Image 2025-10-14 at 00.16.11_91a35be3.jpg"
]
# Make sure these paths exist before running!
# -----------------------------------------

def det_in_roi(bbox, frame_shape, roi_top_frac):
    """Check if detection lies in bottom region of frame."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cy = (y1 + y2) / 2
    return cy >= h * roi_top_frac

def calculate_green_time(vehicle_count):
    g = BASE_GREEN_TIME + PER_VEHICLE_EXTRA * vehicle_count
    return round(max(BASE_GREEN_TIME, min(g, MAX_GREEN_TIME)), 2)

def main():
    model = YOLO(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)
    print("Model classes:", model.names)

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    for i, img_path in enumerate(IMAGE_PATHS):
        if not os.path.exists(img_path):
            print(f"[WARN] File not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        # ---- Use filtered detections ----
        def post_filter_detections(res, model, img_shape,conf_thresh=0.4, min_area_frac=0.001,roi_top_frac=0.75, allowed_labels=None):
        h, w = img_shape[:2]
        filtered = []
        seen_classes = {}
    for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.cls.cpu().numpy(),
                               res.boxes.conf.cpu().numpy()):
        label = model.names[int(cls)]
        if allowed_labels and label not in allowed_labels:
            continue
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
        if area_frac < min_area_frac:
            continue
        cy = (y1 + y2) / 2.0
        if cy < h * roi_top_frac:
            continue
        prev = seen_classes.get(label)
        if prev is None or conf > prev['conf']:
            seen_classes[label] = {'label': label, 'conf': float(conf), 'bbox': [x1, y1, x2, y2]}
    return list(seen_classes.values())

# Now apply it
allowed = ['ambulance', 'firetruck']
filtered = post_filter_detections(res, model, img.shape,
                                  conf_thresh=0.45,
                                  min_area_frac=0.0012,
                                  roi_top_frac=0.75,
                                  allowed_labels=allowed)

vehicle_count = len(filtered)
emergency = any(d['label'] in allowed for d in filtered)
print(f"Filtered detections: {filtered}")


        # Save annotated image
annotated = res.plot()
        out_file = os.path.join(OUT_DIR, f"intersection_{i+1}.jpg")
        cv2.imwrite(out_file, annotated)

        # Save data
        results[f"intersection_{i+1}"] = {
            "image": os.path.basename(img_path),
            "vehicles": vehicle_count,
            "emergency": emergency,
            "last_seen": datetime.utcnow().isoformat() + "Z"
        }
        print(f"Processed {img_path}: vehicles={vehicle_count}, emergency={emergency}")

    # Save traffic summary
    with open(TRAFFIC_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {TRAFFIC_JSON}")

    # Compute green times
    green_times = {}
    for k, v in results.items():
        if v["emergency"]:
            g = EMERGENCY_GREEN_TIME
            mode = "EMERGENCY"
        else:
            g = calculate_green_time(v["vehicles"])
            mode = "NORMAL"
        green_times[k] = {
            "vehicles": v["vehicles"],
            "emergency": v["emergency"],
            "green_time_seconds": g,
            "mode": mode
        }

    with open(GREEN_JSON, "w") as f:
        json.dump(green_times, f, indent=2)

    print(f"Saved {GREEN_JSON}")
    print("Annotated images are in", OUT_DIR)

if __name__ == "__main__":
    main()
