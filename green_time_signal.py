from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Folder where your intersection photos are stored
IMAGE_DIR = "intersection_images"
OUTPUT_JSON = "traffic_data.json"
os.makedirs("out_frames", exist_ok=True)

data = {}

for idx, img_file in enumerate(os.listdir(IMAGE_DIR)):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_file)
    print(f"Processing {img_path}...")
    img = cv2.imread(img_path)

    # Run detection
    results = model.predict(source=img, conf=0.35, verbose=False)
    r = results[0]

    emergency = False
    vehicle_count = len(r.boxes)
    for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        label = model.names[int(cls)]
        if label in ("ambulance", "firetruck"):
            emergency = True

    # Draw detections and save output frame
    annotated = r.plot()
    cv2.imwrite(f"out_frames/detected_{idx:03d}.jpg", annotated)

    # Save detection summary to JSON
    data[f"intersection_{idx+1}"] = {
        "vehicles": vehicle_count,
        "emergency": emergency,
        "last_seen": datetime.utcnow().isoformat() + "Z"
    }

# Write combined results
with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Done! Saved detection images in out_frames/ and JSON in {OUTPUT_JSON}")
