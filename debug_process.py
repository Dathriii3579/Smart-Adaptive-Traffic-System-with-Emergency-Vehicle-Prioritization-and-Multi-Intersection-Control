# debug_all.py
import os, json
from ultralytics import YOLO
import cv2
from datetime import datetime

# ---------- CONFIG ----------
MODEL_PATH = "runs/detect/train/weights/best.pt"   # adjust if different
IMAGE_PATHS = [
   "/Users/dathril/Desktop/b2proj/intersection_images/1.jpeg",
    "intersection_images/WhatsApp Image 2025-10-14 at 00.21.45_03076b87.jpg",
    "/Users/dathril/Desktop/b2proj/intersection_images/WhatsApp Image 2025-10-14 at 00.19.26_9f01fe6a.jpg",
    "/Users/dathril/Desktop/b2proj/intersection_images/WhatsApp Image 2025-10-14 at 00.16.11_91a35be3.jpg"
]
OUT_DIR = "debug_out"
CONF_LIST = [0.5, 0.3, 0.15, 0.1]   # try several thresholds
IMGSZ_LIST = [640, 1280, 1600]      # try different input sizes
# If your model's class names are unknown, we will log model.names
# Temporary emergency fallback mapping (useful if model labeled similarly):
TEMP_EMERGENCY_NAMES = ["ambulance", "firetruck", "truck", "bus"]
# ------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def safe_imread(p):
    img = cv2.imread(p)
    if img is None:
        print("[WARN] cannot read", p)
    return img

# 1) Check model file
diagnostics = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "model_path": MODEL_PATH,
    "model_exists": os.path.exists(MODEL_PATH)
}
print("Model exists:", diagnostics["model_exists"], MODEL_PATH)
if not diagnostics["model_exists"]:
    print("ERROR: model file not found. Set MODEL_PATH to the correct .pt file.")
    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    raise SystemExit(1)

# 2) Load model and print names
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")
diagnostics["model_names"] = model.names
print("Model.names:", model.names)

# 3) Verify images
images = []
for p in IMAGE_PATHS:
    exists = os.path.exists(p)
    diagnostics.setdefault("images", {})[p] = {"exists": exists}
    if not exists:
        print("[WARN] image missing:", p)
        continue
    img = safe_imread(p)
    if img is None:
        diagnostics["images"][p]["readable"] = False
        continue
    diagnostics["images"][p]["readable"] = True
    diagnostics["images"][p]["shape"] = img.shape
    images.append((p, img))

if not images:
    print("No readable images found. Fix IMAGE_PATHS and re-run.")
    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    raise SystemExit(1)

# 4) Run detection across combos and log results
diagnostics["runs"] = []
for imgsz in IMGSZ_LIST:
    for conf in CONF_LIST:
        run_info = {"imgsz": imgsz, "conf": conf, "results": {}}
        print(f"\n--- Running imgsz={imgsz} conf={conf} ---")
        for p, img in images:
            try:
                res = model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)[0]
            except Exception as e:
                print("Inference failed on", p, "error:", e)
                run_info["results"][p] = {"error": str(e)}
                continue

            dets = []
            for box, cls, prob in zip(res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.cls.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy()):
                lbl = model.names[int(cls)] if int(cls) in model.names else str(int(cls))
                b = [int(x) for x in box]
                dets.append({"label": lbl, "conf": float(prob), "bbox": b})
                print(f"  {os.path.basename(p)} -> {lbl} conf={prob:.3f} bbox={b}")

            run_info["results"][p] = {
                "detections_count": len(dets),
                "detections": dets
            }

            # write annotated image for this (imgsz,conf) combo so you can visually inspect
            try:
                ann = res.plot()
                fname = os.path.basename(p)
                outname = f"{os.path.splitext(fname)[0]}_ann_{imgsz}_{int(conf*100)}.jpg"
                outpath = os.path.join(OUT_DIR, outname)
                cv2.imwrite(outpath, ann)
                run_info["results"][p]["annotated_image"] = outpath
            except Exception as e:
                run_info["results"][p]["annotated_image"] = None

        diagnostics["runs"].append(run_info)

# 5) Post-run heuristic: if all detections_count==0, show suggestions
all_zero = True
for r in diagnostics["runs"]:
    for p,res in r["results"].items():
        if isinstance(res, dict) and res.get("detections_count",0) > 0:
            all_zero = False
            break
    if not all_zero:
        break

if all_zero:
    diagnostics["note"] = "NO_DETECTIONS_FOUND in any (imgsz,conf) combination. Possible causes: model class mismatch, model not trained for ambulance/firetruck, or image scale/different appearance."
    diagnostics["suggested_actions"] = [
        "1) Print model.names and confirm it includes 'ambulance' and 'firetruck'.",
        "2) If not present, retrain model with those labels or map similar classes (truck/bus) temporarily.",
        "3) Try training data augmentation or add more examples at camera perspective.",
        "4) If model.names contains classes, paste diagnostics JSON here for further help."
    ]
else:
    diagnostics["note"] = "Some detections were found. Inspect debug_out/ annotated images and detections."

# 6) Save diagnostics file
outdiag = os.path.join(OUT_DIR, "diagnostics.json")
with open(outdiag, "w") as f:
    json.dump(diagnostics, f, indent=2)

print("\nDiagnostics saved to", outdiag)
print("If still failing, paste the contents of debug_out/diagnostics.json here or the printed model.names and a sample 'detections' entry.")
