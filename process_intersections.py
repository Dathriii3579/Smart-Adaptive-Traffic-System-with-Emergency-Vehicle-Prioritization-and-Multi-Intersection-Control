
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import List

import cv2
from ultralytics import YOLO

# defaults
DEFAULT_CONF = 0.25
DEFAULT_IMG_SIZE = 1280
ROI_TOP_FRAC = 0.75     # bottom 25% considered "approaching"
TRAFFIC_JSON = "traffic_data.json"
GREEN_JSON = "green_times.json"

EMERGENCY_CLASSES = ["ambulance", "firetruck"]
BASE_GREEN_TIME = 15
PER_VEHICLE_EXTRA = 0.5
MAX_GREEN_TIME = 60
EMERGENCY_GREEN_TIME = 45

def normalize_path(p: str) -> str:
    """Normalize spaces (including non-breaking) and remove surrounding quotes."""
    p = p.strip().strip('"').strip("'")
    p = re.sub(r'\s+', ' ', p, flags=re.UNICODE)
    return p

def expand_images(images: List[str], images_folder: str = None) -> List[str]:
    imgs = []
    if images_folder:
        # find common image extensions
        pattern = os.path.join(images_folder, '*')
        candidates = sorted(glob.glob(pattern))
        # filter typical image extensions
        for c in candidates:
            if c.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                imgs.append(c)
    # explicit images supplied override / append
    if images:
        for p in images:
            np = normalize_path(p)
            if os.path.exists(np):
                imgs.append(np)
            else:
                # try to fix common non-ascii spaces
                # try a naive replacement of narrow no-break space (u202f) -> space
                np2 = np.replace('\u202f', ' ').replace('\u00A0', ' ')
                if os.path.exists(np2):
                    imgs.append(np2)
                else:
                    print(f"[WARN] image not found: {p!r}", file=sys.stderr)
    # deduplicate while keeping order
    seen = set()
    out = []
    for p in imgs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def post_filter_detections(res, model, img_shape, conf_thresh=0.25, min_area_frac=0.0005, roi_top_frac=ROI_TOP_FRAC):
    h, w = img_shape[:2]
    img_area = max(1, h * w)
    filtered = []

    boxes = getattr(res.boxes, "xyxy", None)
    if boxes is None or len(boxes) == 0:
        return filtered

    boxes_np = res.boxes.xyxy.cpu().numpy()
    cls_np = res.boxes.cls.cpu().numpy()
    conf_np = res.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cls_i, conf in zip(boxes_np, cls_np, conf_np):
        label = model.names[int(cls_i)]
        if conf < conf_thresh:
            continue
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        if (area / img_area) < min_area_frac:
            continue
        cy = (y1 + y2) / 2.0
        if cy < h * roi_top_frac:
            continue
        filtered.append({
            "label": label,
            "conf": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })
    return filtered

def calculate_green_time(vehicle_count: int) -> float:
    g = BASE_GREEN_TIME + PER_VEHICLE_EXTRA * vehicle_count
    return round(max(BASE_GREEN_TIME, min(g, MAX_GREEN_TIME)), 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='path to model .pt')
    p.add_argument('--conf', type=float, default=DEFAULT_CONF, help='confidence threshold')
    p.add_argument('--imgsz', type=int, default=DEFAULT_IMG_SIZE, help='inference image size')
    p.add_argument('--out', default='out_frames_images', help='output folder for annotated images')
    p.add_argument('--images_folder', help='folder containing images to process')
    p.add_argument('--images', nargs='*', help='explicit image paths (overrides images_folder entries with same names)', default=[])
    p.add_argument('--min_area_frac', type=float, default=0.0005, help='min box area fraction of image')
    p.add_argument('--roi_top_frac', type=float, default=ROI_TOP_FRAC, help='roi top fraction (0..1)')
    p.add_argument('--debug', action='store_true', help='print debug info')
    args = p.parse_args()

    model_path = normalize_path(args.model)
    if not os.path.exists(model_path):
        print(f"[ERROR] model not found: {model_path}")
        sys.exit(2)

    imgs = expand_images(args.images, args.images_folder)
    if len(imgs) == 0:
        print("[ERROR] No images found. Provide --images_folder or --images explicitly.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(imgs)} images. Loading model '{model_path}' ...")
    model = YOLO(model_path)
    print("Model classes:", model.names)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for i, img_path in enumerate(imgs):
        try:
            if args.debug:
                print(f"[INFO] Processing {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read {img_path}", file=sys.stderr)
                continue

            preds = model(img, imgsz=args.imgsz, conf=args.conf, verbose=False)
            res = preds[0]

            filtered = post_filter_detections(res, model, img.shape, conf_thresh=args.conf,
                                              min_area_frac=args.min_area_frac, roi_top_frac=args.roi_top_frac)
            vehicle_count = len(filtered)
            emergency = any(d["label"] in EMERGENCY_CLASSES for d in filtered)

            if args.debug:
                print(f"  detections raw count: {len(getattr(res.boxes,'xyxy',[]))}")
                print(f"  filtered: {filtered}")
                print(f"  -> vehicles={vehicle_count}, emergency={emergency}")

            # Annotate: res.plot() usually returns numpy image with boxes
            try:
                annotated = res.plot()  # ultralytics Results.plot()
                # res.plot() may return PIL in some versions; ensure numpy
                if hasattr(annotated, 'convert'):
                    annotated = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            except Exception as e:
                # fallback: draw nothing, use original image
                if args.debug:
                    print("[WARN] res.plot() failed, saving original image:", e)
                annotated = img

            out_file = os.path.join(out_dir, f"intersection_{i+1}.jpg")
            ok = cv2.imwrite(out_file, annotated)
            if not ok:
                print(f"[WARN] cv2.imwrite failed for {out_file}", file=sys.stderr)
            else:
                if args.debug:
                    print(f"  saved annotated -> {out_file}")

            results[f"intersection_{i+1}"] = {
                "image": os.path.basename(img_path),
                "vehicles": vehicle_count,
                "emergency": emergency,
                "last_seen": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            print(f"[ERROR] exception while processing {img_path}:", e, file=sys.stderr)

    # Save outputs
    with open(TRAFFIC_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    if args.debug:
        print(f"Saved {TRAFFIC_JSON}")

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
    with open(GREEN_JSON, 'w') as f:
        json.dump(green_times, f, indent=2)
    if args.debug:
        print(f"Saved {GREEN_JSON}")
    print("Annotated images (if any) are in:", out_dir)
    print("Traffic summary:", TRAFFIC_JSON)
    print("Green timing summary:", GREEN_JSON)

if __name__ == '__main__':
    main()
