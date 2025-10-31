# post_filter.py (snippet)
def post_filter_detections(res, model, img_shape,
                           conf_thresh=0.4, min_area_frac=0.0008,
                           roi_top_frac=0.75, allowed_labels=None):
    """
    res: ultralytics single-image result (r)
    model: loaded YOLO model (for model.names)
    img_shape: frame.shape
    allowed_labels: list of labels to consider for counting/emergency (e.g. ['ambulance','firetruck'])
    """
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
        x1,y1,x2,y2 = [int(v) for v in box]
        area_frac = ((x2-x1)*(y2-y1)) / (w*h)
        if area_frac < min_area_frac:
            continue
        # ROI check (center in bottom area)
        cy = (y1 + y2) / 2.0
        if cy < h * roi_top_frac:
            continue
        # keep only top detection per label (optional)
        prev = seen_classes.get(label)
        if prev is None or conf > prev['conf']:
            seen_classes[label] = {'label': label, 'conf': float(conf), 'bbox':[x1,y1,x2,y2]}
    # build result list
    for v in seen_classes.values():
        filtered.append(v)
    return filtered

# Usage:
allowed = ['ambulance','firetruck']  # only these considered emergency
filtered = post_filter_detections(r, model, img.shape, conf_thresh=0.4, min_area_frac=0.001, roi_top_frac=0.75, allowed_labels=allowed)
print("Filtered detections:", filtered)
