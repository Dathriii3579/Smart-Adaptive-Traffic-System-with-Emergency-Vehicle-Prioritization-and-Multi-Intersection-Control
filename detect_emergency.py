from ultralytics import YOLO
import cv2

# path to trained weights (after training). Use 'best.pt' path from runs/detect/train/weights/
MODEL_PATH = "runs/detect/train/weights/best.pt"  # change as needed
CONF_THRESH = 0.35

# load model
model = YOLO(MODEL_PATH)

# class names (must match your training YAML)
CLASS_NAMES = ['ambulance','firetruck']

def detect_frame(frame):
    # Run inference
    results = model.predict(source=frame, conf=CONF_THRESH, imgsz=640, max_det=20, verbose=False)
    # results is a list; take first
    r = results[0]
    detections = []
    for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        cls = int(cls)
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        x1,y1,x2,y2 = box.astype(int)
        detections.append({'label': label, 'conf': float(conf), 'bbox': [x1,y1,x2,y2]})
    return detections

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # or path to video/cctv stream
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detect_frame(frame)
        for d in dets:
            x1,y1,x2,y2 = d['bbox']
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, f"{d['label']} {d['conf']:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
        cv2.imshow("Emergency detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
