def write_yolo_label(txt_path, class_id, x_min, y_min, x_max, y_max, img_w, img_h):
    x_c = (x_min + x_max) / 2.0 / img_w
    y_c = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    with open(txt_path, 'a') as f:
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

# Example usage:
write_yolo_label("label.txt", 0, 10, 20, 200, 300, 640, 480)
