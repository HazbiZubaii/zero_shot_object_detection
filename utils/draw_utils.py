import cv2

def draw_boxes(frame, detections):
    h, w, _ = frame.shape
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        label = f"{det['label']} ({det['score']:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame
