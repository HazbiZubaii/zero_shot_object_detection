# # main.py

import cv2
from models.owlvit_model import OWLVitDetector
from utils.draw_utils import draw_boxes



if __name__ == "__main__":
    print("Enter the objects to detect (comma-separated, e.g., lightbulb,monitor):")
    categories = input(">> ").strip().split(',')

    print(f"[INFO] Starting webcam with categories: {categories}")

    detector = OWLVitDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, categories)
        frame = draw_boxes(frame, detections)

        cv2.imshow("Zero-Shot Object Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
