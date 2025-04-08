import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np

class OWLVitDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)

    def detect(self, image_np, categories):
        image = Image.fromarray(image_np[..., ::-1])  # Convert BGR to RGB
        inputs = self.processor(text=categories, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1  # LOWERED THRESHOLD
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.tolist()
            label_str = categories[label]
            score_val = score.item()
            print(f"[DEBUG] Detected: {label_str} | Score: {score_val:.2f} | Box: {box}")
            detections.append({
                "box": box,
                "label": label_str,
                "score": score_val
            })

        if not detections:
            print("[DEBUG] No detections in this frame.")

        return detections




