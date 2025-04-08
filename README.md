# 🧠 Zero-Shot Object Detection using OWL-ViT

This project demonstrates **Zero-Shot Object Detection** using **OWL-ViT (Open World Localization Vision Transformer)**, applied on **real-time webcam input**. Unlike traditional detectors trained on a fixed set of classes (e.g., COCO), OWL-ViT enables object detection via **natural language prompts** — making it ideal for **custom, non-COCO categories** such as:

- Lightbulb  
- Monitor  
- Lion  
- Matchstick  
- Gaming console  

---

## ✨ Features

- 🧠 **Zero-shot detection** using natural language
- 🎥 **Real-time webcam feed** analysis
- 🎯 **Custom category detection** with text input
- 📦 Modular code structure for easy editing
- 🖼️ Bounding box drawing over detected objects

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/zero-shot-object-detection.git
cd zero-shot-object-detection

### 2. Install Dependencies

pip install -r requirements.txt


### 3. Run the Project

python main.py

Then enter your desired objects as a comma-separated prompt when asked:

 Press q to quit the webcam window.