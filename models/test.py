# text.py
from ultralytics import YOLO
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import numpy_core_shim

def main():
    # Load a pretrained YOLOv8 model (large, accurate)
    model = YOLO("models/ball_detector_model.pt")  # you can also use yolov8n.pt for faster/smaller

    # Run inference on your basketball image
    results = model("models/shot.jpg")  # replace with your image path if needed

    # Show results visually (opens a window with bounding box)
    results[0].show()

    # Or save results (creates 'runs/detect/predict' folder with annotated image)
    results[0].save()

    # Print raw detections
    for r in results:
        print(r.boxes.xyxy)   # bounding box coordinates
        print(r.boxes.cls)    # class IDs
        print(r.names)        # class name mapping

if __name__ == "__main__":
    main()
