from typing import List, Tuple
import cv2
from ultralytics import YOLO

class YOLOv8PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.4):
        """
        model_path: 'yolov8n.pt' (small, fast) or 'yolov8s.pt' (more accurate)
        conf: confidence threshold
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame) -> List[Tuple[int, int, int, int, float]]:
        """
        Returns list of:
        (x1, y1, x2, y2, confidence) for each person
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]

        boxes = []
        for box in results.boxes:
            cls = int(box.cls[0])  # class index
            if cls == 0:  # 0 = person in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))

        return boxes
