from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    BEST_MODEL_PATH, CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD, IMG_SIZE, CLASS_NAMES, HIGH_RISK_CLASSES,
)


class BrailleDetector:
    def __init__(self, model_path: Path = BEST_MODEL_PATH):
        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            print(f"[BrailleDetector] 체크포인트 없음 → yolov8n.pt 로드")
            self.model = YOLO("yolov8n.pt")

        self.conf  = CONFIDENCE_THRESHOLD
        self.iou   = IOU_THRESHOLD
        self.imgsz = IMG_SIZE

    def predict(self, image: np.ndarray) -> list[dict]:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            cls_id     = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            detections.append({
                "class_name":   class_name,
                "confidence":   float(box.conf[0]),
                "bbox":         box.xyxy[0].tolist(),
                "is_high_risk": class_name in HIGH_RISK_CLASSES,
            })
        return detections

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = (0, 0, 255) if det["is_high_risk"] else (0, 200, 100)
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img