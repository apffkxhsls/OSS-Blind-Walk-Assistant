import cv2
import numpy as np

COLORS = {
    "고위험": (0, 0, 255),
    "주의":   (0, 165, 255),
}

def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = COLORS["고위험"] if det["is_high_risk"] else COLORS["주의"]
        label = f'{det["class_name"]} {det["confidence"]:.0%}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img