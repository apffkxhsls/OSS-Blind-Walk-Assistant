import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from config import DATASET_YAML, CHECKPOINT_DIR, IMG_SIZE

def train():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")  # yolov8s.pt / yolov8m.pt 로 변경 가능
    model.train(
        data=str(DATASET_YAML),
        epochs=100,
        imgsz=IMG_SIZE,
        batch=16,
        project=str(CHECKPOINT_DIR),
        name="braille_guard",
        exist_ok=True,
    )
    print(f"학습 완료 → {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train()