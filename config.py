"""
config.py — 프로젝트 전역 경로 및 설정
모든 파일에서 이 파일을 임포트하여 경로를 참조합니다.
하드코딩된 경로를 쓰지 마세요.
"""
from pathlib import Path

# ── 루트 ──────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

# ── 데이터 ────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_IMAGES_DIR  = DATA_DIR / "raw" / "images"
RAW_LABELS_DIR  = DATA_DIR / "raw" / "labels"
TRAIN_DIR       = DATA_DIR / "processed" / "train"
VAL_DIR         = DATA_DIR / "processed" / "val"
TEST_DIR        = DATA_DIR / "processed" / "test"

# ── 모델 ──────────────────────────────────────
MODELS_DIR      = ROOT / "models"
CHECKPOINT_DIR  = ROOT / "models" / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "v7_best.pt"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last.pt"
DATASET_YAML    = MODELS_DIR / "configs" / "dataset.yaml"
MODEL_YAML      = MODELS_DIR / "configs" / "model.yaml"

# ── 에셋 ──────────────────────────────────────
ASSETS_DIR      = ROOT / "assets"
WARNING_SOUND   = ASSETS_DIR / "sounds" / "warning.mp3"
TEST_IMAGES_DIR = ASSETS_DIR / "test_images"

# ── 탐지 설정 ─────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD        = 0.45
IMG_SIZE             = 640

# ── 클래스 라벨 ───────────────────────────────
CLASS_NAMES = [
    "bicycle",
    "bollard", 
    "car",
    "damaged_braille_block",
    "kickboard",
    "motorcycle",
    "trash",
    "utility_pole",
]

HIGH_RISK_CLASSES = {"kickboard", "bicycle", "car", "motorcycle"}