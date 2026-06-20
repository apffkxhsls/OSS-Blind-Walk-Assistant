# 실행: python analyze.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# 1. 버전별 전체 지표 데이터
versions = {
    "v2\n(Baseline)":           {"images": 408,  "precision": 0.69468, "recall": 0.44115, "mAP50": 0.51363, "mAP50_95": 0.34802},
    "v3\n(Normalization)":         {"images": 500,  "precision": 0.74904, "recall": 0.58637, "mAP50": 0.61914, "mAP50_95": 0.41367},
    "v4\n(+flip)":          {"images": 823,  "precision": 0.75110, "recall": 0.49176, "mAP50": 0.54689, "mAP50_95": 0.40128},
    "v5\n(+brightness)":    {"images": 1355, "precision": 0.83045, "recall": 0.50952, "mAP50": 0.54308, "mAP50_95": 0.42327},
    "v6\n(+rotation)":      {"images": 1359, "precision": 0.80335, "recall": 0.54917, "mAP50": 0.62380, "mAP50_95": 0.45639},
    "v7\n(+noise/blur)":    {"images": 1360, "precision": 0.79089, "recall": 0.56100, "mAP50": 0.58371, "mAP50_95": 0.43193},
    "v7\n(epoch100)":       {"images": 1360, "precision": 0.91225, "recall": 0.46558, "mAP50": 0.54101, "mAP50_95": 0.41639},
    "v7\n(patience15)":     {"images": 1360, "precision": 0.82999, "recall": 0.45763, "mAP50": 0.51809, "mAP50_95": 0.35690},
}

df = pd.DataFrame(versions).T
labels = list(versions.keys())
x = np.arange(len(labels))

# 2. confusion matrix에서 읽은 클래스별 TP 데이터
# 각 버전의 confusion matrix 대각선 값 (TP)
# 순서: bicycle, bollard, car, damaged_braille_block, kickboard, motorcycle, trash, utility_pole
class_names = [
    "bicycle", "bollard", "car", "damaged_braille_block",
    "kickboard", "motorcycle", "trash", "utility_pole"
]

# v2, v3는 클래스 10개라 bollard/fire_hydrant/traffic_cone 포함 → 8클래스 버전과 직접 비교 불가하므로 v4부터 클래스별 분석
tp_data = {
    "v4\n(+flip)":       [10, 0, 3, 3, 3, 1, 4, 2],
    "v5\n(+brightness)": [10, 0, 3, 3, 2, 1, 6, 0],
    "v6\n(+rotation)":   [10, 0, 3, 3, 2, 1, 6, 0],
    "v7\n(+noise/blur)": [9,  0, 3, 2, 3, 1, 7, 0],
    "v7\n(epoch100)":    [9,  0, 3, 2, 3, 1, 7, 0],
}

tp_df = pd.DataFrame(tp_data, index=class_names)