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