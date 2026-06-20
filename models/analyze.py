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

# 3. 시각화
fig = plt.figure(figsize=(20, 18), num="BrailleGuard YOLO11n — Performance Analysis by Version")
fig.suptitle("BrailleGuard YOLO11n — Performance Analysis by Version\n\n", fontsize=16, fontweight="bold", y=0.98)

# 그래프 1: 전체 지표 비교 (라인)
ax1 = fig.add_subplot(3, 2, 1)
metrics = ["precision", "recall", "mAP50", "mAP50_95"]
colors  = ["#2563eb", "#16a34a", "#dc2626", "#d97706"]
markers = ["o", "s", "^", "D"]

for metric, color, marker in zip(metrics, colors, markers):
    ax1.plot(range(len(labels)), df[metric], color=color, marker=marker,
             linewidth=2, markersize=6, label=metric)

ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylim(0.3, 1.0)
ax1.set_title("Overall Metric Trends (By Version)", fontweight="bold")
ax1.set_ylabel("Score")
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.3)
ax1.axvline(x=5, color="gray", linestyle="--", alpha=0.4)  # v6 최고점 표시
ax1.annotate("v6 Best mAP50", xy=(4, 0.624), xytext=(3, 0.72),
             arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=8)

# 그래프 2: Precision vs Recall 트레이드오프
ax2 = fig.add_subplot(3, 2, 2)
short_labels = ["v2", "v3", "v4", "v5", "v6", "v7", "v7\nepoch100", "v7\npatience15"]
scatter_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(labels)))

for i, (label, color) in enumerate(zip(short_labels, scatter_colors)):
    ax2.scatter(df["recall"].iloc[i], df["precision"].iloc[i],
                color=color, s=120, zorder=5)
    ax2.annotate(label, (df["recall"].iloc[i], df["precision"].iloc[i]),
                 textcoords="offset points", xytext=(6, 4), fontsize=7)

ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision vs Recall Trade-off", fontweight="bold")
ax2.set_xlim(0.35, 0.70)
ax2.set_ylim(0.60, 0.95)
ax2.grid(alpha=0.3)
# 이상적인 방향 표시
ax2.annotate("", xy=(0.68, 0.92), xytext=(0.45, 0.72),
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax2.text(0.56, 0.85, "Ideal Direction", fontsize=8, color="gray", rotation=35)

# 그래프 3: mAP50 & mAP50-95 바 차트
ax3 = fig.add_subplot(3, 2, 3)
bar_width = 0.35
b1 = ax3.bar(x - bar_width/2, df["mAP50"],    bar_width, label="mAP50",    color="#2563eb", alpha=0.8)
b2 = ax3.bar(x + bar_width/2, df["mAP50_95"], bar_width, label="mAP50-95", color="#7c3aed", alpha=0.8)

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=8)
ax3.set_ylim(0.25, 0.70)
ax3.set_title("mAP50 & mAP50-95 Comparison", fontweight="bold")
ax3.set_ylabel("Score")
ax3.legend()
ax3.grid(axis="y", alpha=0.3)

# 최고값 강조
best_idx = df["mAP50"].argmax()
ax3.bar(best_idx - bar_width/2, df["mAP50"].iloc[best_idx],
        bar_width, color="#dc2626", alpha=0.9, label="Best mAP50")
ax3.text(best_idx - bar_width/2, df["mAP50"].iloc[best_idx] + 0.005,
         f'{df["mAP50"].iloc[best_idx]:.3f}', ha="center", fontsize=8, color="#dc2626", fontweight="bold")