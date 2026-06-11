#!/usr/bin/env python3
"""Generate performance comparison figure for FlowNSFW.

Creates bar chart comparing FlowNSFW vs baselines.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from BENCHMARK.md
models = ['Traditional\nML', 'YOLOv11\nv16_s', 'YOLOv11\nauto_v14', 'FlowNSFW\n(Ours)']
accuracy = [55.4, 70.0, 64.5, 96.4]
nsfw_recall = [100.0, 60.0, 41.7, 98.3]
sfw_accuracy = [0.0, 82.0, 92.0, 94.0]
speed_ms = [150, 265, 332, 411]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Colors
colors = ['#95a5a6', '#e74c3c', '#e67e22', '#27ae60']
ours_color = colors[3]

# --- Plot 1: Accuracy Metrics ---
x = np.arange(len(models))
width = 0.25

bars1 = ax1.bar(x - width, accuracy, width, label='Overall Accuracy', color=colors, alpha=0.8)
bars2 = ax1.bar(x, nsfw_recall, width, label='NSFW Recall', color=colors, alpha=0.6)
bars3 = ax1.bar(x + width, sfw_accuracy, width, label='SFW Accuracy', color=colors, alpha=0.4)

ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Performance Comparison (224 Test Videos)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=90, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# Highlight our method
ax1.text(3, 105, '⭐ Our Method', ha='center', fontsize=11,
         fontweight='bold', color=ours_color)

# --- Plot 2: Speed vs Accuracy Trade-off ---
ax2.scatter(speed_ms[:3], accuracy[:3], s=200, c=colors[:3], alpha=0.6, edgecolors='black', linewidth=1.5)
ax2.scatter(speed_ms[3], accuracy[3], s=400, c=ours_color, alpha=0.8,
           edgecolors='black', linewidth=2, marker='*', label='FlowNSFW (Ours)')

for i, model in enumerate(models):
    offset_x = 20 if i == 3 else -20
    offset_y = 5 if i == 3 else -5
    ha = 'left' if i == 3 else 'right'
    ax2.annotate(model.replace('\n', ' '),
                (speed_ms[i], accuracy[i]),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
                ha=ha)

ax2.set_xlabel('Inference Speed (ms/video)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Overall Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(100, 500)
ax2.set_ylim(50, 100)

# Add "better" arrows
ax2.annotate('', xy=(120, 95), xytext=(120, 55),
            arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.5))
ax2.text(110, 75, 'Better', rotation=90, fontsize=10, color='green',
        fontweight='bold', ha='right', va='center')

plt.tight_layout()
plt.savefig('assets/performance_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: assets/performance_comparison.png')

# Also save high-res version for paper
plt.savefig('assets/performance_comparison.pdf', bbox_inches='tight')
print('Saved: assets/performance_comparison.pdf')
