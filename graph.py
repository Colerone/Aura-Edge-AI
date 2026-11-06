import pandas as pd
import matplotlib.pyplot as plt


csv_file_path = "results_2.csv"
df = pd.read_csv(csv_file_path)

df.columns = df.columns.str.strip()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Graph 1: Loss Curve (Train Loss vs Val Loss) ---
ax1.plot(df['epoch'], df['train/box_loss'], 'r--', label='train', linewidth=1.5)
ax1.plot(df['epoch'], df['val/box_loss'], 'b--', label='val', linewidth=1.5)
ax1.set_title('Training Loss and Validation Loss', fontsize=13, pad=10)
ax1.set_xlabel('epochs')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper right')


ax2.plot(df['epoch'], df['metrics/precision(B)'], 'r--', label='Precision', linewidth=1.5)
ax2.plot(df['epoch'], df['metrics/mAP50(B)'], 'b--', label='mAP50', linewidth=1.5)
ax2.set_title('mAP50 and Precision in Epochs', fontsize=13, pad=10)
ax2.set_xlabel('epochs')
ax2.set_ylabel('Accuracy / Metrics')
ax2.legend(loc='lower right')

plt.tight_layout()

plt.savefig('YOLO_Learning_Curve_Slide.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph image saved as 'YOLO_Learning_Curve_Slide.png'")