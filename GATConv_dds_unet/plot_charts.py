import os
import pandas as pd
import matplotlib.pyplot as plt

# ============ CONFIG ============ #
CSV_PATH = "models/kvasir_SEG/DDS_kvasir_SEG_result_base_split_GAT_aug/log.csv"          # path to your CSV
OUT_PATH = "models/kvasir_SEG/DDS_kvasir_SEG_result_base_split_GAT_aug/training_summary.png"
# ================================ #

# ---- load csv ---- #
df = pd.read_csv(CSV_PATH)
epochs = df["epoch"]

# ---- create figure ---- #
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Training Summary", fontsize=18)

# ---- 1. Loss ---- #
ax = axes[0, 0]
ax.plot(epochs, df["loss"], label="Train Loss")
ax.plot(epochs, df["val_loss"], label="Val Loss")
ax.set_title("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)

# ---- 2. Dice ---- #
ax = axes[0, 1]
ax.plot(epochs, df["dice"], label="Train Dice")
ax.plot(epochs, df["val_dice"], label="Val Dice")
ax.set_title("Dice")
ax.set_xlabel("Epoch")
ax.set_ylabel("Dice")
ax.legend()
ax.grid(True)

# ---- 3. IoU ---- #
ax = axes[1, 0]
ax.plot(epochs, df["val_iou"], label="Val IoU", color="tab:green")
ax.set_title("IoU")
ax.set_xlabel("Epoch")
ax.set_ylabel("IoU")
ax.legend()
ax.grid(True)

# ---- 4. Precision & Recall ---- #
ax = axes[1, 1]
ax.plot(epochs, df["val_prec"], label="Val Precision")
ax.plot(epochs, df["val_rec"], label="Val Recall")
ax.set_title("Precision & Recall")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.legend()
ax.grid(True)

# ---- 5. Learning Rate ---- #
ax = axes[2, 0]
ax.plot(epochs, df["lr"])
ax.set_title("Learning Rate Schedule")
ax.set_xlabel("Epoch")
ax.set_ylabel("LR")
ax.grid(True)

# ---- empty subplot (for symmetry) ---- #
axes[2, 1].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"[DONE] Saved combined plot -> {OUT_PATH}")
