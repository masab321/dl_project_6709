import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import csv
import subprocess
import pickle
from datetime import datetime

from tqdm.auto import tqdm

from torch.cuda.amp import autocast, GradScaler  # ✅ AMP

from unet_model import CustomUNet
from cvc_dataset import get_cvc_dataloaders, CVCClinicDBDataset
from loss_fnDiceBCE import BCEDiceLoss as ComboLoss
from Augmentation import SegmentationAugmentations




# Mitigate DataLoader shared-memory issues in constrained environments (Docker, small /dev/shm)
# Fallback safely if sharing strategy cannot be changed on the platform.
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass


# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
IMG_SIZE = 256
NUM_CLASSES = 1
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"training_results_{TIMESTAMP}"

# Resolve dataset path relative to this file's location so it works
# regardless of the current working directory from which the script is run.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_BASE_DIR, "..", "dataset", "cvc_clinicDB")


# --- Metrics Calculation ---
def calculate_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()

    pred = pred_bin.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    epsilon = 1e-7

    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)

    return {
        "accuracy": accuracy.item(),
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item()
    }


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            "loss": 0, "accuracy": 0, "dice": 0,
            "iou": 0, "precision": 0, "recall": 0
        }
        self.count = 0

    def update(self, loss, metric_dict):
        self.metrics["loss"] += loss
        for k, v in metric_dict.items():
            self.metrics[k] += v
        self.count += 1

    def get_avg(self):
        return {k: v / self.count for k, v in self.metrics.items()}


# --- Plotting ---
def plot_history(history, save_path="training_plot.png"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    metrics = ["accuracy", "dice", "iou", "precision", "recall"]
    for i, metric in enumerate(metrics, 2):
        plt.subplot(2, 3, i)
        plt.plot(epochs, history[f"train_{metric}"], 'b-', label=f'Train {metric}')
        plt.plot(epochs, history[f"val_{metric}"], 'r-', label=f'Val {metric}')
        plt.title(metric.capitalize())
        plt.xlabel('Epochs')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def visualize_validation_results(model, val_loader, save_dir, device, num_samples=10):
    """Visualize model predictions on validation data.
    
    Creates combined images showing original | ground truth | prediction.
    """
    import cv2
    
    vis_dir = os.path.join(save_dir, "validation_results")
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    samples_saved = 0
    
    print(f"\nGenerating validation visualizations...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if samples_saved >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Run inference
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            
            # Process each image in batch
            for i in range(images.size(0)):
                if samples_saved >= num_samples:
                    break
                
                # Convert to numpy and denormalize
                img = images[i].cpu().permute(1, 2, 0).numpy() * 255
                img = img.astype(np.uint8)
                
                gt = masks[i, 0].cpu().numpy() * 255
                gt = gt.astype(np.uint8)
                
                pred = preds_binary[i, 0].cpu().numpy() * 255
                pred = pred.astype(np.uint8)
                
                # Convert to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
                
                # Combine horizontally
                combined = np.hstack([img_bgr, gt_bgr, pred_bgr])
                
                # Save
                save_path = os.path.join(vis_dir, f"val_sample_{samples_saved + 1}.png")
                cv2.imwrite(save_path, combined)
                samples_saved += 1
    
    print(f"Saved {samples_saved} validation visualizations to: {vis_dir}")


def save_history_to_csv(history, csv_path="training_history.csv"):
    """Save per-epoch training/validation metrics to a CSV file."""
    # All metric lists should be the same length
    num_epochs = len(history["train_loss"])

    # Explicit column order for readability
    fieldnames = [
        "epoch",
        "train_loss", "val_loss",
        "train_accuracy", "val_accuracy",
        "train_dice", "val_dice",
        "train_iou", "val_iou",
        "train_precision", "val_precision",
        "train_recall", "val_recall",
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch_idx in range(num_epochs):
            row = {"epoch": epoch_idx + 1}
            for key in history:
                row[key] = history[key][epoch_idx]
            writer.writerow(row)


def get_cvc_dataloaders_with_test(dataset_root, batch_size=4, img_size=256, train_split=0.8):
    """Create train and validation dataloaders for CVC-ClinicDB dataset.
    
    Args:
        dataset_root: Root directory of CVC-ClinicDB dataset
        batch_size: Batch size for dataloaders
        img_size: Image size to resize to
        train_split: Fraction of data for training (default: 0.8)
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split
    
    image_dir = os.path.join(dataset_root, 'PNG', 'Original')
    mask_dir = os.path.join(dataset_root, 'PNG', 'Ground Truth')
    
    # Create full dataset (shared by all splits)
    dataset = CVCClinicDBDataset(image_dir, mask_dir, img_size=img_size)

    # 2-way split
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Configure augmentations: only apply on training indices
    aug = SegmentationAugmentations()
    dataset.transform = aug
    dataset.set_augment_indices(train_dataset.indices)
    
    print(f"Train set: {train_size} images, Validation set: {val_size} images")
    
    # Create dataloaders
    num_workers = int(os.getenv("NUM_WORKERS", "3"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader


# --- Main Training Loop ---
def train_model():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Setting up CVC-ClinicDB dataset...")
    train_loader, val_loader = get_cvc_dataloaders_with_test(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        train_split=0.8
    )

    print("Initializing model...")
    model = CustomUNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = ComboLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = GradScaler()  # ✅ AMP scaler

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_accuracy", "val_accuracy",
        "train_dice", "val_dice",
        "train_iou", "val_iou",
        "train_precision", "val_precision",
        "train_recall", "val_recall"
    ]}

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        # --- Training ---
        model.train()
        train_tracker = MetricTracker()

        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
            leave=False,
        )

        for batch_idx, (images, masks) in train_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast():  # ✅ AMP forward
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                metrics = calculate_metrics(outputs, masks)

            train_tracker.update(loss.item(), metrics)

            # Show running loss and Dice on the progress bar
            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{metrics['dice']:.4f}",
            })

        train_avg = train_tracker.get_avg()

        # --- Validation ---
        model.eval()
        val_tracker = MetricTracker()

        val_bar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                with autocast():  # ✅ AMP validation
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)

                metrics = calculate_metrics(outputs, masks)
                val_tracker.update(loss.item(), metrics)

                val_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{metrics['dice']:.4f}",
                })

        val_avg = val_tracker.get_avg()

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_avg['loss']:.4f} | Dice: {train_avg['dice']:.4f} | IoU: {train_avg['iou']:.4f}")
        print(f"  Val   Loss: {val_avg['loss']:.4f} | Dice: {val_avg['dice']:.4f} | IoU: {val_avg['iou']:.4f}")

        history["train_loss"].append(train_avg["loss"])
        history["val_loss"].append(val_avg["loss"])

        for k in ["accuracy", "dice", "iou", "precision", "recall"]:
            history[f"train_{k}"].append(train_avg[k])
            history[f"val_{k}"].append(val_avg[k])

    # --- After training: save plots, CSV, and model ---
    metrics_plot_path = os.path.join(SAVE_DIR, "training_metrics.png")
    csv_path = os.path.join(SAVE_DIR, "training_metrics.csv")
    model_path = os.path.join(SAVE_DIR, "unet_model_final.pth")

    plot_history(history, save_path=metrics_plot_path)
    save_history_to_csv(history, csv_path=csv_path)
    torch.save(model.state_dict(), model_path)

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Epochs run: {NUM_EPOCHS}")
    print(f"Metrics CSV saved to: {csv_path}")
    print(f"Metrics plot saved to: {metrics_plot_path}")
    print(f"Model weights saved to: {model_path}")
    
    # Generate validation visualizations
    visualize_validation_results(model, val_loader, SAVE_DIR, DEVICE, num_samples=10)
    
    # Training complete
    print(f"\nTraining session {TIMESTAMP} finished.")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
