import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from unet_model import CustomUNet
from cvc_dataset import get_cvc_dataloaders
from unet_archi.loss_fnDiceBCE import CombinedSegmentationLoss

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 20  # Increased for better training
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", str(NUM_EPOCHS)))
IMG_SIZE = 256
NUM_CLASSES = 1  # Binary segmentation
SAVE_DIR = "training_results"
DATASET_ROOT = "../dataset/cvc_clinicDB"  # Path to CVC-ClinicDB dataset

# --- Metrics Calculation ---
def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculates Accuracy, Dice, IoU, and Precision/Recall for binary segmentation.
    pred: Raw logits or probabilities (B, 1, H, W)
    target: Ground truth (B, 1, H, W)
    """
    # Apply sigmoid if model outputs logits
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()
    
    # Flatten
    pred = pred_bin.view(-1)
    target = target.view(-1)
    
    # True Positives, False Positives, False Negatives, True Negatives
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
    """Helper to track average metrics over an epoch"""
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
    
    # Plot Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot Metrics
    metrics = ["accuracy", "dice", "iou", "precision", "recall"]
    for i, metric in enumerate(metrics, 2):
        plt.subplot(2, 3, i)
        plt.plot(epochs, history[f"train_{metric}"], 'b-', label=f'Train {metric.capitalize()}')
        plt.plot(epochs, history[f"val_{metric}"], 'r-', label=f'Val {metric.capitalize()}')
        plt.title(metric.capitalize())
        plt.xlabel('Epochs')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training graph saved to {save_path}")
    plt.close()

# --- Main Training Loop ---
def train_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Data Setup
    print("Setting up CVC-ClinicDB dataset...")
    print(f"Dataset root: {DATASET_ROOT}")
    train_loader, val_loader = get_cvc_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        train_split=0.8
    )
    
    # 2. Model Setup
    print("Initializing model...")
    model = CustomUNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = CombinedSegmentationLoss(weight_hausdorff=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # History storage
    history = {k: [] for k in ["train_loss", "val_loss", 
                               "train_accuracy", "val_accuracy",
                               "train_dice", "val_dice",
                               "train_iou", "val_iou",
                               "train_precision", "val_precision",
                               "train_recall", "val_recall"]}
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_tracker = MetricTracker()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(outputs, masks)
            train_tracker.update(loss.item(), metrics)
            
        train_avg = train_tracker.get_avg()
        
        # --- Validation Phase ---
        model.eval()
        val_tracker = MetricTracker()
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                metrics = calculate_metrics(outputs, masks)
                val_tracker.update(loss.item(), metrics)
                
        val_avg = val_tracker.get_avg()
        
        # --- Logging & History Update ---
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_avg['loss']:.4f} | Acc: {train_avg['accuracy']:.4f} | Dice: {train_avg['dice']:.4f} | IoU: {train_avg['iou']:.4f}")
        print(f"  Val   Loss: {val_avg['loss']:.4f} | Acc: {val_avg['accuracy']:.4f} | Dice: {val_avg['dice']:.4f} | IoU: {val_avg['iou']:.4f}")
        
        history["train_loss"].append(train_avg["loss"])
        history["val_loss"].append(val_avg["loss"])
        for k in ["accuracy", "dice", "iou", "precision", "recall"]:
            history[f"train_{k}"].append(train_avg[k])
            history[f"val_{k}"].append(val_avg[k])
            
    # --- Save Results ---
    plot_history(history, save_path=os.path.join(SAVE_DIR, "training_metrics.png"))
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "unet_model_final.pth"))
    
    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
