import os
import csv
import argparse
import time
import pickle

import torch
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

from unet_model import CustomUNet
from cvc_dataset import CVCClinicDBDataset


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


class InferenceDataset:
    """Wrapper around CVCClinicDBDataset that returns (image, mask, filename, orig_path, gt_path)."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Handle both regular dataset and Subset
        if hasattr(self.dataset, 'dataset'):
            # This is a Subset
            original_idx = self.dataset.indices[idx]
            filename = self.dataset.dataset.images[original_idx]
            orig_path = os.path.join(self.dataset.dataset.image_dir, filename)
            gt_path = os.path.join(self.dataset.dataset.mask_dir, filename)
        else:
            # This is the regular dataset
            filename = self.dataset.images[idx]
            orig_path = os.path.join(self.dataset.image_dir, filename)
            gt_path = os.path.join(self.dataset.mask_dir, filename)
            
        return image, mask, filename, orig_path, gt_path


def save_mask(mask_tensor, save_path):
    # Accept torch tensors with shapes like:
    #  (1, 1, H, W), (1, H, W), (H, W), or boolean arrays.
    if isinstance(mask_tensor, torch.Tensor):
        arr = mask_tensor.cpu().numpy()
    else:
        arr = np.array(mask_tensor)

    # Remove any singleton batch/channel dimensions until 2D
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    while arr.ndim > 2 and arr.shape[0] != 1 and arr.shape[1] == 1:
        arr = arr.squeeze(1)

    if arr.ndim != 2:
        # As a last resort, try a generic squeeze
        arr = np.squeeze(arr)

    # Convert boolean or float mask to uint8 image (0 or 255)
    arr_bin = (arr > 0).astype(np.uint8) * 255

    pil = Image.fromarray(arr_bin)
    pil.save(save_path)


def save_side_by_side(original_path, gt_path, pred_arr, save_path, img_size=(256, 256)):
    """Save a side-by-side image: original | ground-truth | predicted.

    pred_arr should be a 2D array with values 0 or 255 (uint8) or boolean.
    """
    # Load and resize original
    orig = Image.open(original_path).convert('RGB')
    orig = orig.resize(img_size, Image.BILINEAR)

    # Load and prepare ground truth
    gt = Image.open(gt_path).convert('L')
    gt = gt.resize(img_size, Image.NEAREST)
    gt_arr = np.array(gt)
    gt_bin = (gt_arr > 127).astype(np.uint8) * 255
    gt_img = Image.fromarray(gt_bin).convert('RGB')

    # Prepare predicted (ensure 2D uint8)
    if isinstance(pred_arr, torch.Tensor):
        pred_np = pred_arr.cpu().numpy()
    else:
        pred_np = np.array(pred_arr)

    pred_np = np.squeeze(pred_np)
    pred_bin = (pred_np > 0).astype(np.uint8) * 255
    pred_img = Image.fromarray(pred_bin.astype(np.uint8)).convert('RGB')
    pred_img = pred_img.resize(img_size, Image.NEAREST)

    # Combine horizontally
    w, h = img_size
    combined = Image.new('RGB', (w * 3, h))
    combined.paste(orig, (0, 0))
    combined.paste(gt_img, (w, 0))
    combined.paste(pred_img, (w * 2, 0))

    combined.save(save_path)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = CustomUNet(in_channels=3, num_classes=1).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Dataset
    image_dir = os.path.join(args.dataset_root, 'PNG', 'Original')
    mask_dir = os.path.join(args.dataset_root, 'PNG', 'Ground Truth')
    base_dataset = CVCClinicDBDataset(image_dir, mask_dir, img_size=args.img_size)
    
    # Use test split if provided
    if args.use_test_split:
        with open(args.use_test_split, 'rb') as f:
            test_indices = pickle.load(f)
        dataset = Subset(base_dataset, test_indices)
        print(f"Using test split with {len(test_indices)} images")
        dataset = InferenceDataset(dataset)
    else:
        dataset = InferenceDataset(base_dataset)

    num_workers = int(os.getenv('NUM_WORKERS', '0'))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    preds_dir = os.path.join(args.output_dir, 'pred_masks')
    os.makedirs(preds_dir, exist_ok=True)
    
    if args.save_side_by_side:
        combined_dir = os.path.join(args.output_dir, 'combined_images')
        os.makedirs(combined_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, 'inference_metrics.csv')
    fieldnames = ['filename', 'dice', 'iou', 'precision', 'recall', 'accuracy']

    results = []
    start = time.time()

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for images, masks, filenames, orig_paths, gt_paths in tqdm(loader, desc='Inference'):
                images = images.to(device)
                masks = masks.to(device)

                with autocast():
                    outputs = model(images)

                for i in range(images.shape[0]):
                    out = outputs[i:i+1]
                    gt = masks[i:i+1]
                    fname = filenames[i]
                    orig_path = orig_paths[i]
                    gt_path = gt_paths[i]

                    metrics = calculate_metrics(out, gt, threshold=args.threshold)

                    # Save predicted mask
                    save_path = os.path.join(preds_dir, fname)
                    pred_binary = torch.sigmoid(out).cpu() > args.threshold
                    save_mask(pred_binary, save_path)
                    
                    # Save side-by-side image if requested
                    if args.save_side_by_side:
                        combined_path = os.path.join(combined_dir, fname)
                        save_side_by_side(orig_path, gt_path, pred_binary, combined_path, img_size=(args.img_size, args.img_size))

                    row = {'filename': fname}
                    row.update({k: metrics[k] for k in ['dice', 'iou', 'precision', 'recall', 'accuracy']})
                    writer.writerow(row)
                    results.append(row)

    elapsed = time.time() - start

    # Summary
    if results:
        avg = {k: sum(r[k] for r in results) / len(results) for k in ['dice', 'iou', 'precision', 'recall', 'accuracy']}
        print(f"Inference finished in {elapsed:.1f}s. Processed {len(results)} images.")
        print("Average metrics:")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("No results written.")

    print(f"Per-image CSV saved to: {csv_path}")
    print(f"Predicted masks saved to: {preds_dir}")
    if args.save_side_by_side:
        print(f"Combined images saved to: {combined_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='training_results/unet_model_final.pth')
    parser.add_argument('--dataset-root', type=str, default='../dataset/cvc_clinicDB')
    parser.add_argument('--output-dir', type=str, default='inference_results')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save-side-by-side', dest='save_side_by_side', action='store_true',
                        help='Save combined images (original | ground truth | prediction)')
    parser.add_argument('--use-test-split', type=str, default=None,
                        help='Path to pickle file containing test indices')
    parser.set_defaults(save_side_by_side=True)

    args = parser.parse_args()
    main(args)
