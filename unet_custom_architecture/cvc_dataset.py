import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms


class CVCClinicDBDataset(Dataset):
    """
    Dataset class for CVC-ClinicDB polyp segmentation dataset.
    
    Args:
        image_dir: Path to directory containing original images
        mask_dir: Path to directory containing ground truth masks
        transform: Optional transform to be applied to images and masks
        img_size: Size to resize images (default: 256)
    """
    def __init__(self, image_dir, mask_dir, transform=None, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        # Optional: a set of indices for which augmentation should be applied.
        # If None, transform (if provided) is applied to all samples.
        self._augment_indices = None
        
        # Get all image filenames
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        print(f"Found {len(self.images)} images in {image_dir}")

    def set_augment_indices(self, indices):
        """Specify which original indices should receive augmentation.

        This is useful when using torch.utils.data.random_split, where
        different Subset objects share the same underlying dataset.

        Args:
            indices: Iterable of integer indices in [0, len(dataset)).
        """
        if indices is None:
            self._augment_indices = None
        else:
            self._augment_indices = set(int(i) for i in indices)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Read image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        # Normalize image to [0, 1]
        image = image / 255.0

        # Normalize mask to [0, 1] (binary)
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # Convert to tensors and rearrange dimensions
        # Image: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        # Mask: (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Optional joint transform / augmentation.
        if self.transform is not None:
            apply_transform = (
                self._augment_indices is None or
                idx in self._augment_indices
            )

            if apply_transform:
                result = self.transform(image, mask)
                # Allow compatibility with old-style transforms that only
                # transform the image tensor.
                if isinstance(result, tuple) and len(result) == 2:
                    image, mask = result
                else:
                    image = result

        return image, mask


def get_cvc_dataloaders(dataset_root, batch_size=4, img_size=256, train_split=0.8):
    """
    Create train and validation dataloaders for CVC-ClinicDB dataset.
    
    Args:
        dataset_root: Root directory of CVC-ClinicDB dataset
        batch_size: Batch size for dataloaders
        img_size: Image size to resize to
        train_split: Fraction of data to use for training (default: 0.8)
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split
    
    image_dir = os.path.join(dataset_root, 'PNG', 'Original')
    mask_dir = os.path.join(dataset_root, 'PNG', 'Ground Truth')
    
    # Create full dataset
    dataset = CVCClinicDBDataset(image_dir, mask_dir, img_size=img_size)
    
    # Split into train and validation
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Train set: {train_size} images, Validation set: {val_size} images")
    
    # Create dataloaders
    # Allow configuring num_workers via environment variable `NUM_WORKERS`.
    # Default to 0 to avoid shared-memory allocation issues in constrained environments.
    num_workers = int(os.getenv("NUM_WORKERS", "0"))

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
