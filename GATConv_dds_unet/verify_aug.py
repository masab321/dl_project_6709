
import numpy as np
import torch
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Resize, Compose,
    OneOf, GaussianBlur, GaussNoise, MultiplicativeNoise, MotionBlur, MedianBlur, Blur,
    OpticalDistortion, GridDistortion, ElasticTransform, CLAHE, HueSaturationValue,
    RandomBrightnessContrast, Normalize
)
import os
import sys

# Mocking the Dataset class logic to verify transforms
def test_transforms():
    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        OneOf([
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], p=0.3),
        OneOf([
            GaussNoise(),
            MultiplicativeNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            ElasticTransform(p=0.3),
        ], p=0.2),
        CLAHE(clip_limit=4.0, p=0.5),
        Resize(384, 384),
        Normalize(),
    ])

    img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (500, 500, 1), dtype=np.uint8) * 255

    augmented = train_transform(image=img, mask=mask)
    aug_img = augmented['image']
    aug_mask = augmented['mask']

    print(f"Original shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Augmented shape: {aug_img.shape}, Mask shape: {aug_mask.shape}")
    print(f"Augmented image range: {aug_img.min()} to {aug_img.max()}")
    
    assert aug_img.shape == (384, 384, 3)
    assert aug_mask.shape == (384, 384, 1) or aug_mask.shape == (384, 384)
    print("Verification successful!")

if __name__ == "__main__":
    try:
        test_transforms()
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
