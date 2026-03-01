import os
import time
import torch
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Resize, Compose,
    OneOf, GaussianBlur, GaussNoise, MultiplicativeNoise, MotionBlur, MedianBlur, Blur,
    OpticalDistortion, GridDistortion, ElasticTransform, CLAHE, HueSaturationValue,
    RandomBrightnessContrast, Normalize
)
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import cv2
import numpy as np
import losses
from dataset import Dataset
from metrics import all_score
from utils import AverageMeter
from DDS_UNet.DDS_UNet import DDS_UNet
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'epochs': 150,
    'batch_size': 4,
    'arch': 'DDS_UNet',
    'deep_supervision': False,
    'input_channels': 3,
    'num_classes': 1,
    'input_w': 384,
    'input_h': 384,
    'loss': 'BCEDiceBoundaryLoss',
    'dataset': 'kvasir_SEG',
    'optimizer': 'Adam',
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'scheduler': 'CosineAnnealingLR',
    'min_lr': 1e-5,
    'num_workers': 0,
    'seed': 42,  # Random seed for reproducibility
}
config['name'] = 'DDS_' + config['dataset'] + '_' + str(int(time.time()))

dataset_configs = {
    'kvasir_SEG': {
        'img_dir': '../dataset/kvasir_SEG/Kvasir-SEG/Kvasir-SEG/images',
        'mask_dir': '../dataset/kvasir_SEG/Kvasir-SEG/Kvasir-SEG/masks',
        'img_ext': '.jpg',
        'mask_ext': '.jpg',
        'train_ids': [line.strip() for line in open('../dataset/kvasir_SEG/train.txt')],
        'val_ids': [line.strip() for line in open('../dataset/kvasir_SEG/val.txt')]
    },
    'isic18': {
        'img_dir': '../dataset/isic18/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input',
        'mask_dir': '../dataset/isic18/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth',
        'img_ext': '.jpg',
        'mask_ext': '_segmentation.png'
    },
    'cvc_clinicDB': {
        'img_dir': '../dataset/cvc_clinicDB/PNG/Original',
        'mask_dir': '../dataset/cvc_clinicDB/PNG/Ground Truth',
        'img_ext': '.png',
        'mask_ext': '.png'
    }
}

def get_data_loaders():
    dataset_cfg = dataset_configs[config['dataset']]
    img_dir = dataset_cfg['img_dir']
    mask_dir = dataset_cfg['mask_dir']
    img_ext = dataset_cfg['img_ext']
    mask_ext = dataset_cfg['mask_ext']

    if 'train_ids' in dataset_cfg:
        train_img_ids = dataset_cfg['train_ids']
        val_img_ids = dataset_cfg['val_ids']
    else:
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(img_dir, '*' + img_ext))]
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=100)
        
    
    # print(val_img_ids)
    # c = input()

    # train_img_ids = train_img_ids[:len(train_img_ids)//10]
    # val_img_ids = val_img_ids[:len(val_img_ids)//10]

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
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    train_dataset = Dataset(train_img_ids, img_dir, mask_dir, img_ext, mask_ext, config['num_classes'], transform=train_transform)
    val_dataset = Dataset(val_img_ids, img_dir, mask_dir, img_ext, mask_ext, config['num_classes'], transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], drop_last=False)

    return train_loader, val_loader, val_img_ids

def train_epoch(train_loader, model, criterion, optimizer):
    avg_m = {'loss': AverageMeter(), 'dice': AverageMeter()}
    model.train()
    for input, target, _ in tqdm(train_loader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected: {loss.item()}, skipping batch")
            continue
            
        iou_list, dice_list, _, _, _, _ = all_score(output, target)
        iou = iou_list[0]
        dice = dice_list[0]
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        avg_m['loss'].update(loss.item(), input.size(0))
        avg_m['dice'].update(dice, input.size(0))
    return {'loss': avg_m['loss'].avg, 'dice': avg_m['dice'].avg}

def validate(val_loader, model, criterion):
    avg_m = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(), 'acc': AverageMeter(), 'prec': AverageMeter(), 'rec': AverageMeter()}
    model.eval()
    with torch.no_grad():
        for input, target, _ in tqdm(val_loader):
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)
            iou_list, dice_list, acc_list, prec_list, rec_list, _ = all_score(output, target)
            iou = iou_list[0]
            dice = dice_list[0]
            acc = acc_list[0]
            prec = prec_list[0]
            rec = rec_list[0]
            avg_m['loss'].update(loss.item(), input.size(0))
            avg_m['iou'].update(iou, input.size(0))
            avg_m['dice'].update(dice, input.size(0))
            avg_m['acc'].update(acc, input.size(0))
            avg_m['prec'].update(prec, input.size(0))
            avg_m['rec'].update(rec, input.size(0))
    return {'loss': avg_m['loss'].avg, 'iou': avg_m['iou'].avg, 'dice': avg_m['dice'].avg, 'acc': avg_m['acc'].avg, 'prec': avg_m['prec'].avg, 'rec': avg_m['rec'].avg}

def save_example(model, val_img_ids, img_dir, mask_dir, img_ext, mask_ext, model_dir):
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    model.eval()
    validation_images_dir = os.path.join(model_dir, 'validation_images')
    os.makedirs(validation_images_dir, exist_ok=True)
    for img_id in val_img_ids:
        img_path = os.path.join(img_dir, img_id + img_ext)
        mask_path = os.path.join(mask_dir, img_id + mask_ext)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        resize_transform = Resize(config['input_h'], config['input_w'])
        img_resized = resize_transform(image=img)['image']
        mask_resized = cv2.resize(mask, (config['input_w'], config['input_h']))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5
        mask_display = (mask_resized > 0).astype(np.uint8) * 255
        pred_display = pred.astype(np.uint8) * 255
        gt_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred_display, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([img_resized, gt_bgr, pred_bgr])
        cv2.imwrite(os.path.join(validation_images_dir, f'{img_id}.png'), combined)

def main():
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    model_dir = os.path.join('models', config['dataset'], config['name'])
    os.makedirs(model_dir, exist_ok=True)

    loss_class = getattr(losses, config['loss'])
    criterion = loss_class().to(device)
    cudnn.benchmark = True
    model = DDS_UNet(config['num_classes'], config['input_channels'], config['deep_supervision']).to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])

    train_loader, val_loader, val_img_ids = get_data_loaders()
    dataset_cfg = dataset_configs[config['dataset']]
    img_dir = dataset_cfg['img_dir']
    mask_dir = dataset_cfg['mask_dir']
    img_ext = dataset_cfg['img_ext']
    mask_ext = dataset_cfg['mask_ext']

    best_dice = 0
    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('dice', []), ('val_loss', []), ('val_iou', []), ('val_dice', []), ('val_acc', []), ('val_prec', []), ('val_rec', [])])

    for epoch in range(config['epochs']):
        train_log = train_epoch(train_loader, model, criterion, optimizer)
        val_log = validate(val_loader, model, criterion)
        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_acc'].append(val_log['acc'])
        log['val_prec'].append(val_log['prec'])
        log['val_rec'].append(val_log['rec'])
        pd.DataFrame(log).to_csv(os.path.join(model_dir, 'log.csv'), index=False)
        scheduler.step()
        if val_log['dice'] > best_dice:
            best_dice = val_log['dice']
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
        print(f"Epoch {epoch+1}/{config['epochs']} - loss: {train_log['loss']:.4f} - dice: {train_log['dice']:.4f} - val_loss: {val_log['loss']:.4f} - val_dice: {val_log['dice']:.4f}")

    with open(os.path.join(model_dir, 'results.txt'), 'w') as f:
        f.write(f"Final Validation Loss: {val_log['loss']:.4f}\n")
        f.write(f"Final Validation IoU: {val_log['iou']:.4f}\n")
        f.write(f"Final Validation Dice: {val_log['dice']:.4f}\n")
        f.write(f"Final Validation Accuracy: {val_log['acc']:.4f}\n")
        f.write(f"Final Validation Precision: {val_log['prec']:.4f}\n")
        f.write(f"Final Validation Recall: {val_log['rec']:.4f}\n")
        f.write(f"Best Validation Dice: {best_dice:.4f}\n")

    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    save_example(model, val_img_ids, img_dir, mask_dir, img_ext, mask_ext, model_dir)

if __name__ == '__main__':
    main()
