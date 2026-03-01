"""Simple predictor for `DDS_UNet` models.

Loads a DDS_UNet model (if a weights file is present) and runs
inference on images placed in `./img`. Outputs are saved to
`./img_out` as single-channel prediction images.
"""
import os
import time
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from DDS_UNet.DDS_UNet import DDS_UNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_DIR = './img'
OUT_DIR = './img_out'
os.makedirs(OUT_DIR, exist_ok=True)

# instantiate model
model = DDS_UNet(num_classes=1, input_channels=3).to(DEVICE)

# try common weight locations
weight_paths = ['./model.pth', './models/model.pth']
for p in weight_paths:
    if os.path.exists(p):
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        print(f'Loaded weights from {p}')
        break

model.eval()

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = Image.open(path).convert('RGB')
    input_t = transform(img).unsqueeze(0).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        out = model(input_t)
        out = torch.sigmoid(out)
    dur = time.time() - start

    pred = out.squeeze(0).cpu().numpy()  # C x H x W
    # if multiple channels, take first
    if pred.shape[0] > 1:
        pred = pred[0]
    else:
        pred = pred[0]

    # convert to 0-255 uint8
    pred_img = (pred * 255.0).clip(0, 255).astype('uint8')
    Image.fromarray(pred_img).save(os.path.join(OUT_DIR, fname))
    print(f"Saved {fname} — time: {dur:.3f}s")
