import os
import cv2
import torch
import numpy as np
from albumentations import Resize, Normalize, Compose
from DDS_UNet.DDS_UNet import DDS_UNet
from sklearn.model_selection import train_test_split

# ================= CONFIG ================= #
MODEL_PATH = "models/kvasir_SEG/DDS_kvasir_SEG_result_base_split_GAT_aug/model.pth"

IMG_DIR = "../../dataset/kvasir_SEG/Kvasir-SEG/Kvasir-SEG/images"
MASK_DIR = "../../dataset/kvasir_SEG/Kvasir-SEG/Kvasir-SEG/masks"
IMG_EXT = ".jpg"
MASK_EXT = ".jpg"

OUT_DIR = "models/kvasir_SEG/DDS_kvasir_SEG_result_base_split_GAT_aug/val_predictions_soft"
OUT_DIR_THRESH = "models/kvasir_SEG/DDS_kvasir_SEG_result_base_split_GAT_aug/val_predictions_threshold"
H, W = 384, 384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 100
VAL_RATIO = 0.2
# ========================================== #

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_THRESH, exist_ok=True)

# ---- SAME validation transforms ---- #
transform = Compose([
    Resize(H, W),
    Normalize()
])

# ---- load model ---- #
model = DDS_UNet(
    num_classes=1,
    input_channels=3,
    deep_supervision=False
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---- SAME validation split ---- #
img_ids = [
    os.path.splitext(f)[0]
    for f in os.listdir(IMG_DIR)
    if f.endswith(IMG_EXT)
]

_, val_img_ids = train_test_split(
    img_ids,
    test_size=VAL_RATIO,
    random_state=RANDOM_STATE
)

print(f"[INFO] Saving predictions for {len(val_img_ids)} validation images")

# ---- inference on ALL val images ---- #
with torch.no_grad():
    for img_id in val_img_ids:
        img_path = os.path.join(IMG_DIR, img_id + IMG_EXT)
        mask_path = os.path.join(MASK_DIR, img_id + MASK_EXT)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        aug = transform(image=img, mask=mask[..., None])
        img_t = aug["image"]
        mask_t = aug["mask"]

        img_tensor = (
            torch.from_numpy(img_t)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

        logits = model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # ---- shared visuals ---- #
        img_vis = cv2.resize(img, (W, H))
        gt_vis = (mask_t.squeeze() > 0).astype(np.uint8) * 255
        gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)

        # 🔥 OPTION 1: SOFT MASK (NO THRESHOLD)
        soft_mask = (prob * 255).astype(np.uint8)
        soft_mask_color = cv2.applyColorMap(soft_mask, cv2.COLORMAP_JET)

        combined = np.hstack([img_vis, gt_vis, soft_mask_color])
        cv2.imwrite(os.path.join(OUT_DIR, f"{img_id}.png"), combined)

        # 🔥 OPTION 2: THRESHOLD MASK
        thresh_mask = ((prob > 0.5) * 255).astype(np.uint8)
        thresh_mask_color = cv2.cvtColor(thresh_mask, cv2.COLOR_GRAY2BGR)

        combined_thresh = np.hstack([img_vis, gt_vis, thresh_mask_color])
        cv2.imwrite(os.path.join(OUT_DIR_THRESH, f"{img_id}.png"), combined_thresh)

print("[DONE] All validation predictions saved (SOFT MASK + THRESHOLD visualization)")