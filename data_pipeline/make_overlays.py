import os
import cv2
import numpy as np

IMG_DIR = "images"
MASK_DIR = "masks"
OUT_DIR = "overlays"
os.makedirs(OUT_DIR, exist_ok=True)

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base = os.path.splitext(img_name)[0]

    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, base + ".png")

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        continue

    # ===== tạo mask màu xanh =====
    green_mask = np.zeros_like(img)
    green_mask[:, :, 1] = mask  # kênh G

    # ===== overlay =====
    overlay = cv2.addWeighted(img, 1.0, green_mask, 0.5, 0)

    # ===== tạo viền =====
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(OUT_DIR, base + ".jpg"), overlay)

print("Overlay xong!")