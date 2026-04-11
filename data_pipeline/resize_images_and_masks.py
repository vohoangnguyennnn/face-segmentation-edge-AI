import os
import cv2

IMG_DIR = "segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/masks"
OUT_W, OUT_H = 256, 256

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base = os.path.splitext(img_name)[0]
    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, base + ".png")

    img = cv2.imread(img_path)
    if img is None or not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, 0)

    img_r = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(img_path, img_r)
    cv2.imwrite(mask_path, mask_r)

print("Da resize xong.")