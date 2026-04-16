import os
import cv2

# ─── Guard: must run from repo root ────────────────────────────────────────
_ROOT_GUARD = "requirements.txt"
if not os.path.isfile(_ROOT_GUARD):
    raise RuntimeError(
        f"'{_ROOT_GUARD}' not found — run this script from the repo root:\n"
        f"  python data_pipeline/resize_images_and_masks.py\n"
        f"Current directory: {os.getcwd()}"
    )

SRC_IMG_DIR  = "segmentation_dataset/images"
SRC_MASK_DIR = "segmentation_dataset/masks"
OUT_IMG_DIR  = "segmentation_dataset/images_256"
OUT_MASK_DIR = "segmentation_dataset/masks_256"
OUT_W, OUT_H = 256, 256

os.makedirs(OUT_IMG_DIR,  exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

skipped = 0
for img_name in sorted(os.listdir(SRC_IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base = os.path.splitext(img_name)[0]
    img_path  = os.path.join(SRC_IMG_DIR,  img_name)
    mask_path = os.path.join(SRC_MASK_DIR, base + ".png")

    img = cv2.imread(img_path)
    if img is None or not os.path.exists(mask_path):
        skipped += 1
        continue

    mask = cv2.imread(mask_path, 0)

    img_r   = cv2.resize(img,   (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    mask_r  = cv2.resize(mask,  (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(OUT_IMG_DIR,  img_name),  img_r)
    cv2.imwrite(os.path.join(OUT_MASK_DIR, base + ".png"), mask_r)

print(f"Da resize xong. Skipped: {skipped}")
print(f"  Images -> {OUT_IMG_DIR}/")
print(f"  Masks  -> {OUT_MASK_DIR}/")