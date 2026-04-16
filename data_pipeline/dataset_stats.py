import os

# ─── Guard: must run from repo root ────────────────────────────────────────
_ROOT_GUARD = "requirements.txt"
if not os.path.isfile(_ROOT_GUARD):
    raise RuntimeError(
        f"'{_ROOT_GUARD}' not found — run this script from the repo root.\n"
        f"Current directory: {os.getcwd()}"
    )

# Check resized output directories (produced by resize_images_and_masks.py).
IMG_DIR  = "segmentation_dataset/images_256"
MASK_DIR = "segmentation_dataset/masks_256"

for d, label in [(IMG_DIR, "IMG_DIR"), (MASK_DIR, "MASK_DIR")]:
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"{label} not found: {d}\n"
            "Run data_pipeline/resize_images_and_masks.py first."
        )

images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
masks = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")]

img_bases = set(os.path.splitext(f)[0] for f in images)
mask_bases = set(os.path.splitext(f)[0] for f in masks)

missing_masks = sorted(list(img_bases - mask_bases))
extra_masks = sorted(list(mask_bases - img_bases))

print("So anh:", len(images))
print("So mask:", len(masks))
print("Anh thieu mask:", len(missing_masks))
print("Mask thua:", len(extra_masks))

if missing_masks:
    print("Vi du thieu mask:", missing_masks[:20])
if extra_masks:
    print("Vi du mask thua:", extra_masks[:20])