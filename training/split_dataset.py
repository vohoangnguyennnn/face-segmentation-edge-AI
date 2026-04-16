import os
import random
import shutil

# ─── Guard: must run from repo root ────────────────────────────────────────
_ROOT_GUARD = "requirements.txt"
if not os.path.isfile(_ROOT_GUARD):
    raise RuntimeError(
        f"'{_ROOT_GUARD}' not found — run this script from the repo root:\n"
        f"  python training/split_dataset.py\n"
        f"Current directory: {os.getcwd()}"
    )

IMG_DIR = "segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/masks"
OUT_DIR = "dataset_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

def make_dirs():
    for p in [
        "images/train","images/val","images/test",
        "masks/train","masks/val","masks/test"
    ]:
        os.makedirs(os.path.join(OUT_DIR, p), exist_ok=True)

def main():
    random.seed(SEED)
    imgs = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n*TRAIN_RATIO)
    n_val = int(n*VAL_RATIO)

    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train+n_val],
        "test": imgs[n_train+n_val:]
    }

    make_dirs()
    for split, files in splits.items():
        for f in files:
            base = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(IMG_DIR,f), os.path.join(OUT_DIR,"images",split,f))
            shutil.copy2(os.path.join(MASK_DIR,base+".png"), os.path.join(OUT_DIR,"masks",split,base+".png"))

    print("Done split:", {k:len(v) for k,v in splits.items()})

if __name__ == "__main__":
    main()
