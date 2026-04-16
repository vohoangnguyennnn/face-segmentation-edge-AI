import os
import random
import shutil

from training.models.config import DATASET_SPLIT_DIR, REP_IMAGES_DIR

# ─── Guard: must run from repo root ────────────────────────────────────────
_ROOT_GUARD = "requirements.txt"
if not os.path.isfile(_ROOT_GUARD):
    raise RuntimeError(
        f"'{_ROOT_GUARD}' not found — run this script from the repo root.\n"
        f"Current directory: {os.getcwd()}"
    )

SRC_DIR = os.path.join(DATASET_SPLIT_DIR, "images", "train")
OUT_DIR = REP_IMAGES_DIR
N       = 500
SEED    = 42

def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
    random.shuffle(imgs)
    imgs = imgs[:N]

    for i, f in enumerate(imgs):
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(OUT_DIR, f"{i:05d}_" + f)
        shutil.copy2(src, dst)

    print("Saved rep images:", len(imgs), "to", OUT_DIR)

if __name__ == "__main__":
    main()
