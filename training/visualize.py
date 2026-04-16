"""Run inference and visualization for the trained U-Net model."""

import os
import shutil
import numpy as np
import tensorflow as tf

from training.models.config import DATASET_SPLIT_DIR, KERAS_MODEL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

IMG_SIZE    = (256, 256)
INPUT_SHAPE = (256, 256, 3)
NUM_SAMPLES = 5
OUTPUT_DIR  = "outputs"
SEED        = 42


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.sigmoid(tf.cast(y_pred, tf.float32))
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom    = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    dice     = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)


@tf.keras.utils.register_keras_serializable()
def combined_bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# NOTE: thresholding sigmoid output before Dice is consistent with
# the Keras-side dice_metric in train_unet.py.
@tf.keras.utils.register_keras_serializable()
def dice_metric(y_true, y_pred, smooth: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom    = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


# NOTE: IoU is also computed on thresholded binary masks, matching Keras-side.
@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, smooth: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union    = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


CUSTOM_OBJECTS = {
    "dice_loss": dice_loss,
    "combined_bce_dice_loss": combined_bce_dice_loss,
    "dice_metric": dice_metric,
    "iou_metric": iou_metric,
}


def load_pair(img_path, mask_path):
    """Load and preprocess a single image-mask pair."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape(INPUT_SHAPE)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    mask.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))
    return img, mask


def make_dataset(split: str) -> tf.data.Dataset:
    """Build a simple tf.data pipeline for inference."""
    img_dir = os.path.join(DATASET_SPLIT_DIR, "images", split)
    mask_dir = os.path.join(DATASET_SPLIT_DIR, "masks",  split)

    imgs = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not imgs:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {img_dir}")

    img_paths  = [os.path.join(img_dir,  f) for f in imgs]
    mask_paths = [os.path.join(mask_dir, os.path.splitext(f)[0] + ".png") for f in imgs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def logits_to_pred(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to a binary mask."""
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob > 0.5).astype(np.uint8)
    return pred


def grayscale_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert a grayscale array to RGB when needed."""
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(axis=2)

    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr


def denormalize(img: np.ndarray) -> np.ndarray:
    """[0,1] float32 → [0,255] uint8 RGB."""
    img = np.clip(img, 0.0, 1.0)
    rgb = grayscale_to_rgb(img)
    return (rgb * 255).astype(np.uint8)


def create_overlay(img_rgb: np.ndarray,
                   mask: np.ndarray,
                   color=(0, 255, 0),
                   alpha: float = 0.4) -> np.ndarray:
    """Overlay a binary mask on top of an RGB image."""
    h, w = img_rgb.shape[:2]

    mask_2d = mask.squeeze()
    mask_bool = mask_2d > 0

    color_arr = np.array(color, dtype=np.float32)
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)
    colored_mask[mask_bool] = color_arr

    img_f = img_rgb.astype(np.float32)
    blended = img_f * (1.0 - alpha) + colored_mask * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def save_comparison_panel(img: np.ndarray,
                          gt_mask: np.ndarray,
                          pred_mask: np.ndarray,
                          sample_idx: int,
                          out_dir: str) -> str:
    """Save a three-panel comparison image and return its output path."""
    img_disp   = denormalize(img)
    gt_disp    = (grayscale_to_rgb(gt_mask.squeeze())  * 255).astype(np.uint8)
    pred_disp  = (grayscale_to_rgb(pred_mask.squeeze()) * 255).astype(np.uint8)

    combined = np.hstack([img_disp, gt_disp, pred_disp])

    out_path = os.path.join(out_dir, f"sample_{sample_idx:03d}.png")
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return out_path


def save_overlay_image(img: np.ndarray,
                       mask: np.ndarray,
                       sample_idx: int,
                       out_dir: str,
                       label: str = "pred",
                       color: tuple = (0, 255, 0),
                       alpha: float = 0.4) -> str:
    """Save a masked overlay image and return its output path."""
    img_denorm = (img * 255).astype(np.uint8)
    blended = create_overlay(img_denorm, mask, color=color, alpha=alpha)

    out_path = os.path.join(out_dir, f"overlay_{sample_idx:03d}_{label}.png")
    cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return out_path


def compute_metrics(ds: tf.data.Dataset, model: tf.keras.Model):
    """Compute mean and std for Dice and IoU over a dataset."""
    all_dice, all_iou = [], []

    for img_batch, mask_batch in ds:
        logits  = model(img_batch, training=False)
        y_true   = mask_batch.numpy().ravel()
        y_pred   = logits_to_pred(logits.numpy().ravel())

        inter    = np.sum(y_true * y_pred)
        denom_d  = np.sum(y_true) + np.sum(y_pred)
        union    = denom_d - inter

        dice     = (2.0 * inter + 1e-6) / (denom_d + 1e-6)
        iou      = (inter + 1e-6) / (union + 1e-6)

        all_dice.append(dice)
        all_iou.append(iou)

    return {
        "dice_mean": np.mean(all_dice),
        "dice_std":  np.std(all_dice),
        "iou_mean":  np.mean(all_iou),
        "iou_std":   np.std(all_iou),
        "n":         len(all_dice),
    }


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(KERAS_MODEL):
        raise FileNotFoundError(
            f"Model không tìm thấy: {KERAS_MODEL}\n"
            "Chạy training script trước hoặc chỉ định đường dẫn đúng."
        )
    model = tf.keras.models.load_model(KERAS_MODEL, custom_objects=CUSTOM_OBJECTS)
    print(f"✓ Model loaded: {KERAS_MODEL}")

    infer_ds, infer_tag = None, ""

    if os.path.isdir(os.path.join(DATASET_SPLIT_DIR, "images", "test")):
        infer_ds  = make_dataset("test")
        infer_tag = "test"
    elif os.path.isdir(os.path.join(DATASET_SPLIT_DIR, "images", "val")):
        infer_ds  = make_dataset("val")
        infer_tag = "val"
    else:
        raise FileNotFoundError("Không tìm thấy test hoặc val split trong dataset.")

    print(f"✓ Dataset: {infer_tag}")

    metrics = compute_metrics(infer_ds, model)

    print(f"\n── {infer_tag.upper()} Set Metrics (n={metrics['n']}) ──")
    print(f"  Dice Score : {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"  IoU Score  : {metrics['iou_mean']:.4f}  ± {metrics['iou_std']:.4f}")
    print(f"── Visualization output: {OUTPUT_DIR}/ ──\n")

    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(10, 3 * NUM_SAMPLES), squeeze=False)
    fig.suptitle("Face Segmentation — U-Net Inference", fontsize=14, y=1.01)

    saved_count = 0

    for img_batch, mask_batch in infer_ds:
        if saved_count >= NUM_SAMPLES:
            break

        logits = model(img_batch, training=False)

        for i in range(img_batch.shape[0]):
            if saved_count >= NUM_SAMPLES:
                break

            idx = saved_count

            img   = img_batch[i].numpy()
            mask  = mask_batch[i].numpy()
            logit = logits[i].numpy()
            pred  = logits_to_pred(logit)

            panel_path = save_comparison_panel(img, mask, pred, idx, OUTPUT_DIR)

            pred_overlay = save_overlay_image(
                img, pred, idx, OUTPUT_DIR,
                label="pred", color=(0, 255, 0), alpha=0.4
            )
            gt_overlay = save_overlay_image(
                img, mask, idx, OUTPUT_DIR,
                label="gt", color=(255, 0, 0), alpha=0.4
            )

            img_disp   = denormalize(img)
            gt_disp    = (grayscale_to_rgb(mask.squeeze())  * 255).astype(np.uint8)
            pred_disp  = (grayscale_to_rgb(pred.squeeze()) * 255).astype(np.uint8)

            for col, (arr, title) in enumerate(zip(
                [img_disp, gt_disp, pred_disp],
                ["Input Image", "Ground Truth", "Predicted Mask"]
            )):
                axes[idx][col].imshow(arr)
                axes[idx][col].set_title(title, fontsize=9)
                axes[idx][col].axis("off")

            print(
                f"  [{idx+1:02d}/{NUM_SAMPLES}]  "
                f"sample={os.path.basename(panel_path)}  "
                f"overlay_pred={os.path.basename(pred_overlay)}  "
                f"overlay_gt={os.path.basename(gt_overlay)}"
            )

            saved_count += 1

    plt.tight_layout()
    overview_path = os.path.join(OUTPUT_DIR, "segmentation_comparison.png")
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Overview saved   : {overview_path}")
    print(f"✓ Total samples   : {saved_count}/{NUM_SAMPLES}  →  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
