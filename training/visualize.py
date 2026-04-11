"""
03_visualize.py
===============
Face Segmentation — Inference & Visualization Script
---------------------------------------------------
Chạy inference trên mô hình U-Net đã huấn luyện (logits output).

Pipeline:
  1. Load .keras model với custom_objects
  2. Build tf.data pipeline (no cache) cho val/test split
  3. Inference: logits → sigmoid → binary mask (threshold 0.5)
  4. Tạo ảnh so sánh 3-panel: input | GT mask | predicted mask
  5. Tạo overlay: predicted mask phủ lên input image (green, alpha=0.4)
  6. Lưu: individual PNGs + overview + overlay PNGs
  7. Tính Dice / IoU trên toàn bộ split

 Model output là RAW LOGITS (activation=None).
     Sigmoid CHỈ áp trong inference — không thay đổi model.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# ── Hằng số ────────────────────────────────────────────────────────────────────────
DATA_DIR      = "dataset_split"
MODEL_PATH    = "unet_face_segmentation.keras"   # hoặc dùng checkpoint
IMG_SIZE     = (256, 256)
INPUT_SHAPE  = (256, 256, 3)
NUM_SAMPLES  = 5     # số mẫu hiển thị (lab yêu cầu ≥ 5)
OUTPUT_DIR   = "outputs"
SEED         = 42

# ── 1. Custom objects (phải khớp với training script) ───────────────────────

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


@tf.keras.utils.register_keras_serializable()
def dice_metric(y_true, y_pred, smooth: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom    = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


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


# ── 2. Dataset pipeline ───────────────────────────────────────────────────────

def load_pair(img_path, mask_path):
    """
    Load and preprocess a single image-mask pair.

    Image : decode JPEG → resize(256,256) → cast(float32) / 255.0  → [0,1]
    Mask  : decode PNG → resize(256,256,nearest) → /255.0 → threshold → {0,1}
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape(INPUT_SHAPE)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)   # binary {0.0, 1.0}
    mask.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))
    return img, mask


def make_dataset(split: str) -> tf.data.Dataset:
    """
    Build tf.data pipeline cho split.

    NO .cache() — chỉ đọc file từ disk khi cần, không giữ toàn bộ dataset
    trong RAM. Pipeline:
      from_tensor_slices(paths) → map(load_pair) → batch → prefetch
    """
    img_dir = os.path.join(DATA_DIR, "images", split)
    mask_dir = os.path.join(DATA_DIR, "masks",  split)

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
    ds = ds.batch(1)          # batch=1: từng ảnh, kiểm soát memory tốt nhất
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── 3. Inference helpers ─────────────────────────────────────────────────────

def logits_to_pred(logits: np.ndarray) -> np.ndarray:
    """
    Logits → probability (sigmoid) → binary mask {0, 1}.

    Sigmoid chỉ áp trong inference — model giữ nguyên activation=None.
    """
    prob = 1.0 / (1.0 + np.exp(-logits))       # sigmoid
    pred = (prob > 0.5).astype(np.uint8)        # binary {0, 1}
    return pred


def grayscale_to_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Convert grayscale array (H, W) or (H, W, 1) → RGB array (H, W, 3).

    CRITICAL: matplotlib/channels hiển thị đúng khi all panels cùng shape.
    Tránh bug "Image" of size (256, 256) cannot have width<height và
    ảnh grayscale bị hiển thị sai khi hstack với RGB.
    """
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(axis=2)   # (H, W, 1) → (H, W)

    if arr.ndim == 2:
        # (H, W) → (H, W, 3)  — stack 3 lần để R=G=B (grayscale đúng)
        return np.stack([arr, arr, arr], axis=-1)
    return arr   # đã là (H, W, 3) rồi


def denormalize(img: np.ndarray) -> np.ndarray:
    """[0,1] float32 → [0,255] uint8 RGB."""
    img = np.clip(img, 0.0, 1.0)
    rgb = grayscale_to_rgb(img)
    return (rgb * 255).astype(np.uint8)


def create_overlay(img_rgb: np.ndarray,
                   mask: np.ndarray,
                   color=(0, 255, 0),
                   alpha: float = 0.4) -> np.ndarray:
    """
    Phủ binary mask lên input image với alpha blending.

    Args:
        img_rgb : input image (H, W, 3) uint8 — đã denormalize
        mask    : binary mask {0, 1}  (H, W) hoặc (H, W, 1)
        color   : BGR color cho mask overlay
                  (0,255,0)=green → predicted, (255,0,0)=red → GT
        alpha   : độ trong suốt của mask (0=không thấy, 1=che hoàn toàn)

    Returns:
        Blended RGB image (H, W, 3) uint8.
    """
    h, w = img_rgb.shape[:2]

    # Đảm bảo mask là (H, W) binary
    mask_2d = mask.squeeze()
    mask_bool = mask_2d > 0   # {True, False}

    # Tạo colored mask: những pixel mask=1 có màu, mask=0 trong suốt
    color_arr = np.array(color, dtype=np.float32)   # (3,)
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)
    colored_mask[mask_bool] = color_arr               # chỉ gán màu tại pixel mask=1

    # Alpha blend: result = img*(1-alpha) + mask_color*alpha
    img_f = img_rgb.astype(np.float32)
    blended = img_f * (1.0 - alpha) + colored_mask * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# ── 4. Visualization ────────────────────────────────────────────────────────

def save_comparison_panel(img: np.ndarray,
                          gt_mask: np.ndarray,
                          pred_mask: np.ndarray,
                          sample_idx: int,
                          out_dir: str) -> str:
    """
    Tạo ảnh 3-panel: [input image | GT mask | predicted mask].
    Lưu thành sample_XXX.png trong out_dir.
    Trả về đường dẫn file đã lưu.
    """
    # Denormalize: float[0,1] → uint8[0,255], grayscale → RGB
    img_disp   = denormalize(img)                             # (256, 256, 3)
    gt_disp    = (grayscale_to_rgb(gt_mask.squeeze())  * 255).astype(np.uint8)
    pred_disp  = (grayscale_to_rgb(pred_mask.squeeze()) * 255).astype(np.uint8)

    # Ghép 3 panel ngang: input | GT | predicted
    combined = np.hstack([img_disp, gt_disp, pred_disp])    # (256, 768, 3)

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
    """
    Tạo overlay: input image phủ bởi mask.
    Lưu thành overlay_XXX_label.png trong out_dir.

    Args:
        label: "pred" (green) hoặc "gt" (red)
        color: BGR — (0,255,0)=green cho predicted, (255,0,0)=red cho GT
    """
    # Input image: denormalize trước khi overlay
    img_denorm = img * 255.0 if img.max() <= 1.0 else img   # float [0,1] → float [0,255]
    img_denorm = img_denorm.astype(np.uint8)
    if img_denorm.ndim == 3 and img_denorm.shape[2] == 1:
        img_denorm = np.concatenate([img_denorm] * 3, axis=-1)

    blended = create_overlay(img_denorm, mask, color=color, alpha=alpha)

    out_path = os.path.join(out_dir, f"overlay_{sample_idx:03d}_{label}.png")
    cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return out_path


def compute_metrics(ds: tf.data.Dataset, model: tf.keras.Model):
    """
    Tính Dice và IoU trên toàn bộ dataset split (batch-by-batch).
    Trả về dict với mean ± std.
    """
    all_dice, all_iou = [], []

    for img_batch, mask_batch in ds:
        logits  = model(img_batch, training=False)    # (B, 256, 256, 1) logits
        y_true   = mask_batch.numpy().ravel()           # {0, 1}
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


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model không tìm thấy: {MODEL_PATH}\n"
            "Chạy training script trước hoặc chỉ định đường dẫn đúng."
        )
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    print(f"✓ Model loaded: {MODEL_PATH}")

    # ── Build dataset: ưu tiên test > val ────────────────────────────────────
    infer_ds, infer_tag = None, ""

    if os.path.isdir(os.path.join(DATA_DIR, "images", "test")):
        infer_ds  = make_dataset("test")
        infer_tag = "test"
    elif os.path.isdir(os.path.join(DATA_DIR, "images", "val")):
        infer_ds  = make_dataset("val")
        infer_tag = "val"
    else:
        raise FileNotFoundError("Không tìm thấy test hoặc val split trong dataset.")

    print(f"✓ Dataset: {infer_tag}")

    # ── Metrics trên toàn bộ split ─────────────────────────────────────────────
    metrics = compute_metrics(infer_ds, model)

    print(f"\n── {infer_tag.upper()} Set Metrics (n={metrics['n']}) ──")
    print(f"  Dice Score : {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"  IoU Score  : {metrics['iou_mean']:.4f}  ± {metrics['iou_std']:.4f}")
    print(f"── Visualization output: {OUTPUT_DIR}/ ──\n")

    # ── Visualization: NUM_SAMPLES mẫu đầu tiên ────────────────────────────────
    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(10, 3 * NUM_SAMPLES), squeeze=False)
    fig.suptitle("Face Segmentation — U-Net Inference", fontsize=14, y=1.01)

    saved_count = 0

    for img_batch, mask_batch in infer_ds:
        if saved_count >= NUM_SAMPLES:
            break

        logits = model(img_batch, training=False)   # (B, 256, 256, 1) logits

        for i in range(img_batch.shape[0]):
            if saved_count >= NUM_SAMPLES:
                break

            idx = saved_count

            # Squeeze batch dim → (H, W, C)
            img   = img_batch[i].numpy()                     # (256, 256, 3) float [0,1]
            mask  = mask_batch[i].numpy()                   # (256, 256, 1) float {0,1}
            logit = logits[i].numpy()                        # (256, 256, 1) raw logits
            pred  = logits_to_pred(logit)                    # (256, 256, 1) uint8 {0,1}

            # --- Save individual PNGs ---
            panel_path = save_comparison_panel(img, mask, pred, idx, OUTPUT_DIR)

            # --- Save overlay images ---
            pred_overlay = save_overlay_image(
                img, pred, idx, OUTPUT_DIR,
                label="pred", color=(0, 255, 0), alpha=0.4
            )
            gt_overlay = save_overlay_image(
                img, mask, idx, OUTPUT_DIR,
                label="gt", color=(255, 0, 0), alpha=0.4
            )

            # --- Overview matplotlib figure ---
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
