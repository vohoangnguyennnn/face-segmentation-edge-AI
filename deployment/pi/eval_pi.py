"""Evaluate the INT8 TFLite model on Raspberry Pi."""

import os
import numpy as np
import cv2

from training.models.config import TFLITE_INT8, DATASET_SPLIT_DIR

# ─── Guard: must run from repo root ────────────────────────────────────────
_ROOT_GUARD = "requirements.txt"
if not os.path.isfile(_ROOT_GUARD):
    raise RuntimeError(
        f"'{_ROOT_GUARD}' not found — run this script from the repo root.\n"
        f"Current directory: {os.getcwd()}"
    )

MODEL_PATH   = TFLITE_INT8
IMG_DIR      = os.path.join(DATASET_SPLIT_DIR, "images", "test")
MASK_DIR     = os.path.join(DATASET_SPLIT_DIR, "masks",  "test")

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import sys
    import tensorflow as tf
    from tensorflow.lite.python.interpreter import Interpreter as _TFLiteInterpreter
    Interpreter = _TFLiteInterpreter
    sys.stderr.write(
        "[WARN] tflite_runtime not found; fell back to tensorflow.lite.Interpreter. "
        "On Raspberry Pi, install: pip install tflite-runtime\n"
    )

IMG_SIZE     = (256, 256)
THRESH       = 0.5
NUM_SAMPLES  = 50
OUT_DIR      = "outputs"
DEBUG_SAMPLES = 3


def preprocess_quant(img_bgr, in_det):
    """Normalise image to [0,1] float, then quantise to model's input dtype."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    x = img.astype(np.float32) / 255.0

    q = in_det.get("quantization", (1.0, 0))
    scale, zero = (q[0], q[1]) if len(q) >= 2 else (1.0, 0)
    if scale == 0:
        raise RuntimeError("Input scale=0 — model not quantized correctly")

    if in_det["dtype"] == np.uint8:
        xq = np.round(x / scale + zero).astype(np.uint8)
    else:
        xq = np.round(x / scale + zero).astype(np.int8)
        xq = np.clip(xq, -128, 127)

    return np.expand_dims(xq, axis=0)


def dequantize(yq, out_det):
    """Dequantize INT8 output back to float32 logits."""
    q = out_det.get("quantization", (1.0, 0))
    scale, zero = (q[0], q[1]) if len(q) >= 2 else (1.0, 0)
    if scale == 0:
        return yq.astype(np.float32)
    return (yq.astype(np.float32) - zero) * scale


def sigmoid(x):
    """Numerically stable sigmoid — clip to avoid exp overflow."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def dice_iou(pred, gt):
    """Compute Dice and IoU from binary boolean masks."""
    pred = np.asarray(pred, dtype=bool)
    gt   = np.asarray(gt,   dtype=bool)
    inter = np.logical_and(pred, gt).sum()
    dice  = (2.0 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
    union = np.logical_or(pred, gt).sum()
    iou   = (inter + 1e-6) / (union + 1e-6)
    return float(dice), float(iou)


def print_model_info(in_det, out_det):
    """Print validation info for input and output tensors."""
    q_in  = in_det.get("quantization", (1.0, 0))
    q_out = out_det.get("quantization", (1.0, 0))
    print(f"\n{'─' * 60}")
    print(f"  INPUT  dtype={in_det['dtype']}  shape={in_det['shape']}  "
          f"scale={q_in[0]}  zero_point={q_in[1]}")
    print(f"  OUTPUT dtype={out_det['dtype']}  shape={out_det['shape']}  "
          f"scale={q_out[0]}  zero_point={q_out[1]}")
    print(f"{'─' * 60}\n")


def main():
    itp = Interpreter(model_path=MODEL_PATH, num_threads=4)
    itp.allocate_tensors()
    in_det  = itp.get_input_details()[0]
    out_det = itp.get_output_details()[0]

    print_model_info(in_det, out_det)

    all_files = [f for f in os.listdir(IMG_DIR)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    files = sorted(all_files)[:NUM_SAMPLES]

    if not files:
        raise FileNotFoundError(f"No images found in {IMG_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    dices, ious = [], []
    processed   = 0
    skipped     = 0

    for idx, f in enumerate(files):
        base = os.path.splitext(f)[0]
        img_path = os.path.join(IMG_DIR, f)
        msk_path = os.path.join(MASK_DIR, base + ".png")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️  Skipping (image not found): {img_path}")
            skipped += 1
            continue
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            print(f"  ⚠️  Skipping (mask not found): {msk_path}")
            skipped += 1
            continue

        xq = preprocess_quant(img, in_det)

        itp.set_tensor(in_det["index"], xq)
        itp.invoke()

        yq = itp.get_tensor(out_det["index"])
        y  = dequantize(yq, out_det)[0]

        if y.ndim == 4:
            y = y[0]
        if y.shape[-1] == 1:
            y = y[..., 0]

        # Model outputs raw logits, so apply sigmoid before thresholding.
        y_prob = sigmoid(y)

        if idx < DEBUG_SAMPLES:
            print(f"  [DEBUG {f}] logits  range: [{y.min():.4f}, {y.max():.4f}]  "
                  f"probs range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

        pred_mask = (y_prob >= THRESH)

        gt_mask = cv2.resize(msk, (IMG_SIZE[1], IMG_SIZE[0]),
                             interpolation=cv2.INTER_NEAREST) >= 128

        d, i = dice_iou(pred_mask, gt_mask)
        dices.append(d); ious.append(i)
        processed += 1

        vis = pred_mask.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_pred.png"), vis)

    print(f"\n{'═' * 60}")
    print(f"  Processed : {processed}  |  Skipped : {skipped}")
    print(f"  Dice mean : {np.mean(dices):.4f}  (± {np.std(dices):.4f})")
    print(f"  IoU  mean : {np.mean(ious):.4f}  (± {np.std(ious):.4f})")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
