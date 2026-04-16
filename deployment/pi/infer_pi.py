"""Run single-image inference with the INT8 TFLite model on Raspberry Pi."""

import os
import time
import numpy as np
import cv2

from training.models.config import TFLITE_INT8

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

MODEL_PATH = TFLITE_INT8
IMAGE_PATH = "test.jpg"
IMG_SIZE   = (256, 256)
THRESH     = 0.5
OUT_DIR    = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "masks_pred"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)


def preprocess_quant(img_path, in_det):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    x = img.astype(np.float32) / 255.0

    q = in_det.get("quantization", (1.0, 0))
    scale, zero = (q[0], q[1]) if len(q) >= 2 else (1.0, 0)
    if scale == 0:
        raise RuntimeError("Input scale=0, model not quantized correctly")

    # np.round is required for symmetric quantization to round to nearest int.
    if in_det["dtype"] == np.uint8:
        xq = np.round(x / scale + zero).astype(np.uint8)
    else:
        xq = np.round(x / scale + zero).astype(np.int8)
        xq = np.clip(xq, -128, 127)

    return img_bgr, img, np.expand_dims(xq, axis=0)

def dequant_output(yq, out_det):
    """Dequantize INT8 output back to float32 logits."""
    q = out_det.get("quantization", (1.0, 0))
    scale, zero = (q[0], q[1]) if len(q) >= 2 else (1.0, 0)
    if scale == 0:
        return yq.astype(np.float32)
    return (yq.astype(np.float32) - zero) * scale


def sigmoid(x):
    """Numerically stable sigmoid: clip logits before exp to avoid overflow."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def main():
    itp = Interpreter(model_path=MODEL_PATH, num_threads=4)
    itp.allocate_tensors()
    in_det = itp.get_input_details()[0]
    out_det = itp.get_output_details()[0]

    raw_bgr, rgb_img, xq = preprocess_quant(IMAGE_PATH, in_det)

    for _ in range(5):
        itp.set_tensor(in_det["index"], xq)
        itp.invoke()

    t0 = time.perf_counter()
    itp.set_tensor(in_det["index"], xq)
    itp.invoke()
    t1 = time.perf_counter()

    yq = itp.get_tensor(out_det["index"])
    y = dequant_output(yq, out_det)[0]

    if y.ndim == 4:
        y = y[0]
    if y.shape[-1] == 1:
        y = y[..., 0]

    # Model outputs raw logits, so apply sigmoid before thresholding.
    print(f"    [DEBUG] logits range before sigmoid: [{y.min():.4f}, {y.max():.4f}]")
    y_prob = sigmoid(y)
    print(f"    [DEBUG] probabilities after sigmoid: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

    mask = (y_prob >= THRESH).astype(np.uint8) * 255

    base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    mask_path = os.path.join(OUT_DIR, "masks_pred", base + "_mask.png")
    cv2.imwrite(mask_path, mask)

    overlay = raw_bgr.copy()
    overlay = cv2.resize(overlay, (IMG_SIZE[1], IMG_SIZE[0]))
    color = np.zeros_like(overlay)
    color[:, :, 2] = mask
    blended = cv2.addWeighted(overlay, 0.7, color, 0.3, 0)

    out_overlay = os.path.join(OUT_DIR, "overlays", base + "_overlay.jpg")
    cv2.imwrite(out_overlay, blended)

    print("Latency (ms):", (t1 - t0) * 1000.0)
    print("Saved:", mask_path)
    print("Saved:", out_overlay)

if __name__ == "__main__":
    main()
