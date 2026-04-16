"""Convert the Keras model to a fully quantized INT8 TFLite model."""

import os
import random
import cv2
import numpy as np
import tensorflow as tf

from training.models.config import KERAS_MODEL, TFLITE_INT8, REP_IMAGES_DIR

OUT_TFLITE  = TFLITE_INT8
REP_DIR     = REP_IMAGES_DIR
REP_SAMPLES = 500

IMG_SIZE = (256, 256)


def representative_data_gen():
    """Yield representative samples for PTQ calibration."""
    paths = [
        os.path.join(REP_DIR, f)
        for f in os.listdir(REP_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not paths:
        raise ValueError(f"Representative dataset directory is empty: {REP_DIR}")

    random.shuffle(paths)

    for p in paths[:REP_SAMPLES]:
        img = cv2.imread(p)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))

        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        yield [x]


def validate_full_int8(tflite_path: str) -> bool:
    """
    Check that the converted model uses only quantized internal tensors.
    """
    print("\n─── Post-conversion validation ────────────────────────")
    all_int8 = True

    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    details = interpreter.get_tensor_details()

    for tensor in details:
        name  = tensor.get("name", "?")
        dtype = tensor["dtype"]
        qp    = tensor.get("quantization_parameters", {})
        scales      = qp.get("scales", [])
        zero_points = qp.get("zero_points", [])

        is_int8  = dtype == tf.int8.as_numpy_dtype
        is_int32 = dtype == tf.int32.as_numpy_dtype
        ok       = is_int8 or is_int32

        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         dtype={tf.constant((), dtype=dtype).dtype.name}  "
              f"scales={len(scales)}  zero_points={len(zero_points)}")

        if not ok:
            all_int8 = False

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    print(f"\n  INPUT  dtype={in_det['dtype']}   "
          f"shape={in_det['shape']}   quant={in_det['quantization']}")
    print(f"  OUTPUT dtype={out_det['dtype']}   "
          f"shape={out_det['shape']}   quant={out_det['quantization']}")

    if all_int8:
        print("\n  PASS — all tensors are int8 / int32 (no float fallback).\n")
    else:
        print("\n  FAIL — non-int8 tensors detected. Review model architecture.\n")

    return all_int8


def main():
    model = tf.keras.models.load_model(KERAS_MODEL, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.experimental_enable_resource_variables = True
    converter._experimental_lower_tensor_list_ops = False

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Force strict INT8-only ops so conversion fails instead of falling back to float.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_int8 = converter.convert()

    with open(OUT_TFLITE, "wb") as f:
        f.write(tflite_int8)

    size_mb = len(tflite_int8) / (1024 * 1024)
    print(f"Saved : {OUT_TFLITE}  ({size_mb:.2f} MB)")

    validate_full_int8(OUT_TFLITE)


if __name__ == "__main__":
    main()
