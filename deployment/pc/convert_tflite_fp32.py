"""Convert the Keras model to a plain FP32 TFLite model."""

import os
import tensorflow as tf

from training.models.config import KERAS_MODEL, TFLITE_FP32


def validate_dtypes(tflite_path: str):
    """Print input and output dtypes of the converted TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    print(f"  Input  dtype : {in_det['dtype']}")
    print(f"  Output dtype : {out_det['dtype']}")

    assert in_det["dtype"]  == tf.float32, "Input dtype is not float32!"
    assert out_det["dtype"] == tf.float32, "Output dtype is not float32!"
    print("  PASS — both tensors are float32.")


def main():
    model = tf.keras.models.load_model(KERAS_MODEL, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(TFLITE_FP32, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Saved : {TFLITE_FP32}  ({size_mb:.2f} MB)")

    validate_dtypes(TFLITE_FP32)


if __name__ == "__main__":
    main()
