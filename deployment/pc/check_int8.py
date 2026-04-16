"""Validate whether a TFLite model is fully INT8 quantized internally."""
import os
import numpy as np
import tensorflow as tf

# ─── Guard: must run from repo root ────────────────────────────────────────
_ROOT_GUARD = "requirements.txt"
if not os.path.isfile(_ROOT_GUARD):
    raise RuntimeError(
        f"'{_ROOT_GUARD}' not found — run this script from the repo root.\n"
        f"Current directory: {os.getcwd()}"
    )

from training.models.config import TFLITE_INT8

TFLITE_PATH = TFLITE_INT8

def _fmt_tensor(name: str, dtype: np.dtype, q_params: dict | None) -> str:
    """Return a one-line description of a tensor, including quantization info."""
    dtype_name = {np.int8: "int8", np.int32: "int32",
                  np.float32: "float32", np.uint8: "uint8"}.get(dtype, str(dtype))
    parts = [f"[{dtype_name:8s}]  {name}"]
    if q_params:
        parts.append(f"  scale={q_params.get('scale', 'N/A')}  "
                     f"zero_point={q_params.get('zero_point', 'N/A')}")
    return "  ".join(parts)

def check_full_int8(tflite_path: str, verbose: bool = True) -> bool:
    """
    Check whether `tflite_path` is a fully INT8-quantized TFLite model.

    Parameters
    ----------
    tflite_path : str
        Path to the .tflite model file.
    verbose : bool
        When True (default) prints detailed section output.
        When False prints only the final verdict line.

    Returns
    -------
    bool
        True  → model IS fully INT8 quantized.
        False → model is NOT fully INT8 quantized.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    tensors = interpreter.get_tensor_details()

    WIDTH     = 62

    def _print(msg: str = ""):
        if verbose:
            print(msg)

    reasons_not_full_int8: list[str] = []

    _print(f"\n{'═' * WIDTH}")
    _print("  INPUT / OUTPUT")
    _print(f"{'═' * WIDTH}")

    for det, label in [(in_det, "INPUT"), (out_det, "OUTPUT")]:
        dtype = det["dtype"]
        shape = det["shape"]
        q     = det.get("quantization", None)

        dtype_ok = dtype in (np.int8, np.uint8)
        if not dtype_ok:
            reasons_not_full_int8.append(
                f"{label} dtype is {dtype}, expected int8 or uint8")

        scale_ok = True
        io_type  = "uint8" if dtype == np.uint8 else "int8"
        if q is not None and len(q) >= 2:
            scale = q[0] if isinstance(q, (list, tuple)) else q
            zp    = q[1] if isinstance(q, (list, tuple)) and len(q) > 1 else 0
            scale_ok = (scale != 0.0)

            _print(f"\n  {label}:")
            _print(f"    dtype : {dtype}  ({io_type} I/O — RPi compatible)")
            _print(f"    shape : {shape}")
            _print(f"    scale       : {scale}  "
                   f"{'✅' if scale_ok else '❌ (must be non-zero)'}")
            _print(f"    zero_point  : {zp}")
        else:
            _print(f"\n  {label}:")
            _print(f"    dtype : {dtype}  ({io_type} I/O — RPi compatible)")
            _print(f"    shape : {shape}")
            _print(f"    ⚠️  quantization parameters MISSING")
            reasons_not_full_int8.append(f"{label} has no quantization params")

        _print(f"    dtype valid (int8/uint8): {'✅ PASS' if dtype_ok else '❌ FAIL'}")

    _print(f"\n{'═' * WIDTH}")
    _print("  TENSOR SUMMARY")
    _print(f"{'═' * WIDTH}")

    total_count = len(tensors)
    int8_count  = 0
    int32_count = 0
    fp32_count  = 0
    other_count = 0

    float32_tensors: list[str] = []

    for t in tensors:
        dtype = t["dtype"]
        name  = t["name"]
        q     = t.get("quantization", None)

        if   dtype == np.int8:   int8_count  += 1
        elif dtype == np.int32:  int32_count += 1
        elif dtype == np.float32:
            fp32_count += 1
            float32_tensors.append(name)
        else:
            other_count += 1

    _print(f"\n  Total tensors : {total_count}")
    _print(f"  ├── int8  : {int8_count}  "
           f"({'100%' if total_count==0 else f'{int8_count*100//total_count}%'})")
    _print(f"  ├── int32 : {int32_count}  "
           f"({'0%'   if total_count==0 else f'{int32_count*100//total_count}%'})")
    _print(f"  ├── float32 : {fp32_count}  "
           f"{'❌ HYBRID / FAKE INT8' if fp32_count > 0 else '✅'}")
    _print(f"  └── other : {other_count}")

    pct_quantized = 0
    if total_count > 0:
        pct_quantized = (int8_count + int32_count) * 100 // total_count
    _print(f"\n  Quantized tensors : {pct_quantized}%")

    if verbose:
        _print(f"\n{'─' * WIDTH}")
        _print("  ALL TENSORS")
        _print(f"{'─' * WIDTH}")
        for t in tensors:
            dtype = t["dtype"]
            name  = t["name"]
            q     = t.get("quantization", None)
            q_info = {}
            if q is not None and len(q) >= 2:
                q_info["scale"]       = q[0]
                q_info["zero_point"]  = q[1]

            if dtype == np.float32:
                mark = "🚨"
                reasons_not_full_int8.append(f"FLOAT32 tensor: {name}")
            elif dtype in (np.int8, np.int32, np.uint8):
                mark = "✅"
            else:
                mark = "⚠️"

            _print(f"  {mark}  {_fmt_tensor(name, dtype, q_info)}")

        if fp32_count > 0:
            _print(f"\n  🚨 FLOAT32 OPERATIONS DETECTED ({fp32_count}):")
            for fn in float32_tensors:
                _print(f"     🚨 FLOAT OP DETECTED: {fn}")

    _print(f"\n{'═' * WIDTH}")
    _print("  VALIDATION RESULT")
    _print(f"{'═' * WIDTH}")

    for det, label in [(in_det, "INPUT"), (out_det, "OUTPUT")]:
        q = det.get("quantization", None)
        if q is None:
            reasons_not_full_int8.append(f"{label} quantization params missing")
        elif len(q) >= 1:
            scale = q[0] if isinstance(q, (list, tuple)) else q
            if scale == 0.0:
                reasons_not_full_int8.append(
                    f"{label} quantization scale is zero → invalid INT8")

    if in_det["dtype"] not in (np.int8, np.uint8):
        reasons_not_full_int8.append(
            f"INPUT dtype is {in_det['dtype']}, expected int8 or uint8")
    if out_det["dtype"] not in (np.int8, np.uint8):
        reasons_not_full_int8.append(
            f"OUTPUT dtype is {out_det['dtype']}, expected int8 or uint8")

    if fp32_count > 0:
        reasons_not_full_int8.append(
            f"Model contains {fp32_count} float32 tensor(s) → hybrid / fake INT8")

    if not reasons_not_full_int8:
        io_label = "UINT8" if in_det["dtype"] == np.uint8 else "INT8"
        _print(f"\n  ✅ FULL INT8 — All checks passed!")
        _print(f"     ✅ Internal tensors: INT8 / INT32 (FULL INT8 INTERNAL)")
        _print(f"     ✅ I/O dtype: {io_label} (RPi compatible)")
        _print("     Model is fully INT8 quantized and safe for INT8 delegate.")
    else:
        _print("\n  ❌ NOT FULL INT8 — Issues found:")
        for i, reason in enumerate(reasons_not_full_int8, 1):
            _print(f"     {i}. ❌ {reason}")

    _print(f"\n{'═' * WIDTH}")

    is_full_int8 = len(reasons_not_full_int8) == 0

    io_label  = "UINT8" if in_det["dtype"] == np.uint8 else "INT8"
    out_label = "UINT8" if out_det["dtype"] == np.uint8 else "INT8"
    _print(f"\n  Inference smoke-test ({io_label} input → {out_label} output):")
    try:
        shape = tuple(in_det["shape"])
        scale = in_det["quantization"][0] if in_det.get("quantization") else 1.0
        zp    = in_det["quantization"][1] if (in_det.get("quantization")
                                                and len(in_det["quantization"]) > 1) else 0

        if in_det["dtype"] == np.uint8:
            dummy_u8 = np.random.randint(0, 256, size=shape).astype(np.uint8)
            _print(f"    Input  shape : {dummy_u8.shape}")
            _print(f"    Input  dtype : {dummy_u8.dtype}")
            _print(f"    Input  range : [{int(dummy_u8.min())}, {int(dummy_u8.max())}]")
            interpreter.set_tensor(in_det["index"], dummy_u8)
        else:
            float_min = (np.int8(-120) - zp) * scale
            float_max = (np.int8( 120) - zp) * scale
            dummy_f32 = np.random.uniform(float_min, float_max, size=shape).astype(np.float32)
            dummy_i8  = np.round(dummy_f32 / scale + zp).astype(np.int8)
            np.clip(dummy_i8, -128, 127, out=dummy_i8)
            _print(f"    Input  shape : {dummy_i8.shape}")
            _print(f"    Input  dtype : {dummy_i8.dtype}")
            _print(f"    Input  range : [{int(dummy_i8.min())}, {int(dummy_i8.max())}]")
            interpreter.set_tensor(in_det["index"], dummy_i8)

        interpreter.invoke()

        out = interpreter.get_tensor(out_det["index"])
        _print(f"    Output shape : {out.shape}")
        _print(f"    Output dtype : {out.dtype}")
        _print(f"    Output range : [{out.min()}, {out.max()}]")
        _print(f"    Inference ran successfully — output is {out.dtype.name}.")
    except Exception as exc:
        _print(f"      Inference failed: {exc}")

    _print()
    return is_full_int8

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate that a TFLite model is fully INT8 quantized.")
    parser.add_argument(
        "model_path", nargs="?", default=TFLITE_PATH,
        help="Path to the .tflite model file "
             f"(default: {TFLITE_PATH})")
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Print only the final verdict line.")
    args = parser.parse_args()

    is_full = check_full_int8(args.model_path, verbose=not args.quiet)
    exit(0 if is_full else 1)
