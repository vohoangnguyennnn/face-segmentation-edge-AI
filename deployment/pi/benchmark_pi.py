"""Benchmark INT8 TFLite inference on Raspberry Pi."""

import time
import numpy as np

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

MODEL_PATH  = TFLITE_INT8
WARMUP      = 30
N_RUNS      = 300
THREADS     = [1, 2, 4]


def make_realistic_input(in_det):
    """Generate a plausible uint8 input in the model's quantized range."""
    dtype  = in_det["dtype"]
    shape  = tuple(in_det["shape"])
    q      = in_det.get("quantization", (1.0, 0))

    if len(q) >= 2:
        scale, zero = q[0], q[1]
    else:
        scale, zero = 1.0, 0

    if dtype == np.uint8:
        x = np.full(shape, zero, dtype=np.uint8)
        x = (x.astype(np.int32) + np.random.randint(-20, 20, size=shape)).astype(np.uint8)
        x = np.clip(x, 0, 255)
    elif dtype == np.int8:
        x = np.full(shape, zero, dtype=np.int8)
        x = (x.astype(np.int32) + np.random.randint(-20, 20, size=shape)).astype(np.int8)
        x = np.clip(x, -128, 127)
    else:
        x = np.zeros(shape, dtype=dtype)

    return x


def print_model_info(interpreter):
    """Print input and output tensor details."""
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    q_in  = in_det.get("quantization", (1.0, 0))
    q_out = out_det.get("quantization", (1.0, 0))

    print(f"\n{'─' * 60}")
    print(f"  INPUT  dtype={in_det['dtype']}  shape={in_det['shape']}  "
          f"scale={q_in[0]}  zero_point={q_in[1]}")
    print(f"  OUTPUT dtype={out_det['dtype']}  shape={out_det['shape']}  "
          f"scale={q_out[0]}  zero_point={q_out[1]}")
    print(f"{'─' * 60}")


def benchmark(interpreter, x, n_runs=N_RUNS, warmup=WARMUP):
    """Run warmup + timed inference loop, return latency array in ms."""
    for _ in range(warmup):
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
        interpreter.invoke()

    ts = []
    in_index = interpreter.get_input_details()[0]["index"]
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(in_index, x)
        interpreter.invoke()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000.0)

    return np.array(ts)


def main():
    print(f"TFLite INT8 Benchmark  |  Model: {MODEL_PATH}")
    print(f"Warmup runs: {WARMUP}  |  Benchmark runs: {N_RUNS}")

    results = {}

    for n_threads in THREADS:
        print(f"\n{'═' * 60}")
        print(f"  Threads: {n_threads}")
        print(f"{'═' * 60}")

        interpreter = Interpreter(model_path=MODEL_PATH, num_threads=n_threads)
        interpreter.allocate_tensors()

        if n_threads == THREADS[0]:
            print_model_info(interpreter)

        x = make_realistic_input(interpreter.get_input_details()[0])

        ts = benchmark(interpreter, x)

        mean_ms  = float(ts.mean())
        p50_ms   = float(np.percentile(ts, 50))
        p90_ms   = float(np.percentile(ts, 90))
        p99_ms   = float(np.percentile(ts, 99))
        fps      = 1000.0 / mean_ms

        results[n_threads] = dict(mean=mean_ms, p50=p50_ms, p90=p90_ms, p99=p99_ms, fps=fps)

        print(f"  Mean  : {mean_ms:8.2f} ms")
        print(f"  P50   : {p50_ms:8.2f} ms")
        print(f"  P90   : {p90_ms:8.2f} ms")
        print(f"  P99   : {p99_ms:8.2f} ms")
        print(f"  FPS   : {fps:8.2f}")

    print(f"\n{'═' * 60}")
    print(f"  SUMMARY  (best FPS in bold)")
    print(f"{'─' * 60}")
    best_fps = max(r["fps"] for r in results.values())
    for n_threads, r in results.items():
        marker = "◀ BEST" if r["fps"] == best_fps else ""
        print(f"  {n_threads} thread(s)  "
              f"Mean={r['mean']:6.2f}ms  "
              f"P90={r['p90']:6.2f}ms  "
              f"FPS={r['fps']:6.2f}  {marker}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
