#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Check weight quantization format in TFLite model."""

import numpy as np
import tensorflow as tf
from pathlib import Path

TFLITE_PATH = Path(__file__).parent / "palm_detection.tflite"

def main():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    print("=== Weight tensor dtypes and quantization ===")
    for d in details:
        name = d["name"]
        shape = tuple(d["shape"])
        dtype = d["dtype"]
        quant = d.get("quantization", None)
        quant_params = d.get("quantization_parameters", {})

        # Only show weight-like tensors (small, non-activation)
        total = np.prod(shape) if len(shape) > 0 else 0
        if total == 0 or total > 200000:
            continue

        # Check if it's a weight (has data)
        try:
            data = interpreter.get_tensor(d["index"])
            if np.all(data == 0):
                continue
        except:
            continue

        scales = quant_params.get("scales", np.array([]))
        zero_points = quant_params.get("zero_points", np.array([]))

        is_quant = len(scales) > 0 and not (len(scales) == 1 and scales[0] == 0.0)

        if is_quant or dtype != np.float32:
            name_short = name.split("/")[-1][:50] if "/" in name else name[:50]
            print(f"  [{d['index']:3d}] dtype={dtype.__name__:>8s} shape={str(shape):>25s} quant_scales={scales[:3]}... quant_zp={zero_points[:3]}... {name_short}")

    # Also do a layer-by-layer comparison for the first DW+PW block
    print("\n=== First few weight tensors raw dtype ===")
    for d in details[:30]:
        name = d["name"]
        shape = tuple(d["shape"])
        dtype = d["dtype"]
        if len(shape) == 0 or np.prod(shape) == 0:
            continue
        try:
            data = interpreter.get_tensor(d["index"])
            print(f"  [{d['index']:3d}] dtype={dtype.__name__:>8s} stored_dtype={data.dtype}  shape={shape}  {name[:80]}")
        except:
            pass


if __name__ == "__main__":
    main()
