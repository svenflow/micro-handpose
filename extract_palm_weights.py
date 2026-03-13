#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Extract ALL weights from the MediaPipe palm detection TFLite model
and save them as a single binary file (float32) with a JSON manifest.

The palm detection model uses a BlazeNet-style architecture with:
- Regular conv2d layers
- Depthwise separable convolutions
- PReLU activations (learnable alpha per channel)
- No batch normalization (BN is folded into conv weights at export time)

Output:
  palm_detection_weights.bin  - raw float32 weight data
  palm_detection_weights.json - manifest with keys, shapes, offsets, dtypes
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "palm_detection.tflite"
OUTPUT_BIN = SCRIPT_DIR / "palm_detection_weights.bin"
OUTPUT_JSON = SCRIPT_DIR / "palm_detection_weights.json"


def main():
    print(f"Loading TFLite model: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    details = interpreter.get_tensor_details()

    # Collect all weight tensors (non-input, non-output, constant tensors)
    input_indices = {d["index"] for d in interpreter.get_input_details()}
    output_indices = {d["index"] for d in interpreter.get_output_details()}

    # Get all ops to understand which tensors are weights vs activations
    # We identify weights by: they have data (non-zero), are not inputs/outputs,
    # and their names suggest they are parameters
    weight_tensors = []
    for d in details:
        idx = d["index"]
        name = d["name"]
        shape = tuple(d["shape"])

        # Skip model inputs and outputs
        if idx in input_indices or idx in output_indices:
            continue

        # Skip scalar or empty tensors
        if len(shape) == 0 or any(s == 0 for s in shape):
            continue

        # Try to get the tensor data - weights are statically allocated
        try:
            data = interpreter.get_tensor(idx).astype(np.float32)
        except Exception:
            continue

        # Skip if all zeros (likely an activation buffer)
        if np.all(data == 0):
            continue

        weight_tensors.append({
            "index": idx,
            "name": name,
            "shape": list(shape),
            "data": data,
        })

    print(f"Found {len(weight_tensors)} weight tensors")
    print()

    # Sort by tensor index to maintain consistent order
    weight_tensors.sort(key=lambda x: x["index"])

    # Build binary buffer and manifest
    keys = []
    shapes = []
    offsets = []
    buf = bytearray()

    print(f"{'Idx':>4s}  {'Name':<60s}  {'Shape':<25s}  {'Offset':>10s}  {'Bytes':>10s}  {'Min':>10s}  {'Max':>10s}")
    print("-" * 140)

    for t in weight_tensors:
        name = t["name"]
        shape = t["shape"]
        data = t["data"]

        keys.append(name)
        shapes.append([int(s) for s in shape])
        offsets.append(len(buf))

        # Store as float32
        raw = data.astype(np.float32).tobytes()
        nbytes = len(raw)
        buf.extend(raw)

        print(f"{t['index']:>4d}  {name:<60s}  {str(shape):<25s}  {offsets[-1]:>10d}  {nbytes:>10d}  {data.min():>10.6f}  {data.max():>10.6f}")

    print("-" * 140)
    total_params = sum(np.prod(s) for s in shapes)
    print(f"Total: {len(keys)} tensors, {total_params:,} parameters, {len(buf):,} bytes")

    # Save binary
    OUTPUT_BIN.write_bytes(bytes(buf))
    print(f"\nSaved weights: {OUTPUT_BIN} ({len(buf):,} bytes)")

    # Save manifest
    manifest = {
        "keys": keys,
        "shapes": shapes,
        "offsets": offsets,
        "dtype": "float32",
    }
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest: {OUTPUT_JSON}")

    # Print summary by weight type
    print("\n=== Weight Type Summary ===")
    conv_weights = [t for t in weight_tensors if len(t["shape"]) == 4 and "Kernel" in t["name"]]
    biases = [t for t in weight_tensors if "Bias" in t["name"]]
    prelu_alphas = [t for t in weight_tensors if "alpha" in t["name"].lower() or "Alpha" in t["name"]]
    other = [t for t in weight_tensors if t not in conv_weights and t not in biases and t not in prelu_alphas]

    print(f"  Conv kernels:  {len(conv_weights)}")
    print(f"  Biases:        {len(biases)}")
    print(f"  PReLU alphas:  {len(prelu_alphas)}")
    print(f"  Other:         {len(other)}")

    if other:
        print("\n  Other tensors:")
        for t in other:
            print(f"    {t['name']} shape={t['shape']}")


if __name__ == "__main__":
    main()
