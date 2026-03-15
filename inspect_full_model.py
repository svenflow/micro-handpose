#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Inspect the FULL hand landmark tflite model to understand its weight storage."""

import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
FULL_PATH = SCRIPT_DIR / "hand_landmark_full.tflite"
LITE_PATH = SCRIPT_DIR / "hand_landmark.tflite"


def inspect_model(path, label):
    import tensorflow as tf
    print(f"\n{'='*80}")
    print(f"Inspecting {label}: {path.name} ({path.stat().st_size:,} bytes)")
    print(f"{'='*80}")

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    print(f"\nTotal tensors: {len(details)}")

    # Categorize tensors
    categories = {}
    for d in details:
        name = d["name"]
        shape = list(d["shape"])
        dtype = str(d["dtype"])
        n_params = int(np.prod(shape)) if len(shape) > 0 else 0

        # Check if tensor has data
        try:
            tensor = interpreter.get_tensor(d["index"])
            has_data = True
        except ValueError:
            has_data = False

        # Categorize
        if "Kernel" in name:
            cat = "Kernel"
        elif "Bias" in name:
            cat = "Bias"
        elif "_dequantize" in name:
            cat = "Dequantize"
        elif name.startswith("serving_default"):
            cat = "Input"
        elif name.startswith("StatefulPartitioned"):
            cat = "Output"
        else:
            cat = "Other"

        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "name": name, "shape": shape, "dtype": dtype,
            "n_params": n_params, "has_data": has_data
        })

    for cat, tensors in sorted(categories.items()):
        total_p = sum(t["n_params"] for t in tensors)
        data_count = sum(1 for t in tensors if t["has_data"])
        print(f"\n  {cat}: {len(tensors)} tensors ({total_p:,} params), {data_count} with data")

    # Show all tensors with their properties
    print(f"\n{'='*80}")
    print(f"ALL TENSORS (first 50)")
    print(f"{'='*80}")
    for i, d in enumerate(details[:50]):
        name = d["name"]
        shape = list(d["shape"])
        dtype = str(d["dtype"])
        quant = d.get("quantization", None)
        quant_params = d.get("quantization_parameters", {})

        try:
            tensor = interpreter.get_tensor(d["index"])
            data_info = f"dtype={tensor.dtype} min={tensor.min():.4f} max={tensor.max():.4f}"
        except ValueError:
            data_info = "NO DATA"

        print(f"  {i:>3d}. [{d['index']:>3d}] {name:<55s} {str(shape):<30s} {data_info}")

    if len(details) > 50:
        print(f"\n  ... and {len(details) - 50} more tensors")

    # Print ALL tensors that have "Kernel" or "Bias" or "weight" in name, or are f16
    print(f"\n{'='*80}")
    print(f"WEIGHT-LIKE TENSORS")
    print(f"{'='*80}")
    weight_count = 0
    for d in details:
        name = d["name"]
        shape = list(d["shape"])
        n_params = int(np.prod(shape)) if len(shape) > 0 else 0

        is_weight = ("Kernel" in name or "Bias" in name or "weight" in name.lower()
                     or "kernel" in name.lower() or "bias" in name.lower())
        if not is_weight and n_params < 10:
            continue
        if "_dequantize" in name:
            continue

        try:
            tensor = interpreter.get_tensor(d["index"])
            data_info = f"dtype={tensor.dtype}"
        except ValueError:
            data_info = "NO DATA"

        if is_weight or (n_params > 100 and data_info != "NO DATA"):
            print(f"  [{d['index']:>3d}] {name:<60s} {str(shape):<30s} {n_params:>10,} {data_info}")
            weight_count += 1

    print(f"\nTotal weight-like tensors: {weight_count}")

    # Check if the model uses float16 storage
    print(f"\n{'='*80}")
    print(f"TENSOR DTYPES")
    print(f"{'='*80}")
    dtype_counts = {}
    for d in details:
        dtype = str(d["dtype"])
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count}")

    # Look for dequantize patterns - the FULL model might store f16 weights
    # with dequantize ops
    print(f"\n{'='*80}")
    print(f"DEQUANTIZE TENSORS")
    print(f"{'='*80}")
    dequant_count = 0
    for d in details:
        name = d["name"]
        if "_dequantize" not in name:
            continue
        shape = list(d["shape"])
        n_params = int(np.prod(shape))
        try:
            tensor = interpreter.get_tensor(d["index"])
            data_info = f"dtype={tensor.dtype} min={tensor.min():.4f} max={tensor.max():.4f}"
        except ValueError:
            data_info = "NO DATA"
        if n_params > 10:
            print(f"  [{d['index']:>3d}] {name:<60s} {str(shape):<30s} {n_params:>10,} {data_info}")
            dequant_count += 1
    print(f"\nTotal dequantize tensors with >10 params: {dequant_count}")


inspect_model(LITE_PATH, "LITE")
inspect_model(FULL_PATH, "FULL")
