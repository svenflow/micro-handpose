#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Check if DW conv biases for 128ch and 256ch are truly zero."""

import numpy as np
import tensorflow as tf
from pathlib import Path

TFLITE_PATH = Path(__file__).parent / "palm_detection.tflite"

def main():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    # Find ALL bias-like tensors (1D tensors that could be biases)
    print("=== All 1D tensors (potential biases) ===")
    for d in details:
        shape = tuple(d["shape"])
        if len(shape) == 1 and shape[0] > 1:
            data = interpreter.get_tensor(d["index"])
            is_zero = np.all(data == 0)
            name = d["name"]
            print(f"  [{d['index']:3d}] shape={shape} dtype={d['dtype'].__name__} all_zero={is_zero} name={name}")
            if not is_zero:
                print(f"         min={data.min():.6f} max={data.max():.6f} mean={data.mean():.6f}")

    # Also check specific tensor indices mentioned
    print("\n=== Specific tensor checks (indices 32, 43) ===")
    for idx in [32, 43]:
        d = details[idx]
        data = interpreter.get_tensor(idx)
        print(f"  [{idx}] shape={tuple(d['shape'])} dtype={d['dtype'].__name__} name={d['name']}")
        print(f"       all_zero={np.all(data == 0)} min={data.min():.6f} max={data.max():.6f}")

    # Find all DW conv weight tensors (shape [1, H, W, C]) and their associated biases
    print("\n=== DW conv weights and biases ===")
    ops = interpreter._get_ops_details()
    for i, op in enumerate(ops):
        if op['op_name'] == 'DEPTHWISE_CONV_2D':
            inputs = list(op['inputs'])
            # inputs: [input_tensor, weights, bias]
            if len(inputs) >= 3:
                weight_idx = inputs[1]
                bias_idx = inputs[2]
                w_detail = None
                b_detail = None
                for d in details:
                    if d['index'] == weight_idx:
                        w_detail = d
                    if d['index'] == bias_idx:
                        b_detail = d
                if w_detail and b_detail:
                    bias_data = interpreter.get_tensor(bias_idx)
                    is_zero = np.all(bias_data == 0)
                    w_shape = tuple(w_detail['shape'])
                    b_shape = tuple(b_detail['shape'])
                    print(f"  Op {i:3d}: DW weight={w_shape} bias={b_shape} bias_zero={is_zero} bias_name={b_detail['name']}")
                    if not is_zero:
                        print(f"           bias min={bias_data.min():.6f} max={bias_data.max():.6f}")

if __name__ == "__main__":
    main()
