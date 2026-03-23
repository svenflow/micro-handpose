#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Check pad values and trace full FPN operation order."""

import numpy as np
import tensorflow as tf
from pathlib import Path

TFLITE_PATH = Path(__file__).parent / "palm_detection.tflite"

def main():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    details = interpreter.get_tensor_details()

    # Read specific tensors
    print("=== Channel padding tensors ===")
    for d in details:
        name = d["name"]
        if "channel_padding" in name or "Pad/paddings" in name:
            try:
                data = interpreter.get_tensor(d["index"])
                print(f"  [{d['index']}] {name}")
                print(f"    shape={tuple(d['shape'])} data=\n{data}")
            except:
                pass

    # Trace all ops in order
    print("\n=== Full operation trace (ops 200-272) ===")
    ops = interpreter._get_ops_details()
    for i, op in enumerate(ops):
        if i >= 200:
            inputs = list(op['inputs'])
            outputs = list(op['outputs'])
            inp_info = []
            for idx in inputs:
                for d in details:
                    if d['index'] == idx:
                        inp_info.append(f"{idx}:{tuple(d['shape'])}")
            out_info = []
            for idx in outputs:
                for d in details:
                    if d['index'] == idx:
                        out_info.append(f"{idx}:{tuple(d['shape'])}")
            print(f"  Op {i:3d}: {op['op_name']:25s} inputs={inp_info}  outputs={out_info}")

    # Also trace ops around the channel padding (skip connections)
    print("\n=== Channel padding ops (ops 40-50, 85-92, 128-135) ===")
    for i, op in enumerate(ops):
        if (40 <= i <= 50) or (85 <= i <= 92) or (128 <= i <= 135):
            inputs = list(op['inputs'])
            outputs = list(op['outputs'])
            inp_info = []
            for idx in inputs:
                for d in details:
                    if d['index'] == idx:
                        name_short = d['name'].split('/')[-1][:40]
                        inp_info.append(f"{idx}:{tuple(d['shape'])}({name_short})")
            out_info = []
            for idx in outputs:
                for d in details:
                    if d['index'] == idx:
                        name_short = d['name'].split('/')[-1][:40]
                        out_info.append(f"{idx}:{tuple(d['shape'])}({name_short})")
            print(f"  Op {i:3d}: {op['op_name']:25s} in={inp_info}  out={out_info}")


if __name__ == "__main__":
    main()
