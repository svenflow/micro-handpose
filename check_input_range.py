#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Check input tensor range expected by palm detection model."""

import numpy as np
import tensorflow as tf
from pathlib import Path

TFLITE_PATH = Path(__file__).parent / "palm_detection.tflite"

def main():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("=== Input details ===")
    for d in input_details:
        print(f"  name: {d['name']}")
        print(f"  shape: {d['shape']}")
        print(f"  dtype: {d['dtype']}")
        quant = d.get('quantization_parameters', {})
        print(f"  quantization: {quant}")

    print("\n=== Output details ===")
    for d in output_details:
        print(f"  name: {d['name']}")
        print(f"  shape: {d['shape']}")
        print(f"  dtype: {d['dtype']}")

    # Test with different input ranges to see which produces reasonable outputs
    for input_range_name, make_input in [
        ("[0,1]", lambda: np.random.rand(1, 192, 192, 3).astype(np.float32)),
        ("[-1,1]", lambda: (np.random.rand(1, 192, 192, 3).astype(np.float32) * 2 - 1)),
        ("[0,255]", lambda: (np.random.rand(1, 192, 192, 3).astype(np.float32) * 255)),
    ]:
        inp = make_input()
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        out0 = interpreter.get_tensor(output_details[0]['index'])
        out1 = interpreter.get_tensor(output_details[1]['index'])
        print(f"\n=== Input range {input_range_name} ===")
        print(f"  Output[0] (regressors): shape={out0.shape} min={out0.min():.4f} max={out0.max():.4f} mean={out0.mean():.4f}")
        print(f"  Output[1] (classifiers): shape={out1.shape} min={out1.min():.4f} max={out1.max():.4f} mean={out1.mean():.4f}")

    # Check first conv weights to deduce expected input scale
    print("\n=== First conv weight statistics ===")
    details = interpreter.get_tensor_details()
    ops = interpreter._get_ops_details()
    first_op = ops[0]
    print(f"  First op: {first_op['op_name']} inputs={list(first_op['inputs'])} outputs={list(first_op['outputs'])}")

    # Get input tensor details
    inp_idx = list(first_op['inputs'])[0]
    for d in details:
        if d['index'] == inp_idx:
            print(f"  Input tensor [{inp_idx}]: shape={tuple(d['shape'])} name={d['name']}")

if __name__ == "__main__":
    main()
