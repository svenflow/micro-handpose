#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Inspect the TFLite palm detection model for depthwise conv biases.
Check if any DW conv operations have non-zero bias tensors that we might be missing.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

TFLITE_PATH = Path(__file__).parent / "palm_detection.tflite"


def main():
    print(f"Loading: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    details = interpreter.get_tensor_details()

    # Get all ops
    # TFLite Python API doesn't directly expose ops, but we can examine tensors

    print("\n=== ALL tensors in model ===")
    for d in details:
        idx = d["index"]
        name = d["name"]
        shape = tuple(d["shape"])

        try:
            data = interpreter.get_tensor(idx).astype(np.float32)
            is_zero = np.all(data == 0)
            data_min = data.min()
            data_max = data.max()
            data_mean = data.mean()
        except Exception:
            is_zero = None
            data_min = data_max = data_mean = None

        # Flag potential DW bias tensors
        is_dw_related = "depthwise" in name.lower()
        is_1d = len(shape) == 1
        is_conv = "conv" in name.lower()
        is_bn = "batch_normalization" in name.lower() or "FusedBatchNorm" in name

        marker = ""
        if is_dw_related:
            marker = " *** DW ***"
        if is_1d and (is_dw_related or (is_zero is not None and not is_zero)):
            marker += " [1D]"

        if is_zero is not None:
            print(f"  [{idx:3d}] {str(shape):>20s}  zero={is_zero!s:>5s}  min={data_min:>10.6f}  max={data_max:>10.6f}  mean={data_mean:>10.6f}  {name}{marker}")
        else:
            print(f"  [{idx:3d}] {str(shape):>20s}  (no data)  {name}{marker}")

    # Now specifically look for 1D tensors that could be DW biases
    print("\n\n=== 1D tensors (potential biases) ===")
    for d in details:
        shape = tuple(d["shape"])
        if len(shape) != 1:
            continue
        try:
            data = interpreter.get_tensor(d["index"]).astype(np.float32)
            is_zero = np.all(data == 0)
        except Exception:
            continue

        print(f"  [{d['index']:3d}] shape={shape}  zero={is_zero!s:>5s}  {d['name']}")

    # Check the model's internal operations via the experimental API
    print("\n\n=== TFLite model operations ===")
    try:
        # Use the internal _interpreter to get op details
        op_details = interpreter._get_ops_details()
        for op in op_details:
            op_name = op.get("op_name", "unknown")
            if "DEPTHWISE" in op_name.upper():
                inputs = op.get("inputs", [])
                outputs = op.get("outputs", [])
                print(f"\n  OP: {op_name}")
                print(f"    inputs: {inputs}")
                print(f"    outputs: {outputs}")
                for inp_idx in inputs:
                    for d in details:
                        if d["index"] == inp_idx:
                            try:
                                data = interpreter.get_tensor(inp_idx).astype(np.float32)
                                is_zero = np.all(data == 0)
                                print(f"      input[{inp_idx}]: {d['name']}  shape={tuple(d['shape'])}  zero={is_zero}  min={data.min():.6f} max={data.max():.6f}")
                            except:
                                print(f"      input[{inp_idx}]: {d['name']}  shape={tuple(d['shape'])}  (no data)")
    except Exception as e:
        print(f"  Could not get ops: {e}")

        # Alternative: use flatbuffers to parse the model
        try:
            from tensorflow.lite.python.schema_py_generated import Model
            with open(TFLITE_PATH, 'rb') as f:
                buf = f.read()
            model = Model.GetRootAs(buf)
            subgraph = model.Subgraphs(0)

            for i in range(subgraph.OperatorsLength()):
                op = subgraph.Operators(i)
                opcode_idx = op.OpcodeIndex()
                opcode = model.OperatorCodes(opcode_idx)

                # Check if this is a depthwise conv
                builtin_code = opcode.DeprecatedBuiltinCode()
                # DEPTHWISE_CONV_2D = 4 in TFLite schema
                if builtin_code == 4:
                    input_count = op.InputsLength()
                    inputs = [op.Inputs(j) for j in range(input_count)]

                    print(f"\n  DEPTHWISE_CONV_2D (op {i}): inputs={inputs}")
                    for inp_idx in inputs:
                        tensor = subgraph.Tensors(inp_idx)
                        name = tensor.Name().decode() if tensor.Name() else "unnamed"
                        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]

                        # Check if this tensor has data
                        try:
                            data = interpreter.get_tensor(inp_idx).astype(np.float32)
                            is_zero = np.all(data == 0)
                            print(f"      tensor[{inp_idx}]: {name}  shape={shape}  zero={is_zero}  min={data.min():.6f} max={data.max():.6f}")
                        except:
                            print(f"      tensor[{inp_idx}]: {name}  shape={shape}  (activation)")
        except Exception as e2:
            print(f"  Flatbuffer parse also failed: {e2}")


if __name__ == "__main__":
    main()
