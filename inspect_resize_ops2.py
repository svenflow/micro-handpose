#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Check RESIZE_BILINEAR ops for half_pixel_centers using TF's interpreter."""

import numpy as np
import tensorflow as tf
from pathlib import Path

TFLITE_PATH = Path(__file__).parent / "palm_detection.tflite"

def main():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    # Get detailed op info
    try:
        ops = interpreter._get_ops_details()
        for i, op in enumerate(ops):
            op_name = op.get("op_name", "")
            if "RESIZE" in op_name.upper() or "UPSAMPLE" in op_name.upper():
                print(f"Op {i}: {op}")
            if "PAD" in op_name.upper() and "PADDING" not in str(op.get("inputs", [])):
                print(f"Op {i}: {op}")
    except Exception as e:
        print(f"_get_ops_details failed: {e}")

    # Alternative: parse the TFLite model schema directly
    print("\n=== Parsing TFLite schema for resize ops ===")
    try:
        from tensorflow.lite.python import schema_py_generated as schema
        with open(TFLITE_PATH, 'rb') as f:
            model_buf = f.read()

        model = schema.ModelT.InitFromPackedBuf(model_buf, 0)
        subgraph = model.subgraphs[0]

        for i, op in enumerate(subgraph.operators):
            opcode = model.operatorCodes[op.opcodeIndex]
            builtin_code = opcode.deprecatedBuiltinCode
            if builtin_code == 127:  # Use builtinCode for newer ops
                builtin_code = opcode.builtinCode

            # RESIZE_BILINEAR = 23
            if builtin_code == 23:
                print(f"\nOp {i}: RESIZE_BILINEAR")
                print(f"  inputs: {list(op.inputs)}")
                print(f"  outputs: {list(op.outputs)}")
                opts = op.builtinOptions
                if opts:
                    print(f"  options type: {type(opts).__name__}")
                    if hasattr(opts, 'alignCorners'):
                        print(f"  alignCorners: {opts.alignCorners}")
                    if hasattr(opts, 'halfPixelCenters'):
                        print(f"  halfPixelCenters: {opts.halfPixelCenters}")
                else:
                    print(f"  No builtin options (defaults apply)")

            # Also check for any ADD ops (for skip connections)
            # ADD = 0
            if builtin_code == 0:
                inputs = list(op.inputs)
                outputs = list(op.outputs)
                # Only print if interesting
                inp_shapes = []
                for idx in inputs:
                    t = subgraph.tensors[idx]
                    inp_shapes.append(list(t.shape))

    except Exception as e:
        print(f"Schema parse failed: {e}")
        import traceback
        traceback.print_exc()

    # Try yet another approach - use tflite_support or manual flatbuffer
    print("\n=== Manual flatbuffer inspection ===")
    try:
        import flatbuffers
        from tensorflow.lite.python.schema_py_generated import Model

        with open(TFLITE_PATH, 'rb') as f:
            raw = bytearray(f.read())

        model = Model.GetRootAs(raw)
        subgraph = model.Subgraphs(0)

        opcodes = []
        for i in range(model.OperatorCodesLength()):
            oc = model.OperatorCodes(i)
            code = oc.DeprecatedBuiltinCode()
            if code == 127:
                code = oc.BuiltinCode()
            opcodes.append(code)

        print(f"Operator codes: {opcodes}")

        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode = opcodes[op.OpcodeIndex()]

            # RESIZE_BILINEAR = 23
            if opcode == 23:
                inputs = [op.Inputs(j) for j in range(op.InputsLength())]
                outputs = [op.Outputs(j) for j in range(op.OutputsLength())]

                print(f"\nOp {i}: RESIZE_BILINEAR")
                print(f"  inputs: {inputs}")
                print(f"  outputs: {outputs}")
                print(f"  builtinOptionsType: {op.BuiltinOptionsType()}")

                # Read builtin options
                from tensorflow.lite.python.schema_py_generated import ResizeBilinearOptions
                opts_tab = op.BuiltinOptions()
                if opts_tab:
                    opts = ResizeBilinearOptions()
                    opts.Init(opts_tab.Bytes, opts_tab.Pos)
                    print(f"  align_corners: {opts.AlignCorners()}")
                    print(f"  half_pixel_centers: {opts.HalfPixelCenters()}")
                else:
                    print(f"  No options table")

                # Print input tensor info
                for inp_idx in inputs:
                    t = subgraph.Tensors(inp_idx)
                    name = t.Name().decode() if t.Name() else "?"
                    shape = [t.Shape(j) for j in range(t.ShapeLength())]
                    print(f"  input[{inp_idx}]: shape={shape} name={name}")

                for out_idx in outputs:
                    t = subgraph.Tensors(out_idx)
                    name = t.Name().decode() if t.Name() else "?"
                    shape = [t.Shape(j) for j in range(t.ShapeLength())]
                    print(f"  output[{out_idx}]: shape={shape} name={name}")

    except Exception as e:
        print(f"Manual parse failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
