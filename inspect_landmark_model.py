#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "flatbuffers", "tensorflow"]
# ///
"""Check hand landmark TFLite model for resize ops and their settings."""

from pathlib import Path
import tensorflow as tf

# Try different possible model paths
MODELS = [
    "hand_landmark_full.tflite",
    "hand_landmark.tflite",
    "hand_landmark_lite.tflite",
]

SCRIPT_DIR = Path(__file__).parent

def inspect_model(path):
    print(f"\n=== Inspecting {path.name} ===")
    from tensorflow.lite.python.schema_py_generated import Model, ResizeBilinearOptions

    with open(path, 'rb') as f:
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

    resize_count = 0
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        opcode = opcodes[op.OpcodeIndex()]

        # RESIZE_BILINEAR = 23
        if opcode == 23:
            resize_count += 1
            inputs = [op.Inputs(j) for j in range(op.InputsLength())]
            outputs = [op.Outputs(j) for j in range(op.OutputsLength())]

            print(f"\n  Op {i}: RESIZE_BILINEAR")
            print(f"    inputs: {inputs}")
            print(f"    outputs: {outputs}")

            opts_tab = op.BuiltinOptions()
            if opts_tab:
                opts = ResizeBilinearOptions()
                opts.Init(opts_tab.Bytes, opts_tab.Pos)
                print(f"    align_corners: {opts.AlignCorners()}")
                print(f"    half_pixel_centers: {opts.HalfPixelCenters()}")

            for inp_idx in inputs:
                t = subgraph.Tensors(inp_idx)
                name = t.Name().decode() if t.Name() else "?"
                shape = [t.Shape(j) for j in range(t.ShapeLength())]
                print(f"    input[{inp_idx}]: shape={shape} name={name[:60]}")

            for out_idx in outputs:
                t = subgraph.Tensors(out_idx)
                name = t.Name().decode() if t.Name() else "?"
                shape = [t.Shape(j) for j in range(t.ShapeLength())]
                print(f"    output[{out_idx}]: shape={shape} name={name[:60]}")

    if resize_count == 0:
        print("  No RESIZE_BILINEAR ops found")

def main():
    for name in MODELS:
        path = SCRIPT_DIR / name
        if path.exists():
            inspect_model(path)
        else:
            # Check docs/weights
            path = SCRIPT_DIR / "docs" / "weights" / name
            if path.exists():
                inspect_model(path)

if __name__ == "__main__":
    main()
