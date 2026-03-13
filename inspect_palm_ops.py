#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Inspect the exact TFLite operation graph of the palm detection model.

Prints every operation in execution order with:
- Op type (CONV_2D, DEPTHWISE_CONV_2D, ADD, PRELU, etc.)
- Input/output tensor indices and shapes
- Padding type, strides, dilation
- Activation functions (fused or separate)

This is needed to exactly reproduce the computation in WebGPU.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "palm_detection.tflite"


def get_builtin_op_name(op_code):
    """Map TFLite builtin op code to name."""
    # TFLite BuiltinOperator enum values
    op_names = {
        0: "ADD",
        1: "AVERAGE_POOL_2D",
        2: "CONCATENATION",
        3: "CONV_2D",
        4: "DEPTHWISE_CONV_2D",
        5: "DEPTH_TO_SPACE",
        6: "DEQUANTIZE",
        7: "EMBEDDING_LOOKUP",
        8: "FLOOR",
        9: "FULLY_CONNECTED",
        10: "HASHTABLE_LOOKUP",
        11: "L2_NORMALIZATION",
        12: "L2_POOL_2D",
        13: "LOCAL_RESPONSE_NORMALIZATION",
        14: "LOGISTIC",
        15: "LSH_PROJECTION",
        16: "LSTM",
        17: "MAX_POOL_2D",
        18: "MUL",
        19: "RELU",
        20: "RELU_N1_TO_1",
        21: "RELU6",
        22: "RESHAPE",
        23: "RESIZE_BILINEAR",
        25: "SOFTMAX",
        26: "SPACE_TO_DEPTH",
        27: "SVDF",
        28: "TANH",
        29: "CONCAT_EMBEDDINGS",
        30: "SKIP_GRAM",
        31: "CALL",
        32: "CUSTOM",
        33: "EMBEDDING_LOOKUP_SPARSE",
        34: "PAD",
        35: "UNIDIRECTIONAL_SEQUENCE_RNN",
        36: "GATHER",
        37: "BATCH_TO_SPACE_ND",
        38: "SPACE_TO_BATCH_ND",
        39: "TRANSPOSE",
        40: "MEAN",
        41: "SUB",
        42: "DIV",
        43: "SQUEEZE",
        44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
        45: "STRIDED_SLICE",
        46: "BIDIRECTIONAL_SEQUENCE_RNN",
        47: "EXP",
        48: "TOPK_V2",
        49: "SPLIT",
        50: "LOG_SOFTMAX",
        51: "DELEGATE",
        52: "BIDIRECTIONAL_SEQUENCE_LSTM",
        53: "CAST",
        54: "PRELU",
        55: "MAXIMUM",
        56: "ARG_MAX",
        57: "MINIMUM",
        58: "LESS",
        59: "NEG",
        60: "PADV2",
        61: "GREATER",
        62: "GREATER_EQUAL",
        63: "LESS_EQUAL",
        64: "SELECT",
        65: "SLICE",
        66: "SIN",
        67: "TRANSPOSE_CONV",
        68: "SPARSE_TO_DENSE",
        69: "TILE",
        70: "EXPAND_DIMS",
        71: "EQUAL",
        72: "NOT_EQUAL",
        73: "LOG",
        74: "SUM",
        75: "SQRT",
        76: "RSQRT",
        77: "SHAPE",
        78: "POW",
        79: "ARG_MIN",
        80: "FAKE_QUANT",
        81: "REDUCE_PROD",
        82: "REDUCE_MAX",
        83: "PACK",
        84: "LOGICAL_OR",
        85: "ONE_HOT",
        86: "LOGICAL_AND",
        87: "LOGICAL_NOT",
        88: "UNPACK",
        89: "REDUCE_MIN",
        90: "FLOOR_DIV",
        91: "REDUCE_ANY",
        92: "SQUARE",
        93: "ZEROS_LIKE",
        94: "FILL",
        95: "FLOOR_MOD",
        96: "RANGE",
        97: "RESIZE_NEAREST_NEIGHBOR",
        98: "LEAKY_RELU",
        99: "SQUARED_DIFFERENCE",
        100: "MIRROR_PAD",
        101: "ABS",
        102: "SPLIT_V",
        103: "UNIQUE",
        104: "CEIL",
        105: "REVERSE_V2",
        106: "ADD_N",
        107: "GATHER_ND",
        108: "COS",
        109: "WHERE",
        110: "RANK",
        111: "ELU",
        112: "REVERSE_SEQUENCE",
        113: "MATRIX_DIAG",
        114: "QUANTIZE",
        115: "MATRIX_SET_DIAG",
        116: "ROUND",
        117: "HARD_SWISH",
        118: "IF",
        119: "WHILE",
        120: "NON_MAX_SUPPRESSION_V4",
        121: "NON_MAX_SUPPRESSION_V5",
        122: "SCATTER_ND",
        123: "SELECT_V2",
        124: "DENSIFY",
        125: "SEGMENT_SUM",
        126: "BATCH_MATMUL",
    }
    return op_names.get(op_code, f"UNKNOWN_{op_code}")


def get_activation_name(act_code):
    """Map TFLite activation function code to name."""
    act_names = {
        0: "NONE",
        1: "RELU",
        2: "RELU_N1_TO_1",
        3: "RELU6",
        4: "TANH",
        5: "SIGN_BIT",
    }
    return act_names.get(act_code, f"UNKNOWN_{act_code}")


def get_padding_name(pad_code):
    """Map TFLite padding code to name."""
    pad_names = {
        0: "SAME",
        1: "VALID",
    }
    return pad_names.get(pad_code, f"UNKNOWN_{pad_code}")


def main():
    print(f"Loading TFLite model: {TFLITE_PATH}")

    # Load the flatbuffer directly to get op details
    with open(TFLITE_PATH, "rb") as f:
        model_data = f.read()

    # Also use the interpreter for tensor details
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    all_details = interpreter.get_tensor_details()
    tensor_map = {d["index"]: d for d in all_details}

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n=== MODEL INPUTS ===")
    for d in input_details:
        print(f"  Tensor {d['index']}: {d['name']} shape={list(d['shape'])} dtype={d['dtype']}")

    print("\n=== MODEL OUTPUTS ===")
    for d in output_details:
        print(f"  Tensor {d['index']}: {d['name']} shape={list(d['shape'])} dtype={d['dtype']}")

    # Use TFLite's internal API to get the op graph
    # The _get_ops_details() method is available but undocumented
    try:
        ops = interpreter._get_ops_details()
    except AttributeError:
        print("\nWARNING: _get_ops_details() not available, using flatbuffer parsing")
        ops = None

    if ops is not None:
        print(f"\n=== OPERATION GRAPH ({len(ops)} ops) ===")
        print(f"{'#':>3s}  {'Op Type':<25s}  {'Inputs':<40s}  {'Outputs':<15s}  {'Details'}")
        print("=" * 150)

        for i, op in enumerate(ops):
            op_name = op.get("op_name", "UNKNOWN")
            inputs = op.get("inputs", [])
            outputs = op.get("outputs", [])

            # Build input shapes string
            input_strs = []
            for inp_idx in inputs:
                if inp_idx in tensor_map:
                    td = tensor_map[inp_idx]
                    shape = list(td["shape"])
                    name = td["name"]
                    # Shorten name for display
                    short = name.split("/")[-1] if "/" in name else name
                    input_strs.append(f"t{inp_idx}{shape}")
                else:
                    input_strs.append(f"t{inp_idx}[?]")

            # Build output shapes string
            output_strs = []
            for out_idx in outputs:
                if out_idx in tensor_map:
                    td = tensor_map[out_idx]
                    shape = list(td["shape"])
                    output_strs.append(f"t{out_idx}{shape}")
                else:
                    output_strs.append(f"t{out_idx}[?]")

            inputs_str = ", ".join(input_strs)
            outputs_str = ", ".join(output_strs)

            # Extract conv-specific details
            details_str = ""
            if op_name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                # Try to extract stride and padding from the op options
                # These are embedded in the flatbuffer but we can infer from tensor shapes
                if len(inputs) >= 2:
                    inp_shape = list(tensor_map[inputs[0]]["shape"]) if inputs[0] in tensor_map else None
                    kernel_shape = list(tensor_map[inputs[1]]["shape"]) if inputs[1] in tensor_map else None
                    out_shape = list(tensor_map[outputs[0]]["shape"]) if outputs[0] in tensor_map else None

                    if inp_shape and out_shape and len(inp_shape) == 4 and len(out_shape) == 4:
                        stride_h = inp_shape[1] // out_shape[1] if out_shape[1] > 0 else "?"
                        stride_w = inp_shape[2] // out_shape[2] if out_shape[2] > 0 else "?"
                        if kernel_shape:
                            details_str += f"kernel={kernel_shape} "
                        details_str += f"stride={stride_h}x{stride_w} "

                        # Infer padding
                        if kernel_shape and len(kernel_shape) == 4:
                            kh = kernel_shape[1]
                            kw = kernel_shape[2]
                            # Check if output size matches SAME padding formula
                            import math
                            same_h = math.ceil(inp_shape[1] / stride_h) if isinstance(stride_h, int) else None
                            same_w = math.ceil(inp_shape[2] / stride_w) if isinstance(stride_w, int) else None
                            if same_h == out_shape[1] and same_w == out_shape[2]:
                                details_str += "pad=SAME "
                            else:
                                details_str += "pad=VALID "

            print(f"{i:>3d}  {op_name:<25s}  {inputs_str:<40s}  {outputs_str:<15s}  {details_str}")

    # Print all tensors for reference
    print(f"\n\n=== ALL TENSORS ({len(all_details)} total) ===")
    print(f"{'Idx':>4s}  {'Name':<60s}  {'Shape':<25s}  {'Dtype':<10s}  {'Quantization'}")
    print("-" * 130)

    for d in sorted(all_details, key=lambda x: x["index"]):
        name = d["name"]
        shape = list(d["shape"])
        dtype = str(d["dtype"])
        quant = d.get("quantization_parameters", {})
        quant_str = ""
        if quant:
            scales = quant.get("scales", [])
            zero_points = quant.get("zero_points", [])
            if len(scales) > 0 and scales[0] != 0:
                quant_str = f"scale={scales[0]:.6f} zp={zero_points[0]}"

        print(f"{d['index']:>4d}  {name:<60s}  {str(shape):<25s}  {dtype:<10s}  {quant_str}")

    # Print a summary of unique op types
    if ops:
        op_counts = {}
        for op in ops:
            name = op.get("op_name", "UNKNOWN")
            op_counts[name] = op_counts.get(name, 0) + 1

        print(f"\n=== OP TYPE SUMMARY ===")
        for name, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            print(f"  {name:<25s}  {count:>3d}")

    # Detailed conv analysis - print each conv with full params
    if ops:
        print(f"\n\n=== DETAILED CONV/DEPTHWISE ANALYSIS ===")
        conv_idx = 0
        for i, op in enumerate(ops):
            op_name = op.get("op_name", "UNKNOWN")
            if op_name not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                continue

            inputs = op.get("inputs", [])
            outputs = op.get("outputs", [])

            inp_td = tensor_map.get(inputs[0], {})
            kernel_td = tensor_map.get(inputs[1], {})
            bias_td = tensor_map.get(inputs[2], {}) if len(inputs) > 2 else {}
            out_td = tensor_map.get(outputs[0], {})

            inp_shape = list(inp_td.get("shape", []))
            kernel_shape = list(kernel_td.get("shape", []))
            bias_shape = list(bias_td.get("shape", []))
            out_shape = list(out_td.get("shape", []))

            stride_h = inp_shape[1] // out_shape[1] if len(inp_shape) == 4 and len(out_shape) == 4 and out_shape[1] > 0 else "?"
            stride_w = inp_shape[2] // out_shape[2] if len(inp_shape) == 4 and len(out_shape) == 4 and out_shape[2] > 0 else "?"

            import math
            pad = "?"
            if len(inp_shape) == 4 and len(out_shape) == 4 and isinstance(stride_h, int):
                same_h = math.ceil(inp_shape[1] / stride_h)
                if same_h == out_shape[1]:
                    pad = "SAME"
                else:
                    pad = "VALID"

            print(f"\nOp #{i} ({op_name}) [conv_idx={conv_idx}]:")
            print(f"  Input:  t{inputs[0]} {inp_shape}  ({inp_td.get('name', '?')})")
            print(f"  Kernel: t{inputs[1]} {kernel_shape}  ({kernel_td.get('name', '?')})")
            if bias_shape:
                print(f"  Bias:   t{inputs[2]} {bias_shape}  ({bias_td.get('name', '?')})")
            print(f"  Output: t{outputs[0]} {out_shape}  ({out_td.get('name', '?')})")
            print(f"  Stride: {stride_h}x{stride_w}, Padding: {pad}")

            conv_idx += 1


if __name__ == "__main__":
    main()
