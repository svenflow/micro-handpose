#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "flatbuffers"]
# ///
"""
Inspect TFLite model's RESIZE_BILINEAR ops for half_pixel_centers setting.
Also check all op options/attributes.
"""

import struct
from pathlib import Path

TFLITE_PATH = Path(__file__).parent / "palm_detection.tflite"


def main():
    with open(TFLITE_PATH, 'rb') as f:
        buf = bytearray(f.read())

    # Parse TFLite flatbuffer manually
    # TFLite schema: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs

    # Root table offset
    root_offset = struct.unpack_from('<I', buf, 0)[0]

    def read_offset(pos):
        """Read a flatbuffer offset (relative) and return absolute position"""
        rel = struct.unpack_from('<I', buf, pos)[0]
        return pos + rel if rel else 0

    def read_vtable(table_pos):
        """Read vtable for a table"""
        vtable_offset = struct.unpack_from('<i', buf, table_pos)[0]
        vtable_pos = table_pos - vtable_offset
        vtable_size = struct.unpack_from('<H', buf, vtable_pos)[0]
        table_size = struct.unpack_from('<H', buf, vtable_pos + 2)[0]
        num_fields = (vtable_size - 4) // 2
        fields = []
        for i in range(num_fields):
            fields.append(struct.unpack_from('<H', buf, vtable_pos + 4 + i * 2)[0])
        return fields, table_pos

    # Use tensorflow to load and inspect
    import importlib
    try:
        # Try using tflite_runtime or tensorflow
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
        interpreter.allocate_tensors()

        # Get op details using experimental API
        try:
            ops = interpreter._get_ops_details()
            print("=== All operations ===")
            for i, op in enumerate(ops):
                print(f"  Op {i}: {op}")
        except:
            pass
    except ImportError:
        pass

    # Direct flatbuffer parsing for resize ops
    # Parse using the raw flatbuffer
    print("\n=== Searching for RESIZE_BILINEAR options in raw flatbuffer ===")

    # TFLite BuiltinOperator enum: RESIZE_BILINEAR = 23
    # ResizeBilinearOptions has:
    #   align_corners: bool (field 0)
    #   half_pixel_centers: bool (field 1)

    # Let's search for the string patterns that indicate resize ops
    text = buf.decode('latin-1')

    # Actually, let's use a proper approach: search for the op in the flatbuffer
    # The model structure is: Model -> Subgraphs -> Operators -> BuiltinOptions

    # First, find operator codes
    # Model table fields: version(0), operator_codes(1), subgraphs(2), description(3), buffers(4)

    # Read model table
    model_fields, model_pos = read_vtable(root_offset)

    # operator_codes vector
    if len(model_fields) > 1 and model_fields[1]:
        opcodes_offset = read_offset(model_pos + model_fields[1])
        num_opcodes = struct.unpack_from('<I', buf, opcodes_offset)[0]
        print(f"\nOperator codes ({num_opcodes}):")
        for i in range(num_opcodes):
            opcode_pos = read_offset(opcodes_offset + 4 + i * 4)
            opcode_fields, opcode_table_pos = read_vtable(opcode_pos)
            # OperatorCode fields: deprecated_builtin_code(0), custom_code(1), version(2), builtin_code(3)
            if len(opcode_fields) > 0 and opcode_fields[0]:
                deprecated_code = buf[opcode_table_pos + opcode_fields[0]]
                print(f"  [{i}] deprecated_builtin_code={deprecated_code}")

    # subgraphs vector
    if len(model_fields) > 2 and model_fields[2]:
        subgraphs_offset = read_offset(model_pos + model_fields[2])
        num_subgraphs = struct.unpack_from('<I', buf, subgraphs_offset)[0]

        for sg_i in range(num_subgraphs):
            sg_pos = read_offset(subgraphs_offset + 4 + sg_i * 4)
            sg_fields, sg_table_pos = read_vtable(sg_pos)

            # Subgraph fields: tensors(0), inputs(1), outputs(2), operators(3), name(4)
            if len(sg_fields) > 3 and sg_fields[3]:
                ops_offset = read_offset(sg_table_pos + sg_fields[3])
                num_ops = struct.unpack_from('<I', buf, ops_offset)[0]
                print(f"\nSubgraph {sg_i}: {num_ops} operators")

                for op_i in range(num_ops):
                    op_pos = read_offset(ops_offset + 4 + op_i * 4)
                    op_fields, op_table_pos = read_vtable(op_pos)

                    # Operator fields: opcode_index(0), inputs(1), outputs(2), builtin_options_type(3), builtin_options(4), custom_options(5), custom_options_format(6), mutating_variable_inputs(7), intermediates(8), large_custom_options_offset(9), large_custom_options_size(10), builtin_options_2_type(11), builtin_options_2(12)

                    opcode_index = 0
                    if len(op_fields) > 0 and op_fields[0]:
                        opcode_index = struct.unpack_from('<H', buf, op_table_pos + op_fields[0])[0]

                    builtin_options_type = 0
                    if len(op_fields) > 3 and op_fields[3]:
                        builtin_options_type = buf[op_table_pos + op_fields[3]]

                    # BuiltinOptions_ResizeBilinearOptions = 23
                    if builtin_options_type == 23:
                        print(f"\n  *** Op {op_i}: RESIZE_BILINEAR (opcode_index={opcode_index}, options_type={builtin_options_type})")

                        # Read the options table
                        if len(op_fields) > 4 and op_fields[4]:
                            options_pos = read_offset(op_table_pos + op_fields[4])
                            opt_fields, opt_table_pos = read_vtable(options_pos)

                            # ResizeBilinearOptions: align_corners(0), half_pixel_centers(1)
                            align_corners = False
                            half_pixel_centers = False

                            if len(opt_fields) > 0 and opt_fields[0]:
                                align_corners = bool(buf[opt_table_pos + opt_fields[0]])
                            if len(opt_fields) > 1 and opt_fields[1]:
                                half_pixel_centers = bool(buf[opt_table_pos + opt_fields[1]])

                            print(f"      align_corners = {align_corners}")
                            print(f"      half_pixel_centers = {half_pixel_centers}")

                        # Read inputs/outputs
                        if len(op_fields) > 1 and op_fields[1]:
                            inputs_vec = op_table_pos + op_fields[1]
                            inputs_offset = read_offset(inputs_vec)
                            num_inputs = struct.unpack_from('<I', buf, inputs_offset)[0]
                            input_indices = [struct.unpack_from('<i', buf, inputs_offset + 4 + j * 4)[0] for j in range(num_inputs)]
                            print(f"      inputs: {input_indices}")

                        if len(op_fields) > 2 and op_fields[2]:
                            outputs_vec = op_table_pos + op_fields[2]
                            outputs_offset = read_offset(outputs_vec)
                            num_outputs = struct.unpack_from('<I', buf, outputs_offset)[0]
                            output_indices = [struct.unpack_from('<i', buf, outputs_offset + 4 + j * 4)[0] for j in range(num_outputs)]
                            print(f"      outputs: {output_indices}")

                    # Also check DEPTHWISE_CONV_2D (opcode 4) for its options
                    # BuiltinOptions_DepthwiseConv2DOptions = 6
                    if builtin_options_type == 6:
                        if len(op_fields) > 4 and op_fields[4]:
                            options_pos = read_offset(op_table_pos + op_fields[4])
                            opt_fields, opt_table_pos = read_vtable(options_pos)
                            # DepthwiseConv2DOptions: padding(0), stride_w(1), stride_h(2), depth_multiplier(3), fused_activation_function(4), dilation_w_factor(5), dilation_h_factor(6)
                            padding = 0
                            stride_w = 1
                            stride_h = 1
                            depth_mult = 1
                            activation = 0

                            if len(opt_fields) > 0 and opt_fields[0]:
                                padding = buf[opt_table_pos + opt_fields[0]]
                            if len(opt_fields) > 1 and opt_fields[1]:
                                stride_w = struct.unpack_from('<i', buf, opt_table_pos + opt_fields[1])[0]
                            if len(opt_fields) > 2 and opt_fields[2]:
                                stride_h = struct.unpack_from('<i', buf, opt_table_pos + opt_fields[2])[0]
                            if len(opt_fields) > 4 and opt_fields[4]:
                                activation = buf[opt_table_pos + opt_fields[4]]

                            # Only print if stride != 1 (interesting cases)
                            if stride_w != 1 or stride_h != 1 or activation != 0:
                                print(f"  Op {op_i}: DEPTHWISE_CONV_2D padding={'SAME' if padding==1 else 'VALID'} stride=({stride_h},{stride_w}) activation={activation}")


if __name__ == "__main__":
    main()
