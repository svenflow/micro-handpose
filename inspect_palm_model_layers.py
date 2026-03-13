#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Inspect all layers/ops in the palm detection TFLite model to understand
its architecture for WebGPU porting.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

project_dir = Path(__file__).parent
tflite_path = project_dir / "palm_detection.tflite"

interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

# Get all tensor details
tensor_details = interpreter.get_tensor_details()

print(f"Total tensors: {len(tensor_details)}")
print()

# Group by type
weights = []
inputs = []
outputs = []
intermediates = []

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_indices = {d['index'] for d in input_details}
output_indices = {d['index'] for d in output_details}

for td in tensor_details:
    shape = td['shape']
    name = td['name']
    idx = td['index']
    dtype = td['dtype']

    if idx in input_indices:
        print(f"INPUT  [{idx}] {name}: shape={shape} dtype={dtype}")
    elif idx in output_indices:
        print(f"OUTPUT [{idx}] {name}: shape={shape} dtype={dtype}")
    elif len(shape) >= 2 and np.prod(shape) > 1:
        # Likely a weight or intermediate
        tensor = interpreter.get_tensor(idx)
        is_const = not np.all(tensor == 0)
        if is_const and len(shape) >= 2:
            kind = "WEIGHT"
            # Identify type
            if len(shape) == 4:
                if shape[3] == 1 or shape[0] == 1:
                    kind = f"WEIGHT(dw {shape})"
                else:
                    kind = f"WEIGHT(conv {shape})"
            elif len(shape) == 1:
                kind = f"BIAS({shape[0]})"
            print(f"{kind:30s} [{idx}] {name}: shape={shape}")

# Now list just the conv/dw layers in order
print("\n\n=== Layer Architecture (weights only) ===")
conv_layers = []
for td in sorted(tensor_details, key=lambda x: x['index']):
    shape = td['shape']
    name = td['name']
    idx = td['index']
    if len(shape) == 4:
        tensor = interpreter.get_tensor(idx)
        if not np.all(tensor == 0):
            # TFLite NHWC format
            n, h, w, c = shape
            if h > 1 or w > 1:  # Not 1x1 "bias"
                # Check if depthwise (groups == input channels)
                is_dw = (n == 1)  # TFLite depthwise: [1, kH, kW, C_out*multiplier]
                if is_dw:
                    print(f"  DW Conv: kernel={h}x{w}, channels={c}")
                else:
                    print(f"  Conv:    kernel={h}x{w}, in_ch=?, out_ch={n}")
                conv_layers.append((name, shape, is_dw))
    elif len(shape) == 1:
        tensor = interpreter.get_tensor(idx)
        if not np.all(tensor == 0):
            pass  # bias, skip for cleaner output

print(f"\nTotal conv layers: {len(conv_layers)}")
