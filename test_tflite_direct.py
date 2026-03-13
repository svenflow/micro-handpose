#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "pillow", "tensorflow"]
# ///
"""
Run the TFLite hand landmark model DIRECTLY (no palm detection crop)
on the same 256x256 input to compare with our PyTorch/WebGPU output.
This proves whether the 1.84% error is from the model or from MediaPipe's preprocessing.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
tflite = tf.lite

project_dir = Path(__file__).parent

# Load TFLite model
tflite_path = project_dir / "hand_landmark.tflite"
if not tflite_path.exists():
    print(f"ERROR: {tflite_path} not found. Run extract_tflite_weights.py first.")
    exit(1)

interpreter = tflite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== TFLite Model Info ===")
print(f"Input: {input_details[0]['name']}, shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
for od in output_details:
    print(f"Output: {od['name']}, shape={od['shape']}, dtype={od['dtype']}")

# Load and preprocess image — SAME as our WebGPU/PyTorch pipeline
img = Image.open(project_dir / "docs" / "hand_nikhil.jpg").convert("RGB")
img = img.resize((256, 256), Image.BILINEAR)
pixels = np.array(img, dtype=np.float32) / 255.0  # [256, 256, 3] in [0, 1]

# TFLite expects NHWC: [1, 256, 256, 3]
input_data = pixels.reshape(1, 256, 256, 3)
print(f"\nInput: shape={input_data.shape}, min={input_data.min():.4f}, max={input_data.max():.4f}")

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get outputs
outputs = {}
for od in output_details:
    tensor = interpreter.get_tensor(od['index'])
    outputs[od['name']] = tensor
    print(f"Output '{od['name']}': shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}")

# Parse landmarks — figure out which output is which
# Hand landmark model outputs: landmarks (63 values), handflag (1), handedness (1)
for name, tensor in outputs.items():
    flat = tensor.flatten()
    if flat.shape[0] == 63:
        landmarks_raw = flat
        print(f"\n  Found landmarks in '{name}'")
    elif flat.shape[0] == 1:
        val = flat[0]
        if 'flag' in name.lower() or 'confidence' in name.lower():
            print(f"  Handflag: {val:.6f} (sigmoid={1/(1+np.exp(-val)):.6f})")
        else:
            print(f"  Output '{name}': {val:.6f} (sigmoid={1/(1+np.exp(-val)):.6f})")

# If we found landmarks, compare
if 'landmarks_raw' in dir():
    landmarks = landmarks_raw / 256.0
else:
    # Try each output
    print("\nTrying to identify landmarks output...")
    for name, tensor in outputs.items():
        flat = tensor.flatten()
        print(f"  '{name}': size={flat.shape[0]}, first 5: {flat[:5]}")
    # The one with 63 elements or that reshapes to 21x3
    for name, tensor in outputs.items():
        flat = tensor.flatten()
        if flat.shape[0] >= 63:
            landmarks_raw = flat[:63]
            landmarks = landmarks_raw / 256.0
            print(f"\n  Using '{name}' as landmarks (first 63 values)")
            break

NAMES = [
    'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]

# Load PyTorch reference output for comparison
# (from reference_model.py output with fixed padding)
# Also compare with MediaPipe browser output
mp_landmarks = [
    (0.3962, 0.8616, 0.0000),
    (0.5835, 0.7839, -0.0348),
    (0.7186, 0.6426, -0.0183),
    (0.8162, 0.5565, -0.0060),
    (0.8999, 0.4988, 0.0071),
    (0.5850, 0.4422, 0.0806),
    (0.6091, 0.3305, 0.1045),
    (0.6168, 0.2742, 0.0999),
    (0.6181, 0.2199, 0.0893),
    (0.4791, 0.4260, 0.0796),
    (0.4736, 0.3009, 0.1145),
    (0.4794, 0.2317, 0.1014),
    (0.4766, 0.1766, 0.0844),
    (0.3800, 0.4487, 0.0665),
    (0.3461, 0.3340, 0.0833),
    (0.3402, 0.2674, 0.0666),
    (0.3340, 0.2097, 0.0482),
    (0.2868, 0.5024, 0.0454),
    (0.2221, 0.4186, 0.0454),
    (0.1913, 0.3589, 0.0390),
    (0.1719, 0.3026, 0.0328),
]

# WebGPU output from accuracy test (with the fix)
webgpu_landmarks = [
    (0.4536, 0.7985),
    (0.5924, 0.7628),
    (0.6984, 0.6461),
    (0.7824, 0.5556),
    (0.8766, 0.5117),
    (0.5825, 0.4478),
    (0.6050, 0.3376),
    (0.6102, 0.2715),
    (0.6083, 0.2116),
    (0.4780, 0.4292),
    (0.4741, 0.3119),
    (0.4728, 0.2294),
    (0.4719, 0.1567),
    (0.3849, 0.4512),
    (0.3540, 0.3274),
    (0.3443, 0.2520),
    (0.3429, 0.1861),
    (0.2959, 0.5088),
    (0.2393, 0.4157),
    (0.2084, 0.3626),
    (0.1858, 0.3064),
]

print(f"\n{'name':12s} {'tflite_x':>9s} {'tflite_y':>9s} {'webgpu_x':>9s} {'webgpu_y':>9s} {'mp_x':>9s} {'mp_y':>9s} {'tfl_vs_mp':>10s} {'tfl_vs_wg':>10s} {'wg_vs_mp':>10s}")

total_tfl_mp = 0
total_tfl_wg = 0
total_wg_mp = 0

for i in range(21):
    tfl_x, tfl_y = landmarks[i*3], landmarks[i*3+1]
    wg_x, wg_y = webgpu_landmarks[i]
    mp_x, mp_y = mp_landmarks[i][:2]

    tfl_mp = np.sqrt((tfl_x - mp_x)**2 + (tfl_y - mp_y)**2)
    tfl_wg = np.sqrt((tfl_x - wg_x)**2 + (tfl_y - wg_y)**2)
    wg_mp = np.sqrt((wg_x - mp_x)**2 + (wg_y - mp_y)**2)

    total_tfl_mp += tfl_mp
    total_tfl_wg += tfl_wg
    total_wg_mp += wg_mp

    print(f"  {NAMES[i]:12s} {tfl_x:9.4f} {tfl_y:9.4f} {wg_x:9.4f} {wg_y:9.4f} {mp_x:9.4f} {mp_y:9.4f} {tfl_mp*100:9.2f}% {tfl_wg*100:9.2f}% {wg_mp*100:9.2f}%")

print(f"\n  Summary:")
print(f"    TFLite direct vs MediaPipe browser: {total_tfl_mp/21*100:.2f}%")
print(f"    TFLite direct vs WebGPU:            {total_tfl_wg/21*100:.2f}%")
print(f"    WebGPU vs MediaPipe browser:         {total_wg_mp/21*100:.2f}%")
print(f"\n  If TFLite-direct ≈ WebGPU but both ≠ MediaPipe, the error is from MediaPipe's palm detection crop.")
print(f"  If TFLite-direct ≈ MediaPipe but ≠ WebGPU, we have a computation bug.")
