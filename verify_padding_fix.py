#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "pillow", "tensorflow"]
# ///
"""
Verify that SAME padding (with offset -1) matches TFLite's first conv layer output,
while no-padding does not.

First conv: 5x5 kernel, stride 2, SAME padding, 3->32 channels, followed by PReLU.
TFLite tensor layout:
  - idx 121: conv output (post batch-norm), shape [1, 96, 96, 32]
  - idx 122: PReLU output, shape [1, 96, 96, 32]
  - idx 11:  conv weights [32, 5, 5, 3] (quantized)
  - idx 314: conv weights [32, 5, 5, 3] (dequantized)
  - idx 67:  fused batchnorm bias [32] (quantized)
  - idx 304: fused batchnorm bias [32] (dequantized)
  - idx 68:  PReLU alpha [1, 1, 32] (quantized)
  - idx 319: PReLU alpha [1, 1, 32] (dequantized)
"""

import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf

project_dir = Path(__file__).parent

# --- Load TFLite model ---
# experimental_preserve_all_tensors=True keeps intermediate tensor values after invoke()
interpreter = tf.lite.Interpreter(
    model_path=str(project_dir / "palm_detection.tflite"),
    experimental_preserve_all_tensors=True
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

# --- Load and preprocess image ---
img = Image.open(project_dir / "docs" / "hand_nikhil.jpg").convert("RGB")
img = img.resize((192, 192), Image.BILINEAR)
pixels = np.array(img, dtype=np.float32) / 255.0  # [192, 192, 3] in [0, 1]
input_data = pixels.reshape(1, 192, 192, 3)
print(f"Input: shape={input_data.shape}, min={input_data.min():.4f}, max={input_data.max():.4f}")

# --- Run TFLite inference ---
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get TFLite reference outputs
tflite_conv_out = interpreter.get_tensor(121)   # [1, 96, 96, 32] - conv+batchnorm output
tflite_prelu_out = interpreter.get_tensor(122)  # [1, 96, 96, 32] - after PReLU

print(f"TFLite conv output (idx 121): shape={tflite_conv_out.shape}, "
      f"min={tflite_conv_out.min():.6f}, max={tflite_conv_out.max():.6f}")
print(f"TFLite PReLU output (idx 122): shape={tflite_prelu_out.shape}, "
      f"min={tflite_prelu_out.min():.6f}, max={tflite_prelu_out.max():.6f}")

# --- Extract weights ---
# The model has quantized and dequantized versions. Use the ones TFLite actually uses.
# Try dequantized first (idx 314, 304, 319), fall back to quantized (idx 11, 67, 68).
conv_weight = interpreter.get_tensor(11)   # [32, 5, 5, 3]
conv_bias = interpreter.get_tensor(67)     # [32] - fused batchnorm bias
prelu_alpha = interpreter.get_tensor(68)   # [1, 1, 32]

print(f"\nConv weight: shape={conv_weight.shape}, dtype={conv_weight.dtype}")
print(f"Conv bias: shape={conv_bias.shape}, dtype={conv_bias.dtype}")
print(f"PReLU alpha: shape={prelu_alpha.shape}, dtype={prelu_alpha.dtype}")
print(f"  weight range: [{conv_weight.min():.6f}, {conv_weight.max():.6f}]")
print(f"  bias range: [{conv_bias.min():.6f}, {conv_bias.max():.6f}]")
print(f"  alpha range: [{prelu_alpha.min():.6f}, {prelu_alpha.max():.6f}]")

# Flatten alpha to [32]
alpha = prelu_alpha.flatten()

# --- Manual conv implementations ---
# Input: NHWC [1, 192, 192, 3]
# Weight: TFLite format [outCh, kH, kW, inCh] = [32, 5, 5, 3]
# Output: [1, 96, 96, 32]
# Stride: 2, Kernel: 5x5

inp = input_data[0]  # [192, 192, 3]
out_h, out_w = 96, 96
out_ch = 32
kH, kW = 5, 5
in_ch = 3


def run_conv_no_padding(inp, weight, bias):
    """WITHOUT padding (broken): in_y = out_y*2 + ky, in_x = out_x*2 + kx
    This treats the input as if no padding is needed, so the kernel starts at (0,0).
    For stride=2 with 5x5 kernel on 192x192, this produces output positions where
    the last kernel positions may go out of bounds, so we clamp to valid range."""
    out = np.zeros((out_h, out_w, out_ch), dtype=np.float32)
    for oc in range(out_ch):
        for ky in range(kH):
            for kx in range(kW):
                for ic in range(in_ch):
                    y_start = ky
                    x_start = kx
                    # Slice: inp[y_start::2, x_start::2, ic] but only first out_h/out_w
                    y_indices = np.arange(y_start, y_start + out_h * 2, 2)
                    x_indices = np.arange(x_start, x_start + out_w * 2, 2)
                    # Clamp to valid input range
                    valid_y = y_indices < 192
                    valid_x = x_indices < 192
                    if not (valid_y.all() and valid_x.all()):
                        # Some go out of bounds - handle element by element
                        for oy in range(out_h):
                            iy = y_start + oy * 2
                            if iy >= 192:
                                continue
                            for ox in range(out_w):
                                ix = x_start + ox * 2
                                if ix >= 192:
                                    continue
                                out[oy, ox, oc] += inp[iy, ix, ic] * weight[oc, ky, kx, ic]
                    else:
                        out[:, :, oc] += (
                            inp[y_start:y_start + out_h * 2:2, x_start:x_start + out_w * 2:2, ic]
                            * weight[oc, ky, kx, ic]
                        )
        out[:, :, oc] += bias[oc]
    return out


def run_conv_same_padding(inp, weight, bias):
    """WITH SAME padding (fixed):
    For 5x5 stride 2, SAME padding on 192->96:
      total_pad = (96-1)*2 + 5 - 192 = 190 + 5 - 192 = 3
      pad_top = floor(3/2) = 1, pad_bottom = ceil(3/2) = 2
      pad_left = 1, pad_right = 2
    So we pad: top=1, bottom=2, left=1, right=2 (asymmetric 1,2,1,2)
    """
    # Pad input
    padded = np.zeros((192 + 3, 192 + 3, 3), dtype=np.float32)
    padded[1:193, 1:193, :] = inp  # 1 top, 2 bottom, 1 left, 2 right

    out = np.zeros((out_h, out_w, out_ch), dtype=np.float32)
    for oc in range(out_ch):
        for ky in range(kH):
            for kx in range(kW):
                for ic in range(in_ch):
                    y_start = ky
                    x_start = kx
                    out[:, :, oc] += (
                        padded[y_start:y_start + out_h * 2:2, x_start:x_start + out_w * 2:2, ic]
                        * weight[oc, ky, kx, ic]
                    )
        out[:, :, oc] += bias[oc]
    return out


def prelu(x, alpha):
    """PReLU: max(0,x) + alpha * min(0,x)"""
    return np.maximum(0, x) + alpha * np.minimum(0, x)


print("\nRunning manual convolutions...")

# No padding version
conv_no_pad = run_conv_no_padding(inp, conv_weight, conv_bias)
out_no_pad = prelu(conv_no_pad, alpha)

# SAME padding version
conv_same_pad = run_conv_same_padding(inp, conv_weight, conv_bias)
out_same_pad = prelu(conv_same_pad, alpha)

# Reference: PReLU output
ref = tflite_prelu_out[0]  # [96, 96, 32]

# Also compare pre-PReLU
ref_conv = tflite_conv_out[0]  # [96, 96, 32]

# Compare PReLU outputs
err_no_pad = np.abs(out_no_pad - ref)
err_same_pad = np.abs(out_same_pad - ref)

# Compare pre-PReLU (conv only)
err_conv_no_pad = np.abs(conv_no_pad - ref_conv)
err_conv_same_pad = np.abs(conv_same_pad - ref_conv)

print(f"\n{'='*60}")
print("PRE-PReLU (conv + bias only):")
print(f"  WITHOUT padding:")
print(f"    Max absolute error:  {err_conv_no_pad.max():.6f}")
print(f"    Mean absolute error: {err_conv_no_pad.mean():.6f}")
print(f"  WITH SAME padding:")
print(f"    Max absolute error:  {err_conv_same_pad.max():.6f}")
print(f"    Mean absolute error: {err_conv_same_pad.mean():.6f}")

print(f"\nPOST-PReLU:")
print(f"  WITHOUT padding (broken):")
print(f"    Max absolute error:  {err_no_pad.max():.6f}")
print(f"    Mean absolute error: {err_no_pad.mean():.6f}")
print(f"    Output range: [{out_no_pad.min():.6f}, {out_no_pad.max():.6f}]")

print(f"\n  WITH SAME padding (fixed):")
print(f"    Max absolute error:  {err_same_pad.max():.6f}")
print(f"    Mean absolute error: {err_same_pad.mean():.6f}")
print(f"    Output range: [{out_same_pad.min():.6f}, {out_same_pad.max():.6f}]")

print(f"\n  TFLite reference range: [{ref.min():.6f}, {ref.max():.6f}]")
print(f"{'='*60}")

if err_same_pad.max() < 1e-4:
    print("\nSAME-padded version matches TFLite output (max err < 1e-4)")
elif err_same_pad.max() < 1e-2:
    print(f"\nSAME-padded version nearly matches TFLite (max err = {err_same_pad.max():.6f})")
else:
    print(f"\nSAME-padded version has notable error (max err = {err_same_pad.max():.6f})")

improvement = err_no_pad.mean() / err_same_pad.mean() if err_same_pad.mean() > 0 else float('inf')
print(f"SAME padding is {improvement:.1f}x better than no padding (mean error ratio).")
