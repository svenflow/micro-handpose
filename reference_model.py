#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""
Reference numpy implementation of the micro-handpose model.
Loads weights and runs inference to compare with WebGPU output.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path

# Load weights
weights_dir = Path(__file__).parent / "weights"
meta = json.loads((weights_dir / "weights_f16.json").read_text())
bin_data = (weights_dir / "weights_f16.bin").read_bytes()

def read_tensor(name):
    idx = meta["keys"].index(name)
    shape = meta["shapes"][idx]
    offset = meta["offsets"][idx]
    n = 1
    for s in shape:
        n *= s
    raw = bin_data[offset : offset + n * 2]
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)

weights = {k: read_tensor(k) for k in meta["keys"]}

# Load and preprocess image (same as WebGPU: resize to 256x256, 0-1 range)
img = Image.open(weights_dir.parent / "docs" / "hand_nikhil.jpg").convert("RGB")
img = img.resize((256, 256), Image.BILINEAR)
pixels = np.array(img, dtype=np.float32) / 255.0  # [256, 256, 3] in [0, 1]

# Convert to NCHW and pad to 257x257
x = pixels.transpose(2, 0, 1)  # [3, 256, 256]
padded = np.zeros((3, 257, 257), dtype=np.float32)
padded[:, 1:257, 1:257] = x  # Same as CANVAS_INPUT_SHADER: pixel at (y+1, x+1)

def conv2d(input, weight, bias, stride=1, padding=0):
    """Standard 2D convolution. input: [C_in, H, W], weight: [C_out, C_in, kH, kW]"""
    c_out, c_in, kh, kw = weight.shape
    h, w = input.shape[1], input.shape[2]

    # Add padding
    if padding > 0:
        padded_input = np.zeros((c_in, h + 2*padding, w + 2*padding), dtype=np.float32)
        padded_input[:, padding:h+padding, padding:w+padding] = input
    else:
        padded_input = input

    ph, pw = padded_input.shape[1], padded_input.shape[2]
    oh = (ph - kh) // stride + 1
    ow = (pw - kw) // stride + 1

    output = np.zeros((c_out, oh, ow), dtype=np.float32)
    for oc in range(c_out):
        for ky in range(kh):
            for kx in range(kw):
                for ic in range(c_in):
                    output[oc] += padded_input[ic, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[oc, ic, ky, kx]
        output[oc] += bias[oc]
    return output

def depthwise_conv2d(input, weight, bias, stride=1, padding=0):
    """Depthwise 2D convolution. input: [C, H, W], weight: [C, 1, kH, kW]"""
    c, h, w = input.shape
    kh, kw = weight.shape[2], weight.shape[3]

    if padding > 0:
        padded_input = np.zeros((c, h + 2*padding, w + 2*padding), dtype=np.float32)
        padded_input[:, padding:h+padding, padding:w+padding] = input
    else:
        padded_input = input

    ph, pw = padded_input.shape[1], padded_input.shape[2]
    oh = (ph - kh) // stride + 1
    ow = (pw - kw) // stride + 1

    output = np.zeros((c, oh, ow), dtype=np.float32)
    for ch in range(c):
        for ky in range(kh):
            for kx in range(kw):
                output[ch] += padded_input[ch, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[ch, 0, ky, kx]
        output[ch] += bias[ch]
    return output

def pointwise_conv2d(input, weight, bias):
    """1x1 convolution. input: [C_in, H, W], weight: [C_out, C_in, 1, 1]"""
    c_out, c_in = weight.shape[0], weight.shape[1]
    h, w = input.shape[1], input.shape[2]
    output = np.zeros((c_out, h, w), dtype=np.float32)
    for oc in range(c_out):
        for ic in range(c_in):
            output[oc] += input[ic] * weight[oc, ic, 0, 0]
        output[oc] += bias[oc]
    return output

def maxpool2x2(input):
    """2x2 max pooling with stride 2"""
    c, h, w = input.shape
    oh, ow = h // 2, w // 2
    output = np.zeros((c, oh, ow), dtype=np.float32)
    for y in range(oh):
        for x in range(ow):
            output[:, y, x] = np.max(input[:, y*2:y*2+2, x*2:x*2+2].reshape(c, 4), axis=1)
    return output

def bilinear_upsample_2x(input):
    """Bilinear 2x upsample (align_corners=False, half-pixel centers)"""
    c, h, w = input.shape
    oh, ow = h * 2, w * 2
    output = np.zeros((c, oh, ow), dtype=np.float32)

    scale_y = h / oh
    scale_x = w / ow

    for oy in range(oh):
        for ox in range(ow):
            src_y = (oy + 0.5) * scale_y - 0.5
            src_x = (ox + 0.5) * scale_x - 0.5
            y0 = max(0, int(np.floor(src_y)))
            x0 = max(0, int(np.floor(src_x)))
            y1 = min(y0 + 1, h - 1)
            x1 = min(x0 + 1, w - 1)
            ly = max(0.0, src_y) - y0
            lx = max(0.0, src_x) - x0

            output[:, oy, ox] = (
                input[:, y0, x0] * (1 - ly) * (1 - lx) +
                input[:, y0, x1] * (1 - ly) * lx +
                input[:, y1, x0] * ly * (1 - lx) +
                input[:, y1, x1] * ly * lx
            )
    return output

def resmodule(input, prefix, stride=1):
    """ResModule: depthwise 5x5 + pointwise 1x1 + skip + relu"""
    dw_weight = weights[f"{prefix}convs.0.weight"]
    dw_bias = weights[f"{prefix}convs.0.bias"]
    pw_weight = weights[f"{prefix}convs.1.weight"]
    pw_bias = weights[f"{prefix}convs.1.bias"]

    in_ch = input.shape[0]
    out_ch = pw_weight.shape[0]

    # Depthwise conv
    dw_out = depthwise_conv2d(input, dw_weight, dw_bias, stride=stride, padding=2)

    # Pointwise conv
    pw_out = pointwise_conv2d(dw_out, pw_weight, pw_bias)

    # Skip connection
    if stride == 2:
        skip = maxpool2x2(input)
    else:
        skip = input

    # Add skip (only for matching channels)
    pw_out[:in_ch] += skip[:in_ch]

    # ReLU
    return np.maximum(0, pw_out)

print("Running reference model...")

# Step 1: Input conv3x3 stride=2 + ReLU: 3ch 257x257 -> 24ch 128x128
input_conv_w = weights["backbone1.1.weight"]  # [24, 3, 3, 3]
input_conv_b = weights["backbone1.1.bias"]     # [24]
x = conv2d(padded, input_conv_w, input_conv_b, stride=2)
x = np.maximum(0, x)
print(f"After input conv: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

# Step 2: backbone1 ResBlock(2) = 2 resmodules with stride=1
x = resmodule(x, "backbone1.3.f.0.", stride=1)
x = resmodule(x, "backbone1.3.f.1.", stride=1)
# ResModule stride=2: 24->48
x = resmodule(x, "backbone1.4.", stride=2)
b1 = x.copy()  # Save for skip connection
print(f"After backbone1: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}")

# Step 3: backbone2
x = resmodule(x, "backbone2.0.f.0.", stride=1)
x = resmodule(x, "backbone2.0.f.1.", stride=1)
x = resmodule(x, "backbone2.1.", stride=2)
b2 = x.copy()
print(f"After backbone2: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}")

# Step 4: backbone3
x = resmodule(x, "backbone3.0.f.0.", stride=1)
x = resmodule(x, "backbone3.0.f.1.", stride=1)
x = resmodule(x, "backbone3.1.", stride=2)
b3 = x.copy()
print(f"After backbone3: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}")

# Step 5: backbone4 + upsample + add b3
x = resmodule(x, "backbone4.0.f.0.", stride=1)
x = resmodule(x, "backbone4.0.f.1.", stride=1)
x = resmodule(x, "backbone4.1.", stride=2)
print(f"After backbone4 (before upsample): shape={x.shape}")
x = bilinear_upsample_2x(x)
x = x + b3
print(f"After backbone4 upsample+add b3: shape={x.shape}")

# Step 6: backbone5 + upsample + add b2
x = resmodule(x, "backbone5.0.", stride=1)
x = bilinear_upsample_2x(x)
x = x + b2
print(f"After backbone5 upsample+add b2: shape={x.shape}")

# Step 7: backbone6 + conv1x1(96->48) + upsample + add b1
x = resmodule(x, "backbone6.0.", stride=1)
conv1x1_w = weights["backbone6.1.weight"]  # [48, 96, 1, 1]
conv1x1_b = weights["backbone6.1.bias"]     # [48]
x = pointwise_conv2d(x, conv1x1_w, conv1x1_b)
x = bilinear_upsample_2x(x)
x = x + b1
print(f"After backbone6: shape={x.shape}")

# Step 8: ff layers
# ff.0: ResBlock(4) = 4 resmodules stride=1, then ff.1: ResModule stride=2
for i in range(4):
    x = resmodule(x, f"ff.0.f.{i}.", stride=1)
x = resmodule(x, "ff.1.", stride=2)
print(f"After ff.0+ff.1: shape={x.shape}")

for i in range(4):
    x = resmodule(x, f"ff.2.f.{i}.", stride=1)
x = resmodule(x, "ff.3.", stride=2)
print(f"After ff.2+ff.3: shape={x.shape}")

for i in range(4):
    x = resmodule(x, f"ff.4.f.{i}.", stride=1)
x = resmodule(x, "ff.5.", stride=2)
print(f"After ff.4+ff.5: shape={x.shape}")

for i in range(4):
    x = resmodule(x, f"ff.6.f.{i}.", stride=1)
x = resmodule(x, "ff.7.", stride=2)
print(f"After ff.6+ff.7: shape={x.shape}")

for i in range(4):
    x = resmodule(x, f"ff.8.f.{i}.", stride=1)
x = resmodule(x, "ff.9.", stride=2)
print(f"After ff.8+ff.9: shape={x.shape}")

# Final ResBlock(4) at 2x2
for i in range(4):
    x = resmodule(x, f"ff.10.f.{i}.", stride=1)
print(f"After ff.10: shape={x.shape}")

# Step 9: Output heads (FC over 2x2 spatial → flatten → matmul)
feature = x.reshape(-1)  # [288 * 2 * 2] = [1152]
print(f"Flattened features: shape={feature.shape}")

# Handflag
hf_w = weights["handflag.weight"].reshape(1, -1)  # [1, 1152]
hf_b = weights["handflag.bias"]
handflag_raw = hf_w @ feature + hf_b
handflag = 1.0 / (1.0 + np.exp(-handflag_raw))
print(f"Handflag: raw={handflag_raw[0]:.4f}, sigmoid={handflag[0]:.4f}")

# Handedness
hd_w = weights["handedness.weight"].reshape(1, -1)
hd_b = weights["handedness.bias"]
handedness_raw = hd_w @ feature + hd_b
handedness = 1.0 / (1.0 + np.exp(-handedness_raw))
print(f"Handedness: raw={handedness_raw[0]:.4f}, sigmoid={handedness[0]:.4f}")

# Landmarks
lm_w = weights["reg_3d.weight"].reshape(63, -1)  # [63, 1152]
lm_b = weights["reg_3d.bias"]
landmarks_raw = lm_w @ feature + lm_b
landmarks = landmarks_raw / 256.0
print(f"\nLandmarks (raw / 256.0):")

NAMES = [
    'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]

for i in range(21):
    x_val = landmarks[i*3]
    y_val = landmarks[i*3+1]
    z_val = landmarks[i*3+2]
    print(f"  {NAMES[i]:12s}: ({x_val:.4f}, {y_val:.4f}, {z_val:.4f})  raw=({landmarks_raw[i*3]:.1f}, {landmarks_raw[i*3+1]:.1f}, {landmarks_raw[i*3+2]:.1f})")

# Compare with WebGPU output (from accuracy test)
webgpu_landmarks = [
    (0.5024, 0.7348, 0.0000),
    (0.6328, 0.7029, -0.0317),
    (0.6895, 0.6452, -0.0491),
    (0.7380, 0.5942, -0.0646),
    (0.7907, 0.5655, -0.0801),
    (0.6048, 0.5267, -0.0765),
    (0.6238, 0.4450, -0.0998),
    (0.6364, 0.3894, -0.1069),
    (0.6318, 0.3611, -0.1100),
    (0.5183, 0.5101, -0.0745),
    (0.5292, 0.4014, -0.0975),
    (0.5406, 0.3181, -0.1002),
    (0.5456, 0.2814, -0.1040),
    (0.4418, 0.5240, -0.0743),
    (0.4167, 0.4104, -0.0996),
    (0.4262, 0.3307, -0.1112),
    (0.4477, 0.2818, -0.1196),
    (0.3762, 0.5592, -0.0752),
    (0.3074, 0.4699, -0.1076),
    (0.2844, 0.4127, -0.1330),
    (0.2782, 0.3662, -0.1572),
]

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

print("\n=== Comparison: numpy vs WebGPU vs MediaPipe ===")
print(f"{'landmark':12s} {'numpy_x':>8s} {'numpy_y':>8s} {'webgpu_x':>8s} {'webgpu_y':>8s} {'mp_x':>8s} {'mp_y':>8s} {'np_err':>8s} {'wg_err':>8s}")
total_np_err = 0
total_wg_err = 0
for i in range(21):
    np_x, np_y = landmarks[i*3], landmarks[i*3+1]
    wg_x, wg_y = webgpu_landmarks[i][:2]
    mp_x, mp_y = mp_landmarks[i][:2]
    np_err = np.sqrt((np_x - mp_x)**2 + (np_y - mp_y)**2)
    wg_err = np.sqrt((wg_x - mp_x)**2 + (wg_y - mp_y)**2)
    total_np_err += np_err
    total_wg_err += wg_err
    print(f"  {NAMES[i]:12s} {np_x:8.4f} {np_y:8.4f} {wg_x:8.4f} {wg_y:8.4f} {mp_x:8.4f} {mp_y:8.4f} {np_err*100:7.2f}% {wg_err*100:7.2f}%")

print(f"\n  Average error: numpy={total_np_err/21*100:.2f}%, webgpu={total_wg_err/21*100:.2f}%")
print(f"  numpy-webgpu agreement:")
np_wg_diff = 0
for i in range(21):
    np_x, np_y = landmarks[i*3], landmarks[i*3+1]
    wg_x, wg_y = webgpu_landmarks[i][:2]
    np_wg_diff += np.sqrt((np_x - wg_x)**2 + (np_y - wg_y)**2)
print(f"  Avg numpy vs webgpu diff: {np_wg_diff/21*100:.2f}%")
