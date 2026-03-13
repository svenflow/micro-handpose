#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""
Test: if we crop the hand region first (simulating palm detection), do landmarks match MediaPipe?
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path

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
    return np.frombuffer(bin_data[offset:offset+n*2], dtype=np.float16).astype(np.float32).reshape(shape)

weights = {k: read_tensor(k) for k in meta["keys"]}

def depthwise_conv2d(input, weight, bias, stride=1, padding=0):
    c, h, w = input.shape
    kh, kw = weight.shape[2], weight.shape[3]
    if padding > 0:
        pi = np.zeros((c, h+2*padding, w+2*padding), dtype=np.float32)
        pi[:, padding:h+padding, padding:w+padding] = input
    else:
        pi = input
    ph, pw = pi.shape[1], pi.shape[2]
    oh = (ph - kh) // stride + 1
    ow = (pw - kw) // stride + 1
    out = np.zeros((c, oh, ow), dtype=np.float32)
    for ch in range(c):
        for ky in range(kh):
            for kx in range(kw):
                out[ch] += pi[ch, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[ch, 0, ky, kx]
        out[ch] += bias[ch]
    return out

def pointwise_conv2d(input, weight, bias):
    c_out, c_in = weight.shape[0], weight.shape[1]
    h, w = input.shape[1], input.shape[2]
    out = np.zeros((c_out, h, w), dtype=np.float32)
    for oc in range(c_out):
        for ic in range(c_in):
            out[oc] += input[ic] * weight[oc, ic, 0, 0]
        out[oc] += bias[oc]
    return out

def maxpool2x2(input):
    c, h, w = input.shape
    oh, ow = h // 2, w // 2
    out = np.zeros((c, oh, ow), dtype=np.float32)
    for y in range(oh):
        for x in range(ow):
            out[:, y, x] = np.max(input[:, y*2:y*2+2, x*2:x*2+2].reshape(c, 4), axis=1)
    return out

def bilinear_upsample_2x(input):
    c, h, w = input.shape
    oh, ow = h*2, w*2
    out = np.zeros((c, oh, ow), dtype=np.float32)
    sy, sx = h/oh, w/ow
    for oy in range(oh):
        for ox in range(ow):
            src_y = (oy+0.5)*sy - 0.5
            src_x = (ox+0.5)*sx - 0.5
            y0 = max(0, int(np.floor(src_y)))
            x0 = max(0, int(np.floor(src_x)))
            y1 = min(y0+1, h-1)
            x1 = min(x0+1, w-1)
            ly = max(0.0, src_y) - y0
            lx = max(0.0, src_x) - x0
            out[:, oy, ox] = (input[:, y0, x0]*(1-ly)*(1-lx) + input[:, y0, x1]*(1-ly)*lx +
                              input[:, y1, x0]*ly*(1-lx) + input[:, y1, x1]*ly*lx)
    return out

def conv2d(input, weight, bias, stride=1):
    c_out, c_in, kh, kw = weight.shape
    h, w = input.shape[1], input.shape[2]
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((c_out, oh, ow), dtype=np.float32)
    for oc in range(c_out):
        for ky in range(kh):
            for kx in range(kw):
                for ic in range(c_in):
                    out[oc] += input[ic, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[oc, ic, ky, kx]
        out[oc] += bias[oc]
    return out

def resmodule(x, prefix, stride=1):
    dw_w = weights[f"{prefix}convs.0.weight"]
    dw_b = weights[f"{prefix}convs.0.bias"]
    pw_w = weights[f"{prefix}convs.1.weight"]
    pw_b = weights[f"{prefix}convs.1.bias"]
    in_ch = x.shape[0]
    dw = depthwise_conv2d(x, dw_w, dw_b, stride=stride, padding=2)
    pw = pointwise_conv2d(dw, pw_w, pw_b)
    skip = maxpool2x2(x) if stride == 2 else x
    pw[:in_ch] += skip[:in_ch]
    return np.maximum(0, pw)

def run_model(input_nchw_padded):
    x = conv2d(input_nchw_padded, weights["backbone1.1.weight"], weights["backbone1.1.bias"], stride=2)
    x = np.maximum(0, x)
    x = resmodule(x, "backbone1.3.f.0."); x = resmodule(x, "backbone1.3.f.1.")
    x = resmodule(x, "backbone1.4.", stride=2); b1 = x.copy()
    x = resmodule(x, "backbone2.0.f.0."); x = resmodule(x, "backbone2.0.f.1.")
    x = resmodule(x, "backbone2.1.", stride=2); b2 = x.copy()
    x = resmodule(x, "backbone3.0.f.0."); x = resmodule(x, "backbone3.0.f.1.")
    x = resmodule(x, "backbone3.1.", stride=2); b3 = x.copy()
    x = resmodule(x, "backbone4.0.f.0."); x = resmodule(x, "backbone4.0.f.1.")
    x = resmodule(x, "backbone4.1.", stride=2)
    x = bilinear_upsample_2x(x) + b3
    x = resmodule(x, "backbone5.0.")
    x = bilinear_upsample_2x(x) + b2
    x = resmodule(x, "backbone6.0.")
    x = pointwise_conv2d(x, weights["backbone6.1.weight"], weights["backbone6.1.bias"])
    x = bilinear_upsample_2x(x) + b1
    for i in range(4): x = resmodule(x, f"ff.0.f.{i}.")
    x = resmodule(x, "ff.1.", stride=2)
    for i in range(4): x = resmodule(x, f"ff.2.f.{i}.")
    x = resmodule(x, "ff.3.", stride=2)
    for i in range(4): x = resmodule(x, f"ff.4.f.{i}.")
    x = resmodule(x, "ff.5.", stride=2)
    for i in range(4): x = resmodule(x, f"ff.6.f.{i}.")
    x = resmodule(x, "ff.7.", stride=2)
    for i in range(4): x = resmodule(x, f"ff.8.f.{i}.")
    x = resmodule(x, "ff.9.", stride=2)
    for i in range(4): x = resmodule(x, f"ff.10.f.{i}.")
    feat = x.reshape(-1)
    hf_raw = weights["handflag.weight"].reshape(1,-1) @ feat + weights["handflag.bias"]
    handflag = 1.0/(1.0+np.exp(-hf_raw[0]))
    lm_raw = weights["reg_3d.weight"].reshape(63,-1) @ feat + weights["reg_3d.bias"]
    return lm_raw / 256.0, handflag

# MediaPipe landmark coordinates (from accuracy test on 256x256 input)
mp_landmarks = [
    (0.3962, 0.8616), (0.5835, 0.7839), (0.7186, 0.6426), (0.8162, 0.5565), (0.8999, 0.4988),
    (0.5850, 0.4422), (0.6091, 0.3305), (0.6168, 0.2742), (0.6181, 0.2199),
    (0.4791, 0.4260), (0.4736, 0.3009), (0.4794, 0.2317), (0.4766, 0.1766),
    (0.3800, 0.4487), (0.3461, 0.3340), (0.3402, 0.2674), (0.3340, 0.2097),
    (0.2868, 0.5024), (0.2221, 0.4186), (0.1913, 0.3589), (0.1719, 0.3026),
]

NAMES = [
    'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]

def compute_error(landmarks, ref):
    total = 0
    for i in range(21):
        dx = landmarks[i*3] - ref[i][0]
        dy = landmarks[i*3+1] - ref[i][1]
        total += np.sqrt(dx**2 + dy**2)
    return total / 21 * 100

# Load original image
img = Image.open(weights_dir.parent / "docs" / "hand_nikhil.jpg").convert("RGB")
orig_w, orig_h = img.size
print(f"Original image: {orig_w}x{orig_h}")

# Test 1: Full image (current behavior)
print("\n" + "="*60)
print("Test 1: Full image → 256×256 (current behavior)")
print("="*60)
full = img.resize((256, 256), Image.BILINEAR)
pixels = np.array(full, dtype=np.float32) / 255.0
padded = np.zeros((3, 257, 257), dtype=np.float32)
padded[:, 1:257, 1:257] = pixels.transpose(2, 0, 1)
lm, hf = run_model(padded)
print(f"Handflag: {hf:.4f}")
print(f"Avg error vs MediaPipe: {compute_error(lm, mp_landmarks):.2f}%")

# Test 2: Simulate palm detection crop
# MediaPipe landmarks tell us the hand spans roughly:
# X: 0.17 to 0.90 (pinky_tip to thumb_tip)
# Y: 0.18 to 0.86 (middle_tip to wrist)
# Add some margin (palm detection typically adds ~20% padding)
min_x = min(p[0] for p in mp_landmarks)
max_x = max(p[0] for p in mp_landmarks)
min_y = min(p[1] for p in mp_landmarks)
max_y = max(p[1] for p in mp_landmarks)

# MediaPipe palm detection provides a rotated square crop with padding
# Let's use the MP landmarks to derive the approximate crop box
cx = (min_x + max_x) / 2
cy = (min_y + max_y) / 2
span = max(max_x - min_x, max_y - min_y)

# Test various crop margins
for margin_pct in [0, 10, 20, 30, 40, 50]:
    margin = span * margin_pct / 100
    crop_size = span + 2 * margin

    # Crop box in pixel coordinates of 256x256 image
    x1 = max(0, int((cx - crop_size/2) * 256))
    y1 = max(0, int((cy - crop_size/2) * 256))
    x2 = min(256, int((cx + crop_size/2) * 256))
    y2 = min(256, int((cy + crop_size/2) * 256))

    # Crop from the 256x256 image
    full_np = np.array(full)
    cropped = Image.fromarray(full_np[y1:y2, x1:x2])
    cropped_256 = cropped.resize((256, 256), Image.BILINEAR)

    pixels_c = np.array(cropped_256, dtype=np.float32) / 255.0
    padded_c = np.zeros((3, 257, 257), dtype=np.float32)
    padded_c[:, 1:257, 1:257] = pixels_c.transpose(2, 0, 1)

    lm_c, hf_c = run_model(padded_c)

    # Map landmarks back to original 256x256 space
    crop_w = x2 - x1
    crop_h = y2 - y1
    lm_mapped = np.zeros_like(lm_c)
    for i in range(21):
        lm_mapped[i*3] = lm_c[i*3] * crop_w / 256.0 + x1 / 256.0
        lm_mapped[i*3+1] = lm_c[i*3+1] * crop_h / 256.0 + y1 / 256.0
        lm_mapped[i*3+2] = lm_c[i*3+2]

    err = compute_error(lm_mapped, mp_landmarks)
    print(f"\nMargin {margin_pct}%: crop=({x1},{y1})-({x2},{y2}) [{crop_w}x{crop_h}px] handflag={hf_c:.4f} error={err:.2f}%")
    if margin_pct == 20:
        # Show per-landmark for the 20% margin case
        for idx in [0, 4, 8, 12, 20]:
            x_val = lm_mapped[idx*3]
            y_val = lm_mapped[idx*3+1]
            mp_x, mp_y = mp_landmarks[idx]
            dist = np.sqrt((x_val-mp_x)**2 + (y_val-mp_y)**2) * 100
            print(f"  {NAMES[idx]:12s}: ours=({x_val:.4f}, {y_val:.4f}) mp=({mp_x:.4f}, {mp_y:.4f}) err={dist:.2f}%")
