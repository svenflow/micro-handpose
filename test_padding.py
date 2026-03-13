#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""
Test: does fixing asymmetric padding (1,2,1,2) for stride-2 fix the accuracy?
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

def depthwise_conv2d(input, weight, bias, stride=1, padding=0, asymmetric_pad=False):
    """Depthwise conv. When asymmetric_pad=True and stride=2, use (1,2,1,2) padding."""
    c, h, w = input.shape
    kh, kw = weight.shape[2], weight.shape[3]

    if asymmetric_pad and stride == 2:
        # Original PyTorch: F.pad(x, (1, 2, 1, 2), "constant", 0)
        # (left, right, top, bottom) = (1, 2, 1, 2)
        pi = np.zeros((c, h+3, w+3), dtype=np.float32)
        pi[:, 1:h+1, 1:w+1] = input  # 1 top, 2 bottom, 1 left, 2 right
        padding = 0  # padding already applied
    elif padding > 0:
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

def resmodule(x, prefix, stride=1, use_asymmetric=False):
    dw_w = weights[f"{prefix}convs.0.weight"]
    dw_b = weights[f"{prefix}convs.0.bias"]
    pw_w = weights[f"{prefix}convs.1.weight"]
    pw_b = weights[f"{prefix}convs.1.bias"]
    in_ch = x.shape[0]

    if use_asymmetric and stride == 2:
        dw = depthwise_conv2d(x, dw_w, dw_b, stride=2, asymmetric_pad=True)
    else:
        dw = depthwise_conv2d(x, dw_w, dw_b, stride=stride, padding=2)

    pw = pointwise_conv2d(dw, pw_w, pw_b)

    skip = maxpool2x2(x) if stride == 2 else x

    # Channel padding (not skip add for extra channels)
    out_ch = pw_w.shape[0]
    if out_ch > in_ch:
        padded_skip = np.zeros((out_ch, skip.shape[1], skip.shape[2]), dtype=np.float32)
        padded_skip[:in_ch] = skip
        skip = padded_skip

    pw += skip
    return np.maximum(0, pw)

def run_model(input_padded, use_asymmetric=False):
    x = conv2d(input_padded, weights["backbone1.1.weight"], weights["backbone1.1.bias"], stride=2)
    x = np.maximum(0, x)
    x = resmodule(x, "backbone1.3.f.0.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone1.3.f.1.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone1.4.", stride=2, use_asymmetric=use_asymmetric); b1 = x.copy()
    x = resmodule(x, "backbone2.0.f.0.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone2.0.f.1.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone2.1.", stride=2, use_asymmetric=use_asymmetric); b2 = x.copy()
    x = resmodule(x, "backbone3.0.f.0.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone3.0.f.1.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone3.1.", stride=2, use_asymmetric=use_asymmetric); b3 = x.copy()
    x = resmodule(x, "backbone4.0.f.0.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone4.0.f.1.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "backbone4.1.", stride=2, use_asymmetric=use_asymmetric)
    x = bilinear_upsample_2x(x) + b3
    x = resmodule(x, "backbone5.0.", use_asymmetric=use_asymmetric)
    x = bilinear_upsample_2x(x) + b2
    x = resmodule(x, "backbone6.0.", use_asymmetric=use_asymmetric)
    x = pointwise_conv2d(x, weights["backbone6.1.weight"], weights["backbone6.1.bias"])
    x = bilinear_upsample_2x(x) + b1
    for i in range(4): x = resmodule(x, f"ff.0.f.{i}.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "ff.1.", stride=2, use_asymmetric=use_asymmetric)
    for i in range(4): x = resmodule(x, f"ff.2.f.{i}.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "ff.3.", stride=2, use_asymmetric=use_asymmetric)
    for i in range(4): x = resmodule(x, f"ff.4.f.{i}.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "ff.5.", stride=2, use_asymmetric=use_asymmetric)
    for i in range(4): x = resmodule(x, f"ff.6.f.{i}.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "ff.7.", stride=2, use_asymmetric=use_asymmetric)
    for i in range(4): x = resmodule(x, f"ff.8.f.{i}.", use_asymmetric=use_asymmetric)
    x = resmodule(x, "ff.9.", stride=2, use_asymmetric=use_asymmetric)
    for i in range(4): x = resmodule(x, f"ff.10.f.{i}.", use_asymmetric=use_asymmetric)
    feat = x.reshape(-1)
    lm_raw = weights["reg_3d.weight"].reshape(63,-1) @ feat + weights["reg_3d.bias"]
    return lm_raw / 256.0

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

def compute_error(landmarks):
    total = 0
    for i in range(21):
        dx = landmarks[i*3] - mp_landmarks[i][0]
        dy = landmarks[i*3+1] - mp_landmarks[i][1]
        total += np.sqrt(dx**2 + dy**2)
    return total / 21 * 100

# Load and preprocess image
img = Image.open(weights_dir.parent / "docs" / "hand_nikhil.jpg").convert("RGB")
img = img.resize((256, 256), Image.BILINEAR)
pixels = np.array(img, dtype=np.float32) / 255.0
padded = np.zeros((3, 257, 257), dtype=np.float32)
padded[:, 1:257, 1:257] = pixels.transpose(2, 0, 1)

print("="*60)
print("Test: Symmetric padding (current, pad=2 on all sides)")
print("="*60)
lm_sym = run_model(padded, use_asymmetric=False)
err_sym = compute_error(lm_sym)
print(f"Average error vs MediaPipe: {err_sym:.2f}%")
for idx in [0, 4, 8, 12, 20]:
    x, y = lm_sym[idx*3], lm_sym[idx*3+1]
    mx, my = mp_landmarks[idx]
    print(f"  {NAMES[idx]:12s}: ({x:.4f}, {y:.4f}) vs MP({mx:.4f}, {my:.4f})")

print()
print("="*60)
print("Test: Asymmetric padding (original PyTorch: 1,2,1,2)")
print("="*60)
lm_asym = run_model(padded, use_asymmetric=True)
err_asym = compute_error(lm_asym)
print(f"Average error vs MediaPipe: {err_asym:.2f}%")
for idx in [0, 4, 8, 12, 20]:
    x, y = lm_asym[idx*3], lm_asym[idx*3+1]
    mx, my = mp_landmarks[idx]
    print(f"  {NAMES[idx]:12s}: ({x:.4f}, {y:.4f}) vs MP({mx:.4f}, {my:.4f})")

print(f"\nImprovement: {err_sym:.2f}% → {err_asym:.2f}% ({err_sym - err_asym:.2f}% reduction)")
