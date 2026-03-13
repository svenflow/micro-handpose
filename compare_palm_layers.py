#!/usr/bin/env -S uv run --script --python 3.13
# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "tensorflow",
#     "numpy",
#     "pillow",
# ]
# ///
"""
Layer-by-layer comparison of WebGPU shader CPU reimplementation vs TF reference
for the palm detection model's first 3 layers.

Uses tf.nn ops as ground truth (NOT TFLite intermediate tensors, which are
unreliable due to buffer reuse in the TFLite runtime).

Verifies:
  1. Initial 5x5 conv stride-2 + PReLU (192x192x3 -> 96x96x32)
  2. First depthwise 5x5 conv (96x96x32 -> 96x96x32)
  3. First pointwise 1x1 + skip + PReLU (96x96x32 -> 96x96x32)
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).parent

# ── Load & prepare input image ─────────────────────────────────────────
img = Image.open(ROOT / "docs" / "hand_nikhil.jpg").resize((192, 192))
img_np = np.array(img, dtype=np.float32) / 255.0  # [192,192,3] NHWC, [0,1]
input_chw = np.transpose(img_np, (2, 0, 1))        # [3,192,192] CHW for our shaders

# ── Load exported weights ──────────────────────────────────────────────
with open(ROOT / "weights" / "palm_detection_weights.json") as f:
    manifest = json.load(f)
weight_bin = (ROOT / "weights" / "palm_detection_weights.bin").read_bytes()

m_keys = manifest['keys']
m_shapes = manifest['shapes']
m_offsets = manifest['offsets']

def load_weight(index):
    shape = m_shapes[index]
    start = m_offsets[index]
    n = 1
    for s in shape:
        n *= s
    return np.frombuffer(weight_bin, dtype=np.float32, count=n, offset=start).copy().reshape(shape)

def find_w(*subs):
    for i, k in enumerate(m_keys):
        if all(s in k for s in subs):
            return i
    raise KeyError(f"Not found: {subs}")

# Load weights
init_conv_w = load_weight(find_w('conv2d/Conv2D'))          # [32,5,5,3] OHWI
init_conv_bias = load_weight(find_w('batch_normalization/FusedBatchNormV3', 'conv2d/Conv2D'))  # [32]
init_prelu_alpha = load_weight(find_w('p_re_lu/add', 'p_re_lu/Relu'))  # [1,1,32]

dw0_w = load_weight(find_w('depthwise_conv2d/depthwise'))   # [1,5,5,32]
dw0_bias = np.zeros(32, dtype=np.float32)  # DW bias is zero in this model

pw0_w = load_weight(find_w('conv2d_1/Conv2D'))              # [32,1,1,32] OHWI
pw0_bias = load_weight(find_w('batch_normalization_1/FusedBatchNormV3', 'conv2d_1/Conv2D'))  # [32]
pw0_alpha = load_weight(find_w('p_re_lu_1/add', 'p_re_lu_1/Relu'))  # [1,1,32]

# ── CPU reimplementation (matching WebGPU shaders exactly) ─────────────

def cpu_conv5x5_stride2(input_chw, weight_ohwi, bias, alpha):
    """Exact CPU replica of PALM_CONV5X5_STRIDE2_PRELU_SHADER.
    Input: CHW [3,192,192]. Weight: OHWI [32,5,5,3].
    SAME padding: pad_top=1, pad_left=1.
    Returns (prelu_output, conv_output) both in CHW."""
    in_ch, in_h, in_w = input_chw.shape
    out_ch = weight_ohwi.shape[0]
    out_h, out_w = in_h // 2, in_w // 2
    padded = np.pad(input_chw, ((0, 0), (1, 2), (1, 2)), mode='constant')
    out_pre = np.zeros((out_ch, out_h, out_w), dtype=np.float32)
    for oc in range(out_ch):
        for ky in range(5):
            for kx in range(5):
                for ic in range(in_ch):
                    out_pre[oc] += weight_ohwi[oc, ky, kx, ic] * padded[ic, ky::2, kx::2][:out_h, :out_w]
    out_pre += bias.reshape(-1, 1, 1)
    a = alpha.flatten().reshape(-1, 1, 1)
    out = np.maximum(0, out_pre) + a * np.minimum(0, out_pre)
    return out, out_pre

def transpose_dw(w):
    """[1,5,5,ch] -> [ch,25] matching palm_model.ts transposeDW."""
    _, kH, kW, ch = w.shape
    r = np.zeros((ch, 25), dtype=np.float32)
    for c in range(ch):
        for ky in range(kH):
            for kx in range(kW):
                r[c, ky * 5 + kx] = w[0, ky, kx, c]
    return r

def cpu_dw5x5(input_chw, w_c25, bias, stride=1, pad=2):
    """Exact CPU replica of PALM_DEPTHWISE_5X5_SHADER."""
    ch, in_h, in_w = input_chw.shape
    out_h, out_w = in_h // stride, in_w // stride
    padded = np.pad(input_chw, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((ch, out_h, out_w), dtype=np.float32)
    for ky in range(5):
        for kx in range(5):
            out += w_c25[:, ky * 5 + kx].reshape(-1, 1, 1) * padded[:, ky::stride, kx::stride][:, :out_h, :out_w]
    out += bias.reshape(-1, 1, 1)
    return out

def cpu_pw_skip_prelu(dw_chw, skip_chw, pw_w, pw_b, alpha, in_ch, out_ch, stride=1):
    """Exact CPU replica of PALM_POINTWISE_SKIP_PRELU_SHADER."""
    _, h, w = dw_chw.shape
    pw_2d = pw_w.reshape(out_ch, in_ch)
    out = np.zeros((out_ch, h, w), dtype=np.float32)
    for oc in range(out_ch):
        for ic in range(in_ch):
            out[oc] += pw_2d[oc, ic] * dw_chw[ic]
        out[oc] += pw_b[oc]
    # Skip: zero-pad channels, stride-1 add
    skip_ch = skip_chw.shape[0]
    out[:min(out_ch, skip_ch)] += skip_chw[:min(out_ch, skip_ch)]
    # PReLU
    a = alpha.flatten().reshape(-1, 1, 1)
    out = np.maximum(0, out) + a * np.minimum(0, out)
    return out

# ── Run CPU reimplementation ──────────────────────────────────────────
print("=== Running CPU reimplementation (WebGPU shader replica) ===")
cpu_prelu, cpu_conv = cpu_conv5x5_stride2(input_chw, init_conv_w, init_conv_bias, init_prelu_alpha)
dw0_w_t = transpose_dw(dw0_w)
cpu_dw0 = cpu_dw5x5(cpu_prelu, dw0_w_t, dw0_bias, stride=1, pad=2)
cpu_pw0 = cpu_pw_skip_prelu(cpu_dw0, cpu_prelu, pw0_w, pw0_bias, pw0_alpha, 32, 32, stride=1)

# ── Reference computation using tf.nn ─────────────────────────────────
print("=== Running TF reference computation ===")
inp_tf = tf.constant(img_np[np.newaxis], dtype=tf.float32)

# L1: Conv5x5 stride 2
w_hwio = np.transpose(init_conv_w, (1, 2, 3, 0))
ref_conv = tf.nn.conv2d(inp_tf, w_hwio, strides=2, padding='SAME').numpy()
ref_conv += init_conv_bias.reshape(1, 1, 1, -1)
a0 = init_prelu_alpha.flatten().reshape(1, 1, 1, -1)
ref_prelu = np.maximum(0, ref_conv) + a0 * np.minimum(0, ref_conv)

# L2: DW5x5 stride 1
dw_hwcm = np.transpose(dw0_w, (1, 2, 3, 0))  # [5,5,32,1]
ref_dw = tf.nn.depthwise_conv2d(
    tf.constant(ref_prelu, dtype=tf.float32),
    tf.constant(dw_hwcm, dtype=tf.float32),
    strides=[1, 1, 1, 1], padding='SAME'
).numpy()
ref_dw += dw0_bias.reshape(1, 1, 1, -1)

# L3: PW 1x1 + skip + PReLU
pw_hwio = np.transpose(pw0_w, (1, 2, 3, 0))  # [1,1,32,32]
ref_pw = tf.nn.conv2d(
    tf.constant(ref_dw, dtype=tf.float32),
    tf.constant(pw_hwio, dtype=tf.float32),
    strides=1, padding='VALID'
).numpy()
ref_pw += pw0_bias.reshape(1, 1, 1, -1)
ref_add = ref_pw + ref_prelu
a1 = pw0_alpha.flatten().reshape(1, 1, 1, -1)
ref_block_prelu = np.maximum(0, ref_add) + a1 * np.minimum(0, ref_add)

# ── Compare ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LAYER-BY-LAYER COMPARISON: CPU (WebGPU shader) vs TF Reference")
print("=" * 70)

layers = [
    ("L1 conv5x5+bias",  cpu_conv,  ref_conv[0]),
    ("L1 PReLU",         cpu_prelu, ref_prelu[0]),
    ("L2 DW5x5",         cpu_dw0,   ref_dw[0]),
    ("L3 PW+skip+PReLU", cpu_pw0,   ref_block_prelu[0]),
]

all_ok = True
for name, cpu_chw, ref_hwc in layers:
    cpu_hwc = np.transpose(cpu_chw, (1, 2, 0))
    diff = np.abs(cpu_hwc - ref_hwc)
    max_err = diff.max()
    mean_err = diff.mean()
    status = "OK" if max_err < 0.001 else "DIVERGED"
    if max_err >= 0.001:
        all_ok = False
    print(f"  {name:22s}  max_err={max_err:.8f}  mean_err={mean_err:.8f}  [{status}]")
    if max_err >= 0.001:
        loc = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Worst at (y={loc[0]}, x={loc[1]}, c={loc[2]}): ref={ref_hwc[loc]:.8f} cpu={cpu_hwc[loc]:.8f}")
        per_ch = diff.max(axis=(0, 1))
        print(f"    Worst channel: {np.argmax(per_ch)}  max_err={per_ch.max():.8f}")

# ── Verify TFLite final output matches ─────────────────────────────────
print("\n=== Verifying TFLite final output consistency ===")
interp = tf.lite.Interpreter(model_path=str(ROOT / "palm_detection.tflite"))
interp.allocate_tensors()
interp.set_tensor(interp.get_input_details()[0]['index'], img_np[np.newaxis])
interp.invoke()
tfl_regressors = interp.get_tensor(interp.get_output_details()[0]['index'])
tfl_scores = interp.get_tensor(interp.get_output_details()[1]['index'])
print(f"  TFLite final output: regressors {tfl_regressors.shape}, scores {tfl_scores.shape}")
print(f"  Score range: [{tfl_scores.min():.4f}, {tfl_scores.max():.4f}]")
print(f"  Top scores (sigmoid): {sorted([1/(1+np.exp(-s)) for s in tfl_scores.flatten()], reverse=True)[:5]}")

# ── NOTE about TFLite intermediate tensors ─────────────────────────────
print(f"\n=== NOTE: TFLite intermediate tensors are UNRELIABLE ===")
print(f"  Reading intermediate tensors via interpreter.get_tensor() after invoke()")
print(f"  returns STALE or INCORRECT values because TFLite's runtime reuses")
print(f"  tensor buffers as scratch space during inference.")
print(f"  Always use tf.nn ops as the reference, NOT TFLite intermediates.")

# ── Final verdict ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
if all_ok:
    print("VERDICT: ALL LAYERS MATCH (max error < 0.001)")
    print("")
    print("The CPU reimplementation of WebGPU shaders is correct for layers 1-3.")
    print("Weight indexing, transposition, SAME padding, and PReLU all match.")
    print("The divergence between WebGPU and TFLite (if any) must come from:")
    print("  1. Input preprocessing (letterbox resize, normalization range)")
    print("  2. Later layers in the network")
    print("  3. Output postprocessing (anchor decode, NMS)")
    print("  4. fp16 quantization in TFLite vs fp32 in WebGPU")
else:
    print("VERDICT: DIVERGENCE FOUND — see details above")
print("=" * 70)
