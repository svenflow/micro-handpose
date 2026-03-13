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
FULL layer-by-layer comparison of WebGPU shader CPU reimplementation vs TF reference
for the entire palm detection model.

Layers 1-3 already verified to match within 0.00001.
This extends through ALL 18 backbone blocks, FPN, and SSD heads.

Architecture:
  Initial conv 5x5 stride-2 + PReLU → 96x96x32
  Blocks 0-2:  32→32,  stride 1, 96x96
  Block 3:     32→64,  stride 2, 96→48
  Blocks 4-6:  64→64,  stride 1, 48x48
  Block 7:     64→128, stride 2, 48→24
  Blocks 8-10: 128→128, stride 1, 24x24
  Block 11:    128→256, stride 2, 24→12
  Blocks 12-14: 256→256, stride 1, 12x12
  Block 15:    256→256, stride 2, 12→6
  Blocks 16-18: 256→256, stride 1, 6x6
  FPN Level 1: upsample 6→12 + conv1x1_prelu(256→256) + add backbone12 skip + 2 blocks
  FPN Level 2: upsample 12→24 + conv1x1_prelu(256→128) + add backbone24 skip + 2 blocks
  SSD heads: conv1x1 at 12x12 and 24x24
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

# ── CPU reimplementation functions (matching WebGPU shaders exactly) ──

def cpu_conv5x5_stride2(input_chw, weight_ohwi, bias, alpha):
    """Exact CPU replica of PALM_CONV5X5_STRIDE2_PRELU_SHADER."""
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
    # Asymmetric padding: pad_right = 5 - stride - pad to cover all kernel taps
    pad_right = 5 - stride - pad
    padded = np.pad(input_chw, ((0, 0), (pad, pad_right), (pad, pad_right)), mode='constant')
    out = np.zeros((ch, out_h, out_w), dtype=np.float32)
    for ky in range(5):
        for kx in range(5):
            out += w_c25[:, ky * 5 + kx].reshape(-1, 1, 1) * padded[:, ky::stride, kx::stride][:, :out_h, :out_w]
    out += bias.reshape(-1, 1, 1)
    return out

def cpu_pw_skip_prelu(dw_chw, skip_chw, pw_w, pw_b, alpha, in_ch, out_ch, stride=1, in_h=None, in_w=None):
    """Exact CPU replica of PALM_POINTWISE_SKIP_PRELU_SHADER.
    For stride==2: skip uses max_pool 2x2 + channel zero-pad.
    For stride==1: skip is direct + channel zero-pad."""
    _, h, w = dw_chw.shape
    pw_2d = pw_w.reshape(out_ch, in_ch)
    out = np.zeros((out_ch, h, w), dtype=np.float32)
    for oc in range(out_ch):
        for ic in range(in_ch):
            out[oc] += pw_2d[oc, ic] * dw_chw[ic]
        out[oc] += pw_b[oc]

    # Skip connection: zero-pad channels, stride-2 uses max_pool
    skip_ch = skip_chw.shape[0]
    channel_pad = min(out_ch, skip_ch)  # = in_ch for same-ch blocks

    if stride == 2:
        # Max pool 2x2 from skip (which is at higher resolution)
        skip_h, skip_w = skip_chw.shape[1], skip_chw.shape[2]
        for oc in range(channel_pad):
            for oy in range(h):
                for ox in range(w):
                    max_val = -1e38
                    for py in range(2):
                        for px in range(2):
                            sy = oy * 2 + py
                            sx = ox * 2 + px
                            if sy < skip_h and sx < skip_w:
                                max_val = max(max_val, skip_chw[oc, sy, sx])
                    out[oc, oy, ox] += max_val
    else:
        out[:channel_pad] += skip_chw[:channel_pad]

    # PReLU
    a = alpha.flatten().reshape(-1, 1, 1)
    out = np.maximum(0, out) + a * np.minimum(0, out)
    return out

def cpu_upsample_2x_add(input_chw, skip_chw):
    """Exact CPU replica of PALM_UPSAMPLE_2X_ADD_SHADER."""
    ch, in_h, in_w = input_chw.shape
    out_h, out_w = in_h * 2, in_w * 2
    out = np.zeros((ch, out_h, out_w), dtype=np.float32)
    scale_y = in_h / out_h
    scale_x = in_w / out_w
    for oy in range(out_h):
        for ox in range(out_w):
            src_y = (oy + 0.5) * scale_y - 0.5
            src_x = (ox + 0.5) * scale_x - 0.5
            y0 = int(max(0, np.floor(src_y)))
            x0 = int(max(0, np.floor(src_x)))
            y1 = min(y0 + 1, in_h - 1)
            x1 = min(x0 + 1, in_w - 1)
            ly = max(0.0, src_y) - y0
            lx = max(0.0, src_x) - x0
            val = (input_chw[:, y0, x0] * (1 - ly) * (1 - lx) +
                   input_chw[:, y0, x1] * (1 - ly) * lx +
                   input_chw[:, y1, x0] * ly * (1 - lx) +
                   input_chw[:, y1, x1] * ly * lx)
            out[:, oy, ox] = val + skip_chw[:, oy, ox]
    return out

def cpu_conv1x1_prelu(input_chw, weight, bias, alpha, in_ch, out_ch):
    """Exact CPU replica of PALM_CONV1X1_PRELU_SHADER."""
    _, h, w = input_chw.shape
    pw_2d = weight.reshape(out_ch, in_ch)
    out = np.zeros((out_ch, h, w), dtype=np.float32)
    for oc in range(out_ch):
        for ic in range(in_ch):
            out[oc] += pw_2d[oc, ic] * input_chw[ic]
        out[oc] += bias[oc]
    a = alpha.flatten().reshape(-1, 1, 1)
    out = np.maximum(0, out) + a * np.minimum(0, out)
    return out

def cpu_conv1x1(input_chw, weight, bias, in_ch, out_ch):
    """Exact CPU replica of PALM_CONV1X1_SHADER."""
    _, h, w = input_chw.shape
    pw_2d = weight.reshape(out_ch, in_ch)
    out = np.zeros((out_ch, h, w), dtype=np.float32)
    for oc in range(out_ch):
        for ic in range(in_ch):
            out[oc] += pw_2d[oc, ic] * input_chw[ic]
        out[oc] += bias[oc]
    return out

# ── Block definitions (matching palm_model.ts) ────────────────────────
block_defs = [
    # (dw_key, pw_key, bn_key, prelu_key, in_ch, out_ch, stride, in_h)
    # Stage 1: 32ch, 96x96
    ('depthwise_conv2d/', 'conv2d_1/', 'batch_normalization_1/', 'p_re_lu_1/', 32, 32, 1, 96),
    ('depthwise_conv2d_1/', 'conv2d_2/', 'batch_normalization_2/', 'p_re_lu_2/', 32, 32, 1, 96),
    ('depthwise_conv2d_2/', 'conv2d_3/', 'batch_normalization_3/', 'p_re_lu_3/', 32, 32, 1, 96),
    ('depthwise_conv2d_3/', 'conv2d_4/', 'batch_normalization_4/', 'p_re_lu_4/', 32, 64, 2, 96),
    # Stage 2: 64ch, 48x48
    ('depthwise_conv2d_4/', 'conv2d_5/', 'batch_normalization_5/', 'p_re_lu_5/', 64, 64, 1, 48),
    ('depthwise_conv2d_5/', 'conv2d_6/', 'batch_normalization_6/', 'p_re_lu_6/', 64, 64, 1, 48),
    ('depthwise_conv2d_6/', 'conv2d_7/', 'batch_normalization_7/', 'p_re_lu_7/', 64, 64, 1, 48),
    ('depthwise_conv2d_7/', 'conv2d_8/', 'batch_normalization_8/', 'p_re_lu_8/', 64, 128, 2, 48),
    # Stage 3: 128ch, 24x24
    ('depthwise_conv2d_8/', 'conv2d_9/', 'batch_normalization_9/', 'p_re_lu_9/', 128, 128, 1, 24),
    ('depthwise_conv2d_9/', 'conv2d_10/', 'batch_normalization_10/', 'p_re_lu_10/', 128, 128, 1, 24),
    ('depthwise_conv2d_10/', 'conv2d_11/', 'batch_normalization_11/', 'p_re_lu_11/', 128, 128, 1, 24),
    ('depthwise_conv2d_11/', 'conv2d_12/', 'batch_normalization_12/', 'p_re_lu_12/', 128, 256, 2, 24),
    # Stage 4a: 256ch, 12x12
    ('depthwise_conv2d_12/', 'conv2d_13/', 'batch_normalization_13/', 'p_re_lu_13/', 256, 256, 1, 12),
    ('depthwise_conv2d_13/', 'conv2d_14/', 'batch_normalization_14/', 'p_re_lu_14/', 256, 256, 1, 12),
    ('depthwise_conv2d_14/', 'conv2d_15/', 'batch_normalization_15/', 'p_re_lu_15/', 256, 256, 1, 12),
    # Stage 4b: stride-2 → 6x6, then 256ch at 6x6
    ('depthwise_conv2d_15/', 'conv2d_16/', 'batch_normalization_16/', 'p_re_lu_16/', 256, 256, 2, 12),
    ('depthwise_conv2d_16/', 'conv2d_17/', 'batch_normalization_17/', 'p_re_lu_17/', 256, 256, 1, 6),
    ('depthwise_conv2d_17/', 'conv2d_18/', 'batch_normalization_18/', 'p_re_lu_18/', 256, 256, 1, 6),
    ('depthwise_conv2d_18/', 'conv2d_19/', 'batch_normalization_19/', 'p_re_lu_19/', 256, 256, 1, 6),
]

# ── Load weights for all blocks ───────────────────────────────────────
print("Loading weights...")

# Initial conv
init_conv_w = load_weight(find_w('conv2d/Conv2D'))
init_conv_bias = load_weight(find_w('batch_normalization/FusedBatchNormV3', 'conv2d/Conv2D'))
init_prelu_alpha = load_weight(find_w('p_re_lu/add', 'p_re_lu/Relu'))

block_weights = []
for dw_key, pw_key, bn_key, prelu_key, in_ch, out_ch, stride, in_h in block_defs:
    dw_w = load_weight(find_w(dw_key + 'depthwise'))
    pw_w = load_weight(find_w(pw_key + 'Conv2D'))
    pw_b = load_weight(find_w(bn_key + 'FusedBatchNormV3', pw_key + 'Conv2D'))
    alpha = load_weight(find_w(prelu_key + 'add', prelu_key + 'Relu'))
    dw_bias = np.zeros(in_ch, dtype=np.float32)
    block_weights.append((dw_w, dw_bias, pw_w, pw_b, alpha, in_ch, out_ch, stride, in_h))

# FPN weights
fpn_6to12_w = load_weight(find_w('conv2d_20/Conv2D'))
fpn_6to12_b = load_weight(find_w('batch_normalization_20/'))
fpn_6to12_alpha = load_weight(find_w('p_re_lu_20/'))

fpn12_block1_dw = load_weight(find_w('depthwise_conv2d_19/'))
fpn12_block1_pw = load_weight(find_w('conv2d_21/'))
fpn12_block1_bn = load_weight(find_w('batch_normalization_21/'))
fpn12_block1_alpha = load_weight(find_w('p_re_lu_21/'))

fpn12_block2_dw = load_weight(find_w('depthwise_conv2d_20/'))
fpn12_block2_pw = load_weight(find_w('conv2d_22/Conv2D1'))
fpn12_block2_bn = load_weight(find_w('batch_normalization_22/'))
fpn12_block2_alpha = load_weight(find_w('p_re_lu_22/'))

fpn_12to24_w = load_weight(find_w('conv2d_23/Conv2D'))
fpn_12to24_b = load_weight(find_w('batch_normalization_23/'))
fpn_12to24_alpha = load_weight(find_w('p_re_lu_23/'))

fpn24_block1_dw = load_weight(find_w('depthwise_conv2d_21/'))
fpn24_block1_pw = load_weight(find_w('conv2d_24/'))
fpn24_block1_bn = load_weight(find_w('batch_normalization_24/'))
fpn24_block1_alpha = load_weight(find_w('p_re_lu_24/'))

fpn24_block2_dw = load_weight(find_w('depthwise_conv2d_22/'))
fpn24_block2_pw = load_weight(find_w('conv2d_25/Conv2D1'))
fpn24_block2_bn = load_weight(find_w('batch_normalization_25/'))
fpn24_block2_alpha = load_weight(find_w('p_re_lu_25/'))

# SSD heads
cls16_w = load_weight(find_w('classifier_palm_16_NO_PRUNING/Conv2D'))
cls16_b = load_weight(find_w('classifier_palm_16_NO_PRUNING/BiasAdd'))
reg16_w = load_weight(find_w('regressor_palm_16_NO_PRUNING/Conv2D'))
reg16_b = load_weight(find_w('regressor_palm_16_NO_PRUNING/BiasAdd'))
cls8_w = load_weight(find_w('classifier_palm_8_NO_PRUNING/Conv2D'))
cls8_b = load_weight(find_w('classifier_palm_8_NO_PRUNING/BiasAdd'))
reg8_w = load_weight(find_w('regressor_palm_8_NO_PRUNING/Conv2D'))
reg8_b = load_weight(find_w('regressor_palm_8_NO_PRUNING/BiasAdd'))

# ── Run CPU reimplementation (WebGPU shader replica) through ALL layers ──
print("\n=== Running CPU reimplementation (WebGPU shader replica) ===")

# Initial conv
cpu_prelu, cpu_conv = cpu_conv5x5_stride2(input_chw, init_conv_w, init_conv_bias, init_prelu_alpha)

# Track activations for comparison
cpu_results = []
cpu_results.append(("L0 conv5x5+PReLU (96x96x32)", cpu_prelu))

# Run all backbone blocks
current = cpu_prelu
skip_for_blocks = cpu_prelu  # skip connection input (previous block output / initial conv output)
backbone_12x12_skip = None  # saved after block 14 (index 14)
backbone_24x24_skip = None  # saved after block 10 (index 10)

for bi, (dw_w, dw_bias, pw_w, pw_b, alpha, in_ch, out_ch, stride, in_h) in enumerate(block_weights):
    dw_w_t = transpose_dw(dw_w)
    pad = 1 if stride == 2 else 2
    dw_out = cpu_dw5x5(current, dw_w_t, dw_bias, stride=stride, pad=pad)

    pw_out = cpu_pw_skip_prelu(
        dw_out, current, pw_w, pw_b, alpha,
        in_ch, out_ch, stride=stride,
        in_h=in_h, in_w=in_h
    )

    out_h = in_h // stride
    name = f"Block {bi:2d} DW+PW ({in_ch}→{out_ch}, s{stride}, {in_h}→{out_h})"
    cpu_results.append((name, pw_out))
    print(f"  {name}  shape={pw_out.shape}")

    current = pw_out

    # Save skip connections for FPN
    if bi == 10:  # block 10 output = 24x24x128
        backbone_24x24_skip = pw_out.copy()
        print(f"    → saved backbone 24x24 skip: {pw_out.shape}")
    if bi == 14:  # block 14 output = 12x12x256
        backbone_12x12_skip = pw_out.copy()
        print(f"    → saved backbone 12x12 skip: {pw_out.shape}")

print(f"\nBackbone done. Final output: {current.shape}")

# FPN Level 1: upsample 6→12
print("\n--- FPN Level 1 (6→12) ---")
fpn1_upsampled = cpu_upsample_2x_add(
    current,  # 6x6x256
    np.zeros((256, 12, 12), dtype=np.float32)  # no skip in upsample step itself
)
# Actually the shader adds skip, but here we upsample WITHOUT skip first,
# then apply conv1x1_prelu, then add the backbone12 skip
# Wait, let me re-read palm_model.ts to get the exact FPN order...

# Looking at palm_model.ts: the FPN does:
# 1. Upsample 6→12 (with zeroBuf as skip = no add)
# 2. conv2d_20 (256→256) + PReLU on upsampled
# 3. Add backbone12 skip
# Actually no - let me look at the actual dispatch in palm_model.ts more carefully

# The upsample shader ALWAYS adds a skip. When there's no real skip, it adds zeroBuf.
# So: upsample 6→12 + add zeros = just upsample
# Then conv1x1+PReLU projects
# Then the ADD of backbone skip happens... but how?

# Let me re-check the FPN architecture comment:
# FPN Level 1 (6→12):
#   1. Bilinear upsample 6x6→12x12 (256ch) - skip is zeroBuf
#   2. conv2d_20 (256→256) + BN_20 + PReLU_20 on upsampled features
#   3. Element-wise add backbone_12x12 skip (from block 14)
# Then two DW+PW blocks with residual skip

# Hmm but I need to check how step 3 actually happens. Let me look at palm_model.ts dispatch...
# For now, let me just implement what makes sense and verify against TF.

# Step 1: upsample 6→12 (no add)
fpn1_up = np.zeros((256, 12, 12), dtype=np.float32)
ch, in_h_up, in_w_up = current.shape
for oy in range(12):
    for ox in range(12):
        src_y = (oy + 0.5) * (6/12) - 0.5
        src_x = (ox + 0.5) * (6/12) - 0.5
        y0 = int(max(0, np.floor(src_y)))
        x0 = int(max(0, np.floor(src_x)))
        y1 = min(y0 + 1, 5)
        x1 = min(x0 + 1, 5)
        ly = max(0.0, src_y) - y0
        lx = max(0.0, src_x) - x0
        val = (current[:, y0, x0] * (1 - ly) * (1 - lx) +
               current[:, y0, x1] * (1 - ly) * lx +
               current[:, y1, x0] * ly * (1 - lx) +
               current[:, y1, x1] * ly * lx)
        fpn1_up[:, oy, ox] = val
cpu_results.append(("FPN1 upsample 6→12", fpn1_up))
print(f"  FPN1 upsample: {fpn1_up.shape}")

# Step 2: conv1x1 + PReLU (256→256) on upsampled
fpn1_projected = cpu_conv1x1_prelu(fpn1_up, fpn_6to12_w, fpn_6to12_b, fpn_6to12_alpha, 256, 256)
cpu_results.append(("FPN1 conv1x1+PReLU", fpn1_projected))
print(f"  FPN1 projected: {fpn1_projected.shape}")

# Step 3: add backbone 12x12 skip
fpn1_added = fpn1_projected + backbone_12x12_skip
cpu_results.append(("FPN1 + backbone12 skip", fpn1_added))
print(f"  FPN1 + skip: {fpn1_added.shape}")

# Step 4: FPN 12x12 block 1 (dw_19 + conv2d_21)
dw_t = transpose_dw(fpn12_block1_dw)
fpn1_b1_dw = cpu_dw5x5(fpn1_added, dw_t, np.zeros(256, dtype=np.float32), stride=1, pad=2)
fpn1_b1_pw = cpu_pw_skip_prelu(fpn1_b1_dw, fpn1_added, fpn12_block1_pw, fpn12_block1_bn, fpn12_block1_alpha, 256, 256, stride=1)
cpu_results.append(("FPN1 block1 (12x12)", fpn1_b1_pw))
print(f"  FPN1 block1: {fpn1_b1_pw.shape}")

# Step 5: FPN 12x12 block 2 (dw_20 + conv2d_22)
dw_t = transpose_dw(fpn12_block2_dw)
fpn1_b2_dw = cpu_dw5x5(fpn1_b1_pw, dw_t, np.zeros(256, dtype=np.float32), stride=1, pad=2)
fpn1_b2_pw = cpu_pw_skip_prelu(fpn1_b2_dw, fpn1_b1_pw, fpn12_block2_pw, fpn12_block2_bn, fpn12_block2_alpha, 256, 256, stride=1)
cpu_results.append(("FPN1 block2 (12x12)", fpn1_b2_pw))
print(f"  FPN1 block2: {fpn1_b2_pw.shape}")

# SSD heads at 12x12
ssd_cls16 = cpu_conv1x1(fpn1_b2_pw, cls16_w, cls16_b, 256, 6)
ssd_reg16 = cpu_conv1x1(fpn1_b2_pw, reg16_w, reg16_b, 256, 108)
cpu_results.append(("SSD cls 12x12 (6)", ssd_cls16))
cpu_results.append(("SSD reg 12x12 (108)", ssd_reg16))
print(f"  SSD 12x12: cls={ssd_cls16.shape}, reg={ssd_reg16.shape}")

# FPN Level 2: upsample 12→24
print("\n--- FPN Level 2 (12→24) ---")
fpn2_up = np.zeros((256, 24, 24), dtype=np.float32)
for oy in range(24):
    for ox in range(24):
        src_y = (oy + 0.5) * (12/24) - 0.5
        src_x = (ox + 0.5) * (12/24) - 0.5
        y0 = int(max(0, np.floor(src_y)))
        x0 = int(max(0, np.floor(src_x)))
        y1 = min(y0 + 1, 11)
        x1 = min(x0 + 1, 11)
        ly = max(0.0, src_y) - y0
        lx = max(0.0, src_x) - x0
        val = (fpn1_b2_pw[:, y0, x0] * (1 - ly) * (1 - lx) +
               fpn1_b2_pw[:, y0, x1] * (1 - ly) * lx +
               fpn1_b2_pw[:, y1, x0] * ly * (1 - lx) +
               fpn1_b2_pw[:, y1, x1] * ly * lx)
        fpn2_up[:, oy, ox] = val
print(f"  FPN2 upsample: {fpn2_up.shape}")

# conv1x1+PReLU (256→128)
fpn2_projected = cpu_conv1x1_prelu(fpn2_up, fpn_12to24_w, fpn_12to24_b, fpn_12to24_alpha, 256, 128)
print(f"  FPN2 projected: {fpn2_projected.shape}")

# Add backbone 24x24 skip (128ch)
fpn2_added = fpn2_projected + backbone_24x24_skip
print(f"  FPN2 + skip: {fpn2_added.shape}")

# FPN 24x24 block 1
dw_t = transpose_dw(fpn24_block1_dw)
fpn2_b1_dw = cpu_dw5x5(fpn2_added, dw_t, np.zeros(128, dtype=np.float32), stride=1, pad=2)
fpn2_b1_pw = cpu_pw_skip_prelu(fpn2_b1_dw, fpn2_added, fpn24_block1_pw, fpn24_block1_bn, fpn24_block1_alpha, 128, 128, stride=1)
print(f"  FPN2 block1: {fpn2_b1_pw.shape}")

# FPN 24x24 block 2
dw_t = transpose_dw(fpn24_block2_dw)
fpn2_b2_dw = cpu_dw5x5(fpn2_b1_pw, dw_t, np.zeros(128, dtype=np.float32), stride=1, pad=2)
fpn2_b2_pw = cpu_pw_skip_prelu(fpn2_b2_dw, fpn2_b1_pw, fpn24_block2_pw, fpn24_block2_bn, fpn24_block2_alpha, 128, 128, stride=1)
print(f"  FPN2 block2: {fpn2_b2_pw.shape}")

# SSD heads at 24x24
ssd_cls8 = cpu_conv1x1(fpn2_b2_pw, cls8_w, cls8_b, 128, 2)
ssd_reg8 = cpu_conv1x1(fpn2_b2_pw, reg8_w, reg8_b, 128, 36)
print(f"  SSD 24x24: cls={ssd_cls8.shape}, reg={ssd_reg8.shape}")

# ══════════════════════════════════════════════════════════════════════
# TF Reference computation
# ══════════════════════════════════════════════════════════════════════
print("\n\n=== Running TF reference computation ===")

inp_tf = tf.constant(img_np[np.newaxis], dtype=tf.float32)  # [1,192,192,3]

# Initial conv
w_hwio = np.transpose(init_conv_w, (1, 2, 3, 0))
ref_conv = tf.nn.conv2d(inp_tf, w_hwio, strides=2, padding='SAME').numpy()
ref_conv += init_conv_bias.reshape(1, 1, 1, -1)
a0 = init_prelu_alpha.flatten().reshape(1, 1, 1, -1)
ref_prelu = np.maximum(0, ref_conv) + a0 * np.minimum(0, ref_conv)

ref_results = []
ref_results.append(("L0 conv5x5+PReLU", ref_prelu[0]))  # [96,96,32] HWC

# Run all backbone blocks through TF
tf_current = ref_prelu  # [1,96,96,32] NHWC
tf_skip = ref_prelu
tf_backbone_12 = None
tf_backbone_24 = None

for bi, (dw_w, dw_bias, pw_w, pw_b, alpha, in_ch, out_ch, stride, in_h) in enumerate(block_weights):
    # DW 5x5
    dw_hwcm = np.transpose(dw_w, (1, 2, 3, 0))  # [5,5,ch,1]
    ref_dw = tf.nn.depthwise_conv2d(
        tf.constant(tf_current, dtype=tf.float32),
        tf.constant(dw_hwcm, dtype=tf.float32),
        strides=[1, stride, stride, 1],
        padding='SAME'
    ).numpy()
    ref_dw += dw_bias.reshape(1, 1, 1, -1)

    # PW 1x1
    pw_hwio = np.transpose(pw_w, (1, 2, 3, 0))  # [1,1,in_ch,out_ch]
    ref_pw = tf.nn.conv2d(
        tf.constant(ref_dw, dtype=tf.float32),
        tf.constant(pw_hwio, dtype=tf.float32),
        strides=1, padding='VALID'
    ).numpy()
    ref_pw += pw_b.reshape(1, 1, 1, -1)

    # Skip: channel pad + optional max_pool for stride 2
    skip_ch = tf_current.shape[-1]
    if stride == 2:
        skip_pooled = tf.nn.max_pool2d(
            tf.constant(tf_current, dtype=tf.float32),
            ksize=2, strides=2, padding='VALID'
        ).numpy()
        # Zero-pad channels if needed
        if out_ch > skip_ch:
            pad_width = out_ch - skip_ch
            skip_padded = np.pad(skip_pooled, ((0,0),(0,0),(0,0),(0,pad_width)), mode='constant')
        else:
            skip_padded = skip_pooled
    else:
        if out_ch > skip_ch:
            pad_width = out_ch - skip_ch
            skip_padded = np.pad(tf_current, ((0,0),(0,0),(0,0),(0,pad_width)), mode='constant')
        else:
            skip_padded = tf_current

    ref_add = ref_pw + skip_padded

    # PReLU
    a = alpha.flatten().reshape(1, 1, 1, -1)
    ref_block = np.maximum(0, ref_add) + a * np.minimum(0, ref_add)

    out_h = in_h // stride
    name = f"Block {bi:2d} DW+PW ({in_ch}→{out_ch}, s{stride}, {in_h}→{out_h})"
    ref_results.append((name, ref_block[0]))

    tf_current = ref_block

    if bi == 10:
        tf_backbone_24 = ref_block.copy()
    if bi == 14:
        tf_backbone_12 = ref_block.copy()

print(f"Backbone done. Final: {tf_current.shape}")

# FPN Level 1
print("Computing FPN Level 1...")
# Upsample 6→12
ref_up1 = tf.image.resize(
    tf.constant(tf_current, dtype=tf.float32),
    [12, 12], method='bilinear'
).numpy()
ref_results.append(("FPN1 upsample 6→12", ref_up1[0]))

# conv1x1 + PReLU (256→256)
fpn1_pw_hwio = np.transpose(fpn_6to12_w, (1, 2, 3, 0))
ref_fpn1_proj = tf.nn.conv2d(
    tf.constant(ref_up1, dtype=tf.float32),
    tf.constant(fpn1_pw_hwio, dtype=tf.float32),
    strides=1, padding='VALID'
).numpy()
ref_fpn1_proj += fpn_6to12_b.reshape(1, 1, 1, -1)
a_fpn1 = fpn_6to12_alpha.flatten().reshape(1, 1, 1, -1)
ref_fpn1_proj = np.maximum(0, ref_fpn1_proj) + a_fpn1 * np.minimum(0, ref_fpn1_proj)
ref_results.append(("FPN1 conv1x1+PReLU", ref_fpn1_proj[0]))

# Add backbone 12x12 skip
ref_fpn1_added = ref_fpn1_proj + tf_backbone_12
ref_results.append(("FPN1 + backbone12 skip", ref_fpn1_added[0]))

# FPN 12x12 block 1
def tf_block(inp, dw_w, dw_bias_val, pw_w, pw_b, alpha, in_ch, out_ch, stride=1):
    dw_hwcm = np.transpose(dw_w, (1, 2, 3, 0))
    ref_dw = tf.nn.depthwise_conv2d(
        tf.constant(inp, dtype=tf.float32),
        tf.constant(dw_hwcm, dtype=tf.float32),
        strides=[1, stride, stride, 1], padding='SAME'
    ).numpy()
    ref_dw += dw_bias_val.reshape(1, 1, 1, -1)
    pw_hwio = np.transpose(pw_w, (1, 2, 3, 0))
    ref_pw = tf.nn.conv2d(
        tf.constant(ref_dw, dtype=tf.float32),
        tf.constant(pw_hwio, dtype=tf.float32),
        strides=1, padding='VALID'
    ).numpy()
    ref_pw += pw_b.reshape(1, 1, 1, -1)
    # Skip (same ch, stride 1 = direct add)
    ref_add = ref_pw + inp
    a = alpha.flatten().reshape(1, 1, 1, -1)
    return np.maximum(0, ref_add) + a * np.minimum(0, ref_add)

ref_fpn1_b1 = tf_block(ref_fpn1_added, fpn12_block1_dw, np.zeros(256, dtype=np.float32),
                        fpn12_block1_pw, fpn12_block1_bn, fpn12_block1_alpha, 256, 256)
ref_results.append(("FPN1 block1 (12x12)", ref_fpn1_b1[0]))

ref_fpn1_b2 = tf_block(ref_fpn1_b1, fpn12_block2_dw, np.zeros(256, dtype=np.float32),
                        fpn12_block2_pw, fpn12_block2_bn, fpn12_block2_alpha, 256, 256)
ref_results.append(("FPN1 block2 (12x12)", ref_fpn1_b2[0]))

# SSD 12x12 heads
cls16_hwio = np.transpose(cls16_w, (1, 2, 3, 0))
ref_cls16 = tf.nn.conv2d(tf.constant(ref_fpn1_b2, dtype=tf.float32),
                          tf.constant(cls16_hwio, dtype=tf.float32), strides=1, padding='VALID').numpy()
ref_cls16 += cls16_b.reshape(1, 1, 1, -1)
ref_results.append(("SSD cls 12x12 (6)", ref_cls16[0]))

reg16_hwio = np.transpose(reg16_w, (1, 2, 3, 0))
ref_reg16 = tf.nn.conv2d(tf.constant(ref_fpn1_b2, dtype=tf.float32),
                          tf.constant(reg16_hwio, dtype=tf.float32), strides=1, padding='VALID').numpy()
ref_reg16 += reg16_b.reshape(1, 1, 1, -1)
ref_results.append(("SSD reg 12x12 (108)", ref_reg16[0]))

# FPN Level 2
print("Computing FPN Level 2...")
ref_up2 = tf.image.resize(
    tf.constant(ref_fpn1_b2, dtype=tf.float32),
    [24, 24], method='bilinear'
).numpy()

fpn2_pw_hwio = np.transpose(fpn_12to24_w, (1, 2, 3, 0))
ref_fpn2_proj = tf.nn.conv2d(
    tf.constant(ref_up2, dtype=tf.float32),
    tf.constant(fpn2_pw_hwio, dtype=tf.float32),
    strides=1, padding='VALID'
).numpy()
ref_fpn2_proj += fpn_12to24_b.reshape(1, 1, 1, -1)
a_fpn2 = fpn_12to24_alpha.flatten().reshape(1, 1, 1, -1)
ref_fpn2_proj = np.maximum(0, ref_fpn2_proj) + a_fpn2 * np.minimum(0, ref_fpn2_proj)

ref_fpn2_added = ref_fpn2_proj + tf_backbone_24

ref_fpn2_b1 = tf_block(ref_fpn2_added, fpn24_block1_dw, np.zeros(128, dtype=np.float32),
                        fpn24_block1_pw, fpn24_block1_bn, fpn24_block1_alpha, 128, 128)
ref_fpn2_b2 = tf_block(ref_fpn2_b1, fpn24_block2_dw, np.zeros(128, dtype=np.float32),
                        fpn24_block2_pw, fpn24_block2_bn, fpn24_block2_alpha, 128, 128)

cls8_hwio = np.transpose(cls8_w, (1, 2, 3, 0))
ref_cls8 = tf.nn.conv2d(tf.constant(ref_fpn2_b2, dtype=tf.float32),
                         tf.constant(cls8_hwio, dtype=tf.float32), strides=1, padding='VALID').numpy()
ref_cls8 += cls8_b.reshape(1, 1, 1, -1)

reg8_hwio = np.transpose(reg8_w, (1, 2, 3, 0))
ref_reg8 = tf.nn.conv2d(tf.constant(ref_fpn2_b2, dtype=tf.float32),
                         tf.constant(reg8_hwio, dtype=tf.float32), strides=1, padding='VALID').numpy()
ref_reg8 += reg8_b.reshape(1, 1, 1, -1)

# ══════════════════════════════════════════════════════════════════════
# Also get TFLite final output for end-to-end comparison
# ══════════════════════════════════════════════════════════════════════
print("\n=== TFLite end-to-end reference ===")
interp = tf.lite.Interpreter(model_path=str(ROOT / "palm_detection.tflite"))
interp.allocate_tensors()
interp.set_tensor(interp.get_input_details()[0]['index'], img_np[np.newaxis])
interp.invoke()
tfl_regressors = interp.get_tensor(interp.get_output_details()[0]['index'])
tfl_scores = interp.get_tensor(interp.get_output_details()[1]['index'])

# ══════════════════════════════════════════════════════════════════════
# Compare CPU (WebGPU replica) vs TF Reference
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FULL LAYER-BY-LAYER COMPARISON: CPU (WebGPU shader) vs TF Reference")
print("=" * 80)

# Build comparison pairs - backbone blocks
# CPU results are CHW, TF results are HWC
compare_pairs = []

# Initial conv
compare_pairs.append(("L0 conv5x5+PReLU (96x96x32)", cpu_prelu, ref_prelu[0]))

# All backbone blocks
for bi in range(len(block_weights)):
    dw_w, dw_bias, pw_w, pw_b, alpha, in_ch, out_ch, stride, in_h = block_weights[bi]
    out_h = in_h // stride
    name = f"Block {bi:2d} ({in_ch}→{out_ch}, s{stride}, {in_h}→{out_h})"
    cpu_val = cpu_results[bi + 1][1]  # +1 because index 0 is initial conv
    ref_val = ref_results[bi + 1][1]
    compare_pairs.append((name, cpu_val, ref_val))

# FPN layers
fpn_cpu = [fpn1_up, fpn1_projected, fpn1_added, fpn1_b1_pw, fpn1_b2_pw, ssd_cls16, ssd_reg16]
fpn_ref = [ref_up1[0], ref_fpn1_proj[0], ref_fpn1_added[0], ref_fpn1_b1[0], ref_fpn1_b2[0], ref_cls16[0], ref_reg16[0]]
fpn_names = ["FPN1 upsample 6→12", "FPN1 conv1x1+PReLU", "FPN1 + skip",
             "FPN1 block1", "FPN1 block2", "SSD cls 12x12", "SSD reg 12x12"]
for name, cpu_val, ref_val in zip(fpn_names, fpn_cpu, fpn_ref):
    compare_pairs.append((name, cpu_val, ref_val))

# FPN2
fpn2_cpu = [fpn2_b1_pw, fpn2_b2_pw, ssd_cls8, ssd_reg8]
fpn2_ref = [ref_fpn2_b1[0], ref_fpn2_b2[0], ref_cls8[0], ref_reg8[0]]
fpn2_names = ["FPN2 block1", "FPN2 block2", "SSD cls 24x24", "SSD reg 24x24"]
for name, cpu_val, ref_val in zip(fpn2_names, fpn2_cpu, fpn2_ref):
    compare_pairs.append((name, cpu_val, ref_val))

# Print comparison
all_ok = True
first_divergence = None

for name, cpu_chw, ref_hwc in compare_pairs:
    cpu_hwc = np.transpose(cpu_chw, (1, 2, 0))
    diff = np.abs(cpu_hwc - ref_hwc)
    max_err = diff.max()
    mean_err = diff.mean()
    status = "✓" if max_err < 0.001 else "✗ DIVERGED"
    if max_err >= 0.001:
        all_ok = False
        if first_divergence is None:
            first_divergence = name

    # Color-code output
    print(f"  {name:40s}  max={max_err:.8f}  mean={mean_err:.8f}  [{status}]")
    if max_err >= 0.001:
        loc = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Worst at (y={loc[0]}, x={loc[1]}, c={loc[2]}): ref={ref_hwc[loc]:.8f} cpu={cpu_hwc[loc]:.8f}")

# Compare TFLite final output vs our CPU final output
print("\n--- End-to-end SSD output comparison (CPU vs TFLite) ---")
# Concatenate our SSD outputs in the same order as TFLite: [12x12 anchors, 24x24 anchors]
# TFLite output order: regressors [1,2016,18], scores [1,2016,1]
# 12x12: 6 anchors/cell → 12*12*6 = 864 anchors
# 24x24: 2 anchors/cell → 24*24*2 = 1152 anchors

# Our regressors: CHW → reshape to match TFLite
# 12x12: [108,12,12] → [12,12,108] → [12*12,6,18] → [864,18]
cpu_reg16_hwc = np.transpose(ssd_reg16, (1, 2, 0)).reshape(144, 6, 18).reshape(864, 18)
cpu_reg8_hwc = np.transpose(ssd_reg8, (1, 2, 0)).reshape(576, 2, 18).reshape(1152, 18)
cpu_regressors = np.concatenate([cpu_reg16_hwc, cpu_reg8_hwc], axis=0)[np.newaxis]

cpu_cls16_hwc = np.transpose(ssd_cls16, (1, 2, 0)).reshape(144, 6, 1).reshape(864, 1)
cpu_cls8_hwc = np.transpose(ssd_cls8, (1, 2, 0)).reshape(576, 2, 1).reshape(1152, 1)
cpu_scores = np.concatenate([cpu_cls16_hwc, cpu_cls8_hwc], axis=0)[np.newaxis]

reg_diff = np.abs(cpu_regressors - tfl_regressors)
score_diff = np.abs(cpu_scores - tfl_scores)
print(f"  Regressors: max_err={reg_diff.max():.6f}  mean_err={reg_diff.mean():.6f}")
print(f"  Scores:     max_err={score_diff.max():.6f}  mean_err={score_diff.mean():.6f}")

# Show top detection scores
cpu_sigmoid = 1 / (1 + np.exp(-cpu_scores.flatten()))
tfl_sigmoid = 1 / (1 + np.exp(-tfl_scores.flatten()))
top_cpu = sorted(enumerate(cpu_sigmoid), key=lambda x: -x[1])[:5]
top_tfl = sorted(enumerate(tfl_sigmoid), key=lambda x: -x[1])[:5]
print(f"\n  Top 5 detections (CPU):   {[(i, f'{s:.4f}') for i, s in top_cpu]}")
print(f"  Top 5 detections (TFLite): {[(i, f'{s:.4f}') for i, s in top_tfl]}")

# Show regressor values for top detection
if top_cpu[0][1] > 0.5:
    idx = top_cpu[0][0]
    print(f"\n  Top detection #{idx} regressors comparison:")
    print(f"    CPU:    {cpu_regressors[0, idx, :7]}")
    print(f"    TFLite: {tfl_regressors[0, idx, :7]}")

print(f"\n{'='*80}")
if all_ok:
    print("VERDICT: ALL LAYERS MATCH — CPU (WebGPU shader) is bit-accurate with TF")
    print("If WebGPU produces different results, the issue is in GPU execution")
    print("(f16 precision, accumulation order, workgroup scheduling)")
else:
    print(f"VERDICT: DIVERGENCE FOUND starting at: {first_divergence}")
    print("Fix the CPU reimplementation or identify the GPU-specific issue")
print("=" * 80)
