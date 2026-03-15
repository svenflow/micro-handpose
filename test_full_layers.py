#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""Compare TFLite intermediate layer outputs with our weight-based numpy reference."""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "hand_landmark_full_from_task.tflite"
WEIGHTS_DIR = SCRIPT_DIR / "weights"

# Load model
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

# Create synthetic input (same as WebGPU test)
img = np.full((1, 224, 224, 3), [136/255, 102/255, 68/255], dtype=np.float32)
img[0, 50:174, 50:174, :] = [204/255, 153/255, 102/255]

# Set input and run
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# Get ALL tensor values
details = interpreter.get_tensor_details()

# Find key intermediate tensors by analyzing the graph
print("=" * 80)
print("All tensors with their shapes and stats (non-zero tensors only)")
print("=" * 80)

for d in details:
    try:
        t = interpreter.get_tensor(d['index'])
        if t.size > 0 and np.any(t != 0):
            name = d['name']
            shape = tuple(d['shape'])
            # Only show 4D (activations) and small tensors (outputs)
            if len(shape) == 4 and shape[0] == 1 and shape[1] > 1 and shape[2] > 1:
                # Activation tensor NHWC
                print(f"  idx={d['index']:3d}  {name[:80]:<80s} {str(shape):<25s}  min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")
    except:
        pass

# Get outputs
output_details = interpreter.get_output_details()
for od in output_details:
    t = interpreter.get_tensor(od['index'])
    print(f"\nOutput: {od['name']} shape={tuple(od['shape'])} values={t.flatten()[:6]}")

# Now run our numpy reference through the blocks and compare
print("\n" + "=" * 80)
print("Block-by-block comparison: TFLite vs Numpy with extracted weights")
print("=" * 80)

meta = json.loads((WEIGHTS_DIR / "weights_f16_full.json").read_text())
bin_data = (WEIGHTS_DIR / "weights_f16_full.bin").read_bytes()

# Build weight map with duplicate handling
weight_map = {}
key_counts = {}
for i in range(len(meta['keys'])):
    base_key = meta['keys'][i]
    shape = meta['shapes'][i]
    offset = meta['offsets'][i]
    size = 1
    for s in shape:
        size *= s
    raw = bin_data[offset:offset + size * 2]
    data = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    
    count = key_counts.get(base_key, 0)
    key_counts[base_key] = count + 1
    key = base_key if count == 0 else f"{base_key}__{count}"
    weight_map[key] = data

def conv2d_nchw(inp, weight, bias, stride=1, pad=0):
    """Conv2d in NCHW format. weight: [Cout, Cin, kH, kW]"""
    cout, cin, kh, kw = weight.shape
    h, w = inp.shape[1], inp.shape[2]
    
    if pad > 0:
        padded = np.zeros((cin, h + 2*pad, w + 2*pad), dtype=np.float32)
        padded[:, pad:pad+h, pad:pad+w] = inp
        inp = padded
        h, w = h + 2*pad, w + 2*pad
    
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((cout, out_h, out_w), dtype=np.float32)
    
    for oc in range(cout):
        for oy in range(out_h):
            for ox in range(out_w):
                iy = oy * stride
                ix = ox * stride
                patch = inp[:, iy:iy+kh, ix:ix+kw]
                output[oc, oy, ox] = np.sum(patch * weight[oc]) + bias[oc]
    return output

def dw_conv_nchw(inp, weight, bias, stride=1, pad=0, kernel=3):
    """Depthwise conv in NCHW. weight: [C, 1, kH, kW]"""
    ch = inp.shape[0]
    h, w = inp.shape[1], inp.shape[2]
    
    if pad > 0:
        padded = np.zeros((ch, h + 2*pad, w + 2*pad), dtype=np.float32)
        padded[:, pad:pad+h, pad:pad+w] = inp
        inp = padded
        h, w = h + 2*pad, w + 2*pad
    
    out_h = (h - kernel) // stride + 1
    out_w = (w - kernel) // stride + 1
    output = np.zeros((ch, out_h, out_w), dtype=np.float32)
    
    for c in range(ch):
        for oy in range(out_h):
            for ox in range(out_w):
                iy = oy * stride
                ix = ox * stride
                patch = inp[c, iy:iy+kernel, ix:ix+kernel]
                output[c, oy, ox] = np.sum(patch * weight[c, 0]) + bias[c]
    return output

def relu6(x):
    return np.clip(x, 0, 6)

# Convert input to NCHW
x_nchw = img[0].transpose(2, 0, 1)  # [3, 224, 224]

# Initial conv: 3x3 stride 2 with SAME padding
# SAME padding for stride 2, kernel 3: pad = 1 (asymmetric)
init_w = weight_map['conv2d']  # [24, 3, 3, 3]
init_b = weight_map['batch_normalization']  # [24]
x = conv2d_nchw(x_nchw, init_w, init_b, stride=2, pad=1)
x = relu6(x)
print(f"\nInitial conv output: shape={x.shape}  min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f}")

# Find TFLite's first activation (should be 1x112x112x24)
for d in details:
    try:
        t = interpreter.get_tensor(d['index'])
        shape = tuple(d['shape'])
        if shape == (1, 112, 112, 24) and np.any(t != 0):
            tf_x = t[0].transpose(2, 0, 1)  # NHWC -> CHW
            diff = np.abs(x - tf_x)
            print(f"  vs TFLite {d['name'][:60]}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")
            if diff.max() > 0.01:
                print(f"  MISMATCH! Sample ours: {x[0,0,:5]} tflite: {tf_x[0,0,:5]}")
            break
    except:
        pass

# Now go through blocks
BLOCK_SPECS = [
    (24, 24, 3, 1, 16, False), # Block 1: no expand
    (16, 64, 3, 2, 24, False),
    (24, 144, 3, 1, 24, True),
    (24, 144, 5, 2, 40, False),
    (40, 240, 5, 1, 40, True),
    (40, 240, 3, 2, 80, False),
    (80, 480, 3, 1, 80, True),
    (80, 480, 3, 1, 80, True),
    (80, 480, 5, 1, 112, False),
    (112, 672, 5, 1, 112, True),
    (112, 672, 5, 1, 112, True),
    (112, 672, 5, 2, 192, False),
    (192, 1152, 5, 1, 192, True),
    (192, 1152, 5, 1, 192, True),
    (192, 1152, 5, 1, 192, True),
    (192, 1152, 3, 1, 1152, False), # Block 16: no project
]

BLOCK_NAMES = [
    {'dw': 'batch_normalization_1/FusedBatchNormV3', 'dwbn': 'batch_normalization_1', 'pw': 'conv2d_1', 'pwbn': 'batch_normalization_2/FusedBatchNormV3'},
    {'ex': 'conv2d_2', 'exbn': 'batch_normalization_3', 'dw': 'batch_normalization_4/FusedBatchNormV3', 'dwbn': 'batch_normalization_4', 'pw': 'conv2d_3', 'pwbn': 'batch_normalization_5/FusedBatchNormV3'},
    {'ex': 'conv2d_4', 'exbn': 'batch_normalization_6', 'dw': 'batch_normalization_7/FusedBatchNormV3', 'dwbn': 'batch_normalization_7', 'pw': 'conv2d_5', 'pwbn': 'batch_normalization_8/FusedBatchNormV3'},
    {'ex': 'conv2d_6', 'exbn': 'batch_normalization_9', 'dw': 'batch_normalization_10/FusedBatchNormV3', 'dwbn': 'batch_normalization_10', 'pw': 'conv2d_7', 'pwbn': 'batch_normalization_11/FusedBatchNormV3'},
    {'ex': 'conv2d_8', 'exbn': 'batch_normalization_12', 'dw': 'batch_normalization_13/FusedBatchNormV3', 'dwbn': 'batch_normalization_13', 'pw': 'conv2d_9', 'pwbn': 'batch_normalization_14/FusedBatchNormV3'},
]

for bi, (inCh, expandCh, dwK, stride, outCh, hasRes) in enumerate(BLOCK_SPECS[:5]):
    if bi >= len(BLOCK_NAMES):
        break
    names = BLOCK_NAMES[bi]
    residual = x.copy() if hasRes else None
    
    # Expand (if expandCh != inCh)
    if 'ex' in names:
        ew = weight_map[names['ex']]
        eb = weight_map[names['exbn']]
        x = conv2d_nchw(x, ew, eb)
        x = relu6(x)
    
    # DW conv
    dww = weight_map[names['dw']]
    dwb = weight_map[names['dwbn']]
    pad = dwK // 2
    x = dw_conv_nchw(x, dww, dwb, stride=stride, pad=pad, kernel=dwK)
    x = relu6(x)
    
    # Project
    if bi < 15:  # Not block 16
        pw = weight_map[names['pw']]
        pb = weight_map[names['pwbn']]
        x = conv2d_nchw(x, pw, pb)
        # NO activation for project
    
    # Residual
    if hasRes and residual is not None:
        x = x + residual
    
    print(f"Block {bi+1}: shape={x.shape}  min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f}  (expand={expandCh}, dw={dwK}x{dwK}/s{stride}, out={outCh}, res={hasRes})")
