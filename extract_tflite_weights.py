#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Extract weights from the official MediaPipe hand_landmark TFLite model
and convert them to the format used by micro-handpose (NCHW numpy arrays
with PyTorch-style layer names).

Downloads hand_landmark.tflite from Google's MediaPipe assets if not present,
then maps each TFLite tensor to the corresponding micro-handpose weight name,
transposes NHWC -> NCHW where needed, and saves the result.

Also compares extracted weights against the existing vidursatija/BlazePalm
weights to quantify differences.
"""

import json
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "hand_landmark.tflite"
TFLITE_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmark.tflite"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
OUTPUT_NPZ = SCRIPT_DIR / "weights_tflite.npz"
OUTPUT_BIN = WEIGHTS_DIR / "weights_tflite_f16.bin"
OUTPUT_JSON = WEIGHTS_DIR / "weights_tflite_f16.json"


# ---------------------------------------------------------------------------
# Step 1: Download the TFLite model if needed
# ---------------------------------------------------------------------------
def download_model():
    if TFLITE_PATH.exists():
        print(f"TFLite model already exists: {TFLITE_PATH}")
        return
    print(f"Downloading {TFLITE_URL} ...")
    urllib.request.urlretrieve(TFLITE_URL, TFLITE_PATH)
    print(f"Saved to {TFLITE_PATH} ({TFLITE_PATH.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Step 2: Load TFLite model and extract tensors
# ---------------------------------------------------------------------------
def load_tflite_weights():
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    weights = {}
    for d in details:
        name = d["name"]
        if "_dequantize" in name:
            continue
        if not ("Kernel" in name or "Bias" in name):
            continue
        tensor = interpreter.get_tensor(d["index"]).astype(np.float32)
        weights[name] = tensor
    return weights


# ---------------------------------------------------------------------------
# Step 3: Build the mapping from TFLite names -> micro-handpose names
# ---------------------------------------------------------------------------
def build_mapping():
    """
    Returns list of (our_name, tflite_name, kind) tuples.

    kind is one of:
        'conv3x3'   - regular 3x3 conv (input layer), NHWC kernel [O,H,W,I]
        'depthwise' - depthwise conv, NHWC kernel [1,H,W,C]
        'pointwise' - 1x1 conv, NHWC kernel [O,1,1,I]
        'output_conv' - output head conv (2x2), NHWC kernel [O,H,W,I]
        'bias'      - bias vector (no transpose needed)
    """
    mapping = []

    # Input conv: conv2d (index 0) -> backbone1.1
    mapping.append(("backbone1.1.weight", "conv2d/Kernel", "conv3x3"))
    mapping.append(("backbone1.1.bias", "conv2d/Bias", "bias"))

    # Backbone layers: each ResModule = depthwise + pointwise
    backbone_dw = [
        "backbone1.3.f.0.convs.0", "backbone1.3.f.1.convs.0", "backbone1.4.convs.0",
        "backbone2.0.f.0.convs.0", "backbone2.0.f.1.convs.0", "backbone2.1.convs.0",
        "backbone3.0.f.0.convs.0", "backbone3.0.f.1.convs.0", "backbone3.1.convs.0",
        "backbone4.0.f.0.convs.0", "backbone4.0.f.1.convs.0", "backbone4.1.convs.0",
        "backbone5.0.convs.0",
        "backbone6.0.convs.0",
    ]
    backbone_pw = [
        "backbone1.3.f.0.convs.1", "backbone1.3.f.1.convs.1", "backbone1.4.convs.1",
        "backbone2.0.f.0.convs.1", "backbone2.0.f.1.convs.1", "backbone2.1.convs.1",
        "backbone3.0.f.0.convs.1", "backbone3.0.f.1.convs.1", "backbone3.1.convs.1",
        "backbone4.0.f.0.convs.1", "backbone4.0.f.1.convs.1", "backbone4.1.convs.1",
        "backbone5.0.convs.1",
        "backbone6.0.convs.1",
    ]

    dw_idx = 0
    conv_idx = 1  # conv2d_0 was the input conv

    for i in range(len(backbone_dw)):
        dw_name = "depthwise_conv2d" if dw_idx == 0 else f"depthwise_conv2d_{dw_idx}"
        conv_name = f"conv2d_{conv_idx}"

        mapping.append((f"{backbone_dw[i]}.weight", f"{dw_name}/Kernel", "depthwise"))
        mapping.append((f"{backbone_dw[i]}.bias", f"{dw_name}/Bias", "bias"))
        mapping.append((f"{backbone_pw[i]}.weight", f"{conv_name}/Kernel", "pointwise"))
        mapping.append((f"{backbone_pw[i]}.bias", f"{conv_name}/Bias", "bias"))

        dw_idx += 1
        conv_idx += 1

    # backbone6.1 (channel reduction 96->48, a standalone 1x1 conv)
    mapping.append(("backbone6.1.weight", f"conv2d_{conv_idx}/Kernel", "pointwise"))
    mapping.append(("backbone6.1.bias", f"conv2d_{conv_idx}/Bias", "bias"))
    conv_idx += 1

    # ff layers: blocks of ResBlock(4) + stride-2 ResModule
    for block in range(6):  # ff.0, ff.2, ff.4, ff.6, ff.8, ff.10
        ff_block = block * 2
        for sub in range(4):
            dw_name = f"depthwise_conv2d_{dw_idx}"
            conv_name = f"conv2d_{conv_idx}"
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.0.weight", f"{dw_name}/Kernel", "depthwise"))
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.0.bias", f"{dw_name}/Bias", "bias"))
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.1.weight", f"{conv_name}/Kernel", "pointwise"))
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.1.bias", f"{conv_name}/Bias", "bias"))
            dw_idx += 1
            conv_idx += 1

        if block < 5:  # ff.1, ff.3, ff.5, ff.7, ff.9
            ff_stride = ff_block + 1
            dw_name = f"depthwise_conv2d_{dw_idx}"
            conv_name = f"conv2d_{conv_idx}"
            mapping.append((f"ff.{ff_stride}.convs.0.weight", f"{dw_name}/Kernel", "depthwise"))
            mapping.append((f"ff.{ff_stride}.convs.0.bias", f"{dw_name}/Bias", "bias"))
            mapping.append((f"ff.{ff_stride}.convs.1.weight", f"{conv_name}/Kernel", "pointwise"))
            mapping.append((f"ff.{ff_stride}.convs.1.bias", f"{conv_name}/Bias", "bias"))
            dw_idx += 1
            conv_idx += 1

    # Output heads
    mapping.append(("handflag.weight", "conv_handflag/Kernel", "output_conv"))
    mapping.append(("handflag.bias", "conv_handflag/Bias", "bias"))
    mapping.append(("handedness.weight", "conv_handedness/Kernel", "output_conv"))
    mapping.append(("handedness.bias", "conv_handedness/Bias", "bias"))
    mapping.append(("reg_3d.weight", "convld_21_3d/Kernel", "output_conv"))
    mapping.append(("reg_3d.bias", "convld_21_3d/Bias", "bias"))

    return mapping


# ---------------------------------------------------------------------------
# Step 4: Convert NHWC -> NCHW
# ---------------------------------------------------------------------------
def nhwc_to_nchw(tensor, kind):
    """
    Convert a TFLite weight tensor from NHWC to NCHW format.

    TFLite conv2d kernel: [O, H, W, I] -> PyTorch: [O, I, H, W]
    TFLite depthwise kernel: [1, H, W, C] -> PyTorch: [C, 1, H, W]
    TFLite output conv kernel: [O, H, W, I] -> PyTorch: [O, I, H, W]
    Bias: no change needed.
    """
    if kind == "bias":
        return tensor

    if kind == "depthwise":
        # TFLite: [1, H, W, C] -> PyTorch: [C, 1, H, W]
        return tensor.transpose(3, 0, 1, 2)

    if kind in ("conv3x3", "pointwise", "output_conv"):
        # TFLite: [O, H, W, I] -> PyTorch: [O, I, H, W]
        return tensor.transpose(0, 3, 1, 2)

    raise ValueError(f"Unknown kind: {kind}")


# ---------------------------------------------------------------------------
# Step 5: Load existing weights for comparison
# ---------------------------------------------------------------------------
def load_existing_weights():
    meta_path = WEIGHTS_DIR / "weights_f16.json"
    bin_path = WEIGHTS_DIR / "weights_f16.bin"
    meta = json.loads(meta_path.read_text())
    bin_data = bin_path.read_bytes()

    weights = {}
    for i, key in enumerate(meta["keys"]):
        shape = meta["shapes"][i]
        offset = meta["offsets"][i]
        n = 1
        for s in shape:
            n *= s
        raw = bin_data[offset : offset + n * 2]
        weights[key] = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    return weights


# ---------------------------------------------------------------------------
# Step 6: Save in micro-handpose format (f16 bin + json manifest)
# ---------------------------------------------------------------------------
def save_micro_handpose_format(converted, keys):
    """Save weights in the same binary format as weights_f16.bin + weights_f16.json"""
    shapes = []
    offsets = []
    buf = bytearray()

    for key in keys:
        tensor = converted[key]
        shape = list(tensor.shape)
        shapes.append(shape)
        offsets.append(len(buf))
        f16 = tensor.astype(np.float16)
        buf.extend(f16.tobytes())

    manifest = {"keys": keys, "shapes": shapes, "offsets": offsets}
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2))
    OUTPUT_BIN.write_bytes(bytes(buf))
    print(f"\nSaved {OUTPUT_JSON} ({len(keys)} tensors)")
    print(f"Saved {OUTPUT_BIN} ({len(buf)} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    download_model()

    print("\nLoading TFLite model...")
    tflite_weights = load_tflite_weights()
    print(f"Found {len(tflite_weights)} weight tensors in TFLite model")

    mapping = build_mapping()
    print(f"Built {len(mapping)} tensor mappings")

    # Convert all weights
    converted = {}
    for our_name, tflite_name, kind in mapping:
        # TFLite may store weights as f16 with a separate dequantize op.
        # Try dequantized version first (float32), fall back to original.
        dequant_name = tflite_name + "_dequantize"
        if dequant_name in tflite_weights:
            raw = tflite_weights[dequant_name]
        elif tflite_name in tflite_weights:
            raw = tflite_weights[tflite_name]
        else:
            print(f"ERROR: TFLite tensor not found: {tflite_name}")
            continue

        converted[our_name] = nhwc_to_nchw(raw, kind)

    print(f"Converted {len(converted)} tensors to NCHW format")

    # Verify shapes match our model
    existing = load_existing_weights()
    print(f"\nLoaded {len(existing)} existing (BlazePalm) weights for comparison")

    all_match = True
    for key in sorted(converted.keys()):
        our_shape = list(existing[key].shape)
        tflite_shape = list(converted[key].shape)
        if our_shape != tflite_shape:
            print(f"  SHAPE MISMATCH: {key}: ours={our_shape}, tflite={tflite_shape}")
            all_match = False

    if all_match:
        print("All shapes match!")
    else:
        print("WARNING: Some shapes don't match!")
        return

    # Compare weight values
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON: BlazePalm (vidursatija) vs Official MediaPipe TFLite")
    print("=" * 80)

    total_params = 0
    total_diff = 0
    max_diffs = []

    for key in existing.keys():
        ours = existing[key]
        theirs = converted[key]
        n = ours.size
        total_params += n

        abs_diff = np.abs(ours - theirs)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        rel_diff = mean_diff / (np.abs(theirs).mean() + 1e-10)
        total_diff += abs_diff.sum()

        is_same = np.allclose(ours, theirs, atol=1e-3, rtol=1e-3)
        marker = "SAME" if is_same else "DIFFERENT"

        max_diffs.append((key, max_diff, mean_diff, rel_diff, marker, n))

    # Print summary sorted by relative difference
    max_diffs.sort(key=lambda x: -x[3])

    print(f"\n{'Layer':<45s} {'Params':>8s} {'MaxDiff':>10s} {'MeanDiff':>10s} {'RelDiff':>10s} {'Status':>10s}")
    print("-" * 95)
    for key, max_diff, mean_diff, rel_diff, marker, n in max_diffs[:30]:
        print(f"  {key:<43s} {n:>8d} {max_diff:>10.6f} {mean_diff:>10.6f} {rel_diff:>9.4f}% {marker:>10s}")

    if len(max_diffs) > 30:
        print(f"  ... ({len(max_diffs) - 30} more layers)")

    same_count = sum(1 for x in max_diffs if x[4] == "SAME")
    diff_count = sum(1 for x in max_diffs if x[4] == "DIFFERENT")
    avg_diff = total_diff / total_params

    print(f"\nSummary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Layers SAME (atol=1e-3): {same_count}/{len(max_diffs)}")
    print(f"  Layers DIFFERENT: {diff_count}/{len(max_diffs)}")
    print(f"  Global mean absolute difference: {avg_diff:.8f}")

    # Show a few specific tensor value comparisons
    print("\n" + "=" * 80)
    print("SAMPLE VALUE COMPARISONS (first 5 values of key tensors)")
    print("=" * 80)
    sample_keys = [
        "backbone1.1.weight", "backbone1.1.bias",
        "backbone1.3.f.0.convs.0.weight",
        "ff.10.f.3.convs.1.weight",
        "reg_3d.weight", "reg_3d.bias",
        "handflag.weight", "handflag.bias",
    ]
    for key in sample_keys:
        if key not in existing:
            continue
        ours_flat = existing[key].flatten()[:5]
        theirs_flat = converted[key].flatten()[:5]
        print(f"\n  {key}:")
        print(f"    BlazePalm: {ours_flat}")
        print(f"    TFLite:    {theirs_flat}")

    # Save converted weights
    keys_in_order = list(existing.keys())  # preserve original key order
    save_micro_handpose_format(converted, keys_in_order)

    # Also save as .npz for easy loading
    np.savez(str(OUTPUT_NPZ), **converted)
    print(f"Saved {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / 1024:.0f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
