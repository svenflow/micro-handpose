#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Compare LITE vs FULL hand landmark models and extract FULL weights.

1. Extracts hand_landmarks_detector.tflite from hand_landmarker.task (zip)
2. Loads both LITE and FULL tflite models
3. Compares architectures (layers, tensor shapes, ops)
4. Extracts FULL model weights in micro-handpose format
"""

import json
import os
import struct
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
TASK_PATH = SCRIPT_DIR / "hand_landmarker.task"
LITE_PATH = SCRIPT_DIR / "hand_landmark.tflite"
FULL_PATH = SCRIPT_DIR / "hand_landmark_full.tflite"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
OUTPUT_BIN = WEIGHTS_DIR / "weights_f16_full.bin"
OUTPUT_JSON = WEIGHTS_DIR / "weights_f16_full.json"


# ---------------------------------------------------------------------------
# Step 1: Extract FULL model from .task bundle
# ---------------------------------------------------------------------------
def extract_from_task():
    print("=" * 80)
    print("STEP 1: Extract FULL model from hand_landmarker.task")
    print("=" * 80)

    if not TASK_PATH.exists():
        print(f"ERROR: {TASK_PATH} not found")
        return None

    with zipfile.ZipFile(TASK_PATH, 'r') as z:
        print(f"Contents of {TASK_PATH.name}:")
        for info in z.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

        # Look for the landmark detector tflite
        landmark_files = [f for f in z.namelist() if 'landmark' in f.lower() and f.endswith('.tflite')]
        print(f"\nLandmark tflite files found: {landmark_files}")

        if landmark_files:
            extracted_path = SCRIPT_DIR / "hand_landmark_full_from_task.tflite"
            with z.open(landmark_files[0]) as src, open(extracted_path, 'wb') as dst:
                dst.write(src.read())
            print(f"Extracted to: {extracted_path} ({extracted_path.stat().st_size:,} bytes)")
            return extracted_path

    return None


# ---------------------------------------------------------------------------
# Step 2: Compare architectures
# ---------------------------------------------------------------------------
def analyze_model(path, label):
    """Load a tflite model and return detailed info about its structure."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()

    # Separate weights from intermediate tensors
    weight_tensors = []
    all_tensors = []
    for d in tensor_details:
        name = d["name"]
        shape = list(d["shape"])
        dtype = str(d["dtype"])
        all_tensors.append({"name": name, "shape": shape, "dtype": dtype, "index": d["index"]})
        if "Kernel" in name or "Bias" in name:
            if "_dequantize" not in name:
                try:
                    tensor = interpreter.get_tensor(d["index"])
                except ValueError:
                    tensor = None
                n_params = int(np.prod(shape))
                weight_tensors.append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "n_params": n_params, "tensor": tensor
                })

    total_params = sum(w["n_params"] for w in weight_tensors)

    info = {
        "label": label,
        "path": str(path),
        "file_size": path.stat().st_size,
        "n_tensors_total": len(all_tensors),
        "n_weight_tensors": len(weight_tensors),
        "total_params": total_params,
        "inputs": [{"name": d["name"], "shape": list(d["shape"]), "dtype": str(d["dtype"])} for d in input_details],
        "outputs": [{"name": d["name"], "shape": list(d["shape"]), "dtype": str(d["dtype"])} for d in output_details],
        "weight_tensors": weight_tensors,
        "all_tensors": all_tensors,
    }
    return info, interpreter


def compare_architectures(lite_info, full_info):
    print("\n" + "=" * 80)
    print("STEP 2: Architecture Comparison (LITE vs FULL)")
    print("=" * 80)

    print(f"\n{'Metric':<35s} {'LITE':>15s} {'FULL':>15s}")
    print("-" * 65)
    print(f"{'File size (bytes)':<35s} {lite_info['file_size']:>15,} {full_info['file_size']:>15,}")
    print(f"{'Total tensors':<35s} {lite_info['n_tensors_total']:>15d} {full_info['n_tensors_total']:>15d}")
    print(f"{'Weight tensors (Kernel+Bias)':<35s} {lite_info['n_weight_tensors']:>15d} {full_info['n_weight_tensors']:>15d}")
    print(f"{'Total parameters':<35s} {lite_info['total_params']:>15,} {full_info['total_params']:>15,}")

    print(f"\nInputs:")
    for inp in lite_info["inputs"]:
        print(f"  LITE: {inp['name']} shape={inp['shape']} dtype={inp['dtype']}")
    for inp in full_info["inputs"]:
        print(f"  FULL: {inp['name']} shape={inp['shape']} dtype={inp['dtype']}")

    print(f"\nOutputs:")
    for out in lite_info["outputs"]:
        print(f"  LITE: {out['name']} shape={out['shape']} dtype={out['dtype']}")
    for out in full_info["outputs"]:
        print(f"  FULL: {out['name']} shape={out['shape']} dtype={out['dtype']}")

    # Compare weight tensors side by side
    print(f"\n{'='*80}")
    print("WEIGHT TENSOR COMPARISON")
    print(f"{'='*80}")

    lite_weights = {w["name"]: w for w in lite_info["weight_tensors"]}
    full_weights = {w["name"]: w for w in full_info["weight_tensors"]}

    lite_names = [w["name"] for w in lite_info["weight_tensors"]]
    full_names = [w["name"] for w in full_info["weight_tensors"]]

    # Find common, lite-only, full-only
    lite_set = set(lite_names)
    full_set = set(full_names)

    common = lite_set & full_set
    lite_only = lite_set - full_set
    full_only = full_set - lite_set

    print(f"\nCommon tensors: {len(common)}")
    print(f"LITE-only tensors: {len(lite_only)}")
    print(f"FULL-only tensors: {len(full_only)}")

    # Show shape differences for common tensors
    shape_diffs = []
    for name in sorted(common):
        ls = lite_weights[name]["shape"]
        fs = full_weights[name]["shape"]
        if ls != fs:
            shape_diffs.append((name, ls, fs))

    if shape_diffs:
        print(f"\nShape differences in common tensors ({len(shape_diffs)}):")
        for name, ls, fs in shape_diffs:
            print(f"  {name}:")
            print(f"    LITE: {ls} ({int(np.prod(ls)):,} params)")
            print(f"    FULL: {fs} ({int(np.prod(fs)):,} params)")
    else:
        print("\nNo shape differences in common tensors.")

    if lite_only:
        print(f"\nTensors ONLY in LITE ({len(lite_only)}):")
        for name in sorted(lite_only):
            w = lite_weights[name]
            print(f"  {name}: shape={w['shape']} ({w['n_params']:,} params)")

    if full_only:
        print(f"\nTensors ONLY in FULL ({len(full_only)}):")
        for name in sorted(full_only):
            w = full_weights[name]
            print(f"  {name}: shape={w['shape']} ({w['n_params']:,} params)")

    # Print all FULL weights in order with shapes
    print(f"\n{'='*80}")
    print(f"ALL FULL MODEL WEIGHT TENSORS ({len(full_names)} tensors)")
    print(f"{'='*80}")
    print(f"\n{'#':>3s}  {'Name':<50s} {'Shape':<25s} {'Params':>10s}")
    print("-" * 92)
    for i, w in enumerate(full_info["weight_tensors"]):
        in_lite = "  " if w["name"] in lite_set else "* "
        print(f"{i:>3d}  {in_lite}{w['name']:<48s} {str(w['shape']):<25s} {w['n_params']:>10,}")

    # Print all LITE weights in order with shapes for comparison
    print(f"\n{'='*80}")
    print(f"ALL LITE MODEL WEIGHT TENSORS ({len(lite_names)} tensors)")
    print(f"{'='*80}")
    print(f"\n{'#':>3s}  {'Name':<50s} {'Shape':<25s} {'Params':>10s}")
    print("-" * 92)
    for i, w in enumerate(lite_info["weight_tensors"]):
        in_full = "  " if w["name"] in full_set else "* "
        print(f"{i:>3d}  {in_full}{w['name']:<48s} {str(w['shape']):<25s} {w['n_params']:>10,}")

    return lite_names, full_names, lite_weights, full_weights


# ---------------------------------------------------------------------------
# Step 3: Build mapping for FULL model and extract weights
# ---------------------------------------------------------------------------
def discover_full_mapping(full_info):
    """
    Auto-discover the mapping from TFLite tensor names to micro-handpose names.
    We parse the sequence of conv2d and depthwise_conv2d operations to figure out
    the architecture.
    """
    weight_tensors = full_info["weight_tensors"]
    names = [w["name"] for w in weight_tensors]

    # Separate into categories
    conv_kernels = [(i, w) for i, w in enumerate(weight_tensors) if w["name"].startswith("conv2d") and "Kernel" in w["name"]]
    dw_kernels = [(i, w) for i, w in enumerate(weight_tensors) if w["name"].startswith("depthwise_conv2d") and "Kernel" in w["name"]]
    other_kernels = [(i, w) for i, w in enumerate(weight_tensors) if "Kernel" in w["name"] and not w["name"].startswith("conv2d") and not w["name"].startswith("depthwise_conv2d")]

    print(f"\n{'='*80}")
    print("FULL MODEL LAYER ANALYSIS")
    print(f"{'='*80}")
    print(f"\nConv2d layers: {len(conv_kernels)}")
    print(f"Depthwise conv2d layers: {len(dw_kernels)}")
    print(f"Other kernel layers: {len(other_kernels)}")

    # Print interleaved sequence of operations
    print(f"\nOperation sequence (interleaved conv + depthwise):")
    ops = []
    for w in weight_tensors:
        name = w["name"]
        if "Kernel" not in name:
            continue
        shape = w["shape"]
        if name.startswith("depthwise_conv2d"):
            op_type = "DW"
            # TFLite depthwise: [1, H, W, C]
            channels = shape[3] if len(shape) == 4 else shape[0]
            kernel_size = f"{shape[1]}x{shape[2]}" if len(shape) == 4 else "?"
            desc = f"DW {kernel_size} ch={channels}"
        elif name.startswith("conv2d"):
            op_type = "PW"
            # TFLite conv: [O, H, W, I]
            out_ch = shape[0]
            in_ch = shape[3] if len(shape) == 4 else shape[1]
            kernel_size = f"{shape[1]}x{shape[2]}" if len(shape) == 4 else "?"
            desc = f"Conv {kernel_size} {in_ch}->{out_ch}"
        elif name.startswith("conv_"):
            op_type = "OUT"
            out_ch = shape[0]
            in_ch = shape[3] if len(shape) == 4 else shape[1]
            kernel_size = f"{shape[1]}x{shape[2]}" if len(shape) == 4 else "?"
            desc = f"Output {kernel_size} {in_ch}->{out_ch}"
        else:
            op_type = "?"
            desc = f"Unknown {shape}"

        ops.append((name, op_type, desc, shape))
        print(f"  {name:<50s}  {desc}")

    return ops


def build_full_mapping(full_info, lite_info):
    """
    Build the mapping from TFLite names to micro-handpose names for the FULL model.
    We need to figure out what's different from LITE and adjust naming accordingly.
    """
    full_weights = {w["name"]: w for w in full_info["weight_tensors"]}
    lite_weights = {w["name"]: w for w in lite_info["weight_tensors"]}

    # Get the ordered list of kernel tensors to understand the structure
    full_kernels = [w for w in full_info["weight_tensors"] if "Kernel" in w["name"]]
    lite_kernels = [w for w in lite_info["weight_tensors"] if "Kernel" in w["name"]]

    print(f"\n{'='*80}")
    print("BUILDING FULL MODEL MAPPING")
    print(f"{'='*80}")

    # First, let's see if the FULL model has the same TFLite naming convention
    # by checking if the same conv2d/depthwise_conv2d indices exist
    full_conv_names = sorted([w["name"] for w in full_info["weight_tensors"] if w["name"].startswith("conv2d") and "Kernel" in w["name"]])
    full_dw_names = sorted([w["name"] for w in full_info["weight_tensors"] if w["name"].startswith("depthwise_conv2d") and "Kernel" in w["name"]])
    lite_conv_names = sorted([w["name"] for w in lite_info["weight_tensors"] if w["name"].startswith("conv2d") and "Kernel" in w["name"]])
    lite_dw_names = sorted([w["name"] for w in lite_info["weight_tensors"] if w["name"].startswith("depthwise_conv2d") and "Kernel" in w["name"]])

    print(f"\nLITE: {len(lite_conv_names)} conv2d, {len(lite_dw_names)} depthwise")
    print(f"FULL: {len(full_conv_names)} conv2d, {len(full_dw_names)} depthwise")

    # Parse conv indices
    def get_conv_idx(name):
        # conv2d/Kernel -> 0, conv2d_1/Kernel -> 1, etc.
        base = name.split("/")[0]
        if base == "conv2d":
            return 0
        return int(base.split("_")[1])

    def get_dw_idx(name):
        base = name.split("/")[0]
        if base == "depthwise_conv2d":
            return 0
        return int(base.split("_")[-1])

    lite_conv_indices = [get_conv_idx(n) for n in lite_conv_names]
    full_conv_indices = [get_conv_idx(n) for n in full_conv_names]
    lite_dw_indices = [get_dw_idx(n) for n in lite_dw_names]
    full_dw_indices = [get_dw_idx(n) for n in full_dw_names]

    print(f"\nLITE conv2d indices: {lite_conv_indices}")
    print(f"FULL conv2d indices: {full_conv_indices}")
    print(f"\nLITE DW indices range: {min(lite_dw_indices)}-{max(lite_dw_indices)} ({len(lite_dw_indices)} total)")
    print(f"FULL DW indices range: {min(full_dw_indices)}-{max(full_dw_indices)} ({len(full_dw_indices)} total)")

    # Extra indices in FULL
    extra_conv = set(full_conv_indices) - set(lite_conv_indices)
    extra_dw = set(full_dw_indices) - set(lite_dw_indices)
    print(f"\nExtra conv2d indices in FULL: {sorted(extra_conv)}")
    print(f"Extra DW indices in FULL: {sorted(extra_dw)}")

    # Now let's trace through the FULL model architecture by looking at shapes
    # in the order they appear
    print(f"\n{'='*80}")
    print("FULL MODEL ARCHITECTURE TRACE")
    print(f"{'='*80}")

    # Go through weight tensors in order and identify blocks
    all_w = full_info["weight_tensors"]
    i = 0
    block_num = 0
    while i < len(all_w):
        w = all_w[i]
        name = w["name"]
        shape = w["shape"]

        if "Kernel" in name:
            if name.startswith("depthwise"):
                ch = shape[3]
                ks = f"{shape[1]}x{shape[2]}"
                # Next should be pointwise
                if i+2 < len(all_w) and "conv2d" in all_w[i+2]["name"] and "Kernel" in all_w[i+2]["name"]:
                    pw = all_w[i+2]
                    pw_shape = pw["shape"]
                    print(f"  Block {block_num}: DW {ks} ch={ch} + PW {pw_shape[3]}->{pw_shape[0]}  [{name} + {pw['name']}]")
                    block_num += 1
                    i += 4  # skip both kernel+bias pairs
                    continue
                else:
                    print(f"  Block {block_num}: DW {ks} ch={ch}  [{name}]")
                    block_num += 1
            elif name.startswith("conv2d"):
                out_ch = shape[0]
                in_ch = shape[3]
                ks = f"{shape[1]}x{shape[2]}"
                print(f"  Block {block_num}: Conv {ks} {in_ch}->{out_ch}  [{name}]")
                block_num += 1
            elif name.startswith("conv_"):
                out_ch = shape[0]
                in_ch = shape[3]
                ks = f"{shape[1]}x{shape[2]}"
                print(f"  Block {block_num}: Output {ks} {in_ch}->{out_ch}  [{name}]")
                block_num += 1

        i += 1

    return None  # We'll build the actual mapping below


def build_mapping_by_structure(full_info):
    """
    Build the micro-handpose name mapping for the FULL model by examining
    the actual tensor sequence and shapes, matching the architecture description.

    Architecture (from model.ts):
    1. Input conv3x3 (3→24, stride=2) + ReLU → 128x128x24
    2. backbone1: ResBlock(2) + ResModule(24→48, stride=2) → 64x64x48
    3. backbone2: ResBlock(2) + ResModule(48→96, stride=2) → 32x32x96
    4. backbone3: ResBlock(2) + ResModule(96→96, stride=2) → 16x16x96
    5. backbone4: ResBlock(2) + ResModule(96→96, stride=2) + upsample + add b3
    6. backbone5: ResModule(96→96) + upsample + add b2
    7. backbone6: ResModule(96→96) + conv1x1(96→48) + upsample + add b1
    8. ff layers: 5x (ResBlock(4) + ResModule(stride=2)) + ResBlock(4) → 2x2x288
    9. Output heads: handflag, handedness, landmarks
    """

    weight_tensors = full_info["weight_tensors"]

    # Extract only Kernel tensors in order to understand the sequence
    kernel_sequence = []
    for w in weight_tensors:
        if "Kernel" in w["name"]:
            kernel_sequence.append(w)

    print(f"\n{'='*80}")
    print(f"FULL kernel sequence ({len(kernel_sequence)} kernels):")
    print(f"{'='*80}")
    for i, k in enumerate(kernel_sequence):
        shape = k["shape"]
        name = k["name"]
        if name.startswith("depthwise"):
            desc = f"DW {shape[1]}x{shape[2]} ch={shape[3]}"
        elif name.startswith("conv2d") or name.startswith("conv_"):
            desc = f"Conv {shape[1]}x{shape[2]} {shape[3]}->{shape[0]}"
        else:
            desc = f"??? {shape}"
        print(f"  {i:>3d}. {name:<50s} {str(shape):<25s} {desc}")

    # Now build the mapping based on this sequence
    # Each DW+PW pair is a depthwise separable conv
    # The sequence follows the architecture above

    mapping = []  # (our_name, tflite_name, kind)

    # Parse the sequence of operations
    dw_names = []
    conv_names = []
    output_names = []

    for w in weight_tensors:
        name = w["name"]
        if "Kernel" not in name:
            continue
        if name.startswith("depthwise"):
            dw_names.append(name.replace("/Kernel", ""))
        elif name.startswith("conv2d"):
            conv_names.append(name.replace("/Kernel", ""))
        elif name.startswith("conv_"):
            output_names.append(name.replace("/Kernel", ""))

    print(f"\n  DW layers: {len(dw_names)}")
    print(f"  Conv layers: {len(conv_names)}")
    print(f"  Output layers: {len(output_names)}")

    # The first conv is always the input conv (3x3, 3->24)
    # Then DW+PW pairs follow the backbone/ff structure

    # Let's trace through the architecture by examining shapes
    # and build the mapping dynamically

    # Get shapes for each layer
    full_w_dict = {w["name"]: w for w in weight_tensors}

    conv_idx = 0
    dw_idx = 0

    # Input conv: conv2d (first one)
    conv_base = conv_names[conv_idx]
    mapping.append(("backbone1.1.weight", f"{conv_base}/Kernel", "conv3x3"))
    mapping.append(("backbone1.1.bias", f"{conv_base}/Bias", "bias"))
    conv_idx += 1

    # Now trace DW+PW pairs through the backbone
    # backbone1: ResBlock(2) = 2 DW+PW pairs, then ResModule(24→48) = 1 DW+PW pair
    # ResBlock(N) = N pairs of (DW, PW) with residual (same channels)
    # ResModule(in→out) = 1 pair of (DW, PW) that changes channels

    # backbone1.3.f.0 and backbone1.3.f.1 are the ResBlock(2) pairs
    backbone_sections = [
        # (prefix, n_resblock, has_resmodule, resmodule_prefix)
        # backbone1: ResBlock(2) + ResModule(24→48)
        ("backbone1.3", 2, True, "backbone1.4"),
        # backbone2: ResBlock(2) + ResModule(48→96)
        ("backbone2.0", 2, True, "backbone2.1"),
        # backbone3: ResBlock(2) + ResModule(96→96)
        ("backbone3.0", 2, True, "backbone3.1"),
        # backbone4: ResBlock(2) + ResModule(96→96)
        ("backbone4.0", 2, True, "backbone4.1"),
        # backbone5: just ResModule(96→96), no ResBlock
        (None, 0, True, "backbone5.0"),
        # backbone6: just ResModule(96→96) + 1x1 conv
        (None, 0, True, "backbone6.0"),
    ]

    for section_prefix, n_resblock, has_resmodule, resmodule_prefix in backbone_sections:
        # ResBlock pairs
        for j in range(n_resblock):
            dw_base = dw_names[dw_idx]
            conv_base = conv_names[conv_idx]
            mapping.append((f"{section_prefix}.f.{j}.convs.0.weight", f"{dw_base}/Kernel", "depthwise"))
            mapping.append((f"{section_prefix}.f.{j}.convs.0.bias", f"{dw_base}/Bias", "bias"))
            mapping.append((f"{section_prefix}.f.{j}.convs.1.weight", f"{conv_base}/Kernel", "pointwise"))
            mapping.append((f"{section_prefix}.f.{j}.convs.1.bias", f"{conv_base}/Bias", "bias"))
            dw_idx += 1
            conv_idx += 1

        # ResModule pair
        if has_resmodule:
            dw_base = dw_names[dw_idx]
            conv_base = conv_names[conv_idx]
            mapping.append((f"{resmodule_prefix}.convs.0.weight", f"{dw_base}/Kernel", "depthwise"))
            mapping.append((f"{resmodule_prefix}.convs.0.bias", f"{dw_base}/Bias", "bias"))
            mapping.append((f"{resmodule_prefix}.convs.1.weight", f"{conv_base}/Kernel", "pointwise"))
            mapping.append((f"{resmodule_prefix}.convs.1.bias", f"{conv_base}/Bias", "bias"))
            dw_idx += 1
            conv_idx += 1

    # backbone6.1 (channel reduction 96->48, standalone 1x1 conv)
    conv_base = conv_names[conv_idx]
    mapping.append(("backbone6.1.weight", f"{conv_base}/Kernel", "pointwise"))
    mapping.append(("backbone6.1.bias", f"{conv_base}/Bias", "bias"))
    conv_idx += 1

    # ff layers: 5x (ResBlock(4) + ResModule(stride=2)) + ResBlock(4)
    # That's: ff.0(ResBlock4) + ff.1(ResModule) + ff.2(ResBlock4) + ff.3(ResModule) + ...
    # ff.0, ff.2, ff.4, ff.6, ff.8 = ResBlock(4) each (4 DW+PW pairs)
    # ff.1, ff.3, ff.5, ff.7, ff.9 = ResModule (1 DW+PW pair each)
    # ff.10 = final ResBlock(4) (no following ResModule)

    for block in range(6):  # 6 ResBlock(4) blocks: ff.0, ff.2, ff.4, ff.6, ff.8, ff.10
        ff_block = block * 2
        for sub in range(4):
            dw_base = dw_names[dw_idx]
            conv_base = conv_names[conv_idx]
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.0.weight", f"{dw_base}/Kernel", "depthwise"))
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.0.bias", f"{dw_base}/Bias", "bias"))
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.1.weight", f"{conv_base}/Kernel", "pointwise"))
            mapping.append((f"ff.{ff_block}.f.{sub}.convs.1.bias", f"{conv_base}/Bias", "bias"))
            dw_idx += 1
            conv_idx += 1

        if block < 5:  # ResModule between ResBlocks
            ff_stride = ff_block + 1
            dw_base = dw_names[dw_idx]
            conv_base = conv_names[conv_idx]
            mapping.append((f"ff.{ff_stride}.convs.0.weight", f"{dw_base}/Kernel", "depthwise"))
            mapping.append((f"ff.{ff_stride}.convs.0.bias", f"{dw_base}/Bias", "bias"))
            mapping.append((f"ff.{ff_stride}.convs.1.weight", f"{conv_base}/Kernel", "pointwise"))
            mapping.append((f"ff.{ff_stride}.convs.1.bias", f"{conv_base}/Bias", "bias"))
            dw_idx += 1
            conv_idx += 1

    print(f"\n  Used {dw_idx}/{len(dw_names)} DW layers")
    print(f"  Used {conv_idx}/{len(conv_names)} conv layers")
    print(f"  Remaining DW: {len(dw_names) - dw_idx}")
    print(f"  Remaining conv: {len(conv_names) - conv_idx}")

    # Output heads
    for out_name in output_names:
        if "handflag" in out_name:
            mapping.append(("handflag.weight", f"{out_name}/Kernel", "output_conv"))
            mapping.append(("handflag.bias", f"{out_name}/Bias", "bias"))
        elif "handedness" in out_name:
            mapping.append(("handedness.weight", f"{out_name}/Kernel", "output_conv"))
            mapping.append(("handedness.bias", f"{out_name}/Bias", "bias"))
        elif "ld_21_3d" in out_name or "landmarks" in out_name:
            mapping.append(("reg_3d.weight", f"{out_name}/Kernel", "output_conv"))
            mapping.append(("reg_3d.bias", f"{out_name}/Bias", "bias"))

    print(f"\n  Total mapping entries: {len(mapping)}")
    return mapping


# ---------------------------------------------------------------------------
# Step 4: Convert and save
# ---------------------------------------------------------------------------
def nhwc_to_nchw(tensor, kind):
    if kind == "bias":
        return tensor.astype(np.float32)
    if kind == "depthwise":
        return tensor.astype(np.float32).transpose(3, 0, 1, 2)
    if kind in ("conv3x3", "pointwise", "output_conv"):
        return tensor.astype(np.float32).transpose(0, 3, 1, 2)
    raise ValueError(f"Unknown kind: {kind}")


def extract_and_save_full_weights(full_info, mapping):
    print(f"\n{'='*80}")
    print("STEP 3: Extracting FULL model weights")
    print(f"{'='*80}")

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(FULL_PATH))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    # Build lookup
    tflite_tensors = {}
    for d in details:
        name = d["name"]
        tensor = interpreter.get_tensor(d["index"]).astype(np.float32)
        tflite_tensors[name] = tensor

    # Convert all weights
    converted = {}
    keys = []
    errors = []

    for our_name, tflite_name, kind in mapping:
        if tflite_name in tflite_tensors:
            raw = tflite_tensors[tflite_name]
        else:
            errors.append(f"  NOT FOUND: {tflite_name} (for {our_name})")
            continue

        converted[our_name] = nhwc_to_nchw(raw, kind)
        keys.append(our_name)

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(e)

    print(f"\nConverted {len(converted)} tensors to NCHW format")

    # Print shapes
    total_params = 0
    print(f"\n{'#':>3s}  {'Name':<45s} {'Shape':<25s} {'Params':>10s}")
    print("-" * 87)
    for i, key in enumerate(keys):
        shape = list(converted[key].shape)
        n = int(np.prod(shape))
        total_params += n
        print(f"{i:>3d}  {key:<45s} {str(shape):<25s} {n:>10,}")
    print(f"\nTotal parameters: {total_params:,}")

    # Save in micro-handpose format
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

    manifest = {"keys": keys, "shapes": shapes, "offsets": offsets, "dtype": "float16"}
    OUTPUT_JSON.write_text(json.dumps(manifest))
    OUTPUT_BIN.write_bytes(bytes(buf))
    print(f"\nSaved {OUTPUT_JSON} ({len(keys)} tensors)")
    print(f"Saved {OUTPUT_BIN} ({len(buf):,} bytes)")

    return converted, keys


# ---------------------------------------------------------------------------
# Step 5: Compare LITE vs FULL weight counts
# ---------------------------------------------------------------------------
def print_summary(lite_info, full_info, converted_full, full_keys):
    print(f"\n{'='*80}")
    print("STEP 4: SUMMARY OF DIFFERENCES (LITE vs FULL)")
    print(f"{'='*80}")

    # Load existing LITE weights for comparison
    meta_path = WEIGHTS_DIR / "weights_f16.json"
    meta = json.loads(meta_path.read_text())
    lite_keys = meta["keys"]
    lite_shapes = {k: s for k, s in zip(meta["keys"], meta["shapes"])}

    full_shapes = {k: list(converted_full[k].shape) for k in full_keys}

    # Compare key sets
    lite_set = set(lite_keys)
    full_set = set(full_keys)

    common = lite_set & full_set
    lite_only = lite_set - full_set
    full_only = full_set - lite_set

    print(f"\nWeight tensor counts:")
    print(f"  LITE: {len(lite_keys)} tensors")
    print(f"  FULL: {len(full_keys)} tensors")
    print(f"  Common: {len(common)}")
    print(f"  LITE-only: {len(lite_only)}")
    print(f"  FULL-only: {len(full_only)}")

    if lite_only:
        print(f"\nTensors only in LITE:")
        for k in sorted(lite_only):
            print(f"  {k}: {lite_shapes[k]}")

    if full_only:
        print(f"\nTensors only in FULL:")
        for k in sorted(full_only):
            print(f"  {k}: {full_shapes[k]}")

    # Shape differences in common tensors
    shape_diffs = []
    for k in sorted(common):
        ls = lite_shapes[k]
        fs = full_shapes[k]
        if ls != fs:
            shape_diffs.append((k, ls, fs))

    if shape_diffs:
        print(f"\nShape differences ({len(shape_diffs)}):")
        for k, ls, fs in shape_diffs:
            lite_p = int(np.prod(ls))
            full_p = int(np.prod(fs))
            print(f"  {k}:")
            print(f"    LITE: {ls} ({lite_p:,} params)")
            print(f"    FULL: {fs} ({full_p:,} params)")
    else:
        print(f"\nAll {len(common)} common tensors have identical shapes.")

    lite_total = sum(int(np.prod(lite_shapes[k])) for k in lite_keys)
    full_total = sum(int(np.prod(full_shapes[k])) for k in full_keys)
    print(f"\nTotal parameter counts:")
    print(f"  LITE: {lite_total:,}")
    print(f"  FULL: {full_total:,}")
    print(f"  Difference: {full_total - lite_total:,} ({(full_total/lite_total - 1)*100:.1f}% more)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Step 1: Extract from .task
    extracted = extract_from_task()

    # Step 2: Compare architectures
    print("\nLoading LITE model...")
    lite_info, lite_interp = analyze_model(LITE_PATH, "LITE")
    print(f"  {lite_info['n_weight_tensors']} weight tensors, {lite_info['total_params']:,} params")

    print("Loading FULL model...")
    full_info, full_interp = analyze_model(FULL_PATH, "FULL")
    print(f"  {full_info['n_weight_tensors']} weight tensors, {full_info['total_params']:,} params")

    compare_architectures(lite_info, full_info)

    # Discover FULL model structure
    ops = discover_full_mapping(full_info)

    # Step 3: Build mapping and extract
    mapping = build_mapping_by_structure(full_info)

    if mapping is None:
        print("ERROR: Could not build mapping")
        return

    converted, keys = extract_and_save_full_weights(full_info, mapping)

    # Step 4: Summary
    print_summary(lite_info, full_info, converted, keys)


if __name__ == "__main__":
    main()
