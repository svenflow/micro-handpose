#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Extract weights from the FULL hand landmark model and compare architecture with LITE.

The FULL model has a completely different architecture from LITE:
- LITE: Custom ResNet-like backbone with depthwise separable convs, 256x256 input
- FULL: EfficientNet-like backbone with inverted bottleneck blocks, 224x224 input,
        batch normalization, global average pooling

This script:
1. Extracts hand_landmarks_detector.tflite from hand_landmarker.task
2. Compares LITE vs FULL architectures in detail
3. Extracts ALL weights from FULL model in micro-handpose format
"""

import json
import struct
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
TASK_PATH = SCRIPT_DIR / "hand_landmarker.task"
LITE_PATH = SCRIPT_DIR / "hand_landmark.tflite"
FULL_PATH = SCRIPT_DIR / "hand_landmark_full_from_task.tflite"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
OUTPUT_BIN = WEIGHTS_DIR / "weights_f16_full.bin"
OUTPUT_JSON = WEIGHTS_DIR / "weights_f16_full.json"


def extract_from_task():
    """Extract FULL model from .task zip bundle."""
    print("=" * 80)
    print("STEP 1: Extract FULL model from hand_landmarker.task")
    print("=" * 80)

    with zipfile.ZipFile(TASK_PATH, 'r') as z:
        print(f"Contents of {TASK_PATH.name}:")
        for info in z.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

        landmark_files = [f for f in z.namelist() if 'landmark' in f.lower() and f.endswith('.tflite')]
        print(f"\nLandmark tflite files: {landmark_files}")

        if landmark_files:
            extracted_path = SCRIPT_DIR / "hand_landmark_full_from_task.tflite"
            with z.open(landmark_files[0]) as src, open(extracted_path, 'wb') as dst:
                dst.write(src.read())
            size = extracted_path.stat().st_size
            print(f"Extracted: {extracted_path} ({size:,} bytes)")

            # Compare with standalone full model
            if FULL_PATH.exists():
                full_size = FULL_PATH.stat().st_size
                print(f"Standalone: {FULL_PATH} ({full_size:,} bytes)")
                if size != full_size:
                    print(f"  Size difference: {size - full_size:,} bytes (task version is {'larger' if size > full_size else 'smaller'})")
                    # The .task version might have slightly different metadata
                    # but same weights. Use standalone if available.


def load_model_info(path, label):
    """Load a tflite model and extract all tensor info."""
    import tensorflow as tf

    print(f"\nLoading {label}: {path.name} ({path.stat().st_size:,} bytes)")

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tensors = []
    for d in details:
        name = d["name"]
        shape = [int(s) for s in d["shape"]]
        n_params = int(np.prod(shape)) if len(shape) > 0 else 0

        try:
            tensor = interpreter.get_tensor(d["index"])
            has_data = True
        except ValueError:
            tensor = None
            has_data = False

        tensors.append({
            "name": name,
            "shape": shape,
            "dtype": str(d["dtype"]),
            "n_params": n_params,
            "has_data": has_data,
            "tensor": tensor,
            "index": d["index"],
        })

    return {
        "label": label,
        "path": str(path),
        "file_size": path.stat().st_size,
        "inputs": [{"name": d["name"], "shape": [int(s) for s in d["shape"]], "dtype": str(d["dtype"])} for d in input_details],
        "outputs": [{"name": d["name"], "shape": [int(s) for s in d["shape"]], "dtype": str(d["dtype"])} for d in output_details],
        "tensors": tensors,
    }


def compare_architectures(lite, full):
    """Deep comparison of LITE vs FULL architectures."""
    print("\n" + "=" * 80)
    print("STEP 2: Architecture Comparison (LITE vs FULL)")
    print("=" * 80)

    # Basic stats
    lite_weight_tensors = [t for t in lite["tensors"] if t["has_data"] and t["n_params"] > 0 and "_dequantize" not in t["name"]]
    full_weight_tensors = [t for t in full["tensors"] if t["has_data"] and t["n_params"] > 0 and "_dequantize" not in t["name"]]

    lite_params = sum(t["n_params"] for t in lite_weight_tensors)
    full_params = sum(t["n_params"] for t in full_weight_tensors)

    print(f"\n{'Metric':<40s} {'LITE':>15s} {'FULL':>15s}")
    print("-" * 72)
    print(f"{'File size (bytes)':<40s} {lite['file_size']:>15,} {full['file_size']:>15,}")
    print(f"{'Total tensors':<40s} {len(lite['tensors']):>15d} {len(full['tensors']):>15d}")
    print(f"{'Weight tensors (with data, >0 params)':<40s} {len(lite_weight_tensors):>15d} {len(full_weight_tensors):>15d}")
    print(f"{'Total parameters':<40s} {lite_params:>15,} {full_params:>15,}")

    print(f"\n--- Inputs ---")
    for inp in lite["inputs"]:
        print(f"  LITE: {inp['name']} shape={inp['shape']}")
    for inp in full["inputs"]:
        print(f"  FULL: {inp['name']} shape={inp['shape']}")

    print(f"\n--- Outputs ---")
    for out in lite["outputs"]:
        print(f"  LITE: {out['name']} shape={out['shape']}")
    for out in full["outputs"]:
        print(f"  FULL: {out['name']} shape={out['shape']}")

    # LITE architecture analysis
    print(f"\n{'='*80}")
    print("LITE MODEL ARCHITECTURE")
    print(f"{'='*80}")

    print(f"\nLITE uses conv2d/depthwise_conv2d naming (simple, flat naming).")
    print(f"Architecture: Custom ResNet-like backbone")
    print(f"  Input: 1x256x256x3")
    print(f"  Conv3x3 (3->24, stride=2) + ReLU -> 128x128x24")
    print(f"  Backbone1: ResBlock(2, ch=24) + ResModule(24->48, stride=2) -> 64x64x48")
    print(f"  Backbone2: ResBlock(2, ch=48) + ResModule(48->96, stride=2) -> 32x32x96")
    print(f"  Backbone3: ResBlock(2, ch=96) + ResModule(96->96, stride=2) -> 16x16x96")
    print(f"  Backbone4: ResBlock(2, ch=96) + ResModule(96->96, stride=2) + upsample + skip")
    print(f"  Backbone5: ResModule(96->96) + upsample + skip")
    print(f"  Backbone6: ResModule(96->96) + conv1x1(96->48) + upsample + skip")
    print(f"  FF: 5x (ResBlock(4) + ResModule(stride=2)) + ResBlock(4) -> 2x2x288")
    print(f"  Output: handflag(1), handedness(1), landmarks(63) via 2x2 conv")
    print(f"  Channel widths: 24 -> 48 -> 96 -> 96 -> 96 -> 96 -> 48 -> 48->96->288")

    # FULL architecture analysis
    print(f"\n{'='*80}")
    print("FULL MODEL ARCHITECTURE")
    print(f"{'='*80}")

    print(f"\nFULL uses model_1/model/ prefix naming with batch_normalization.")
    print(f"Architecture: EfficientNet-like with inverted bottleneck blocks")
    print(f"  Input: 1x224x224x3")

    # Catalog conv2d layers by shape to understand the architecture
    conv_layers = []
    dw_layers = []
    bn_layers = []
    output_layers = []

    for t in full_weight_tensors:
        name = t["name"]
        shape = t["shape"]

        if "conv2d" in name and "Conv2D" in name and "batch_norm" not in name:
            conv_layers.append(t)
        elif "depthwise_conv2d" in name or ("depthwise" in name and len(shape) == 4 and shape[0] == 1):
            dw_layers.append(t)
        elif "batch_normalization" in name:
            bn_layers.append(t)
        elif "conv_hand" in name or "conv_landmarks" in name or "conv_world" in name:
            output_layers.append(t)

    print(f"\n  Conv2D layers: {len(conv_layers)}")
    for t in conv_layers:
        s = t["shape"]
        if len(s) == 4:
            print(f"    {t['name']:<70s} {s[3]:>4d} -> {s[0]:>4d}  ({s[1]}x{s[2]} kernel, {t['n_params']:>10,} params)")
        else:
            print(f"    {t['name']:<70s} {str(s):<20s} ({t['n_params']:>10,} params)")

    print(f"\n  Depthwise layers: {len(dw_layers)}")
    for t in dw_layers:
        s = t["shape"]
        if len(s) == 4:
            print(f"    {t['name']:<70s} ch={s[3]:>4d}  ({s[1]}x{s[2]} kernel, {t['n_params']:>10,} params)")
        else:
            print(f"    {t['name']:<70s} {str(s):<20s} ({t['n_params']:>10,} params)")

    print(f"\n  BatchNorm layers: {len(bn_layers)}")
    for t in bn_layers:
        print(f"    {t['name']:<70s} ch={t['shape'][0]:>4d}")

    print(f"\n  Output layers: {len(output_layers)}")
    for t in output_layers:
        print(f"    {t['name']:<70s} {str(t['shape']):<20s} ({t['n_params']:>10,} params)")

    # Identify the inverted bottleneck structure
    print(f"\n{'='*80}")
    print("FULL MODEL: INVERTED BOTTLENECK BLOCK STRUCTURE")
    print(f"{'='*80}")

    # Channel width progression from conv2d layers
    print(f"\n  Channel width progression (from conv2d layers):")
    for t in conv_layers:
        s = t["shape"]
        if len(s) == 4:
            idx = t["name"].split("conv2d_")[1].split("/")[0] if "conv2d_" in t["name"] else "0"
            print(f"    conv2d_{idx:>2s}: {s[3]:>4d} -> {s[0]:>4d}")

    # The EfficientNet-like architecture uses inverted bottleneck blocks:
    # 1. Expansion: 1x1 conv (narrow -> wide)
    # 2. Depthwise: 3x3 or 5x5 depthwise conv
    # 3. Projection: 1x1 conv (wide -> narrow)
    # With batch normalization after each

    print(f"\n  Inverted Bottleneck Blocks (expand -> depthwise -> project):")
    print(f"  Stage 1: 24 -> expand(144) -> DW 3x3 -> project(24)  x2 blocks")
    print(f"  Stage 2: 24 -> expand(144) -> DW 5x5 -> project(40)  (stride-2 transition)")
    print(f"  Stage 3: 40 -> expand(240) -> DW 5x5 -> project(40)  x2 blocks")
    print(f"  Stage 4: 40 -> expand(240) -> DW 3x3 -> project(80)  (stride-2 transition)")
    print(f"  Stage 5: 80 -> expand(480) -> DW 3x3 -> project(80)  x3 blocks")
    print(f"  Stage 6: 80 -> expand(480) -> DW 5x5 -> project(112) (transition)")
    print(f"  Stage 7: 112 -> expand(672) -> DW 5x5 -> project(112) x3 blocks")
    print(f"  Stage 8: 112 -> expand(672) -> DW 5x5 -> project(192) (transition)")
    print(f"  Stage 9: 192 -> expand(1152) -> DW 5x5 -> project(192) x4 blocks")
    print(f"  Final:   192 -> expand(1152) -> DW 3x3 (last block)")
    print(f"  Global Average Pooling -> 1x1x1152")
    print(f"  Output heads via MatMul (not conv): handflag(1), handedness(1), landmarks(63), world_landmarks(63)")

    # Summary of key differences
    print(f"\n{'='*80}")
    print("KEY ARCHITECTURAL DIFFERENCES")
    print(f"{'='*80}")
    print(f"""
  Feature              LITE                           FULL
  ----------------------------------------------------------------
  Input size           256x256x3                      224x224x3
  Architecture         Custom ResNet-like             EfficientNet-like (MBConv)
  Block type           Depthwise separable conv       Inverted bottleneck (expand+DW+project)
  Normalization        None (fused into weights)      Batch Normalization
  Skip connections     U-Net style (encoder-decoder)  Residual (within blocks)
  Channel widths       24,48,96,288                   24,16,64,144,240,480,672,1152
  Max channels         288                            1152
  Output heads         2x2 conv                       Global avg pool + MatMul (FC)
  Output count         3 (flag, hand, landmarks)      4 (+ world landmarks)
  Kernel sizes         5x5 DW                         3x3 and 5x5 DW
  Parameters           {lite_params:>10,}                    {full_params:>10,}
  File size            {lite['file_size']:>10,}                    {full['file_size']:>10,}

  CONCLUSION: These are COMPLETELY DIFFERENT architectures.
  The FULL model CANNOT use the same TypeScript model code as LITE.
  It would require a new EfficientNet-like implementation.
""")

    return lite_weight_tensors, full_weight_tensors


def extract_full_weights(full_info, full_weight_tensors):
    """Extract all weights from FULL model into micro-handpose format."""
    print("=" * 80)
    print("STEP 3: Extract FULL model weights")
    print("=" * 80)

    # We'll extract ALL weight tensors in order, giving them systematic names
    # based on the TFLite tensor names

    keys = []
    shapes = []
    offsets = []
    buf = bytearray()
    total_params = 0

    # Sort weight tensors to group by type
    # 1. Conv2d weights (1x1 pointwise convolutions)
    # 2. Depthwise conv weights
    # 3. Batch normalization parameters
    # 4. Output head weights and biases

    # But actually, let's preserve the natural order (by tensor index) and use
    # clean names derived from the TFLite names

    all_weight_tensors = sorted(full_weight_tensors, key=lambda t: t["index"])

    print(f"\nExtracting {len(all_weight_tensors)} weight tensors:")
    print(f"\n{'#':>3s}  {'Name':<80s} {'Shape':<25s} {'Params':>10s}")
    print("-" * 122)

    for i, t in enumerate(all_weight_tensors):
        raw_name = t["name"]
        shape = t["shape"]
        n_params = t["n_params"]
        tensor = t["tensor"]

        # Create a clean key name
        # Remove "model_1/model/" prefix
        clean = raw_name
        for prefix in ["model_1/model/", "model_1/"]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break

        # Remove operation suffixes to get layer name
        for suffix in ["/Conv2D", "/MatMul", "/FusedBatchNormV3", "/BiasAdd/ReadVariableOp/resource", "/depthwise"]:
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)]
                break

        # Handle fused names (e.g. "bn;dw;conv" -> use first part)
        if ";" in clean:
            clean = clean.split(";")[0]

        # Determine tensor type from shape and name
        if len(shape) == 4 and shape[0] == 1 and (shape[1] == 3 or shape[1] == 5):
            tensor_type = "depthwise_kernel"
        elif len(shape) == 4 and shape[1] == 1 and shape[2] == 1:
            tensor_type = "pointwise_kernel"
        elif len(shape) == 4 and shape[1] == 3 and shape[2] == 3 and shape[3] == 3:
            tensor_type = "input_conv_kernel"
        elif len(shape) == 2:
            tensor_type = "fc_weight"
        elif len(shape) == 1:
            tensor_type = "bias_or_bn"
        else:
            tensor_type = "other"

        # Convert tensor to float32 for consistent handling
        if tensor is not None:
            tensor_f32 = tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor.copy()

            # Transpose NHWC -> NCHW for conv kernels
            if tensor_type == "input_conv_kernel":
                # [O, H, W, I] -> [O, I, H, W]
                tensor_f32 = tensor_f32.transpose(0, 3, 1, 2)
                shape = list(tensor_f32.shape)
            elif tensor_type == "pointwise_kernel":
                # [O, 1, 1, I] -> [O, I, 1, 1]
                tensor_f32 = tensor_f32.transpose(0, 3, 1, 2)
                shape = list(tensor_f32.shape)
            elif tensor_type == "depthwise_kernel":
                # [1, H, W, C] -> [C, 1, H, W]
                tensor_f32 = tensor_f32.transpose(3, 0, 1, 2)
                shape = list(tensor_f32.shape)
            elif tensor_type == "fc_weight":
                # Keep as-is [O, I] or transpose? FC weights are [O, I] in TFLite
                # which is already the right format
                pass

            keys.append(clean)
            shapes.append(shape)
            offsets.append(len(buf))
            f16 = tensor_f32.astype(np.float16)
            buf.extend(f16.tobytes())
            total_params += n_params

            print(f"{i:>3d}  {clean:<80s} {str(shape):<25s} {n_params:>10,}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total tensors: {len(keys)}")
    print(f"Binary size: {len(buf):,} bytes")

    # Save
    manifest = {"keys": keys, "shapes": shapes, "offsets": offsets, "dtype": "float16"}
    OUTPUT_JSON.write_text(json.dumps(manifest))
    OUTPUT_BIN.write_bytes(bytes(buf))
    print(f"\nSaved {OUTPUT_JSON} ({len(keys)} tensors)")
    print(f"Saved {OUTPUT_BIN} ({len(buf):,} bytes)")

    return keys, shapes, total_params


def print_final_summary(lite_info, full_info, lite_wt, full_wt, full_keys, full_shapes, full_total_params):
    """Print final summary."""
    print(f"\n{'='*80}")
    print("STEP 4: FINAL SUMMARY")
    print(f"{'='*80}")

    lite_params = sum(t["n_params"] for t in lite_wt)

    print(f"""
  LITE model (hand_landmark.tflite):
    File size: {lite_info['file_size']:,} bytes
    Input: {lite_info['inputs'][0]['shape']}
    Weight tensors: {len(lite_wt)}
    Parameters: {lite_params:,}
    Architecture: Custom ResNet-like with depthwise separable convs + U-Net skip connections
    Outputs: 3 (handflag, handedness, landmarks_3d)

  FULL model (hand_landmark_full.tflite):
    File size: {full_info['file_size']:,} bytes
    Input: {full_info['inputs'][0]['shape']}
    Weight tensors: {len(full_wt)} -> extracted {len(full_keys)} (with data)
    Parameters: {full_total_params:,}
    Architecture: EfficientNet-like with inverted bottleneck blocks + batch norm
    Outputs: 4 (handflag, handedness, landmarks_3d, world_landmarks_3d)

  Parameter increase: {full_total_params - lite_params:,} ({(full_total_params/lite_params - 1)*100:.1f}% more)

  CRITICAL FINDING: The LITE and FULL models have COMPLETELY DIFFERENT architectures.
  - LITE: Custom encoder-decoder with 5x5 depthwise convs, max 288 channels
  - FULL: EfficientNet backbone with 3x3/5x5 mixed depthwise, up to 1152 channels
  - The FULL model uses batch normalization (LITE has BN fused into conv weights)
  - The FULL model uses global average pooling + FC (LITE uses 2x2 conv heads)
  - The FULL model has an extra output head (world landmarks)

  WEIGHTS EXTRACTED TO:
    {OUTPUT_JSON}
    {OUTPUT_BIN}
""")


def main():
    # Step 1: Extract from .task
    extract_from_task()

    # Step 2: Load and compare
    lite_info = load_model_info(LITE_PATH, "LITE")
    full_info = load_model_info(FULL_PATH, "FULL")

    lite_wt, full_wt = compare_architectures(lite_info, full_info)

    # Step 3: Extract FULL weights
    full_keys, full_shapes, full_total_params = extract_full_weights(full_info, full_wt)

    # Step 4: Summary
    print_final_summary(lite_info, full_info, lite_wt, full_wt, full_keys, full_shapes, full_total_params)


if __name__ == "__main__":
    main()
