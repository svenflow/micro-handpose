#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""Dump intermediate activations from TFLite - no delegate for full op trace."""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "palm_detection.tflite"


def letterbox_resize(img_array, target_size=192):
    h, w = img_array.shape[:2]
    scale = min(target_size / w, target_size / h)
    scaled_w = round(w * scale)
    scaled_h = round(h * scale)
    offset_x = (target_size - scaled_w) // 2
    offset_y = (target_size - scaled_h) // 2
    pil_img = Image.fromarray(img_array)
    pil_resized = pil_img.resize((scaled_w, scaled_h), Image.BILINEAR)
    resized = np.array(pil_resized).astype(np.float32) / 255.0
    result = np.zeros((target_size, target_size, 3), dtype=np.float32)
    result[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = resized
    return result, offset_x, offset_y, scale


def main():
    import sys
    image_name = sys.argv[1] if len(sys.argv) > 1 else "hand_07.jpg"

    img_path = SCRIPT_DIR / "docs" / "test-hands" / image_name
    if not img_path.exists():
        img_path = SCRIPT_DIR / "docs" / image_name
    if not img_path.exists():
        print(f"Image not found: {image_name}")
        return

    img = Image.open(img_path)
    img_array = np.array(img)
    print(f"Image: {image_name} shape={img_array.shape}")

    letterboxed, offset_x, offset_y, scale = letterbox_resize(img_array)
    input_tensor = letterboxed[np.newaxis, ...]

    # Disable XNNPACK delegate to get individual ops
    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_PATH),
        experimental_preserve_all_tensors=True,
        num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()

    details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    ops = interpreter._get_ops_details()
    out_to_op = {}
    for i, op in enumerate(ops):
        for oidx in op['outputs']:
            out_to_op[oidx] = (i, op['op_name'])

    result = {}

    # Input in CHW
    input_chw = letterboxed.transpose(2, 0, 1)
    flat_input = input_chw.flatten()

    # Sample spatial center: 3x3 patch at (H/2, W/2) for each channel
    C_in, H_in, W_in = input_chw.shape
    center_samples_input = []
    cr_in, cc_in = H_in // 2, W_in // 2
    for ch in range(C_in):
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, cc = cr_in + dr, cc_in + dc
                if 0 <= r < H_in and 0 <= cc < W_in and len(center_samples_input) < 500:
                    center_samples_input.append(float(input_chw[ch, r, cc]))

    result['input'] = {
        'shape': list(input_chw.shape),
        'min': float(flat_input.min()),
        'max': float(flat_input.max()),
        'mean': float(flat_input.mean()),
        'sample': flat_input[:10].tolist(),
        'data500': flat_input[:500].tolist(),
        'dataCenter500': center_samples_input,
        'totalLength': len(flat_input),
    }

    print("\n=== Key activation tensors ===")
    for d in details:
        idx = d['index']
        shape = tuple(d['shape'])
        name = d['name']

        if len(shape) != 4 or shape[0] != 1 or shape[1] <= 1:
            continue

        op_idx, op_name = out_to_op.get(idx, (-1, "?"))
        if op_idx == -1:
            continue

        try:
            data = interpreter.get_tensor(idx)
            h, w, c = shape[1], shape[2], shape[3]
            flat = data.flatten()

            label = None

            # Key layers — only label the FIRST PReLU at 96x96x32 as initConv
            if h == 96 and c == 32 and op_name == 'PRELU' and 'initConv' not in result:
                label = 'initConv'
            elif h == 12 and c == 256 and op_name == 'RESIZE_BILINEAR':
                label = 'fpnUpsample6to12'
            elif h == 24 and c == 256 and op_name == 'RESIZE_BILINEAR':
                label = 'fpnUpsample12to24'
            elif 'conv2d_25' in name and (op_name == 'PRELU' or op_name == 'CONV_2D'):
                label = 'fpn6to12Conv'
            elif 'conv2d_28' in name and (op_name == 'PRELU' or op_name == 'CONV_2D'):
                label = 'fpn12to24Conv'
            elif 'classifier_palm_16' in name and op_name == 'CONV_2D':
                label = 'cls16'
            elif 'regressor_palm_16' in name and op_name == 'CONV_2D':
                label = 'reg16'
            elif 'classifier_palm_8' in name and op_name == 'CONV_2D':
                label = 'cls8'
            elif 'regressor_palm_8' in name and op_name == 'CONV_2D':
                label = 'reg8'
            # Stage boundaries - last ADD in each stage
            elif op_name == 'ADD':
                if h == 96 and c == 32:
                    label = f'add_96x32_op{op_idx}'
                elif h == 48 and c == 64:
                    label = f'add_48x64_op{op_idx}'
                elif h == 24 and c == 128:
                    label = f'add_24x128_op{op_idx}'
                elif h == 12 and c == 256:
                    label = f'add_12x256_op{op_idx}'
                elif h == 6 and c == 256:
                    label = f'add_6x256_op{op_idx}'

            if label:
                tensor = data[0].transpose(2, 0, 1)  # NHWC → CHW
                C, H, W = tensor.shape
                flat_chw = tensor.flatten()
                mid_start = max(0, len(flat_chw) // 2 - 250)

                # Sample spatial center: 3x3 patch at (H/2, W/2) for each channel
                center_samples = []
                center_r, center_c = H // 2, W // 2
                for ch in range(C):
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            r, cc = center_r + dr, center_c + dc
                            if 0 <= r < H and 0 <= cc < W and len(center_samples) < 500:
                                center_samples.append(float(tensor[ch, r, cc]))
                    if len(center_samples) >= 500:
                        break

                result[label] = {
                    'shape': list(tensor.shape),
                    'min': float(flat_chw.min()),
                    'max': float(flat_chw.max()),
                    'mean': float(flat_chw.mean()),
                    'nonZero': int(np.count_nonzero(flat_chw[:1000])),
                    'sample': flat_chw[:10].tolist(),
                    'data500': flat_chw[:500].tolist(),
                    'dataMid500': flat_chw[mid_start:mid_start+500].tolist(),
                    'dataCenter500': center_samples,
                    'totalLength': len(flat_chw),
                }
                print(f"  Op {op_idx:3d} [{label:25s}] {shape} min={flat.min():.6f} max={flat.max():.6f} mean={flat.mean():.6f}")

        except:
            pass

    out_path = SCRIPT_DIR / "docs" / f"tflite_ref_{image_name.replace('.jpg','')}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path} ({len(result)} layers)")


if __name__ == "__main__":
    main()
