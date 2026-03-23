#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""Dump intermediate activations from TFLite palm detection - v2.

Dumps all non-weight tensors after inference for layer comparison.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "palm_detection.tflite"


def letterbox_resize(img_array, target_size=192):
    """Match our WebGPU letterbox resize."""
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
    print(f"Letterbox: offset=({offset_x},{offset_y}) scale={scale}")

    input_tensor = letterboxed[np.newaxis, ...]

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    details = interpreter.get_tensor_details()

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Now read ALL tensors and find the activation ones
    result = {}

    # Input in CHW format
    input_chw = letterboxed.transpose(2, 0, 1)
    result['input'] = {
        'shape': list(input_chw.shape),
        'min': float(input_chw.min()),
        'max': float(input_chw.max()),
        'mean': float(input_chw.mean()),
        'sample': input_chw.flatten()[:10].tolist(),
    }

    # Get op trace
    ops = interpreter._get_ops_details()

    # Map each op's output tensor index to op info
    out_to_op = {}
    for i, op in enumerate(ops):
        for oidx in op['outputs']:
            out_to_op[oidx] = (i, op['op_name'])

    # Read all tensors, filtering for activations (not weights/constants)
    activation_tensors = []
    for d in details:
        idx = d['index']
        shape = tuple(d['shape'])
        name = d['name']

        # Skip 1D tensors (biases, constants), small constants
        if len(shape) < 2:
            continue
        # Skip weight tensors (4D with dim 0 == 1 is activation, but [OC, 1, 1, IC] is weight)
        # Keep only [1, H, W, C] tensors (batch 1 activations)
        if len(shape) == 4 and shape[0] == 1 and shape[1] > 1:
            # This is likely an activation
            try:
                data = interpreter.get_tensor(idx)
                if data is None:
                    continue

                op_idx, op_name = out_to_op.get(idx, (-1, "unknown"))

                # Convert to CHW
                tensor = data[0]  # remove batch
                tensor_chw = tensor.transpose(2, 0, 1)
                flat = tensor_chw.flatten()

                activation_tensors.append({
                    'idx': idx,
                    'op_idx': op_idx,
                    'op_name': op_name,
                    'shape': shape,
                    'name': name,
                    'min': float(flat.min()),
                    'max': float(flat.max()),
                    'mean': float(flat.mean()),
                })

            except Exception as e:
                pass

    # Sort by op index
    activation_tensors.sort(key=lambda x: x['op_idx'])

    # Print and save key activations
    for info in activation_tensors:
        shape = info['shape']
        label = None
        h, w, c = shape[1], shape[2], shape[3]

        # Identify key layers
        if h == 96 and w == 96 and c == 32 and info['op_name'] == 'PRELU':
            label = 'initConv'
        elif h == 12 and w == 12 and c == 256 and info['op_name'] == 'RESIZE_BILINEAR':
            label = 'fpnUpsample6to12'
        elif h == 24 and w == 24 and c == 256 and info['op_name'] == 'RESIZE_BILINEAR':
            label = 'fpnUpsample12to24'
        elif 'conv2d_25' in info['name'] and info['op_name'] == 'PRELU':
            label = 'fpn6to12Conv'
        elif 'conv2d_28' in info['name'] and info['op_name'] == 'PRELU':
            label = 'fpn12to24Conv'
        elif 'classifier_palm_16' in info['name'] and info['op_name'] == 'CONV_2D':
            label = 'cls16'
        elif 'regressor_palm_16' in info['name'] and info['op_name'] == 'CONV_2D':
            label = 'reg16'
        elif 'classifier_palm_8' in info['name'] and info['op_name'] == 'CONV_2D':
            label = 'cls8'
        elif 'regressor_palm_8' in info['name'] and info['op_name'] == 'CONV_2D':
            label = 'reg8'
        elif info['op_name'] == 'ADD':
            # Identify stage based on spatial dims
            if h == 48 and c == 64:
                label = f'add_48x64_{info["op_idx"]}'
            elif h == 24 and c == 128:
                label = f'add_24x128_{info["op_idx"]}'
            elif h == 12 and c == 256:
                label = f'add_12x256_{info["op_idx"]}'
            elif h == 6 and c == 256:
                label = f'add_6x256_{info["op_idx"]}'

        if label:
            data = interpreter.get_tensor(info['idx'])
            tensor = data[0].transpose(2, 0, 1)
            flat = tensor.flatten()
            result[label] = {
                'shape': list(tensor.shape),
                'min': float(flat.min()),
                'max': float(flat.max()),
                'mean': float(flat.mean()),
                'nonZero': int(np.count_nonzero(flat[:1000])),
                'sample': flat[:10].tolist(),
                'data500': flat[:500].tolist(),
            }
            print(f"  Op {info['op_idx']:3d} [{label:25s}] {shape} min={flat.min():.6f} max={flat.max():.6f} mean={flat.mean():.6f}")

    # Final outputs
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        name = od['name']
        flat = data.flatten()
        result[f"output_{name}"] = {
            'shape': list(data.shape),
            'min': float(flat.min()),
            'max': float(flat.max()),
            'mean': float(flat.mean()),
            'sample': flat[:20].tolist(),
        }
        print(f"  Output [{name}] shape={data.shape} min={flat.min():.6f} max={flat.max():.6f}")

    out_path = SCRIPT_DIR / "docs" / f"tflite_ref_{image_name.replace('.jpg','')}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
