#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""Dump ALL tensor values from TFLite after inference - debug version."""

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

    # Use preserve_all_tensors to keep intermediates
    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_PATH),
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()

    details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Get op trace
    ops = interpreter._get_ops_details()
    out_to_op = {}
    for i, op in enumerate(ops):
        for oidx in op['outputs']:
            out_to_op[oidx] = (i, op['op_name'])

    # Read ALL 4D tensors with batch=1
    result = {}
    print("\n=== All activation-like tensors ===")
    for d in details:
        idx = d['index']
        shape = tuple(d['shape'])
        name = d['name']

        if len(shape) != 4 or shape[0] != 1:
            continue
        if shape[1] <= 1 and shape[2] <= 1:
            continue  # Skip [1,1,1,C] weight-like

        try:
            data = interpreter.get_tensor(idx)
            flat = data.flatten()
            op_idx, op_name = out_to_op.get(idx, (-1, "?"))

            # Only show non-zero tensors with spatial dims > 1
            if np.all(data == 0) and op_idx == -1:
                continue

            h, w, c = shape[1], shape[2], shape[3]
            label = None

            # Identify key layers
            if h == 96 and w == 96 and c == 32 and op_name == 'PRELU':
                label = 'initConv'
            elif h == 12 and w == 12 and c == 256 and op_name == 'RESIZE_BILINEAR':
                label = 'fpnUpsample6to12'
            elif h == 24 and w == 24 and c == 256 and op_name == 'RESIZE_BILINEAR':
                label = 'fpnUpsample12to24'
            elif 'conv2d_25' in name and op_name == 'PRELU':
                label = 'fpn6to12Conv'
            elif 'conv2d_28' in name and op_name == 'PRELU':
                label = 'fpn12to24Conv'
            elif 'classifier_palm_16' in name:
                label = 'cls16'
            elif 'regressor_palm_16' in name:
                label = 'reg16'
            elif 'classifier_palm_8' in name:
                label = 'cls8'
            elif 'regressor_palm_8' in name:
                label = 'reg8'

            print(f"  [{idx:3d}] op={op_idx:3d} {op_name:25s} shape={shape} min={flat.min():.4f} max={flat.max():.4f} mean={flat.mean():.4f} {name[:60]} {'<-- '+label if label else ''}")

            if label:
                tensor = data[0].transpose(2, 0, 1) if len(data[0].shape) == 3 else data[0]
                flat_chw = tensor.flatten()
                result[label] = {
                    'shape': list(tensor.shape),
                    'min': float(flat_chw.min()),
                    'max': float(flat_chw.max()),
                    'mean': float(flat_chw.mean()),
                    'nonZero': int(np.count_nonzero(flat_chw[:1000])),
                    'sample': flat_chw[:10].tolist(),
                    'data500': flat_chw[:500].tolist(),
                }

        except Exception as e:
            pass

    # Also add input
    input_chw = letterboxed.transpose(2, 0, 1)
    result['input'] = {
        'shape': list(input_chw.shape),
        'min': float(input_chw.min()),
        'max': float(input_chw.max()),
        'mean': float(input_chw.mean()),
        'sample': input_chw.flatten()[:10].tolist(),
        'data500': input_chw.flatten()[:500].tolist(),
    }

    out_path = SCRIPT_DIR / "docs" / f"tflite_ref_{image_name.replace('.jpg','')}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path} ({len(result)} layers)")


if __name__ == "__main__":
    main()
