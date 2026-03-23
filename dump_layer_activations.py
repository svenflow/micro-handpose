#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""Dump intermediate activations from TFLite palm detection for layer-by-layer comparison.

Preprocesses test image with exact same letterbox resize as our WebGPU code,
runs TFLite inference, and dumps key intermediate tensors to JSON.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "palm_detection.tflite"


def letterbox_resize(img_array, target_size=192):
    """Match our WebGPU letterbox resize exactly."""
    h, w = img_array.shape[:2]
    scale = min(target_size / w, target_size / h)
    scaled_w = round(w * scale)
    scaled_h = round(h * scale)
    offset_x = (target_size - scaled_w) // 2
    offset_y = (target_size - scaled_h) // 2

    # Use PIL for bilinear resize (matches GPU hardware bilinear)
    pil_img = Image.fromarray(img_array)
    pil_resized = pil_img.resize((scaled_w, scaled_h), Image.BILINEAR)
    resized = np.array(pil_resized).astype(np.float32) / 255.0

    # Place in letterbox
    result = np.zeros((target_size, target_size, 3), dtype=np.float32)
    result[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = resized

    return result, offset_x, offset_y, scale


def main():
    import sys
    image_name = sys.argv[1] if len(sys.argv) > 1 else "hand_07.jpg"

    # Find image
    img_path = SCRIPT_DIR / "docs" / image_name
    if not img_path.exists():
        img_path = SCRIPT_DIR / "docs" / "test-hands" / image_name
    if not img_path.exists():
        print(f"Image not found: {image_name}")
        return

    # Load and preprocess
    img = Image.open(img_path)
    img_array = np.array(img)
    print(f"Image: {image_name} shape={img_array.shape}")

    letterboxed, offset_x, offset_y, scale = letterbox_resize(img_array)
    print(f"Letterbox: offset=({offset_x},{offset_y}) scale={scale}")

    # Prepare input: TFLite expects [1, 192, 192, 3] in [0, 1]
    input_tensor = letterboxed[np.newaxis, ...]  # [1, 192, 192, 3]

    # Setup interpreter with tensor allocation
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    details = interpreter.get_tensor_details()

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Get ops info
    ops = interpreter._get_ops_details()

    # Build tensor name → details map
    tensor_map = {d['index']: d for d in details}

    # Dump activations at key layers
    result = {}

    # Input
    input_data = interpreter.get_tensor(input_details[0]['index'])
    # Convert to CHW [0,1] for comparison with our GPU format
    input_chw = input_data[0].transpose(2, 0, 1)  # [3, 192, 192]
    result['input'] = {
        'shape': list(input_chw.shape),
        'min': float(input_chw.min()),
        'max': float(input_chw.max()),
        'mean': float(input_chw.mean()),
        'sample': input_chw.flatten()[:10].tolist(),
    }

    # Find key ops and their output tensors
    # We want:
    # 1. After initial conv2d_0 + PReLU (96x96x32)
    # 2. After each stage boundary (stride-2 blocks)
    # 3. FPN intermediate tensors
    # 4. SSD head outputs

    # Map op names to their output tensors
    print(f"\n=== Key ops ===")
    key_tensors = {}

    for i, op in enumerate(ops):
        op_name = op['op_name']
        outputs = list(op['outputs'])

        # Skip DEQUANTIZE ops (just weight conversion)
        if op_name == 'DEQUANTIZE':
            continue

        # For key layers, grab the output tensor
        if outputs:
            out_idx = outputs[0]
            if out_idx in tensor_map:
                out_shape = tuple(tensor_map[out_idx]['shape'])
                out_name = tensor_map[out_idx]['name']

                # Print significant ops
                if op_name in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'ADD', 'RESIZE_BILINEAR', 'PRELU', 'PAD'):
                    if any(s > 1 for s in out_shape[1:3]):  # spatial dims > 1
                        key_tensors[i] = {
                            'op_name': op_name,
                            'out_idx': out_idx,
                            'out_shape': out_shape,
                            'out_name': out_name[:80],
                        }

    # Now read back intermediate tensors at key points
    # TFLite doesn't easily let us read intermediates after invoke(),
    # but the tensor data IS available after invoke
    print("\n=== Reading intermediate activations ===")

    # Find specific layers we care about
    for i, info in sorted(key_tensors.items()):
        try:
            data = interpreter.get_tensor(info['out_idx'])
            if data.size == 0:
                continue

            # Only dump key layers (by shape and name)
            shape = info['out_shape']
            name = info['out_name']
            op_name = info['op_name']

            # Key shapes we want:
            interesting = False
            label = None

            if shape == (1, 96, 96, 32) and op_name == 'PRELU':
                interesting = True
                label = 'initConv'  # After initial conv + PReLU
            elif shape == (1, 48, 48, 64) and op_name == 'ADD':
                interesting = True
                label = f'stage1_add_{i}'
            elif shape == (1, 24, 24, 128) and op_name == 'ADD':
                interesting = True
                label = f'stage2_add_{i}'
            elif shape == (1, 12, 12, 256) and op_name == 'ADD':
                interesting = True
                label = f'stage3_add_{i}'
            elif shape == (1, 6, 6, 256) and op_name == 'ADD':
                interesting = True
                label = f'stage4_add_{i}'
            elif shape == (1, 12, 12, 256) and op_name == 'RESIZE_BILINEAR':
                interesting = True
                label = 'fpnUpsample6to12'
            elif shape == (1, 24, 24, 256) and op_name == 'RESIZE_BILINEAR':
                interesting = True
                label = 'fpnUpsample12to24'
            elif 'conv2d_25' in name and op_name == 'PRELU':
                interesting = True
                label = 'fpn6to12Conv'
            elif 'conv2d_28' in name and op_name == 'PRELU':
                interesting = True
                label = 'fpn12to24Conv'
            # SSD heads
            elif 'classifier_palm_16' in name and op_name == 'CONV_2D':
                interesting = True
                label = 'cls16'
            elif 'regressor_palm_16' in name and op_name == 'CONV_2D':
                interesting = True
                label = 'reg16'
            elif 'classifier_palm_8' in name and op_name == 'CONV_2D':
                interesting = True
                label = 'cls8'
            elif 'regressor_palm_8' in name and op_name == 'CONV_2D':
                interesting = True
                label = 'reg8'

            if interesting:
                tensor = data[0]  # Remove batch dim
                # Convert to CHW for comparison with our GPU format
                if len(tensor.shape) == 3:
                    tensor_chw = tensor.transpose(2, 0, 1)  # [C, H, W]
                else:
                    tensor_chw = tensor

                flat = tensor_chw.flatten()
                result[label] = {
                    'shape': list(tensor_chw.shape),
                    'min': float(flat.min()),
                    'max': float(flat.max()),
                    'mean': float(flat.mean()),
                    'nonZero': int(np.count_nonzero(flat[:1000])),
                    'sample': flat[:10].tolist(),
                    # Store first 500 values for detailed comparison
                    'data500': flat[:500].tolist(),
                }
                print(f"  Op {i:3d} [{label:25s}] shape={shape} min={flat.min():.6f} max={flat.max():.6f} mean={flat.mean():.6f}")

        except Exception as e:
            pass

    # Final outputs
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        name = od['name']
        flat = data.flatten()
        label = f"output_{name}"
        result[label] = {
            'shape': list(data.shape),
            'min': float(flat.min()),
            'max': float(flat.max()),
            'mean': float(flat.mean()),
            'sample': flat[:20].tolist(),
        }
        print(f"  Output [{name}] shape={data.shape} min={flat.min():.6f} max={flat.max():.6f}")

    # Write to JSON
    out_path = SCRIPT_DIR / "docs" / f"tflite_activations_{image_name.replace('.jpg','')}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
