#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""Verify X/Y order in SSD decode by checking if decoded keypoints make sense."""

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
    result[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = resized
    return result, offset_x, offset_y, scale


def generate_anchors():
    anchors = []
    # Layer 0: 12x12, 6 anchors per cell
    for y in range(12):
        for x in range(12):
            cx = (x + 0.5) / 12
            cy = (y + 0.5) / 12
            for _ in range(6):
                anchors.append((cx, cy))
    # Layer 1: 24x24, 2 anchors per cell
    for y in range(24):
        for x in range(24):
            cx = (x + 0.5) / 24
            cy = (y + 0.5) / 24
            for _ in range(2):
                anchors.append((cx, cy))
    return anchors


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_detections(scores, regressors, anchors, threshold=0.3, xy_order="xy"):
    """Decode SSD output. xy_order='xy' means raw[0]=x, 'yx' means raw[0]=y."""
    detections = []
    for i in range(len(anchors)):
        score = sigmoid(float(scores[i]))
        if score < threshold:
            continue

        anchor_x, anchor_y = anchors[i]
        base = i * 18

        if xy_order == "xy":
            # Our current code: raw[0]=x, raw[1]=y
            cx = anchor_x + regressors[base + 0] / 192
            cy = anchor_y + regressors[base + 1] / 192
            w = regressors[base + 2] / 192
            h = regressors[base + 3] / 192
            kps = []
            for k in range(7):
                kx = anchor_x + regressors[base + 4 + k * 2] / 192
                ky = anchor_y + regressors[base + 4 + k * 2 + 1] / 192
                kps.append((kx, ky))
        else:
            # MediaPipe reverse_output_order: raw[0]=y, raw[1]=x
            cy = anchor_y + regressors[base + 0] / 192
            cx = anchor_x + regressors[base + 1] / 192
            h = regressors[base + 2] / 192
            w = regressors[base + 3] / 192
            kps = []
            for k in range(7):
                ky = anchor_y + regressors[base + 4 + k * 2] / 192
                kx = anchor_x + regressors[base + 4 + k * 2 + 1] / 192
                kps.append((kx, ky))

        detections.append({
            'score': score,
            'cx': cx, 'cy': cy, 'w': w, 'h': h,
            'wrist': kps[0],
            'middle_mcp': kps[2],
            'all_kps': kps,
            'anchor_idx': i,
        })

    # Sort by score
    detections.sort(key=lambda d: d['score'], reverse=True)
    return detections


def main():
    img_path = SCRIPT_DIR / "docs" / "test-hands" / "hand_07.jpg"
    img = Image.open(img_path)
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    print(f"Image: hand_07.jpg ({w}x{h})")

    letterboxed, offset_x, offset_y, scale = letterbox_resize(img_array)
    input_tensor = letterboxed[np.newaxis, ...]
    print(f"Letterbox: offset=({offset_x},{offset_y}) scale={scale:.4f}")

    lb_pad_x = offset_x / 192
    lb_pad_y = offset_y / 192

    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_PATH),
        num_threads=4,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Get outputs
    for od in output_details:
        print(f"  Output [{od['name']}] shape={od['shape']}")

    # Identity = regressors (1, 2016, 18), Identity_1 = scores (1, 2016, 1)
    regressors = None
    scores = None
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        if data.shape[-1] == 18:
            regressors = data[0].reshape(-1)  # (2016*18,)
        elif data.shape[-1] == 1:
            scores = data[0].reshape(-1)  # (2016,)

    anchors = generate_anchors()
    print(f"\nAnchors: {len(anchors)}")

    # Decode both ways
    for order in ["xy", "yx"]:
        dets = decode_detections(scores, regressors, anchors, threshold=0.3, xy_order=order)
        print(f"\n=== Decode order: {order} ===")
        print(f"  Detections above 0.3: {len(dets)}")
        if len(dets) == 0:
            continue

        top = dets[0]
        print(f"  Top detection:")
        print(f"    score={top['score']:.4f}  anchor_idx={top['anchor_idx']}")
        print(f"    box: cx={top['cx']:.4f} cy={top['cy']:.4f} w={top['w']:.4f} h={top['h']:.4f}")
        print(f"    wrist(kp0): ({top['wrist'][0]:.4f}, {top['wrist'][1]:.4f})")
        print(f"    middle_mcp(kp2): ({top['middle_mcp'][0]:.4f}, {top['middle_mcp'][1]:.4f})")

        # Remove letterbox
        sx = 1 / (1 - 2 * lb_pad_x)
        sy = 1 / (1 - 2 * lb_pad_y)
        wrist_img = ((top['wrist'][0] - lb_pad_x) * sx, (top['wrist'][1] - lb_pad_y) * sy)
        mcp_img = ((top['middle_mcp'][0] - lb_pad_x) * sx, (top['middle_mcp'][1] - lb_pad_y) * sy)

        print(f"    After letterbox removal:")
        print(f"      wrist: ({wrist_img[0]:.4f}, {wrist_img[1]:.4f}) -> pixel ({wrist_img[0]*w:.1f}, {wrist_img[1]*h:.1f})")
        print(f"      mcp:   ({mcp_img[0]:.4f}, {mcp_img[1]:.4f}) -> pixel ({mcp_img[0]*w:.1f}, {mcp_img[1]*h:.1f})")

        # Which makes more sense for hand_07?
        # hand_07 shows a hand in the center-right area of the image


if __name__ == "__main__":
    main()
