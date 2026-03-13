#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "pillow", "tensorflow"]
# ///
"""
Inspect the palm detection TFLite model structure and run inference
to understand its output format and how to decode detections.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf

project_dir = Path(__file__).parent
tflite_path = project_dir / "palm_detection.tflite"

interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== Palm Detection Model ===")
for inp in input_details:
    print(f"Input: {inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")
for out in output_details:
    print(f"Output: {out['name']}, shape={out['shape']}, dtype={out['dtype']}")

# Load test image
img = Image.open(project_dir / "docs" / "hand_nikhil.jpg").convert("RGB")
input_shape = input_details[0]['shape']
h, w = input_shape[1], input_shape[2]
print(f"\nModel input size: {w}x{h}")
img_resized = img.resize((w, h), Image.BILINEAR)
pixels = np.array(img_resized, dtype=np.float32) / 255.0
input_data = pixels.reshape(1, h, w, 3)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

for out in output_details:
    tensor = interpreter.get_tensor(out['index'])
    print(f"\nOutput '{out['name']}':")
    print(f"  shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}")
    if tensor.shape[-1] == 1:
        # Scores - find top detections
        scores = tensor.flatten()
        # Apply sigmoid
        scores_sigmoid = 1.0 / (1.0 + np.exp(-scores))
        top_indices = np.argsort(scores_sigmoid)[-10:][::-1]
        print(f"  Top 10 scores (sigmoid):")
        for idx in top_indices:
            print(f"    [{idx}] score={scores_sigmoid[idx]:.4f} (raw={scores[idx]:.4f})")
    elif tensor.shape[-1] == 18:
        # Boxes: 18 values = 4 bbox + 7 keypoints * 2
        print(f"  Format: [cx, cy, w, h, kp0_x, kp0_y, kp1_x, kp1_y, ..., kp6_x, kp6_y]")
        # Find the box corresponding to highest score
        scores_tensor = None
        for out2 in output_details:
            t2 = interpreter.get_tensor(out2['index'])
            if t2.shape[-1] == 1:
                scores_tensor = t2.flatten()
                break
        if scores_tensor is not None:
            scores_sigmoid = 1.0 / (1.0 + np.exp(-scores_tensor))
            best_idx = np.argmax(scores_sigmoid)
            box = tensor[0, best_idx]
            print(f"  Best detection (idx={best_idx}, score={scores_sigmoid[best_idx]:.4f}):")
            print(f"    bbox: cx={box[0]:.1f}, cy={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}")
            print(f"    Keypoints (in pixels of {w}x{h} input):")
            kp_names = ['wrist_center', 'kp1', 'middle_finger_mcp', 'kp3', 'kp4', 'kp5', 'kp6']
            for ki in range(7):
                kx = box[4 + ki*2]
                ky = box[5 + ki*2]
                print(f"      kp{ki} ({kp_names[ki]}): ({kx:.1f}, {ky:.1f}) = ({kx/w:.3f}, {ky/h:.3f}) normalized")

# Generate anchors (same as MediaPipe config)
print("\n=== Anchor Generation ===")
def generate_anchors(input_size=192, strides=[8, 16, 16, 16]):
    """Generate SSD anchors matching MediaPipe's config."""
    anchors = []
    for stride in strides:
        grid_h = input_size // stride
        grid_w = input_size // stride
        # 2 anchors per grid cell for stride 8, 6 for others?
        # Actually MediaPipe uses fixed_anchor_size=true with num_layers=4
        # Let's compute based on the actual anchor count
        for y in range(grid_h):
            for x in range(grid_w):
                cx = (x + 0.5) / grid_w
                cy = (y + 0.5) / grid_h
                anchors.append((cx, cy))
                anchors.append((cx, cy))  # 2 anchors per location
    return np.array(anchors)

anchors = generate_anchors()
print(f"Generated {len(anchors)} anchors")

# Decode the best detection using anchors
scores_tensor = None
boxes_tensor = None
for out in output_details:
    t = interpreter.get_tensor(out['index'])
    if t.shape[-1] == 1:
        scores_tensor = t[0].flatten()
    elif t.shape[-1] == 18:
        boxes_tensor = t[0]

if scores_tensor is not None and boxes_tensor is not None:
    scores_sigmoid = 1.0 / (1.0 + np.exp(-scores_tensor))
    best_idx = np.argmax(scores_sigmoid)
    print(f"\nBest detection: idx={best_idx}, score={scores_sigmoid[best_idx]:.4f}")

    raw = boxes_tensor[best_idx]
    anchor = anchors[best_idx]

    # Decode: cx = raw_cx / input_size + anchor_cx, etc.
    input_size = w
    cx = raw[0] / input_size + anchor[0]
    cy = raw[1] / input_size + anchor[1]
    bw = raw[2] / input_size
    bh = raw[3] / input_size

    print(f"Decoded bbox (normalized): cx={cx:.3f}, cy={cy:.3f}, w={bw:.3f}, h={bh:.3f}")
    print(f"  x1={cx-bw/2:.3f}, y1={cy-bh/2:.3f}, x2={cx+bw/2:.3f}, y2={cy+bh/2:.3f}")

    # Decode keypoints
    print(f"Decoded keypoints (normalized):")
    kp_names = ['wrist_center', 'kp1', 'middle_finger_mcp', 'kp3', 'kp4', 'kp5', 'kp6']
    kps = []
    for ki in range(7):
        kx = raw[4 + ki*2] / input_size + anchor[0]
        ky = raw[5 + ki*2] / input_size + anchor[1]
        kps.append((kx, ky))
        print(f"  kp{ki} ({kp_names[ki]}): ({kx:.3f}, {ky:.3f})")

    # Compute rotation (wrist center to middle finger MCP)
    # keypoint 0 = wrist center, keypoint 2 = middle finger MCP
    kp0 = kps[0]  # wrist center
    kp2 = kps[2]  # middle finger MCP
    target_angle = np.pi / 2  # 90 degrees

    angle = target_angle - np.arctan2(-(kp2[1] - kp0[1]), kp2[0] - kp0[0])
    # Normalize to [-pi, pi]
    angle = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
    print(f"\nRotation: {np.degrees(angle):.1f} degrees")

    # Compute ROI (scale 2.6, shift_y -0.5)
    # 1. Compute rotated rect from detection
    # 2. Scale by 2.6x2.6
    # 3. Shift by -0.5 along y-axis of the rotated rect
    # 4. Make square (use longer side)
    rect_cx = (cx + kp0[0] + kp2[0]) / 3  # rough center
    rect_cy = (cy + kp0[1] + kp2[1]) / 3

    print(f"\nPalm detection center: ({cx:.3f}, {cy:.3f})")
    print(f"Wrist: ({kp0[0]:.3f}, {kp0[1]:.3f})")
    print(f"Middle finger MCP: ({kp2[0]:.3f}, {kp2[1]:.3f})")
