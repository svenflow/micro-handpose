#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""
Ground truth palm detection script using TFLite.
Loads palm_detection.tflite, runs inference on docs/hand_nikhil.jpg,
decodes SSD outputs, runs NMS, and computes the hand crop ROI.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf

project_dir = Path(__file__).parent

# ─── 1. Load model ───────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOADING PALM DETECTION MODEL")
print("=" * 60)

tflite_path = project_dir / "palm_detection.tflite"
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for inp in input_details:
    print(f"  Input:  {inp['name']}, shape={inp['shape']}, dtype={inp['dtype'].__name__}")
for out in output_details:
    print(f"  Output: {out['name']}, shape={out['shape']}, dtype={out['dtype'].__name__}")

# ─── 2. Load & preprocess image ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. LOADING AND PREPROCESSING IMAGE")
print("=" * 60)

img_path = project_dir / "docs" / "hand_nikhil.jpg"
img_orig = Image.open(img_path).convert("RGB")
orig_w, orig_h = img_orig.size
print(f"  Original image: {orig_w}x{orig_h}")

INPUT_SIZE = 192
img_resized = img_orig.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
pixels = np.array(img_resized, dtype=np.float32) / 255.0
input_data = pixels[np.newaxis, ...]  # [1, 192, 192, 3]
print(f"  Resized to: {INPUT_SIZE}x{INPUT_SIZE}, normalized to [0, 1]")
print(f"  Input tensor shape: {input_data.shape}, min={input_data.min():.4f}, max={input_data.max():.4f}")

# ─── 3. Run inference ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. RUNNING INFERENCE")
print("=" * 60)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Extract outputs — Identity [1,2016,18] = regressors, Identity_1 [1,2016,1] = scores
raw_regs = None
raw_scores = None
for out in output_details:
    tensor = interpreter.get_tensor(out['index'])
    print(f"  Output '{out['name']}': shape={tensor.shape}, "
          f"min={tensor.min():.4f}, max={tensor.max():.4f}")
    if tensor.shape[-1] == 18:
        raw_regs = tensor[0]    # [2016, 18]
    elif tensor.shape[-1] == 1:
        raw_scores = tensor[0].flatten()  # [2016]

assert raw_regs is not None and raw_scores is not None, "Could not find expected outputs"

# ─── 4. Generate SSD anchors ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. GENERATING SSD ANCHORS")
print("=" * 60)

def generate_anchors():
    """
    MediaPipe palm detection anchor layout:
      Layer 0: 12x12 grid, 6 anchors per cell  → 12*12*6 = 864 anchors (0..863)
      Layer 1: 24x24 grid, 2 anchors per cell  → 24*24*2 = 1152 anchors (864..2015)
    Anchor center: (col + 0.5) / grid_size, (row + 0.5) / grid_size  (normalized)
    """
    anchors = []
    configs = [
        (12, 6),   # layer 0: 12x12 grid, 6 anchors per cell
        (24, 2),   # layer 1: 24x24 grid, 2 anchors per cell
    ]
    for grid_size, anchors_per_cell in configs:
        for row in range(grid_size):
            for col in range(grid_size):
                cx = (col + 0.5) / grid_size
                cy = (row + 0.5) / grid_size
                for _ in range(anchors_per_cell):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)  # [2016, 2]

anchors = generate_anchors()
print(f"  Generated {len(anchors)} anchors")
print(f"  Layer 0: anchors 0..863   (12x12 grid, 6/cell)")
print(f"  Layer 1: anchors 864..2015 (24x24 grid, 2/cell)")

# ─── 5. Decode SSD detections ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. DECODING SSD OUTPUT (SIGMOID SCORES + ANCHOR DECODE)")
print("=" * 60)

# Sigmoid on scores
scores = 1.0 / (1.0 + np.exp(-raw_scores))  # [2016]

# Decode boxes: regression values are in pixel space relative to INPUT_SIZE,
# offset from anchor centers (normalized).
#   decoded_cx = raw[0] / INPUT_SIZE + anchor_cx
#   decoded_cy = raw[1] / INPUT_SIZE + anchor_cy
#   decoded_w  = raw[2] / INPUT_SIZE
#   decoded_h  = raw[3] / INPUT_SIZE
decoded_cx = raw_regs[:, 0] / INPUT_SIZE + anchors[:, 0]
decoded_cy = raw_regs[:, 1] / INPUT_SIZE + anchors[:, 1]
decoded_w  = raw_regs[:, 2] / INPUT_SIZE
decoded_h  = raw_regs[:, 3] / INPUT_SIZE

# Decode keypoints (7 kps × 2 coords, starting at index 4)
# kp_x = raw[4 + 2*k] / INPUT_SIZE + anchor_cx
# kp_y = raw[5 + 2*k] / INPUT_SIZE + anchor_cy
num_kps = 7
kps = np.zeros((2016, num_kps, 2), dtype=np.float32)
for k in range(num_kps):
    kps[:, k, 0] = raw_regs[:, 4 + 2*k] / INPUT_SIZE + anchors[:, 0]
    kps[:, k, 1] = raw_regs[:, 5 + 2*k] / INPUT_SIZE + anchors[:, 1]

# Convert cx/cy/w/h → x1/y1/x2/y2 for NMS
x1 = decoded_cx - decoded_w / 2
y1 = decoded_cy - decoded_h / 2
x2 = decoded_cx + decoded_w / 2
y2 = decoded_cy + decoded_h / 2

print(f"  Score statistics: min={scores.min():.4f}, max={scores.max():.4f}, "
      f"mean={scores.mean():.4f}")
print(f"  Detections with score > 0.5: {(scores > 0.5).sum()}")
print(f"  Detections with score > 0.1: {(scores > 0.1).sum()}")

# ─── 6. Non-Maximum Suppression ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. NON-MAXIMUM SUPPRESSION")
print("=" * 60)

SCORE_THRESHOLD = 0.5
IOU_THRESHOLD   = 0.3
MAX_DETECTIONS  = 10

def nms(x1, y1, x2, y2, scores, score_thresh, iou_thresh, max_det):
    """Simple greedy NMS."""
    keep_mask = scores >= score_thresh
    idxs = np.where(keep_mask)[0]
    if len(idxs) == 0:
        return []

    # Sort by score descending
    order = idxs[np.argsort(scores[idxs])[::-1]]
    kept = []

    while len(order) > 0 and len(kept) < max_det:
        i = order[0]
        kept.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]

        # Compute IoU of i with rest
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        intersection = inter_w * inter_h
        area_i    = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_rest = (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])
        union = area_i + area_rest - intersection
        iou = intersection / (union + 1e-6)

        order = rest[iou <= iou_thresh]

    return kept

kept_indices = nms(x1, y1, x2, y2, scores,
                   SCORE_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS)
print(f"  Score threshold: {SCORE_THRESHOLD}")
print(f"  IoU   threshold: {IOU_THRESHOLD}")
print(f"  Detections after NMS: {len(kept_indices)}")

# ─── 7. Print top detections ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TOP DETECTIONS")
print("=" * 60)

KP_NAMES = [
    "wrist_center (kp0)",
    "index_mcp    (kp1)",
    "middle_mcp   (kp2)",
    "ring_mcp     (kp3)",
    "pinky_mcp    (kp4)",
    "kp5",
    "kp6",
]

for rank, idx in enumerate(kept_indices):
    s  = scores[idx]
    bx1, by1, bx2, by2 = x1[idx], y1[idx], x2[idx], y2[idx]
    bcx, bcy, bw, bh   = decoded_cx[idx], decoded_cy[idx], decoded_w[idx], decoded_h[idx]
    print(f"\n  Detection #{rank + 1}  (anchor idx={idx}, score={s:.4f})")
    print(f"    BBox normalized: cx={bcx:.4f} cy={bcy:.4f} w={bw:.4f} h={bh:.4f}")
    print(f"    BBox normalized: x1={bx1:.4f} y1={by1:.4f} x2={bx2:.4f} y2={by2:.4f}")
    print(f"    BBox pixels:     x1={bx1*INPUT_SIZE:.1f} y1={by1*INPUT_SIZE:.1f} "
          f"x2={bx2*INPUT_SIZE:.1f} y2={by2*INPUT_SIZE:.1f}")
    print(f"    Keypoints (normalized):")
    for k in range(num_kps):
        print(f"      {KP_NAMES[k]}: ({kps[idx, k, 0]:.4f}, {kps[idx, k, 1]:.4f})")

# ─── 8. Compute hand crop ROI for top detection ───────────────────────────────
print("\n" + "=" * 60)
print("8. HAND CROP ROI (MediaPipe parameters)")
print("=" * 60)

if not kept_indices:
    print("  No detections — cannot compute ROI.")
else:
    best = kept_indices[0]
    print(f"  Using detection #{1} (anchor idx={best}, score={scores[best]:.4f})")

    # MediaPipe uses kp0 (wrist center) and kp2 (middle finger MCP) to define rotation.
    # All coordinates are normalized [0, 1].
    kp0 = kps[best, 0]   # wrist center  (x, y)
    kp2 = kps[best, 2]   # middle MCP    (x, y)

    print(f"\n  kp0 wrist center: ({kp0[0]:.4f}, {kp0[1]:.4f})")
    print(f"  kp2 middle  MCP : ({kp2[0]:.4f}, {kp2[1]:.4f})")

    # ── Rotation ──────────────────────────────────────────────────────────────
    # MediaPipe target angle = π/2 (pointing upward in image coords).
    # Angle of vector from kp0→kp2 in image space (y-axis flipped for screen coords).
    TARGET_ANGLE = np.pi / 2.0
    dx = kp2[0] - kp0[0]
    dy = kp2[1] - kp0[1]
    # In image coordinates y increases downward, so negate dy for math angle.
    angle = TARGET_ANGLE - np.arctan2(-dy, dx)
    # Normalize to [-π, π]
    angle = angle - 2.0 * np.pi * np.floor((angle + np.pi) / (2.0 * np.pi))

    print(f"\n  Rotation angle: {np.degrees(angle):.4f} degrees ({angle:.6f} radians)")

    # ── ROI center: midpoint of kp0 and kp2 ──────────────────────────────────
    # MediaPipe centers the crop at the midpoint of wrist–middle_MCP.
    roi_cx = (kp0[0] + kp2[0]) / 2.0
    roi_cy = (kp0[1] + kp2[1]) / 2.0

    print(f"  ROI center (normalized): ({roi_cx:.4f}, {roi_cy:.4f})")

    # ── ROI size: distance kp0→kp2, scaled by 2.6 ────────────────────────────
    # MediaPipe scale factor = 2.6
    SCALE = 2.6
    dist = np.sqrt(dx**2 + dy**2)
    roi_size = dist * SCALE   # normalized units

    print(f"  kp0→kp2 distance (normalized): {dist:.4f}")
    print(f"  ROI size (normalized, scale={SCALE}): {roi_size:.4f}")
    print(f"  ROI size (pixels, INPUT_SIZE={INPUT_SIZE}): {roi_size * INPUT_SIZE:.1f}")

    # ── Shift ROI center along the palm axis ──────────────────────────────────
    # MediaPipe shifts the center by –0.5 × dist along the kp0→kp2 direction
    # (i.e., toward kp2 / fingertips by half the kp0→kp2 distance).
    SHIFT = -0.5   # in units of dist, along the kp0→kp2 axis
    # Unit vector along kp0→kp2
    if dist > 1e-8:
        ux = dx / dist
        uy = dy / dist
    else:
        ux, uy = 0.0, 0.0

    roi_cx_shifted = roi_cx + SHIFT * dist * ux
    roi_cy_shifted = roi_cy + SHIFT * dist * uy

    print(f"  ROI center after shift (normalized): ({roi_cx_shifted:.4f}, {roi_cy_shifted:.4f})")
    print(f"  ROI center after shift (pixels):     "
          f"({roi_cx_shifted * INPUT_SIZE:.1f}, {roi_cy_shifted * INPUT_SIZE:.1f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │              CROP PARAMETERS                │")
    print("  ├─────────────────────────────────────────────┤")
    print(f"  │  center_x (norm):  {roi_cx_shifted:.6f}             │")
    print(f"  │  center_y (norm):  {roi_cy_shifted:.6f}             │")
    print(f"  │  size     (norm):  {roi_size:.6f}             │")
    print(f"  │  rotation (deg):   {np.degrees(angle):.4f}             │")
    print(f"  │  rotation (rad):   {angle:.6f}             │")
    print("  └─────────────────────────────────────────────┘")

    # ── Pixel-space corners of the (un-rotated) crop box ─────────────────────
    half = roi_size / 2.0
    print(f"\n  Unrotated crop box corners (normalized):")
    print(f"    top-left:     ({roi_cx_shifted - half:.4f}, {roi_cy_shifted - half:.4f})")
    print(f"    top-right:    ({roi_cx_shifted + half:.4f}, {roi_cy_shifted - half:.4f})")
    print(f"    bottom-right: ({roi_cx_shifted + half:.4f}, {roi_cy_shifted + half:.4f})")
    print(f"    bottom-left:  ({roi_cx_shifted - half:.4f}, {roi_cy_shifted + half:.4f})")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
