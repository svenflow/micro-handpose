#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""Check NMS cluster sizes for different test images."""

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
    for y in range(12):
        for x in range(12):
            cx, cy = (x + 0.5) / 12, (y + 0.5) / 12
            for _ in range(6):
                anchors.append((cx, cy))
    for y in range(24):
        for x in range(24):
            cx, cy = (x + 0.5) / 24, (y + 0.5) / 24
            for _ in range(2):
                anchors.append((cx, cy))
    return anchors


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))


def compute_iou(a, b):
    ax1, ay1 = a['cx'] - a['w']/2, a['cy'] - a['h']/2
    ax2, ay2 = a['cx'] + a['w']/2, a['cy'] + a['h']/2
    bx1, by1 = b['cx'] - b['w']/2, b['cy'] - b['h']/2
    bx2, by2 = b['cx'] + b['w']/2, b['cy'] + b['h']/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def analyze_image(img_path, interpreter, anchors, input_details, output_details):
    img = Image.open(img_path)
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    letterboxed, offset_x, offset_y, scale = letterbox_resize(img_array)
    input_tensor = letterboxed[np.newaxis, ...]

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    regressors, scores = None, None
    for od in output_details:
        data = interpreter.get_tensor(od['index'])
        if data.shape[-1] == 18:
            regressors = data[0].reshape(-1)
        elif data.shape[-1] == 1:
            scores = data[0].reshape(-1)

    # Decode all detections above 0.5
    dets = []
    for i in range(len(anchors)):
        score = float(sigmoid(scores[i]))
        if score < 0.5:
            continue
        ax, ay = anchors[i]
        base = i * 18
        cx = ax + regressors[base] / 192
        cy = ay + regressors[base + 1] / 192
        bw = regressors[base + 2] / 192
        bh = regressors[base + 3] / 192
        kps = []
        for k in range(7):
            kx = ax + regressors[base + 4 + k*2] / 192
            ky = ay + regressors[base + 4 + k*2 + 1] / 192
            kps.append((float(kx), float(ky)))
        dets.append({
            'score': score, 'cx': float(cx), 'cy': float(cy),
            'w': float(bw), 'h': float(bh), 'kps': kps, 'idx': i
        })

    dets.sort(key=lambda d: d['score'], reverse=True)

    # NMS with cluster analysis
    suppressed = set()
    clusters = []
    for i in range(len(dets)):
        if i in suppressed:
            continue
        cluster = [i]
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            if compute_iou(dets[i], dets[j]) > 0.3:
                cluster.append(j)
                suppressed.add(j)
        clusters.append(cluster)

    name = img_path.name
    print(f"\n{'='*60}")
    print(f"{name} ({w}x{h})")
    print(f"  Detections above 0.5: {len(dets)}")
    print(f"  NMS clusters: {len(clusters)}")

    for ci, cluster in enumerate(clusters):
        top = dets[cluster[0]]
        print(f"  Cluster {ci}: size={len(cluster)}, top_score={top['score']:.4f}")

        # Show all detections in cluster with their wrist keypoints
        if len(cluster) > 1:
            # Compute weighted average vs top-1 keypoint positions
            total_w = 0
            avg_wrist = [0, 0]
            avg_mcp = [0, 0]
            for idx in cluster:
                d = dets[idx]
                sw = d['score']
                total_w += sw
                avg_wrist[0] += d['kps'][0][0] * sw
                avg_wrist[1] += d['kps'][0][1] * sw
                avg_mcp[0] += d['kps'][2][0] * sw
                avg_mcp[1] += d['kps'][2][1] * sw
            avg_wrist = [avg_wrist[0]/total_w, avg_wrist[1]/total_w]
            avg_mcp = [avg_mcp[0]/total_w, avg_mcp[1]/total_w]

            top_wrist = top['kps'][0]
            top_mcp = top['kps'][2]

            # Compute rotation for weighted avg vs top-1
            def compute_rot(wrist, mcp):
                dx, dy = mcp[0] - wrist[0], mcp[1] - wrist[1]
                return np.degrees(np.arctan2(dy, dx))

            rot_avg = compute_rot(avg_wrist, avg_mcp)
            rot_top = compute_rot(top_wrist, top_mcp)

            print(f"    Top-1 wrist: ({top_wrist[0]:.5f}, {top_wrist[1]:.5f})")
            print(f"    Avg   wrist: ({avg_wrist[0]:.5f}, {avg_wrist[1]:.5f})")
            print(f"    Top-1 MCP:   ({top_mcp[0]:.5f}, {top_mcp[1]:.5f})")
            print(f"    Avg   MCP:   ({avg_mcp[0]:.5f}, {avg_mcp[1]:.5f})")
            print(f"    Top-1 rotation: {rot_top:.3f}°")
            print(f"    Avg   rotation: {rot_avg:.3f}°")
            print(f"    Rotation diff:  {abs(rot_avg - rot_top):.3f}°")

            # Show individual scores in cluster
            scores_str = ", ".join(f"{dets[i]['score']:.4f}" for i in cluster[:10])
            if len(cluster) > 10:
                scores_str += f", ... ({len(cluster)} total)"
            print(f"    Scores: [{scores_str}]")


def main():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    anchors = generate_anchors()

    test_images = ['hand_05.jpg', 'hand_06.jpg', 'hand_07.jpg', 'hand_08.jpg', 'hand_09.jpg', 'hand_10.jpg']
    for name in test_images:
        path = SCRIPT_DIR / "docs" / "test-hands" / name
        if path.exists():
            analyze_image(path, interpreter, anchors, input_details, output_details)


if __name__ == "__main__":
    main()
