#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""Check TFLite model output ordering and do element-by-element comparison with our GPU."""

import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def main():
    interpreter = tf.lite.Interpreter(model_path=str(SCRIPT_DIR / 'docs/weights/palm_detection.tflite'))
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    print("=== Small constant tensors (reshape shapes, etc.) ===")
    for d in details:
        idx = d['index']
        name = d['name']
        shape = tuple(d['shape'])
        try:
            data = interpreter.get_tensor(idx)
            if data.size <= 10 and data.size > 0 and not np.all(data == 0):
                print(f"  {idx:4d} {str(shape):15s} val={data.flatten().tolist()} {name[:80]}")
        except:
            pass

    # Load and preprocess image (same as our GPU pipeline)
    img = np.array(Image.open(SCRIPT_DIR / 'docs/test-hands/hand_05.jpg').convert('RGB'))
    h, w = img.shape[:2]

    # For 1500x1500 square image, no letterbox needed (scale=1, no padding)
    pil_img = Image.fromarray(img).resize((192, 192), Image.BILINEAR)
    input_tensor = np.array(pil_img, dtype=np.float32) / 255.0
    input_data = input_tensor.reshape(1, 192, 192, 3)

    print(f"\nInput shape: {input_data.shape}, range: [{input_data.min():.4f}, {input_data.max():.4f}]")
    print(f"Input NHWC [0,0,:] (first pixel RGB): {input_data[0, 0, 0, :].tolist()}")

    # What our GPU sees (CHW format): input_chw[c, y, x] = input_nhwc[y, x, c]
    input_chw = input_data[0].transpose(2, 0, 1)  # [3, 192, 192]
    print(f"Input CHW [c, 0, 0] (first pixel in CHW): {input_chw[:, 0, 0].tolist()}")
    print(f"Input CHW flat[0:10]: {input_chw.flatten()[:10].tolist()}")

    # Run TFLite inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get outputs
    for d in output_details:
        tensor = interpreter.get_tensor(d['index'])
        print(f"\nOutput '{d['name']}': shape={tensor.shape}")

    regressors = None
    scores = None
    for d in output_details:
        tensor = interpreter.get_tensor(d['index'])
        if tensor.shape[-1] == 18:
            regressors = tensor.reshape(2016, 18)
        elif tensor.shape[-1] == 1:
            scores = tensor.reshape(2016)

    print(f"\n=== TFLite output ordering analysis ===")

    # The TFLite output is [1, 2016, 1]. Let's figure out if 12x12 or 24x24 comes first.
    # Strategy: check if scores[0:864] look like 12x12 head or 24x24 head
    # by looking at the score distribution.

    # 12x12 head has 6 anchors per cell, 24x24 has 2 anchors per cell
    # They should have very different score distributions

    block1 = scores[:864]
    block2 = scores[864:]
    print(f"Block 1 (indices 0-863): max_score=sigmoid({block1.max():.4f})={1/(1+np.exp(-block1.max())):.4f}")
    print(f"Block 2 (indices 864-2015): max_score=sigmoid({block2.max():.4f})={1/(1+np.exp(-block2.max())):.4f}")

    # Check if block1 has 6-fold repetition pattern (12x12, 6 anchors)
    # If anchors at same position have similar scores, it's NHWC ordering
    block1_reshaped_nhwc = block1.reshape(12, 12, 6)  # if NHWC
    block1_reshaped_nchw = block1.reshape(6, 12, 12)  # if NCHW (unlikely for TFLite)

    # For NHWC, adjacent elements should be anchors at same position (similar scores)
    nhwc_var = np.mean(np.var(block1_reshaped_nhwc, axis=2))  # var across anchors at each position
    nchw_var = np.mean(np.var(block1_reshaped_nchw, axis=0))  # var across anchors at each position (wrong)

    print(f"\nBlock 1 as NHWC [12,12,6] - var across anchors per position: {nhwc_var:.4f}")
    print(f"Block 1 as NCHW [6,12,12] - var across channels per position: {nchw_var:.4f}")
    print(f"{'NHWC' if nhwc_var < nchw_var else 'NCHW'} has lower variance → likely that format")

    # Now let's check which block has the hand detection
    # For hand_05.jpg (hand at center), the best detection should be near center
    sigmoid_scores = 1 / (1 + np.exp(-scores))
    top_idx = np.argmax(sigmoid_scores)
    print(f"\nTop detection: anchor={top_idx}, score={sigmoid_scores[top_idx]:.6f}, logit={scores[top_idx]:.6f}")

    if top_idx < 864:
        # In block 1 - either 12x12 NHWC or something else
        local_idx = top_idx
        # NHWC: y = local / (12*6), x = (local % (12*6)) / 6, a = local % 6
        nhwc_y = local_idx // (12 * 6)
        nhwc_x = (local_idx % (12 * 6)) // 6
        nhwc_a = local_idx % 6
        print(f"  In block 1, local_idx={local_idx}")
        print(f"  As NHWC [12,12,6]: y={nhwc_y}, x={nhwc_x}, a={nhwc_a} → pos=({(nhwc_x+0.5)/12:.3f}, {(nhwc_y+0.5)/12:.3f})")
    else:
        local_idx = top_idx - 864
        nhwc_y = local_idx // (24 * 2)
        nhwc_x = (local_idx % (24 * 2)) // 2
        nhwc_a = local_idx % 2
        print(f"  In block 2, local_idx={local_idx}")
        print(f"  As NHWC [24,24,2]: y={nhwc_y}, x={nhwc_x}, a={nhwc_a} → pos=({(nhwc_x+0.5)/24:.3f}, {(nhwc_y+0.5)/24:.3f})")

    # Decode the top detection
    reg = regressors[top_idx]
    if top_idx < 864:
        anchor_x = (nhwc_x + 0.5) / 12
        anchor_y = (nhwc_y + 0.5) / 12
    else:
        anchor_x = (nhwc_x + 0.5) / 24
        anchor_y = (nhwc_y + 0.5) / 24

    decoded_cx = anchor_x + reg[0] / 192
    decoded_cy = anchor_y + reg[1] / 192
    decoded_w = reg[2] / 192
    decoded_h = reg[3] / 192
    print(f"  Regressors[0:6]: {reg[:6].tolist()}")
    print(f"  Decoded: cx={decoded_cx:.4f}, cy={decoded_cy:.4f}, w={decoded_w:.4f}, h={decoded_h:.4f}")

    # Show top 10 in both blocks
    print(f"\n=== Top 10 detections ===")
    top10 = np.argsort(-sigmoid_scores)[:10]
    for rank, idx in enumerate(top10):
        s = sigmoid_scores[idx]
        if idx < 864:
            block = "12x12"
            local = idx
            y = local // (12*6)
            x = (local % (12*6)) // 6
            a = local % 6
            ax = (x + 0.5) / 12
            ay = (y + 0.5) / 12
        else:
            block = "24x24"
            local = idx - 864
            y = local // (24*2)
            x = (local % (24*2)) // 2
            a = local % 2
            ax = (x + 0.5) / 24
            ay = (y + 0.5) / 24

        reg = regressors[idx]
        cx = ax + reg[0]/192
        cy = ay + reg[1]/192
        w = reg[2]/192
        h = reg[3]/192
        print(f"  {rank+1:2d}. [{block}] anchor={idx} pos=({ax:.3f},{ay:.3f}) a={a} score={s:.6f}")
        print(f"      decoded: cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")


if __name__ == '__main__':
    main()
