#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""
Create a 192x192 test image and run TFLite inference on it.
The 192x192 size eliminates any resize differences between GPU and CPU.
Also exports the exact pixel values for verification against GPU.

Saves:
- docs/test-hands/hand_192.png: 192x192 center crop of hand_05.jpg
- docs/exact_cpu_ssd.json: raw TFLite outputs for comparison
"""

import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def main():
    # Create a 192x192 version of hand_05.jpg via center crop + resize
    img = Image.open(SCRIPT_DIR / 'docs/test-hands/hand_05.jpg').convert('RGB')
    w, h = img.size

    # Just resize directly to 192x192 (no crop — same as what model expects)
    img192 = img.resize((192, 192), Image.BILINEAR)
    img192.save(SCRIPT_DIR / 'docs/test-hands/hand_192.png')
    print(f"Saved 192x192 test image: docs/test-hands/hand_192.png")

    # Get exact pixel values
    pixels = np.array(img192, dtype=np.float32) / 255.0  # [192, 192, 3] in [0,1]
    print(f"Input shape: {pixels.shape}, range: [{pixels.min():.6f}, {pixels.max():.6f}]")
    print(f"Input mean: {pixels.mean():.6f}")

    # Show first few pixels in NHWC format
    print(f"\nNHWC pixel values:")
    print(f"  [0,0,:]: {pixels[0, 0, :].tolist()}")
    print(f"  [0,1,:]: {pixels[0, 1, :].tolist()}")
    print(f"  [0,2,:]: {pixels[0, 2, :].tolist()}")

    # CHW format (what our GPU uses)
    chw = pixels.transpose(2, 0, 1)  # [3, 192, 192]
    print(f"\nCHW values:")
    print(f"  flat[0:10]: {chw.flatten()[:10].tolist()}")
    print(f"  flat[192*192:192*192+5]: {chw.flatten()[192*192:192*192+5].tolist()}")  # start of G channel

    # Run TFLite
    interpreter = tf.lite.Interpreter(model_path=str(SCRIPT_DIR / 'docs/weights/palm_detection.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = pixels.reshape(1, 192, 192, 3)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get outputs
    scores = None
    regressors = None
    for d in output_details:
        tensor = interpreter.get_tensor(d['index'])
        if tensor.shape[-1] == 1:
            scores = tensor.reshape(2016)
        elif tensor.shape[-1] == 18:
            regressors = tensor.reshape(2016, 18)

    # Sigmoid scores
    sigmoid_scores = 1 / (1 + np.exp(-scores))

    # Show top detections
    print(f"\n=== Top 20 TFLite CPU detections ===")
    top20 = np.argsort(-sigmoid_scores)[:20]
    for rank, idx in enumerate(top20):
        s = sigmoid_scores[idx]
        logit = scores[idx]
        if idx < 864:  # 12x12 block
            local = idx
            y = local // 72
            x = (local % 72) // 6
            a = local % 6
            grid = "12x12"
            ax = (x + 0.5) / 12
            ay = (y + 0.5) / 12
        else:  # 24x24 block
            local = idx - 864
            y = local // 48
            x = (local % 48) // 2
            a = local % 2
            grid = "24x24"
            ax = (x + 0.5) / 24
            ay = (y + 0.5) / 24

        reg = regressors[idx]
        cx = ax + reg[0] / 192
        cy = ay + reg[1] / 192

        print(f"  {rank+1:2d}. [{grid}] anchor={idx} (y={y},x={x},a={a}) pos=({ax:.3f},{ay:.3f}) score={s:.6f} logit={logit:.4f}")
        print(f"      decoded: cx={cx:.4f} cy={cy:.4f} reg[0:6]=[{', '.join(f'{v:.2f}' for v in reg[:6])}]")

    # Save full results
    result = {
        'input_nhwc_sample': pixels[0, 0:3, :].tolist(),  # First 3 rows, all columns, all channels
        'input_chw_flat_first20': chw.flatten()[:20].tolist(),
        'scores_logit': scores.tolist(),
        'scores_sigmoid': sigmoid_scores.tolist(),
        'regressors': regressors.tolist(),
        'top_20': [
            {
                'anchor': int(idx),
                'score': float(sigmoid_scores[idx]),
                'logit': float(scores[idx]),
                'regressors': regressors[int(idx)].tolist(),
            }
            for idx in top20
        ],
    }

    out_path = SCRIPT_DIR / 'docs' / 'exact_cpu_ssd.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
