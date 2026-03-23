#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""
Run the TFLite palm detection model on CPU and dump raw SSD outputs.
This gives us a CPU ground truth to compare against our WebGPU outputs.

Usage:
  uv run compare_ssd_cpu.py <image_path>

Outputs a JSON file with:
- raw_scores: [2016] sigmoid'd scores
- raw_logits: [2016] raw classifier logits
- raw_regressors: [2016 * 18] raw regressor outputs
- input_tensor: first 20 values of the preprocessed input (for verification)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "docs" / "weights" / "palm_detection.tflite"


def letterbox_resize(img: np.ndarray, target_size: int = 192) -> tuple[np.ndarray, float, float]:
    """Letterbox resize matching MediaPipe's ImageToTensorCalculator.

    Returns (resized_img, pad_x_frac, pad_y_frac) where pad fractions are in [0, 0.5].
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    scaled_w = round(w * scale)
    scaled_h = round(h * scale)
    offset_x = (target_size - scaled_w) // 2
    offset_y = (target_size - scaled_h) // 2

    # Resize using bilinear interpolation
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img)
    pil_resized = pil_img.resize((scaled_w, scaled_h), PILImage.BILINEAR)
    resized = np.array(pil_resized)

    # Create letterboxed image (zero-padded)
    result = np.zeros((target_size, target_size, 3), dtype=np.float32)
    result[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = resized / 255.0

    pad_x = offset_x / target_size
    pad_y = offset_y / target_size

    return result, pad_x, pad_y


def run_tflite_palm(image_path: str):
    """Run TFLite palm detection on CPU and dump raw outputs."""
    # Load image
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]
    print(f"Image: {image_path} ({w}x{h})")

    # Letterbox resize to 192x192
    input_tensor, pad_x, pad_y = letterbox_resize(img, 192)
    print(f"Letterbox: pad_x={pad_x:.4f}, pad_y={pad_y:.4f}")
    print(f"Input range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
    print(f"Input mean: {input_tensor.mean():.6f}")
    print(f"Input sample[0:10]: {input_tensor.reshape(-1)[:10].tolist()}")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nModel inputs:")
    for d in input_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

    print(f"\nModel outputs:")
    for d in output_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

    # Set input - the model expects NHWC [1, 192, 192, 3] in [0, 1] range
    input_data = input_tensor.reshape(1, 192, 192, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get outputs
    # The palm detection model has 2 outputs:
    # - regressors: [1, 2016, 18] or similar
    # - classifiers: [1, 2016, 1] or similar
    outputs = {}
    for d in output_details:
        tensor = interpreter.get_tensor(d['index'])
        name = d['name']
        outputs[name] = tensor
        print(f"\nOutput '{name}': shape={tensor.shape}, range=[{tensor.min():.6f}, {tensor.max():.6f}]")
        print(f"  Sample[0:10]: {tensor.reshape(-1)[:10].tolist()}")

    # Figure out which output is scores and which is regressors
    # The classifier output has shape [1, N, 1] and the regressor has [1, N, 18]
    scores_tensor = None
    regressors_tensor = None
    for name, tensor in outputs.items():
        flat = tensor.reshape(-1)
        if tensor.shape[-1] == 1 or (len(tensor.shape) == 2 and tensor.shape[1] == 2016):
            scores_tensor = flat
            scores_name = name
        elif tensor.shape[-1] == 18 or (len(tensor.shape) == 2 and tensor.shape[1] == 2016 * 18):
            regressors_tensor = flat
            regressors_name = name

    if scores_tensor is None or regressors_tensor is None:
        # Try alternative: maybe outputs are reshaped differently
        for name, tensor in outputs.items():
            size = tensor.size
            if size == 2016:
                scores_tensor = tensor.reshape(-1)
                scores_name = name
            elif size == 2016 * 18:
                regressors_tensor = tensor.reshape(-1)
                regressors_name = name

    if scores_tensor is None or regressors_tensor is None:
        print("\nCouldn't identify score/regressor outputs. Dumping all outputs:")
        for name, tensor in outputs.items():
            print(f"  {name}: size={tensor.size}, shape={tensor.shape}")
        return

    print(f"\nScores ({scores_name}): {scores_tensor.shape}")
    print(f"Regressors ({regressors_name}): {regressors_tensor.shape}")

    # Apply sigmoid to scores to get probabilities
    scores_sigmoid = 1.0 / (1.0 + np.exp(-scores_tensor))

    # Find top detections
    top_indices = np.argsort(-scores_sigmoid)[:20]
    print(f"\nTop 20 detections:")
    print(f"{'Rank':>4} {'Anchor':>6} {'Score':>10} {'Logit':>10} {'Reg[0:6]':>50}")
    for rank, idx in enumerate(top_indices):
        score = scores_sigmoid[idx]
        logit = scores_tensor[idx]
        reg_start = idx * 18
        regs = regressors_tensor[reg_start:reg_start+6]
        reg_str = ', '.join(f'{v:.2f}' for v in regs)
        print(f"{rank+1:>4} {idx:>6} {score:>10.6f} {logit:>10.6f} {reg_str:>50}")

    # Save results as JSON for comparison
    result = {
        'image': image_path,
        'image_size': [w, h],
        'pad_x': float(pad_x),
        'pad_y': float(pad_y),
        'input_sample': input_tensor.reshape(-1)[:20].tolist(),
        'raw_logits': scores_tensor.tolist(),
        'raw_regressors': regressors_tensor.tolist(),
        'top_20': [
            {
                'anchor': int(idx),
                'score': float(scores_sigmoid[idx]),
                'logit': float(scores_tensor[idx]),
                'regressors': regressors_tensor[int(idx)*18:int(idx)*18+18].tolist(),
            }
            for idx in top_indices
        ],
    }

    out_path = Path(image_path).stem + '_cpu_ssd.json'
    with open(SCRIPT_DIR / 'docs' / out_path, 'w') as f:
        json.dump(result, f)
    print(f"\nSaved CPU reference: docs/{out_path}")

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default: run on all test images
        test_dir = SCRIPT_DIR / 'docs' / 'test-hands'
        images = sorted(test_dir.glob('hand_*.jpg'))
        if not images:
            print("No test images found in docs/test-hands/")
            sys.exit(1)
        for img_path in images:
            print(f"\n{'='*80}")
            run_tflite_palm(str(img_path))
    else:
        run_tflite_palm(sys.argv[1])
