#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow"]
# ///
"""
Feed the GPU's exact preprocessed input tensor into TFLite CPU and compare outputs.

This isolates model computation differences from preprocessing differences.
If GPU and CPU produce the same SSD outputs given the same input, the model
implementation is correct and any detection differences come from preprocessing.

Expects:
- docs/gpu_input_192x192_chw.bin: float32 [3, 192, 192] from export-input.html
- docs/gpu_ssd_raw.json: {scores: [...], regressors: [...]} from export-ssd.html
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def main():
    # Load GPU input tensor (CHW float32)
    input_path = SCRIPT_DIR / 'docs' / 'gpu_input_192x192_chw.bin'
    if not input_path.exists():
        print(f"Missing {input_path}")
        print("Open http://localhost:9999/export-input.html to download it")
        return

    raw = np.fromfile(input_path, dtype=np.float32)
    print(f"GPU input: {raw.shape[0]} floats ({raw.shape[0] * 4} bytes)")
    assert raw.shape[0] == 3 * 192 * 192, f"Expected {3*192*192}, got {raw.shape[0]}"

    # CHW -> NHWC
    chw = raw.reshape(3, 192, 192)
    nhwc = chw.transpose(1, 2, 0)  # [192, 192, 3]
    input_data = nhwc.reshape(1, 192, 192, 3)

    print(f"Input range: [{nhwc.min():.6f}, {nhwc.max():.6f}]")
    print(f"Input mean: {nhwc.mean():.6f}")
    print(f"First pixel (R,G,B): {nhwc[0, 0, :].tolist()}")

    # Load GPU SSD outputs from JSON
    ssd_path = SCRIPT_DIR / 'docs' / 'gpu_ssd_raw.json'
    gpu_scores = None
    gpu_regressors = None
    if ssd_path.exists():
        with open(ssd_path) as f:
            gpu_data = json.load(f)
        gpu_scores_raw = np.array(gpu_data['scores'], dtype=np.float32)
        gpu_regressors_raw = np.array(gpu_data['regressors'], dtype=np.float32)
        print(f"\nGPU SSD from JSON: {len(gpu_scores_raw)} scores, {len(gpu_regressors_raw)} regressors")
        print(f"GPU score range: [{gpu_scores_raw.min():.6f}, {gpu_scores_raw.max():.6f}]")
    else:
        print(f"\nNo GPU SSD JSON found at {ssd_path}")
        gpu_scores_raw = None
        gpu_regressors_raw = None

    # Run TFLite with GPU's exact input
    print("\n=== Running TFLite with GPU's exact input ===")
    interpreter = tf.lite.Interpreter(
        model_path=str(SCRIPT_DIR / 'docs/weights/palm_detection.tflite')
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get CPU outputs
    cpu_scores = None
    cpu_regressors = None
    for d in output_details:
        tensor = interpreter.get_tensor(d['index'])
        if tensor.shape[-1] == 1:
            cpu_scores = tensor.reshape(-1)  # [2016]
        elif tensor.shape[-1] == 18:
            cpu_regressors = tensor.reshape(2016, 18)

    print(f"CPU scores: {cpu_scores.shape}, range: [{cpu_scores.min():.6f}, {cpu_scores.max():.6f}]")
    cpu_regressors_flat = cpu_regressors.reshape(-1)

    # Show CPU top detections
    cpu_sigmoid = 1 / (1 + np.exp(-cpu_scores))
    top_cpu = np.argsort(-cpu_sigmoid)[:10]
    print(f"\nTop 10 CPU detections (TFLite order: 24x24 first, then 12x12):")
    for rank, idx in enumerate(top_cpu):
        s = cpu_sigmoid[idx]
        logit = cpu_scores[idx]
        if idx < 1152:  # 24x24 block
            local = idx
            y = local // (24 * 2)
            x = (local % (24 * 2)) // 2
            a = local % 2
            grid = "24x24"
            ax, ay = (x + 0.5) / 24, (y + 0.5) / 24
        else:  # 12x12 block
            local = idx - 1152
            y = local // (12 * 6)
            x = (local % (12 * 6)) // 6
            a = local % 6
            grid = "12x12"
            ax, ay = (x + 0.5) / 12, (y + 0.5) / 12
        print(f"  {rank+1:2d}. [{grid}] anchor={idx} (y={y},x={x},a={a}) pos=({ax:.3f},{ay:.3f}) score={s:.6f} logit={logit:.4f}")

    # Compare with GPU outputs
    if gpu_scores_raw is not None:
        print(f"\n{'='*70}")
        print(f"=== GPU vs CPU comparison (SAME INPUT) ===")
        print(f"{'='*70}")

        # GPU debugRun outputs are in NCHW order:
        #   First 864 scores: 12x12 grid, 6 anchors/cell, layout [6, 12, 12] (anchor, y, x)
        #   Next 1152 scores: 24x24 grid, 2 anchors/cell, layout [2, 24, 24]
        #
        # CPU (TFLite) outputs are in NHWC order:
        #   First 1152 scores: 24x24 grid, 2 anchors/cell, layout [24, 24, 2] (y, x, anchor)
        #   Next 864 scores: 12x12 grid, 6 anchors/cell, layout [12, 12, 6]

        # Reorder GPU NCHW -> TFLite NHWC order
        # 12x12 block: GPU [6, 12, 12] -> TFLite [12, 12, 6]
        gpu_12x12_scores = gpu_scores_raw[:864].reshape(6, 12, 12).transpose(1, 2, 0).reshape(-1)
        gpu_24x24_scores = gpu_scores_raw[864:].reshape(2, 24, 24).transpose(1, 2, 0).reshape(-1)

        # TFLite order: 24x24 first, then 12x12
        gpu_reordered_scores = np.concatenate([gpu_24x24_scores, gpu_12x12_scores])

        score_diff = np.abs(gpu_reordered_scores - cpu_scores)
        print(f"\nScore logit comparison:")
        print(f"  Avg diff:    {score_diff.mean():.8f}")
        print(f"  Max diff:    {score_diff.max():.8f}")
        print(f"  Median diff: {np.median(score_diff):.8f}")
        print(f"  Std diff:    {score_diff.std():.8f}")

        # Histogram
        print(f"\nLogit diff distribution:")
        for threshold in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0]:
            count = np.sum(score_diff < threshold)
            pct = count / len(score_diff) * 100
            print(f"  < {threshold:.0e}: {count}/{len(score_diff)} ({pct:.1f}%)")

        # Compare top detections
        gpu_sigmoid = 1 / (1 + np.exp(-gpu_reordered_scores))
        top_gpu = np.argsort(-gpu_sigmoid)[:10]

        print(f"\nTop 10 element-by-element comparison:")
        print(f"  {'Rank':>4} {'Anchor':>6} {'CPU logit':>12} {'GPU logit':>12} {'Diff':>12} {'CPU score':>10} {'GPU score':>10}")
        for rank, idx in enumerate(top_cpu[:10]):
            cl = cpu_scores[idx]
            gl = gpu_reordered_scores[idx]
            cs = cpu_sigmoid[idx]
            gs = gpu_sigmoid[idx]
            print(f"  {rank+1:4d} {idx:6d} {cl:12.6f} {gl:12.6f} {abs(cl-gl):12.8f} {cs:10.6f} {gs:10.6f}")

        # Check if same anchors are selected
        print(f"\n  CPU top 10 anchors: {list(top_cpu)}")
        print(f"  GPU top 10 anchors: {list(top_gpu)}")
        print(f"  Same anchors: {set(top_cpu) == set(top_gpu)}")

        # Regressors comparison
        if gpu_regressors_raw is not None:
            # GPU regressors: [12x12 block (864*18), 24x24 block (1152*18)]
            # Each block is NCHW: [numAnchors * 18, gridH, gridW] or similar
            # Actually regressors from debugRun: flat array of 2016*18 values
            # Let me check if they're interleaved differently

            # GPU regressor layout for 12x12: [6*18, 12, 12] in NCHW = [108, 12, 12]
            # -> flatten is [a0_r0_y0x0, a0_r0_y0x1, ..., a0_r0_y11x11, a0_r1_y0x0, ...]
            # TFLite NHWC for 12x12: [12, 12, 6*18] = [12, 12, 108]
            # -> flatten is [y0x0_a0r0, y0x0_a0r1, ..., y0x0_a5r17, y0x1_a0r0, ...]

            gpu_12x12_reg = gpu_regressors_raw[:864*18].reshape(6*18, 12, 12).transpose(1, 2, 0).reshape(864, 18)
            gpu_24x24_reg = gpu_regressors_raw[864*18:].reshape(2*18, 24, 24).transpose(1, 2, 0).reshape(1152, 18)

            gpu_reordered_reg = np.concatenate([gpu_24x24_reg, gpu_12x12_reg], axis=0).reshape(-1)
            cpu_reg_flat = cpu_regressors.reshape(-1)

            reg_diff = np.abs(gpu_reordered_reg - cpu_reg_flat)
            print(f"\nRegressor comparison:")
            print(f"  Avg diff:    {reg_diff.mean():.8f}")
            print(f"  Max diff:    {reg_diff.max():.8f}")
            print(f"  Median diff: {np.median(reg_diff):.8f}")

            # Show regressors for top detection
            top_idx = top_cpu[0]
            print(f"\n  Top detection (anchor {top_idx}) regressors:")
            print(f"    CPU: {cpu_regressors[top_idx].tolist()}")
            gpu_reordered_reg_2d = np.concatenate([gpu_24x24_reg, gpu_12x12_reg], axis=0)
            print(f"    GPU: {gpu_reordered_reg_2d[top_idx].tolist()}")
            print(f"    Diff: {np.abs(cpu_regressors[top_idx] - gpu_reordered_reg_2d[top_idx]).tolist()}")

        # VERDICT
        print(f"\n{'='*70}")
        if score_diff.max() < 0.01:
            print("VERDICT: GPU and CPU produce IDENTICAL outputs (within float precision)")
            print("Any detection differences come from INPUT PREPROCESSING only.")
        elif score_diff.max() < 0.1:
            print("VERDICT: GPU and CPU produce VERY SIMILAR outputs")
            print("Small differences likely from float precision / op ordering.")
        else:
            print("VERDICT: GPU and CPU produce DIFFERENT outputs")
            print("There may be a model implementation bug in the WGSL shaders.")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
