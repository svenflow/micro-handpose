#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "pillow"]
# ///
"""
Compare FULL landmark model: TFLite reference vs our weight extraction.

Loads the FULL TFLite model, runs inference on a test image,
and prints the raw output (landmarks, handflag, handedness).

Also loads OUR extracted weights and runs the same computation in numpy
to find exactly where the divergence is.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TFLITE_PATH = SCRIPT_DIR / "hand_landmark_full.tflite"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
TEST_IMAGE = SCRIPT_DIR / "docs" / "hand_nikhil.jpg"


def run_tflite_model(image_224: np.ndarray) -> dict:
    """Run the TFLite model on a 224x224x3 float32 image in [0,1] range."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model inputs:")
    for d in input_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")
    print("TFLite model outputs:")
    for d in output_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

    # Input is NHWC [1, 224, 224, 3]
    input_tensor = image_224.reshape(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    results = {}
    for d in output_details:
        tensor = interpreter.get_tensor(d['index'])
        results[d['name']] = tensor.copy()
        print(f"  Output '{d['name']}': shape={tensor.shape}, "
              f"min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}")

    return results


def run_numpy_model(image_224: np.ndarray) -> dict:
    """Run our extracted weights through numpy to match the TFLite computation."""
    meta = json.loads((WEIGHTS_DIR / "weights_f16_full.json").read_text())
    bin_data = (WEIGHTS_DIR / "weights_f16_full.bin").read_bytes()

    # Track duplicate keys
    key_counts = {}
    weights = {}

    for i, key in enumerate(meta["keys"]):
        shape = meta["shapes"][i]
        offset = meta["offsets"][i]
        n = 1
        for s in shape:
            n *= s

        raw = bin_data[offset:offset + n * 2]
        tensor = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)

        if key in key_counts:
            key_counts[key] += 1
            actual_key = f"{key}__{key_counts[key]}"
        else:
            key_counts[key] = 1
            actual_key = key

        weights[actual_key] = tensor

    print(f"\nLoaded {len(weights)} weight tensors")

    # NOTE: Our weights are stored in NCHW format (transposed during extraction)
    # TFLite uses NHWC. We need to work in NCHW to match our WebGPU shaders.

    # Input: [224, 224, 3] in HWC -> [3, 224, 224] in CHW
    x = image_224.transpose(2, 0, 1)  # [3, 224, 224]

    def conv2d_nchw(inp, weight, bias, stride=1, pad=0):
        """Conv2D in NCHW format. weight: [O, I, kH, kW]"""
        c_out, c_in, kh, kw = weight.shape
        c, h, w = inp.shape

        # Pad input
        if pad > 0:
            padded = np.zeros((c, h + 2*pad, w + 2*pad), dtype=np.float32)
            padded[:, pad:h+pad, pad:w+pad] = inp
        else:
            padded = inp

        ph, pw = padded.shape[1], padded.shape[2]
        oh = (ph - kh) // stride + 1
        ow = (pw - kw) // stride + 1

        output = np.zeros((c_out, oh, ow), dtype=np.float32)
        for oc in range(c_out):
            for ic in range(c_in):
                for ky in range(kh):
                    for kx in range(kw):
                        output[oc] += padded[ic, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[oc, ic, ky, kx]
            output[oc] += bias[oc]
        return output

    def conv2d_nchw_asympad(inp, weight, bias, stride, pad_before, pad_after):
        """Conv2D with asymmetric padding (TFLite SAME)."""
        c_out, c_in, kh, kw = weight.shape
        c, h, w = inp.shape

        padded = np.zeros((c, h + pad_before + pad_after, w + pad_before + pad_after), dtype=np.float32)
        padded[:, pad_before:pad_before+h, pad_before:pad_before+w] = inp

        ph, pw = padded.shape[1], padded.shape[2]
        oh = (ph - kh) // stride + 1
        ow = (pw - kw) // stride + 1

        output = np.zeros((c_out, oh, ow), dtype=np.float32)
        for oc in range(c_out):
            for ic in range(c_in):
                for ky in range(kh):
                    for kx in range(kw):
                        output[oc] += padded[ic, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[oc, ic, ky, kx]
            output[oc] += bias[oc]
        return output

    def dw_conv_nchw(inp, weight, bias, stride=1, pad=0):
        """Depthwise conv in NCHW. weight: [C, 1, kH, kW]"""
        c, h, w = inp.shape
        kh, kw = weight.shape[2], weight.shape[3]

        if pad > 0:
            padded = np.zeros((c, h + 2*pad, w + 2*pad), dtype=np.float32)
            padded[:, pad:h+pad, pad:w+pad] = inp
        else:
            padded = inp

        ph, pw = padded.shape[1], padded.shape[2]
        oh = (ph - kh) // stride + 1
        ow = (pw - kw) // stride + 1

        output = np.zeros((c, oh, ow), dtype=np.float32)
        for ch in range(c):
            for ky in range(kh):
                for kx in range(kw):
                    output[ch] += padded[ch, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[ch, 0, ky, kx]
            output[ch] += bias[ch]
        return output

    def dw_conv_nchw_asympad(inp, weight, bias, stride, pad_before, pad_after):
        """Depthwise conv with asymmetric padding."""
        c, h, w = inp.shape
        kh, kw = weight.shape[2], weight.shape[3]

        padded = np.zeros((c, h + pad_before + pad_after, w + pad_before + pad_after), dtype=np.float32)
        padded[:, pad_before:pad_before+h, pad_before:pad_before+w] = inp

        ph, pw = padded.shape[1], padded.shape[2]
        oh = (ph - kh) // stride + 1
        ow = (pw - kw) // stride + 1

        output = np.zeros((c, oh, ow), dtype=np.float32)
        for ch in range(c):
            for ky in range(kh):
                for kx in range(kw):
                    output[ch] += padded[ch, ky:ky+oh*stride:stride, kx:kx+ow*stride:stride] * weight[ch, 0, ky, kx]
            output[ch] += bias[ch]
        return output

    def pointwise_nchw(inp, weight, bias):
        """1x1 conv in NCHW. weight: [O, I, 1, 1]"""
        c_out, c_in = weight.shape[0], weight.shape[1]
        h, w = inp.shape[1], inp.shape[2]
        output = np.zeros((c_out, h, w), dtype=np.float32)
        for oc in range(c_out):
            for ic in range(c_in):
                output[oc] += inp[ic] * weight[oc, ic, 0, 0]
            output[oc] += bias[oc]
        return output

    def relu6(x):
        return np.minimum(np.maximum(x, 0), 6.0)

    # Block specs matching the TypeScript
    BLOCK_SPECS = [
        {"inCh": 24, "expandCh": 24, "dwKernel": 3, "stride": 1, "outCh": 16, "hasResidual": False, "hasProject": True},
        {"inCh": 16, "expandCh": 64, "dwKernel": 3, "stride": 2, "outCh": 24, "hasResidual": False, "hasProject": True},
        {"inCh": 24, "expandCh": 144, "dwKernel": 3, "stride": 1, "outCh": 24, "hasResidual": True, "hasProject": True},
        {"inCh": 24, "expandCh": 144, "dwKernel": 5, "stride": 2, "outCh": 40, "hasResidual": False, "hasProject": True},
        {"inCh": 40, "expandCh": 240, "dwKernel": 5, "stride": 1, "outCh": 40, "hasResidual": True, "hasProject": True},
        {"inCh": 40, "expandCh": 240, "dwKernel": 3, "stride": 2, "outCh": 80, "hasResidual": False, "hasProject": True},
        {"inCh": 80, "expandCh": 480, "dwKernel": 3, "stride": 1, "outCh": 80, "hasResidual": True, "hasProject": True},
        {"inCh": 80, "expandCh": 480, "dwKernel": 3, "stride": 1, "outCh": 80, "hasResidual": True, "hasProject": True},
        {"inCh": 80, "expandCh": 480, "dwKernel": 5, "stride": 1, "outCh": 112, "hasResidual": False, "hasProject": True},
        {"inCh": 112, "expandCh": 672, "dwKernel": 5, "stride": 1, "outCh": 112, "hasResidual": True, "hasProject": True},
        {"inCh": 112, "expandCh": 672, "dwKernel": 5, "stride": 1, "outCh": 112, "hasResidual": True, "hasProject": True},
        {"inCh": 112, "expandCh": 672, "dwKernel": 5, "stride": 2, "outCh": 192, "hasResidual": False, "hasProject": True},
        {"inCh": 192, "expandCh": 1152, "dwKernel": 5, "stride": 1, "outCh": 192, "hasResidual": True, "hasProject": True},
        {"inCh": 192, "expandCh": 1152, "dwKernel": 5, "stride": 1, "outCh": 192, "hasResidual": True, "hasProject": True},
        {"inCh": 192, "expandCh": 1152, "dwKernel": 5, "stride": 1, "outCh": 192, "hasResidual": True, "hasProject": True},
        {"inCh": 192, "expandCh": 1152, "dwKernel": 3, "stride": 1, "outCh": 1152, "hasResidual": False, "hasProject": False},
    ]

    # Weight name mapping (same as TypeScript)
    BLOCK_NAMES = [
        {"dw": ("batch_normalization_1/FusedBatchNormV3", "batch_normalization_1"),
         "proj": ("conv2d_1", "batch_normalization_2/FusedBatchNormV3")},
        {"expand": ("conv2d_2", "batch_normalization_3"),
         "dw": ("batch_normalization_4/FusedBatchNormV3", "batch_normalization_4"),
         "proj": ("conv2d_3", "batch_normalization_5/FusedBatchNormV3")},
        {"expand": ("conv2d_4", "batch_normalization_6"),
         "dw": ("batch_normalization_7/FusedBatchNormV3", "batch_normalization_7"),
         "proj": ("conv2d_5", "batch_normalization_8/FusedBatchNormV3")},
        {"expand": ("conv2d_6", "batch_normalization_9"),
         "dw": ("batch_normalization_10/FusedBatchNormV3", "batch_normalization_10"),
         "proj": ("conv2d_7", "batch_normalization_11/FusedBatchNormV3")},
        {"expand": ("conv2d_8", "batch_normalization_12"),
         "dw": ("batch_normalization_13/FusedBatchNormV3", "batch_normalization_13"),
         "proj": ("conv2d_9", "batch_normalization_14/FusedBatchNormV3")},
        {"expand": ("conv2d_10", "batch_normalization_15"),
         "dw": ("batch_normalization_16/FusedBatchNormV3", "batch_normalization_16"),
         "proj": ("conv2d_11", "batch_normalization_17/FusedBatchNormV3")},
        {"expand": ("conv2d_12", "batch_normalization_18"),
         "dw": ("batch_normalization_19/FusedBatchNormV3", "batch_normalization_19"),
         "proj": ("conv2d_13", "batch_normalization_20/FusedBatchNormV3")},
        {"expand": ("conv2d_14", "batch_normalization_21"),
         "dw": ("batch_normalization_22/FusedBatchNormV3", "batch_normalization_22"),
         "proj": ("conv2d_15", "batch_normalization_23/FusedBatchNormV3")},
        {"expand": ("conv2d_16", "batch_normalization_24"),
         "dw": ("batch_normalization_25/FusedBatchNormV3", "batch_normalization_25"),
         "proj": ("conv2d_17", "batch_normalization_26/FusedBatchNormV3")},
        {"expand": ("conv2d_18", "batch_normalization_27"),
         "dw": ("batch_normalization_28/FusedBatchNormV3", "batch_normalization_28"),
         "proj": ("conv2d_19", "batch_normalization_29/FusedBatchNormV3")},
        {"expand": ("conv2d_20", "batch_normalization_30"),
         "dw": ("batch_normalization_31/FusedBatchNormV3", "batch_normalization_31"),
         "proj": ("conv2d_21", "batch_normalization_32/FusedBatchNormV3")},
        {"expand": ("conv2d_22", "batch_normalization_33"),
         "dw": ("batch_normalization_34/FusedBatchNormV3", "batch_normalization_34"),
         "proj": ("conv2d_23", "batch_normalization_35/FusedBatchNormV3")},
        {"expand": ("conv2d_24", "batch_normalization_36"),
         "dw": ("batch_normalization_37/FusedBatchNormV3", "batch_normalization_37"),
         "proj": ("conv2d_25", "batch_normalization_38/FusedBatchNormV3")},
        {"expand": ("conv2d_26", "batch_normalization_39"),
         "dw": ("batch_normalization_40/FusedBatchNormV3", "batch_normalization_40"),
         "proj": ("conv2d_27", "batch_normalization_41/FusedBatchNormV3")},
        {"expand": ("conv2d_28", "batch_normalization_42"),
         "dw": ("batch_normalization_43/FusedBatchNormV3", "batch_normalization_43"),
         "proj": ("conv2d_29", "batch_normalization_44/FusedBatchNormV3")},
        {"expand": ("conv2d_30", "batch_normalization_45"),
         "dw": ("batch_normalization_46/FusedBatchNormV3", "batch_normalization_46")},
    ]

    def tflite_same_pad(kernel, stride):
        """Compute TFLite SAME padding: pad_before = (kernel - stride) // 2"""
        pad_total = kernel - stride  # for even input sizes
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return pad_before, pad_after

    # Initial conv: 3x3 stride 2 + bias + ReLU6
    init_w = weights["conv2d"]        # [24, 3, 3, 3] in OIHW
    init_b = weights["batch_normalization"]  # [24]
    pad_before, pad_after = tflite_same_pad(3, 2)
    x = conv2d_nchw_asympad(x, init_w, init_b, stride=2, pad_before=pad_before, pad_after=pad_after)
    x = relu6(x)
    print(f"After initial conv: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.6f}")

    # Run blocks
    for block_idx, (spec, names) in enumerate(zip(BLOCK_SPECS, BLOCK_NAMES)):
        residual = x.copy() if spec["hasResidual"] else None

        # Expand 1x1
        if "expand" in names:
            ew, eb = weights[names["expand"][0]], weights[names["expand"][1]]
            x = pointwise_nchw(x, ew, eb)
            x = relu6(x)

        # Depthwise
        dw_w = weights[names["dw"][0]]
        dw_b = weights[names["dw"][1]]
        stride = spec["stride"]
        kernel = spec["dwKernel"]
        if stride == 1:
            pad = kernel // 2
            x = dw_conv_nchw(x, dw_w, dw_b, stride=1, pad=pad)
        else:
            pb, pa = tflite_same_pad(kernel, stride)
            x = dw_conv_nchw_asympad(x, dw_w, dw_b, stride=stride, pad_before=pb, pad_after=pa)
        x = relu6(x)

        # Project 1x1
        if spec["hasProject"] and "proj" in names:
            pw, pb_proj = weights[names["proj"][0]], weights[names["proj"][1]]
            x = pointwise_nchw(x, pw, pb_proj)
            # NO activation after project

            # Residual
            if spec["hasResidual"] and residual is not None:
                x = x + residual

        print(f"Block {block_idx:2d}: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.6f}")

    # Global average pooling: [1152, 7, 7] → [1152]
    gap = x.mean(axis=(1, 2))  # average over spatial dims
    print(f"After GAP: shape={gap.shape}, min={gap.min():.4f}, max={gap.max():.4f}")

    # FC heads
    # Landmarks: weight [63, 1152], bias from Identity [1, 63]
    lm_w = weights["conv_landmarks__2"]  # Second occurrence = [63, 1152]
    lm_b = weights["Identity"]           # [1, 63]
    landmarks_raw = lm_w @ gap + lm_b.flatten()

    # Handflag
    hf_w = weights["conv_handflag__2"]   # [1, 1152]
    hf_b = weights["Identity_1"]         # [1, 1]
    handflag_raw = (hf_w @ gap + hf_b.flatten())[0]
    handflag = 1.0 / (1.0 + np.exp(-handflag_raw))

    # Handedness
    hd_w = weights["conv_handedness__2"] # [1, 1152]
    hd_b = weights["Identity_2"]         # [1, 1]
    handedness_raw = (hd_w @ gap + hd_b.flatten())[0]
    handedness = 1.0 / (1.0 + np.exp(-handedness_raw))

    print(f"\nHandflag: raw={handflag_raw:.4f}, sigmoid={handflag:.4f}")
    print(f"Handedness: raw={handedness_raw:.4f}, sigmoid={handedness:.4f}")
    print(f"Landmarks (first 6): {landmarks_raw[:6]}")

    return {
        "landmarks": landmarks_raw,
        "handflag": handflag,
        "handedness": handedness,
        "gap": gap,
    }


def main():
    # Load test image and resize to 224x224
    img = Image.open(TEST_IMAGE).convert("RGB")
    img_224 = img.resize((224, 224), Image.BILINEAR)
    pixels = np.array(img_224, dtype=np.float32) / 255.0  # [224, 224, 3] in [0, 1]

    print("=" * 80)
    print("STEP 1: Run TFLite model (ground truth)")
    print("=" * 80)
    tflite_results = run_tflite_model(pixels)

    print("\n" + "=" * 80)
    print("STEP 2: Run numpy model (our weights)")
    print("=" * 80)
    numpy_results = run_numpy_model(pixels)

    # Compare
    print("\n" + "=" * 80)
    print("STEP 3: Compare TFLite vs numpy")
    print("=" * 80)

    # Find the TFLite landmarks output
    for name, tensor in tflite_results.items():
        if "landmark" in name.lower() and "world" not in name.lower() and tensor.size == 63:
            tflite_landmarks = tensor.flatten()
            print(f"\nTFLite landmarks key: '{name}', shape={tensor.shape}")
            break
    else:
        # Try identity outputs
        print("\nTFLite output keys:", list(tflite_results.keys()))
        for name, tensor in tflite_results.items():
            print(f"  {name}: shape={tensor.shape}")
        return

    numpy_landmarks = numpy_results["landmarks"]

    print(f"\nTFLite landmarks (first 6): {tflite_landmarks[:6]}")
    print(f"Numpy landmarks  (first 6): {numpy_landmarks[:6]}")

    diff = np.abs(tflite_landmarks - numpy_landmarks)
    print(f"\nAbsolute diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
    print(f"Relative diff: mean={diff.mean() / (np.abs(tflite_landmarks).mean() + 1e-10) * 100:.4f}%")

    # Per-landmark comparison
    print(f"\n{'#':>2s} {'Name':>12s} {'TFLite_x':>10s} {'Numpy_x':>10s} {'diff_x':>10s}")
    NAMES = [
        'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
        'index_mcp', 'index_pip', 'index_dip', 'index_tip',
        'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
        'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
        'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
    ]
    for i in range(21):
        tfl_x = tflite_landmarks[i*3]
        np_x = numpy_landmarks[i*3]
        d = abs(tfl_x - np_x)
        print(f"{i:2d} {NAMES[i]:>12s} {tfl_x:10.4f} {np_x:10.4f} {d:10.6f}")


if __name__ == "__main__":
    main()
