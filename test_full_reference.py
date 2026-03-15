#!/usr/bin/env -S uv run --script --python 3.12
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "tensorflow", "Pillow"]
# ///
"""
Test the FULL hand landmark model (hand_landmark_full_from_task.tflite).

1. Runs TFLite inference on a synthetic 224x224 test image
2. Prints raw output values (landmarks, handflag, handedness)
3. Verifies extracted WebGPU weights by computing first layer (3x3 conv + fused BN bias + ReLU6)
   and comparing against TFLite's fused first-layer output
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "hand_landmark_full_from_task.tflite"
WEIGHTS_DIR = SCRIPT_DIR / "weights"


def create_test_image():
    """Create the same synthetic 224x224 test image used by the WebGPU test.
    Fill with RGB(136, 102, 68) = #886644, lighter rect at (50,50)-(174,174) RGB(204, 153, 102).
    Returns NHWC float32 tensor normalized to [0, 1].
    """
    img = np.full((1, 224, 224, 3), [136 / 255.0, 102 / 255.0, 68 / 255.0], dtype=np.float32)
    img[0, 50:174, 50:174, :] = [204 / 255.0, 153 / 255.0, 102 / 255.0]
    return img


def run_tflite_inference(img):
    """Run the FULL landmark model via TFLite and print outputs."""
    import tensorflow as tf

    print("=" * 80)
    print("PART 1: TFLite inference on synthetic test image")
    print("=" * 80)
    print("Model: %s" % MODEL_PATH.name)
    print("Image: 224x224, brown (#886644) fill + lighter rect (#cc9966) at (50,50)-(174,174)")
    print()

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input:  %s shape=%s dtype=%s" % (
        input_details[0]["name"], list(input_details[0]["shape"]), input_details[0]["dtype"]))
    print("Outputs:")
    for o in output_details:
        print("  %s shape=%s dtype=%s" % (o["name"], list(o["shape"]), o["dtype"]))
    print()

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    # Collect outputs
    results = {}
    for o in output_details:
        t = interpreter.get_tensor(o["index"]).flatten()
        results[o["name"]] = t

    # Identify outputs by shape
    # Identity: [1, 63] = landmarks (pixel-space coords)
    # Identity_1: [1, 1] = handflag (presence confidence)
    # Identity_2: [1, 1] = handedness
    # Identity_3: [1, 63] = world landmarks (meters)
    landmarks = results.get("Identity", None)
    handflag = results.get("Identity_1", None)
    handedness = results.get("Identity_2", None)
    world_landmarks = results.get("Identity_3", None)

    print("--- Handflag (Identity_1) ---")
    if handflag is not None:
        print("  Raw value: %.6f" % handflag[0])
        print("  (This is a presence/confidence score, already sigmoid-applied)")
    print()

    print("--- Handedness (Identity_2) ---")
    if handedness is not None:
        print("  Raw value: %.6f" % handedness[0])
        print("  (0 = left hand, 1 = right hand, already sigmoid-applied)")
    print()

    print("--- Landmarks (Identity) - first 9 values ---")
    if landmarks is not None:
        print("  Values: %s" % landmarks[:9])
        print("  These are in PIXEL SPACE (0-224 range), NOT normalized 0-1")
        print("  To normalize: divide by 224")
        print("  First 3 landmarks (x, y, z):")
        for i in range(3):
            x, y, z = landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2]
            print("    landmark %d: x=%.2f y=%.2f z=%.4f  (normalized: %.4f, %.4f)" % (
                i, x, y, z, x / 224.0, y / 224.0))
    print()

    print("--- World Landmarks (Identity_3) - first 9 values ---")
    if world_landmarks is not None:
        print("  Values: %s" % world_landmarks[:9])
        print("  These are in METERS (real-world 3D coordinates)")
    print()

    # Also extract the fused first layer kernel and bias for part 2
    details = interpreter.get_tensor_details()
    conv_kernel = None
    bn_bias = None
    for d in details:
        if d["name"] == "model_1/model/conv2d/Conv2D":
            conv_kernel = interpreter.get_tensor(d["index"]).astype(np.float32)
        elif d["name"] == "model_1/model/batch_normalization/FusedBatchNormV3":
            bn_bias = interpreter.get_tensor(d["index"]).astype(np.float32)

    return conv_kernel, bn_bias, interpreter, details


def compute_first_layer_numpy(img_nhwc, conv_kernel_ohwi, bn_bias):
    """Compute first layer: 3x3 conv (stride 2, padding SAME) + fused BN bias + ReLU6.

    In TFLite, the FusedBatchNormV3 parameters are already folded into the conv kernel
    and bias. So the operation is:
        output = relu6(conv(input, kernel) + bias)

    Args:
        img_nhwc: [1, 224, 224, 3] float32 input
        conv_kernel_ohwi: [24, 3, 3, 3] float32 kernel (OHWI format from TFLite)
        bn_bias: [24] float32 fused bias
    Returns:
        [1, 112, 112, 24] float32 output
    """
    print("=" * 80)
    print("PART 2: Numpy reference first layer (conv2d + fused BN bias + ReLU6)")
    print("=" * 80)
    print()

    # TFLite conv2d with SAME padding and stride 2:
    # Input: 224x224, kernel 3x3, stride 2 -> output 112x112
    # SAME padding for stride 2: pad so that ceil(224/2) = 112
    # Total pad = max(0, (112-1)*2 + 3 - 224) = max(0, 225-224) = 1
    # pad_top = 0, pad_bottom = 1, pad_left = 0, pad_right = 1 (TF convention: less on top/left)

    x = img_nhwc[0]  # [224, 224, 3]
    h, w, c_in = x.shape
    c_out = conv_kernel_ohwi.shape[0]
    kh, kw = conv_kernel_ohwi.shape[1], conv_kernel_ohwi.shape[2]
    stride = 2

    # SAME padding
    out_h = (h + stride - 1) // stride  # ceil(224/2) = 112
    out_w = (w + stride - 1) // stride
    pad_h = max(0, (out_h - 1) * stride + kh - h)  # 1
    pad_w = max(0, (out_w - 1) * stride + kw - w)  # 1
    pad_top = pad_h // 2      # 0
    pad_bottom = pad_h - pad_top  # 1
    pad_left = pad_w // 2     # 0
    pad_right = pad_w - pad_left  # 1

    print("Input shape: %s" % str(x.shape))
    print("Kernel shape: %s (OHWI)" % str(conv_kernel_ohwi.shape))
    print("Bias shape: %s" % str(bn_bias.shape))
    print("Stride: %d" % stride)
    print("SAME padding: top=%d bottom=%d left=%d right=%d" % (pad_top, pad_bottom, pad_left, pad_right))
    print("Output shape: %dx%dx%d" % (out_h, out_w, c_out))
    print()

    # Pad input
    x_padded = np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                       mode='constant', constant_values=0)

    # Convolution in NHWC
    output = np.zeros((out_h, out_w, c_out), dtype=np.float32)
    for oc in range(c_out):
        for ky in range(kh):
            for kx in range(kw):
                for ic in range(c_in):
                    output[:, :, oc] += (
                        x_padded[ky:ky + out_h * stride:stride,
                                 kx:kx + out_w * stride:stride, ic]
                        * conv_kernel_ohwi[oc, ky, kx, ic]
                    )
        output[:, :, oc] += bn_bias[oc]

    print("Conv output (before ReLU6): min=%.6f max=%.6f mean=%.6f" % (
        output.min(), output.max(), output.mean()))

    # ReLU6
    output = np.clip(output, 0, 6)

    print("After ReLU6: min=%.6f max=%.6f mean=%.6f" % (
        output.min(), output.max(), output.mean()))
    print()

    return output[np.newaxis]  # [1, 112, 112, 24]


def compare_with_extracted_weights(img_nhwc):
    """Load extracted WebGPU weights and compute first layer, compare with TFLite."""
    print("=" * 80)
    print("PART 3: Compare extracted WebGPU weights vs TFLite first layer")
    print("=" * 80)
    print()

    json_path = WEIGHTS_DIR / "weights_f16_full.json"
    bin_path = WEIGHTS_DIR / "weights_f16_full.bin"

    if not json_path.exists() or not bin_path.exists():
        print("Extracted weights not found at %s" % WEIGHTS_DIR)
        print("Run extract_full_weights.py first.")
        return None

    meta = json.loads(json_path.read_text())
    bin_data = bin_path.read_bytes()

    def read_tensor(name):
        idx = meta["keys"].index(name)
        shape = meta["shapes"][idx]
        offset = meta["offsets"][idx]
        n = 1
        for s in shape:
            n *= s
        raw = bin_data[offset: offset + n * 2]
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)

    # The extract_full_weights.py transposes kernels to NCHW format:
    # input_conv_kernel: OHWI -> OIHW  (i.e., [24, 3, 3, 3] -> [24, 3, 3, 3])
    # BN bias: [24] stays as-is
    # Find the right keys
    print("Available weight keys (first 10):")
    for i, k in enumerate(meta["keys"][:10]):
        print("  %s  shape=%s" % (k, meta["shapes"][i]))
    print()

    # The first conv kernel was stored with key derived from TFLite name
    # Looking for conv2d kernel and batch_normalization bias
    conv_key = None
    bn_key = None
    for k in meta["keys"]:
        if "conv2d" in k and "depthwise" not in k and "conv2d_" not in k:
            conv_key = k
        if k == "batch_normalization":
            bn_key = k

    if conv_key is None or bn_key is None:
        print("Could not find first layer weights in extracted data.")
        print("Conv key: %s, BN key: %s" % (conv_key, bn_key))
        return None

    print("Using conv kernel: '%s'" % conv_key)
    print("Using BN bias: '%s'" % bn_key)

    ext_kernel = read_tensor(conv_key)
    ext_bias = read_tensor(bn_key)

    print("Extracted kernel shape: %s" % str(ext_kernel.shape))
    print("Extracted bias shape: %s" % str(ext_bias.shape))
    print()

    # The extract_full_weights.py transposes input_conv_kernel from OHWI -> OIHW.
    # For the first conv [24,3,3,3], both OHWI and OIHW have the same shape,
    # so we must explicitly transpose OIHW -> OHWI regardless of shape ambiguity.
    # OIHW = [O, I, H, W] -> OHWI = [O, H, W, I]
    kernel_ohwi = ext_kernel.transpose(0, 2, 3, 1)
    print("Transposed kernel from OIHW to OHWI")

    print("Kernel (OHWI) shape: %s" % str(kernel_ohwi.shape))
    print()

    # Compute first layer with extracted weights
    output_ext = compute_first_layer_numpy(img_nhwc, kernel_ohwi, ext_bias)

    return output_ext


def main():
    # Create test image
    img = create_test_image()
    print("Test image: shape=%s, pixel range=[%.4f, %.4f]" % (
        img.shape, img.min(), img.max()))
    print("  Brown fill: RGB(%.0f, %.0f, %.0f) = (%.4f, %.4f, %.4f)" % (
        136, 102, 68, 136 / 255.0, 102 / 255.0, 68 / 255.0))
    print("  Lighter rect: RGB(%.0f, %.0f, %.0f) = (%.4f, %.4f, %.4f)" % (
        204, 153, 102, 204 / 255.0, 153 / 255.0, 102 / 255.0))
    print()

    # Part 1: TFLite inference
    conv_kernel, bn_bias, interpreter, details = run_tflite_inference(img)

    if conv_kernel is None or bn_bias is None:
        print("ERROR: Could not extract first layer weights from TFLite model")
        return

    # Part 2: Numpy reference first layer using TFLite's own weights
    print()
    tflite_first_layer = compute_first_layer_numpy(img, conv_kernel, bn_bias)

    # Part 3: Compare with extracted WebGPU weights
    print()
    ext_first_layer = compare_with_extracted_weights(img)

    if ext_first_layer is not None and tflite_first_layer is not None:
        print()
        print("=" * 80)
        print("PART 4: Comparison results")
        print("=" * 80)
        print()

        diff = np.abs(tflite_first_layer - ext_first_layer)
        print("TFLite vs extracted weights first layer:")
        print("  Max absolute difference: %.6f" % diff.max())
        print("  Mean absolute difference: %.6f" % diff.mean())
        print("  RMS difference: %.6f" % np.sqrt(np.mean(diff ** 2)))
        print()

        # Show sample values at a few positions
        positions = [(0, 0, 0), (0, 56, 56), (0, 111, 111)]
        for b, y, x in positions:
            tfl_vals = tflite_first_layer[b, y, x, :5]
            ext_vals = ext_first_layer[b, y, x, :5]
            print("  Position (%d,%d,%d) first 5 channels:" % (b, y, x))
            print("    TFLite:    %s" % tfl_vals)
            print("    Extracted: %s" % ext_vals)
            print("    Diff:      %s" % np.abs(tfl_vals - ext_vals))
            print()

        match_threshold = 0.01
        if diff.max() < match_threshold:
            print("PASS: Extracted weights match TFLite first layer (max diff < %.4f)" % match_threshold)
        else:
            print("MISMATCH: Extracted weights differ from TFLite (max diff = %.6f)" % diff.max())
            print("This may be due to float16 quantization in extracted weights.")
            if diff.max() < 0.1:
                print("However, the difference is small and likely due to f16 precision loss.")


if __name__ == "__main__":
    main()
