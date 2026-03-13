#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Inspect bias values from micro-handpose weights."""

import json
import numpy as np
from pathlib import Path

weights_dir = Path(__file__).parent / "weights"
meta = json.loads((weights_dir / "weights_f16.json").read_text())

keys = meta["keys"]
shapes = meta["shapes"]
offsets = meta["offsets"]

bin_data = (weights_dir / "weights_f16.bin").read_bytes()

def read_tensor(name):
    idx = keys.index(name)
    shape = shapes[idx]
    offset = offsets[idx]
    n_elements = 1
    for s in shape:
        n_elements *= s
    raw = bin_data[offset : offset + n_elements * 2]  # float16 = 2 bytes
    return np.frombuffer(raw, dtype=np.float16).reshape(shape)

# --- reg_3d.bias ---
print("=" * 60)
print("reg_3d.bias (63 values = 21 landmarks x 3 coords)")
print("=" * 60)
bias = read_tensor("reg_3d.bias")
for i in range(21):
    x, y, z = bias[i*3], bias[i*3+1], bias[i*3+2]
    print(f"  Landmark {i:2d}: x={float(x):8.3f}  y={float(y):8.3f}  z={float(z):8.3f}")

x_vals = [float(bias[i]) for i in range(0, 63, 3)]
y_vals = [float(bias[i]) for i in range(1, 63, 3)]
z_vals = [float(bias[i]) for i in range(2, 63, 3)]

print(f"\n  Mean x-bias: {np.mean(x_vals):.4f}")
print(f"  Mean y-bias: {np.mean(y_vals):.4f}")
print(f"  Mean z-bias: {np.mean(z_vals):.4f}")
print(f"  Min/Max x:   {min(x_vals):.4f} / {max(x_vals):.4f}")
print(f"  Min/Max y:   {min(y_vals):.4f} / {max(y_vals):.4f}")
print(f"  Min/Max z:   {min(z_vals):.4f} / {max(z_vals):.4f}")

# --- handflag.bias ---
print("\n" + "=" * 60)
print("handflag.bias")
print("=" * 60)
hf = read_tensor("handflag.bias")
print(f"  Value: {float(hf.item()):.6f}")

# --- handedness.bias ---
print("\n" + "=" * 60)
print("handedness.bias")
print("=" * 60)
hd = read_tensor("handedness.bias")
print(f"  Value: {float(hd.item()):.6f}")

# --- Conclusion ---
print("\n" + "=" * 60)
mean_x = np.mean(x_vals)
if abs(mean_x - 112) < abs(mean_x - 128):
    print(f"CONCLUSION: Mean x-bias ({mean_x:.1f}) is closer to 112 → trained on 224x224")
else:
    print(f"CONCLUSION: Mean x-bias ({mean_x:.1f}) is closer to 128 → trained on 256x256")
print("=" * 60)
