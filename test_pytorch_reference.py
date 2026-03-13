#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch"]
# ///
"""
Run our exact weights through the original PyTorch model to get ground truth.
Compare with our numpy reference to find where error accumulates.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# ============ Load the PyTorch model definition ============
class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResModule, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        kernel_size = 5
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        return self.act(self.convs(h) + x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, number=2):
        super(ResBlock, self).__init__()
        layers = [ResModule(in_channels, in_channels) for _ in range(number)]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class HandLandmarks(nn.Module):
    def __init__(self):
        super(HandLandmarks, self).__init__()
        self.backbone1 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), value=0.0),
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            ResBlock(24),
            ResModule(24, 48, stride=2)
        )
        self.backbone2 = nn.Sequential(ResBlock(48), ResModule(48, 96, stride=2))
        self.backbone3 = nn.Sequential(ResBlock(96), ResModule(96, 96, stride=2))
        self.backbone4 = nn.Sequential(
            ResBlock(96),
            ResModule(96, 96, stride=2),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.backbone5 = nn.Sequential(
            ResModule(96, 96),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.backbone6 = nn.Sequential(
            ResModule(96, 96),
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        ff_layers = []
        ResBlockChannels = [48, 96, 288, 288, 288]
        ResModuleChannels = [96, 288, 288, 288, 288]
        for rbc, rmc in zip(ResBlockChannels, ResModuleChannels):
            ff_layers.append(ResBlock(rbc, number=4))
            ff_layers.append(ResModule(rbc, rmc, stride=2))
        ff_layers.append(ResBlock(288, number=4))
        self.ff = nn.Sequential(*ff_layers)
        self.handflag = nn.Conv2d(in_channels=288, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
        self.handedness = nn.Conv2d(in_channels=288, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)
        self.reg_3d = nn.Conv2d(in_channels=288, out_channels=63, kernel_size=2, stride=1, padding=0, bias=True)

    def forward(self, x):
        b1 = self.backbone1(x)
        b2 = self.backbone2(b1)
        b3 = self.backbone3(b2)
        b4 = self.backbone4(b3) + b3
        b5 = self.backbone5(b4) + b2
        b6 = self.backbone6(b5) + b1
        ff = self.ff(b6)
        hand = self.handflag(ff)
        hand = hand.squeeze().sigmoid().reshape(-1, 1)
        handedness = self.handedness(ff)
        handedness = handedness.squeeze().sigmoid().reshape(-1, 1)
        reg_3d = self.reg_3d(ff)
        reg_3d = reg_3d.permute(0, 2, 3, 1).reshape(-1, 63) / 256.0
        return hand, handedness, reg_3d


# ============ Load our weights ============
weights_dir = Path(__file__).parent / "weights"
meta = json.loads((weights_dir / "weights_f16.json").read_text())
bin_data = (weights_dir / "weights_f16.bin").read_bytes()

def read_tensor(name):
    idx = meta["keys"].index(name)
    shape = meta["shapes"][idx]
    offset = meta["offsets"][idx]
    n = 1
    for s in shape:
        n *= s
    raw = bin_data[offset : offset + n * 2]
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)

# Also load f32 weights for comparison
meta_f32 = json.loads((weights_dir / "weights.json").read_text())
bin_f32 = (weights_dir / "weights.bin").read_bytes()

def read_tensor_f32(name):
    idx = meta_f32["keys"].index(name)
    shape = meta_f32["shapes"][idx]
    offset = meta_f32["offsets"][idx]
    n = 1
    for s in shape:
        n *= s
    raw = bin_f32[offset : offset + n * 4]
    return np.frombuffer(raw, dtype=np.float32).reshape(shape)

# Create model and load weights
model = HandLandmarks()
state_dict = model.state_dict()

# Map our weight names to PyTorch state dict names
print("Loading weights into PyTorch model...")
for key in meta["keys"]:
    tensor_f16 = read_tensor(key)
    tensor_f32 = read_tensor_f32(key)

    if key in state_dict:
        state_dict[key] = torch.from_numpy(tensor_f32)
    else:
        print(f"  WARNING: {key} not in state dict")

model.load_state_dict(state_dict)
model.eval()

# Also create f16-weights model for comparison
model_f16 = HandLandmarks()
state_dict_f16 = model_f16.state_dict()
for key in meta["keys"]:
    tensor_f16 = read_tensor(key)
    if key in state_dict_f16:
        state_dict_f16[key] = torch.from_numpy(tensor_f16)
model_f16.load_state_dict(state_dict_f16)
model_f16.eval()

# Load and preprocess image
img = Image.open(weights_dir.parent / "docs" / "hand_nikhil.jpg").convert("RGB")
img = img.resize((256, 256), Image.BILINEAR)
pixels = np.array(img, dtype=np.float32) / 255.0  # [256, 256, 3] in [0, 1]
x = pixels.transpose(2, 0, 1)  # [3, 256, 256]
input_tensor = torch.from_numpy(x).unsqueeze(0)  # [1, 3, 256, 256]

print(f"\nInput: shape={input_tensor.shape}, min={input_tensor.min():.4f}, max={input_tensor.max():.4f}")

# Run inference
with torch.no_grad():
    hand_f32, handedness_f32, landmarks_f32 = model(input_tensor)
    hand_f16, handedness_f16, landmarks_f16 = model_f16(input_tensor)

NAMES = [
    'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]

mp_landmarks = [
    (0.3962, 0.8616, 0.0000),
    (0.5835, 0.7839, -0.0348),
    (0.7186, 0.6426, -0.0183),
    (0.8162, 0.5565, -0.0060),
    (0.8999, 0.4988, 0.0071),
    (0.5850, 0.4422, 0.0806),
    (0.6091, 0.3305, 0.1045),
    (0.6168, 0.2742, 0.0999),
    (0.6181, 0.2199, 0.0893),
    (0.4791, 0.4260, 0.0796),
    (0.4736, 0.3009, 0.1145),
    (0.4794, 0.2317, 0.1014),
    (0.4766, 0.1766, 0.0844),
    (0.3800, 0.4487, 0.0665),
    (0.3461, 0.3340, 0.0833),
    (0.3402, 0.2674, 0.0666),
    (0.3340, 0.2097, 0.0482),
    (0.2868, 0.5024, 0.0454),
    (0.2221, 0.4186, 0.0454),
    (0.1913, 0.3589, 0.0390),
    (0.1719, 0.3026, 0.0328),
]

lm_f32 = landmarks_f32[0].numpy()
lm_f16 = landmarks_f16[0].numpy()

print(f"\nHandflag: f32={hand_f32.item():.6f}, f16={hand_f16.item():.6f}")
print(f"Handedness: f32={handedness_f32.item():.6f}, f16={handedness_f16.item():.6f}")

print(f"\n{'name':12s} {'pt_f32_x':>9s} {'pt_f32_y':>9s} {'pt_f16_x':>9s} {'pt_f16_y':>9s} {'mp_x':>9s} {'mp_y':>9s} {'f32_err':>8s} {'f16_err':>8s} {'f16_drift':>9s}")

total_f32_err = 0
total_f16_err = 0
total_f16_drift = 0

for i in range(21):
    f32_x, f32_y = lm_f32[i*3], lm_f32[i*3+1]
    f16_x, f16_y = lm_f16[i*3], lm_f16[i*3+1]
    mp_x, mp_y = mp_landmarks[i][:2]

    f32_err = np.sqrt((f32_x - mp_x)**2 + (f32_y - mp_y)**2)
    f16_err = np.sqrt((f16_x - mp_x)**2 + (f16_y - mp_y)**2)
    f16_drift = np.sqrt((f32_x - f16_x)**2 + (f32_y - f16_y)**2)

    total_f32_err += f32_err
    total_f16_err += f16_err
    total_f16_drift += f16_drift

    print(f"  {NAMES[i]:12s} {f32_x:9.4f} {f32_y:9.4f} {f16_x:9.4f} {f16_y:9.4f} {mp_x:9.4f} {mp_y:9.4f} {f32_err*100:7.2f}% {f16_err*100:7.2f}% {f16_drift*100:8.2f}%")

print(f"\n  Avg error vs MediaPipe:  f32={total_f32_err/21*100:.2f}%, f16={total_f16_err/21*100:.2f}%")
print(f"  Avg f16 quantization drift: {total_f16_drift/21*100:.2f}%")

# Check z-axis
print(f"\n=== Z-axis comparison (f32 PyTorch vs MediaPipe) ===")
for i in range(21):
    f32_z = lm_f32[i*3+2]
    mp_z = mp_landmarks[i][2]
    print(f"  {NAMES[i]:12s}: pytorch_z={f32_z:7.4f}  mp_z={mp_z:7.4f}  diff={abs(f32_z-mp_z)*100:.2f}%")
