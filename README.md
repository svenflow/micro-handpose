# micro-handpose

[![npm](https://img.shields.io/npm/v/@svenflow/micro-handpose)](https://www.npmjs.com/package/@svenflow/micro-handpose)
[![license](https://img.shields.io/npm/l/@svenflow/micro-handpose)](./LICENSE)

**WebGPU hand tracking for the browser. Multi-hand detection with ROI tracking, 21 landmarks per hand. No WASM, no ONNX Runtime — just 15 compute shaders. 80KB JS + model weights downloaded at runtime.**

[**Live Demo**](https://svenflow.github.io/micro-handpose/) | [npm](https://www.npmjs.com/package/@svenflow/micro-handpose)

---

## Quick Start

```bash
npm install @svenflow/micro-handpose
```

```typescript
import { createHandpose } from '@svenflow/micro-handpose'

const handpose = await createHandpose()
const hands = await handpose.detect(videoElement)

for (const hand of hands) {
  console.log(hand.handedness)       // 'left' | 'right'
  console.log(hand.keypoints.wrist)  // { x, y, z }
}
```

Create once, detect per frame. Weights download on first call from CDN and are cached by the browser. Full TypeScript types included.

## Benchmarks

### Mac Mini M4 Pro — Chrome 134

| | Median | p99 | Backend |
|---|---|---|---|
| **micro-handpose** | **2.2ms** | **3.1ms** | WebGPU |
| MediaPipe | 4.0ms | 6.5ms | WebGPU |
| MediaPipe | 4.5ms | 8.2ms | WASM |

**~2x faster than MediaPipe** on the same hardware. With ROI tracking, most frames skip palm detection entirely — only landmark inference runs (~1.5ms).

## Features

- **80KB** minified JS + 7.7MB weights (served via CDN)
- **~2x faster** than MediaPipe on the same hardware
- **Multi-hand tracking** — detects up to 3 hands simultaneously
- **ROI tracking** — uses previous landmarks to track between frames (same approach as MediaPipe), skipping palm detection for smoother, faster results
- **21 landmarks** per hand following MediaPipe ordering
- **Named keypoints** — `hand.keypoints.thumb_tip`, `hand.keypoints.wrist`, etc.
- **Zero dependencies** — pure WebGPU compute shaders, no WASM or ONNX Runtime

## API

### `createHandpose(options?)`

Creates and initializes the detector. Downloads weights and compiles WebGPU pipelines.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `weightsUrl` | `string` | jsdelivr CDN | Base URL for weight files |
| `scoreThreshold` | `number` | `0.5` | Minimum hand confidence (0-1) |
| `maxHands` | `number` | `3` | Maximum hands to detect |

### `handpose.detect(source)`

Detect hands in an image or video frame. Returns `HandposeResult[]` (empty if no hands found).

Accepts: `HTMLVideoElement`, `HTMLCanvasElement`, `OffscreenCanvas`, `ImageBitmap`, `HTMLImageElement`, `ImageData`

```typescript
interface HandposeResult {
  score: number                // Confidence (0-1)
  handedness: 'left' | 'right'
  landmarks: Landmark[]        // 21 points, normalized [0,1]
  keypoints: Keypoints         // Named access: .wrist, .thumb_tip, etc.
}
```

### `handpose.reset()`

Reset tracking state. Call when switching between unrelated images to force palm re-detection.

### `handpose.dispose()`

Release GPU resources.

## How It Works

```
Video frame → Palm Detection (192×192, 15 compute shaders)
           → ROI crop (affine transform on GPU)
           → Landmark model (224×224 EfficientNet-B0, 42 compute shaders)
           → 21 landmarks + hand score
           → ROI tracking (landmarks → next frame's crop region)
```

On the first frame, palm detection finds hand bounding boxes. On subsequent frames, landmarks from the previous frame compute the crop region directly — palm detection is skipped entirely. This matches MediaPipe's tracking approach: smoother results and ~40% less compute per frame.

## Self-Hosting Weights

```typescript
const handpose = await createHandpose({
  weightsUrl: '/models/handpose'
})
```

Copy the `weights/` directory from the npm package to your server.

## Browser Support

| Browser | Status |
|---------|--------|
| Chrome 113+ | ✅ |
| Edge 113+ | ✅ |
| Safari 18+ (macOS/iOS) | ✅ |
| Firefox Nightly | Experimental |

## License

MIT
