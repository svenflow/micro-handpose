# micro-handpose

[![npm](https://img.shields.io/npm/v/@svenflow/micro-handpose)](https://www.npmjs.com/package/@svenflow/micro-handpose)
[![license](https://img.shields.io/npm/l/@svenflow/micro-handpose)](./LICENSE)

**WebGPU hand tracking for the browser. Multi-hand detection with ROI tracking, 21 landmarks per hand. No WASM, no ONNX Runtime — just 15 compute shaders. 74KB JS + model weights downloaded at runtime.**

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

### iPhone 17 Pro — Safari (iOS 26, WebGPU)

| | Inference | FPS | Backend |
|---|---|---|---|
| **micro-handpose** | **4.1ms** | **60** | WebGPU |
| MediaPipe | 11.4ms | 60 | WebGPU |

### Mac Mini M4 Pro — Chrome 134

| | Median | p99 | Backend |
|---|---|---|---|
| **micro-handpose** | **2.2ms** | **3.1ms** | WebGPU |
| MediaPipe | 4.0ms | 6.5ms | WebGPU |
| MediaPipe | 4.5ms | 8.2ms | WASM |

[**Run this benchmark on your device →**](https://svenflow.github.io/micro-handpose/)

**~3x faster than MediaPipe GPU on iPhone Safari.** On desktop, ~2x faster. With ROI tracking, 99% of frames skip palm detection entirely — only landmark inference runs.

## Features

- **74KB** minified JS (17KB gzipped) + 7.7MB weights (served via CDN)
- **~3x faster** than MediaPipe on iPhone, ~2x on desktop
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
| `palmScoreThreshold` | `number` | `0.5` | Minimum palm detection score (0-1) |
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

## Error Handling

Check for WebGPU support before initializing:

```typescript
if (!navigator.gpu) {
  console.error('WebGPU is not supported in this browser')
  // Fall back to a non-WebGPU solution or show a message
}
```

Wrap initialization in a try/catch to handle GPU adapter or device failures:

```typescript
try {
  const handpose = await createHandpose()
  const hands = await handpose.detect(videoElement)
} catch (err) {
  console.error('Failed to initialize hand tracking:', err)
}
```

## SSR / Server-Side Rendering

`micro-handpose` requires WebGPU and browser APIs (`navigator.gpu`, `OffscreenCanvas`, etc.) that are not available in server environments. If you use a framework with server-side rendering (Next.js, Nuxt, SvelteKit, etc.), make sure to only import and initialize it on the client:

```typescript
// Next.js example (app router)
'use client'

import { useEffect, useState } from 'react'
import type { Handpose } from '@svenflow/micro-handpose'

export default function HandTracker() {
  const [handpose, setHandpose] = useState<Handpose | null>(null)

  useEffect(() => {
    import('@svenflow/micro-handpose').then(({ createHandpose }) => {
      createHandpose().then(setHandpose)
    })
  }, [])

  // ...
}
```

## FAQ

**Does it work on mobile?**
Yes. WebGPU is supported in Chrome on Android and Safari on iOS 18+. On iPhone (iOS 18, Safari), we measured 72ms median (14 FPS) for single-image detection — 1.4x faster than MediaPipe GPU, 1.9x faster than MediaPipe CPU on the same device. With ROI tracking in video mode, sustained performance is ~37ms (27 FPS).

**How many hands can it track?**
Up to 3 by default. Set `maxHands` in the options to change this.

**Does it work offline?**
Model weights are downloaded on first use and cached by the browser. After that, it works offline. You can also self-host the weights (see [Self-Hosting Weights](#self-hosting-weights)).

**What license is the model under?**
The model architecture and weights are derived from MediaPipe's hand landmark model, which is published under the Apache 2.0 license.

## Development

```bash
git clone https://github.com/svenflow/micro-handpose.git
cd micro-handpose
npm install
npm run dev    # Watch mode with hot reload
npm run build  # Production build
```

## Credits

- Hand landmark model architecture and weights adapted from [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe) (Apache 2.0 license).
- ROI tracking approach follows MediaPipe's pipeline design (palm detection + landmark tracking with re-detection on loss).

## License

MIT
