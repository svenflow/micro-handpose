# @svenflow/micro-handpose

Tiny, fast hand landmark detection for the browser. WebGPU-powered, zero dependencies.

[**Live Demo**](https://svenflow.github.io/micro-handpose/)

- **80KB** JS (minified) + 7.7MB weights (served via CDN)
- **~2x faster** than MediaPipe on the same hardware
- **Multi-hand tracking** with ROI-based frame-to-frame tracking (same approach as MediaPipe)
- **21 landmarks** per hand, matching MediaPipe's output format
- TypeScript types included

## Install

```bash
npm install @svenflow/micro-handpose
```

## Usage

```typescript
import { createHandpose } from '@svenflow/micro-handpose'

const handpose = await createHandpose()

// In your render loop:
const hands = await handpose.detect(videoElement)

for (const hand of hands) {
  console.log(hand.score)      // 0.99
  console.log(hand.handedness) // 'left' | 'right'
  console.log(hand.landmarks)  // 21 { x, y, z } points
  console.log(hand.keypoints)  // named access: hand.keypoints.index_tip
}

// Clean up GPU resources when done
handpose.dispose()
```

## API

### `createHandpose(options?): Promise<Handpose>`

Creates and initializes the detector. Downloads weights and compiles the WebGPU pipeline. Call once, then reuse.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `weightsUrl` | `string` | jsdelivr CDN | Base URL for weight files. Set this to self-host weights. |
| `scoreThreshold` | `number` | `0.5` | Minimum hand confidence to return a detection (0-1). |
| `palmScoreThreshold` | `number` | `0.5` | Minimum palm detection confidence (0-1). |
| `maxHands` | `number` | `3` | Maximum number of hands to detect. |

### `handpose.detect(source): Promise<HandposeResult[]>`

Runs inference on an image source. Returns an array of detected hands (empty if none found).

Uses ROI tracking between frames: after the first palm detection, subsequent calls compute the crop region from previous landmarks instead of re-running palm detection. This is faster and produces smoother tracking (same approach as MediaPipe).

**Accepted input types:** `HTMLCanvasElement`, `OffscreenCanvas`, `ImageBitmap`, `HTMLImageElement`, `HTMLVideoElement`, `ImageData`

#### `HandposeResult`

```typescript
{
  score: number           // Confidence (0-1)
  handedness: 'left' | 'right'
  landmarks: Landmark[]   // 21 points (normalized 0-1 coords)
  keypoints: Keypoints    // Named access to landmarks
}
```

Each `Landmark` has `x`, `y` (normalized image coordinates, 0-1) and `z` (relative depth).

The 21 landmarks follow MediaPipe ordering: `wrist`, `thumb_cmc`, `thumb_mcp`, `thumb_ip`, `thumb_tip`, `index_mcp` ... `pinky_tip`.

### `handpose.reset(): void`

Resets tracking state. Call this when switching between unrelated images to force palm re-detection on the next frame.

### `handpose.dispose(): void`

Releases GPU resources.

## Self-hosting weights

By default, weights are fetched from jsdelivr CDN. To self-host:

```typescript
const handpose = await createHandpose({
  weightsUrl: '/models/handpose'
})
```

Copy the `weights/` directory from the npm package to your server.

## Browser requirements

Requires [WebGPU](https://webgpureport.org). Supported in Chrome 113+, Edge 113+, Safari 18+, and Firefox Nightly.

## Performance

Benchmarked on Apple M4:

| | Median | p99 | Backend |
|---|---|---|---|
| **micro-handpose** | 2.2ms | 3.1ms | WebGPU |
| MediaPipe | 4.0ms | 6.5ms | WebGPU |
| MediaPipe | 4.5ms | 8.2ms | WASM |

With ROI tracking enabled, most frames skip palm detection entirely, making the effective per-frame cost even lower after initial detection.

## License

MIT
