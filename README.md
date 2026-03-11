# @svenflow/micro-handpose

Tiny, fast hand landmark detection for the browser. WebGPU-powered, zero dependencies.

- **57KB** JS (9KB gzipped) + 7.7MB weights (served via CDN)
- **2.2ms** median inference (455 FPS) — ~2x faster than MediaPipe
- **21 landmarks** per hand, 100% identical output to the PyTorch reference
- TypeScript types included

## Install

```bash
npm install @svenflow/micro-handpose
```

## Usage

```typescript
import { createHandpose } from '@svenflow/micro-handpose'

const handpose = await createHandpose()
const result = await handpose.detect(canvas)

if (result) {
  console.log(result.score)      // 0.99
  console.log(result.handedness) // 'left' | 'right'
  console.log(result.landmarks)  // 21 { x, y, z } points
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
| `weightsUrl` | `string` | jsdelivr CDN | Base URL for `weights.json` and `weights.bin`. Set this to self-host weights. |
| `scoreThreshold` | `number` | `0.5` | Minimum confidence to return a detection (0-1). |

### `handpose.detect(source): Promise<HandposeResult | null>`

Runs inference on an image source. Returns `null` if no hand is detected.

**Accepted input types:** `HTMLCanvasElement`, `OffscreenCanvas`, `ImageBitmap`, `HTMLImageElement`, `HTMLVideoElement`, `ImageData`

#### `HandposeResult`

```typescript
{
  score: number           // Confidence (0-1)
  handedness: 'left' | 'right'
  landmarks: Landmark[]   // 21 points
}
```

Each `Landmark` has `x`, `y` (normalized image coordinates, 0-1) and `z` (relative depth).

The 21 landmarks follow MediaPipe ordering: `wrist`, `thumb_cmc`, `thumb_mcp`, `thumb_ip`, `thumb_tip`, `index_mcp` ... `pinky_tip`.

### `handpose.dispose(): void`

Releases GPU resources.

## Self-hosting weights

By default, weights are fetched from jsdelivr CDN. To self-host:

```typescript
const handpose = await createHandpose({
  weightsUrl: '/models/handpose'
})
```

The detector expects `weights.json` and `weights.bin` at that path.

## Browser requirements

Requires [WebGPU](https://webgpureport.org). Supported in Chrome 113+, Edge 113+, and Firefox Nightly.

## Performance

Benchmarked on Apple M4:

| | Median | p99 | Backend |
|---|---|---|---|
| **micro-handpose** | 2.2ms | 3.1ms | WebGPU |
| MediaPipe | 4.0ms | 6.5ms | WebGPU |
| MediaPipe | 4.5ms | 8.2ms | WASM |

## License

MIT
