/**
 * @svenflow/micro-handpose
 *
 * Tiny, fast hand landmark detection. WebGPU-powered, 2ms inference.
 *
 * @example
 * ```typescript
 * import { createHandpose } from '@svenflow/micro-handpose'
 *
 * const handpose = await createHandpose()
 * const result = await handpose.detect(canvas)
 *
 * if (result) {
 *   console.log(result.score)      // 0.99
 *   console.log(result.handedness) // 'left' or 'right'
 *   console.log(result.landmarks)  // 21 { x, y, z } points
 * }
 * ```
 */

export { createHandpose } from './handpose.js';
export type { Handpose, HandposeResult, HandposeOptions, Landmark } from './types.js';
