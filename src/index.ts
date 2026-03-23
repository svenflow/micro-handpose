/**
 * @svenflow/micro-handpose
 *
 * WebGPU hand tracking. Faster than MediaPipe, zero dependencies.
 *
 * @example
 * ```typescript
 * import { createHandpose } from '@svenflow/micro-handpose'
 *
 * const handpose = await createHandpose()
 * const hands = await handpose.detect(videoElement)
 *
 * for (const hand of hands) {
 *   console.log(hand.keypoints.index_tip) // {x, y, z}
 *   console.log(hand.handedness)          // 'left' or 'right'
 * }
 * ```
 */

export { createHandpose } from './handpose.js';
export { LANDMARK_NAMES } from './types.js';
export type {
  Handpose,
  HandposeResult,
  HandposeOptions,
  Landmark,
  Keypoints,
  HandposeInput,
} from './types.js';
