/**
 * @svenflow/micro-handpose
 *
 * WebGPU-powered hand tracking in an 8KB bundle. Zero dependencies.
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
export { createLandmarkSmoother } from './filter.js';
export { toKeypoints, LANDMARK_NAMES } from './types.js';
export type { Handpose, HandposeResult, HandposeOptions, Landmark, Keypoints, HandposeInput } from './types.js';
export type { LandmarkSmoother, SmootherOptions } from './filter.js';
export { compileFullModel } from './model_full.js';
export { loadWeightsFromBuffer } from './model.js';
export type { CompiledModel, WeightsMetadata, Tensor } from './model.js';
