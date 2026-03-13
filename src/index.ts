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

export { createHandpose, createFullHandpose } from './handpose.js';
export { loadWeightsFromBuffer } from './model.js';
export { compilePalmModel } from './palm_model.js';
export { createPalmDetector, computeCropTransform, projectLandmarksToOriginal } from './palm_detection.js';
export { createCropPipeline } from './crop_shader.js';
export { createLandmarkSmoother } from './filter.js';
export type { LandmarkSmoother, SmootherOptions } from './filter.js';
export { toKeypoints, LANDMARK_NAMES } from './types.js';
export type { Handpose, HandposeResult, HandposeOptions, Landmark, FullHandpose, FullHandposeResult, Keypoints } from './types.js';
export type { PalmDetection, HandROI, PalmDetector, PalmDetectorOptions } from './palm_detection.js';
export type { PalmDetectionOutput, CompiledPalmModel } from './palm_model.js';
export type { CropPipeline } from './crop_shader.js';
