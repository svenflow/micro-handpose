/** A 3D landmark point (x, y in [0,1] image coords, z is relative depth) */
interface Landmark {
    x: number;
    y: number;
    z: number;
}
/** Detection result for a single hand */
interface HandposeResult {
    /** Confidence score (0-1) that a hand is present */
    score: number;
    /** Whether this is a left or right hand */
    handedness: 'left' | 'right';
    /** 21 hand landmarks in order (wrist, thumb_cmc, ..., pinky_tip) */
    landmarks: Landmark[];
}
/** Options for creating a handpose detector */
interface HandposeOptions {
    /** URL to fetch weights from. Defaults to bundled weights or CDN. */
    weightsUrl?: string;
    /** Minimum confidence score to return a detection (0-1). Default: 0.5 */
    scoreThreshold?: number;
    /** Force f32 weights even when shader-f16 is available. Default: false */
    forceF32?: boolean;
    /** URL to fetch palm detection weights from. Required for full-frame detection. */
    palmWeightsUrl?: string;
    /** Minimum palm detection score (0-1). Default: 0.5 */
    palmScoreThreshold?: number;
    /** Maximum number of hands to detect. Default: 2 */
    maxHands?: number;
}
/** A handpose detector instance */
interface Handpose {
    /**
     * Detect hand landmarks from an image source.
     *
     * Accepts: HTMLCanvasElement, OffscreenCanvas, ImageBitmap, HTMLImageElement,
     * HTMLVideoElement, or ImageData.
     *
     * Returns null if no hand is detected (below score threshold).
     */
    detect: (source: HandposeInput) => Promise<HandposeResult | null>;
    /** Pipelined detection: returns previous frame's result, one frame latency, lower overhead */
    detectPipelined: (source: HandposeInput) => Promise<HandposeResult | null>;
    /** Flush pipelined readback to get last frame's result */
    flushPipelined: () => Promise<HandposeResult | null>;
    /** Run diagnostic benchmark measuring GPU time, mapAsync time, pipelining separately */
    benchmarkDiagnostic: (source: HandposeInput, iterations?: number) => Promise<any>;
    /** Debug: read intermediate layer outputs to find where activations die */
    debugLayerOutputs: (source: HandposeInput) => Promise<any>;
    /** Dispose GPU resources */
    dispose: () => void;
}
/** Detection result for a single hand with full-frame coordinates */
interface FullHandposeResult {
    /** Confidence score (0-1) that a hand is present */
    score: number;
    /** Whether this is a left or right hand */
    handedness: 'left' | 'right';
    /** 21 hand landmarks in original image coordinates [0,1] */
    landmarks: Landmark[];
    /** Palm detection score */
    palmScore: number;
}
/** A full-frame handpose detector instance (palm detection + landmarks) */
interface FullHandpose {
    /**
     * Detect hand landmarks from a full camera frame.
     *
     * Runs palm detection to find hands, crops each detected hand,
     * runs landmark detection, and projects landmarks back to original coordinates.
     *
     * Returns array of detected hands (empty if none found).
     */
    detect: (source: HandposeInput) => Promise<FullHandposeResult[]>;
    /** Dispose GPU resources */
    dispose: () => void;
}
/** Accepted input types for detection */
type HandposeInput = HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLImageElement | HTMLVideoElement | ImageData;

/**
 * Create a handpose detector.
 *
 * Downloads model weights and compiles the WebGPU pipeline.
 * Call this once, then use `detect()` repeatedly.
 *
 * @example
 * ```typescript
 * const handpose = await createHandpose()
 * const result = await handpose.detect(canvas)
 * ```
 */
declare function createHandpose(options?: HandposeOptions): Promise<Handpose>;
/**
 * Create a full-frame handpose detector with palm detection + landmarks.
 *
 * Pipeline:
 * 1. Run palm detection on the full camera frame (192x192)
 * 2. For each detected palm, compute crop ROI
 * 3. Crop + rotate + resize to 256x256
 * 4. Run hand landmark model on the cropped image
 * 5. Project landmarks back to original image coordinates
 *
 * @example
 * ```typescript
 * const detector = await createFullHandpose({
 *   palmWeightsUrl: '/palm_detection_weights',
 * })
 * const hands = await detector.detect(videoElement)
 * for (const hand of hands) {
 *   console.log(hand.landmarks) // in original image coords
 * }
 * ```
 */
declare function createFullHandpose(options?: HandposeOptions): Promise<FullHandpose>;

/**
 * Optimized HandLandmarks Model - FULL PIPELINE
 *
 * Two-pass fused pipeline achieving 4.0ms (253 FPS) - faster than MediaPipe!
 *
 * Performance comparison:
 * - This implementation: ~4.3ms (233 FPS)
 * - MediaPipe WebGL: 5.0ms (200 FPS)
 * - torchjs generic WebGPU: 11ms (90 FPS)
 *
 * FULL ARCHITECTURE:
 * 1. Input conv3x3 (3→24, stride=2) + ReLU → 128x128x24
 * 2. backbone1: ResBlock(2) + ResModule(24→48, stride=2) → 64x64x48 (save b1)
 * 3. backbone2: ResBlock(2) + ResModule(48→96, stride=2) → 32x32x96 (save b2)
 * 4. backbone3: ResBlock(2) + ResModule(96→96, stride=2) → 16x16x96 (save b3)
 * 5. backbone4: ResBlock(2) + ResModule(96→96, stride=2) + upsample + add b3
 * 6. backbone5: ResModule(96→96) + upsample + add b2
 * 7. backbone6: ResModule(96→96) + conv1x1(96→48) + upsample + add b1
 * 8. ff layers: 5x (ResBlock(4) + ResModule(stride=2)) + ResBlock(4) → 2x2x288
 * 9. Output heads: handflag, handedness, landmarks
 */
interface Tensor {
    data: Float32Array;
    shape: number[];
    rawF16?: ArrayBufferLike;
}
interface WeightsMetadata {
    keys: string[];
    shapes: number[][];
    offsets: number[];
    dtype?: 'float32' | 'float16';
}
/**
 * Load weights from JSON metadata + binary buffer
 */
declare function loadWeightsFromBuffer(metadata: WeightsMetadata, buffer: ArrayBuffer): Map<string, Tensor>;

/**
 * Palm Detection WebGPU Model
 *
 * BlazeNet backbone with PReLU activations, FPN, and SSD output heads.
 *
 * Architecture:
 * 1. Initial conv 5x5 stride-2 + PReLU → 96x96x32
 * 2. Stage 1: 4 blocks (32ch), stride-2 transition → 48x48x64
 * 3. Stage 2: 4 blocks (64ch), stride-2 transition → 24x24x128 (save backbone24 skip)
 * 4. Stage 3: 4 blocks (128ch), stride-2 transition → 12x12x256 (save backbone12 skip after block 14)
 * 5. Stage 4a: 3 blocks (256ch) at 12x12
 * 6. Stage 4b: stride-2 transition → 6x6x256, then 3 more blocks at 6x6
 * 7. FPN Level 1: upsample 6→12 → conv2d_20 (256→256) → add backbone12 skip → 2 blocks at 12x12
 * 8. FPN Level 2: upsample 12→24 → conv2d_23 (256→128) → add backbone24 skip → 2 blocks at 24x24
 * 9. SSD heads:
 *    - 12x12: 6 classifiers + 108 regressors (6 anchors × 18 values)
 *    - 24x24: 2 classifiers + 36 regressors (2 anchors × 18 values)
 *
 * Output: 2016 anchors total (864 from 12x12 + 1152 from 24x24)
 *   Per anchor: 1 score + 18 regression values
 */

interface PalmDetectionOutput {
    scores: Float32Array;
    regressors: Float32Array;
}
interface CompiledPalmModel {
    device: GPUDevice;
    run: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<PalmDetectionOutput>;
    debugRun: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<any>;
}
declare function compilePalmModel(weights: Map<string, Tensor>, existingDevice?: GPUDevice): Promise<CompiledPalmModel>;

/**
 * Palm detection post-processing: anchor generation, decode, NMS, and crop ROI.
 *
 * All operations run on CPU (JavaScript) — fast enough for 2016 anchors.
 */

interface PalmDetection {
    /** Confidence score (sigmoid of raw logit) */
    score: number;
    /** Bounding box in normalized [0,1] coords: [center_x, center_y, width, height] */
    box: [number, number, number, number];
    /** 7 keypoints in normalized [0,1] coords: [[x,y], ...] */
    keypoints: [number, number][];
}
interface HandROI {
    /** Center of crop region in original image coords [0,1] */
    centerX: number;
    centerY: number;
    /** Size of crop region in original image coords [0,1] */
    width: number;
    height: number;
    /** Rotation angle in radians (from wrist to middle finger MCP, aligned to 90 degrees) */
    rotation: number;
}
interface PalmDetector {
    /** Run palm detection and return ROIs for detected hands */
    detect: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandROI[]>;
    /** Run palm detection and return raw detections (before ROI conversion) */
    detectRaw: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<PalmDetection[]>;
    /** Get the compiled palm model (for resource sharing) */
    model: CompiledPalmModel;
}
interface PalmDetectorOptions {
    /** Minimum confidence score (0-1). Default: 0.5 */
    scoreThreshold?: number;
    /** IoU threshold for NMS. Default: 0.3 */
    nmsThreshold?: number;
    /** Maximum number of hands to detect. Default: 2 */
    maxHands?: number;
}
/**
 * Create a palm detector from a compiled model.
 */
declare function createPalmDetector(model: CompiledPalmModel, options?: PalmDetectorOptions): PalmDetector;
/**
 * Compute the affine transform matrix for cropping a hand region.
 *
 * Returns a 2x3 matrix [a, b, tx, c, d, ty] that maps from crop space [0,256]
 * to original image space [0,1] (normalized).
 *
 * Usage: originalX = a * cropX + b * cropY + tx
 *        originalY = c * cropX + d * cropY + ty
 */
declare function computeCropTransform(roi: HandROI, cropSize?: number): {
    forward: [number, number, number, number, number, number];
    inverse: [number, number, number, number, number, number];
};
/**
 * Project landmarks from crop space back to original image coordinates.
 *
 * @param landmarks Array of {x, y, z} in crop space [0, 1] (from 256x256 crop)
 * @param roi The hand ROI used for cropping
 * @returns Array of {x, y, z} in original image space [0, 1]
 */
declare function projectLandmarksToOriginal(landmarks: Array<{
    x: number;
    y: number;
    z: number;
}>, roi: HandROI): Array<{
    x: number;
    y: number;
    z: number;
}>;

interface CropPipeline {
    /** Execute the crop transform and write output to the given buffer */
    crop: (encoder: GPUCommandEncoder, sourceTexture: GPUTexture, outputBuffer: GPUBuffer, transform: [number, number, number, number, number, number], srcWidth: number, srcHeight: number, dstSize: number) => void;
}
/**
 * Create a reusable crop pipeline on the given device.
 */
declare function createCropPipeline(device: GPUDevice): CropPipeline;

export { type CompiledPalmModel, type CropPipeline, type FullHandpose, type FullHandposeResult, type HandROI, type Handpose, type HandposeOptions, type HandposeResult, type Landmark, type PalmDetection, type PalmDetectionOutput, type PalmDetector, type PalmDetectorOptions, compilePalmModel, computeCropTransform, createCropPipeline, createFullHandpose, createHandpose, createPalmDetector, loadWeightsFromBuffer, projectLandmarksToOriginal };
