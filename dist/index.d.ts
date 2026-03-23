/** A 3D landmark point (x, y in [0,1] image coords, z is relative depth) */
interface Landmark {
    x: number;
    y: number;
    z: number;
}
/** Hand landmark names in order (21 landmarks) */
declare const LANDMARK_NAMES: readonly ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"];
/** Named landmark type — each key is a landmark name */
type Keypoints = {
    [K in (typeof LANDMARK_NAMES)[number]]: Landmark;
};
/** Build a Keypoints object from an array of 21 landmarks */
declare function toKeypoints(landmarks: Landmark[]): Keypoints;
/** Detection result for a single hand */
interface HandposeResult {
    /** Confidence score (0-1) that a hand is present */
    score: number;
    /** Whether this is a left or right hand */
    handedness: 'left' | 'right';
    /** 21 hand landmarks in original image coordinates [0,1] */
    landmarks: Landmark[];
    /** Named landmarks for ergonomic access: result.keypoints.thumb_tip */
    keypoints: Keypoints;
}
/** Options for creating a handpose detector */
interface HandposeOptions {
    /** URL to fetch weights from. Defaults to CDN. All weight files must be in this directory. */
    weightsUrl?: string;
    /** Minimum landmark confidence score (0-1). Default: 0.5 */
    scoreThreshold?: number;
    /** Minimum palm detection score (0-1). Default: 0.5 */
    palmScoreThreshold?: number;
    /** Maximum number of hands to detect. Default: 3 */
    maxHands?: number;
    /** Force f32 weights even when shader-f16 is available. Default: false */
    forceF32?: boolean;
}
/** A handpose detector instance */
interface Handpose {
    /**
     * Detect hands from a camera frame or image.
     *
     * Accepts: HTMLCanvasElement, OffscreenCanvas, ImageBitmap, HTMLImageElement,
     * HTMLVideoElement, or ImageData.
     *
     * Returns array of detected hands (empty if none found).
     */
    detect: (source: HandposeInput) => Promise<HandposeResult[]>;
    /** Detect hands with full debug information (intermediate pipeline values) */
    detectWithDebug: (source: HandposeInput) => Promise<HandposeDebugResult[]>;
    /**
     * Run landmarks from externally-provided palm detections.
     * Skips internal palm detection — useful for hybrid pipelines
     * (e.g., WASM palm detection + WebGPU landmarks).
     *
     * @param source - Image to crop from
     * @param detections - Pre-computed palm detections (in image-normalized [0,1] coords, letterbox already removed)
     */
    detectFromDetections: (source: HandposeInput, detections: Array<{
        score: number;
        box: [number, number, number, number];
        keypoints: [number, number][];
    }>) => Promise<HandposeResult[]>;
    /** Dispose GPU resources */
    dispose: () => void;
    /** Reset temporal smoothing state (call between unrelated images/scenes) */
    reset: () => void;
    /** Internal debug access (not part of public API) */
    _debug?: any;
}
/** Debug result including intermediate pipeline values */
interface HandposeDebugResult extends HandposeResult {
    /** Raw crop-space landmarks [0,1] from the model (before back-projection) */
    cropLandmarks: Landmark[];
    /** Crop ROI in pixel space */
    roi: {
        centerXpx: number;
        centerYpx: number;
        sizePx: number;
        rotation: number;
    };
    /** Palm detection in image-normalized coords (after letterbox removal) */
    palmDetection: {
        score: number;
        box: [number, number, number, number];
        keypoints: [number, number][];
    };
}
/** Accepted input types for detection */
type HandposeInput = HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLImageElement | HTMLVideoElement | ImageData;

/**
 * Create a hand detector.
 *
 * Downloads model weights and compiles the WebGPU pipeline.
 * Call this once, then use `detect()` repeatedly.
 *
 * @example
 * ```typescript
 * const handpose = await createHandpose()
 * const hands = await handpose.detect(videoElement)
 * for (const hand of hands) {
 *   console.log(hand.keypoints.index_tip) // {x, y, z}
 * }
 * ```
 */
declare function createHandpose(options?: HandposeOptions): Promise<Handpose>;

/**
 * One Euro Filter — adaptive low-pass filter for noisy signals.
 *
 * Used by MediaPipe for landmark smoothing. The key insight: slow movements
 * get heavy smoothing (removes jitter), fast movements get light smoothing
 * (preserves responsiveness).
 *
 * Reference: https://gery.casiez.net/1euro/
 */
/**
 * Landmark smoother using One Euro Filter.
 *
 * Applies independent filtering to each (x, y, z) of 21 landmarks.
 * Matches MediaPipe's VelocityFilter behavior.
 */
interface LandmarkSmoother {
    /** Apply smoothing to landmarks. Returns new smoothed landmark array. */
    apply(landmarks: Array<{
        x: number;
        y: number;
        z: number;
    }>, timestamp?: number): Array<{
        x: number;
        y: number;
        z: number;
    }>;
    /** Reset filter state (e.g., when hand is lost and re-detected) */
    reset(): void;
}
interface SmootherOptions {
    /** Minimum cutoff frequency (Hz). Lower = more smoothing. Default: 1.0 */
    minCutoff?: number;
    /** Speed coefficient. Higher = less lag during fast movement. Default: 0.0 */
    beta?: number;
    /** Derivative cutoff frequency (Hz). Default: 1.0 */
    dCutoff?: number;
}
declare function createLandmarkSmoother(options?: SmootherOptions): LandmarkSmoother;

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
interface HandLandmarksOutput {
    handflag: Float32Array;
    handedness: Float32Array;
    landmarks: Float32Array;
}
interface WeightsMetadata {
    keys: string[];
    shapes: number[][];
    offsets: number[];
    dtype?: 'float32' | 'float16';
}
interface CompiledModel {
    device: GPUDevice;
    run: (input: Float32Array) => Promise<HandLandmarksOutput>;
    runFromCanvas: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandLandmarksOutput>;
    runFromGPUBuffer: (inputBuffer: GPUBuffer) => Promise<HandLandmarksOutput>;
    runFromCanvasPipelined: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandLandmarksOutput | null>;
    flushPipelined: () => Promise<HandLandmarksOutput | null>;
    benchmark: (iterations?: number) => Promise<{
        avgMs: number;
        fps: number;
    }>;
    benchmarkGPU: (iterations?: number) => Promise<{
        avgMs: number;
        fps: number;
        medianMs: number;
        minMs: number;
    }>;
    runFromCanvasViaRender: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandLandmarksOutput>;
    benchmarkDiagnostic: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap, iterations?: number) => Promise<{
        gpuOnly: {
            median: number;
            min: number;
        };
        mapAsyncOnly: {
            median: number;
            min: number;
        };
        mapAsyncNoWait: {
            median: number;
            min: number;
        };
        total: {
            median: number;
            min: number;
        };
        pipelined: {
            median: number;
            min: number;
        };
        renderReadback: {
            median: number;
            min: number;
        } | null;
    }>;
    debugLayerOutputs: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<any>;
}
/**
 * Load weights from JSON metadata + binary buffer
 */
declare function loadWeightsFromBuffer(metadata: WeightsMetadata, buffer: ArrayBuffer): Map<string, Tensor>;

/**
 * FULL Hand Landmark Model — EfficientNet-B0-like architecture
 *
 * Input: 224x224x3
 * Architecture: Initial Conv + 16 MBConv blocks + Global Average Pool + FC heads
 * Output: landmarks(63), world_landmarks(63), handflag(1), handedness(1)
 */

declare function compileFullModel(weights: Map<string, Tensor>, options?: {
    forceF32?: boolean;
}): Promise<CompiledModel>;

export { type CompiledModel, type Handpose, type HandposeDebugResult, type HandposeInput, type HandposeOptions, type HandposeResult, type Keypoints, LANDMARK_NAMES, type Landmark, type LandmarkSmoother, type SmootherOptions, type Tensor, type WeightsMetadata, compileFullModel, createHandpose, createLandmarkSmoother, loadWeightsFromBuffer, toKeypoints };
