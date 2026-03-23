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
    /** Dispose GPU resources */
    dispose: () => void;
    /** Reset temporal smoothing state (call between unrelated images/scenes) */
    reset: () => void;
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

export { type Handpose, type HandposeInput, type HandposeOptions, type HandposeResult, type Keypoints, LANDMARK_NAMES, type Landmark, createHandpose };
