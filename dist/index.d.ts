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

export { type Handpose, type HandposeOptions, type HandposeResult, type Landmark, createHandpose };
