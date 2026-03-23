/** A 3D landmark point (x, y in [0,1] image coords, z is relative depth) */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** Hand landmark names in order (21 landmarks) */
export const LANDMARK_NAMES = [
  'wrist',
  'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_mcp', 'index_pip', 'index_dip', 'index_tip',
  'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
] as const;

/** Named landmark type — each key is a landmark name */
export type Keypoints = {
  [K in (typeof LANDMARK_NAMES)[number]]: Landmark;
};

/** Build a Keypoints object from an array of 21 landmarks */
export function toKeypoints(landmarks: Landmark[]): Keypoints {
  const kp = {} as Record<string, Landmark>;
  for (let i = 0; i < LANDMARK_NAMES.length; i++) {
    kp[LANDMARK_NAMES[i]] = landmarks[i]!;
  }
  return kp as Keypoints;
}

/** Detection result for a single hand */
export interface HandposeResult {
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
export interface HandposeOptions {
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
export interface Handpose {
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
  detectFromDetections: (
    source: HandposeInput,
    detections: Array<{ score: number; box: [number, number, number, number]; keypoints: [number, number][] }>,
  ) => Promise<HandposeResult[]>;

  /** Dispose GPU resources */
  dispose: () => void;

  /** Reset temporal smoothing state (call between unrelated images/scenes) */
  reset: () => void;

  /** Internal debug access (not part of public API) */
  _debug?: any;
}

/** Debug result including intermediate pipeline values */
export interface HandposeDebugResult extends HandposeResult {
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
export type HandposeInput =
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | HTMLImageElement
  | HTMLVideoElement
  | ImageData;
