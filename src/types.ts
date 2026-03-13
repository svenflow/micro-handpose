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

/** Detection result for a single hand */
export interface HandposeResult {
  /** Confidence score (0-1) that a hand is present */
  score: number;
  /** Whether this is a left or right hand */
  handedness: 'left' | 'right';
  /** 21 hand landmarks in order (wrist, thumb_cmc, ..., pinky_tip) */
  landmarks: Landmark[];
}

/** Options for creating a handpose detector */
export interface HandposeOptions {
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
export interface Handpose {
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
export interface FullHandposeResult {
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
export interface FullHandpose {
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
export type HandposeInput =
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | HTMLImageElement
  | HTMLVideoElement
  | ImageData;
