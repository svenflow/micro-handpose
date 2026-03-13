/**
 * Palm detection post-processing: anchor generation, decode, NMS, and crop ROI.
 *
 * All operations run on CPU (JavaScript) — fast enough for 2016 anchors.
 */

import type { CompiledPalmModel, PalmDetectionOutput } from './palm_model.js';

export interface PalmDetection {
  /** Confidence score (sigmoid of raw logit) */
  score: number;
  /** Bounding box in normalized [0,1] coords: [center_x, center_y, width, height] */
  box: [number, number, number, number];
  /** 7 keypoints in normalized [0,1] coords: [[x,y], ...] */
  keypoints: [number, number][];
}

export interface HandROI {
  /** Center of crop region in original image coords [0,1] */
  centerX: number;
  centerY: number;
  /** Size of crop region in original image coords [0,1] */
  width: number;
  height: number;
  /** Rotation angle in radians (from wrist to middle finger MCP, aligned to 90 degrees) */
  rotation: number;
}

// ============ Anchor Generation ============

interface Anchor {
  x: number;  // center x in [0,1]
  y: number;  // center y in [0,1]
}

/**
 * Generate SSD anchors for palm detection model.
 *
 * Layer 0: 12x12 grid, 6 anchors per cell (stride 16, relative to 192)
 * Layer 1: 24x24 grid, 2 anchors per cell (stride 8, relative to 192)
 *
 * Anchor positions are at grid cell centers. No anchor size needed since
 * the model regresses offsets from centers directly.
 */
function generateAnchors(): Anchor[] {
  const anchors: Anchor[] = [];

  // Layer 0: 12x12, 6 anchors per cell
  for (let y = 0; y < 12; y++) {
    for (let x = 0; x < 12; x++) {
      const cx = (x + 0.5) / 12;
      const cy = (y + 0.5) / 12;
      for (let a = 0; a < 6; a++) {
        anchors.push({ x: cx, y: cy });
      }
    }
  }

  // Layer 1: 24x24, 2 anchors per cell
  for (let y = 0; y < 24; y++) {
    for (let x = 0; x < 24; x++) {
      const cx = (x + 0.5) / 24;
      const cy = (y + 0.5) / 24;
      for (let a = 0; a < 2; a++) {
        anchors.push({ x: cx, y: cy });
      }
    }
  }

  return anchors;
}

// Pre-generate anchors (same for every inference)
const ANCHORS = generateAnchors();

// ============ Decode + NMS ============

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Decode raw model output into palm detections.
 *
 * Each anchor has:
 * - 1 score (logit → sigmoid)
 * - 18 regression values:
 *   [0-3]: center_x_offset, center_y_offset, width, height (in pixels relative to 192)
 *   [4-17]: 7 keypoints × 2 (x_offset, y_offset) (in pixels relative to 192)
 *
 * Offsets are relative to anchor centers, scaled by 192 (input size).
 */
function decodeDetections(
  output: PalmDetectionOutput,
  scoreThreshold: number,
): PalmDetection[] {
  const detections: PalmDetection[] = [];
  const { scores, regressors } = output;
  const inputSize = 192;

  for (let i = 0; i < ANCHORS.length; i++) {
    const score = sigmoid(scores[i]);
    if (score < scoreThreshold) continue;

    const anchor = ANCHORS[i];
    const regBase = i * 18;

    // Decode box: offsets are in pixel space relative to anchor center
    const cx = anchor.x + regressors[regBase + 0] / inputSize;
    const cy = anchor.y + regressors[regBase + 1] / inputSize;
    const w = regressors[regBase + 2] / inputSize;
    const h = regressors[regBase + 3] / inputSize;

    // Decode 7 keypoints
    const keypoints: [number, number][] = [];
    for (let k = 0; k < 7; k++) {
      const kx = anchor.x + regressors[regBase + 4 + k * 2] / inputSize;
      const ky = anchor.y + regressors[regBase + 4 + k * 2 + 1] / inputSize;
      keypoints.push([kx, ky]);
    }

    detections.push({
      score,
      box: [cx, cy, w, h],
      keypoints,
    });
  }

  return detections;
}

/**
 * Non-maximum suppression using IoU (intersection over union).
 */
function nms(detections: PalmDetection[], iouThreshold: number): PalmDetection[] {
  if (detections.length === 0) return [];

  // Sort by score descending
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: PalmDetection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(sorted[i]);

    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      if (computeIoU(sorted[i], sorted[j]) > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return kept;
}

function computeIoU(a: PalmDetection, b: PalmDetection): number {
  // Convert center format to corner format
  const ax1 = a.box[0] - a.box[2] / 2;
  const ay1 = a.box[1] - a.box[3] / 2;
  const ax2 = a.box[0] + a.box[2] / 2;
  const ay2 = a.box[1] + a.box[3] / 2;

  const bx1 = b.box[0] - b.box[2] / 2;
  const by1 = b.box[1] - b.box[3] / 2;
  const bx2 = b.box[0] + b.box[2] / 2;
  const by2 = b.box[1] + b.box[3] / 2;

  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const intersection = iw * ih;

  const aArea = (ax2 - ax1) * (ay2 - ay1);
  const bArea = (bx2 - bx1) * (by2 - by1);
  const union = aArea + bArea - intersection;

  return union > 0 ? intersection / union : 0;
}

// ============ ROI Computation ============

/**
 * Convert a palm detection to a hand crop ROI.
 *
 * Uses keypoint 0 (wrist) and keypoint 2 (middle finger MCP) to determine
 * hand orientation. The crop is rotated so the hand is upright, and scaled
 * by 2.6x to include fingers.
 *
 * Keypoint indices in palm detection:
 * 0: wrist center
 * 1: index finger MCP (approximate)
 * 2: middle finger MCP
 * 3: ring finger MCP (approximate)
 * 4: pinky MCP (approximate)
 * 5: thumb CMC (approximate)
 * 6: thumb tip (approximate)
 */
export function detectionToROI(detection: PalmDetection): HandROI {
  const [cx, cy, w, h] = detection.box;

  // Compute rotation from wrist (kp0) to middle finger MCP (kp2)
  const wrist = detection.keypoints[0];
  const middleMCP = detection.keypoints[2];

  const dx = middleMCP[0] - wrist[0];
  const dy = middleMCP[1] - wrist[1];

  // Compute the angle of the wrist→MCP vector from the positive X axis
  // In image coords: +X = right, +Y = down
  // A hand pointing up has angle ≈ -90° (or -π/2)
  const angle = Math.atan2(dy, dx);

  // Target angle: hand should point up in the crop = -π/2 in image coords
  // Rotation = how much to rotate the image so the hand becomes upright
  const targetAngle = -Math.PI / 2;
  const rotation = targetAngle - angle;

  // MediaPipe's RectTransformationCalculator:
  // 1. long_side = max(w, h) of the UNSCALED palm box
  // 2. Shift center by (shift_x, shift_y) in the ROTATED frame (shift uses unscaled long_side)
  // 3. Then scale: final_size = long_side * scale
  const longSide = Math.max(w, h);
  const scale = 2.6;
  const size = longSide * scale;

  // Shift in rotated frame: shift_x=0, shift_y=-0.5 (toward fingers in crop-up direction)
  // In image space: apply rotation to the shift vector
  // MediaPipe's convention: x_shift = shift_y * longSide * sin(r_mp), y_shift = shift_y * longSide * cos(r_mp)
  // Our rotation = -r_mp, so: x_shift = -shift_y * L * sin(r_ours), y_shift = shift_y * L * cos(r_ours)
  const shiftAmount = -0.5 * longSide;
  const cosR = Math.cos(rotation);
  const sinR = Math.sin(rotation);
  const shiftX = shiftAmount * sinR;
  const shiftY = shiftAmount * cosR;

  return {
    centerX: cx + shiftX,
    centerY: cy + shiftY,
    width: size,
    height: size,
    rotation,
  };
}

// ============ Public API ============

export interface PalmDetector {
  /** Run palm detection and return ROIs for detected hands */
  detect: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandROI[]>;
  /** Run palm detection and return raw detections (before ROI conversion) */
  detectRaw: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<PalmDetection[]>;
  /** Run palm detection with GPU letterbox resize (matches MediaPipe's bilinear exactly) */
  detectRawWithResize: (source: any, srcW: number, srcH: number) => Promise<{ detections: PalmDetection[]; lbPadX: number; lbPadY: number }>;
  /** Get the compiled palm model (for resource sharing) */
  model: CompiledPalmModel;
}

export interface PalmDetectorOptions {
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
export function createPalmDetector(
  model: CompiledPalmModel,
  options: PalmDetectorOptions = {},
): PalmDetector {
  const {
    scoreThreshold = 0.5,
    nmsThreshold = 0.3,
    maxHands = 2,
  } = options;

  async function detect(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<HandROI[]> {
    const output = await model.run(source);
    const detections = decodeDetections(output, scoreThreshold);
    const filtered = nms(detections, nmsThreshold);
    const limited = filtered.slice(0, maxHands);
    return limited.map(detectionToROI);
  }

  async function detectRaw(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<PalmDetection[]> {
    const output = await model.run(source);
    const detections = decodeDetections(output, scoreThreshold);
    return nms(detections, nmsThreshold).slice(0, maxHands);
  }

  async function detectRawWithResize(source: any, srcW: number, srcH: number): Promise<{ detections: PalmDetection[]; lbPadX: number; lbPadY: number }> {
    const { output, lbPadX, lbPadY } = await model.runWithResize(source, srcW, srcH);
    const detections = decodeDetections(output, scoreThreshold);
    return { detections: nms(detections, nmsThreshold).slice(0, maxHands), lbPadX, lbPadY };
  }

  return { detect, detectRaw, detectRawWithResize, model };
}

/**
 * Compute the affine transform matrix for cropping a hand region.
 *
 * Returns a 2x3 matrix [a, b, tx, c, d, ty] that maps from crop space [0,256]
 * to original image space [0,1] (normalized).
 *
 * Usage: originalX = a * cropX + b * cropY + tx
 *        originalY = c * cropX + d * cropY + ty
 */
export function computeCropTransform(roi: HandROI, cropSize: number = 256): {
  forward: [number, number, number, number, number, number];  // crop → original
  inverse: [number, number, number, number, number, number];  // original → crop
} {
  const cos = Math.cos(roi.rotation);
  const sin = Math.sin(roi.rotation);
  const sx = roi.width / cropSize;
  const sy = roi.height / cropSize;

  // Forward: crop [0,cropSize] → original [0,1]
  // The crop applies R(θ) to go from original → crop space.
  // So crop → original requires R(-θ), the inverse rotation.
  //
  // Steps: center → scale → rotate_inverse → translate to ROI center
  // R(-θ) = [cos, sin; -sin, cos]
  //
  // For non-square images (sx ≠ sy), the rotation happens in uniform physical
  // space. The correct decomposition is:
  //   x_out = sx * (cos * dx + sin * dy) + cx
  //   y_out = sy * (-sin * dx + cos * dy) + cy
  // where dx = x - cropSize/2, dy = y - cropSize/2

  const a = sx * cos;
  const b = sx * sin;
  const c = -sy * sin;
  const d = sy * cos;
  const tx = roi.centerX - (a * cropSize / 2 + b * cropSize / 2);
  const ty = roi.centerY - (c * cropSize / 2 + d * cropSize / 2);

  // Inverse: original [0,1] → crop [0,cropSize]
  // Inverse of: [a b tx; c d ty; 0 0 1]
  const det = a * d - b * c;
  const ia = d / det;
  const ib = -b / det;
  const ic = -c / det;
  const id = a / det;
  const itx = -(ia * tx + ib * ty);
  const ity = -(ic * tx + id * ty);

  return {
    forward: [a, b, tx, c, d, ty],
    inverse: [ia, ib, itx, ic, id, ity],
  };
}

/**
 * Project landmarks from crop space back to original image coordinates.
 *
 * The crop is always square in pixel space (using refDim = min(srcW, srcH)).
 * For non-square images, the X and Y normalization factors differ, so we
 * need srcWidth/srcHeight to correctly project back.
 *
 * @param landmarks Array of {x, y, z} in crop space [0, 1] (from 256x256 crop)
 * @param roi The hand ROI used for cropping (width/height in normalized [0,1] coords using refDim)
 * @param srcWidth Original image width in pixels
 * @param srcHeight Original image height in pixels
 * @returns Array of {x, y, z} in original image space [0, 1]
 */
export function projectLandmarksToOriginal(
  landmarks: Array<{ x: number; y: number; z: number }>,
  roi: HandROI,
  srcWidth: number,
  srcHeight: number,
): Array<{ x: number; y: number; z: number }> {
  const cos = Math.cos(roi.rotation);
  const sin = Math.sin(roi.rotation);
  const refDim = Math.min(srcWidth, srcHeight);
  const physicalSize = roi.width * refDim; // crop size in pixels (square)
  const wx = physicalSize / srcWidth;  // X span in normalized image coords
  const wy = physicalSize / srcHeight; // Y span in normalized image coords

  // Crop → original: undo rotation R(-θ) in uniform physical space,
  // then normalize to image coordinates.
  //   x_out = wx * (cos*(x-0.5) + sin*(y-0.5)) + roi.centerX
  //   y_out = wy * (-sin*(x-0.5) + cos*(y-0.5)) + roi.centerY
  return landmarks.map(lm => {
    const dx = lm.x - 0.5;
    const dy = lm.y - 0.5;
    return {
      x: wx * (cos * dx + sin * dy) + roi.centerX,
      y: wy * (-sin * dx + cos * dy) + roi.centerY,
      z: lm.z,
    };
  });
}
