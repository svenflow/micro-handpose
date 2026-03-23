import { loadWeightsFromBuffer } from './model.js';
import type { CompiledModel, WeightsMetadata } from './model.js';
import { compileFullModel, loadFullWeightsFromBuffer } from './model_full.js';
import { compilePalmModel } from './palm_model.js';
import { createPalmDetector } from './palm_detection.js';
import type { PalmDetection } from './palm_detection.js';
import type { Handpose, HandposeInput, HandposeOptions, HandposeResult, Landmark } from './types.js';
import { toKeypoints } from './types.js';
import { createCropPipeline } from './crop_shader.js';
import type { CropPipeline } from './crop_shader.js';

// Default: jsdelivr CDN (auto-mirrors npm packages)
const DEFAULT_WEIGHTS_BASE = 'https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@0.3.0/weights';

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
export async function createHandpose(options: HandposeOptions = {}): Promise<Handpose> {
  const {
    weightsUrl,
    scoreThreshold = 0.5,
    palmScoreThreshold = 0.5,
    maxHands = 3,
  } = options;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('micro-handpose requires WebGPU. Check browser support at https://webgpureport.org');
  }

  // Load all weights in parallel from CDN (or custom URL)
  const base = (weightsUrl ?? DEFAULT_WEIGHTS_BASE).replace(/\/$/, '') + '/';

  const [landmarkMetaRes, landmarkBinRes, palmMetaRes, palmBinRes] = await Promise.all([
    fetch(`${base}weights_f16_full.json`),
    fetch(`${base}weights_f16_full.bin`),
    fetch(`${base}palm_detection_weights.json`),
    fetch(`${base}palm_detection_weights.bin`),
  ]);

  if (!landmarkMetaRes.ok) throw new Error(`Failed to fetch landmark weights: ${landmarkMetaRes.status}`);
  if (!landmarkBinRes.ok) throw new Error(`Failed to fetch landmark weights: ${landmarkBinRes.status}`);
  if (!palmMetaRes.ok) throw new Error(`Failed to fetch palm detection weights: ${palmMetaRes.status}`);
  if (!palmBinRes.ok) throw new Error(`Failed to fetch palm detection weights: ${palmBinRes.status}`);

  const [landmarkMeta, landmarkBuf, palmMeta, palmBuf] = await Promise.all([
    landmarkMetaRes.json() as Promise<WeightsMetadata>,
    landmarkBinRes.arrayBuffer(),
    palmMetaRes.json() as Promise<WeightsMetadata>,
    palmBinRes.arrayBuffer(),
  ]);

  const landmarkWeights = loadFullWeightsFromBuffer(landmarkMeta, landmarkBuf);
  const palmWeights = loadWeightsFromBuffer(palmMeta, palmBuf);

  // FULL landmark model input size
  const LANDMARK_SIZE = 224;

  // Compile FULL landmark model (EfficientNet-B0, 224x224)
  // When f16 is available, use f16 compute to match MediaPipe's mediump
  // WebGL behavior (reduces ~1.5% precision gap to near-zero)
  const landmarkModel = await compileFullModel(landmarkWeights);

  // f16 self-test (FULL model doesn't have f16 toggle — always f32 BN weights)
  {
    const testCanvas = new OffscreenCanvas(LANDMARK_SIZE, LANDMARK_SIZE);
    const testCtx = testCanvas.getContext('2d')!;
    testCtx.fillStyle = '#886644';
    testCtx.fillRect(0, 0, LANDMARK_SIZE, LANDMARK_SIZE);
    testCtx.fillStyle = '#cc9966';
    testCtx.fillRect(50, 50, 124, 124);
    const testOutput = await landmarkModel.runFromCanvas(testCanvas);
    const allZero = testOutput.landmarks.every(v => v === 0) &&
                    testOutput.handflag.every(v => v === 0);
    if (allZero) {
      console.warn('[micro-handpose] FULL model produced all-zero output on self-test');
    }
  }

  // Compile palm detection model
  const palmModel = await compilePalmModel(palmWeights);
  const palmDetector = createPalmDetector(palmModel, {
    scoreThreshold: palmScoreThreshold,
    maxHands,
  });

  // ROI tracking state: previous frame's landmarks per hand (for tracking mode)
  // When tracking, we compute the next-frame crop ROI from previous landmarks
  // instead of re-running palm detection (matches MediaPipe's approach)
  let trackedHands: Array<{ landmarks: Landmark[]; handedness: 'left' | 'right' }> = [];

  /**
   * Compute ROI from previous frame's landmarks (MediaPipe's tracking path).
   * Uses HandLandmarksToRectCalculator + RectTransformationCalculator.
   *
   * 1. Extract 12 partial landmarks (palm + finger MCPs/PIPs)
   * 2. Compute rotation from wrist → averaged finger MCPs
   * 3. Find axis-aligned bounding box in rotated space
   * 4. Apply shift_y=-0.1, scale=2.0, square_long=true
   */
  function landmarksToROI(
    landmarks: Landmark[], imgW: number, imgH: number
  ): { centerXpx: number; centerYpx: number; sizePx: number; rotation: number } {
    // Step 1: Compute rotation from wrist + finger MCPs
    // MediaPipe uses wrist(0), indexMCP(5), middleMCP(9), ringMCP(13)
    const wrist = landmarks[0]!;
    const indexMCP = landmarks[5]!;
    const middleMCP = landmarks[9]!;
    const ringMCP = landmarks[13]!;

    const x0 = wrist.x * imgW;
    const y0 = wrist.y * imgH;

    // Average index+ring MCPs, then average with middle MCP
    let x1 = (indexMCP.x + ringMCP.x) / 2;
    let y1 = (indexMCP.y + ringMCP.y) / 2;
    x1 = (x1 + middleMCP.x) / 2 * imgW;
    y1 = (y1 + middleMCP.y) / 2 * imgH;

    const rawRotation = Math.PI / 2 - Math.atan2(-(y1 - y0), x1 - x0);
    const rotation = rawRotation - 2 * Math.PI * Math.floor((rawRotation + Math.PI) / (2 * Math.PI));

    // Step 2: Extract 12 partial landmarks for bounding box
    const partialIndices = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);

    // Rotate landmarks and find axis-aligned bounding box in rotated space
    let minRx = Infinity, maxRx = -Infinity;
    let minRy = Infinity, maxRy = -Infinity;
    for (const idx of partialIndices) {
      const lm = landmarks[idx]!;
      const px = lm.x * imgW;
      const py = lm.y * imgH;
      // Rotate to aligned space: R(-rotation)
      const rx = cosR * px + sinR * py;
      const ry = -sinR * px + cosR * py;
      minRx = Math.min(minRx, rx);
      maxRx = Math.max(maxRx, rx);
      minRy = Math.min(minRy, ry);
      maxRy = Math.max(maxRy, ry);
    }

    // Bounding box center and size in rotated space
    const rcx = (minRx + maxRx) / 2;
    const rcy = (minRy + maxRy) / 2;
    let boxW = maxRx - minRx;
    let boxH = maxRy - minRy;

    // Rotate center back to original space
    let cx = (cosR * rcx - sinR * rcy) / imgW; // normalized
    let cy = (sinR * rcx + cosR * rcy) / imgH; // normalized
    boxW /= imgW; // normalized
    boxH /= imgH; // normalized

    // Step 3: RectTransformationCalculator (shift_y=-0.1, shift upward toward fingers)
    // MediaPipe formula with shift_x=0:
    //   x_shift = (-imgH * h * shift_y * sin(rot)) / imgW
    //   y_shift = (imgH * h * shift_y * cos(rot)) / imgH = h * shift_y * cos(rot)
    const shiftY = -0.1;
    cx += (-imgH * boxH * shiftY * sinR) / imgW;
    cy += boxH * shiftY * cosR;

    // square_long: use max dimension in pixels
    const longSidePx = Math.max(boxW * imgW, boxH * imgH);

    // scale: 2.0 (landmark path uses 2.0, palm detection path uses 2.6)
    const scale = 2.0;
    const sizePx = longSidePx * scale;

    return {
      centerXpx: cx * imgW,
      centerYpx: cy * imgH,
      sizePx,
      rotation,
    };
  }

  // GPU crop resources (lazy-initialized, reused across frames)
  const cropDevice = landmarkModel.device;
  let gpuCropPipeline: CropPipeline | null = null;
  let gpuCropOutputBuffer: GPUBuffer | null = null;
  let gpuCropSourceTexture: GPUTexture | null = null;
  let gpuCropSourceWidth = 0;
  let gpuCropSourceHeight = 0;

  function ensureCropPipeline(): CropPipeline {
    if (!gpuCropPipeline) {
      gpuCropPipeline = createCropPipeline(cropDevice);
    }
    return gpuCropPipeline;
  }

  function ensureCropOutputBuffer(): GPUBuffer {
    if (!gpuCropOutputBuffer) {
      gpuCropOutputBuffer = cropDevice.createBuffer({
        size: 3 * LANDMARK_SIZE * LANDMARK_SIZE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
    }
    return gpuCropOutputBuffer;
  }

  function ensureCropSourceTexture(width: number, height: number): GPUTexture {
    if (!gpuCropSourceTexture || gpuCropSourceWidth !== width || gpuCropSourceHeight !== height) {
      if (gpuCropSourceTexture) gpuCropSourceTexture.destroy();
      gpuCropSourceTexture = cropDevice.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      gpuCropSourceWidth = width;
      gpuCropSourceHeight = height;
    }
    return gpuCropSourceTexture;
  }

  // Letterbox padding for removing letterbox from detections
  let lbPadX = 0; // normalized padding on left (and right)
  let lbPadY = 0; // normalized padding on top (and bottom)

  /**
   * Remove letterbox from a palm detection.
   * Matches MediaPipe's DetectionLetterboxRemovalCalculator.
   * Converts from letterbox [0,1] coords to image [0,1] coords.
   */
  function removeLetterbox(det: PalmDetection): PalmDetection {
    const sx = 1 / (1 - 2 * lbPadX);
    const sy = 1 / (1 - 2 * lbPadY);
    return {
      score: det.score,
      box: [
        (det.box[0] - lbPadX) * sx,
        (det.box[1] - lbPadY) * sy,
        det.box[2] * sx,
        det.box[3] * sy,
      ],
      keypoints: det.keypoints.map(([kx, ky]) => [
        (kx - lbPadX) * sx,
        (ky - lbPadY) * sy,
      ]),
    };
  }

  /**
   * Compute hand crop ROI from a detection in image-normalized coords.
   * Matches MediaPipe's DetectionsToRectsCalculator + RectTransformationCalculator.
   *
   * Returns pixel-based ROI for cropping (center in pixels, size in pixels).
   * The crop is always SQUARE in pixel space.
   */
  function detectionToPixelROI(
    det: PalmDetection, imgW: number, imgH: number
  ): { centerXpx: number; centerYpx: number; sizePx: number; rotation: number } {
    // Step 1: DetectionsToRectsCalculator — compute rotation from keypoints
    // MediaPipe computes atan2 in PIXEL space (critical for non-square images!)
    const wrist = det.keypoints[0];
    const middleMCP = det.keypoints[2];
    const dxPx = (middleMCP[0] - wrist[0]) * imgW;
    const dyPx = (middleMCP[1] - wrist[1]) * imgH;
    const angle = Math.atan2(-dyPx, dxPx);
    const targetAngle = Math.PI / 2;
    // NormalizeRadians: wrap to [-PI, PI] matching MediaPipe's DetectionsToRectsCalculator
    const rawRotation = targetAngle - angle;
    const rotation = rawRotation - 2 * Math.PI * Math.floor((rawRotation + Math.PI) / (2 * Math.PI));

    // Step 2: RectTransformationCalculator
    // MediaPipe applies shift BEFORE square_long, using original box dimensions!
    // Reference: rect_transformation_calculator.cc TransformNormalizedRect()
    const [cx, cy, w, h] = det.box;
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);

    // Shift: shift_x=0, shift_y=-0.5
    // MediaPipe formula (from C++ source):
    //   x_shift = (imgW*w*shift_x*cos(r) - imgH*h*shift_y*sin(r)) / imgW
    //   y_shift = (imgW*w*shift_x*sin(r) + imgH*h*shift_y*cos(r)) / imgH
    // With shift_x=0, shift_y=-0.5:
    //   x_shift = (imgH*h*0.5*sin(r)) / imgW
    //   y_shift = h*(-0.5)*cos(r)
    const boxHpx = h * imgH; // original box height in pixels (BEFORE square_long!)
    const newCx = cx + (0.5 * boxHpx * sinR) / imgW;
    const newCy = cy + (-0.5 * h * cosR);

    // THEN square_long: long_side in pixels
    const longSidePx = Math.max(w * imgW, h * imgH);

    // Scale: final size in pixels
    const scale = 2.6;
    const sizePx = longSidePx * scale;

    return {
      centerXpx: newCx * imgW,
      centerYpx: newCy * imgH,
      sizePx,
      rotation,
    };
  }

  function getSourceDimensions(source: HandposeInput): [number, number] {
    if (source instanceof HTMLCanvasElement || source instanceof OffscreenCanvas) {
      return [source.width, source.height];
    }
    if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
      return [source.width, source.height];
    }
    if (source instanceof ImageData) {
      return [source.width, source.height];
    }
    if (source instanceof HTMLVideoElement) {
      return [source.videoWidth, source.videoHeight];
    }
    if (source instanceof HTMLImageElement) {
      return [source.naturalWidth, source.naturalHeight];
    }
    return [LANDMARK_SIZE, LANDMARK_SIZE];
  }

  /**
   * Run landmark inference for a single ROI. Shared between palm-detection and tracking paths.
   * Returns landmarks + metadata, or null if hand score is below threshold.
   */
  async function runLandmarkForROI(
    pxROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number },
    srcTexture: GPUTexture,
    srcWidth: number, srcHeight: number,
    cropPipeline: CropPipeline, cropOutputBuf: GPUBuffer,
  ): Promise<{ landmarks: Landmark[]; score: number; handedness: 'left' | 'right' } | null> {
    const cosR = Math.cos(pxROI.rotation);
    const sinR = Math.sin(pxROI.rotation);
    const s = pxROI.sizePx / LANDMARK_SIZE;

    const halfLM = LANDMARK_SIZE / 2;
    const a = cosR * s / srcWidth;
    const b = -sinR * s / srcWidth;
    const tx = pxROI.centerXpx / srcWidth - halfLM * (a + b);
    const c = sinR * s / srcHeight;
    const d = cosR * s / srcHeight;
    const ty = pxROI.centerYpx / srcHeight - halfLM * (c + d);

    const encoder = cropDevice.createCommandEncoder();
    cropPipeline.crop(
      encoder, srcTexture, cropOutputBuf,
      [a, b, tx, c, d, ty],
      srcWidth, srcHeight, LANDMARK_SIZE,
    );
    cropDevice.queue.submit([encoder.finish()]);

    const output = await landmarkModel.runFromGPUBuffer(cropOutputBuf);
    const handScore = output.handflag[0]!;

    if (handScore < scoreThreshold) return null;

    const isRight = output.handedness[0]! > 0.5;
    const landmarks: Landmark[] = [];

    for (let i = 0; i < 21; i++) {
      const lx = output.landmarks[i * 3]!;
      const ly = output.landmarks[i * 3 + 1]!;
      const lz = output.landmarks[i * 3 + 2]!;

      const dx = (lx - 0.5) * pxROI.sizePx;
      const dy = (ly - 0.5) * pxROI.sizePx;

      const origXpx = cosR * dx - sinR * dy + pxROI.centerXpx;
      const origYpx = sinR * dx + cosR * dy + pxROI.centerYpx;

      landmarks.push({
        x: origXpx / srcWidth,
        y: origYpx / srcHeight,
        z: lz,
      });
    }

    return { landmarks, score: handScore, handedness: isRight ? 'right' : 'left' };
  }

  async function detect(source: HandposeInput): Promise<HandposeResult[]> {
    // Normalize video/image to ImageBitmap early to handle mobile camera rotation.
    let normalizedSource: HandposeInput = source;
    let srcWidth: number;
    let srcHeight: number;

    if (source instanceof HTMLVideoElement || source instanceof HTMLImageElement) {
      const bmp = await createImageBitmap(source, { colorSpaceConversion: 'none' });
      normalizedSource = bmp;
      srcWidth = bmp.width;
      srcHeight = bmp.height;
    } else {
      [srcWidth, srcHeight] = getSourceDimensions(source);
    }

    // Upload source to GPU texture (shared by both tracking and detection paths)
    const cropPipeline = ensureCropPipeline();
    const cropOutputBuf = ensureCropOutputBuffer();
    const srcTexture = ensureCropSourceTexture(srcWidth, srcHeight);
    let uploadSource: HTMLCanvasElement | OffscreenCanvas | ImageBitmap;
    if (normalizedSource instanceof ImageData) {
      uploadSource = await createImageBitmap(normalizedSource, { colorSpaceConversion: 'none' });
    } else {
      uploadSource = normalizedSource as HTMLCanvasElement | OffscreenCanvas | ImageBitmap;
    }
    cropDevice.queue.copyExternalImageToTexture(
      { source: uploadSource },
      { texture: srcTexture },
      [srcWidth, srcHeight],
    );

    // ---- TRACKING PATH ----
    // If we have previous landmarks, try to track using landmark-derived ROI
    // (skip palm detection — matches MediaPipe's approach for smooth, fast tracking)
    if (trackedHands.length > 0) {
      const results: HandposeResult[] = [];

      for (const tracked of trackedHands) {
        // Compute ROI from previous landmarks (MediaPipe's landmark-to-ROI path)
        const pxROI = landmarksToROI(tracked.landmarks, srcWidth, srcHeight);

        const result = await runLandmarkForROI(
          pxROI, srcTexture, srcWidth, srcHeight, cropPipeline, cropOutputBuf
        );

        if (result) {
          results.push({
            score: result.score,
            handedness: result.handedness,
            landmarks: result.landmarks,
            keypoints: toKeypoints(result.landmarks),
          });
        }
        // If hand score < threshold, this tracked hand is lost — don't add to results
      }

      if (results.length > 0) {
        // Tracking succeeded — update tracked hands for next frame
        trackedHands = results.map(r => ({ landmarks: r.landmarks, handedness: r.handedness }));
        return results;
      }

      // All tracked hands lost — fall through to palm detection
      trackedHands = [];
    }

    // ---- DETECTION PATH ----
    // No tracked hands (first frame or tracking lost) — run palm detection
    const { detections: rawDetections, lbPadX: gpuLbPadX, lbPadY: gpuLbPadY } =
      await palmDetector.detectRawWithResize(normalizedSource, srcWidth, srcHeight);
    lbPadX = gpuLbPadX;
    lbPadY = gpuLbPadY;

    if (rawDetections.length === 0) {
      trackedHands = [];
      return [];
    }

    const results: HandposeResult[] = [];

    for (const rawDet of rawDetections) {
      const det = removeLetterbox(rawDet);
      const pxROI = detectionToPixelROI(det, srcWidth, srcHeight);

      const result = await runLandmarkForROI(
        pxROI, srcTexture, srcWidth, srcHeight, cropPipeline, cropOutputBuf
      );

      if (result) {
        results.push({
          score: result.score,
          handedness: result.handedness,
          landmarks: result.landmarks,
          keypoints: toKeypoints(result.landmarks),
        });
      }
    }

    // Store for tracking on next frame
    trackedHands = results.map(r => ({ landmarks: r.landmarks, handedness: r.handedness }));

    return results;
  }

  function dispose(): void {
    if (gpuCropSourceTexture) gpuCropSourceTexture.destroy();
    if (gpuCropOutputBuffer) gpuCropOutputBuffer.destroy();
    gpuCropSourceTexture = null;
    gpuCropOutputBuffer = null;
    gpuCropPipeline = null;
    landmarkModel.device.destroy();
    palmModel.device.destroy();
  }

  /** Reset tracking state (call between unrelated images or when hand is lost) */
  function reset(): void {
    trackedHands = [];
  }

  return { detect, dispose, reset };
}
