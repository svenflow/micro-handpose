import { compileModel, loadWeightsFromBuffer } from './model.js';
import type { CompiledModel, WeightsMetadata } from './model.js';
import { compilePalmModel } from './palm_model.js';
import { createPalmDetector } from './palm_detection.js';
import type { PalmDetection } from './palm_detection.js';
import type { Handpose, HandposeInput, HandposeOptions, HandposeResult, Landmark } from './types.js';
import { toKeypoints } from './types.js';
import { createLandmarkSmoother } from './filter.js';
import type { LandmarkSmoother } from './filter.js';
import { createCropPipeline } from './crop_shader.js';
import type { CropPipeline } from './crop_shader.js';

// Default: jsdelivr CDN (auto-mirrors npm packages)
const DEFAULT_WEIGHTS_BASE = 'https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights';

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
    forceF32 = false,
  } = options;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('micro-handpose requires WebGPU. Check browser support at https://webgpureport.org');
  }

  // Load all weights in parallel from CDN (or custom URL)
  const base = (weightsUrl ?? DEFAULT_WEIGHTS_BASE).replace(/\/$/, '') + '/';

  const [landmarkMetaRes, landmarkBinRes, palmMetaRes, palmBinRes] = await Promise.all([
    fetch(`${base}weights_f16.json`),
    fetch(`${base}weights_f16.bin`),
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

  const landmarkWeights = loadWeightsFromBuffer(landmarkMeta, landmarkBuf);
  const palmWeights = loadWeightsFromBuffer(palmMeta, palmBuf);

  // Compile landmark model (with f16 self-test)
  let landmarkModel = await compileModel(landmarkWeights, { forceF32 });

  if (!forceF32) {
    const testCanvas = new OffscreenCanvas(256, 256);
    const testCtx = testCanvas.getContext('2d')!;
    testCtx.fillStyle = '#886644';
    testCtx.fillRect(0, 0, 256, 256);
    testCtx.fillStyle = '#cc9966';
    testCtx.fillRect(50, 50, 156, 156);
    const testOutput = await landmarkModel.runFromCanvas(testCanvas);
    const allZero = testOutput.landmarks.every(v => v === 0) &&
                    testOutput.handflag.every(v => v === 0);
    if (allZero) {
      console.warn('[micro-handpose] f16 model produced all-zero output — recompiling with f32');
      landmarkModel.device.destroy();
      landmarkModel = await compileModel(landmarkWeights, { forceF32: true });
    }
  }

  // Compile palm detection model
  const palmModel = await compilePalmModel(palmWeights);
  const palmDetector = createPalmDetector(palmModel, {
    scoreThreshold: palmScoreThreshold,
    maxHands,
  });

  // Per-hand one euro filters for temporal smoothing
  const smoothers: LandmarkSmoother[] = [];
  for (let i = 0; i < maxHands; i++) {
    smoothers.push(createLandmarkSmoother());
  }
  let prevHandCount = 0;

  // Scratch canvases
  let palmScratchCanvas: OffscreenCanvas | null = null;
  let cropScratchCanvas: OffscreenCanvas | null = null;

  function getPalmScratchCanvas(): OffscreenCanvas {
    if (!palmScratchCanvas) palmScratchCanvas = new OffscreenCanvas(192, 192);
    return palmScratchCanvas;
  }

  function getCropScratchCanvas(): OffscreenCanvas {
    if (!cropScratchCanvas) cropScratchCanvas = new OffscreenCanvas(256, 256);
    return cropScratchCanvas;
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
        size: 3 * 256 * 256 * 4,
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
   * Convert input to 192x192 for palm detection using LETTERBOXING.
   * MediaPipe uses FIT mode (maintain aspect ratio, zero-pad) not STRETCH.
   */
  function toPalmInputLetterbox(source: HandposeInput, srcW: number, srcH: number): OffscreenCanvas {
    const canvas = getPalmScratchCanvas();
    canvas.width = 192;
    canvas.height = 192;
    const ctx = canvas.getContext('2d')!;

    ctx.clearRect(0, 0, 192, 192);

    const scale = Math.min(192 / srcW, 192 / srcH);
    const dstW = Math.round(srcW * scale);
    const dstH = Math.round(srcH * scale);
    const offsetX = (192 - dstW) / 2;
    const offsetY = (192 - dstH) / 2;

    lbPadX = offsetX / 192;
    lbPadY = offsetY / 192;

    if (source instanceof ImageData) {
      const tmp = new OffscreenCanvas(source.width, source.height);
      const tmpCtx = tmp.getContext('2d')!;
      tmpCtx.putImageData(source, 0, 0);
      ctx.drawImage(tmp, offsetX, offsetY, dstW, dstH);
    } else {
      ctx.drawImage(source as CanvasImageSource, 0, 0, srcW, srcH, offsetX, offsetY, dstW, dstH);
    }
    return canvas;
  }

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
    const wrist = det.keypoints[0];
    const middleMCP = det.keypoints[2];

    // MediaPipe uses: rotation = target_angle - atan2(-(y2-y0), x2-x0)
    // The negated Y accounts for image coords (Y down) vs math coords (Y up)
    const dx = middleMCP[0] - wrist[0];
    const dy = middleMCP[1] - wrist[1];
    const angle = Math.atan2(-dy, dx); // MediaPipe's convention: negate Y
    const targetAngle = Math.PI / 2;   // 90 degrees (MediaPipe's target)
    const rotation = targetAngle - angle;

    // Step 2: RectTransformationCalculator
    // long_side is computed in PIXELS (critical for non-square images!)
    const [cx, cy, w, h] = det.box;
    const longSidePx = Math.max(w * imgW, h * imgH);

    // Shift in normalized coords, but scaled by pixel-based long_side
    // MediaPipe: shift_x = shift_x_ * long_side / image_width
    //            shift_y = shift_y_ * long_side / image_height
    const shiftXnorm = 0; // shift_x_ = 0
    const shiftYnorm = -0.5 * longSidePx / imgH;

    // Apply rotation to shift vector
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);
    const newCx = cx + (shiftXnorm * cosR - shiftYnorm * sinR);
    const newCy = cy + (shiftXnorm * sinR + shiftYnorm * cosR);

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

  /**
   * Crop a hand region from the source image and render to 256x256 canvas.
   * Uses pixel-based ROI (center and size in original image pixels).
   * The crop is always SQUARE in pixel space, matching MediaPipe.
   */
  function cropHandRegion(
    source: HandposeInput,
    pxROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number },
  ): OffscreenCanvas {
    const canvas = getCropScratchCanvas();
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    ctx.clearRect(0, 0, 256, 256);

    const s = 256 / pxROI.sizePx;
    const cosR = Math.cos(pxROI.rotation);
    const sinR = Math.sin(pxROI.rotation);

    // Apply R(-rotation) to straighten the hand in the crop.
    // setTransform(a, b, c, d, e, f) maps source (x,y) → canvas:
    //   canvas_x = a*x + c*y + e
    //   canvas_y = b*x + d*y + f
    // R(-θ) = [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]]
    const a = cosR * s;
    const b = -sinR * s;
    const c = sinR * s;
    const d = cosR * s;
    const e = -pxROI.centerXpx * a - pxROI.centerYpx * c + 128;
    const f = -pxROI.centerXpx * b - pxROI.centerYpx * d + 128;

    ctx.setTransform(a, b, c, d, e, f);

    if (source instanceof ImageData) {
      const tmp = new OffscreenCanvas(source.width, source.height);
      const tmpCtx = tmp.getContext('2d')!;
      tmpCtx.putImageData(source, 0, 0);
      ctx.drawImage(tmp, 0, 0);
    } else {
      ctx.drawImage(source as CanvasImageSource, 0, 0);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return canvas;
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
    return [256, 256];
  }

  async function detect(source: HandposeInput): Promise<HandposeResult[]> {
    const [srcWidth, srcHeight] = getSourceDimensions(source);

    // Step 1: Palm detection with GPU letterbox resize (matches MediaPipe's bilinear exactly)
    const { detections: rawDetections, lbPadX: gpuLbPadX, lbPadY: gpuLbPadY } =
      await palmDetector.detectRawWithResize(source, srcWidth, srcHeight);
    // Update letterbox padding for removeLetterbox
    lbPadX = gpuLbPadX;
    lbPadY = gpuLbPadY;

    if (rawDetections.length === 0) {
      if (prevHandCount > 0) {
        for (let i = 0; i < prevHandCount && i < smoothers.length; i++) {
          smoothers[i]!.reset();
        }
      }
      prevHandCount = 0;
      return [];
    }

    const results: HandposeResult[] = [];

    // Step 2: For each detection, remove letterbox → compute ROI → GPU crop → landmark
    // Upload source image to GPU texture once per frame (reused across all hand crops)
    const cropPipeline = ensureCropPipeline();
    const cropOutputBuf = ensureCropOutputBuffer();

    // Upload source to GPU texture (only for types supported by copyExternalImageToTexture)
    const srcTexture = ensureCropSourceTexture(srcWidth, srcHeight);
    if (source instanceof ImageData) {
      // ImageData needs to go through a canvas first
      const tmp = new OffscreenCanvas(source.width, source.height);
      const tmpCtx = tmp.getContext('2d')!;
      tmpCtx.putImageData(source, 0, 0);
      cropDevice.queue.copyExternalImageToTexture(
        { source: tmp },
        { texture: srcTexture },
        [srcWidth, srcHeight],
      );
    } else {
      cropDevice.queue.copyExternalImageToTexture(
        { source: source as HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement | HTMLImageElement },
        { texture: srcTexture },
        [srcWidth, srcHeight],
      );
    }

    for (const rawDet of rawDetections) {
      // Remove letterbox from detection (letterbox → image normalized coords)
      const det = removeLetterbox(rawDet);

      // Compute pixel-based ROI matching MediaPipe's exact pipeline
      const pxROI = detectionToPixelROI(det, srcWidth, srcHeight);

      // Compute affine transform: crop pixel [0,256] → source normalized [0,1]
      // The crop shader expects: source_x = a * crop_x + b * crop_y + tx
      //                          source_y = c * crop_x + d * crop_y + ty
      // where crop coords include +0.5 pixel center offset (handled in shader)
      const cosR = Math.cos(pxROI.rotation);
      const sinR = Math.sin(pxROI.rotation);
      const s = pxROI.sizePx / 256; // scale from crop pixels to source pixels

      // Affine: crop pixel (fx,fy) → source normalized coords via R(-rotation) + scale + translate
      // fx = px + 0.5 (pixel center, added in shader)
      // src_norm = R(-rot) * s/srcDim * (fx - 128) + center/srcDim
      const a = cosR * s / srcWidth;
      const b = sinR * s / srcWidth;
      const tx = pxROI.centerXpx / srcWidth - 128 * (a + b);
      const c = -sinR * s / srcHeight;
      const d = cosR * s / srcHeight;
      const ty = pxROI.centerYpx / srcHeight - 128 * (c + d);

      // Run GPU crop shader
      const encoder = cropDevice.createCommandEncoder();
      cropPipeline.crop(
        encoder, srcTexture, cropOutputBuf,
        [a, b, tx, c, d, ty],
        srcWidth, srcHeight, 256,
      );
      cropDevice.queue.submit([encoder.finish()]);

      // Run landmark inference directly from GPU buffer (no CPU roundtrip)
      const output = await landmarkModel.runFromGPUBuffer(cropOutputBuf);
      const handScore = output.handflag[0]!;

      if (handScore < scoreThreshold) continue;

      const isRight = output.handedness[0]! > 0.5;

      // Parse landmarks in crop space [0, 1] and project to original image [0, 1]
      // cosR and sinR already computed above for the affine transform
      const landmarks: Landmark[] = [];

      for (let i = 0; i < 21; i++) {
        const lx = output.landmarks[i * 3]!;     // crop x [0,1]
        const ly = output.landmarks[i * 3 + 1]!; // crop y [0,1]
        const lz = output.landmarks[i * 3 + 2]!;

        // Project from 256x256 crop back to original image pixels
        // Crop [0,1] → centered [-0.5,0.5] → scale by physical size → inverse rotate → add center
        const dx = (lx - 0.5) * pxROI.sizePx;
        const dy = (ly - 0.5) * pxROI.sizePx;

        // Inverse of R(-rotation) = R(+rotation): crop → original image
        // R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        const origXpx = cosR * dx - sinR * dy + pxROI.centerXpx;
        const origYpx = sinR * dx + cosR * dy + pxROI.centerYpx;

        landmarks.push({
          x: origXpx / srcWidth,
          y: origYpx / srcHeight,
          z: lz,
        });
      }

      // Apply one euro filter for temporal smoothing
      const handIdx = results.length;
      const smoothedLandmarks = handIdx < smoothers.length
        ? smoothers[handIdx]!.apply(landmarks)
        : landmarks;

      results.push({
        score: handScore,
        handedness: isRight ? 'right' : 'left',
        landmarks: smoothedLandmarks,
        keypoints: toKeypoints(smoothedLandmarks),
      });
    }

    // Reset smoothers for hands that disappeared
    if (results.length < prevHandCount) {
      for (let i = results.length; i < prevHandCount; i++) {
        if (i < smoothers.length) smoothers[i]!.reset();
      }
    }
    prevHandCount = results.length;

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
    palmScratchCanvas = null;
    cropScratchCanvas = null;
  }

  // Expose internals for debugging
  const _debug = {
    palmDetector,
    palmModel,
    landmarkModel,
    removeLetterbox,
    detectionToPixelROI,
    cropHandRegion,
  };

  return { detect, dispose, _debug };
}
