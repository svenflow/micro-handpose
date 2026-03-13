import { compileModel, loadWeightsFromBuffer } from './model.js';
import type { CompiledModel, WeightsMetadata } from './model.js';
import { compilePalmModel } from './palm_model.js';
import { createPalmDetector, computeCropTransform, projectLandmarksToOriginal } from './palm_detection.js';
import type { HandROI } from './palm_detection.js';
import type { Handpose, HandposeInput, HandposeOptions, HandposeResult, Landmark } from './types.js';
import { toKeypoints } from './types.js';
import { createLandmarkSmoother } from './filter.js';
import type { LandmarkSmoother } from './filter.js';

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

  /** Convert input to 192x192 for palm detection */
  async function toPalmInput(source: HandposeInput): Promise<HTMLCanvasElement | OffscreenCanvas | ImageBitmap> {
    if (source instanceof HTMLCanvasElement || source instanceof OffscreenCanvas) {
      if (source.width === 192 && source.height === 192) return source;
      const canvas = getPalmScratchCanvas();
      canvas.width = 192;
      canvas.height = 192;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(source, 0, 0, 192, 192);
      return canvas;
    }
    if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
      if (source.width === 192 && source.height === 192) return source;
      const canvas = getPalmScratchCanvas();
      canvas.width = 192;
      canvas.height = 192;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(source, 0, 0, 192, 192);
      return canvas;
    }
    const canvas = getPalmScratchCanvas();
    canvas.width = 192;
    canvas.height = 192;
    const ctx = canvas.getContext('2d')!;
    if (source instanceof ImageData) {
      const tmp = new OffscreenCanvas(source.width, source.height);
      const tmpCtx = tmp.getContext('2d')!;
      tmpCtx.putImageData(source, 0, 0);
      ctx.drawImage(tmp, 0, 0, 192, 192);
    } else {
      ctx.drawImage(source, 0, 0, 192, 192);
    }
    return canvas;
  }

  /** Crop a hand region from the source image and render to 256x256 canvas */
  function cropHandRegion(
    source: HandposeInput,
    roi: HandROI,
    sourceWidth: number,
    sourceHeight: number,
  ): OffscreenCanvas {
    const canvas = getCropScratchCanvas();
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    ctx.clearRect(0, 0, 256, 256);
    ctx.save();
    ctx.translate(128, 128);
    ctx.scale(roi.width * sourceWidth / 256, roi.height * sourceHeight / 256);
    ctx.rotate(-roi.rotation);
    ctx.translate(-128, -128);

    const srcCenterX = roi.centerX * sourceWidth;
    const srcCenterY = roi.centerY * sourceHeight;
    ctx.restore();

    const refDim = Math.min(sourceWidth, sourceHeight);
    const scaleX = 256 / (roi.width * refDim);
    const scaleY = 256 / (roi.height * refDim);

    const cosR = Math.cos(roi.rotation);
    const sinR = Math.sin(roi.rotation);

    const a = cosR * scaleX;
    const b = sinR * scaleX;
    const c = -sinR * scaleY;
    const d = cosR * scaleY;
    const e = -srcCenterX * a - srcCenterY * c + 128;
    const f = -srcCenterX * b - srcCenterY * d + 128;

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
    // Step 1: Palm detection on full frame
    const palmInput = await toPalmInput(source);
    const rois = await palmDetector.detect(palmInput);

    if (rois.length === 0) {
      // Reset all smoothers when no hands detected
      if (prevHandCount > 0) {
        for (let i = 0; i < prevHandCount && i < smoothers.length; i++) {
          smoothers[i]!.reset();
        }
      }
      prevHandCount = 0;
      return [];
    }

    const [srcWidth, srcHeight] = getSourceDimensions(source);
    const results: HandposeResult[] = [];

    // Step 2: For each detected palm, crop and run landmark model
    for (const roi of rois) {
      const croppedCanvas = cropHandRegion(source, roi, srcWidth, srcHeight);
      const output = await landmarkModel.runFromCanvas(croppedCanvas);
      const handScore = output.handflag[0]!;

      if (handScore < scoreThreshold) continue;

      const isRight = output.handedness[0]! > 0.5;

      // Parse landmarks in crop space [0, 1]
      const cropLandmarks: Landmark[] = [];
      for (let i = 0; i < 21; i++) {
        cropLandmarks.push({
          x: output.landmarks[i * 3]!,
          y: output.landmarks[i * 3 + 1]!,
          z: output.landmarks[i * 3 + 2]!,
        });
      }

      // Project back to original image coordinates
      const originalLandmarks = projectLandmarksToOriginal(cropLandmarks, roi, srcWidth, srcHeight);

      // Apply one euro filter for temporal smoothing
      const handIdx = results.length;
      const smoothedLandmarks = handIdx < smoothers.length
        ? smoothers[handIdx]!.apply(originalLandmarks)
        : originalLandmarks;

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
    landmarkModel.device.destroy();
    palmModel.device.destroy();
    palmScratchCanvas = null;
    cropScratchCanvas = null;
  }

  return { detect, dispose };
}
