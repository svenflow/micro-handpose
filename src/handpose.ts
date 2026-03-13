import { compileModel, loadWeightsFromBuffer } from './model.js';
import type { CompiledModel, WeightsMetadata } from './model.js';
import { compilePalmModel } from './palm_model.js';
import { createPalmDetector, computeCropTransform, projectLandmarksToOriginal } from './palm_detection.js';
import type { HandROI } from './palm_detection.js';
import type { Handpose, HandposeInput, HandposeOptions, HandposeResult, FullHandpose, FullHandposeResult, Landmark } from './types.js';

// Default: jsdelivr CDN (auto-mirrors npm packages)
const DEFAULT_WEIGHTS_BASE = 'https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights';

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
export async function createHandpose(options: HandposeOptions = {}): Promise<Handpose> {
  const {
    weightsUrl,
    scoreThreshold = 0.5,
    forceF32 = false,
  } = options;

  // Check WebGPU support
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('micro-handpose requires WebGPU. Check browser support at https://webgpureport.org');
  }

  // Load weights (prefer f16 — smaller download + enables GPU f16 path)
  const baseUrl = weightsUrl ?? DEFAULT_WEIGHTS_BASE;
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const metaUrl = `${base}weights_f16.json`;
  const binUrl = `${base}weights_f16.bin`;

  const [metaRes, binRes] = await Promise.all([
    fetch(metaUrl),
    fetch(binUrl),
  ]);

  if (!metaRes.ok) throw new Error(`Failed to fetch weights metadata: ${metaRes.status}`);
  if (!binRes.ok) throw new Error(`Failed to fetch weights binary: ${binRes.status}`);

  const metadata: WeightsMetadata = await metaRes.json();
  const buffer = await binRes.arrayBuffer();
  const weights = loadWeightsFromBuffer(metadata, buffer);

  // Compile model (with f16 auto-detection unless forceF32)
  let model: CompiledModel = await compileModel(weights, { forceF32 });

  // Self-test: run inference on a patterned canvas and verify non-zero output.
  // Some platforms (macOS Chrome) pass f16 validation but silently produce zeros
  // with the actual model shaders. This catches those cases.
  if (!forceF32) {
    const testCanvas = new OffscreenCanvas(256, 256);
    const testCtx = testCanvas.getContext('2d')!;
    testCtx.fillStyle = '#886644';
    testCtx.fillRect(0, 0, 256, 256);
    testCtx.fillStyle = '#cc9966';
    testCtx.fillRect(50, 50, 156, 156);
    const testOutput = await model.runFromCanvas(testCanvas);
    const allZero = testOutput.landmarks.every(v => v === 0) &&
                    testOutput.handflag.every(v => v === 0);
    if (allZero) {
      console.warn('[micro-handpose] f16 model produced all-zero output — recompiling with f32');
      model.device.destroy();
      model = await compileModel(weights, { forceF32: true });
    }
  }

  // Scratch canvas for converting various input types to ImageBitmap
  let scratchCanvas: OffscreenCanvas | null = null;

  function getScratchCanvas(): OffscreenCanvas {
    if (!scratchCanvas) {
      scratchCanvas = new OffscreenCanvas(256, 256);
    }
    return scratchCanvas;
  }

  /**
   * Convert any supported input type to something the model can consume.
   * The model's runFromCanvas accepts: HTMLCanvasElement, OffscreenCanvas, ImageBitmap.
   * For other types, we draw to a scratch canvas first.
   */
  async function toModelInput(source: HandposeInput): Promise<HTMLCanvasElement | OffscreenCanvas | ImageBitmap> {
    // These types work directly with copyExternalImageToTexture
    if (source instanceof HTMLCanvasElement || source instanceof OffscreenCanvas) {
      return source;
    }

    if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
      return source;
    }

    // HTMLImageElement, HTMLVideoElement, ImageData — draw to scratch canvas
    const canvas = getScratchCanvas();
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    if (source instanceof ImageData) {
      ctx.putImageData(source, 0, 0);
    } else {
      // HTMLImageElement or HTMLVideoElement
      ctx.drawImage(source, 0, 0, 256, 256);
    }

    return canvas;
  }

  /** Parse raw model output into clean result */
  function parseOutput(
    handflag: Float32Array,
    handedness: Float32Array,
    landmarks: Float32Array,
  ): HandposeResult | null {
    const score = handflag[0]!;

    if (score < scoreThreshold) {
      return null;
    }

    // handedness: 0 = left, 1 = right (sigmoid output)
    const isRight = handedness[0]! > 0.5;

    // landmarks: 63 values = 21 points × 3 (x, y, z)
    const points: Landmark[] = [];
    for (let i = 0; i < 21; i++) {
      points.push({
        x: landmarks[i * 3]!,
        y: landmarks[i * 3 + 1]!,
        z: landmarks[i * 3 + 2]!,
      });
    }

    return {
      score,
      handedness: isRight ? 'right' : 'left',
      landmarks: points,
    };
  }

  async function detect(source: HandposeInput): Promise<HandposeResult | null> {
    const input = await toModelInput(source);
    const output = await model.runFromCanvas(input);
    return parseOutput(output.handflag, output.handedness, output.landmarks);
  }

  async function detectPipelined(source: HandposeInput): Promise<HandposeResult | null> {
    const input = await toModelInput(source);
    const output = await model.runFromCanvasPipelined(input);
    if (!output) return null;
    return parseOutput(output.handflag, output.handedness, output.landmarks);
  }

  async function flushPipelined(): Promise<HandposeResult | null> {
    const output = await model.flushPipelined();
    if (!output) return null;
    return parseOutput(output.handflag, output.handedness, output.landmarks);
  }

  function dispose(): void {
    model.device.destroy();
    scratchCanvas = null;
  }

  async function benchmarkDiagnostic(source: HandposeInput) {
    const input = await toModelInput(source);
    return model.benchmarkDiagnostic(input);
  }

  async function debugLayerOutputs(source: HandposeInput) {
    const input = await toModelInput(source);
    return model.debugLayerOutputs(input);
  }

  return { detect, detectPipelined, flushPipelined, dispose, benchmarkDiagnostic, debugLayerOutputs };
}

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
export async function createFullHandpose(options: HandposeOptions = {}): Promise<FullHandpose> {
  const {
    weightsUrl,
    palmWeightsUrl,
    scoreThreshold = 0.5,
    palmScoreThreshold = 0.5,
    maxHands = 2,
    forceF32 = false,
  } = options;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('micro-handpose requires WebGPU. Check browser support at https://webgpureport.org');
  }

  // Load both weight sets in parallel
  const baseUrl = weightsUrl ?? DEFAULT_WEIGHTS_BASE;
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;

  if (!palmWeightsUrl) {
    throw new Error('palmWeightsUrl is required for createFullHandpose');
  }
  const palmBase = palmWeightsUrl.endsWith('/') ? palmWeightsUrl : `${palmWeightsUrl}/`;

  const [landmarkMetaRes, landmarkBinRes, palmMetaRes, palmBinRes] = await Promise.all([
    fetch(`${base}weights_f16.json`),
    fetch(`${base}weights_f16.bin`),
    fetch(`${palmBase}palm_detection_weights.json`),
    fetch(`${palmBase}palm_detection_weights.bin`),
  ]);

  if (!landmarkMetaRes.ok) throw new Error(`Failed to fetch landmark weights metadata: ${landmarkMetaRes.status}`);
  if (!landmarkBinRes.ok) throw new Error(`Failed to fetch landmark weights binary: ${landmarkBinRes.status}`);
  if (!palmMetaRes.ok) throw new Error(`Failed to fetch palm weights metadata: ${palmMetaRes.status}`);
  if (!palmBinRes.ok) throw new Error(`Failed to fetch palm weights binary: ${palmBinRes.status}`);

  const [landmarkMeta, landmarkBuf, palmMeta, palmBuf] = await Promise.all([
    landmarkMetaRes.json() as Promise<WeightsMetadata>,
    landmarkBinRes.arrayBuffer(),
    palmMetaRes.json() as Promise<WeightsMetadata>,
    palmBinRes.arrayBuffer(),
  ]);

  const landmarkWeights = loadWeightsFromBuffer(landmarkMeta, landmarkBuf);
  const palmWeights = loadWeightsFromBuffer(palmMeta, palmBuf);

  // Compile both models
  const landmarkModel = await compileModel(landmarkWeights, { forceF32 });

  // Compile palm model (uses its own device for now; could share in future)
  const palmModel = await compilePalmModel(palmWeights);
  const palmDetector = createPalmDetector(palmModel, {
    scoreThreshold: palmScoreThreshold,
    maxHands,
  });

  // Scratch canvases for input conversion and cropping
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

  /** Convert input to a form suitable for palm detection (192x192) */
  async function toPalmInput(source: HandposeInput): Promise<HTMLCanvasElement | OffscreenCanvas | ImageBitmap> {
    if (source instanceof HTMLCanvasElement || source instanceof OffscreenCanvas) {
      // Resize to 192x192 if needed
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
      // Create temp canvas at source size, then resize
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

    // Apply the inverse affine transform:
    // We want to sample from the source at positions determined by the ROI
    const cos = Math.cos(-roi.rotation);
    const sin = Math.sin(-roi.rotation);

    ctx.clearRect(0, 0, 256, 256);
    ctx.save();

    // Map from 256x256 crop space to source space
    // 1. Translate crop center to origin
    ctx.translate(128, 128);
    // 2. Scale from crop pixels to source pixels
    ctx.scale(roi.width * sourceWidth / 256, roi.height * sourceHeight / 256);
    // 3. Rotate
    ctx.rotate(-roi.rotation);
    // 4. Translate back to crop origin, then translate to source ROI center
    ctx.translate(-128, -128);

    // Now draw the source image centered at the ROI
    // The transform above maps crop space to a window in source space
    // We need to draw the source such that the ROI center maps to crop center
    const srcCenterX = roi.centerX * sourceWidth;
    const srcCenterY = roi.centerY * sourceHeight;

    // Undo the transform to figure out source drawing position
    // Actually, let's use setTransform directly for precision
    ctx.restore();

    // Direct approach: set transform matrix, then draw source
    // Canvas 2D transform: [a, b, c, d, e, f]
    // Maps (srcX, srcY) → (a*srcX + c*srcY + e, b*srcX + d*srcY + f)
    // We want: crop_x = (src_x - srcCenterX) * cos * scale + ... + 128
    const scaleX = 256 / (roi.width * sourceWidth);
    const scaleY = 256 / (roi.height * sourceHeight);

    const cosR = Math.cos(roi.rotation);
    const sinR = Math.sin(roi.rotation);

    // Transform from source pixels to crop pixels:
    // 1. Translate so ROI center is at origin: x' = x - srcCenterX
    // 2. Rotate by -rotation: x'' = x'*cos + y'*sin
    // 3. Scale to crop size: x''' = x'' * scaleX
    // 4. Translate to crop center: x'''' = x''' + 128

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

  async function detect(source: HandposeInput): Promise<FullHandposeResult[]> {
    // Step 1: Palm detection
    const palmInput = await toPalmInput(source);
    const rois = await palmDetector.detect(palmInput);

    if (rois.length === 0) return [];

    const [srcWidth, srcHeight] = getSourceDimensions(source);
    const results: FullHandposeResult[] = [];

    // Step 2: For each detected palm, crop and run landmark model
    for (const roi of rois) {
      // Crop hand region to 256x256
      const croppedCanvas = cropHandRegion(source, roi, srcWidth, srcHeight);

      // Run landmark model
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

      // Project landmarks back to original image coordinates
      const originalLandmarks = projectLandmarksToOriginal(cropLandmarks, roi);

      // Get the palm detection score from detectRaw for this ROI
      // (For simplicity, we use the landmark model's confidence score)
      results.push({
        score: handScore,
        handedness: isRight ? 'right' : 'left',
        landmarks: originalLandmarks,
        palmScore: 0, // TODO: pass through from palm detection
      });
    }

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
