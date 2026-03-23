import { loadWeightsFromBuffer } from './model.js';
import type { CompiledModel, WeightsMetadata } from './model.js';
import { compileFullModel, loadFullWeightsFromBuffer } from './model_full.js';
import { compilePalmModel } from './palm_model.js';
import { createPalmDetector } from './palm_detection.js';
import type { PalmDetection } from './palm_detection.js';
import type { Handpose, HandposeInput, HandposeOptions, HandposeResult, HandposeDebugResult, Landmark } from './types.js';
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
  // When f16 is available and forceF32 is not set, use f16 compute to match
  // MediaPipe's mediump WebGL behavior (reduces ~1.5% precision gap to near-zero)
  let landmarkModel = await compileFullModel(landmarkWeights, { forceF32 });

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
    if (!cropScratchCanvas) cropScratchCanvas = new OffscreenCanvas(LANDMARK_SIZE, LANDMARK_SIZE);
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

  /**
   * Crop a hand region from the source image and render to LANDMARK_SIZE canvas.
   * Uses pixel-based ROI (center and size in original image pixels).
   * The crop is always SQUARE in pixel space, matching MediaPipe.
   */
  function cropHandRegion(
    source: HandposeInput,
    pxROI: { centerXpx: number; centerYpx: number; sizePx: number; rotation: number },
  ): OffscreenCanvas {
    const canvas = getCropScratchCanvas();
    canvas.width = LANDMARK_SIZE;
    canvas.height = LANDMARK_SIZE;
    const ctx = canvas.getContext('2d')!;

    ctx.clearRect(0, 0, LANDMARK_SIZE, LANDMARK_SIZE);

    const s = LANDMARK_SIZE / pxROI.sizePx;
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
    const half = LANDMARK_SIZE / 2;
    const e = -pxROI.centerXpx * a - pxROI.centerYpx * c + half;
    const f = -pxROI.centerXpx * b - pxROI.centerYpx * d + half;

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
    return [LANDMARK_SIZE, LANDMARK_SIZE];
  }

  async function detect(source: HandposeInput): Promise<HandposeResult[]> {
    // Normalize video/image to ImageBitmap early to handle mobile camera rotation.
    // On iOS Safari, video.videoWidth/videoHeight may report sensor dimensions (landscape)
    // while createImageBitmap returns display-oriented pixels (portrait). Using ImageBitmap
    // ensures consistent dimensions throughout the pipeline.
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

    // Step 1: Palm detection with GPU letterbox resize (matches MediaPipe's bilinear exactly)
    const { detections: rawDetections, lbPadX: gpuLbPadX, lbPadY: gpuLbPadY } =
      await palmDetector.detectRawWithResize(normalizedSource, srcWidth, srcHeight);
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

    // Upload source to GPU texture
    // normalizedSource is already an ImageBitmap for video/image, or canvas/ImageData
    // colorSpaceConversion:'none' prevents sRGB→P3 conversion on wide-gamut displays
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

    for (const rawDet of rawDetections) {
      // Remove letterbox from detection (letterbox → image normalized coords)
      const det = removeLetterbox(rawDet);

      // Compute pixel-based ROI matching MediaPipe's exact pipeline
      const pxROI = detectionToPixelROI(det, srcWidth, srcHeight);

      // Compute affine transform: crop pixel [0,LANDMARK_SIZE] → source normalized [0,1]
      // The crop shader expects: source_x = a * crop_x + b * crop_y + tx
      //                          source_y = c * crop_x + d * crop_y + ty
      // where crop coords include +0.5 pixel center offset (handled in shader)
      const cosR = Math.cos(pxROI.rotation);
      const sinR = Math.sin(pxROI.rotation);
      const s = pxROI.sizePx / LANDMARK_SIZE; // scale from crop pixels to source pixels

      // Affine: crop pixel (fx,fy) → source normalized coords via R(+rotation) + scale + translate
      // The canvas crop applied R(-rotation) to go source→crop.
      // Inverse is R(+rotation): crop→source.
      // R(θ) = [[cos, -sin], [sin, cos]]
      // src_x = cos(r)/s * crop_x - sin(r)/s * crop_y + ...  (in pixel space)
      // Then normalize by dividing by srcWidth/srcHeight
      const halfLM = LANDMARK_SIZE / 2;
      const a = cosR * s / srcWidth;
      const b = -sinR * s / srcWidth;
      const tx = pxROI.centerXpx / srcWidth - halfLM * (a + b);
      const c = sinR * s / srcHeight;
      const d = cosR * s / srcHeight;
      const ty = pxROI.centerYpx / srcHeight - halfLM * (c + d);

      // Run GPU crop shader
      const encoder = cropDevice.createCommandEncoder();
      cropPipeline.crop(
        encoder, srcTexture, cropOutputBuf,
        [a, b, tx, c, d, ty],
        srcWidth, srcHeight, LANDMARK_SIZE,
      );
      cropDevice.queue.submit([encoder.finish()]);

      // Run landmark inference directly from GPU buffer (no CPU roundtrip)
      const output = await landmarkModel.runFromGPUBuffer(cropOutputBuf);
      const handScore = output.handflag[0]!;

      if (handScore < scoreThreshold) continue;

      const isRight = output.handedness[0]! > 0.5;

      // Parse landmarks in crop space [0, 1] and project to original image [0, 1]
      // Matches MediaPipe's LandmarkProjectionCalculator (NORM_RECT path):
      //   Step 1: Rotate isotropically in centered [-0.5, 0.5] space
      //   Step 2: Scale by rect.width()/rect.height() and add center
      // Reference: mediapipe/calculators/util/landmark_projection_calculator.cc
      const landmarks: Landmark[] = [];

      for (let i = 0; i < 21; i++) {
        const lx = output.landmarks[i * 3]!;     // crop x [0,1]
        const ly = output.landmarks[i * 3 + 1]!; // crop y [0,1]
        const lz = output.landmarks[i * 3 + 2]!;

        // Project from LANDMARK_SIZE crop back to original image pixels
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

  /**
   * Detect hands with full debug information.
   * Returns intermediate pipeline values for each hand:
   * - cropLandmarks: raw [0,1] landmarks in crop space (before back-projection)
   * - roi: crop ROI in pixel space
   * - palmDetection: palm detection after letterbox removal
   */
  async function detectWithDebug(source: HandposeInput): Promise<HandposeDebugResult[]> {
    // Normalize video/image to ImageBitmap (same as detect() — handles mobile rotation)
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

    const { detections: rawDetections, lbPadX: gpuLbPadX, lbPadY: gpuLbPadY } =
      await palmDetector.detectRawWithResize(normalizedSource, srcWidth, srcHeight);
    lbPadX = gpuLbPadX;
    lbPadY = gpuLbPadY;

    if (rawDetections.length === 0) return [];

    const results: HandposeDebugResult[] = [];

    const cropPipeline = ensureCropPipeline();
    const cropOutputBuf = ensureCropOutputBuffer();
    const srcTexture = ensureCropSourceTexture(srcWidth, srcHeight);

    // Upload source to GPU texture
    let debugUploadSource: HTMLCanvasElement | OffscreenCanvas | ImageBitmap;
    if (normalizedSource instanceof ImageData) {
      debugUploadSource = await createImageBitmap(normalizedSource, { colorSpaceConversion: 'none' });
    } else {
      debugUploadSource = normalizedSource as HTMLCanvasElement | OffscreenCanvas | ImageBitmap;
    }
    cropDevice.queue.copyExternalImageToTexture(
      { source: debugUploadSource }, { texture: srcTexture }, [srcWidth, srcHeight],
    );

    for (const rawDet of rawDetections) {
      const det = removeLetterbox(rawDet);
      const pxROI = detectionToPixelROI(det, srcWidth, srcHeight);

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

      if (handScore < scoreThreshold) continue;

      const isRight = output.handedness[0]! > 0.5;

      // Save raw crop-space landmarks
      const cropLandmarks: Landmark[] = [];
      const landmarks: Landmark[] = [];

      for (let i = 0; i < 21; i++) {
        const lx = output.landmarks[i * 3]!;
        const ly = output.landmarks[i * 3 + 1]!;
        const lz = output.landmarks[i * 3 + 2]!;

        cropLandmarks.push({ x: lx, y: ly, z: lz });

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

      results.push({
        score: handScore,
        handedness: isRight ? 'right' : 'left',
        landmarks,
        keypoints: toKeypoints(landmarks),
        cropLandmarks,
        roi: { ...pxROI },
        palmDetection: {
          score: det.score,
          box: [...det.box] as [number, number, number, number],
          keypoints: det.keypoints.map(([kx, ky]) => [kx, ky] as [number, number]),
        },
      });
    }

    return results;
  }

  /**
   * Run crop + landmark inference from externally-provided palm detections.
   * Useful for hybrid pipelines (e.g., WASM palm detection + WebGPU landmarks).
   */
  async function detectFromDetections(
    source: HandposeInput,
    detections: Array<{ score: number; box: [number, number, number, number]; keypoints: [number, number][] }>,
  ): Promise<HandposeResult[]> {
    const [srcWidth, srcHeight] = getSourceDimensions(source);
    if (detections.length === 0) return [];

    const results: HandposeResult[] = [];

    const cropPipeline = ensureCropPipeline();
    const cropOutputBuf = ensureCropOutputBuffer();
    const srcTexture = ensureCropSourceTexture(srcWidth, srcHeight);

    let uploadSource: HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement;
    if (source instanceof ImageData) {
      uploadSource = await createImageBitmap(source, { colorSpaceConversion: 'none' });
    } else if (source instanceof HTMLImageElement) {
      uploadSource = await createImageBitmap(source, { colorSpaceConversion: 'none' });
    } else {
      uploadSource = source as HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement;
    }
    cropDevice.queue.copyExternalImageToTexture(
      { source: uploadSource }, { texture: srcTexture }, [srcWidth, srcHeight],
    );

    for (const det of detections) {
      // Compute pixel-based ROI (detections should already be in image-normalized coords)
      const pxROI = detectionToPixelROI(det as any, srcWidth, srcHeight);

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
      if (handScore < scoreThreshold) continue;

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
        landmarks.push({ x: origXpx / srcWidth, y: origYpx / srcHeight, z: lz });
      }

      results.push({
        score: handScore,
        handedness: isRight ? 'right' : 'left',
        landmarks,
        keypoints: toKeypoints(landmarks),
      });
    }

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

  /** Reset temporal smoothing state (call between unrelated images) */
  function reset(): void {
    for (const s of smoothers) s.reset();
    prevHandCount = 0;
  }

  return { detect, detectWithDebug, detectFromDetections, dispose, reset, _debug };
}
