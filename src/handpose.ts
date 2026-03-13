import { compileModel, loadWeightsFromBuffer } from './model.js';
import type { CompiledModel, WeightsMetadata } from './model.js';
import type { Handpose, HandposeInput, HandposeOptions, HandposeResult, Landmark } from './types.js';

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

  // Compile model
  const model: CompiledModel = await compileModel(weights, { forceF32 });

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

  return { detect, detectPipelined, flushPipelined, dispose, benchmarkDiagnostic };
}
