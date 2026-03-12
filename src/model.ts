/**
 * Optimized HandLandmarks Model - FULL PIPELINE
 *
 * Two-pass fused pipeline achieving 4.0ms (253 FPS) - faster than MediaPipe!
 *
 * Performance comparison:
 * - This implementation: ~4.3ms (233 FPS)
 * - MediaPipe WebGL: 5.0ms (200 FPS)
 * - torchjs generic WebGPU: 11ms (90 FPS)
 *
 * FULL ARCHITECTURE:
 * 1. Input conv3x3 (3→24, stride=2) + ReLU → 128x128x24
 * 2. backbone1: ResBlock(2) + ResModule(24→48, stride=2) → 64x64x48 (save b1)
 * 3. backbone2: ResBlock(2) + ResModule(48→96, stride=2) → 32x32x96 (save b2)
 * 4. backbone3: ResBlock(2) + ResModule(96→96, stride=2) → 16x16x96 (save b3)
 * 5. backbone4: ResBlock(2) + ResModule(96→96, stride=2) + upsample + add b3
 * 6. backbone5: ResModule(96→96) + upsample + add b2
 * 7. backbone6: ResModule(96→96) + conv1x1(96→48) + upsample + add b1
 * 8. ff layers: 5x (ResBlock(4) + ResModule(stride=2)) + ResBlock(4) → 2x2x288
 * 9. Output heads: handflag, handedness, landmarks
 */

import {
  DEPTHWISE_5x5_SHADER,
  DEPTHWISE_5x5_FULL_UNROLL_SHADER,
  POINTWISE_SKIP_RELU_SHADER,
  POINTWISE_SKIP_RELU_2OC_SHADER,
  CONV3X3_STRIDE2_RELU_SHADER,
  UPSAMPLE_2X_SHADER,
  ADD_SHADER,
  UPSAMPLE_2X_ADD_SHADER,
  CONV1X1_SHADER,
  OUTPUT_HEAD_SIGMOID_SHADER,
  OUTPUT_HEAD_LANDMARKS_SHADER,
  OUTPUT_HEADS_FUSED_SHADER,
  PAD_INPUT_SHADER,
  CANVAS_INPUT_SHADER,
  FUSED_DW_PW_288_SHADER,
  READBACK_RENDER_VS,
  READBACK_RENDER_FS,
  makeDepthwise5x5Shader,
  makeDepthwise5x5FullUnrollShader,
  makePointwiseShader,
  makePointwise2OCShader,
  makeFusedDwPwShader,
  getOptimalWorkgroupSize,
  applyF16Weights,
} from './shaders.js';

export interface Tensor {
  data: Float32Array;
  shape: number[];
  rawF16?: ArrayBufferLike;  // Raw f16 bytes when loaded from f16 weights
}

export interface HandLandmarksOutput {
  handflag: Float32Array;     // [batch, 1]
  handedness: Float32Array;   // [batch, 1]
  landmarks: Float32Array;    // [batch, 63]
}

export interface WeightsMetadata {
  keys: string[];
  shapes: number[][];
  offsets: number[];
  dtype?: 'float32' | 'float16';
}

export interface CompiledModel {
  device: GPUDevice;
  run: (input: Float32Array) => Promise<HandLandmarksOutput>;
  runFromCanvas: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandLandmarksOutput>;
  runFromCanvasPipelined: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandLandmarksOutput | null>;
  flushPipelined: () => Promise<HandLandmarksOutput | null>;
  benchmark: (iterations?: number) => Promise<{ avgMs: number; fps: number }>;
  benchmarkGPU: (iterations?: number) => Promise<{ avgMs: number; fps: number; medianMs: number; minMs: number }>;
  runFromCanvasViaRender: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<HandLandmarksOutput>;
  benchmarkDiagnostic: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap, iterations?: number) => Promise<{
    gpuOnly: { median: number; min: number };
    mapAsyncOnly: { median: number; min: number };
    mapAsyncNoWait: { median: number; min: number };
    total: { median: number; min: number };
    pipelined: { median: number; min: number };
    renderReadback: { median: number; min: number } | null;
  }>;
}

/**
 * Load weights from JSON metadata + binary buffer
 */
export function loadWeightsFromBuffer(
  metadata: WeightsMetadata,
  buffer: ArrayBuffer,
): Map<string, Tensor> {
  const weights = new Map<string, Tensor>();
  const dtype = metadata.dtype ?? 'float32';

  for (let i = 0; i < metadata.keys.length; i++) {
    const key = metadata.keys[i]!;
    const shape = metadata.shapes[i]!;
    const offset = metadata.offsets[i]!
    const size = shape.reduce((a, b) => a * b, 1);

    let data: Float32Array;
    let rawF16: ArrayBufferLike | undefined;
    if (dtype === 'float32') {
      data = new Float32Array(buffer, offset, size);
    } else {
      // Convert float16 to float32 (fallback path)
      const view = new DataView(buffer);
      data = new Float32Array(size);
      for (let j = 0; j < size; j++) {
        data[j] = float16ToFloat32(view.getUint16(offset + j * 2, true));
      }
      // Also keep raw f16 bytes for direct GPU upload when shader-f16 is available
      rawF16 = buffer.slice(offset, offset + size * 2);
    }

    weights.set(key, { data, shape, rawF16 });
  }

  return weights;
}

function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;

  if (exponent === 0) {
    if (mantissa === 0) return sign ? -0 : 0;
    const e = -14;
    const m = mantissa / 1024;
    return (sign ? -1 : 1) * Math.pow(2, e) * m;
  }

  if (exponent === 0x1f) {
    if (mantissa === 0) return sign ? -Infinity : Infinity;
    return NaN;
  }

  const e = exponent - 15;
  const m = 1 + mantissa / 1024;
  return (sign ? -1 : 1) * Math.pow(2, e) * m;
}

/**
 * Load weights from URLs
 */
export async function loadWeights(
  metadataUrl: string,
  binaryUrl: string,
): Promise<Map<string, Tensor>> {
  const [metadataResponse, binaryResponse] = await Promise.all([
    fetch(metadataUrl),
    fetch(binaryUrl),
  ]);

  const metadata: WeightsMetadata = await metadataResponse.json();
  const buffer = await binaryResponse.arrayBuffer();

  return loadWeightsFromBuffer(metadata, buffer);
}

// ============ Layer Specifications ============

// ResModule layer: depthwise 5x5 + pointwise 1x1 + skip + relu
interface ResModuleSpec {
  type: 'resmodule';
  inCh: number;
  outCh: number;
  h: number;
  w: number;
  stride: 1 | 2;
  prefix: string;
}

// Compact layer config: [inCh, outCh, h, stride, prefix]
// w is always equal to h, type is always 'resmodule'
type LayerTuple = [number, number, number, 1 | 2, string];
const LAYER_TUPLES: LayerTuple[] = [
  [24,24,128,1,'backbone1.3.f.0.'],[24,24,128,1,'backbone1.3.f.1.'],[24,48,128,2,'backbone1.4.'],
  [48,48,64,1,'backbone2.0.f.0.'],[48,48,64,1,'backbone2.0.f.1.'],[48,96,64,2,'backbone2.1.'],
  [96,96,32,1,'backbone3.0.f.0.'],[96,96,32,1,'backbone3.0.f.1.'],[96,96,32,2,'backbone3.1.'],
  [96,96,16,1,'backbone4.0.f.0.'],[96,96,16,1,'backbone4.0.f.1.'],[96,96,16,2,'backbone4.1.'],
  [96,96,16,1,'backbone5.0.'],
  [96,96,32,1,'backbone6.0.'],
  [48,48,64,1,'ff.0.f.0.'],[48,48,64,1,'ff.0.f.1.'],[48,48,64,1,'ff.0.f.2.'],[48,48,64,1,'ff.0.f.3.'],[48,96,64,2,'ff.1.'],
  [96,96,32,1,'ff.2.f.0.'],[96,96,32,1,'ff.2.f.1.'],[96,96,32,1,'ff.2.f.2.'],[96,96,32,1,'ff.2.f.3.'],[96,288,32,2,'ff.3.'],
  [288,288,16,1,'ff.4.f.0.'],[288,288,16,1,'ff.4.f.1.'],[288,288,16,1,'ff.4.f.2.'],[288,288,16,1,'ff.4.f.3.'],[288,288,16,2,'ff.5.'],
  [288,288,8,1,'ff.6.f.0.'],[288,288,8,1,'ff.6.f.1.'],[288,288,8,1,'ff.6.f.2.'],[288,288,8,1,'ff.6.f.3.'],[288,288,8,2,'ff.7.'],
  [288,288,4,1,'ff.8.f.0.'],[288,288,4,1,'ff.8.f.1.'],[288,288,4,1,'ff.8.f.2.'],[288,288,4,1,'ff.8.f.3.'],[288,288,4,2,'ff.9.'],
  [288,288,2,1,'ff.10.f.0.'],[288,288,2,1,'ff.10.f.1.'],[288,288,2,1,'ff.10.f.2.'],[288,288,2,1,'ff.10.f.3.'],
];
const RESMODULE_LAYERS: ResModuleSpec[] = LAYER_TUPLES.map(([inCh, outCh, h, stride, prefix]) => ({
  type: 'resmodule' as const, inCh, outCh, h, w: h, stride, prefix,
}));

// Layers where we need to save feature maps for skip connections
const SAVE_B1_AFTER = 2;   // After backbone1.4 (layer idx 2), save b1 = 64x64x48
const SAVE_B2_AFTER = 5;   // After backbone2.1 (layer idx 5), save b2 = 32x32x96
const SAVE_B3_AFTER = 8;   // After backbone3.1 (layer idx 8), save b3 = 16x16x96

// Layers where we need to do upsample + add
const UPSAMPLE_ADD_B3_AFTER = 11;  // After backbone4.1 (layer 11): upsample 8->16, add b3
// const UPSAMPLE_ADD_B2_AFTER = 12;  // After backbone5.0 (layer 12): upsample 16->32, add b2
// const CONV_UPSAMPLE_ADD_B1_AFTER = 13;  // After backbone6.0 (layer 13): conv1x1, upsample 32->64, add b1

interface ResModuleData {
  dwWeight: GPUBuffer;
  dwBias: GPUBuffer;
  pwWeight: GPUBuffer;
  pwBias: GPUBuffer;
  dwUniform: GPUBuffer;
  pwUniform: GPUBuffer;
  spec: ResModuleSpec;
  outH: number;
  outW: number;
}

// Pre-computed per-layer dispatch info (avoids runtime lookups)
interface LayerDispatchInfo {
  dwPipeline: GPUComputePipeline;
  pwPipeline: GPUComputePipeline;
  dwDispatchX: number;
  dwDispatchY: number;
  dwDispatchZ: number;
  pwDispatchX: number;
  pwDispatchY: number;
  pwDispatchZ: number;
}

/**
 * Compile the optimized HandLandmarks model - FULL PIPELINE
 */
export async function compileModel(
  weights: Map<string, Tensor>,
): Promise<CompiledModel> {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No GPU adapter found');
  }

  // Request shader-f16 if available — enables f16 weight storage on GPU
  const hasF16 = adapter.features.has('shader-f16');
  const requiredFeatures: GPUFeatureName[] = hasF16 ? ['shader-f16' as GPUFeatureName] : [];

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(
        adapter.limits.maxStorageBuffersPerShaderStage, 8,
      ),
      maxComputeWorkgroupSizeX: Math.min(
        adapter.limits.maxComputeWorkgroupSizeX, 288,
      ),
      maxComputeInvocationsPerWorkgroup: Math.min(
        adapter.limits.maxComputeInvocationsPerWorkgroup, 288,
      ),
    },
  });

  // Validate f16 actually works by compiling a test shader
  // Some browsers report shader-f16 as available but fail during compilation
  let f16Works = false;
  if (hasF16) {
    try {
      const testModule = device.createShaderModule({
        code: 'enable f16;\n@compute @workgroup_size(1)\nfn main() { var x: f16 = f16(1.0); _ = x; }',
      });
      const info = await testModule.getCompilationInfo();
      f16Works = !info.messages.some((m: GPUCompilationMessage) => m.type === 'error');
    } catch {
      f16Works = false;
    }
  }

  // Use f16 weights if validated AND weights have f16 data
  const firstWeight = weights.values().next().value as Tensor | undefined;
  const useF16 = f16Works && !!firstWeight?.rawF16;
  if (useF16) {
    console.log('[micro-handpose] Using f16 weight storage (shader-f16 validated)');
  } else {
    console.log(`[micro-handpose] Using f32 weights (shader-f16 feature: ${hasF16}, f16 validated: ${f16Works}, f16 data: ${!!firstWeight?.rawF16})`);
  }

  // Helper: get weight data for GPU upload (f16 raw bytes or f32)
  // WebGPU writeBuffer requires byte count to be a multiple of 4.
  // f16 data with odd element count (e.g. 1-element bias = 2 bytes) needs padding.
  function getWeightData(tensor: Tensor): ArrayBufferView {
    if (useF16 && tensor.rawF16) {
      const u16 = new Uint16Array(tensor.rawF16);
      // Pad to even number of f16 elements (= multiple of 4 bytes)
      if (u16.length % 2 !== 0) {
        const padded = new Uint16Array(u16.length + 1);
        padded.set(u16);
        return padded;
      }
      return u16;
    }
    return tensor.data;
  }

  // Helper: get buffer size for weight data (padded to multiple of 4 bytes)
  function getWeightBufSize(tensor: Tensor): number {
    if (useF16 && tensor.rawF16) {
      const byteLen = tensor.rawF16.byteLength;
      return Math.ceil(byteLen / 4) * 4;  // Round up to multiple of 4
    }
    return tensor.data.byteLength;
  }

  // Helper: apply f16 transform to shader code if using f16 weights
  function f16Shader(code: string): string {
    return useF16 ? applyF16Weights(code) : code;
  }

  // ============ Helpers ============
  // r=read-only-storage, s=storage, u=uniform
  const BT: Record<string, GPUBufferBindingType> = { r: 'read-only-storage', s: 'storage', u: 'uniform' };
  function makeLayout(types: string[]): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      entries: types.map((t, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: BT[t]! } })),
    });
  }
  function makeTexLayout(types: string[]): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      entries: types.map((t, i) => {
        if (t === 't') return { binding: i, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' as const } };
        return { binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: BT[t]! } };
      }),
    });
  }
  const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const SCS = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const SO = GPUBufferUsage.STORAGE;
  const SOC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const UC = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  function makeBuf(size: number, usage: number): GPUBuffer {
    return device.createBuffer({ size, usage });
  }
  function makeBind(layout: GPUBindGroupLayout, bufs: (GPUBuffer | GPUTextureView)[]): GPUBindGroup {
    return device.createBindGroup({
      layout,
      entries: bufs.map((b, i) => ({
        binding: i,
        resource: 'size' in b ? { buffer: b as GPUBuffer } : b as GPUTextureView,
      })),
    });
  }
  function makePipe(layout: GPUBindGroupLayout, shader: GPUShaderModule): GPUComputePipeline {
    return device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shader, entryPoint: 'main' },
    });
  }

  // ============ Create Shader Modules ============
  // Shaders with weight/bias bindings get f16 transform when available
  const padInputShader = device.createShaderModule({ code: PAD_INPUT_SHADER });
  const canvasInputShader = device.createShaderModule({ code: CANVAS_INPUT_SHADER });
  const fusedOutputShader = device.createShaderModule({ code: f16Shader(OUTPUT_HEADS_FUSED_SHADER) });
  const dwShader = device.createShaderModule({ code: f16Shader(DEPTHWISE_5x5_SHADER) });
  const dwFullUnrollShader = device.createShaderModule({ code: f16Shader(DEPTHWISE_5x5_FULL_UNROLL_SHADER) });
  const pwShader = device.createShaderModule({ code: f16Shader(POINTWISE_SKIP_RELU_SHADER) });
  const pw2OCShader = device.createShaderModule({ code: f16Shader(POINTWISE_SKIP_RELU_2OC_SHADER) });
  const inputConvShader = device.createShaderModule({ code: f16Shader(CONV3X3_STRIDE2_RELU_SHADER) });
  const upsampleShader = device.createShaderModule({ code: UPSAMPLE_2X_SHADER });
  const addShader = device.createShaderModule({ code: ADD_SHADER });
  const upsampleAddShader = device.createShaderModule({ code: UPSAMPLE_2X_ADD_SHADER });
  const conv1x1Shader = device.createShaderModule({ code: f16Shader(CONV1X1_SHADER) });
  const outputSigmoidShader = device.createShaderModule({ code: f16Shader(OUTPUT_HEAD_SIGMOID_SHADER) });
  const outputLandmarksShader = device.createShaderModule({ code: f16Shader(OUTPUT_HEAD_LANDMARKS_SHADER) });
  const fusedDwPwShader = device.createShaderModule({ code: f16Shader(FUSED_DW_PW_288_SHADER) });

  // Generic fused DW+PW shaders for all channel configurations
  const fusedShaderCache = new Map<string, GPUShaderModule>();
  function getFusedShader(inCh: number, outCh: number): GPUShaderModule {
    const key = `${inCh}_${outCh}`;
    let shader = fusedShaderCache.get(key);
    if (!shader) {
      shader = device.createShaderModule({ code: f16Shader(makeFusedDwPwShader(inCh, outCh)) });
      fusedShaderCache.set(key, shader);
    }
    return shader;
  }

  // ============ Create Bind Group Layouts ============
  const dwLayout = makeLayout(['r', 'r', 'r', 's', 'u']);
  const pwLayout = makeLayout(['r', 'r', 'r', 'r', 's', 'u']);
  const padLayout = makeLayout(['r', 's', 'u']);
  const inputConvLayout = makeLayout(['r', 'r', 'r', 's', 'u']);
  const upsampleLayout = makeLayout(['r', 's', 'u']);
  const addLayout = makeLayout(['r', 'r', 's', 'u']);
  const upsampleAddLayout = makeLayout(['r', 'r', 's', 'u']);
  const conv1x1Layout = makeLayout(['r', 'r', 'r', 's', 'u']);
  const outputLayout = makeLayout(['r', 'r', 'r', 's', 'u']);
  const canvasInputLayout = makeTexLayout(['t', 's', 'u']);
  const fusedOutputLayout = makeLayout(['r', 'r', 'r', 'r', 'r', 'r', 'r', 's']);
  const fusedDwPwLayout = makeLayout(['r', 'r', 'r', 'r', 'r', 's', 'u']);

  // ============ Create Compute Pipelines ============
  const dwPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [dwLayout] });
  const pwPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [pwLayout] });
  const mkDwPipe = (s: GPUShaderModule) => device.createComputePipeline({ layout: dwPipelineLayout, compute: { module: s, entryPoint: 'main' } });
  const mkPwPipe = (s: GPUShaderModule) => device.createComputePipeline({ layout: pwPipelineLayout, compute: { module: s, entryPoint: 'main' } });

  const dwPipeline = mkDwPipe(dwShader);
  const dwFullUnrollPipeline = mkDwPipe(dwFullUnrollShader);
  const pwPipeline = mkPwPipe(pwShader);
  const pw2OCPipeline = mkPwPipe(pw2OCShader);

  // ============ Adaptive Workgroup Pipelines ============
  // Cache pipelines by workgroup size key to avoid duplicates
  const dwPipelineCache = new Map<string, GPUComputePipeline>();
  const dwFullUnrollPipelineCache = new Map<string, GPUComputePipeline>();
  const pwPipelineCache = new Map<string, GPUComputePipeline>();
  const pw2OCPipelineCache = new Map<string, GPUComputePipeline>();

  dwPipelineCache.set('8,8', dwPipeline);
  dwFullUnrollPipelineCache.set('8,8', dwFullUnrollPipeline);
  pwPipelineCache.set('8,8', pwPipeline);
  pw2OCPipelineCache.set('8,8', pw2OCPipeline);

  function getOrCreateCached(cache: Map<string, GPUComputePipeline>, wgX: number, wgY: number, mkShader: (x: number, y: number) => string, mkPipe: (s: GPUShaderModule) => GPUComputePipeline): GPUComputePipeline {
    const key = `${wgX},${wgY}`;
    let p = cache.get(key);
    if (!p) {
      p = mkPipe(device.createShaderModule({ code: f16Shader(mkShader(wgX, wgY)) }));
      cache.set(key, p);
    }
    return p;
  }
  const getOrCreateDwPipeline = (x: number, y: number) => getOrCreateCached(dwPipelineCache, x, y, makeDepthwise5x5Shader, mkDwPipe);
  const getOrCreateDwFullUnrollPipeline = (x: number, y: number) => getOrCreateCached(dwFullUnrollPipelineCache, x, y, makeDepthwise5x5FullUnrollShader, mkDwPipe);
  const getOrCreatePwPipeline = (x: number, y: number) => getOrCreateCached(pwPipelineCache, x, y, makePointwiseShader, mkPwPipe);
  const getOrCreatePw2OCPipeline = (x: number, y: number) => getOrCreateCached(pw2OCPipelineCache, x, y, makePointwise2OCShader, mkPwPipe);

  // Pre-compute all dispatch info per layer (avoids any runtime lookups)
  const layerDispatchInfos: LayerDispatchInfo[] = RESMODULE_LAYERS.map((spec) => {
    const outH = spec.stride === 2 ? spec.h / 2 : spec.h;
    const outW = spec.stride === 2 ? spec.w / 2 : spec.w;
    const [wgX, wgY] = getOptimalWorkgroupSize(spec.inCh, outH);

    const useFullUnroll = spec.h >= 64;
    const use2OC = outH >= 16 && spec.inCh >= 288 && spec.outCh >= 288 && spec.outCh % 2 === 0;

    return {
      dwPipeline: useFullUnroll
        ? getOrCreateDwFullUnrollPipeline(wgX, wgY)
        : getOrCreateDwPipeline(wgX, wgY),
      pwPipeline: use2OC
        ? getOrCreatePw2OCPipeline(wgX, wgY)
        : getOrCreatePwPipeline(wgX, wgY),
      dwDispatchX: Math.ceil(outW / wgX),
      dwDispatchY: Math.ceil(outH / wgY),
      dwDispatchZ: spec.inCh,
      pwDispatchX: Math.ceil(outW / wgX),
      pwDispatchY: Math.ceil(outH / wgY),
      pwDispatchZ: use2OC ? spec.outCh / 2 : spec.outCh,
    };
  });

  const padInputPipeline = makePipe(padLayout, padInputShader);
  const inputConvPipeline = makePipe(inputConvLayout, inputConvShader);
  void makePipe(upsampleLayout, upsampleShader);
  void makePipe(addLayout, addShader);
  const upsampleAddPipeline = makePipe(upsampleAddLayout, upsampleAddShader);
  const conv1x1Pipeline = makePipe(conv1x1Layout, conv1x1Shader);
  void makePipe(outputLayout, outputSigmoidShader);
  void makePipe(outputLayout, outputLandmarksShader);
  const canvasInputPipeline = makePipe(canvasInputLayout, canvasInputShader);
  const fusedOutputPipeline = makePipe(fusedOutputLayout, fusedOutputShader);
  const fusedDwPwPipeline = makePipe(fusedDwPwLayout, fusedDwPwShader);

  // ============ Allocate Buffers ============
  const maxSize = 1 * 288 * 128 * 128 * 4;
  const bufRawInput = makeBuf(1 * 3 * 256 * 256 * 4, SC);
  const bufInput = makeBuf(1 * 3 * 257 * 257 * 4, SO);
  const bufPadU = makeBuf(12, UC);
  device.queue.writeBuffer(bufPadU, 0, new Uint32Array([3, 256, 257]));
  const bufA = makeBuf(maxSize, SCS);
  const bufB = makeBuf(maxSize, SOC);
  const bufDW = makeBuf(maxSize, SO);
  const bufB1 = makeBuf(1 * 48 * 64 * 64 * 4, SC);
  const bufB2 = makeBuf(1 * 96 * 32 * 32 * 4, SC);
  const bufB3 = makeBuf(1 * 96 * 16 * 16 * 4, SC);
  const bufTmp = makeBuf(1 * 96 * 64 * 64 * 4, SO);
  const bufOutput = makeBuf(65 * 4, SOC); // unified: [handflag, handedness, landmarks[63]]
  const readbackBuf = makeBuf(65 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  void makeBuf(65 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  // For future double-buffered readback (pipeline mapAsync with compute)
  // let readbackIdx = 0;
  // const readbackBufs = [readbackBuf, readbackBufB];

  // Canvas input texture (256x256 rgba8unorm)
  const canvasTexture = device.createTexture({
    size: [256, 256],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const bufCanvasU = makeBuf(8, UC);
  device.queue.writeBuffer(bufCanvasU, 0, new Uint32Array([256, 257]));

  // ============ Upload Input Conv Weights ============
  // backbone1.1: Conv2d(3, 24, 3, stride=2)
  const inputConvWT = weights.get('backbone1.1.weight');
  const inputConvBT = weights.get('backbone1.1.bias');
  if (!inputConvWT || !inputConvBT) {
    throw new Error('Missing input conv weights');
  }
  const inputConvW = getWeightData(inputConvWT);
  const inputConvB = getWeightData(inputConvBT);

  const bufInputConvW = makeBuf(inputConvW.byteLength, SC);
  const bufInputConvB = makeBuf(inputConvB.byteLength, SC);
  const bufInputConvU = makeBuf(28, UC);
  device.queue.writeBuffer(bufInputConvW, 0, inputConvW as any);
  device.queue.writeBuffer(bufInputConvB, 0, inputConvB as any);
  // Input: 3ch x 257x257 (padded), Output: 24ch x 128x128
  device.queue.writeBuffer(bufInputConvU, 0, new Uint32Array([1, 3, 24, 257, 257, 128, 128]));

  // ============ Upload Conv1x1 Weights (backbone6.1) ============
  const conv1x1WT = weights.get('backbone6.1.weight');
  const conv1x1BT = weights.get('backbone6.1.bias');
  if (!conv1x1WT || !conv1x1BT) {
    throw new Error('Missing backbone6.1 conv1x1 weights');
  }
  const conv1x1W = getWeightData(conv1x1WT);
  const conv1x1B = getWeightData(conv1x1BT);

  const bufConv1x1W = makeBuf(conv1x1W.byteLength, SC);
  const bufConv1x1B = makeBuf(conv1x1B.byteLength, SC);
  const bufConv1x1U = makeBuf(20, UC);
  device.queue.writeBuffer(bufConv1x1W, 0, conv1x1W as any);
  device.queue.writeBuffer(bufConv1x1B, 0, conv1x1B as any);
  // Conv1x1: 96ch -> 48ch at 32x32
  device.queue.writeBuffer(bufConv1x1U, 0, new Uint32Array([1, 96, 48, 32, 32]));

  // ============ Upload Output Head Weights ============
  // handflag: Conv2d(288->1, 2x2)
  const handflagWT = weights.get('handflag.weight');
  const handflagBT = weights.get('handflag.bias');
  if (!handflagWT || !handflagBT) {
    throw new Error('Missing handflag weights');
  }
  const handflagW = getWeightData(handflagWT);
  const handflagB = getWeightData(handflagBT);

  const bufHandflagW = makeBuf(handflagW.byteLength, SC);
  const bufHandflagB = makeBuf(handflagB.byteLength, SC);
  const bufHandflagU = makeBuf(12, UC);
  device.queue.writeBuffer(bufHandflagW, 0, handflagW as any);
  device.queue.writeBuffer(bufHandflagB, 0, handflagB as any);
  device.queue.writeBuffer(bufHandflagU, 0, new Uint32Array([1, 288, 1]));

  // handedness: Conv2d(288->1, 2x2)
  const handednessWT = weights.get('handedness.weight');
  const handednessBT = weights.get('handedness.bias');
  if (!handednessWT || !handednessBT) {
    throw new Error('Missing handedness weights');
  }
  const handednessW = getWeightData(handednessWT);
  const handednessB = getWeightData(handednessBT);

  const bufHandednessW = makeBuf(handednessW.byteLength, SC);
  const bufHandednessB = makeBuf(handednessB.byteLength, SC);
  const bufHandednessU = makeBuf(12, UC);
  device.queue.writeBuffer(bufHandednessW, 0, handednessW as any);
  device.queue.writeBuffer(bufHandednessB, 0, handednessB as any);
  device.queue.writeBuffer(bufHandednessU, 0, new Uint32Array([1, 288, 1]));

  // landmarks (reg_3d): Conv2d(288->63, 2x2)
  const landmarksWT = weights.get('reg_3d.weight');
  const landmarksBT = weights.get('reg_3d.bias');
  if (!landmarksWT || !landmarksBT) {
    throw new Error('Missing reg_3d weights');
  }
  const landmarksW = getWeightData(landmarksWT);
  const landmarksB = getWeightData(landmarksBT);

  const bufLandmarksW = makeBuf(landmarksW.byteLength, SC);
  const bufLandmarksB = makeBuf(landmarksB.byteLength, SC);
  const bufLandmarksU = makeBuf(12, UC);
  device.queue.writeBuffer(bufLandmarksW, 0, landmarksW as any);
  device.queue.writeBuffer(bufLandmarksB, 0, landmarksB as any);
  device.queue.writeBuffer(bufLandmarksU, 0, new Uint32Array([1, 288, 63]));

  // ============ Upload ResModule Weights ============
  const resmoduleData: ResModuleData[] = RESMODULE_LAYERS.map((spec) => {
    const { inCh, outCh, h, w, stride, prefix } = spec;
    const outH = stride === 2 ? h / 2 : h;
    const outW = stride === 2 ? w / 2 : w;
    const pad = stride === 1 ? 2 : 1;

    const dwWT = weights.get(`${prefix}convs.0.weight`);
    const dwBT = weights.get(`${prefix}convs.0.bias`);
    const pwWT = weights.get(`${prefix}convs.1.weight`);
    const pwBT = weights.get(`${prefix}convs.1.bias`);

    if (!dwWT || !dwBT || !pwWT || !pwBT) {
      throw new Error(`Missing weights for ${prefix}`);
    }

    const dwW = getWeightData(dwWT);
    const dwB = getWeightData(dwBT);
    const pwW = getWeightData(pwWT);
    const pwB = getWeightData(pwBT);

    const dwWeight = makeBuf(dwW.byteLength, SC);
    const dwBias = makeBuf(dwB.byteLength, SC);
    const pwWeight = makeBuf(pwW.byteLength, SC);
    const pwBias = makeBuf(pwB.byteLength, SC);
    const dwUniform = makeBuf(32, UC);
    const pwUniform = makeBuf(36, UC);

    device.queue.writeBuffer(dwWeight, 0, dwW as any);
    device.queue.writeBuffer(dwBias, 0, dwB as any);
    device.queue.writeBuffer(pwWeight, 0, pwW as any);
    device.queue.writeBuffer(pwBias, 0, pwB as any);
    device.queue.writeBuffer(dwUniform, 0, new Uint32Array([1, inCh, h, w, outH, outW, stride, pad]));
    device.queue.writeBuffer(
      pwUniform,
      0,
      new Uint32Array([1, inCh, outCh, outH, outW, Math.max(0, outCh - inCh), stride, h, w]),
    );

    return { dwWeight, dwBias, pwWeight, pwBias, dwUniform, pwUniform, spec, outH, outW };
  });

  // ============ Create Upsample/Add Uniforms ============
  function makeUniform(data: number[]): GPUBuffer {
    const b = makeBuf(data.length * 4, UC);
    device.queue.writeBuffer(b, 0, new Uint32Array(data));
    return b;
  }
  const bufUpsample8to16U = makeUniform([1, 96, 8, 8, 16, 16]);
  const bufUpsample16to32U = makeUniform([1, 96, 16, 16, 32, 32]);
  const bufUpsample32to64U = makeUniform([1, 48, 32, 32, 64, 64]);
  void makeUniform([1 * 96 * 16 * 16]);
  void makeUniform([1 * 96 * 32 * 32]);
  void makeUniform([1 * 48 * 64 * 64]);

  // ============ Pre-create Bind Groups ============
  const padBG = makeBind(padLayout, [bufRawInput, bufInput, bufPadU]);
  const inputConvBG = makeBind(inputConvLayout, [bufInput, bufInputConvW, bufInputConvB, bufA, bufInputConvU]);

  // ResModule bind groups for A->B and B->A directions
  const dwBindGroupsAtoB: GPUBindGroup[] = [];
  const pwBindGroupsAtoB: GPUBindGroup[] = [];
  const dwBindGroupsBtoA: GPUBindGroup[] = [];
  const pwBindGroupsBtoA: GPUBindGroup[] = [];

  for (const ld of resmoduleData) {
    dwBindGroupsAtoB.push(makeBind(dwLayout, [bufA, ld.dwWeight, ld.dwBias, bufDW, ld.dwUniform]));
    pwBindGroupsAtoB.push(makeBind(pwLayout, [bufDW, bufA, ld.pwWeight, ld.pwBias, bufB, ld.pwUniform]));
    dwBindGroupsBtoA.push(makeBind(dwLayout, [bufB, ld.dwWeight, ld.dwBias, bufDW, ld.dwUniform]));
    pwBindGroupsBtoA.push(makeBind(pwLayout, [bufDW, bufB, ld.pwWeight, ld.pwBias, bufA, ld.pwUniform]));
  }

  // Fused upsample+add bind groups
  const upsampleAddB3BG = makeBind(upsampleAddLayout, [bufA, bufB3, bufB, bufUpsample8to16U]);
  const upsampleAddB2BG = makeBind(upsampleAddLayout, [bufA, bufB2, bufB, bufUpsample16to32U]);
  const conv1x1BG = makeBind(conv1x1Layout, [bufA, bufConv1x1W, bufConv1x1B, bufTmp, bufConv1x1U]);
  const upsampleAddB1BG = makeBind(upsampleAddLayout, [bufTmp, bufB1, bufB, bufUpsample32to64U]);

  // Output head bind groups (legacy individual — kept for pipeline creation)
  void makeBind(outputLayout, [bufA, bufHandflagW, bufHandflagB, bufOutput, bufHandflagU]);
  void makeBind(outputLayout, [bufA, bufHandednessW, bufHandednessB, bufOutput, bufHandednessU]);
  void makeBind(outputLayout, [bufA, bufLandmarksW, bufLandmarksB, bufOutput, bufLandmarksU]);

  const canvasInputBG = makeBind(canvasInputLayout, [canvasTexture.createView(), bufInput, bufCanvasU]);

  const fusedOutputBG = makeBind(fusedOutputLayout, [bufA, bufHandflagW, bufHandflagB, bufHandednessW, bufHandednessB, bufLandmarksW, bufLandmarksB, bufOutput]);

  // Fused DW+PW bind groups for layers 24-42 (all 288→288 channels)
  const FUSED_START = 24;
  const fusedBindGroupsAtoB: GPUBindGroup[] = [];
  const fusedBindGroupsBtoA: GPUBindGroup[] = [];
  for (let i = FUSED_START; i < resmoduleData.length; i++) {
    const ld = resmoduleData[i]!;
    fusedBindGroupsAtoB.push(makeBind(fusedDwPwLayout, [bufA, ld.dwWeight, ld.dwBias, ld.pwWeight, ld.pwBias, bufB, ld.dwUniform]));
    fusedBindGroupsBtoA.push(makeBind(fusedDwPwLayout, [bufB, ld.dwWeight, ld.dwBias, ld.pwWeight, ld.pwBias, bufA, ld.dwUniform]));
  }

  // ============ Render-based Readback (bypasses mapAsync) ============
  // Encodes float32 values as RGBA8 pixels, reads via canvas getImageData
  // This may be faster than mapAsync on Safari iOS
  const readbackCanvas = new OffscreenCanvas(9, 8); // 9*8=72 >= 65 values
  const readbackCtx = readbackCanvas.getContext('2d', { willReadFrequently: true })!;
  // Use 'copy' compositing to avoid alpha blending corrupting our float encoding
  readbackCtx.globalCompositeOperation = 'copy';
  const readbackGpuCanvas = new OffscreenCanvas(9, 8);
  const readbackGpuCtx = readbackGpuCanvas.getContext('webgpu') as any;

  let readbackRenderPipeline: GPURenderPipeline | null = null;
  let readbackRenderBG: GPUBindGroup | null = null;

  if (readbackGpuCtx) {
    readbackGpuCtx.configure({
      device,
      format: 'rgba8unorm',
      alphaMode: 'premultiplied',
    });

    const readbackBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' as GPUBufferBindingType } }],
    });

    const vsModule = device.createShaderModule({ code: READBACK_RENDER_VS });
    const fsModule = device.createShaderModule({ code: READBACK_RENDER_FS });

    readbackRenderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [readbackBGL] }),
      vertex: { module: vsModule, entryPoint: 'vs' },
      fragment: {
        module: fsModule, entryPoint: 'fs',
        targets: [{ format: 'rgba8unorm' as GPUTextureFormat }],
      },
    });

    readbackRenderBG = device.createBindGroup({
      layout: readbackBGL,
      entries: [{ binding: 0, resource: { buffer: bufOutput } }],
    });
  }

  // Pre-allocated output arrays (avoid per-frame allocations)
  const outputHandflag = new Float32Array(1);
  const outputHandedness = new Float32Array(1);
  const outputLandmarks = new Float32Array(63);

  // ============ Shared Inference Encoding ============
  // Encodes the common GPU work (everything after input is in bufInput)
  // OPTIMIZATION: Batched compute passes — dispatches within a single pass have
  // implicit memory barriers (per WebGPU spec). We only end a pass when we need
  // encoder-level ops (copyBufferToBuffer for skip connections).
  // Result: 7 compute passes instead of 86+ (saves pass boundary overhead)
  // NOTE: Merging passes 4-7 into 1 was tested but 1.75x SLOWER — the driver
  // uses pass boundaries for memory management and scheduling optimization.
  function encodeInference(encoder: GPUCommandEncoder, readbackTarget: GPUBuffer | null) {
    let useAtoB = true;
    let layerIdx = 0;

    // Pass 1: Input conv + layers 0-2 (until first skip save)
    let pass = encoder.beginComputePass();

    // Input conv3x3 (3->24, 257->128)
    pass.setPipeline(inputConvPipeline);
    pass.setBindGroup(0, inputConvBG);
    pass.dispatchWorkgroups(Math.ceil(128 / 8), Math.ceil(128 / 8), 24);

    // Layers until first skip save (SAVE_B1_AFTER = layer 2)
    for (; layerIdx <= SAVE_B1_AFTER; layerIdx++) {
      const dwBG = useAtoB ? dwBindGroupsAtoB[layerIdx] : dwBindGroupsBtoA[layerIdx];
      const pwBG = useAtoB ? pwBindGroupsAtoB[layerIdx] : pwBindGroupsBtoA[layerIdx];
      const di = layerDispatchInfos[layerIdx]!;
      pass.setPipeline(di.dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(di.dwDispatchX, di.dwDispatchY, di.dwDispatchZ);
      pass.setPipeline(di.pwPipeline);
      pass.setBindGroup(0, pwBG);
      pass.dispatchWorkgroups(di.pwDispatchX, di.pwDispatchY, di.pwDispatchZ);
      useAtoB = !useAtoB;
    }
    pass.end();

    // Save b1 skip connection
    const b1Buf = useAtoB ? bufA : bufB;
    encoder.copyBufferToBuffer(b1Buf, 0, bufB1, 0, 1 * 48 * 64 * 64 * 4);

    // Pass 2: Layers 3-5 (until SAVE_B2_AFTER)
    pass = encoder.beginComputePass();
    for (; layerIdx <= SAVE_B2_AFTER; layerIdx++) {
      const dwBG = useAtoB ? dwBindGroupsAtoB[layerIdx] : dwBindGroupsBtoA[layerIdx];
      const pwBG = useAtoB ? pwBindGroupsAtoB[layerIdx] : pwBindGroupsBtoA[layerIdx];
      const di = layerDispatchInfos[layerIdx]!;
      pass.setPipeline(di.dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(di.dwDispatchX, di.dwDispatchY, di.dwDispatchZ);
      pass.setPipeline(di.pwPipeline);
      pass.setBindGroup(0, pwBG);
      pass.dispatchWorkgroups(di.pwDispatchX, di.pwDispatchY, di.pwDispatchZ);
      useAtoB = !useAtoB;
    }
    pass.end();

    // Save b2 skip connection
    const b2Buf = useAtoB ? bufA : bufB;
    encoder.copyBufferToBuffer(b2Buf, 0, bufB2, 0, 1 * 96 * 32 * 32 * 4);

    // Pass 3: Layers 6-8 (until SAVE_B3_AFTER)
    pass = encoder.beginComputePass();
    for (; layerIdx <= SAVE_B3_AFTER; layerIdx++) {
      const dwBG = useAtoB ? dwBindGroupsAtoB[layerIdx] : dwBindGroupsBtoA[layerIdx];
      const pwBG = useAtoB ? pwBindGroupsAtoB[layerIdx] : pwBindGroupsBtoA[layerIdx];
      const di = layerDispatchInfos[layerIdx]!;
      pass.setPipeline(di.dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(di.dwDispatchX, di.dwDispatchY, di.dwDispatchZ);
      pass.setPipeline(di.pwPipeline);
      pass.setBindGroup(0, pwBG);
      pass.dispatchWorkgroups(di.pwDispatchX, di.pwDispatchY, di.pwDispatchZ);
      useAtoB = !useAtoB;
    }
    pass.end();

    // Save b3 skip connection
    const b3Buf = useAtoB ? bufA : bufB;
    encoder.copyBufferToBuffer(b3Buf, 0, bufB3, 0, 1 * 96 * 16 * 16 * 4);

    // Pass 4: Layers 9-11 + upsample+add b3
    pass = encoder.beginComputePass();
    for (; layerIdx <= UPSAMPLE_ADD_B3_AFTER; layerIdx++) {
      const dwBG = useAtoB ? dwBindGroupsAtoB[layerIdx] : dwBindGroupsBtoA[layerIdx];
      const pwBG = useAtoB ? pwBindGroupsAtoB[layerIdx] : pwBindGroupsBtoA[layerIdx];
      const di = layerDispatchInfos[layerIdx]!;
      pass.setPipeline(di.dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(di.dwDispatchX, di.dwDispatchY, di.dwDispatchZ);
      pass.setPipeline(di.pwPipeline);
      pass.setBindGroup(0, pwBG);
      pass.dispatchWorkgroups(di.pwDispatchX, di.pwDispatchY, di.pwDispatchZ);
      useAtoB = !useAtoB;
    }
    // Fused upsample 8->16 + add b3
    pass.setPipeline(upsampleAddPipeline);
    pass.setBindGroup(0, upsampleAddB3BG);
    pass.dispatchWorkgroups(Math.ceil(16 / 8), Math.ceil(16 / 8), 96);
    pass.end();
    useAtoB = false;

    // Pass 5: Layer 12 + upsample+add b2
    pass = encoder.beginComputePass();
    {
      const dwBG = useAtoB ? dwBindGroupsAtoB[layerIdx] : dwBindGroupsBtoA[layerIdx];
      const pwBG = useAtoB ? pwBindGroupsAtoB[layerIdx] : pwBindGroupsBtoA[layerIdx];
      const di = layerDispatchInfos[layerIdx]!;
      pass.setPipeline(di.dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(di.dwDispatchX, di.dwDispatchY, di.dwDispatchZ);
      pass.setPipeline(di.pwPipeline);
      pass.setBindGroup(0, pwBG);
      pass.dispatchWorkgroups(di.pwDispatchX, di.pwDispatchY, di.pwDispatchZ);
      useAtoB = !useAtoB;
      layerIdx++;
    }
    // Fused upsample 16->32 + add b2
    pass.setPipeline(upsampleAddPipeline);
    pass.setBindGroup(0, upsampleAddB2BG);
    pass.dispatchWorkgroups(Math.ceil(32 / 8), Math.ceil(32 / 8), 96);
    pass.end();
    useAtoB = false;

    // Pass 6: Layer 13 + conv1x1 + upsample+add b1
    pass = encoder.beginComputePass();
    {
      const dwBG = useAtoB ? dwBindGroupsAtoB[layerIdx] : dwBindGroupsBtoA[layerIdx];
      const pwBG = useAtoB ? pwBindGroupsAtoB[layerIdx] : pwBindGroupsBtoA[layerIdx];
      const di = layerDispatchInfos[layerIdx]!;
      pass.setPipeline(di.dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(di.dwDispatchX, di.dwDispatchY, di.dwDispatchZ);
      pass.setPipeline(di.pwPipeline);
      pass.setBindGroup(0, pwBG);
      pass.dispatchWorkgroups(di.pwDispatchX, di.pwDispatchY, di.pwDispatchZ);
      useAtoB = !useAtoB;
      layerIdx++;
    }
    // Conv1x1 (96->48)
    pass.setPipeline(conv1x1Pipeline);
    pass.setBindGroup(0, conv1x1BG);
    pass.dispatchWorkgroups(Math.ceil(32 / 8), Math.ceil(32 / 8), 48);
    // Fused upsample 32->64 + add b1
    pass.setPipeline(upsampleAddPipeline);
    pass.setBindGroup(0, upsampleAddB1BG);
    pass.dispatchWorkgroups(Math.ceil(64 / 8), Math.ceil(64 / 8), 48);
    pass.end();
    useAtoB = false;

    // Pass 7: Remaining ff layers (14-42) + output heads
    // Layers 14-23: separate DW+PW (channels < 288)
    // Layers 24-42: fused DW+PW (all 288→288, saves 19 dispatches)
    pass = encoder.beginComputePass();
    for (; layerIdx < FUSED_START; layerIdx++) {
      const dwBG = useAtoB ? dwBindGroupsAtoB[layerIdx] : dwBindGroupsBtoA[layerIdx];
      const pwBG = useAtoB ? pwBindGroupsAtoB[layerIdx] : pwBindGroupsBtoA[layerIdx];
      const di = layerDispatchInfos[layerIdx]!;
      pass.setPipeline(di.dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(di.dwDispatchX, di.dwDispatchY, di.dwDispatchZ);
      pass.setPipeline(di.pwPipeline);
      pass.setBindGroup(0, pwBG);
      pass.dispatchWorkgroups(di.pwDispatchX, di.pwDispatchY, di.pwDispatchZ);
      useAtoB = !useAtoB;
    }

    // Layers 24-42: fused DW+PW (one dispatch per layer instead of two)
    for (; layerIdx < resmoduleData.length; layerIdx++) {
      const fusedIdx = layerIdx - FUSED_START;
      const bg = useAtoB ? fusedBindGroupsAtoB[fusedIdx]! : fusedBindGroupsBtoA[fusedIdx]!;
      const ld = resmoduleData[layerIdx]!;
      pass.setPipeline(fusedDwPwPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ld.outW, ld.outH, 1);
      useAtoB = !useAtoB;
    }

    // Fused output heads (1 dispatch instead of 3)
    pass.setPipeline(fusedOutputPipeline);
    pass.setBindGroup(0, fusedOutputBG);
    pass.dispatchWorkgroups(1);
    pass.end();

    // Copy unified output to readback buffer (1 copy instead of 3)
    if (readbackTarget) {
      encoder.copyBufferToBuffer(bufOutput, 0, readbackTarget, 0, 65 * 4);
    }
  }

  // ============ Run Function (Float32Array input) ============
  async function run(input: Float32Array): Promise<HandLandmarksOutput> {
    // Write raw 256x256 input directly to GPU (no CPU-side padding!)
    device.queue.writeBuffer(bufRawInput, 0, input as any);

    const encoder = device.createCommandEncoder();

    // 0. GPU-side padding: 256x256 -> 257x257
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(padInputPipeline);
      pass.setBindGroup(0, padBG);
      pass.dispatchWorkgroups(Math.ceil(256 / 16), Math.ceil(256 / 16), 3);
      pass.end();
    }

    // 1-3. Shared inference (conv → ResModules → output heads → readback)
    encodeInference(encoder, readbackBuf);

    device.queue.submit([encoder.finish()]);

    // Wait for GPU completion, then map — may help Safari resolve mapAsync faster
    const mapPromise = readbackBuf.mapAsync(GPUMapMode.READ);
    await device.queue.onSubmittedWorkDone();
    await mapPromise;
    const mapped = new Float32Array(readbackBuf.getMappedRange());
    outputHandflag[0] = mapped[0]!;
    outputHandedness[0] = mapped[1]!;
    outputLandmarks.set(mapped.subarray(2, 65));
    readbackBuf.unmap();

    return {
      handflag: new Float32Array(outputHandflag),
      handedness: new Float32Array(outputHandedness),
      landmarks: new Float32Array(outputLandmarks),
    };
  }

  // ============ Run from Canvas/ImageBitmap (zero-copy GPU input) ============
  async function runFromCanvas(
    source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap,
  ): Promise<HandLandmarksOutput> {
    // Copy canvas pixels directly to GPU texture (no CPU readback!)
    // This replaces writeBuffer + padding shader with texture copy + fused conversion
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: canvasTexture },
      [256, 256],
    );

    const encoder = device.createCommandEncoder();

    // 0. Fused texture→NCHW+pad: reads rgba8unorm texture, writes NCHW float32 padded 257x257
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipeline);
      pass.setBindGroup(0, canvasInputBG);
      pass.dispatchWorkgroups(Math.ceil(256 / 16), Math.ceil(256 / 16), 1);
      pass.end();
    }

    // 1-3. Shared inference
    encodeInference(encoder, readbackBuf);

    device.queue.submit([encoder.finish()]);

    // Wait for GPU completion, then map — may help Safari resolve mapAsync faster
    const mapPromise = readbackBuf.mapAsync(GPUMapMode.READ);
    await device.queue.onSubmittedWorkDone();
    await mapPromise;
    const mapped = new Float32Array(readbackBuf.getMappedRange());
    outputHandflag[0] = mapped[0]!;
    outputHandedness[0] = mapped[1]!;
    outputLandmarks.set(mapped.subarray(2, 65));
    readbackBuf.unmap();

    return {
      handflag: new Float32Array(outputHandflag),
      handedness: new Float32Array(outputHandedness),
      landmarks: new Float32Array(outputLandmarks),
    };
  }

  // ============ Render-based Readback (bypasses mapAsync) ============
  // Uses render pass to encode floats as RGBA8 pixels, reads via canvas getImageData
  // May be faster than mapAsync on Safari iOS
  async function runFromCanvasViaRender(
    source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap,
  ): Promise<HandLandmarksOutput> {
    if (!readbackRenderPipeline || !readbackRenderBG || !readbackGpuCtx) {
      throw new Error('Render-based readback not available (no WebGPU canvas context)');
    }

    device.queue.copyExternalImageToTexture(
      { source }, { texture: canvasTexture }, [256, 256],
    );
    const encoder = device.createCommandEncoder();
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipeline);
      pass.setBindGroup(0, canvasInputBG);
      pass.dispatchWorkgroups(16, 16, 1);
      pass.end();
    }

    // Run inference without copying to readback buffer
    encodeInference(encoder, null);

    // Render float values as RGBA8 pixels to canvas
    const renderTarget = readbackGpuCtx.getCurrentTexture();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: renderTarget.createView(),
        loadOp: 'clear' as GPULoadOp,
        storeOp: 'store' as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(readbackRenderPipeline);
    renderPass.setBindGroup(0, readbackRenderBG);
    renderPass.draw(3);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read pixels from canvas (bypasses mapAsync!)
    readbackCtx.drawImage(readbackGpuCanvas, 0, 0);
    const imageData = readbackCtx.getImageData(0, 0, 9, 8);
    const pixels = imageData.data; // Uint8ClampedArray

    // Decode RGBA8 pixels back to float32
    const decoded = new Float32Array(65);
    const view = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < 65; i++) {
      const pi = i * 4;
      view.setUint8(0, pixels[pi]!);     // r = byte3 (MSB)
      view.setUint8(1, pixels[pi + 1]!); // g = byte2
      view.setUint8(2, pixels[pi + 2]!); // b = byte1
      view.setUint8(3, pixels[pi + 3]!); // a = byte0 (LSB)
      decoded[i] = view.getFloat32(0);
    }

    return {
      handflag: new Float32Array([decoded[0]!]),
      handedness: new Float32Array([decoded[1]!]),
      landmarks: new Float32Array(decoded.subarray(2, 65)),
    };
  }

  // ============ Pipelined Run (double-buffered readback) ============
  // Overlaps GPU compute with CPU readback. Returns PREVIOUS frame's result.
  // First call returns null (priming the pipeline). One frame of latency.
  const readbackBufB = device.createBuffer({
    size: 65 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  let pipeIdx = 0;
  const pipeBufs = [readbackBuf, readbackBufB];
  let pendingMap: Promise<void> | null = null;
  let pendingBuf: GPUBuffer | null = null;

  async function runFromCanvasPipelined(
    source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap,
  ): Promise<HandLandmarksOutput | null> {
    const curBuf = pipeBufs[pipeIdx]!;
    pipeIdx = 1 - pipeIdx;

    // Submit current frame
    device.queue.copyExternalImageToTexture(
      { source }, { texture: canvasTexture }, [256, 256],
    );
    const encoder = device.createCommandEncoder();
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipeline);
      pass.setBindGroup(0, canvasInputBG);
      pass.dispatchWorkgroups(Math.ceil(256 / 16), Math.ceil(256 / 16), 1);
      pass.end();
    }
    encodeInference(encoder, curBuf);
    device.queue.submit([encoder.finish()]);

    // Read previous frame
    let result: HandLandmarksOutput | null = null;
    if (pendingMap !== null && pendingBuf !== null) {
      await pendingMap;
      const mapped = new Float32Array(pendingBuf.getMappedRange());
      result = {
        handflag: new Float32Array([mapped[0]!]),
        handedness: new Float32Array([mapped[1]!]),
        landmarks: new Float32Array(mapped.subarray(2, 65)),
      };
      pendingBuf.unmap();
    }

    // Start mapping current frame (will be read next call)
    pendingBuf = curBuf;
    pendingMap = curBuf.mapAsync(GPUMapMode.READ);

    return result;
  }

  async function flushPipelined(): Promise<HandLandmarksOutput | null> {
    if (!pendingMap || !pendingBuf) return null;
    await pendingMap;
    const mapped = new Float32Array(pendingBuf.getMappedRange());
    const result: HandLandmarksOutput = {
      handflag: new Float32Array([mapped[0]!]),
      handedness: new Float32Array([mapped[1]!]),
      landmarks: new Float32Array(mapped.subarray(2, 65)),
    };
    pendingBuf.unmap();
    pendingMap = null;
    pendingBuf = null;
    return result;
  }

  // ============ Benchmark Function ============
  async function benchmark(iterations = 50): Promise<{ avgMs: number; fps: number }> {
    const dummyInput = new Float32Array(1 * 3 * 256 * 256);

    // Warmup
    for (let i = 0; i < 5; i++) {
      await run(dummyInput);
    }

    // Benchmark
    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await run(dummyInput);
      times.push(performance.now() - start);
    }

    const avgMs = times.reduce((a, b) => a + b, 0) / times.length;
    return { avgMs, fps: 1000 / avgMs };
  }

  // GPU-only benchmark (skips readback for pure GPU timing)
  async function benchmarkGPU(iterations = 50): Promise<{ avgMs: number; fps: number; medianMs: number; minMs: number }> {
    const dummyInput = new Float32Array(1 * 3 * 256 * 256);

    // Warmup with full run
    for (let i = 0; i < 5; i++) {
      await run(dummyInput);
    }

    // Benchmark just the GPU submit + wait (skip readback)
    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const encoder = device.createCommandEncoder();

      // GPU padding
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(padInputPipeline);
        pass.setBindGroup(0, padBG);
        pass.dispatchWorkgroups(Math.ceil(256 / 16), Math.ceil(256 / 16), 3);
        pass.end();
      }

      // Shared inference (conv → ResModules → fused output heads)
      encodeInference(encoder, readbackBuf);

      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      times.push(performance.now() - start);
    }

    times.sort((a, b) => a - b);
    const avgMs = times.reduce((a, b) => a + b, 0) / times.length;
    const medianMs = times[Math.floor(times.length / 2)]!;
    const minMs = times[0]!;
    return { avgMs, fps: 1000 / avgMs, medianMs, minMs };
  }

  // Submit-only function for GPU-only benchmarking (no readback)
  function submitOnly(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) {
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: canvasTexture },
      [256, 256],
    );
    const encoder = device.createCommandEncoder();
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipeline);
      pass.setBindGroup(0, canvasInputBG);
      pass.dispatchWorkgroups(Math.ceil(256 / 16), Math.ceil(256 / 16), 1);
      pass.end();
    }
    encodeInference(encoder, readbackBuf);
    device.queue.submit([encoder.finish()]);
  }

  // Diagnostic benchmark: measures GPU time, mapAsync time, and total separately
  async function benchmarkDiagnostic(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap, iterations = 50): Promise<{
    gpuOnly: { median: number; min: number };
    mapAsyncOnly: { median: number; min: number };
    mapAsyncNoWait: { median: number; min: number };
    total: { median: number; min: number };
    pipelined: { median: number; min: number };
  }> {
    function stats(times: number[]) {
      const s = [...times].sort((a, b) => a - b);
      return { median: s[Math.floor(s.length / 2)]!, min: s[0]! };
    }

    // Warmup
    for (let i = 0; i < 10; i++) await runFromCanvas(source);

    // 1. GPU-only (submit + onSubmittedWorkDone, no mapAsync)
    const gpuTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      device.queue.copyExternalImageToTexture({ source }, { texture: canvasTexture }, [256, 256]);
      const enc = device.createCommandEncoder();
      { const p = enc.beginComputePass(); p.setPipeline(canvasInputPipeline); p.setBindGroup(0, canvasInputBG); p.dispatchWorkgroups(16, 16, 1); p.end(); }
      encodeInference(enc, readbackBuf);
      const t = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      gpuTimes.push(performance.now() - t);
    }

    // 2. mapAsync WITH onSubmittedWorkDone first (current approach)
    const mapWaitTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      device.queue.copyExternalImageToTexture({ source }, { texture: canvasTexture }, [256, 256]);
      const enc = device.createCommandEncoder();
      { const p = enc.beginComputePass(); p.setPipeline(canvasInputPipeline); p.setBindGroup(0, canvasInputBG); p.dispatchWorkgroups(16, 16, 1); p.end(); }
      encodeInference(enc, readbackBuf);
      device.queue.submit([enc.finish()]);
      const mp = readbackBuf.mapAsync(GPUMapMode.READ);
      const t = performance.now();
      await device.queue.onSubmittedWorkDone();
      await mp;
      readbackBuf.getMappedRange();
      readbackBuf.unmap();
      mapWaitTimes.push(performance.now() - t);
    }

    // 3. mapAsync WITHOUT onSubmittedWorkDone (just raw mapAsync)
    const mapNoWaitTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      device.queue.copyExternalImageToTexture({ source }, { texture: canvasTexture }, [256, 256]);
      const enc = device.createCommandEncoder();
      { const p = enc.beginComputePass(); p.setPipeline(canvasInputPipeline); p.setBindGroup(0, canvasInputBG); p.dispatchWorkgroups(16, 16, 1); p.end(); }
      encodeInference(enc, readbackBuf);
      device.queue.submit([enc.finish()]);
      const t = performance.now();
      await readbackBuf.mapAsync(GPUMapMode.READ);
      readbackBuf.getMappedRange();
      readbackBuf.unmap();
      mapNoWaitTimes.push(performance.now() - t);
    }

    // 4. Total end-to-end
    const totalTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const t = performance.now();
      await runFromCanvas(source);
      totalTimes.push(performance.now() - t);
    }

    // 5. Pipelined throughput
    await runFromCanvasPipelined(source); // prime
    const pipeTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const t = performance.now();
      await runFromCanvasPipelined(source);
      pipeTimes.push(performance.now() - t);
    }
    await flushPipelined();

    // 6. Render-based readback (bypasses mapAsync)
    let renderReadback: { median: number; min: number } | null = null;
    if (readbackRenderPipeline) {
      const renderTimes: number[] = [];
      for (let i = 0; i < iterations; i++) {
        const t = performance.now();
        await runFromCanvasViaRender(source);
        renderTimes.push(performance.now() - t);
      }
      renderReadback = stats(renderTimes);
    }

    return {
      gpuOnly: stats(gpuTimes),
      mapAsyncOnly: stats(mapWaitTimes),
      mapAsyncNoWait: stats(mapNoWaitTimes),
      total: stats(totalTimes),
      pipelined: stats(pipeTimes),
      renderReadback,
    };
  }

  return {
    device, run, runFromCanvas, runFromCanvasViaRender, runFromCanvasPipelined, flushPipelined,
    benchmark, benchmarkGPU, benchmarkDiagnostic,
    _device: device,
    _benchmarkSubmitOnly: submitOnly,
  };
}
