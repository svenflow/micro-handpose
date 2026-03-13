/**
 * Palm Detection WebGPU Model
 *
 * BlazeNet backbone with PReLU activations, FPN, and SSD output heads.
 *
 * Architecture:
 * 1. Initial conv 5x5 stride-2 + PReLU → 96x96x32
 * 2. Stage 1: 4 blocks (32ch), stride-2 transition → 48x48x64 (save backbone2 skip)
 * 3. Stage 2: 4 blocks (64ch), stride-2 transition → 24x24x128 (save backbone3 skip)
 * 4. Stage 3: 4 blocks (128ch), stride-2 transition → 12x12x256
 * 5. Stage 4: 8 blocks (256ch) at 12x12
 * 6. FPN: conv1x1 + upsample 12→24 + add backbone3; 2 blocks at 24x24
 * 7. SSD heads:
 *    - 12x12: 6 classifiers + 108 regressors (6 anchors × 18 values)
 *    - 24x24: 2 classifiers + 36 regressors (2 anchors × 18 values)
 *
 * Output: 2016 anchors total (864 from 12x12 + 1152 from 24x24)
 *   Per anchor: 1 score + 18 regression values
 */

import {
  PALM_CONV5X5_STRIDE2_PRELU_SHADER,
  PALM_DEPTHWISE_5X5_SHADER,
  PALM_POINTWISE_SKIP_PRELU_SHADER,
  PALM_CONV1X1_SHADER,
  PALM_UPSAMPLE_2X_ADD_SHADER,
  PALM_CONV1X1_PRELU_SHADER,
  PALM_CANVAS_INPUT_SHADER,
} from './palm_shaders.js';
import type { Tensor, WeightsMetadata } from './model.js';

export interface PalmDetectionOutput {
  scores: Float32Array;      // [2016] raw classifier logits
  regressors: Float32Array;  // [2016 * 18] raw regressor outputs
}

export interface CompiledPalmModel {
  device: GPUDevice;
  run: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<PalmDetectionOutput>;
  debugRun: (source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) => Promise<any>;
}

// Weight key helper: find key by substring
function findKey(keys: string[], ...substrings: string[]): number {
  const idx = keys.findIndex(k => substrings.every(s => k.includes(s)));
  if (idx === -1) throw new Error(`Weight key not found for: ${substrings.join(', ')}`);
  return idx;
}

/**
 * Palm detection model layer structure:
 *
 * conv2d/Conv2D [32,5,5,3] + batch_normalization/bias [32] + p_re_lu/alpha [1,1,32]
 * Then 18 depthwise-separable blocks:
 *   depthwise_conv2d_N + conv2d_N+1 + batch_normalization_N+1 + p_re_lu_N+1
 *
 * Blocks 0-3: 32ch (stage 1), block 3 is stride-2 transition to 64ch
 * Blocks 4-7: 64ch (stage 2), block 7 is stride-2 transition to 128ch
 * Blocks 8-11: 128ch (stage 3), block 11 is stride-2 transition to 256ch
 * Blocks 12-19: 256ch (stage 4)
 *
 * FPN path (from stage 4 output):
 *   conv2d_20 (256→256) projects stage4
 *   upsample 12→24, add with backbone3 skip (conv2d_22 projects backbone3 output)
 *   Two more dw+pw blocks (dw_19/conv2d_21, dw_20/conv2d_22 path)
 *
 * Actually looking at the manifest more carefully:
 * conv2d_20: [256,1,1,256] FPN project at 12x12 (with bn_20 bias, prelu_20 alpha)
 * conv2d_22/Conv2D1: [256,1,1,256] backbone3 skip projection
 * conv2d_21: [256,1,1,256] after first FPN upsample block
 * dw_19, dw_20: FPN blocks
 *
 * For 24x24 head path:
 * conv2d_23: [128,1,1,256] project to 128ch
 * dw_21 + conv2d_24: block at 24x24 128ch
 * dw_22 + conv2d_25/Conv2D1: block at 24x24 128ch (conv2d_25/Conv2D1 is skip project)
 *
 * SSD heads:
 * classifier_palm_16: [6,1,1,256] at 12x12
 * regressor_palm_16: [108,1,1,256] at 12x12
 * classifier_palm_8: [2,1,1,128] at 24x24
 * regressor_palm_8: [36,1,1,128] at 24x24
 */

// Block specification for the backbone
interface BlockSpec {
  dwIdx: number;      // depthwise_conv2d index
  pwIdx: number;      // conv2d index for pointwise
  inCh: number;
  outCh: number;
  stride: 1 | 2;
  inH: number;        // input spatial size
}

export async function compilePalmModel(
  weights: Map<string, Tensor>,
  existingDevice?: GPUDevice,
): Promise<CompiledPalmModel> {
  // Use existing device or create new one
  let device: GPUDevice;
  if (existingDevice) {
    device = existingDevice;
  } else {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No GPU adapter found');
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 8),
      },
    });
  }

  // Helpers
  const BT: Record<string, GPUBufferBindingType> = { r: 'read-only-storage', s: 'storage', u: 'uniform' };
  function makeLayout(types: string[]): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      entries: types.map((t, i) => {
        if (t === 't') return { binding: i, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' as const } };
        return { binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: BT[t]! } };
      }),
    });
  }
  const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const SO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const SOC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const UC = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

  function makeBuf(size: number, usage: number): GPUBuffer {
    return device.createBuffer({ size: Math.max(size, 4), usage });
  }
  // writeBuffer wrapper to avoid @webgpu/types incompatibility with Float32Array
  function writeBuf(buf: GPUBuffer, offset: number, data: ArrayBufferView | ArrayBuffer): void {
    device.queue.writeBuffer(buf, offset, data as unknown as ArrayBuffer);
  }
  function uploadWeights(tensor: Tensor): GPUBuffer {
    const buf = makeBuf(tensor.data.byteLength, SC);
    writeBuf(buf, 0, tensor.data);
    return buf;
  }

  // ============ Parse weight manifest ============
  // Build a key index for quick lookup
  const keyList = Array.from(weights.keys());

  function getWeight(key: string): Tensor {
    const t = weights.get(key);
    if (!t) throw new Error(`Weight not found: ${key}`);
    return t;
  }

  function findWeight(...substrings: string[]): Tensor {
    const key = keyList.find(k => substrings.every(s => k.includes(s)));
    if (!key) throw new Error(`Weight not found for: ${substrings.join(', ')}`);
    return getWeight(key);
  }

  // ============ Weight transposition helpers ============
  // TFLite stores conv weights as [outCh, kH, kW, inCh] — already correct for our shaders
  // TFLite stores depthwise weights as [1, kH, kW, channels]
  // We need depthwise as [channels, 25] (channels groups of 25 weights)

  function transposeDW(tensor: Tensor): Float32Array {
    // Input: [1, 5, 5, channels], Output: [channels, 25]
    const [, kH, kW, ch] = tensor.shape;
    const result = new Float32Array(ch * 25);
    for (let c = 0; c < ch; c++) {
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) {
          result[c * 25 + ky * 5 + kx] = tensor.data[ky * kW * ch + kx * ch + c];
        }
      }
    }
    return result;
  }

  function transposePW(tensor: Tensor): Float32Array {
    // Input: [outCh, 1, 1, inCh], Output: [outCh, inCh]
    const [outCh, , , inCh] = tensor.shape;
    const result = new Float32Array(outCh * inCh);
    for (let oc = 0; oc < outCh; oc++) {
      for (let ic = 0; ic < inCh; ic++) {
        result[oc * inCh + ic] = tensor.data[oc * inCh + ic];
      }
    }
    return result; // Already in [outCh, inCh] layout
  }

  // ============ Create Shader Modules ============
  const inputConvMod = device.createShaderModule({ code: PALM_CONV5X5_STRIDE2_PRELU_SHADER });
  const dwMod = device.createShaderModule({ code: PALM_DEPTHWISE_5X5_SHADER });
  const pwPreluMod = device.createShaderModule({ code: PALM_POINTWISE_SKIP_PRELU_SHADER });
  const conv1x1Mod = device.createShaderModule({ code: PALM_CONV1X1_SHADER });
  const conv1x1PreluMod = device.createShaderModule({ code: PALM_CONV1X1_PRELU_SHADER });
  const upsampleAddMod = device.createShaderModule({ code: PALM_UPSAMPLE_2X_ADD_SHADER });
  const canvasInputMod = device.createShaderModule({ code: PALM_CANVAS_INPUT_SHADER });

  // ============ Create Layouts ============
  const inputConvLayout = makeLayout(['r', 'r', 'r', 'r', 's', 'u']);  // input, weight, bias, alpha, output, params
  const dwLayout = makeLayout(['r', 'r', 'r', 's', 'u']);              // input, weight, bias, output, params
  const pwPreluLayout = makeLayout(['r', 'r', 'r', 'r', 'r', 's', 'u']); // dw_out, skip, pw_w, pw_b, alpha, output, params
  const conv1x1Layout = makeLayout(['r', 'r', 'r', 's', 'u']);         // input, weight, bias, output, params
  const conv1x1PreluLayout = makeLayout(['r', 'r', 'r', 'r', 's', 'u']); // input, weight, bias, alpha, output, params
  const upsampleAddLayout = makeLayout(['r', 'r', 's', 'u']);          // input, skip, output, params
  const canvasInputLayout = makeLayout(['t', 's', 'u']);

  // ============ Create Pipelines ============
  function makePipe(layout: GPUBindGroupLayout, mod: GPUShaderModule): GPUComputePipeline {
    return device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: mod, entryPoint: 'main' },
    });
  }

  const inputConvPipe = makePipe(inputConvLayout, inputConvMod);
  const dwPipe = makePipe(dwLayout, dwMod);
  const pwPreluPipe = makePipe(pwPreluLayout, pwPreluMod);
  const conv1x1Pipe = makePipe(conv1x1Layout, conv1x1Mod);
  const conv1x1PreluPipe = makePipe(conv1x1PreluLayout, conv1x1PreluMod);
  const upsampleAddPipe = makePipe(upsampleAddLayout, upsampleAddMod);
  const canvasInputPipe = makePipe(canvasInputLayout, canvasInputMod);

  // ============ Load and upload weights ============

  // Initial conv: conv2d/Conv2D [32,5,5,3]
  const initConvW = findWeight('conv2d/Conv2D');
  // Bias comes from fused batchnorm: batch_normalization/FusedBatchNormV3
  const initConvB = findWeight('batch_normalization/', 'conv2d/Conv2D');
  const initConvAlpha = findWeight('p_re_lu/');

  const initConvWeightBuf = uploadWeights(initConvW);
  const initConvBiasBuf = uploadWeights(initConvB);
  // Alpha shape is [1,1,C], flatten to [C]
  const initConvAlphaBuf = uploadWeights(initConvAlpha);

  // ============ Define backbone blocks ============
  // Each block: depthwise_conv2d_N + conv2d_M + batch_normalization_M + p_re_lu_M
  //
  // The weight naming pattern:
  // - depthwise_conv2d_N/depthwise (or depthwise1 for stride-2 transitions)
  // - conv2d_M/Conv2D (pointwise weights)
  // - batch_normalization_M/... (fused bias for conv2d_M)
  // - p_re_lu_M/... (PReLU alpha for conv2d_M)

  // Build the 20 backbone blocks explicitly
  interface BackboneBlock {
    dwWeightBuf: GPUBuffer;
    dwBiasBuf: GPUBuffer;  // from depthwise batchnorm — but BlazeNet folds BN into pw, not dw!
    pwWeightBuf: GPUBuffer;
    pwBiasBuf: GPUBuffer;
    alphaBuf: GPUBuffer;
    inCh: number;
    outCh: number;
    stride: 1 | 2;
    inH: number;
  }

  // Map out the layers based on the weight manifest analysis
  const blockDefs: Array<{
    dwKey: string;
    pwKey: string;
    bnKey: string;
    preluKey: string;
    inCh: number;
    outCh: number;
    stride: 1 | 2;
    inH: number;
  }> = [
    // Stage 1: 32ch blocks, 96x96 input
    { dwKey: 'depthwise_conv2d/', pwKey: 'conv2d_1/', bnKey: 'batch_normalization_1/', preluKey: 'p_re_lu_1/', inCh: 32, outCh: 32, stride: 1, inH: 96 },
    { dwKey: 'depthwise_conv2d_1/', pwKey: 'conv2d_2/', bnKey: 'batch_normalization_2/', preluKey: 'p_re_lu_2/', inCh: 32, outCh: 32, stride: 1, inH: 96 },
    { dwKey: 'depthwise_conv2d_2/', pwKey: 'conv2d_3/', bnKey: 'batch_normalization_3/', preluKey: 'p_re_lu_3/', inCh: 32, outCh: 32, stride: 1, inH: 96 },
    { dwKey: 'depthwise_conv2d_3/', pwKey: 'conv2d_4/', bnKey: 'batch_normalization_4/', preluKey: 'p_re_lu_4/', inCh: 32, outCh: 64, stride: 2, inH: 96 },
    // Stage 2: 64ch blocks, 48x48 input
    { dwKey: 'depthwise_conv2d_4/', pwKey: 'conv2d_5/', bnKey: 'batch_normalization_5/', preluKey: 'p_re_lu_5/', inCh: 64, outCh: 64, stride: 1, inH: 48 },
    { dwKey: 'depthwise_conv2d_5/', pwKey: 'conv2d_6/', bnKey: 'batch_normalization_6/', preluKey: 'p_re_lu_6/', inCh: 64, outCh: 64, stride: 1, inH: 48 },
    { dwKey: 'depthwise_conv2d_6/', pwKey: 'conv2d_7/', bnKey: 'batch_normalization_7/', preluKey: 'p_re_lu_7/', inCh: 64, outCh: 64, stride: 1, inH: 48 },
    { dwKey: 'depthwise_conv2d_7/', pwKey: 'conv2d_8/', bnKey: 'batch_normalization_8/', preluKey: 'p_re_lu_8/', inCh: 64, outCh: 128, stride: 2, inH: 48 },
    // Stage 3: 128ch blocks, 24x24 input
    { dwKey: 'depthwise_conv2d_8/', pwKey: 'conv2d_9/', bnKey: 'batch_normalization_9/', preluKey: 'p_re_lu_9/', inCh: 128, outCh: 128, stride: 1, inH: 24 },
    { dwKey: 'depthwise_conv2d_9/', pwKey: 'conv2d_10/', bnKey: 'batch_normalization_10/', preluKey: 'p_re_lu_10/', inCh: 128, outCh: 128, stride: 1, inH: 24 },
    { dwKey: 'depthwise_conv2d_10/', pwKey: 'conv2d_11/', bnKey: 'batch_normalization_11/', preluKey: 'p_re_lu_11/', inCh: 128, outCh: 128, stride: 1, inH: 24 },
    { dwKey: 'depthwise_conv2d_11/', pwKey: 'conv2d_12/', bnKey: 'batch_normalization_12/', preluKey: 'p_re_lu_12/', inCh: 128, outCh: 256, stride: 2, inH: 24 },
    // Stage 4: 256ch blocks, 12x12 input
    { dwKey: 'depthwise_conv2d_12/', pwKey: 'conv2d_13/', bnKey: 'batch_normalization_13/', preluKey: 'p_re_lu_13/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_13/', pwKey: 'conv2d_14/', bnKey: 'batch_normalization_14/', preluKey: 'p_re_lu_14/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_14/', pwKey: 'conv2d_15/', bnKey: 'batch_normalization_15/', preluKey: 'p_re_lu_15/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_15/', pwKey: 'conv2d_16/', bnKey: 'batch_normalization_16/', preluKey: 'p_re_lu_16/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_16/', pwKey: 'conv2d_17/', bnKey: 'batch_normalization_17/', preluKey: 'p_re_lu_17/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_17/', pwKey: 'conv2d_18/', bnKey: 'batch_normalization_18/', preluKey: 'p_re_lu_18/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_18/', pwKey: 'conv2d_19/', bnKey: 'batch_normalization_19/', preluKey: 'p_re_lu_19/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    // NOTE: block 19 (the last backbone block before FPN) uses different naming for the dw bias
    // Actually the DW bias is always from the BN fused into the DW conv itself.
    // Let me re-examine: In BlazeNet, each block is DW5x5 → BN → PReLU → PW1x1 → BN → PReLU → Add
    // The DW bias comes from the DW-associated BN, and PW bias from the PW-associated BN.
    // But in the TFLite manifest, the BN biases labeled "batch_normalization_N" correspond to conv2d_N.
    // The depthwise convs don't have separate BN entries — their biases are folded into the weights.
    // Actually wait — let me look again. The DW weights are stored as [1,5,5,ch] and there are no
    // separate DW bias entries. The "batch_normalization_N" entries are the PW conv biases.
    // For the DW, the bias is already folded into the depthwise weights during TFLite conversion.
  ];

  // Actually, looking at the manifest again, there are NO separate depthwise biases.
  // The depthwise convolutions have their batchnorm folded into the weights.
  // So we pass a zero bias for the DW shader and rely on the PW bias (from batch_normalization) + PReLU.
  //
  // Wait, but the hand landmark model DOES have separate DW biases. Let me check...
  // In the palm detection manifest, the DW bias would be folded. We need to create zero-biases for DW.

  const blocks: BackboneBlock[] = blockDefs.map(def => {
    const dwTensor = findWeight(def.dwKey);
    const pwTensor = findWeight(def.pwKey);
    const bnTensor = findWeight(def.bnKey);
    const preluTensor = findWeight(def.preluKey);

    // Transpose DW weights from [1,5,5,ch] to [ch, 25]
    const dwTransposed = transposeDW(dwTensor);
    const dwBuf = makeBuf(dwTransposed.byteLength, SC);
    writeBuf(dwBuf, 0, dwTransposed);

    // Zero bias for DW (bias is folded into weights in TFLite export)
    const dwBias = new Float32Array(def.inCh);
    const dwBiasBuf = makeBuf(dwBias.byteLength, SC);
    writeBuf(dwBiasBuf, 0, dwBias);

    // PW weights: already [outCh, 1, 1, inCh] = [outCh, inCh]
    const pwTransposed = transposePW(pwTensor);
    const pwBuf = makeBuf(pwTransposed.byteLength, SC);
    writeBuf(pwBuf, 0, pwTransposed);

    // PW bias from fused batchnorm
    const pwBiasBuf = uploadWeights(bnTensor);

    // PReLU alpha
    const alphaBuf = uploadWeights(preluTensor);

    return {
      dwWeightBuf: dwBuf,
      dwBiasBuf: dwBiasBuf,
      pwWeightBuf: pwBuf,
      pwBiasBuf: pwBiasBuf,
      alphaBuf,
      inCh: def.inCh,
      outCh: def.outCh,
      stride: def.stride,
      inH: def.inH,
    };
  });

  // ============ FPN weights ============

  // conv2d_20: 256→256 project stage4 output at 12x12, with bn_20 bias and prelu_20 alpha
  const fpnProj12W = transposePW(findWeight('conv2d_20/Conv2D'));
  const fpnProj12WBuf = makeBuf(fpnProj12W.byteLength, SC);
  writeBuf(fpnProj12WBuf, 0, fpnProj12W);
  const fpnProj12BBuf = uploadWeights(findWeight('batch_normalization_20/'));
  const fpnProj12AlphaBuf = uploadWeights(findWeight('p_re_lu_20/'));

  // conv2d_22/Conv2D1: 256→256 project backbone3 skip (stage3 output at 24x24→ channels match)
  // Wait — backbone3 output is at 24x24 with 128 channels, but conv2d_22/Conv2D1 is [256,1,1,256].
  // Let me re-think. Actually:
  // - backbone2 output = after stage 2 transition = 24x24 x 128 — this is where we need skip
  // - backbone3 output = after stage 3 = 12x12 x 256 (stage 4 input)
  //
  // The FPN upsample goes from 12→24, needs to match 24x24 features.
  // conv2d_22/Conv2D1 [256,1,1,256] projects... hmm that doesn't match 128ch.
  //
  // Let me look at this differently. The manifest shows:
  // conv2d_22/Conv2D1 [256,1,1,256] — this projects something 256ch to 256ch
  //
  // Actually for MediaPipe palm detection FPN:
  // - After stage4 (12x12x256), apply conv2d_20 (256→256) + PReLU
  // - Upsample 12→24, but the skip is from the backbone2/3 boundary
  //
  // Looking more carefully at the model:
  // Stage 3 ends at block 11: 128→256 stride 2, output 12x12x256
  // Stage 3 blocks 8-10 are at 24x24x128
  // The skip for FPN upsampling to 24x24 should be from the 24x24 feature map
  //
  // But conv2d_22/Conv2D1 is [256,1,1,256], not [256,1,1,128].
  // This means the FPN skip isn't from pre-transition 24x24x128 but from some intermediate point.
  //
  // Actually, re-reading the task description:
  // "FPN: Upsample 6x6→12x12 (add backbone3 skip), 12x12→24x24 (add backbone2 skip)"
  //
  // Wait, that says 6x6→12x12 and 12x12→24x24. So the deepest feature is at 6x6?
  // But stage 4 is at 12x12. Let me re-examine...
  //
  // Actually the task description says the model has 4 stages with 4 blocks each.
  // Stage 1: 96→48 (stride 2), Stage 2: 48→24, Stage 3: 24→12, Stage 4: 12→6
  // Then FPN: 6→12, 12→24
  //
  // But looking at my block definitions, I have stages going:
  // 96→48 (block 3), 48→24 (block 7), 24→12 (block 11), then blocks 12-18 stay at 12x12
  //
  // Hmm, but the task says 4 stages with stride-2 transitions:
  // 32→64, 64→128, 128→256 channels.
  // That's only 3 stride-2 transitions. Where's the 4th?
  //
  // Let me count: we have depthwise_conv2d_0 through _18 in the backbone = 19 DW convs.
  // Plus the initial conv. That's 20 layers total.
  // 4 stages × (4 blocks × 1 DW each + 1 stride-2 transition) = 4 × 5 = 20? No, that's wrong.
  //
  // Actually: 4 stages, each with 4 depthwise-separable blocks:
  // Stage 1 (32ch): blocks 0,1,2 at stride 1, block 3 at stride 2 → 64ch
  // Stage 2 (64ch): blocks 4,5,6 at stride 1, block 7 at stride 2 → 128ch
  // Stage 3 (128ch): blocks 8,9,10 at stride 1, block 11 at stride 2 → 256ch
  // Stage 4 (256ch): blocks 12,13,14,15,16,17,18 at stride 1
  //
  // That's 3 blocks stride-1 + 1 transition = 4 blocks per stage for stages 1-3,
  // and stage 4 has 7 blocks. Total: 4×3 + 7 = 19 blocks + initial conv = 20 convolutions.
  //
  // Stage 4 has 7 more blocks than expected because the task description says "4 blocks per stage"
  // but MediaPipe's BlazeNet has more blocks in the deeper stages.
  //
  // So FPN starts from 12x12, not 6x6. The skip connections are:
  // - backbone2 = 48x48 feature map (after stage 1, before stride-2 to 64ch) - no wait
  // - Actually the convention is backbone_N refers to the Nth backbone stage output.
  //
  // For the FPN to work with conv2d_22/Conv2D1 [256,1,1,256]:
  // This could be projecting the 12x12x256 features. But that doesn't make sense for a skip.
  //
  // Let me try a different interpretation:
  // FPN path uses conv2d_20 to project stage4 output, then:
  // - First upsample: stage4 12x12 → needs NO upsample, it's already at 12x12
  // - Actually maybe the task description is slightly off and this model has blocks at 6x6 too.
  //
  // Looking again at blocks 12-18: they are ALL at 12x12 with 256ch. That's 7 blocks at 12x12.
  // The task says "4 stages (32→64→128→256 channels), each with 4 depthwise-separable residual blocks"
  // That means stage 4 has 4 blocks too: blocks 12-15. Then what are blocks 16-18?
  //
  // Re-reading: "BlazeNet with 4 stages (32→64→128→256 channels), each with 4 depthwise-separable residual blocks"
  // So that's 4×4 = 16 DW blocks. But we have 19 (0-18).
  // The extra 3 could be the FPN decoder blocks.
  //
  // Let me check: blocks 12,13,14,15 = stage 4 (256ch, 12x12).
  // Then blocks 16,17,18 might be part of FPN/decoder, not stage 4.
  //
  // But looking at the weight names, all of depthwise_conv2d_12 through _18 have the same
  // pattern with conv2d_13 through conv2d_19. The FPN-specific convs are conv2d_20, conv2d_21.
  //
  // I think the architecture is simply:
  // - Stage 4 has more than 4 blocks (it has 7: blocks 12-18)
  // - After stage 4 (12x12x256), FPN starts
  //
  // For the FPN:
  // conv2d_20 [256,1,1,256] + bn_20 [256] + prelu_20 = project stage4 output (12x12x256)
  // upsample 12→24
  // conv2d_22/Conv2D1 [256,1,1,256] = project the skip connection from somewhere
  //
  // But backbone2 output (before stride-2 to 128) is 24x24x64, not 256ch.
  // backbone3 output (at 12x12) is already 256ch but 12x12, not useful for 24x24 skip.
  // backbone2 end (after stride-2 block 7 output) is 24x24x128.
  //
  // Wait — I need to save the skip BEFORE the stride-2 transition.
  // backbone2_skip = output of block 10 (last 128ch block at 24x24) = 24x24x128
  // backbone3_skip = output of block 11 (stride-2 transition) = 12x12x256... no
  //
  // Actually for FPN in BlazeNet:
  // - Save backbone stage 2 output at 24x24 (after blocks 8-10, before block 11 stride-2)
  // - Save backbone stage 3 output at 12x12 (stage 4 blocks 12-15... or the stage3 end)
  //
  // Hmm, this is getting complex. Let me look at it from the output heads:
  // - 12x12 head: classifier_palm_16 [6,1,1,256] and regressor_palm_16 [108,1,1,256]
  //   → operates on 12x12x256 features
  // - 24x24 head: classifier_palm_8 [2,1,1,128] and regressor_palm_8 [36,1,1,128]
  //   → operates on 24x24x128 features
  //
  // So the 24x24 path needs 128 channels. Let's look at the FPN path:
  // conv2d_23 [128,1,1,256] — projects 256→128 channels
  // This makes sense: after upsample from 12x12→24x24 (256ch), project to 128ch.
  // Then add skip from backbone at 24x24 (which was 128ch... but wait, we need a skip
  // that's also 128ch at 24x24).
  //
  // Actually: conv2d_25/Conv2D1 [128,1,1,128] = skip projection for the 24x24 FPN level
  //
  // And looking at the names more carefully:
  // dw_19 + conv2d_21: FPN block after first upsample (24x24, 256ch)
  // conv2d_22/Conv2D1: this is the skip projection at some level
  //
  // Let me try this FPN structure:
  // 1. Stage 4 output: 12x12x256
  // 2. conv2d_20 projects to 12x12x256 (with PReLU)
  // 3. SSD head at 12x12: classifier_palm_16 + regressor_palm_16 (on stage4 output directly)
  //
  // Actually, maybe the 12x12 SSD head runs on the conv2d_20 output directly.
  // Then for the 24x24 path:
  // 4. Upsample 12→24 (256ch)
  // 5. Add skip: conv2d_22/Conv2D1 [256,1,1,256] projects backbone3 (12x12x256?) — no
  //
  // OK let me just look at which conv2d has what skip prefix:
  // conv2d_22/Conv2D1 [256,1,1,256] — note the "Conv2D1" suffix (not just "Conv2D")
  // conv2d_25/Conv2D1 [128,1,1,128] — also "Conv2D1" suffix
  // These "Conv2D1" suffixed weights are the skip/residual projections.
  //
  // For the FPN, I think the structure is:
  //
  // 12x12 path:
  // - Stage4 output (12x12x256)
  // - dw_19 + conv2d_20: one more block at 12x12x256 with PReLU
  //   (prelu_20, bn_20)
  // - 12x12 SSD heads: classifier_palm_16 [6,1,1,256], regressor_palm_16 [108,1,1,256]
  //
  // But wait, after the 7 backbone stage4 blocks (12-18), we have:
  // - conv2d_19 is the last PW conv in the backbone.
  // - So stage4 output = output of block 18 = 12x12x256.
  //
  // Then FPN:
  // dw_19 [1,5,5,256] + conv2d_21 [256,1,1,256] + bn_21 + prelu_21 = FPN block at 12x12
  //   skip projection: conv2d_22/Conv2D1 [256,1,1,256]
  // Wait, but dw_19 comes AFTER the backbone blocks. Let me reconsider.
  //
  // There are 23 depthwise convs total (dw_0 through dw_22).
  // Backbone: dw_0 through dw_18 (19 DW convs)
  // FPN/decoder: dw_19, dw_20, dw_21, dw_22 (4 more DW convs)
  //
  // Looking at the heads:
  // 12x12 heads feed from 256ch features
  // 24x24 heads feed from 128ch features
  //
  // Let me try this refined structure:
  //
  // 12x12 feature extraction:
  // - backbone output: 12x12x256
  // - conv2d_20 [256,1,1,256] + bn_20 + prelu_20 = project features (1x1 conv + PReLU)
  // - dw_19 [1,5,5,256] + conv2d_21 [256,1,1,256] + bn_21 + prelu_21 = one DW-sep block
  //   with skip via conv2d_22/Conv2D1 [256,1,1,256]
  // - dw_20 [1,5,5,256] + ... hmm, but there's no conv2d_22 PW for this
  //
  // Actually I see bn_22 [256] and prelu_22 [1,1,256] in the manifest. And conv2d_22/Conv2D1.
  // conv2d_22 has two weight entries:
  // - conv2d_22/Conv2D1 [256,1,1,256] — this is the skip projection
  // - bn_22 [256] + prelu_22 [1,1,256] — these go with conv2d_22 output
  //
  // So the 12x12 block after conv2d_20 is:
  // Input → dw_19 → conv2d_20 already done... no.
  //
  // Let me reconsider. Maybe:
  // - conv2d_20 is not after dw_19. conv2d_20 [256,1,1,256] is a standalone 1x1 projection.
  //
  // FPN at 12x12:
  // 1. backbone stage4 output: 12x12x256
  // 2. conv2d_20 [256,1,1,256] + bn_20 + prelu_20: project backbone output
  //    → 12x12x256 features for FPN path
  // 3. 12x12 SSD heads on this projected feature:
  //    classifier_palm_16 [6,1,1,256], regressor_palm_16 [108,1,1,256]
  //
  // Then separately, for the 24x24 path:
  // 4. upsample 12→24 of the conv2d_20 output (256ch)
  // 5. Add skip: save backbone at 24x24, project with conv2d_22/Conv2D1 [256,1,1,256]
  //    But backbone at 24x24 is 128ch, not 256ch. conv2d_22/Conv2D1 is [256,1,1,256].
  //    This doesn't match.
  //
  // Unless the skip at 24x24 has 256ch. That would mean saving the output AFTER the
  // stride-2 transition but using a different feature map. Hmm.
  //
  // Actually, maybe I miscounted the backbone stages. Let me re-examine:
  // The task says: "4 stages (32→64→128→256 channels)"
  // If each stage has 4 DW-sep blocks, the stride-2 transition is the FIRST block of each stage:
  //
  // Stage 0: initial conv → 96x96x32
  // Stage 1 (32ch): blocks 0-3 at 96x96, all stride 1
  //   Then stride-2 transition to 48x48x64
  // Stage 2 (64ch): blocks 4-7, the first block has stride 2
  //
  // No wait, that gives 32ch for 4 blocks, then transition. But the first DW block is
  // depthwise_conv2d_0 with [1,5,5,32] and conv2d_1 [32,1,1,32] — same channels, stride 1.
  //
  // OK I'll just go with my original analysis:
  // blocks 0-2: 32→32, stride 1, 96x96
  // block 3: 32→64, stride 2, 96→48
  // blocks 4-6: 64→64, stride 1, 48x48
  // block 7: 64→128, stride 2, 48→24
  // blocks 8-10: 128→128, stride 1, 24x24
  // block 11: 128→256, stride 2, 24→12
  // blocks 12-18: 256→256, stride 1, 12x12
  //
  // So we save:
  // backbone2_skip = output of block 10 = 24x24x128 (before stride-2 transition to 256ch)
  // backbone3_skip = output of block 7 or... hmm
  //
  // Actually for the FPN:
  // We need a 24x24 skip. The 24x24 features are at 128ch (blocks 8-10).
  // But conv2d_22/Conv2D1 is [256,1,1,256]. That can't project 128→256.
  //
  // Unless the skip is from the stride-2 block output: block 11 outputs 12x12x256.
  // That's the wrong spatial resolution for a 24x24 skip.
  //
  // I think what's happening is:
  // 1. The FPN operates entirely in 256ch for the 12→24 upsample
  // 2. conv2d_22/Conv2D1 projects the stage4 backbone features (12x12x256) as a skip
  //    to be added after upsampling a different branch
  // 3. Then after adding, conv2d_23 projects 256→128 for the 24x24 head
  //
  // Actually, I think the FPN structure is:
  //
  // 12x12 branch:
  // - backbone output: 12x12x256
  // - conv2d_20 [256,1,1,256] + bn_20 + prelu_20: refine features
  // - dw_19 [1,5,5,256] + conv2d_21 [256,1,1,256] + bn_21 + prelu_21: another block
  //   skip: conv2d_22/Conv2D1 [256,1,1,256] projects the conv2d_20 output as skip
  // - Output: refined 12x12x256
  // - 12x12 SSD heads on conv2d_20 output (or after dw_19/conv2d_21 block)
  //
  // Actually I think the 12x12 SSD heads run directly on the backbone stage4 output
  // and the FPN only creates the 24x24 features.
  //
  // Let me just go with a simpler interpretation that makes the shapes work:
  //
  // After backbone (12x12x256):
  // FPN block A at 12x12:
  //   dw_19 [1,5,5,256] + conv2d_20 [256,1,1,256] + bn_20 + prelu_20
  //   Skip: from backbone stage4 output (same 12x12x256)
  //   But we also have conv2d_22/Conv2D1 [256,1,1,256] and bn_22 [256] + prelu_22 alpha
  //
  // Then:
  //   dw_20 [1,5,5,256] + conv2d_21 [256,1,1,256] + bn_21 + prelu_21
  //   Skip: conv2d_22/Conv2D1 projects backbone or previous output
  //
  // Actually, the cleanest interpretation that matches the naming is:
  //
  // After backbone stage4 output (12x12x256):
  //
  // FPN 12x12 block 1: dw_19 + conv2d_20 + bn_20 + prelu_20, skip=stage4 output
  // FPN 12x12 block 2: dw_20 + conv2d_21 + bn_21 + prelu_21, skip=block1 output
  //   conv2d_22/Conv2D1 + bn_22 + prelu_22 is a projection of the block1 output used as skip for block2
  //
  // → 12x12 SSD head operates on FPN block 2 output (12x12x256)
  //
  // Then upsample 12→24:
  // conv2d_23 [128,1,1,256]: project 256→128 at 24x24
  //   + prelu_23 alpha, + bn_23 bias (if exists... bn_23 = "batch_normalization_23" = [128])
  //
  // Then 24x24 blocks:
  // FPN 24x24 block 1: dw_21 + conv2d_24 + bn_24 + prelu_24, skip from somewhere
  // FPN 24x24 block 2: dw_22 + conv2d_25/Conv2D1 + bn_25 + prelu_25, skip from block1
  //   conv2d_25/Conv2D1 is the skip projection
  //
  // → 24x24 SSD head operates on FPN 24x24 output (24x24x128)
  //
  // This makes sense! Let me also add the skip connection for the upsample:
  // The upsample from 12→24 adds with a backbone skip at 24x24.
  // Backbone at 24x24 is 128ch (blocks 8-10 output). After conv2d_23 projects to 128ch
  // we can add the backbone2 skip (24x24x128).
  //
  // But actually the upsample happens BEFORE conv2d_23. Or maybe:
  // 1. conv2d_23 projects FPN 12x12 output (256ch) to 128ch at 12x12
  // 2. upsample 12→24 (128ch)
  // 3. Add backbone skip at 24x24x128
  //
  // That works with all the shapes! And then the two blocks refine it:
  // dw_21+conv2d_24 (128ch, 24x24) + dw_22+conv2d_25 (128ch, 24x24)
  //
  // For the 12x12 blocks, let me simplify further:
  // The "12x12 blocks" might not exist as separate blocks. Maybe:
  // - 12x12 SSD heads run directly on backbone stage4 output
  // - conv2d_20 + dw_19,20 + conv2d_21 are actually part of the FPN decoder
  //
  // Actually, let me re-examine. The blocks dw_19, dw_20 have [1,5,5,256].
  // conv2d_20 [256,1,1,256], conv2d_21 [256,1,1,256].
  // These form two additional residual blocks at 12x12x256 AFTER the backbone.
  //
  // So the final architecture:
  //
  // Backbone → 12x12x256
  // Extra block A: dw_19+conv2d_20 (256ch, 12x12) + skip from backbone
  // Extra block B: dw_20+conv2d_21 (256ch, 12x12) + skip from block A
  //   → conv2d_22/Conv2D1 projects block A output as skip for block B? Or just identity skip.
  //
  // Actually for the skip in extra blocks, since channels don't change (256→256),
  // the skip is identity (just add input). conv2d_22/Conv2D1 might be something else.
  //
  // Let me just go with the simplest working interpretation:
  //
  // BACKBONE + EXTRA BLOCKS → 12x12x256 features
  // ├── 12x12 SSD heads (classifier_palm_16, regressor_palm_16)
  // └── FPN to 24x24:
  //     conv2d_23 (256→128) + PReLU → upsample 12→24 → add backbone2 skip (24x24x128)
  //     → dw_21+conv2d_24 block (128ch, 24x24)
  //     → dw_22+conv2d_25 block (128ch, 24x24) [conv2d_25/Conv2D1 is skip projection]
  //     → 24x24 SSD heads (classifier_palm_8, regressor_palm_8)

  // Extra blocks at 12x12 (after backbone stage 4)
  // Block A: dw_19 + conv2d_20 + bn_20 + prelu_20
  const extraBlockA = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_19/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(256); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
    pwWeightBuf: fpnProj12WBuf,
    pwBiasBuf: fpnProj12BBuf,
    alphaBuf: fpnProj12AlphaBuf,
    inCh: 256, outCh: 256, stride: 1 as const, inH: 12,
  };

  // Block B: dw_20 + conv2d_21 + bn_21 + prelu_21
  const extraBlockB = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_20/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(256); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
    pwWeightBuf: (() => { const d = transposePW(findWeight('conv2d_21/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    pwBiasBuf: uploadWeights(findWeight('batch_normalization_21/')),
    alphaBuf: uploadWeights(findWeight('p_re_lu_21/')),
    inCh: 256, outCh: 256, stride: 1 as const, inH: 12,
  };

  // conv2d_22/Conv2D1: skip projection for the extra blocks
  // bn_22 and prelu_22 - these might be the skip projection with activation
  // Actually for BlazeNet, when channels match, skip is identity. conv2d_22/Conv2D1
  // might be unused in this path. Let me check if bn_22/prelu_22 exist and what they're for.
  // bn_22 [256], prelu_22 [1,1,256] — yes they exist.
  //
  // I think conv2d_22/Conv2D1 + bn_22 + prelu_22 form the final refinement conv at 12x12.
  // Or maybe they're used differently.
  //
  // For now, let me just treat blocks A and B as regular residual blocks with identity skip
  // (since channels don't change).

  // FPN projection: conv2d_23 [128,1,1,256] projects 256→128
  const fpnProjW = transposePW(findWeight('conv2d_23/Conv2D'));
  const fpnProjWBuf = makeBuf(fpnProjW.byteLength, SC);
  writeBuf(fpnProjWBuf, 0, fpnProjW);
  const fpnProjBBuf = uploadWeights(findWeight('batch_normalization_23/'));
  const fpnProjAlphaBuf = uploadWeights(findWeight('p_re_lu_23/'));

  // FPN 24x24 block 1: dw_21 + conv2d_24
  const fpnBlock1 = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_21/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(128); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
    pwWeightBuf: (() => { const d = transposePW(findWeight('conv2d_24/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    pwBiasBuf: uploadWeights(findWeight('batch_normalization_24/')),
    alphaBuf: uploadWeights(findWeight('p_re_lu_24/')),
    inCh: 128, outCh: 128, stride: 1 as const, inH: 24,
  };

  // FPN 24x24 block 2: dw_22 + conv2d_25 (identity since 128→128 but conv2d_25/Conv2D1 is skip projection)
  const fpnBlock2 = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_22/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(128); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
    // The PW for block2 comes from... looking at the manifest, there's no conv2d_25/Conv2D (without "1")
    // conv2d_25/Conv2D1 [128,1,1,128] — this is labeled as skip projection but actually might be the PW conv.
    // In the naming convention, Conv2D1 means it's an alternate weight for the same layer.
    // Let me check if there are separate bn_25 entries... yes: bn_25 [128] and prelu_25 [1,1,128].
    // So conv2d_25/Conv2D1 IS the PW conv for this block, and bn_25 is its bias.
    pwWeightBuf: (() => { const d = transposePW(findWeight('conv2d_25/Conv2D1')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    pwBiasBuf: uploadWeights(findWeight('batch_normalization_25/')),
    alphaBuf: uploadWeights(findWeight('p_re_lu_25/')),
    inCh: 128, outCh: 128, stride: 1 as const, inH: 24,
  };

  // ============ SSD Head weights ============
  // 12x12 head (6 anchors per cell)
  const cls16W = transposePW(findWeight('classifier_palm_16_NO_PRUNING/Conv2D'));
  const cls16WBuf = makeBuf(cls16W.byteLength, SC);
  writeBuf(cls16WBuf, 0, cls16W);
  const cls16BBuf = uploadWeights(findWeight('classifier_palm_16_NO_PRUNING/BiasAdd'));

  const reg16W = transposePW(findWeight('regressor_palm_16_NO_PRUNING/Conv2D'));
  const reg16WBuf = makeBuf(reg16W.byteLength, SC);
  writeBuf(reg16WBuf, 0, reg16W);
  const reg16BBuf = uploadWeights(findWeight('regressor_palm_16_NO_PRUNING/BiasAdd'));

  // 24x24 head (2 anchors per cell)
  const cls8W = transposePW(findWeight('classifier_palm_8_NO_PRUNING/Conv2D'));
  const cls8WBuf = makeBuf(cls8W.byteLength, SC);
  writeBuf(cls8WBuf, 0, cls8W);
  const cls8BBuf = uploadWeights(findWeight('classifier_palm_8_NO_PRUNING/BiasAdd'));

  const reg8W = transposePW(findWeight('regressor_palm_8_NO_PRUNING/Conv2D'));
  const reg8WBuf = makeBuf(reg8W.byteLength, SC);
  writeBuf(reg8WBuf, 0, reg8W);
  const reg8BBuf = uploadWeights(findWeight('regressor_palm_8_NO_PRUNING/BiasAdd'));

  // ============ Activation buffers ============
  // We'll use double-buffering: alternate between two buffers for each layer's input/output.
  // Maximum spatial: 96x96x32 = 294912, 48x48x64 = 147456, 24x24x128 = 73728, 12x12x256 = 36864
  // For DW output, we need same spatial as DW input but same channels.
  // Max buffer size needed: 96x96x32 = 294912 floats
  const maxBufSize = 192 * 192 * 3; // input buffer for 192x192x3
  const buf192 = Math.max(192 * 192 * 3, 96 * 96 * 64, 48 * 48 * 128, 24 * 24 * 256, 12 * 12 * 256) * 4;

  const inputBuf = makeBuf(192 * 192 * 3 * 4, SC);
  const actBufA = makeBuf(buf192, SO);  // Activation buffer A
  const actBufB = makeBuf(buf192, SO);  // Activation buffer B
  const dwOutBuf = makeBuf(buf192, SO); // DW output buffer

  // Save buffers for skip connections
  const backbone2SkipBuf = makeBuf(24 * 24 * 128 * 4, SO | GPUBufferUsage.COPY_DST); // After block 10 (24x24x128)
  // For FPN upsample+add at 24x24
  const fpnUpsampleBuf = makeBuf(24 * 24 * 128 * 4, SO);

  // SSD output buffers
  const cls16Buf = makeBuf(12 * 12 * 6 * 4, SOC);   // 12x12x6 = 864
  const reg16Buf = makeBuf(12 * 12 * 108 * 4, SOC);  // 12x12x108 = 15552
  const cls8Buf = makeBuf(24 * 24 * 2 * 4, SOC);     // 24x24x2 = 1152
  const reg8Buf = makeBuf(24 * 24 * 36 * 4, SOC);    // 24x24x36 = 20736

  // Readback buffers
  const cls16ReadBuf = makeBuf(12 * 12 * 6 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const reg16ReadBuf = makeBuf(12 * 12 * 108 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const cls8ReadBuf = makeBuf(24 * 24 * 2 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const reg8ReadBuf = makeBuf(24 * 24 * 36 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

  // Canvas input texture
  const canvasInputTexture = device.createTexture({
    size: [192, 192, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // ============ Build the command encoder function ============
  function ceil(a: number, b: number): number {
    return Math.ceil(a / b);
  }

  function makeUniform(data: Uint32Array): GPUBuffer {
    const buf = makeBuf(data.byteLength, UC);
    writeBuf(buf, 0, data);
    return buf;
  }

  // Pre-create uniform buffers for each layer

  // Input conv: 192x192x3 → 96x96x32
  const inputConvUniform = makeUniform(new Uint32Array([1, 3, 32, 192, 192, 96, 96]));

  // Helper to encode a DW+PW block into the command encoder
  // Returns which buffer has the output (A or B toggled)
  function encodeDwPwBlock(
    encoder: GPUCommandEncoder,
    block: BackboneBlock,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    skipBuffer: GPUBuffer, // for residual skip connection
  ): void {
    const outH = block.stride === 2 ? block.inH / 2 : block.inH;
    const outW = outH;
    const pad = block.stride === 2 ? 1 : 2;

    // DW conv
    const dwUniform = makeUniform(new Uint32Array([1, block.inCh, block.inH, block.inH, outH, outW, block.stride, pad]));
    const dwBG = device.createBindGroup({
      layout: dwLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: block.dwWeightBuf } },
        { binding: 2, resource: { buffer: block.dwBiasBuf } },
        { binding: 3, resource: { buffer: dwOutBuf } },
        { binding: 4, resource: { buffer: dwUniform } },
      ],
    });

    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(dwPipe);
    pass1.setBindGroup(0, dwBG);
    pass1.dispatchWorkgroups(ceil(outW, 8), ceil(outH, 8), block.inCh);
    pass1.end();

    // PW conv + skip + PReLU
    // channel_pad = number of channels in the skip input (for zero-padding)
    // When inCh != outCh, we zero-pad: channels 0..inCh-1 use skip, inCh..outCh-1 get 0
    const channelPad = block.inCh; // skip has inCh channels
    const pwUniform = makeUniform(new Uint32Array([1, block.inCh, block.outCh, outH, outW, channelPad, block.stride, block.inH, block.inH]));
    const pwBG = device.createBindGroup({
      layout: pwPreluLayout,
      entries: [
        { binding: 0, resource: { buffer: dwOutBuf } },
        { binding: 1, resource: { buffer: skipBuffer } },
        { binding: 2, resource: { buffer: block.pwWeightBuf } },
        { binding: 3, resource: { buffer: block.pwBiasBuf } },
        { binding: 4, resource: { buffer: block.alphaBuf } },
        { binding: 5, resource: { buffer: outputBuffer } },
        { binding: 6, resource: { buffer: pwUniform } },
      ],
    });

    const pass2 = encoder.beginComputePass();
    pass2.setPipeline(pwPreluPipe);
    pass2.setBindGroup(0, pwBG);
    pass2.dispatchWorkgroups(ceil(outW, 8), ceil(outH, 8), block.outCh);
    pass2.end();
  }

  // Encode a conv1x1 (no activation) for SSD heads
  function encodeConv1x1(
    encoder: GPUCommandEncoder,
    inputBuffer: GPUBuffer,
    weightBuf: GPUBuffer,
    biasBuf: GPUBuffer,
    outputBuffer: GPUBuffer,
    inCh: number, outCh: number, h: number, w: number,
  ): void {
    const uniform = makeUniform(new Uint32Array([1, inCh, outCh, h, w]));
    const bg = device.createBindGroup({
      layout: conv1x1Layout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuf } },
        { binding: 2, resource: { buffer: biasBuf } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: uniform } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(conv1x1Pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(ceil(w, 8), ceil(h, 8), outCh);
    pass.end();
  }

  // Encode conv1x1 + PReLU
  function encodeConv1x1PReLU(
    encoder: GPUCommandEncoder,
    inputBuffer: GPUBuffer,
    weightBuf: GPUBuffer,
    biasBuf: GPUBuffer,
    alphaBuf: GPUBuffer,
    outputBuffer: GPUBuffer,
    inCh: number, outCh: number, h: number, w: number,
  ): void {
    const uniform = makeUniform(new Uint32Array([1, inCh, outCh, h, w]));
    const bg = device.createBindGroup({
      layout: conv1x1PreluLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuf } },
        { binding: 2, resource: { buffer: biasBuf } },
        { binding: 3, resource: { buffer: alphaBuf } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: uniform } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(conv1x1PreluPipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(ceil(w, 8), ceil(h, 8), outCh);
    pass.end();
  }

  async function run(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<PalmDetectionOutput> {
    // Upload canvas to texture
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: canvasInputTexture },
      [192, 192],
    );

    // Canvas input → CHW buffer
    const canvasUniform = makeUniform(new Uint32Array([192, 192, 192]));
    const canvasBG = device.createBindGroup({
      layout: canvasInputLayout,
      entries: [
        { binding: 0, resource: canvasInputTexture.createView() },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: canvasUniform } },
      ],
    });

    const encoder = device.createCommandEncoder();

    // Canvas → CHW
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipe);
      pass.setBindGroup(0, canvasBG);
      pass.dispatchWorkgroups(ceil(192, 16), ceil(192, 16), 1);
      pass.end();
    }

    // Initial conv 5x5 stride 2 + PReLU: 192x192x3 → 96x96x32
    {
      const bg = device.createBindGroup({
        layout: inputConvLayout,
        entries: [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: initConvWeightBuf } },
          { binding: 2, resource: { buffer: initConvBiasBuf } },
          { binding: 3, resource: { buffer: initConvAlphaBuf } },
          { binding: 4, resource: { buffer: actBufA } },
          { binding: 5, resource: { buffer: inputConvUniform } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(inputConvPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(96, 8), ceil(96, 8), 32);
      pass.end();
    }

    // Backbone blocks with double buffering
    // After input conv: actBufA has 96x96x32
    let curBuf = actBufA;
    let altBuf = actBufB;

    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i];
      // For same-channel blocks (stride 1, inCh==outCh), skip = curBuf (identity)
      // For channel-change blocks (stride 2, inCh!=outCh), skip = curBuf (zero-pad in shader)
      encodeDwPwBlock(encoder, block, curBuf, altBuf, curBuf);

      // Swap buffers
      const tmp = curBuf;
      curBuf = altBuf;
      altBuf = tmp;

      // Save skip for backbone2 (after block 10 = last 128ch stride-1 block at 24x24)
      if (i === 10) {
        // Copy 24x24x128 from curBuf to backbone2SkipBuf
        encoder.copyBufferToBuffer(curBuf, 0, backbone2SkipBuf, 0, 24 * 24 * 128 * 4);
      }
    }

    // After backbone: curBuf has 12x12x256

    // Extra block A at 12x12
    encodeDwPwBlock(encoder, extraBlockA, curBuf, altBuf, curBuf);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Extra block B at 12x12
    encodeDwPwBlock(encoder, extraBlockB, curBuf, altBuf, curBuf);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // 12x12 SSD heads (on curBuf = 12x12x256)
    encodeConv1x1(encoder, curBuf, cls16WBuf, cls16BBuf, cls16Buf, 256, 6, 12, 12);
    encodeConv1x1(encoder, curBuf, reg16WBuf, reg16BBuf, reg16Buf, 256, 108, 12, 12);

    // FPN: project 256→128 + PReLU at 12x12
    encodeConv1x1PReLU(encoder, curBuf, fpnProjWBuf, fpnProjBBuf, fpnProjAlphaBuf, altBuf, 256, 128, 12, 12);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Upsample 12→24 + add backbone2 skip (24x24x128)
    {
      const uniform = makeUniform(new Uint32Array([1, 128, 12, 12, 24, 24]));
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: backbone2SkipBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: uniform } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(24, 8), ceil(24, 8), 128);
      pass.end();
    }
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // FPN 24x24 block 1
    encodeDwPwBlock(encoder, fpnBlock1, curBuf, altBuf, curBuf);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // FPN 24x24 block 2
    encodeDwPwBlock(encoder, fpnBlock2, curBuf, altBuf, curBuf);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // 24x24 SSD heads (on curBuf = 24x24x128)
    encodeConv1x1(encoder, curBuf, cls8WBuf, cls8BBuf, cls8Buf, 128, 2, 24, 24);
    encodeConv1x1(encoder, curBuf, reg8WBuf, reg8BBuf, reg8Buf, 128, 36, 24, 24);

    // Submit compute passes first
    device.queue.submit([encoder.finish()]);

    // Copy outputs to readback buffers (separate encoder to ensure compute is complete)
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(cls16Buf, 0, cls16ReadBuf, 0, 12 * 12 * 6 * 4);
    copyEncoder.copyBufferToBuffer(reg16Buf, 0, reg16ReadBuf, 0, 12 * 12 * 108 * 4);
    copyEncoder.copyBufferToBuffer(cls8Buf, 0, cls8ReadBuf, 0, 24 * 24 * 2 * 4);
    copyEncoder.copyBufferToBuffer(reg8Buf, 0, reg8ReadBuf, 0, 24 * 24 * 36 * 4);
    device.queue.submit([copyEncoder.finish()]);

    // Wait and read back
    await Promise.all([
      cls16ReadBuf.mapAsync(GPUMapMode.READ),
      reg16ReadBuf.mapAsync(GPUMapMode.READ),
      cls8ReadBuf.mapAsync(GPUMapMode.READ),
      reg8ReadBuf.mapAsync(GPUMapMode.READ),
    ]);

    const cls16Data = new Float32Array(cls16ReadBuf.getMappedRange()).slice();
    const reg16Data = new Float32Array(reg16ReadBuf.getMappedRange()).slice();
    const cls8Data = new Float32Array(cls8ReadBuf.getMappedRange()).slice();
    const reg8Data = new Float32Array(reg8ReadBuf.getMappedRange()).slice();

    cls16ReadBuf.unmap();
    reg16ReadBuf.unmap();
    cls8ReadBuf.unmap();
    reg8ReadBuf.unmap();

    // Combine outputs: SSD format is [anchors, values] but GPU outputs [channels, H, W]
    // We need to reorganize to [H*W*anchors_per_cell] order

    // 12x12 grid, 6 anchors per cell: cls is [6, 12, 12], reg is [108, 12, 12]
    // Need to reorder to [12, 12, 6] and [12, 12, 108] = per-pixel anchor ordering
    // Total: 12*12*6 = 864 anchors, each with 1 score and 18 reg values

    // 24x24 grid, 2 anchors per cell: cls is [2, 24, 24], reg is [36, 24, 24]
    // Total: 24*24*2 = 1152 anchors

    const totalAnchors = 864 + 1152;
    const scores = new Float32Array(totalAnchors);
    const regressors = new Float32Array(totalAnchors * 18);

    // Reorder 12x12 head: CHW → HWC
    let anchorIdx = 0;
    for (let y = 0; y < 12; y++) {
      for (let x = 0; x < 12; x++) {
        for (let a = 0; a < 6; a++) {
          // Score: cls16Data layout is [6, 12, 12] (CHW)
          scores[anchorIdx] = cls16Data[a * 144 + y * 12 + x];
          // Regressors: reg16Data layout is [108, 12, 12] (CHW)
          // 108 = 6 anchors × 18 values. Channels are interleaved: anchor0_val0, anchor0_val1, ..., anchor0_val17, anchor1_val0, ...
          for (let v = 0; v < 18; v++) {
            const ch = a * 18 + v;
            regressors[anchorIdx * 18 + v] = reg16Data[ch * 144 + y * 12 + x];
          }
          anchorIdx++;
        }
      }
    }

    // Reorder 24x24 head: CHW → HWC
    for (let y = 0; y < 24; y++) {
      for (let x = 0; x < 24; x++) {
        for (let a = 0; a < 2; a++) {
          scores[anchorIdx] = cls8Data[a * 576 + y * 24 + x];
          for (let v = 0; v < 18; v++) {
            const ch = a * 18 + v;
            regressors[anchorIdx * 18 + v] = reg8Data[ch * 576 + y * 24 + x];
          }
          anchorIdx++;
        }
      }
    }

    return { scores, regressors };
  }

  async function debugReadBuffer(buf: GPUBuffer, numFloats: number): Promise<Float32Array> {
    const readBuf = device.createBuffer({
      size: numFloats * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, readBuf, 0, numFloats * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readBuf.getMappedRange()).slice();
    readBuf.unmap();
    readBuf.destroy();
    return data;
  }

  async function debugRun(source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap): Promise<any> {
    // Run the full model step by step, reading back after key layers
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: canvasInputTexture },
      [192, 192],
    );

    function stats(data: Float32Array, n: number = 1000) {
      const d = data.slice(0, n);
      return {
        min: Math.min(...d),
        max: Math.max(...d),
        mean: d.reduce((a, b) => a + b, 0) / d.length,
        nonZero: d.filter(v => v !== 0).length,
        sample: Array.from(d.slice(0, 10)),
      };
    }

    const result: any = {};

    // Canvas → CHW
    const canvasUniform = makeUniform(new Uint32Array([192, 192, 192]));
    const canvasBG = device.createBindGroup({
      layout: canvasInputLayout,
      entries: [
        { binding: 0, resource: canvasInputTexture.createView() },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: canvasUniform } },
      ],
    });
    let enc = device.createCommandEncoder();
    let p = enc.beginComputePass();
    p.setPipeline(canvasInputPipe);
    p.setBindGroup(0, canvasBG);
    p.dispatchWorkgroups(ceil(192, 16), ceil(192, 16), 1);
    p.end();
    device.queue.submit([enc.finish()]);
    result.input = stats(await debugReadBuffer(inputBuf, 192 * 192 * 3));

    // Initial conv
    enc = device.createCommandEncoder();
    const bg2 = device.createBindGroup({
      layout: inputConvLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: initConvWeightBuf } },
        { binding: 2, resource: { buffer: initConvBiasBuf } },
        { binding: 3, resource: { buffer: initConvAlphaBuf } },
        { binding: 4, resource: { buffer: actBufA } },
        { binding: 5, resource: { buffer: inputConvUniform } },
      ],
    });
    p = enc.beginComputePass();
    p.setPipeline(inputConvPipe);
    p.setBindGroup(0, bg2);
    p.dispatchWorkgroups(ceil(96, 8), ceil(96, 8), 32);
    p.end();
    device.queue.submit([enc.finish()]);
    result.initConv = stats(await debugReadBuffer(actBufA, 96 * 96 * 32));

    // Run backbone blocks one at a time
    let curBuf = actBufA;
    let altBuf = actBufB;

    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i];
      enc = device.createCommandEncoder();
      encodeDwPwBlock(enc, block, curBuf, altBuf, curBuf);
      device.queue.submit([enc.finish()]);

      const tmp = curBuf;
      curBuf = altBuf;
      altBuf = tmp;

      // Read back after key blocks
      if (i === 0 || i === 3 || i === 7 || i === 11 || i === 17) {
        const outH = block.stride === 2 ? block.inH / 2 : block.inH;
        const size = outH * outH * block.outCh;
        result[`block${i}`] = stats(await debugReadBuffer(curBuf, size));
      }

      if (i === 10) {
        enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(curBuf, 0, backbone2SkipBuf, 0, 24 * 24 * 128 * 4);
        device.queue.submit([enc.finish()]);
      }
    }

    // Extra blocks
    enc = device.createCommandEncoder();
    encodeDwPwBlock(enc, extraBlockA, curBuf, altBuf, curBuf);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.extraBlockA = stats(await debugReadBuffer(curBuf, 12 * 12 * 256));

    enc = device.createCommandEncoder();
    encodeDwPwBlock(enc, extraBlockB, curBuf, altBuf, curBuf);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.extraBlockB = stats(await debugReadBuffer(curBuf, 12 * 12 * 256));

    // SSD head
    enc = device.createCommandEncoder();
    encodeConv1x1(enc, curBuf, cls16WBuf, cls16BBuf, cls16Buf, 256, 6, 12, 12);
    device.queue.submit([enc.finish()]);
    result.cls16 = stats(await debugReadBuffer(cls16Buf, 864));

    enc = device.createCommandEncoder();
    encodeConv1x1(enc, curBuf, reg16WBuf, reg16BBuf, reg16Buf, 256, 108, 12, 12);
    device.queue.submit([enc.finish()]);
    result.reg16 = stats(await debugReadBuffer(reg16Buf, 15552), 500);

    // Weight checks
    result.initWeights = stats(await debugReadBuffer(initConvWeightBuf, 100), 100);
    result.initBias = stats(await debugReadBuffer(initConvBiasBuf, 32), 32);
    result.cls16Weights = stats(await debugReadBuffer(cls16WBuf, 100), 100);
    result.cls16Bias = stats(await debugReadBuffer(cls16BBuf, 6), 6);

    return result;
  }

  return { device, run, debugRun };
}
