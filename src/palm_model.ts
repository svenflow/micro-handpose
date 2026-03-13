/**
 * Palm Detection WebGPU Model
 *
 * BlazeNet backbone with PReLU activations, FPN, and SSD output heads.
 *
 * Architecture:
 * 1. Initial conv 5x5 stride-2 + PReLU → 96x96x32
 * 2. Stage 1: 4 blocks (32ch), stride-2 transition → 48x48x64
 * 3. Stage 2: 4 blocks (64ch), stride-2 transition → 24x24x128 (save backbone24 skip)
 * 4. Stage 3: 4 blocks (128ch), stride-2 transition → 12x12x256 (save backbone12 skip after block 14)
 * 5. Stage 4a: 3 blocks (256ch) at 12x12
 * 6. Stage 4b: stride-2 transition → 6x6x256, then 3 more blocks at 6x6
 * 7. FPN Level 1: upsample 6→12 → conv2d_20 (256→256) → add backbone12 skip → 2 blocks at 12x12
 * 8. FPN Level 2: upsample 12→24 → conv2d_23 (256→128) → add backbone24 skip → 2 blocks at 24x24
 * 9. SSD heads:
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
    // Stage 4a: 256ch blocks, 12x12 input
    { dwKey: 'depthwise_conv2d_12/', pwKey: 'conv2d_13/', bnKey: 'batch_normalization_13/', preluKey: 'p_re_lu_13/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_13/', pwKey: 'conv2d_14/', bnKey: 'batch_normalization_14/', preluKey: 'p_re_lu_14/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    { dwKey: 'depthwise_conv2d_14/', pwKey: 'conv2d_15/', bnKey: 'batch_normalization_15/', preluKey: 'p_re_lu_15/', inCh: 256, outCh: 256, stride: 1, inH: 12 },
    // Stage 4b: stride-2 transition 12→6, then 256ch at 6x6
    { dwKey: 'depthwise_conv2d_15/', pwKey: 'conv2d_16/', bnKey: 'batch_normalization_16/', preluKey: 'p_re_lu_16/', inCh: 256, outCh: 256, stride: 2, inH: 12 },
    { dwKey: 'depthwise_conv2d_16/', pwKey: 'conv2d_17/', bnKey: 'batch_normalization_17/', preluKey: 'p_re_lu_17/', inCh: 256, outCh: 256, stride: 1, inH: 6 },
    { dwKey: 'depthwise_conv2d_17/', pwKey: 'conv2d_18/', bnKey: 'batch_normalization_18/', preluKey: 'p_re_lu_18/', inCh: 256, outCh: 256, stride: 1, inH: 6 },
    { dwKey: 'depthwise_conv2d_18/', pwKey: 'conv2d_19/', bnKey: 'batch_normalization_19/', preluKey: 'p_re_lu_19/', inCh: 256, outCh: 256, stride: 1, inH: 6 },
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
  //
  // ARCHITECTURE (verified against TFLite intermediate tensors):
  //
  // Backbone ends at 6x6x256 (block 17 = dw_18+conv2d_19)
  //
  // FPN Level 1 (6→12):
  //   1. Bilinear upsample 6x6→12x12 (256ch)
  //   2. conv2d_20 (256→256) + BN_20 + PReLU_20 on upsampled features
  //   3. Element-wise add backbone_12x12 skip (from block 14 = 12x12x256)
  //   4. Block: dw_19 + conv2d_21 + BN_21 + PReLU_21 (12x12x256, residual skip)
  //   5. Block: dw_20 + conv2d_22 + BN_22 + PReLU_22 (12x12x256, residual skip)
  //   6. 12x12 SSD heads on block 5 output
  //
  // FPN Level 2 (12→24):
  //   7. Bilinear upsample 12x12→24x24 (256ch) from block 5 output
  //   8. conv2d_23 (256→128) + BN_23 + PReLU_23 on upsampled features
  //   9. Element-wise add backbone_24x24 skip (from block 10 = 24x24x128)
  //  10. Block: dw_21 + conv2d_24 + BN_24 + PReLU_24 (24x24x128, residual skip)
  //  11. Block: dw_22 + conv2d_25 + BN_25 + PReLU_25 (24x24x128, residual skip)
  //  12. 24x24 SSD heads on block 11 output

  // FPN Level 1: conv2d_20 (256→256) applied after 6→12 upsample
  const fpn6to12W = transposePW(findWeight('conv2d_20/Conv2D'));
  const fpn6to12WBuf = makeBuf(fpn6to12W.byteLength, SC);
  writeBuf(fpn6to12WBuf, 0, fpn6to12W);
  const fpn6to12BBuf = uploadWeights(findWeight('batch_normalization_20/'));
  const fpn6to12AlphaBuf = uploadWeights(findWeight('p_re_lu_20/'));

  // FPN 12x12 block 1: dw_19 + conv2d_21
  const fpn12Block1 = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_19/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(256); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
    pwWeightBuf: (() => { const d = transposePW(findWeight('conv2d_21/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    pwBiasBuf: uploadWeights(findWeight('batch_normalization_21/')),
    alphaBuf: uploadWeights(findWeight('p_re_lu_21/')),
    inCh: 256, outCh: 256, stride: 1 as const, inH: 12,
  };

  // FPN 12x12 block 2: dw_20 + conv2d_22/Conv2D1
  const fpn12Block2 = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_20/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(256); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
    pwWeightBuf: (() => { const d = transposePW(findWeight('conv2d_22/Conv2D1')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    pwBiasBuf: uploadWeights(findWeight('batch_normalization_22/')),
    alphaBuf: uploadWeights(findWeight('p_re_lu_22/')),
    inCh: 256, outCh: 256, stride: 1 as const, inH: 12,
  };

  // FPN Level 2: conv2d_23 (256→128) applied after 12→24 upsample
  const fpn12to24W = transposePW(findWeight('conv2d_23/Conv2D'));
  const fpn12to24WBuf = makeBuf(fpn12to24W.byteLength, SC);
  writeBuf(fpn12to24WBuf, 0, fpn12to24W);
  const fpn12to24BBuf = uploadWeights(findWeight('batch_normalization_23/'));
  const fpn12to24AlphaBuf = uploadWeights(findWeight('p_re_lu_23/'));

  // FPN 24x24 block 1: dw_21 + conv2d_24
  const fpn24Block1 = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_21/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(128); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
    pwWeightBuf: (() => { const d = transposePW(findWeight('conv2d_24/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    pwBiasBuf: uploadWeights(findWeight('batch_normalization_24/')),
    alphaBuf: uploadWeights(findWeight('p_re_lu_24/')),
    inCh: 128, outCh: 128, stride: 1 as const, inH: 24,
  };

  // FPN 24x24 block 2: dw_22 + conv2d_25/Conv2D1
  const fpn24Block2 = {
    dwWeightBuf: (() => { const d = transposeDW(findWeight('depthwise_conv2d_22/')); const b = makeBuf(d.byteLength, SC); writeBuf(b, 0, d); return b; })(),
    dwBiasBuf: (() => { const z = new Float32Array(128); const b = makeBuf(z.byteLength, SC); writeBuf(b, 0, z); return b; })(),
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
  const buf192 = Math.max(192 * 192 * 3, 96 * 96 * 64, 48 * 48 * 128, 24 * 24 * 256, 12 * 12 * 256) * 4;

  const inputBuf = makeBuf(192 * 192 * 3 * 4, SC);
  const actBufA = makeBuf(buf192, SO);  // Activation buffer A
  const actBufB = makeBuf(buf192, SO);  // Activation buffer B
  const dwOutBuf = makeBuf(buf192, SO); // DW output buffer

  // Zero buffer for upsample-without-add (same size as largest upsample output)
  const zeroBuf = makeBuf(24 * 24 * 256 * 4, SO);  // Pre-zeroed by WebGPU

  // Save buffers for skip connections
  const backbone12SkipBuf = makeBuf(12 * 12 * 256 * 4, SO | GPUBufferUsage.COPY_DST); // After block 14 (12x12x256)
  const backbone24SkipBuf = makeBuf(24 * 24 * 128 * 4, SO | GPUBufferUsage.COPY_DST); // After block 10 (24x24x128)

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

  // Pre-create uniform buffers for backbone blocks (creating per-frame leaks GPU memory!)
  const blockUniforms = blocks.map(block => {
    const outH = block.stride === 2 ? block.inH / 2 : block.inH;
    const outW = outH;
    const pad = block.stride === 2 ? 1 : 2;
    const channelPad = block.inCh;
    return {
      dw: makeUniform(new Uint32Array([1, block.inCh, block.inH, block.inH, outH, outW, block.stride, pad])),
      pw: makeUniform(new Uint32Array([1, block.inCh, block.outCh, outH, outW, channelPad, block.stride, block.inH, block.inH])),
      outH, outW,
    };
  });

  // Pre-create uniform buffers for FPN blocks
  const fpn12Block1Uniforms = (() => {
    const b = fpn12Block1;
    const outH = b.stride === 2 ? b.inH / 2 : b.inH;
    const pad = b.stride === 2 ? 1 : 2;
    return {
      dw: makeUniform(new Uint32Array([1, b.inCh, b.inH, b.inH, outH, outH, b.stride, pad])),
      pw: makeUniform(new Uint32Array([1, b.inCh, b.outCh, outH, outH, b.inCh, b.stride, b.inH, b.inH])),
      outH,
    };
  })();
  const fpn12Block2Uniforms = (() => {
    const b = fpn12Block2;
    const outH = b.stride === 2 ? b.inH / 2 : b.inH;
    const pad = b.stride === 2 ? 1 : 2;
    return {
      dw: makeUniform(new Uint32Array([1, b.inCh, b.inH, b.inH, outH, outH, b.stride, pad])),
      pw: makeUniform(new Uint32Array([1, b.inCh, b.outCh, outH, outH, b.inCh, b.stride, b.inH, b.inH])),
      outH,
    };
  })();
  const fpn24Block1Uniforms = (() => {
    const b = fpn24Block1;
    const outH = b.stride === 2 ? b.inH / 2 : b.inH;
    const pad = b.stride === 2 ? 1 : 2;
    return {
      dw: makeUniform(new Uint32Array([1, b.inCh, b.inH, b.inH, outH, outH, b.stride, pad])),
      pw: makeUniform(new Uint32Array([1, b.inCh, b.outCh, outH, outH, b.inCh, b.stride, b.inH, b.inH])),
      outH,
    };
  })();
  const fpn24Block2Uniforms = (() => {
    const b = fpn24Block2;
    const outH = b.stride === 2 ? b.inH / 2 : b.inH;
    const pad = b.stride === 2 ? 1 : 2;
    return {
      dw: makeUniform(new Uint32Array([1, b.inCh, b.inH, b.inH, outH, outH, b.stride, pad])),
      pw: makeUniform(new Uint32Array([1, b.inCh, b.outCh, outH, outH, b.inCh, b.stride, b.inH, b.inH])),
      outH,
    };
  })();

  // Pre-create uniform buffers for FPN upsample/add steps and SSD heads
  const fpnUpsample6to12Uniform = makeUniform(new Uint32Array([1, 256, 6, 6, 12, 12]));
  const fpnAdd12Uniform = makeUniform(new Uint32Array([1, 256, 12, 12, 12, 12]));
  const fpnUpsample12to24Uniform = makeUniform(new Uint32Array([1, 256, 12, 12, 24, 24]));
  const fpnAdd24Uniform = makeUniform(new Uint32Array([1, 128, 24, 24, 24, 24]));
  const fpnConv6to12Uniform = makeUniform(new Uint32Array([1, 256, 256, 12, 12]));
  const fpnConv12to24Uniform = makeUniform(new Uint32Array([1, 256, 128, 24, 24]));
  const ssdCls16Uniform = makeUniform(new Uint32Array([1, 256, 6, 12, 12]));
  const ssdReg16Uniform = makeUniform(new Uint32Array([1, 256, 108, 12, 12]));
  const ssdCls8Uniform = makeUniform(new Uint32Array([1, 128, 2, 24, 24]));
  const ssdReg8Uniform = makeUniform(new Uint32Array([1, 128, 36, 24, 24]));

  // Pre-create canvas input bind group (texture view is stable)
  const canvasInputUniform = makeUniform(new Uint32Array([192, 192, 192]));
  const canvasInputBG = device.createBindGroup({
    layout: canvasInputLayout,
    entries: [
      { binding: 0, resource: canvasInputTexture.createView() },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: canvasInputUniform } },
    ],
  });

  // Pre-create initial conv bind group
  const initConvBG = device.createBindGroup({
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

  // Helper to encode a DW+PW block into the command encoder
  function encodeDwPwBlock(
    encoder: GPUCommandEncoder,
    block: BackboneBlock,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    skipBuffer: GPUBuffer,
    uniforms: { dw: GPUBuffer; pw: GPUBuffer; outH: number },
  ): void {
    const outW = uniforms.outH;

    const dwBG = device.createBindGroup({
      layout: dwLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: block.dwWeightBuf } },
        { binding: 2, resource: { buffer: block.dwBiasBuf } },
        { binding: 3, resource: { buffer: dwOutBuf } },
        { binding: 4, resource: { buffer: uniforms.dw } },
      ],
    });

    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(dwPipe);
    pass1.setBindGroup(0, dwBG);
    pass1.dispatchWorkgroups(ceil(outW, 8), ceil(uniforms.outH, 8), block.inCh);
    pass1.end();

    const pwBG = device.createBindGroup({
      layout: pwPreluLayout,
      entries: [
        { binding: 0, resource: { buffer: dwOutBuf } },
        { binding: 1, resource: { buffer: skipBuffer } },
        { binding: 2, resource: { buffer: block.pwWeightBuf } },
        { binding: 3, resource: { buffer: block.pwBiasBuf } },
        { binding: 4, resource: { buffer: block.alphaBuf } },
        { binding: 5, resource: { buffer: outputBuffer } },
        { binding: 6, resource: { buffer: uniforms.pw } },
      ],
    });

    const pass2 = encoder.beginComputePass();
    pass2.setPipeline(pwPreluPipe);
    pass2.setBindGroup(0, pwBG);
    pass2.dispatchWorkgroups(ceil(outW, 8), ceil(uniforms.outH, 8), block.outCh);
    pass2.end();
  }

  // Encode a conv1x1 (no activation) for SSD heads
  function encodeConv1x1(
    encoder: GPUCommandEncoder,
    inputBuffer: GPUBuffer,
    weightBuf: GPUBuffer,
    biasBuf: GPUBuffer,
    outputBuffer: GPUBuffer,
    uniform: GPUBuffer,
    outCh: number, h: number, w: number,
  ): void {
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
    uniform: GPUBuffer,
    outCh: number, h: number, w: number,
  ): void {
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

    const encoder = device.createCommandEncoder();

    // Canvas → CHW (using pre-created bind group)
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipe);
      pass.setBindGroup(0, canvasInputBG);
      pass.dispatchWorkgroups(ceil(192, 16), ceil(192, 16), 1);
      pass.end();
    }

    // Initial conv 5x5 stride 2 + PReLU: 192x192x3 → 96x96x32 (pre-created bind group)
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(inputConvPipe);
      pass.setBindGroup(0, initConvBG);
      pass.dispatchWorkgroups(ceil(96, 8), ceil(96, 8), 32);
      pass.end();
    }

    // Backbone blocks 0-18 with double buffering
    // After input conv: actBufA has 96x96x32
    let curBuf = actBufA;
    let altBuf = actBufB;

    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i];
      encodeDwPwBlock(encoder, block, curBuf, altBuf, curBuf, blockUniforms[i]);

      // Swap buffers
      const tmp = curBuf;
      curBuf = altBuf;
      altBuf = tmp;

      // Save backbone skip connections for FPN
      if (i === 10) {
        // After block 10: 24x24x128 — skip for FPN level 2 (12→24)
        encoder.copyBufferToBuffer(curBuf, 0, backbone24SkipBuf, 0, 24 * 24 * 128 * 4);
      }
      if (i === 14) {
        // After block 14: 12x12x256 — skip for FPN level 1 (6→12)
        encoder.copyBufferToBuffer(curBuf, 0, backbone12SkipBuf, 0, 12 * 12 * 256 * 4);
      }
    }

    // After backbone block 18: curBuf has 6x6x256

    // ============ FPN Level 1: 6→12 ============
    // Step 1: Upsample 6→12 (no add — use zeroBuf as skip)
    {
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: zeroBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: fpnUpsample6to12Uniform } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(12, 8), ceil(12, 8), 256);
      pass.end();
    }
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 2: conv1x1+PReLU (conv2d_20) on upsampled 12x12x256
    encodeConv1x1PReLU(encoder, curBuf, fpn6to12WBuf, fpn6to12BBuf, fpn6to12AlphaBuf, altBuf, fpnConv6to12Uniform, 256, 12, 12);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 3: Element-wise add backbone12Skip (identity upsample: 12→12)
    {
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: backbone12SkipBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: fpnAdd12Uniform } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(12, 8), ceil(12, 8), 256);
      pass.end();
    }
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 4: FPN 12x12 refinement blocks
    encodeDwPwBlock(encoder, fpn12Block1, curBuf, altBuf, curBuf, fpn12Block1Uniforms);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    encodeDwPwBlock(encoder, fpn12Block2, curBuf, altBuf, curBuf, fpn12Block2Uniforms);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 5: 12x12 SSD heads (on curBuf = 12x12x256)
    encodeConv1x1(encoder, curBuf, cls16WBuf, cls16BBuf, cls16Buf, ssdCls16Uniform, 6, 12, 12);
    encodeConv1x1(encoder, curBuf, reg16WBuf, reg16BBuf, reg16Buf, ssdReg16Uniform, 108, 12, 12);

    // ============ FPN Level 2: 12→24 ============
    // Step 6: Upsample 12→24 (no add — use zeroBuf as skip)
    {
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: zeroBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: fpnUpsample12to24Uniform } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(24, 8), ceil(24, 8), 256);
      pass.end();
    }
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 7: conv1x1+PReLU (conv2d_23) on upsampled 24x24: 256→128
    encodeConv1x1PReLU(encoder, curBuf, fpn12to24WBuf, fpn12to24BBuf, fpn12to24AlphaBuf, altBuf, fpnConv12to24Uniform, 128, 24, 24);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 8: Element-wise add backbone24Skip (identity upsample: 24→24)
    {
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: backbone24SkipBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: fpnAdd24Uniform } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(24, 8), ceil(24, 8), 128);
      pass.end();
    }
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 9: FPN 24x24 refinement blocks
    encodeDwPwBlock(encoder, fpn24Block1, curBuf, altBuf, curBuf, fpn24Block1Uniforms);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    encodeDwPwBlock(encoder, fpn24Block2, curBuf, altBuf, curBuf, fpn24Block2Uniforms);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }

    // Step 10: 24x24 SSD heads (on curBuf = 24x24x128)
    encodeConv1x1(encoder, curBuf, cls8WBuf, cls8BBuf, cls8Buf, ssdCls8Uniform, 2, 24, 24);
    encodeConv1x1(encoder, curBuf, reg8WBuf, reg8BBuf, reg8Buf, ssdReg8Uniform, 36, 24, 24);

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
      encodeDwPwBlock(enc, block, curBuf, altBuf, curBuf, blockUniforms[i]);
      device.queue.submit([enc.finish()]);

      const tmp = curBuf;
      curBuf = altBuf;
      altBuf = tmp;

      // Read back after key blocks
      if (i === 0 || i === 3 || i === 7 || i === 11 || i === 14 || i === 15 || i === 18) {
        const outH = block.stride === 2 ? block.inH / 2 : block.inH;
        const size = outH * outH * block.outCh;
        result[`block${i}`] = stats(await debugReadBuffer(curBuf, size));
      }

      // Save backbone skip connections for FPN
      if (i === 10) {
        enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(curBuf, 0, backbone24SkipBuf, 0, 24 * 24 * 128 * 4);
        device.queue.submit([enc.finish()]);
      }
      if (i === 14) {
        enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(curBuf, 0, backbone12SkipBuf, 0, 12 * 12 * 256 * 4);
        device.queue.submit([enc.finish()]);
      }
    }

    // ============ FPN Level 1: 6→12 ============

    // Upsample 6→12 (no add)
    enc = device.createCommandEncoder();
    {
      const uniform = makeUniform(new Uint32Array([1, 256, 6, 6, 12, 12]));
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: zeroBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: uniform } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(12, 8), ceil(12, 8), 256);
      pass.end();
    }
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpnUpsample6to12 = stats(await debugReadBuffer(curBuf, 12 * 12 * 256));

    // conv1x1+PReLU (conv2d_20) on upsampled 12x12x256
    enc = device.createCommandEncoder();
    encodeConv1x1PReLU(enc, curBuf, fpn6to12WBuf, fpn6to12BBuf, fpn6to12AlphaBuf, altBuf, fpnConv6to12Uniform, 256, 12, 12);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpn6to12Conv = stats(await debugReadBuffer(curBuf, 12 * 12 * 256));

    // backbone12 skip check
    result.backbone12Skip = stats(await debugReadBuffer(backbone12SkipBuf, 12 * 12 * 256));

    // Element-wise add backbone12Skip (identity upsample 12→12)
    enc = device.createCommandEncoder();
    {
      const uniform = makeUniform(new Uint32Array([1, 256, 12, 12, 12, 12]));
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: backbone12SkipBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: uniform } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(12, 8), ceil(12, 8), 256);
      pass.end();
    }
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpnAdd12 = stats(await debugReadBuffer(curBuf, 12 * 12 * 256));

    // FPN 12x12 block 1 (dw_19 + conv2d_21)
    enc = device.createCommandEncoder();
    encodeDwPwBlock(enc, fpn12Block1, curBuf, altBuf, curBuf, fpn12Block1Uniforms);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpn12Block1 = stats(await debugReadBuffer(curBuf, 12 * 12 * 256));

    // FPN 12x12 block 2 (dw_20 + conv2d_22)
    enc = device.createCommandEncoder();
    encodeDwPwBlock(enc, fpn12Block2, curBuf, altBuf, curBuf, fpn12Block2Uniforms);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpn12Block2 = stats(await debugReadBuffer(curBuf, 12 * 12 * 256));

    // 12x12 SSD heads
    enc = device.createCommandEncoder();
    encodeConv1x1(enc, curBuf, cls16WBuf, cls16BBuf, cls16Buf, ssdCls16Uniform, 6, 12, 12);
    device.queue.submit([enc.finish()]);
    result.cls16 = stats(await debugReadBuffer(cls16Buf, 864));

    enc = device.createCommandEncoder();
    encodeConv1x1(enc, curBuf, reg16WBuf, reg16BBuf, reg16Buf, ssdReg16Uniform, 108, 12, 12);
    device.queue.submit([enc.finish()]);
    result.reg16 = stats(await debugReadBuffer(reg16Buf, 15552), 500);

    // ============ FPN Level 2: 12→24 ============

    // Upsample 12→24 (no add — 256ch from fpn12Block2 output)
    enc = device.createCommandEncoder();
    {
      const uniform = makeUniform(new Uint32Array([1, 256, 12, 12, 24, 24]));
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: zeroBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: uniform } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(24, 8), ceil(24, 8), 256);
      pass.end();
    }
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpnUpsample12to24 = stats(await debugReadBuffer(curBuf, 24 * 24 * 256));

    // conv1x1+PReLU (conv2d_23) on upsampled 24x24: 256→128
    enc = device.createCommandEncoder();
    encodeConv1x1PReLU(enc, curBuf, fpn12to24WBuf, fpn12to24BBuf, fpn12to24AlphaBuf, altBuf, 256, 128, 24, 24);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpn12to24Conv = stats(await debugReadBuffer(curBuf, 24 * 24 * 128));

    // backbone24 skip check
    result.backbone24Skip = stats(await debugReadBuffer(backbone24SkipBuf, 24 * 24 * 128));

    // Element-wise add backbone24Skip (identity upsample 24→24)
    enc = device.createCommandEncoder();
    {
      const uniform = makeUniform(new Uint32Array([1, 128, 24, 24, 24, 24]));
      const bg = device.createBindGroup({
        layout: upsampleAddLayout,
        entries: [
          { binding: 0, resource: { buffer: curBuf } },
          { binding: 1, resource: { buffer: backbone24SkipBuf } },
          { binding: 2, resource: { buffer: altBuf } },
          { binding: 3, resource: { buffer: uniform } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(upsampleAddPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(ceil(24, 8), ceil(24, 8), 128);
      pass.end();
    }
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpnAdd24 = stats(await debugReadBuffer(curBuf, 24 * 24 * 128));

    // FPN 24x24 block 1 (dw_21 + conv2d_24)
    enc = device.createCommandEncoder();
    encodeDwPwBlock(enc, fpn24Block1, curBuf, altBuf, curBuf, fpn24Block1Uniforms);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpn24Block1 = stats(await debugReadBuffer(curBuf, 24 * 24 * 128));

    // FPN 24x24 block 2 (dw_22 + conv2d_25)
    enc = device.createCommandEncoder();
    encodeDwPwBlock(enc, fpn24Block2, curBuf, altBuf, curBuf, fpn24Block2Uniforms);
    device.queue.submit([enc.finish()]);
    { const tmp = curBuf; curBuf = altBuf; altBuf = tmp; }
    result.fpn24Block2 = stats(await debugReadBuffer(curBuf, 24 * 24 * 128));

    // 24x24 SSD heads
    enc = device.createCommandEncoder();
    encodeConv1x1(enc, curBuf, cls8WBuf, cls8BBuf, cls8Buf, 128, 2, 24, 24);
    device.queue.submit([enc.finish()]);
    result.cls8 = stats(await debugReadBuffer(cls8Buf, 24 * 24 * 2));

    enc = device.createCommandEncoder();
    encodeConv1x1(enc, curBuf, reg8WBuf, reg8BBuf, reg8Buf, 128, 36, 24, 24);
    device.queue.submit([enc.finish()]);
    result.reg8 = stats(await debugReadBuffer(reg8Buf, 24 * 24 * 36));

    // Weight checks
    result.initWeights = stats(await debugReadBuffer(initConvWeightBuf, 100), 100);
    result.initBias = stats(await debugReadBuffer(initConvBiasBuf, 32), 32);
    result.cls16Weights = stats(await debugReadBuffer(cls16WBuf, 100), 100);
    result.cls16Bias = stats(await debugReadBuffer(cls16BBuf, 6), 6);
    result.cls8Weights = stats(await debugReadBuffer(cls8WBuf, 100), 100);
    result.cls8Bias = stats(await debugReadBuffer(cls8BBuf, 2), 2);
    result.fpn6to12Weights = stats(await debugReadBuffer(fpn6to12WBuf, 100), 100);

    return result;
  }

  return { device, run, debugRun };
}
