/**
 * FULL Hand Landmark Model — EfficientNet-B0-like architecture
 *
 * Input: 224x224x3
 * Architecture: Initial Conv + 16 MBConv blocks + Global Average Pool + FC heads
 * Output: landmarks(63), world_landmarks(63), handflag(1), handedness(1)
 */

import type { Tensor, HandLandmarksOutput, CompiledModel, WeightsMetadata } from './model.js';

/**
 * Load weights from buffer, handling duplicate keys by keeping ALL entries.
 * The standard loadWeightsFromBuffer loses duplicate keys (conv_landmarks appears
 * twice — once as bias [63] and once as weight [63,1152]). This version appends
 * a shape suffix for disambiguation.
 */
export function loadFullWeightsFromBuffer(
  metadata: WeightsMetadata,
  buffer: ArrayBuffer,
): Map<string, Tensor> {
  const weights = new Map<string, Tensor>();
  const dtype = metadata.dtype ?? 'float32';
  const keyCounts = new Map<string, number>();

  for (let i = 0; i < metadata.keys.length; i++) {
    const baseKey = metadata.keys[i]!;
    const shape = metadata.shapes[i]!;
    const offset = metadata.offsets[i]!;
    const size = shape.reduce((a, b) => a * b, 1);

    let data: Float32Array;
    let rawF16: ArrayBufferLike | undefined;
    if (dtype === 'float32') {
      data = new Float32Array(buffer, offset, size);
    } else {
      const view = new DataView(buffer);
      data = new Float32Array(size);
      for (let j = 0; j < size; j++) {
        data[j] = float16ToFloat32Full(view.getUint16(offset + j * 2, true));
      }
      rawF16 = buffer.slice(offset, offset + size * 2);
    }

    // Use shape-based disambiguation for duplicate keys
    const count = keyCounts.get(baseKey) ?? 0;
    keyCounts.set(baseKey, count + 1);
    const key = count === 0 ? baseKey : `${baseKey}__${count}`;

    weights.set(key, { data, shape, rawF16 });
  }

  return weights;
}

function float16ToFloat32Full(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa = h & 0x3ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  }
  if (exponent === 0x1f) {
    if (mantissa === 0) return sign ? -Infinity : Infinity;
    return NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

// ============ Shader code (minified inline) ============
function S(s: string): string {
  return s.replace(/\/\/[^\n]*/g, '').replace(/\s+/g, ' ').replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g, '$1').trim();
}

// Canvas to NCHW 224x224 (no padding needed for FULL model)
const CANVAS_INPUT_224_SHADER = S(`
struct CanvasParams { in_size:u32, }
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CanvasParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y;
  if(x>=params.in_size||y>=params.in_size){return;}
  let pixel=textureLoad(input_tex,vec2<u32>(x,y),0);
  let stride=params.in_size*params.in_size;
  output[0u*stride+y*params.in_size+x]=pixel.r;
  output[1u*stride+y*params.in_size+x]=pixel.g;
  output[2u*stride+y*params.in_size+x]=pixel.b;
}
`);

// Initial conv 3x3 stride 2 + BN + ReLU6
// Input: NCHW [1, 3, 224, 224], Output: [1, 24, 112, 112]
// TFLite SAME padding for stride 2: pad_before=0, pad_after=1 (asymmetric)
const CONV3X3_S2_BN_RELU6_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    for(var ky:u32=0u;ky<3u;ky++){
      for(var kx:u32=0u;kx<3u;kx++){
        let iy=i32(out_y*2u+ky); let ix=i32(out_x*2u+kx);
        if(iy>=0 && iy<in_h && ix>=0 && ix<in_w){
          let in_idx=ic*params.in_h*params.in_w+u32(iy)*params.in_w+u32(ix);
          let w_idx=oc*params.in_channels*9u+ic*9u+ky*3u+kx;
          sum+=input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum+=bias[oc];
  sum=min(max(sum,0.0),6.0);
  let out_idx=oc*params.out_h*params.out_w+out_y*params.out_w+out_x;
  output[out_idx]=sum;
}
`);

// Expand 1x1 conv + BN + ReLU6
const EXPAND_1X1_BN_RELU6_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  sum=min(max(sum,0.0),6.0);
  output[oc*spatial+pix]=sum;
}
`);

// Depthwise conv kxk + BN + ReLU6 (parameterized kernel size)
const DEPTHWISE_BN_RELU6_SHADER = S(`
struct Params { channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, stride:u32, pad:u32, kernel:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||c>=params.channels){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  let kk=params.kernel*params.kernel;
  for(var ky:u32=0u;ky<params.kernel;ky++){
    for(var kx:u32=0u;kx<params.kernel;kx++){
      let iy=i32(out_y*params.stride+ky)-i32(params.pad);
      let ix=i32(out_x*params.stride+kx)-i32(params.pad);
      if(iy>=0 && iy<in_h && ix>=0 && ix<in_w){
        sum+=input[c*params.in_h*params.in_w+u32(iy)*params.in_w+u32(ix)]*weight[c*kk+ky*params.kernel+kx];
      }
    }
  }
  sum+=bias[c];
  sum=min(max(sum,0.0),6.0);
  output[c*params.out_h*params.out_w+out_y*params.out_w+out_x]=sum;
}
`);

// Project 1x1 conv + BN (NO activation)
const PROJECT_1X1_BN_SHADER = S(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  output[oc*spatial+pix]=sum;
}
`);

// Element-wise add (residual connection)
const ADD_RESIDUAL_SHADER = S(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`);

// Global average pooling: [1, C, H, W] → [1, C]
const GLOBAL_AVG_POOL_SHADER = S(`
struct Params { channels:u32, spatial:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let c=gid.x;
  if(c>=params.channels){return;}
  var sum:f32=0.0;
  let base=c*params.spatial;
  for(var i:u32=0u;i<params.spatial;i++){
    sum+=input[base+i];
  }
  output[c]=sum/f32(params.spatial);
}
`);

// FC MatMul + bias (for landmarks, world_landmarks)
const FC_MATMUL_SHADER = S(`
struct Params { in_features:u32, out_features:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  if(oc>=params.out_features){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_features;ic++){
    sum+=input[ic]*weight[oc*params.in_features+ic];
  }
  output[oc]=sum+bias[oc];
}
`);

// FC MatMul + bias + sigmoid (for handflag, handedness)
const FC_SIGMOID_SHADER = S(`
struct Params { in_features:u32, out_features:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  if(oc>=params.out_features){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_features;ic++){
    sum+=input[ic]*weight[oc*params.in_features+ic];
  }
  sum+=bias[oc];
  output[oc]=1.0/(1.0+exp(-sum));
}
`);

// ============ Block specification ============
interface MBConvBlockSpec {
  inCh: number;
  expandCh: number;
  dwKernel: number;
  stride: number;
  outCh: number;
  hasResidual: boolean;
  hasProject: boolean; // false for block 16 (special)
}

const BLOCK_SPECS: MBConvBlockSpec[] = [
  { inCh: 24,  expandCh: 24,   dwKernel: 3, stride: 1, outCh: 16,   hasResidual: false, hasProject: true },
  { inCh: 16,  expandCh: 64,   dwKernel: 3, stride: 2, outCh: 24,   hasResidual: false, hasProject: true },
  { inCh: 24,  expandCh: 144,  dwKernel: 3, stride: 1, outCh: 24,   hasResidual: true,  hasProject: true },
  { inCh: 24,  expandCh: 144,  dwKernel: 5, stride: 2, outCh: 40,   hasResidual: false, hasProject: true },
  { inCh: 40,  expandCh: 240,  dwKernel: 5, stride: 1, outCh: 40,   hasResidual: true,  hasProject: true },
  { inCh: 40,  expandCh: 240,  dwKernel: 3, stride: 2, outCh: 80,   hasResidual: false, hasProject: true },
  { inCh: 80,  expandCh: 480,  dwKernel: 3, stride: 1, outCh: 80,   hasResidual: true,  hasProject: true },
  { inCh: 80,  expandCh: 480,  dwKernel: 3, stride: 1, outCh: 80,   hasResidual: true,  hasProject: true },
  { inCh: 80,  expandCh: 480,  dwKernel: 5, stride: 1, outCh: 112,  hasResidual: false, hasProject: true },
  { inCh: 112, expandCh: 672,  dwKernel: 5, stride: 1, outCh: 112,  hasResidual: true,  hasProject: true },
  { inCh: 112, expandCh: 672,  dwKernel: 5, stride: 1, outCh: 112,  hasResidual: true,  hasProject: true },
  { inCh: 112, expandCh: 672,  dwKernel: 5, stride: 2, outCh: 192,  hasResidual: false, hasProject: true },
  { inCh: 192, expandCh: 1152, dwKernel: 5, stride: 1, outCh: 192,  hasResidual: true,  hasProject: true },
  { inCh: 192, expandCh: 1152, dwKernel: 5, stride: 1, outCh: 192,  hasResidual: true,  hasProject: true },
  { inCh: 192, expandCh: 1152, dwKernel: 5, stride: 1, outCh: 192,  hasResidual: true,  hasProject: true },
  { inCh: 192, expandCh: 1152, dwKernel: 3, stride: 1, outCh: 1152, hasResidual: false, hasProject: false },
];

// ============ Weight name mapping ============

// Each block's weight names. Derived from the actual TFLite weight manifest.
// Block indices are 0-based (Block 1 in spec = index 0 here)
//
// Pattern:
// - Initial conv: conv2d, batch_normalization
// - Block N expand 1x1: conv2d_{X} where X is the conv index
// - Block N DW: batch_normalization_{Y}/FusedBatchNormV3 (for weights), batch_normalization_{Y} (for bias)
// - Block N project 1x1: conv2d_{X+1}, batch_normalization_{Z}/FusedBatchNormV3 (for bias)
//
// Block 1 (expandCh==inCh=24): no expand conv, just DW + project
// Block 16 (special): expand + DW only, no project

interface BlockWeightNames {
  expandConvKey?: string;   // conv2d key for expand 1x1 (undefined if expandCh==inCh)
  expandBNKey?: string;     // BN bias key for expand
  dwWeightKey: string;      // DW weight key (FusedBatchNormV3)
  dwBNKey: string;          // DW BN bias key
  projectConvKey?: string;  // conv2d key for project 1x1 (undefined for block 16)
  projectBNKey?: string;    // BN bias key for project
}

const BLOCK_WEIGHT_NAMES: BlockWeightNames[] = [
  // Block 1: no expand (expandCh == inCh = 24)
  {
    dwWeightKey: 'batch_normalization_1/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_1',
    projectConvKey: 'conv2d_1',
    projectBNKey: 'batch_normalization_2/FusedBatchNormV3',
  },
  // Block 2
  {
    expandConvKey: 'conv2d_2',
    expandBNKey: 'batch_normalization_3',
    dwWeightKey: 'batch_normalization_4/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_4',
    projectConvKey: 'conv2d_3',
    projectBNKey: 'batch_normalization_5/FusedBatchNormV3',
  },
  // Block 3
  {
    expandConvKey: 'conv2d_4',
    expandBNKey: 'batch_normalization_6',
    dwWeightKey: 'batch_normalization_7/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_7',
    projectConvKey: 'conv2d_5',
    projectBNKey: 'batch_normalization_8/FusedBatchNormV3',
  },
  // Block 4
  {
    expandConvKey: 'conv2d_6',
    expandBNKey: 'batch_normalization_9',
    dwWeightKey: 'batch_normalization_10/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_10',
    projectConvKey: 'conv2d_7',
    projectBNKey: 'batch_normalization_11/FusedBatchNormV3',
  },
  // Block 5
  {
    expandConvKey: 'conv2d_8',
    expandBNKey: 'batch_normalization_12',
    dwWeightKey: 'batch_normalization_13/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_13',
    projectConvKey: 'conv2d_9',
    projectBNKey: 'batch_normalization_14/FusedBatchNormV3',
  },
  // Block 6
  {
    expandConvKey: 'conv2d_10',
    expandBNKey: 'batch_normalization_15',
    dwWeightKey: 'batch_normalization_16/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_16',
    projectConvKey: 'conv2d_11',
    projectBNKey: 'batch_normalization_17/FusedBatchNormV3',
  },
  // Block 7
  {
    expandConvKey: 'conv2d_12',
    expandBNKey: 'batch_normalization_18',
    dwWeightKey: 'batch_normalization_19/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_19',
    projectConvKey: 'conv2d_13',
    projectBNKey: 'batch_normalization_20/FusedBatchNormV3',
  },
  // Block 8
  {
    expandConvKey: 'conv2d_14',
    expandBNKey: 'batch_normalization_21',
    dwWeightKey: 'batch_normalization_22/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_22',
    projectConvKey: 'conv2d_15',
    projectBNKey: 'batch_normalization_23/FusedBatchNormV3',
  },
  // Block 9
  {
    expandConvKey: 'conv2d_16',
    expandBNKey: 'batch_normalization_24',
    dwWeightKey: 'batch_normalization_25/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_25',
    projectConvKey: 'conv2d_17',
    projectBNKey: 'batch_normalization_26/FusedBatchNormV3',
  },
  // Block 10
  {
    expandConvKey: 'conv2d_18',
    expandBNKey: 'batch_normalization_27',
    dwWeightKey: 'batch_normalization_28/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_28',
    projectConvKey: 'conv2d_19',
    projectBNKey: 'batch_normalization_29/FusedBatchNormV3',
  },
  // Block 11
  {
    expandConvKey: 'conv2d_20',
    expandBNKey: 'batch_normalization_30',
    dwWeightKey: 'batch_normalization_31/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_31',
    projectConvKey: 'conv2d_21',
    projectBNKey: 'batch_normalization_32/FusedBatchNormV3',
  },
  // Block 12
  {
    expandConvKey: 'conv2d_22',
    expandBNKey: 'batch_normalization_33',
    dwWeightKey: 'batch_normalization_34/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_34',
    projectConvKey: 'conv2d_23',
    projectBNKey: 'batch_normalization_35/FusedBatchNormV3',
  },
  // Block 13
  {
    expandConvKey: 'conv2d_24',
    expandBNKey: 'batch_normalization_36',
    dwWeightKey: 'batch_normalization_37/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_37',
    projectConvKey: 'conv2d_25',
    projectBNKey: 'batch_normalization_38/FusedBatchNormV3',
  },
  // Block 14
  {
    expandConvKey: 'conv2d_26',
    expandBNKey: 'batch_normalization_39',
    dwWeightKey: 'batch_normalization_40/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_40',
    projectConvKey: 'conv2d_27',
    projectBNKey: 'batch_normalization_41/FusedBatchNormV3',
  },
  // Block 15
  {
    expandConvKey: 'conv2d_28',
    expandBNKey: 'batch_normalization_42',
    dwWeightKey: 'batch_normalization_43/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_43',
    projectConvKey: 'conv2d_29',
    projectBNKey: 'batch_normalization_44/FusedBatchNormV3',
  },
  // Block 16 (special: expand + DW only, no project)
  {
    expandConvKey: 'conv2d_30',
    expandBNKey: 'batch_normalization_45',
    dwWeightKey: 'batch_normalization_46/FusedBatchNormV3',
    dwBNKey: 'batch_normalization_46',
    // no project
  },
];

// ============ Compile the FULL model ============
export async function compileFullModel(
  weights: Map<string, Tensor>,
  options?: { forceF32?: boolean },
): Promise<CompiledModel> {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No GPU adapter found');
  }

  const hasF16 = adapter.features.has('shader-f16');
  const requiredFeatures: GPUFeatureName[] = hasF16 ? ['shader-f16' as GPUFeatureName] : [];

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(
        adapter.limits.maxStorageBuffersPerShaderStage, 8,
      ),
    },
  });

  // Check f16 support (simplified — skip full validation for now)
  const firstWeight = weights.values().next().value as Tensor | undefined;
  const useF16 = hasF16 && !!firstWeight?.rawF16 && !options?.forceF32;
  // Debug: console.log('[micro-handpose] useF16:', useF16, 'hasF16:', hasF16, 'rawF16:', !!firstWeight?.rawF16, 'forceF32:', options?.forceF32);

  function getWeightData(tensor: Tensor): ArrayBufferView {
    if (useF16 && tensor.rawF16) {
      const u16 = new Uint16Array(tensor.rawF16);
      if (u16.length % 2 !== 0) {
        const padded = new Uint16Array(u16.length + 1);
        padded.set(u16);
        return padded;
      }
      return u16;
    }
    return tensor.data;
  }

  function getWeightBufSize(tensor: Tensor): number {
    if (useF16 && tensor.rawF16) {
      return Math.ceil(tensor.rawF16.byteLength / 4) * 4;
    }
    return tensor.data.byteLength;
  }

  // Bytes per element: 2 for f16, 4 for f32
  const bpe = useF16 ? 2 : 4;

  /**
   * Full f16 compute shader: ALL arrays become f16, accumulators become f16,
   * literals become h-suffixed. Matches MediaPipe's mediump behavior.
   */
  function f16ComputeShader(code: string): string {
    if (!useF16) return code;
    let s = code;
    // Convert all storage arrays from f32 to f16
    s = s.replace(/array<f32>/g, 'array<f16>');
    s = s.replace(/array<f32,/g, 'array<f16,');
    // Convert accumulator declarations
    s = s.replace(/var sum:f32=0\.0/g, 'var sum:f16=0.0h');
    s = s.replace(/var sum0:f32=0\.0/g, 'var sum0:f16=0.0h');
    s = s.replace(/var sum1:f32=0\.0/g, 'var sum1:f16=0.0h');
    s = s.replace(/var sum2:f32=0\.0/g, 'var sum2:f16=0.0h');
    s = s.replace(/var sum3:f32=0\.0/g, 'var sum3:f16=0.0h');
    // Convert f32 casts to f16 in division (GAP: sum/f32(x) -> sum/f16(x))
    s = s.replace(/\/f32\(params/g, '/f16(params');
    // Convert clamp literals: min(max(sum,0.0),6.0) -> min(max(sum,0.0h),6.0h)
    // After minification: min(max(sum,0.0),6.0)
    s = s.replace(/,0\.0\),6\.0\)/g, ',0.0h),6.0h)');
    // Convert helper function return types
    s = s.replace(/->f32\{/g, '->f16{');
    s = s.replace(/->f32 \{/g, '->f16 {');
    // Convert return 0.0 to return 0.0h in helper functions
    s = s.replace(/return 0\.0;/g, 'return 0.0h;');
    return 'enable f16;' + s;
  }

  /**
   * Initial conv f16 shader: input stays array<f32> (reads from f32 crop/canvas buffer),
   * but weights, bias, output, and computation are all f16.
   */
  function f16InitConvShader(code: string): string {
    if (!useF16) return code;
    // First apply full f16 conversion
    let s = f16ComputeShader(code);
    // Restore the FIRST storage binding (input) back to f32.
    // After f16ComputeShader, pattern is: read>input:array<f16>
    // We need to revert just the input binding.
    s = s.replace('read>input:array<f16>', 'read>input:array<f32>');
    // Input reads need f16 cast: input[in_idx] -> f16(input[in_idx])
    // After minification the pattern is: input[in_idx]*weight[w_idx]
    s = s.replace(/input\[in_idx\]/g, 'f16(input[in_idx])');
    return s;
  }

  /**
   * FC head shader (f32 compute, matching MediaPipe highp):
   * Input is f16 (from GAP), weights/bias are f16, but computation is f32, output is f32.
   * This is essentially what applyF16Weights does, plus converting input to f16.
   */
  function f16FcShader(code: string): string {
    if (!useF16) return code;
    let s = code;
    // Convert input, weight, bias arrays to f16 (they store f16 data)
    s = s.replace('read>input:array<f32>', 'read>input:array<f16>');
    s = s.replace('read>weight:array<f32>', 'read>weight:array<f16>');
    s = s.replace('read>bias:array<f32>', 'read>bias:array<f16>');
    // Output stays f32, computation stays f32 (var sum:f32=0.0)
    // WGSL does NOT auto-promote f16 to f32 — must cast explicitly
    // FC matmul: sum+=input[ic]*weight[oc*params.in_features+ic]
    s = s.replace(/input\[ic\]/g, 'f32(input[ic])');
    s = s.replace(/weight\[oc\*params\.in_features\+ic\]/g, 'f32(weight[oc*params.in_features+ic])');
    // FC matmul: output[oc]=sum+bias[oc]
    s = s.replace(/bias\[oc\]/g, 'f32(bias[oc])');
    return 'enable f16;' + s;
  }

  // Helper functions
  const BT: Record<string, GPUBufferBindingType> = { r: 'read-only-storage', s: 'storage', u: 'uniform' };
  function makeLayout(types: string[]): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      entries: types.map((t, i) => {
        if (t === 't') return { binding: i, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' as const } };
        return { binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: BT[t]! } };
      }),
    });
  }
  const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const SO = GPUBufferUsage.STORAGE;
  const SOC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const SOCD = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const UC = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  function makeBuf(size: number, usage: number): GPUBuffer {
    return device.createBuffer({ size: Math.max(size, 4), usage });
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

  function requireWeight(key: string): Tensor {
    const t = weights.get(key);
    if (!t) throw new Error(`Missing weight: ${key}`);
    return t;
  }

  // ============ Create Shader Modules ============
  const canvasInputShaderMod = device.createShaderModule({ code: CANVAS_INPUT_224_SHADER });
  const initConvShaderMod = device.createShaderModule({ code: f16InitConvShader(CONV3X3_S2_BN_RELU6_SHADER) });
  const expand1x1ShaderMod = device.createShaderModule({ code: f16ComputeShader(EXPAND_1X1_BN_RELU6_SHADER) });
  const dwShaderMod = device.createShaderModule({ code: f16ComputeShader(DEPTHWISE_BN_RELU6_SHADER) });
  const project1x1ShaderMod = device.createShaderModule({ code: f16ComputeShader(PROJECT_1X1_BN_SHADER) });
  const addResShaderMod = device.createShaderModule({ code: f16ComputeShader(ADD_RESIDUAL_SHADER) });
  const gapShaderMod = device.createShaderModule({ code: f16ComputeShader(GLOBAL_AVG_POOL_SHADER) });
  const fcShaderMod = device.createShaderModule({ code: f16FcShader(FC_MATMUL_SHADER) });
  const fcSigmoidShaderMod = device.createShaderModule({ code: f16FcShader(FC_SIGMOID_SHADER) });

  // ============ Create Layouts ============
  const conv5Layout = makeLayout(['r', 'r', 'r', 's', 'u']); // input, weight, bias, output, params
  const addLayout = makeLayout(['r', 'r', 's', 'u']);
  const gapLayout = makeLayout(['r', 's', 'u']);
  const fcLayout = makeLayout(['r', 'r', 'r', 's', 'u']);
  const canvasLayout = makeLayout(['t', 's', 'u']);

  // ============ Create Pipelines ============
  const canvasInputPipeline = makePipe(canvasLayout, canvasInputShaderMod);
  const initConvPipeline = makePipe(conv5Layout, initConvShaderMod);
  const expand1x1Pipeline = makePipe(conv5Layout, expand1x1ShaderMod);
  const dwPipeline = makePipe(conv5Layout, dwShaderMod);
  const project1x1Pipeline = makePipe(conv5Layout, project1x1ShaderMod);
  const addResPipeline = makePipe(addLayout, addResShaderMod);
  const gapPipeline = makePipe(gapLayout, gapShaderMod);
  const fcPipeline = makePipe(fcLayout, fcShaderMod);
  const fcSigmoidPipeline = makePipe(fcLayout, fcSigmoidShaderMod);

  // ============ Allocate Buffers ============
  // We use a ping-pong pattern with two large buffers
  // Max intermediate size: 1152 channels * 7 * 7 = 56448 floats
  // But we also need to handle the largest expand output: 1152 * 7 * 7 = 56448
  // Actually largest is 24 * 112 * 112 = 301056 after initial conv
  const maxElements = 1152 * 112 * 112; // generous upper bound
  const maxSize = maxElements * 4;

  const bufA = makeBuf(maxSize, SOCD);  // primary buffer (needs CopySrc+CopyDst for ping-pong + copy)
  const bufB = makeBuf(maxSize, SOCD);  // secondary buffer
  const bufExpand = makeBuf(maxSize, SO); // expand output / DW input
  const bufDW = makeBuf(maxSize, SO);     // DW output
  const bufResidual = makeBuf(maxSize, SC); // for saving input before block (residual)

  // Input buffer for raw 224x224x3 (needs CopySrc for copyToBuffer, CopyDst for writeBuffer)
  const bufRawInput = makeBuf(3 * 224 * 224 * 4, SOCD);

  // GAP output: [1, 1152]
  const bufGAP = makeBuf(1152 * 4, SOC);

  // Output buffers
  const bufLandmarks = makeBuf(63 * 4, SOC);
  const bufWorldLandmarks = makeBuf(63 * 4, SOC);
  const bufHandflag = makeBuf(4, SOC);
  const bufHandedness = makeBuf(4, SOC);

  // Unified output: [handflag, handedness, landmarks[63]] = 65 floats
  const bufOutput = makeBuf(65 * 4, SOCD);  // needs CopyDst for copyBufferToBuffer dest
  const readbackBuf = makeBuf(65 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

  // Canvas input texture (224x224 rgba8unorm)
  const canvasTexture = device.createTexture({
    size: [224, 224],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const bufCanvasU = makeBuf(4, UC);
  device.queue.writeBuffer(bufCanvasU, 0, new Uint32Array([224]));

  // ============ Upload Initial Conv Weights ============
  const initConvWT = requireWeight('conv2d');
  const initConvBT = requireWeight('batch_normalization');

  const initConvWData = getWeightData(initConvWT);
  const initConvBData = getWeightData(initConvBT);

  const bufInitConvW = makeBuf(getWeightBufSize(initConvWT), SC);
  const bufInitConvB = makeBuf(getWeightBufSize(initConvBT), SC);
  const bufInitConvU = makeBuf(24, UC);
  device.queue.writeBuffer(bufInitConvW, 0, initConvWData as any);
  device.queue.writeBuffer(bufInitConvB, 0, initConvBData as any);
  device.queue.writeBuffer(bufInitConvU, 0, new Uint32Array([3, 24, 224, 224, 112, 112]));

  // ============ Upload Block Weights ============
  interface BlockData {
    expandW?: GPUBuffer;
    expandB?: GPUBuffer;
    expandU?: GPUBuffer;
    dwW: GPUBuffer;
    dwB: GPUBuffer;
    dwU: GPUBuffer;
    projectW?: GPUBuffer;
    projectB?: GPUBuffer;
    projectU?: GPUBuffer;
    spec: MBConvBlockSpec;
    inH: number;
    inW: number;
    outH: number;
    outW: number;
  }

  // Compute spatial dims through the network
  let curH = 112;
  let curW = 112;

  const blockData: BlockData[] = [];

  for (let i = 0; i < BLOCK_SPECS.length; i++) {
    const spec = BLOCK_SPECS[i]!;
    const names = BLOCK_WEIGHT_NAMES[i]!;
    const inH = curH;
    const inW = curW;
    const outH = spec.stride === 2 ? Math.floor(curH / 2) : curH;
    const outW = spec.stride === 2 ? Math.floor(curW / 2) : curW;

    const bd: BlockData = {
      spec,
      inH, inW, outH, outW,
      dwW: makeBuf(4, SC),
      dwB: makeBuf(4, SC),
      dwU: makeBuf(32, UC),
    };

    // Expand 1x1 (skip if expandCh == inCh, i.e., block 1)
    if (names.expandConvKey) {
      const ewt = requireWeight(names.expandConvKey);
      const ebt = requireWeight(names.expandBNKey!);
      bd.expandW = makeBuf(getWeightBufSize(ewt), SC);
      bd.expandB = makeBuf(getWeightBufSize(ebt), SC);
      bd.expandU = makeBuf(16, UC);
      device.queue.writeBuffer(bd.expandW, 0, getWeightData(ewt) as any);
      device.queue.writeBuffer(bd.expandB, 0, getWeightData(ebt) as any);
      device.queue.writeBuffer(bd.expandU, 0, new Uint32Array([spec.inCh, spec.expandCh, inH, inW]));
    }

    // DW conv
    const dwwt = requireWeight(names.dwWeightKey);
    const dwbt = requireWeight(names.dwBNKey);
    bd.dwW = makeBuf(getWeightBufSize(dwwt), SC);
    bd.dwB = makeBuf(getWeightBufSize(dwbt), SC);
    device.queue.writeBuffer(bd.dwW, 0, getWeightData(dwwt) as any);
    device.queue.writeBuffer(bd.dwB, 0, getWeightData(dwbt) as any);

    // DW params: channels, in_h, in_w, out_h, out_w, stride, pad, kernel
    // TFLite SAME padding: pad_before = floor((kernel - stride) / 2)
    // For stride=1: symmetric (e.g. k=3→pad=1, k=5→pad=2)
    // For stride=2: asymmetric (e.g. k=3→pad=0, k=5→pad=1)
    const dwPad = Math.floor((spec.dwKernel - spec.stride) / 2);
    device.queue.writeBuffer(bd.dwU, 0, new Uint32Array([
      spec.expandCh, inH, inW, outH, outW, spec.stride, dwPad, spec.dwKernel
    ]));

    // Project 1x1 (skip for block 16)
    if (spec.hasProject && names.projectConvKey) {
      const pwt = requireWeight(names.projectConvKey);
      const pbt = requireWeight(names.projectBNKey!);
      bd.projectW = makeBuf(getWeightBufSize(pwt), SC);
      bd.projectB = makeBuf(getWeightBufSize(pbt), SC);
      bd.projectU = makeBuf(16, UC);
      device.queue.writeBuffer(bd.projectW, 0, getWeightData(pwt) as any);
      device.queue.writeBuffer(bd.projectB, 0, getWeightData(pbt) as any);
      device.queue.writeBuffer(bd.projectU, 0, new Uint32Array([spec.expandCh, spec.outCh, outH, outW]));
    }

    blockData.push(bd);
    curH = outH;
    curW = outW;
  }

  // ============ Upload Head Weights ============
  // Head keys: conv_landmarks (weight [63,1152], bias [63])
  //            conv_world_landmarks (weight [63,1152], bias [63])
  //            conv_handflag (weight [1,1152], bias [1])
  //            conv_handedness (weight [1,1152], bias [1])
  // Note: conv_landmarks appears twice — shape [63] is bias, shape [63,1152] is weight.
  // We need to find by shape.
  function findWeightByKeyAndShape(key: string, rank: number): Tensor {
    // The weights map may have duplicate keys. We need to iterate.
    // Actually Map can't have duplicate keys — but looking at the manifest,
    // the key names like "conv_landmarks" appear at index 3 (bias [63]) and index 40 (weight [63,1152]).
    // Since Map.set overwrites, only the LAST one is stored.
    // So we need to use the keys/shapes from the original manifest.
    // But we receive a Map<string, Tensor>. Let's just use what we get.
    const t = weights.get(key);
    if (!t) throw new Error(`Missing weight: ${key}`);
    if (t.shape.length !== rank) {
      throw new Error(`Weight ${key} has rank ${t.shape.length}, expected ${rank}`);
    }
    return t;
  }

  // Since the Map has duplicate keys and the LAST one wins, let's verify which we get.
  // From the manifest order: conv_landmarks appears at index 3 (shape=[63], bias) and index 40 (shape=[63,1152], weight)
  // Since index 40 comes LAST, weights.get('conv_landmarks') returns the [63,1152] weight tensor.
  // The bias at index 3 is overwritten. We need to use the Identity keys for biases.
  // Identity = [1, 63] → landmarks bias
  // Identity_1 = [1, 1] → handflag bias? Or handedness?
  // Identity_2 = [1, 1]
  // Identity_3 = [1, 63] → world landmarks bias

  // Let's look at head weights more carefully:
  // conv_handflag shape=[1] (bias, idx 4) → overwritten by conv_handflag shape=[1,1152] (weight, idx 39)
  // conv_handedness shape=[1] (bias, idx 5) → overwritten by conv_handedness shape=[1,1152] (weight, idx 38)
  // conv_landmarks shape=[63] (bias, idx 3) → overwritten by conv_landmarks shape=[63,1152] (weight, idx 40)
  // conv_world_landmarks shape=[63] (bias, idx 2) → overwritten by conv_world_landmarks shape=[63,1152] (weight, idx 41)
  //
  // So for biases we need to use Identity_N keys:
  // Identity = [1, 63] → probably landmarks bias
  // Identity_1 = [1, 1] → probably handflag bias (since it's sigmoid)
  // Identity_2 = [1, 1] → probably handedness bias
  // Identity_3 = [1, 63] → probably world landmarks bias

  // Duplicate keys in manifest: first occurrence = bias, second = weight.
  // loadFullWeightsFromBuffer names them: base key, then key__1.
  const landmarksWT = requireWeight('conv_landmarks__1');     // [63, 1152]
  const worldLandmarksWT = requireWeight('conv_world_landmarks__1'); // [63, 1152]
  const handflagWT = requireWeight('conv_handflag__1');        // [1, 1152]
  const handednessWT = requireWeight('conv_handedness__1');    // [1, 1152]

  // Biases from Identity keys
  const landmarksBT = requireWeight('Identity');           // [1, 63]
  const handflagBT = requireWeight('Identity_1');          // [1, 1]
  const handednessBT = requireWeight('Identity_2');        // [1, 1]
  const worldLandmarksBT = requireWeight('Identity_3');    // [1, 63]

  const bufLandmarksW = makeBuf(getWeightBufSize(landmarksWT), SC);
  const bufLandmarksB = makeBuf(getWeightBufSize(landmarksBT), SC);
  const bufWorldLandmarksW = makeBuf(getWeightBufSize(worldLandmarksWT), SC);
  const bufWorldLandmarksB = makeBuf(getWeightBufSize(worldLandmarksBT), SC);
  const bufHandflagW = makeBuf(getWeightBufSize(handflagWT), SC);
  const bufHandflagB = makeBuf(getWeightBufSize(handflagBT), SC);
  const bufHandednessW = makeBuf(getWeightBufSize(handednessWT), SC);
  const bufHandednessB = makeBuf(getWeightBufSize(handednessBT), SC);

  device.queue.writeBuffer(bufLandmarksW, 0, getWeightData(landmarksWT) as any);
  device.queue.writeBuffer(bufLandmarksB, 0, getWeightData(landmarksBT) as any);
  device.queue.writeBuffer(bufWorldLandmarksW, 0, getWeightData(worldLandmarksWT) as any);
  device.queue.writeBuffer(bufWorldLandmarksB, 0, getWeightData(worldLandmarksBT) as any);
  device.queue.writeBuffer(bufHandflagW, 0, getWeightData(handflagWT) as any);
  device.queue.writeBuffer(bufHandflagB, 0, getWeightData(handflagBT) as any);
  device.queue.writeBuffer(bufHandednessW, 0, getWeightData(handednessWT) as any);
  device.queue.writeBuffer(bufHandednessB, 0, getWeightData(handednessBT) as any);

  const bufFCLandmarksU = makeBuf(8, UC);
  const bufFCWorldLandmarksU = makeBuf(8, UC);
  const bufFCHandflagU = makeBuf(8, UC);
  const bufFCHandednessU = makeBuf(8, UC);
  device.queue.writeBuffer(bufFCLandmarksU, 0, new Uint32Array([1152, 63]));
  device.queue.writeBuffer(bufFCWorldLandmarksU, 0, new Uint32Array([1152, 63]));
  device.queue.writeBuffer(bufFCHandflagU, 0, new Uint32Array([1152, 1]));
  device.queue.writeBuffer(bufFCHandednessU, 0, new Uint32Array([1152, 1]));

  // GAP uniform
  const bufGAPU = makeBuf(8, UC);
  device.queue.writeBuffer(bufGAPU, 0, new Uint32Array([1152, curH * curW])); // 7*7=49

  // Add residual uniform
  // Pre-create uniforms for each block that has residual
  const residualUniforms: Map<number, GPUBuffer> = new Map();
  for (let i = 0; i < BLOCK_SPECS.length; i++) {
    if (BLOCK_SPECS[i]!.hasResidual) {
      const bd = blockData[i]!;
      const u = makeBuf(4, UC);
      device.queue.writeBuffer(u, 0, new Uint32Array([BLOCK_SPECS[i]!.outCh * bd.outH * bd.outW]));
      residualUniforms.set(i, u);
    }
  }

  // ============ Pre-create Bind Groups ============
  const canvasInputBG = makeBind(canvasLayout, [canvasTexture.createView(), bufA, bufCanvasU]);
  const initConvBG = makeBind(conv5Layout, [bufA, bufInitConvW, bufInitConvB, bufB, bufInitConvU]);

  // Pre-allocated output arrays
  const outputHandflag = new Float32Array(1);
  const outputHandedness = new Float32Array(1);
  const outputLandmarks = new Float32Array(63);

  // ============ Encode Inference ============
  // Uses a simple per-block approach: each block that needs a residual copy
  // gets its own encoder-level boundary. Non-residual blocks can share passes.
  function encodeInference(encoder: GPUCommandEncoder, readbackTarget: GPUBuffer | null) {
    // Initial conv: bufA (224x224x3) → bufB (112x112x24)
    let pass = encoder.beginComputePass();
    pass.setPipeline(initConvPipeline);
    pass.setBindGroup(0, initConvBG);
    pass.dispatchWorkgroups(Math.ceil(112 / 8), Math.ceil(112 / 8), 24);
    pass.end();

    // After initial conv: data is in bufB
    let curBuf = bufB;
    let altBuf = bufA;

    for (let blockIdx = 0; blockIdx < BLOCK_SPECS.length; blockIdx++) {
      const spec = BLOCK_SPECS[blockIdx]!;
      const bd = blockData[blockIdx]!;

      // Save input for residual if needed (requires encoder-level copy)
      if (spec.hasResidual) {
        const residualSize = spec.inCh * bd.inH * bd.inW * bpe;
        encoder.copyBufferToBuffer(curBuf, 0, bufResidual, 0, residualSize);
      }

      pass = encoder.beginComputePass();

      // Expand step (1x1 conv + BN + ReLU6)
      if (bd.expandW) {
        const expandBG = makeBind(conv5Layout, [curBuf, bd.expandW, bd.expandB!, bufExpand, bd.expandU!]);
        pass.setPipeline(expand1x1Pipeline);
        pass.setBindGroup(0, expandBG);
        pass.dispatchWorkgroups(Math.ceil(bd.inW / 8), Math.ceil(bd.inH / 8), spec.expandCh);
      }

      // DW step (depthwise conv + BN + ReLU6)
      const dwInput = bd.expandW ? bufExpand : curBuf;
      const dwBG = makeBind(conv5Layout, [dwInput, bd.dwW, bd.dwB, bufDW, bd.dwU]);
      pass.setPipeline(dwPipeline);
      pass.setBindGroup(0, dwBG);
      pass.dispatchWorkgroups(Math.ceil(bd.outW / 8), Math.ceil(bd.outH / 8), spec.expandCh);

      // Project step (1x1 conv + BN, NO activation)
      if (spec.hasProject && bd.projectW) {
        const projTarget = spec.hasResidual ? altBuf : altBuf;
        const projBG = makeBind(conv5Layout, [bufDW, bd.projectW, bd.projectB!, projTarget, bd.projectU!]);
        pass.setPipeline(project1x1Pipeline);
        pass.setBindGroup(0, projBG);
        pass.dispatchWorkgroups(Math.ceil(bd.outW / 8), Math.ceil(bd.outH / 8), spec.outCh);

        if (spec.hasResidual) {
          // Add residual: altBuf + bufResidual → curBuf
          const addU = residualUniforms.get(blockIdx)!;
          const addBG = makeBind(addLayout, [altBuf, bufResidual, curBuf, addU]);
          pass.setPipeline(addResPipeline);
          pass.setBindGroup(0, addBG);
          pass.dispatchWorkgroups(Math.ceil(spec.outCh * bd.outH * bd.outW / 256));
          // curBuf still has the result (same buffer as input for residual blocks)
        } else {
          // Swap buffers
          const tmp = curBuf;
          curBuf = altBuf;
          altBuf = tmp;
        }
      }

      pass.end();

      // Block 16 special case: no project, DW output is final feature map
      if (!spec.hasProject) {
        // bufDW has [1152, 7, 7] — use it directly for GAP
        pass = encoder.beginComputePass();

        // GAP: [1152, 7, 7] → [1152]
        const gapBG = makeBind(gapLayout, [bufDW, bufGAP, bufGAPU]);
        pass.setPipeline(gapPipeline);
        pass.setBindGroup(0, gapBG);
        pass.dispatchWorkgroups(Math.ceil(1152 / 256));

        // FC heads — all read from bufGAP
        // Landmarks: [1152] → [63]
        const fcLandmarksBG = makeBind(fcLayout, [bufGAP, bufLandmarksW, bufLandmarksB, bufLandmarks, bufFCLandmarksU]);
        pass.setPipeline(fcPipeline);
        pass.setBindGroup(0, fcLandmarksBG);
        pass.dispatchWorkgroups(1);

        // Handflag: [1152] → [1] + sigmoid
        const fcHandflagBG = makeBind(fcLayout, [bufGAP, bufHandflagW, bufHandflagB, bufHandflag, bufFCHandflagU]);
        pass.setPipeline(fcSigmoidPipeline);
        pass.setBindGroup(0, fcHandflagBG);
        pass.dispatchWorkgroups(1);

        // Handedness: [1152] → [1] + sigmoid
        const fcHandednessBG = makeBind(fcLayout, [bufGAP, bufHandednessW, bufHandednessB, bufHandedness, bufFCHandednessU]);
        pass.setPipeline(fcSigmoidPipeline);
        pass.setBindGroup(0, fcHandednessBG);
        pass.dispatchWorkgroups(1);

        pass.end();

        // Copy results to unified output buffer
        // Layout: [handflag(1), handedness(1), landmarks(63)] = 65 floats
        encoder.copyBufferToBuffer(bufHandflag, 0, bufOutput, 0, 4);
        encoder.copyBufferToBuffer(bufHandedness, 0, bufOutput, 4, 4);
        encoder.copyBufferToBuffer(bufLandmarks, 0, bufOutput, 8, 63 * 4);

        if (readbackTarget) {
          encoder.copyBufferToBuffer(bufOutput, 0, readbackTarget, 0, 65 * 4);
        }
        return;
      }
    }
  }

  // ============ Run from Float32Array ============
  async function run(input: Float32Array): Promise<HandLandmarksOutput> {
    device.queue.writeBuffer(bufRawInput, 0, input as any);

    const encoder = device.createCommandEncoder();

    // Copy raw input to bufA
    encoder.copyBufferToBuffer(bufRawInput, 0, bufA, 0, 3 * 224 * 224 * 4);

    encodeInference(encoder, readbackBuf);

    device.queue.submit([encoder.finish()]);

    const mapPromise = readbackBuf.mapAsync(GPUMapMode.READ);
    await device.queue.onSubmittedWorkDone();
    await mapPromise;
    const mapped = new Float32Array(readbackBuf.getMappedRange());
    outputHandflag[0] = mapped[0]!;
    outputHandedness[0] = mapped[1]!;
    // Normalize landmarks from 224-pixel space to [0,1]
    for (let i = 0; i < 63; i++) {
      outputLandmarks[i] = mapped[2 + i]! / 224;
    }
    readbackBuf.unmap();

    return {
      handflag: new Float32Array(outputHandflag),
      handedness: new Float32Array(outputHandedness),
      landmarks: new Float32Array(outputLandmarks),
    };
  }

  // ============ Run from Canvas ============
  async function runFromCanvas(
    source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap,
  ): Promise<HandLandmarksOutput> {
    device.queue.copyExternalImageToTexture(
      { source },
      { texture: canvasTexture },
      [224, 224],
    );

    const encoder = device.createCommandEncoder();

    // Canvas → NCHW in bufA
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(canvasInputPipeline);
      pass.setBindGroup(0, canvasInputBG);
      pass.dispatchWorkgroups(Math.ceil(224 / 16), Math.ceil(224 / 16), 1);
      pass.end();
    }

    encodeInference(encoder, readbackBuf);

    device.queue.submit([encoder.finish()]);

    const mapPromise = readbackBuf.mapAsync(GPUMapMode.READ);
    await device.queue.onSubmittedWorkDone();
    await mapPromise;
    const mapped = new Float32Array(readbackBuf.getMappedRange());
    outputHandflag[0] = mapped[0]!;
    outputHandedness[0] = mapped[1]!;
    // Normalize landmarks from 224-pixel space to [0,1]
    for (let i = 0; i < 63; i++) {
      outputLandmarks[i] = mapped[2 + i]! / 224;
    }
    readbackBuf.unmap();

    return {
      handflag: new Float32Array(outputHandflag),
      handedness: new Float32Array(outputHandedness),
      landmarks: new Float32Array(outputLandmarks),
    };
  }

  // ============ Run from GPUBuffer ============
  async function runFromGPUBuffer(inputBuffer: GPUBuffer): Promise<HandLandmarksOutput> {
    const encoder = device.createCommandEncoder();

    // Copy CHW data from external buffer to bufA
    encoder.copyBufferToBuffer(inputBuffer, 0, bufA, 0, 3 * 224 * 224 * 4);

    encodeInference(encoder, readbackBuf);

    device.queue.submit([encoder.finish()]);

    const mapPromise = readbackBuf.mapAsync(GPUMapMode.READ);
    await device.queue.onSubmittedWorkDone();
    await mapPromise;
    const mapped = new Float32Array(readbackBuf.getMappedRange());
    outputHandflag[0] = mapped[0]!;
    outputHandedness[0] = mapped[1]!;
    // Normalize landmarks from 224-pixel space to [0,1]
    for (let i = 0; i < 63; i++) {
      outputLandmarks[i] = mapped[2 + i]! / 224;
    }
    readbackBuf.unmap();

    return {
      handflag: new Float32Array(outputHandflag),
      handedness: new Float32Array(outputHandedness),
      landmarks: new Float32Array(outputLandmarks),
    };
  }

  // ============ Stub methods to satisfy CompiledModel interface ============
  async function runFromCanvasPipelined(): Promise<HandLandmarksOutput | null> {
    return null;
  }

  async function flushPipelined(): Promise<HandLandmarksOutput | null> {
    return null;
  }

  async function benchmark(iterations = 100): Promise<{ avgMs: number; fps: number }> {
    const canvas = new OffscreenCanvas(224, 224);
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#886644';
    ctx.fillRect(0, 0, 224, 224);

    // Warmup
    for (let i = 0; i < 5; i++) await runFromCanvas(canvas);

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      await runFromCanvas(canvas);
    }
    const elapsed = performance.now() - start;
    const avgMs = elapsed / iterations;
    return { avgMs, fps: 1000 / avgMs };
  }

  async function benchmarkGPU(iterations = 100): Promise<{ avgMs: number; fps: number; medianMs: number; minMs: number }> {
    const result = await benchmark(iterations);
    return { ...result, medianMs: result.avgMs, minMs: result.avgMs };
  }

  async function runFromCanvasViaRender(
    source: HTMLCanvasElement | OffscreenCanvas | ImageBitmap,
  ): Promise<HandLandmarksOutput> {
    return runFromCanvas(source);
  }

  async function benchmarkDiagnostic() {
    return {
      gpuOnly: { median: 0, min: 0 },
      mapAsyncOnly: { median: 0, min: 0 },
      mapAsyncNoWait: { median: 0, min: 0 },
      total: { median: 0, min: 0 },
      pipelined: { median: 0, min: 0 },
      renderReadback: null,
    };
  }

  async function debugLayerOutputs(source?: HTMLCanvasElement | OffscreenCanvas | ImageBitmap) {
    // Read back intermediate buffers after running
    const results: Record<string, any> = {};

    async function readBufStats(buf: GPUBuffer, numFloats: number, label: string) {
      const byteSize = numFloats * 4;
      const rb = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, rb, 0, byteSize);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      await rb.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(rb.getMappedRange());
      let min = Infinity, max = -Infinity, nonZero = 0;
      for (let i = 0; i < data.length; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
        if (data[i] !== 0) nonZero++;
      }
      const sample = Array.from(data.slice(0, 5));
      rb.unmap();
      rb.destroy();
      results[label] = { min, max, nonZero, total: numFloats, sample };
    }

    // Run inference with synthetic input
    const input = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224*224; i++) {
      input[i] = 0.5;
      input[224*224+i] = 0.3;
      input[2*224*224+i] = 0.7;
    }
    device.queue.writeBuffer(bufRawInput, 0, input as any);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufRawInput, 0, bufA, 0, 3 * 224 * 224 * 4);
    encodeInference(encoder, readbackBuf);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read key buffers
    await readBufStats(bufA, 3 * 224 * 224, 'inputBufA');
    await readBufStats(bufB, 24 * 112 * 112, 'afterInitConvBufB');
    await readBufStats(bufGAP, 1152, 'gapOutput');
    await readBufStats(bufLandmarks, 63, 'landmarks');
    await readBufStats(bufHandflag, 1, 'handflag');
    await readBufStats(bufOutput, 65, 'unifiedOutput');

    return results;
  }

  return {
    device,
    run,
    runFromCanvas,
    runFromGPUBuffer,
    runFromCanvasPipelined,
    flushPipelined,
    benchmark,
    benchmarkGPU,
    runFromCanvasViaRender,
    benchmarkDiagnostic,
    debugLayerOutputs,
  };
}
