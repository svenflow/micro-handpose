/**
 * Optimized WebGPU shaders for HandLandmarks model
 *
 * PERFORMANCE: ~4.1ms (244 FPS) - 1.2x faster than MediaPipe WebGL!
 *
 * KEY OPTIMIZATIONS:
 *
 * 1. Two-Pass DW+PW Pipeline (vs fused)
 *    - Separate depthwise and pointwise dispatches
 *    - Each pass has optimal parallelism for its workload
 *    - Fused kernels were 700x SLOWER due to O(N²) work per thread
 *
 * 2. 4x Loop Unrolling in Pointwise (2.1x speedup)
 *    - Manual unroll of channel reduction loop
 *    - Enables instruction-level parallelism (4 independent MADs per iteration)
 *    - Reduces loop overhead and keeps more values in registers
 *
 * 3. Workgroup Size (8,8,1)
 *    - Balanced for both small and large spatial sizes
 *    - Z-parallel (1,1,256) was SLOWER in full model due to dispatch overhead
 *
 * WHAT DIDN'T WORK:
 * - FP16 + register blocking: 19% slower (register pressure)
 * - Fused DW+PW kernel: 700x slower (wrong parallelism)
 * - Z-parallel workgroups: fake 12x speedup (z-limit is 64, not 256!)
 * - NHWC layout: mixed results, hurts large spatial sizes
 *
 * See OPTIMIZATIONS.md for full details.
 */

// Full 5x5 unrolled depthwise - 2x faster on large spatial (128x128)
// Loads all 25 weights and inputs, then computes in one expression
export const DEPTHWISE_5x5_FULL_UNROLL_SHADER = /* wgsl */ `
struct DepthwiseParams {
  batch: u32,
  channels: u32,
  in_height: u32,
  in_width: u32,
  out_height: u32,
  out_width: u32,
  stride: u32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: DepthwiseParams;

fn load_input(base: u32, y: i32, x: i32, in_h: i32, in_w: i32) -> f32 {
  if (y >= 0 && y < in_h && x >= 0 && x < in_w) {
    return input[base + u32(y) * u32(in_w) + u32(x)];
  }
  return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let c_batch = gid.z;

  let c = c_batch % params.channels;
  let batch = c_batch / params.channels;

  if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
    return;
  }

  let in_base = batch * params.channels * params.in_height * params.in_width
              + c * params.in_height * params.in_width;
  let w_base = c * 25u;
  let in_h = i32(params.in_height);
  let in_w = i32(params.in_width);
  let by = i32(out_y * params.stride) - i32(params.pad);
  let bx = i32(out_x * params.stride) - i32(params.pad);

  // Load all 25 weights into registers
  let w00 = weight[w_base]; let w01 = weight[w_base + 1u]; let w02 = weight[w_base + 2u]; let w03 = weight[w_base + 3u]; let w04 = weight[w_base + 4u];
  let w10 = weight[w_base + 5u]; let w11 = weight[w_base + 6u]; let w12 = weight[w_base + 7u]; let w13 = weight[w_base + 8u]; let w14 = weight[w_base + 9u];
  let w20 = weight[w_base + 10u]; let w21 = weight[w_base + 11u]; let w22 = weight[w_base + 12u]; let w23 = weight[w_base + 13u]; let w24 = weight[w_base + 14u];
  let w30 = weight[w_base + 15u]; let w31 = weight[w_base + 16u]; let w32 = weight[w_base + 17u]; let w33 = weight[w_base + 18u]; let w34 = weight[w_base + 19u];
  let w40 = weight[w_base + 20u]; let w41 = weight[w_base + 21u]; let w42 = weight[w_base + 22u]; let w43 = weight[w_base + 23u]; let w44 = weight[w_base + 24u];

  // Load all 25 inputs (with boundary checks)
  let i00 = load_input(in_base, by, bx, in_h, in_w);
  let i01 = load_input(in_base, by, bx+1, in_h, in_w);
  let i02 = load_input(in_base, by, bx+2, in_h, in_w);
  let i03 = load_input(in_base, by, bx+3, in_h, in_w);
  let i04 = load_input(in_base, by, bx+4, in_h, in_w);
  let i10 = load_input(in_base, by+1, bx, in_h, in_w);
  let i11 = load_input(in_base, by+1, bx+1, in_h, in_w);
  let i12 = load_input(in_base, by+1, bx+2, in_h, in_w);
  let i13 = load_input(in_base, by+1, bx+3, in_h, in_w);
  let i14 = load_input(in_base, by+1, bx+4, in_h, in_w);
  let i20 = load_input(in_base, by+2, bx, in_h, in_w);
  let i21 = load_input(in_base, by+2, bx+1, in_h, in_w);
  let i22 = load_input(in_base, by+2, bx+2, in_h, in_w);
  let i23 = load_input(in_base, by+2, bx+3, in_h, in_w);
  let i24 = load_input(in_base, by+2, bx+4, in_h, in_w);
  let i30 = load_input(in_base, by+3, bx, in_h, in_w);
  let i31 = load_input(in_base, by+3, bx+1, in_h, in_w);
  let i32_ = load_input(in_base, by+3, bx+2, in_h, in_w);
  let i33 = load_input(in_base, by+3, bx+3, in_h, in_w);
  let i34 = load_input(in_base, by+3, bx+4, in_h, in_w);
  let i40 = load_input(in_base, by+4, bx, in_h, in_w);
  let i41 = load_input(in_base, by+4, bx+1, in_h, in_w);
  let i42 = load_input(in_base, by+4, bx+2, in_h, in_w);
  let i43 = load_input(in_base, by+4, bx+3, in_h, in_w);
  let i44 = load_input(in_base, by+4, bx+4, in_h, in_w);

  // Compute all 25 MADs in one expression
  let sum = i00*w00 + i01*w01 + i02*w02 + i03*w03 + i04*w04
          + i10*w10 + i11*w11 + i12*w12 + i13*w13 + i14*w14
          + i20*w20 + i21*w21 + i22*w22 + i23*w23 + i24*w24
          + i30*w30 + i31*w31 + i32_*w32 + i33*w33 + i34*w34
          + i40*w40 + i41*w41 + i42*w42 + i43*w43 + i44*w44
          + bias[c];

  let out_idx = batch * params.channels * params.out_height * params.out_width
              + c * params.out_height * params.out_width
              + out_y * params.out_width
              + out_x;
  output[out_idx] = sum;
}
`;

// 5-row unrolled depthwise 5x5 - 1.35x faster (2.07ms → 1.53ms isolated)
// Pre-computed row offsets and manual unroll enables better instruction scheduling
export const DEPTHWISE_5x5_SHADER = /* wgsl */ `
struct DepthwiseParams {
  batch: u32,
  channels: u32,
  in_height: u32,
  in_width: u32,
  out_height: u32,
  out_width: u32,
  stride: u32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: DepthwiseParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let c_batch = gid.z;

  let c = c_batch % params.channels;
  let batch = c_batch / params.channels;

  if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
    return;
  }

  // Pre-compute base indices once
  let in_base = batch * params.channels * params.in_height * params.in_width
              + c * params.in_height * params.in_width;
  let w_base = c * 25u;
  let in_h = i32(params.in_height);
  let in_w = i32(params.in_width);
  let pad = i32(params.pad);

  // Pre-compute base input position
  let base_in_y = i32(out_y * params.stride) - pad;
  let base_in_x = i32(out_x * params.stride) - pad;

  // 5 separate accumulators for each row (enables parallel execution)
  var row0: f32 = 0.0;
  var row1: f32 = 0.0;
  var row2: f32 = 0.0;
  var row3: f32 = 0.0;
  var row4: f32 = 0.0;

  // Row 0 (ky=0)
  let y0 = base_in_y;
  if (y0 >= 0 && y0 < in_h) {
    let row_base = in_base + u32(y0) * params.in_width;
    for (var kx: u32 = 0u; kx < 5u; kx = kx + 1u) {
      let in_x = base_in_x + i32(kx);
      if (in_x >= 0 && in_x < in_w) {
        row0 += input[row_base + u32(in_x)] * weight[w_base + kx];
      }
    }
  }

  // Row 1 (ky=1)
  let y1 = base_in_y + 1;
  if (y1 >= 0 && y1 < in_h) {
    let row_base = in_base + u32(y1) * params.in_width;
    for (var kx: u32 = 0u; kx < 5u; kx = kx + 1u) {
      let in_x = base_in_x + i32(kx);
      if (in_x >= 0 && in_x < in_w) {
        row1 += input[row_base + u32(in_x)] * weight[w_base + 5u + kx];
      }
    }
  }

  // Row 2 (ky=2)
  let y2 = base_in_y + 2;
  if (y2 >= 0 && y2 < in_h) {
    let row_base = in_base + u32(y2) * params.in_width;
    for (var kx: u32 = 0u; kx < 5u; kx = kx + 1u) {
      let in_x = base_in_x + i32(kx);
      if (in_x >= 0 && in_x < in_w) {
        row2 += input[row_base + u32(in_x)] * weight[w_base + 10u + kx];
      }
    }
  }

  // Row 3 (ky=3)
  let y3 = base_in_y + 3;
  if (y3 >= 0 && y3 < in_h) {
    let row_base = in_base + u32(y3) * params.in_width;
    for (var kx: u32 = 0u; kx < 5u; kx = kx + 1u) {
      let in_x = base_in_x + i32(kx);
      if (in_x >= 0 && in_x < in_w) {
        row3 += input[row_base + u32(in_x)] * weight[w_base + 15u + kx];
      }
    }
  }

  // Row 4 (ky=4)
  let y4 = base_in_y + 4;
  if (y4 >= 0 && y4 < in_h) {
    let row_base = in_base + u32(y4) * params.in_width;
    for (var kx: u32 = 0u; kx < 5u; kx = kx + 1u) {
      let in_x = base_in_x + i32(kx);
      if (in_x >= 0 && in_x < in_w) {
        row4 += input[row_base + u32(in_x)] * weight[w_base + 20u + kx];
      }
    }
  }

  let sum = row0 + row1 + row2 + row3 + row4 + bias[c];

  let out_idx = batch * params.channels * params.out_height * params.out_width
              + c * params.out_height * params.out_width
              + out_y * params.out_width
              + out_x;
  output[out_idx] = sum;
}
`;

// 4x loop unrolled version - up to 2x faster on 288-channel layers
export const POINTWISE_SKIP_RELU_SHADER = /* wgsl */ `
struct PointwiseParams {
  batch: u32,
  in_channels: u32,
  out_channels: u32,
  height: u32,
  width: u32,
  channel_pad: u32,
  stride: u32,
  in_height: u32,
  in_width: u32,
}

@group(0) @binding(0) var<storage, read> dw_output: array<f32>;
@group(0) @binding(1) var<storage, read> skip_input: array<f32>;
@group(0) @binding(2) var<storage, read> pw_weight: array<f32>;
@group(0) @binding(3) var<storage, read> pw_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: PointwiseParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let oc_batch = gid.z;

  let oc = oc_batch % params.out_channels;
  let batch = oc_batch / params.out_channels;

  if (out_x >= params.width || out_y >= params.height || batch >= params.batch) {
    return;
  }

  // 4x loop unroll for instruction-level parallelism
  var sum0: f32 = 0.0;
  var sum1: f32 = 0.0;
  var sum2: f32 = 0.0;
  var sum3: f32 = 0.0;

  let dw_base = batch * params.in_channels * params.height * params.width
              + out_y * params.width + out_x;
  let w_base = oc * params.in_channels;
  let spatial_stride = params.height * params.width;

  var ic: u32 = 0u;
  let ic_end4 = (params.in_channels / 4u) * 4u;

  // Process 4 channels at a time
  while (ic < ic_end4) {
    let dw0 = dw_output[dw_base + ic * spatial_stride];
    let dw1 = dw_output[dw_base + (ic + 1u) * spatial_stride];
    let dw2 = dw_output[dw_base + (ic + 2u) * spatial_stride];
    let dw3 = dw_output[dw_base + (ic + 3u) * spatial_stride];

    let w0 = pw_weight[w_base + ic];
    let w1 = pw_weight[w_base + ic + 1u];
    let w2 = pw_weight[w_base + ic + 2u];
    let w3 = pw_weight[w_base + ic + 3u];

    sum0 = sum0 + dw0 * w0;
    sum1 = sum1 + dw1 * w1;
    sum2 = sum2 + dw2 * w2;
    sum3 = sum3 + dw3 * w3;

    ic = ic + 4u;
  }

  // Handle remaining channels
  while (ic < params.in_channels) {
    let dw_val = dw_output[dw_base + ic * spatial_stride];
    let w_val = pw_weight[w_base + ic];
    sum0 = sum0 + dw_val * w_val;
    ic = ic + 1u;
  }

  var pw_sum = sum0 + sum1 + sum2 + sum3 + pw_bias[oc];

  var skip_val: f32 = 0.0;
  if (oc < params.in_channels) {
    if (params.stride == 2u) {
      var max_val: f32 = -1e38;
      for (var py: u32 = 0u; py < 2u; py = py + 1u) {
        for (var px: u32 = 0u; px < 2u; px = px + 1u) {
          let skip_y = out_y * 2u + py;
          let skip_x = out_x * 2u + px;
          if (skip_y < params.in_height && skip_x < params.in_width) {
            let skip_idx = batch * params.in_channels * params.in_height * params.in_width
                         + oc * params.in_height * params.in_width
                         + skip_y * params.in_width
                         + skip_x;
            max_val = max(max_val, skip_input[skip_idx]);
          }
        }
      }
      skip_val = max_val;
    } else {
      let skip_idx = batch * params.in_channels * params.height * params.width
                   + oc * params.height * params.width
                   + out_y * params.width
                   + out_x;
      skip_val = skip_input[skip_idx];
    }
  }

  let result = max(0.0, pw_sum + skip_val);
  let out_idx = batch * params.out_channels * params.height * params.width
              + oc * params.height * params.width
              + out_y * params.width
              + out_x;
  output[out_idx] = result;
}
`;

// 2 output channels per thread - 3x faster on large spatial (>=16) with high channels (>=288)
// Each thread computes 2 output channels, reusing input loads for better efficiency
export const POINTWISE_SKIP_RELU_2OC_SHADER = /* wgsl */ `
struct PointwiseParams {
  batch: u32,
  in_channels: u32,
  out_channels: u32,
  height: u32,
  width: u32,
  channel_pad: u32,
  stride: u32,
  in_height: u32,
  in_width: u32,
}

@group(0) @binding(0) var<storage, read> dw_output: array<f32>;
@group(0) @binding(1) var<storage, read> skip_input: array<f32>;
@group(0) @binding(2) var<storage, read> pw_weight: array<f32>;
@group(0) @binding(3) var<storage, read> pw_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: PointwiseParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let oc_pair_batch = gid.z;  // Each thread handles 2 output channels

  let half_out_channels = params.out_channels / 2u;
  let oc_pair = oc_pair_batch % half_out_channels;
  let batch = oc_pair_batch / half_out_channels;

  if (out_x >= params.width || out_y >= params.height || batch >= params.batch) {
    return;
  }

  let oc0 = oc_pair * 2u;
  let oc1 = oc0 + 1u;

  // 4x unroll with 2 output channels = 8 accumulators
  var sum0_a: f32 = 0.0;
  var sum0_b: f32 = 0.0;
  var sum0_c: f32 = 0.0;
  var sum0_d: f32 = 0.0;
  var sum1_a: f32 = 0.0;
  var sum1_b: f32 = 0.0;
  var sum1_c: f32 = 0.0;
  var sum1_d: f32 = 0.0;

  let dw_base = batch * params.in_channels * params.height * params.width
              + out_y * params.width + out_x;
  let w_base0 = oc0 * params.in_channels;
  let w_base1 = oc1 * params.in_channels;
  let spatial_stride = params.height * params.width;

  var ic: u32 = 0u;
  let ic_end4 = (params.in_channels / 4u) * 4u;

  // Process 4 input channels at a time, compute both output channels
  while (ic < ic_end4) {
    // Load input once, reuse for both output channels
    let in0 = dw_output[dw_base + ic * spatial_stride];
    let in1 = dw_output[dw_base + (ic + 1u) * spatial_stride];
    let in2 = dw_output[dw_base + (ic + 2u) * spatial_stride];
    let in3 = dw_output[dw_base + (ic + 3u) * spatial_stride];

    // Output channel 0
    sum0_a += in0 * pw_weight[w_base0 + ic];
    sum0_b += in1 * pw_weight[w_base0 + ic + 1u];
    sum0_c += in2 * pw_weight[w_base0 + ic + 2u];
    sum0_d += in3 * pw_weight[w_base0 + ic + 3u];

    // Output channel 1
    sum1_a += in0 * pw_weight[w_base1 + ic];
    sum1_b += in1 * pw_weight[w_base1 + ic + 1u];
    sum1_c += in2 * pw_weight[w_base1 + ic + 2u];
    sum1_d += in3 * pw_weight[w_base1 + ic + 3u];

    ic += 4u;
  }

  // Handle remaining channels
  while (ic < params.in_channels) {
    let in_val = dw_output[dw_base + ic * spatial_stride];
    sum0_a += in_val * pw_weight[w_base0 + ic];
    sum1_a += in_val * pw_weight[w_base1 + ic];
    ic += 1u;
  }

  let pw_sum0 = sum0_a + sum0_b + sum0_c + sum0_d + pw_bias[oc0];
  let pw_sum1 = sum1_a + sum1_b + sum1_c + sum1_d + pw_bias[oc1];

  // Skip connection for oc0
  var skip_val0: f32 = 0.0;
  if (oc0 < params.in_channels) {
    if (params.stride == 2u) {
      var max_val: f32 = -1e38;
      for (var py: u32 = 0u; py < 2u; py++) {
        for (var px: u32 = 0u; px < 2u; px++) {
          let skip_y = out_y * 2u + py;
          let skip_x = out_x * 2u + px;
          if (skip_y < params.in_height && skip_x < params.in_width) {
            let skip_idx = batch * params.in_channels * params.in_height * params.in_width
                         + oc0 * params.in_height * params.in_width
                         + skip_y * params.in_width + skip_x;
            max_val = max(max_val, skip_input[skip_idx]);
          }
        }
      }
      skip_val0 = max_val;
    } else {
      let skip_idx = batch * params.in_channels * params.height * params.width
                   + oc0 * params.height * params.width
                   + out_y * params.width + out_x;
      skip_val0 = skip_input[skip_idx];
    }
  }

  // Skip connection for oc1
  var skip_val1: f32 = 0.0;
  if (oc1 < params.in_channels) {
    if (params.stride == 2u) {
      var max_val: f32 = -1e38;
      for (var py: u32 = 0u; py < 2u; py++) {
        for (var px: u32 = 0u; px < 2u; px++) {
          let skip_y = out_y * 2u + py;
          let skip_x = out_x * 2u + px;
          if (skip_y < params.in_height && skip_x < params.in_width) {
            let skip_idx = batch * params.in_channels * params.in_height * params.in_width
                         + oc1 * params.in_height * params.in_width
                         + skip_y * params.in_width + skip_x;
            max_val = max(max_val, skip_input[skip_idx]);
          }
        }
      }
      skip_val1 = max_val;
    } else {
      let skip_idx = batch * params.in_channels * params.height * params.width
                   + oc1 * params.height * params.width
                   + out_y * params.width + out_x;
      skip_val1 = skip_input[skip_idx];
    }
  }

  let result0 = max(0.0, pw_sum0 + skip_val0);
  let result1 = max(0.0, pw_sum1 + skip_val1);

  let out_base = batch * params.out_channels * params.height * params.width;
  output[out_base + oc0 * params.height * params.width + out_y * params.width + out_x] = result0;
  if (oc1 < params.out_channels) {
    output[out_base + oc1 * params.height * params.width + out_y * params.width + out_x] = result1;
  }
}
`;



// ============ Adaptive Workgroup Size Shader Factories ============
// Generate shader variants with different workgroup sizes for optimal performance per layer
// Benchmark results:
//   288ch 16x16: wg(16,1,1) is 1.6x faster than wg(8,8,1)
//   48ch 64x64:  wg(16,1,1) is 1.1x faster than wg(8,8,1)
//   288ch 4x4:   wg(4,4,1) is 1.2x faster than wg(8,8,1)
//   288ch 8x8:   wg(8,8,1) is optimal
//   24ch 128x128: wg(8,8,1) is optimal

export function makeDepthwise5x5Shader(wgX: number, wgY: number): string {
  return DEPTHWISE_5x5_SHADER.replace(
    '@compute @workgroup_size(8, 8, 1)',
    `@compute @workgroup_size(${wgX}, ${wgY}, 1)`,
  );
}

export function makeDepthwise5x5FullUnrollShader(wgX: number, wgY: number): string {
  return DEPTHWISE_5x5_FULL_UNROLL_SHADER.replace(
    '@compute @workgroup_size(8, 8, 1)',
    `@compute @workgroup_size(${wgX}, ${wgY}, 1)`,
  );
}

export function makePointwiseShader(wgX: number, wgY: number): string {
  return POINTWISE_SKIP_RELU_SHADER.replace(
    '@compute @workgroup_size(8, 8, 1)',
    `@compute @workgroup_size(${wgX}, ${wgY}, 1)`,
  );
}

export function makePointwise2OCShader(wgX: number, wgY: number): string {
  return POINTWISE_SKIP_RELU_2OC_SHADER.replace(
    '@compute @workgroup_size(8, 8, 1)',
    `@compute @workgroup_size(${wgX}, ${wgY}, 1)`,
  );
}

// Determine optimal workgroup size for a given layer configuration
// Conservative: only change workgroup for layers with biggest proven gains
// to minimize pipeline switching overhead
export function getOptimalWorkgroupSize(channels: number, spatialH: number): [number, number] {
  // Adaptive WG tested but didn't help in full model (pipeline switching overhead > isolated gains)
  void channels; void spatialH;
  return [8, 8];
}

// GPU-side input padding (256x256 -> 257x257, avoids CPU copy overhead)
export const PAD_INPUT_SHADER = /* wgsl */ `
struct PadParams {
  channels: u32,
  in_size: u32,   // 256
  out_size: u32,  // 257
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PadParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let c = gid.z;

  if (x >= params.in_size || y >= params.in_size || c >= params.channels) {
    return;
  }

  let in_idx = c * params.in_size * params.in_size + y * params.in_size + x;
  let out_idx = c * params.out_size * params.out_size + y * params.out_size + x;
  output[out_idx] = input[in_idx];
}
`;

// Initial conv3x3 stride=2 (input layer)
export const CONV3X3_STRIDE2_RELU_SHADER = /* wgsl */ `
struct Conv3x3Params {
  batch: u32,
  in_channels: u32,
  out_channels: u32,
  in_height: u32,
  in_width: u32,
  out_height: u32,
  out_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv3x3Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let oc_batch = gid.z;

  let oc = oc_batch % params.out_channels;
  let batch = oc_batch / params.out_channels;

  if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
    return;
  }

  var sum: f32 = 0.0;

  for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
    for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
      for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
        // Stride 2, padding (0,1,0,1)
        let in_y = out_y * 2u + ky;
        let in_x = out_x * 2u + kx;

        if (in_y < params.in_height && in_x < params.in_width) {
          let in_idx = batch * params.in_channels * params.in_height * params.in_width
                     + ic * params.in_height * params.in_width
                     + in_y * params.in_width
                     + in_x;
          let w_idx = oc * params.in_channels * 9u + ic * 9u + ky * 3u + kx;
          sum = sum + input[in_idx] * weight[w_idx];
        }
      }
    }
  }

  sum = sum + bias[oc];
  sum = max(0.0, sum);  // ReLU

  let out_idx = batch * params.out_channels * params.out_height * params.out_width
              + oc * params.out_height * params.out_width
              + out_y * params.out_width
              + out_x;
  output[out_idx] = sum;
}
`;

// Bilinear upsample 2x
export const UPSAMPLE_2X_SHADER = /* wgsl */ `
struct UpsampleParams {
  batch: u32,
  channels: u32,
  in_height: u32,
  in_width: u32,
  out_height: u32,
  out_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: UpsampleParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let c_batch = gid.z;

  let c = c_batch % params.channels;
  let batch = c_batch / params.channels;

  if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
    return;
  }

  // Bilinear interpolation (align_corners=false)
  let scale_y = f32(params.in_height) / f32(params.out_height);
  let scale_x = f32(params.in_width) / f32(params.out_width);

  let src_y = (f32(out_y) + 0.5) * scale_y - 0.5;
  let src_x = (f32(out_x) + 0.5) * scale_x - 0.5;

  let y0 = u32(max(0.0, floor(src_y)));
  let x0 = u32(max(0.0, floor(src_x)));
  let y1 = min(y0 + 1u, params.in_height - 1u);
  let x1 = min(x0 + 1u, params.in_width - 1u);

  let ly = max(0.0, src_y) - f32(y0);
  let lx = max(0.0, src_x) - f32(x0);

  let base = batch * params.channels * params.in_height * params.in_width
           + c * params.in_height * params.in_width;

  let v00 = input[base + y0 * params.in_width + x0];
  let v01 = input[base + y0 * params.in_width + x1];
  let v10 = input[base + y1 * params.in_width + x0];
  let v11 = input[base + y1 * params.in_width + x1];

  let val = v00 * (1.0 - ly) * (1.0 - lx)
          + v01 * (1.0 - ly) * lx
          + v10 * ly * (1.0 - lx)
          + v11 * ly * lx;

  let out_idx = batch * params.channels * params.out_height * params.out_width
              + c * params.out_height * params.out_width
              + out_y * params.out_width
              + out_x;
  output[out_idx] = val;
}
`;

// Element-wise add
export const ADD_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= size) { return; }
  output[idx] = a[idx] + b[idx];
}
`;

// Fused bilinear upsample 2x + element-wise add (saves one dispatch)
export const UPSAMPLE_2X_ADD_SHADER = /* wgsl */ `
struct UpsampleAddParams {
  batch: u32,
  channels: u32,
  in_height: u32,
  in_width: u32,
  out_height: u32,
  out_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> skip: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: UpsampleAddParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let c_batch = gid.z;

  let c = c_batch % params.channels;
  let batch = c_batch / params.channels;

  if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
    return;
  }

  // Bilinear interpolation (align_corners=false)
  let scale_y = f32(params.in_height) / f32(params.out_height);
  let scale_x = f32(params.in_width) / f32(params.out_width);

  let src_y = (f32(out_y) + 0.5) * scale_y - 0.5;
  let src_x = (f32(out_x) + 0.5) * scale_x - 0.5;

  let y0 = u32(max(0.0, floor(src_y)));
  let x0 = u32(max(0.0, floor(src_x)));
  let y1 = min(y0 + 1u, params.in_height - 1u);
  let x1 = min(x0 + 1u, params.in_width - 1u);

  let ly = max(0.0, src_y) - f32(y0);
  let lx = max(0.0, src_x) - f32(x0);

  let in_base = batch * params.channels * params.in_height * params.in_width
              + c * params.in_height * params.in_width;

  let v00 = input[in_base + y0 * params.in_width + x0];
  let v01 = input[in_base + y0 * params.in_width + x1];
  let v10 = input[in_base + y1 * params.in_width + x0];
  let v11 = input[in_base + y1 * params.in_width + x1];

  let upsampled = v00 * (1.0 - ly) * (1.0 - lx)
                + v01 * (1.0 - ly) * lx
                + v10 * ly * (1.0 - lx)
                + v11 * ly * lx;

  // Add skip connection (skip is already at output resolution)
  let out_idx = batch * params.channels * params.out_height * params.out_width
              + c * params.out_height * params.out_width
              + out_y * params.out_width
              + out_x;

  output[out_idx] = upsampled + skip[out_idx];
}
`;

// 1x1 convolution (channel reduction)
export const CONV1X1_SHADER = /* wgsl */ `
struct Conv1x1Params {
  batch: u32,
  in_channels: u32,
  out_channels: u32,
  height: u32,
  width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv1x1Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_x = gid.x;
  let out_y = gid.y;
  let oc_batch = gid.z;

  let oc = oc_batch % params.out_channels;
  let batch = oc_batch / params.out_channels;

  if (out_x >= params.width || out_y >= params.height || batch >= params.batch) {
    return;
  }

  var sum: f32 = 0.0;
  for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
    let in_idx = batch * params.in_channels * params.height * params.width
               + ic * params.height * params.width
               + out_y * params.width
               + out_x;
    let w_idx = oc * params.in_channels + ic;
    sum = sum + input[in_idx] * weight[w_idx];
  }
  sum = sum + bias[oc];

  let out_idx = batch * params.out_channels * params.height * params.width
              + oc * params.height * params.width
              + out_y * params.width
              + out_x;
  output[out_idx] = sum;
}
`;

// Output head: 2x2 conv to scalar + sigmoid
export const OUTPUT_HEAD_SIGMOID_SHADER = /* wgsl */ `
struct OutputParams {
  batch: u32,
  in_channels: u32,
  out_channels: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: OutputParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc_batch = gid.x;
  let oc = oc_batch % params.out_channels;
  let batch = oc_batch / params.out_channels;

  if (batch >= params.batch) { return; }

  var sum: f32 = 0.0;
  for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
    for (var y: u32 = 0u; y < 2u; y = y + 1u) {
      for (var x: u32 = 0u; x < 2u; x = x + 1u) {
        let in_idx = batch * params.in_channels * 4u + ic * 4u + y * 2u + x;
        let w_idx = oc * params.in_channels * 4u + ic * 4u + y * 2u + x;
        sum = sum + input[in_idx] * weight[w_idx];
      }
    }
  }
  sum = sum + bias[oc];

  // Sigmoid
  let sigmoid_val = 1.0 / (1.0 + exp(-sum));

  output[batch * params.out_channels + oc] = sigmoid_val;
}
`;

// Output head: 2x2 conv to landmarks (no sigmoid, /256)
export const OUTPUT_HEAD_LANDMARKS_SHADER = /* wgsl */ `
struct OutputParams {
  batch: u32,
  in_channels: u32,
  out_channels: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: OutputParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc_batch = gid.x;
  let oc = oc_batch % params.out_channels;
  let batch = oc_batch / params.out_channels;

  if (batch >= params.batch) { return; }

  var sum: f32 = 0.0;
  for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
    for (var y: u32 = 0u; y < 2u; y = y + 1u) {
      for (var x: u32 = 0u; x < 2u; x = x + 1u) {
        let in_idx = batch * params.in_channels * 4u + ic * 4u + y * 2u + x;
        let w_idx = oc * params.in_channels * 4u + ic * 4u + y * 2u + x;
        sum = sum + input[in_idx] * weight[w_idx];
      }
    }
  }
  sum = sum + bias[oc];

  // Normalize by 256
  output[batch * params.out_channels + oc] = sum / 256.0;
}
`;

// Fused all 3 output heads in one dispatch (handflag, handedness, landmarks)
// Saves 2 dispatches and improves cache locality for input features
export const OUTPUT_HEADS_FUSED_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;         // 2x2x288
@group(0) @binding(1) var<storage, read> handflag_w: array<f32>;    // 1x288x2x2
@group(0) @binding(2) var<storage, read> handflag_b: array<f32>;    // 1
@group(0) @binding(3) var<storage, read> handedness_w: array<f32>;  // 1x288x2x2
@group(0) @binding(4) var<storage, read> handedness_b: array<f32>;  // 1
@group(0) @binding(5) var<storage, read> landmarks_w: array<f32>;   // 63x288x2x2
@group(0) @binding(6) var<storage, read> landmarks_b: array<f32>;   // 63
@group(0) @binding(7) var<storage, read_write> handflag: array<f32>;
@group(0) @binding(8) var<storage, read_write> handedness: array<f32>;
@group(0) @binding(9) var<storage, read_write> landmarks: array<f32>;

const IN_CHANNELS: u32 = 288u;

@compute @workgroup_size(65)  // 1 + 1 + 63 = 65 output channels total
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc = gid.x;

  // Compute conv 2x2 over 288 input channels
  var sum: f32 = 0.0;

  // Select weights and bias based on which output we're computing
  var w_base: u32;
  var bias_val: f32;

  if (oc == 0u) {
    // handflag
    for (var ic: u32 = 0u; ic < IN_CHANNELS; ic++) {
      for (var y: u32 = 0u; y < 2u; y++) {
        for (var x: u32 = 0u; x < 2u; x++) {
          let in_idx = ic * 4u + y * 2u + x;
          let w_idx = ic * 4u + y * 2u + x;
          sum += input[in_idx] * handflag_w[w_idx];
        }
      }
    }
    sum += handflag_b[0];
    // Sigmoid
    handflag[0] = 1.0 / (1.0 + exp(-sum));
  } else if (oc == 1u) {
    // handedness
    for (var ic: u32 = 0u; ic < IN_CHANNELS; ic++) {
      for (var y: u32 = 0u; y < 2u; y++) {
        for (var x: u32 = 0u; x < 2u; x++) {
          let in_idx = ic * 4u + y * 2u + x;
          let w_idx = ic * 4u + y * 2u + x;
          sum += input[in_idx] * handedness_w[w_idx];
        }
      }
    }
    sum += handedness_b[0];
    // Sigmoid
    handedness[0] = 1.0 / (1.0 + exp(-sum));
  } else {
    // landmarks (oc 2-64 → landmark 0-62)
    let landmark_oc = oc - 2u;
    for (var ic: u32 = 0u; ic < IN_CHANNELS; ic++) {
      for (var y: u32 = 0u; y < 2u; y++) {
        for (var x: u32 = 0u; x < 2u; x++) {
          let in_idx = ic * 4u + y * 2u + x;
          let w_idx = landmark_oc * IN_CHANNELS * 4u + ic * 4u + y * 2u + x;
          sum += input[in_idx] * landmarks_w[w_idx];
        }
      }
    }
    sum += landmarks_b[landmark_oc];
    // /256 normalization
    landmarks[landmark_oc] = sum / 256.0;
  }
}
`;

// Canvas/texture input → NCHW float32 padded buffer (fused conversion)
// Reads from rgba8unorm texture (NHWC uint8), writes NCHW float32 padded 257x257
// Fuses: RGBA→RGB + NHWC→NCHW + uint8→float32 + 256→257 padding in one dispatch
export const CANVAS_INPUT_SHADER = /* wgsl */ `
struct CanvasParams {
  in_size: u32,   // 256
  out_size: u32,  // 257
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: CanvasParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;

  if (x >= params.in_size || y >= params.in_size) {
    return;
  }

  // Read RGBA from texture (automatically converted to f32 [0,1] range)
  let pixel = textureLoad(input_tex, vec2<u32>(x, y), 0);

  // Write RGB channels in NCHW layout with 257 padding
  let out_stride = params.out_size * params.out_size;
  output[0u * out_stride + y * params.out_size + x] = pixel.r;
  output[1u * out_stride + y * params.out_size + x] = pixel.g;
  output[2u * out_stride + y * params.out_size + x] = pixel.b;
}
`;
