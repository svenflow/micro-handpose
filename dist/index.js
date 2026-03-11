var Ge=`
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
`,ke=`
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
`,Oe=`
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
`,Ee=`
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
`;function di(u,o){return ke.replace("@compute @workgroup_size(8, 8, 1)",`@compute @workgroup_size(${u}, ${o}, 1)`)}function fi(u,o){return Ge.replace("@compute @workgroup_size(8, 8, 1)",`@compute @workgroup_size(${u}, ${o}, 1)`)}function ci(u,o){return Oe.replace("@compute @workgroup_size(8, 8, 1)",`@compute @workgroup_size(${u}, ${o}, 1)`)}function li(u,o){return Ee.replace("@compute @workgroup_size(8, 8, 1)",`@compute @workgroup_size(${u}, ${o}, 1)`)}function hi(u,o){return[8,8]}var _i=`
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
`,gi=`
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
`,bi=`
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
`,mi=`
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
`,wi=`
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
`,yi=`
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
`,Pi=`
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
`,xi=`
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
`,Ui=`
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
    // landmarks (oc 2-64 \u2192 landmark 0-62)
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
`,vi=`
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
`;function Ci(u,o){let e=new Map,l=u.dtype??"float32";for(let m=0;m<u.keys.length;m++){let E=u.keys[m],P=u.shapes[m],x=u.offsets[m],D=P.reduce((M,w)=>M*w,1),B;if(l==="float32")B=new Float32Array(o,x,D);else{let M=new DataView(o);B=new Float32Array(D);for(let w=0;w<D;w++)B[w]=ia(M.getUint16(x+w*2,!0))}e.set(E,{data:B,shape:P})}return e}function ia(u){let o=u>>15&1,e=u>>10&31,l=u&1023;if(e===0){if(l===0)return o?-0:0;let P=-14,x=l/1024;return(o?-1:1)*Math.pow(2,P)*x}if(e===31)return l===0?o?-1/0:1/0:NaN;let m=e-15,E=1+l/1024;return(o?-1:1)*Math.pow(2,m)*E}var Bi=[{type:"resmodule",inCh:24,outCh:24,h:128,w:128,stride:1,prefix:"backbone1.3.f.0."},{type:"resmodule",inCh:24,outCh:24,h:128,w:128,stride:1,prefix:"backbone1.3.f.1."},{type:"resmodule",inCh:24,outCh:48,h:128,w:128,stride:2,prefix:"backbone1.4."},{type:"resmodule",inCh:48,outCh:48,h:64,w:64,stride:1,prefix:"backbone2.0.f.0."},{type:"resmodule",inCh:48,outCh:48,h:64,w:64,stride:1,prefix:"backbone2.0.f.1."},{type:"resmodule",inCh:48,outCh:96,h:64,w:64,stride:2,prefix:"backbone2.1."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:1,prefix:"backbone3.0.f.0."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:1,prefix:"backbone3.0.f.1."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:2,prefix:"backbone3.1."},{type:"resmodule",inCh:96,outCh:96,h:16,w:16,stride:1,prefix:"backbone4.0.f.0."},{type:"resmodule",inCh:96,outCh:96,h:16,w:16,stride:1,prefix:"backbone4.0.f.1."},{type:"resmodule",inCh:96,outCh:96,h:16,w:16,stride:2,prefix:"backbone4.1."},{type:"resmodule",inCh:96,outCh:96,h:16,w:16,stride:1,prefix:"backbone5.0."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:1,prefix:"backbone6.0."},{type:"resmodule",inCh:48,outCh:48,h:64,w:64,stride:1,prefix:"ff.0.f.0."},{type:"resmodule",inCh:48,outCh:48,h:64,w:64,stride:1,prefix:"ff.0.f.1."},{type:"resmodule",inCh:48,outCh:48,h:64,w:64,stride:1,prefix:"ff.0.f.2."},{type:"resmodule",inCh:48,outCh:48,h:64,w:64,stride:1,prefix:"ff.0.f.3."},{type:"resmodule",inCh:48,outCh:96,h:64,w:64,stride:2,prefix:"ff.1."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:1,prefix:"ff.2.f.0."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:1,prefix:"ff.2.f.1."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:1,prefix:"ff.2.f.2."},{type:"resmodule",inCh:96,outCh:96,h:32,w:32,stride:1,prefix:"ff.2.f.3."},{type:"resmodule",inCh:96,outCh:288,h:32,w:32,stride:2,prefix:"ff.3."},{type:"resmodule",inCh:288,outCh:288,h:16,w:16,stride:1,prefix:"ff.4.f.0."},{type:"resmodule",inCh:288,outCh:288,h:16,w:16,stride:1,prefix:"ff.4.f.1."},{type:"resmodule",inCh:288,outCh:288,h:16,w:16,stride:1,prefix:"ff.4.f.2."},{type:"resmodule",inCh:288,outCh:288,h:16,w:16,stride:1,prefix:"ff.4.f.3."},{type:"resmodule",inCh:288,outCh:288,h:16,w:16,stride:2,prefix:"ff.5."},{type:"resmodule",inCh:288,outCh:288,h:8,w:8,stride:1,prefix:"ff.6.f.0."},{type:"resmodule",inCh:288,outCh:288,h:8,w:8,stride:1,prefix:"ff.6.f.1."},{type:"resmodule",inCh:288,outCh:288,h:8,w:8,stride:1,prefix:"ff.6.f.2."},{type:"resmodule",inCh:288,outCh:288,h:8,w:8,stride:1,prefix:"ff.6.f.3."},{type:"resmodule",inCh:288,outCh:288,h:8,w:8,stride:2,prefix:"ff.7."},{type:"resmodule",inCh:288,outCh:288,h:4,w:4,stride:1,prefix:"ff.8.f.0."},{type:"resmodule",inCh:288,outCh:288,h:4,w:4,stride:1,prefix:"ff.8.f.1."},{type:"resmodule",inCh:288,outCh:288,h:4,w:4,stride:1,prefix:"ff.8.f.2."},{type:"resmodule",inCh:288,outCh:288,h:4,w:4,stride:1,prefix:"ff.8.f.3."},{type:"resmodule",inCh:288,outCh:288,h:4,w:4,stride:2,prefix:"ff.9."},{type:"resmodule",inCh:288,outCh:288,h:2,w:2,stride:1,prefix:"ff.10.f.0."},{type:"resmodule",inCh:288,outCh:288,h:2,w:2,stride:1,prefix:"ff.10.f.1."},{type:"resmodule",inCh:288,outCh:288,h:2,w:2,stride:1,prefix:"ff.10.f.2."},{type:"resmodule",inCh:288,outCh:288,h:2,w:2,stride:1,prefix:"ff.10.f.3."}],aa=2,ta=5,ra=8,na=11;async function Si(u){if(!navigator.gpu)throw new Error("WebGPU not supported");let o=await navigator.gpu.requestAdapter();if(!o)throw new Error("No GPU adapter found");let e=await o.requestDevice(),l=e.createShaderModule({code:_i}),m=e.createShaderModule({code:vi}),E=e.createShaderModule({code:Ui}),P=e.createShaderModule({code:ke}),x=e.createShaderModule({code:Ge}),D=e.createShaderModule({code:Oe}),B=e.createShaderModule({code:Ee}),M=e.createShaderModule({code:gi}),w=e.createShaderModule({code:bi}),R=e.createShaderModule({code:mi}),X=e.createShaderModule({code:wi}),Z=e.createShaderModule({code:yi}),V=e.createShaderModule({code:Pi}),j=e.createShaderModule({code:xi}),L=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),c=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),y=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),g=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),H=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),K=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),T=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),C=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),z=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),De=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),Me=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),I=e.createPipelineLayout({bindGroupLayouts:[L]}),W=e.createPipelineLayout({bindGroupLayouts:[c]}),Gi=e.createComputePipeline({layout:I,compute:{module:P,entryPoint:"main"}}),ki=e.createComputePipeline({layout:I,compute:{module:x,entryPoint:"main"}}),Oi=e.createComputePipeline({layout:W,compute:{module:D,entryPoint:"main"}}),Ei=e.createComputePipeline({layout:W,compute:{module:B,entryPoint:"main"}}),J=new Map,Q=new Map,ee=new Map,ie=new Map;J.set("8,8",Gi),Q.set("8,8",ki),ee.set("8,8",Oi),ie.set("8,8",Ei);function Di(n,s){let t=`${n},${s}`,a=J.get(t);if(!a){let i=e.createShaderModule({code:di(n,s)});a=e.createComputePipeline({layout:I,compute:{module:i,entryPoint:"main"}}),J.set(t,a)}return a}function Mi(n,s){let t=`${n},${s}`,a=Q.get(t);if(!a){let i=e.createShaderModule({code:fi(n,s)});a=e.createComputePipeline({layout:I,compute:{module:i,entryPoint:"main"}}),Q.set(t,a)}return a}function Ti(n,s){let t=`${n},${s}`,a=ee.get(t);if(!a){let i=e.createShaderModule({code:ci(n,s)});a=e.createComputePipeline({layout:W,compute:{module:i,entryPoint:"main"}}),ee.set(t,a)}return a}function Ai(n,s){let t=`${n},${s}`,a=ie.get(t);if(!a){let i=e.createShaderModule({code:li(n,s)});a=e.createComputePipeline({layout:W,compute:{module:i,entryPoint:"main"}}),ie.set(t,a)}return a}let A=Bi.map(n=>{let s=n.stride===2?n.h/2:n.h,t=n.stride===2?n.w/2:n.w,[a,i]=hi(n.inCh,s),h=n.h>=64,p=s>=16&&n.inCh>=288&&n.outCh>=288&&n.outCh%2===0;return{dwPipeline:h?Mi(a,i):Di(a,i),pwPipeline:p?Ai(a,i):Ti(a,i),dwDispatchX:Math.ceil(t/a),dwDispatchY:Math.ceil(s/i),dwDispatchZ:n.inCh,pwDispatchX:Math.ceil(t/a),pwDispatchY:Math.ceil(s/i),pwDispatchZ:p?n.outCh/2:n.outCh}}),Te=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[y]}),compute:{module:l,entryPoint:"main"}}),Ri=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[g]}),compute:{module:M,entryPoint:"main"}});e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[H]}),compute:{module:w,entryPoint:"main"}}),e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[K]}),compute:{module:R,entryPoint:"main"}});let ae=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[T]}),compute:{module:X,entryPoint:"main"}}),Li=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[C]}),compute:{module:Z,entryPoint:"main"}}),Ae=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[z]}),compute:{module:V,entryPoint:"main"}}),zi=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[z]}),compute:{module:j,entryPoint:"main"}}),Hi=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[De]}),compute:{module:m,entryPoint:"main"}});e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Me]}),compute:{module:E,entryPoint:"main"}});let te=1*288*128*128*4,Re=e.createBuffer({size:3*256*256*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),re=e.createBuffer({size:3*257*257*4,usage:GPUBufferUsage.STORAGE}),Le=e.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(Le,0,new Uint32Array([3,256,257]));let _=e.createBuffer({size:te,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),U=e.createBuffer({size:te,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),F=e.createBuffer({size:te,usage:GPUBufferUsage.STORAGE}),ze=e.createBuffer({size:3072*64*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),He=e.createBuffer({size:3072*32*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),Ie=e.createBuffer({size:1536*16*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),We=e.createBuffer({size:6144*64*4,usage:GPUBufferUsage.STORAGE}),ne=e.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),se=e.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),ue=e.createBuffer({size:252,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),v=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});let Fe=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ye=e.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(Ye,0,new Uint32Array([256,257]));let oe=u.get("backbone1.1.weight")?.data,pe=u.get("backbone1.1.bias")?.data;if(!oe||!pe)throw new Error("Missing input conv weights");let Ne=e.createBuffer({size:oe.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),qe=e.createBuffer({size:pe.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),$e=e.createBuffer({size:28,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(Ne,0,oe),e.queue.writeBuffer(qe,0,pe),e.queue.writeBuffer($e,0,new Uint32Array([1,3,24,257,257,128,128]));let de=u.get("backbone6.1.weight")?.data,fe=u.get("backbone6.1.bias")?.data;if(!de||!fe)throw new Error("Missing backbone6.1 conv1x1 weights");let Xe=e.createBuffer({size:de.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),Ze=e.createBuffer({size:fe.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),Ve=e.createBuffer({size:20,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(Xe,0,de),e.queue.writeBuffer(Ze,0,fe),e.queue.writeBuffer(Ve,0,new Uint32Array([1,96,48,32,32]));let ce=u.get("handflag.weight")?.data,le=u.get("handflag.bias")?.data;if(!ce||!le)throw new Error("Missing handflag weights");let he=e.createBuffer({size:ce.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),_e=e.createBuffer({size:le.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),je=e.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(he,0,ce),e.queue.writeBuffer(_e,0,le),e.queue.writeBuffer(je,0,new Uint32Array([1,288,1]));let ge=u.get("handedness.weight")?.data,be=u.get("handedness.bias")?.data;if(!ge||!be)throw new Error("Missing handedness weights");let me=e.createBuffer({size:ge.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),we=e.createBuffer({size:be.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),Ke=e.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(me,0,ge),e.queue.writeBuffer(we,0,be),e.queue.writeBuffer(Ke,0,new Uint32Array([1,288,1]));let ye=u.get("reg_3d.weight")?.data,Pe=u.get("reg_3d.bias")?.data;if(!ye||!Pe)throw new Error("Missing reg_3d weights");let xe=e.createBuffer({size:ye.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),Ue=e.createBuffer({size:Pe.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),Je=e.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(xe,0,ye),e.queue.writeBuffer(Ue,0,Pe),e.queue.writeBuffer(Je,0,new Uint32Array([1,288,63]));let Qe=Bi.map(n=>{let{inCh:s,outCh:t,h:a,w:i,stride:h,prefix:p}=n,b=h===2?a/2:a,f=h===2?i/2:i,d=h===1?2:1,r=u.get(`${p}convs.0.weight`)?.data,Be=u.get(`${p}convs.0.bias`)?.data,Ce=u.get(`${p}convs.1.weight`)?.data,Se=u.get(`${p}convs.1.bias`)?.data;if(!r||!Be||!Ce||!Se)throw new Error(`Missing weights for ${p}`);let ri=e.createBuffer({size:r.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),ni=e.createBuffer({size:Be.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),si=e.createBuffer({size:Ce.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),ui=e.createBuffer({size:Se.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),oi=e.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),pi=e.createBuffer({size:36,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});return e.queue.writeBuffer(ri,0,r),e.queue.writeBuffer(ni,0,Be),e.queue.writeBuffer(si,0,Ce),e.queue.writeBuffer(ui,0,Se),e.queue.writeBuffer(oi,0,new Uint32Array([1,s,a,i,b,f,h,d])),e.queue.writeBuffer(pi,0,new Uint32Array([1,s,t,b,f,Math.max(0,t-s),h,a,i])),{dwWeight:ri,dwBias:ni,pwWeight:si,pwBias:ui,dwUniform:oi,pwUniform:pi,spec:n,outH:b,outW:f}}),ei=e.createBuffer({size:24,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ei,0,new Uint32Array([1,96,8,8,16,16]));let ii=e.createBuffer({size:24,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ii,0,new Uint32Array([1,96,16,16,32,32]));let ai=e.createBuffer({size:24,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ai,0,new Uint32Array([1,48,32,32,64,64]));let Ii=e.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(Ii,0,new Uint32Array([1536*16]));let Wi=e.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(Wi,0,new Uint32Array([3072*32]));let Fi=e.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(Fi,0,new Uint32Array([3072*64]));let ti=e.createBindGroup({layout:y,entries:[{binding:0,resource:{buffer:Re}},{binding:1,resource:{buffer:re}},{binding:2,resource:{buffer:Le}}]}),Yi=e.createBindGroup({layout:g,entries:[{binding:0,resource:{buffer:re}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:qe}},{binding:3,resource:{buffer:_}},{binding:4,resource:{buffer:$e}}]}),S=[],G=[],k=[],O=[];for(let n of Qe)S.push(e.createBindGroup({layout:L,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:n.dwWeight}},{binding:2,resource:{buffer:n.dwBias}},{binding:3,resource:{buffer:F}},{binding:4,resource:{buffer:n.dwUniform}}]})),G.push(e.createBindGroup({layout:c,entries:[{binding:0,resource:{buffer:F}},{binding:1,resource:{buffer:_}},{binding:2,resource:{buffer:n.pwWeight}},{binding:3,resource:{buffer:n.pwBias}},{binding:4,resource:{buffer:U}},{binding:5,resource:{buffer:n.pwUniform}}]})),k.push(e.createBindGroup({layout:L,entries:[{binding:0,resource:{buffer:U}},{binding:1,resource:{buffer:n.dwWeight}},{binding:2,resource:{buffer:n.dwBias}},{binding:3,resource:{buffer:F}},{binding:4,resource:{buffer:n.dwUniform}}]})),O.push(e.createBindGroup({layout:c,entries:[{binding:0,resource:{buffer:F}},{binding:1,resource:{buffer:U}},{binding:2,resource:{buffer:n.pwWeight}},{binding:3,resource:{buffer:n.pwBias}},{binding:4,resource:{buffer:_}},{binding:5,resource:{buffer:n.pwUniform}}]}));let Ni=e.createBindGroup({layout:T,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:Ie}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:ei}}]}),qi=e.createBindGroup({layout:T,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:He}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:ii}}]}),$i=e.createBindGroup({layout:C,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:Xe}},{binding:2,resource:{buffer:Ze}},{binding:3,resource:{buffer:We}},{binding:4,resource:{buffer:Ve}}]}),Xi=e.createBindGroup({layout:T,entries:[{binding:0,resource:{buffer:We}},{binding:1,resource:{buffer:ze}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:ai}}]}),Zi=e.createBindGroup({layout:z,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:he}},{binding:2,resource:{buffer:_e}},{binding:3,resource:{buffer:ne}},{binding:4,resource:{buffer:je}}]}),Vi=e.createBindGroup({layout:z,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:me}},{binding:2,resource:{buffer:we}},{binding:3,resource:{buffer:se}},{binding:4,resource:{buffer:Ke}}]}),ji=e.createBindGroup({layout:z,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:xe}},{binding:2,resource:{buffer:Ue}},{binding:3,resource:{buffer:ue}},{binding:4,resource:{buffer:Je}}]}),Ki=e.createBindGroup({layout:De,entries:[{binding:0,resource:Fe.createView()},{binding:1,resource:{buffer:re}},{binding:2,resource:{buffer:Ye}}]});e.createBindGroup({layout:Me,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:he}},{binding:2,resource:{buffer:_e}},{binding:3,resource:{buffer:me}},{binding:4,resource:{buffer:we}},{binding:5,resource:{buffer:xe}},{binding:6,resource:{buffer:Ue}},{binding:7,resource:{buffer:ne}},{binding:8,resource:{buffer:se}},{binding:9,resource:{buffer:ue}}]});let Y=new Float32Array(1),N=new Float32Array(1),q=new Float32Array(63);function ve(n,s){let t=!0,a=0,i=n.beginComputePass();for(i.setPipeline(Ri),i.setBindGroup(0,Yi),i.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);a<=aa;a++){let f=t?S[a]:k[a],d=t?G[a]:O[a],r=A[a];i.setPipeline(r.dwPipeline),i.setBindGroup(0,f),i.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),i.setPipeline(r.pwPipeline),i.setBindGroup(0,d),i.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),t=!t}i.end();let h=t?_:U;for(n.copyBufferToBuffer(h,0,ze,0,3072*64*4),i=n.beginComputePass();a<=ta;a++){let f=t?S[a]:k[a],d=t?G[a]:O[a],r=A[a];i.setPipeline(r.dwPipeline),i.setBindGroup(0,f),i.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),i.setPipeline(r.pwPipeline),i.setBindGroup(0,d),i.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),t=!t}i.end();let p=t?_:U;for(n.copyBufferToBuffer(p,0,He,0,3072*32*4),i=n.beginComputePass();a<=ra;a++){let f=t?S[a]:k[a],d=t?G[a]:O[a],r=A[a];i.setPipeline(r.dwPipeline),i.setBindGroup(0,f),i.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),i.setPipeline(r.pwPipeline),i.setBindGroup(0,d),i.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),t=!t}i.end();let b=t?_:U;for(n.copyBufferToBuffer(b,0,Ie,0,1536*16*4),i=n.beginComputePass();a<=na;a++){let f=t?S[a]:k[a],d=t?G[a]:O[a],r=A[a];i.setPipeline(r.dwPipeline),i.setBindGroup(0,f),i.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),i.setPipeline(r.pwPipeline),i.setBindGroup(0,d),i.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),t=!t}i.setPipeline(ae),i.setBindGroup(0,Ni),i.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),i.end(),t=!1,i=n.beginComputePass();{let f=t?S[a]:k[a],d=t?G[a]:O[a],r=A[a];i.setPipeline(r.dwPipeline),i.setBindGroup(0,f),i.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),i.setPipeline(r.pwPipeline),i.setBindGroup(0,d),i.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),t=!t,a++}i.setPipeline(ae),i.setBindGroup(0,qi),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),i.end(),t=!1,i=n.beginComputePass();{let f=t?S[a]:k[a],d=t?G[a]:O[a],r=A[a];i.setPipeline(r.dwPipeline),i.setBindGroup(0,f),i.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),i.setPipeline(r.pwPipeline),i.setBindGroup(0,d),i.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),t=!t,a++}for(i.setPipeline(Li),i.setBindGroup(0,$i),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),i.setPipeline(ae),i.setBindGroup(0,Xi),i.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),i.end(),t=!1,i=n.beginComputePass();a<Qe.length;a++){let f=t?S[a]:k[a],d=t?G[a]:O[a],r=A[a];i.setPipeline(r.dwPipeline),i.setBindGroup(0,f),i.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),i.setPipeline(r.pwPipeline),i.setBindGroup(0,d),i.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),t=!t}i.setPipeline(Ae),i.setBindGroup(0,Zi),i.dispatchWorkgroups(1),i.setPipeline(Ae),i.setBindGroup(0,Vi),i.dispatchWorkgroups(1),i.setPipeline(zi),i.setBindGroup(0,ji),i.dispatchWorkgroups(1),i.end(),n.copyBufferToBuffer(ne,0,s,0,4),n.copyBufferToBuffer(se,0,s,4,4),n.copyBufferToBuffer(ue,0,s,8,252)}async function $(n){e.queue.writeBuffer(Re,0,n);let s=e.createCommandEncoder();{let a=s.beginComputePass();a.setPipeline(Te),a.setBindGroup(0,ti),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),a.end()}ve(s,v),e.queue.submit([s.finish()]),await v.mapAsync(GPUMapMode.READ);let t=new Float32Array(v.getMappedRange());return Y[0]=t[0],N[0]=t[1],q.set(t.subarray(2,65)),v.unmap(),{handflag:new Float32Array(Y),handedness:new Float32Array(N),landmarks:new Float32Array(q)}}async function Ji(n){e.queue.copyExternalImageToTexture({source:n},{texture:Fe},[256,256]);let s=e.createCommandEncoder();{let a=s.beginComputePass();a.setPipeline(Hi),a.setBindGroup(0,Ki),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}ve(s,v),e.queue.submit([s.finish()]),await v.mapAsync(GPUMapMode.READ);let t=new Float32Array(v.getMappedRange());return Y[0]=t[0],N[0]=t[1],q.set(t.subarray(2,65)),v.unmap(),{handflag:new Float32Array(Y),handedness:new Float32Array(N),landmarks:new Float32Array(q)}}async function Qi(n=50){let s=new Float32Array(196608);for(let i=0;i<5;i++)await $(s);let t=[];for(let i=0;i<n;i++){let h=performance.now();await $(s),t.push(performance.now()-h)}let a=t.reduce((i,h)=>i+h,0)/t.length;return{avgMs:a,fps:1e3/a}}async function ea(n=50){let s=new Float32Array(196608);for(let p=0;p<5;p++)await $(s);let t=[];for(let p=0;p<n;p++){let b=e.createCommandEncoder();{let d=b.beginComputePass();d.setPipeline(Te),d.setBindGroup(0,ti),d.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),d.end()}ve(b,v);let f=performance.now();e.queue.submit([b.finish()]),await e.queue.onSubmittedWorkDone(),t.push(performance.now()-f)}t.sort((p,b)=>p-b);let a=t.reduce((p,b)=>p+b,0)/t.length,i=t[Math.floor(t.length/2)],h=t[0];return{avgMs:a,fps:1e3/a,medianMs:i,minMs:h}}return{device:e,run:$,runFromCanvas:Ji,benchmark:Qi,benchmarkGPU:ea}}var sa="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ua(u={}){let{weightsUrl:o,scoreThreshold:e=.5}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let l=o??sa,m=l.endsWith("/")?`${l}weights.json`:`${l}/weights.json`,E=l.endsWith("/")?`${l}weights.bin`:`${l}/weights.bin`,[P,x]=await Promise.all([fetch(m),fetch(E)]);if(!P.ok)throw new Error(`Failed to fetch weights metadata: ${P.status}`);if(!x.ok)throw new Error(`Failed to fetch weights binary: ${x.status}`);let D=await P.json(),B=await x.arrayBuffer(),M=Ci(D,B),w=await Si(M),R=null;function X(){return R||(R=new OffscreenCanvas(256,256)),R}async function Z(c){if(c instanceof HTMLCanvasElement||c instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&c instanceof ImageBitmap)return c;let y=X();y.width=256,y.height=256;let g=y.getContext("2d");return c instanceof ImageData?g.putImageData(c,0,0):g.drawImage(c,0,0,256,256),y}function V(c,y,g){let H=c[0];if(H<e)return null;let K=y[0]>.5,T=[];for(let C=0;C<21;C++)T.push({x:g[C*3],y:g[C*3+1],z:g[C*3+2]});return{score:H,handedness:K?"right":"left",landmarks:T}}async function j(c){let y=await Z(c),g=await w.runFromCanvas(y);return V(g.handflag,g.handedness,g.landmarks)}function L(){w.device.destroy(),R=null}return{detect:j,dispose:L}}export{ua as createHandpose};
