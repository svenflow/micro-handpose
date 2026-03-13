function W(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function pt(o){let w=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],h="enable f16;"+o;for(let b of w)for(;h.includes(`${b}:array<f32>`);)h=h.replace(`${b}:array<f32>`,`${b}:array<f16>`);return h}var ea=W(`
struct DepthwiseParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:DepthwiseParams;
fn load_input(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c_batch=gid.z;
  let c=c_batch%params.channels; let batch=c_batch/params.channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  let in_base=batch*params.channels*params.in_height*params.in_width+c*params.in_height*params.in_width;
  let w_base=c*25u; let in_h=i32(params.in_height); let in_w=i32(params.in_width);
  let by=i32(out_y*params.stride)-i32(params.pad); let bx=i32(out_x*params.stride)-i32(params.pad);
  let w00=weight[w_base];let w01=weight[w_base+1u];let w02=weight[w_base+2u];let w03=weight[w_base+3u];let w04=weight[w_base+4u];
  let w10=weight[w_base+5u];let w11=weight[w_base+6u];let w12=weight[w_base+7u];let w13=weight[w_base+8u];let w14=weight[w_base+9u];
  let w20=weight[w_base+10u];let w21=weight[w_base+11u];let w22=weight[w_base+12u];let w23=weight[w_base+13u];let w24=weight[w_base+14u];
  let w30=weight[w_base+15u];let w31=weight[w_base+16u];let w32=weight[w_base+17u];let w33=weight[w_base+18u];let w34=weight[w_base+19u];
  let w40=weight[w_base+20u];let w41=weight[w_base+21u];let w42=weight[w_base+22u];let w43=weight[w_base+23u];let w44=weight[w_base+24u];
  let i00=load_input(in_base,by,bx,in_h,in_w);
  let i01=load_input(in_base,by,bx+1,in_h,in_w);
  let i02=load_input(in_base,by,bx+2,in_h,in_w);
  let i03=load_input(in_base,by,bx+3,in_h,in_w);
  let i04=load_input(in_base,by,bx+4,in_h,in_w);
  let i10=load_input(in_base,by+1,bx,in_h,in_w);
  let i11=load_input(in_base,by+1,bx+1,in_h,in_w);
  let i12=load_input(in_base,by+1,bx+2,in_h,in_w);
  let i13=load_input(in_base,by+1,bx+3,in_h,in_w);
  let i14=load_input(in_base,by+1,bx+4,in_h,in_w);
  let i20=load_input(in_base,by+2,bx,in_h,in_w);
  let i21=load_input(in_base,by+2,bx+1,in_h,in_w);
  let i22=load_input(in_base,by+2,bx+2,in_h,in_w);
  let i23=load_input(in_base,by+2,bx+3,in_h,in_w);
  let i24=load_input(in_base,by+2,bx+4,in_h,in_w);
  let i30=load_input(in_base,by+3,bx,in_h,in_w);
  let i31=load_input(in_base,by+3,bx+1,in_h,in_w);
  let i32_=load_input(in_base,by+3,bx+2,in_h,in_w);
  let i33=load_input(in_base,by+3,bx+3,in_h,in_w);
  let i34=load_input(in_base,by+3,bx+4,in_h,in_w);
  let i40=load_input(in_base,by+4,bx,in_h,in_w);
  let i41=load_input(in_base,by+4,bx+1,in_h,in_w);
  let i42=load_input(in_base,by+4,bx+2,in_h,in_w);
  let i43=load_input(in_base,by+4,bx+3,in_h,in_w);
  let i44=load_input(in_base,by+4,bx+4,in_h,in_w);
  let sum=i00*w00+i01*w01+i02*w02+i03*w03+i04*w04+i10*w10+i11*w11+i12*w12+i13*w13+i14*w14+i20*w20+i21*w21+i22*w22+i23*w23+i24*w24+i30*w30+i31*w31+i32_*w32+i33*w33+i34*w34+i40*w40+i41*w41+i42*w42+i43*w43+i44*w44+bias[c];
  let out_idx=batch*params.channels*params.out_height*params.out_width+c*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=sum;
}
`),aa=W(`
struct DepthwiseParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:DepthwiseParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c_batch=gid.z;
  let c=c_batch%params.channels; let batch=c_batch/params.channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  let in_base=batch*params.channels*params.in_height*params.in_width+c*params.in_height*params.in_width;
  let w_base=c*25u; let in_h=i32(params.in_height); let in_w=i32(params.in_width); let pad=i32(params.pad);
  let base_in_y=i32(out_y*params.stride)-pad; let base_in_x=i32(out_x*params.stride)-pad;
  var row0:f32=0.0; var row1:f32=0.0; var row2:f32=0.0; var row3:f32=0.0; var row4:f32=0.0;
  let y0=base_in_y;
  if(y0>=0 && y0<in_h){ let row_base=in_base+u32(y0)*params.in_width; for(var kx:u32=0u;kx<5u;kx=kx+1u){ let in_x=base_in_x+i32(kx); if(in_x>=0 && in_x<in_w){ row0+=input[row_base+u32(in_x)]*weight[w_base+kx]; } } }
  let y1=base_in_y+1;
  if(y1>=0 && y1<in_h){ let row_base=in_base+u32(y1)*params.in_width; for(var kx:u32=0u;kx<5u;kx=kx+1u){ let in_x=base_in_x+i32(kx); if(in_x>=0 && in_x<in_w){ row1+=input[row_base+u32(in_x)]*weight[w_base+5u+kx]; } } }
  let y2=base_in_y+2;
  if(y2>=0 && y2<in_h){ let row_base=in_base+u32(y2)*params.in_width; for(var kx:u32=0u;kx<5u;kx=kx+1u){ let in_x=base_in_x+i32(kx); if(in_x>=0 && in_x<in_w){ row2+=input[row_base+u32(in_x)]*weight[w_base+10u+kx]; } } }
  let y3=base_in_y+3;
  if(y3>=0 && y3<in_h){ let row_base=in_base+u32(y3)*params.in_width; for(var kx:u32=0u;kx<5u;kx=kx+1u){ let in_x=base_in_x+i32(kx); if(in_x>=0 && in_x<in_w){ row3+=input[row_base+u32(in_x)]*weight[w_base+15u+kx]; } } }
  let y4=base_in_y+4;
  if(y4>=0 && y4<in_h){ let row_base=in_base+u32(y4)*params.in_width; for(var kx:u32=0u;kx<5u;kx=kx+1u){ let in_x=base_in_x+i32(kx); if(in_x>=0 && in_x<in_w){ row4+=input[row_base+u32(in_x)]*weight[w_base+20u+kx]; } } }
  let sum=row0+row1+row2+row3+row4+bias[c];
  let out_idx=batch*params.channels*params.out_height*params.out_width+c*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=sum;
}
`),ta=W(`
struct PointwiseParams { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, channel_pad:u32, stride:u32, in_height:u32, in_width:u32, }
@group(0)@binding(0) var<storage,read> dw_output:array<f32>;
@group(0)@binding(1) var<storage,read> skip_input:array<f32>;
@group(0)@binding(2) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(3) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:PointwiseParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
  let dw_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels; let spatial_stride=params.height*params.width;
  var ic:u32=0u; let ic_end4=(params.in_channels/4u)*4u;
  while(ic<ic_end4){
    let dw0=dw_output[dw_base+ic*spatial_stride]; let dw1=dw_output[dw_base+(ic+1u)*spatial_stride];
    let dw2=dw_output[dw_base+(ic+2u)*spatial_stride]; let dw3=dw_output[dw_base+(ic+3u)*spatial_stride];
    let w0=pw_weight[w_base+ic]; let w1=pw_weight[w_base+ic+1u]; let w2=pw_weight[w_base+ic+2u]; let w3=pw_weight[w_base+ic+3u];
    sum0=sum0+dw0*w0; sum1=sum1+dw1*w1; sum2=sum2+dw2*w2; sum3=sum3+dw3*w3;
    ic=ic+4u;
  }
  while(ic<params.in_channels){ let dw_val=dw_output[dw_base+ic*spatial_stride]; let w_val=pw_weight[w_base+ic]; sum0=sum0+dw_val*w_val; ic=ic+1u; }
  var pw_sum=sum0+sum1+sum2+sum3+pw_bias[oc];
  var skip_val:f32=0.0;
  if(oc<params.in_channels){
    if(params.stride==2u){
      var max_val:f32=-1e38;
      for(var py:u32=0u;py<2u;py=py+1u){ for(var px:u32=0u;px<2u;px=px+1u){ let skip_y=out_y*2u+py; let skip_x=out_x*2u+px; if(skip_y<params.in_height && skip_x<params.in_width){ let skip_idx=batch*params.in_channels*params.in_height*params.in_width+oc*params.in_height*params.in_width+skip_y*params.in_width+skip_x; max_val=max(max_val,skip_input[skip_idx]); } } }
      skip_val=max_val;
    } else {
      let skip_idx=batch*params.in_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
      skip_val=skip_input[skip_idx];
    }
  }
  let result=max(0.0,pw_sum+skip_val);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),ia=W(`
struct PointwiseParams { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, channel_pad:u32, stride:u32, in_height:u32, in_width:u32, }
@group(0)@binding(0) var<storage,read> dw_output:array<f32>;
@group(0)@binding(1) var<storage,read> skip_input:array<f32>;
@group(0)@binding(2) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(3) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:PointwiseParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_pair_batch=gid.z;
  let half_out_channels=params.out_channels/2u; let oc_pair=oc_pair_batch%half_out_channels; let batch=oc_pair_batch/half_out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  let oc0=oc_pair*2u; let oc1=oc0+1u;
  var sum0_a:f32=0.0; var sum0_b:f32=0.0; var sum0_c:f32=0.0; var sum0_d:f32=0.0;
  var sum1_a:f32=0.0; var sum1_b:f32=0.0; var sum1_c:f32=0.0; var sum1_d:f32=0.0;
  let dw_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base0=oc0*params.in_channels; let w_base1=oc1*params.in_channels; let spatial_stride=params.height*params.width;
  var ic:u32=0u; let ic_end4=(params.in_channels/4u)*4u;
  while(ic<ic_end4){
    let in0=dw_output[dw_base+ic*spatial_stride]; let in1=dw_output[dw_base+(ic+1u)*spatial_stride];
    let in2=dw_output[dw_base+(ic+2u)*spatial_stride]; let in3=dw_output[dw_base+(ic+3u)*spatial_stride];
    sum0_a+=in0*pw_weight[w_base0+ic]; sum0_b+=in1*pw_weight[w_base0+ic+1u]; sum0_c+=in2*pw_weight[w_base0+ic+2u]; sum0_d+=in3*pw_weight[w_base0+ic+3u];
    sum1_a+=in0*pw_weight[w_base1+ic]; sum1_b+=in1*pw_weight[w_base1+ic+1u]; sum1_c+=in2*pw_weight[w_base1+ic+2u]; sum1_d+=in3*pw_weight[w_base1+ic+3u];
    ic+=4u;
  }
  while(ic<params.in_channels){ let in_val=dw_output[dw_base+ic*spatial_stride]; sum0_a+=in_val*pw_weight[w_base0+ic]; sum1_a+=in_val*pw_weight[w_base1+ic]; ic+=1u; }
  let pw_sum0=sum0_a+sum0_b+sum0_c+sum0_d+pw_bias[oc0];
  let pw_sum1=sum1_a+sum1_b+sum1_c+sum1_d+pw_bias[oc1];
  var skip_val0:f32=0.0;
  if(oc0<params.in_channels){
    if(params.stride==2u){ var max_val:f32=-1e38; for(var py:u32=0u;py<2u;py++){ for(var px:u32=0u;px<2u;px++){ let skip_y=out_y*2u+py; let skip_x=out_x*2u+px; if(skip_y<params.in_height && skip_x<params.in_width){ let skip_idx=batch*params.in_channels*params.in_height*params.in_width+oc0*params.in_height*params.in_width+skip_y*params.in_width+skip_x; max_val=max(max_val,skip_input[skip_idx]); } } } skip_val0=max_val;
    } else { let skip_idx=batch*params.in_channels*params.height*params.width+oc0*params.height*params.width+out_y*params.width+out_x; skip_val0=skip_input[skip_idx]; }
  }
  var skip_val1:f32=0.0;
  if(oc1<params.in_channels){
    if(params.stride==2u){ var max_val:f32=-1e38; for(var py:u32=0u;py<2u;py++){ for(var px:u32=0u;px<2u;px++){ let skip_y=out_y*2u+py; let skip_x=out_x*2u+px; if(skip_y<params.in_height && skip_x<params.in_width){ let skip_idx=batch*params.in_channels*params.in_height*params.in_width+oc1*params.in_height*params.in_width+skip_y*params.in_width+skip_x; max_val=max(max_val,skip_input[skip_idx]); } } } skip_val1=max_val;
    } else { let skip_idx=batch*params.in_channels*params.height*params.width+oc1*params.height*params.width+out_y*params.width+out_x; skip_val1=skip_input[skip_idx]; }
  }
  let result0=max(0.0,pw_sum0+skip_val0); let result1=max(0.0,pw_sum1+skip_val1);
  let out_base=batch*params.out_channels*params.height*params.width;
  output[out_base+oc0*params.height*params.width+out_y*params.width+out_x]=result0;
  if(oc1<params.out_channels){ output[out_base+oc1*params.height*params.width+out_y*params.width+out_x]=result1; }
}
`);function dt(o,w){return aa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function ct(o,w){return ea.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function _t(o,w){return ta.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function lt(o,w){return ia.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function wt(o,w){return[8,8]}var mt=W(`
struct PadParams { channels:u32, in_size:u32, out_size:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:PadParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y; let c=gid.z;
  if(x>=params.in_size||y>=params.in_size||c>=params.channels){return;}
  let in_idx=c*params.in_size*params.in_size+y*params.in_size+x;
  let out_idx=c*params.out_size*params.out_size+y*params.out_size+x;
  output[out_idx]=input[in_idx];
}
`),ft=W(`
struct Conv3x3Params { batch:u32, in_channels:u32, out_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Conv3x3Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){ for(var ky:u32=0u;ky<3u;ky=ky+1u){ for(var kx:u32=0u;kx<3u;kx=kx+1u){
    let in_y=out_y*2u+ky; let in_x=out_x*2u+kx;
    if(in_y<params.in_height && in_x<params.in_width){
      let in_idx=batch*params.in_channels*params.in_height*params.in_width+ic*params.in_height*params.in_width+in_y*params.in_width+in_x;
      let w_idx=oc*params.in_channels*9u+ic*9u+ky*3u+kx;
      sum=sum+input[in_idx]*weight[w_idx];
    }
  } } }
  sum=sum+bias[oc]; sum=max(0.0,sum);
  let out_idx=batch*params.out_channels*params.out_height*params.out_width+oc*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=sum;
}
`);function ht(o){return W(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${o?`@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> skip:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> params:UpsampleParams;`:`@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:UpsampleParams;`}
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c_batch=gid.z;
  let c=c_batch%params.channels; let batch=c_batch/params.channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  let scale_y=f32(params.in_height)/f32(params.out_height); let scale_x=f32(params.in_width)/f32(params.out_width);
  let src_y=(f32(out_y)+0.5)*scale_y-0.5; let src_x=(f32(out_x)+0.5)*scale_x-0.5;
  let y0=u32(max(0.0,floor(src_y))); let x0=u32(max(0.0,floor(src_x)));
  let y1=min(y0+1u,params.in_height-1u); let x1=min(x0+1u,params.in_width-1u);
  let ly=max(0.0,src_y)-f32(y0); let lx=max(0.0,src_x)-f32(x0);
  let base=batch*params.channels*params.in_height*params.in_width+c*params.in_height*params.in_width;
  let v00=input[base+y0*params.in_width+x0]; let v01=input[base+y0*params.in_width+x1];
  let v10=input[base+y1*params.in_width+x0]; let v11=input[base+y1*params.in_width+x1];
  let val=v00*(1.0-ly)*(1.0-lx)+v01*(1.0-ly)*lx+v10*ly*(1.0-lx)+v11*ly*lx;
  let out_idx=batch*params.channels*params.out_height*params.out_width+c*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=val${o?"+skip[out_idx]":""};
}
`)}var bt=ht(!1),gt=ht(!0),yt=W(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),xt=W(`
struct Conv1x1Params { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Conv1x1Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){
    let in_idx=batch*params.in_channels*params.height*params.width+ic*params.height*params.width+out_y*params.width+out_x;
    let w_idx=oc*params.in_channels+ic;
    sum=sum+input[in_idx]*weight[w_idx];
  }
  sum=sum+bias[oc];
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=sum;
}
`);function vt(o){return W(`
struct OutputParams { batch:u32, in_channels:u32, out_channels:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:OutputParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc_batch=gid.x; let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(batch>=params.batch){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){ for(var y:u32=0u;y<2u;y=y+1u){ for(var x:u32=0u;x<2u;x=x+1u){
    let in_idx=batch*params.in_channels*4u+ic*4u+y*2u+x;
    let w_idx=oc*params.in_channels*4u+ic*4u+y*2u+x;
    sum=sum+input[in_idx]*weight[w_idx];
  } } }
  sum=sum+bias[oc];
  ${o==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var Pt=vt("sigmoid"),kt=vt("div256"),Bt=W(`
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> handflag_w:array<f32>;
@group(0)@binding(2) var<storage,read> handflag_b:array<f32>;
@group(0)@binding(3) var<storage,read> handedness_w:array<f32>;
@group(0)@binding(4) var<storage,read> handedness_b:array<f32>;
@group(0)@binding(5) var<storage,read> landmarks_w:array<f32>;
@group(0)@binding(6) var<storage,read> landmarks_b:array<f32>;
@group(0)@binding(7) var<storage,read_write> output:array<f32>;
const IN_CHANNELS:u32=288u;
@compute @workgroup_size(65)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  var sum:f32=0.0;
  if(oc==0u){
    for(var ic:u32=0u;ic<IN_CHANNELS;ic++){ for(var y:u32=0u;y<2u;y++){ for(var x:u32=0u;x<2u;x++){
      let in_idx=ic*4u+y*2u+x; sum+=input[in_idx]*handflag_w[in_idx];
    } } }
    sum+=handflag_b[0]; output[0]=1.0/(1.0+exp(-sum));
  } else if(oc==1u){
    for(var ic:u32=0u;ic<IN_CHANNELS;ic++){ for(var y:u32=0u;y<2u;y++){ for(var x:u32=0u;x<2u;x++){
      let in_idx=ic*4u+y*2u+x; sum+=input[in_idx]*handedness_w[in_idx];
    } } }
    sum+=handedness_b[0]; output[1]=1.0/(1.0+exp(-sum));
  } else {
    let landmark_oc=oc-2u;
    for(var ic:u32=0u;ic<IN_CHANNELS;ic++){ for(var y:u32=0u;y<2u;y++){ for(var x:u32=0u;x<2u;x++){
      let in_idx=ic*4u+y*2u+x; let w_idx=landmark_oc*IN_CHANNELS*4u+ic*4u+y*2u+x;
      sum+=input[in_idx]*landmarks_w[w_idx];
    } } }
    sum+=landmarks_b[landmark_oc]; output[oc]=sum/256.0;
  }
}
`),Ut=W(`
struct FusedParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,288>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(256,1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution \u2014 stride loop over 288 channels with 256 threads
  for(var c:u32=lid.x;c<288u;c+=256u){
    let in_base=c*params.in_height*params.in_width;
    let w_base=c*25u;
    let by=i32(out_y*params.stride)-i32(params.pad);
    let bx=i32(out_x*params.stride)-i32(params.pad);
    var dw_sum:f32=0.0;
    let w00=dw_weight[w_base];let w01=dw_weight[w_base+1u];let w02=dw_weight[w_base+2u];let w03=dw_weight[w_base+3u];let w04=dw_weight[w_base+4u];
    let w10=dw_weight[w_base+5u];let w11=dw_weight[w_base+6u];let w12=dw_weight[w_base+7u];let w13=dw_weight[w_base+8u];let w14=dw_weight[w_base+9u];
    let w20=dw_weight[w_base+10u];let w21=dw_weight[w_base+11u];let w22=dw_weight[w_base+12u];let w23=dw_weight[w_base+13u];let w24=dw_weight[w_base+14u];
    let w30=dw_weight[w_base+15u];let w31=dw_weight[w_base+16u];let w32=dw_weight[w_base+17u];let w33=dw_weight[w_base+18u];let w34=dw_weight[w_base+19u];
    let w40=dw_weight[w_base+20u];let w41=dw_weight[w_base+21u];let w42=dw_weight[w_base+22u];let w43=dw_weight[w_base+23u];let w44=dw_weight[w_base+24u];
    let i00=load_input_f(in_base,by,bx,inH,inW);
    let i01=load_input_f(in_base,by,bx+1,inH,inW);
    let i02=load_input_f(in_base,by,bx+2,inH,inW);
    let i03=load_input_f(in_base,by,bx+3,inH,inW);
    let i04=load_input_f(in_base,by,bx+4,inH,inW);
    let i10=load_input_f(in_base,by+1,bx,inH,inW);
    let i11=load_input_f(in_base,by+1,bx+1,inH,inW);
    let i12=load_input_f(in_base,by+1,bx+2,inH,inW);
    let i13=load_input_f(in_base,by+1,bx+3,inH,inW);
    let i14=load_input_f(in_base,by+1,bx+4,inH,inW);
    let i20=load_input_f(in_base,by+2,bx,inH,inW);
    let i21=load_input_f(in_base,by+2,bx+1,inH,inW);
    let i22=load_input_f(in_base,by+2,bx+2,inH,inW);
    let i23=load_input_f(in_base,by+2,bx+3,inH,inW);
    let i24=load_input_f(in_base,by+2,bx+4,inH,inW);
    let i30=load_input_f(in_base,by+3,bx,inH,inW);
    let i31=load_input_f(in_base,by+3,bx+1,inH,inW);
    let i32_=load_input_f(in_base,by+3,bx+2,inH,inW);
    let i33=load_input_f(in_base,by+3,bx+3,inH,inW);
    let i34=load_input_f(in_base,by+3,bx+4,inH,inW);
    let i40=load_input_f(in_base,by+4,bx,inH,inW);
    let i41=load_input_f(in_base,by+4,bx+1,inH,inW);
    let i42=load_input_f(in_base,by+4,bx+2,inH,inW);
    let i43=load_input_f(in_base,by+4,bx+3,inH,inW);
    let i44=load_input_f(in_base,by+4,bx+4,inH,inW);
    dw_sum=i00*w00+i01*w01+i02*w02+i03*w03+i04*w04+i10*w10+i11*w11+i12*w12+i13*w13+i14*w14+i20*w20+i21*w21+i22*w22+i23*w23+i24*w24+i30*w30+i31*w31+i32_*w32+i33*w33+i34*w34+i40*w40+i41*w41+i42*w42+i43*w43+i44*w44+dw_bias[c];
    shared_dw[c]=dw_sum;
  }
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU \u2014 stride loop over 288 channels
  for(var c:u32=lid.x;c<288u;c+=256u){
    let pw_base=c*288u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    var ic:u32=0u;
    while(ic<288u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];
    // Skip connection
    var skip_val:f32=0.0;
    if(params.stride==2u){
      var max_val:f32=-1e38;
      for(var py:u32=0u;py<2u;py++){
        for(var px:u32=0u;px<2u;px++){
          let skip_y=out_y*2u+py; let skip_x=out_x*2u+px;
          if(skip_y<params.in_height && skip_x<params.in_width){
            let skip_idx=c*params.in_height*params.in_width+skip_y*params.in_width+skip_x;
            max_val=max(max_val,input[skip_idx]);
          }
        }
      }
      skip_val=max_val;
    } else {
      skip_val=input[c*outH*outW+out_y*outW+out_x];
    }
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`);function St(o,w){let b=Math.min(w,256),U=w>b,D=o%4===0?`var ic:u32=0u;
    while(ic<${o}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${o}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,R=`var skip_val:f32=0.0;
    if(c<${o}u){
      if(params.stride==2u){
        var max_val:f32=-1e38;
        for(var py:u32=0u;py<2u;py++){
          for(var px:u32=0u;px<2u;px++){
            let skip_y=out_y*2u+py; let skip_x=out_x*2u+px;
            if(skip_y<params.in_height && skip_x<params.in_width){
              let skip_idx=c*params.in_height*params.in_width+skip_y*params.in_width+skip_x;
              max_val=max(max_val,input[skip_idx]);
            }
          }
        }
        skip_val=max_val;
      } else {
        skip_val=input[c*outH*outW+out_y*outW+out_x];
      }
    }`,E=o===w?"":`if(c<${o}u){`,M=o===w?"":"}",x=U?`for(var c:u32=lid.x;c<${o}u;c+=${b}u){`:`let c=lid.x;
  ${E}`,X=U?"}":M,g=U?`for(var c:u32=lid.x;c<${w}u;c+=${b}u){`:"{let c=lid.x;";return W(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${o}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${b},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${x}
  let in_base=c*params.in_height*params.in_width;
  let w_base=c*25u;
  let by=i32(out_y*params.stride)-i32(params.pad);
  let bx=i32(out_x*params.stride)-i32(params.pad);
  var dw_sum:f32=0.0;
  let w00=dw_weight[w_base];let w01=dw_weight[w_base+1u];let w02=dw_weight[w_base+2u];let w03=dw_weight[w_base+3u];let w04=dw_weight[w_base+4u];
  let w10=dw_weight[w_base+5u];let w11=dw_weight[w_base+6u];let w12=dw_weight[w_base+7u];let w13=dw_weight[w_base+8u];let w14=dw_weight[w_base+9u];
  let w20=dw_weight[w_base+10u];let w21=dw_weight[w_base+11u];let w22=dw_weight[w_base+12u];let w23=dw_weight[w_base+13u];let w24=dw_weight[w_base+14u];
  let w30=dw_weight[w_base+15u];let w31=dw_weight[w_base+16u];let w32=dw_weight[w_base+17u];let w33=dw_weight[w_base+18u];let w34=dw_weight[w_base+19u];
  let w40=dw_weight[w_base+20u];let w41=dw_weight[w_base+21u];let w42=dw_weight[w_base+22u];let w43=dw_weight[w_base+23u];let w44=dw_weight[w_base+24u];
  let i00=load_input_f(in_base,by,bx,inH,inW);
  let i01=load_input_f(in_base,by,bx+1,inH,inW);
  let i02=load_input_f(in_base,by,bx+2,inH,inW);
  let i03=load_input_f(in_base,by,bx+3,inH,inW);
  let i04=load_input_f(in_base,by,bx+4,inH,inW);
  let i10=load_input_f(in_base,by+1,bx,inH,inW);
  let i11=load_input_f(in_base,by+1,bx+1,inH,inW);
  let i12=load_input_f(in_base,by+1,bx+2,inH,inW);
  let i13=load_input_f(in_base,by+1,bx+3,inH,inW);
  let i14=load_input_f(in_base,by+1,bx+4,inH,inW);
  let i20=load_input_f(in_base,by+2,bx,inH,inW);
  let i21=load_input_f(in_base,by+2,bx+1,inH,inW);
  let i22=load_input_f(in_base,by+2,bx+2,inH,inW);
  let i23=load_input_f(in_base,by+2,bx+3,inH,inW);
  let i24=load_input_f(in_base,by+2,bx+4,inH,inW);
  let i30=load_input_f(in_base,by+3,bx,inH,inW);
  let i31=load_input_f(in_base,by+3,bx+1,inH,inW);
  let i32_=load_input_f(in_base,by+3,bx+2,inH,inW);
  let i33=load_input_f(in_base,by+3,bx+3,inH,inW);
  let i34=load_input_f(in_base,by+3,bx+4,inH,inW);
  let i40=load_input_f(in_base,by+4,bx,inH,inW);
  let i41=load_input_f(in_base,by+4,bx+1,inH,inW);
  let i42=load_input_f(in_base,by+4,bx+2,inH,inW);
  let i43=load_input_f(in_base,by+4,bx+3,inH,inW);
  let i44=load_input_f(in_base,by+4,bx+4,inH,inW);
  dw_sum=i00*w00+i01*w01+i02*w02+i03*w03+i04*w04+i10*w10+i11*w11+i12*w12+i13*w13+i14*w14+i20*w20+i21*w21+i22*w22+i23*w23+i24*w24+i30*w30+i31*w31+i32_*w32+i33*w33+i34*w34+i40*w40+i41*w41+i42*w42+i43*w43+i44*w44+dw_bias[c];
  shared_dw[c]=dw_sum;
  ${X}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${g}
    let pw_base=c*${o}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${D}
    // Skip connection (only for c < inCh)
    ${R}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var Gt=W(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),Ct=W(`
@group(0)@binding(0) var<storage,read> data:array<f32>;
@fragment fn fs(@builtin(position) pos:vec4<f32>)->@location(0) vec4<f32>{
  let x=u32(pos.x); let y=u32(pos.y);
  let idx=y*9u+x;
  if(idx>=65u){return vec4(0,0,0,1);}
  let bits=bitcast<u32>(data[idx]);
  let r=f32((bits>>24u)&0xFFu)/255.0;
  let g=f32((bits>>16u)&0xFFu)/255.0;
  let b=f32((bits>>8u)&0xFFu)/255.0;
  let a=f32(bits&0xFFu)/255.0;
  return vec4(r,g,b,a);
}
`),Dt=W(`
struct CanvasParams { in_size:u32, out_size:u32, }
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CanvasParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y;
  if(x>=params.in_size||y>=params.in_size){return;}
  let pixel=textureLoad(input_tex,vec2<u32>(x,y),0);
  let out_stride=params.out_size*params.out_size;
  output[0u*out_stride+y*params.out_size+x]=pixel.r;
  output[1u*out_stride+y*params.out_size+x]=pixel.g;
  output[2u*out_stride+y*params.out_size+x]=pixel.b;
}
`);function Wt(o,w){let h=new Map,b=o.dtype??"float32";for(let U=0;U<o.keys.length;U++){let a=o.keys[U],D=o.shapes[U],R=o.offsets[U],E=D.reduce((X,g)=>X*g,1),M,x;if(b==="float32")M=new Float32Array(w,R,E);else{let X=new DataView(w);M=new Float32Array(E);for(let g=0;g<E;g++)M[g]=xi(X.getUint16(R+g*2,!0));x=w.slice(R,R+E*2)}h.set(a,{data:M,shape:D,rawF16:x})}return h}function xi(o){let w=o>>15&1,h=o>>10&31,b=o&1023;if(h===0){if(b===0)return w?-0:0;let D=-14,R=b/1024;return(w?-1:1)*Math.pow(2,D)*R}if(h===31)return b===0?w?-1/0:1/0:NaN;let U=h-15,a=1+b/1024;return(w?-1:1)*Math.pow(2,U)*a}var vi=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],At=vi.map(([o,w,h,b,U])=>({type:"resmodule",inCh:o,outCh:w,h,w:h,stride:b,prefix:U})),Pi=2,ki=5,Bi=8,Ui=11;async function na(o,w){if(!navigator.gpu)throw new Error("WebGPU not supported");let h=await navigator.gpu.requestAdapter();if(!h)throw new Error("No GPU adapter found");let b=h.features.has("shader-f16"),U=b?["shader-f16"]:[],a=await h.requestDevice({requiredFeatures:U,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(h.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(h.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(h.limits.maxComputeInvocationsPerWorkgroup,288)}}),D=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(b)try{let t=`enable f16;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> dw_weight: array<f16>;
@group(0) @binding(2) var<storage, read> dw_bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> intermediate: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= 96u) { return; }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < 25u; i = i + 1u) {
    sum = sum + input[ch * 25u + i] * f32(dw_weight[ch * 25u + i]);
  }
  intermediate[ch] = max(0.0, sum + f32(dw_bias[ch]));
}`,s=`enable f16;
@group(0) @binding(0) var<storage, read> intermediate: array<f32>;
@group(0) @binding(1) var<storage, read> pw_weight: array<f16>;
@group(0) @binding(2) var<storage, read> pw_bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc = gid.x;
  if (oc >= 96u) { return; }
  var sum: f32 = 0.0;
  for (var ic: u32 = 0u; ic < 96u; ic = ic + 1u) {
    sum = sum + intermediate[ic] * f32(pw_weight[oc * 96u + ic]);
  }
  output[oc] = sum + f32(pw_bias[oc]);
}`,n=a.createShaderModule({code:t}),i=a.createShaderModule({code:s}),e=await n.getCompilationInfo(),c=await i.getCompilationInfo();if(e.messages.some(d=>d.type==="error")||c.messages.some(d=>d.type==="error"))D=!1;else{let d=new Float32Array(2400);d.fill(1);let _=new Uint16Array(2400);_.fill(10516);let p=new Uint16Array(96);p.fill(14336);let u=new Uint16Array(9216);u.fill(8478);let r=new Uint16Array(96);r.fill(12288);let m=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,P=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,C=a.createBuffer({size:d.byteLength,usage:m}),se=a.createBuffer({size:_.byteLength,usage:m}),ue=a.createBuffer({size:p.byteLength,usage:m}),oe=a.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),pe=a.createBuffer({size:u.byteLength,usage:m}),de=a.createBuffer({size:r.byteLength,usage:m}),ce=a.createBuffer({size:384,usage:P}),J=a.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});a.queue.writeBuffer(C,0,d),a.queue.writeBuffer(se,0,_),a.queue.writeBuffer(ue,0,p),a.queue.writeBuffer(pe,0,u),a.queue.writeBuffer(de,0,r);let Y="read-only-storage",be="storage",ge=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Y}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Y}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Y}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:be}}]}),st=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Y}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Y}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Y}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:be}}]}),hi=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[ge]}),compute:{module:n,entryPoint:"main"}}),bi=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[st]}),compute:{module:i,entryPoint:"main"}}),gi=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:C}},{binding:1,resource:{buffer:se}},{binding:2,resource:{buffer:ue}},{binding:3,resource:{buffer:oe}}]}),yi=a.createBindGroup({layout:st,entries:[{binding:0,resource:{buffer:oe}},{binding:1,resource:{buffer:pe}},{binding:2,resource:{buffer:de}},{binding:3,resource:{buffer:ce}}]}),We=a.createCommandEncoder(),Me=We.beginComputePass();Me.setPipeline(hi),Me.setBindGroup(0,gi),Me.dispatchWorkgroups(2),Me.end();let Ee=We.beginComputePass();Ee.setPipeline(bi),Ee.setBindGroup(0,yi),Ee.dispatchWorkgroups(2),Ee.end(),We.copyBufferToBuffer(ce,0,J,0,384),a.queue.submit([We.finish()]),await a.queue.onSubmittedWorkDone(),await J.mapAsync(GPUMapMode.READ);let _e=new Float32Array(J.getMappedRange()),Qe=1.5*.0104*96+.25,ut=_e[0]!==0&&_e[47]!==0&&_e[95]!==0,ot=Math.abs(_e[0]-Qe)<1;D=ut&&ot,console.log(`[micro-handpose] f16 validation: result[0]=${_e[0]}, expected=${Qe.toFixed(2)}, allNonZero=${ut}, closeEnough=${ot}, f16Works=${D}`),J.unmap(),C.destroy(),se.destroy(),ue.destroy(),oe.destroy(),pe.destroy(),de.destroy(),ce.destroy(),J.destroy(),D||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${_e[0]}, expected ~${Qe.toFixed(2)}) \u2014 falling back to f32`)}}catch{D=!1}let E=o.values().next().value,M=D&&!!E?.rawF16&&!w?.forceF32;console.log(M?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${b}, f16 validated: ${D}, f16 data: ${!!E?.rawF16})`);function x(t){if(M&&t.rawF16){let s=new Uint16Array(t.rawF16);if(s.length%2!==0){let n=new Uint16Array(s.length+1);return n.set(s),n}return s}return t.data}function X(t){if(M&&t.rawF16){let s=t.rawF16.byteLength;return Math.ceil(s/4)*4}return t.data.byteLength}function g(t){return M?pt(t):t}let H={r:"read-only-storage",s:"storage",u:"uniform"};function A(t){return a.createBindGroupLayout({entries:t.map((s,n)=>({binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[s]}}))})}function He(t){return a.createBindGroupLayout({entries:t.map((s,n)=>s==="t"?{binding:n,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[s]}})})}let y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,le=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,we=GPUBufferUsage.STORAGE,ye=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,L=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(t,s){return a.createBuffer({size:t,usage:s})}function S(t,s){return a.createBindGroup({layout:t,entries:s.map((n,i)=>({binding:i,resource:"size"in n?{buffer:n}:n}))})}function T(t,s){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:s,entryPoint:"main"}})}let f=a.createShaderModule({code:mt}),v=a.createShaderModule({code:Dt}),k=a.createShaderModule({code:g(Bt)}),me=a.createShaderModule({code:g(aa)}),Q=a.createShaderModule({code:g(ea)}),xe=a.createShaderModule({code:g(ta)}),ee=a.createShaderModule({code:g(ia)}),Mt=a.createShaderModule({code:g(ft)}),Et=a.createShaderModule({code:bt}),Ht=a.createShaderModule({code:yt}),Tt=a.createShaderModule({code:gt}),Lt=a.createShaderModule({code:g(xt)}),Ot=a.createShaderModule({code:g(Pt)}),Rt=a.createShaderModule({code:g(kt)}),Ft=a.createShaderModule({code:g(Ut)}),ra=new Map;function Ci(t,s){let n=`${t}_${s}`,i=ra.get(n);return i||(i=a.createShaderModule({code:g(St(t,s))}),ra.set(n,i)),i}let Te=A(["r","r","r","s","u"]),Le=A(["r","r","r","r","s","u"]),sa=A(["r","s","u"]),ua=A(["r","r","r","s","u"]),It=A(["r","s","u"]),zt=A(["r","r","s","u"]),ve=A(["r","r","s","u"]),oa=A(["r","r","r","s","u"]),fe=A(["r","r","r","s","u"]),pa=He(["t","s","u"]),da=A(["r","r","r","r","r","r","r","s"]),Oe=A(["r","r","r","r","r","s","u"]),qt=a.createPipelineLayout({bindGroupLayouts:[Te]}),$t=a.createPipelineLayout({bindGroupLayouts:[Le]}),Pe=t=>a.createComputePipeline({layout:qt,compute:{module:t,entryPoint:"main"}}),ke=t=>a.createComputePipeline({layout:$t,compute:{module:t,entryPoint:"main"}}),Nt=Pe(me),Yt=Pe(Q),Xt=ke(xe),Zt=ke(ee),ca=new Map,_a=new Map,la=new Map,wa=new Map;ca.set("8,8",Nt),_a.set("8,8",Yt),la.set("8,8",Xt),wa.set("8,8",Zt);function Be(t,s,n,i,e){let c=`${s},${n}`,d=t.get(c);return d||(d=e(a.createShaderModule({code:g(i(s,n))})),t.set(c,d)),d}let Vt=(t,s)=>Be(ca,t,s,dt,Pe),Kt=(t,s)=>Be(_a,t,s,ct,Pe),jt=(t,s)=>Be(la,t,s,_t,ke),Jt=(t,s)=>Be(wa,t,s,lt,ke),Z=At.map(t=>{let s=t.stride===2?t.h/2:t.h,n=t.stride===2?t.w/2:t.w,[i,e]=wt(t.inCh,s),c=t.h>=64,d=s>=16&&t.inCh>=288&&t.outCh>=288&&t.outCh%2===0;return{dwPipeline:c?Kt(i,e):Vt(i,e),pwPipeline:d?Jt(i,e):jt(i,e),dwDispatchX:Math.ceil(n/i),dwDispatchY:Math.ceil(s/e),dwDispatchZ:t.inCh,pwDispatchX:Math.ceil(n/i),pwDispatchY:Math.ceil(s/e),pwDispatchZ:d?t.outCh/2:t.outCh}}),ma=T(sa,f),fa=T(ua,Mt);T(It,Et),T(zt,Ht);let Re=T(ve,Tt),Qt=T(oa,Lt);T(fe,Ot),T(fe,Rt);let V=T(pa,v),ei=T(da,k),ai=T(Oe,Ft),Fe=1*288*128*128*4,ha=l(3*256*256*4,y),he=l(3*257*257*4,we),ba=l(12,L);a.queue.writeBuffer(ba,0,new Uint32Array([3,256,257]));let G=l(Fe,le),O=l(Fe,ye),te=l(Fe,we),ga=l(3072*64*4,y),ya=l(3072*32*4,y),xa=l(1536*16*4,y),va=l(6144*64*4,we),ie=l(260,ye),B=l(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);l(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let F=a.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Pa=l(8,L);a.queue.writeBuffer(Pa,0,new Uint32Array([256,257]));let ka=o.get("backbone1.1.weight"),Ba=o.get("backbone1.1.bias");if(!ka||!Ba)throw new Error("Missing input conv weights");let Ua=x(ka),Sa=x(Ba),Ga=l(Ua.byteLength,y),Ca=l(Sa.byteLength,y),Da=l(28,L);a.queue.writeBuffer(Ga,0,Ua),a.queue.writeBuffer(Ca,0,Sa),a.queue.writeBuffer(Da,0,new Uint32Array([1,3,24,257,257,128,128]));let Aa=o.get("backbone6.1.weight"),Wa=o.get("backbone6.1.bias");if(!Aa||!Wa)throw new Error("Missing backbone6.1 conv1x1 weights");let Ma=x(Aa),Ea=x(Wa),Ha=l(Ma.byteLength,y),Ta=l(Ea.byteLength,y),La=l(20,L);a.queue.writeBuffer(Ha,0,Ma),a.queue.writeBuffer(Ta,0,Ea),a.queue.writeBuffer(La,0,new Uint32Array([1,96,48,32,32]));let Oa=o.get("handflag.weight"),Ra=o.get("handflag.bias");if(!Oa||!Ra)throw new Error("Missing handflag weights");let Fa=x(Oa),Ia=x(Ra),Ie=l(Fa.byteLength,y),ze=l(Ia.byteLength,y),za=l(12,L);a.queue.writeBuffer(Ie,0,Fa),a.queue.writeBuffer(ze,0,Ia),a.queue.writeBuffer(za,0,new Uint32Array([1,288,1]));let qa=o.get("handedness.weight"),$a=o.get("handedness.bias");if(!qa||!$a)throw new Error("Missing handedness weights");let Na=x(qa),Ya=x($a),qe=l(Na.byteLength,y),$e=l(Ya.byteLength,y),Xa=l(12,L);a.queue.writeBuffer(qe,0,Na),a.queue.writeBuffer($e,0,Ya),a.queue.writeBuffer(Xa,0,new Uint32Array([1,288,1]));let Za=o.get("reg_3d.weight"),Va=o.get("reg_3d.bias");if(!Za||!Va)throw new Error("Missing reg_3d weights");let Ka=x(Za),ja=x(Va),Ne=l(Ka.byteLength,y),Ye=l(ja.byteLength,y),Ja=l(12,L);a.queue.writeBuffer(Ne,0,Ka),a.queue.writeBuffer(Ye,0,ja),a.queue.writeBuffer(Ja,0,new Uint32Array([1,288,63]));let ae=At.map(t=>{let{inCh:s,outCh:n,h:i,w:e,stride:c,prefix:d}=t,_=c===2?i/2:i,p=c===2?e/2:e,u=c===1?2:1,r=o.get(`${d}convs.0.weight`),m=o.get(`${d}convs.0.bias`),P=o.get(`${d}convs.1.weight`),C=o.get(`${d}convs.1.bias`);if(!r||!m||!P||!C)throw new Error(`Missing weights for ${d}`);let se=x(r),ue=x(m),oe=x(P),pe=x(C),de=l(se.byteLength,y),ce=l(ue.byteLength,y),J=l(oe.byteLength,y),Y=l(pe.byteLength,y),be=l(32,L),ge=l(36,L);return a.queue.writeBuffer(de,0,se),a.queue.writeBuffer(ce,0,ue),a.queue.writeBuffer(J,0,oe),a.queue.writeBuffer(Y,0,pe),a.queue.writeBuffer(be,0,new Uint32Array([1,s,i,e,_,p,c,u])),a.queue.writeBuffer(ge,0,new Uint32Array([1,s,n,_,p,Math.max(0,n-s),c,i,e])),{dwWeight:de,dwBias:ce,pwWeight:J,pwBias:Y,dwUniform:be,pwUniform:ge,spec:t,outH:_,outW:p}});function ne(t){let s=l(t.length*4,L);return a.queue.writeBuffer(s,0,new Uint32Array(t)),s}let ti=ne([1,96,8,8,16,16]),ii=ne([1,96,16,16,32,32]),ni=ne([1,48,32,32,64,64]);ne([1536*16]),ne([3072*32]),ne([3072*64]);let Qa=S(sa,[ha,he,ba]),et=S(ua,[he,Ga,Ca,G,Da]),I=[],z=[],q=[],$=[];for(let t of ae)I.push(S(Te,[G,t.dwWeight,t.dwBias,te,t.dwUniform])),z.push(S(Le,[te,G,t.pwWeight,t.pwBias,O,t.pwUniform])),q.push(S(Te,[O,t.dwWeight,t.dwBias,te,t.dwUniform])),$.push(S(Le,[te,O,t.pwWeight,t.pwBias,G,t.pwUniform]));let ri=S(ve,[G,xa,O,ti]),si=S(ve,[G,ya,O,ii]),ui=S(oa,[G,Ha,Ta,va,La]),oi=S(ve,[va,ga,O,ni]);S(fe,[G,Ie,ze,ie,za]),S(fe,[G,qe,$e,ie,Xa]),S(fe,[G,Ne,Ye,ie,Ja]);let K=S(pa,[F.createView(),he,Pa]),pi=S(da,[G,Ie,ze,qe,$e,Ne,Ye,ie]),Xe=24,at=[],tt=[];for(let t=Xe;t<ae.length;t++){let s=ae[t];at.push(S(Oe,[G,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,O,s.dwUniform])),tt.push(S(Oe,[O,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,G,s.dwUniform]))}let Ze=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Ze.globalCompositeOperation="copy";let it=new OffscreenCanvas(9,8),Ue=it.getContext("webgpu"),Se=null,Ve=null;if(Ue){Ue.configure({device:a,format:"rgba8unorm",alphaMode:"premultiplied"});let t=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),s=a.createShaderModule({code:Gt}),n=a.createShaderModule({code:Ct});Se=a.createRenderPipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),vertex:{module:s,entryPoint:"vs"},fragment:{module:n,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),Ve=a.createBindGroup({layout:t,entries:[{binding:0,resource:{buffer:ie}}]})}let Ge=new Float32Array(1),Ce=new Float32Array(1),De=new Float32Array(63);function N(t,s){let n=!0,i=0,e=t.beginComputePass();for(e.setPipeline(fa),e.setBindGroup(0,et),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=Pi;i++){let p=n?I[i]:q[i],u=n?z[i]:$[i],r=Z[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let c=n?G:O;for(t.copyBufferToBuffer(c,0,ga,0,3072*64*4),e=t.beginComputePass();i<=ki;i++){let p=n?I[i]:q[i],u=n?z[i]:$[i],r=Z[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let d=n?G:O;for(t.copyBufferToBuffer(d,0,ya,0,3072*32*4),e=t.beginComputePass();i<=Bi;i++){let p=n?I[i]:q[i],u=n?z[i]:$[i],r=Z[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let _=n?G:O;for(t.copyBufferToBuffer(_,0,xa,0,1536*16*4),e=t.beginComputePass();i<=Ui;i++){let p=n?I[i]:q[i],u=n?z[i]:$[i],r=Z[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.setPipeline(Re),e.setBindGroup(0,ri),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),n=!1,e=t.beginComputePass();{let p=n?I[i]:q[i],u=n?z[i]:$[i],r=Z[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,i++}e.setPipeline(Re),e.setBindGroup(0,si),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),n=!1,e=t.beginComputePass();{let p=n?I[i]:q[i],u=n?z[i]:$[i],r=Z[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,i++}for(e.setPipeline(Qt),e.setBindGroup(0,ui),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(Re),e.setBindGroup(0,oi),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),n=!1,e=t.beginComputePass();i<Xe;i++){let p=n?I[i]:q[i],u=n?z[i]:$[i],r=Z[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}for(;i<ae.length;i++){let p=i-Xe,u=n?at[p]:tt[p],r=ae[i];e.setPipeline(ai),e.setBindGroup(0,u),e.dispatchWorkgroups(r.outW,r.outH,1),n=!n}e.setPipeline(ei),e.setBindGroup(0,pi),e.dispatchWorkgroups(1),e.end(),s&&t.copyBufferToBuffer(ie,0,s,0,260)}async function Ae(t){a.queue.writeBuffer(ha,0,t);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(ma),e.setBindGroup(0,Qa),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}N(s,B),a.queue.submit([s.finish()]);let n=B.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(B.getMappedRange());return Ge[0]=i[0],Ce[0]=i[1],De.set(i.subarray(2,65)),B.unmap(),{handflag:new Float32Array(Ge),handedness:new Float32Array(Ce),landmarks:new Float32Array(De)}}async function Ke(t){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(V),e.setBindGroup(0,K),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}N(s,B),a.queue.submit([s.finish()]);let n=B.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(B.getMappedRange());return Ge[0]=i[0],Ce[0]=i[1],De.set(i.subarray(2,65)),B.unmap(),{handflag:new Float32Array(Ge),handedness:new Float32Array(Ce),landmarks:new Float32Array(De)}}async function nt(t){if(!Se||!Ve||!Ue)throw new Error("Render-based readback not available (no WebGPU canvas context)");a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let s=a.createCommandEncoder();{let p=s.beginComputePass();p.setPipeline(V),p.setBindGroup(0,K),p.dispatchWorkgroups(16,16,1),p.end()}N(s,null);let n=Ue.getCurrentTexture(),i=s.beginRenderPass({colorAttachments:[{view:n.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});i.setPipeline(Se),i.setBindGroup(0,Ve),i.draw(3),i.end(),a.queue.submit([s.finish()]),await a.queue.onSubmittedWorkDone(),Ze.drawImage(it,0,0);let c=Ze.getImageData(0,0,9,8).data,d=new Float32Array(65),_=new DataView(new ArrayBuffer(4));for(let p=0;p<65;p++){let u=p*4;_.setUint8(0,c[u]),_.setUint8(1,c[u+1]),_.setUint8(2,c[u+2]),_.setUint8(3,c[u+3]),d[p]=_.getFloat32(0)}return{handflag:new Float32Array([d[0]]),handedness:new Float32Array([d[1]]),landmarks:new Float32Array(d.subarray(2,65))}}let di=a.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),je=0,ci=[B,di],re=null,j=null;async function Je(t){let s=ci[je];je=1-je,a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let n=a.createCommandEncoder();{let e=n.beginComputePass();e.setPipeline(V),e.setBindGroup(0,K),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}N(n,s),a.queue.submit([n.finish()]);let i=null;if(re!==null&&j!==null){await re;let e=new Float32Array(j.getMappedRange());i={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},j.unmap()}return j=s,re=s.mapAsync(GPUMapMode.READ),i}async function rt(){if(!re||!j)return null;await re;let t=new Float32Array(j.getMappedRange()),s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))};return j.unmap(),re=null,j=null,s}async function _i(t=50){let s=new Float32Array(196608);for(let e=0;e<5;e++)await Ae(s);let n=[];for(let e=0;e<t;e++){let c=performance.now();await Ae(s),n.push(performance.now()-c)}let i=n.reduce((e,c)=>e+c,0)/n.length;return{avgMs:i,fps:1e3/i}}async function li(t=50){let s=new Float32Array(196608);for(let d=0;d<5;d++)await Ae(s);let n=[];for(let d=0;d<t;d++){let _=a.createCommandEncoder();{let u=_.beginComputePass();u.setPipeline(ma),u.setBindGroup(0,Qa),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),u.end()}N(_,B);let p=performance.now();a.queue.submit([_.finish()]),await a.queue.onSubmittedWorkDone(),n.push(performance.now()-p)}n.sort((d,_)=>d-_);let i=n.reduce((d,_)=>d+_,0)/n.length,e=n[Math.floor(n.length/2)],c=n[0];return{avgMs:i,fps:1e3/i,medianMs:e,minMs:c}}function wi(t){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let s=a.createCommandEncoder();{let n=s.beginComputePass();n.setPipeline(V),n.setBindGroup(0,K),n.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),n.end()}N(s,B),a.queue.submit([s.finish()])}async function mi(t,s=50){function n(u){let r=[...u].sort((m,P)=>m-P);return{median:r[Math.floor(r.length/2)],min:r[0]}}for(let u=0;u<10;u++)await Ke(t);let i=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let r=a.createCommandEncoder();{let P=r.beginComputePass();P.setPipeline(V),P.setBindGroup(0,K),P.dispatchWorkgroups(16,16,1),P.end()}N(r,B);let m=performance.now();a.queue.submit([r.finish()]),await a.queue.onSubmittedWorkDone(),i.push(performance.now()-m)}let e=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let r=a.createCommandEncoder();{let C=r.beginComputePass();C.setPipeline(V),C.setBindGroup(0,K),C.dispatchWorkgroups(16,16,1),C.end()}N(r,B),a.queue.submit([r.finish()]);let m=B.mapAsync(GPUMapMode.READ),P=performance.now();await a.queue.onSubmittedWorkDone(),await m,B.getMappedRange(),B.unmap(),e.push(performance.now()-P)}let c=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let r=a.createCommandEncoder();{let P=r.beginComputePass();P.setPipeline(V),P.setBindGroup(0,K),P.dispatchWorkgroups(16,16,1),P.end()}N(r,B),a.queue.submit([r.finish()]);let m=performance.now();await B.mapAsync(GPUMapMode.READ),B.getMappedRange(),B.unmap(),c.push(performance.now()-m)}let d=[];for(let u=0;u<s;u++){let r=performance.now();await Ke(t),d.push(performance.now()-r)}await Je(t);let _=[];for(let u=0;u<s;u++){let r=performance.now();await Je(t),_.push(performance.now()-r)}await rt();let p=null;if(Se){let u=[];for(let r=0;r<s;r++){let m=performance.now();await nt(t),u.push(performance.now()-m)}p=n(u)}return{gpuOnly:n(i),mapAsyncOnly:n(e),mapAsyncNoWait:n(c),total:n(d),pipelined:n(_),renderReadback:p}}async function fi(t){let s=[];async function n(e,c,d){let _=a.createBuffer({size:c,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),p=a.createCommandEncoder();p.copyBufferToBuffer(e,0,_,0,c),a.queue.submit([p.finish()]),await a.queue.onSubmittedWorkDone(),await _.mapAsync(GPUMapMode.READ);let u=new Float32Array(_.getMappedRange()),r=1/0,m=-1/0,P=0;for(let C=0;C<u.length;C++)u[C]<r&&(r=u[C]),u[C]>m&&(m=u[C]),u[C]!==0&&P++;_.unmap(),_.destroy(),s.push({layer:d,stats:{min:r,max:m,nonZero:P,total:u.length}})}a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);{let e=a.createCommandEncoder(),c=e.beginComputePass();c.setPipeline(V),c.setBindGroup(0,K),c.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),c.end(),a.queue.submit([e.finish()])}await n(he,Math.min(he.size,3*257*257*4),"canvas\u2192bufInput");{let e=a.createCommandEncoder(),c=e.beginComputePass();c.setPipeline(fa),c.setBindGroup(0,et),c.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),c.end(),a.queue.submit([e.finish()])}await n(G,Math.min(G.size,3072*128*4),"inputConv\u2192bufA");let i=!0;for(let e=0;e<Math.min(ae.length,6);e++){let c=i?I[e]:q[e],d=i?z[e]:$[e],_=Z[e],p=ae[e];{let r=a.createCommandEncoder(),m=r.beginComputePass();m.setPipeline(_.dwPipeline),m.setBindGroup(0,c),m.dispatchWorkgroups(_.dwDispatchX,_.dwDispatchY,_.dwDispatchZ),m.end(),a.queue.submit([r.finish()])}await n(te,Math.min(te.size,p.spec.inCh*p.outH*p.outW*4),`layer${e}.DW\u2192bufDW (${p.spec.prefix})`);{let r=a.createCommandEncoder(),m=r.beginComputePass();m.setPipeline(_.pwPipeline),m.setBindGroup(0,d),m.dispatchWorkgroups(_.pwDispatchX,_.pwDispatchY,_.pwDispatchZ),m.end(),a.queue.submit([r.finish()])}let u=i?O:G;await n(u,Math.min(u.size,p.spec.outCh*p.outH*p.outW*4),`layer${e}.PW\u2192buf${i?"B":"A"} (${p.spec.prefix})`),i=!i}return s}return{device:a,run:Ae,runFromCanvas:Ke,runFromCanvasViaRender:nt,runFromCanvasPipelined:Je,flushPipelined:rt,benchmark:_i,benchmarkGPU:li,benchmarkDiagnostic:mi,debugLayerOutputs:fi,_device:a,_benchmarkSubmitOnly:wi}}var Si="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Gi(o={}){let{weightsUrl:w,scoreThreshold:h=.5,forceF32:b=!1}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let U=w??Si,a=U.endsWith("/")?U:`${U}/`,D=`${a}weights_f16.json`,R=`${a}weights_f16.bin`,[E,M]=await Promise.all([fetch(D),fetch(R)]);if(!E.ok)throw new Error(`Failed to fetch weights metadata: ${E.status}`);if(!M.ok)throw new Error(`Failed to fetch weights binary: ${M.status}`);let x=await E.json(),X=await M.arrayBuffer(),g=Wt(x,X),H=await na(g,{forceF32:b});if(!b){let f=new OffscreenCanvas(256,256),v=f.getContext("2d");v.fillStyle="#886644",v.fillRect(0,0,256,256),v.fillStyle="#cc9966",v.fillRect(50,50,156,156);let k=await H.runFromCanvas(f);k.landmarks.every(Q=>Q===0)&&k.handflag.every(Q=>Q===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),H.device.destroy(),H=await na(g,{forceF32:!0}))}let A=null;function He(){return A||(A=new OffscreenCanvas(256,256)),A}async function y(f){if(f instanceof HTMLCanvasElement||f instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&f instanceof ImageBitmap)return f;let v=He();v.width=256,v.height=256;let k=v.getContext("2d");return f instanceof ImageData?k.putImageData(f,0,0):k.drawImage(f,0,0,256,256),v}function le(f,v,k){let me=f[0];if(me<h)return null;let Q=v[0]>.5,xe=[];for(let ee=0;ee<21;ee++)xe.push({x:k[ee*3],y:k[ee*3+1],z:k[ee*3+2]});return{score:me,handedness:Q?"right":"left",landmarks:xe}}async function we(f){let v=await y(f),k=await H.runFromCanvas(v);return le(k.handflag,k.handedness,k.landmarks)}async function ye(f){let v=await y(f),k=await H.runFromCanvasPipelined(v);return k?le(k.handflag,k.handedness,k.landmarks):null}async function L(){let f=await H.flushPipelined();return f?le(f.handflag,f.handedness,f.landmarks):null}function l(){H.device.destroy(),A=null}async function S(f){let v=await y(f);return H.benchmarkDiagnostic(v)}async function T(f){let v=await y(f);return H.debugLayerOutputs(v)}return{detect:we,detectPipelined:ye,flushPipelined:L,dispose:l,benchmarkDiagnostic:S,debugLayerOutputs:T}}export{Gi as createHandpose};
