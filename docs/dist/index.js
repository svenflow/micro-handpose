function _e(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ta(n){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],i="enable f16;"+n;for(let y of _)for(;i.includes(`${y}:array<f32>`);)i=i.replace(`${y}:array<f32>`,`${y}:array<f16>`);return i}var _a=_e(`
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
`),la=_e(`
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
`),ma=_e(`
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
`),fa=_e(`
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
`);function La(n,_){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Ra(n,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Oa(n,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Ia(n,_){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Fa(n,_){return[8,8]}var za=_e(`
struct PadParams { channels:u32, in_size:u32, out_size:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:PadParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y; let c=gid.z;
  if(x>=params.in_size||y>=params.in_size||c>=params.channels){return;}
  let in_idx=c*params.in_size*params.in_size+y*params.in_size+x;
  // PyTorch uses ConstantPad2d((0,1,0,1)): pad right+bottom, image stays at (0,0)
  let out_idx=c*params.out_size*params.out_size+y*params.out_size+x;
  output[out_idx]=input[in_idx];
}
`),Na=_e(`
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
`);function Ka(n){return _e(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${n?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${n?"+skip[out_idx]":""};
}
`)}var qa=Ka(!1),$a=Ka(!0),Ya=_e(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Xa=_e(`
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
`);function Va(n){return _e(`
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
  ${n==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var ja=Va("sigmoid"),Za=Va("div256"),Ja=_e(`
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
`),Qa=_e(`
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
`);function en(n,_){let y=Math.min(_,256),w=_>y,g=n%4===0?`var ic:u32=0u;
    while(ic<${n}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${n}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,S=`var skip_val:f32=0.0;
    if(c<${n}u){
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
    }`,D=n===_?"":`if(c<${n}u){`,d=n===_?"":"}",h=w?`for(var c:u32=lid.x;c<${n}u;c+=${y}u){`:`let c=lid.x;
  ${D}`,v=w?"}":d,U=w?`for(var c:u32=lid.x;c<${_}u;c+=${y}u){`:"{let c=lid.x;";return _e(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${n}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${y},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${h}
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
  ${v}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${U}
    let pw_base=c*${n}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${g}
    // Skip connection (only for c < inCh)
    ${S}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var tn=_e(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),an=_e(`
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
`),nn=_e(`
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
  // PyTorch uses ConstantPad2d((0,1,0,1)): pad right+bottom, image stays at (0,0)
  output[0u*out_stride+y*params.out_size+x]=pixel.r;
  output[1u*out_stride+y*params.out_size+x]=pixel.g;
  output[2u*out_stride+y*params.out_size+x]=pixel.b;
}
`);function Dt(n,_){let i=new Map,y=n.dtype??"float32";for(let w=0;w<n.keys.length;w++){let e=n.keys[w],g=n.shapes[w],S=n.offsets[w],D=g.reduce((v,U)=>v*U,1),d,h;if(y==="float32")d=new Float32Array(_,S,D);else{let v=new DataView(_);d=new Float32Array(D);for(let U=0;U<D;U++)d[U]=Rn(v.getUint16(S+U*2,!0));h=_.slice(S,S+D*2)}i.set(e,{data:d,shape:g,rawF16:h})}return i}function Rn(n){let _=n>>15&1,i=n>>10&31,y=n&1023;if(i===0){if(y===0)return _?-0:0;let g=-14,S=y/1024;return(_?-1:1)*Math.pow(2,g)*S}if(i===31)return y===0?_?-1/0:1/0:NaN;let w=i-15,e=1+y/1024;return(_?-1:1)*Math.pow(2,w)*e}var On=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],rn=On.map(([n,_,i,y,w])=>({type:"resmodule",inCh:n,outCh:_,h:i,w:i,stride:y,prefix:w})),In=2,Fn=5,zn=8,Nn=11;async function Zt(n,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let i=await navigator.gpu.requestAdapter();if(!i)throw new Error("No GPU adapter found");let y=i.features.has("shader-f16"),w=y?["shader-f16"]:[],e=await i.requestDevice({requiredFeatures:w,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(i.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(i.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(i.limits.maxComputeInvocationsPerWorkgroup,288)}}),g=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(y)try{let a=`enable f16;
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
}`,p=`enable f16;
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
}`,o=e.createShaderModule({code:a}),s=e.createShaderModule({code:p}),t=await o.getCompilationInfo(),E=await s.getCompilationInfo();if(t.messages.some(A=>A.type==="error")||E.messages.some(A=>A.type==="error"))g=!1;else{let A=new Float32Array(2400);A.fill(1);let W=new Uint16Array(2400);W.fill(10516);let x=new Uint16Array(96);x.fill(14336);let f=new Uint16Array(9216);f.fill(8478);let u=new Uint16Array(96);u.fill(12288);let I=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,te=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ie=e.createBuffer({size:A.byteLength,usage:I}),wt=e.createBuffer({size:W.byteLength,usage:I}),gt=e.createBuffer({size:x.byteLength,usage:I}),bt=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),yt=e.createBuffer({size:f.byteLength,usage:I}),xt=e.createBuffer({size:u.byteLength,usage:I}),vt=e.createBuffer({size:384,usage:te}),je=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ie,0,A),e.queue.writeBuffer(wt,0,W),e.queue.writeBuffer(gt,0,x),e.queue.writeBuffer(yt,0,f),e.queue.writeBuffer(xt,0,u);let qe="read-only-storage",Gt="storage",At=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:qe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:qe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:qe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Gt}}]}),Wa=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:qe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:qe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:qe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Gt}}]}),Mn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[At]}),compute:{module:o,entryPoint:"main"}}),En=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Wa]}),compute:{module:s,entryPoint:"main"}}),Wn=e.createBindGroup({layout:At,entries:[{binding:0,resource:{buffer:ie}},{binding:1,resource:{buffer:wt}},{binding:2,resource:{buffer:gt}},{binding:3,resource:{buffer:bt}}]}),Hn=e.createBindGroup({layout:Wa,entries:[{binding:0,resource:{buffer:bt}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:xt}},{binding:3,resource:{buffer:vt}}]}),Xt=e.createCommandEncoder(),Vt=Xt.beginComputePass();Vt.setPipeline(Mn),Vt.setBindGroup(0,Wn),Vt.dispatchWorkgroups(2),Vt.end();let jt=Xt.beginComputePass();jt.setPipeline(En),jt.setBindGroup(0,Hn),jt.dispatchWorkgroups(2),jt.end(),Xt.copyBufferToBuffer(vt,0,je,0,384),e.queue.submit([Xt.finish()]),await e.queue.onSubmittedWorkDone(),await je.mapAsync(GPUMapMode.READ);let St=new Float32Array(je.getMappedRange()),Ha=1.5*.0104*96+.25,Tn=St[0]!==0&&St[47]!==0&&St[95]!==0,Ln=Math.abs(St[0]-Ha)<1;g=Tn&&Ln,je.unmap(),ie.destroy(),wt.destroy(),gt.destroy(),bt.destroy(),yt.destroy(),xt.destroy(),vt.destroy(),je.destroy(),g||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${St[0]}, expected ~${Ha.toFixed(2)}) \u2014 falling back to f32`)}}catch{g=!1}let D=n.values().next().value,d=g&&!!D?.rawF16&&!_?.forceF32;console.log(d?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${y}, f16 validated: ${g}, f16 data: ${!!D?.rawF16})`);function h(a){if(d&&a.rawF16){let p=new Uint16Array(a.rawF16);if(p.length%2!==0){let o=new Uint16Array(p.length+1);return o.set(p),o}return p}return a.data}function v(a){if(d&&a.rawF16){let p=a.rawF16.byteLength;return Math.ceil(p/4)*4}return a.data.byteLength}function U(a){return d?Ta(a):a}let T={r:"read-only-storage",s:"storage",u:"uniform"};function m(a){return e.createBindGroupLayout({entries:a.map((p,o)=>({binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[p]}}))})}function z(a){return e.createBindGroupLayout({entries:a.map((p,o)=>p==="t"?{binding:o,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[p]}})})}let C=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,pe=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,re=GPUBufferUsage.STORAGE,ve=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ce=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function M(a,p){return e.createBuffer({size:a,usage:p})}function Y(a,p){return e.createBindGroup({layout:a,entries:p.map((o,s)=>({binding:s,resource:"size"in o?{buffer:o}:o}))})}function ae(a,p){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:p,entryPoint:"main"}})}let H=e.createShaderModule({code:za}),F=e.createShaderModule({code:nn}),N=e.createShaderModule({code:U(Ja)}),Ae=e.createShaderModule({code:U(la)}),ge=e.createShaderModule({code:U(_a)}),He=e.createShaderModule({code:U(ma)}),fe=e.createShaderModule({code:U(fa)}),Pe=e.createShaderModule({code:U(Na)}),P=e.createShaderModule({code:qa}),V=e.createShaderModule({code:Ya}),he=e.createShaderModule({code:$a}),j=e.createShaderModule({code:U(Xa)}),we=e.createShaderModule({code:U(ja)}),se=e.createShaderModule({code:U(Za)}),$e=e.createShaderModule({code:U(Qa)}),Ze=new Map;function Be(a,p){let o=`${a}_${p}`,s=Ze.get(o);return s||(s=e.createShaderModule({code:U(en(a,p))}),Ze.set(o,s)),s}let Ce=m(["r","r","r","s","u"]),be=m(["r","r","r","r","s","u"]),ye=m(["r","s","u"]),Se=m(["r","r","r","s","u"]),ke=m(["r","s","u"]),Te=m(["r","r","s","u"]),De=m(["r","r","s","u"]),Ye=m(["r","r","r","s","u"]),Me=m(["r","r","r","s","u"]),ot=z(["t","s","u"]),Je=m(["r","r","r","r","r","r","r","s"]),Le=m(["r","r","r","r","r","s","u"]),Pt=e.createPipelineLayout({bindGroupLayouts:[Ce]}),Mt=e.createPipelineLayout({bindGroupLayouts:[be]}),ut=a=>e.createComputePipeline({layout:Pt,compute:{module:a,entryPoint:"main"}}),pt=a=>e.createComputePipeline({layout:Mt,compute:{module:a,entryPoint:"main"}}),Jt=ut(Ae),Qt=ut(ge),Et=pt(He),ct=pt(fe),dt=new Map,Bt=new Map,_t=new Map,Ct=new Map;dt.set("8,8",Jt),Bt.set("8,8",Qt),_t.set("8,8",Et),Ct.set("8,8",ct);function Qe(a,p,o,s,t){let E=`${p},${o}`,A=a.get(E);return A||(A=t(e.createShaderModule({code:U(s(p,o))})),a.set(E,A)),A}let Wt=(a,p)=>Qe(dt,a,p,La,ut),ea=(a,p)=>Qe(Bt,a,p,Ra,ut),Ht=(a,p)=>Qe(_t,a,p,Oa,pt),Tt=(a,p)=>Qe(Ct,a,p,Ia,pt),Ee=rn.map(a=>{let p=a.stride===2?a.h/2:a.h,o=a.stride===2?a.w/2:a.w,[s,t]=Fa(a.inCh,p),E=a.h>=64,A=p>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:E?ea(s,t):Wt(s,t),pwPipeline:A?Tt(s,t):Ht(s,t),dwDispatchX:Math.ceil(o/s),dwDispatchY:Math.ceil(p/t),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(o/s),pwDispatchY:Math.ceil(p/t),pwDispatchZ:A?a.outCh/2:a.outCh}}),ta=ae(ye,H),lt=ae(Se,Pe);ae(ke,P),ae(Te,V);let Re=ae(De,he),et=ae(Ye,j);ae(Me,we),ae(Me,se);let Ue=ae(ot,F),Lt=ae(Je,N),kt=ae(Le,$e),Rt=1*288*128*128*4,tt=M(3*256*256*4,C),We=M(3*257*257*4,re),Ut=M(12,ce);e.queue.writeBuffer(Ut,0,new Uint32Array([3,256,257]));let Z=M(Rt,pe),de=M(Rt,ve),Ge=M(Rt,re),at=M(3072*64*4,C),nt=M(3072*32*4,C),it=M(1536*16*4,C),ee=M(6144*64*4,re),le=M(260,ve),X=M(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);M(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let oe=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Oe=M(8,ce);e.queue.writeBuffer(Oe,0,new Uint32Array([256,257]));let Ot=n.get("backbone1.1.weight"),It=n.get("backbone1.1.bias");if(!Ot||!It)throw new Error("Missing input conv weights");let me=h(Ot),Ft=h(It),c=M(me.byteLength,C),r=M(Ft.byteLength,C),k=M(28,ce);e.queue.writeBuffer(c,0,me),e.queue.writeBuffer(r,0,Ft),e.queue.writeBuffer(k,0,new Uint32Array([1,3,24,257,257,128,128]));let G=n.get("backbone6.1.weight"),b=n.get("backbone6.1.bias");if(!G||!b)throw new Error("Missing backbone6.1 conv1x1 weights");let l=h(G),O=h(b),J=M(l.byteLength,C),L=M(O.byteLength,C),q=M(20,ce);e.queue.writeBuffer(J,0,l),e.queue.writeBuffer(L,0,O),e.queue.writeBuffer(q,0,new Uint32Array([1,96,48,32,32]));let R=n.get("handflag.weight"),K=n.get("handflag.bias");if(!R||!K)throw new Error("Missing handflag weights");let $=h(R),ue=h(K),B=M($.byteLength,C),Q=M(ue.byteLength,C),ne=M(12,ce);e.queue.writeBuffer(B,0,$),e.queue.writeBuffer(Q,0,ue),e.queue.writeBuffer(ne,0,new Uint32Array([1,288,1]));let xe=n.get("handedness.weight"),mt=n.get("handedness.bias");if(!xe||!mt)throw new Error("Missing handedness weights");let ba=h(xe),ya=h(mt),aa=M(ba.byteLength,C),na=M(ya.byteLength,C),xa=M(12,ce);e.queue.writeBuffer(aa,0,ba),e.queue.writeBuffer(na,0,ya),e.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let va=n.get("reg_3d.weight"),Pa=n.get("reg_3d.bias");if(!va||!Pa)throw new Error("Missing reg_3d weights");let Ba=h(va),Ca=h(Pa),ia=M(Ba.byteLength,C),ra=M(Ca.byteLength,C),ka=M(12,ce);e.queue.writeBuffer(ia,0,Ba),e.queue.writeBuffer(ra,0,Ca),e.queue.writeBuffer(ka,0,new Uint32Array([1,288,63]));let rt=rn.map(a=>{let{inCh:p,outCh:o,h:s,w:t,stride:E,prefix:A}=a,W=E===2?s/2:s,x=E===2?t/2:t,f=E===2?1:2,u=n.get(`${A}convs.0.weight`),I=n.get(`${A}convs.0.bias`),te=n.get(`${A}convs.1.weight`),ie=n.get(`${A}convs.1.bias`);if(!u||!I||!te||!ie)throw new Error(`Missing weights for ${A}`);let wt=h(u),gt=h(I),bt=h(te),yt=h(ie),xt=M(wt.byteLength,C),vt=M(gt.byteLength,C),je=M(bt.byteLength,C),qe=M(yt.byteLength,C),Gt=M(32,ce),At=M(36,ce);return e.queue.writeBuffer(xt,0,wt),e.queue.writeBuffer(vt,0,gt),e.queue.writeBuffer(je,0,bt),e.queue.writeBuffer(qe,0,yt),e.queue.writeBuffer(Gt,0,new Uint32Array([1,p,s,t,W,x,E,f])),e.queue.writeBuffer(At,0,new Uint32Array([1,p,o,W,x,Math.max(0,o-p),E,s,t])),{dwWeight:xt,dwBias:vt,pwWeight:je,pwBias:qe,dwUniform:Gt,pwUniform:At,spec:a,outH:W,outW:x}});function ft(a){let p=M(a.length*4,ce);return e.queue.writeBuffer(p,0,new Uint32Array(a)),p}let gn=ft([1,96,8,8,16,16]),bn=ft([1,96,16,16,32,32]),yn=ft([1,48,32,32,64,64]);ft([1536*16]),ft([3072*32]),ft([3072*64]);let Ua=Y(ye,[tt,We,Ut]),Ga=Y(Se,[We,c,r,Z,k]),Ie=[],Fe=[],ze=[],Ne=[];for(let a of rt)Ie.push(Y(Ce,[Z,a.dwWeight,a.dwBias,Ge,a.dwUniform])),Fe.push(Y(be,[Ge,Z,a.pwWeight,a.pwBias,de,a.pwUniform])),ze.push(Y(Ce,[de,a.dwWeight,a.dwBias,Ge,a.dwUniform])),Ne.push(Y(be,[Ge,de,a.pwWeight,a.pwBias,Z,a.pwUniform]));let xn=Y(De,[Z,it,de,gn]),vn=Y(De,[Z,nt,de,bn]),Pn=Y(Ye,[Z,J,L,ee,q]),Bn=Y(De,[ee,at,de,yn]);Y(Me,[Z,B,Q,le,ne]),Y(Me,[Z,aa,na,le,xa]),Y(Me,[Z,ia,ra,le,ka]);let Xe=Y(ot,[oe.createView(),We,Oe]),Cn=Y(Je,[Z,B,Q,aa,na,ia,ra,le]),sa=24,Aa=[],Sa=[];for(let a=sa;a<rt.length;a++){let p=rt[a];Aa.push(Y(Le,[Z,p.dwWeight,p.dwBias,p.pwWeight,p.pwBias,de,p.dwUniform])),Sa.push(Y(Le,[de,p.dwWeight,p.dwBias,p.pwWeight,p.pwBias,Z,p.dwUniform]))}let oa=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});oa.globalCompositeOperation="copy";let Da=new OffscreenCanvas(9,8),zt=Da.getContext("webgpu"),Nt=null,ua=null;if(zt){zt.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let a=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),p=e.createShaderModule({code:tn}),o=e.createShaderModule({code:an});Nt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:p,entryPoint:"vs"},fragment:{module:o,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),ua=e.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:le}}]})}let Kt=new Float32Array(1),qt=new Float32Array(1),$t=new Float32Array(63);function Ke(a,p){let o=!0,s=0,t=a.beginComputePass();for(t.setPipeline(lt),t.setBindGroup(0,Ga),t.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);s<=In;s++){let x=o?Ie[s]:ze[s],f=o?Fe[s]:Ne[s],u=Ee[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let E=o?Z:de;for(a.copyBufferToBuffer(E,0,at,0,3072*64*4),t=a.beginComputePass();s<=Fn;s++){let x=o?Ie[s]:ze[s],f=o?Fe[s]:Ne[s],u=Ee[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let A=o?Z:de;for(a.copyBufferToBuffer(A,0,nt,0,3072*32*4),t=a.beginComputePass();s<=zn;s++){let x=o?Ie[s]:ze[s],f=o?Fe[s]:Ne[s],u=Ee[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let W=o?Z:de;for(a.copyBufferToBuffer(W,0,it,0,1536*16*4),t=a.beginComputePass();s<=Nn;s++){let x=o?Ie[s]:ze[s],f=o?Fe[s]:Ne[s],u=Ee[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.setPipeline(Re),t.setBindGroup(0,xn),t.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),t.end(),o=!1,t=a.beginComputePass();{let x=o?Ie[s]:ze[s],f=o?Fe[s]:Ne[s],u=Ee[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}t.setPipeline(Re),t.setBindGroup(0,vn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),t.end(),o=!1,t=a.beginComputePass();{let x=o?Ie[s]:ze[s],f=o?Fe[s]:Ne[s],u=Ee[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}for(t.setPipeline(et),t.setBindGroup(0,Pn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),t.setPipeline(Re),t.setBindGroup(0,Bn),t.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),t.end(),o=!1,t=a.beginComputePass();s<sa;s++){let x=o?Ie[s]:ze[s],f=o?Fe[s]:Ne[s],u=Ee[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}for(;s<rt.length;s++){let x=s-sa,f=o?Aa[x]:Sa[x],u=rt[s];t.setPipeline(kt),t.setBindGroup(0,f),t.dispatchWorkgroups(u.outW,u.outH,1),o=!o}t.setPipeline(Lt),t.setBindGroup(0,Cn),t.dispatchWorkgroups(1),t.end(),p&&a.copyBufferToBuffer(le,0,p,0,260)}async function Yt(a){e.queue.writeBuffer(tt,0,a);let p=e.createCommandEncoder();{let t=p.beginComputePass();t.setPipeline(ta),t.setBindGroup(0,Ua),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}Ke(p,X),e.queue.submit([p.finish()]);let o=X.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(X.getMappedRange());return Kt[0]=s[0],qt[0]=s[1],$t.set(s.subarray(2,65)),X.unmap(),{handflag:new Float32Array(Kt),handedness:new Float32Array(qt),landmarks:new Float32Array($t)}}async function pa(a){e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);let p=e.createCommandEncoder();{let t=p.beginComputePass();t.setPipeline(Ue),t.setBindGroup(0,Xe),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Ke(p,X),e.queue.submit([p.finish()]);let o=X.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(X.getMappedRange());return Kt[0]=s[0],qt[0]=s[1],$t.set(s.subarray(2,65)),X.unmap(),{handflag:new Float32Array(Kt),handedness:new Float32Array(qt),landmarks:new Float32Array($t)}}async function Ma(a){if(!Nt||!ua||!zt)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);let p=e.createCommandEncoder();{let x=p.beginComputePass();x.setPipeline(Ue),x.setBindGroup(0,Xe),x.dispatchWorkgroups(16,16,1),x.end()}Ke(p,null);let o=zt.getCurrentTexture(),s=p.beginRenderPass({colorAttachments:[{view:o.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});s.setPipeline(Nt),s.setBindGroup(0,ua),s.draw(3),s.end(),e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),oa.drawImage(Da,0,0);let E=oa.getImageData(0,0,9,8).data,A=new Float32Array(65),W=new DataView(new ArrayBuffer(4));for(let x=0;x<65;x++){let f=x*4;W.setUint8(0,E[f]),W.setUint8(1,E[f+1]),W.setUint8(2,E[f+2]),W.setUint8(3,E[f+3]),A[x]=W.getFloat32(0)}return{handflag:new Float32Array([A[0]]),handedness:new Float32Array([A[1]]),landmarks:new Float32Array(A.subarray(2,65))}}let kn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),ca=0,Un=[X,kn],ht=null,Ve=null;async function da(a){let p=Un[ca];ca=1-ca,e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);let o=e.createCommandEncoder();{let t=o.beginComputePass();t.setPipeline(Ue),t.setBindGroup(0,Xe),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Ke(o,p),e.queue.submit([o.finish()]);let s=null;if(ht!==null&&Ve!==null){await ht;let t=new Float32Array(Ve.getMappedRange());s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))},Ve.unmap()}return Ve=p,ht=p.mapAsync(GPUMapMode.READ),s}async function Ea(){if(!ht||!Ve)return null;await ht;let a=new Float32Array(Ve.getMappedRange()),p={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return Ve.unmap(),ht=null,Ve=null,p}async function Gn(a=50){let p=new Float32Array(196608);for(let t=0;t<5;t++)await Yt(p);let o=[];for(let t=0;t<a;t++){let E=performance.now();await Yt(p),o.push(performance.now()-E)}let s=o.reduce((t,E)=>t+E,0)/o.length;return{avgMs:s,fps:1e3/s}}async function An(a=50){let p=new Float32Array(196608);for(let A=0;A<5;A++)await Yt(p);let o=[];for(let A=0;A<a;A++){let W=e.createCommandEncoder();{let f=W.beginComputePass();f.setPipeline(ta),f.setBindGroup(0,Ua),f.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),f.end()}Ke(W,X);let x=performance.now();e.queue.submit([W.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-x)}o.sort((A,W)=>A-W);let s=o.reduce((A,W)=>A+W,0)/o.length,t=o[Math.floor(o.length/2)],E=o[0];return{avgMs:s,fps:1e3/s,medianMs:t,minMs:E}}function ei(a){e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);let p=e.createCommandEncoder();{let o=p.beginComputePass();o.setPipeline(Ue),o.setBindGroup(0,Xe),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),o.end()}Ke(p,X),e.queue.submit([p.finish()])}async function Sn(a,p=50){function o(f){let u=[...f].sort((I,te)=>I-te);return{median:u[Math.floor(u.length/2)],min:u[0]}}for(let f=0;f<10;f++)await pa(a);let s=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);let u=e.createCommandEncoder();{let te=u.beginComputePass();te.setPipeline(Ue),te.setBindGroup(0,Xe),te.dispatchWorkgroups(16,16,1),te.end()}Ke(u,X);let I=performance.now();e.queue.submit([u.finish()]),await e.queue.onSubmittedWorkDone(),s.push(performance.now()-I)}let t=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);let u=e.createCommandEncoder();{let ie=u.beginComputePass();ie.setPipeline(Ue),ie.setBindGroup(0,Xe),ie.dispatchWorkgroups(16,16,1),ie.end()}Ke(u,X),e.queue.submit([u.finish()]);let I=X.mapAsync(GPUMapMode.READ),te=performance.now();await e.queue.onSubmittedWorkDone(),await I,X.getMappedRange(),X.unmap(),t.push(performance.now()-te)}let E=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);let u=e.createCommandEncoder();{let te=u.beginComputePass();te.setPipeline(Ue),te.setBindGroup(0,Xe),te.dispatchWorkgroups(16,16,1),te.end()}Ke(u,X),e.queue.submit([u.finish()]);let I=performance.now();await X.mapAsync(GPUMapMode.READ),X.getMappedRange(),X.unmap(),E.push(performance.now()-I)}let A=[];for(let f=0;f<p;f++){let u=performance.now();await pa(a),A.push(performance.now()-u)}await da(a);let W=[];for(let f=0;f<p;f++){let u=performance.now();await da(a),W.push(performance.now()-u)}await Ea();let x=null;if(Nt){let f=[];for(let u=0;u<p;u++){let I=performance.now();await Ma(a),f.push(performance.now()-I)}x=o(f)}return{gpuOnly:o(s),mapAsyncOnly:o(t),mapAsyncNoWait:o(E),total:o(A),pipelined:o(W),renderReadback:x}}async function Dn(a){let p=[];async function o(t,E,A){let W=e.createBuffer({size:E,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),x=e.createCommandEncoder();x.copyBufferToBuffer(t,0,W,0,E),e.queue.submit([x.finish()]),await e.queue.onSubmittedWorkDone(),await W.mapAsync(GPUMapMode.READ);let f=new Float32Array(W.getMappedRange()),u=1/0,I=-1/0,te=0;for(let ie=0;ie<f.length;ie++)f[ie]<u&&(u=f[ie]),f[ie]>I&&(I=f[ie]),f[ie]!==0&&te++;W.unmap(),W.destroy(),p.push({layer:A,stats:{min:u,max:I,nonZero:te,total:f.length}})}e.queue.copyExternalImageToTexture({source:a},{texture:oe},[256,256]);{let t=e.createCommandEncoder(),E=t.beginComputePass();E.setPipeline(Ue),E.setBindGroup(0,Xe),E.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),E.end(),e.queue.submit([t.finish()])}await o(We,Math.min(We.size,3*257*257*4),"canvas\u2192bufInput");{let t=e.createCommandEncoder(),E=t.beginComputePass();E.setPipeline(lt),E.setBindGroup(0,Ga),E.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),E.end(),e.queue.submit([t.finish()])}await o(Z,Math.min(Z.size,3072*128*4),"inputConv\u2192bufA");let s=!0;for(let t=0;t<Math.min(rt.length,6);t++){let E=s?Ie[t]:ze[t],A=s?Fe[t]:Ne[t],W=Ee[t],x=rt[t];{let u=e.createCommandEncoder(),I=u.beginComputePass();I.setPipeline(W.dwPipeline),I.setBindGroup(0,E),I.dispatchWorkgroups(W.dwDispatchX,W.dwDispatchY,W.dwDispatchZ),I.end(),e.queue.submit([u.finish()])}await o(Ge,Math.min(Ge.size,x.spec.inCh*x.outH*x.outW*4),`layer${t}.DW\u2192bufDW (${x.spec.prefix})`);{let u=e.createCommandEncoder(),I=u.beginComputePass();I.setPipeline(W.pwPipeline),I.setBindGroup(0,A),I.dispatchWorkgroups(W.pwDispatchX,W.pwDispatchY,W.pwDispatchZ),I.end(),e.queue.submit([u.finish()])}let f=s?de:Z;await o(f,Math.min(f.size,x.spec.outCh*x.outH*x.outW*4),`layer${t}.PW\u2192buf${s?"B":"A"} (${x.spec.prefix})`),s=!s}return p}return{device:e,run:Yt,runFromCanvas:pa,runFromCanvasViaRender:Ma,runFromCanvasPipelined:da,flushPipelined:Ea,benchmark:Gn,benchmarkGPU:An,benchmarkDiagnostic:Sn,debugLayerOutputs:Dn}}function st(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var sn=st(`
struct ConvParams { batch:u32, in_channels:u32, out_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read> alpha:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:ConvParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){
    for(var ky:u32=0u;ky<5u;ky=ky+1u){
      for(var kx:u32=0u;kx<5u;kx=kx+1u){
        let in_y=out_y*2u+ky;
        let in_x=out_x*2u+kx;
        if(in_y<params.in_height && in_x<params.in_width){
          let in_idx=batch*params.in_channels*params.in_height*params.in_width+ic*params.in_height*params.in_width+in_y*params.in_width+in_x;
          let w_idx=oc*5u*5u*params.in_channels+ky*5u*params.in_channels+kx*params.in_channels+ic;
          sum=sum+input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum=sum+bias[oc];
  // PReLU
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.out_height*params.out_width+oc*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=result;
}
`),on=st(`
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
  var sum:f32=0.0;
  for(var ky:u32=0u;ky<5u;ky=ky+1u){
    let in_y=base_in_y+i32(ky);
    if(in_y>=0 && in_y<in_h){
      let row_base=in_base+u32(in_y)*params.in_width;
      for(var kx:u32=0u;kx<5u;kx=kx+1u){
        let in_x=base_in_x+i32(kx);
        if(in_x>=0 && in_x<in_w){
          sum+=input[row_base+u32(in_x)]*weight[w_base+ky*5u+kx];
        }
      }
    }
  }
  sum+=bias[c];
  let out_idx=batch*params.channels*params.out_height*params.out_width+c*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=sum;
}
`),un=st(`
struct PointwiseParams { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, channel_pad:u32, stride:u32, in_height:u32, in_width:u32, }
@group(0)@binding(0) var<storage,read> dw_output:array<f32>;
@group(0)@binding(1) var<storage,read> skip_input:array<f32>;
@group(0)@binding(2) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(3) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(4) var<storage,read> alpha:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:PointwiseParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  var sum:f32=0.0;
  let dw_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels; let spatial_stride=params.height*params.width;
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){
    sum+=dw_output[dw_base+ic*spatial_stride]*pw_weight[w_base+ic];
  }
  sum+=pw_bias[oc];
  // Skip connection: zero-pad channels
  var skip_val:f32=0.0;
  if(oc<params.channel_pad){
    if(params.stride==2u){
      var max_val:f32=-1e38;
      for(var py:u32=0u;py<2u;py=py+1u){
        for(var px:u32=0u;px<2u;px=px+1u){
          let skip_y=out_y*2u+py; let skip_x=out_x*2u+px;
          if(skip_y<params.in_height && skip_x<params.in_width){
            let skip_idx=batch*params.channel_pad*params.in_height*params.in_width+oc*params.in_height*params.in_width+skip_y*params.in_width+skip_x;
            max_val=max(max_val,skip_input[skip_idx]);
          }
        }
      }
      skip_val=max_val;
    } else {
      let skip_idx=batch*params.channel_pad*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
      skip_val=skip_input[skip_idx];
    }
  }
  let v=sum+skip_val;
  let a=alpha[oc];
  let result=max(0.0,v)+a*min(0.0,v);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),pn=st(`
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
`),cn=st(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> skip:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> params:UpsampleParams;
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
  output[out_idx]=val+skip[out_idx];
}
`),dn=st(`
struct Conv1x1Params { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read> alpha:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:Conv1x1Params;
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
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),_n=st(`
struct CanvasParams { in_width:u32, in_height:u32, out_size:u32, }
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CanvasParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y;
  if(x>=params.in_width||y>=params.in_height){return;}
  let pixel=textureLoad(input_tex,vec2<u32>(x,y),0);
  let out_stride=params.out_size*params.out_size;
  output[0u*out_stride+y*params.out_size+x]=pixel.r;
  output[1u*out_stride+y*params.out_size+x]=pixel.g;
  output[2u*out_stride+y*params.out_size+x]=pixel.b;
}
`);async function ha(n,_){let i;if(_)i=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let c=await navigator.gpu.requestAdapter();if(!c)throw new Error("No GPU adapter found");i=await c.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(c.limits.maxStorageBuffersPerShaderStage,8)}})}let y={r:"read-only-storage",s:"storage",u:"uniform"};function w(c){return i.createBindGroupLayout({entries:c.map((r,k)=>r==="t"?{binding:k,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:k,visibility:GPUShaderStage.COMPUTE,buffer:{type:y[r]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,g=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,S=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,D=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(c,r){return i.createBuffer({size:Math.max(c,4),usage:r})}function h(c,r,k){i.queue.writeBuffer(c,r,k)}function v(c){let r=d(c.data.byteLength,e);return h(r,0,c.data),r}let U=Array.from(n.keys());function T(c){let r=n.get(c);if(!r)throw new Error(`Weight not found: ${c}`);return r}function m(...c){let r=U.find(k=>c.every(G=>k.includes(G)));if(!r)throw new Error(`Weight not found for: ${c.join(", ")}`);return T(r)}function z(c){let[,r,k,G]=c.shape,b=new Float32Array(G*25);for(let l=0;l<G;l++)for(let O=0;O<r;O++)for(let J=0;J<k;J++)b[l*25+O*5+J]=c.data[O*k*G+J*G+l];return b}function C(c){let[r,,,k]=c.shape,G=new Float32Array(r*k);for(let b=0;b<r;b++)for(let l=0;l<k;l++)G[b*k+l]=c.data[b*k+l];return G}let pe=i.createShaderModule({code:sn}),re=i.createShaderModule({code:on}),ve=i.createShaderModule({code:un}),ce=i.createShaderModule({code:pn}),M=i.createShaderModule({code:dn}),Y=i.createShaderModule({code:cn}),ae=i.createShaderModule({code:_n}),H=w(["r","r","r","r","s","u"]),F=w(["r","r","r","s","u"]),N=w(["r","r","r","r","r","s","u"]),Ae=w(["r","r","r","s","u"]),ge=w(["r","r","r","r","s","u"]),He=w(["r","r","s","u"]),fe=w(["t","s","u"]);function Pe(c,r){return i.createComputePipeline({layout:i.createPipelineLayout({bindGroupLayouts:[c]}),compute:{module:r,entryPoint:"main"}})}let P=Pe(H,pe),V=Pe(F,re),he=Pe(N,ve),j=Pe(Ae,ce),we=Pe(ge,M),se=Pe(He,Y),$e=Pe(fe,ae),Ze=m("conv2d/Conv2D"),Be=m("batch_normalization/","conv2d/Conv2D"),Ce=m("p_re_lu/"),be=v(Ze),ye=v(Be),Se=v(Ce),Te=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12}].map(c=>{let r=m(c.dwKey),k=m(c.pwKey),G=m(c.bnKey),b=m(c.preluKey),l=z(r),O=d(l.byteLength,e);h(O,0,l);let J=new Float32Array(c.inCh),L=d(J.byteLength,e);h(L,0,J);let q=C(k),R=d(q.byteLength,e);h(R,0,q);let K=v(G),$=v(b);return{dwWeightBuf:O,dwBiasBuf:L,pwWeightBuf:R,pwBiasBuf:K,alphaBuf:$,inCh:c.inCh,outCh:c.outCh,stride:c.stride,inH:c.inH}}),De=C(m("conv2d_20/Conv2D")),Ye=d(De.byteLength,e);h(Ye,0,De);let Me=v(m("batch_normalization_20/")),ot=v(m("p_re_lu_20/")),Je={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_19/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(256),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:Ye,pwBiasBuf:Me,alphaBuf:ot,inCh:256,outCh:256,stride:1,inH:12},Le={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_20/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(256),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:(()=>{let c=C(m("conv2d_21/")),r=d(c.byteLength,e);return h(r,0,c),r})(),pwBiasBuf:v(m("batch_normalization_21/")),alphaBuf:v(m("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},Pt=C(m("conv2d_23/Conv2D")),Mt=d(Pt.byteLength,e);h(Mt,0,Pt);let ut=v(m("batch_normalization_23/")),pt=v(m("p_re_lu_23/")),Jt={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_21/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(128),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:(()=>{let c=C(m("conv2d_24/")),r=d(c.byteLength,e);return h(r,0,c),r})(),pwBiasBuf:v(m("batch_normalization_24/")),alphaBuf:v(m("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},Qt={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_22/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(128),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:(()=>{let c=C(m("conv2d_25/Conv2D1")),r=d(c.byteLength,e);return h(r,0,c),r})(),pwBiasBuf:v(m("batch_normalization_25/")),alphaBuf:v(m("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},Et=C(m("classifier_palm_16_NO_PRUNING/Conv2D")),ct=d(Et.byteLength,e);h(ct,0,Et);let dt=v(m("classifier_palm_16_NO_PRUNING/BiasAdd")),Bt=C(m("regressor_palm_16_NO_PRUNING/Conv2D")),_t=d(Bt.byteLength,e);h(_t,0,Bt);let Ct=v(m("regressor_palm_16_NO_PRUNING/BiasAdd")),Qe=C(m("classifier_palm_8_NO_PRUNING/Conv2D")),Wt=d(Qe.byteLength,e);h(Wt,0,Qe);let ea=v(m("classifier_palm_8_NO_PRUNING/BiasAdd")),Ht=C(m("regressor_palm_8_NO_PRUNING/Conv2D")),Tt=d(Ht.byteLength,e);h(Tt,0,Ht);let Ee=v(m("regressor_palm_8_NO_PRUNING/BiasAdd")),ta=36864*3,lt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Re=d(36864*3*4,e),et=d(lt,g),Ue=d(lt,g),Lt=d(lt,g),kt=d(576*128*4,g),Rt=d(576*128*4,g),tt=d(864*4,S),We=d(15552*4,S),Ut=d(576*2*4,S),Z=d(576*36*4,S),de=d(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ge=d(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),at=d(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),nt=d(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),it=i.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function ee(c,r){return Math.ceil(c/r)}function le(c){let r=d(c.byteLength,D);return h(r,0,c),r}let X=le(new Uint32Array([1,3,32,192,192,96,96]));function oe(c,r,k,G,b){let l=r.stride===2?r.inH/2:r.inH,O=l,J=r.stride===2?1:2,L=le(new Uint32Array([1,r.inCh,r.inH,r.inH,l,O,r.stride,J])),q=i.createBindGroup({layout:F,entries:[{binding:0,resource:{buffer:k}},{binding:1,resource:{buffer:r.dwWeightBuf}},{binding:2,resource:{buffer:r.dwBiasBuf}},{binding:3,resource:{buffer:Lt}},{binding:4,resource:{buffer:L}}]}),R=c.beginComputePass();R.setPipeline(V),R.setBindGroup(0,q),R.dispatchWorkgroups(ee(O,8),ee(l,8),r.inCh),R.end();let K=r.inCh,$=le(new Uint32Array([1,r.inCh,r.outCh,l,O,K,r.stride,r.inH,r.inH])),ue=i.createBindGroup({layout:N,entries:[{binding:0,resource:{buffer:Lt}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:r.pwWeightBuf}},{binding:3,resource:{buffer:r.pwBiasBuf}},{binding:4,resource:{buffer:r.alphaBuf}},{binding:5,resource:{buffer:G}},{binding:6,resource:{buffer:$}}]}),B=c.beginComputePass();B.setPipeline(he),B.setBindGroup(0,ue),B.dispatchWorkgroups(ee(O,8),ee(l,8),r.outCh),B.end()}function Oe(c,r,k,G,b,l,O,J,L){let q=le(new Uint32Array([1,l,O,J,L])),R=i.createBindGroup({layout:Ae,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:k}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:b}},{binding:4,resource:{buffer:q}}]}),K=c.beginComputePass();K.setPipeline(j),K.setBindGroup(0,R),K.dispatchWorkgroups(ee(L,8),ee(J,8),O),K.end()}function Ot(c,r,k,G,b,l,O,J,L,q){let R=le(new Uint32Array([1,O,J,L,q])),K=i.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:k}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:b}},{binding:4,resource:{buffer:l}},{binding:5,resource:{buffer:R}}]}),$=c.beginComputePass();$.setPipeline(we),$.setBindGroup(0,K),$.dispatchWorkgroups(ee(q,8),ee(L,8),J),$.end()}async function It(c){i.queue.copyExternalImageToTexture({source:c},{texture:it},[192,192]);let r=le(new Uint32Array([192,192,192])),k=i.createBindGroup({layout:fe,entries:[{binding:0,resource:it.createView()},{binding:1,resource:{buffer:Re}},{binding:2,resource:{buffer:r}}]}),G=i.createCommandEncoder();{let B=G.beginComputePass();B.setPipeline($e),B.setBindGroup(0,k),B.dispatchWorkgroups(ee(192,16),ee(192,16),1),B.end()}{let B=i.createBindGroup({layout:H,entries:[{binding:0,resource:{buffer:Re}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:ye}},{binding:3,resource:{buffer:Se}},{binding:4,resource:{buffer:et}},{binding:5,resource:{buffer:X}}]}),Q=G.beginComputePass();Q.setPipeline(P),Q.setBindGroup(0,B),Q.dispatchWorkgroups(ee(96,8),ee(96,8),32),Q.end()}let b=et,l=Ue;for(let B=0;B<Te.length;B++){let Q=Te[B];oe(G,Q,b,l,b);let ne=b;b=l,l=ne,B===10&&G.copyBufferToBuffer(b,0,kt,0,576*128*4)}oe(G,Je,b,l,b);{let B=b;b=l,l=B}oe(G,Le,b,l,b);{let B=b;b=l,l=B}Oe(G,b,ct,dt,tt,256,6,12,12),Oe(G,b,_t,Ct,We,256,108,12,12),Ot(G,b,Mt,ut,pt,l,256,128,12,12);{let B=b;b=l,l=B}{let B=le(new Uint32Array([1,128,12,12,24,24])),Q=i.createBindGroup({layout:He,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:kt}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:B}}]}),ne=G.beginComputePass();ne.setPipeline(se),ne.setBindGroup(0,Q),ne.dispatchWorkgroups(ee(24,8),ee(24,8),128),ne.end()}{let B=b;b=l,l=B}oe(G,Jt,b,l,b);{let B=b;b=l,l=B}oe(G,Qt,b,l,b);{let B=b;b=l,l=B}Oe(G,b,Wt,ea,Ut,128,2,24,24),Oe(G,b,Tt,Ee,Z,128,36,24,24),G.copyBufferToBuffer(tt,0,de,0,864*4),G.copyBufferToBuffer(We,0,Ge,0,15552*4),G.copyBufferToBuffer(Ut,0,at,0,576*2*4),G.copyBufferToBuffer(Z,0,nt,0,576*36*4),i.queue.submit([G.finish()]),await Promise.all([de.mapAsync(GPUMapMode.READ),Ge.mapAsync(GPUMapMode.READ),at.mapAsync(GPUMapMode.READ),nt.mapAsync(GPUMapMode.READ)]);let O=new Float32Array(de.getMappedRange()).slice(),J=new Float32Array(Ge.getMappedRange()).slice(),L=new Float32Array(at.getMappedRange()).slice(),q=new Float32Array(nt.getMappedRange()).slice();de.unmap(),Ge.unmap(),at.unmap(),nt.unmap();let R=2016,K=new Float32Array(R),$=new Float32Array(R*18),ue=0;for(let B=0;B<12;B++)for(let Q=0;Q<12;Q++)for(let ne=0;ne<6;ne++){K[ue]=O[ne*144+B*12+Q];for(let xe=0;xe<18;xe++){let mt=ne*18+xe;$[ue*18+xe]=J[mt*144+B*12+Q]}ue++}for(let B=0;B<24;B++)for(let Q=0;Q<24;Q++)for(let ne=0;ne<2;ne++){K[ue]=L[ne*576+B*24+Q];for(let xe=0;xe<18;xe++){let mt=ne*18+xe;$[ue*18+xe]=q[mt*576+B*24+Q]}ue++}return{scores:K,regressors:$}}async function me(c,r){let k=i.createBuffer({size:r*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),G=i.createCommandEncoder();G.copyBufferToBuffer(c,0,k,0,r*4),i.queue.submit([G.finish()]),await k.mapAsync(GPUMapMode.READ);let b=new Float32Array(k.getMappedRange()).slice();return k.unmap(),k.destroy(),b}async function Ft(c){i.queue.copyExternalImageToTexture({source:c},{texture:it},[192,192]);function r(R,K=1e3){let $=R.slice(0,K);return{min:Math.min(...$),max:Math.max(...$),mean:$.reduce((ue,B)=>ue+B,0)/$.length,nonZero:$.filter(ue=>ue!==0).length,sample:Array.from($.slice(0,10))}}let k={},G=le(new Uint32Array([192,192,192])),b=i.createBindGroup({layout:fe,entries:[{binding:0,resource:it.createView()},{binding:1,resource:{buffer:Re}},{binding:2,resource:{buffer:G}}]}),l=i.createCommandEncoder(),O=l.beginComputePass();O.setPipeline($e),O.setBindGroup(0,b),O.dispatchWorkgroups(ee(192,16),ee(192,16),1),O.end(),i.queue.submit([l.finish()]),k.input=r(await me(Re,36864*3)),l=i.createCommandEncoder();let J=i.createBindGroup({layout:H,entries:[{binding:0,resource:{buffer:Re}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:ye}},{binding:3,resource:{buffer:Se}},{binding:4,resource:{buffer:et}},{binding:5,resource:{buffer:X}}]});O=l.beginComputePass(),O.setPipeline(P),O.setBindGroup(0,J),O.dispatchWorkgroups(ee(96,8),ee(96,8),32),O.end(),i.queue.submit([l.finish()]),k.initConv=r(await me(et,9216*32));let L=et,q=Ue;for(let R=0;R<Te.length;R++){let K=Te[R];l=i.createCommandEncoder(),oe(l,K,L,q,L),i.queue.submit([l.finish()]);let $=L;if(L=q,q=$,R===0||R===3||R===7||R===11||R===17){let ue=K.stride===2?K.inH/2:K.inH,B=ue*ue*K.outCh;k[`block${R}`]=r(await me(L,B))}R===10&&(l=i.createCommandEncoder(),l.copyBufferToBuffer(L,0,kt,0,576*128*4),i.queue.submit([l.finish()]))}l=i.createCommandEncoder(),oe(l,Je,L,q,L),i.queue.submit([l.finish()]);{let R=L;L=q,q=R}k.extraBlockA=r(await me(L,144*256)),l=i.createCommandEncoder(),oe(l,Le,L,q,L),i.queue.submit([l.finish()]);{let R=L;L=q,q=R}return k.extraBlockB=r(await me(L,144*256)),l=i.createCommandEncoder(),Oe(l,L,ct,dt,tt,256,6,12,12),i.queue.submit([l.finish()]),k.cls16=r(await me(tt,864)),l=i.createCommandEncoder(),Oe(l,L,_t,Ct,We,256,108,12,12),i.queue.submit([l.finish()]),k.reg16=r(await me(We,15552),500),k.initWeights=r(await me(be,100),100),k.initBias=r(await me(ye,32),32),k.cls16Weights=r(await me(ct,100),100),k.cls16Bias=r(await me(dt,6),6),k}return{device:i,run:It,debugRun:Ft}}function Kn(){let n=[];for(let _=0;_<12;_++)for(let i=0;i<12;i++){let y=(i+.5)/12,w=(_+.5)/12;for(let e=0;e<6;e++)n.push({x:y,y:w})}for(let _=0;_<24;_++)for(let i=0;i<24;i++){let y=(i+.5)/24,w=(_+.5)/24;for(let e=0;e<2;e++)n.push({x:y,y:w})}return n}var ln=Kn();function qn(n){return 1/(1+Math.exp(-n))}function mn(n,_){let i=[],{scores:y,regressors:w}=n,e=192;for(let g=0;g<ln.length;g++){let S=qn(y[g]);if(S<_)continue;let D=ln[g],d=g*18,h=D.x+w[d+0]/e,v=D.y+w[d+1]/e,U=w[d+2]/e,T=w[d+3]/e,m=[];for(let z=0;z<7;z++){let C=D.x+w[d+4+z*2]/e,pe=D.y+w[d+4+z*2+1]/e;m.push([C,pe])}i.push({score:S,box:[h,v,U,T],keypoints:m})}return i}function fn(n,_){if(n.length===0)return[];let i=[...n].sort((e,g)=>g.score-e.score),y=[],w=new Set;for(let e=0;e<i.length;e++)if(!w.has(e)){y.push(i[e]);for(let g=e+1;g<i.length;g++)w.has(g)||$n(i[e],i[g])>_&&w.add(g)}return y}function $n(n,_){let i=n.box[0]-n.box[2]/2,y=n.box[1]-n.box[3]/2,w=n.box[0]+n.box[2]/2,e=n.box[1]+n.box[3]/2,g=_.box[0]-_.box[2]/2,S=_.box[1]-_.box[3]/2,D=_.box[0]+_.box[2]/2,d=_.box[1]+_.box[3]/2,h=Math.max(i,g),v=Math.max(y,S),U=Math.min(w,D),T=Math.min(e,d),m=Math.max(0,U-h),z=Math.max(0,T-v),C=m*z,pe=(w-i)*(e-y),re=(D-g)*(d-S),ve=pe+re-C;return ve>0?C/ve:0}function Yn(n){let[_,i,y,w]=n.box,e=n.keypoints[0],g=n.keypoints[2],S=g[0]-e[0],D=g[1]-e[1],d=Math.atan2(S,D),v=Math.max(y,w)*2.6,U=.5,T=Math.sqrt(S*S+D*D),m=T>0?S/T*v*U*.5:0,z=T>0?D/T*v*U*.5:0;return{centerX:_+m,centerY:i+z,width:v,height:v,rotation:d}}function wa(n,_={}){let{scoreThreshold:i=.5,nmsThreshold:y=.3,maxHands:w=2}=_;async function e(S){let D=await n.run(S),d=mn(D,i);return fn(d,y).slice(0,w).map(Yn)}async function g(S){let D=await n.run(S),d=mn(D,i);return fn(d,y).slice(0,w)}return{detect:e,detectRaw:g,model:n}}function hn(n,_=256){let i=Math.cos(n.rotation),y=Math.sin(n.rotation),w=n.width/_,e=n.height/_,g=w*i,S=-e*y,D=w*y,d=e*i,h=n.centerX-(g*_/2+S*_/2),v=n.centerY-(D*_/2+d*_/2),U=g*d-S*D,T=d/U,m=-S/U,z=-D/U,C=g/U,pe=-(T*h+m*v),re=-(z*h+C*v);return{forward:[g,S,h,D,d,v],inverse:[T,m,pe,z,C,re]}}function ga(n,_){let{forward:i}=hn(_,1),[y,w,e,g,S,D]=i;return n.map(d=>({x:y*d.x+w*d.y+e,y:g*d.x+S*d.y+D,z:d.z}))}var wn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Xn(n={}){let{weightsUrl:_,scoreThreshold:i=.5,forceF32:y=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let w=_??wn,e=w.endsWith("/")?w:`${w}/`,g=`${e}weights_f16.json`,S=`${e}weights_f16.bin`,[D,d]=await Promise.all([fetch(g),fetch(S)]);if(!D.ok)throw new Error(`Failed to fetch weights metadata: ${D.status}`);if(!d.ok)throw new Error(`Failed to fetch weights binary: ${d.status}`);let h=await D.json(),v=await d.arrayBuffer(),U=Dt(h,v),T=await Zt(U,{forceF32:y});if(!y){let H=new OffscreenCanvas(256,256),F=H.getContext("2d");F.fillStyle="#886644",F.fillRect(0,0,256,256),F.fillStyle="#cc9966",F.fillRect(50,50,156,156);let N=await T.runFromCanvas(H);N.landmarks.every(ge=>ge===0)&&N.handflag.every(ge=>ge===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),T.device.destroy(),T=await Zt(U,{forceF32:!0}))}let m=null;function z(){return m||(m=new OffscreenCanvas(256,256)),m}async function C(H){if(H instanceof HTMLCanvasElement||H instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&H instanceof ImageBitmap)return H;let F=z();F.width=256,F.height=256;let N=F.getContext("2d");return H instanceof ImageData?N.putImageData(H,0,0):N.drawImage(H,0,0,256,256),F}function pe(H,F,N){let Ae=H[0];if(Ae<i)return null;let ge=F[0]>.5,He=[];for(let fe=0;fe<21;fe++)He.push({x:N[fe*3],y:N[fe*3+1],z:N[fe*3+2]});return{score:Ae,handedness:ge?"right":"left",landmarks:He}}async function re(H){let F=await C(H),N=await T.runFromCanvas(F);return pe(N.handflag,N.handedness,N.landmarks)}async function ve(H){let F=await C(H),N=await T.runFromCanvasPipelined(F);return N?pe(N.handflag,N.handedness,N.landmarks):null}async function ce(){let H=await T.flushPipelined();return H?pe(H.handflag,H.handedness,H.landmarks):null}function M(){T.device.destroy(),m=null}async function Y(H){let F=await C(H);return T.benchmarkDiagnostic(F)}async function ae(H){let F=await C(H);return T.debugLayerOutputs(F)}return{detect:re,detectPipelined:ve,flushPipelined:ce,dispose:M,benchmarkDiagnostic:Y,debugLayerOutputs:ae}}async function Vn(n={}){let{weightsUrl:_,palmWeightsUrl:i,scoreThreshold:y=.5,palmScoreThreshold:w=.5,maxHands:e=2,forceF32:g=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let S=_??wn,D=S.endsWith("/")?S:`${S}/`;if(!i)throw new Error("palmWeightsUrl is required for createFullHandpose");let d=i.endsWith("/")?i:`${i}/`,[h,v,U,T]=await Promise.all([fetch(`${D}weights_f16.json`),fetch(`${D}weights_f16.bin`),fetch(`${d}palm_detection_weights.json`),fetch(`${d}palm_detection_weights.bin`)]);if(!h.ok)throw new Error(`Failed to fetch landmark weights metadata: ${h.status}`);if(!v.ok)throw new Error(`Failed to fetch landmark weights binary: ${v.status}`);if(!U.ok)throw new Error(`Failed to fetch palm weights metadata: ${U.status}`);if(!T.ok)throw new Error(`Failed to fetch palm weights binary: ${T.status}`);let[m,z,C,pe]=await Promise.all([h.json(),v.arrayBuffer(),U.json(),T.arrayBuffer()]),re=Dt(m,z),ve=Dt(C,pe),ce=await Zt(re,{forceF32:g}),M=await ha(ve),Y=wa(M,{scoreThreshold:w,maxHands:e}),ae=null,H=null;function F(){return ae||(ae=new OffscreenCanvas(192,192)),ae}function N(){return H||(H=new OffscreenCanvas(256,256)),H}async function Ae(P){if(P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas){if(P.width===192&&P.height===192)return P;let j=F();return j.width=192,j.height=192,j.getContext("2d").drawImage(P,0,0,192,192),j}if(typeof ImageBitmap<"u"&&P instanceof ImageBitmap){if(P.width===192&&P.height===192)return P;let j=F();return j.width=192,j.height=192,j.getContext("2d").drawImage(P,0,0,192,192),j}let V=F();V.width=192,V.height=192;let he=V.getContext("2d");if(P instanceof ImageData){let j=new OffscreenCanvas(P.width,P.height);j.getContext("2d").putImageData(P,0,0),he.drawImage(j,0,0,192,192)}else he.drawImage(P,0,0,192,192);return V}function ge(P,V,he,j){let we=N();we.width=256,we.height=256;let se=we.getContext("2d"),$e=Math.cos(-V.rotation),Ze=Math.sin(-V.rotation);se.clearRect(0,0,256,256),se.save(),se.translate(128,128),se.scale(V.width*he/256,V.height*j/256),se.rotate(-V.rotation),se.translate(-128,-128);let Be=V.centerX*he,Ce=V.centerY*j;se.restore();let be=256/(V.width*he),ye=256/(V.height*j),Se=Math.cos(V.rotation),ke=Math.sin(V.rotation),Te=Se*be,De=ke*be,Ye=-ke*ye,Me=Se*ye,ot=-Be*Te-Ce*Ye+128,Je=-Be*De-Ce*Me+128;if(se.setTransform(Te,De,Ye,Me,ot,Je),P instanceof ImageData){let Le=new OffscreenCanvas(P.width,P.height);Le.getContext("2d").putImageData(P,0,0),se.drawImage(Le,0,0)}else se.drawImage(P,0,0);return se.setTransform(1,0,0,1,0,0),we}function He(P){return P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas?[P.width,P.height]:typeof ImageBitmap<"u"&&P instanceof ImageBitmap?[P.width,P.height]:P instanceof ImageData?[P.width,P.height]:P instanceof HTMLVideoElement?[P.videoWidth,P.videoHeight]:P instanceof HTMLImageElement?[P.naturalWidth,P.naturalHeight]:[256,256]}async function fe(P){let V=await Ae(P),he=await Y.detect(V);if(he.length===0)return[];let[j,we]=He(P),se=[];for(let $e of he){let Ze=ge(P,$e,j,we),Be=await ce.runFromCanvas(Ze),Ce=Be.handflag[0];if(Ce<y)continue;let be=Be.handedness[0]>.5,ye=[];for(let ke=0;ke<21;ke++)ye.push({x:Be.landmarks[ke*3],y:Be.landmarks[ke*3+1],z:Be.landmarks[ke*3+2]});let Se=ga(ye,$e);se.push({score:Ce,handedness:be?"right":"left",landmarks:Se,palmScore:0})}return se}function Pe(){ce.device.destroy(),M.device.destroy(),ae=null,H=null}return{detect:fe,dispose:Pe}}function jn(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Zn=jn(`
struct CropParams { src_width:u32, src_height:u32, dst_size:u32, _pad:u32, }
struct AffineTransform { a:f32, b:f32, tx:f32, c:f32, d:f32, ty:f32, }

@group(0)@binding(0) var src_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CropParams;
@group(0)@binding(3) var<uniform> transform:AffineTransform;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let dst_x=gid.x; let dst_y=gid.y;
  if(dst_x>=params.dst_size||dst_y>=params.dst_size){return;}

  // Map crop pixel to source normalized coordinates
  let fx=f32(dst_x)+0.5;
  let fy=f32(dst_y)+0.5;
  let src_nx=transform.a*fx+transform.b*fy+transform.tx;
  let src_ny=transform.c*fx+transform.d*fy+transform.ty;

  // Convert to pixel coordinates
  let src_px=src_nx*f32(params.src_width)-0.5;
  let src_py=src_ny*f32(params.src_height)-0.5;

  // Bilinear interpolation
  let x0=i32(floor(src_px)); let y0=i32(floor(src_py));
  let x1=x0+1; let y1=y0+1;
  let lx=src_px-f32(x0); let ly=src_py-f32(y0);

  let sw=i32(params.src_width); let sh=i32(params.src_height);

  // Clamp to bounds (or return 0 for out-of-bounds)
  var p00=vec4<f32>(0.0); var p01=vec4<f32>(0.0);
  var p10=vec4<f32>(0.0); var p11=vec4<f32>(0.0);

  if(x0>=0 && x0<sw && y0>=0 && y0<sh){ p00=textureLoad(src_tex,vec2<u32>(u32(x0),u32(y0)),0); }
  if(x1>=0 && x1<sw && y0>=0 && y0<sh){ p01=textureLoad(src_tex,vec2<u32>(u32(x1),u32(y0)),0); }
  if(x0>=0 && x0<sw && y1>=0 && y1<sh){ p10=textureLoad(src_tex,vec2<u32>(u32(x0),u32(y1)),0); }
  if(x1>=0 && x1<sw && y1>=0 && y1<sh){ p11=textureLoad(src_tex,vec2<u32>(u32(x1),u32(y1)),0); }

  let pixel=p00*(1.0-lx)*(1.0-ly)+p01*lx*(1.0-ly)+p10*(1.0-lx)*ly+p11*lx*ly;

  // Write CHW format
  let out_stride=params.dst_size*params.dst_size;
  output[0u*out_stride+dst_y*params.dst_size+dst_x]=pixel.r;
  output[1u*out_stride+dst_y*params.dst_size+dst_x]=pixel.g;
  output[2u*out_stride+dst_y*params.dst_size+dst_x]=pixel.b;
}
`);function Jn(n){let _=n.createShaderModule({code:Zn}),i=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),y=n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[i]}),compute:{module:_,entryPoint:"main"}});function w(e,g,S,D,d,h,v){let U=new Uint32Array([d,h,v,0]),T=n.createBuffer({size:U.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(T,0,U);let m=new Float32Array(D),z=new Float32Array(8);z.set(m);let C=n.createBuffer({size:z.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(C,0,z);let pe=n.createBindGroup({layout:i,entries:[{binding:0,resource:g.createView()},{binding:1,resource:{buffer:S}},{binding:2,resource:{buffer:T}},{binding:3,resource:{buffer:C}}]}),re=e.beginComputePass();re.setPipeline(y),re.setBindGroup(0,pe),re.dispatchWorkgroups(Math.ceil(v/16),Math.ceil(v/16),1),re.end()}return{crop:w}}export{ha as compilePalmModel,hn as computeCropTransform,Jn as createCropPipeline,Vn as createFullHandpose,Xn as createHandpose,wa as createPalmDetector,Dt as loadWeightsFromBuffer,ga as projectLandmarksToOriginal};
