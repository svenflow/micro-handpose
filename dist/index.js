function we(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ta(s){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+s;for(let B of _)for(;a.includes(`${B}:array<f32>`);)a=a.replace(`${B}:array<f32>`,`${B}:array<f16>`);return a}var _a=we(`
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
`),fa=we(`
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
`),ma=we(`
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
`),ha=we(`
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
`);function La(s,_){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Ra(s,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Oa(s,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Ia(s,_){return ha.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Fa(s,_){return[8,8]}var za=we(`
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
`),Na=we(`
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
`);function qa(s){return we(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${s?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${s?"+skip[out_idx]":""};
}
`)}var Ka=qa(!1),$a=qa(!0),Ya=we(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Xa=we(`
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
`);function Va(s){return we(`
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
  ${s==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var Za=Va("sigmoid"),ja=Va("div256"),Ja=we(`
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
`),Qa=we(`
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
`);function en(s,_){let B=Math.min(_,256),x=_>B,v=s%4===0?`var ic:u32=0u;
    while(ic<${s}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${s}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,T=`var skip_val:f32=0.0;
    if(c<${s}u){
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
    }`,L=s===_?"":`if(c<${s}u){`,l=s===_?"":"}",b=x?`for(var c:u32=lid.x;c<${s}u;c+=${B}u){`:`let c=lid.x;
  ${L}`,C=x?"}":l,D=x?`for(var c:u32=lid.x;c<${_}u;c+=${B}u){`:"{let c=lid.x;";return we(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${s}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${B},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${b}
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
  ${C}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${D}
    let pw_base=c*${s}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${v}
    // Skip connection (only for c < inCh)
    ${T}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var tn=we(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),an=we(`
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
`),nn=we(`
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
`);function Ot(s,_){let a=new Map,B=s.dtype??"float32";for(let x=0;x<s.keys.length;x++){let e=s.keys[x],v=s.shapes[x],T=s.offsets[x],L=v.reduce((C,D)=>C*D,1),l,b;if(B==="float32")l=new Float32Array(_,T,L);else{let C=new DataView(_);l=new Float32Array(L);for(let D=0;D<L;D++)l[D]=Rn(C.getUint16(T+D*2,!0));b=_.slice(T,T+L*2)}a.set(e,{data:l,shape:v,rawF16:b})}return a}function Rn(s){let _=s>>15&1,a=s>>10&31,B=s&1023;if(a===0){if(B===0)return _?-0:0;let v=-14,T=B/1024;return(_?-1:1)*Math.pow(2,v)*T}if(a===31)return B===0?_?-1/0:1/0:NaN;let x=a-15,e=1+B/1024;return(_?-1:1)*Math.pow(2,x)*e}var On=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],rn=On.map(([s,_,a,B,x])=>({type:"resmodule",inCh:s,outCh:_,h:a,w:a,stride:B,prefix:x})),In=2,Fn=5,zn=8,Nn=11;async function It(s,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let B=a.features.has("shader-f16"),x=B?["shader-f16"]:[],e=await a.requestDevice({requiredFeatures:x,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(a.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(a.limits.maxComputeInvocationsPerWorkgroup,288)}}),v=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(B)try{let i=`enable f16;
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
}`,c=`enable f16;
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
}`,u=e.createShaderModule({code:i}),o=e.createShaderModule({code:c}),t=await u.getCompilationInfo(),H=await o.getCompilationInfo();if(t.messages.some(M=>M.type==="error")||H.messages.some(M=>M.type==="error"))v=!1;else{let M=new Float32Array(2400);M.fill(1);let R=new Uint16Array(2400);R.fill(10516);let U=new Uint16Array(96);U.fill(14336);let w=new Uint16Array(9216);w.fill(8478);let p=new Uint16Array(96);p.fill(12288);let q=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,se=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,le=e.createBuffer({size:M.byteLength,usage:q}),kt=e.createBuffer({size:R.byteLength,usage:q}),Ut=e.createBuffer({size:U.byteLength,usage:q}),Gt=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),At=e.createBuffer({size:w.byteLength,usage:q}),St=e.createBuffer({size:p.byteLength,usage:q}),Dt=e.createBuffer({size:384,usage:se}),et=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(le,0,M),e.queue.writeBuffer(kt,0,R),e.queue.writeBuffer(Ut,0,U),e.queue.writeBuffer(At,0,w),e.queue.writeBuffer(St,0,p);let Ye="read-only-storage",Tt="storage",Lt=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ye}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ye}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ye}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Tt}}]}),Wa=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ye}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ye}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ye}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Tt}}]}),Mn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Lt]}),compute:{module:u,entryPoint:"main"}}),En=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Wa]}),compute:{module:o,entryPoint:"main"}}),Wn=e.createBindGroup({layout:Lt,entries:[{binding:0,resource:{buffer:le}},{binding:1,resource:{buffer:kt}},{binding:2,resource:{buffer:Ut}},{binding:3,resource:{buffer:Gt}}]}),Hn=e.createBindGroup({layout:Wa,entries:[{binding:0,resource:{buffer:Gt}},{binding:1,resource:{buffer:At}},{binding:2,resource:{buffer:St}},{binding:3,resource:{buffer:Dt}}]}),ea=e.createCommandEncoder(),ta=ea.beginComputePass();ta.setPipeline(Mn),ta.setBindGroup(0,Wn),ta.dispatchWorkgroups(2),ta.end();let aa=ea.beginComputePass();aa.setPipeline(En),aa.setBindGroup(0,Hn),aa.dispatchWorkgroups(2),aa.end(),ea.copyBufferToBuffer(Dt,0,et,0,384),e.queue.submit([ea.finish()]),await e.queue.onSubmittedWorkDone(),await et.mapAsync(GPUMapMode.READ);let Rt=new Float32Array(et.getMappedRange()),Ha=1.5*.0104*96+.25,Tn=Rt[0]!==0&&Rt[47]!==0&&Rt[95]!==0,Ln=Math.abs(Rt[0]-Ha)<1;v=Tn&&Ln,et.unmap(),le.destroy(),kt.destroy(),Ut.destroy(),Gt.destroy(),At.destroy(),St.destroy(),Dt.destroy(),et.destroy(),v||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Rt[0]}, expected ~${Ha.toFixed(2)}) \u2014 falling back to f32`)}}catch{v=!1}let L=s.values().next().value,l=v&&!!L?.rawF16&&!_?.forceF32;console.log(l?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${B}, f16 validated: ${v}, f16 data: ${!!L?.rawF16})`);function b(i){if(l&&i.rawF16){let c=new Uint16Array(i.rawF16);if(c.length%2!==0){let u=new Uint16Array(c.length+1);return u.set(c),u}return c}return i.data}function C(i){if(l&&i.rawF16){let c=i.rawF16.byteLength;return Math.ceil(c/4)*4}return i.data.byteLength}function D(i){return l?Ta(i):i}let K={r:"read-only-storage",s:"storage",u:"uniform"};function m(i){return e.createBindGroupLayout({entries:i.map((c,u)=>({binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:K[c]}}))})}function Y(i){return e.createBindGroupLayout({entries:i.map((c,u)=>c==="t"?{binding:u,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:K[c]}})})}let G=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ce=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,re=GPUBufferUsage.STORAGE,ge=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,oe=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function W(i,c){return e.createBuffer({size:i,usage:c})}function ee(i,c){return e.createBindGroup({layout:i,entries:c.map((u,o)=>({binding:o,resource:"size"in u?{buffer:u}:u}))})}function ue(i,c){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),compute:{module:c,entryPoint:"main"}})}let O=e.createShaderModule({code:za}),X=e.createShaderModule({code:nn}),j=e.createShaderModule({code:D(Ja)}),We=e.createShaderModule({code:D(fa)}),xe=e.createShaderModule({code:D(_a)}),me=e.createShaderModule({code:D(ma)}),ye=e.createShaderModule({code:D(ha)}),Ce=e.createShaderModule({code:D(Na)}),k=e.createShaderModule({code:Ka}),V=e.createShaderModule({code:Ya}),he=e.createShaderModule({code:$a}),te=e.createShaderModule({code:D(Xa)}),_e=e.createShaderModule({code:D(Za)}),J=e.createShaderModule({code:D(ja)}),Xe=e.createShaderModule({code:D(Qa)}),tt=new Map;function ke(i,c){let u=`${i}_${c}`,o=tt.get(u);return o||(o=e.createShaderModule({code:D(en(i,c))}),tt.set(u,o)),o}let Ue=m(["r","r","r","s","u"]),ve=m(["r","r","r","r","s","u"]),Pe=m(["r","s","u"]),He=m(["r","r","r","s","u"]),Ge=m(["r","s","u"]),Oe=m(["r","r","s","u"]),Te=m(["r","r","s","u"]),Le=m(["r","r","r","s","u"]),Ae=m(["r","r","r","s","u"]),Ve=Y(["t","s","u"]),at=m(["r","r","r","r","r","r","r","s"]),Ze=m(["r","r","r","r","r","s","u"]),nt=e.createPipelineLayout({bindGroupLayouts:[Ue]}),ht=e.createPipelineLayout({bindGroupLayouts:[ve]}),it=i=>e.createComputePipeline({layout:nt,compute:{module:i,entryPoint:"main"}}),rt=i=>e.createComputePipeline({layout:ht,compute:{module:i,entryPoint:"main"}}),Ft=it(We),zt=it(xe),Nt=rt(me),wt=rt(ye),bt=new Map,Mt=new Map,gt=new Map,Et=new Map;bt.set("8,8",Ft),Mt.set("8,8",zt),gt.set("8,8",Nt),Et.set("8,8",wt);function st(i,c,u,o,t){let H=`${c},${u}`,M=i.get(H);return M||(M=t(e.createShaderModule({code:D(o(c,u))})),i.set(H,M)),M}let yt=(i,c)=>st(bt,i,c,La,it),Wt=(i,c)=>st(Mt,i,c,Ra,it),qt=(i,c)=>st(gt,i,c,Oa,rt),Ht=(i,c)=>st(Et,i,c,Ia,rt),Se=rn.map(i=>{let c=i.stride===2?i.h/2:i.h,u=i.stride===2?i.w/2:i.w,[o,t]=Fa(i.inCh,c),H=i.h>=64,M=c>=16&&i.inCh>=288&&i.outCh>=288&&i.outCh%2===0;return{dwPipeline:H?Wt(o,t):yt(o,t),pwPipeline:M?Ht(o,t):qt(o,t),dwDispatchX:Math.ceil(u/o),dwDispatchY:Math.ceil(c/t),dwDispatchZ:i.inCh,pwDispatchX:Math.ceil(u/o),pwDispatchY:Math.ceil(c/t),pwDispatchZ:M?i.outCh/2:i.outCh}}),xt=ue(Pe,O),je=ue(He,Ce);ue(Ge,k),ue(Oe,V);let Ie=ue(Te,he),Kt=ue(Le,te);ue(Ae,_e),ue(Ae,J);let De=ue(Ve,X),vt=ue(at,j),ot=ue(Ze,Xe),Fe=1*288*128*128*4,ut=W(3*256*256*4,G),Re=W(3*257*257*4,re),pt=W(12,oe);e.queue.writeBuffer(pt,0,new Uint32Array([3,256,257]));let Q=W(Fe,ce),fe=W(Fe,ge),Me=W(Fe,re),ct=W(3072*64*4,G),dt=W(3072*32*4,G),lt=W(1536*16*4,G),F=W(6144*64*4,re),ne=W(260,ge),ae=W(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);W(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let pe=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ee=W(8,oe);e.queue.writeBuffer(Ee,0,new Uint32Array([256,257]));let _t=s.get("backbone1.1.weight"),$t=s.get("backbone1.1.bias");if(!_t||!$t)throw new Error("Missing input conv weights");let $=b(_t),Yt=b($t),d=W($.byteLength,G),n=W(Yt.byteLength,G),g=W(28,oe);e.queue.writeBuffer(d,0,$),e.queue.writeBuffer(n,0,Yt),e.queue.writeBuffer(g,0,new Uint32Array([1,3,24,257,257,128,128]));let S=s.get("backbone6.1.weight"),h=s.get("backbone6.1.bias");if(!S||!h)throw new Error("Missing backbone6.1 conv1x1 weights");let r=b(S),I=b(h),ie=W(r.byteLength,G),f=W(I.byteLength,G),A=W(20,oe);e.queue.writeBuffer(ie,0,r),e.queue.writeBuffer(f,0,I),e.queue.writeBuffer(A,0,new Uint32Array([1,96,48,32,32]));let y=s.get("handflag.weight"),N=s.get("handflag.bias");if(!y||!N)throw new Error("Missing handflag weights");let E=b(y),be=b(N),de=W(E.byteLength,G),P=W(be.byteLength,G),Z=W(12,oe);e.queue.writeBuffer(de,0,E),e.queue.writeBuffer(P,0,be),e.queue.writeBuffer(Z,0,new Uint32Array([1,288,1]));let z=s.get("handedness.weight"),Be=s.get("handedness.bias");if(!z||!Be)throw new Error("Missing handedness weights");let Pt=b(z),ya=b(Be),na=W(Pt.byteLength,G),ia=W(ya.byteLength,G),xa=W(12,oe);e.queue.writeBuffer(na,0,Pt),e.queue.writeBuffer(ia,0,ya),e.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let va=s.get("reg_3d.weight"),Pa=s.get("reg_3d.bias");if(!va||!Pa)throw new Error("Missing reg_3d weights");let Ba=b(va),Ca=b(Pa),ra=W(Ba.byteLength,G),sa=W(Ca.byteLength,G),ka=W(12,oe);e.queue.writeBuffer(ra,0,Ba),e.queue.writeBuffer(sa,0,Ca),e.queue.writeBuffer(ka,0,new Uint32Array([1,288,63]));let ft=rn.map(i=>{let{inCh:c,outCh:u,h:o,w:t,stride:H,prefix:M}=i,R=H===2?o/2:o,U=H===2?t/2:t,w=H===2?1:2,p=s.get(`${M}convs.0.weight`),q=s.get(`${M}convs.0.bias`),se=s.get(`${M}convs.1.weight`),le=s.get(`${M}convs.1.bias`);if(!p||!q||!se||!le)throw new Error(`Missing weights for ${M}`);let kt=b(p),Ut=b(q),Gt=b(se),At=b(le),St=W(kt.byteLength,G),Dt=W(Ut.byteLength,G),et=W(Gt.byteLength,G),Ye=W(At.byteLength,G),Tt=W(32,oe),Lt=W(36,oe);return e.queue.writeBuffer(St,0,kt),e.queue.writeBuffer(Dt,0,Ut),e.queue.writeBuffer(et,0,Gt),e.queue.writeBuffer(Ye,0,At),e.queue.writeBuffer(Tt,0,new Uint32Array([1,c,o,t,R,U,H,w])),e.queue.writeBuffer(Lt,0,new Uint32Array([1,c,u,R,U,Math.max(0,u-c),H,o,t])),{dwWeight:St,dwBias:Dt,pwWeight:et,pwBias:Ye,dwUniform:Tt,pwUniform:Lt,spec:i,outH:R,outW:U}});function Bt(i){let c=W(i.length*4,oe);return e.queue.writeBuffer(c,0,new Uint32Array(i)),c}let bn=Bt([1,96,8,8,16,16]),gn=Bt([1,96,16,16,32,32]),yn=Bt([1,48,32,32,64,64]);Bt([1536*16]),Bt([3072*32]),Bt([3072*64]);let Ua=ee(Pe,[ut,Re,pt]),Ga=ee(He,[Re,d,n,Q,g]),ze=[],Ne=[],qe=[],Ke=[];for(let i of ft)ze.push(ee(Ue,[Q,i.dwWeight,i.dwBias,Me,i.dwUniform])),Ne.push(ee(ve,[Me,Q,i.pwWeight,i.pwBias,fe,i.pwUniform])),qe.push(ee(Ue,[fe,i.dwWeight,i.dwBias,Me,i.dwUniform])),Ke.push(ee(ve,[Me,fe,i.pwWeight,i.pwBias,Q,i.pwUniform]));let xn=ee(Te,[Q,lt,fe,bn]),vn=ee(Te,[Q,dt,fe,gn]),Pn=ee(Le,[Q,ie,f,F,A]),Bn=ee(Te,[F,ct,fe,yn]);ee(Ae,[Q,de,P,ne,Z]),ee(Ae,[Q,na,ia,ne,xa]),ee(Ae,[Q,ra,sa,ne,ka]);let Je=ee(Ve,[pe.createView(),Re,Ee]),Cn=ee(at,[Q,de,P,na,ia,ra,sa,ne]),oa=24,Aa=[],Sa=[];for(let i=oa;i<ft.length;i++){let c=ft[i];Aa.push(ee(Ze,[Q,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,fe,c.dwUniform])),Sa.push(ee(Ze,[fe,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,Q,c.dwUniform]))}let ua=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});ua.globalCompositeOperation="copy";let Da=new OffscreenCanvas(9,8),Xt=Da.getContext("webgpu"),Vt=null,pa=null;if(Xt){Xt.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let i=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),c=e.createShaderModule({code:tn}),u=e.createShaderModule({code:an});Vt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),vertex:{module:c,entryPoint:"vs"},fragment:{module:u,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),pa=e.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:ne}}]})}let Zt=new Float32Array(1),jt=new Float32Array(1),Jt=new Float32Array(63);function $e(i,c){let u=!0,o=0,t=i.beginComputePass();for(t.setPipeline(je),t.setBindGroup(0,Ga),t.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);o<=In;o++){let U=u?ze[o]:qe[o],w=u?Ne[o]:Ke[o],p=Se[o];t.setPipeline(p.dwPipeline),t.setBindGroup(0,U),t.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),t.setPipeline(p.pwPipeline),t.setBindGroup(0,w),t.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}t.end();let H=u?Q:fe;for(i.copyBufferToBuffer(H,0,ct,0,3072*64*4),t=i.beginComputePass();o<=Fn;o++){let U=u?ze[o]:qe[o],w=u?Ne[o]:Ke[o],p=Se[o];t.setPipeline(p.dwPipeline),t.setBindGroup(0,U),t.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),t.setPipeline(p.pwPipeline),t.setBindGroup(0,w),t.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}t.end();let M=u?Q:fe;for(i.copyBufferToBuffer(M,0,dt,0,3072*32*4),t=i.beginComputePass();o<=zn;o++){let U=u?ze[o]:qe[o],w=u?Ne[o]:Ke[o],p=Se[o];t.setPipeline(p.dwPipeline),t.setBindGroup(0,U),t.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),t.setPipeline(p.pwPipeline),t.setBindGroup(0,w),t.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}t.end();let R=u?Q:fe;for(i.copyBufferToBuffer(R,0,lt,0,1536*16*4),t=i.beginComputePass();o<=Nn;o++){let U=u?ze[o]:qe[o],w=u?Ne[o]:Ke[o],p=Se[o];t.setPipeline(p.dwPipeline),t.setBindGroup(0,U),t.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),t.setPipeline(p.pwPipeline),t.setBindGroup(0,w),t.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}t.setPipeline(Ie),t.setBindGroup(0,xn),t.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),t.end(),u=!1,t=i.beginComputePass();{let U=u?ze[o]:qe[o],w=u?Ne[o]:Ke[o],p=Se[o];t.setPipeline(p.dwPipeline),t.setBindGroup(0,U),t.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),t.setPipeline(p.pwPipeline),t.setBindGroup(0,w),t.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}t.setPipeline(Ie),t.setBindGroup(0,vn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),t.end(),u=!1,t=i.beginComputePass();{let U=u?ze[o]:qe[o],w=u?Ne[o]:Ke[o],p=Se[o];t.setPipeline(p.dwPipeline),t.setBindGroup(0,U),t.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),t.setPipeline(p.pwPipeline),t.setBindGroup(0,w),t.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}for(t.setPipeline(Kt),t.setBindGroup(0,Pn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),t.setPipeline(Ie),t.setBindGroup(0,Bn),t.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),t.end(),u=!1,t=i.beginComputePass();o<oa;o++){let U=u?ze[o]:qe[o],w=u?Ne[o]:Ke[o],p=Se[o];t.setPipeline(p.dwPipeline),t.setBindGroup(0,U),t.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),t.setPipeline(p.pwPipeline),t.setBindGroup(0,w),t.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}for(;o<ft.length;o++){let U=o-oa,w=u?Aa[U]:Sa[U],p=ft[o];t.setPipeline(ot),t.setBindGroup(0,w),t.dispatchWorkgroups(p.outW,p.outH,1),u=!u}t.setPipeline(vt),t.setBindGroup(0,Cn),t.dispatchWorkgroups(1),t.end(),c&&i.copyBufferToBuffer(ne,0,c,0,260)}async function Qt(i){e.queue.writeBuffer(ut,0,i);let c=e.createCommandEncoder();{let t=c.beginComputePass();t.setPipeline(xt),t.setBindGroup(0,Ua),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}$e(c,ae),e.queue.submit([c.finish()]);let u=ae.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(ae.getMappedRange());return Zt[0]=o[0],jt[0]=o[1],Jt.set(o.subarray(2,65)),ae.unmap(),{handflag:new Float32Array(Zt),handedness:new Float32Array(jt),landmarks:new Float32Array(Jt)}}async function ca(i){e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);let c=e.createCommandEncoder();{let t=c.beginComputePass();t.setPipeline(De),t.setBindGroup(0,Je),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}$e(c,ae),e.queue.submit([c.finish()]);let u=ae.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(ae.getMappedRange());return Zt[0]=o[0],jt[0]=o[1],Jt.set(o.subarray(2,65)),ae.unmap(),{handflag:new Float32Array(Zt),handedness:new Float32Array(jt),landmarks:new Float32Array(Jt)}}async function Ma(i){if(!Vt||!pa||!Xt)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);let c=e.createCommandEncoder();{let U=c.beginComputePass();U.setPipeline(De),U.setBindGroup(0,Je),U.dispatchWorkgroups(16,16,1),U.end()}$e(c,null);let u=Xt.getCurrentTexture(),o=c.beginRenderPass({colorAttachments:[{view:u.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});o.setPipeline(Vt),o.setBindGroup(0,pa),o.draw(3),o.end(),e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),ua.drawImage(Da,0,0);let H=ua.getImageData(0,0,9,8).data,M=new Float32Array(65),R=new DataView(new ArrayBuffer(4));for(let U=0;U<65;U++){let w=U*4;R.setUint8(0,H[w]),R.setUint8(1,H[w+1]),R.setUint8(2,H[w+2]),R.setUint8(3,H[w+3]),M[U]=R.getFloat32(0)}return{handflag:new Float32Array([M[0]]),handedness:new Float32Array([M[1]]),landmarks:new Float32Array(M.subarray(2,65))}}let kn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),da=0,Un=[ae,kn],Ct=null,Qe=null;async function la(i){let c=Un[da];da=1-da,e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);let u=e.createCommandEncoder();{let t=u.beginComputePass();t.setPipeline(De),t.setBindGroup(0,Je),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}$e(u,c),e.queue.submit([u.finish()]);let o=null;if(Ct!==null&&Qe!==null){await Ct;let t=new Float32Array(Qe.getMappedRange());o={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))},Qe.unmap()}return Qe=c,Ct=c.mapAsync(GPUMapMode.READ),o}async function Ea(){if(!Ct||!Qe)return null;await Ct;let i=new Float32Array(Qe.getMappedRange()),c={handflag:new Float32Array([i[0]]),handedness:new Float32Array([i[1]]),landmarks:new Float32Array(i.subarray(2,65))};return Qe.unmap(),Ct=null,Qe=null,c}async function Gn(i=50){let c=new Float32Array(196608);for(let t=0;t<5;t++)await Qt(c);let u=[];for(let t=0;t<i;t++){let H=performance.now();await Qt(c),u.push(performance.now()-H)}let o=u.reduce((t,H)=>t+H,0)/u.length;return{avgMs:o,fps:1e3/o}}async function An(i=50){let c=new Float32Array(196608);for(let M=0;M<5;M++)await Qt(c);let u=[];for(let M=0;M<i;M++){let R=e.createCommandEncoder();{let w=R.beginComputePass();w.setPipeline(xt),w.setBindGroup(0,Ua),w.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),w.end()}$e(R,ae);let U=performance.now();e.queue.submit([R.finish()]),await e.queue.onSubmittedWorkDone(),u.push(performance.now()-U)}u.sort((M,R)=>M-R);let o=u.reduce((M,R)=>M+R,0)/u.length,t=u[Math.floor(u.length/2)],H=u[0];return{avgMs:o,fps:1e3/o,medianMs:t,minMs:H}}function ei(i){e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);let c=e.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(De),u.setBindGroup(0,Je),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),u.end()}$e(c,ae),e.queue.submit([c.finish()])}async function Sn(i,c=50){function u(w){let p=[...w].sort((q,se)=>q-se);return{median:p[Math.floor(p.length/2)],min:p[0]}}for(let w=0;w<10;w++)await ca(i);let o=[];for(let w=0;w<c;w++){e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);let p=e.createCommandEncoder();{let se=p.beginComputePass();se.setPipeline(De),se.setBindGroup(0,Je),se.dispatchWorkgroups(16,16,1),se.end()}$e(p,ae);let q=performance.now();e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-q)}let t=[];for(let w=0;w<c;w++){e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);let p=e.createCommandEncoder();{let le=p.beginComputePass();le.setPipeline(De),le.setBindGroup(0,Je),le.dispatchWorkgroups(16,16,1),le.end()}$e(p,ae),e.queue.submit([p.finish()]);let q=ae.mapAsync(GPUMapMode.READ),se=performance.now();await e.queue.onSubmittedWorkDone(),await q,ae.getMappedRange(),ae.unmap(),t.push(performance.now()-se)}let H=[];for(let w=0;w<c;w++){e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);let p=e.createCommandEncoder();{let se=p.beginComputePass();se.setPipeline(De),se.setBindGroup(0,Je),se.dispatchWorkgroups(16,16,1),se.end()}$e(p,ae),e.queue.submit([p.finish()]);let q=performance.now();await ae.mapAsync(GPUMapMode.READ),ae.getMappedRange(),ae.unmap(),H.push(performance.now()-q)}let M=[];for(let w=0;w<c;w++){let p=performance.now();await ca(i),M.push(performance.now()-p)}await la(i);let R=[];for(let w=0;w<c;w++){let p=performance.now();await la(i),R.push(performance.now()-p)}await Ea();let U=null;if(Vt){let w=[];for(let p=0;p<c;p++){let q=performance.now();await Ma(i),w.push(performance.now()-q)}U=u(w)}return{gpuOnly:u(o),mapAsyncOnly:u(t),mapAsyncNoWait:u(H),total:u(M),pipelined:u(R),renderReadback:U}}async function Dn(i){let c=[];async function u(t,H,M){let R=e.createBuffer({size:H,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),U=e.createCommandEncoder();U.copyBufferToBuffer(t,0,R,0,H),e.queue.submit([U.finish()]),await e.queue.onSubmittedWorkDone(),await R.mapAsync(GPUMapMode.READ);let w=new Float32Array(R.getMappedRange()),p=1/0,q=-1/0,se=0;for(let le=0;le<w.length;le++)w[le]<p&&(p=w[le]),w[le]>q&&(q=w[le]),w[le]!==0&&se++;R.unmap(),R.destroy(),c.push({layer:M,stats:{min:p,max:q,nonZero:se,total:w.length}})}e.queue.copyExternalImageToTexture({source:i},{texture:pe},[256,256]);{let t=e.createCommandEncoder(),H=t.beginComputePass();H.setPipeline(De),H.setBindGroup(0,Je),H.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),H.end(),e.queue.submit([t.finish()])}await u(Re,Math.min(Re.size,3*257*257*4),"canvas\u2192bufInput");{let t=e.createCommandEncoder(),H=t.beginComputePass();H.setPipeline(je),H.setBindGroup(0,Ga),H.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),H.end(),e.queue.submit([t.finish()])}await u(Q,Math.min(Q.size,3072*128*4),"inputConv\u2192bufA");let o=!0;for(let t=0;t<Math.min(ft.length,6);t++){let H=o?ze[t]:qe[t],M=o?Ne[t]:Ke[t],R=Se[t],U=ft[t];{let p=e.createCommandEncoder(),q=p.beginComputePass();q.setPipeline(R.dwPipeline),q.setBindGroup(0,H),q.dispatchWorkgroups(R.dwDispatchX,R.dwDispatchY,R.dwDispatchZ),q.end(),e.queue.submit([p.finish()])}await u(Me,Math.min(Me.size,U.spec.inCh*U.outH*U.outW*4),`layer${t}.DW\u2192bufDW (${U.spec.prefix})`);{let p=e.createCommandEncoder(),q=p.beginComputePass();q.setPipeline(R.pwPipeline),q.setBindGroup(0,M),q.dispatchWorkgroups(R.pwDispatchX,R.pwDispatchY,R.pwDispatchZ),q.end(),e.queue.submit([p.finish()])}let w=o?fe:Q;await u(w,Math.min(w.size,U.spec.outCh*U.outH*U.outW*4),`layer${t}.PW\u2192buf${o?"B":"A"} (${U.spec.prefix})`),o=!o}return c}return{device:e,run:Qt,runFromCanvas:ca,runFromCanvasViaRender:Ma,runFromCanvasPipelined:la,flushPipelined:Ea,benchmark:Gn,benchmarkGPU:An,benchmarkDiagnostic:Sn,debugLayerOutputs:Dn}}function mt(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var sn=mt(`
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
`),on=mt(`
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
`),un=mt(`
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
`),pn=mt(`
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
`),cn=mt(`
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
`),dn=mt(`
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
`),ln=mt(`
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
`);async function wa(s,_){let a;if(_)a=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");a=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8)}})}let B={r:"read-only-storage",s:"storage",u:"uniform"};function x(d){return a.createBindGroupLayout({entries:d.map((n,g)=>n==="t"?{binding:g,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:g,visibility:GPUShaderStage.COMPUTE,buffer:{type:B[n]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,T=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,L=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(d,n){return a.createBuffer({size:Math.max(d,4),usage:n})}function b(d,n,g){a.queue.writeBuffer(d,n,g)}function C(d){let n=l(d.data.byteLength,e);return b(n,0,d.data),n}let D=Array.from(s.keys());function K(d){let n=s.get(d);if(!n)throw new Error(`Weight not found: ${d}`);return n}function m(...d){let n=D.find(g=>d.every(S=>g.includes(S)));if(!n)throw new Error(`Weight not found for: ${d.join(", ")}`);return K(n)}function Y(d){let[,n,g,S]=d.shape,h=new Float32Array(S*25);for(let r=0;r<S;r++)for(let I=0;I<n;I++)for(let ie=0;ie<g;ie++)h[r*25+I*5+ie]=d.data[I*g*S+ie*S+r];return h}function G(d){let[n,,,g]=d.shape,S=new Float32Array(n*g);for(let h=0;h<n;h++)for(let r=0;r<g;r++)S[h*g+r]=d.data[h*g+r];return S}let ce=a.createShaderModule({code:sn}),re=a.createShaderModule({code:on}),ge=a.createShaderModule({code:un}),oe=a.createShaderModule({code:pn}),W=a.createShaderModule({code:dn}),ee=a.createShaderModule({code:cn}),ue=a.createShaderModule({code:ln}),O=x(["r","r","r","r","s","u"]),X=x(["r","r","r","s","u"]),j=x(["r","r","r","r","r","s","u"]),We=x(["r","r","r","s","u"]),xe=x(["r","r","r","r","s","u"]),me=x(["r","r","s","u"]),ye=x(["t","s","u"]);function Ce(d,n){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:n,entryPoint:"main"}})}let k=Ce(O,ce),V=Ce(X,re),he=Ce(j,ge),te=Ce(We,oe),_e=Ce(xe,W),J=Ce(me,ee),Xe=Ce(ye,ue),tt=m("conv2d/Conv2D"),ke=m("batch_normalization/","conv2d/Conv2D"),Ue=m("p_re_lu/"),ve=C(tt),Pe=C(ke),He=C(Ue),Oe=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(d=>{let n=m(d.dwKey),g=m(d.pwKey),S=m(d.bnKey),h=m(d.preluKey),r=Y(n),I=l(r.byteLength,e);b(I,0,r);let ie=new Float32Array(d.inCh),f=l(ie.byteLength,e);b(f,0,ie);let A=G(g),y=l(A.byteLength,e);b(y,0,A);let N=C(S),E=C(h);return{dwWeightBuf:I,dwBiasBuf:f,pwWeightBuf:y,pwBiasBuf:N,alphaBuf:E,inCh:d.inCh,outCh:d.outCh,stride:d.stride,inH:d.inH}}),Te=G(m("conv2d_20/Conv2D")),Le=l(Te.byteLength,e);b(Le,0,Te);let Ae=C(m("batch_normalization_20/")),Ve=C(m("p_re_lu_20/")),at={dwWeightBuf:(()=>{let d=Y(m("depthwise_conv2d_19/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=G(m("conv2d_21/")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:C(m("batch_normalization_21/")),alphaBuf:C(m("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},Ze={dwWeightBuf:(()=>{let d=Y(m("depthwise_conv2d_20/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=G(m("conv2d_22/Conv2D1")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:C(m("batch_normalization_22/")),alphaBuf:C(m("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},nt=G(m("conv2d_23/Conv2D")),ht=l(nt.byteLength,e);b(ht,0,nt);let it=C(m("batch_normalization_23/")),rt=C(m("p_re_lu_23/")),Ft={dwWeightBuf:(()=>{let d=Y(m("depthwise_conv2d_21/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=G(m("conv2d_24/")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:C(m("batch_normalization_24/")),alphaBuf:C(m("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},zt={dwWeightBuf:(()=>{let d=Y(m("depthwise_conv2d_22/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=G(m("conv2d_25/Conv2D1")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:C(m("batch_normalization_25/")),alphaBuf:C(m("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},Nt=G(m("classifier_palm_16_NO_PRUNING/Conv2D")),wt=l(Nt.byteLength,e);b(wt,0,Nt);let bt=C(m("classifier_palm_16_NO_PRUNING/BiasAdd")),Mt=G(m("regressor_palm_16_NO_PRUNING/Conv2D")),gt=l(Mt.byteLength,e);b(gt,0,Mt);let Et=C(m("regressor_palm_16_NO_PRUNING/BiasAdd")),st=G(m("classifier_palm_8_NO_PRUNING/Conv2D")),yt=l(st.byteLength,e);b(yt,0,st);let Wt=C(m("classifier_palm_8_NO_PRUNING/BiasAdd")),qt=G(m("regressor_palm_8_NO_PRUNING/Conv2D")),Ht=l(qt.byteLength,e);b(Ht,0,qt);let Se=C(m("regressor_palm_8_NO_PRUNING/BiasAdd")),xt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,je=l(36864*3*4,e),Ie=l(xt,v),Kt=l(xt,v),De=l(xt,v),vt=l(576*256*4,v),ot=l(144*256*4,v|GPUBufferUsage.COPY_DST),Fe=l(576*128*4,v|GPUBufferUsage.COPY_DST),ut=l(864*4,T),Re=l(15552*4,T),pt=l(576*2*4,T),Q=l(576*36*4,T),fe=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Me=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),dt=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),lt=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function F(d,n){return Math.ceil(d/n)}function ne(d){let n=l(d.byteLength,L);return b(n,0,d),n}let ae=ne(new Uint32Array([1,3,32,192,192,96,96]));function pe(d,n,g,S,h){let r=n.stride===2?n.inH/2:n.inH,I=r,ie=n.stride===2?1:2,f=ne(new Uint32Array([1,n.inCh,n.inH,n.inH,r,I,n.stride,ie])),A=a.createBindGroup({layout:X,entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:n.dwWeightBuf}},{binding:2,resource:{buffer:n.dwBiasBuf}},{binding:3,resource:{buffer:De}},{binding:4,resource:{buffer:f}}]}),y=d.beginComputePass();y.setPipeline(V),y.setBindGroup(0,A),y.dispatchWorkgroups(F(I,8),F(r,8),n.inCh),y.end();let N=n.inCh,E=ne(new Uint32Array([1,n.inCh,n.outCh,r,I,N,n.stride,n.inH,n.inH])),be=a.createBindGroup({layout:j,entries:[{binding:0,resource:{buffer:De}},{binding:1,resource:{buffer:h}},{binding:2,resource:{buffer:n.pwWeightBuf}},{binding:3,resource:{buffer:n.pwBiasBuf}},{binding:4,resource:{buffer:n.alphaBuf}},{binding:5,resource:{buffer:S}},{binding:6,resource:{buffer:E}}]}),de=d.beginComputePass();de.setPipeline(he),de.setBindGroup(0,be),de.dispatchWorkgroups(F(I,8),F(r,8),n.outCh),de.end()}function Ee(d,n,g,S,h,r,I,ie,f){let A=ne(new Uint32Array([1,r,I,ie,f])),y=a.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:g}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:h}},{binding:4,resource:{buffer:A}}]}),N=d.beginComputePass();N.setPipeline(te),N.setBindGroup(0,y),N.dispatchWorkgroups(F(f,8),F(ie,8),I),N.end()}function _t(d,n,g,S,h,r,I,ie,f,A){let y=ne(new Uint32Array([1,I,ie,f,A])),N=a.createBindGroup({layout:xe,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:g}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:h}},{binding:4,resource:{buffer:r}},{binding:5,resource:{buffer:y}}]}),E=d.beginComputePass();E.setPipeline(_e),E.setBindGroup(0,N),E.dispatchWorkgroups(F(A,8),F(f,8),ie),E.end()}async function $t(d){a.queue.copyExternalImageToTexture({source:d},{texture:lt},[192,192]);let n=ne(new Uint32Array([192,192,192])),g=a.createBindGroup({layout:ye,entries:[{binding:0,resource:lt.createView()},{binding:1,resource:{buffer:je}},{binding:2,resource:{buffer:n}}]}),S=a.createCommandEncoder();{let P=S.beginComputePass();P.setPipeline(Xe),P.setBindGroup(0,g),P.dispatchWorkgroups(F(192,16),F(192,16),1),P.end()}{let P=a.createBindGroup({layout:O,entries:[{binding:0,resource:{buffer:je}},{binding:1,resource:{buffer:ve}},{binding:2,resource:{buffer:Pe}},{binding:3,resource:{buffer:He}},{binding:4,resource:{buffer:Ie}},{binding:5,resource:{buffer:ae}}]}),Z=S.beginComputePass();Z.setPipeline(k),Z.setBindGroup(0,P),Z.dispatchWorkgroups(F(96,8),F(96,8),32),Z.end()}let h=Ie,r=Kt;for(let P=0;P<Oe.length;P++){let Z=Oe[P];pe(S,Z,h,r,h);let z=h;h=r,r=z,P===10&&S.copyBufferToBuffer(h,0,Fe,0,576*128*4),P===14&&S.copyBufferToBuffer(h,0,ot,0,144*256*4)}{let P=ne(new Uint32Array([1,256,6,6,12,12])),Z=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:P}}]}),z=S.beginComputePass();z.setPipeline(J),z.setBindGroup(0,Z),z.dispatchWorkgroups(F(12,8),F(12,8),256),z.end()}{let P=h;h=r,r=P}_t(S,h,Le,Ae,Ve,r,256,256,12,12);{let P=h;h=r,r=P}{let P=ne(new Uint32Array([1,256,12,12,12,12])),Z=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:ot}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:P}}]}),z=S.beginComputePass();z.setPipeline(J),z.setBindGroup(0,Z),z.dispatchWorkgroups(F(12,8),F(12,8),256),z.end()}{let P=h;h=r,r=P}pe(S,at,h,r,h);{let P=h;h=r,r=P}pe(S,Ze,h,r,h);{let P=h;h=r,r=P}Ee(S,h,wt,bt,ut,256,6,12,12),Ee(S,h,gt,Et,Re,256,108,12,12);{let P=ne(new Uint32Array([1,256,12,12,24,24])),Z=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:P}}]}),z=S.beginComputePass();z.setPipeline(J),z.setBindGroup(0,Z),z.dispatchWorkgroups(F(24,8),F(24,8),256),z.end()}{let P=h;h=r,r=P}_t(S,h,ht,it,rt,r,256,128,24,24);{let P=h;h=r,r=P}{let P=ne(new Uint32Array([1,128,24,24,24,24])),Z=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:P}}]}),z=S.beginComputePass();z.setPipeline(J),z.setBindGroup(0,Z),z.dispatchWorkgroups(F(24,8),F(24,8),128),z.end()}{let P=h;h=r,r=P}pe(S,Ft,h,r,h);{let P=h;h=r,r=P}pe(S,zt,h,r,h);{let P=h;h=r,r=P}Ee(S,h,yt,Wt,pt,128,2,24,24),Ee(S,h,Ht,Se,Q,128,36,24,24),a.queue.submit([S.finish()]);let I=a.createCommandEncoder();I.copyBufferToBuffer(ut,0,fe,0,864*4),I.copyBufferToBuffer(Re,0,Me,0,15552*4),I.copyBufferToBuffer(pt,0,ct,0,576*2*4),I.copyBufferToBuffer(Q,0,dt,0,576*36*4),a.queue.submit([I.finish()]),await Promise.all([fe.mapAsync(GPUMapMode.READ),Me.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ),dt.mapAsync(GPUMapMode.READ)]);let ie=new Float32Array(fe.getMappedRange()).slice(),f=new Float32Array(Me.getMappedRange()).slice(),A=new Float32Array(ct.getMappedRange()).slice(),y=new Float32Array(dt.getMappedRange()).slice();fe.unmap(),Me.unmap(),ct.unmap(),dt.unmap();let N=2016,E=new Float32Array(N),be=new Float32Array(N*18),de=0;for(let P=0;P<12;P++)for(let Z=0;Z<12;Z++)for(let z=0;z<6;z++){E[de]=ie[z*144+P*12+Z];for(let Be=0;Be<18;Be++){let Pt=z*18+Be;be[de*18+Be]=f[Pt*144+P*12+Z]}de++}for(let P=0;P<24;P++)for(let Z=0;Z<24;Z++)for(let z=0;z<2;z++){E[de]=A[z*576+P*24+Z];for(let Be=0;Be<18;Be++){let Pt=z*18+Be;be[de*18+Be]=y[Pt*576+P*24+Z]}de++}return{scores:E,regressors:be}}async function $(d,n){let g=a.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),S=a.createCommandEncoder();S.copyBufferToBuffer(d,0,g,0,n*4),a.queue.submit([S.finish()]),await g.mapAsync(GPUMapMode.READ);let h=new Float32Array(g.getMappedRange()).slice();return g.unmap(),g.destroy(),h}async function Yt(d){a.queue.copyExternalImageToTexture({source:d},{texture:lt},[192,192]);function n(y,N=1e3){let E=y.slice(0,N);return{min:Math.min(...E),max:Math.max(...E),mean:E.reduce((be,de)=>be+de,0)/E.length,nonZero:E.filter(be=>be!==0).length,sample:Array.from(E.slice(0,10))}}let g={},S=ne(new Uint32Array([192,192,192])),h=a.createBindGroup({layout:ye,entries:[{binding:0,resource:lt.createView()},{binding:1,resource:{buffer:je}},{binding:2,resource:{buffer:S}}]}),r=a.createCommandEncoder(),I=r.beginComputePass();I.setPipeline(Xe),I.setBindGroup(0,h),I.dispatchWorkgroups(F(192,16),F(192,16),1),I.end(),a.queue.submit([r.finish()]),g.input=n(await $(je,36864*3)),r=a.createCommandEncoder();let ie=a.createBindGroup({layout:O,entries:[{binding:0,resource:{buffer:je}},{binding:1,resource:{buffer:ve}},{binding:2,resource:{buffer:Pe}},{binding:3,resource:{buffer:He}},{binding:4,resource:{buffer:Ie}},{binding:5,resource:{buffer:ae}}]});I=r.beginComputePass(),I.setPipeline(k),I.setBindGroup(0,ie),I.dispatchWorkgroups(F(96,8),F(96,8),32),I.end(),a.queue.submit([r.finish()]),g.initConv=n(await $(Ie,9216*32));let f=Ie,A=Kt;for(let y=0;y<Oe.length;y++){let N=Oe[y];r=a.createCommandEncoder(),pe(r,N,f,A,f),a.queue.submit([r.finish()]);let E=f;if(f=A,A=E,y===0||y===3||y===7||y===11||y===14||y===15||y===18){let be=N.stride===2?N.inH/2:N.inH,de=be*be*N.outCh;g[`block${y}`]=n(await $(f,de))}y===10&&(r=a.createCommandEncoder(),r.copyBufferToBuffer(f,0,Fe,0,576*128*4),a.queue.submit([r.finish()])),y===14&&(r=a.createCommandEncoder(),r.copyBufferToBuffer(f,0,ot,0,144*256*4),a.queue.submit([r.finish()]))}r=a.createCommandEncoder();{let y=ne(new Uint32Array([1,256,6,6,12,12])),N=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:A}},{binding:3,resource:{buffer:y}}]}),E=r.beginComputePass();E.setPipeline(J),E.setBindGroup(0,N),E.dispatchWorkgroups(F(12,8),F(12,8),256),E.end()}a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpnUpsample6to12=n(await $(f,144*256)),r=a.createCommandEncoder(),_t(r,f,Le,Ae,Ve,A,256,256,12,12),a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpn6to12Conv=n(await $(f,144*256)),g.backbone12Skip=n(await $(ot,144*256)),r=a.createCommandEncoder();{let y=ne(new Uint32Array([1,256,12,12,12,12])),N=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:ot}},{binding:2,resource:{buffer:A}},{binding:3,resource:{buffer:y}}]}),E=r.beginComputePass();E.setPipeline(J),E.setBindGroup(0,N),E.dispatchWorkgroups(F(12,8),F(12,8),256),E.end()}a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpnAdd12=n(await $(f,144*256)),r=a.createCommandEncoder(),pe(r,at,f,A,f),a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpn12Block1=n(await $(f,144*256)),r=a.createCommandEncoder(),pe(r,Ze,f,A,f),a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpn12Block2=n(await $(f,144*256)),r=a.createCommandEncoder(),Ee(r,f,wt,bt,ut,256,6,12,12),a.queue.submit([r.finish()]),g.cls16=n(await $(ut,864)),r=a.createCommandEncoder(),Ee(r,f,gt,Et,Re,256,108,12,12),a.queue.submit([r.finish()]),g.reg16=n(await $(Re,15552),500),r=a.createCommandEncoder();{let y=ne(new Uint32Array([1,256,12,12,24,24])),N=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:A}},{binding:3,resource:{buffer:y}}]}),E=r.beginComputePass();E.setPipeline(J),E.setBindGroup(0,N),E.dispatchWorkgroups(F(24,8),F(24,8),256),E.end()}a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpnUpsample12to24=n(await $(f,576*256)),r=a.createCommandEncoder(),_t(r,f,ht,it,rt,A,256,128,24,24),a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpn12to24Conv=n(await $(f,576*128)),g.backbone24Skip=n(await $(Fe,576*128)),r=a.createCommandEncoder();{let y=ne(new Uint32Array([1,128,24,24,24,24])),N=a.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:A}},{binding:3,resource:{buffer:y}}]}),E=r.beginComputePass();E.setPipeline(J),E.setBindGroup(0,N),E.dispatchWorkgroups(F(24,8),F(24,8),128),E.end()}a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpnAdd24=n(await $(f,576*128)),r=a.createCommandEncoder(),pe(r,Ft,f,A,f),a.queue.submit([r.finish()]);{let y=f;f=A,A=y}g.fpn24Block1=n(await $(f,576*128)),r=a.createCommandEncoder(),pe(r,zt,f,A,f),a.queue.submit([r.finish()]);{let y=f;f=A,A=y}return g.fpn24Block2=n(await $(f,576*128)),r=a.createCommandEncoder(),Ee(r,f,yt,Wt,pt,128,2,24,24),a.queue.submit([r.finish()]),g.cls8=n(await $(pt,576*2)),r=a.createCommandEncoder(),Ee(r,f,Ht,Se,Q,128,36,24,24),a.queue.submit([r.finish()]),g.reg8=n(await $(Q,576*36)),g.initWeights=n(await $(ve,100),100),g.initBias=n(await $(Pe,32),32),g.cls16Weights=n(await $(wt,100),100),g.cls16Bias=n(await $(bt,6),6),g.cls8Weights=n(await $(yt,100),100),g.cls8Bias=n(await $(Wt,2),2),g.fpn6to12Weights=n(await $(Le,100),100),g}return{device:a,run:$t,debugRun:Yt}}function qn(){let s=[];for(let _=0;_<12;_++)for(let a=0;a<12;a++){let B=(a+.5)/12,x=(_+.5)/12;for(let e=0;e<6;e++)s.push({x:B,y:x})}for(let _=0;_<24;_++)for(let a=0;a<24;a++){let B=(a+.5)/24,x=(_+.5)/24;for(let e=0;e<2;e++)s.push({x:B,y:x})}return s}var _n=qn();function Kn(s){return 1/(1+Math.exp(-s))}function fn(s,_){let a=[],{scores:B,regressors:x}=s,e=192;for(let v=0;v<_n.length;v++){let T=Kn(B[v]);if(T<_)continue;let L=_n[v],l=v*18,b=L.x+x[l+0]/e,C=L.y+x[l+1]/e,D=x[l+2]/e,K=x[l+3]/e,m=[];for(let Y=0;Y<7;Y++){let G=L.x+x[l+4+Y*2]/e,ce=L.y+x[l+4+Y*2+1]/e;m.push([G,ce])}a.push({score:T,box:[b,C,D,K],keypoints:m})}return a}function mn(s,_){if(s.length===0)return[];let a=[...s].sort((e,v)=>v.score-e.score),B=[],x=new Set;for(let e=0;e<a.length;e++)if(!x.has(e)){B.push(a[e]);for(let v=e+1;v<a.length;v++)x.has(v)||$n(a[e],a[v])>_&&x.add(v)}return B}function $n(s,_){let a=s.box[0]-s.box[2]/2,B=s.box[1]-s.box[3]/2,x=s.box[0]+s.box[2]/2,e=s.box[1]+s.box[3]/2,v=_.box[0]-_.box[2]/2,T=_.box[1]-_.box[3]/2,L=_.box[0]+_.box[2]/2,l=_.box[1]+_.box[3]/2,b=Math.max(a,v),C=Math.max(B,T),D=Math.min(x,L),K=Math.min(e,l),m=Math.max(0,D-b),Y=Math.max(0,K-C),G=m*Y,ce=(x-a)*(e-B),re=(L-v)*(l-T),ge=ce+re-G;return ge>0?G/ge:0}function Yn(s){let[_,a,B,x]=s.box,e=s.keypoints[0],v=s.keypoints[2],T=v[0]-e[0],L=v[1]-e[1],l=Math.atan2(L,T),C=-Math.PI/2-l,D=Math.max(B,x),m=D*2.6,Y=-.5*D,G=Math.cos(C),ce=Math.sin(C),re=-Y*ce,ge=Y*G;return{centerX:_+re,centerY:a+ge,width:m,height:m,rotation:C}}function ba(s,_={}){let{scoreThreshold:a=.5,nmsThreshold:B=.3,maxHands:x=2}=_;async function e(T){let L=await s.run(T),l=fn(L,a);return mn(l,B).slice(0,x).map(Yn)}async function v(T){let L=await s.run(T),l=fn(L,a);return mn(l,B).slice(0,x)}return{detect:e,detectRaw:v,model:s}}function hn(s,_=256){let a=Math.cos(s.rotation),B=Math.sin(s.rotation),x=s.width/_,e=s.height/_,v=x*a,T=-e*B,L=x*B,l=e*a,b=s.centerX-(v*_/2+T*_/2),C=s.centerY-(L*_/2+l*_/2),D=v*l-T*L,K=l/D,m=-T/D,Y=-L/D,G=v/D,ce=-(K*b+m*C),re=-(Y*b+G*C);return{forward:[v,T,b,L,l,C],inverse:[K,m,ce,Y,G,re]}}function ga(s,_){let{forward:a}=hn(_,1),[B,x,e,v,T,L]=a;return s.map(l=>({x:B*l.x+x*l.y+e,y:v*l.x+T*l.y+L,z:l.z}))}var wn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Xn(s={}){let{weightsUrl:_,scoreThreshold:a=.5,forceF32:B=!1}=s;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let x=_??wn,e=x.endsWith("/")?x:`${x}/`,v=`${e}weights_f16.json`,T=`${e}weights_f16.bin`,[L,l]=await Promise.all([fetch(v),fetch(T)]);if(!L.ok)throw new Error(`Failed to fetch weights metadata: ${L.status}`);if(!l.ok)throw new Error(`Failed to fetch weights binary: ${l.status}`);let b=await L.json(),C=await l.arrayBuffer(),D=Ot(b,C),K=await It(D,{forceF32:B});if(!B){let O=new OffscreenCanvas(256,256),X=O.getContext("2d");X.fillStyle="#886644",X.fillRect(0,0,256,256),X.fillStyle="#cc9966",X.fillRect(50,50,156,156);let j=await K.runFromCanvas(O);j.landmarks.every(xe=>xe===0)&&j.handflag.every(xe=>xe===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),K.device.destroy(),K=await It(D,{forceF32:!0}))}let m=null;function Y(){return m||(m=new OffscreenCanvas(256,256)),m}async function G(O){if(O instanceof HTMLCanvasElement||O instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&O instanceof ImageBitmap)return O;let X=Y();X.width=256,X.height=256;let j=X.getContext("2d");return O instanceof ImageData?j.putImageData(O,0,0):j.drawImage(O,0,0,256,256),X}function ce(O,X,j){let We=O[0];if(We<a)return null;let xe=X[0]>.5,me=[];for(let ye=0;ye<21;ye++)me.push({x:j[ye*3],y:j[ye*3+1],z:j[ye*3+2]});return{score:We,handedness:xe?"right":"left",landmarks:me}}async function re(O){let X=await G(O),j=await K.runFromCanvas(X);return ce(j.handflag,j.handedness,j.landmarks)}async function ge(O){let X=await G(O),j=await K.runFromCanvasPipelined(X);return j?ce(j.handflag,j.handedness,j.landmarks):null}async function oe(){let O=await K.flushPipelined();return O?ce(O.handflag,O.handedness,O.landmarks):null}function W(){K.device.destroy(),m=null}async function ee(O){let X=await G(O);return K.benchmarkDiagnostic(X)}async function ue(O){let X=await G(O);return K.debugLayerOutputs(X)}return{detect:re,detectPipelined:ge,flushPipelined:oe,dispose:W,benchmarkDiagnostic:ee,debugLayerOutputs:ue}}async function Vn(s={}){let{weightsUrl:_,palmWeightsUrl:a,scoreThreshold:B=.5,palmScoreThreshold:x=.5,maxHands:e=2,forceF32:v=!1}=s;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let T=_??wn,L=T.endsWith("/")?T:`${T}/`;if(!a)throw new Error("palmWeightsUrl is required for createFullHandpose");let l=a.endsWith("/")?a:`${a}/`,[b,C,D,K]=await Promise.all([fetch(`${L}weights_f16.json`),fetch(`${L}weights_f16.bin`),fetch(`${l}palm_detection_weights.json`),fetch(`${l}palm_detection_weights.bin`)]);if(!b.ok)throw new Error(`Failed to fetch landmark weights metadata: ${b.status}`);if(!C.ok)throw new Error(`Failed to fetch landmark weights binary: ${C.status}`);if(!D.ok)throw new Error(`Failed to fetch palm weights metadata: ${D.status}`);if(!K.ok)throw new Error(`Failed to fetch palm weights binary: ${K.status}`);let[m,Y,G,ce]=await Promise.all([b.json(),C.arrayBuffer(),D.json(),K.arrayBuffer()]),re=Ot(m,Y),ge=Ot(G,ce),oe=await It(re,{forceF32:v});if(!v){let k=new OffscreenCanvas(256,256),V=k.getContext("2d");V.fillStyle="#886644",V.fillRect(0,0,256,256),V.fillStyle="#cc9966",V.fillRect(50,50,156,156);let he=await oe.runFromCanvas(k);he.landmarks.every(_e=>_e===0)&&he.handflag.every(_e=>_e===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),oe.device.destroy(),oe=await It(re,{forceF32:!0}))}let W=await wa(ge),ee=ba(W,{scoreThreshold:x,maxHands:e}),ue=null,O=null;function X(){return ue||(ue=new OffscreenCanvas(192,192)),ue}function j(){return O||(O=new OffscreenCanvas(256,256)),O}async function We(k){if(k instanceof HTMLCanvasElement||k instanceof OffscreenCanvas){if(k.width===192&&k.height===192)return k;let te=X();return te.width=192,te.height=192,te.getContext("2d").drawImage(k,0,0,192,192),te}if(typeof ImageBitmap<"u"&&k instanceof ImageBitmap){if(k.width===192&&k.height===192)return k;let te=X();return te.width=192,te.height=192,te.getContext("2d").drawImage(k,0,0,192,192),te}let V=X();V.width=192,V.height=192;let he=V.getContext("2d");if(k instanceof ImageData){let te=new OffscreenCanvas(k.width,k.height);te.getContext("2d").putImageData(k,0,0),he.drawImage(te,0,0,192,192)}else he.drawImage(k,0,0,192,192);return V}function xe(k,V,he,te){let _e=j();_e.width=256,_e.height=256;let J=_e.getContext("2d"),Xe=Math.cos(-V.rotation),tt=Math.sin(-V.rotation);J.clearRect(0,0,256,256),J.save(),J.translate(128,128),J.scale(V.width*he/256,V.height*te/256),J.rotate(-V.rotation),J.translate(-128,-128);let ke=V.centerX*he,Ue=V.centerY*te;J.restore();let ve=Math.min(he,te),Pe=256/(V.width*ve),He=256/(V.height*ve),Ge=Math.cos(V.rotation),Oe=Math.sin(V.rotation),Te=Ge*Pe,Le=Oe*Pe,Ae=-Oe*He,Ve=Ge*He,at=-ke*Te-Ue*Ae+128,Ze=-ke*Le-Ue*Ve+128;if(J.setTransform(Te,Le,Ae,Ve,at,Ze),k instanceof ImageData){let nt=new OffscreenCanvas(k.width,k.height);nt.getContext("2d").putImageData(k,0,0),J.drawImage(nt,0,0)}else J.drawImage(k,0,0);return J.setTransform(1,0,0,1,0,0),_e}function me(k){return k instanceof HTMLCanvasElement||k instanceof OffscreenCanvas?[k.width,k.height]:typeof ImageBitmap<"u"&&k instanceof ImageBitmap?[k.width,k.height]:k instanceof ImageData?[k.width,k.height]:k instanceof HTMLVideoElement?[k.videoWidth,k.videoHeight]:k instanceof HTMLImageElement?[k.naturalWidth,k.naturalHeight]:[256,256]}async function ye(k){let V=await We(k),he=await ee.detect(V);if(he.length===0)return[];let[te,_e]=me(k),J=[];for(let Xe of he){let tt=xe(k,Xe,te,_e),ke=await oe.runFromCanvas(tt),Ue=ke.handflag[0];if(Ue<B)continue;let ve=ke.handedness[0]>.5,Pe=[];for(let Ge=0;Ge<21;Ge++)Pe.push({x:ke.landmarks[Ge*3],y:ke.landmarks[Ge*3+1],z:ke.landmarks[Ge*3+2]});let He=ga(Pe,Xe);J.push({score:Ue,handedness:ve?"right":"left",landmarks:He,palmScore:0})}return J}function Ce(){oe.device.destroy(),W.device.destroy(),ue=null,O=null}return{detect:ye,dispose:Ce}}function Zn(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var jn=Zn(`
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
`);function Jn(s){let _=s.createShaderModule({code:jn}),a=s.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),B=s.createComputePipeline({layout:s.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:_,entryPoint:"main"}});function x(e,v,T,L,l,b,C){let D=new Uint32Array([l,b,C,0]),K=s.createBuffer({size:D.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});s.queue.writeBuffer(K,0,D);let m=new Float32Array(L),Y=new Float32Array(8);Y.set(m);let G=s.createBuffer({size:Y.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});s.queue.writeBuffer(G,0,Y);let ce=s.createBindGroup({layout:a,entries:[{binding:0,resource:v.createView()},{binding:1,resource:{buffer:T}},{binding:2,resource:{buffer:K}},{binding:3,resource:{buffer:G}}]}),re=e.beginComputePass();re.setPipeline(B),re.setBindGroup(0,ce),re.dispatchWorkgroups(Math.ceil(C/16),Math.ceil(C/16),1),re.end()}return{crop:x}}export{wa as compilePalmModel,hn as computeCropTransform,Jn as createCropPipeline,Vn as createFullHandpose,Xn as createHandpose,ba as createPalmDetector,Ot as loadWeightsFromBuffer,ga as projectLandmarksToOriginal};
