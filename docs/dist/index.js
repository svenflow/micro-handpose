function _e(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ta(n){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],i="enable f16;"+n;for(let y of _)for(;i.includes(`${y}:array<f32>`);)i=i.replace(`${y}:array<f32>`,`${y}:array<f16>`);return i}var la=_e(`
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
`),ma=_e(`
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
`),ha=_e(`
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
`);function La(n,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Ra(n,_){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Oa(n,_){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Ia(n,_){return ha.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Fa(n,_){return[8,8]}var za=_e(`
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
    var pw_sum=sum0+pw_bias[c];`,G=`var skip_val:f32=0.0;
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
    }`,A=n===_?"":`if(c<${n}u){`,d=n===_?"":"}",h=w?`for(var c:u32=lid.x;c<${n}u;c+=${y}u){`:`let c=lid.x;
  ${A}`,v=w?"}":d,k=w?`for(var c:u32=lid.x;c<${_}u;c+=${y}u){`:"{let c=lid.x;";return _e(`
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
  ${k}
    let pw_base=c*${n}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${g}
    // Skip connection (only for c < inCh)
    ${G}
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
`);function Mt(n,_){let i=new Map,y=n.dtype??"float32";for(let w=0;w<n.keys.length;w++){let e=n.keys[w],g=n.shapes[w],G=n.offsets[w],A=g.reduce((v,k)=>v*k,1),d,h;if(y==="float32")d=new Float32Array(_,G,A);else{let v=new DataView(_);d=new Float32Array(A);for(let k=0;k<A;k++)d[k]=Rn(v.getUint16(G+k*2,!0));h=_.slice(G,G+A*2)}i.set(e,{data:d,shape:g,rawF16:h})}return i}function Rn(n){let _=n>>15&1,i=n>>10&31,y=n&1023;if(i===0){if(y===0)return _?-0:0;let g=-14,G=y/1024;return(_?-1:1)*Math.pow(2,g)*G}if(i===31)return y===0?_?-1/0:1/0:NaN;let w=i-15,e=1+y/1024;return(_?-1:1)*Math.pow(2,w)*e}var On=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],rn=On.map(([n,_,i,y,w])=>({type:"resmodule",inCh:n,outCh:_,h:i,w:i,stride:y,prefix:w})),In=2,Fn=5,zn=8,Nn=11;async function Jt(n,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let i=await navigator.gpu.requestAdapter();if(!i)throw new Error("No GPU adapter found");let y=i.features.has("shader-f16"),w=y?["shader-f16"]:[],e=await i.requestDevice({requiredFeatures:w,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(i.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(i.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(i.limits.maxComputeInvocationsPerWorkgroup,288)}}),g=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(y)try{let a=`enable f16;
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
}`,o=e.createShaderModule({code:a}),s=e.createShaderModule({code:p}),t=await o.getCompilationInfo(),D=await s.getCompilationInfo();if(t.messages.some(U=>U.type==="error")||D.messages.some(U=>U.type==="error"))g=!1;else{let U=new Float32Array(2400);U.fill(1);let E=new Uint16Array(2400);E.fill(10516);let x=new Uint16Array(96);x.fill(14336);let f=new Uint16Array(9216);f.fill(8478);let u=new Uint16Array(96);u.fill(12288);let I=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,te=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,re=e.createBuffer({size:U.byteLength,usage:I}),gt=e.createBuffer({size:E.byteLength,usage:I}),bt=e.createBuffer({size:x.byteLength,usage:I}),yt=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),xt=e.createBuffer({size:f.byteLength,usage:I}),vt=e.createBuffer({size:u.byteLength,usage:I}),Pt=e.createBuffer({size:384,usage:te}),Ze=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(re,0,U),e.queue.writeBuffer(gt,0,E),e.queue.writeBuffer(bt,0,x),e.queue.writeBuffer(xt,0,f),e.queue.writeBuffer(vt,0,u);let $e="read-only-storage",At="storage",St=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:At}}]}),Wa=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:At}}]}),Mn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[St]}),compute:{module:o,entryPoint:"main"}}),En=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Wa]}),compute:{module:s,entryPoint:"main"}}),Wn=e.createBindGroup({layout:St,entries:[{binding:0,resource:{buffer:re}},{binding:1,resource:{buffer:gt}},{binding:2,resource:{buffer:bt}},{binding:3,resource:{buffer:yt}}]}),Hn=e.createBindGroup({layout:Wa,entries:[{binding:0,resource:{buffer:yt}},{binding:1,resource:{buffer:xt}},{binding:2,resource:{buffer:vt}},{binding:3,resource:{buffer:Pt}}]}),Vt=e.createCommandEncoder(),jt=Vt.beginComputePass();jt.setPipeline(Mn),jt.setBindGroup(0,Wn),jt.dispatchWorkgroups(2),jt.end();let Zt=Vt.beginComputePass();Zt.setPipeline(En),Zt.setBindGroup(0,Hn),Zt.dispatchWorkgroups(2),Zt.end(),Vt.copyBufferToBuffer(Pt,0,Ze,0,384),e.queue.submit([Vt.finish()]),await e.queue.onSubmittedWorkDone(),await Ze.mapAsync(GPUMapMode.READ);let Dt=new Float32Array(Ze.getMappedRange()),Ha=1.5*.0104*96+.25,Tn=Dt[0]!==0&&Dt[47]!==0&&Dt[95]!==0,Ln=Math.abs(Dt[0]-Ha)<1;g=Tn&&Ln,Ze.unmap(),re.destroy(),gt.destroy(),bt.destroy(),yt.destroy(),xt.destroy(),vt.destroy(),Pt.destroy(),Ze.destroy(),g||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Dt[0]}, expected ~${Ha.toFixed(2)}) \u2014 falling back to f32`)}}catch{g=!1}let A=n.values().next().value,d=g&&!!A?.rawF16&&!_?.forceF32;console.log(d?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${y}, f16 validated: ${g}, f16 data: ${!!A?.rawF16})`);function h(a){if(d&&a.rawF16){let p=new Uint16Array(a.rawF16);if(p.length%2!==0){let o=new Uint16Array(p.length+1);return o.set(p),o}return p}return a.data}function v(a){if(d&&a.rawF16){let p=a.rawF16.byteLength;return Math.ceil(p/4)*4}return a.data.byteLength}function k(a){return d?Ta(a):a}let L={r:"read-only-storage",s:"storage",u:"uniform"};function m(a){return e.createBindGroupLayout({entries:a.map((p,o)=>({binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:L[p]}}))})}function z(a){return e.createBindGroupLayout({entries:a.map((p,o)=>p==="t"?{binding:o,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:L[p]}})})}let B=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,pe=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,se=GPUBufferUsage.STORAGE,Pe=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ce=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function S(a,p){return e.createBuffer({size:a,usage:p})}function Y(a,p){return e.createBindGroup({layout:a,entries:p.map((o,s)=>({binding:s,resource:"size"in o?{buffer:o}:o}))})}function ae(a,p){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:p,entryPoint:"main"}})}let H=e.createShaderModule({code:za}),F=e.createShaderModule({code:nn}),N=e.createShaderModule({code:k(Ja)}),Se=e.createShaderModule({code:k(ma)}),be=e.createShaderModule({code:k(la)}),Te=e.createShaderModule({code:k(fa)}),he=e.createShaderModule({code:k(ha)}),Be=e.createShaderModule({code:k(Na)}),P=e.createShaderModule({code:qa}),V=e.createShaderModule({code:Ya}),we=e.createShaderModule({code:$a}),j=e.createShaderModule({code:k(Xa)}),ge=e.createShaderModule({code:k(ja)}),oe=e.createShaderModule({code:k(Za)}),Ye=e.createShaderModule({code:k(Qa)}),Je=new Map;function Ce(a,p){let o=`${a}_${p}`,s=Je.get(o);return s||(s=e.createShaderModule({code:k(en(a,p))}),Je.set(o,s)),s}let ke=m(["r","r","r","s","u"]),ye=m(["r","r","r","r","s","u"]),xe=m(["r","s","u"]),De=m(["r","r","r","s","u"]),Ue=m(["r","s","u"]),Le=m(["r","r","s","u"]),Me=m(["r","r","s","u"]),Xe=m(["r","r","r","s","u"]),Ee=m(["r","r","r","s","u"]),ut=z(["t","s","u"]),Qe=m(["r","r","r","r","r","r","r","s"]),Re=m(["r","r","r","r","r","s","u"]),Bt=e.createPipelineLayout({bindGroupLayouts:[ke]}),Et=e.createPipelineLayout({bindGroupLayouts:[ye]}),pt=a=>e.createComputePipeline({layout:Bt,compute:{module:a,entryPoint:"main"}}),ct=a=>e.createComputePipeline({layout:Et,compute:{module:a,entryPoint:"main"}}),Qt=pt(Se),ea=pt(be),Wt=ct(Te),dt=ct(he),_t=new Map,Ct=new Map,lt=new Map,kt=new Map;_t.set("8,8",Qt),Ct.set("8,8",ea),lt.set("8,8",Wt),kt.set("8,8",dt);function et(a,p,o,s,t){let D=`${p},${o}`,U=a.get(D);return U||(U=t(e.createShaderModule({code:k(s(p,o))})),a.set(D,U)),U}let Ht=(a,p)=>et(_t,a,p,La,pt),ta=(a,p)=>et(Ct,a,p,Ra,pt),Tt=(a,p)=>et(lt,a,p,Oa,ct),Lt=(a,p)=>et(kt,a,p,Ia,ct),We=rn.map(a=>{let p=a.stride===2?a.h/2:a.h,o=a.stride===2?a.w/2:a.w,[s,t]=Fa(a.inCh,p),D=a.h>=64,U=p>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:D?ta(s,t):Ht(s,t),pwPipeline:U?Lt(s,t):Tt(s,t),dwDispatchX:Math.ceil(o/s),dwDispatchY:Math.ceil(p/t),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(o/s),pwDispatchY:Math.ceil(p/t),pwDispatchZ:U?a.outCh/2:a.outCh}}),aa=ae(xe,H),mt=ae(De,Be);ae(Ue,P),ae(Le,V);let Oe=ae(Me,we),tt=ae(Xe,j);ae(Ee,ge),ae(Ee,oe);let Ge=ae(ut,F),Rt=ae(Qe,N),Ut=ae(Re,Ye),Ot=1*288*128*128*4,at=S(3*256*256*4,B),He=S(3*257*257*4,se),Gt=S(12,ce);e.queue.writeBuffer(Gt,0,new Uint32Array([3,256,257]));let Z=S(Ot,pe),de=S(Ot,Pe),Ae=S(Ot,se),nt=S(3072*64*4,B),it=S(3072*32*4,B),rt=S(1536*16*4,B),Q=S(6144*64*4,se),le=S(260,Pe),X=S(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);S(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let ue=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ie=S(8,ce);e.queue.writeBuffer(Ie,0,new Uint32Array([256,257]));let It=n.get("backbone1.1.weight"),Ft=n.get("backbone1.1.bias");if(!It||!Ft)throw new Error("Missing input conv weights");let fe=h(It),zt=h(Ft),c=S(fe.byteLength,B),r=S(zt.byteLength,B),C=S(28,ce);e.queue.writeBuffer(c,0,fe),e.queue.writeBuffer(r,0,zt),e.queue.writeBuffer(C,0,new Uint32Array([1,3,24,257,257,128,128]));let M=n.get("backbone6.1.weight"),b=n.get("backbone6.1.bias");if(!M||!b)throw new Error("Missing backbone6.1 conv1x1 weights");let l=h(M),T=h(b),J=S(l.byteLength,B),R=S(T.byteLength,B),K=S(20,ce);e.queue.writeBuffer(J,0,l),e.queue.writeBuffer(R,0,T),e.queue.writeBuffer(K,0,new Uint32Array([1,96,48,32,32]));let O=n.get("handflag.weight"),q=n.get("handflag.bias");if(!O||!q)throw new Error("Missing handflag weights");let $=h(O),me=h(q),ne=S($.byteLength,B),W=S(me.byteLength,B),ee=S(12,ce);e.queue.writeBuffer(ne,0,$),e.queue.writeBuffer(W,0,me),e.queue.writeBuffer(ee,0,new Uint32Array([1,288,1]));let ie=n.get("handedness.weight"),ve=n.get("handedness.bias");if(!ie||!ve)throw new Error("Missing handedness weights");let ft=h(ie),ya=h(ve),na=S(ft.byteLength,B),ia=S(ya.byteLength,B),xa=S(12,ce);e.queue.writeBuffer(na,0,ft),e.queue.writeBuffer(ia,0,ya),e.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let va=n.get("reg_3d.weight"),Pa=n.get("reg_3d.bias");if(!va||!Pa)throw new Error("Missing reg_3d weights");let Ba=h(va),Ca=h(Pa),ra=S(Ba.byteLength,B),sa=S(Ca.byteLength,B),ka=S(12,ce);e.queue.writeBuffer(ra,0,Ba),e.queue.writeBuffer(sa,0,Ca),e.queue.writeBuffer(ka,0,new Uint32Array([1,288,63]));let st=rn.map(a=>{let{inCh:p,outCh:o,h:s,w:t,stride:D,prefix:U}=a,E=D===2?s/2:s,x=D===2?t/2:t,f=D===2?1:2,u=n.get(`${U}convs.0.weight`),I=n.get(`${U}convs.0.bias`),te=n.get(`${U}convs.1.weight`),re=n.get(`${U}convs.1.bias`);if(!u||!I||!te||!re)throw new Error(`Missing weights for ${U}`);let gt=h(u),bt=h(I),yt=h(te),xt=h(re),vt=S(gt.byteLength,B),Pt=S(bt.byteLength,B),Ze=S(yt.byteLength,B),$e=S(xt.byteLength,B),At=S(32,ce),St=S(36,ce);return e.queue.writeBuffer(vt,0,gt),e.queue.writeBuffer(Pt,0,bt),e.queue.writeBuffer(Ze,0,yt),e.queue.writeBuffer($e,0,xt),e.queue.writeBuffer(At,0,new Uint32Array([1,p,s,t,E,x,D,f])),e.queue.writeBuffer(St,0,new Uint32Array([1,p,o,E,x,Math.max(0,o-p),D,s,t])),{dwWeight:vt,dwBias:Pt,pwWeight:Ze,pwBias:$e,dwUniform:At,pwUniform:St,spec:a,outH:E,outW:x}});function ht(a){let p=S(a.length*4,ce);return e.queue.writeBuffer(p,0,new Uint32Array(a)),p}let gn=ht([1,96,8,8,16,16]),bn=ht([1,96,16,16,32,32]),yn=ht([1,48,32,32,64,64]);ht([1536*16]),ht([3072*32]),ht([3072*64]);let Ua=Y(xe,[at,He,Gt]),Ga=Y(De,[He,c,r,Z,C]),Fe=[],ze=[],Ne=[],Ke=[];for(let a of st)Fe.push(Y(ke,[Z,a.dwWeight,a.dwBias,Ae,a.dwUniform])),ze.push(Y(ye,[Ae,Z,a.pwWeight,a.pwBias,de,a.pwUniform])),Ne.push(Y(ke,[de,a.dwWeight,a.dwBias,Ae,a.dwUniform])),Ke.push(Y(ye,[Ae,de,a.pwWeight,a.pwBias,Z,a.pwUniform]));let xn=Y(Me,[Z,rt,de,gn]),vn=Y(Me,[Z,it,de,bn]),Pn=Y(Xe,[Z,J,R,Q,K]),Bn=Y(Me,[Q,nt,de,yn]);Y(Ee,[Z,ne,W,le,ee]),Y(Ee,[Z,na,ia,le,xa]),Y(Ee,[Z,ra,sa,le,ka]);let Ve=Y(ut,[ue.createView(),He,Ie]),Cn=Y(Qe,[Z,ne,W,na,ia,ra,sa,le]),oa=24,Aa=[],Sa=[];for(let a=oa;a<st.length;a++){let p=st[a];Aa.push(Y(Re,[Z,p.dwWeight,p.dwBias,p.pwWeight,p.pwBias,de,p.dwUniform])),Sa.push(Y(Re,[de,p.dwWeight,p.dwBias,p.pwWeight,p.pwBias,Z,p.dwUniform]))}let ua=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});ua.globalCompositeOperation="copy";let Da=new OffscreenCanvas(9,8),Nt=Da.getContext("webgpu"),Kt=null,pa=null;if(Nt){Nt.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let a=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),p=e.createShaderModule({code:tn}),o=e.createShaderModule({code:an});Kt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:p,entryPoint:"vs"},fragment:{module:o,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),pa=e.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:le}}]})}let qt=new Float32Array(1),$t=new Float32Array(1),Yt=new Float32Array(63);function qe(a,p){let o=!0,s=0,t=a.beginComputePass();for(t.setPipeline(mt),t.setBindGroup(0,Ga),t.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);s<=In;s++){let x=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=We[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let D=o?Z:de;for(a.copyBufferToBuffer(D,0,nt,0,3072*64*4),t=a.beginComputePass();s<=Fn;s++){let x=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=We[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let U=o?Z:de;for(a.copyBufferToBuffer(U,0,it,0,3072*32*4),t=a.beginComputePass();s<=zn;s++){let x=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=We[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let E=o?Z:de;for(a.copyBufferToBuffer(E,0,rt,0,1536*16*4),t=a.beginComputePass();s<=Nn;s++){let x=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=We[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.setPipeline(Oe),t.setBindGroup(0,xn),t.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),t.end(),o=!1,t=a.beginComputePass();{let x=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=We[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}t.setPipeline(Oe),t.setBindGroup(0,vn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),t.end(),o=!1,t=a.beginComputePass();{let x=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=We[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}for(t.setPipeline(tt),t.setBindGroup(0,Pn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),t.setPipeline(Oe),t.setBindGroup(0,Bn),t.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),t.end(),o=!1,t=a.beginComputePass();s<oa;s++){let x=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=We[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}for(;s<st.length;s++){let x=s-oa,f=o?Aa[x]:Sa[x],u=st[s];t.setPipeline(Ut),t.setBindGroup(0,f),t.dispatchWorkgroups(u.outW,u.outH,1),o=!o}t.setPipeline(Rt),t.setBindGroup(0,Cn),t.dispatchWorkgroups(1),t.end(),p&&a.copyBufferToBuffer(le,0,p,0,260)}async function Xt(a){e.queue.writeBuffer(at,0,a);let p=e.createCommandEncoder();{let t=p.beginComputePass();t.setPipeline(aa),t.setBindGroup(0,Ua),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}qe(p,X),e.queue.submit([p.finish()]);let o=X.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(X.getMappedRange());return qt[0]=s[0],$t[0]=s[1],Yt.set(s.subarray(2,65)),X.unmap(),{handflag:new Float32Array(qt),handedness:new Float32Array($t),landmarks:new Float32Array(Yt)}}async function ca(a){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let p=e.createCommandEncoder();{let t=p.beginComputePass();t.setPipeline(Ge),t.setBindGroup(0,Ve),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}qe(p,X),e.queue.submit([p.finish()]);let o=X.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(X.getMappedRange());return qt[0]=s[0],$t[0]=s[1],Yt.set(s.subarray(2,65)),X.unmap(),{handflag:new Float32Array(qt),handedness:new Float32Array($t),landmarks:new Float32Array(Yt)}}async function Ma(a){if(!Kt||!pa||!Nt)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let p=e.createCommandEncoder();{let x=p.beginComputePass();x.setPipeline(Ge),x.setBindGroup(0,Ve),x.dispatchWorkgroups(16,16,1),x.end()}qe(p,null);let o=Nt.getCurrentTexture(),s=p.beginRenderPass({colorAttachments:[{view:o.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});s.setPipeline(Kt),s.setBindGroup(0,pa),s.draw(3),s.end(),e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),ua.drawImage(Da,0,0);let D=ua.getImageData(0,0,9,8).data,U=new Float32Array(65),E=new DataView(new ArrayBuffer(4));for(let x=0;x<65;x++){let f=x*4;E.setUint8(0,D[f]),E.setUint8(1,D[f+1]),E.setUint8(2,D[f+2]),E.setUint8(3,D[f+3]),U[x]=E.getFloat32(0)}return{handflag:new Float32Array([U[0]]),handedness:new Float32Array([U[1]]),landmarks:new Float32Array(U.subarray(2,65))}}let kn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),da=0,Un=[X,kn],wt=null,je=null;async function _a(a){let p=Un[da];da=1-da,e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let o=e.createCommandEncoder();{let t=o.beginComputePass();t.setPipeline(Ge),t.setBindGroup(0,Ve),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}qe(o,p),e.queue.submit([o.finish()]);let s=null;if(wt!==null&&je!==null){await wt;let t=new Float32Array(je.getMappedRange());s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))},je.unmap()}return je=p,wt=p.mapAsync(GPUMapMode.READ),s}async function Ea(){if(!wt||!je)return null;await wt;let a=new Float32Array(je.getMappedRange()),p={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return je.unmap(),wt=null,je=null,p}async function Gn(a=50){let p=new Float32Array(196608);for(let t=0;t<5;t++)await Xt(p);let o=[];for(let t=0;t<a;t++){let D=performance.now();await Xt(p),o.push(performance.now()-D)}let s=o.reduce((t,D)=>t+D,0)/o.length;return{avgMs:s,fps:1e3/s}}async function An(a=50){let p=new Float32Array(196608);for(let U=0;U<5;U++)await Xt(p);let o=[];for(let U=0;U<a;U++){let E=e.createCommandEncoder();{let f=E.beginComputePass();f.setPipeline(aa),f.setBindGroup(0,Ua),f.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),f.end()}qe(E,X);let x=performance.now();e.queue.submit([E.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-x)}o.sort((U,E)=>U-E);let s=o.reduce((U,E)=>U+E,0)/o.length,t=o[Math.floor(o.length/2)],D=o[0];return{avgMs:s,fps:1e3/s,medianMs:t,minMs:D}}function ei(a){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let p=e.createCommandEncoder();{let o=p.beginComputePass();o.setPipeline(Ge),o.setBindGroup(0,Ve),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),o.end()}qe(p,X),e.queue.submit([p.finish()])}async function Sn(a,p=50){function o(f){let u=[...f].sort((I,te)=>I-te);return{median:u[Math.floor(u.length/2)],min:u[0]}}for(let f=0;f<10;f++)await ca(a);let s=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let u=e.createCommandEncoder();{let te=u.beginComputePass();te.setPipeline(Ge),te.setBindGroup(0,Ve),te.dispatchWorkgroups(16,16,1),te.end()}qe(u,X);let I=performance.now();e.queue.submit([u.finish()]),await e.queue.onSubmittedWorkDone(),s.push(performance.now()-I)}let t=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let u=e.createCommandEncoder();{let re=u.beginComputePass();re.setPipeline(Ge),re.setBindGroup(0,Ve),re.dispatchWorkgroups(16,16,1),re.end()}qe(u,X),e.queue.submit([u.finish()]);let I=X.mapAsync(GPUMapMode.READ),te=performance.now();await e.queue.onSubmittedWorkDone(),await I,X.getMappedRange(),X.unmap(),t.push(performance.now()-te)}let D=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let u=e.createCommandEncoder();{let te=u.beginComputePass();te.setPipeline(Ge),te.setBindGroup(0,Ve),te.dispatchWorkgroups(16,16,1),te.end()}qe(u,X),e.queue.submit([u.finish()]);let I=performance.now();await X.mapAsync(GPUMapMode.READ),X.getMappedRange(),X.unmap(),D.push(performance.now()-I)}let U=[];for(let f=0;f<p;f++){let u=performance.now();await ca(a),U.push(performance.now()-u)}await _a(a);let E=[];for(let f=0;f<p;f++){let u=performance.now();await _a(a),E.push(performance.now()-u)}await Ea();let x=null;if(Kt){let f=[];for(let u=0;u<p;u++){let I=performance.now();await Ma(a),f.push(performance.now()-I)}x=o(f)}return{gpuOnly:o(s),mapAsyncOnly:o(t),mapAsyncNoWait:o(D),total:o(U),pipelined:o(E),renderReadback:x}}async function Dn(a){let p=[];async function o(t,D,U){let E=e.createBuffer({size:D,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),x=e.createCommandEncoder();x.copyBufferToBuffer(t,0,E,0,D),e.queue.submit([x.finish()]),await e.queue.onSubmittedWorkDone(),await E.mapAsync(GPUMapMode.READ);let f=new Float32Array(E.getMappedRange()),u=1/0,I=-1/0,te=0;for(let re=0;re<f.length;re++)f[re]<u&&(u=f[re]),f[re]>I&&(I=f[re]),f[re]!==0&&te++;E.unmap(),E.destroy(),p.push({layer:U,stats:{min:u,max:I,nonZero:te,total:f.length}})}e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);{let t=e.createCommandEncoder(),D=t.beginComputePass();D.setPipeline(Ge),D.setBindGroup(0,Ve),D.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),D.end(),e.queue.submit([t.finish()])}await o(He,Math.min(He.size,3*257*257*4),"canvas\u2192bufInput");{let t=e.createCommandEncoder(),D=t.beginComputePass();D.setPipeline(mt),D.setBindGroup(0,Ga),D.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),D.end(),e.queue.submit([t.finish()])}await o(Z,Math.min(Z.size,3072*128*4),"inputConv\u2192bufA");let s=!0;for(let t=0;t<Math.min(st.length,6);t++){let D=s?Fe[t]:Ne[t],U=s?ze[t]:Ke[t],E=We[t],x=st[t];{let u=e.createCommandEncoder(),I=u.beginComputePass();I.setPipeline(E.dwPipeline),I.setBindGroup(0,D),I.dispatchWorkgroups(E.dwDispatchX,E.dwDispatchY,E.dwDispatchZ),I.end(),e.queue.submit([u.finish()])}await o(Ae,Math.min(Ae.size,x.spec.inCh*x.outH*x.outW*4),`layer${t}.DW\u2192bufDW (${x.spec.prefix})`);{let u=e.createCommandEncoder(),I=u.beginComputePass();I.setPipeline(E.pwPipeline),I.setBindGroup(0,U),I.dispatchWorkgroups(E.pwDispatchX,E.pwDispatchY,E.pwDispatchZ),I.end(),e.queue.submit([u.finish()])}let f=s?de:Z;await o(f,Math.min(f.size,x.spec.outCh*x.outH*x.outW*4),`layer${t}.PW\u2192buf${s?"B":"A"} (${x.spec.prefix})`),s=!s}return p}return{device:e,run:Xt,runFromCanvas:ca,runFromCanvasViaRender:Ma,runFromCanvasPipelined:_a,flushPipelined:Ea,benchmark:Gn,benchmarkGPU:An,benchmarkDiagnostic:Sn,debugLayerOutputs:Dn}}function ot(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var sn=ot(`
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
`),on=ot(`
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
`),un=ot(`
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
`),pn=ot(`
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
`),cn=ot(`
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
`),dn=ot(`
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
`),_n=ot(`
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
`);async function wa(n,_){let i;if(_)i=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let c=await navigator.gpu.requestAdapter();if(!c)throw new Error("No GPU adapter found");i=await c.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(c.limits.maxStorageBuffersPerShaderStage,8)}})}let y={r:"read-only-storage",s:"storage",u:"uniform"};function w(c){return i.createBindGroupLayout({entries:c.map((r,C)=>r==="t"?{binding:C,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:C,visibility:GPUShaderStage.COMPUTE,buffer:{type:y[r]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,g=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,G=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(c,r){return i.createBuffer({size:Math.max(c,4),usage:r})}function h(c,r,C){i.queue.writeBuffer(c,r,C)}function v(c){let r=d(c.data.byteLength,e);return h(r,0,c.data),r}let k=Array.from(n.keys());function L(c){let r=n.get(c);if(!r)throw new Error(`Weight not found: ${c}`);return r}function m(...c){let r=k.find(C=>c.every(M=>C.includes(M)));if(!r)throw new Error(`Weight not found for: ${c.join(", ")}`);return L(r)}function z(c){let[,r,C,M]=c.shape,b=new Float32Array(M*25);for(let l=0;l<M;l++)for(let T=0;T<r;T++)for(let J=0;J<C;J++)b[l*25+T*5+J]=c.data[T*C*M+J*M+l];return b}function B(c){let[r,,,C]=c.shape,M=new Float32Array(r*C);for(let b=0;b<r;b++)for(let l=0;l<C;l++)M[b*C+l]=c.data[b*C+l];return M}let pe=i.createShaderModule({code:sn}),se=i.createShaderModule({code:on}),Pe=i.createShaderModule({code:un}),ce=i.createShaderModule({code:pn}),S=i.createShaderModule({code:dn}),Y=i.createShaderModule({code:cn}),ae=i.createShaderModule({code:_n}),H=w(["r","r","r","r","s","u"]),F=w(["r","r","r","s","u"]),N=w(["r","r","r","r","r","s","u"]),Se=w(["r","r","r","s","u"]),be=w(["r","r","r","r","s","u"]),Te=w(["r","r","s","u"]),he=w(["t","s","u"]);function Be(c,r){return i.createComputePipeline({layout:i.createPipelineLayout({bindGroupLayouts:[c]}),compute:{module:r,entryPoint:"main"}})}let P=Be(H,pe),V=Be(F,se),we=Be(N,Pe),j=Be(Se,ce),ge=Be(be,S),oe=Be(Te,Y),Ye=Be(he,ae),Je=m("conv2d/Conv2D"),Ce=m("batch_normalization/","conv2d/Conv2D"),ke=m("p_re_lu/"),ye=v(Je),xe=v(Ce),De=v(ke),Le=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12}].map(c=>{let r=m(c.dwKey),C=m(c.pwKey),M=m(c.bnKey),b=m(c.preluKey),l=z(r),T=d(l.byteLength,e);h(T,0,l);let J=new Float32Array(c.inCh),R=d(J.byteLength,e);h(R,0,J);let K=B(C),O=d(K.byteLength,e);h(O,0,K);let q=v(M),$=v(b);return{dwWeightBuf:T,dwBiasBuf:R,pwWeightBuf:O,pwBiasBuf:q,alphaBuf:$,inCh:c.inCh,outCh:c.outCh,stride:c.stride,inH:c.inH}}),Me=B(m("conv2d_20/Conv2D")),Xe=d(Me.byteLength,e);h(Xe,0,Me);let Ee=v(m("batch_normalization_20/")),ut=v(m("p_re_lu_20/")),Qe={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_19/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(256),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:Xe,pwBiasBuf:Ee,alphaBuf:ut,inCh:256,outCh:256,stride:1,inH:12},Re={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_20/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(256),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:(()=>{let c=B(m("conv2d_21/")),r=d(c.byteLength,e);return h(r,0,c),r})(),pwBiasBuf:v(m("batch_normalization_21/")),alphaBuf:v(m("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},Bt=B(m("conv2d_23/Conv2D")),Et=d(Bt.byteLength,e);h(Et,0,Bt);let pt=v(m("batch_normalization_23/")),ct=v(m("p_re_lu_23/")),Qt={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_21/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(128),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:(()=>{let c=B(m("conv2d_24/")),r=d(c.byteLength,e);return h(r,0,c),r})(),pwBiasBuf:v(m("batch_normalization_24/")),alphaBuf:v(m("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},ea={dwWeightBuf:(()=>{let c=z(m("depthwise_conv2d_22/")),r=d(c.byteLength,e);return h(r,0,c),r})(),dwBiasBuf:(()=>{let c=new Float32Array(128),r=d(c.byteLength,e);return h(r,0,c),r})(),pwWeightBuf:(()=>{let c=B(m("conv2d_25/Conv2D1")),r=d(c.byteLength,e);return h(r,0,c),r})(),pwBiasBuf:v(m("batch_normalization_25/")),alphaBuf:v(m("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},Wt=B(m("classifier_palm_16_NO_PRUNING/Conv2D")),dt=d(Wt.byteLength,e);h(dt,0,Wt);let _t=v(m("classifier_palm_16_NO_PRUNING/BiasAdd")),Ct=B(m("regressor_palm_16_NO_PRUNING/Conv2D")),lt=d(Ct.byteLength,e);h(lt,0,Ct);let kt=v(m("regressor_palm_16_NO_PRUNING/BiasAdd")),et=B(m("classifier_palm_8_NO_PRUNING/Conv2D")),Ht=d(et.byteLength,e);h(Ht,0,et);let ta=v(m("classifier_palm_8_NO_PRUNING/BiasAdd")),Tt=B(m("regressor_palm_8_NO_PRUNING/Conv2D")),Lt=d(Tt.byteLength,e);h(Lt,0,Tt);let We=v(m("regressor_palm_8_NO_PRUNING/BiasAdd")),aa=36864*3,mt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Oe=d(36864*3*4,e),tt=d(mt,g),Ge=d(mt,g),Rt=d(mt,g),Ut=d(576*128*4,g),Ot=d(576*128*4,g),at=d(864*4,G),He=d(15552*4,G),Gt=d(576*2*4,G),Z=d(576*36*4,G),de=d(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ae=d(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),nt=d(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),it=d(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),rt=i.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function Q(c,r){return Math.ceil(c/r)}function le(c){let r=d(c.byteLength,A);return h(r,0,c),r}let X=le(new Uint32Array([1,3,32,192,192,96,96]));function ue(c,r,C,M,b){let l=r.stride===2?r.inH/2:r.inH,T=l,J=r.stride===2?1:2,R=le(new Uint32Array([1,r.inCh,r.inH,r.inH,l,T,r.stride,J])),K=i.createBindGroup({layout:F,entries:[{binding:0,resource:{buffer:C}},{binding:1,resource:{buffer:r.dwWeightBuf}},{binding:2,resource:{buffer:r.dwBiasBuf}},{binding:3,resource:{buffer:Rt}},{binding:4,resource:{buffer:R}}]}),O=c.beginComputePass();O.setPipeline(V),O.setBindGroup(0,K),O.dispatchWorkgroups(Q(T,8),Q(l,8),r.inCh),O.end();let q=r.inCh,$=le(new Uint32Array([1,r.inCh,r.outCh,l,T,q,r.stride,r.inH,r.inH])),me=i.createBindGroup({layout:N,entries:[{binding:0,resource:{buffer:Rt}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:r.pwWeightBuf}},{binding:3,resource:{buffer:r.pwBiasBuf}},{binding:4,resource:{buffer:r.alphaBuf}},{binding:5,resource:{buffer:M}},{binding:6,resource:{buffer:$}}]}),ne=c.beginComputePass();ne.setPipeline(we),ne.setBindGroup(0,me),ne.dispatchWorkgroups(Q(T,8),Q(l,8),r.outCh),ne.end()}function Ie(c,r,C,M,b,l,T,J,R){let K=le(new Uint32Array([1,l,T,J,R])),O=i.createBindGroup({layout:Se,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:C}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:b}},{binding:4,resource:{buffer:K}}]}),q=c.beginComputePass();q.setPipeline(j),q.setBindGroup(0,O),q.dispatchWorkgroups(Q(R,8),Q(J,8),T),q.end()}function It(c,r,C,M,b,l,T,J,R,K){let O=le(new Uint32Array([1,T,J,R,K])),q=i.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:C}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:b}},{binding:4,resource:{buffer:l}},{binding:5,resource:{buffer:O}}]}),$=c.beginComputePass();$.setPipeline(ge),$.setBindGroup(0,q),$.dispatchWorkgroups(Q(K,8),Q(R,8),J),$.end()}async function Ft(c){i.queue.copyExternalImageToTexture({source:c},{texture:rt},[192,192]);let r=le(new Uint32Array([192,192,192])),C=i.createBindGroup({layout:he,entries:[{binding:0,resource:rt.createView()},{binding:1,resource:{buffer:Oe}},{binding:2,resource:{buffer:r}}]}),M=i.createCommandEncoder();{let W=M.beginComputePass();W.setPipeline(Ye),W.setBindGroup(0,C),W.dispatchWorkgroups(Q(192,16),Q(192,16),1),W.end()}{let W=i.createBindGroup({layout:H,entries:[{binding:0,resource:{buffer:Oe}},{binding:1,resource:{buffer:ye}},{binding:2,resource:{buffer:xe}},{binding:3,resource:{buffer:De}},{binding:4,resource:{buffer:tt}},{binding:5,resource:{buffer:X}}]}),ee=M.beginComputePass();ee.setPipeline(P),ee.setBindGroup(0,W),ee.dispatchWorkgroups(Q(96,8),Q(96,8),32),ee.end()}let b=tt,l=Ge;for(let W=0;W<Le.length;W++){let ee=Le[W];ue(M,ee,b,l,b);let ie=b;b=l,l=ie,W===10&&M.copyBufferToBuffer(b,0,Ut,0,576*128*4)}ue(M,Qe,b,l,b);{let W=b;b=l,l=W}ue(M,Re,b,l,b);{let W=b;b=l,l=W}Ie(M,b,dt,_t,at,256,6,12,12),Ie(M,b,lt,kt,He,256,108,12,12),It(M,b,Et,pt,ct,l,256,128,12,12);{let W=b;b=l,l=W}{let W=le(new Uint32Array([1,128,12,12,24,24])),ee=i.createBindGroup({layout:Te,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:Ut}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:W}}]}),ie=M.beginComputePass();ie.setPipeline(oe),ie.setBindGroup(0,ee),ie.dispatchWorkgroups(Q(24,8),Q(24,8),128),ie.end()}{let W=b;b=l,l=W}ue(M,Qt,b,l,b);{let W=b;b=l,l=W}ue(M,ea,b,l,b);{let W=b;b=l,l=W}Ie(M,b,Ht,ta,Gt,128,2,24,24),Ie(M,b,Lt,We,Z,128,36,24,24),i.queue.submit([M.finish()]);let T=i.createCommandEncoder();T.copyBufferToBuffer(at,0,de,0,864*4),T.copyBufferToBuffer(He,0,Ae,0,15552*4),T.copyBufferToBuffer(Gt,0,nt,0,576*2*4),T.copyBufferToBuffer(Z,0,it,0,576*36*4),i.queue.submit([T.finish()]),await Promise.all([de.mapAsync(GPUMapMode.READ),Ae.mapAsync(GPUMapMode.READ),nt.mapAsync(GPUMapMode.READ),it.mapAsync(GPUMapMode.READ)]);let J=new Float32Array(de.getMappedRange()).slice(),R=new Float32Array(Ae.getMappedRange()).slice(),K=new Float32Array(nt.getMappedRange()).slice(),O=new Float32Array(it.getMappedRange()).slice();de.unmap(),Ae.unmap(),nt.unmap(),it.unmap();let q=2016,$=new Float32Array(q),me=new Float32Array(q*18),ne=0;for(let W=0;W<12;W++)for(let ee=0;ee<12;ee++)for(let ie=0;ie<6;ie++){$[ne]=J[ie*144+W*12+ee];for(let ve=0;ve<18;ve++){let ft=ie*18+ve;me[ne*18+ve]=R[ft*144+W*12+ee]}ne++}for(let W=0;W<24;W++)for(let ee=0;ee<24;ee++)for(let ie=0;ie<2;ie++){$[ne]=K[ie*576+W*24+ee];for(let ve=0;ve<18;ve++){let ft=ie*18+ve;me[ne*18+ve]=O[ft*576+W*24+ee]}ne++}return{scores:$,regressors:me}}async function fe(c,r){let C=i.createBuffer({size:r*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),M=i.createCommandEncoder();M.copyBufferToBuffer(c,0,C,0,r*4),i.queue.submit([M.finish()]),await C.mapAsync(GPUMapMode.READ);let b=new Float32Array(C.getMappedRange()).slice();return C.unmap(),C.destroy(),b}async function zt(c){i.queue.copyExternalImageToTexture({source:c},{texture:rt},[192,192]);function r(O,q=1e3){let $=O.slice(0,q);return{min:Math.min(...$),max:Math.max(...$),mean:$.reduce((me,ne)=>me+ne,0)/$.length,nonZero:$.filter(me=>me!==0).length,sample:Array.from($.slice(0,10))}}let C={},M=le(new Uint32Array([192,192,192])),b=i.createBindGroup({layout:he,entries:[{binding:0,resource:rt.createView()},{binding:1,resource:{buffer:Oe}},{binding:2,resource:{buffer:M}}]}),l=i.createCommandEncoder(),T=l.beginComputePass();T.setPipeline(Ye),T.setBindGroup(0,b),T.dispatchWorkgroups(Q(192,16),Q(192,16),1),T.end(),i.queue.submit([l.finish()]),C.input=r(await fe(Oe,36864*3)),l=i.createCommandEncoder();let J=i.createBindGroup({layout:H,entries:[{binding:0,resource:{buffer:Oe}},{binding:1,resource:{buffer:ye}},{binding:2,resource:{buffer:xe}},{binding:3,resource:{buffer:De}},{binding:4,resource:{buffer:tt}},{binding:5,resource:{buffer:X}}]});T=l.beginComputePass(),T.setPipeline(P),T.setBindGroup(0,J),T.dispatchWorkgroups(Q(96,8),Q(96,8),32),T.end(),i.queue.submit([l.finish()]),C.initConv=r(await fe(tt,9216*32));let R=tt,K=Ge;for(let O=0;O<Le.length;O++){let q=Le[O];l=i.createCommandEncoder(),ue(l,q,R,K,R),i.queue.submit([l.finish()]);let $=R;if(R=K,K=$,O===0||O===3||O===7||O===11||O===17){let me=q.stride===2?q.inH/2:q.inH,ne=me*me*q.outCh;C[`block${O}`]=r(await fe(R,ne))}O===10&&(l=i.createCommandEncoder(),l.copyBufferToBuffer(R,0,Ut,0,576*128*4),i.queue.submit([l.finish()]))}l=i.createCommandEncoder(),ue(l,Qe,R,K,R),i.queue.submit([l.finish()]);{let O=R;R=K,K=O}C.extraBlockA=r(await fe(R,144*256)),l=i.createCommandEncoder(),ue(l,Re,R,K,R),i.queue.submit([l.finish()]);{let O=R;R=K,K=O}return C.extraBlockB=r(await fe(R,144*256)),l=i.createCommandEncoder(),Ie(l,R,dt,_t,at,256,6,12,12),i.queue.submit([l.finish()]),C.cls16=r(await fe(at,864)),l=i.createCommandEncoder(),Ie(l,R,lt,kt,He,256,108,12,12),i.queue.submit([l.finish()]),C.reg16=r(await fe(He,15552),500),C.initWeights=r(await fe(ye,100),100),C.initBias=r(await fe(xe,32),32),C.cls16Weights=r(await fe(dt,100),100),C.cls16Bias=r(await fe(_t,6),6),C}return{device:i,run:Ft,debugRun:zt}}function Kn(){let n=[];for(let _=0;_<12;_++)for(let i=0;i<12;i++){let y=(i+.5)/12,w=(_+.5)/12;for(let e=0;e<6;e++)n.push({x:y,y:w})}for(let _=0;_<24;_++)for(let i=0;i<24;i++){let y=(i+.5)/24,w=(_+.5)/24;for(let e=0;e<2;e++)n.push({x:y,y:w})}return n}var ln=Kn();function qn(n){return 1/(1+Math.exp(-n))}function mn(n,_){let i=[],{scores:y,regressors:w}=n,e=192;for(let g=0;g<ln.length;g++){let G=qn(y[g]);if(G<_)continue;let A=ln[g],d=g*18,h=A.x+w[d+0]/e,v=A.y+w[d+1]/e,k=w[d+2]/e,L=w[d+3]/e,m=[];for(let z=0;z<7;z++){let B=A.x+w[d+4+z*2]/e,pe=A.y+w[d+4+z*2+1]/e;m.push([B,pe])}i.push({score:G,box:[h,v,k,L],keypoints:m})}return i}function fn(n,_){if(n.length===0)return[];let i=[...n].sort((e,g)=>g.score-e.score),y=[],w=new Set;for(let e=0;e<i.length;e++)if(!w.has(e)){y.push(i[e]);for(let g=e+1;g<i.length;g++)w.has(g)||$n(i[e],i[g])>_&&w.add(g)}return y}function $n(n,_){let i=n.box[0]-n.box[2]/2,y=n.box[1]-n.box[3]/2,w=n.box[0]+n.box[2]/2,e=n.box[1]+n.box[3]/2,g=_.box[0]-_.box[2]/2,G=_.box[1]-_.box[3]/2,A=_.box[0]+_.box[2]/2,d=_.box[1]+_.box[3]/2,h=Math.max(i,g),v=Math.max(y,G),k=Math.min(w,A),L=Math.min(e,d),m=Math.max(0,k-h),z=Math.max(0,L-v),B=m*z,pe=(w-i)*(e-y),se=(A-g)*(d-G),Pe=pe+se-B;return Pe>0?B/Pe:0}function Yn(n){let[_,i,y,w]=n.box,e=n.keypoints[0],g=n.keypoints[2],G=g[0]-e[0],A=g[1]-e[1],d=Math.atan2(G,A),v=Math.max(y,w)*2.6,k=.5,L=Math.sqrt(G*G+A*A),m=L>0?G/L*v*k*.5:0,z=L>0?A/L*v*k*.5:0;return{centerX:_+m,centerY:i+z,width:v,height:v,rotation:d}}function ga(n,_={}){let{scoreThreshold:i=.5,nmsThreshold:y=.3,maxHands:w=2}=_;async function e(G){let A=await n.run(G),d=mn(A,i);return fn(d,y).slice(0,w).map(Yn)}async function g(G){let A=await n.run(G),d=mn(A,i);return fn(d,y).slice(0,w)}return{detect:e,detectRaw:g,model:n}}function hn(n,_=256){let i=Math.cos(n.rotation),y=Math.sin(n.rotation),w=n.width/_,e=n.height/_,g=w*i,G=-e*y,A=w*y,d=e*i,h=n.centerX-(g*_/2+G*_/2),v=n.centerY-(A*_/2+d*_/2),k=g*d-G*A,L=d/k,m=-G/k,z=-A/k,B=g/k,pe=-(L*h+m*v),se=-(z*h+B*v);return{forward:[g,G,h,A,d,v],inverse:[L,m,pe,z,B,se]}}function ba(n,_){let{forward:i}=hn(_,1),[y,w,e,g,G,A]=i;return n.map(d=>({x:y*d.x+w*d.y+e,y:g*d.x+G*d.y+A,z:d.z}))}var wn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Xn(n={}){let{weightsUrl:_,scoreThreshold:i=.5,forceF32:y=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let w=_??wn,e=w.endsWith("/")?w:`${w}/`,g=`${e}weights_f16.json`,G=`${e}weights_f16.bin`,[A,d]=await Promise.all([fetch(g),fetch(G)]);if(!A.ok)throw new Error(`Failed to fetch weights metadata: ${A.status}`);if(!d.ok)throw new Error(`Failed to fetch weights binary: ${d.status}`);let h=await A.json(),v=await d.arrayBuffer(),k=Mt(h,v),L=await Jt(k,{forceF32:y});if(!y){let H=new OffscreenCanvas(256,256),F=H.getContext("2d");F.fillStyle="#886644",F.fillRect(0,0,256,256),F.fillStyle="#cc9966",F.fillRect(50,50,156,156);let N=await L.runFromCanvas(H);N.landmarks.every(be=>be===0)&&N.handflag.every(be=>be===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),L.device.destroy(),L=await Jt(k,{forceF32:!0}))}let m=null;function z(){return m||(m=new OffscreenCanvas(256,256)),m}async function B(H){if(H instanceof HTMLCanvasElement||H instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&H instanceof ImageBitmap)return H;let F=z();F.width=256,F.height=256;let N=F.getContext("2d");return H instanceof ImageData?N.putImageData(H,0,0):N.drawImage(H,0,0,256,256),F}function pe(H,F,N){let Se=H[0];if(Se<i)return null;let be=F[0]>.5,Te=[];for(let he=0;he<21;he++)Te.push({x:N[he*3],y:N[he*3+1],z:N[he*3+2]});return{score:Se,handedness:be?"right":"left",landmarks:Te}}async function se(H){let F=await B(H),N=await L.runFromCanvas(F);return pe(N.handflag,N.handedness,N.landmarks)}async function Pe(H){let F=await B(H),N=await L.runFromCanvasPipelined(F);return N?pe(N.handflag,N.handedness,N.landmarks):null}async function ce(){let H=await L.flushPipelined();return H?pe(H.handflag,H.handedness,H.landmarks):null}function S(){L.device.destroy(),m=null}async function Y(H){let F=await B(H);return L.benchmarkDiagnostic(F)}async function ae(H){let F=await B(H);return L.debugLayerOutputs(F)}return{detect:se,detectPipelined:Pe,flushPipelined:ce,dispose:S,benchmarkDiagnostic:Y,debugLayerOutputs:ae}}async function Vn(n={}){let{weightsUrl:_,palmWeightsUrl:i,scoreThreshold:y=.5,palmScoreThreshold:w=.5,maxHands:e=2,forceF32:g=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let G=_??wn,A=G.endsWith("/")?G:`${G}/`;if(!i)throw new Error("palmWeightsUrl is required for createFullHandpose");let d=i.endsWith("/")?i:`${i}/`,[h,v,k,L]=await Promise.all([fetch(`${A}weights_f16.json`),fetch(`${A}weights_f16.bin`),fetch(`${d}palm_detection_weights.json`),fetch(`${d}palm_detection_weights.bin`)]);if(!h.ok)throw new Error(`Failed to fetch landmark weights metadata: ${h.status}`);if(!v.ok)throw new Error(`Failed to fetch landmark weights binary: ${v.status}`);if(!k.ok)throw new Error(`Failed to fetch palm weights metadata: ${k.status}`);if(!L.ok)throw new Error(`Failed to fetch palm weights binary: ${L.status}`);let[m,z,B,pe]=await Promise.all([h.json(),v.arrayBuffer(),k.json(),L.arrayBuffer()]),se=Mt(m,z),Pe=Mt(B,pe),ce=await Jt(se,{forceF32:g}),S=await wa(Pe),Y=ga(S,{scoreThreshold:w,maxHands:e}),ae=null,H=null;function F(){return ae||(ae=new OffscreenCanvas(192,192)),ae}function N(){return H||(H=new OffscreenCanvas(256,256)),H}async function Se(P){if(P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas){if(P.width===192&&P.height===192)return P;let j=F();return j.width=192,j.height=192,j.getContext("2d").drawImage(P,0,0,192,192),j}if(typeof ImageBitmap<"u"&&P instanceof ImageBitmap){if(P.width===192&&P.height===192)return P;let j=F();return j.width=192,j.height=192,j.getContext("2d").drawImage(P,0,0,192,192),j}let V=F();V.width=192,V.height=192;let we=V.getContext("2d");if(P instanceof ImageData){let j=new OffscreenCanvas(P.width,P.height);j.getContext("2d").putImageData(P,0,0),we.drawImage(j,0,0,192,192)}else we.drawImage(P,0,0,192,192);return V}function be(P,V,we,j){let ge=N();ge.width=256,ge.height=256;let oe=ge.getContext("2d"),Ye=Math.cos(-V.rotation),Je=Math.sin(-V.rotation);oe.clearRect(0,0,256,256),oe.save(),oe.translate(128,128),oe.scale(V.width*we/256,V.height*j/256),oe.rotate(-V.rotation),oe.translate(-128,-128);let Ce=V.centerX*we,ke=V.centerY*j;oe.restore();let ye=256/(V.width*we),xe=256/(V.height*j),De=Math.cos(V.rotation),Ue=Math.sin(V.rotation),Le=De*ye,Me=Ue*ye,Xe=-Ue*xe,Ee=De*xe,ut=-Ce*Le-ke*Xe+128,Qe=-Ce*Me-ke*Ee+128;if(oe.setTransform(Le,Me,Xe,Ee,ut,Qe),P instanceof ImageData){let Re=new OffscreenCanvas(P.width,P.height);Re.getContext("2d").putImageData(P,0,0),oe.drawImage(Re,0,0)}else oe.drawImage(P,0,0);return oe.setTransform(1,0,0,1,0,0),ge}function Te(P){return P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas?[P.width,P.height]:typeof ImageBitmap<"u"&&P instanceof ImageBitmap?[P.width,P.height]:P instanceof ImageData?[P.width,P.height]:P instanceof HTMLVideoElement?[P.videoWidth,P.videoHeight]:P instanceof HTMLImageElement?[P.naturalWidth,P.naturalHeight]:[256,256]}async function he(P){let V=await Se(P),we=await Y.detect(V);if(we.length===0)return[];let[j,ge]=Te(P),oe=[];for(let Ye of we){let Je=be(P,Ye,j,ge),Ce=await ce.runFromCanvas(Je),ke=Ce.handflag[0];if(ke<y)continue;let ye=Ce.handedness[0]>.5,xe=[];for(let Ue=0;Ue<21;Ue++)xe.push({x:Ce.landmarks[Ue*3],y:Ce.landmarks[Ue*3+1],z:Ce.landmarks[Ue*3+2]});let De=ba(xe,Ye);oe.push({score:ke,handedness:ye?"right":"left",landmarks:De,palmScore:0})}return oe}function Be(){ce.device.destroy(),S.device.destroy(),ae=null,H=null}return{detect:he,dispose:Be}}function jn(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Zn=jn(`
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
`);function Jn(n){let _=n.createShaderModule({code:Zn}),i=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),y=n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[i]}),compute:{module:_,entryPoint:"main"}});function w(e,g,G,A,d,h,v){let k=new Uint32Array([d,h,v,0]),L=n.createBuffer({size:k.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(L,0,k);let m=new Float32Array(A),z=new Float32Array(8);z.set(m);let B=n.createBuffer({size:z.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(B,0,z);let pe=n.createBindGroup({layout:i,entries:[{binding:0,resource:g.createView()},{binding:1,resource:{buffer:G}},{binding:2,resource:{buffer:L}},{binding:3,resource:{buffer:B}}]}),se=e.beginComputePass();se.setPipeline(y),se.setBindGroup(0,pe),se.dispatchWorkgroups(Math.ceil(v/16),Math.ceil(v/16),1),se.end()}return{crop:w}}export{wa as compilePalmModel,hn as computeCropTransform,Jn as createCropPipeline,Vn as createFullHandpose,Xn as createHandpose,ga as createPalmDetector,Mt as loadWeightsFromBuffer,ba as projectLandmarksToOriginal};
