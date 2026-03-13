function _e(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function La(s){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],t="enable f16;"+s;for(let k of _)for(;t.includes(`${k}:array<f32>`);)t=t.replace(`${k}:array<f32>`,`${k}:array<f16>`);return t}var da=_e(`
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
`),_a=_e(`
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
`),la=_e(`
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
`);function Ra(s,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Oa(s,_){return da.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Ia(s,_){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function za(s,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Fa(s,_){return[8,8]}var Ka=_e(`
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
`);function qa(s){return _e(`
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
`)}var $a=qa(!1),Ya=qa(!0),Xa=_e(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Va=_e(`
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
`);function Za(s){return _e(`
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
`)}var ja=Za("sigmoid"),Ja=Za("div256"),Qa=_e(`
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
`),en=_e(`
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
`);function tn(s,_){let k=Math.min(_,256),b=_>k,P=s%4===0?`var ic:u32=0u;
    while(ic<${s}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${s}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,R=`var skip_val:f32=0.0;
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
    }`,H=s===_?"":`if(c<${s}u){`,l=s===_?"":"}",w=b?`for(var c:u32=lid.x;c<${s}u;c+=${k}u){`:`let c=lid.x;
  ${H}`,A=b?"}":l,T=b?`for(var c:u32=lid.x;c<${_}u;c+=${k}u){`:"{let c=lid.x;";return _e(`
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
@compute @workgroup_size(${k},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${w}
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
  ${A}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${T}
    let pw_base=c*${s}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${P}
    // Skip connection (only for c < inCh)
    ${R}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var an=_e(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),nn=_e(`
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
`),rn=_e(`
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
`);function fa(s,_){let t=new Map,k=s.dtype??"float32";for(let b=0;b<s.keys.length;b++){let e=s.keys[b],P=s.shapes[b],R=s.offsets[b],H=P.reduce((A,T)=>A*T,1),l,w;if(k==="float32")l=new Float32Array(_,R,H);else{let A=new DataView(_);l=new Float32Array(H);for(let T=0;T<H;T++)l[T]=Fn(A.getUint16(R+T*2,!0));w=_.slice(R,R+H*2)}t.set(e,{data:l,shape:P,rawF16:w})}return t}function Fn(s){let _=s>>15&1,t=s>>10&31,k=s&1023;if(t===0){if(k===0)return _?-0:0;let P=-14,R=k/1024;return(_?-1:1)*Math.pow(2,P)*R}if(t===31)return k===0?_?-1/0:1/0:NaN;let b=t-15,e=1+k/1024;return(_?-1:1)*Math.pow(2,b)*e}var Kn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],sn=Kn.map(([s,_,t,k,b])=>({type:"resmodule",inCh:s,outCh:_,h:t,w:t,stride:k,prefix:b})),Nn=2,qn=5,$n=8,Yn=11;async function ha(s,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No GPU adapter found");let k=t.features.has("shader-f16"),b=k?["shader-f16"]:[],e=await t.requestDevice({requiredFeatures:b,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(t.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(t.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(t.limits.maxComputeInvocationsPerWorkgroup,288)}}),P=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(k)try{let i=`enable f16;
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
}`,u=e.createShaderModule({code:i}),o=e.createShaderModule({code:c}),a=await u.getCompilationInfo(),E=await o.getCompilationInfo();if(a.messages.some(D=>D.type==="error")||E.messages.some(D=>D.type==="error"))P=!1;else{let D=new Float32Array(2400);D.fill(1);let W=new Uint16Array(2400);W.fill(10516);let C=new Uint16Array(96);C.fill(14336);let h=new Uint16Array(9216);h.fill(8478);let p=new Uint16Array(96);p.fill(12288);let K=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,te=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,se=e.createBuffer({size:D.byteLength,usage:K}),Bt=e.createBuffer({size:W.byteLength,usage:K}),kt=e.createBuffer({size:C.byteLength,usage:K}),Ct=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),Ut=e.createBuffer({size:h.byteLength,usage:K}),At=e.createBuffer({size:p.byteLength,usage:K}),Gt=e.createBuffer({size:384,usage:te}),Ve=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(se,0,D),e.queue.writeBuffer(Bt,0,W),e.queue.writeBuffer(kt,0,C),e.queue.writeBuffer(Ut,0,h),e.queue.writeBuffer(At,0,p);let Fe="read-only-storage",Ht="storage",Tt=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ht}}]}),Ha=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ht}}]}),Tn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Tt]}),compute:{module:u,entryPoint:"main"}}),Ln=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Ha]}),compute:{module:o,entryPoint:"main"}}),Rn=e.createBindGroup({layout:Tt,entries:[{binding:0,resource:{buffer:se}},{binding:1,resource:{buffer:Bt}},{binding:2,resource:{buffer:kt}},{binding:3,resource:{buffer:Ct}}]}),On=e.createBindGroup({layout:Ha,entries:[{binding:0,resource:{buffer:Ct}},{binding:1,resource:{buffer:Ut}},{binding:2,resource:{buffer:At}},{binding:3,resource:{buffer:Gt}}]}),Jt=e.createCommandEncoder(),Qt=Jt.beginComputePass();Qt.setPipeline(Tn),Qt.setBindGroup(0,Rn),Qt.dispatchWorkgroups(2),Qt.end();let ea=Jt.beginComputePass();ea.setPipeline(Ln),ea.setBindGroup(0,On),ea.dispatchWorkgroups(2),ea.end(),Jt.copyBufferToBuffer(Gt,0,Ve,0,384),e.queue.submit([Jt.finish()]),await e.queue.onSubmittedWorkDone(),await Ve.mapAsync(GPUMapMode.READ);let Lt=new Float32Array(Ve.getMappedRange()),Ta=1.5*.0104*96+.25,In=Lt[0]!==0&&Lt[47]!==0&&Lt[95]!==0,zn=Math.abs(Lt[0]-Ta)<1;P=In&&zn,Ve.unmap(),se.destroy(),Bt.destroy(),kt.destroy(),Ct.destroy(),Ut.destroy(),At.destroy(),Gt.destroy(),Ve.destroy(),P||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Lt[0]}, expected ~${Ta.toFixed(2)}) \u2014 falling back to f32`)}}catch{P=!1}let H=s.values().next().value,l=P&&!!H?.rawF16&&!_?.forceF32;console.log(l?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${k}, f16 validated: ${P}, f16 data: ${!!H?.rawF16})`);function w(i){if(l&&i.rawF16){let c=new Uint16Array(i.rawF16);if(c.length%2!==0){let u=new Uint16Array(c.length+1);return u.set(c),u}return c}return i.data}function A(i){if(l&&i.rawF16){let c=i.rawF16.byteLength;return Math.ceil(c/4)*4}return i.data.byteLength}function T(i){return l?La(i):i}let ce={r:"read-only-storage",s:"storage",u:"uniform"};function x(i){return e.createBindGroupLayout({entries:i.map((c,u)=>({binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:ce[c]}}))})}function ae(i){return e.createBindGroupLayout({entries:i.map((c,u)=>c==="t"?{binding:u,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:ce[c]}})})}let L=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,le=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,ge=GPUBufferUsage.STORAGE,ve=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ne=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function S(i,c){return e.createBuffer({size:i,usage:c})}function X(i,c){return e.createBindGroup({layout:i,entries:c.map((u,o)=>({binding:o,resource:"size"in u?{buffer:u}:u}))})}function oe(i,c){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),compute:{module:c,entryPoint:"main"}})}let Me=e.createShaderModule({code:Ka}),pt=e.createShaderModule({code:rn}),ct=e.createShaderModule({code:T(Qa)}),dt=e.createShaderModule({code:T(_a)}),_t=e.createShaderModule({code:T(da)}),he=e.createShaderModule({code:T(la)}),Ze=e.createShaderModule({code:T(ma)}),v=e.createShaderModule({code:T(Na)}),$=e.createShaderModule({code:$a}),de=e.createShaderModule({code:Xa}),V=e.createShaderModule({code:Ya}),ue=e.createShaderModule({code:T(Va)}),J=e.createShaderModule({code:T(ja)}),Y=e.createShaderModule({code:T(Ja)}),Ee=e.createShaderModule({code:T(en)}),we=new Map;function Ke(i,c){let u=`${i}_${c}`,o=we.get(u);return o||(o=e.createShaderModule({code:T(tn(i,c))}),we.set(u,o)),o}let Ae=x(["r","r","r","s","u"]),be=x(["r","r","r","r","s","u"]),ye=x(["r","s","u"]),Pe=x(["r","r","r","s","u"]),Ne=x(["r","s","u"]),me=x(["r","r","s","u"]),Ge=x(["r","r","s","u"]),We=x(["r","r","r","s","u"]),Se=x(["r","r","r","s","u"]),qe=ae(["t","s","u"]),lt=x(["r","r","r","r","r","r","r","s"]),mt=x(["r","r","r","r","r","s","u"]),Rt=e.createPipelineLayout({bindGroupLayouts:[Ae]}),St=e.createPipelineLayout({bindGroupLayouts:[be]}),je=i=>e.createComputePipeline({layout:Rt,compute:{module:i,entryPoint:"main"}}),Je=i=>e.createComputePipeline({layout:St,compute:{module:i,entryPoint:"main"}}),Ot=je(dt),It=je(_t),zt=Je(he),ft=Je(Ze),ht=new Map,Dt=new Map,wt=new Map,Mt=new Map;ht.set("8,8",Ot),Dt.set("8,8",It),wt.set("8,8",zt),Mt.set("8,8",ft);function Qe(i,c,u,o,a){let E=`${c},${u}`,D=i.get(E);return D||(D=a(e.createShaderModule({code:T(o(c,u))})),i.set(E,D)),D}let bt=(i,c)=>Qe(ht,i,c,Ra,je),Et=(i,c)=>Qe(Dt,i,c,Oa,je),Ft=(i,c)=>Qe(wt,i,c,Ia,Je),Wt=(i,c)=>Qe(Mt,i,c,za,Je),Be=sn.map(i=>{let c=i.stride===2?i.h/2:i.h,u=i.stride===2?i.w/2:i.w,[o,a]=Fa(i.inCh,c),E=i.h>=64,D=c>=16&&i.inCh>=288&&i.outCh>=288&&i.outCh%2===0;return{dwPipeline:E?Et(o,a):bt(o,a),pwPipeline:D?Wt(o,a):Ft(o,a),dwDispatchX:Math.ceil(u/o),dwDispatchY:Math.ceil(c/a),dwDispatchZ:i.inCh,pwDispatchX:Math.ceil(u/o),pwDispatchY:Math.ceil(c/a),pwDispatchZ:D?i.outCh/2:i.outCh}}),gt=oe(ye,Me),$e=oe(Pe,v);oe(Ne,$),oe(me,de);let He=oe(Ge,V),Kt=oe(We,ue);oe(Se,J),oe(Se,Y);let ke=oe(qe,pt),yt=oe(lt,ct),et=oe(mt,Ee),Te=1*288*128*128*4,tt=S(3*256*256*4,L),De=S(3*257*257*4,ge),at=S(12,ne);e.queue.writeBuffer(at,0,new Uint32Array([3,256,257]));let Z=S(Te,le),pe=S(Te,ve),Ce=S(Te,ge),nt=S(3072*64*4,L),it=S(3072*32*4,L),rt=S(1536*16*4,L),I=S(6144*64*4,ge),Q=S(260,ve),j=S(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);S(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let ie=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ue=S(8,ne);e.queue.writeBuffer(Ue,0,new Uint32Array([256,257]));let st=s.get("backbone1.1.weight"),Nt=s.get("backbone1.1.bias");if(!st||!Nt)throw new Error("Missing input conv weights");let N=w(st),qt=w(Nt),d=S(N.byteLength,L),n=S(qt.byteLength,L),g=S(28,ne);e.queue.writeBuffer(d,0,N),e.queue.writeBuffer(n,0,qt),e.queue.writeBuffer(g,0,new Uint32Array([1,3,24,257,257,128,128]));let G=s.get("backbone6.1.weight"),f=s.get("backbone6.1.bias");if(!G||!f)throw new Error("Missing backbone6.1 conv1x1 weights");let r=w(G),O=w(f),ee=S(r.byteLength,L),m=S(O.byteLength,L),U=S(20,ne);e.queue.writeBuffer(ee,0,r),e.queue.writeBuffer(m,0,O),e.queue.writeBuffer(U,0,new Uint32Array([1,96,48,32,32]));let y=s.get("handflag.weight"),F=s.get("handflag.bias");if(!y||!F)throw new Error("Missing handflag weights");let M=w(y),fe=w(F),re=S(M.byteLength,L),B=S(fe.byteLength,L),q=S(12,ne);e.queue.writeBuffer(re,0,M),e.queue.writeBuffer(B,0,fe),e.queue.writeBuffer(q,0,new Uint32Array([1,288,1]));let z=s.get("handedness.weight"),xe=s.get("handedness.bias");if(!z||!xe)throw new Error("Missing handedness weights");let xt=w(z),xa=w(xe),ta=S(xt.byteLength,L),aa=S(xa.byteLength,L),va=S(12,ne);e.queue.writeBuffer(ta,0,xt),e.queue.writeBuffer(aa,0,xa),e.queue.writeBuffer(va,0,new Uint32Array([1,288,1]));let Pa=s.get("reg_3d.weight"),Ba=s.get("reg_3d.bias");if(!Pa||!Ba)throw new Error("Missing reg_3d weights");let ka=w(Pa),Ca=w(Ba),na=S(ka.byteLength,L),ia=S(Ca.byteLength,L),Ua=S(12,ne);e.queue.writeBuffer(na,0,ka),e.queue.writeBuffer(ia,0,Ca),e.queue.writeBuffer(Ua,0,new Uint32Array([1,288,63]));let ot=sn.map(i=>{let{inCh:c,outCh:u,h:o,w:a,stride:E,prefix:D}=i,W=E===2?o/2:o,C=E===2?a/2:a,h=E===2?1:2,p=s.get(`${D}convs.0.weight`),K=s.get(`${D}convs.0.bias`),te=s.get(`${D}convs.1.weight`),se=s.get(`${D}convs.1.bias`);if(!p||!K||!te||!se)throw new Error(`Missing weights for ${D}`);let Bt=w(p),kt=w(K),Ct=w(te),Ut=w(se),At=S(Bt.byteLength,L),Gt=S(kt.byteLength,L),Ve=S(Ct.byteLength,L),Fe=S(Ut.byteLength,L),Ht=S(32,ne),Tt=S(36,ne);return e.queue.writeBuffer(At,0,Bt),e.queue.writeBuffer(Gt,0,kt),e.queue.writeBuffer(Ve,0,Ct),e.queue.writeBuffer(Fe,0,Ut),e.queue.writeBuffer(Ht,0,new Uint32Array([1,c,o,a,W,C,E,h])),e.queue.writeBuffer(Tt,0,new Uint32Array([1,c,u,W,C,Math.max(0,u-c),E,o,a])),{dwWeight:At,dwBias:Gt,pwWeight:Ve,pwBias:Fe,dwUniform:Ht,pwUniform:Tt,spec:i,outH:W,outW:C}});function vt(i){let c=S(i.length*4,ne);return e.queue.writeBuffer(c,0,new Uint32Array(i)),c}let vn=vt([1,96,8,8,16,16]),Pn=vt([1,96,16,16,32,32]),Bn=vt([1,48,32,32,64,64]);vt([1536*16]),vt([3072*32]),vt([3072*64]);let Aa=X(ye,[tt,De,at]),Ga=X(Pe,[De,d,n,Z,g]),Le=[],Re=[],Oe=[],Ie=[];for(let i of ot)Le.push(X(Ae,[Z,i.dwWeight,i.dwBias,Ce,i.dwUniform])),Re.push(X(be,[Ce,Z,i.pwWeight,i.pwBias,pe,i.pwUniform])),Oe.push(X(Ae,[pe,i.dwWeight,i.dwBias,Ce,i.dwUniform])),Ie.push(X(be,[Ce,pe,i.pwWeight,i.pwBias,Z,i.pwUniform]));let kn=X(Ge,[Z,rt,pe,vn]),Cn=X(Ge,[Z,it,pe,Pn]),Un=X(We,[Z,ee,m,I,U]),An=X(Ge,[I,nt,pe,Bn]);X(Se,[Z,re,B,Q,q]),X(Se,[Z,ta,aa,Q,va]),X(Se,[Z,na,ia,Q,Ua]);let Ye=X(qe,[ie.createView(),De,Ue]),Gn=X(lt,[Z,re,B,ta,aa,na,ia,Q]),ra=24,Sa=[],Da=[];for(let i=ra;i<ot.length;i++){let c=ot[i];Sa.push(X(mt,[Z,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,pe,c.dwUniform])),Da.push(X(mt,[pe,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,Z,c.dwUniform]))}let sa=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});sa.globalCompositeOperation="copy";let Ma=new OffscreenCanvas(9,8),$t=Ma.getContext("webgpu"),Yt=null,oa=null;if($t){$t.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let i=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),c=e.createShaderModule({code:an}),u=e.createShaderModule({code:nn});Yt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),vertex:{module:c,entryPoint:"vs"},fragment:{module:u,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),oa=e.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:Q}}]})}let Xt=new Float32Array(1),Vt=new Float32Array(1),Zt=new Float32Array(63);function ze(i,c){let u=!0,o=0,a=i.beginComputePass();for(a.setPipeline($e),a.setBindGroup(0,Ga),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);o<=Nn;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Be[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let E=u?Z:pe;for(i.copyBufferToBuffer(E,0,nt,0,3072*64*4),a=i.beginComputePass();o<=qn;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Be[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let D=u?Z:pe;for(i.copyBufferToBuffer(D,0,it,0,3072*32*4),a=i.beginComputePass();o<=$n;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Be[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let W=u?Z:pe;for(i.copyBufferToBuffer(W,0,rt,0,1536*16*4),a=i.beginComputePass();o<=Yn;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Be[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.setPipeline(He),a.setBindGroup(0,kn),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),u=!1,a=i.beginComputePass();{let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Be[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}a.setPipeline(He),a.setBindGroup(0,Cn),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),u=!1,a=i.beginComputePass();{let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Be[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}for(a.setPipeline(Kt),a.setBindGroup(0,Un),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(He),a.setBindGroup(0,An),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),u=!1,a=i.beginComputePass();o<ra;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Be[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}for(;o<ot.length;o++){let C=o-ra,h=u?Sa[C]:Da[C],p=ot[o];a.setPipeline(et),a.setBindGroup(0,h),a.dispatchWorkgroups(p.outW,p.outH,1),u=!u}a.setPipeline(yt),a.setBindGroup(0,Gn),a.dispatchWorkgroups(1),a.end(),c&&i.copyBufferToBuffer(Q,0,c,0,260)}async function jt(i){e.queue.writeBuffer(tt,0,i);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(gt),a.setBindGroup(0,Aa),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),a.end()}ze(c,j),e.queue.submit([c.finish()]);let u=j.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(j.getMappedRange());return Xt[0]=o[0],Vt[0]=o[1],Zt.set(o.subarray(2,65)),j.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function ua(i){e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(ke),a.setBindGroup(0,Ye),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}ze(c,j),e.queue.submit([c.finish()]);let u=j.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(j.getMappedRange());return Xt[0]=o[0],Vt[0]=o[1],Zt.set(o.subarray(2,65)),j.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function Ea(i){if(!Yt||!oa||!$t)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);let c=e.createCommandEncoder();{let C=c.beginComputePass();C.setPipeline(ke),C.setBindGroup(0,Ye),C.dispatchWorkgroups(16,16,1),C.end()}ze(c,null);let u=$t.getCurrentTexture(),o=c.beginRenderPass({colorAttachments:[{view:u.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});o.setPipeline(Yt),o.setBindGroup(0,oa),o.draw(3),o.end(),e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),sa.drawImage(Ma,0,0);let E=sa.getImageData(0,0,9,8).data,D=new Float32Array(65),W=new DataView(new ArrayBuffer(4));for(let C=0;C<65;C++){let h=C*4;W.setUint8(0,E[h]),W.setUint8(1,E[h+1]),W.setUint8(2,E[h+2]),W.setUint8(3,E[h+3]),D[C]=W.getFloat32(0)}return{handflag:new Float32Array([D[0]]),handedness:new Float32Array([D[1]]),landmarks:new Float32Array(D.subarray(2,65))}}let Sn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),pa=0,Dn=[j,Sn],Pt=null,Xe=null;async function ca(i){let c=Dn[pa];pa=1-pa,e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);let u=e.createCommandEncoder();{let a=u.beginComputePass();a.setPipeline(ke),a.setBindGroup(0,Ye),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}ze(u,c),e.queue.submit([u.finish()]);let o=null;if(Pt!==null&&Xe!==null){await Pt;let a=new Float32Array(Xe.getMappedRange());o={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))},Xe.unmap()}return Xe=c,Pt=c.mapAsync(GPUMapMode.READ),o}async function Wa(){if(!Pt||!Xe)return null;await Pt;let i=new Float32Array(Xe.getMappedRange()),c={handflag:new Float32Array([i[0]]),handedness:new Float32Array([i[1]]),landmarks:new Float32Array(i.subarray(2,65))};return Xe.unmap(),Pt=null,Xe=null,c}async function Mn(i=50){let c=new Float32Array(196608);for(let a=0;a<5;a++)await jt(c);let u=[];for(let a=0;a<i;a++){let E=performance.now();await jt(c),u.push(performance.now()-E)}let o=u.reduce((a,E)=>a+E,0)/u.length;return{avgMs:o,fps:1e3/o}}async function En(i=50){let c=new Float32Array(196608);for(let D=0;D<5;D++)await jt(c);let u=[];for(let D=0;D<i;D++){let W=e.createCommandEncoder();{let h=W.beginComputePass();h.setPipeline(gt),h.setBindGroup(0,Aa),h.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),h.end()}ze(W,j);let C=performance.now();e.queue.submit([W.finish()]),await e.queue.onSubmittedWorkDone(),u.push(performance.now()-C)}u.sort((D,W)=>D-W);let o=u.reduce((D,W)=>D+W,0)/u.length,a=u[Math.floor(u.length/2)],E=u[0];return{avgMs:o,fps:1e3/o,medianMs:a,minMs:E}}function ai(i){e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);let c=e.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(ke),u.setBindGroup(0,Ye),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),u.end()}ze(c,j),e.queue.submit([c.finish()])}async function Wn(i,c=50){function u(h){let p=[...h].sort((K,te)=>K-te);return{median:p[Math.floor(p.length/2)],min:p[0]}}for(let h=0;h<10;h++)await ua(i);let o=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let te=p.beginComputePass();te.setPipeline(ke),te.setBindGroup(0,Ye),te.dispatchWorkgroups(16,16,1),te.end()}ze(p,j);let K=performance.now();e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-K)}let a=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let se=p.beginComputePass();se.setPipeline(ke),se.setBindGroup(0,Ye),se.dispatchWorkgroups(16,16,1),se.end()}ze(p,j),e.queue.submit([p.finish()]);let K=j.mapAsync(GPUMapMode.READ),te=performance.now();await e.queue.onSubmittedWorkDone(),await K,j.getMappedRange(),j.unmap(),a.push(performance.now()-te)}let E=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let te=p.beginComputePass();te.setPipeline(ke),te.setBindGroup(0,Ye),te.dispatchWorkgroups(16,16,1),te.end()}ze(p,j),e.queue.submit([p.finish()]);let K=performance.now();await j.mapAsync(GPUMapMode.READ),j.getMappedRange(),j.unmap(),E.push(performance.now()-K)}let D=[];for(let h=0;h<c;h++){let p=performance.now();await ua(i),D.push(performance.now()-p)}await ca(i);let W=[];for(let h=0;h<c;h++){let p=performance.now();await ca(i),W.push(performance.now()-p)}await Wa();let C=null;if(Yt){let h=[];for(let p=0;p<c;p++){let K=performance.now();await Ea(i),h.push(performance.now()-K)}C=u(h)}return{gpuOnly:u(o),mapAsyncOnly:u(a),mapAsyncNoWait:u(E),total:u(D),pipelined:u(W),renderReadback:C}}async function Hn(i){let c=[];async function u(a,E,D){let W=e.createBuffer({size:E,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),C=e.createCommandEncoder();C.copyBufferToBuffer(a,0,W,0,E),e.queue.submit([C.finish()]),await e.queue.onSubmittedWorkDone(),await W.mapAsync(GPUMapMode.READ);let h=new Float32Array(W.getMappedRange()),p=1/0,K=-1/0,te=0;for(let se=0;se<h.length;se++)h[se]<p&&(p=h[se]),h[se]>K&&(K=h[se]),h[se]!==0&&te++;W.unmap(),W.destroy(),c.push({layer:D,stats:{min:p,max:K,nonZero:te,total:h.length}})}e.queue.copyExternalImageToTexture({source:i},{texture:ie},[256,256]);{let a=e.createCommandEncoder(),E=a.beginComputePass();E.setPipeline(ke),E.setBindGroup(0,Ye),E.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),E.end(),e.queue.submit([a.finish()])}await u(De,Math.min(De.size,3*257*257*4),"canvas\u2192bufInput");{let a=e.createCommandEncoder(),E=a.beginComputePass();E.setPipeline($e),E.setBindGroup(0,Ga),E.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),E.end(),e.queue.submit([a.finish()])}await u(Z,Math.min(Z.size,3072*128*4),"inputConv\u2192bufA");let o=!0;for(let a=0;a<Math.min(ot.length,6);a++){let E=o?Le[a]:Oe[a],D=o?Re[a]:Ie[a],W=Be[a],C=ot[a];{let p=e.createCommandEncoder(),K=p.beginComputePass();K.setPipeline(W.dwPipeline),K.setBindGroup(0,E),K.dispatchWorkgroups(W.dwDispatchX,W.dwDispatchY,W.dwDispatchZ),K.end(),e.queue.submit([p.finish()])}await u(Ce,Math.min(Ce.size,C.spec.inCh*C.outH*C.outW*4),`layer${a}.DW\u2192bufDW (${C.spec.prefix})`);{let p=e.createCommandEncoder(),K=p.beginComputePass();K.setPipeline(W.pwPipeline),K.setBindGroup(0,D),K.dispatchWorkgroups(W.pwDispatchX,W.pwDispatchY,W.pwDispatchZ),K.end(),e.queue.submit([p.finish()])}let h=o?pe:Z;await u(h,Math.min(h.size,C.spec.outCh*C.outH*C.outW*4),`layer${a}.PW\u2192buf${o?"B":"A"} (${C.spec.prefix})`),o=!o}return c}return{device:e,run:jt,runFromCanvas:ua,runFromCanvasViaRender:Ea,runFromCanvasPipelined:ca,flushPipelined:Wa,benchmark:Mn,benchmarkGPU:En,benchmarkDiagnostic:Wn,debugLayerOutputs:Hn}}function ut(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var on=ut(`
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
`),un=ut(`
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
`),pn=ut(`
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
`),cn=ut(`
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
`),dn=ut(`
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
`),_n=ut(`
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
`),ln=ut(`
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
`);async function mn(s,_){let t;if(_)t=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");t=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8)}})}let k={r:"read-only-storage",s:"storage",u:"uniform"};function b(d){return t.createBindGroupLayout({entries:d.map((n,g)=>n==="t"?{binding:g,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:g,visibility:GPUShaderStage.COMPUTE,buffer:{type:k[n]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,P=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,R=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,H=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(d,n){return t.createBuffer({size:Math.max(d,4),usage:n})}function w(d,n,g){t.queue.writeBuffer(d,n,g)}function A(d){let n=l(d.data.byteLength,e);return w(n,0,d.data),n}let T=Array.from(s.keys());function ce(d){let n=s.get(d);if(!n)throw new Error(`Weight not found: ${d}`);return n}function x(...d){let n=T.find(g=>d.every(G=>g.includes(G)));if(!n)throw new Error(`Weight not found for: ${d.join(", ")}`);return ce(n)}function ae(d){let[,n,g,G]=d.shape,f=new Float32Array(G*25);for(let r=0;r<G;r++)for(let O=0;O<n;O++)for(let ee=0;ee<g;ee++)f[r*25+O*5+ee]=d.data[O*g*G+ee*G+r];return f}function L(d){let[n,,,g]=d.shape,G=new Float32Array(n*g);for(let f=0;f<n;f++)for(let r=0;r<g;r++)G[f*g+r]=d.data[f*g+r];return G}let le=t.createShaderModule({code:on}),ge=t.createShaderModule({code:un}),ve=t.createShaderModule({code:pn}),ne=t.createShaderModule({code:cn}),S=t.createShaderModule({code:_n}),X=t.createShaderModule({code:dn}),oe=t.createShaderModule({code:ln}),Me=b(["r","r","r","r","s","u"]),pt=b(["r","r","r","s","u"]),ct=b(["r","r","r","r","r","s","u"]),dt=b(["r","r","r","s","u"]),_t=b(["r","r","r","r","s","u"]),he=b(["r","r","s","u"]),Ze=b(["t","s","u"]);function v(d,n){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:n,entryPoint:"main"}})}let $=v(Me,le),de=v(pt,ge),V=v(ct,ve),ue=v(dt,ne),J=v(_t,S),Y=v(he,X),Ee=v(Ze,oe),we=x("conv2d/Conv2D"),Ke=x("batch_normalization/","conv2d/Conv2D"),Ae=x("p_re_lu/"),be=A(we),ye=A(Ke),Pe=A(Ae),me=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(d=>{let n=x(d.dwKey),g=x(d.pwKey),G=x(d.bnKey),f=x(d.preluKey),r=ae(n),O=l(r.byteLength,e);w(O,0,r);let ee=new Float32Array(d.inCh),m=l(ee.byteLength,e);w(m,0,ee);let U=L(g),y=l(U.byteLength,e);w(y,0,U);let F=A(G),M=A(f);return{dwWeightBuf:O,dwBiasBuf:m,pwWeightBuf:y,pwBiasBuf:F,alphaBuf:M,inCh:d.inCh,outCh:d.outCh,stride:d.stride,inH:d.inH}}),Ge=L(x("conv2d_20/Conv2D")),We=l(Ge.byteLength,e);w(We,0,Ge);let Se=A(x("batch_normalization_20/")),qe=A(x("p_re_lu_20/")),lt={dwWeightBuf:(()=>{let d=ae(x("depthwise_conv2d_19/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=L(x("conv2d_21/")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:A(x("batch_normalization_21/")),alphaBuf:A(x("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},mt={dwWeightBuf:(()=>{let d=ae(x("depthwise_conv2d_20/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=L(x("conv2d_22/Conv2D1")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:A(x("batch_normalization_22/")),alphaBuf:A(x("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},Rt=L(x("conv2d_23/Conv2D")),St=l(Rt.byteLength,e);w(St,0,Rt);let je=A(x("batch_normalization_23/")),Je=A(x("p_re_lu_23/")),Ot={dwWeightBuf:(()=>{let d=ae(x("depthwise_conv2d_21/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=L(x("conv2d_24/")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:A(x("batch_normalization_24/")),alphaBuf:A(x("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},It={dwWeightBuf:(()=>{let d=ae(x("depthwise_conv2d_22/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=L(x("conv2d_25/Conv2D1")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:A(x("batch_normalization_25/")),alphaBuf:A(x("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},zt=L(x("classifier_palm_16_NO_PRUNING/Conv2D")),ft=l(zt.byteLength,e);w(ft,0,zt);let ht=A(x("classifier_palm_16_NO_PRUNING/BiasAdd")),Dt=L(x("regressor_palm_16_NO_PRUNING/Conv2D")),wt=l(Dt.byteLength,e);w(wt,0,Dt);let Mt=A(x("regressor_palm_16_NO_PRUNING/BiasAdd")),Qe=L(x("classifier_palm_8_NO_PRUNING/Conv2D")),bt=l(Qe.byteLength,e);w(bt,0,Qe);let Et=A(x("classifier_palm_8_NO_PRUNING/BiasAdd")),Ft=L(x("regressor_palm_8_NO_PRUNING/Conv2D")),Wt=l(Ft.byteLength,e);w(Wt,0,Ft);let Be=A(x("regressor_palm_8_NO_PRUNING/BiasAdd")),gt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,$e=l(36864*3*4,e),He=l(gt,P),Kt=l(gt,P),ke=l(gt,P),yt=l(576*256*4,P),et=l(144*256*4,P|GPUBufferUsage.COPY_DST),Te=l(576*128*4,P|GPUBufferUsage.COPY_DST),tt=l(864*4,R),De=l(15552*4,R),at=l(576*2*4,R),Z=l(576*36*4,R),pe=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ce=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),nt=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),it=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),rt=t.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function I(d,n){return Math.ceil(d/n)}function Q(d){let n=l(d.byteLength,H);return w(n,0,d),n}let j=Q(new Uint32Array([1,3,32,192,192,96,96]));function ie(d,n,g,G,f){let r=n.stride===2?n.inH/2:n.inH,O=r,ee=n.stride===2?1:2,m=Q(new Uint32Array([1,n.inCh,n.inH,n.inH,r,O,n.stride,ee])),U=t.createBindGroup({layout:pt,entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:n.dwWeightBuf}},{binding:2,resource:{buffer:n.dwBiasBuf}},{binding:3,resource:{buffer:ke}},{binding:4,resource:{buffer:m}}]}),y=d.beginComputePass();y.setPipeline(de),y.setBindGroup(0,U),y.dispatchWorkgroups(I(O,8),I(r,8),n.inCh),y.end();let F=n.inCh,M=Q(new Uint32Array([1,n.inCh,n.outCh,r,O,F,n.stride,n.inH,n.inH])),fe=t.createBindGroup({layout:ct,entries:[{binding:0,resource:{buffer:ke}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:n.pwWeightBuf}},{binding:3,resource:{buffer:n.pwBiasBuf}},{binding:4,resource:{buffer:n.alphaBuf}},{binding:5,resource:{buffer:G}},{binding:6,resource:{buffer:M}}]}),re=d.beginComputePass();re.setPipeline(V),re.setBindGroup(0,fe),re.dispatchWorkgroups(I(O,8),I(r,8),n.outCh),re.end()}function Ue(d,n,g,G,f,r,O,ee,m){let U=Q(new Uint32Array([1,r,O,ee,m])),y=t.createBindGroup({layout:dt,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:g}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:U}}]}),F=d.beginComputePass();F.setPipeline(ue),F.setBindGroup(0,y),F.dispatchWorkgroups(I(m,8),I(ee,8),O),F.end()}function st(d,n,g,G,f,r,O,ee,m,U){let y=Q(new Uint32Array([1,O,ee,m,U])),F=t.createBindGroup({layout:_t,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:g}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:r}},{binding:5,resource:{buffer:y}}]}),M=d.beginComputePass();M.setPipeline(J),M.setBindGroup(0,F),M.dispatchWorkgroups(I(U,8),I(m,8),ee),M.end()}async function Nt(d){t.queue.copyExternalImageToTexture({source:d},{texture:rt},[192,192]);let n=Q(new Uint32Array([192,192,192])),g=t.createBindGroup({layout:Ze,entries:[{binding:0,resource:rt.createView()},{binding:1,resource:{buffer:$e}},{binding:2,resource:{buffer:n}}]}),G=t.createCommandEncoder();{let B=G.beginComputePass();B.setPipeline(Ee),B.setBindGroup(0,g),B.dispatchWorkgroups(I(192,16),I(192,16),1),B.end()}{let B=t.createBindGroup({layout:Me,entries:[{binding:0,resource:{buffer:$e}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:ye}},{binding:3,resource:{buffer:Pe}},{binding:4,resource:{buffer:He}},{binding:5,resource:{buffer:j}}]}),q=G.beginComputePass();q.setPipeline($),q.setBindGroup(0,B),q.dispatchWorkgroups(I(96,8),I(96,8),32),q.end()}let f=He,r=Kt;for(let B=0;B<me.length;B++){let q=me[B];ie(G,q,f,r,f);let z=f;f=r,r=z,B===10&&G.copyBufferToBuffer(f,0,Te,0,576*128*4),B===14&&G.copyBufferToBuffer(f,0,et,0,144*256*4)}{let B=Q(new Uint32Array([1,256,6,6,12,12])),q=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(Y),z.setBindGroup(0,q),z.dispatchWorkgroups(I(12,8),I(12,8),256),z.end()}{let B=f;f=r,r=B}st(G,f,We,Se,qe,r,256,256,12,12);{let B=f;f=r,r=B}{let B=Q(new Uint32Array([1,256,12,12,12,12])),q=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:et}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(Y),z.setBindGroup(0,q),z.dispatchWorkgroups(I(12,8),I(12,8),256),z.end()}{let B=f;f=r,r=B}ie(G,lt,f,r,f);{let B=f;f=r,r=B}ie(G,mt,f,r,f);{let B=f;f=r,r=B}Ue(G,f,ft,ht,tt,256,6,12,12),Ue(G,f,wt,Mt,De,256,108,12,12);{let B=Q(new Uint32Array([1,256,12,12,24,24])),q=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(Y),z.setBindGroup(0,q),z.dispatchWorkgroups(I(24,8),I(24,8),256),z.end()}{let B=f;f=r,r=B}st(G,f,St,je,Je,r,256,128,24,24);{let B=f;f=r,r=B}{let B=Q(new Uint32Array([1,128,24,24,24,24])),q=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Te}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(Y),z.setBindGroup(0,q),z.dispatchWorkgroups(I(24,8),I(24,8),128),z.end()}{let B=f;f=r,r=B}ie(G,Ot,f,r,f);{let B=f;f=r,r=B}ie(G,It,f,r,f);{let B=f;f=r,r=B}Ue(G,f,bt,Et,at,128,2,24,24),Ue(G,f,Wt,Be,Z,128,36,24,24),t.queue.submit([G.finish()]);let O=t.createCommandEncoder();O.copyBufferToBuffer(tt,0,pe,0,864*4),O.copyBufferToBuffer(De,0,Ce,0,15552*4),O.copyBufferToBuffer(at,0,nt,0,576*2*4),O.copyBufferToBuffer(Z,0,it,0,576*36*4),t.queue.submit([O.finish()]),await Promise.all([pe.mapAsync(GPUMapMode.READ),Ce.mapAsync(GPUMapMode.READ),nt.mapAsync(GPUMapMode.READ),it.mapAsync(GPUMapMode.READ)]);let ee=new Float32Array(pe.getMappedRange()).slice(),m=new Float32Array(Ce.getMappedRange()).slice(),U=new Float32Array(nt.getMappedRange()).slice(),y=new Float32Array(it.getMappedRange()).slice();pe.unmap(),Ce.unmap(),nt.unmap(),it.unmap();let F=2016,M=new Float32Array(F),fe=new Float32Array(F*18),re=0;for(let B=0;B<12;B++)for(let q=0;q<12;q++)for(let z=0;z<6;z++){M[re]=ee[z*144+B*12+q];for(let xe=0;xe<18;xe++){let xt=z*18+xe;fe[re*18+xe]=m[xt*144+B*12+q]}re++}for(let B=0;B<24;B++)for(let q=0;q<24;q++)for(let z=0;z<2;z++){M[re]=U[z*576+B*24+q];for(let xe=0;xe<18;xe++){let xt=z*18+xe;fe[re*18+xe]=y[xt*576+B*24+q]}re++}return{scores:M,regressors:fe}}async function N(d,n){let g=t.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),G=t.createCommandEncoder();G.copyBufferToBuffer(d,0,g,0,n*4),t.queue.submit([G.finish()]),await g.mapAsync(GPUMapMode.READ);let f=new Float32Array(g.getMappedRange()).slice();return g.unmap(),g.destroy(),f}async function qt(d){t.queue.copyExternalImageToTexture({source:d},{texture:rt},[192,192]);function n(y,F=1e3){let M=y.slice(0,F);return{min:Math.min(...M),max:Math.max(...M),mean:M.reduce((fe,re)=>fe+re,0)/M.length,nonZero:M.filter(fe=>fe!==0).length,sample:Array.from(M.slice(0,10))}}let g={},G=Q(new Uint32Array([192,192,192])),f=t.createBindGroup({layout:Ze,entries:[{binding:0,resource:rt.createView()},{binding:1,resource:{buffer:$e}},{binding:2,resource:{buffer:G}}]}),r=t.createCommandEncoder(),O=r.beginComputePass();O.setPipeline(Ee),O.setBindGroup(0,f),O.dispatchWorkgroups(I(192,16),I(192,16),1),O.end(),t.queue.submit([r.finish()]),g.input=n(await N($e,36864*3)),r=t.createCommandEncoder();let ee=t.createBindGroup({layout:Me,entries:[{binding:0,resource:{buffer:$e}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:ye}},{binding:3,resource:{buffer:Pe}},{binding:4,resource:{buffer:He}},{binding:5,resource:{buffer:j}}]});O=r.beginComputePass(),O.setPipeline($),O.setBindGroup(0,ee),O.dispatchWorkgroups(I(96,8),I(96,8),32),O.end(),t.queue.submit([r.finish()]),g.initConv=n(await N(He,9216*32));let m=He,U=Kt;for(let y=0;y<me.length;y++){let F=me[y];r=t.createCommandEncoder(),ie(r,F,m,U,m),t.queue.submit([r.finish()]);let M=m;if(m=U,U=M,y===0||y===3||y===7||y===11||y===14||y===15||y===18){let fe=F.stride===2?F.inH/2:F.inH,re=fe*fe*F.outCh;g[`block${y}`]=n(await N(m,re))}y===10&&(r=t.createCommandEncoder(),r.copyBufferToBuffer(m,0,Te,0,576*128*4),t.queue.submit([r.finish()])),y===14&&(r=t.createCommandEncoder(),r.copyBufferToBuffer(m,0,et,0,144*256*4),t.queue.submit([r.finish()]))}r=t.createCommandEncoder();{let y=Q(new Uint32Array([1,256,6,6,12,12])),F=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:y}}]}),M=r.beginComputePass();M.setPipeline(Y),M.setBindGroup(0,F),M.dispatchWorkgroups(I(12,8),I(12,8),256),M.end()}t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpnUpsample6to12=n(await N(m,144*256)),r=t.createCommandEncoder(),st(r,m,We,Se,qe,U,256,256,12,12),t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpn6to12Conv=n(await N(m,144*256)),g.backbone12Skip=n(await N(et,144*256)),r=t.createCommandEncoder();{let y=Q(new Uint32Array([1,256,12,12,12,12])),F=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:et}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:y}}]}),M=r.beginComputePass();M.setPipeline(Y),M.setBindGroup(0,F),M.dispatchWorkgroups(I(12,8),I(12,8),256),M.end()}t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpnAdd12=n(await N(m,144*256)),r=t.createCommandEncoder(),ie(r,lt,m,U,m),t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpn12Block1=n(await N(m,144*256)),r=t.createCommandEncoder(),ie(r,mt,m,U,m),t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpn12Block2=n(await N(m,144*256)),r=t.createCommandEncoder(),Ue(r,m,ft,ht,tt,256,6,12,12),t.queue.submit([r.finish()]),g.cls16=n(await N(tt,864)),r=t.createCommandEncoder(),Ue(r,m,wt,Mt,De,256,108,12,12),t.queue.submit([r.finish()]),g.reg16=n(await N(De,15552),500),r=t.createCommandEncoder();{let y=Q(new Uint32Array([1,256,12,12,24,24])),F=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:y}}]}),M=r.beginComputePass();M.setPipeline(Y),M.setBindGroup(0,F),M.dispatchWorkgroups(I(24,8),I(24,8),256),M.end()}t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpnUpsample12to24=n(await N(m,576*256)),r=t.createCommandEncoder(),st(r,m,St,je,Je,U,256,128,24,24),t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpn12to24Conv=n(await N(m,576*128)),g.backbone24Skip=n(await N(Te,576*128)),r=t.createCommandEncoder();{let y=Q(new Uint32Array([1,128,24,24,24,24])),F=t.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Te}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:y}}]}),M=r.beginComputePass();M.setPipeline(Y),M.setBindGroup(0,F),M.dispatchWorkgroups(I(24,8),I(24,8),128),M.end()}t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpnAdd24=n(await N(m,576*128)),r=t.createCommandEncoder(),ie(r,Ot,m,U,m),t.queue.submit([r.finish()]);{let y=m;m=U,U=y}g.fpn24Block1=n(await N(m,576*128)),r=t.createCommandEncoder(),ie(r,It,m,U,m),t.queue.submit([r.finish()]);{let y=m;m=U,U=y}return g.fpn24Block2=n(await N(m,576*128)),r=t.createCommandEncoder(),Ue(r,m,bt,Et,at,128,2,24,24),t.queue.submit([r.finish()]),g.cls8=n(await N(at,576*2)),r=t.createCommandEncoder(),Ue(r,m,Wt,Be,Z,128,36,24,24),t.queue.submit([r.finish()]),g.reg8=n(await N(Z,576*36)),g.initWeights=n(await N(be,100),100),g.initBias=n(await N(ye,32),32),g.cls16Weights=n(await N(ft,100),100),g.cls16Bias=n(await N(ht,6),6),g.cls8Weights=n(await N(bt,100),100),g.cls8Bias=n(await N(Et,2),2),g.fpn6to12Weights=n(await N(We,100),100),g}return{device:t,run:Nt,debugRun:qt}}function Xn(){let s=[];for(let _=0;_<12;_++)for(let t=0;t<12;t++){let k=(t+.5)/12,b=(_+.5)/12;for(let e=0;e<6;e++)s.push({x:k,y:b})}for(let _=0;_<24;_++)for(let t=0;t<24;t++){let k=(t+.5)/24,b=(_+.5)/24;for(let e=0;e<2;e++)s.push({x:k,y:b})}return s}var fn=Xn();function Vn(s){return 1/(1+Math.exp(-s))}function hn(s,_){let t=[],{scores:k,regressors:b}=s,e=192;for(let P=0;P<fn.length;P++){let R=Vn(k[P]);if(R<_)continue;let H=fn[P],l=P*18,w=H.x+b[l+0]/e,A=H.y+b[l+1]/e,T=b[l+2]/e,ce=b[l+3]/e,x=[];for(let ae=0;ae<7;ae++){let L=H.x+b[l+4+ae*2]/e,le=H.y+b[l+4+ae*2+1]/e;x.push([L,le])}t.push({score:R,box:[w,A,T,ce],keypoints:x})}return t}function wn(s,_){if(s.length===0)return[];let t=[...s].sort((e,P)=>P.score-e.score),k=[],b=new Set;for(let e=0;e<t.length;e++)if(!b.has(e)){k.push(t[e]);for(let P=e+1;P<t.length;P++)b.has(P)||Zn(t[e],t[P])>_&&b.add(P)}return k}function Zn(s,_){let t=s.box[0]-s.box[2]/2,k=s.box[1]-s.box[3]/2,b=s.box[0]+s.box[2]/2,e=s.box[1]+s.box[3]/2,P=_.box[0]-_.box[2]/2,R=_.box[1]-_.box[3]/2,H=_.box[0]+_.box[2]/2,l=_.box[1]+_.box[3]/2,w=Math.max(t,P),A=Math.max(k,R),T=Math.min(b,H),ce=Math.min(e,l),x=Math.max(0,T-w),ae=Math.max(0,ce-A),L=x*ae,le=(b-t)*(e-k),ge=(H-P)*(l-R),ve=le+ge-L;return ve>0?L/ve:0}function jn(s){let[_,t,k,b]=s.box,e=s.keypoints[0],P=s.keypoints[2],R=P[0]-e[0],H=P[1]-e[1],l=Math.atan2(H,R),A=-Math.PI/2-l,T=Math.max(k,b),x=T*2.6,ae=-.5*T,L=Math.cos(A),le=Math.sin(A),ge=-ae*le,ve=ae*L;return{centerX:_+ge,centerY:t+ve,width:x,height:x,rotation:A}}function bn(s,_={}){let{scoreThreshold:t=.5,nmsThreshold:k=.3,maxHands:b=2}=_;async function e(R){let H=await s.run(R),l=hn(H,t);return wn(l,k).slice(0,b).map(jn)}async function P(R){let H=await s.run(R),l=hn(H,t);return wn(l,k).slice(0,b)}return{detect:e,detectRaw:P,model:s}}function gn(s,_,t,k){let b=Math.cos(_.rotation),e=Math.sin(_.rotation),P=Math.min(t,k),R=_.width*P,H=R/t,l=R/k;return s.map(w=>{let A=w.x-.5,T=w.y-.5;return{x:H*(b*A+e*T)+_.centerX,y:l*(-e*A+b*T)+_.centerY,z:w.z}})}var wa=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function ba(s){let _={};for(let t=0;t<wa.length;t++)_[wa[t]]=s[t];return _}function yn(s,_,t){return s.initialized?(s.value=t*_+(1-t)*s.value,s.value):(s.value=_,s.initialized=!0,_)}function xn(s,_){let t=2*Math.PI*_*s;return t/(t+1)}function Jn(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function ga(s,_,t,k,b,e){let P=s.lastTime<0?.03333333333333333:t-s.lastTime;s.lastTime=t;let R=xn(P,e),H=s.x.initialized?(_-s.x.value)/P:0,l=yn(s.dx,H,R),w=k+b*Math.abs(l),A=xn(P,w);return yn(s.x,_,A)}function ya(s={}){let{minCutoff:_=.05,beta:t=80,dCutoff:k=1}=s,b=[];function e(H){b.length!==H&&(b=Array.from({length:H},()=>Jn()))}function P(H,l){let w=l??performance.now()/1e3,A=H.length*3;return e(A),H.map((T,ce)=>({x:ga(b[ce*3],T.x,w,_,t,k),y:ga(b[ce*3+1],T.y,w,_,t,k),z:ga(b[ce*3+2],T.z,w,_,t,k)}))}function R(){b=[]}return{apply:P,reset:R}}var Qn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ei(s={}){let{weightsUrl:_,scoreThreshold:t=.5,palmScoreThreshold:k=.5,maxHands:b=3,forceF32:e=!1}=s;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let P=(_??Qn).replace(/\/$/,"")+"/",[R,H,l,w]=await Promise.all([fetch(`${P}weights_f16.json`),fetch(`${P}weights_f16.bin`),fetch(`${P}palm_detection_weights.json`),fetch(`${P}palm_detection_weights.bin`)]);if(!R.ok)throw new Error(`Failed to fetch landmark weights: ${R.status}`);if(!H.ok)throw new Error(`Failed to fetch landmark weights: ${H.status}`);if(!l.ok)throw new Error(`Failed to fetch palm detection weights: ${l.status}`);if(!w.ok)throw new Error(`Failed to fetch palm detection weights: ${w.status}`);let[A,T,ce,x]=await Promise.all([R.json(),H.arrayBuffer(),l.json(),w.arrayBuffer()]),ae=fa(A,T),L=fa(ce,x),le=await ha(ae,{forceF32:e});if(!e){let v=new OffscreenCanvas(256,256),$=v.getContext("2d");$.fillStyle="#886644",$.fillRect(0,0,256,256),$.fillStyle="#cc9966",$.fillRect(50,50,156,156);let de=await le.runFromCanvas(v);de.landmarks.every(ue=>ue===0)&&de.handflag.every(ue=>ue===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),le.device.destroy(),le=await ha(ae,{forceF32:!0}))}let ge=await mn(L),ve=bn(ge,{scoreThreshold:k,maxHands:b}),ne=[];for(let v=0;v<b;v++)ne.push(ya());let S=0,X=null,oe=null;function Me(){return X||(X=new OffscreenCanvas(192,192)),X}function pt(){return oe||(oe=new OffscreenCanvas(256,256)),oe}async function ct(v){if(v instanceof HTMLCanvasElement||v instanceof OffscreenCanvas){if(v.width===192&&v.height===192)return v;let V=Me();return V.width=192,V.height=192,V.getContext("2d").drawImage(v,0,0,192,192),V}if(typeof ImageBitmap<"u"&&v instanceof ImageBitmap){if(v.width===192&&v.height===192)return v;let V=Me();return V.width=192,V.height=192,V.getContext("2d").drawImage(v,0,0,192,192),V}let $=Me();$.width=192,$.height=192;let de=$.getContext("2d");if(v instanceof ImageData){let V=new OffscreenCanvas(v.width,v.height);V.getContext("2d").putImageData(v,0,0),de.drawImage(V,0,0,192,192)}else de.drawImage(v,0,0,192,192);return $}function dt(v,$,de,V){let ue=pt();ue.width=256,ue.height=256;let J=ue.getContext("2d");J.clearRect(0,0,256,256),J.save(),J.translate(128,128),J.scale($.width*de/256,$.height*V/256),J.rotate(-$.rotation),J.translate(-128,-128);let Y=$.centerX*de,Ee=$.centerY*V;J.restore();let we=Math.min(de,V),Ke=256/($.width*we),Ae=256/($.height*we),be=Math.cos($.rotation),ye=Math.sin($.rotation),Pe=be*Ke,Ne=ye*Ke,me=-ye*Ae,Ge=be*Ae,We=-Y*Pe-Ee*me+128,Se=-Y*Ne-Ee*Ge+128;if(J.setTransform(Pe,Ne,me,Ge,We,Se),v instanceof ImageData){let qe=new OffscreenCanvas(v.width,v.height);qe.getContext("2d").putImageData(v,0,0),J.drawImage(qe,0,0)}else J.drawImage(v,0,0);return J.setTransform(1,0,0,1,0,0),ue}function _t(v){return v instanceof HTMLCanvasElement||v instanceof OffscreenCanvas?[v.width,v.height]:typeof ImageBitmap<"u"&&v instanceof ImageBitmap?[v.width,v.height]:v instanceof ImageData?[v.width,v.height]:v instanceof HTMLVideoElement?[v.videoWidth,v.videoHeight]:v instanceof HTMLImageElement?[v.naturalWidth,v.naturalHeight]:[256,256]}async function he(v){let $=await ct(v),de=await ve.detect($);if(de.length===0){if(S>0)for(let Y=0;Y<S&&Y<ne.length;Y++)ne[Y].reset();return S=0,[]}let[V,ue]=_t(v),J=[];for(let Y of de){let Ee=dt(v,Y,V,ue),we=await le.runFromCanvas(Ee),Ke=we.handflag[0];if(Ke<t)continue;let Ae=we.handedness[0]>.5,be=[];for(let me=0;me<21;me++)be.push({x:we.landmarks[me*3],y:we.landmarks[me*3+1],z:we.landmarks[me*3+2]});let ye=gn(be,Y,V,ue),Pe=J.length,Ne=Pe<ne.length?ne[Pe].apply(ye):ye;J.push({score:Ke,handedness:Ae?"right":"left",landmarks:Ne,keypoints:ba(Ne)})}if(J.length<S)for(let Y=J.length;Y<S;Y++)Y<ne.length&&ne[Y].reset();return S=J.length,J}function Ze(){le.device.destroy(),ge.device.destroy(),X=null,oe=null}return{detect:he,dispose:Ze}}export{wa as LANDMARK_NAMES,ei as createHandpose,ya as createLandmarkSmoother,ba as toKeypoints};
