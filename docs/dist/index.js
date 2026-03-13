function be(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function La(s){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],t="enable f16;"+s;for(let C of _)for(;t.includes(`${C}:array<f32>`);)t=t.replace(`${C}:array<f32>`,`${C}:array<f16>`);return t}var da=be(`
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
`),_a=be(`
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
`),la=be(`
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
`),ma=be(`
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
`);function Ra(s,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Oa(s,_){return da.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Ia(s,_){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function za(s,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${s},${_},1)`)}function Fa(s,_){return[8,8]}var Ka=be(`
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
`),Na=be(`
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
`);function qa(s){return be(`
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
`)}var $a=qa(!1),Ya=qa(!0),Xa=be(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Va=be(`
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
`);function Za(s){return be(`
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
`)}var ja=Za("sigmoid"),Ja=Za("div256"),Qa=be(`
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
`),en=be(`
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
`);function tn(s,_){let C=Math.min(_,256),x=_>C,P=s%4===0?`var ic:u32=0u;
    while(ic<${s}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${s}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,I=`var skip_val:f32=0.0;
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
    }`,T=s===_?"":`if(c<${s}u){`,l=s===_?"":"}",g=x?`for(var c:u32=lid.x;c<${s}u;c+=${C}u){`:`let c=lid.x;
  ${T}`,M=x?"}":l,L=x?`for(var c:u32=lid.x;c<${_}u;c+=${C}u){`:"{let c=lid.x;";return be(`
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
@compute @workgroup_size(${C},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${g}
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
  ${M}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${L}
    let pw_base=c*${s}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${P}
    // Skip connection (only for c < inCh)
    ${I}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var an=be(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),nn=be(`
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
`),rn=be(`
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
`);function fa(s,_){let t=new Map,C=s.dtype??"float32";for(let x=0;x<s.keys.length;x++){let e=s.keys[x],P=s.shapes[x],I=s.offsets[x],T=P.reduce((M,L)=>M*L,1),l,g;if(C==="float32")l=new Float32Array(_,I,T);else{let M=new DataView(_);l=new Float32Array(T);for(let L=0;L<T;L++)l[L]=zn(M.getUint16(I+L*2,!0));g=_.slice(I,I+T*2)}t.set(e,{data:l,shape:P,rawF16:g})}return t}function zn(s){let _=s>>15&1,t=s>>10&31,C=s&1023;if(t===0){if(C===0)return _?-0:0;let P=-14,I=C/1024;return(_?-1:1)*Math.pow(2,P)*I}if(t===31)return C===0?_?-1/0:1/0:NaN;let x=t-15,e=1+C/1024;return(_?-1:1)*Math.pow(2,x)*e}var Fn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],sn=Fn.map(([s,_,t,C,x])=>({type:"resmodule",inCh:s,outCh:_,h:t,w:t,stride:C,prefix:x})),Kn=2,Nn=5,qn=8,$n=11;async function ha(s,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No GPU adapter found");let C=t.features.has("shader-f16"),x=C?["shader-f16"]:[],e=await t.requestDevice({requiredFeatures:x,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(t.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(t.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(t.limits.maxComputeInvocationsPerWorkgroup,288)}}),P=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(C)try{let i=`enable f16;
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
}`,u=e.createShaderModule({code:i}),o=e.createShaderModule({code:c}),a=await u.getCompilationInfo(),E=await o.getCompilationInfo();if(a.messages.some(S=>S.type==="error")||E.messages.some(S=>S.type==="error"))P=!1;else{let S=new Float32Array(2400);S.fill(1);let W=new Uint16Array(2400);W.fill(10516);let k=new Uint16Array(96);k.fill(14336);let h=new Uint16Array(9216);h.fill(8478);let p=new Uint16Array(96);p.fill(12288);let N=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ne=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ue=e.createBuffer({size:S.byteLength,usage:N}),Ut=e.createBuffer({size:W.byteLength,usage:N}),At=e.createBuffer({size:k.byteLength,usage:N}),Gt=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),St=e.createBuffer({size:h.byteLength,usage:N}),Dt=e.createBuffer({size:p.byteLength,usage:N}),Mt=e.createBuffer({size:384,usage:ne}),tt=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ue,0,S),e.queue.writeBuffer(Ut,0,W),e.queue.writeBuffer(At,0,k),e.queue.writeBuffer(St,0,h),e.queue.writeBuffer(Dt,0,p);let Ze="read-only-storage",Ot="storage",It=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ot}}]}),Ha=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ot}}]}),Hn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[It]}),compute:{module:u,entryPoint:"main"}}),Tn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Ha]}),compute:{module:o,entryPoint:"main"}}),Ln=e.createBindGroup({layout:It,entries:[{binding:0,resource:{buffer:ue}},{binding:1,resource:{buffer:Ut}},{binding:2,resource:{buffer:At}},{binding:3,resource:{buffer:Gt}}]}),Rn=e.createBindGroup({layout:Ha,entries:[{binding:0,resource:{buffer:Gt}},{binding:1,resource:{buffer:St}},{binding:2,resource:{buffer:Dt}},{binding:3,resource:{buffer:Mt}}]}),Jt=e.createCommandEncoder(),Qt=Jt.beginComputePass();Qt.setPipeline(Hn),Qt.setBindGroup(0,Ln),Qt.dispatchWorkgroups(2),Qt.end();let ea=Jt.beginComputePass();ea.setPipeline(Tn),ea.setBindGroup(0,Rn),ea.dispatchWorkgroups(2),ea.end(),Jt.copyBufferToBuffer(Mt,0,tt,0,384),e.queue.submit([Jt.finish()]),await e.queue.onSubmittedWorkDone(),await tt.mapAsync(GPUMapMode.READ);let zt=new Float32Array(tt.getMappedRange()),Ta=1.5*.0104*96+.25,On=zt[0]!==0&&zt[47]!==0&&zt[95]!==0,In=Math.abs(zt[0]-Ta)<1;P=On&&In,tt.unmap(),ue.destroy(),Ut.destroy(),At.destroy(),Gt.destroy(),St.destroy(),Dt.destroy(),Mt.destroy(),tt.destroy(),P||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${zt[0]}, expected ~${Ta.toFixed(2)}) \u2014 falling back to f32`)}}catch{P=!1}let T=s.values().next().value,l=P&&!!T?.rawF16&&!_?.forceF32;console.log(l?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${C}, f16 validated: ${P}, f16 data: ${!!T?.rawF16})`);function g(i){if(l&&i.rawF16){let c=new Uint16Array(i.rawF16);if(c.length%2!==0){let u=new Uint16Array(c.length+1);return u.set(c),u}return c}return i.data}function M(i){if(l&&i.rawF16){let c=i.rawF16.byteLength;return Math.ceil(c/4)*4}return i.data.byteLength}function L(i){return l?La(i):i}let me={r:"read-only-storage",s:"storage",u:"uniform"};function y(i){return e.createBindGroupLayout({entries:i.map((c,u)=>({binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:me[c]}}))})}function ie(i){return e.createBindGroupLayout({entries:i.map((c,u)=>c==="t"?{binding:u,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:me[c]}})})}let H=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,we=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,ke=GPUBufferUsage.STORAGE,Ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,re=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function G(i,c){return e.createBuffer({size:i,usage:c})}function Z(i,c){return e.createBindGroup({layout:i,entries:c.map((u,o)=>({binding:o,resource:"size"in u?{buffer:u}:u}))})}function pe(i,c){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),compute:{module:c,entryPoint:"main"}})}let at=e.createShaderModule({code:Ka}),ft=e.createShaderModule({code:rn}),Oe=e.createShaderModule({code:L(Qa)}),Ie=e.createShaderModule({code:L(_a)}),ht=e.createShaderModule({code:L(da)}),ve=e.createShaderModule({code:L(la)}),nt=e.createShaderModule({code:L(ma)}),Ge=e.createShaderModule({code:L(Na)}),bt=e.createShaderModule({code:$a}),Et=e.createShaderModule({code:Xa}),Wt=e.createShaderModule({code:Ya}),B=e.createShaderModule({code:L(Va)}),q=e.createShaderModule({code:L(ja)}),z=e.createShaderModule({code:L(Ja)}),J=e.createShaderModule({code:L(en)}),V=new Map;function _e(i,c){let u=`${i}_${c}`,o=V.get(u);return o||(o=e.createShaderModule({code:L(tn(i,c))}),V.set(u,o)),o}let X=y(["r","r","r","s","u"]),ce=y(["r","r","r","r","s","u"]),ee=y(["r","s","u"]),de=y(["r","r","r","s","u"]),fe=y(["r","s","u"]),xe=y(["r","r","s","u"]),Pe=y(["r","r","s","u"]),he=y(["r","r","r","s","u"]),ge=y(["r","r","r","s","u"]),Se=ie(["t","s","u"]),De=y(["r","r","r","r","r","r","r","s"]),Ce=y(["r","r","r","r","r","s","u"]),Be=e.createPipelineLayout({bindGroupLayouts:[X]}),ze=e.createPipelineLayout({bindGroupLayouts:[ce]}),Me=i=>e.createComputePipeline({layout:Be,compute:{module:i,entryPoint:"main"}}),Le=i=>e.createComputePipeline({layout:ze,compute:{module:i,entryPoint:"main"}}),Fe=Me(Ie),it=Me(ht),wt=Le(ve),je=Le(nt),gt=new Map,Ht=new Map,yt=new Map,Tt=new Map;gt.set("8,8",Fe),Ht.set("8,8",it),yt.set("8,8",wt),Tt.set("8,8",je);function rt(i,c,u,o,a){let E=`${c},${u}`,S=i.get(E);return S||(S=a(e.createShaderModule({code:L(o(c,u))})),i.set(E,S)),S}let xt=(i,c)=>rt(gt,i,c,Ra,Me),Lt=(i,c)=>rt(Ht,i,c,Oa,Me),Ft=(i,c)=>rt(yt,i,c,Ia,Le),Rt=(i,c)=>rt(Tt,i,c,za,Le),Ee=sn.map(i=>{let c=i.stride===2?i.h/2:i.h,u=i.stride===2?i.w/2:i.w,[o,a]=Fa(i.inCh,c),E=i.h>=64,S=c>=16&&i.inCh>=288&&i.outCh>=288&&i.outCh%2===0;return{dwPipeline:E?Lt(o,a):xt(o,a),pwPipeline:S?Rt(o,a):Ft(o,a),dwDispatchX:Math.ceil(u/o),dwDispatchY:Math.ceil(c/a),dwDispatchZ:i.inCh,pwDispatchX:Math.ceil(u/o),pwDispatchY:Math.ceil(c/a),pwDispatchZ:S?i.outCh/2:i.outCh}}),vt=pe(ee,at),Je=pe(de,Ge);pe(fe,bt),pe(xe,Et);let Ke=pe(Pe,Wt),Kt=pe(he,B);pe(ge,q),pe(ge,z);let We=pe(Se,ft),Pt=pe(De,Oe),st=pe(Ce,J),Ne=1*288*128*128*4,ot=G(3*256*256*4,H),Re=G(3*257*257*4,ke),ut=G(12,re);e.queue.writeBuffer(ut,0,new Uint32Array([3,256,257]));let j=G(Ne,we),le=G(Ne,Ae),He=G(Ne,ke),pt=G(3072*64*4,H),ct=G(3072*32*4,H),dt=G(1536*16*4,H),O=G(6144*64*4,ke),te=G(260,Ae),Q=G(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);G(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let se=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Te=G(8,re);e.queue.writeBuffer(Te,0,new Uint32Array([256,257]));let _t=s.get("backbone1.1.weight"),Nt=s.get("backbone1.1.bias");if(!_t||!Nt)throw new Error("Missing input conv weights");let $=g(_t),qt=g(Nt),d=G($.byteLength,H),n=G(qt.byteLength,H),b=G(28,re);e.queue.writeBuffer(d,0,$),e.queue.writeBuffer(n,0,qt),e.queue.writeBuffer(b,0,new Uint32Array([1,3,24,257,257,128,128]));let A=s.get("backbone6.1.weight"),f=s.get("backbone6.1.bias");if(!A||!f)throw new Error("Missing backbone6.1 conv1x1 weights");let r=g(A),R=g(f),ae=G(r.byteLength,H),m=G(R.byteLength,H),U=G(20,re);e.queue.writeBuffer(ae,0,r),e.queue.writeBuffer(m,0,R),e.queue.writeBuffer(U,0,new Uint32Array([1,96,48,32,32]));let w=s.get("handflag.weight"),K=s.get("handflag.bias");if(!w||!K)throw new Error("Missing handflag weights");let D=g(w),ye=g(K),oe=G(D.byteLength,H),v=G(ye.byteLength,H),Y=G(12,re);e.queue.writeBuffer(oe,0,D),e.queue.writeBuffer(v,0,ye),e.queue.writeBuffer(Y,0,new Uint32Array([1,288,1]));let F=s.get("handedness.weight"),Ue=s.get("handedness.bias");if(!F||!Ue)throw new Error("Missing handedness weights");let Bt=g(F),xa=g(Ue),ta=G(Bt.byteLength,H),aa=G(xa.byteLength,H),va=G(12,re);e.queue.writeBuffer(ta,0,Bt),e.queue.writeBuffer(aa,0,xa),e.queue.writeBuffer(va,0,new Uint32Array([1,288,1]));let Pa=s.get("reg_3d.weight"),Ba=s.get("reg_3d.bias");if(!Pa||!Ba)throw new Error("Missing reg_3d weights");let ka=g(Pa),Ca=g(Ba),na=G(ka.byteLength,H),ia=G(Ca.byteLength,H),Ua=G(12,re);e.queue.writeBuffer(na,0,ka),e.queue.writeBuffer(ia,0,Ca),e.queue.writeBuffer(Ua,0,new Uint32Array([1,288,63]));let lt=sn.map(i=>{let{inCh:c,outCh:u,h:o,w:a,stride:E,prefix:S}=i,W=E===2?o/2:o,k=E===2?a/2:a,h=E===2?1:2,p=s.get(`${S}convs.0.weight`),N=s.get(`${S}convs.0.bias`),ne=s.get(`${S}convs.1.weight`),ue=s.get(`${S}convs.1.bias`);if(!p||!N||!ne||!ue)throw new Error(`Missing weights for ${S}`);let Ut=g(p),At=g(N),Gt=g(ne),St=g(ue),Dt=G(Ut.byteLength,H),Mt=G(At.byteLength,H),tt=G(Gt.byteLength,H),Ze=G(St.byteLength,H),Ot=G(32,re),It=G(36,re);return e.queue.writeBuffer(Dt,0,Ut),e.queue.writeBuffer(Mt,0,At),e.queue.writeBuffer(tt,0,Gt),e.queue.writeBuffer(Ze,0,St),e.queue.writeBuffer(Ot,0,new Uint32Array([1,c,o,a,W,k,E,h])),e.queue.writeBuffer(It,0,new Uint32Array([1,c,u,W,k,Math.max(0,u-c),E,o,a])),{dwWeight:Dt,dwBias:Mt,pwWeight:tt,pwBias:Ze,dwUniform:Ot,pwUniform:It,spec:i,outH:W,outW:k}});function kt(i){let c=G(i.length*4,re);return e.queue.writeBuffer(c,0,new Uint32Array(i)),c}let xn=kt([1,96,8,8,16,16]),vn=kt([1,96,16,16,32,32]),Pn=kt([1,48,32,32,64,64]);kt([1536*16]),kt([3072*32]),kt([3072*64]);let Aa=Z(ee,[ot,Re,ut]),Ga=Z(de,[Re,d,n,j,b]),qe=[],$e=[],Ye=[],Xe=[];for(let i of lt)qe.push(Z(X,[j,i.dwWeight,i.dwBias,He,i.dwUniform])),$e.push(Z(ce,[He,j,i.pwWeight,i.pwBias,le,i.pwUniform])),Ye.push(Z(X,[le,i.dwWeight,i.dwBias,He,i.dwUniform])),Xe.push(Z(ce,[He,le,i.pwWeight,i.pwBias,j,i.pwUniform]));let Bn=Z(Pe,[j,dt,le,xn]),kn=Z(Pe,[j,ct,le,vn]),Cn=Z(he,[j,ae,m,O,U]),Un=Z(Pe,[O,pt,le,Pn]);Z(ge,[j,oe,v,te,Y]),Z(ge,[j,ta,aa,te,va]),Z(ge,[j,na,ia,te,Ua]);let Qe=Z(Se,[se.createView(),Re,Te]),An=Z(De,[j,oe,v,ta,aa,na,ia,te]),ra=24,Sa=[],Da=[];for(let i=ra;i<lt.length;i++){let c=lt[i];Sa.push(Z(Ce,[j,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,le,c.dwUniform])),Da.push(Z(Ce,[le,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,j,c.dwUniform]))}let sa=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});sa.globalCompositeOperation="copy";let Ma=new OffscreenCanvas(9,8),$t=Ma.getContext("webgpu"),Yt=null,oa=null;if($t){$t.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let i=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),c=e.createShaderModule({code:an}),u=e.createShaderModule({code:nn});Yt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),vertex:{module:c,entryPoint:"vs"},fragment:{module:u,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),oa=e.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:te}}]})}let Xt=new Float32Array(1),Vt=new Float32Array(1),Zt=new Float32Array(63);function Ve(i,c){let u=!0,o=0,a=i.beginComputePass();for(a.setPipeline(Je),a.setBindGroup(0,Ga),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);o<=Kn;o++){let k=u?qe[o]:Ye[o],h=u?$e[o]:Xe[o],p=Ee[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,k),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let E=u?j:le;for(i.copyBufferToBuffer(E,0,pt,0,3072*64*4),a=i.beginComputePass();o<=Nn;o++){let k=u?qe[o]:Ye[o],h=u?$e[o]:Xe[o],p=Ee[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,k),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let S=u?j:le;for(i.copyBufferToBuffer(S,0,ct,0,3072*32*4),a=i.beginComputePass();o<=qn;o++){let k=u?qe[o]:Ye[o],h=u?$e[o]:Xe[o],p=Ee[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,k),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let W=u?j:le;for(i.copyBufferToBuffer(W,0,dt,0,1536*16*4),a=i.beginComputePass();o<=$n;o++){let k=u?qe[o]:Ye[o],h=u?$e[o]:Xe[o],p=Ee[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,k),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.setPipeline(Ke),a.setBindGroup(0,Bn),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),u=!1,a=i.beginComputePass();{let k=u?qe[o]:Ye[o],h=u?$e[o]:Xe[o],p=Ee[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,k),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}a.setPipeline(Ke),a.setBindGroup(0,kn),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),u=!1,a=i.beginComputePass();{let k=u?qe[o]:Ye[o],h=u?$e[o]:Xe[o],p=Ee[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,k),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}for(a.setPipeline(Kt),a.setBindGroup(0,Cn),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(Ke),a.setBindGroup(0,Un),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),u=!1,a=i.beginComputePass();o<ra;o++){let k=u?qe[o]:Ye[o],h=u?$e[o]:Xe[o],p=Ee[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,k),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}for(;o<lt.length;o++){let k=o-ra,h=u?Sa[k]:Da[k],p=lt[o];a.setPipeline(st),a.setBindGroup(0,h),a.dispatchWorkgroups(p.outW,p.outH,1),u=!u}a.setPipeline(Pt),a.setBindGroup(0,An),a.dispatchWorkgroups(1),a.end(),c&&i.copyBufferToBuffer(te,0,c,0,260)}async function jt(i){e.queue.writeBuffer(ot,0,i);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(vt),a.setBindGroup(0,Aa),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),a.end()}Ve(c,Q),e.queue.submit([c.finish()]);let u=Q.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(Q.getMappedRange());return Xt[0]=o[0],Vt[0]=o[1],Zt.set(o.subarray(2,65)),Q.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function ua(i){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(We),a.setBindGroup(0,Qe),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}Ve(c,Q),e.queue.submit([c.finish()]);let u=Q.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(Q.getMappedRange());return Xt[0]=o[0],Vt[0]=o[1],Zt.set(o.subarray(2,65)),Q.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function Ea(i){if(!Yt||!oa||!$t)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let c=e.createCommandEncoder();{let k=c.beginComputePass();k.setPipeline(We),k.setBindGroup(0,Qe),k.dispatchWorkgroups(16,16,1),k.end()}Ve(c,null);let u=$t.getCurrentTexture(),o=c.beginRenderPass({colorAttachments:[{view:u.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});o.setPipeline(Yt),o.setBindGroup(0,oa),o.draw(3),o.end(),e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),sa.drawImage(Ma,0,0);let E=sa.getImageData(0,0,9,8).data,S=new Float32Array(65),W=new DataView(new ArrayBuffer(4));for(let k=0;k<65;k++){let h=k*4;W.setUint8(0,E[h]),W.setUint8(1,E[h+1]),W.setUint8(2,E[h+2]),W.setUint8(3,E[h+3]),S[k]=W.getFloat32(0)}return{handflag:new Float32Array([S[0]]),handedness:new Float32Array([S[1]]),landmarks:new Float32Array(S.subarray(2,65))}}let Gn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),pa=0,Sn=[Q,Gn],Ct=null,et=null;async function ca(i){let c=Sn[pa];pa=1-pa,e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let u=e.createCommandEncoder();{let a=u.beginComputePass();a.setPipeline(We),a.setBindGroup(0,Qe),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}Ve(u,c),e.queue.submit([u.finish()]);let o=null;if(Ct!==null&&et!==null){await Ct;let a=new Float32Array(et.getMappedRange());o={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))},et.unmap()}return et=c,Ct=c.mapAsync(GPUMapMode.READ),o}async function Wa(){if(!Ct||!et)return null;await Ct;let i=new Float32Array(et.getMappedRange()),c={handflag:new Float32Array([i[0]]),handedness:new Float32Array([i[1]]),landmarks:new Float32Array(i.subarray(2,65))};return et.unmap(),Ct=null,et=null,c}async function Dn(i=50){let c=new Float32Array(196608);for(let a=0;a<5;a++)await jt(c);let u=[];for(let a=0;a<i;a++){let E=performance.now();await jt(c),u.push(performance.now()-E)}let o=u.reduce((a,E)=>a+E,0)/u.length;return{avgMs:o,fps:1e3/o}}async function Mn(i=50){let c=new Float32Array(196608);for(let S=0;S<5;S++)await jt(c);let u=[];for(let S=0;S<i;S++){let W=e.createCommandEncoder();{let h=W.beginComputePass();h.setPipeline(vt),h.setBindGroup(0,Aa),h.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),h.end()}Ve(W,Q);let k=performance.now();e.queue.submit([W.finish()]),await e.queue.onSubmittedWorkDone(),u.push(performance.now()-k)}u.sort((S,W)=>S-W);let o=u.reduce((S,W)=>S+W,0)/u.length,a=u[Math.floor(u.length/2)],E=u[0];return{avgMs:o,fps:1e3/o,medianMs:a,minMs:E}}function ti(i){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let c=e.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(We),u.setBindGroup(0,Qe),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),u.end()}Ve(c,Q),e.queue.submit([c.finish()])}async function En(i,c=50){function u(h){let p=[...h].sort((N,ne)=>N-ne);return{median:p[Math.floor(p.length/2)],min:p[0]}}for(let h=0;h<10;h++)await ua(i);let o=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let p=e.createCommandEncoder();{let ne=p.beginComputePass();ne.setPipeline(We),ne.setBindGroup(0,Qe),ne.dispatchWorkgroups(16,16,1),ne.end()}Ve(p,Q);let N=performance.now();e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-N)}let a=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let p=e.createCommandEncoder();{let ue=p.beginComputePass();ue.setPipeline(We),ue.setBindGroup(0,Qe),ue.dispatchWorkgroups(16,16,1),ue.end()}Ve(p,Q),e.queue.submit([p.finish()]);let N=Q.mapAsync(GPUMapMode.READ),ne=performance.now();await e.queue.onSubmittedWorkDone(),await N,Q.getMappedRange(),Q.unmap(),a.push(performance.now()-ne)}let E=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let p=e.createCommandEncoder();{let ne=p.beginComputePass();ne.setPipeline(We),ne.setBindGroup(0,Qe),ne.dispatchWorkgroups(16,16,1),ne.end()}Ve(p,Q),e.queue.submit([p.finish()]);let N=performance.now();await Q.mapAsync(GPUMapMode.READ),Q.getMappedRange(),Q.unmap(),E.push(performance.now()-N)}let S=[];for(let h=0;h<c;h++){let p=performance.now();await ua(i),S.push(performance.now()-p)}await ca(i);let W=[];for(let h=0;h<c;h++){let p=performance.now();await ca(i),W.push(performance.now()-p)}await Wa();let k=null;if(Yt){let h=[];for(let p=0;p<c;p++){let N=performance.now();await Ea(i),h.push(performance.now()-N)}k=u(h)}return{gpuOnly:u(o),mapAsyncOnly:u(a),mapAsyncNoWait:u(E),total:u(S),pipelined:u(W),renderReadback:k}}async function Wn(i){let c=[];async function u(a,E,S){let W=e.createBuffer({size:E,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),k=e.createCommandEncoder();k.copyBufferToBuffer(a,0,W,0,E),e.queue.submit([k.finish()]),await e.queue.onSubmittedWorkDone(),await W.mapAsync(GPUMapMode.READ);let h=new Float32Array(W.getMappedRange()),p=1/0,N=-1/0,ne=0;for(let ue=0;ue<h.length;ue++)h[ue]<p&&(p=h[ue]),h[ue]>N&&(N=h[ue]),h[ue]!==0&&ne++;W.unmap(),W.destroy(),c.push({layer:S,stats:{min:p,max:N,nonZero:ne,total:h.length}})}e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);{let a=e.createCommandEncoder(),E=a.beginComputePass();E.setPipeline(We),E.setBindGroup(0,Qe),E.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),E.end(),e.queue.submit([a.finish()])}await u(Re,Math.min(Re.size,3*257*257*4),"canvas\u2192bufInput");{let a=e.createCommandEncoder(),E=a.beginComputePass();E.setPipeline(Je),E.setBindGroup(0,Ga),E.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),E.end(),e.queue.submit([a.finish()])}await u(j,Math.min(j.size,3072*128*4),"inputConv\u2192bufA");let o=!0;for(let a=0;a<Math.min(lt.length,6);a++){let E=o?qe[a]:Ye[a],S=o?$e[a]:Xe[a],W=Ee[a],k=lt[a];{let p=e.createCommandEncoder(),N=p.beginComputePass();N.setPipeline(W.dwPipeline),N.setBindGroup(0,E),N.dispatchWorkgroups(W.dwDispatchX,W.dwDispatchY,W.dwDispatchZ),N.end(),e.queue.submit([p.finish()])}await u(He,Math.min(He.size,k.spec.inCh*k.outH*k.outW*4),`layer${a}.DW\u2192bufDW (${k.spec.prefix})`);{let p=e.createCommandEncoder(),N=p.beginComputePass();N.setPipeline(W.pwPipeline),N.setBindGroup(0,S),N.dispatchWorkgroups(W.pwDispatchX,W.pwDispatchY,W.pwDispatchZ),N.end(),e.queue.submit([p.finish()])}let h=o?le:j;await u(h,Math.min(h.size,k.spec.outCh*k.outH*k.outW*4),`layer${a}.PW\u2192buf${o?"B":"A"} (${k.spec.prefix})`),o=!o}return c}return{device:e,run:jt,runFromCanvas:ua,runFromCanvasViaRender:Ea,runFromCanvasPipelined:ca,flushPipelined:Wa,benchmark:Dn,benchmarkGPU:Mn,benchmarkDiagnostic:En,debugLayerOutputs:Wn}}function mt(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var on=mt(`
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
`),un=mt(`
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
`),pn=mt(`
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
`),cn=mt(`
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
`),dn=mt(`
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
`),_n=mt(`
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
`);async function mn(s,_){let t;if(_)t=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");t=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8)}})}let C={r:"read-only-storage",s:"storage",u:"uniform"};function x(d){return t.createBindGroupLayout({entries:d.map((n,b)=>n==="t"?{binding:b,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:b,visibility:GPUShaderStage.COMPUTE,buffer:{type:C[n]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,P=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,I=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,T=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(d,n){return t.createBuffer({size:Math.max(d,4),usage:n})}function g(d,n,b){t.queue.writeBuffer(d,n,b)}function M(d){let n=l(d.data.byteLength,e);return g(n,0,d.data),n}let L=Array.from(s.keys());function me(d){let n=s.get(d);if(!n)throw new Error(`Weight not found: ${d}`);return n}function y(...d){let n=L.find(b=>d.every(A=>b.includes(A)));if(!n)throw new Error(`Weight not found for: ${d.join(", ")}`);return me(n)}function ie(d){let[,n,b,A]=d.shape,f=new Float32Array(A*25);for(let r=0;r<A;r++)for(let R=0;R<n;R++)for(let ae=0;ae<b;ae++)f[r*25+R*5+ae]=d.data[R*b*A+ae*A+r];return f}function H(d){let[n,,,b]=d.shape,A=new Float32Array(n*b);for(let f=0;f<n;f++)for(let r=0;r<b;r++)A[f*b+r]=d.data[f*b+r];return A}let we=t.createShaderModule({code:on}),ke=t.createShaderModule({code:un}),Ae=t.createShaderModule({code:pn}),re=t.createShaderModule({code:cn}),G=t.createShaderModule({code:_n}),Z=t.createShaderModule({code:dn}),pe=t.createShaderModule({code:ln}),at=x(["r","r","r","r","s","u"]),ft=x(["r","r","r","s","u"]),Oe=x(["r","r","r","r","r","s","u"]),Ie=x(["r","r","r","s","u"]),ht=x(["r","r","r","r","s","u"]),ve=x(["r","r","s","u"]),nt=x(["t","s","u"]);function Ge(d,n){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:n,entryPoint:"main"}})}let bt=Ge(at,we),Et=Ge(ft,ke),Wt=Ge(Oe,Ae),B=Ge(Ie,re),q=Ge(ht,G),z=Ge(ve,Z),J=Ge(nt,pe),V=y("conv2d/Conv2D"),_e=y("batch_normalization/","conv2d/Conv2D"),X=y("p_re_lu/"),ce=M(V),ee=M(_e),de=M(X),xe=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(d=>{let n=y(d.dwKey),b=y(d.pwKey),A=y(d.bnKey),f=y(d.preluKey),r=ie(n),R=l(r.byteLength,e);g(R,0,r);let ae=new Float32Array(d.inCh),m=l(ae.byteLength,e);g(m,0,ae);let U=H(b),w=l(U.byteLength,e);g(w,0,U);let K=M(A),D=M(f);return{dwWeightBuf:R,dwBiasBuf:m,pwWeightBuf:w,pwBiasBuf:K,alphaBuf:D,inCh:d.inCh,outCh:d.outCh,stride:d.stride,inH:d.inH}}),Pe=H(y("conv2d_20/Conv2D")),he=l(Pe.byteLength,e);g(he,0,Pe);let ge=M(y("batch_normalization_20/")),Se=M(y("p_re_lu_20/")),De={dwWeightBuf:(()=>{let d=ie(y("depthwise_conv2d_19/")),n=l(d.byteLength,e);return g(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return g(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(y("conv2d_21/")),n=l(d.byteLength,e);return g(n,0,d),n})(),pwBiasBuf:M(y("batch_normalization_21/")),alphaBuf:M(y("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},Ce={dwWeightBuf:(()=>{let d=ie(y("depthwise_conv2d_20/")),n=l(d.byteLength,e);return g(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return g(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(y("conv2d_22/Conv2D1")),n=l(d.byteLength,e);return g(n,0,d),n})(),pwBiasBuf:M(y("batch_normalization_22/")),alphaBuf:M(y("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},Be=H(y("conv2d_23/Conv2D")),ze=l(Be.byteLength,e);g(ze,0,Be);let Me=M(y("batch_normalization_23/")),Le=M(y("p_re_lu_23/")),Fe={dwWeightBuf:(()=>{let d=ie(y("depthwise_conv2d_21/")),n=l(d.byteLength,e);return g(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return g(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(y("conv2d_24/")),n=l(d.byteLength,e);return g(n,0,d),n})(),pwBiasBuf:M(y("batch_normalization_24/")),alphaBuf:M(y("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},it={dwWeightBuf:(()=>{let d=ie(y("depthwise_conv2d_22/")),n=l(d.byteLength,e);return g(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return g(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(y("conv2d_25/Conv2D1")),n=l(d.byteLength,e);return g(n,0,d),n})(),pwBiasBuf:M(y("batch_normalization_25/")),alphaBuf:M(y("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},wt=H(y("classifier_palm_16_NO_PRUNING/Conv2D")),je=l(wt.byteLength,e);g(je,0,wt);let gt=M(y("classifier_palm_16_NO_PRUNING/BiasAdd")),Ht=H(y("regressor_palm_16_NO_PRUNING/Conv2D")),yt=l(Ht.byteLength,e);g(yt,0,Ht);let Tt=M(y("regressor_palm_16_NO_PRUNING/BiasAdd")),rt=H(y("classifier_palm_8_NO_PRUNING/Conv2D")),xt=l(rt.byteLength,e);g(xt,0,rt);let Lt=M(y("classifier_palm_8_NO_PRUNING/BiasAdd")),Ft=H(y("regressor_palm_8_NO_PRUNING/Conv2D")),Rt=l(Ft.byteLength,e);g(Rt,0,Ft);let Ee=M(y("regressor_palm_8_NO_PRUNING/BiasAdd")),vt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Je=l(36864*3*4,e),Ke=l(vt,P),Kt=l(vt,P),We=l(vt,P),Pt=l(576*256*4,P),st=l(144*256*4,P|GPUBufferUsage.COPY_DST),Ne=l(576*128*4,P|GPUBufferUsage.COPY_DST),ot=l(864*4,I),Re=l(15552*4,I),ut=l(576*2*4,I),j=l(576*36*4,I),le=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),He=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pt=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),dt=t.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function O(d,n){return Math.ceil(d/n)}function te(d){let n=l(d.byteLength,T);return g(n,0,d),n}let Q=te(new Uint32Array([1,3,32,192,192,96,96]));function se(d,n,b,A,f){let r=n.stride===2?n.inH/2:n.inH,R=r,ae=n.stride===2?1:2,m=te(new Uint32Array([1,n.inCh,n.inH,n.inH,r,R,n.stride,ae])),U=t.createBindGroup({layout:ft,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:n.dwWeightBuf}},{binding:2,resource:{buffer:n.dwBiasBuf}},{binding:3,resource:{buffer:We}},{binding:4,resource:{buffer:m}}]}),w=d.beginComputePass();w.setPipeline(Et),w.setBindGroup(0,U),w.dispatchWorkgroups(O(R,8),O(r,8),n.inCh),w.end();let K=n.inCh,D=te(new Uint32Array([1,n.inCh,n.outCh,r,R,K,n.stride,n.inH,n.inH])),ye=t.createBindGroup({layout:Oe,entries:[{binding:0,resource:{buffer:We}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:n.pwWeightBuf}},{binding:3,resource:{buffer:n.pwBiasBuf}},{binding:4,resource:{buffer:n.alphaBuf}},{binding:5,resource:{buffer:A}},{binding:6,resource:{buffer:D}}]}),oe=d.beginComputePass();oe.setPipeline(Wt),oe.setBindGroup(0,ye),oe.dispatchWorkgroups(O(R,8),O(r,8),n.outCh),oe.end()}function Te(d,n,b,A,f,r,R,ae,m){let U=te(new Uint32Array([1,r,R,ae,m])),w=t.createBindGroup({layout:Ie,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:A}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:U}}]}),K=d.beginComputePass();K.setPipeline(B),K.setBindGroup(0,w),K.dispatchWorkgroups(O(m,8),O(ae,8),R),K.end()}function _t(d,n,b,A,f,r,R,ae,m,U){let w=te(new Uint32Array([1,R,ae,m,U])),K=t.createBindGroup({layout:ht,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:A}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:r}},{binding:5,resource:{buffer:w}}]}),D=d.beginComputePass();D.setPipeline(q),D.setBindGroup(0,K),D.dispatchWorkgroups(O(U,8),O(m,8),ae),D.end()}async function Nt(d){t.queue.copyExternalImageToTexture({source:d},{texture:dt},[192,192]);let n=te(new Uint32Array([192,192,192])),b=t.createBindGroup({layout:nt,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:Je}},{binding:2,resource:{buffer:n}}]}),A=t.createCommandEncoder();{let v=A.beginComputePass();v.setPipeline(J),v.setBindGroup(0,b),v.dispatchWorkgroups(O(192,16),O(192,16),1),v.end()}{let v=t.createBindGroup({layout:at,entries:[{binding:0,resource:{buffer:Je}},{binding:1,resource:{buffer:ce}},{binding:2,resource:{buffer:ee}},{binding:3,resource:{buffer:de}},{binding:4,resource:{buffer:Ke}},{binding:5,resource:{buffer:Q}}]}),Y=A.beginComputePass();Y.setPipeline(bt),Y.setBindGroup(0,v),Y.dispatchWorkgroups(O(96,8),O(96,8),32),Y.end()}let f=Ke,r=Kt;for(let v=0;v<xe.length;v++){let Y=xe[v];se(A,Y,f,r,f);let F=f;f=r,r=F,v===10&&A.copyBufferToBuffer(f,0,Ne,0,576*128*4),v===14&&A.copyBufferToBuffer(f,0,st,0,144*256*4)}{let v=te(new Uint32Array([1,256,6,6,12,12])),Y=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Pt}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:v}}]}),F=A.beginComputePass();F.setPipeline(z),F.setBindGroup(0,Y),F.dispatchWorkgroups(O(12,8),O(12,8),256),F.end()}{let v=f;f=r,r=v}_t(A,f,he,ge,Se,r,256,256,12,12);{let v=f;f=r,r=v}{let v=te(new Uint32Array([1,256,12,12,12,12])),Y=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:v}}]}),F=A.beginComputePass();F.setPipeline(z),F.setBindGroup(0,Y),F.dispatchWorkgroups(O(12,8),O(12,8),256),F.end()}{let v=f;f=r,r=v}se(A,De,f,r,f);{let v=f;f=r,r=v}se(A,Ce,f,r,f);{let v=f;f=r,r=v}Te(A,f,je,gt,ot,256,6,12,12),Te(A,f,yt,Tt,Re,256,108,12,12);{let v=te(new Uint32Array([1,256,12,12,24,24])),Y=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Pt}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:v}}]}),F=A.beginComputePass();F.setPipeline(z),F.setBindGroup(0,Y),F.dispatchWorkgroups(O(24,8),O(24,8),256),F.end()}{let v=f;f=r,r=v}_t(A,f,ze,Me,Le,r,256,128,24,24);{let v=f;f=r,r=v}{let v=te(new Uint32Array([1,128,24,24,24,24])),Y=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:v}}]}),F=A.beginComputePass();F.setPipeline(z),F.setBindGroup(0,Y),F.dispatchWorkgroups(O(24,8),O(24,8),128),F.end()}{let v=f;f=r,r=v}se(A,Fe,f,r,f);{let v=f;f=r,r=v}se(A,it,f,r,f);{let v=f;f=r,r=v}Te(A,f,xt,Lt,ut,128,2,24,24),Te(A,f,Rt,Ee,j,128,36,24,24),t.queue.submit([A.finish()]);let R=t.createCommandEncoder();R.copyBufferToBuffer(ot,0,le,0,864*4),R.copyBufferToBuffer(Re,0,He,0,15552*4),R.copyBufferToBuffer(ut,0,pt,0,576*2*4),R.copyBufferToBuffer(j,0,ct,0,576*36*4),t.queue.submit([R.finish()]),await Promise.all([le.mapAsync(GPUMapMode.READ),He.mapAsync(GPUMapMode.READ),pt.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ)]);let ae=new Float32Array(le.getMappedRange()).slice(),m=new Float32Array(He.getMappedRange()).slice(),U=new Float32Array(pt.getMappedRange()).slice(),w=new Float32Array(ct.getMappedRange()).slice();le.unmap(),He.unmap(),pt.unmap(),ct.unmap();let K=2016,D=new Float32Array(K),ye=new Float32Array(K*18),oe=0;for(let v=0;v<12;v++)for(let Y=0;Y<12;Y++)for(let F=0;F<6;F++){D[oe]=ae[F*144+v*12+Y];for(let Ue=0;Ue<18;Ue++){let Bt=F*18+Ue;ye[oe*18+Ue]=m[Bt*144+v*12+Y]}oe++}for(let v=0;v<24;v++)for(let Y=0;Y<24;Y++)for(let F=0;F<2;F++){D[oe]=U[F*576+v*24+Y];for(let Ue=0;Ue<18;Ue++){let Bt=F*18+Ue;ye[oe*18+Ue]=w[Bt*576+v*24+Y]}oe++}return{scores:D,regressors:ye}}async function $(d,n){let b=t.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),A=t.createCommandEncoder();A.copyBufferToBuffer(d,0,b,0,n*4),t.queue.submit([A.finish()]),await b.mapAsync(GPUMapMode.READ);let f=new Float32Array(b.getMappedRange()).slice();return b.unmap(),b.destroy(),f}async function qt(d){t.queue.copyExternalImageToTexture({source:d},{texture:dt},[192,192]);function n(w,K=1e3){let D=w.slice(0,K);return{min:Math.min(...D),max:Math.max(...D),mean:D.reduce((ye,oe)=>ye+oe,0)/D.length,nonZero:D.filter(ye=>ye!==0).length,sample:Array.from(D.slice(0,10))}}let b={},A=te(new Uint32Array([192,192,192])),f=t.createBindGroup({layout:nt,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:Je}},{binding:2,resource:{buffer:A}}]}),r=t.createCommandEncoder(),R=r.beginComputePass();R.setPipeline(J),R.setBindGroup(0,f),R.dispatchWorkgroups(O(192,16),O(192,16),1),R.end(),t.queue.submit([r.finish()]),b.input=n(await $(Je,36864*3)),r=t.createCommandEncoder();let ae=t.createBindGroup({layout:at,entries:[{binding:0,resource:{buffer:Je}},{binding:1,resource:{buffer:ce}},{binding:2,resource:{buffer:ee}},{binding:3,resource:{buffer:de}},{binding:4,resource:{buffer:Ke}},{binding:5,resource:{buffer:Q}}]});R=r.beginComputePass(),R.setPipeline(bt),R.setBindGroup(0,ae),R.dispatchWorkgroups(O(96,8),O(96,8),32),R.end(),t.queue.submit([r.finish()]),b.initConv=n(await $(Ke,9216*32));let m=Ke,U=Kt;for(let w=0;w<xe.length;w++){let K=xe[w];r=t.createCommandEncoder(),se(r,K,m,U,m),t.queue.submit([r.finish()]);let D=m;if(m=U,U=D,w===0||w===3||w===7||w===11||w===14||w===15||w===18){let ye=K.stride===2?K.inH/2:K.inH,oe=ye*ye*K.outCh;b[`block${w}`]=n(await $(m,oe))}w===10&&(r=t.createCommandEncoder(),r.copyBufferToBuffer(m,0,Ne,0,576*128*4),t.queue.submit([r.finish()])),w===14&&(r=t.createCommandEncoder(),r.copyBufferToBuffer(m,0,st,0,144*256*4),t.queue.submit([r.finish()]))}r=t.createCommandEncoder();{let w=te(new Uint32Array([1,256,6,6,12,12])),K=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Pt}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:w}}]}),D=r.beginComputePass();D.setPipeline(z),D.setBindGroup(0,K),D.dispatchWorkgroups(O(12,8),O(12,8),256),D.end()}t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpnUpsample6to12=n(await $(m,144*256)),r=t.createCommandEncoder(),_t(r,m,he,ge,Se,U,256,256,12,12),t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpn6to12Conv=n(await $(m,144*256)),b.backbone12Skip=n(await $(st,144*256)),r=t.createCommandEncoder();{let w=te(new Uint32Array([1,256,12,12,12,12])),K=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:w}}]}),D=r.beginComputePass();D.setPipeline(z),D.setBindGroup(0,K),D.dispatchWorkgroups(O(12,8),O(12,8),256),D.end()}t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpnAdd12=n(await $(m,144*256)),r=t.createCommandEncoder(),se(r,De,m,U,m),t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpn12Block1=n(await $(m,144*256)),r=t.createCommandEncoder(),se(r,Ce,m,U,m),t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpn12Block2=n(await $(m,144*256)),r=t.createCommandEncoder(),Te(r,m,je,gt,ot,256,6,12,12),t.queue.submit([r.finish()]),b.cls16=n(await $(ot,864)),r=t.createCommandEncoder(),Te(r,m,yt,Tt,Re,256,108,12,12),t.queue.submit([r.finish()]),b.reg16=n(await $(Re,15552),500),r=t.createCommandEncoder();{let w=te(new Uint32Array([1,256,12,12,24,24])),K=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Pt}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:w}}]}),D=r.beginComputePass();D.setPipeline(z),D.setBindGroup(0,K),D.dispatchWorkgroups(O(24,8),O(24,8),256),D.end()}t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpnUpsample12to24=n(await $(m,576*256)),r=t.createCommandEncoder(),_t(r,m,ze,Me,Le,U,256,128,24,24),t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpn12to24Conv=n(await $(m,576*128)),b.backbone24Skip=n(await $(Ne,576*128)),r=t.createCommandEncoder();{let w=te(new Uint32Array([1,128,24,24,24,24])),K=t.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:w}}]}),D=r.beginComputePass();D.setPipeline(z),D.setBindGroup(0,K),D.dispatchWorkgroups(O(24,8),O(24,8),128),D.end()}t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpnAdd24=n(await $(m,576*128)),r=t.createCommandEncoder(),se(r,Fe,m,U,m),t.queue.submit([r.finish()]);{let w=m;m=U,U=w}b.fpn24Block1=n(await $(m,576*128)),r=t.createCommandEncoder(),se(r,it,m,U,m),t.queue.submit([r.finish()]);{let w=m;m=U,U=w}return b.fpn24Block2=n(await $(m,576*128)),r=t.createCommandEncoder(),Te(r,m,xt,Lt,ut,128,2,24,24),t.queue.submit([r.finish()]),b.cls8=n(await $(ut,576*2)),r=t.createCommandEncoder(),Te(r,m,Rt,Ee,j,128,36,24,24),t.queue.submit([r.finish()]),b.reg8=n(await $(j,576*36)),b.initWeights=n(await $(ce,100),100),b.initBias=n(await $(ee,32),32),b.cls16Weights=n(await $(je,100),100),b.cls16Bias=n(await $(gt,6),6),b.cls8Weights=n(await $(xt,100),100),b.cls8Bias=n(await $(Lt,2),2),b.fpn6to12Weights=n(await $(he,100),100),b}return{device:t,run:Nt,debugRun:qt}}function Yn(){let s=[];for(let _=0;_<12;_++)for(let t=0;t<12;t++){let C=(t+.5)/12,x=(_+.5)/12;for(let e=0;e<6;e++)s.push({x:C,y:x})}for(let _=0;_<24;_++)for(let t=0;t<24;t++){let C=(t+.5)/24,x=(_+.5)/24;for(let e=0;e<2;e++)s.push({x:C,y:x})}return s}var fn=Yn();function Xn(s){return 1/(1+Math.exp(-s))}function hn(s,_){let t=[],{scores:C,regressors:x}=s,e=192;for(let P=0;P<fn.length;P++){let I=Xn(C[P]);if(I<_)continue;let T=fn[P],l=P*18,g=T.x+x[l+0]/e,M=T.y+x[l+1]/e,L=x[l+2]/e,me=x[l+3]/e,y=[];for(let ie=0;ie<7;ie++){let H=T.x+x[l+4+ie*2]/e,we=T.y+x[l+4+ie*2+1]/e;y.push([H,we])}t.push({score:I,box:[g,M,L,me],keypoints:y})}return t}function bn(s,_){if(s.length===0)return[];let t=[...s].sort((e,P)=>P.score-e.score),C=[],x=new Set;for(let e=0;e<t.length;e++)if(!x.has(e)){C.push(t[e]);for(let P=e+1;P<t.length;P++)x.has(P)||Vn(t[e],t[P])>_&&x.add(P)}return C}function Vn(s,_){let t=s.box[0]-s.box[2]/2,C=s.box[1]-s.box[3]/2,x=s.box[0]+s.box[2]/2,e=s.box[1]+s.box[3]/2,P=_.box[0]-_.box[2]/2,I=_.box[1]-_.box[3]/2,T=_.box[0]+_.box[2]/2,l=_.box[1]+_.box[3]/2,g=Math.max(t,P),M=Math.max(C,I),L=Math.min(x,T),me=Math.min(e,l),y=Math.max(0,L-g),ie=Math.max(0,me-M),H=y*ie,we=(x-t)*(e-C),ke=(T-P)*(l-I),Ae=we+ke-H;return Ae>0?H/Ae:0}function Zn(s){let[_,t,C,x]=s.box,e=s.keypoints[0],P=s.keypoints[2],I=P[0]-e[0],T=P[1]-e[1],l=Math.atan2(T,I),M=-Math.PI/2-l,L=Math.max(C,x),y=L*2.6,ie=-.5*L,H=Math.cos(M),we=Math.sin(M),ke=ie*we,Ae=ie*H;return{centerX:_+ke,centerY:t+Ae,width:y,height:y,rotation:M}}function wn(s,_={}){let{scoreThreshold:t=.5,nmsThreshold:C=.3,maxHands:x=2}=_;async function e(I){let T=await s.run(I),l=hn(T,t);return bn(l,C).slice(0,x).map(Zn)}async function P(I){let T=await s.run(I),l=hn(T,t);return bn(l,C).slice(0,x)}return{detect:e,detectRaw:P,model:s}}var ba=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function wa(s){let _={};for(let t=0;t<ba.length;t++)_[ba[t]]=s[t];return _}function gn(s,_,t){return s.initialized?(s.value=t*_+(1-t)*s.value,s.value):(s.value=_,s.initialized=!0,_)}function yn(s,_){let t=2*Math.PI*_*s;return t/(t+1)}function jn(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function ga(s,_,t,C,x,e){let P=s.lastTime<0?.03333333333333333:t-s.lastTime;s.lastTime=t;let I=yn(P,e),T=s.x.initialized?(_-s.x.value)/P:0,l=gn(s.dx,T,I),g=C+x*Math.abs(l),M=yn(P,g);return gn(s.x,_,M)}function ya(s={}){let{minCutoff:_=.05,beta:t=80,dCutoff:C=1}=s,x=[];function e(T){x.length!==T&&(x=Array.from({length:T},()=>jn()))}function P(T,l){let g=l??performance.now()/1e3,M=T.length*3;return e(M),T.map((L,me)=>({x:ga(x[me*3],L.x,g,_,t,C),y:ga(x[me*3+1],L.y,g,_,t,C),z:ga(x[me*3+2],L.z,g,_,t,C)}))}function I(){x=[]}return{apply:P,reset:I}}var Jn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Qn(s={}){let{weightsUrl:_,scoreThreshold:t=.5,palmScoreThreshold:C=.5,maxHands:x=3,forceF32:e=!1}=s;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let P=(_??Jn).replace(/\/$/,"")+"/",[I,T,l,g]=await Promise.all([fetch(`${P}weights_f16.json`),fetch(`${P}weights_f16.bin`),fetch(`${P}palm_detection_weights.json`),fetch(`${P}palm_detection_weights.bin`)]);if(!I.ok)throw new Error(`Failed to fetch landmark weights: ${I.status}`);if(!T.ok)throw new Error(`Failed to fetch landmark weights: ${T.status}`);if(!l.ok)throw new Error(`Failed to fetch palm detection weights: ${l.status}`);if(!g.ok)throw new Error(`Failed to fetch palm detection weights: ${g.status}`);let[M,L,me,y]=await Promise.all([I.json(),T.arrayBuffer(),l.json(),g.arrayBuffer()]),ie=fa(M,L),H=fa(me,y),we=await ha(ie,{forceF32:e});if(!e){let B=new OffscreenCanvas(256,256),q=B.getContext("2d");q.fillStyle="#886644",q.fillRect(0,0,256,256),q.fillStyle="#cc9966",q.fillRect(50,50,156,156);let z=await we.runFromCanvas(B);z.landmarks.every(V=>V===0)&&z.handflag.every(V=>V===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),we.device.destroy(),we=await ha(ie,{forceF32:!0}))}let ke=await mn(H),Ae=wn(ke,{scoreThreshold:C,maxHands:x}),re=[];for(let B=0;B<x;B++)re.push(ya());let G=0,Z=null,pe=null;function at(){return Z||(Z=new OffscreenCanvas(192,192)),Z}function ft(){return pe||(pe=new OffscreenCanvas(256,256)),pe}let Oe=0,Ie=0;function ht(B,q,z){let J=at();J.width=192,J.height=192;let V=J.getContext("2d");V.clearRect(0,0,192,192);let _e=Math.min(192/q,192/z),X=Math.round(q*_e),ce=Math.round(z*_e),ee=(192-X)/2,de=(192-ce)/2;if(Oe=ee/192,Ie=de/192,B instanceof ImageData){let fe=new OffscreenCanvas(B.width,B.height);fe.getContext("2d").putImageData(B,0,0),V.drawImage(fe,ee,de,X,ce)}else V.drawImage(B,0,0,q,z,ee,de,X,ce);return J}function ve(B){let q=1/(1-2*Oe),z=1/(1-2*Ie);return{score:B.score,box:[(B.box[0]-Oe)*q,(B.box[1]-Ie)*z,B.box[2]*q,B.box[3]*z],keypoints:B.keypoints.map(([J,V])=>[(J-Oe)*q,(V-Ie)*z])}}function nt(B,q,z){let J=B.keypoints[0],V=B.keypoints[2],_e=V[0]-J[0],X=V[1]-J[1],ce=Math.atan2(-X,_e),de=Math.PI/2-ce,[fe,xe,Pe,he]=B.box,ge=Math.max(Pe*q,he*z),Se=0,De=-.5*ge/z,Ce=Math.cos(de),Be=Math.sin(de),ze=fe+(Se*Ce-De*Be),Me=xe+(Se*Be+De*Ce),Fe=ge*2.6;return{centerXpx:ze*q,centerYpx:Me*z,sizePx:Fe,rotation:de}}function Ge(B,q){let z=ft();z.width=256,z.height=256;let J=z.getContext("2d");J.clearRect(0,0,256,256);let V=256/q.sizePx,_e=Math.cos(q.rotation),X=Math.sin(q.rotation),ce=_e*V,ee=-X*V,de=X*V,fe=_e*V,xe=-q.centerXpx*ce-q.centerYpx*de+128,Pe=-q.centerXpx*ee-q.centerYpx*fe+128;if(J.setTransform(ce,ee,de,fe,xe,Pe),B instanceof ImageData){let he=new OffscreenCanvas(B.width,B.height);he.getContext("2d").putImageData(B,0,0),J.drawImage(he,0,0)}else J.drawImage(B,0,0);return J.setTransform(1,0,0,1,0,0),z}function bt(B){return B instanceof HTMLCanvasElement||B instanceof OffscreenCanvas?[B.width,B.height]:typeof ImageBitmap<"u"&&B instanceof ImageBitmap?[B.width,B.height]:B instanceof ImageData?[B.width,B.height]:B instanceof HTMLVideoElement?[B.videoWidth,B.videoHeight]:B instanceof HTMLImageElement?[B.naturalWidth,B.naturalHeight]:[256,256]}async function Et(B){let[q,z]=bt(B),J=ht(B,q,z),V=await Ae.detectRaw(J);if(V.length===0){if(G>0)for(let X=0;X<G&&X<re.length;X++)re[X].reset();return G=0,[]}let _e=[];for(let X of V){let ce=ve(X),ee=nt(ce,q,z),de=Ge(B,ee),fe=await we.runFromCanvas(de),xe=fe.handflag[0];if(xe<t)continue;let Pe=fe.handedness[0]>.5,he=[],ge=Math.cos(ee.rotation),Se=Math.sin(ee.rotation);for(let Be=0;Be<21;Be++){let ze=fe.landmarks[Be*3],Me=fe.landmarks[Be*3+1],Le=fe.landmarks[Be*3+2],Fe=(ze-.5)*ee.sizePx,it=(Me-.5)*ee.sizePx,wt=ge*Fe-Se*it+ee.centerXpx,je=Se*Fe+ge*it+ee.centerYpx;he.push({x:wt/q,y:je/z,z:Le})}let De=_e.length,Ce=De<re.length?re[De].apply(he):he;_e.push({score:xe,handedness:Pe?"right":"left",landmarks:Ce,keypoints:wa(Ce)})}if(_e.length<G)for(let X=_e.length;X<G;X++)X<re.length&&re[X].reset();return G=_e.length,_e}function Wt(){we.device.destroy(),ke.device.destroy(),Z=null,pe=null}return{detect:Et,dispose:Wt}}export{ba as LANDMARK_NAMES,Qn as createHandpose,ya as createLandmarkSmoother,wa as toKeypoints};
