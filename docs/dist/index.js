function me(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ta(r){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+r;for(let v of _)for(;a.includes(`${v}:array<f32>`);)a=a.replace(`${v}:array<f32>`,`${v}:array<f16>`);return a}var _a=me(`
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
`),ma=me(`
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
`),fa=me(`
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
`),ha=me(`
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
`);function La(r,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Ra(r,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Oa(r,_){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Ia(r,_){return ha.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Fa(r,_){return[8,8]}var za=me(`
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
`),Na=me(`
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
`);function Ka(r){return me(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${r?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${r?"+skip[out_idx]":""};
}
`)}var qa=Ka(!1),$a=Ka(!0),Ya=me(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Xa=me(`
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
`);function ja(r){return me(`
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
  ${r==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var Va=ja("sigmoid"),Za=ja("div256"),Ja=me(`
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
`),Qa=me(`
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
`);function en(r,_){let v=Math.min(_,256),w=_>v,g=r%4===0?`var ic:u32=0u;
    while(ic<${r}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${r}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,A=`var skip_val:f32=0.0;
    if(c<${r}u){
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
    }`,S=r===_?"":`if(c<${r}u){`,l=r===_?"":"}",h=w?`for(var c:u32=lid.x;c<${r}u;c+=${v}u){`:`let c=lid.x;
  ${S}`,B=w?"}":l,U=w?`for(var c:u32=lid.x;c<${_}u;c+=${v}u){`:"{let c=lid.x;";return me(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${r}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${v},1,1)
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
  ${B}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${U}
    let pw_base=c*${r}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${g}
    // Skip connection (only for c < inCh)
    ${A}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var tn=me(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),an=me(`
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
`),nn=me(`
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
`);function Lt(r,_){let a=new Map,v=r.dtype??"float32";for(let w=0;w<r.keys.length;w++){let e=r.keys[w],g=r.shapes[w],A=r.offsets[w],S=g.reduce((B,U)=>B*U,1),l,h;if(v==="float32")l=new Float32Array(_,A,S);else{let B=new DataView(_);l=new Float32Array(S);for(let U=0;U<S;U++)l[U]=Rn(B.getUint16(A+U*2,!0));h=_.slice(A,A+S*2)}a.set(e,{data:l,shape:g,rawF16:h})}return a}function Rn(r){let _=r>>15&1,a=r>>10&31,v=r&1023;if(a===0){if(v===0)return _?-0:0;let g=-14,A=v/1024;return(_?-1:1)*Math.pow(2,g)*A}if(a===31)return v===0?_?-1/0:1/0:NaN;let w=a-15,e=1+v/1024;return(_?-1:1)*Math.pow(2,w)*e}var On=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],rn=On.map(([r,_,a,v,w])=>({type:"resmodule",inCh:r,outCh:_,h:a,w:a,stride:v,prefix:w})),In=2,Fn=5,zn=8,Nn=11;async function ta(r,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let v=a.features.has("shader-f16"),w=v?["shader-f16"]:[],e=await a.requestDevice({requiredFeatures:w,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(a.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(a.limits.maxComputeInvocationsPerWorkgroup,288)}}),g=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(v)try{let n=`enable f16;
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
}`,o=e.createShaderModule({code:n}),s=e.createShaderModule({code:p}),t=await o.getCompilationInfo(),E=await s.getCompilationInfo();if(t.messages.some(G=>G.type==="error")||E.messages.some(G=>G.type==="error"))g=!1;else{let G=new Float32Array(2400);G.fill(1);let H=new Uint16Array(2400);H.fill(10516);let P=new Uint16Array(96);P.fill(14336);let f=new Uint16Array(9216);f.fill(8478);let u=new Uint16Array(96);u.fill(12288);let F=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ue=e.createBuffer({size:G.byteLength,usage:F}),vt=e.createBuffer({size:H.byteLength,usage:F}),Pt=e.createBuffer({size:P.byteLength,usage:F}),Bt=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),Ct=e.createBuffer({size:f.byteLength,usage:F}),kt=e.createBuffer({size:u.byteLength,usage:F}),Ut=e.createBuffer({size:384,usage:ae}),Ze=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ue,0,G),e.queue.writeBuffer(vt,0,H),e.queue.writeBuffer(Pt,0,P),e.queue.writeBuffer(Ct,0,f),e.queue.writeBuffer(kt,0,u);let $e="read-only-storage",Wt="storage",Ht=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Wt}}]}),Wa=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:$e}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Wt}}]}),Mn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Ht]}),compute:{module:o,entryPoint:"main"}}),En=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Wa]}),compute:{module:s,entryPoint:"main"}}),Wn=e.createBindGroup({layout:Ht,entries:[{binding:0,resource:{buffer:ue}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:Pt}},{binding:3,resource:{buffer:Bt}}]}),Hn=e.createBindGroup({layout:Wa,entries:[{binding:0,resource:{buffer:Bt}},{binding:1,resource:{buffer:Ct}},{binding:2,resource:{buffer:kt}},{binding:3,resource:{buffer:Ut}}]}),Jt=e.createCommandEncoder(),Qt=Jt.beginComputePass();Qt.setPipeline(Mn),Qt.setBindGroup(0,Wn),Qt.dispatchWorkgroups(2),Qt.end();let ea=Jt.beginComputePass();ea.setPipeline(En),ea.setBindGroup(0,Hn),ea.dispatchWorkgroups(2),ea.end(),Jt.copyBufferToBuffer(Ut,0,Ze,0,384),e.queue.submit([Jt.finish()]),await e.queue.onSubmittedWorkDone(),await Ze.mapAsync(GPUMapMode.READ);let Tt=new Float32Array(Ze.getMappedRange()),Ha=1.5*.0104*96+.25,Tn=Tt[0]!==0&&Tt[47]!==0&&Tt[95]!==0,Ln=Math.abs(Tt[0]-Ha)<1;g=Tn&&Ln,Ze.unmap(),ue.destroy(),vt.destroy(),Pt.destroy(),Bt.destroy(),Ct.destroy(),kt.destroy(),Ut.destroy(),Ze.destroy(),g||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Tt[0]}, expected ~${Ha.toFixed(2)}) \u2014 falling back to f32`)}}catch{g=!1}let S=r.values().next().value,l=g&&!!S?.rawF16&&!_?.forceF32;console.log(l?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${v}, f16 validated: ${g}, f16 data: ${!!S?.rawF16})`);function h(n){if(l&&n.rawF16){let p=new Uint16Array(n.rawF16);if(p.length%2!==0){let o=new Uint16Array(p.length+1);return o.set(p),o}return p}return n.data}function B(n){if(l&&n.rawF16){let p=n.rawF16.byteLength;return Math.ceil(p/4)*4}return n.data.byteLength}function U(n){return l?Ta(n):n}let I={r:"read-only-storage",s:"storage",u:"uniform"};function m(n){return e.createBindGroupLayout({entries:n.map((p,o)=>({binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:I[p]}}))})}function K(n){return e.createBindGroupLayout({entries:n.map((p,o)=>p==="t"?{binding:o,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:I[p]}})})}let k=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ce=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,pe=GPUBufferUsage.STORAGE,Pe=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,de=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function D(n,p){return e.createBuffer({size:n,usage:p})}function j(n,p){return e.createBindGroup({layout:n,entries:p.map((o,s)=>({binding:s,resource:"size"in o?{buffer:o}:o}))})}function ne(n,p){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:p,entryPoint:"main"}})}let L=e.createShaderModule({code:za}),N=e.createShaderModule({code:nn}),$=e.createShaderModule({code:U(Ja)}),Me=e.createShaderModule({code:U(ma)}),ge=e.createShaderModule({code:U(_a)}),Ee=e.createShaderModule({code:U(fa)}),he=e.createShaderModule({code:U(ha)}),Be=e.createShaderModule({code:U(Na)}),C=e.createShaderModule({code:qa}),J=e.createShaderModule({code:Ya}),we=e.createShaderModule({code:$a}),Q=e.createShaderModule({code:U(Xa)}),be=e.createShaderModule({code:U(Va)}),re=e.createShaderModule({code:U(Za)}),Ye=e.createShaderModule({code:U(Qa)}),Je=new Map;function Ce(n,p){let o=`${n}_${p}`,s=Je.get(o);return s||(s=e.createShaderModule({code:U(en(n,p))}),Je.set(o,s)),s}let ke=m(["r","r","r","s","u"]),ye=m(["r","r","r","r","s","u"]),xe=m(["r","s","u"]),We=m(["r","r","r","s","u"]),Ue=m(["r","s","u"]),Re=m(["r","r","s","u"]),He=m(["r","r","s","u"]),Xe=m(["r","r","r","s","u"]),Te=m(["r","r","r","s","u"]),lt=K(["t","s","u"]),Qe=m(["r","r","r","r","r","r","r","s"]),Oe=m(["r","r","r","r","r","s","u"]),Gt=e.createPipelineLayout({bindGroupLayouts:[ke]}),_t=e.createPipelineLayout({bindGroupLayouts:[ye]}),et=n=>e.createComputePipeline({layout:Gt,compute:{module:n,entryPoint:"main"}}),tt=n=>e.createComputePipeline({layout:_t,compute:{module:n,entryPoint:"main"}}),Rt=et(Me),Ot=et(ge),It=tt(Ee),mt=tt(he),ft=new Map,At=new Map,ht=new Map,St=new Map;ft.set("8,8",Rt),At.set("8,8",Ot),ht.set("8,8",It),St.set("8,8",mt);function at(n,p,o,s,t){let E=`${p},${o}`,G=n.get(E);return G||(G=t(e.createShaderModule({code:U(s(p,o))})),n.set(E,G)),G}let wt=(n,p)=>at(ft,n,p,La,et),Dt=(n,p)=>at(At,n,p,Ra,et),Ft=(n,p)=>at(ht,n,p,Oa,tt),Mt=(n,p)=>at(St,n,p,Ia,tt),Ge=rn.map(n=>{let p=n.stride===2?n.h/2:n.h,o=n.stride===2?n.w/2:n.w,[s,t]=Fa(n.inCh,p),E=n.h>=64,G=p>=16&&n.inCh>=288&&n.outCh>=288&&n.outCh%2===0;return{dwPipeline:E?Dt(s,t):wt(s,t),pwPipeline:G?Mt(s,t):Ft(s,t),dwDispatchX:Math.ceil(o/s),dwDispatchY:Math.ceil(p/t),dwDispatchZ:n.inCh,pwDispatchX:Math.ceil(o/s),pwDispatchY:Math.ceil(p/t),pwDispatchZ:G?n.outCh/2:n.outCh}}),aa=ne(xe,L),bt=ne(We,Be);ne(Ue,C),ne(Re,J);let Ie=ne(He,we),nt=ne(Xe,Q);ne(Te,be),ne(Te,re);let Ae=ne(lt,N),zt=ne(Qe,$),it=ne(Oe,Ye),Nt=1*288*128*128*4,rt=D(3*256*256*4,k),Le=D(3*257*257*4,pe),st=D(12,de);e.queue.writeBuffer(st,0,new Uint32Array([3,256,257]));let Y=D(Nt,ce),le=D(Nt,Pe),Se=D(Nt,pe),ot=D(3072*64*4,k),ut=D(3072*32*4,k),pt=D(1536*16*4,k),V=D(6144*64*4,pe),_e=D(260,Pe),Z=D(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);D(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let ie=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),De=D(8,de);e.queue.writeBuffer(De,0,new Uint32Array([256,257]));let Et=r.get("backbone1.1.weight"),Kt=r.get("backbone1.1.bias");if(!Et||!Kt)throw new Error("Missing input conv weights");let X=h(Et),qt=h(Kt),c=D(X.byteLength,k),i=D(qt.byteLength,k),b=D(28,de);e.queue.writeBuffer(c,0,X),e.queue.writeBuffer(i,0,qt),e.queue.writeBuffer(b,0,new Uint32Array([1,3,24,257,257,128,128]));let W=r.get("backbone6.1.weight"),y=r.get("backbone6.1.bias");if(!W||!y)throw new Error("Missing backbone6.1 conv1x1 weights");let d=h(W),R=h(y),ee=D(d.byteLength,k),x=D(R.byteLength,k),O=D(20,de);e.queue.writeBuffer(ee,0,d),e.queue.writeBuffer(x,0,R),e.queue.writeBuffer(O,0,new Uint32Array([1,96,48,32,32]));let M=r.get("handflag.weight"),q=r.get("handflag.bias");if(!M||!q)throw new Error("Missing handflag weights");let z=h(M),fe=h(q),se=D(z.byteLength,k),T=D(fe.byteLength,k),te=D(12,de);e.queue.writeBuffer(se,0,z),e.queue.writeBuffer(T,0,fe),e.queue.writeBuffer(te,0,new Uint32Array([1,288,1]));let oe=r.get("handedness.weight"),ve=r.get("handedness.bias");if(!oe||!ve)throw new Error("Missing handedness weights");let gt=h(oe),ya=h(ve),na=D(gt.byteLength,k),ia=D(ya.byteLength,k),xa=D(12,de);e.queue.writeBuffer(na,0,gt),e.queue.writeBuffer(ia,0,ya),e.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let va=r.get("reg_3d.weight"),Pa=r.get("reg_3d.bias");if(!va||!Pa)throw new Error("Missing reg_3d weights");let Ba=h(va),Ca=h(Pa),ra=D(Ba.byteLength,k),sa=D(Ca.byteLength,k),ka=D(12,de);e.queue.writeBuffer(ra,0,Ba),e.queue.writeBuffer(sa,0,Ca),e.queue.writeBuffer(ka,0,new Uint32Array([1,288,63]));let ct=rn.map(n=>{let{inCh:p,outCh:o,h:s,w:t,stride:E,prefix:G}=n,H=E===2?s/2:s,P=E===2?t/2:t,f=E===2?1:2,u=r.get(`${G}convs.0.weight`),F=r.get(`${G}convs.0.bias`),ae=r.get(`${G}convs.1.weight`),ue=r.get(`${G}convs.1.bias`);if(!u||!F||!ae||!ue)throw new Error(`Missing weights for ${G}`);let vt=h(u),Pt=h(F),Bt=h(ae),Ct=h(ue),kt=D(vt.byteLength,k),Ut=D(Pt.byteLength,k),Ze=D(Bt.byteLength,k),$e=D(Ct.byteLength,k),Wt=D(32,de),Ht=D(36,de);return e.queue.writeBuffer(kt,0,vt),e.queue.writeBuffer(Ut,0,Pt),e.queue.writeBuffer(Ze,0,Bt),e.queue.writeBuffer($e,0,Ct),e.queue.writeBuffer(Wt,0,new Uint32Array([1,p,s,t,H,P,E,f])),e.queue.writeBuffer(Ht,0,new Uint32Array([1,p,o,H,P,Math.max(0,o-p),E,s,t])),{dwWeight:kt,dwBias:Ut,pwWeight:Ze,pwBias:$e,dwUniform:Wt,pwUniform:Ht,spec:n,outH:H,outW:P}});function yt(n){let p=D(n.length*4,de);return e.queue.writeBuffer(p,0,new Uint32Array(n)),p}let bn=yt([1,96,8,8,16,16]),gn=yt([1,96,16,16,32,32]),yn=yt([1,48,32,32,64,64]);yt([1536*16]),yt([3072*32]),yt([3072*64]);let Ua=j(xe,[rt,Le,st]),Ga=j(We,[Le,c,i,Y,b]),Fe=[],ze=[],Ne=[],Ke=[];for(let n of ct)Fe.push(j(ke,[Y,n.dwWeight,n.dwBias,Se,n.dwUniform])),ze.push(j(ye,[Se,Y,n.pwWeight,n.pwBias,le,n.pwUniform])),Ne.push(j(ke,[le,n.dwWeight,n.dwBias,Se,n.dwUniform])),Ke.push(j(ye,[Se,le,n.pwWeight,n.pwBias,Y,n.pwUniform]));let xn=j(He,[Y,pt,le,bn]),vn=j(He,[Y,ut,le,gn]),Pn=j(Xe,[Y,ee,x,V,O]),Bn=j(He,[V,ot,le,yn]);j(Te,[Y,se,T,_e,te]),j(Te,[Y,na,ia,_e,xa]),j(Te,[Y,ra,sa,_e,ka]);let je=j(lt,[ie.createView(),Le,De]),Cn=j(Qe,[Y,se,T,na,ia,ra,sa,_e]),oa=24,Aa=[],Sa=[];for(let n=oa;n<ct.length;n++){let p=ct[n];Aa.push(j(Oe,[Y,p.dwWeight,p.dwBias,p.pwWeight,p.pwBias,le,p.dwUniform])),Sa.push(j(Oe,[le,p.dwWeight,p.dwBias,p.pwWeight,p.pwBias,Y,p.dwUniform]))}let ua=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});ua.globalCompositeOperation="copy";let Da=new OffscreenCanvas(9,8),$t=Da.getContext("webgpu"),Yt=null,pa=null;if($t){$t.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let n=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),p=e.createShaderModule({code:tn}),o=e.createShaderModule({code:an});Yt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[n]}),vertex:{module:p,entryPoint:"vs"},fragment:{module:o,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),pa=e.createBindGroup({layout:n,entries:[{binding:0,resource:{buffer:_e}}]})}let Xt=new Float32Array(1),jt=new Float32Array(1),Vt=new Float32Array(63);function qe(n,p){let o=!0,s=0,t=n.beginComputePass();for(t.setPipeline(bt),t.setBindGroup(0,Ga),t.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);s<=In;s++){let P=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=Ge[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,P),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let E=o?Y:le;for(n.copyBufferToBuffer(E,0,ot,0,3072*64*4),t=n.beginComputePass();s<=Fn;s++){let P=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=Ge[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,P),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let G=o?Y:le;for(n.copyBufferToBuffer(G,0,ut,0,3072*32*4),t=n.beginComputePass();s<=zn;s++){let P=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=Ge[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,P),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let H=o?Y:le;for(n.copyBufferToBuffer(H,0,pt,0,1536*16*4),t=n.beginComputePass();s<=Nn;s++){let P=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=Ge[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,P),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.setPipeline(Ie),t.setBindGroup(0,xn),t.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),t.end(),o=!1,t=n.beginComputePass();{let P=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=Ge[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,P),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}t.setPipeline(Ie),t.setBindGroup(0,vn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),t.end(),o=!1,t=n.beginComputePass();{let P=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=Ge[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,P),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}for(t.setPipeline(nt),t.setBindGroup(0,Pn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),t.setPipeline(Ie),t.setBindGroup(0,Bn),t.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),t.end(),o=!1,t=n.beginComputePass();s<oa;s++){let P=o?Fe[s]:Ne[s],f=o?ze[s]:Ke[s],u=Ge[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,P),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}for(;s<ct.length;s++){let P=s-oa,f=o?Aa[P]:Sa[P],u=ct[s];t.setPipeline(it),t.setBindGroup(0,f),t.dispatchWorkgroups(u.outW,u.outH,1),o=!o}t.setPipeline(zt),t.setBindGroup(0,Cn),t.dispatchWorkgroups(1),t.end(),p&&n.copyBufferToBuffer(_e,0,p,0,260)}async function Zt(n){e.queue.writeBuffer(rt,0,n);let p=e.createCommandEncoder();{let t=p.beginComputePass();t.setPipeline(aa),t.setBindGroup(0,Ua),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}qe(p,Z),e.queue.submit([p.finish()]);let o=Z.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(Z.getMappedRange());return Xt[0]=s[0],jt[0]=s[1],Vt.set(s.subarray(2,65)),Z.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(jt),landmarks:new Float32Array(Vt)}}async function ca(n){e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let t=p.beginComputePass();t.setPipeline(Ae),t.setBindGroup(0,je),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}qe(p,Z),e.queue.submit([p.finish()]);let o=Z.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(Z.getMappedRange());return Xt[0]=s[0],jt[0]=s[1],Vt.set(s.subarray(2,65)),Z.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(jt),landmarks:new Float32Array(Vt)}}async function Ma(n){if(!Yt||!pa||!$t)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let P=p.beginComputePass();P.setPipeline(Ae),P.setBindGroup(0,je),P.dispatchWorkgroups(16,16,1),P.end()}qe(p,null);let o=$t.getCurrentTexture(),s=p.beginRenderPass({colorAttachments:[{view:o.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});s.setPipeline(Yt),s.setBindGroup(0,pa),s.draw(3),s.end(),e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),ua.drawImage(Da,0,0);let E=ua.getImageData(0,0,9,8).data,G=new Float32Array(65),H=new DataView(new ArrayBuffer(4));for(let P=0;P<65;P++){let f=P*4;H.setUint8(0,E[f]),H.setUint8(1,E[f+1]),H.setUint8(2,E[f+2]),H.setUint8(3,E[f+3]),G[P]=H.getFloat32(0)}return{handflag:new Float32Array([G[0]]),handedness:new Float32Array([G[1]]),landmarks:new Float32Array(G.subarray(2,65))}}let kn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),da=0,Un=[Z,kn],xt=null,Ve=null;async function la(n){let p=Un[da];da=1-da,e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);let o=e.createCommandEncoder();{let t=o.beginComputePass();t.setPipeline(Ae),t.setBindGroup(0,je),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}qe(o,p),e.queue.submit([o.finish()]);let s=null;if(xt!==null&&Ve!==null){await xt;let t=new Float32Array(Ve.getMappedRange());s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))},Ve.unmap()}return Ve=p,xt=p.mapAsync(GPUMapMode.READ),s}async function Ea(){if(!xt||!Ve)return null;await xt;let n=new Float32Array(Ve.getMappedRange()),p={handflag:new Float32Array([n[0]]),handedness:new Float32Array([n[1]]),landmarks:new Float32Array(n.subarray(2,65))};return Ve.unmap(),xt=null,Ve=null,p}async function Gn(n=50){let p=new Float32Array(196608);for(let t=0;t<5;t++)await Zt(p);let o=[];for(let t=0;t<n;t++){let E=performance.now();await Zt(p),o.push(performance.now()-E)}let s=o.reduce((t,E)=>t+E,0)/o.length;return{avgMs:s,fps:1e3/s}}async function An(n=50){let p=new Float32Array(196608);for(let G=0;G<5;G++)await Zt(p);let o=[];for(let G=0;G<n;G++){let H=e.createCommandEncoder();{let f=H.beginComputePass();f.setPipeline(aa),f.setBindGroup(0,Ua),f.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),f.end()}qe(H,Z);let P=performance.now();e.queue.submit([H.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-P)}o.sort((G,H)=>G-H);let s=o.reduce((G,H)=>G+H,0)/o.length,t=o[Math.floor(o.length/2)],E=o[0];return{avgMs:s,fps:1e3/s,medianMs:t,minMs:E}}function ei(n){e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let o=p.beginComputePass();o.setPipeline(Ae),o.setBindGroup(0,je),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),o.end()}qe(p,Z),e.queue.submit([p.finish()])}async function Sn(n,p=50){function o(f){let u=[...f].sort((F,ae)=>F-ae);return{median:u[Math.floor(u.length/2)],min:u[0]}}for(let f=0;f<10;f++)await ca(n);let s=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);let u=e.createCommandEncoder();{let ae=u.beginComputePass();ae.setPipeline(Ae),ae.setBindGroup(0,je),ae.dispatchWorkgroups(16,16,1),ae.end()}qe(u,Z);let F=performance.now();e.queue.submit([u.finish()]),await e.queue.onSubmittedWorkDone(),s.push(performance.now()-F)}let t=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);let u=e.createCommandEncoder();{let ue=u.beginComputePass();ue.setPipeline(Ae),ue.setBindGroup(0,je),ue.dispatchWorkgroups(16,16,1),ue.end()}qe(u,Z),e.queue.submit([u.finish()]);let F=Z.mapAsync(GPUMapMode.READ),ae=performance.now();await e.queue.onSubmittedWorkDone(),await F,Z.getMappedRange(),Z.unmap(),t.push(performance.now()-ae)}let E=[];for(let f=0;f<p;f++){e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);let u=e.createCommandEncoder();{let ae=u.beginComputePass();ae.setPipeline(Ae),ae.setBindGroup(0,je),ae.dispatchWorkgroups(16,16,1),ae.end()}qe(u,Z),e.queue.submit([u.finish()]);let F=performance.now();await Z.mapAsync(GPUMapMode.READ),Z.getMappedRange(),Z.unmap(),E.push(performance.now()-F)}let G=[];for(let f=0;f<p;f++){let u=performance.now();await ca(n),G.push(performance.now()-u)}await la(n);let H=[];for(let f=0;f<p;f++){let u=performance.now();await la(n),H.push(performance.now()-u)}await Ea();let P=null;if(Yt){let f=[];for(let u=0;u<p;u++){let F=performance.now();await Ma(n),f.push(performance.now()-F)}P=o(f)}return{gpuOnly:o(s),mapAsyncOnly:o(t),mapAsyncNoWait:o(E),total:o(G),pipelined:o(H),renderReadback:P}}async function Dn(n){let p=[];async function o(t,E,G){let H=e.createBuffer({size:E,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),P=e.createCommandEncoder();P.copyBufferToBuffer(t,0,H,0,E),e.queue.submit([P.finish()]),await e.queue.onSubmittedWorkDone(),await H.mapAsync(GPUMapMode.READ);let f=new Float32Array(H.getMappedRange()),u=1/0,F=-1/0,ae=0;for(let ue=0;ue<f.length;ue++)f[ue]<u&&(u=f[ue]),f[ue]>F&&(F=f[ue]),f[ue]!==0&&ae++;H.unmap(),H.destroy(),p.push({layer:G,stats:{min:u,max:F,nonZero:ae,total:f.length}})}e.queue.copyExternalImageToTexture({source:n},{texture:ie},[256,256]);{let t=e.createCommandEncoder(),E=t.beginComputePass();E.setPipeline(Ae),E.setBindGroup(0,je),E.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),E.end(),e.queue.submit([t.finish()])}await o(Le,Math.min(Le.size,3*257*257*4),"canvas\u2192bufInput");{let t=e.createCommandEncoder(),E=t.beginComputePass();E.setPipeline(bt),E.setBindGroup(0,Ga),E.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),E.end(),e.queue.submit([t.finish()])}await o(Y,Math.min(Y.size,3072*128*4),"inputConv\u2192bufA");let s=!0;for(let t=0;t<Math.min(ct.length,6);t++){let E=s?Fe[t]:Ne[t],G=s?ze[t]:Ke[t],H=Ge[t],P=ct[t];{let u=e.createCommandEncoder(),F=u.beginComputePass();F.setPipeline(H.dwPipeline),F.setBindGroup(0,E),F.dispatchWorkgroups(H.dwDispatchX,H.dwDispatchY,H.dwDispatchZ),F.end(),e.queue.submit([u.finish()])}await o(Se,Math.min(Se.size,P.spec.inCh*P.outH*P.outW*4),`layer${t}.DW\u2192bufDW (${P.spec.prefix})`);{let u=e.createCommandEncoder(),F=u.beginComputePass();F.setPipeline(H.pwPipeline),F.setBindGroup(0,G),F.dispatchWorkgroups(H.pwDispatchX,H.pwDispatchY,H.pwDispatchZ),F.end(),e.queue.submit([u.finish()])}let f=s?le:Y;await o(f,Math.min(f.size,P.spec.outCh*P.outH*P.outW*4),`layer${t}.PW\u2192buf${s?"B":"A"} (${P.spec.prefix})`),s=!s}return p}return{device:e,run:Zt,runFromCanvas:ca,runFromCanvasViaRender:Ma,runFromCanvasPipelined:la,flushPipelined:Ea,benchmark:Gn,benchmarkGPU:An,benchmarkDiagnostic:Sn,debugLayerOutputs:Dn}}function dt(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var sn=dt(`
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
`),on=dt(`
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
`),un=dt(`
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
`),pn=dt(`
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
`),cn=dt(`
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
`),dn=dt(`
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
`),ln=dt(`
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
`);async function wa(r,_){let a;if(_)a=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let c=await navigator.gpu.requestAdapter();if(!c)throw new Error("No GPU adapter found");a=await c.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(c.limits.maxStorageBuffersPerShaderStage,8)}})}let v={r:"read-only-storage",s:"storage",u:"uniform"};function w(c){return a.createBindGroupLayout({entries:c.map((i,b)=>i==="t"?{binding:b,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:b,visibility:GPUShaderStage.COMPUTE,buffer:{type:v[i]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,g=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,S=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(c,i){return a.createBuffer({size:Math.max(c,4),usage:i})}function h(c,i,b){a.queue.writeBuffer(c,i,b)}function B(c){let i=l(c.data.byteLength,e);return h(i,0,c.data),i}let U=Array.from(r.keys());function I(c){let i=r.get(c);if(!i)throw new Error(`Weight not found: ${c}`);return i}function m(...c){let i=U.find(b=>c.every(W=>b.includes(W)));if(!i)throw new Error(`Weight not found for: ${c.join(", ")}`);return I(i)}function K(c){let[,i,b,W]=c.shape,y=new Float32Array(W*25);for(let d=0;d<W;d++)for(let R=0;R<i;R++)for(let ee=0;ee<b;ee++)y[d*25+R*5+ee]=c.data[R*b*W+ee*W+d];return y}function k(c){let[i,,,b]=c.shape,W=new Float32Array(i*b);for(let y=0;y<i;y++)for(let d=0;d<b;d++)W[y*b+d]=c.data[y*b+d];return W}let ce=a.createShaderModule({code:sn}),pe=a.createShaderModule({code:on}),Pe=a.createShaderModule({code:un}),de=a.createShaderModule({code:pn}),D=a.createShaderModule({code:dn}),j=a.createShaderModule({code:cn}),ne=a.createShaderModule({code:ln}),L=w(["r","r","r","r","s","u"]),N=w(["r","r","r","s","u"]),$=w(["r","r","r","r","r","s","u"]),Me=w(["r","r","r","s","u"]),ge=w(["r","r","r","r","s","u"]),Ee=w(["r","r","s","u"]),he=w(["t","s","u"]);function Be(c,i){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[c]}),compute:{module:i,entryPoint:"main"}})}let C=Be(L,ce),J=Be(N,pe),we=Be($,Pe),Q=Be(Me,de),be=Be(ge,D),re=Be(Ee,j),Ye=Be(he,ne),Je=m("conv2d/Conv2D"),Ce=m("batch_normalization/","conv2d/Conv2D"),ke=m("p_re_lu/"),ye=B(Je),xe=B(Ce),We=B(ke),Re=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12}].map(c=>{let i=m(c.dwKey),b=m(c.pwKey),W=m(c.bnKey),y=m(c.preluKey),d=K(i),R=l(d.byteLength,e);h(R,0,d);let ee=new Float32Array(c.inCh),x=l(ee.byteLength,e);h(x,0,ee);let O=k(b),M=l(O.byteLength,e);h(M,0,O);let q=B(W),z=B(y);return{dwWeightBuf:R,dwBiasBuf:x,pwWeightBuf:M,pwBiasBuf:q,alphaBuf:z,inCh:c.inCh,outCh:c.outCh,stride:c.stride,inH:c.inH}}),He=k(m("conv2d_20/Conv2D")),Xe=l(He.byteLength,e);h(Xe,0,He);let Te=B(m("batch_normalization_20/")),lt=B(m("p_re_lu_20/")),Qe={dwWeightBuf:(()=>{let c=K(m("depthwise_conv2d_19/")),i=l(c.byteLength,e);return h(i,0,c),i})(),dwBiasBuf:(()=>{let c=new Float32Array(256),i=l(c.byteLength,e);return h(i,0,c),i})(),pwWeightBuf:Xe,pwBiasBuf:Te,alphaBuf:lt,inCh:256,outCh:256,stride:1,inH:12},Oe={dwWeightBuf:(()=>{let c=K(m("depthwise_conv2d_20/")),i=l(c.byteLength,e);return h(i,0,c),i})(),dwBiasBuf:(()=>{let c=new Float32Array(256),i=l(c.byteLength,e);return h(i,0,c),i})(),pwWeightBuf:(()=>{let c=k(m("conv2d_21/")),i=l(c.byteLength,e);return h(i,0,c),i})(),pwBiasBuf:B(m("batch_normalization_21/")),alphaBuf:B(m("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},Gt=k(m("conv2d_23/Conv2D")),_t=l(Gt.byteLength,e);h(_t,0,Gt);let et=B(m("batch_normalization_23/")),tt=B(m("p_re_lu_23/")),Rt={dwWeightBuf:(()=>{let c=K(m("depthwise_conv2d_21/")),i=l(c.byteLength,e);return h(i,0,c),i})(),dwBiasBuf:(()=>{let c=new Float32Array(128),i=l(c.byteLength,e);return h(i,0,c),i})(),pwWeightBuf:(()=>{let c=k(m("conv2d_24/")),i=l(c.byteLength,e);return h(i,0,c),i})(),pwBiasBuf:B(m("batch_normalization_24/")),alphaBuf:B(m("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},Ot={dwWeightBuf:(()=>{let c=K(m("depthwise_conv2d_22/")),i=l(c.byteLength,e);return h(i,0,c),i})(),dwBiasBuf:(()=>{let c=new Float32Array(128),i=l(c.byteLength,e);return h(i,0,c),i})(),pwWeightBuf:(()=>{let c=k(m("conv2d_25/Conv2D1")),i=l(c.byteLength,e);return h(i,0,c),i})(),pwBiasBuf:B(m("batch_normalization_25/")),alphaBuf:B(m("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},It=k(m("classifier_palm_16_NO_PRUNING/Conv2D")),mt=l(It.byteLength,e);h(mt,0,It);let ft=B(m("classifier_palm_16_NO_PRUNING/BiasAdd")),At=k(m("regressor_palm_16_NO_PRUNING/Conv2D")),ht=l(At.byteLength,e);h(ht,0,At);let St=B(m("regressor_palm_16_NO_PRUNING/BiasAdd")),at=k(m("classifier_palm_8_NO_PRUNING/Conv2D")),wt=l(at.byteLength,e);h(wt,0,at);let Dt=B(m("classifier_palm_8_NO_PRUNING/BiasAdd")),Ft=k(m("regressor_palm_8_NO_PRUNING/Conv2D")),Mt=l(Ft.byteLength,e);h(Mt,0,Ft);let Ge=B(m("regressor_palm_8_NO_PRUNING/BiasAdd")),aa=36864*3,bt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Ie=l(36864*3*4,e),nt=l(bt,g),Ae=l(bt,g),zt=l(bt,g),it=l(576*128*4,g|GPUBufferUsage.COPY_DST),Nt=l(576*128*4,g),rt=l(864*4,A),Le=l(15552*4,A),st=l(576*2*4,A),Y=l(576*36*4,A),le=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Se=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ot=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ut=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pt=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function V(c,i){return Math.ceil(c/i)}function _e(c){let i=l(c.byteLength,S);return h(i,0,c),i}let Z=_e(new Uint32Array([1,3,32,192,192,96,96]));function ie(c,i,b,W,y){let d=i.stride===2?i.inH/2:i.inH,R=d,ee=i.stride===2?1:2,x=_e(new Uint32Array([1,i.inCh,i.inH,i.inH,d,R,i.stride,ee])),O=a.createBindGroup({layout:N,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:i.dwWeightBuf}},{binding:2,resource:{buffer:i.dwBiasBuf}},{binding:3,resource:{buffer:zt}},{binding:4,resource:{buffer:x}}]}),M=c.beginComputePass();M.setPipeline(J),M.setBindGroup(0,O),M.dispatchWorkgroups(V(R,8),V(d,8),i.inCh),M.end();let q=i.inCh,z=_e(new Uint32Array([1,i.inCh,i.outCh,d,R,q,i.stride,i.inH,i.inH])),fe=a.createBindGroup({layout:$,entries:[{binding:0,resource:{buffer:zt}},{binding:1,resource:{buffer:y}},{binding:2,resource:{buffer:i.pwWeightBuf}},{binding:3,resource:{buffer:i.pwBiasBuf}},{binding:4,resource:{buffer:i.alphaBuf}},{binding:5,resource:{buffer:W}},{binding:6,resource:{buffer:z}}]}),se=c.beginComputePass();se.setPipeline(we),se.setBindGroup(0,fe),se.dispatchWorkgroups(V(R,8),V(d,8),i.outCh),se.end()}function De(c,i,b,W,y,d,R,ee,x){let O=_e(new Uint32Array([1,d,R,ee,x])),M=a.createBindGroup({layout:Me,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:W}},{binding:3,resource:{buffer:y}},{binding:4,resource:{buffer:O}}]}),q=c.beginComputePass();q.setPipeline(Q),q.setBindGroup(0,M),q.dispatchWorkgroups(V(x,8),V(ee,8),R),q.end()}function Et(c,i,b,W,y,d,R,ee,x,O){let M=_e(new Uint32Array([1,R,ee,x,O])),q=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:W}},{binding:3,resource:{buffer:y}},{binding:4,resource:{buffer:d}},{binding:5,resource:{buffer:M}}]}),z=c.beginComputePass();z.setPipeline(be),z.setBindGroup(0,q),z.dispatchWorkgroups(V(O,8),V(x,8),ee),z.end()}async function Kt(c){a.queue.copyExternalImageToTexture({source:c},{texture:pt},[192,192]);let i=_e(new Uint32Array([192,192,192])),b=a.createBindGroup({layout:he,entries:[{binding:0,resource:pt.createView()},{binding:1,resource:{buffer:Ie}},{binding:2,resource:{buffer:i}}]}),W=a.createCommandEncoder();{let T=W.beginComputePass();T.setPipeline(Ye),T.setBindGroup(0,b),T.dispatchWorkgroups(V(192,16),V(192,16),1),T.end()}{let T=a.createBindGroup({layout:L,entries:[{binding:0,resource:{buffer:Ie}},{binding:1,resource:{buffer:ye}},{binding:2,resource:{buffer:xe}},{binding:3,resource:{buffer:We}},{binding:4,resource:{buffer:nt}},{binding:5,resource:{buffer:Z}}]}),te=W.beginComputePass();te.setPipeline(C),te.setBindGroup(0,T),te.dispatchWorkgroups(V(96,8),V(96,8),32),te.end()}let y=nt,d=Ae;for(let T=0;T<Re.length;T++){let te=Re[T];ie(W,te,y,d,y);let oe=y;y=d,d=oe,T===10&&W.copyBufferToBuffer(y,0,it,0,576*128*4)}ie(W,Qe,y,d,y);{let T=y;y=d,d=T}ie(W,Oe,y,d,y);{let T=y;y=d,d=T}De(W,y,mt,ft,rt,256,6,12,12),De(W,y,ht,St,Le,256,108,12,12),Et(W,y,_t,et,tt,d,256,128,12,12);{let T=y;y=d,d=T}{let T=_e(new Uint32Array([1,128,12,12,24,24])),te=a.createBindGroup({layout:Ee,entries:[{binding:0,resource:{buffer:y}},{binding:1,resource:{buffer:it}},{binding:2,resource:{buffer:d}},{binding:3,resource:{buffer:T}}]}),oe=W.beginComputePass();oe.setPipeline(re),oe.setBindGroup(0,te),oe.dispatchWorkgroups(V(24,8),V(24,8),128),oe.end()}{let T=y;y=d,d=T}ie(W,Rt,y,d,y);{let T=y;y=d,d=T}ie(W,Ot,y,d,y);{let T=y;y=d,d=T}De(W,y,wt,Dt,st,128,2,24,24),De(W,y,Mt,Ge,Y,128,36,24,24),a.queue.submit([W.finish()]);let R=a.createCommandEncoder();R.copyBufferToBuffer(rt,0,le,0,864*4),R.copyBufferToBuffer(Le,0,Se,0,15552*4),R.copyBufferToBuffer(st,0,ot,0,576*2*4),R.copyBufferToBuffer(Y,0,ut,0,576*36*4),a.queue.submit([R.finish()]),await Promise.all([le.mapAsync(GPUMapMode.READ),Se.mapAsync(GPUMapMode.READ),ot.mapAsync(GPUMapMode.READ),ut.mapAsync(GPUMapMode.READ)]);let ee=new Float32Array(le.getMappedRange()).slice(),x=new Float32Array(Se.getMappedRange()).slice(),O=new Float32Array(ot.getMappedRange()).slice(),M=new Float32Array(ut.getMappedRange()).slice();le.unmap(),Se.unmap(),ot.unmap(),ut.unmap();let q=2016,z=new Float32Array(q),fe=new Float32Array(q*18),se=0;for(let T=0;T<12;T++)for(let te=0;te<12;te++)for(let oe=0;oe<6;oe++){z[se]=ee[oe*144+T*12+te];for(let ve=0;ve<18;ve++){let gt=oe*18+ve;fe[se*18+ve]=x[gt*144+T*12+te]}se++}for(let T=0;T<24;T++)for(let te=0;te<24;te++)for(let oe=0;oe<2;oe++){z[se]=O[oe*576+T*24+te];for(let ve=0;ve<18;ve++){let gt=oe*18+ve;fe[se*18+ve]=M[gt*576+T*24+te]}se++}return{scores:z,regressors:fe}}async function X(c,i){let b=a.createBuffer({size:i*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),W=a.createCommandEncoder();W.copyBufferToBuffer(c,0,b,0,i*4),a.queue.submit([W.finish()]),await b.mapAsync(GPUMapMode.READ);let y=new Float32Array(b.getMappedRange()).slice();return b.unmap(),b.destroy(),y}async function qt(c){a.queue.copyExternalImageToTexture({source:c},{texture:pt},[192,192]);function i(M,q=1e3){let z=M.slice(0,q);return{min:Math.min(...z),max:Math.max(...z),mean:z.reduce((fe,se)=>fe+se,0)/z.length,nonZero:z.filter(fe=>fe!==0).length,sample:Array.from(z.slice(0,10))}}let b={},W=_e(new Uint32Array([192,192,192])),y=a.createBindGroup({layout:he,entries:[{binding:0,resource:pt.createView()},{binding:1,resource:{buffer:Ie}},{binding:2,resource:{buffer:W}}]}),d=a.createCommandEncoder(),R=d.beginComputePass();R.setPipeline(Ye),R.setBindGroup(0,y),R.dispatchWorkgroups(V(192,16),V(192,16),1),R.end(),a.queue.submit([d.finish()]),b.input=i(await X(Ie,36864*3)),d=a.createCommandEncoder();let ee=a.createBindGroup({layout:L,entries:[{binding:0,resource:{buffer:Ie}},{binding:1,resource:{buffer:ye}},{binding:2,resource:{buffer:xe}},{binding:3,resource:{buffer:We}},{binding:4,resource:{buffer:nt}},{binding:5,resource:{buffer:Z}}]});R=d.beginComputePass(),R.setPipeline(C),R.setBindGroup(0,ee),R.dispatchWorkgroups(V(96,8),V(96,8),32),R.end(),a.queue.submit([d.finish()]),b.initConv=i(await X(nt,9216*32));let x=nt,O=Ae;for(let M=0;M<Re.length;M++){let q=Re[M];d=a.createCommandEncoder(),ie(d,q,x,O,x),a.queue.submit([d.finish()]);let z=x;if(x=O,O=z,M===0||M===3||M===7||M===11||M===17){let fe=q.stride===2?q.inH/2:q.inH,se=fe*fe*q.outCh;b[`block${M}`]=i(await X(x,se))}M===10&&(d=a.createCommandEncoder(),d.copyBufferToBuffer(x,0,it,0,576*128*4),a.queue.submit([d.finish()]))}d=a.createCommandEncoder(),ie(d,Qe,x,O,x),a.queue.submit([d.finish()]);{let M=x;x=O,O=M}b.extraBlockA=i(await X(x,144*256)),d=a.createCommandEncoder(),ie(d,Oe,x,O,x),a.queue.submit([d.finish()]);{let M=x;x=O,O=M}b.extraBlockB=i(await X(x,144*256)),d=a.createCommandEncoder(),De(d,x,mt,ft,rt,256,6,12,12),a.queue.submit([d.finish()]),b.cls16=i(await X(rt,864)),d=a.createCommandEncoder(),De(d,x,ht,St,Le,256,108,12,12),a.queue.submit([d.finish()]),b.reg16=i(await X(Le,15552),500),d=a.createCommandEncoder(),Et(d,x,_t,et,tt,O,256,128,12,12),a.queue.submit([d.finish()]);{let M=x;x=O,O=M}b.fpnProj=i(await X(x,18432)),b.backbone2Skip=i(await X(it,576*128)),d=a.createCommandEncoder();{let M=_e(new Uint32Array([1,128,12,12,24,24])),q=a.createBindGroup({layout:Ee,entries:[{binding:0,resource:{buffer:x}},{binding:1,resource:{buffer:it}},{binding:2,resource:{buffer:O}},{binding:3,resource:{buffer:M}}]}),z=d.beginComputePass();z.setPipeline(re),z.setBindGroup(0,q),z.dispatchWorkgroups(V(24,8),V(24,8),128),z.end()}a.queue.submit([d.finish()]);{let M=x;x=O,O=M}b.fpnUpsample=i(await X(x,576*128)),d=a.createCommandEncoder(),ie(d,Rt,x,O,x),a.queue.submit([d.finish()]);{let M=x;x=O,O=M}b.fpnBlock1=i(await X(x,576*128)),d=a.createCommandEncoder(),ie(d,Ot,x,O,x),a.queue.submit([d.finish()]);{let M=x;x=O,O=M}return b.fpnBlock2=i(await X(x,576*128)),d=a.createCommandEncoder(),De(d,x,wt,Dt,st,128,2,24,24),a.queue.submit([d.finish()]),b.cls8=i(await X(st,576*2)),d=a.createCommandEncoder(),De(d,x,Mt,Ge,Y,128,36,24,24),a.queue.submit([d.finish()]),b.reg8=i(await X(Y,576*36)),b.initWeights=i(await X(ye,100),100),b.initBias=i(await X(xe,32),32),b.cls16Weights=i(await X(mt,100),100),b.cls16Bias=i(await X(ft,6),6),b.cls8Weights=i(await X(wt,100),100),b.cls8Bias=i(await X(Dt,2),2),b.fpnProjWeights=i(await X(_t,100),100),b}return{device:a,run:Kt,debugRun:qt}}function Kn(){let r=[];for(let _=0;_<12;_++)for(let a=0;a<12;a++){let v=(a+.5)/12,w=(_+.5)/12;for(let e=0;e<6;e++)r.push({x:v,y:w})}for(let _=0;_<24;_++)for(let a=0;a<24;a++){let v=(a+.5)/24,w=(_+.5)/24;for(let e=0;e<2;e++)r.push({x:v,y:w})}return r}var _n=Kn();function qn(r){return 1/(1+Math.exp(-r))}function mn(r,_){let a=[],{scores:v,regressors:w}=r,e=192;for(let g=0;g<_n.length;g++){let A=qn(v[g]);if(A<_)continue;let S=_n[g],l=g*18,h=S.x+w[l+0]/e,B=S.y+w[l+1]/e,U=w[l+2]/e,I=w[l+3]/e,m=[];for(let K=0;K<7;K++){let k=S.x+w[l+4+K*2]/e,ce=S.y+w[l+4+K*2+1]/e;m.push([k,ce])}a.push({score:A,box:[h,B,U,I],keypoints:m})}return a}function fn(r,_){if(r.length===0)return[];let a=[...r].sort((e,g)=>g.score-e.score),v=[],w=new Set;for(let e=0;e<a.length;e++)if(!w.has(e)){v.push(a[e]);for(let g=e+1;g<a.length;g++)w.has(g)||$n(a[e],a[g])>_&&w.add(g)}return v}function $n(r,_){let a=r.box[0]-r.box[2]/2,v=r.box[1]-r.box[3]/2,w=r.box[0]+r.box[2]/2,e=r.box[1]+r.box[3]/2,g=_.box[0]-_.box[2]/2,A=_.box[1]-_.box[3]/2,S=_.box[0]+_.box[2]/2,l=_.box[1]+_.box[3]/2,h=Math.max(a,g),B=Math.max(v,A),U=Math.min(w,S),I=Math.min(e,l),m=Math.max(0,U-h),K=Math.max(0,I-B),k=m*K,ce=(w-a)*(e-v),pe=(S-g)*(l-A),Pe=ce+pe-k;return Pe>0?k/Pe:0}function Yn(r){let[_,a,v,w]=r.box,e=r.keypoints[0],g=r.keypoints[2],A=g[0]-e[0],S=g[1]-e[1],l=Math.atan2(A,S),B=Math.max(v,w)*2.6,U=.5,I=Math.sqrt(A*A+S*S),m=I>0?A/I*B*U*.5:0,K=I>0?S/I*B*U*.5:0;return{centerX:_+m,centerY:a+K,width:B,height:B,rotation:l}}function ba(r,_={}){let{scoreThreshold:a=.5,nmsThreshold:v=.3,maxHands:w=2}=_;async function e(A){let S=await r.run(A),l=mn(S,a);return fn(l,v).slice(0,w).map(Yn)}async function g(A){let S=await r.run(A),l=mn(S,a);return fn(l,v).slice(0,w)}return{detect:e,detectRaw:g,model:r}}function hn(r,_=256){let a=Math.cos(r.rotation),v=Math.sin(r.rotation),w=r.width/_,e=r.height/_,g=w*a,A=-e*v,S=w*v,l=e*a,h=r.centerX-(g*_/2+A*_/2),B=r.centerY-(S*_/2+l*_/2),U=g*l-A*S,I=l/U,m=-A/U,K=-S/U,k=g/U,ce=-(I*h+m*B),pe=-(K*h+k*B);return{forward:[g,A,h,S,l,B],inverse:[I,m,ce,K,k,pe]}}function ga(r,_){let{forward:a}=hn(_,1),[v,w,e,g,A,S]=a;return r.map(l=>({x:v*l.x+w*l.y+e,y:g*l.x+A*l.y+S,z:l.z}))}var wn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Xn(r={}){let{weightsUrl:_,scoreThreshold:a=.5,forceF32:v=!1}=r;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let w=_??wn,e=w.endsWith("/")?w:`${w}/`,g=`${e}weights_f16.json`,A=`${e}weights_f16.bin`,[S,l]=await Promise.all([fetch(g),fetch(A)]);if(!S.ok)throw new Error(`Failed to fetch weights metadata: ${S.status}`);if(!l.ok)throw new Error(`Failed to fetch weights binary: ${l.status}`);let h=await S.json(),B=await l.arrayBuffer(),U=Lt(h,B),I=await ta(U,{forceF32:v});if(!v){let L=new OffscreenCanvas(256,256),N=L.getContext("2d");N.fillStyle="#886644",N.fillRect(0,0,256,256),N.fillStyle="#cc9966",N.fillRect(50,50,156,156);let $=await I.runFromCanvas(L);$.landmarks.every(ge=>ge===0)&&$.handflag.every(ge=>ge===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),I.device.destroy(),I=await ta(U,{forceF32:!0}))}let m=null;function K(){return m||(m=new OffscreenCanvas(256,256)),m}async function k(L){if(L instanceof HTMLCanvasElement||L instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&L instanceof ImageBitmap)return L;let N=K();N.width=256,N.height=256;let $=N.getContext("2d");return L instanceof ImageData?$.putImageData(L,0,0):$.drawImage(L,0,0,256,256),N}function ce(L,N,$){let Me=L[0];if(Me<a)return null;let ge=N[0]>.5,Ee=[];for(let he=0;he<21;he++)Ee.push({x:$[he*3],y:$[he*3+1],z:$[he*3+2]});return{score:Me,handedness:ge?"right":"left",landmarks:Ee}}async function pe(L){let N=await k(L),$=await I.runFromCanvas(N);return ce($.handflag,$.handedness,$.landmarks)}async function Pe(L){let N=await k(L),$=await I.runFromCanvasPipelined(N);return $?ce($.handflag,$.handedness,$.landmarks):null}async function de(){let L=await I.flushPipelined();return L?ce(L.handflag,L.handedness,L.landmarks):null}function D(){I.device.destroy(),m=null}async function j(L){let N=await k(L);return I.benchmarkDiagnostic(N)}async function ne(L){let N=await k(L);return I.debugLayerOutputs(N)}return{detect:pe,detectPipelined:Pe,flushPipelined:de,dispose:D,benchmarkDiagnostic:j,debugLayerOutputs:ne}}async function jn(r={}){let{weightsUrl:_,palmWeightsUrl:a,scoreThreshold:v=.5,palmScoreThreshold:w=.5,maxHands:e=2,forceF32:g=!1}=r;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let A=_??wn,S=A.endsWith("/")?A:`${A}/`;if(!a)throw new Error("palmWeightsUrl is required for createFullHandpose");let l=a.endsWith("/")?a:`${a}/`,[h,B,U,I]=await Promise.all([fetch(`${S}weights_f16.json`),fetch(`${S}weights_f16.bin`),fetch(`${l}palm_detection_weights.json`),fetch(`${l}palm_detection_weights.bin`)]);if(!h.ok)throw new Error(`Failed to fetch landmark weights metadata: ${h.status}`);if(!B.ok)throw new Error(`Failed to fetch landmark weights binary: ${B.status}`);if(!U.ok)throw new Error(`Failed to fetch palm weights metadata: ${U.status}`);if(!I.ok)throw new Error(`Failed to fetch palm weights binary: ${I.status}`);let[m,K,k,ce]=await Promise.all([h.json(),B.arrayBuffer(),U.json(),I.arrayBuffer()]),pe=Lt(m,K),Pe=Lt(k,ce),de=await ta(pe,{forceF32:g}),D=await wa(Pe),j=ba(D,{scoreThreshold:w,maxHands:e}),ne=null,L=null;function N(){return ne||(ne=new OffscreenCanvas(192,192)),ne}function $(){return L||(L=new OffscreenCanvas(256,256)),L}async function Me(C){if(C instanceof HTMLCanvasElement||C instanceof OffscreenCanvas){if(C.width===192&&C.height===192)return C;let Q=N();return Q.width=192,Q.height=192,Q.getContext("2d").drawImage(C,0,0,192,192),Q}if(typeof ImageBitmap<"u"&&C instanceof ImageBitmap){if(C.width===192&&C.height===192)return C;let Q=N();return Q.width=192,Q.height=192,Q.getContext("2d").drawImage(C,0,0,192,192),Q}let J=N();J.width=192,J.height=192;let we=J.getContext("2d");if(C instanceof ImageData){let Q=new OffscreenCanvas(C.width,C.height);Q.getContext("2d").putImageData(C,0,0),we.drawImage(Q,0,0,192,192)}else we.drawImage(C,0,0,192,192);return J}function ge(C,J,we,Q){let be=$();be.width=256,be.height=256;let re=be.getContext("2d"),Ye=Math.cos(-J.rotation),Je=Math.sin(-J.rotation);re.clearRect(0,0,256,256),re.save(),re.translate(128,128),re.scale(J.width*we/256,J.height*Q/256),re.rotate(-J.rotation),re.translate(-128,-128);let Ce=J.centerX*we,ke=J.centerY*Q;re.restore();let ye=256/(J.width*we),xe=256/(J.height*Q),We=Math.cos(J.rotation),Ue=Math.sin(J.rotation),Re=We*ye,He=Ue*ye,Xe=-Ue*xe,Te=We*xe,lt=-Ce*Re-ke*Xe+128,Qe=-Ce*He-ke*Te+128;if(re.setTransform(Re,He,Xe,Te,lt,Qe),C instanceof ImageData){let Oe=new OffscreenCanvas(C.width,C.height);Oe.getContext("2d").putImageData(C,0,0),re.drawImage(Oe,0,0)}else re.drawImage(C,0,0);return re.setTransform(1,0,0,1,0,0),be}function Ee(C){return C instanceof HTMLCanvasElement||C instanceof OffscreenCanvas?[C.width,C.height]:typeof ImageBitmap<"u"&&C instanceof ImageBitmap?[C.width,C.height]:C instanceof ImageData?[C.width,C.height]:C instanceof HTMLVideoElement?[C.videoWidth,C.videoHeight]:C instanceof HTMLImageElement?[C.naturalWidth,C.naturalHeight]:[256,256]}async function he(C){let J=await Me(C),we=await j.detect(J);if(we.length===0)return[];let[Q,be]=Ee(C),re=[];for(let Ye of we){let Je=ge(C,Ye,Q,be),Ce=await de.runFromCanvas(Je),ke=Ce.handflag[0];if(ke<v)continue;let ye=Ce.handedness[0]>.5,xe=[];for(let Ue=0;Ue<21;Ue++)xe.push({x:Ce.landmarks[Ue*3],y:Ce.landmarks[Ue*3+1],z:Ce.landmarks[Ue*3+2]});let We=ga(xe,Ye);re.push({score:ke,handedness:ye?"right":"left",landmarks:We,palmScore:0})}return re}function Be(){de.device.destroy(),D.device.destroy(),ne=null,L=null}return{detect:he,dispose:Be}}function Vn(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Zn=Vn(`
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
`);function Jn(r){let _=r.createShaderModule({code:Zn}),a=r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),v=r.createComputePipeline({layout:r.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:_,entryPoint:"main"}});function w(e,g,A,S,l,h,B){let U=new Uint32Array([l,h,B,0]),I=r.createBuffer({size:U.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});r.queue.writeBuffer(I,0,U);let m=new Float32Array(S),K=new Float32Array(8);K.set(m);let k=r.createBuffer({size:K.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});r.queue.writeBuffer(k,0,K);let ce=r.createBindGroup({layout:a,entries:[{binding:0,resource:g.createView()},{binding:1,resource:{buffer:A}},{binding:2,resource:{buffer:I}},{binding:3,resource:{buffer:k}}]}),pe=e.beginComputePass();pe.setPipeline(v),pe.setBindGroup(0,ce),pe.dispatchWorkgroups(Math.ceil(B/16),Math.ceil(B/16),1),pe.end()}return{crop:w}}export{wa as compilePalmModel,hn as computeCropTransform,Jn as createCropPipeline,jn as createFullHandpose,Xn as createHandpose,ba as createPalmDetector,Lt as loadWeightsFromBuffer,ga as projectLandmarksToOriginal};
