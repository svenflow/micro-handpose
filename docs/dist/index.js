function me(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ta(r){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+r;for(let v of _)for(;a.includes(`${v}:array<f32>`);)a=a.replace(`${v}:array<f32>`,`${v}:array<f16>`);return a}var la=me(`
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
`),_a=me(`
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
`),ma=me(`
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
`);function La(r,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Oa(r,_){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Ra(r,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Ia(r,_){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${_},1)`)}function Fa(r,_){return[8,8]}var za=me(`
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
`);function qa(r){return me(`
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
`)}var Ka=qa(!1),$a=qa(!0),Ya=me(`
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
`);function Va(r){return me(`
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
`)}var Za=Va("sigmoid"),ja=Va("div256"),Ja=me(`
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
`);function en(r,_){let v=Math.min(_,256),g=_>v,x=r%4===0?`var ic:u32=0u;
    while(ic<${r}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${r}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,D=`var skip_val:f32=0.0;
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
    }`,M=r===_?"":`if(c<${r}u){`,l=r===_?"":"}",w=g?`for(var c:u32=lid.x;c<${r}u;c+=${v}u){`:`let c=lid.x;
  ${M}`,P=g?"}":l,A=g?`for(var c:u32=lid.x;c<${_}u;c+=${v}u){`:"{let c=lid.x;";return me(`
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
  ${P}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${A}
    let pw_base=c*${r}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${x}
    // Skip connection (only for c < inCh)
    ${D}
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
`);function Ot(r,_){let a=new Map,v=r.dtype??"float32";for(let g=0;g<r.keys.length;g++){let e=r.keys[g],x=r.shapes[g],D=r.offsets[g],M=x.reduce((P,A)=>P*A,1),l,w;if(v==="float32")l=new Float32Array(_,D,M);else{let P=new DataView(_);l=new Float32Array(M);for(let A=0;A<M;A++)l[A]=On(P.getUint16(D+A*2,!0));w=_.slice(D,D+M*2)}a.set(e,{data:l,shape:x,rawF16:w})}return a}function On(r){let _=r>>15&1,a=r>>10&31,v=r&1023;if(a===0){if(v===0)return _?-0:0;let x=-14,D=v/1024;return(_?-1:1)*Math.pow(2,x)*D}if(a===31)return v===0?_?-1/0:1/0:NaN;let g=a-15,e=1+v/1024;return(_?-1:1)*Math.pow(2,g)*e}var Rn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],rn=Rn.map(([r,_,a,v,g])=>({type:"resmodule",inCh:r,outCh:_,h:a,w:a,stride:v,prefix:g})),In=2,Fn=5,zn=8,Nn=11;async function ta(r,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let v=a.features.has("shader-f16"),g=v?["shader-f16"]:[],e=await a.requestDevice({requiredFeatures:g,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(a.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(a.limits.maxComputeInvocationsPerWorkgroup,288)}}),x=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(v)try{let i=`enable f16;
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
}`,o=e.createShaderModule({code:i}),s=e.createShaderModule({code:c}),t=await o.getCompilationInfo(),H=await s.getCompilationInfo();if(t.messages.some(S=>S.type==="error")||H.messages.some(S=>S.type==="error"))x=!1;else{let S=new Float32Array(2400);S.fill(1);let L=new Uint16Array(2400);L.fill(10516);let C=new Uint16Array(96);C.fill(14336);let f=new Uint16Array(9216);f.fill(8478);let u=new Uint16Array(96);u.fill(12288);let z=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ie=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ue=e.createBuffer({size:S.byteLength,usage:z}),Pt=e.createBuffer({size:L.byteLength,usage:z}),Bt=e.createBuffer({size:C.byteLength,usage:z}),Ct=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),kt=e.createBuffer({size:f.byteLength,usage:z}),Ut=e.createBuffer({size:u.byteLength,usage:z}),Gt=e.createBuffer({size:384,usage:ie}),Qe=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ue,0,S),e.queue.writeBuffer(Pt,0,L),e.queue.writeBuffer(Bt,0,C),e.queue.writeBuffer(kt,0,f),e.queue.writeBuffer(Ut,0,u);let Xe="read-only-storage",Ht="storage",Tt=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ht}}]}),Wa=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ht}}]}),Mn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Tt]}),compute:{module:o,entryPoint:"main"}}),En=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Wa]}),compute:{module:s,entryPoint:"main"}}),Wn=e.createBindGroup({layout:Tt,entries:[{binding:0,resource:{buffer:ue}},{binding:1,resource:{buffer:Pt}},{binding:2,resource:{buffer:Bt}},{binding:3,resource:{buffer:Ct}}]}),Hn=e.createBindGroup({layout:Wa,entries:[{binding:0,resource:{buffer:Ct}},{binding:1,resource:{buffer:kt}},{binding:2,resource:{buffer:Ut}},{binding:3,resource:{buffer:Gt}}]}),Jt=e.createCommandEncoder(),Qt=Jt.beginComputePass();Qt.setPipeline(Mn),Qt.setBindGroup(0,Wn),Qt.dispatchWorkgroups(2),Qt.end();let ea=Jt.beginComputePass();ea.setPipeline(En),ea.setBindGroup(0,Hn),ea.dispatchWorkgroups(2),ea.end(),Jt.copyBufferToBuffer(Gt,0,Qe,0,384),e.queue.submit([Jt.finish()]),await e.queue.onSubmittedWorkDone(),await Qe.mapAsync(GPUMapMode.READ);let Lt=new Float32Array(Qe.getMappedRange()),Ha=1.5*.0104*96+.25,Tn=Lt[0]!==0&&Lt[47]!==0&&Lt[95]!==0,Ln=Math.abs(Lt[0]-Ha)<1;x=Tn&&Ln,Qe.unmap(),ue.destroy(),Pt.destroy(),Bt.destroy(),Ct.destroy(),kt.destroy(),Ut.destroy(),Gt.destroy(),Qe.destroy(),x||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Lt[0]}, expected ~${Ha.toFixed(2)}) \u2014 falling back to f32`)}}catch{x=!1}let M=r.values().next().value,l=x&&!!M?.rawF16&&!_?.forceF32;console.log(l?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${v}, f16 validated: ${x}, f16 data: ${!!M?.rawF16})`);function w(i){if(l&&i.rawF16){let c=new Uint16Array(i.rawF16);if(c.length%2!==0){let o=new Uint16Array(c.length+1);return o.set(c),o}return c}return i.data}function P(i){if(l&&i.rawF16){let c=i.rawF16.byteLength;return Math.ceil(c/4)*4}return i.data.byteLength}function A(i){return l?Ta(i):i}let I={r:"read-only-storage",s:"storage",u:"uniform"};function m(i){return e.createBindGroupLayout({entries:i.map((c,o)=>({binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:I[c]}}))})}function K(i){return e.createBindGroupLayout({entries:i.map((c,o)=>c==="t"?{binding:o,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:I[c]}})})}let k=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,de=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,pe=GPUBufferUsage.STORAGE,Be=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,le=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function W(i,c){return e.createBuffer({size:i,usage:c})}function j(i,c){return e.createBindGroup({layout:i,entries:c.map((o,s)=>({binding:s,resource:"size"in o?{buffer:o}:o}))})}function re(i,c){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),compute:{module:c,entryPoint:"main"}})}let O=e.createShaderModule({code:za}),N=e.createShaderModule({code:nn}),X=e.createShaderModule({code:A(Ja)}),We=e.createShaderModule({code:A(_a)}),ge=e.createShaderModule({code:A(la)}),ye=e.createShaderModule({code:A(ma)}),he=e.createShaderModule({code:A(fa)}),Ce=e.createShaderModule({code:A(Na)}),U=e.createShaderModule({code:Ka}),ee=e.createShaderModule({code:Ya}),we=e.createShaderModule({code:$a}),te=e.createShaderModule({code:A(Xa)}),be=e.createShaderModule({code:A(Za)}),ne=e.createShaderModule({code:A(ja)}),Ve=e.createShaderModule({code:A(Qa)}),et=new Map;function ke(i,c){let o=`${i}_${c}`,s=et.get(o);return s||(s=e.createShaderModule({code:A(en(i,c))}),et.set(o,s)),s}let Ue=m(["r","r","r","s","u"]),xe=m(["r","r","r","r","s","u"]),ve=m(["r","s","u"]),He=m(["r","r","r","s","u"]),Ge=m(["r","s","u"]),Re=m(["r","r","s","u"]),Te=m(["r","r","s","u"]),Le=m(["r","r","r","s","u"]),Ae=m(["r","r","r","s","u"]),tt=K(["t","s","u"]),at=m(["r","r","r","r","r","r","r","s"]),Ie=m(["r","r","r","r","r","s","u"]),At=e.createPipelineLayout({bindGroupLayouts:[Ue]}),St=e.createPipelineLayout({bindGroupLayouts:[xe]}),nt=i=>e.createComputePipeline({layout:At,compute:{module:i,entryPoint:"main"}}),it=i=>e.createComputePipeline({layout:St,compute:{module:i,entryPoint:"main"}}),Rt=nt(We),It=nt(ge),Ft=it(ye),ft=it(he),ht=new Map,Dt=new Map,wt=new Map,Mt=new Map;ht.set("8,8",Rt),Dt.set("8,8",It),wt.set("8,8",Ft),Mt.set("8,8",ft);function rt(i,c,o,s,t){let H=`${c},${o}`,S=i.get(H);return S||(S=t(e.createShaderModule({code:A(s(c,o))})),i.set(H,S)),S}let bt=(i,c)=>rt(ht,i,c,La,nt),Et=(i,c)=>rt(Dt,i,c,Oa,nt),zt=(i,c)=>rt(wt,i,c,Ra,it),Wt=(i,c)=>rt(Mt,i,c,Ia,it),Se=rn.map(i=>{let c=i.stride===2?i.h/2:i.h,o=i.stride===2?i.w/2:i.w,[s,t]=Fa(i.inCh,c),H=i.h>=64,S=c>=16&&i.inCh>=288&&i.outCh>=288&&i.outCh%2===0;return{dwPipeline:H?Et(s,t):bt(s,t),pwPipeline:S?Wt(s,t):zt(s,t),dwDispatchX:Math.ceil(o/s),dwDispatchY:Math.ceil(c/t),dwDispatchZ:i.inCh,pwDispatchX:Math.ceil(o/s),pwDispatchY:Math.ceil(c/t),pwDispatchZ:S?i.outCh/2:i.outCh}}),gt=re(ve,O),Ze=re(He,Ce);re(Ge,U),re(Re,ee);let Fe=re(Te,we),Nt=re(Le,te);re(Ae,be),re(Ae,ne);let De=re(tt,N),ga=re(at,X),st=re(Ie,Ve),ze=1*288*128*128*4,ot=W(3*256*256*4,k),Oe=W(3*257*257*4,pe),ut=W(12,le);e.queue.writeBuffer(ut,0,new Uint32Array([3,256,257]));let V=W(ze,de),_e=W(ze,Be),Me=W(ze,pe),pt=W(3072*64*4,k),ct=W(3072*32*4,k),dt=W(1536*16*4,k),$=W(6144*64*4,pe),ce=W(260,Be),J=W(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);W(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let se=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ee=W(8,le);e.queue.writeBuffer(Ee,0,new Uint32Array([256,257]));let lt=r.get("backbone1.1.weight"),qt=r.get("backbone1.1.bias");if(!lt||!qt)throw new Error("Missing input conv weights");let Y=w(lt),Kt=w(qt),d=W(Y.byteLength,k),n=W(Kt.byteLength,k),y=W(28,le);e.queue.writeBuffer(d,0,Y),e.queue.writeBuffer(n,0,Kt),e.queue.writeBuffer(y,0,new Uint32Array([1,3,24,257,257,128,128]));let E=r.get("backbone6.1.weight"),b=r.get("backbone6.1.bias");if(!E||!b)throw new Error("Missing backbone6.1 conv1x1 weights");let p=w(E),R=w(b),ae=W(p.byteLength,k),h=W(R.byteLength,k),T=W(20,le);e.queue.writeBuffer(ae,0,p),e.queue.writeBuffer(h,0,R),e.queue.writeBuffer(T,0,new Uint32Array([1,96,48,32,32]));let B=r.get("handflag.weight"),q=r.get("handflag.bias");if(!B||!q)throw new Error("Missing handflag weights");let F=w(B),fe=w(q),oe=W(F.byteLength,k),G=W(fe.byteLength,k),Q=W(12,le);e.queue.writeBuffer(oe,0,F),e.queue.writeBuffer(G,0,fe),e.queue.writeBuffer(Q,0,new Uint32Array([1,288,1]));let Z=r.get("handedness.weight"),Pe=r.get("handedness.bias");if(!Z||!Pe)throw new Error("Missing handedness weights");let yt=w(Z),ya=w(Pe),aa=W(yt.byteLength,k),na=W(ya.byteLength,k),xa=W(12,le);e.queue.writeBuffer(aa,0,yt),e.queue.writeBuffer(na,0,ya),e.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let va=r.get("reg_3d.weight"),Pa=r.get("reg_3d.bias");if(!va||!Pa)throw new Error("Missing reg_3d weights");let Ba=w(va),Ca=w(Pa),ia=W(Ba.byteLength,k),ra=W(Ca.byteLength,k),ka=W(12,le);e.queue.writeBuffer(ia,0,Ba),e.queue.writeBuffer(ra,0,Ca),e.queue.writeBuffer(ka,0,new Uint32Array([1,288,63]));let _t=rn.map(i=>{let{inCh:c,outCh:o,h:s,w:t,stride:H,prefix:S}=i,L=H===2?s/2:s,C=H===2?t/2:t,f=H===2?1:2,u=r.get(`${S}convs.0.weight`),z=r.get(`${S}convs.0.bias`),ie=r.get(`${S}convs.1.weight`),ue=r.get(`${S}convs.1.bias`);if(!u||!z||!ie||!ue)throw new Error(`Missing weights for ${S}`);let Pt=w(u),Bt=w(z),Ct=w(ie),kt=w(ue),Ut=W(Pt.byteLength,k),Gt=W(Bt.byteLength,k),Qe=W(Ct.byteLength,k),Xe=W(kt.byteLength,k),Ht=W(32,le),Tt=W(36,le);return e.queue.writeBuffer(Ut,0,Pt),e.queue.writeBuffer(Gt,0,Bt),e.queue.writeBuffer(Qe,0,Ct),e.queue.writeBuffer(Xe,0,kt),e.queue.writeBuffer(Ht,0,new Uint32Array([1,c,s,t,L,C,H,f])),e.queue.writeBuffer(Tt,0,new Uint32Array([1,c,o,L,C,Math.max(0,o-c),H,s,t])),{dwWeight:Ut,dwBias:Gt,pwWeight:Qe,pwBias:Xe,dwUniform:Ht,pwUniform:Tt,spec:i,outH:L,outW:C}});function xt(i){let c=W(i.length*4,le);return e.queue.writeBuffer(c,0,new Uint32Array(i)),c}let bn=xt([1,96,8,8,16,16]),gn=xt([1,96,16,16,32,32]),yn=xt([1,48,32,32,64,64]);xt([1536*16]),xt([3072*32]),xt([3072*64]);let Ua=j(ve,[ot,Oe,ut]),Ga=j(He,[Oe,d,n,V,y]),Ne=[],qe=[],Ke=[],$e=[];for(let i of _t)Ne.push(j(Ue,[V,i.dwWeight,i.dwBias,Me,i.dwUniform])),qe.push(j(xe,[Me,V,i.pwWeight,i.pwBias,_e,i.pwUniform])),Ke.push(j(Ue,[_e,i.dwWeight,i.dwBias,Me,i.dwUniform])),$e.push(j(xe,[Me,_e,i.pwWeight,i.pwBias,V,i.pwUniform]));let xn=j(Te,[V,dt,_e,bn]),vn=j(Te,[V,ct,_e,gn]),Pn=j(Le,[V,ae,h,$,T]),Bn=j(Te,[$,pt,_e,yn]);j(Ae,[V,oe,G,ce,Q]),j(Ae,[V,aa,na,ce,xa]),j(Ae,[V,ia,ra,ce,ka]);let je=j(tt,[se.createView(),Oe,Ee]),Cn=j(at,[V,oe,G,aa,na,ia,ra,ce]),sa=24,Aa=[],Sa=[];for(let i=sa;i<_t.length;i++){let c=_t[i];Aa.push(j(Ie,[V,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,_e,c.dwUniform])),Sa.push(j(Ie,[_e,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,V,c.dwUniform]))}let oa=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});oa.globalCompositeOperation="copy";let Da=new OffscreenCanvas(9,8),$t=Da.getContext("webgpu"),Yt=null,ua=null;if($t){$t.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let i=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),c=e.createShaderModule({code:tn}),o=e.createShaderModule({code:an});Yt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[i]}),vertex:{module:c,entryPoint:"vs"},fragment:{module:o,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),ua=e.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:ce}}]})}let Xt=new Float32Array(1),Vt=new Float32Array(1),Zt=new Float32Array(63);function Ye(i,c){let o=!0,s=0,t=i.beginComputePass();for(t.setPipeline(Ze),t.setBindGroup(0,Ga),t.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);s<=In;s++){let C=o?Ne[s]:Ke[s],f=o?qe[s]:$e[s],u=Se[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,C),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let H=o?V:_e;for(i.copyBufferToBuffer(H,0,pt,0,3072*64*4),t=i.beginComputePass();s<=Fn;s++){let C=o?Ne[s]:Ke[s],f=o?qe[s]:$e[s],u=Se[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,C),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let S=o?V:_e;for(i.copyBufferToBuffer(S,0,ct,0,3072*32*4),t=i.beginComputePass();s<=zn;s++){let C=o?Ne[s]:Ke[s],f=o?qe[s]:$e[s],u=Se[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,C),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.end();let L=o?V:_e;for(i.copyBufferToBuffer(L,0,dt,0,1536*16*4),t=i.beginComputePass();s<=Nn;s++){let C=o?Ne[s]:Ke[s],f=o?qe[s]:$e[s],u=Se[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,C),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}t.setPipeline(Fe),t.setBindGroup(0,xn),t.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),t.end(),o=!1,t=i.beginComputePass();{let C=o?Ne[s]:Ke[s],f=o?qe[s]:$e[s],u=Se[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,C),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}t.setPipeline(Fe),t.setBindGroup(0,vn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),t.end(),o=!1,t=i.beginComputePass();{let C=o?Ne[s]:Ke[s],f=o?qe[s]:$e[s],u=Se[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,C),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o,s++}for(t.setPipeline(Nt),t.setBindGroup(0,Pn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),t.setPipeline(Fe),t.setBindGroup(0,Bn),t.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),t.end(),o=!1,t=i.beginComputePass();s<sa;s++){let C=o?Ne[s]:Ke[s],f=o?qe[s]:$e[s],u=Se[s];t.setPipeline(u.dwPipeline),t.setBindGroup(0,C),t.dispatchWorkgroups(u.dwDispatchX,u.dwDispatchY,u.dwDispatchZ),t.setPipeline(u.pwPipeline),t.setBindGroup(0,f),t.dispatchWorkgroups(u.pwDispatchX,u.pwDispatchY,u.pwDispatchZ),o=!o}for(;s<_t.length;s++){let C=s-sa,f=o?Aa[C]:Sa[C],u=_t[s];t.setPipeline(st),t.setBindGroup(0,f),t.dispatchWorkgroups(u.outW,u.outH,1),o=!o}t.setPipeline(ga),t.setBindGroup(0,Cn),t.dispatchWorkgroups(1),t.end(),c&&i.copyBufferToBuffer(ce,0,c,0,260)}async function jt(i){e.queue.writeBuffer(ot,0,i);let c=e.createCommandEncoder();{let t=c.beginComputePass();t.setPipeline(gt),t.setBindGroup(0,Ua),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}Ye(c,J),e.queue.submit([c.finish()]);let o=J.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(J.getMappedRange());return Xt[0]=s[0],Vt[0]=s[1],Zt.set(s.subarray(2,65)),J.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function pa(i){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let c=e.createCommandEncoder();{let t=c.beginComputePass();t.setPipeline(De),t.setBindGroup(0,je),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Ye(c,J),e.queue.submit([c.finish()]);let o=J.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await o;let s=new Float32Array(J.getMappedRange());return Xt[0]=s[0],Vt[0]=s[1],Zt.set(s.subarray(2,65)),J.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function Ma(i){if(!Yt||!ua||!$t)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let c=e.createCommandEncoder();{let C=c.beginComputePass();C.setPipeline(De),C.setBindGroup(0,je),C.dispatchWorkgroups(16,16,1),C.end()}Ye(c,null);let o=$t.getCurrentTexture(),s=c.beginRenderPass({colorAttachments:[{view:o.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});s.setPipeline(Yt),s.setBindGroup(0,ua),s.draw(3),s.end(),e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),oa.drawImage(Da,0,0);let H=oa.getImageData(0,0,9,8).data,S=new Float32Array(65),L=new DataView(new ArrayBuffer(4));for(let C=0;C<65;C++){let f=C*4;L.setUint8(0,H[f]),L.setUint8(1,H[f+1]),L.setUint8(2,H[f+2]),L.setUint8(3,H[f+3]),S[C]=L.getFloat32(0)}return{handflag:new Float32Array([S[0]]),handedness:new Float32Array([S[1]]),landmarks:new Float32Array(S.subarray(2,65))}}let kn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),ca=0,Un=[J,kn],vt=null,Je=null;async function da(i){let c=Un[ca];ca=1-ca,e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let o=e.createCommandEncoder();{let t=o.beginComputePass();t.setPipeline(De),t.setBindGroup(0,je),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Ye(o,c),e.queue.submit([o.finish()]);let s=null;if(vt!==null&&Je!==null){await vt;let t=new Float32Array(Je.getMappedRange());s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))},Je.unmap()}return Je=c,vt=c.mapAsync(GPUMapMode.READ),s}async function Ea(){if(!vt||!Je)return null;await vt;let i=new Float32Array(Je.getMappedRange()),c={handflag:new Float32Array([i[0]]),handedness:new Float32Array([i[1]]),landmarks:new Float32Array(i.subarray(2,65))};return Je.unmap(),vt=null,Je=null,c}async function Gn(i=50){let c=new Float32Array(196608);for(let t=0;t<5;t++)await jt(c);let o=[];for(let t=0;t<i;t++){let H=performance.now();await jt(c),o.push(performance.now()-H)}let s=o.reduce((t,H)=>t+H,0)/o.length;return{avgMs:s,fps:1e3/s}}async function An(i=50){let c=new Float32Array(196608);for(let S=0;S<5;S++)await jt(c);let o=[];for(let S=0;S<i;S++){let L=e.createCommandEncoder();{let f=L.beginComputePass();f.setPipeline(gt),f.setBindGroup(0,Ua),f.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),f.end()}Ye(L,J);let C=performance.now();e.queue.submit([L.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-C)}o.sort((S,L)=>S-L);let s=o.reduce((S,L)=>S+L,0)/o.length,t=o[Math.floor(o.length/2)],H=o[0];return{avgMs:s,fps:1e3/s,medianMs:t,minMs:H}}function ei(i){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let c=e.createCommandEncoder();{let o=c.beginComputePass();o.setPipeline(De),o.setBindGroup(0,je),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),o.end()}Ye(c,J),e.queue.submit([c.finish()])}async function Sn(i,c=50){function o(f){let u=[...f].sort((z,ie)=>z-ie);return{median:u[Math.floor(u.length/2)],min:u[0]}}for(let f=0;f<10;f++)await pa(i);let s=[];for(let f=0;f<c;f++){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let u=e.createCommandEncoder();{let ie=u.beginComputePass();ie.setPipeline(De),ie.setBindGroup(0,je),ie.dispatchWorkgroups(16,16,1),ie.end()}Ye(u,J);let z=performance.now();e.queue.submit([u.finish()]),await e.queue.onSubmittedWorkDone(),s.push(performance.now()-z)}let t=[];for(let f=0;f<c;f++){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let u=e.createCommandEncoder();{let ue=u.beginComputePass();ue.setPipeline(De),ue.setBindGroup(0,je),ue.dispatchWorkgroups(16,16,1),ue.end()}Ye(u,J),e.queue.submit([u.finish()]);let z=J.mapAsync(GPUMapMode.READ),ie=performance.now();await e.queue.onSubmittedWorkDone(),await z,J.getMappedRange(),J.unmap(),t.push(performance.now()-ie)}let H=[];for(let f=0;f<c;f++){e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);let u=e.createCommandEncoder();{let ie=u.beginComputePass();ie.setPipeline(De),ie.setBindGroup(0,je),ie.dispatchWorkgroups(16,16,1),ie.end()}Ye(u,J),e.queue.submit([u.finish()]);let z=performance.now();await J.mapAsync(GPUMapMode.READ),J.getMappedRange(),J.unmap(),H.push(performance.now()-z)}let S=[];for(let f=0;f<c;f++){let u=performance.now();await pa(i),S.push(performance.now()-u)}await da(i);let L=[];for(let f=0;f<c;f++){let u=performance.now();await da(i),L.push(performance.now()-u)}await Ea();let C=null;if(Yt){let f=[];for(let u=0;u<c;u++){let z=performance.now();await Ma(i),f.push(performance.now()-z)}C=o(f)}return{gpuOnly:o(s),mapAsyncOnly:o(t),mapAsyncNoWait:o(H),total:o(S),pipelined:o(L),renderReadback:C}}async function Dn(i){let c=[];async function o(t,H,S){let L=e.createBuffer({size:H,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),C=e.createCommandEncoder();C.copyBufferToBuffer(t,0,L,0,H),e.queue.submit([C.finish()]),await e.queue.onSubmittedWorkDone(),await L.mapAsync(GPUMapMode.READ);let f=new Float32Array(L.getMappedRange()),u=1/0,z=-1/0,ie=0;for(let ue=0;ue<f.length;ue++)f[ue]<u&&(u=f[ue]),f[ue]>z&&(z=f[ue]),f[ue]!==0&&ie++;L.unmap(),L.destroy(),c.push({layer:S,stats:{min:u,max:z,nonZero:ie,total:f.length}})}e.queue.copyExternalImageToTexture({source:i},{texture:se},[256,256]);{let t=e.createCommandEncoder(),H=t.beginComputePass();H.setPipeline(De),H.setBindGroup(0,je),H.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),H.end(),e.queue.submit([t.finish()])}await o(Oe,Math.min(Oe.size,3*257*257*4),"canvas\u2192bufInput");{let t=e.createCommandEncoder(),H=t.beginComputePass();H.setPipeline(Ze),H.setBindGroup(0,Ga),H.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),H.end(),e.queue.submit([t.finish()])}await o(V,Math.min(V.size,3072*128*4),"inputConv\u2192bufA");let s=!0;for(let t=0;t<Math.min(_t.length,6);t++){let H=s?Ne[t]:Ke[t],S=s?qe[t]:$e[t],L=Se[t],C=_t[t];{let u=e.createCommandEncoder(),z=u.beginComputePass();z.setPipeline(L.dwPipeline),z.setBindGroup(0,H),z.dispatchWorkgroups(L.dwDispatchX,L.dwDispatchY,L.dwDispatchZ),z.end(),e.queue.submit([u.finish()])}await o(Me,Math.min(Me.size,C.spec.inCh*C.outH*C.outW*4),`layer${t}.DW\u2192bufDW (${C.spec.prefix})`);{let u=e.createCommandEncoder(),z=u.beginComputePass();z.setPipeline(L.pwPipeline),z.setBindGroup(0,S),z.dispatchWorkgroups(L.pwDispatchX,L.pwDispatchY,L.pwDispatchZ),z.end(),e.queue.submit([u.finish()])}let f=s?_e:V;await o(f,Math.min(f.size,C.spec.outCh*C.outH*C.outW*4),`layer${t}.PW\u2192buf${s?"B":"A"} (${C.spec.prefix})`),s=!s}return c}return{device:e,run:jt,runFromCanvas:pa,runFromCanvasViaRender:Ma,runFromCanvasPipelined:da,flushPipelined:Ea,benchmark:Gn,benchmarkGPU:An,benchmarkDiagnostic:Sn,debugLayerOutputs:Dn}}function mt(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var sn=mt(`
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
`);async function ha(r,_){let a;if(_)a=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");a=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8)}})}let v={r:"read-only-storage",s:"storage",u:"uniform"};function g(d){return a.createBindGroupLayout({entries:d.map((n,y)=>n==="t"?{binding:y,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:y,visibility:GPUShaderStage.COMPUTE,buffer:{type:v[n]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,x=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,D=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,M=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(d,n){return a.createBuffer({size:Math.max(d,4),usage:n})}function w(d,n,y){a.queue.writeBuffer(d,n,y)}function P(d){let n=l(d.data.byteLength,e);return w(n,0,d.data),n}let A=Array.from(r.keys());function I(d){let n=r.get(d);if(!n)throw new Error(`Weight not found: ${d}`);return n}function m(...d){let n=A.find(y=>d.every(E=>y.includes(E)));if(!n)throw new Error(`Weight not found for: ${d.join(", ")}`);return I(n)}function K(d){let[,n,y,E]=d.shape,b=new Float32Array(E*25);for(let p=0;p<E;p++)for(let R=0;R<n;R++)for(let ae=0;ae<y;ae++)b[p*25+R*5+ae]=d.data[R*y*E+ae*E+p];return b}function k(d){let[n,,,y]=d.shape,E=new Float32Array(n*y);for(let b=0;b<n;b++)for(let p=0;p<y;p++)E[b*y+p]=d.data[b*y+p];return E}let de=a.createShaderModule({code:sn}),pe=a.createShaderModule({code:on}),Be=a.createShaderModule({code:un}),le=a.createShaderModule({code:pn}),W=a.createShaderModule({code:dn}),j=a.createShaderModule({code:cn}),re=a.createShaderModule({code:ln}),O=g(["r","r","r","r","s","u"]),N=g(["r","r","r","s","u"]),X=g(["r","r","r","r","r","s","u"]),We=g(["r","r","r","s","u"]),ge=g(["r","r","r","r","s","u"]),ye=g(["r","r","s","u"]),he=g(["t","s","u"]);function Ce(d,n){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:n,entryPoint:"main"}})}let U=Ce(O,de),ee=Ce(N,pe),we=Ce(X,Be),te=Ce(We,le),be=Ce(ge,W),ne=Ce(ye,j),Ve=Ce(he,re),et=m("conv2d/Conv2D"),ke=m("batch_normalization/","conv2d/Conv2D"),Ue=m("p_re_lu/"),xe=P(et),ve=P(ke),He=P(Ue),Re=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(d=>{let n=m(d.dwKey),y=m(d.pwKey),E=m(d.bnKey),b=m(d.preluKey),p=K(n),R=l(p.byteLength,e);w(R,0,p);let ae=new Float32Array(d.inCh),h=l(ae.byteLength,e);w(h,0,ae);let T=k(y),B=l(T.byteLength,e);w(B,0,T);let q=P(E),F=P(b);return{dwWeightBuf:R,dwBiasBuf:h,pwWeightBuf:B,pwBiasBuf:q,alphaBuf:F,inCh:d.inCh,outCh:d.outCh,stride:d.stride,inH:d.inH}}),Te=k(m("conv2d_20/Conv2D")),Le=l(Te.byteLength,e);w(Le,0,Te);let Ae=P(m("batch_normalization_20/")),tt=P(m("p_re_lu_20/")),at={dwWeightBuf:(()=>{let d=K(m("depthwise_conv2d_19/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=k(m("conv2d_21/")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:P(m("batch_normalization_21/")),alphaBuf:P(m("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},Ie={dwWeightBuf:(()=>{let d=K(m("depthwise_conv2d_20/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=k(m("conv2d_22/Conv2D1")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:P(m("batch_normalization_22/")),alphaBuf:P(m("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},At=k(m("conv2d_23/Conv2D")),St=l(At.byteLength,e);w(St,0,At);let nt=P(m("batch_normalization_23/")),it=P(m("p_re_lu_23/")),Rt={dwWeightBuf:(()=>{let d=K(m("depthwise_conv2d_21/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=k(m("conv2d_24/")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:P(m("batch_normalization_24/")),alphaBuf:P(m("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},It={dwWeightBuf:(()=>{let d=K(m("depthwise_conv2d_22/")),n=l(d.byteLength,e);return w(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return w(n,0,d),n})(),pwWeightBuf:(()=>{let d=k(m("conv2d_25/Conv2D1")),n=l(d.byteLength,e);return w(n,0,d),n})(),pwBiasBuf:P(m("batch_normalization_25/")),alphaBuf:P(m("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},Ft=k(m("classifier_palm_16_NO_PRUNING/Conv2D")),ft=l(Ft.byteLength,e);w(ft,0,Ft);let ht=P(m("classifier_palm_16_NO_PRUNING/BiasAdd")),Dt=k(m("regressor_palm_16_NO_PRUNING/Conv2D")),wt=l(Dt.byteLength,e);w(wt,0,Dt);let Mt=P(m("regressor_palm_16_NO_PRUNING/BiasAdd")),rt=k(m("classifier_palm_8_NO_PRUNING/Conv2D")),bt=l(rt.byteLength,e);w(bt,0,rt);let Et=P(m("classifier_palm_8_NO_PRUNING/BiasAdd")),zt=k(m("regressor_palm_8_NO_PRUNING/Conv2D")),Wt=l(zt.byteLength,e);w(Wt,0,zt);let Se=P(m("regressor_palm_8_NO_PRUNING/BiasAdd")),gt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Ze=l(36864*3*4,e),Fe=l(gt,x),Nt=l(gt,x),De=l(gt,x),ga=l(576*256*4,x),st=l(144*256*4,x|GPUBufferUsage.COPY_DST),ze=l(576*128*4,x|GPUBufferUsage.COPY_DST),ot=l(864*4,D),Oe=l(15552*4,D),ut=l(576*2*4,D),V=l(576*36*4,D),_e=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Me=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pt=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),dt=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function $(d,n){return Math.ceil(d/n)}function ce(d){let n=l(d.byteLength,M);return w(n,0,d),n}let J=ce(new Uint32Array([1,3,32,192,192,96,96]));function se(d,n,y,E,b){let p=n.stride===2?n.inH/2:n.inH,R=p,ae=n.stride===2?1:2,h=ce(new Uint32Array([1,n.inCh,n.inH,n.inH,p,R,n.stride,ae])),T=a.createBindGroup({layout:N,entries:[{binding:0,resource:{buffer:y}},{binding:1,resource:{buffer:n.dwWeightBuf}},{binding:2,resource:{buffer:n.dwBiasBuf}},{binding:3,resource:{buffer:De}},{binding:4,resource:{buffer:h}}]}),B=d.beginComputePass();B.setPipeline(ee),B.setBindGroup(0,T),B.dispatchWorkgroups($(R,8),$(p,8),n.inCh),B.end();let q=n.inCh,F=ce(new Uint32Array([1,n.inCh,n.outCh,p,R,q,n.stride,n.inH,n.inH])),fe=a.createBindGroup({layout:X,entries:[{binding:0,resource:{buffer:De}},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:n.pwWeightBuf}},{binding:3,resource:{buffer:n.pwBiasBuf}},{binding:4,resource:{buffer:n.alphaBuf}},{binding:5,resource:{buffer:E}},{binding:6,resource:{buffer:F}}]}),oe=d.beginComputePass();oe.setPipeline(we),oe.setBindGroup(0,fe),oe.dispatchWorkgroups($(R,8),$(p,8),n.outCh),oe.end()}function Ee(d,n,y,E,b,p,R,ae,h){let T=ce(new Uint32Array([1,p,R,ae,h])),B=a.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:y}},{binding:2,resource:{buffer:E}},{binding:3,resource:{buffer:b}},{binding:4,resource:{buffer:T}}]}),q=d.beginComputePass();q.setPipeline(te),q.setBindGroup(0,B),q.dispatchWorkgroups($(h,8),$(ae,8),R),q.end()}function lt(d,n,y,E,b,p,R,ae,h,T){let B=ce(new Uint32Array([1,R,ae,h,T])),q=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:y}},{binding:2,resource:{buffer:E}},{binding:3,resource:{buffer:b}},{binding:4,resource:{buffer:p}},{binding:5,resource:{buffer:B}}]}),F=d.beginComputePass();F.setPipeline(be),F.setBindGroup(0,q),F.dispatchWorkgroups($(T,8),$(h,8),ae),F.end()}async function qt(d){a.queue.copyExternalImageToTexture({source:d},{texture:dt},[192,192]);let n=ce(new Uint32Array([192,192,192])),y=a.createBindGroup({layout:he,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:Ze}},{binding:2,resource:{buffer:n}}]}),E=a.createCommandEncoder();{let G=E.beginComputePass();G.setPipeline(Ve),G.setBindGroup(0,y),G.dispatchWorkgroups($(192,16),$(192,16),1),G.end()}{let G=a.createBindGroup({layout:O,entries:[{binding:0,resource:{buffer:Ze}},{binding:1,resource:{buffer:xe}},{binding:2,resource:{buffer:ve}},{binding:3,resource:{buffer:He}},{binding:4,resource:{buffer:Fe}},{binding:5,resource:{buffer:J}}]}),Q=E.beginComputePass();Q.setPipeline(U),Q.setBindGroup(0,G),Q.dispatchWorkgroups($(96,8),$(96,8),32),Q.end()}let b=Fe,p=Nt;for(let G=0;G<Re.length;G++){let Q=Re[G];se(E,Q,b,p,b);let Z=b;b=p,p=Z,G===10&&E.copyBufferToBuffer(b,0,ze,0,576*128*4),G===14&&E.copyBufferToBuffer(b,0,st,0,144*256*4)}lt(E,b,Le,Ae,tt,p,256,256,6,6);{let G=b;b=p,p=G}{let G=ce(new Uint32Array([1,256,6,6,12,12])),Q=a.createBindGroup({layout:ye,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:p}},{binding:3,resource:{buffer:G}}]}),Z=E.beginComputePass();Z.setPipeline(ne),Z.setBindGroup(0,Q),Z.dispatchWorkgroups($(12,8),$(12,8),256),Z.end()}{let G=b;b=p,p=G}se(E,at,b,p,b);{let G=b;b=p,p=G}se(E,Ie,b,p,b);{let G=b;b=p,p=G}Ee(E,b,ft,ht,ot,256,6,12,12),Ee(E,b,wt,Mt,Oe,256,108,12,12),lt(E,b,St,nt,it,p,256,128,12,12);{let G=b;b=p,p=G}{let G=ce(new Uint32Array([1,128,12,12,24,24])),Q=a.createBindGroup({layout:ye,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:ze}},{binding:2,resource:{buffer:p}},{binding:3,resource:{buffer:G}}]}),Z=E.beginComputePass();Z.setPipeline(ne),Z.setBindGroup(0,Q),Z.dispatchWorkgroups($(24,8),$(24,8),128),Z.end()}{let G=b;b=p,p=G}se(E,Rt,b,p,b);{let G=b;b=p,p=G}se(E,It,b,p,b);{let G=b;b=p,p=G}Ee(E,b,bt,Et,ut,128,2,24,24),Ee(E,b,Wt,Se,V,128,36,24,24),a.queue.submit([E.finish()]);let R=a.createCommandEncoder();R.copyBufferToBuffer(ot,0,_e,0,864*4),R.copyBufferToBuffer(Oe,0,Me,0,15552*4),R.copyBufferToBuffer(ut,0,pt,0,576*2*4),R.copyBufferToBuffer(V,0,ct,0,576*36*4),a.queue.submit([R.finish()]),await Promise.all([_e.mapAsync(GPUMapMode.READ),Me.mapAsync(GPUMapMode.READ),pt.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ)]);let ae=new Float32Array(_e.getMappedRange()).slice(),h=new Float32Array(Me.getMappedRange()).slice(),T=new Float32Array(pt.getMappedRange()).slice(),B=new Float32Array(ct.getMappedRange()).slice();_e.unmap(),Me.unmap(),pt.unmap(),ct.unmap();let q=2016,F=new Float32Array(q),fe=new Float32Array(q*18),oe=0;for(let G=0;G<12;G++)for(let Q=0;Q<12;Q++)for(let Z=0;Z<6;Z++){F[oe]=ae[Z*144+G*12+Q];for(let Pe=0;Pe<18;Pe++){let yt=Z*18+Pe;fe[oe*18+Pe]=h[yt*144+G*12+Q]}oe++}for(let G=0;G<24;G++)for(let Q=0;Q<24;Q++)for(let Z=0;Z<2;Z++){F[oe]=T[Z*576+G*24+Q];for(let Pe=0;Pe<18;Pe++){let yt=Z*18+Pe;fe[oe*18+Pe]=B[yt*576+G*24+Q]}oe++}return{scores:F,regressors:fe}}async function Y(d,n){let y=a.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),E=a.createCommandEncoder();E.copyBufferToBuffer(d,0,y,0,n*4),a.queue.submit([E.finish()]),await y.mapAsync(GPUMapMode.READ);let b=new Float32Array(y.getMappedRange()).slice();return y.unmap(),y.destroy(),b}async function Kt(d){a.queue.copyExternalImageToTexture({source:d},{texture:dt},[192,192]);function n(B,q=1e3){let F=B.slice(0,q);return{min:Math.min(...F),max:Math.max(...F),mean:F.reduce((fe,oe)=>fe+oe,0)/F.length,nonZero:F.filter(fe=>fe!==0).length,sample:Array.from(F.slice(0,10))}}let y={},E=ce(new Uint32Array([192,192,192])),b=a.createBindGroup({layout:he,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:Ze}},{binding:2,resource:{buffer:E}}]}),p=a.createCommandEncoder(),R=p.beginComputePass();R.setPipeline(Ve),R.setBindGroup(0,b),R.dispatchWorkgroups($(192,16),$(192,16),1),R.end(),a.queue.submit([p.finish()]),y.input=n(await Y(Ze,36864*3)),p=a.createCommandEncoder();let ae=a.createBindGroup({layout:O,entries:[{binding:0,resource:{buffer:Ze}},{binding:1,resource:{buffer:xe}},{binding:2,resource:{buffer:ve}},{binding:3,resource:{buffer:He}},{binding:4,resource:{buffer:Fe}},{binding:5,resource:{buffer:J}}]});R=p.beginComputePass(),R.setPipeline(U),R.setBindGroup(0,ae),R.dispatchWorkgroups($(96,8),$(96,8),32),R.end(),a.queue.submit([p.finish()]),y.initConv=n(await Y(Fe,9216*32));let h=Fe,T=Nt;for(let B=0;B<Re.length;B++){let q=Re[B];p=a.createCommandEncoder(),se(p,q,h,T,h),a.queue.submit([p.finish()]);let F=h;if(h=T,T=F,B===0||B===3||B===7||B===11||B===14||B===15||B===18){let fe=q.stride===2?q.inH/2:q.inH,oe=fe*fe*q.outCh;y[`block${B}`]=n(await Y(h,oe))}B===10&&(p=a.createCommandEncoder(),p.copyBufferToBuffer(h,0,ze,0,576*128*4),a.queue.submit([p.finish()])),B===14&&(p=a.createCommandEncoder(),p.copyBufferToBuffer(h,0,st,0,144*256*4),a.queue.submit([p.finish()]))}p=a.createCommandEncoder(),lt(p,h,Le,Ae,tt,T,256,256,6,6),a.queue.submit([p.finish()]);{let B=h;h=T,T=B}y.fpn6to12Conv=n(await Y(h,36*256)),p=a.createCommandEncoder();{let B=ce(new Uint32Array([1,256,6,6,12,12])),q=a.createBindGroup({layout:ye,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:T}},{binding:3,resource:{buffer:B}}]}),F=p.beginComputePass();F.setPipeline(ne),F.setBindGroup(0,q),F.dispatchWorkgroups($(12,8),$(12,8),256),F.end()}a.queue.submit([p.finish()]);{let B=h;h=T,T=B}y.fpnUpsample6to12=n(await Y(h,144*256)),y.backbone12Skip=n(await Y(st,144*256)),p=a.createCommandEncoder(),se(p,at,h,T,h),a.queue.submit([p.finish()]);{let B=h;h=T,T=B}y.fpn12Block1=n(await Y(h,144*256)),p=a.createCommandEncoder(),se(p,Ie,h,T,h),a.queue.submit([p.finish()]);{let B=h;h=T,T=B}y.fpn12Block2=n(await Y(h,144*256)),p=a.createCommandEncoder(),Ee(p,h,ft,ht,ot,256,6,12,12),a.queue.submit([p.finish()]),y.cls16=n(await Y(ot,864)),p=a.createCommandEncoder(),Ee(p,h,wt,Mt,Oe,256,108,12,12),a.queue.submit([p.finish()]),y.reg16=n(await Y(Oe,15552),500),p=a.createCommandEncoder(),lt(p,h,St,nt,it,T,256,128,12,12),a.queue.submit([p.finish()]);{let B=h;h=T,T=B}y.fpn12to24Conv=n(await Y(h,18432)),y.backbone24Skip=n(await Y(ze,576*128)),p=a.createCommandEncoder();{let B=ce(new Uint32Array([1,128,12,12,24,24])),q=a.createBindGroup({layout:ye,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:ze}},{binding:2,resource:{buffer:T}},{binding:3,resource:{buffer:B}}]}),F=p.beginComputePass();F.setPipeline(ne),F.setBindGroup(0,q),F.dispatchWorkgroups($(24,8),$(24,8),128),F.end()}a.queue.submit([p.finish()]);{let B=h;h=T,T=B}y.fpnUpsample12to24=n(await Y(h,576*128)),p=a.createCommandEncoder(),se(p,Rt,h,T,h),a.queue.submit([p.finish()]);{let B=h;h=T,T=B}y.fpn24Block1=n(await Y(h,576*128)),p=a.createCommandEncoder(),se(p,It,h,T,h),a.queue.submit([p.finish()]);{let B=h;h=T,T=B}return y.fpn24Block2=n(await Y(h,576*128)),p=a.createCommandEncoder(),Ee(p,h,bt,Et,ut,128,2,24,24),a.queue.submit([p.finish()]),y.cls8=n(await Y(ut,576*2)),p=a.createCommandEncoder(),Ee(p,h,Wt,Se,V,128,36,24,24),a.queue.submit([p.finish()]),y.reg8=n(await Y(V,576*36)),y.initWeights=n(await Y(xe,100),100),y.initBias=n(await Y(ve,32),32),y.cls16Weights=n(await Y(ft,100),100),y.cls16Bias=n(await Y(ht,6),6),y.cls8Weights=n(await Y(bt,100),100),y.cls8Bias=n(await Y(Et,2),2),y.fpn6to12Weights=n(await Y(Le,100),100),y}return{device:a,run:qt,debugRun:Kt}}function qn(){let r=[];for(let _=0;_<12;_++)for(let a=0;a<12;a++){let v=(a+.5)/12,g=(_+.5)/12;for(let e=0;e<6;e++)r.push({x:v,y:g})}for(let _=0;_<24;_++)for(let a=0;a<24;a++){let v=(a+.5)/24,g=(_+.5)/24;for(let e=0;e<2;e++)r.push({x:v,y:g})}return r}var _n=qn();function Kn(r){return 1/(1+Math.exp(-r))}function mn(r,_){let a=[],{scores:v,regressors:g}=r,e=192;for(let x=0;x<_n.length;x++){let D=Kn(v[x]);if(D<_)continue;let M=_n[x],l=x*18,w=M.x+g[l+0]/e,P=M.y+g[l+1]/e,A=g[l+2]/e,I=g[l+3]/e,m=[];for(let K=0;K<7;K++){let k=M.x+g[l+4+K*2]/e,de=M.y+g[l+4+K*2+1]/e;m.push([k,de])}a.push({score:D,box:[w,P,A,I],keypoints:m})}return a}function fn(r,_){if(r.length===0)return[];let a=[...r].sort((e,x)=>x.score-e.score),v=[],g=new Set;for(let e=0;e<a.length;e++)if(!g.has(e)){v.push(a[e]);for(let x=e+1;x<a.length;x++)g.has(x)||$n(a[e],a[x])>_&&g.add(x)}return v}function $n(r,_){let a=r.box[0]-r.box[2]/2,v=r.box[1]-r.box[3]/2,g=r.box[0]+r.box[2]/2,e=r.box[1]+r.box[3]/2,x=_.box[0]-_.box[2]/2,D=_.box[1]-_.box[3]/2,M=_.box[0]+_.box[2]/2,l=_.box[1]+_.box[3]/2,w=Math.max(a,x),P=Math.max(v,D),A=Math.min(g,M),I=Math.min(e,l),m=Math.max(0,A-w),K=Math.max(0,I-P),k=m*K,de=(g-a)*(e-v),pe=(M-x)*(l-D),Be=de+pe-k;return Be>0?k/Be:0}function Yn(r){let[_,a,v,g]=r.box,e=r.keypoints[0],x=r.keypoints[2],D=x[0]-e[0],M=x[1]-e[1],l=Math.atan2(D,M),P=Math.max(v,g)*2.6,A=.5,I=Math.sqrt(D*D+M*M),m=I>0?D/I*P*A*.5:0,K=I>0?M/I*P*A*.5:0;return{centerX:_+m,centerY:a+K,width:P,height:P,rotation:l}}function wa(r,_={}){let{scoreThreshold:a=.5,nmsThreshold:v=.3,maxHands:g=2}=_;async function e(D){let M=await r.run(D),l=mn(M,a);return fn(l,v).slice(0,g).map(Yn)}async function x(D){let M=await r.run(D),l=mn(M,a);return fn(l,v).slice(0,g)}return{detect:e,detectRaw:x,model:r}}function hn(r,_=256){let a=Math.cos(r.rotation),v=Math.sin(r.rotation),g=r.width/_,e=r.height/_,x=g*a,D=-e*v,M=g*v,l=e*a,w=r.centerX-(x*_/2+D*_/2),P=r.centerY-(M*_/2+l*_/2),A=x*l-D*M,I=l/A,m=-D/A,K=-M/A,k=x/A,de=-(I*w+m*P),pe=-(K*w+k*P);return{forward:[x,D,w,M,l,P],inverse:[I,m,de,K,k,pe]}}function ba(r,_){let{forward:a}=hn(_,1),[v,g,e,x,D,M]=a;return r.map(l=>({x:v*l.x+g*l.y+e,y:x*l.x+D*l.y+M,z:l.z}))}var wn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Xn(r={}){let{weightsUrl:_,scoreThreshold:a=.5,forceF32:v=!1}=r;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let g=_??wn,e=g.endsWith("/")?g:`${g}/`,x=`${e}weights_f16.json`,D=`${e}weights_f16.bin`,[M,l]=await Promise.all([fetch(x),fetch(D)]);if(!M.ok)throw new Error(`Failed to fetch weights metadata: ${M.status}`);if(!l.ok)throw new Error(`Failed to fetch weights binary: ${l.status}`);let w=await M.json(),P=await l.arrayBuffer(),A=Ot(w,P),I=await ta(A,{forceF32:v});if(!v){let O=new OffscreenCanvas(256,256),N=O.getContext("2d");N.fillStyle="#886644",N.fillRect(0,0,256,256),N.fillStyle="#cc9966",N.fillRect(50,50,156,156);let X=await I.runFromCanvas(O);X.landmarks.every(ge=>ge===0)&&X.handflag.every(ge=>ge===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),I.device.destroy(),I=await ta(A,{forceF32:!0}))}let m=null;function K(){return m||(m=new OffscreenCanvas(256,256)),m}async function k(O){if(O instanceof HTMLCanvasElement||O instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&O instanceof ImageBitmap)return O;let N=K();N.width=256,N.height=256;let X=N.getContext("2d");return O instanceof ImageData?X.putImageData(O,0,0):X.drawImage(O,0,0,256,256),N}function de(O,N,X){let We=O[0];if(We<a)return null;let ge=N[0]>.5,ye=[];for(let he=0;he<21;he++)ye.push({x:X[he*3],y:X[he*3+1],z:X[he*3+2]});return{score:We,handedness:ge?"right":"left",landmarks:ye}}async function pe(O){let N=await k(O),X=await I.runFromCanvas(N);return de(X.handflag,X.handedness,X.landmarks)}async function Be(O){let N=await k(O),X=await I.runFromCanvasPipelined(N);return X?de(X.handflag,X.handedness,X.landmarks):null}async function le(){let O=await I.flushPipelined();return O?de(O.handflag,O.handedness,O.landmarks):null}function W(){I.device.destroy(),m=null}async function j(O){let N=await k(O);return I.benchmarkDiagnostic(N)}async function re(O){let N=await k(O);return I.debugLayerOutputs(N)}return{detect:pe,detectPipelined:Be,flushPipelined:le,dispose:W,benchmarkDiagnostic:j,debugLayerOutputs:re}}async function Vn(r={}){let{weightsUrl:_,palmWeightsUrl:a,scoreThreshold:v=.5,palmScoreThreshold:g=.5,maxHands:e=2,forceF32:x=!1}=r;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let D=_??wn,M=D.endsWith("/")?D:`${D}/`;if(!a)throw new Error("palmWeightsUrl is required for createFullHandpose");let l=a.endsWith("/")?a:`${a}/`,[w,P,A,I]=await Promise.all([fetch(`${M}weights_f16.json`),fetch(`${M}weights_f16.bin`),fetch(`${l}palm_detection_weights.json`),fetch(`${l}palm_detection_weights.bin`)]);if(!w.ok)throw new Error(`Failed to fetch landmark weights metadata: ${w.status}`);if(!P.ok)throw new Error(`Failed to fetch landmark weights binary: ${P.status}`);if(!A.ok)throw new Error(`Failed to fetch palm weights metadata: ${A.status}`);if(!I.ok)throw new Error(`Failed to fetch palm weights binary: ${I.status}`);let[m,K,k,de]=await Promise.all([w.json(),P.arrayBuffer(),A.json(),I.arrayBuffer()]),pe=Ot(m,K),Be=Ot(k,de),le=await ta(pe,{forceF32:x}),W=await ha(Be),j=wa(W,{scoreThreshold:g,maxHands:e}),re=null,O=null;function N(){return re||(re=new OffscreenCanvas(192,192)),re}function X(){return O||(O=new OffscreenCanvas(256,256)),O}async function We(U){if(U instanceof HTMLCanvasElement||U instanceof OffscreenCanvas){if(U.width===192&&U.height===192)return U;let te=N();return te.width=192,te.height=192,te.getContext("2d").drawImage(U,0,0,192,192),te}if(typeof ImageBitmap<"u"&&U instanceof ImageBitmap){if(U.width===192&&U.height===192)return U;let te=N();return te.width=192,te.height=192,te.getContext("2d").drawImage(U,0,0,192,192),te}let ee=N();ee.width=192,ee.height=192;let we=ee.getContext("2d");if(U instanceof ImageData){let te=new OffscreenCanvas(U.width,U.height);te.getContext("2d").putImageData(U,0,0),we.drawImage(te,0,0,192,192)}else we.drawImage(U,0,0,192,192);return ee}function ge(U,ee,we,te){let be=X();be.width=256,be.height=256;let ne=be.getContext("2d"),Ve=Math.cos(-ee.rotation),et=Math.sin(-ee.rotation);ne.clearRect(0,0,256,256),ne.save(),ne.translate(128,128),ne.scale(ee.width*we/256,ee.height*te/256),ne.rotate(-ee.rotation),ne.translate(-128,-128);let ke=ee.centerX*we,Ue=ee.centerY*te;ne.restore();let xe=256/(ee.width*we),ve=256/(ee.height*te),He=Math.cos(ee.rotation),Ge=Math.sin(ee.rotation),Re=He*xe,Te=Ge*xe,Le=-Ge*ve,Ae=He*ve,tt=-ke*Re-Ue*Le+128,at=-ke*Te-Ue*Ae+128;if(ne.setTransform(Re,Te,Le,Ae,tt,at),U instanceof ImageData){let Ie=new OffscreenCanvas(U.width,U.height);Ie.getContext("2d").putImageData(U,0,0),ne.drawImage(Ie,0,0)}else ne.drawImage(U,0,0);return ne.setTransform(1,0,0,1,0,0),be}function ye(U){return U instanceof HTMLCanvasElement||U instanceof OffscreenCanvas?[U.width,U.height]:typeof ImageBitmap<"u"&&U instanceof ImageBitmap?[U.width,U.height]:U instanceof ImageData?[U.width,U.height]:U instanceof HTMLVideoElement?[U.videoWidth,U.videoHeight]:U instanceof HTMLImageElement?[U.naturalWidth,U.naturalHeight]:[256,256]}async function he(U){let ee=await We(U),we=await j.detect(ee);if(we.length===0)return[];let[te,be]=ye(U),ne=[];for(let Ve of we){let et=ge(U,Ve,te,be),ke=await le.runFromCanvas(et),Ue=ke.handflag[0];if(Ue<v)continue;let xe=ke.handedness[0]>.5,ve=[];for(let Ge=0;Ge<21;Ge++)ve.push({x:ke.landmarks[Ge*3],y:ke.landmarks[Ge*3+1],z:ke.landmarks[Ge*3+2]});let He=ba(ve,Ve);ne.push({score:Ue,handedness:xe?"right":"left",landmarks:He,palmScore:0})}return ne}function Ce(){le.device.destroy(),W.device.destroy(),re=null,O=null}return{detect:he,dispose:Ce}}function Zn(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var jn=Zn(`
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
`);function Jn(r){let _=r.createShaderModule({code:jn}),a=r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),v=r.createComputePipeline({layout:r.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:_,entryPoint:"main"}});function g(e,x,D,M,l,w,P){let A=new Uint32Array([l,w,P,0]),I=r.createBuffer({size:A.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});r.queue.writeBuffer(I,0,A);let m=new Float32Array(M),K=new Float32Array(8);K.set(m);let k=r.createBuffer({size:K.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});r.queue.writeBuffer(k,0,K);let de=r.createBindGroup({layout:a,entries:[{binding:0,resource:x.createView()},{binding:1,resource:{buffer:D}},{binding:2,resource:{buffer:I}},{binding:3,resource:{buffer:k}}]}),pe=e.beginComputePass();pe.setPipeline(v),pe.setBindGroup(0,de),pe.dispatchWorkgroups(Math.ceil(P/16),Math.ceil(P/16),1),pe.end()}return{crop:g}}export{ha as compilePalmModel,hn as computeCropTransform,Jn as createCropPipeline,Vn as createFullHandpose,Xn as createHandpose,wa as createPalmDetector,Ot as loadWeightsFromBuffer,ba as projectLandmarksToOriginal};
