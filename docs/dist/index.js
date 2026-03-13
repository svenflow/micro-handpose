function me(i){return i.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function La(i){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],t="enable f16;"+i;for(let k of _)for(;t.includes(`${k}:array<f32>`);)t=t.replace(`${k}:array<f32>`,`${k}:array<f16>`);return t}var da=me(`
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
`),la=me(`
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
`);function Ra(i,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${i},${_},1)`)}function Oa(i,_){return da.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${i},${_},1)`)}function Ia(i,_){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${i},${_},1)`)}function za(i,_){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${i},${_},1)`)}function Fa(i,_){return[8,8]}var Ka=me(`
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
`);function qa(i){return me(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${i?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${i?"+skip[out_idx]":""};
}
`)}var $a=qa(!1),Ya=qa(!0),Xa=me(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Va=me(`
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
`);function Za(i){return me(`
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
  ${i==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var ja=Za("sigmoid"),Ja=Za("div256"),Qa=me(`
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
`),en=me(`
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
`);function tn(i,_){let k=Math.min(_,256),w=_>k,x=i%4===0?`var ic:u32=0u;
    while(ic<${i}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${i}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,T=`var skip_val:f32=0.0;
    if(c<${i}u){
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
    }`,M=i===_?"":`if(c<${i}u){`,l=i===_?"":"}",b=w?`for(var c:u32=lid.x;c<${i}u;c+=${k}u){`:`let c=lid.x;
  ${M}`,A=w?"}":l,L=w?`for(var c:u32=lid.x;c<${_}u;c+=${k}u){`:"{let c=lid.x;";return me(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${i}>;
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
  ${A}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${L}
    let pw_base=c*${i}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${x}
    // Skip connection (only for c < inCh)
    ${T}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var an=me(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),nn=me(`
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
`),rn=me(`
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
`);function fa(i,_){let t=new Map,k=i.dtype??"float32";for(let w=0;w<i.keys.length;w++){let e=i.keys[w],x=i.shapes[w],T=i.offsets[w],M=x.reduce((A,L)=>A*L,1),l,b;if(k==="float32")l=new Float32Array(_,T,M);else{let A=new DataView(_);l=new Float32Array(M);for(let L=0;L<M;L++)l[L]=Fn(A.getUint16(T+L*2,!0));b=_.slice(T,T+M*2)}t.set(e,{data:l,shape:x,rawF16:b})}return t}function Fn(i){let _=i>>15&1,t=i>>10&31,k=i&1023;if(t===0){if(k===0)return _?-0:0;let x=-14,T=k/1024;return(_?-1:1)*Math.pow(2,x)*T}if(t===31)return k===0?_?-1/0:1/0:NaN;let w=t-15,e=1+k/1024;return(_?-1:1)*Math.pow(2,w)*e}var Kn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],sn=Kn.map(([i,_,t,k,w])=>({type:"resmodule",inCh:i,outCh:_,h:t,w:t,stride:k,prefix:w})),Nn=2,qn=5,$n=8,Yn=11;async function ha(i,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No GPU adapter found");let k=t.features.has("shader-f16"),w=k?["shader-f16"]:[],e=await t.requestDevice({requiredFeatures:w,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(t.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(t.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(t.limits.maxComputeInvocationsPerWorkgroup,288)}}),x=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(k)try{let r=`enable f16;
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
}`,u=e.createShaderModule({code:r}),o=e.createShaderModule({code:c}),a=await u.getCompilationInfo(),W=await o.getCompilationInfo();if(a.messages.some(D=>D.type==="error")||W.messages.some(D=>D.type==="error"))x=!1;else{let D=new Float32Array(2400);D.fill(1);let R=new Uint16Array(2400);R.fill(10516);let C=new Uint16Array(96);C.fill(14336);let h=new Uint16Array(9216);h.fill(8478);let p=new Uint16Array(96);p.fill(12288);let K=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ue=e.createBuffer({size:D.byteLength,usage:K}),Bt=e.createBuffer({size:R.byteLength,usage:K}),kt=e.createBuffer({size:C.byteLength,usage:K}),Ct=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),Ut=e.createBuffer({size:h.byteLength,usage:K}),At=e.createBuffer({size:p.byteLength,usage:K}),Gt=e.createBuffer({size:384,usage:ae}),Xe=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ue,0,D),e.queue.writeBuffer(Bt,0,R),e.queue.writeBuffer(kt,0,C),e.queue.writeBuffer(Ut,0,h),e.queue.writeBuffer(At,0,p);let Fe="read-only-storage",Ht="storage",Tt=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ht}}]}),Ha=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ht}}]}),Tn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Tt]}),compute:{module:u,entryPoint:"main"}}),Ln=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Ha]}),compute:{module:o,entryPoint:"main"}}),Rn=e.createBindGroup({layout:Tt,entries:[{binding:0,resource:{buffer:ue}},{binding:1,resource:{buffer:Bt}},{binding:2,resource:{buffer:kt}},{binding:3,resource:{buffer:Ct}}]}),On=e.createBindGroup({layout:Ha,entries:[{binding:0,resource:{buffer:Ct}},{binding:1,resource:{buffer:Ut}},{binding:2,resource:{buffer:At}},{binding:3,resource:{buffer:Gt}}]}),Jt=e.createCommandEncoder(),Qt=Jt.beginComputePass();Qt.setPipeline(Tn),Qt.setBindGroup(0,Rn),Qt.dispatchWorkgroups(2),Qt.end();let ea=Jt.beginComputePass();ea.setPipeline(Ln),ea.setBindGroup(0,On),ea.dispatchWorkgroups(2),ea.end(),Jt.copyBufferToBuffer(Gt,0,Xe,0,384),e.queue.submit([Jt.finish()]),await e.queue.onSubmittedWorkDone(),await Xe.mapAsync(GPUMapMode.READ);let Lt=new Float32Array(Xe.getMappedRange()),Ta=1.5*.0104*96+.25,In=Lt[0]!==0&&Lt[47]!==0&&Lt[95]!==0,zn=Math.abs(Lt[0]-Ta)<1;x=In&&zn,Xe.unmap(),ue.destroy(),Bt.destroy(),kt.destroy(),Ct.destroy(),Ut.destroy(),At.destroy(),Gt.destroy(),Xe.destroy(),x||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Lt[0]}, expected ~${Ta.toFixed(2)}) \u2014 falling back to f32`)}}catch{x=!1}let M=i.values().next().value,l=x&&!!M?.rawF16&&!_?.forceF32;console.log(l?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${k}, f16 validated: ${x}, f16 data: ${!!M?.rawF16})`);function b(r){if(l&&r.rawF16){let c=new Uint16Array(r.rawF16);if(c.length%2!==0){let u=new Uint16Array(c.length+1);return u.set(c),u}return c}return r.data}function A(r){if(l&&r.rawF16){let c=r.rawF16.byteLength;return Math.ceil(c/4)*4}return r.data.byteLength}function L(r){return l?La(r):r}let re={r:"read-only-storage",s:"storage",u:"uniform"};function g(r){return e.createBindGroupLayout({entries:r.map((c,u)=>({binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:re[c]}}))})}function j(r){return e.createBindGroupLayout({entries:r.map((c,u)=>c==="t"?{binding:u,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:re[c]}})})}let H=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ce=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,he=GPUBufferUsage.STORAGE,Be=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ne=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function S(r,c){return e.createBuffer({size:r,usage:c})}function V(r,c){return e.createBindGroup({layout:r,entries:c.map((u,o)=>({binding:o,resource:"size"in u?{buffer:u}:u}))})}function pe(r,c){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[r]}),compute:{module:c,entryPoint:"main"}})}let Ee=e.createShaderModule({code:Ka}),pt=e.createShaderModule({code:rn}),ct=e.createShaderModule({code:L(Qa)}),dt=e.createShaderModule({code:L(_a)}),_t=e.createShaderModule({code:L(da)}),we=e.createShaderModule({code:L(la)}),Ve=e.createShaderModule({code:L(ma)}),P=e.createShaderModule({code:L(Na)}),Y=e.createShaderModule({code:$a}),_e=e.createShaderModule({code:Xa}),X=e.createShaderModule({code:Ya}),se=e.createShaderModule({code:L(Va)}),Q=e.createShaderModule({code:L(ja)}),q=e.createShaderModule({code:L(Ja)}),We=e.createShaderModule({code:L(en)}),be=new Map;function Ke(r,c){let u=`${r}_${c}`,o=be.get(u);return o||(o=e.createShaderModule({code:L(tn(r,c))}),be.set(u,o)),o}let De=g(["r","r","r","s","u"]),ge=g(["r","r","r","r","s","u"]),ke=g(["r","s","u"]),Ce=g(["r","r","r","s","u"]),Ze=g(["r","s","u"]),xe=g(["r","r","s","u"]),ye=g(["r","r","s","u"]),ve=g(["r","r","r","s","u"]),le=g(["r","r","r","s","u"]),Ne=j(["t","s","u"]),lt=g(["r","r","r","r","r","r","r","s"]),mt=g(["r","r","r","r","r","s","u"]),Rt=e.createPipelineLayout({bindGroupLayouts:[De]}),St=e.createPipelineLayout({bindGroupLayouts:[ge]}),je=r=>e.createComputePipeline({layout:Rt,compute:{module:r,entryPoint:"main"}}),Je=r=>e.createComputePipeline({layout:St,compute:{module:r,entryPoint:"main"}}),Ot=je(dt),It=je(_t),zt=Je(we),ft=Je(Ve),ht=new Map,Dt=new Map,wt=new Map,Mt=new Map;ht.set("8,8",Ot),Dt.set("8,8",It),wt.set("8,8",zt),Mt.set("8,8",ft);function Qe(r,c,u,o,a){let W=`${c},${u}`,D=r.get(W);return D||(D=a(e.createShaderModule({code:L(o(c,u))})),r.set(W,D)),D}let bt=(r,c)=>Qe(ht,r,c,Ra,je),Et=(r,c)=>Qe(Dt,r,c,Oa,je),Ft=(r,c)=>Qe(wt,r,c,Ia,Je),Wt=(r,c)=>Qe(Mt,r,c,za,Je),Ue=sn.map(r=>{let c=r.stride===2?r.h/2:r.h,u=r.stride===2?r.w/2:r.w,[o,a]=Fa(r.inCh,c),W=r.h>=64,D=c>=16&&r.inCh>=288&&r.outCh>=288&&r.outCh%2===0;return{dwPipeline:W?Et(o,a):bt(o,a),pwPipeline:D?Wt(o,a):Ft(o,a),dwDispatchX:Math.ceil(u/o),dwDispatchY:Math.ceil(c/a),dwDispatchZ:r.inCh,pwDispatchX:Math.ceil(u/o),pwDispatchY:Math.ceil(c/a),pwDispatchZ:D?r.outCh/2:r.outCh}}),gt=pe(ke,Ee),qe=pe(Ce,P);pe(Ze,Y),pe(xe,_e);let He=pe(ye,X),Kt=pe(ve,se);pe(le,Q),pe(le,q);let Ae=pe(Ne,pt),yt=pe(lt,ct),et=pe(mt,We),Te=1*288*128*128*4,tt=S(3*256*256*4,H),Me=S(3*257*257*4,he),at=S(12,ne);e.queue.writeBuffer(at,0,new Uint32Array([3,256,257]));let Z=S(Te,ce),de=S(Te,Be),Ge=S(Te,he),nt=S(3072*64*4,H),it=S(3072*32*4,H),rt=S(1536*16*4,H),I=S(6144*64*4,he),ee=S(260,Be),J=S(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);S(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let ie=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Se=S(8,ne);e.queue.writeBuffer(Se,0,new Uint32Array([256,257]));let st=i.get("backbone1.1.weight"),Nt=i.get("backbone1.1.bias");if(!st||!Nt)throw new Error("Missing input conv weights");let N=b(st),qt=b(Nt),d=S(N.byteLength,H),n=S(qt.byteLength,H),y=S(28,ne);e.queue.writeBuffer(d,0,N),e.queue.writeBuffer(n,0,qt),e.queue.writeBuffer(y,0,new Uint32Array([1,3,24,257,257,128,128]));let G=i.get("backbone6.1.weight"),f=i.get("backbone6.1.bias");if(!G||!f)throw new Error("Missing backbone6.1 conv1x1 weights");let s=b(G),O=b(f),te=S(s.byteLength,H),m=S(O.byteLength,H),U=S(20,ne);e.queue.writeBuffer(te,0,s),e.queue.writeBuffer(m,0,O),e.queue.writeBuffer(U,0,new Uint32Array([1,96,48,32,32]));let v=i.get("handflag.weight"),F=i.get("handflag.bias");if(!v||!F)throw new Error("Missing handflag weights");let E=b(v),fe=b(F),oe=S(E.byteLength,H),B=S(fe.byteLength,H),$=S(12,ne);e.queue.writeBuffer(oe,0,E),e.queue.writeBuffer(B,0,fe),e.queue.writeBuffer($,0,new Uint32Array([1,288,1]));let z=i.get("handedness.weight"),Pe=i.get("handedness.bias");if(!z||!Pe)throw new Error("Missing handedness weights");let xt=b(z),xa=b(Pe),ta=S(xt.byteLength,H),aa=S(xa.byteLength,H),va=S(12,ne);e.queue.writeBuffer(ta,0,xt),e.queue.writeBuffer(aa,0,xa),e.queue.writeBuffer(va,0,new Uint32Array([1,288,1]));let Pa=i.get("reg_3d.weight"),Ba=i.get("reg_3d.bias");if(!Pa||!Ba)throw new Error("Missing reg_3d weights");let ka=b(Pa),Ca=b(Ba),na=S(ka.byteLength,H),ia=S(Ca.byteLength,H),Ua=S(12,ne);e.queue.writeBuffer(na,0,ka),e.queue.writeBuffer(ia,0,Ca),e.queue.writeBuffer(Ua,0,new Uint32Array([1,288,63]));let ot=sn.map(r=>{let{inCh:c,outCh:u,h:o,w:a,stride:W,prefix:D}=r,R=W===2?o/2:o,C=W===2?a/2:a,h=W===2?1:2,p=i.get(`${D}convs.0.weight`),K=i.get(`${D}convs.0.bias`),ae=i.get(`${D}convs.1.weight`),ue=i.get(`${D}convs.1.bias`);if(!p||!K||!ae||!ue)throw new Error(`Missing weights for ${D}`);let Bt=b(p),kt=b(K),Ct=b(ae),Ut=b(ue),At=S(Bt.byteLength,H),Gt=S(kt.byteLength,H),Xe=S(Ct.byteLength,H),Fe=S(Ut.byteLength,H),Ht=S(32,ne),Tt=S(36,ne);return e.queue.writeBuffer(At,0,Bt),e.queue.writeBuffer(Gt,0,kt),e.queue.writeBuffer(Xe,0,Ct),e.queue.writeBuffer(Fe,0,Ut),e.queue.writeBuffer(Ht,0,new Uint32Array([1,c,o,a,R,C,W,h])),e.queue.writeBuffer(Tt,0,new Uint32Array([1,c,u,R,C,Math.max(0,u-c),W,o,a])),{dwWeight:At,dwBias:Gt,pwWeight:Xe,pwBias:Fe,dwUniform:Ht,pwUniform:Tt,spec:r,outH:R,outW:C}});function vt(r){let c=S(r.length*4,ne);return e.queue.writeBuffer(c,0,new Uint32Array(r)),c}let vn=vt([1,96,8,8,16,16]),Pn=vt([1,96,16,16,32,32]),Bn=vt([1,48,32,32,64,64]);vt([1536*16]),vt([3072*32]),vt([3072*64]);let Aa=V(ke,[tt,Me,at]),Ga=V(Ce,[Me,d,n,Z,y]),Le=[],Re=[],Oe=[],Ie=[];for(let r of ot)Le.push(V(De,[Z,r.dwWeight,r.dwBias,Ge,r.dwUniform])),Re.push(V(ge,[Ge,Z,r.pwWeight,r.pwBias,de,r.pwUniform])),Oe.push(V(De,[de,r.dwWeight,r.dwBias,Ge,r.dwUniform])),Ie.push(V(ge,[Ge,de,r.pwWeight,r.pwBias,Z,r.pwUniform]));let kn=V(ye,[Z,rt,de,vn]),Cn=V(ye,[Z,it,de,Pn]),Un=V(ve,[Z,te,m,I,U]),An=V(ye,[I,nt,de,Bn]);V(le,[Z,oe,B,ee,$]),V(le,[Z,ta,aa,ee,va]),V(le,[Z,na,ia,ee,Ua]);let $e=V(Ne,[ie.createView(),Me,Se]),Gn=V(lt,[Z,oe,B,ta,aa,na,ia,ee]),ra=24,Sa=[],Da=[];for(let r=ra;r<ot.length;r++){let c=ot[r];Sa.push(V(mt,[Z,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,de,c.dwUniform])),Da.push(V(mt,[de,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,Z,c.dwUniform]))}let sa=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});sa.globalCompositeOperation="copy";let Ma=new OffscreenCanvas(9,8),$t=Ma.getContext("webgpu"),Yt=null,oa=null;if($t){$t.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let r=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),c=e.createShaderModule({code:an}),u=e.createShaderModule({code:nn});Yt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[r]}),vertex:{module:c,entryPoint:"vs"},fragment:{module:u,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),oa=e.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:ee}}]})}let Xt=new Float32Array(1),Vt=new Float32Array(1),Zt=new Float32Array(63);function ze(r,c){let u=!0,o=0,a=r.beginComputePass();for(a.setPipeline(qe),a.setBindGroup(0,Ga),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);o<=Nn;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Ue[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let W=u?Z:de;for(r.copyBufferToBuffer(W,0,nt,0,3072*64*4),a=r.beginComputePass();o<=qn;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Ue[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let D=u?Z:de;for(r.copyBufferToBuffer(D,0,it,0,3072*32*4),a=r.beginComputePass();o<=$n;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Ue[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let R=u?Z:de;for(r.copyBufferToBuffer(R,0,rt,0,1536*16*4),a=r.beginComputePass();o<=Yn;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Ue[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.setPipeline(He),a.setBindGroup(0,kn),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),u=!1,a=r.beginComputePass();{let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Ue[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}a.setPipeline(He),a.setBindGroup(0,Cn),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),u=!1,a=r.beginComputePass();{let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Ue[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}for(a.setPipeline(Kt),a.setBindGroup(0,Un),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(He),a.setBindGroup(0,An),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),u=!1,a=r.beginComputePass();o<ra;o++){let C=u?Le[o]:Oe[o],h=u?Re[o]:Ie[o],p=Ue[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,C),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}for(;o<ot.length;o++){let C=o-ra,h=u?Sa[C]:Da[C],p=ot[o];a.setPipeline(et),a.setBindGroup(0,h),a.dispatchWorkgroups(p.outW,p.outH,1),u=!u}a.setPipeline(yt),a.setBindGroup(0,Gn),a.dispatchWorkgroups(1),a.end(),c&&r.copyBufferToBuffer(ee,0,c,0,260)}async function jt(r){e.queue.writeBuffer(tt,0,r);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(gt),a.setBindGroup(0,Aa),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),a.end()}ze(c,J),e.queue.submit([c.finish()]);let u=J.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(J.getMappedRange());return Xt[0]=o[0],Vt[0]=o[1],Zt.set(o.subarray(2,65)),J.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function ua(r){e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(Ae),a.setBindGroup(0,$e),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}ze(c,J),e.queue.submit([c.finish()]);let u=J.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(J.getMappedRange());return Xt[0]=o[0],Vt[0]=o[1],Zt.set(o.subarray(2,65)),J.unmap(),{handflag:new Float32Array(Xt),handedness:new Float32Array(Vt),landmarks:new Float32Array(Zt)}}async function Ea(r){if(!Yt||!oa||!$t)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);let c=e.createCommandEncoder();{let C=c.beginComputePass();C.setPipeline(Ae),C.setBindGroup(0,$e),C.dispatchWorkgroups(16,16,1),C.end()}ze(c,null);let u=$t.getCurrentTexture(),o=c.beginRenderPass({colorAttachments:[{view:u.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});o.setPipeline(Yt),o.setBindGroup(0,oa),o.draw(3),o.end(),e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),sa.drawImage(Ma,0,0);let W=sa.getImageData(0,0,9,8).data,D=new Float32Array(65),R=new DataView(new ArrayBuffer(4));for(let C=0;C<65;C++){let h=C*4;R.setUint8(0,W[h]),R.setUint8(1,W[h+1]),R.setUint8(2,W[h+2]),R.setUint8(3,W[h+3]),D[C]=R.getFloat32(0)}return{handflag:new Float32Array([D[0]]),handedness:new Float32Array([D[1]]),landmarks:new Float32Array(D.subarray(2,65))}}let Sn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),pa=0,Dn=[J,Sn],Pt=null,Ye=null;async function ca(r){let c=Dn[pa];pa=1-pa,e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);let u=e.createCommandEncoder();{let a=u.beginComputePass();a.setPipeline(Ae),a.setBindGroup(0,$e),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}ze(u,c),e.queue.submit([u.finish()]);let o=null;if(Pt!==null&&Ye!==null){await Pt;let a=new Float32Array(Ye.getMappedRange());o={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))},Ye.unmap()}return Ye=c,Pt=c.mapAsync(GPUMapMode.READ),o}async function Wa(){if(!Pt||!Ye)return null;await Pt;let r=new Float32Array(Ye.getMappedRange()),c={handflag:new Float32Array([r[0]]),handedness:new Float32Array([r[1]]),landmarks:new Float32Array(r.subarray(2,65))};return Ye.unmap(),Pt=null,Ye=null,c}async function Mn(r=50){let c=new Float32Array(196608);for(let a=0;a<5;a++)await jt(c);let u=[];for(let a=0;a<r;a++){let W=performance.now();await jt(c),u.push(performance.now()-W)}let o=u.reduce((a,W)=>a+W,0)/u.length;return{avgMs:o,fps:1e3/o}}async function En(r=50){let c=new Float32Array(196608);for(let D=0;D<5;D++)await jt(c);let u=[];for(let D=0;D<r;D++){let R=e.createCommandEncoder();{let h=R.beginComputePass();h.setPipeline(gt),h.setBindGroup(0,Aa),h.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),h.end()}ze(R,J);let C=performance.now();e.queue.submit([R.finish()]),await e.queue.onSubmittedWorkDone(),u.push(performance.now()-C)}u.sort((D,R)=>D-R);let o=u.reduce((D,R)=>D+R,0)/u.length,a=u[Math.floor(u.length/2)],W=u[0];return{avgMs:o,fps:1e3/o,medianMs:a,minMs:W}}function ni(r){e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);let c=e.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(Ae),u.setBindGroup(0,$e),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),u.end()}ze(c,J),e.queue.submit([c.finish()])}async function Wn(r,c=50){function u(h){let p=[...h].sort((K,ae)=>K-ae);return{median:p[Math.floor(p.length/2)],min:p[0]}}for(let h=0;h<10;h++)await ua(r);let o=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let ae=p.beginComputePass();ae.setPipeline(Ae),ae.setBindGroup(0,$e),ae.dispatchWorkgroups(16,16,1),ae.end()}ze(p,J);let K=performance.now();e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-K)}let a=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let ue=p.beginComputePass();ue.setPipeline(Ae),ue.setBindGroup(0,$e),ue.dispatchWorkgroups(16,16,1),ue.end()}ze(p,J),e.queue.submit([p.finish()]);let K=J.mapAsync(GPUMapMode.READ),ae=performance.now();await e.queue.onSubmittedWorkDone(),await K,J.getMappedRange(),J.unmap(),a.push(performance.now()-ae)}let W=[];for(let h=0;h<c;h++){e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);let p=e.createCommandEncoder();{let ae=p.beginComputePass();ae.setPipeline(Ae),ae.setBindGroup(0,$e),ae.dispatchWorkgroups(16,16,1),ae.end()}ze(p,J),e.queue.submit([p.finish()]);let K=performance.now();await J.mapAsync(GPUMapMode.READ),J.getMappedRange(),J.unmap(),W.push(performance.now()-K)}let D=[];for(let h=0;h<c;h++){let p=performance.now();await ua(r),D.push(performance.now()-p)}await ca(r);let R=[];for(let h=0;h<c;h++){let p=performance.now();await ca(r),R.push(performance.now()-p)}await Wa();let C=null;if(Yt){let h=[];for(let p=0;p<c;p++){let K=performance.now();await Ea(r),h.push(performance.now()-K)}C=u(h)}return{gpuOnly:u(o),mapAsyncOnly:u(a),mapAsyncNoWait:u(W),total:u(D),pipelined:u(R),renderReadback:C}}async function Hn(r){let c=[];async function u(a,W,D){let R=e.createBuffer({size:W,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),C=e.createCommandEncoder();C.copyBufferToBuffer(a,0,R,0,W),e.queue.submit([C.finish()]),await e.queue.onSubmittedWorkDone(),await R.mapAsync(GPUMapMode.READ);let h=new Float32Array(R.getMappedRange()),p=1/0,K=-1/0,ae=0;for(let ue=0;ue<h.length;ue++)h[ue]<p&&(p=h[ue]),h[ue]>K&&(K=h[ue]),h[ue]!==0&&ae++;R.unmap(),R.destroy(),c.push({layer:D,stats:{min:p,max:K,nonZero:ae,total:h.length}})}e.queue.copyExternalImageToTexture({source:r},{texture:ie},[256,256]);{let a=e.createCommandEncoder(),W=a.beginComputePass();W.setPipeline(Ae),W.setBindGroup(0,$e),W.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),W.end(),e.queue.submit([a.finish()])}await u(Me,Math.min(Me.size,3*257*257*4),"canvas\u2192bufInput");{let a=e.createCommandEncoder(),W=a.beginComputePass();W.setPipeline(qe),W.setBindGroup(0,Ga),W.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),W.end(),e.queue.submit([a.finish()])}await u(Z,Math.min(Z.size,3072*128*4),"inputConv\u2192bufA");let o=!0;for(let a=0;a<Math.min(ot.length,6);a++){let W=o?Le[a]:Oe[a],D=o?Re[a]:Ie[a],R=Ue[a],C=ot[a];{let p=e.createCommandEncoder(),K=p.beginComputePass();K.setPipeline(R.dwPipeline),K.setBindGroup(0,W),K.dispatchWorkgroups(R.dwDispatchX,R.dwDispatchY,R.dwDispatchZ),K.end(),e.queue.submit([p.finish()])}await u(Ge,Math.min(Ge.size,C.spec.inCh*C.outH*C.outW*4),`layer${a}.DW\u2192bufDW (${C.spec.prefix})`);{let p=e.createCommandEncoder(),K=p.beginComputePass();K.setPipeline(R.pwPipeline),K.setBindGroup(0,D),K.dispatchWorkgroups(R.pwDispatchX,R.pwDispatchY,R.pwDispatchZ),K.end(),e.queue.submit([p.finish()])}let h=o?de:Z;await u(h,Math.min(h.size,C.spec.outCh*C.outH*C.outW*4),`layer${a}.PW\u2192buf${o?"B":"A"} (${C.spec.prefix})`),o=!o}return c}return{device:e,run:jt,runFromCanvas:ua,runFromCanvasViaRender:Ea,runFromCanvasPipelined:ca,flushPipelined:Wa,benchmark:Mn,benchmarkGPU:En,benchmarkDiagnostic:Wn,debugLayerOutputs:Hn}}function ut(i){return i.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var on=ut(`
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
`);async function mn(i,_){let t;if(_)t=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");t=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8)}})}let k={r:"read-only-storage",s:"storage",u:"uniform"};function w(d){return t.createBindGroupLayout({entries:d.map((n,y)=>n==="t"?{binding:y,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:y,visibility:GPUShaderStage.COMPUTE,buffer:{type:k[n]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,x=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,T=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,M=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(d,n){return t.createBuffer({size:Math.max(d,4),usage:n})}function b(d,n,y){t.queue.writeBuffer(d,n,y)}function A(d){let n=l(d.data.byteLength,e);return b(n,0,d.data),n}let L=Array.from(i.keys());function re(d){let n=i.get(d);if(!n)throw new Error(`Weight not found: ${d}`);return n}function g(...d){let n=L.find(y=>d.every(G=>y.includes(G)));if(!n)throw new Error(`Weight not found for: ${d.join(", ")}`);return re(n)}function j(d){let[,n,y,G]=d.shape,f=new Float32Array(G*25);for(let s=0;s<G;s++)for(let O=0;O<n;O++)for(let te=0;te<y;te++)f[s*25+O*5+te]=d.data[O*y*G+te*G+s];return f}function H(d){let[n,,,y]=d.shape,G=new Float32Array(n*y);for(let f=0;f<n;f++)for(let s=0;s<y;s++)G[f*y+s]=d.data[f*y+s];return G}let ce=t.createShaderModule({code:on}),he=t.createShaderModule({code:un}),Be=t.createShaderModule({code:pn}),ne=t.createShaderModule({code:cn}),S=t.createShaderModule({code:_n}),V=t.createShaderModule({code:dn}),pe=t.createShaderModule({code:ln}),Ee=w(["r","r","r","r","s","u"]),pt=w(["r","r","r","s","u"]),ct=w(["r","r","r","r","r","s","u"]),dt=w(["r","r","r","s","u"]),_t=w(["r","r","r","r","s","u"]),we=w(["r","r","s","u"]),Ve=w(["t","s","u"]);function P(d,n){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:n,entryPoint:"main"}})}let Y=P(Ee,ce),_e=P(pt,he),X=P(ct,Be),se=P(dt,ne),Q=P(_t,S),q=P(we,V),We=P(Ve,pe),be=g("conv2d/Conv2D"),Ke=g("batch_normalization/","conv2d/Conv2D"),De=g("p_re_lu/"),ge=A(be),ke=A(Ke),Ce=A(De),xe=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(d=>{let n=g(d.dwKey),y=g(d.pwKey),G=g(d.bnKey),f=g(d.preluKey),s=j(n),O=l(s.byteLength,e);b(O,0,s);let te=new Float32Array(d.inCh),m=l(te.byteLength,e);b(m,0,te);let U=H(y),v=l(U.byteLength,e);b(v,0,U);let F=A(G),E=A(f);return{dwWeightBuf:O,dwBiasBuf:m,pwWeightBuf:v,pwBiasBuf:F,alphaBuf:E,inCh:d.inCh,outCh:d.outCh,stride:d.stride,inH:d.inH}}),ye=H(g("conv2d_20/Conv2D")),ve=l(ye.byteLength,e);b(ve,0,ye);let le=A(g("batch_normalization_20/")),Ne=A(g("p_re_lu_20/")),lt={dwWeightBuf:(()=>{let d=j(g("depthwise_conv2d_19/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(g("conv2d_21/")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:A(g("batch_normalization_21/")),alphaBuf:A(g("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},mt={dwWeightBuf:(()=>{let d=j(g("depthwise_conv2d_20/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(256),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(g("conv2d_22/Conv2D1")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:A(g("batch_normalization_22/")),alphaBuf:A(g("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},Rt=H(g("conv2d_23/Conv2D")),St=l(Rt.byteLength,e);b(St,0,Rt);let je=A(g("batch_normalization_23/")),Je=A(g("p_re_lu_23/")),Ot={dwWeightBuf:(()=>{let d=j(g("depthwise_conv2d_21/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(g("conv2d_24/")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:A(g("batch_normalization_24/")),alphaBuf:A(g("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},It={dwWeightBuf:(()=>{let d=j(g("depthwise_conv2d_22/")),n=l(d.byteLength,e);return b(n,0,d),n})(),dwBiasBuf:(()=>{let d=new Float32Array(128),n=l(d.byteLength,e);return b(n,0,d),n})(),pwWeightBuf:(()=>{let d=H(g("conv2d_25/Conv2D1")),n=l(d.byteLength,e);return b(n,0,d),n})(),pwBiasBuf:A(g("batch_normalization_25/")),alphaBuf:A(g("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},zt=H(g("classifier_palm_16_NO_PRUNING/Conv2D")),ft=l(zt.byteLength,e);b(ft,0,zt);let ht=A(g("classifier_palm_16_NO_PRUNING/BiasAdd")),Dt=H(g("regressor_palm_16_NO_PRUNING/Conv2D")),wt=l(Dt.byteLength,e);b(wt,0,Dt);let Mt=A(g("regressor_palm_16_NO_PRUNING/BiasAdd")),Qe=H(g("classifier_palm_8_NO_PRUNING/Conv2D")),bt=l(Qe.byteLength,e);b(bt,0,Qe);let Et=A(g("classifier_palm_8_NO_PRUNING/BiasAdd")),Ft=H(g("regressor_palm_8_NO_PRUNING/Conv2D")),Wt=l(Ft.byteLength,e);b(Wt,0,Ft);let Ue=A(g("regressor_palm_8_NO_PRUNING/BiasAdd")),gt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,qe=l(36864*3*4,e),He=l(gt,x),Kt=l(gt,x),Ae=l(gt,x),yt=l(576*256*4,x),et=l(144*256*4,x|GPUBufferUsage.COPY_DST),Te=l(576*128*4,x|GPUBufferUsage.COPY_DST),tt=l(864*4,T),Me=l(15552*4,T),at=l(576*2*4,T),Z=l(576*36*4,T),de=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ge=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),nt=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),it=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),rt=t.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function I(d,n){return Math.ceil(d/n)}function ee(d){let n=l(d.byteLength,M);return b(n,0,d),n}let J=ee(new Uint32Array([1,3,32,192,192,96,96]));function ie(d,n,y,G,f){let s=n.stride===2?n.inH/2:n.inH,O=s,te=n.stride===2?1:2,m=ee(new Uint32Array([1,n.inCh,n.inH,n.inH,s,O,n.stride,te])),U=t.createBindGroup({layout:pt,entries:[{binding:0,resource:{buffer:y}},{binding:1,resource:{buffer:n.dwWeightBuf}},{binding:2,resource:{buffer:n.dwBiasBuf}},{binding:3,resource:{buffer:Ae}},{binding:4,resource:{buffer:m}}]}),v=d.beginComputePass();v.setPipeline(_e),v.setBindGroup(0,U),v.dispatchWorkgroups(I(O,8),I(s,8),n.inCh),v.end();let F=n.inCh,E=ee(new Uint32Array([1,n.inCh,n.outCh,s,O,F,n.stride,n.inH,n.inH])),fe=t.createBindGroup({layout:ct,entries:[{binding:0,resource:{buffer:Ae}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:n.pwWeightBuf}},{binding:3,resource:{buffer:n.pwBiasBuf}},{binding:4,resource:{buffer:n.alphaBuf}},{binding:5,resource:{buffer:G}},{binding:6,resource:{buffer:E}}]}),oe=d.beginComputePass();oe.setPipeline(X),oe.setBindGroup(0,fe),oe.dispatchWorkgroups(I(O,8),I(s,8),n.outCh),oe.end()}function Se(d,n,y,G,f,s,O,te,m){let U=ee(new Uint32Array([1,s,O,te,m])),v=t.createBindGroup({layout:dt,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:y}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:U}}]}),F=d.beginComputePass();F.setPipeline(se),F.setBindGroup(0,v),F.dispatchWorkgroups(I(m,8),I(te,8),O),F.end()}function st(d,n,y,G,f,s,O,te,m,U){let v=ee(new Uint32Array([1,O,te,m,U])),F=t.createBindGroup({layout:_t,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:y}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:s}},{binding:5,resource:{buffer:v}}]}),E=d.beginComputePass();E.setPipeline(Q),E.setBindGroup(0,F),E.dispatchWorkgroups(I(U,8),I(m,8),te),E.end()}async function Nt(d){t.queue.copyExternalImageToTexture({source:d},{texture:rt},[192,192]);let n=ee(new Uint32Array([192,192,192])),y=t.createBindGroup({layout:Ve,entries:[{binding:0,resource:rt.createView()},{binding:1,resource:{buffer:qe}},{binding:2,resource:{buffer:n}}]}),G=t.createCommandEncoder();{let B=G.beginComputePass();B.setPipeline(We),B.setBindGroup(0,y),B.dispatchWorkgroups(I(192,16),I(192,16),1),B.end()}{let B=t.createBindGroup({layout:Ee,entries:[{binding:0,resource:{buffer:qe}},{binding:1,resource:{buffer:ge}},{binding:2,resource:{buffer:ke}},{binding:3,resource:{buffer:Ce}},{binding:4,resource:{buffer:He}},{binding:5,resource:{buffer:J}}]}),$=G.beginComputePass();$.setPipeline(Y),$.setBindGroup(0,B),$.dispatchWorkgroups(I(96,8),I(96,8),32),$.end()}let f=He,s=Kt;for(let B=0;B<xe.length;B++){let $=xe[B];ie(G,$,f,s,f);let z=f;f=s,s=z,B===10&&G.copyBufferToBuffer(f,0,Te,0,576*128*4),B===14&&G.copyBufferToBuffer(f,0,et,0,144*256*4)}{let B=ee(new Uint32Array([1,256,6,6,12,12])),$=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(q),z.setBindGroup(0,$),z.dispatchWorkgroups(I(12,8),I(12,8),256),z.end()}{let B=f;f=s,s=B}st(G,f,ve,le,Ne,s,256,256,12,12);{let B=f;f=s,s=B}{let B=ee(new Uint32Array([1,256,12,12,12,12])),$=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:et}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(q),z.setBindGroup(0,$),z.dispatchWorkgroups(I(12,8),I(12,8),256),z.end()}{let B=f;f=s,s=B}ie(G,lt,f,s,f);{let B=f;f=s,s=B}ie(G,mt,f,s,f);{let B=f;f=s,s=B}Se(G,f,ft,ht,tt,256,6,12,12),Se(G,f,wt,Mt,Me,256,108,12,12);{let B=ee(new Uint32Array([1,256,12,12,24,24])),$=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(q),z.setBindGroup(0,$),z.dispatchWorkgroups(I(24,8),I(24,8),256),z.end()}{let B=f;f=s,s=B}st(G,f,St,je,Je,s,256,128,24,24);{let B=f;f=s,s=B}{let B=ee(new Uint32Array([1,128,24,24,24,24])),$=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Te}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:B}}]}),z=G.beginComputePass();z.setPipeline(q),z.setBindGroup(0,$),z.dispatchWorkgroups(I(24,8),I(24,8),128),z.end()}{let B=f;f=s,s=B}ie(G,Ot,f,s,f);{let B=f;f=s,s=B}ie(G,It,f,s,f);{let B=f;f=s,s=B}Se(G,f,bt,Et,at,128,2,24,24),Se(G,f,Wt,Ue,Z,128,36,24,24),t.queue.submit([G.finish()]);let O=t.createCommandEncoder();O.copyBufferToBuffer(tt,0,de,0,864*4),O.copyBufferToBuffer(Me,0,Ge,0,15552*4),O.copyBufferToBuffer(at,0,nt,0,576*2*4),O.copyBufferToBuffer(Z,0,it,0,576*36*4),t.queue.submit([O.finish()]),await Promise.all([de.mapAsync(GPUMapMode.READ),Ge.mapAsync(GPUMapMode.READ),nt.mapAsync(GPUMapMode.READ),it.mapAsync(GPUMapMode.READ)]);let te=new Float32Array(de.getMappedRange()).slice(),m=new Float32Array(Ge.getMappedRange()).slice(),U=new Float32Array(nt.getMappedRange()).slice(),v=new Float32Array(it.getMappedRange()).slice();de.unmap(),Ge.unmap(),nt.unmap(),it.unmap();let F=2016,E=new Float32Array(F),fe=new Float32Array(F*18),oe=0;for(let B=0;B<12;B++)for(let $=0;$<12;$++)for(let z=0;z<6;z++){E[oe]=te[z*144+B*12+$];for(let Pe=0;Pe<18;Pe++){let xt=z*18+Pe;fe[oe*18+Pe]=m[xt*144+B*12+$]}oe++}for(let B=0;B<24;B++)for(let $=0;$<24;$++)for(let z=0;z<2;z++){E[oe]=U[z*576+B*24+$];for(let Pe=0;Pe<18;Pe++){let xt=z*18+Pe;fe[oe*18+Pe]=v[xt*576+B*24+$]}oe++}return{scores:E,regressors:fe}}async function N(d,n){let y=t.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),G=t.createCommandEncoder();G.copyBufferToBuffer(d,0,y,0,n*4),t.queue.submit([G.finish()]),await y.mapAsync(GPUMapMode.READ);let f=new Float32Array(y.getMappedRange()).slice();return y.unmap(),y.destroy(),f}async function qt(d){t.queue.copyExternalImageToTexture({source:d},{texture:rt},[192,192]);function n(v,F=1e3){let E=v.slice(0,F);return{min:Math.min(...E),max:Math.max(...E),mean:E.reduce((fe,oe)=>fe+oe,0)/E.length,nonZero:E.filter(fe=>fe!==0).length,sample:Array.from(E.slice(0,10))}}let y={},G=ee(new Uint32Array([192,192,192])),f=t.createBindGroup({layout:Ve,entries:[{binding:0,resource:rt.createView()},{binding:1,resource:{buffer:qe}},{binding:2,resource:{buffer:G}}]}),s=t.createCommandEncoder(),O=s.beginComputePass();O.setPipeline(We),O.setBindGroup(0,f),O.dispatchWorkgroups(I(192,16),I(192,16),1),O.end(),t.queue.submit([s.finish()]),y.input=n(await N(qe,36864*3)),s=t.createCommandEncoder();let te=t.createBindGroup({layout:Ee,entries:[{binding:0,resource:{buffer:qe}},{binding:1,resource:{buffer:ge}},{binding:2,resource:{buffer:ke}},{binding:3,resource:{buffer:Ce}},{binding:4,resource:{buffer:He}},{binding:5,resource:{buffer:J}}]});O=s.beginComputePass(),O.setPipeline(Y),O.setBindGroup(0,te),O.dispatchWorkgroups(I(96,8),I(96,8),32),O.end(),t.queue.submit([s.finish()]),y.initConv=n(await N(He,9216*32));let m=He,U=Kt;for(let v=0;v<xe.length;v++){let F=xe[v];s=t.createCommandEncoder(),ie(s,F,m,U,m),t.queue.submit([s.finish()]);let E=m;if(m=U,U=E,v===0||v===3||v===7||v===11||v===14||v===15||v===18){let fe=F.stride===2?F.inH/2:F.inH,oe=fe*fe*F.outCh;y[`block${v}`]=n(await N(m,oe))}v===10&&(s=t.createCommandEncoder(),s.copyBufferToBuffer(m,0,Te,0,576*128*4),t.queue.submit([s.finish()])),v===14&&(s=t.createCommandEncoder(),s.copyBufferToBuffer(m,0,et,0,144*256*4),t.queue.submit([s.finish()]))}s=t.createCommandEncoder();{let v=ee(new Uint32Array([1,256,6,6,12,12])),F=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:v}}]}),E=s.beginComputePass();E.setPipeline(q),E.setBindGroup(0,F),E.dispatchWorkgroups(I(12,8),I(12,8),256),E.end()}t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpnUpsample6to12=n(await N(m,144*256)),s=t.createCommandEncoder(),st(s,m,ve,le,Ne,U,256,256,12,12),t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpn6to12Conv=n(await N(m,144*256)),y.backbone12Skip=n(await N(et,144*256)),s=t.createCommandEncoder();{let v=ee(new Uint32Array([1,256,12,12,12,12])),F=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:et}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:v}}]}),E=s.beginComputePass();E.setPipeline(q),E.setBindGroup(0,F),E.dispatchWorkgroups(I(12,8),I(12,8),256),E.end()}t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpnAdd12=n(await N(m,144*256)),s=t.createCommandEncoder(),ie(s,lt,m,U,m),t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpn12Block1=n(await N(m,144*256)),s=t.createCommandEncoder(),ie(s,mt,m,U,m),t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpn12Block2=n(await N(m,144*256)),s=t.createCommandEncoder(),Se(s,m,ft,ht,tt,256,6,12,12),t.queue.submit([s.finish()]),y.cls16=n(await N(tt,864)),s=t.createCommandEncoder(),Se(s,m,wt,Mt,Me,256,108,12,12),t.queue.submit([s.finish()]),y.reg16=n(await N(Me,15552),500),s=t.createCommandEncoder();{let v=ee(new Uint32Array([1,256,12,12,24,24])),F=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:v}}]}),E=s.beginComputePass();E.setPipeline(q),E.setBindGroup(0,F),E.dispatchWorkgroups(I(24,8),I(24,8),256),E.end()}t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpnUpsample12to24=n(await N(m,576*256)),s=t.createCommandEncoder(),st(s,m,St,je,Je,U,256,128,24,24),t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpn12to24Conv=n(await N(m,576*128)),y.backbone24Skip=n(await N(Te,576*128)),s=t.createCommandEncoder();{let v=ee(new Uint32Array([1,128,24,24,24,24])),F=t.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Te}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:v}}]}),E=s.beginComputePass();E.setPipeline(q),E.setBindGroup(0,F),E.dispatchWorkgroups(I(24,8),I(24,8),128),E.end()}t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpnAdd24=n(await N(m,576*128)),s=t.createCommandEncoder(),ie(s,Ot,m,U,m),t.queue.submit([s.finish()]);{let v=m;m=U,U=v}y.fpn24Block1=n(await N(m,576*128)),s=t.createCommandEncoder(),ie(s,It,m,U,m),t.queue.submit([s.finish()]);{let v=m;m=U,U=v}return y.fpn24Block2=n(await N(m,576*128)),s=t.createCommandEncoder(),Se(s,m,bt,Et,at,128,2,24,24),t.queue.submit([s.finish()]),y.cls8=n(await N(at,576*2)),s=t.createCommandEncoder(),Se(s,m,Wt,Ue,Z,128,36,24,24),t.queue.submit([s.finish()]),y.reg8=n(await N(Z,576*36)),y.initWeights=n(await N(ge,100),100),y.initBias=n(await N(ke,32),32),y.cls16Weights=n(await N(ft,100),100),y.cls16Bias=n(await N(ht,6),6),y.cls8Weights=n(await N(bt,100),100),y.cls8Bias=n(await N(Et,2),2),y.fpn6to12Weights=n(await N(ve,100),100),y}return{device:t,run:Nt,debugRun:qt}}function Xn(){let i=[];for(let _=0;_<12;_++)for(let t=0;t<12;t++){let k=(t+.5)/12,w=(_+.5)/12;for(let e=0;e<6;e++)i.push({x:k,y:w})}for(let _=0;_<24;_++)for(let t=0;t<24;t++){let k=(t+.5)/24,w=(_+.5)/24;for(let e=0;e<2;e++)i.push({x:k,y:w})}return i}var fn=Xn();function Vn(i){return 1/(1+Math.exp(-i))}function hn(i,_){let t=[],{scores:k,regressors:w}=i,e=192;for(let x=0;x<fn.length;x++){let T=Vn(k[x]);if(T<_)continue;let M=fn[x],l=x*18,b=M.x+w[l+0]/e,A=M.y+w[l+1]/e,L=w[l+2]/e,re=w[l+3]/e,g=[];for(let j=0;j<7;j++){let H=M.x+w[l+4+j*2]/e,ce=M.y+w[l+4+j*2+1]/e;g.push([H,ce])}t.push({score:T,box:[b,A,L,re],keypoints:g})}return t}function wn(i,_){if(i.length===0)return[];let t=[...i].sort((e,x)=>x.score-e.score),k=[],w=new Set;for(let e=0;e<t.length;e++)if(!w.has(e)){k.push(t[e]);for(let x=e+1;x<t.length;x++)w.has(x)||Zn(t[e],t[x])>_&&w.add(x)}return k}function Zn(i,_){let t=i.box[0]-i.box[2]/2,k=i.box[1]-i.box[3]/2,w=i.box[0]+i.box[2]/2,e=i.box[1]+i.box[3]/2,x=_.box[0]-_.box[2]/2,T=_.box[1]-_.box[3]/2,M=_.box[0]+_.box[2]/2,l=_.box[1]+_.box[3]/2,b=Math.max(t,x),A=Math.max(k,T),L=Math.min(w,M),re=Math.min(e,l),g=Math.max(0,L-b),j=Math.max(0,re-A),H=g*j,ce=(w-t)*(e-k),he=(M-x)*(l-T),Be=ce+he-H;return Be>0?H/Be:0}function jn(i){let[_,t,k,w]=i.box,e=i.keypoints[0],x=i.keypoints[2],T=x[0]-e[0],M=x[1]-e[1],l=Math.atan2(M,T),A=-Math.PI/2-l,L=Math.max(k,w),g=L*2.6,j=-.5*L,H=Math.cos(A),ce=Math.sin(A),he=-j*ce,Be=j*H;return{centerX:_+he,centerY:t+Be,width:g,height:g,rotation:A}}function bn(i,_={}){let{scoreThreshold:t=.5,nmsThreshold:k=.3,maxHands:w=2}=_;async function e(T){let M=await i.run(T),l=hn(M,t);return wn(l,k).slice(0,w).map(jn)}async function x(T){let M=await i.run(T),l=hn(M,t);return wn(l,k).slice(0,w)}return{detect:e,detectRaw:x,model:i}}function Jn(i,_=256){let t=Math.cos(i.rotation),k=Math.sin(i.rotation),w=i.width/_,e=i.height/_,x=w*t,T=-e*k,M=w*k,l=e*t,b=i.centerX-(x*_/2+T*_/2),A=i.centerY-(M*_/2+l*_/2),L=x*l-T*M,re=l/L,g=-T/L,j=-M/L,H=x/L,ce=-(re*b+g*A),he=-(j*b+H*A);return{forward:[x,T,b,M,l,A],inverse:[re,g,ce,j,H,he]}}function gn(i,_){let{forward:t}=Jn(_,1),[k,w,e,x,T,M]=t;return i.map(l=>({x:k*l.x+w*l.y+e,y:x*l.x+T*l.y+M,z:l.z}))}var wa=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function ba(i){let _={};for(let t=0;t<wa.length;t++)_[wa[t]]=i[t];return _}function yn(i,_,t){return i.initialized?(i.value=t*_+(1-t)*i.value,i.value):(i.value=_,i.initialized=!0,_)}function xn(i,_){let t=2*Math.PI*_*i;return t/(t+1)}function Qn(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function ga(i,_,t,k,w,e){let x=i.lastTime<0?.03333333333333333:t-i.lastTime;i.lastTime=t;let T=xn(x,e),M=i.x.initialized?(_-i.x.value)/x:0,l=yn(i.dx,M,T),b=k+w*Math.abs(l),A=xn(x,b);return yn(i.x,_,A)}function ya(i={}){let{minCutoff:_=1,beta:t=10,dCutoff:k=1}=i,w=[];function e(M){w.length!==M&&(w=Array.from({length:M},()=>Qn()))}function x(M,l){let b=l??performance.now()/1e3,A=M.length*3;return e(A),M.map((L,re)=>({x:ga(w[re*3],L.x,b,_,t,k),y:ga(w[re*3+1],L.y,b,_,t,k),z:ga(w[re*3+2],L.z,b,_,t,k)}))}function T(){w=[]}return{apply:x,reset:T}}var ei="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ti(i={}){let{weightsUrl:_,scoreThreshold:t=.5,palmScoreThreshold:k=.5,maxHands:w=3,forceF32:e=!1}=i;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let x=(_??ei).replace(/\/$/,"")+"/",[T,M,l,b]=await Promise.all([fetch(`${x}weights_f16.json`),fetch(`${x}weights_f16.bin`),fetch(`${x}palm_detection_weights.json`),fetch(`${x}palm_detection_weights.bin`)]);if(!T.ok)throw new Error(`Failed to fetch landmark weights: ${T.status}`);if(!M.ok)throw new Error(`Failed to fetch landmark weights: ${M.status}`);if(!l.ok)throw new Error(`Failed to fetch palm detection weights: ${l.status}`);if(!b.ok)throw new Error(`Failed to fetch palm detection weights: ${b.status}`);let[A,L,re,g]=await Promise.all([T.json(),M.arrayBuffer(),l.json(),b.arrayBuffer()]),j=fa(A,L),H=fa(re,g),ce=await ha(j,{forceF32:e});if(!e){let P=new OffscreenCanvas(256,256),Y=P.getContext("2d");Y.fillStyle="#886644",Y.fillRect(0,0,256,256),Y.fillStyle="#cc9966",Y.fillRect(50,50,156,156);let _e=await ce.runFromCanvas(P);_e.landmarks.every(se=>se===0)&&_e.handflag.every(se=>se===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),ce.device.destroy(),ce=await ha(j,{forceF32:!0}))}let he=await mn(H),Be=bn(he,{scoreThreshold:k,maxHands:w}),ne=[];for(let P=0;P<w;P++)ne.push(ya());let S=0,V=null,pe=null;function Ee(){return V||(V=new OffscreenCanvas(192,192)),V}function pt(){return pe||(pe=new OffscreenCanvas(256,256)),pe}async function ct(P){if(P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas){if(P.width===192&&P.height===192)return P;let X=Ee();return X.width=192,X.height=192,X.getContext("2d").drawImage(P,0,0,192,192),X}if(typeof ImageBitmap<"u"&&P instanceof ImageBitmap){if(P.width===192&&P.height===192)return P;let X=Ee();return X.width=192,X.height=192,X.getContext("2d").drawImage(P,0,0,192,192),X}let Y=Ee();Y.width=192,Y.height=192;let _e=Y.getContext("2d");if(P instanceof ImageData){let X=new OffscreenCanvas(P.width,P.height);X.getContext("2d").putImageData(P,0,0),_e.drawImage(X,0,0,192,192)}else _e.drawImage(P,0,0,192,192);return Y}function dt(P,Y,_e,X){let se=pt();se.width=256,se.height=256;let Q=se.getContext("2d");Q.clearRect(0,0,256,256),Q.save(),Q.translate(128,128),Q.scale(Y.width*_e/256,Y.height*X/256),Q.rotate(-Y.rotation),Q.translate(-128,-128);let q=Y.centerX*_e,We=Y.centerY*X;Q.restore();let be=Math.min(_e,X),Ke=256/(Y.width*be),De=256/(Y.height*be),ge=Math.cos(Y.rotation),ke=Math.sin(Y.rotation),Ce=ge*Ke,Ze=ke*Ke,xe=-ke*De,ye=ge*De,ve=-q*Ce-We*xe+128,le=-q*Ze-We*ye+128;if(Q.setTransform(Ce,Ze,xe,ye,ve,le),P instanceof ImageData){let Ne=new OffscreenCanvas(P.width,P.height);Ne.getContext("2d").putImageData(P,0,0),Q.drawImage(Ne,0,0)}else Q.drawImage(P,0,0);return Q.setTransform(1,0,0,1,0,0),se}function _t(P){return P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas?[P.width,P.height]:typeof ImageBitmap<"u"&&P instanceof ImageBitmap?[P.width,P.height]:P instanceof ImageData?[P.width,P.height]:P instanceof HTMLVideoElement?[P.videoWidth,P.videoHeight]:P instanceof HTMLImageElement?[P.naturalWidth,P.naturalHeight]:[256,256]}async function we(P){let Y=await ct(P),_e=await Be.detect(Y);if(_e.length===0){if(S>0)for(let q=0;q<S&&q<ne.length;q++)ne[q].reset();return S=0,[]}let[X,se]=_t(P),Q=[];for(let q of _e){let We=dt(P,q,X,se),be=await ce.runFromCanvas(We),Ke=be.handflag[0];if(Ke<t)continue;let De=be.handedness[0]>.5,ge=[];for(let le=0;le<21;le++)ge.push({x:be.landmarks[le*3],y:be.landmarks[le*3+1],z:be.landmarks[le*3+2]});let ke=Math.min(X,se),Ce=q.width*ke,Ze={...q,width:Ce/X,height:Ce/se},xe=gn(ge,Ze),ye=Q.length,ve=ye<ne.length?ne[ye].apply(xe):xe;Q.push({score:Ke,handedness:De?"right":"left",landmarks:ve,keypoints:ba(ve)})}if(Q.length<S)for(let q=Q.length;q<S;q++)q<ne.length&&ne[q].reset();return S=Q.length,Q}function Ve(){ce.device.destroy(),he.device.destroy(),V=null,pe=null}return{detect:we,dispose:Ve}}export{wa as LANDMARK_NAMES,ti as createHandpose,ya as createLandmarkSmoother,ba as toKeypoints};
