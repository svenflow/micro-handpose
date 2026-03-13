function ce(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ta(n){let l=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],o="enable f16;"+n;for(let y of l)for(;o.includes(`${y}:array<f32>`);)o=o.replace(`${y}:array<f32>`,`${y}:array<f16>`);return o}var la=ce(`
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
`),_a=ce(`
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
`),ma=ce(`
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
`),fa=ce(`
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
`);function La(n,l){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Ra(n,l){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Oa(n,l){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Ia(n,l){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Fa(n,l){return[8,8]}var za=ce(`
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
`),Na=ce(`
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
`);function Ka(n){return ce(`
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
`)}var qa=Ka(!1),$a=Ka(!0),Ya=ce(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Xa=ce(`
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
`);function Va(n){return ce(`
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
`)}var ja=Va("sigmoid"),Za=Va("div256"),Ja=ce(`
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
`),Qa=ce(`
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
`);function en(n,l){let y=Math.min(l,256),h=l>y,g=n%4===0?`var ic:u32=0u;
    while(ic<${n}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${n}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,A=`var skip_val:f32=0.0;
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
    }`,S=n===l?"":`if(c<${n}u){`,d=n===l?"":"}",f=h?`for(var c:u32=lid.x;c<${n}u;c+=${y}u){`:`let c=lid.x;
  ${S}`,v=h?"}":d,U=h?`for(var c:u32=lid.x;c<${l}u;c+=${y}u){`:"{let c=lid.x;";return ce(`
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
  ${f}
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
    ${A}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var tn=ce(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),an=ce(`
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
`),nn=ce(`
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
`);function Ut(n,l){let o=new Map,y=n.dtype??"float32";for(let h=0;h<n.keys.length;h++){let e=n.keys[h],g=n.shapes[h],A=n.offsets[h],S=g.reduce((v,U)=>v*U,1),d,f;if(y==="float32")d=new Float32Array(l,A,S);else{let v=new DataView(l);d=new Float32Array(S);for(let U=0;U<S;U++)d[U]=Rn(v.getUint16(A+U*2,!0));f=l.slice(A,A+S*2)}o.set(e,{data:d,shape:g,rawF16:f})}return o}function Rn(n){let l=n>>15&1,o=n>>10&31,y=n&1023;if(o===0){if(y===0)return l?-0:0;let g=-14,A=y/1024;return(l?-1:1)*Math.pow(2,g)*A}if(o===31)return y===0?l?-1/0:1/0:NaN;let h=o-15,e=1+y/1024;return(l?-1:1)*Math.pow(2,h)*e}var On=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],rn=On.map(([n,l,o,y,h])=>({type:"resmodule",inCh:n,outCh:l,h:o,w:o,stride:y,prefix:h})),In=2,Fn=5,zn=8,Nn=11;async function Zt(n,l){if(!navigator.gpu)throw new Error("WebGPU not supported");let o=await navigator.gpu.requestAdapter();if(!o)throw new Error("No GPU adapter found");let y=o.features.has("shader-f16"),h=y?["shader-f16"]:[],e=await o.requestDevice({requiredFeatures:h,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(o.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(o.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(o.limits.maxComputeInvocationsPerWorkgroup,288)}}),g=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(y)try{let a=`enable f16;
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
}`,u=`enable f16;
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
}`,r=e.createShaderModule({code:a}),i=e.createShaderModule({code:u}),t=await r.getCompilationInfo(),M=await i.getCompilationInfo();if(t.messages.some(G=>G.type==="error")||M.messages.some(G=>G.type==="error"))g=!1;else{let G=new Float32Array(2400);G.fill(1);let E=new Uint16Array(2400);E.fill(10516);let x=new Uint16Array(96);x.fill(14336);let m=new Uint16Array(9216);m.fill(8478);let s=new Uint16Array(96);s.fill(12288);let L=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,J=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ae=e.createBuffer({size:G.byteLength,usage:L}),_t=e.createBuffer({size:E.byteLength,usage:L}),mt=e.createBuffer({size:x.byteLength,usage:L}),ft=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),ht=e.createBuffer({size:m.byteLength,usage:L}),wt=e.createBuffer({size:s.byteLength,usage:L}),gt=e.createBuffer({size:384,usage:J}),Xe=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ae,0,G),e.queue.writeBuffer(_t,0,E),e.queue.writeBuffer(mt,0,x),e.queue.writeBuffer(ht,0,m),e.queue.writeBuffer(wt,0,s);let Ie="read-only-storage",Bt="storage",Ct=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ie}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ie}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ie}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Bt}}]}),Wa=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ie}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ie}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ie}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Bt}}]}),Mn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Ct]}),compute:{module:r,entryPoint:"main"}}),En=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Wa]}),compute:{module:i,entryPoint:"main"}}),Wn=e.createBindGroup({layout:Ct,entries:[{binding:0,resource:{buffer:ae}},{binding:1,resource:{buffer:_t}},{binding:2,resource:{buffer:mt}},{binding:3,resource:{buffer:ft}}]}),Hn=e.createBindGroup({layout:Wa,entries:[{binding:0,resource:{buffer:ft}},{binding:1,resource:{buffer:ht}},{binding:2,resource:{buffer:wt}},{binding:3,resource:{buffer:gt}}]}),Xt=e.createCommandEncoder(),Vt=Xt.beginComputePass();Vt.setPipeline(Mn),Vt.setBindGroup(0,Wn),Vt.dispatchWorkgroups(2),Vt.end();let jt=Xt.beginComputePass();jt.setPipeline(En),jt.setBindGroup(0,Hn),jt.dispatchWorkgroups(2),jt.end(),Xt.copyBufferToBuffer(gt,0,Xe,0,384),e.queue.submit([Xt.finish()]),await e.queue.onSubmittedWorkDone(),await Xe.mapAsync(GPUMapMode.READ);let kt=new Float32Array(Xe.getMappedRange()),Ha=1.5*.0104*96+.25,Tn=kt[0]!==0&&kt[47]!==0&&kt[95]!==0,Ln=Math.abs(kt[0]-Ha)<1;g=Tn&&Ln,Xe.unmap(),ae.destroy(),_t.destroy(),mt.destroy(),ft.destroy(),ht.destroy(),wt.destroy(),gt.destroy(),Xe.destroy(),g||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${kt[0]}, expected ~${Ha.toFixed(2)}) \u2014 falling back to f32`)}}catch{g=!1}let S=n.values().next().value,d=g&&!!S?.rawF16&&!l?.forceF32;console.log(d?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${y}, f16 validated: ${g}, f16 data: ${!!S?.rawF16})`);function f(a){if(d&&a.rawF16){let u=new Uint16Array(a.rawF16);if(u.length%2!==0){let r=new Uint16Array(u.length+1);return r.set(u),r}return u}return a.data}function v(a){if(d&&a.rawF16){let u=a.rawF16.byteLength;return Math.ceil(u/4)*4}return a.data.byteLength}function U(a){return d?Ta(a):a}let T={r:"read-only-storage",s:"storage",u:"uniform"};function _(a){return e.createBindGroupLayout({entries:a.map((u,r)=>({binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[u]}}))})}function I(a){return e.createBindGroupLayout({entries:a.map((u,r)=>u==="t"?{binding:r,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[u]}})})}let B=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,re=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,ne=GPUBufferUsage.STORAGE,xe=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,se=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function D(a,u){return e.createBuffer({size:a,usage:u})}function N(a,u){return e.createBindGroup({layout:a,entries:u.map((r,i)=>({binding:i,resource:"size"in r?{buffer:r}:r}))})}function Q(a,u){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:u,entryPoint:"main"}})}let W=e.createShaderModule({code:za}),O=e.createShaderModule({code:nn}),F=e.createShaderModule({code:U(Ja)}),Ge=e.createShaderModule({code:U(_a)}),we=e.createShaderModule({code:U(la)}),Ee=e.createShaderModule({code:U(ma)}),_e=e.createShaderModule({code:U(fa)}),ve=e.createShaderModule({code:U(Na)}),P=e.createShaderModule({code:qa}),X=e.createShaderModule({code:Ya}),me=e.createShaderModule({code:$a}),V=e.createShaderModule({code:U(Xa)}),fe=e.createShaderModule({code:U(ja)}),ie=e.createShaderModule({code:U(Za)}),Fe=e.createShaderModule({code:U(Qa)}),Ve=new Map;function Pe(a,u){let r=`${a}_${u}`,i=Ve.get(r);return i||(i=e.createShaderModule({code:U(en(a,u))}),Ve.set(r,i)),i}let Be=_(["r","r","r","s","u"]),ge=_(["r","r","r","r","s","u"]),be=_(["r","s","u"]),Ce=_(["r","r","r","s","u"]),ke=_(["r","s","u"]),je=_(["r","r","s","u"]),Ae=_(["r","r","s","u"]),ze=_(["r","r","r","s","u"]),Se=_(["r","r","r","s","u"]),it=I(["t","s","u"]),rt=_(["r","r","r","r","r","r","r","s"]),Ne=_(["r","r","r","r","r","s","u"]),bt=e.createPipelineLayout({bindGroupLayouts:[Be]}),Gt=e.createPipelineLayout({bindGroupLayouts:[ge]}),st=a=>e.createComputePipeline({layout:bt,compute:{module:a,entryPoint:"main"}}),ot=a=>e.createComputePipeline({layout:Gt,compute:{module:a,entryPoint:"main"}}),Jt=st(Ge),Qt=st(we),At=ot(Ee),St=ot(_e),Dt=new Map,yt=new Map,xt=new Map,Mt=new Map;Dt.set("8,8",Jt),yt.set("8,8",Qt),xt.set("8,8",At),Mt.set("8,8",St);function Ze(a,u,r,i,t){let M=`${u},${r}`,G=a.get(M);return G||(G=t(e.createShaderModule({code:U(i(u,r))})),a.set(M,G)),G}let Et=(a,u)=>Ze(Dt,a,u,La,st),ea=(a,u)=>Ze(yt,a,u,Ra,st),Wt=(a,u)=>Ze(xt,a,u,Oa,ot),Ht=(a,u)=>Ze(Mt,a,u,Ia,ot),De=rn.map(a=>{let u=a.stride===2?a.h/2:a.h,r=a.stride===2?a.w/2:a.w,[i,t]=Fa(a.inCh,u),M=a.h>=64,G=u>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:M?ea(i,t):Et(i,t),pwPipeline:G?Ht(i,t):Wt(i,t),dwDispatchX:Math.ceil(r/i),dwDispatchY:Math.ceil(u/t),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(r/i),pwDispatchY:Math.ceil(u/t),pwDispatchZ:G?a.outCh/2:a.outCh}}),ta=Q(be,W),ut=Q(Ce,ve);Q(ke,P),Q(je,X);let We=Q(Ae,me),pt=Q(ze,V);Q(Se,fe),Q(Se,ie);let Me=Q(it,O),Tt=Q(rt,F),Lt=Q(Ne,Fe),Rt=1*288*128*128*4,vt=D(3*256*256*4,B),Ke=D(3*257*257*4,ne),Pt=D(12,se);e.queue.writeBuffer(Pt,0,new Uint32Array([3,256,257]));let j=D(Rt,re),oe=D(Rt,xe),Ue=D(Rt,ne),Je=D(3072*64*4,B),Qe=D(3072*32*4,B),et=D(1536*16*4,B),Z=D(6144*64*4,ne),de=D(260,xe),K=D(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);D(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let le=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),tt=D(8,se);e.queue.writeBuffer(tt,0,new Uint32Array([256,257]));let Ot=n.get("backbone1.1.weight"),It=n.get("backbone1.1.bias");if(!Ot||!It)throw new Error("Missing input conv weights");let qe=f(Ot),Ft=f(It),c=D(qe.byteLength,B),p=D(Ft.byteLength,B),H=D(28,se);e.queue.writeBuffer(c,0,qe),e.queue.writeBuffer(p,0,Ft),e.queue.writeBuffer(H,0,new Uint32Array([1,3,24,257,257,128,128]));let C=n.get("backbone6.1.weight"),w=n.get("backbone6.1.bias");if(!C||!w)throw new Error("Missing backbone6.1 conv1x1 weights");let b=f(C),z=f(w),q=D(b.byteLength,B),ue=D(z.byteLength,B),ee=D(20,se);e.queue.writeBuffer(q,0,b),e.queue.writeBuffer(ue,0,z),e.queue.writeBuffer(ee,0,new Uint32Array([1,96,48,32,32]));let $=n.get("handflag.weight"),te=n.get("handflag.bias");if(!$||!te)throw new Error("Missing handflag weights");let pe=f($),he=f(te),k=D(pe.byteLength,B),R=D(he.byteLength,B),Y=D(12,se);e.queue.writeBuffer(k,0,pe),e.queue.writeBuffer(R,0,he),e.queue.writeBuffer(Y,0,new Uint32Array([1,288,1]));let ye=n.get("handedness.weight"),ct=n.get("handedness.bias");if(!ye||!ct)throw new Error("Missing handedness weights");let ba=f(ye),ya=f(ct),aa=D(ba.byteLength,B),na=D(ya.byteLength,B),xa=D(12,se);e.queue.writeBuffer(aa,0,ba),e.queue.writeBuffer(na,0,ya),e.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let va=n.get("reg_3d.weight"),Pa=n.get("reg_3d.bias");if(!va||!Pa)throw new Error("Missing reg_3d weights");let Ba=f(va),Ca=f(Pa),ia=D(Ba.byteLength,B),ra=D(Ca.byteLength,B),ka=D(12,se);e.queue.writeBuffer(ia,0,Ba),e.queue.writeBuffer(ra,0,Ca),e.queue.writeBuffer(ka,0,new Uint32Array([1,288,63]));let at=rn.map(a=>{let{inCh:u,outCh:r,h:i,w:t,stride:M,prefix:G}=a,E=M===2?i/2:i,x=M===2?t/2:t,m=M===2?1:2,s=n.get(`${G}convs.0.weight`),L=n.get(`${G}convs.0.bias`),J=n.get(`${G}convs.1.weight`),ae=n.get(`${G}convs.1.bias`);if(!s||!L||!J||!ae)throw new Error(`Missing weights for ${G}`);let _t=f(s),mt=f(L),ft=f(J),ht=f(ae),wt=D(_t.byteLength,B),gt=D(mt.byteLength,B),Xe=D(ft.byteLength,B),Ie=D(ht.byteLength,B),Bt=D(32,se),Ct=D(36,se);return e.queue.writeBuffer(wt,0,_t),e.queue.writeBuffer(gt,0,mt),e.queue.writeBuffer(Xe,0,ft),e.queue.writeBuffer(Ie,0,ht),e.queue.writeBuffer(Bt,0,new Uint32Array([1,u,i,t,E,x,M,m])),e.queue.writeBuffer(Ct,0,new Uint32Array([1,u,r,E,x,Math.max(0,r-u),M,i,t])),{dwWeight:wt,dwBias:gt,pwWeight:Xe,pwBias:Ie,dwUniform:Bt,pwUniform:Ct,spec:a,outH:E,outW:x}});function dt(a){let u=D(a.length*4,se);return e.queue.writeBuffer(u,0,new Uint32Array(a)),u}let gn=dt([1,96,8,8,16,16]),bn=dt([1,96,16,16,32,32]),yn=dt([1,48,32,32,64,64]);dt([1536*16]),dt([3072*32]),dt([3072*64]);let Ua=N(be,[vt,Ke,Pt]),Ga=N(Ce,[Ke,c,p,j,H]),He=[],Te=[],Le=[],Re=[];for(let a of at)He.push(N(Be,[j,a.dwWeight,a.dwBias,Ue,a.dwUniform])),Te.push(N(ge,[Ue,j,a.pwWeight,a.pwBias,oe,a.pwUniform])),Le.push(N(Be,[oe,a.dwWeight,a.dwBias,Ue,a.dwUniform])),Re.push(N(ge,[Ue,oe,a.pwWeight,a.pwBias,j,a.pwUniform]));let xn=N(Ae,[j,et,oe,gn]),vn=N(Ae,[j,Qe,oe,bn]),Pn=N(ze,[j,q,ue,Z,ee]),Bn=N(Ae,[Z,Je,oe,yn]);N(Se,[j,k,R,de,Y]),N(Se,[j,aa,na,de,xa]),N(Se,[j,ia,ra,de,ka]);let $e=N(it,[le.createView(),Ke,tt]),Cn=N(rt,[j,k,R,aa,na,ia,ra,de]),sa=24,Aa=[],Sa=[];for(let a=sa;a<at.length;a++){let u=at[a];Aa.push(N(Ne,[j,u.dwWeight,u.dwBias,u.pwWeight,u.pwBias,oe,u.dwUniform])),Sa.push(N(Ne,[oe,u.dwWeight,u.dwBias,u.pwWeight,u.pwBias,j,u.dwUniform]))}let oa=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});oa.globalCompositeOperation="copy";let Da=new OffscreenCanvas(9,8),zt=Da.getContext("webgpu"),Nt=null,ua=null;if(zt){zt.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let a=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),u=e.createShaderModule({code:tn}),r=e.createShaderModule({code:an});Nt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:u,entryPoint:"vs"},fragment:{module:r,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),ua=e.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:de}}]})}let Kt=new Float32Array(1),qt=new Float32Array(1),$t=new Float32Array(63);function Oe(a,u){let r=!0,i=0,t=a.beginComputePass();for(t.setPipeline(ut),t.setBindGroup(0,Ga),t.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=In;i++){let x=r?He[i]:Le[i],m=r?Te[i]:Re[i],s=De[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.end();let M=r?j:oe;for(a.copyBufferToBuffer(M,0,Je,0,3072*64*4),t=a.beginComputePass();i<=Fn;i++){let x=r?He[i]:Le[i],m=r?Te[i]:Re[i],s=De[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.end();let G=r?j:oe;for(a.copyBufferToBuffer(G,0,Qe,0,3072*32*4),t=a.beginComputePass();i<=zn;i++){let x=r?He[i]:Le[i],m=r?Te[i]:Re[i],s=De[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.end();let E=r?j:oe;for(a.copyBufferToBuffer(E,0,et,0,1536*16*4),t=a.beginComputePass();i<=Nn;i++){let x=r?He[i]:Le[i],m=r?Te[i]:Re[i],s=De[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.setPipeline(We),t.setBindGroup(0,xn),t.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),t.end(),r=!1,t=a.beginComputePass();{let x=r?He[i]:Le[i],m=r?Te[i]:Re[i],s=De[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r,i++}t.setPipeline(We),t.setBindGroup(0,vn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),t.end(),r=!1,t=a.beginComputePass();{let x=r?He[i]:Le[i],m=r?Te[i]:Re[i],s=De[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r,i++}for(t.setPipeline(pt),t.setBindGroup(0,Pn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),t.setPipeline(We),t.setBindGroup(0,Bn),t.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),t.end(),r=!1,t=a.beginComputePass();i<sa;i++){let x=r?He[i]:Le[i],m=r?Te[i]:Re[i],s=De[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,x),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}for(;i<at.length;i++){let x=i-sa,m=r?Aa[x]:Sa[x],s=at[i];t.setPipeline(Lt),t.setBindGroup(0,m),t.dispatchWorkgroups(s.outW,s.outH,1),r=!r}t.setPipeline(Tt),t.setBindGroup(0,Cn),t.dispatchWorkgroups(1),t.end(),u&&a.copyBufferToBuffer(de,0,u,0,260)}async function Yt(a){e.queue.writeBuffer(vt,0,a);let u=e.createCommandEncoder();{let t=u.beginComputePass();t.setPipeline(ta),t.setBindGroup(0,Ua),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}Oe(u,K),e.queue.submit([u.finish()]);let r=K.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(K.getMappedRange());return Kt[0]=i[0],qt[0]=i[1],$t.set(i.subarray(2,65)),K.unmap(),{handflag:new Float32Array(Kt),handedness:new Float32Array(qt),landmarks:new Float32Array($t)}}async function pa(a){e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);let u=e.createCommandEncoder();{let t=u.beginComputePass();t.setPipeline(Me),t.setBindGroup(0,$e),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Oe(u,K),e.queue.submit([u.finish()]);let r=K.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(K.getMappedRange());return Kt[0]=i[0],qt[0]=i[1],$t.set(i.subarray(2,65)),K.unmap(),{handflag:new Float32Array(Kt),handedness:new Float32Array(qt),landmarks:new Float32Array($t)}}async function Ma(a){if(!Nt||!ua||!zt)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);let u=e.createCommandEncoder();{let x=u.beginComputePass();x.setPipeline(Me),x.setBindGroup(0,$e),x.dispatchWorkgroups(16,16,1),x.end()}Oe(u,null);let r=zt.getCurrentTexture(),i=u.beginRenderPass({colorAttachments:[{view:r.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});i.setPipeline(Nt),i.setBindGroup(0,ua),i.draw(3),i.end(),e.queue.submit([u.finish()]),await e.queue.onSubmittedWorkDone(),oa.drawImage(Da,0,0);let M=oa.getImageData(0,0,9,8).data,G=new Float32Array(65),E=new DataView(new ArrayBuffer(4));for(let x=0;x<65;x++){let m=x*4;E.setUint8(0,M[m]),E.setUint8(1,M[m+1]),E.setUint8(2,M[m+2]),E.setUint8(3,M[m+3]),G[x]=E.getFloat32(0)}return{handflag:new Float32Array([G[0]]),handedness:new Float32Array([G[1]]),landmarks:new Float32Array(G.subarray(2,65))}}let kn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),ca=0,Un=[K,kn],lt=null,Ye=null;async function da(a){let u=Un[ca];ca=1-ca,e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);let r=e.createCommandEncoder();{let t=r.beginComputePass();t.setPipeline(Me),t.setBindGroup(0,$e),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Oe(r,u),e.queue.submit([r.finish()]);let i=null;if(lt!==null&&Ye!==null){await lt;let t=new Float32Array(Ye.getMappedRange());i={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))},Ye.unmap()}return Ye=u,lt=u.mapAsync(GPUMapMode.READ),i}async function Ea(){if(!lt||!Ye)return null;await lt;let a=new Float32Array(Ye.getMappedRange()),u={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return Ye.unmap(),lt=null,Ye=null,u}async function Gn(a=50){let u=new Float32Array(196608);for(let t=0;t<5;t++)await Yt(u);let r=[];for(let t=0;t<a;t++){let M=performance.now();await Yt(u),r.push(performance.now()-M)}let i=r.reduce((t,M)=>t+M,0)/r.length;return{avgMs:i,fps:1e3/i}}async function An(a=50){let u=new Float32Array(196608);for(let G=0;G<5;G++)await Yt(u);let r=[];for(let G=0;G<a;G++){let E=e.createCommandEncoder();{let m=E.beginComputePass();m.setPipeline(ta),m.setBindGroup(0,Ua),m.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),m.end()}Oe(E,K);let x=performance.now();e.queue.submit([E.finish()]),await e.queue.onSubmittedWorkDone(),r.push(performance.now()-x)}r.sort((G,E)=>G-E);let i=r.reduce((G,E)=>G+E,0)/r.length,t=r[Math.floor(r.length/2)],M=r[0];return{avgMs:i,fps:1e3/i,medianMs:t,minMs:M}}function ei(a){e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);let u=e.createCommandEncoder();{let r=u.beginComputePass();r.setPipeline(Me),r.setBindGroup(0,$e),r.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),r.end()}Oe(u,K),e.queue.submit([u.finish()])}async function Sn(a,u=50){function r(m){let s=[...m].sort((L,J)=>L-J);return{median:s[Math.floor(s.length/2)],min:s[0]}}for(let m=0;m<10;m++)await pa(a);let i=[];for(let m=0;m<u;m++){e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);let s=e.createCommandEncoder();{let J=s.beginComputePass();J.setPipeline(Me),J.setBindGroup(0,$e),J.dispatchWorkgroups(16,16,1),J.end()}Oe(s,K);let L=performance.now();e.queue.submit([s.finish()]),await e.queue.onSubmittedWorkDone(),i.push(performance.now()-L)}let t=[];for(let m=0;m<u;m++){e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);let s=e.createCommandEncoder();{let ae=s.beginComputePass();ae.setPipeline(Me),ae.setBindGroup(0,$e),ae.dispatchWorkgroups(16,16,1),ae.end()}Oe(s,K),e.queue.submit([s.finish()]);let L=K.mapAsync(GPUMapMode.READ),J=performance.now();await e.queue.onSubmittedWorkDone(),await L,K.getMappedRange(),K.unmap(),t.push(performance.now()-J)}let M=[];for(let m=0;m<u;m++){e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);let s=e.createCommandEncoder();{let J=s.beginComputePass();J.setPipeline(Me),J.setBindGroup(0,$e),J.dispatchWorkgroups(16,16,1),J.end()}Oe(s,K),e.queue.submit([s.finish()]);let L=performance.now();await K.mapAsync(GPUMapMode.READ),K.getMappedRange(),K.unmap(),M.push(performance.now()-L)}let G=[];for(let m=0;m<u;m++){let s=performance.now();await pa(a),G.push(performance.now()-s)}await da(a);let E=[];for(let m=0;m<u;m++){let s=performance.now();await da(a),E.push(performance.now()-s)}await Ea();let x=null;if(Nt){let m=[];for(let s=0;s<u;s++){let L=performance.now();await Ma(a),m.push(performance.now()-L)}x=r(m)}return{gpuOnly:r(i),mapAsyncOnly:r(t),mapAsyncNoWait:r(M),total:r(G),pipelined:r(E),renderReadback:x}}async function Dn(a){let u=[];async function r(t,M,G){let E=e.createBuffer({size:M,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),x=e.createCommandEncoder();x.copyBufferToBuffer(t,0,E,0,M),e.queue.submit([x.finish()]),await e.queue.onSubmittedWorkDone(),await E.mapAsync(GPUMapMode.READ);let m=new Float32Array(E.getMappedRange()),s=1/0,L=-1/0,J=0;for(let ae=0;ae<m.length;ae++)m[ae]<s&&(s=m[ae]),m[ae]>L&&(L=m[ae]),m[ae]!==0&&J++;E.unmap(),E.destroy(),u.push({layer:G,stats:{min:s,max:L,nonZero:J,total:m.length}})}e.queue.copyExternalImageToTexture({source:a},{texture:le},[256,256]);{let t=e.createCommandEncoder(),M=t.beginComputePass();M.setPipeline(Me),M.setBindGroup(0,$e),M.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),M.end(),e.queue.submit([t.finish()])}await r(Ke,Math.min(Ke.size,3*257*257*4),"canvas\u2192bufInput");{let t=e.createCommandEncoder(),M=t.beginComputePass();M.setPipeline(ut),M.setBindGroup(0,Ga),M.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),M.end(),e.queue.submit([t.finish()])}await r(j,Math.min(j.size,3072*128*4),"inputConv\u2192bufA");let i=!0;for(let t=0;t<Math.min(at.length,6);t++){let M=i?He[t]:Le[t],G=i?Te[t]:Re[t],E=De[t],x=at[t];{let s=e.createCommandEncoder(),L=s.beginComputePass();L.setPipeline(E.dwPipeline),L.setBindGroup(0,M),L.dispatchWorkgroups(E.dwDispatchX,E.dwDispatchY,E.dwDispatchZ),L.end(),e.queue.submit([s.finish()])}await r(Ue,Math.min(Ue.size,x.spec.inCh*x.outH*x.outW*4),`layer${t}.DW\u2192bufDW (${x.spec.prefix})`);{let s=e.createCommandEncoder(),L=s.beginComputePass();L.setPipeline(E.pwPipeline),L.setBindGroup(0,G),L.dispatchWorkgroups(E.pwDispatchX,E.pwDispatchY,E.pwDispatchZ),L.end(),e.queue.submit([s.finish()])}let m=i?oe:j;await r(m,Math.min(m.size,x.spec.outCh*x.outH*x.outW*4),`layer${t}.PW\u2192buf${i?"B":"A"} (${x.spec.prefix})`),i=!i}return u}return{device:e,run:Yt,runFromCanvas:pa,runFromCanvasViaRender:Ma,runFromCanvasPipelined:da,flushPipelined:Ea,benchmark:Gn,benchmarkGPU:An,benchmarkDiagnostic:Sn,debugLayerOutputs:Dn}}function nt(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var sn=nt(`
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
`),on=nt(`
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
`),un=nt(`
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
`),pn=nt(`
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
`),cn=nt(`
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
`),dn=nt(`
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
`),ln=nt(`
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
`);async function ha(n,l){let o;if(l)o=l;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let c=await navigator.gpu.requestAdapter();if(!c)throw new Error("No GPU adapter found");o=await c.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(c.limits.maxStorageBuffersPerShaderStage,8)}})}let y={r:"read-only-storage",s:"storage",u:"uniform"};function h(c){return o.createBindGroupLayout({entries:c.map((p,H)=>p==="t"?{binding:H,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:H,visibility:GPUShaderStage.COMPUTE,buffer:{type:y[p]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,g=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,S=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(c,p){return o.createBuffer({size:Math.max(c,4),usage:p})}function f(c,p,H){o.queue.writeBuffer(c,p,H)}function v(c){let p=d(c.data.byteLength,e);return f(p,0,c.data),p}let U=Array.from(n.keys());function T(c){let p=n.get(c);if(!p)throw new Error(`Weight not found: ${c}`);return p}function _(...c){let p=U.find(H=>c.every(C=>H.includes(C)));if(!p)throw new Error(`Weight not found for: ${c.join(", ")}`);return T(p)}function I(c){let[,p,H,C]=c.shape,w=new Float32Array(C*25);for(let b=0;b<C;b++)for(let z=0;z<p;z++)for(let q=0;q<H;q++)w[b*25+z*5+q]=c.data[z*H*C+q*C+b];return w}function B(c){let[p,,,H]=c.shape,C=new Float32Array(p*H);for(let w=0;w<p;w++)for(let b=0;b<H;b++)C[w*H+b]=c.data[w*H+b];return C}let re=o.createShaderModule({code:sn}),ne=o.createShaderModule({code:on}),xe=o.createShaderModule({code:un}),se=o.createShaderModule({code:pn}),D=o.createShaderModule({code:dn}),N=o.createShaderModule({code:cn}),Q=o.createShaderModule({code:ln}),W=h(["r","r","r","r","s","u"]),O=h(["r","r","r","s","u"]),F=h(["r","r","r","r","r","s","u"]),Ge=h(["r","r","r","s","u"]),we=h(["r","r","r","r","s","u"]),Ee=h(["r","r","s","u"]),_e=h(["t","s","u"]);function ve(c,p){return o.createComputePipeline({layout:o.createPipelineLayout({bindGroupLayouts:[c]}),compute:{module:p,entryPoint:"main"}})}let P=ve(W,re),X=ve(O,ne),me=ve(F,xe),V=ve(Ge,se),fe=ve(we,D),ie=ve(Ee,N),Fe=ve(_e,Q),Ve=_("conv2d/Conv2D"),Pe=_("batch_normalization/","conv2d/Conv2D"),Be=_("p_re_lu/"),ge=v(Ve),be=v(Pe),Ce=v(Be),je=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12}].map(c=>{let p=_(c.dwKey),H=_(c.pwKey),C=_(c.bnKey),w=_(c.preluKey),b=I(p),z=d(b.byteLength,e);f(z,0,b);let q=new Float32Array(c.inCh),ue=d(q.byteLength,e);f(ue,0,q);let ee=B(H),$=d(ee.byteLength,e);f($,0,ee);let te=v(C),pe=v(w);return{dwWeightBuf:z,dwBiasBuf:ue,pwWeightBuf:$,pwBiasBuf:te,alphaBuf:pe,inCh:c.inCh,outCh:c.outCh,stride:c.stride,inH:c.inH}}),Ae=B(_("conv2d_20/Conv2D")),ze=d(Ae.byteLength,e);f(ze,0,Ae);let Se=v(_("batch_normalization_20/")),it=v(_("p_re_lu_20/")),rt={dwWeightBuf:(()=>{let c=I(_("depthwise_conv2d_19/")),p=d(c.byteLength,e);return f(p,0,c),p})(),dwBiasBuf:(()=>{let c=new Float32Array(256),p=d(c.byteLength,e);return f(p,0,c),p})(),pwWeightBuf:ze,pwBiasBuf:Se,alphaBuf:it,inCh:256,outCh:256,stride:1,inH:12},Ne={dwWeightBuf:(()=>{let c=I(_("depthwise_conv2d_20/")),p=d(c.byteLength,e);return f(p,0,c),p})(),dwBiasBuf:(()=>{let c=new Float32Array(256),p=d(c.byteLength,e);return f(p,0,c),p})(),pwWeightBuf:(()=>{let c=B(_("conv2d_21/")),p=d(c.byteLength,e);return f(p,0,c),p})(),pwBiasBuf:v(_("batch_normalization_21/")),alphaBuf:v(_("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},bt=B(_("conv2d_23/Conv2D")),Gt=d(bt.byteLength,e);f(Gt,0,bt);let st=v(_("batch_normalization_23/")),ot=v(_("p_re_lu_23/")),Jt={dwWeightBuf:(()=>{let c=I(_("depthwise_conv2d_21/")),p=d(c.byteLength,e);return f(p,0,c),p})(),dwBiasBuf:(()=>{let c=new Float32Array(128),p=d(c.byteLength,e);return f(p,0,c),p})(),pwWeightBuf:(()=>{let c=B(_("conv2d_24/")),p=d(c.byteLength,e);return f(p,0,c),p})(),pwBiasBuf:v(_("batch_normalization_24/")),alphaBuf:v(_("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},Qt={dwWeightBuf:(()=>{let c=I(_("depthwise_conv2d_22/")),p=d(c.byteLength,e);return f(p,0,c),p})(),dwBiasBuf:(()=>{let c=new Float32Array(128),p=d(c.byteLength,e);return f(p,0,c),p})(),pwWeightBuf:(()=>{let c=B(_("conv2d_25/Conv2D1")),p=d(c.byteLength,e);return f(p,0,c),p})(),pwBiasBuf:v(_("batch_normalization_25/")),alphaBuf:v(_("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},At=B(_("classifier_palm_16_NO_PRUNING/Conv2D")),St=d(At.byteLength,e);f(St,0,At);let Dt=v(_("classifier_palm_16_NO_PRUNING/BiasAdd")),yt=B(_("regressor_palm_16_NO_PRUNING/Conv2D")),xt=d(yt.byteLength,e);f(xt,0,yt);let Mt=v(_("regressor_palm_16_NO_PRUNING/BiasAdd")),Ze=B(_("classifier_palm_8_NO_PRUNING/Conv2D")),Et=d(Ze.byteLength,e);f(Et,0,Ze);let ea=v(_("classifier_palm_8_NO_PRUNING/BiasAdd")),Wt=B(_("regressor_palm_8_NO_PRUNING/Conv2D")),Ht=d(Wt.byteLength,e);f(Ht,0,Wt);let De=v(_("regressor_palm_8_NO_PRUNING/BiasAdd")),ta=36864*3,ut=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,We=d(36864*3*4,e),pt=d(ut,g),Me=d(ut,g),Tt=d(ut,g),Lt=d(576*128*4,g),Rt=d(576*128*4,g),vt=d(864*4,A),Ke=d(15552*4,A),Pt=d(576*2*4,A),j=d(576*36*4,A),oe=d(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ue=d(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Je=d(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Qe=d(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),et=o.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function Z(c,p){return Math.ceil(c/p)}function de(c){let p=d(c.byteLength,S);return f(p,0,c),p}let K=de(new Uint32Array([1,3,32,192,192,96,96]));function le(c,p,H,C,w){let b=p.stride===2?p.inH/2:p.inH,z=b,q=p.stride===2?1:2,ue=de(new Uint32Array([1,p.inCh,p.inH,p.inH,b,z,p.stride,q])),ee=o.createBindGroup({layout:O,entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:p.dwWeightBuf}},{binding:2,resource:{buffer:p.dwBiasBuf}},{binding:3,resource:{buffer:Tt}},{binding:4,resource:{buffer:ue}}]}),$=c.beginComputePass();$.setPipeline(X),$.setBindGroup(0,ee),$.dispatchWorkgroups(Z(z,8),Z(b,8),p.inCh),$.end();let te=p.inCh,pe=de(new Uint32Array([1,p.inCh,p.outCh,b,z,te,p.stride,p.inH,p.inH])),he=o.createBindGroup({layout:F,entries:[{binding:0,resource:{buffer:Tt}},{binding:1,resource:{buffer:w}},{binding:2,resource:{buffer:p.pwWeightBuf}},{binding:3,resource:{buffer:p.pwBiasBuf}},{binding:4,resource:{buffer:p.alphaBuf}},{binding:5,resource:{buffer:C}},{binding:6,resource:{buffer:pe}}]}),k=c.beginComputePass();k.setPipeline(me),k.setBindGroup(0,he),k.dispatchWorkgroups(Z(z,8),Z(b,8),p.outCh),k.end()}function tt(c,p,H,C,w,b,z,q,ue){let ee=de(new Uint32Array([1,b,z,q,ue])),$=o.createBindGroup({layout:Ge,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:H}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:ee}}]}),te=c.beginComputePass();te.setPipeline(V),te.setBindGroup(0,$),te.dispatchWorkgroups(Z(ue,8),Z(q,8),z),te.end()}function Ot(c,p,H,C,w,b,z,q,ue,ee){let $=de(new Uint32Array([1,z,q,ue,ee])),te=o.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:H}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:b}},{binding:5,resource:{buffer:$}}]}),pe=c.beginComputePass();pe.setPipeline(fe),pe.setBindGroup(0,te),pe.dispatchWorkgroups(Z(ee,8),Z(ue,8),q),pe.end()}async function It(c){o.queue.copyExternalImageToTexture({source:c},{texture:et},[192,192]);let p=de(new Uint32Array([192,192,192])),H=o.createBindGroup({layout:_e,entries:[{binding:0,resource:et.createView()},{binding:1,resource:{buffer:We}},{binding:2,resource:{buffer:p}}]}),C=o.createCommandEncoder();{let k=C.beginComputePass();k.setPipeline(Fe),k.setBindGroup(0,H),k.dispatchWorkgroups(Z(192,16),Z(192,16),1),k.end()}{let k=o.createBindGroup({layout:W,entries:[{binding:0,resource:{buffer:We}},{binding:1,resource:{buffer:ge}},{binding:2,resource:{buffer:be}},{binding:3,resource:{buffer:Ce}},{binding:4,resource:{buffer:pt}},{binding:5,resource:{buffer:K}}]}),R=C.beginComputePass();R.setPipeline(P),R.setBindGroup(0,k),R.dispatchWorkgroups(Z(96,8),Z(96,8),32),R.end()}let w=pt,b=Me;for(let k=0;k<je.length;k++){let R=je[k];le(C,R,w,b,w);let Y=w;w=b,b=Y,k===10&&C.copyBufferToBuffer(w,0,Lt,0,576*128*4)}le(C,rt,w,b,w);{let k=w;w=b,b=k}le(C,Ne,w,b,w);{let k=w;w=b,b=k}tt(C,w,St,Dt,vt,256,6,12,12),tt(C,w,xt,Mt,Ke,256,108,12,12),Ot(C,w,Gt,st,ot,b,256,128,12,12);{let k=w;w=b,b=k}{let k=de(new Uint32Array([1,128,12,12,24,24])),R=o.createBindGroup({layout:Ee,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:Lt}},{binding:2,resource:{buffer:b}},{binding:3,resource:{buffer:k}}]}),Y=C.beginComputePass();Y.setPipeline(ie),Y.setBindGroup(0,R),Y.dispatchWorkgroups(Z(24,8),Z(24,8),128),Y.end()}{let k=w;w=b,b=k}le(C,Jt,w,b,w);{let k=w;w=b,b=k}le(C,Qt,w,b,w);{let k=w;w=b,b=k}tt(C,w,Et,ea,Pt,128,2,24,24),tt(C,w,Ht,De,j,128,36,24,24),C.copyBufferToBuffer(vt,0,oe,0,864*4),C.copyBufferToBuffer(Ke,0,Ue,0,15552*4),C.copyBufferToBuffer(Pt,0,Je,0,576*2*4),C.copyBufferToBuffer(j,0,Qe,0,576*36*4),o.queue.submit([C.finish()]),await Promise.all([oe.mapAsync(GPUMapMode.READ),Ue.mapAsync(GPUMapMode.READ),Je.mapAsync(GPUMapMode.READ),Qe.mapAsync(GPUMapMode.READ)]);let z=new Float32Array(oe.getMappedRange()).slice(),q=new Float32Array(Ue.getMappedRange()).slice(),ue=new Float32Array(Je.getMappedRange()).slice(),ee=new Float32Array(Qe.getMappedRange()).slice();oe.unmap(),Ue.unmap(),Je.unmap(),Qe.unmap();let $=2016,te=new Float32Array($),pe=new Float32Array($*18),he=0;for(let k=0;k<12;k++)for(let R=0;R<12;R++)for(let Y=0;Y<6;Y++){te[he]=z[Y*144+k*12+R];for(let ye=0;ye<18;ye++){let ct=Y*18+ye;pe[he*18+ye]=q[ct*144+k*12+R]}he++}for(let k=0;k<24;k++)for(let R=0;R<24;R++)for(let Y=0;Y<2;Y++){te[he]=ue[Y*576+k*24+R];for(let ye=0;ye<18;ye++){let ct=Y*18+ye;pe[he*18+ye]=ee[ct*576+k*24+R]}he++}return{scores:te,regressors:pe}}async function qe(c,p){let H=o.createBuffer({size:p*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),C=o.createCommandEncoder();C.copyBufferToBuffer(c,0,H,0,p*4),o.queue.submit([C.finish()]),await H.mapAsync(GPUMapMode.READ);let w=new Float32Array(H.getMappedRange()).slice();return H.unmap(),H.destroy(),w}async function Ft(c){o.queue.copyExternalImageToTexture({source:c},{texture:et},[192,192]);let p=de(new Uint32Array([192,192,192])),H=o.createBindGroup({layout:_e,entries:[{binding:0,resource:et.createView()},{binding:1,resource:{buffer:We}},{binding:2,resource:{buffer:p}}]}),C=o.createCommandEncoder(),w=C.beginComputePass();w.setPipeline(Fe),w.setBindGroup(0,H),w.dispatchWorkgroups(Z(192,16),Z(192,16),1),w.end(),o.queue.submit([C.finish()]);let b=await qe(We,36864*3),z={min:Math.min(...b.slice(0,1e3)),max:Math.max(...b.slice(0,1e3)),mean:b.slice(0,1e3).reduce((R,Y)=>R+Y,0)/1e3,nonZero:b.slice(0,1e3).filter(R=>R!==0).length},q=o.createCommandEncoder(),ue=o.createBindGroup({layout:W,entries:[{binding:0,resource:{buffer:We}},{binding:1,resource:{buffer:ge}},{binding:2,resource:{buffer:be}},{binding:3,resource:{buffer:Ce}},{binding:4,resource:{buffer:pt}},{binding:5,resource:{buffer:K}}]}),ee=q.beginComputePass();ee.setPipeline(P),ee.setBindGroup(0,ue),ee.dispatchWorkgroups(Z(96,8),Z(96,8),32),ee.end(),o.queue.submit([q.finish()]);let $=await qe(pt,9216*32),te={min:Math.min(...$.slice(0,1e3)),max:Math.max(...$.slice(0,1e3)),mean:$.slice(0,1e3).reduce((R,Y)=>R+Y,0)/1e3,nonZero:$.slice(0,1e3).filter(R=>R!==0).length},pe=await qe(ge,100),he=await qe(be,32),k=await qe(Ce,32);return{input:z,inputSample:Array.from(b.slice(0,20)),initConv:te,convSample:Array.from($.slice(0,20)),weightSample:Array.from(pe.slice(0,20)),biasSample:Array.from(he.slice(0,10)),alphaSample:Array.from(k.slice(0,10))}}return{device:o,run:It,debugRun:Ft}}function Kn(){let n=[];for(let l=0;l<12;l++)for(let o=0;o<12;o++){let y=(o+.5)/12,h=(l+.5)/12;for(let e=0;e<6;e++)n.push({x:y,y:h})}for(let l=0;l<24;l++)for(let o=0;o<24;o++){let y=(o+.5)/24,h=(l+.5)/24;for(let e=0;e<2;e++)n.push({x:y,y:h})}return n}var _n=Kn();function qn(n){return 1/(1+Math.exp(-n))}function mn(n,l){let o=[],{scores:y,regressors:h}=n,e=192;for(let g=0;g<_n.length;g++){let A=qn(y[g]);if(A<l)continue;let S=_n[g],d=g*18,f=S.x+h[d+0]/e,v=S.y+h[d+1]/e,U=h[d+2]/e,T=h[d+3]/e,_=[];for(let I=0;I<7;I++){let B=S.x+h[d+4+I*2]/e,re=S.y+h[d+4+I*2+1]/e;_.push([B,re])}o.push({score:A,box:[f,v,U,T],keypoints:_})}return o}function fn(n,l){if(n.length===0)return[];let o=[...n].sort((e,g)=>g.score-e.score),y=[],h=new Set;for(let e=0;e<o.length;e++)if(!h.has(e)){y.push(o[e]);for(let g=e+1;g<o.length;g++)h.has(g)||$n(o[e],o[g])>l&&h.add(g)}return y}function $n(n,l){let o=n.box[0]-n.box[2]/2,y=n.box[1]-n.box[3]/2,h=n.box[0]+n.box[2]/2,e=n.box[1]+n.box[3]/2,g=l.box[0]-l.box[2]/2,A=l.box[1]-l.box[3]/2,S=l.box[0]+l.box[2]/2,d=l.box[1]+l.box[3]/2,f=Math.max(o,g),v=Math.max(y,A),U=Math.min(h,S),T=Math.min(e,d),_=Math.max(0,U-f),I=Math.max(0,T-v),B=_*I,re=(h-o)*(e-y),ne=(S-g)*(d-A),xe=re+ne-B;return xe>0?B/xe:0}function Yn(n){let[l,o,y,h]=n.box,e=n.keypoints[0],g=n.keypoints[2],A=g[0]-e[0],S=g[1]-e[1],d=Math.atan2(A,S),v=Math.max(y,h)*2.6,U=.5,T=Math.sqrt(A*A+S*S),_=T>0?A/T*v*U*.5:0,I=T>0?S/T*v*U*.5:0;return{centerX:l+_,centerY:o+I,width:v,height:v,rotation:d}}function wa(n,l={}){let{scoreThreshold:o=.5,nmsThreshold:y=.3,maxHands:h=2}=l;async function e(A){let S=await n.run(A),d=mn(S,o);return fn(d,y).slice(0,h).map(Yn)}async function g(A){let S=await n.run(A),d=mn(S,o);return fn(d,y).slice(0,h)}return{detect:e,detectRaw:g,model:n}}function hn(n,l=256){let o=Math.cos(n.rotation),y=Math.sin(n.rotation),h=n.width/l,e=n.height/l,g=h*o,A=-e*y,S=h*y,d=e*o,f=n.centerX-(g*l/2+A*l/2),v=n.centerY-(S*l/2+d*l/2),U=g*d-A*S,T=d/U,_=-A/U,I=-S/U,B=g/U,re=-(T*f+_*v),ne=-(I*f+B*v);return{forward:[g,A,f,S,d,v],inverse:[T,_,re,I,B,ne]}}function ga(n,l){let{forward:o}=hn(l,1),[y,h,e,g,A,S]=o;return n.map(d=>({x:y*d.x+h*d.y+e,y:g*d.x+A*d.y+S,z:d.z}))}var wn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Xn(n={}){let{weightsUrl:l,scoreThreshold:o=.5,forceF32:y=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let h=l??wn,e=h.endsWith("/")?h:`${h}/`,g=`${e}weights_f16.json`,A=`${e}weights_f16.bin`,[S,d]=await Promise.all([fetch(g),fetch(A)]);if(!S.ok)throw new Error(`Failed to fetch weights metadata: ${S.status}`);if(!d.ok)throw new Error(`Failed to fetch weights binary: ${d.status}`);let f=await S.json(),v=await d.arrayBuffer(),U=Ut(f,v),T=await Zt(U,{forceF32:y});if(!y){let W=new OffscreenCanvas(256,256),O=W.getContext("2d");O.fillStyle="#886644",O.fillRect(0,0,256,256),O.fillStyle="#cc9966",O.fillRect(50,50,156,156);let F=await T.runFromCanvas(W);F.landmarks.every(we=>we===0)&&F.handflag.every(we=>we===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),T.device.destroy(),T=await Zt(U,{forceF32:!0}))}let _=null;function I(){return _||(_=new OffscreenCanvas(256,256)),_}async function B(W){if(W instanceof HTMLCanvasElement||W instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&W instanceof ImageBitmap)return W;let O=I();O.width=256,O.height=256;let F=O.getContext("2d");return W instanceof ImageData?F.putImageData(W,0,0):F.drawImage(W,0,0,256,256),O}function re(W,O,F){let Ge=W[0];if(Ge<o)return null;let we=O[0]>.5,Ee=[];for(let _e=0;_e<21;_e++)Ee.push({x:F[_e*3],y:F[_e*3+1],z:F[_e*3+2]});return{score:Ge,handedness:we?"right":"left",landmarks:Ee}}async function ne(W){let O=await B(W),F=await T.runFromCanvas(O);return re(F.handflag,F.handedness,F.landmarks)}async function xe(W){let O=await B(W),F=await T.runFromCanvasPipelined(O);return F?re(F.handflag,F.handedness,F.landmarks):null}async function se(){let W=await T.flushPipelined();return W?re(W.handflag,W.handedness,W.landmarks):null}function D(){T.device.destroy(),_=null}async function N(W){let O=await B(W);return T.benchmarkDiagnostic(O)}async function Q(W){let O=await B(W);return T.debugLayerOutputs(O)}return{detect:ne,detectPipelined:xe,flushPipelined:se,dispose:D,benchmarkDiagnostic:N,debugLayerOutputs:Q}}async function Vn(n={}){let{weightsUrl:l,palmWeightsUrl:o,scoreThreshold:y=.5,palmScoreThreshold:h=.5,maxHands:e=2,forceF32:g=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let A=l??wn,S=A.endsWith("/")?A:`${A}/`;if(!o)throw new Error("palmWeightsUrl is required for createFullHandpose");let d=o.endsWith("/")?o:`${o}/`,[f,v,U,T]=await Promise.all([fetch(`${S}weights_f16.json`),fetch(`${S}weights_f16.bin`),fetch(`${d}palm_detection_weights.json`),fetch(`${d}palm_detection_weights.bin`)]);if(!f.ok)throw new Error(`Failed to fetch landmark weights metadata: ${f.status}`);if(!v.ok)throw new Error(`Failed to fetch landmark weights binary: ${v.status}`);if(!U.ok)throw new Error(`Failed to fetch palm weights metadata: ${U.status}`);if(!T.ok)throw new Error(`Failed to fetch palm weights binary: ${T.status}`);let[_,I,B,re]=await Promise.all([f.json(),v.arrayBuffer(),U.json(),T.arrayBuffer()]),ne=Ut(_,I),xe=Ut(B,re),se=await Zt(ne,{forceF32:g}),D=await ha(xe),N=wa(D,{scoreThreshold:h,maxHands:e}),Q=null,W=null;function O(){return Q||(Q=new OffscreenCanvas(192,192)),Q}function F(){return W||(W=new OffscreenCanvas(256,256)),W}async function Ge(P){if(P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas){if(P.width===192&&P.height===192)return P;let V=O();return V.width=192,V.height=192,V.getContext("2d").drawImage(P,0,0,192,192),V}if(typeof ImageBitmap<"u"&&P instanceof ImageBitmap){if(P.width===192&&P.height===192)return P;let V=O();return V.width=192,V.height=192,V.getContext("2d").drawImage(P,0,0,192,192),V}let X=O();X.width=192,X.height=192;let me=X.getContext("2d");if(P instanceof ImageData){let V=new OffscreenCanvas(P.width,P.height);V.getContext("2d").putImageData(P,0,0),me.drawImage(V,0,0,192,192)}else me.drawImage(P,0,0,192,192);return X}function we(P,X,me,V){let fe=F();fe.width=256,fe.height=256;let ie=fe.getContext("2d"),Fe=Math.cos(-X.rotation),Ve=Math.sin(-X.rotation);ie.clearRect(0,0,256,256),ie.save(),ie.translate(128,128),ie.scale(X.width*me/256,X.height*V/256),ie.rotate(-X.rotation),ie.translate(-128,-128);let Pe=X.centerX*me,Be=X.centerY*V;ie.restore();let ge=256/(X.width*me),be=256/(X.height*V),Ce=Math.cos(X.rotation),ke=Math.sin(X.rotation),je=Ce*ge,Ae=ke*ge,ze=-ke*be,Se=Ce*be,it=-Pe*je-Be*ze+128,rt=-Pe*Ae-Be*Se+128;if(ie.setTransform(je,Ae,ze,Se,it,rt),P instanceof ImageData){let Ne=new OffscreenCanvas(P.width,P.height);Ne.getContext("2d").putImageData(P,0,0),ie.drawImage(Ne,0,0)}else ie.drawImage(P,0,0);return ie.setTransform(1,0,0,1,0,0),fe}function Ee(P){return P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas?[P.width,P.height]:typeof ImageBitmap<"u"&&P instanceof ImageBitmap?[P.width,P.height]:P instanceof ImageData?[P.width,P.height]:P instanceof HTMLVideoElement?[P.videoWidth,P.videoHeight]:P instanceof HTMLImageElement?[P.naturalWidth,P.naturalHeight]:[256,256]}async function _e(P){let X=await Ge(P),me=await N.detect(X);if(me.length===0)return[];let[V,fe]=Ee(P),ie=[];for(let Fe of me){let Ve=we(P,Fe,V,fe),Pe=await se.runFromCanvas(Ve),Be=Pe.handflag[0];if(Be<y)continue;let ge=Pe.handedness[0]>.5,be=[];for(let ke=0;ke<21;ke++)be.push({x:Pe.landmarks[ke*3],y:Pe.landmarks[ke*3+1],z:Pe.landmarks[ke*3+2]});let Ce=ga(be,Fe);ie.push({score:Be,handedness:ge?"right":"left",landmarks:Ce,palmScore:0})}return ie}function ve(){se.device.destroy(),D.device.destroy(),Q=null,W=null}return{detect:_e,dispose:ve}}function jn(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Zn=jn(`
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
`);function Jn(n){let l=n.createShaderModule({code:Zn}),o=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),y=n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[o]}),compute:{module:l,entryPoint:"main"}});function h(e,g,A,S,d,f,v){let U=new Uint32Array([d,f,v,0]),T=n.createBuffer({size:U.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(T,0,U);let _=new Float32Array(S),I=new Float32Array(8);I.set(_);let B=n.createBuffer({size:I.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(B,0,I);let re=n.createBindGroup({layout:o,entries:[{binding:0,resource:g.createView()},{binding:1,resource:{buffer:A}},{binding:2,resource:{buffer:T}},{binding:3,resource:{buffer:B}}]}),ne=e.beginComputePass();ne.setPipeline(y),ne.setBindGroup(0,re),ne.dispatchWorkgroups(Math.ceil(v/16),Math.ceil(v/16),1),ne.end()}return{crop:h}}export{ha as compilePalmModel,hn as computeCropTransform,Jn as createCropPipeline,Vn as createFullHandpose,Xn as createHandpose,wa as createPalmDetector,Ut as loadWeightsFromBuffer,ga as projectLandmarksToOriginal};
