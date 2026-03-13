function we(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ia(n){let l=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],t="enable f16;"+n;for(let P of l)for(;t.includes(`${P}:array<f32>`);)t=t.replace(`${P}:array<f32>`,`${P}:array<f16>`);return t}var ma=we(`
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
`),ba=we(`
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
`);function Fa(n,l){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function za(n,l){return ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Ka(n,l){return ha.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Na(n,l){return ba.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function qa(n,l){return[8,8]}var $a=we(`
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
`),Ya=we(`
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
`);function Xa(n){return we(`
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
`)}var Va=Xa(!1),Za=Xa(!0),ja=we(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Ja=we(`
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
`);function Qa(n){return we(`
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
`)}var en=Qa("sigmoid"),tn=Qa("div256"),an=we(`
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
`),nn=we(`
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
`);function rn(n,l){let P=Math.min(l,256),h=l>P,y=n%4===0?`var ic:u32=0u;
    while(ic<${n}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${n}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,E=`var skip_val:f32=0.0;
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
    }`,A=n===l?"":`if(c<${n}u){`,_=n===l?"":"}",b=h?`for(var c:u32=lid.x;c<${n}u;c+=${P}u){`:`let c=lid.x;
  ${A}`,B=h?"}":_,S=h?`for(var c:u32=lid.x;c<${l}u;c+=${P}u){`:"{let c=lid.x;";return we(`
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
@compute @workgroup_size(${P},1,1)
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
  ${B}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${S}
    let pw_base=c*${n}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${y}
    // Skip connection (only for c < inCh)
    ${E}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var sn=we(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),on=we(`
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
`),un=we(`
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
`);function Ot(n,l){let t=new Map,P=n.dtype??"float32";for(let h=0;h<n.keys.length;h++){let e=n.keys[h],y=n.shapes[h],E=n.offsets[h],A=y.reduce((B,S)=>B*S,1),_,b;if(P==="float32")_=new Float32Array(l,E,A);else{let B=new DataView(l);_=new Float32Array(A);for(let S=0;S<A;S++)_[S]=Nn(B.getUint16(E+S*2,!0));b=l.slice(E,E+A*2)}t.set(e,{data:_,shape:y,rawF16:b})}return t}function Nn(n){let l=n>>15&1,t=n>>10&31,P=n&1023;if(t===0){if(P===0)return l?-0:0;let y=-14,E=P/1024;return(l?-1:1)*Math.pow(2,y)*E}if(t===31)return P===0?l?-1/0:1/0:NaN;let h=t-15,e=1+P/1024;return(l?-1:1)*Math.pow(2,h)*e}var qn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],pn=qn.map(([n,l,t,P,h])=>({type:"resmodule",inCh:n,outCh:l,h:t,w:t,stride:P,prefix:h})),$n=2,Yn=5,Xn=8,Vn=11;async function It(n,l){if(!navigator.gpu)throw new Error("WebGPU not supported");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No GPU adapter found");let P=t.features.has("shader-f16"),h=P?["shader-f16"]:[],e=await t.requestDevice({requiredFeatures:h,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(t.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(t.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(t.limits.maxComputeInvocationsPerWorkgroup,288)}}),y=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(P)try{let r=`enable f16;
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
}`,u=e.createShaderModule({code:r}),o=e.createShaderModule({code:c}),a=await u.getCompilationInfo(),L=await o.getCompilationInfo();if(a.messages.some(H=>H.type==="error")||L.messages.some(H=>H.type==="error"))y=!1;else{let H=new Float32Array(2400);H.fill(1);let R=new Uint16Array(2400);R.fill(10516);let U=new Uint16Array(96);U.fill(14336);let g=new Uint16Array(9216);g.fill(8478);let p=new Uint16Array(96);p.fill(12288);let $=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,oe=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,_e=e.createBuffer({size:H.byteLength,usage:$}),Ct=e.createBuffer({size:R.byteLength,usage:$}),kt=e.createBuffer({size:U.byteLength,usage:$}),Ut=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),Gt=e.createBuffer({size:g.byteLength,usage:$}),At=e.createBuffer({size:p.byteLength,usage:$}),St=e.createBuffer({size:384,usage:oe}),at=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(_e,0,H),e.queue.writeBuffer(Ct,0,R),e.queue.writeBuffer(kt,0,U),e.queue.writeBuffer(Gt,0,g),e.queue.writeBuffer(At,0,p);let Ve="read-only-storage",Tt="storage",Lt=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ve}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ve}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ve}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Tt}}]}),Ra=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ve}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ve}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ve}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Tt}}]}),Rn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Lt]}),compute:{module:u,entryPoint:"main"}}),On=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Ra]}),compute:{module:o,entryPoint:"main"}}),In=e.createBindGroup({layout:Lt,entries:[{binding:0,resource:{buffer:_e}},{binding:1,resource:{buffer:Ct}},{binding:2,resource:{buffer:kt}},{binding:3,resource:{buffer:Ut}}]}),Fn=e.createBindGroup({layout:Ra,entries:[{binding:0,resource:{buffer:Ut}},{binding:1,resource:{buffer:Gt}},{binding:2,resource:{buffer:At}},{binding:3,resource:{buffer:St}}]}),ea=e.createCommandEncoder(),ta=ea.beginComputePass();ta.setPipeline(Rn),ta.setBindGroup(0,In),ta.dispatchWorkgroups(2),ta.end();let aa=ea.beginComputePass();aa.setPipeline(On),aa.setBindGroup(0,Fn),aa.dispatchWorkgroups(2),aa.end(),ea.copyBufferToBuffer(St,0,at,0,384),e.queue.submit([ea.finish()]),await e.queue.onSubmittedWorkDone(),await at.mapAsync(GPUMapMode.READ);let Rt=new Float32Array(at.getMappedRange()),Oa=1.5*.0104*96+.25,zn=Rt[0]!==0&&Rt[47]!==0&&Rt[95]!==0,Kn=Math.abs(Rt[0]-Oa)<1;y=zn&&Kn,at.unmap(),_e.destroy(),Ct.destroy(),kt.destroy(),Ut.destroy(),Gt.destroy(),At.destroy(),St.destroy(),at.destroy(),y||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Rt[0]}, expected ~${Oa.toFixed(2)}) \u2014 falling back to f32`)}}catch{y=!1}let A=n.values().next().value,_=y&&!!A?.rawF16&&!l?.forceF32;console.log(_?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${P}, f16 validated: ${y}, f16 data: ${!!A?.rawF16})`);function b(r){if(_&&r.rawF16){let c=new Uint16Array(r.rawF16);if(c.length%2!==0){let u=new Uint16Array(c.length+1);return u.set(c),u}return c}return r.data}function B(r){if(_&&r.rawF16){let c=r.rawF16.byteLength;return Math.ceil(c/4)*4}return r.data.byteLength}function S(r){return _?Ia(r):r}let z={r:"read-only-storage",s:"storage",u:"uniform"};function f(r){return e.createBindGroupLayout({entries:r.map((c,u)=>({binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:z[c]}}))})}function V(r){return e.createBindGroupLayout({entries:r.map((c,u)=>c==="t"?{binding:u,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:z[c]}})})}let G=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,de=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,re=GPUBufferUsage.STORAGE,ve=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ue=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function T(r,c){return e.createBuffer({size:r,usage:c})}function ee(r,c){return e.createBindGroup({layout:r,entries:c.map((u,o)=>({binding:o,resource:"size"in u?{buffer:u}:u}))})}function se(r,c){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[r]}),compute:{module:c,entryPoint:"main"}})}let O=e.createShaderModule({code:$a}),Y=e.createShaderModule({code:un}),Z=e.createShaderModule({code:S(an)}),Ce=e.createShaderModule({code:S(fa)}),ke=e.createShaderModule({code:S(ma)}),me=e.createShaderModule({code:S(ha)}),Pe=e.createShaderModule({code:S(ba)}),De=e.createShaderModule({code:S(Ya)}),mt=e.createShaderModule({code:Va}),Dt=e.createShaderModule({code:ja}),k=e.createShaderModule({code:Za}),j=e.createShaderModule({code:S(Ja)}),he=e.createShaderModule({code:S(en)}),N=e.createShaderModule({code:S(tn)}),pe=e.createShaderModule({code:S(nn)}),te=new Map;function Ue(r,c){let u=`${r}_${c}`,o=te.get(u);return o||(o=e.createShaderModule({code:S(rn(r,c))}),te.set(u,o)),o}let Ze=f(["r","r","r","s","u"]),be=f(["r","r","r","r","s","u"]),Ge=f(["r","s","u"]),Oe=f(["r","r","r","s","u"]),je=f(["r","s","u"]),Me=f(["r","r","s","u"]),Be=f(["r","r","s","u"]),Ae=f(["r","r","r","s","u"]),xe=f(["r","r","r","s","u"]),Ee=V(["t","s","u"]),He=f(["r","r","r","r","r","r","r","s"]),ge=f(["r","r","r","r","r","s","u"]),ft=e.createPipelineLayout({bindGroupLayouts:[Ze]}),nt=e.createPipelineLayout({bindGroupLayouts:[be]}),Ie=r=>e.createComputePipeline({layout:ft,compute:{module:r,entryPoint:"main"}}),Je=r=>e.createComputePipeline({layout:nt,compute:{module:r,entryPoint:"main"}}),Ft=Ie(Ce),zt=Ie(ke),Kt=Je(me),ht=Je(Pe),bt=new Map,Mt=new Map,wt=new Map,Et=new Map;bt.set("8,8",Ft),Mt.set("8,8",zt),wt.set("8,8",Kt),Et.set("8,8",ht);function it(r,c,u,o,a){let L=`${c},${u}`,H=r.get(L);return H||(H=a(e.createShaderModule({code:S(o(c,u))})),r.set(L,H)),H}let gt=(r,c)=>it(bt,r,c,Fa,Ie),Ht=(r,c)=>it(Mt,r,c,za,Ie),Nt=(r,c)=>it(wt,r,c,Ka,Je),Wt=(r,c)=>it(Et,r,c,Na,Je),We=pn.map(r=>{let c=r.stride===2?r.h/2:r.h,u=r.stride===2?r.w/2:r.w,[o,a]=qa(r.inCh,c),L=r.h>=64,H=c>=16&&r.inCh>=288&&r.outCh>=288&&r.outCh%2===0;return{dwPipeline:L?Ht(o,a):gt(o,a),pwPipeline:H?Wt(o,a):Nt(o,a),dwDispatchX:Math.ceil(u/o),dwDispatchY:Math.ceil(c/a),dwDispatchZ:r.inCh,pwDispatchX:Math.ceil(u/o),pwDispatchY:Math.ceil(c/a),pwDispatchZ:H?r.outCh/2:r.outCh}}),yt=se(Ge,O),Qe=se(Oe,De);se(je,mt),se(Me,Dt);let ze=se(Be,k),qt=se(Ae,j);se(xe,he),se(xe,N);let Te=se(Ee,Y),xt=se(He,Z),rt=se(ge,pe),Ke=1*288*128*128*4,st=T(3*256*256*4,G),Fe=T(3*257*257*4,re),ot=T(12,ue);e.queue.writeBuffer(ot,0,new Uint32Array([3,256,257]));let Q=T(Ke,de),fe=T(Ke,ve),Le=T(Ke,re),ut=T(3072*64*4,G),pt=T(3072*32*4,G),ct=T(1536*16*4,G),F=T(6144*64*4,re),ne=T(260,ve),ae=T(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);T(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let ce=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Re=T(8,ue);e.queue.writeBuffer(Re,0,new Uint32Array([256,257]));let dt=n.get("backbone1.1.weight"),$t=n.get("backbone1.1.bias");if(!dt||!$t)throw new Error("Missing input conv weights");let X=b(dt),Yt=b($t),d=T(X.byteLength,G),i=T(Yt.byteLength,G),x=T(28,ue);e.queue.writeBuffer(d,0,X),e.queue.writeBuffer(i,0,Yt),e.queue.writeBuffer(x,0,new Uint32Array([1,3,24,257,257,128,128]));let M=n.get("backbone6.1.weight"),w=n.get("backbone6.1.bias");if(!M||!w)throw new Error("Missing backbone6.1 conv1x1 weights");let s=b(M),I=b(w),ie=T(s.byteLength,G),m=T(I.byteLength,G),D=T(20,ue);e.queue.writeBuffer(ie,0,s),e.queue.writeBuffer(m,0,I),e.queue.writeBuffer(D,0,new Uint32Array([1,96,48,32,32]));let v=n.get("handflag.weight"),q=n.get("handflag.bias");if(!v||!q)throw new Error("Missing handflag weights");let W=b(v),ye=b(q),le=T(W.byteLength,G),C=T(ye.byteLength,G),J=T(12,ue);e.queue.writeBuffer(le,0,W),e.queue.writeBuffer(C,0,ye),e.queue.writeBuffer(J,0,new Uint32Array([1,288,1]));let K=n.get("handedness.weight"),Se=n.get("handedness.bias");if(!K||!Se)throw new Error("Missing handedness weights");let vt=b(K),Ba=b(Se),ia=T(vt.byteLength,G),ra=T(Ba.byteLength,G),Ca=T(12,ue);e.queue.writeBuffer(ia,0,vt),e.queue.writeBuffer(ra,0,Ba),e.queue.writeBuffer(Ca,0,new Uint32Array([1,288,1]));let ka=n.get("reg_3d.weight"),Ua=n.get("reg_3d.bias");if(!ka||!Ua)throw new Error("Missing reg_3d weights");let Ga=b(ka),Aa=b(Ua),sa=T(Ga.byteLength,G),oa=T(Aa.byteLength,G),Sa=T(12,ue);e.queue.writeBuffer(sa,0,Ga),e.queue.writeBuffer(oa,0,Aa),e.queue.writeBuffer(Sa,0,new Uint32Array([1,288,63]));let lt=pn.map(r=>{let{inCh:c,outCh:u,h:o,w:a,stride:L,prefix:H}=r,R=L===2?o/2:o,U=L===2?a/2:a,g=L===2?1:2,p=n.get(`${H}convs.0.weight`),$=n.get(`${H}convs.0.bias`),oe=n.get(`${H}convs.1.weight`),_e=n.get(`${H}convs.1.bias`);if(!p||!$||!oe||!_e)throw new Error(`Missing weights for ${H}`);let Ct=b(p),kt=b($),Ut=b(oe),Gt=b(_e),At=T(Ct.byteLength,G),St=T(kt.byteLength,G),at=T(Ut.byteLength,G),Ve=T(Gt.byteLength,G),Tt=T(32,ue),Lt=T(36,ue);return e.queue.writeBuffer(At,0,Ct),e.queue.writeBuffer(St,0,kt),e.queue.writeBuffer(at,0,Ut),e.queue.writeBuffer(Ve,0,Gt),e.queue.writeBuffer(Tt,0,new Uint32Array([1,c,o,a,R,U,L,g])),e.queue.writeBuffer(Lt,0,new Uint32Array([1,c,u,R,U,Math.max(0,u-c),L,o,a])),{dwWeight:At,dwBias:St,pwWeight:at,pwBias:Ve,dwUniform:Tt,pwUniform:Lt,spec:r,outH:R,outW:U}});function Pt(r){let c=T(r.length*4,ue);return e.queue.writeBuffer(c,0,new Uint32Array(r)),c}let Bn=Pt([1,96,8,8,16,16]),Cn=Pt([1,96,16,16,32,32]),kn=Pt([1,48,32,32,64,64]);Pt([1536*16]),Pt([3072*32]),Pt([3072*64]);let Da=ee(Ge,[st,Fe,ot]),Ma=ee(Oe,[Fe,d,i,Q,x]),Ne=[],qe=[],$e=[],Ye=[];for(let r of lt)Ne.push(ee(Ze,[Q,r.dwWeight,r.dwBias,Le,r.dwUniform])),qe.push(ee(be,[Le,Q,r.pwWeight,r.pwBias,fe,r.pwUniform])),$e.push(ee(Ze,[fe,r.dwWeight,r.dwBias,Le,r.dwUniform])),Ye.push(ee(be,[Le,fe,r.pwWeight,r.pwBias,Q,r.pwUniform]));let Un=ee(Be,[Q,ct,fe,Bn]),Gn=ee(Be,[Q,pt,fe,Cn]),An=ee(Ae,[Q,ie,m,F,D]),Sn=ee(Be,[F,ut,fe,kn]);ee(xe,[Q,le,C,ne,J]),ee(xe,[Q,ia,ra,ne,Ca]),ee(xe,[Q,sa,oa,ne,Sa]);let et=ee(Ee,[ce.createView(),Fe,Re]),Dn=ee(He,[Q,le,C,ia,ra,sa,oa,ne]),ua=24,Ea=[],Ha=[];for(let r=ua;r<lt.length;r++){let c=lt[r];Ea.push(ee(ge,[Q,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,fe,c.dwUniform])),Ha.push(ee(ge,[fe,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,Q,c.dwUniform]))}let pa=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});pa.globalCompositeOperation="copy";let Wa=new OffscreenCanvas(9,8),Xt=Wa.getContext("webgpu"),Vt=null,ca=null;if(Xt){Xt.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let r=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),c=e.createShaderModule({code:sn}),u=e.createShaderModule({code:on});Vt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[r]}),vertex:{module:c,entryPoint:"vs"},fragment:{module:u,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),ca=e.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:ne}}]})}let Zt=new Float32Array(1),jt=new Float32Array(1),Jt=new Float32Array(63);function Xe(r,c){let u=!0,o=0,a=r.beginComputePass();for(a.setPipeline(Qe),a.setBindGroup(0,Ma),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);o<=$n;o++){let U=u?Ne[o]:$e[o],g=u?qe[o]:Ye[o],p=We[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let L=u?Q:fe;for(r.copyBufferToBuffer(L,0,ut,0,3072*64*4),a=r.beginComputePass();o<=Yn;o++){let U=u?Ne[o]:$e[o],g=u?qe[o]:Ye[o],p=We[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let H=u?Q:fe;for(r.copyBufferToBuffer(H,0,pt,0,3072*32*4),a=r.beginComputePass();o<=Xn;o++){let U=u?Ne[o]:$e[o],g=u?qe[o]:Ye[o],p=We[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let R=u?Q:fe;for(r.copyBufferToBuffer(R,0,ct,0,1536*16*4),a=r.beginComputePass();o<=Vn;o++){let U=u?Ne[o]:$e[o],g=u?qe[o]:Ye[o],p=We[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.setPipeline(ze),a.setBindGroup(0,Un),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),u=!1,a=r.beginComputePass();{let U=u?Ne[o]:$e[o],g=u?qe[o]:Ye[o],p=We[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}a.setPipeline(ze),a.setBindGroup(0,Gn),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),u=!1,a=r.beginComputePass();{let U=u?Ne[o]:$e[o],g=u?qe[o]:Ye[o],p=We[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}for(a.setPipeline(qt),a.setBindGroup(0,An),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(ze),a.setBindGroup(0,Sn),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),u=!1,a=r.beginComputePass();o<ua;o++){let U=u?Ne[o]:$e[o],g=u?qe[o]:Ye[o],p=We[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}for(;o<lt.length;o++){let U=o-ua,g=u?Ea[U]:Ha[U],p=lt[o];a.setPipeline(rt),a.setBindGroup(0,g),a.dispatchWorkgroups(p.outW,p.outH,1),u=!u}a.setPipeline(xt),a.setBindGroup(0,Dn),a.dispatchWorkgroups(1),a.end(),c&&r.copyBufferToBuffer(ne,0,c,0,260)}async function Qt(r){e.queue.writeBuffer(st,0,r);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(yt),a.setBindGroup(0,Da),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),a.end()}Xe(c,ae),e.queue.submit([c.finish()]);let u=ae.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(ae.getMappedRange());return Zt[0]=o[0],jt[0]=o[1],Jt.set(o.subarray(2,65)),ae.unmap(),{handflag:new Float32Array(Zt),handedness:new Float32Array(jt),landmarks:new Float32Array(Jt)}}async function da(r){e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(Te),a.setBindGroup(0,et),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}Xe(c,ae),e.queue.submit([c.finish()]);let u=ae.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(ae.getMappedRange());return Zt[0]=o[0],jt[0]=o[1],Jt.set(o.subarray(2,65)),ae.unmap(),{handflag:new Float32Array(Zt),handedness:new Float32Array(jt),landmarks:new Float32Array(Jt)}}async function Ta(r){if(!Vt||!ca||!Xt)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);let c=e.createCommandEncoder();{let U=c.beginComputePass();U.setPipeline(Te),U.setBindGroup(0,et),U.dispatchWorkgroups(16,16,1),U.end()}Xe(c,null);let u=Xt.getCurrentTexture(),o=c.beginRenderPass({colorAttachments:[{view:u.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});o.setPipeline(Vt),o.setBindGroup(0,ca),o.draw(3),o.end(),e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),pa.drawImage(Wa,0,0);let L=pa.getImageData(0,0,9,8).data,H=new Float32Array(65),R=new DataView(new ArrayBuffer(4));for(let U=0;U<65;U++){let g=U*4;R.setUint8(0,L[g]),R.setUint8(1,L[g+1]),R.setUint8(2,L[g+2]),R.setUint8(3,L[g+3]),H[U]=R.getFloat32(0)}return{handflag:new Float32Array([H[0]]),handedness:new Float32Array([H[1]]),landmarks:new Float32Array(H.subarray(2,65))}}let Mn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),la=0,En=[ae,Mn],Bt=null,tt=null;async function _a(r){let c=En[la];la=1-la,e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);let u=e.createCommandEncoder();{let a=u.beginComputePass();a.setPipeline(Te),a.setBindGroup(0,et),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}Xe(u,c),e.queue.submit([u.finish()]);let o=null;if(Bt!==null&&tt!==null){await Bt;let a=new Float32Array(tt.getMappedRange());o={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))},tt.unmap()}return tt=c,Bt=c.mapAsync(GPUMapMode.READ),o}async function La(){if(!Bt||!tt)return null;await Bt;let r=new Float32Array(tt.getMappedRange()),c={handflag:new Float32Array([r[0]]),handedness:new Float32Array([r[1]]),landmarks:new Float32Array(r.subarray(2,65))};return tt.unmap(),Bt=null,tt=null,c}async function Hn(r=50){let c=new Float32Array(196608);for(let a=0;a<5;a++)await Qt(c);let u=[];for(let a=0;a<r;a++){let L=performance.now();await Qt(c),u.push(performance.now()-L)}let o=u.reduce((a,L)=>a+L,0)/u.length;return{avgMs:o,fps:1e3/o}}async function Wn(r=50){let c=new Float32Array(196608);for(let H=0;H<5;H++)await Qt(c);let u=[];for(let H=0;H<r;H++){let R=e.createCommandEncoder();{let g=R.beginComputePass();g.setPipeline(yt),g.setBindGroup(0,Da),g.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),g.end()}Xe(R,ae);let U=performance.now();e.queue.submit([R.finish()]),await e.queue.onSubmittedWorkDone(),u.push(performance.now()-U)}u.sort((H,R)=>H-R);let o=u.reduce((H,R)=>H+R,0)/u.length,a=u[Math.floor(u.length/2)],L=u[0];return{avgMs:o,fps:1e3/o,medianMs:a,minMs:L}}function oi(r){e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);let c=e.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(Te),u.setBindGroup(0,et),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),u.end()}Xe(c,ae),e.queue.submit([c.finish()])}async function Tn(r,c=50){function u(g){let p=[...g].sort(($,oe)=>$-oe);return{median:p[Math.floor(p.length/2)],min:p[0]}}for(let g=0;g<10;g++)await da(r);let o=[];for(let g=0;g<c;g++){e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);let p=e.createCommandEncoder();{let oe=p.beginComputePass();oe.setPipeline(Te),oe.setBindGroup(0,et),oe.dispatchWorkgroups(16,16,1),oe.end()}Xe(p,ae);let $=performance.now();e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-$)}let a=[];for(let g=0;g<c;g++){e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);let p=e.createCommandEncoder();{let _e=p.beginComputePass();_e.setPipeline(Te),_e.setBindGroup(0,et),_e.dispatchWorkgroups(16,16,1),_e.end()}Xe(p,ae),e.queue.submit([p.finish()]);let $=ae.mapAsync(GPUMapMode.READ),oe=performance.now();await e.queue.onSubmittedWorkDone(),await $,ae.getMappedRange(),ae.unmap(),a.push(performance.now()-oe)}let L=[];for(let g=0;g<c;g++){e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);let p=e.createCommandEncoder();{let oe=p.beginComputePass();oe.setPipeline(Te),oe.setBindGroup(0,et),oe.dispatchWorkgroups(16,16,1),oe.end()}Xe(p,ae),e.queue.submit([p.finish()]);let $=performance.now();await ae.mapAsync(GPUMapMode.READ),ae.getMappedRange(),ae.unmap(),L.push(performance.now()-$)}let H=[];for(let g=0;g<c;g++){let p=performance.now();await da(r),H.push(performance.now()-p)}await _a(r);let R=[];for(let g=0;g<c;g++){let p=performance.now();await _a(r),R.push(performance.now()-p)}await La();let U=null;if(Vt){let g=[];for(let p=0;p<c;p++){let $=performance.now();await Ta(r),g.push(performance.now()-$)}U=u(g)}return{gpuOnly:u(o),mapAsyncOnly:u(a),mapAsyncNoWait:u(L),total:u(H),pipelined:u(R),renderReadback:U}}async function Ln(r){let c=[];async function u(a,L,H){let R=e.createBuffer({size:L,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),U=e.createCommandEncoder();U.copyBufferToBuffer(a,0,R,0,L),e.queue.submit([U.finish()]),await e.queue.onSubmittedWorkDone(),await R.mapAsync(GPUMapMode.READ);let g=new Float32Array(R.getMappedRange()),p=1/0,$=-1/0,oe=0;for(let _e=0;_e<g.length;_e++)g[_e]<p&&(p=g[_e]),g[_e]>$&&($=g[_e]),g[_e]!==0&&oe++;R.unmap(),R.destroy(),c.push({layer:H,stats:{min:p,max:$,nonZero:oe,total:g.length}})}e.queue.copyExternalImageToTexture({source:r},{texture:ce},[256,256]);{let a=e.createCommandEncoder(),L=a.beginComputePass();L.setPipeline(Te),L.setBindGroup(0,et),L.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),L.end(),e.queue.submit([a.finish()])}await u(Fe,Math.min(Fe.size,3*257*257*4),"canvas\u2192bufInput");{let a=e.createCommandEncoder(),L=a.beginComputePass();L.setPipeline(Qe),L.setBindGroup(0,Ma),L.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),L.end(),e.queue.submit([a.finish()])}await u(Q,Math.min(Q.size,3072*128*4),"inputConv\u2192bufA");let o=!0;for(let a=0;a<Math.min(lt.length,6);a++){let L=o?Ne[a]:$e[a],H=o?qe[a]:Ye[a],R=We[a],U=lt[a];{let p=e.createCommandEncoder(),$=p.beginComputePass();$.setPipeline(R.dwPipeline),$.setBindGroup(0,L),$.dispatchWorkgroups(R.dwDispatchX,R.dwDispatchY,R.dwDispatchZ),$.end(),e.queue.submit([p.finish()])}await u(Le,Math.min(Le.size,U.spec.inCh*U.outH*U.outW*4),`layer${a}.DW\u2192bufDW (${U.spec.prefix})`);{let p=e.createCommandEncoder(),$=p.beginComputePass();$.setPipeline(R.pwPipeline),$.setBindGroup(0,H),$.dispatchWorkgroups(R.pwDispatchX,R.pwDispatchY,R.pwDispatchZ),$.end(),e.queue.submit([p.finish()])}let g=o?fe:Q;await u(g,Math.min(g.size,U.spec.outCh*U.outH*U.outW*4),`layer${a}.PW\u2192buf${o?"B":"A"} (${U.spec.prefix})`),o=!o}return c}return{device:e,run:Qt,runFromCanvas:da,runFromCanvasViaRender:Ta,runFromCanvasPipelined:_a,flushPipelined:La,benchmark:Hn,benchmarkGPU:Wn,benchmarkDiagnostic:Tn,debugLayerOutputs:Ln}}function _t(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var cn=_t(`
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
`),dn=_t(`
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
`),ln=_t(`
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
`),_n=_t(`
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
`),mn=_t(`
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
`),fn=_t(`
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
`),hn=_t(`
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
`);async function wa(n,l){let t;if(l)t=l;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");t=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8)}})}let P={r:"read-only-storage",s:"storage",u:"uniform"};function h(d){return t.createBindGroupLayout({entries:d.map((i,x)=>i==="t"?{binding:x,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:x,visibility:GPUShaderStage.COMPUTE,buffer:{type:P[i]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,E=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function _(d,i){return t.createBuffer({size:Math.max(d,4),usage:i})}function b(d,i,x){t.queue.writeBuffer(d,i,x)}function B(d){let i=_(d.data.byteLength,e);return b(i,0,d.data),i}let S=Array.from(n.keys());function z(d){let i=n.get(d);if(!i)throw new Error(`Weight not found: ${d}`);return i}function f(...d){let i=S.find(x=>d.every(M=>x.includes(M)));if(!i)throw new Error(`Weight not found for: ${d.join(", ")}`);return z(i)}function V(d){let[,i,x,M]=d.shape,w=new Float32Array(M*25);for(let s=0;s<M;s++)for(let I=0;I<i;I++)for(let ie=0;ie<x;ie++)w[s*25+I*5+ie]=d.data[I*x*M+ie*M+s];return w}function G(d){let[i,,,x]=d.shape,M=new Float32Array(i*x);for(let w=0;w<i;w++)for(let s=0;s<x;s++)M[w*x+s]=d.data[w*x+s];return M}let de=t.createShaderModule({code:cn}),re=t.createShaderModule({code:dn}),ve=t.createShaderModule({code:ln}),ue=t.createShaderModule({code:_n}),T=t.createShaderModule({code:fn}),ee=t.createShaderModule({code:mn}),se=t.createShaderModule({code:hn}),O=h(["r","r","r","r","s","u"]),Y=h(["r","r","r","s","u"]),Z=h(["r","r","r","r","r","s","u"]),Ce=h(["r","r","r","s","u"]),ke=h(["r","r","r","r","s","u"]),me=h(["r","r","s","u"]),Pe=h(["t","s","u"]);function De(d,i){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:i,entryPoint:"main"}})}let mt=De(O,de),Dt=De(Y,re),k=De(Z,ve),j=De(Ce,ue),he=De(ke,T),N=De(me,ee),pe=De(Pe,se),te=f("conv2d/Conv2D"),Ue=f("batch_normalization/","conv2d/Conv2D"),Ze=f("p_re_lu/"),be=B(te),Ge=B(Ue),Oe=B(Ze),Me=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(d=>{let i=f(d.dwKey),x=f(d.pwKey),M=f(d.bnKey),w=f(d.preluKey),s=V(i),I=_(s.byteLength,e);b(I,0,s);let ie=new Float32Array(d.inCh),m=_(ie.byteLength,e);b(m,0,ie);let D=G(x),v=_(D.byteLength,e);b(v,0,D);let q=B(M),W=B(w);return{dwWeightBuf:I,dwBiasBuf:m,pwWeightBuf:v,pwBiasBuf:q,alphaBuf:W,inCh:d.inCh,outCh:d.outCh,stride:d.stride,inH:d.inH}}),Be=G(f("conv2d_20/Conv2D")),Ae=_(Be.byteLength,e);b(Ae,0,Be);let xe=B(f("batch_normalization_20/")),Ee=B(f("p_re_lu_20/")),He={dwWeightBuf:(()=>{let d=V(f("depthwise_conv2d_19/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(256),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(f("conv2d_21/")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(f("batch_normalization_21/")),alphaBuf:B(f("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},ge={dwWeightBuf:(()=>{let d=V(f("depthwise_conv2d_20/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(256),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(f("conv2d_22/Conv2D1")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(f("batch_normalization_22/")),alphaBuf:B(f("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},ft=G(f("conv2d_23/Conv2D")),nt=_(ft.byteLength,e);b(nt,0,ft);let Ie=B(f("batch_normalization_23/")),Je=B(f("p_re_lu_23/")),Ft={dwWeightBuf:(()=>{let d=V(f("depthwise_conv2d_21/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(128),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(f("conv2d_24/")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(f("batch_normalization_24/")),alphaBuf:B(f("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},zt={dwWeightBuf:(()=>{let d=V(f("depthwise_conv2d_22/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(128),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(f("conv2d_25/Conv2D1")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(f("batch_normalization_25/")),alphaBuf:B(f("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},Kt=G(f("classifier_palm_16_NO_PRUNING/Conv2D")),ht=_(Kt.byteLength,e);b(ht,0,Kt);let bt=B(f("classifier_palm_16_NO_PRUNING/BiasAdd")),Mt=G(f("regressor_palm_16_NO_PRUNING/Conv2D")),wt=_(Mt.byteLength,e);b(wt,0,Mt);let Et=B(f("regressor_palm_16_NO_PRUNING/BiasAdd")),it=G(f("classifier_palm_8_NO_PRUNING/Conv2D")),gt=_(it.byteLength,e);b(gt,0,it);let Ht=B(f("classifier_palm_8_NO_PRUNING/BiasAdd")),Nt=G(f("regressor_palm_8_NO_PRUNING/Conv2D")),Wt=_(Nt.byteLength,e);b(Wt,0,Nt);let We=B(f("regressor_palm_8_NO_PRUNING/BiasAdd")),yt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Qe=_(36864*3*4,e),ze=_(yt,y),qt=_(yt,y),Te=_(yt,y),xt=_(576*256*4,y),rt=_(144*256*4,y|GPUBufferUsage.COPY_DST),Ke=_(576*128*4,y|GPUBufferUsage.COPY_DST),st=_(864*4,E),Fe=_(15552*4,E),ot=_(576*2*4,E),Q=_(576*36*4,E),fe=_(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Le=_(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ut=_(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pt=_(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=t.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function F(d,i){return Math.ceil(d/i)}function ne(d){let i=_(d.byteLength,A);return b(i,0,d),i}let ae=ne(new Uint32Array([1,3,32,192,192,96,96]));function ce(d,i,x,M,w){let s=i.stride===2?i.inH/2:i.inH,I=s,ie=i.stride===2?1:2,m=ne(new Uint32Array([1,i.inCh,i.inH,i.inH,s,I,i.stride,ie])),D=t.createBindGroup({layout:Y,entries:[{binding:0,resource:{buffer:x}},{binding:1,resource:{buffer:i.dwWeightBuf}},{binding:2,resource:{buffer:i.dwBiasBuf}},{binding:3,resource:{buffer:Te}},{binding:4,resource:{buffer:m}}]}),v=d.beginComputePass();v.setPipeline(Dt),v.setBindGroup(0,D),v.dispatchWorkgroups(F(I,8),F(s,8),i.inCh),v.end();let q=i.inCh,W=ne(new Uint32Array([1,i.inCh,i.outCh,s,I,q,i.stride,i.inH,i.inH])),ye=t.createBindGroup({layout:Z,entries:[{binding:0,resource:{buffer:Te}},{binding:1,resource:{buffer:w}},{binding:2,resource:{buffer:i.pwWeightBuf}},{binding:3,resource:{buffer:i.pwBiasBuf}},{binding:4,resource:{buffer:i.alphaBuf}},{binding:5,resource:{buffer:M}},{binding:6,resource:{buffer:W}}]}),le=d.beginComputePass();le.setPipeline(k),le.setBindGroup(0,ye),le.dispatchWorkgroups(F(I,8),F(s,8),i.outCh),le.end()}function Re(d,i,x,M,w,s,I,ie,m){let D=ne(new Uint32Array([1,s,I,ie,m])),v=t.createBindGroup({layout:Ce,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:D}}]}),q=d.beginComputePass();q.setPipeline(j),q.setBindGroup(0,v),q.dispatchWorkgroups(F(m,8),F(ie,8),I),q.end()}function dt(d,i,x,M,w,s,I,ie,m,D){let v=ne(new Uint32Array([1,I,ie,m,D])),q=t.createBindGroup({layout:ke,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:s}},{binding:5,resource:{buffer:v}}]}),W=d.beginComputePass();W.setPipeline(he),W.setBindGroup(0,q),W.dispatchWorkgroups(F(D,8),F(m,8),ie),W.end()}async function $t(d){t.queue.copyExternalImageToTexture({source:d},{texture:ct},[192,192]);let i=ne(new Uint32Array([192,192,192])),x=t.createBindGroup({layout:Pe,entries:[{binding:0,resource:ct.createView()},{binding:1,resource:{buffer:Qe}},{binding:2,resource:{buffer:i}}]}),M=t.createCommandEncoder();{let C=M.beginComputePass();C.setPipeline(pe),C.setBindGroup(0,x),C.dispatchWorkgroups(F(192,16),F(192,16),1),C.end()}{let C=t.createBindGroup({layout:O,entries:[{binding:0,resource:{buffer:Qe}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:Ge}},{binding:3,resource:{buffer:Oe}},{binding:4,resource:{buffer:ze}},{binding:5,resource:{buffer:ae}}]}),J=M.beginComputePass();J.setPipeline(mt),J.setBindGroup(0,C),J.dispatchWorkgroups(F(96,8),F(96,8),32),J.end()}let w=ze,s=qt;for(let C=0;C<Me.length;C++){let J=Me[C];ce(M,J,w,s,w);let K=w;w=s,s=K,C===10&&M.copyBufferToBuffer(w,0,Ke,0,576*128*4),C===14&&M.copyBufferToBuffer(w,0,rt,0,144*256*4)}{let C=ne(new Uint32Array([1,256,6,6,12,12])),J=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:xt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),K=M.beginComputePass();K.setPipeline(N),K.setBindGroup(0,J),K.dispatchWorkgroups(F(12,8),F(12,8),256),K.end()}{let C=w;w=s,s=C}dt(M,w,Ae,xe,Ee,s,256,256,12,12);{let C=w;w=s,s=C}{let C=ne(new Uint32Array([1,256,12,12,12,12])),J=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:rt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),K=M.beginComputePass();K.setPipeline(N),K.setBindGroup(0,J),K.dispatchWorkgroups(F(12,8),F(12,8),256),K.end()}{let C=w;w=s,s=C}ce(M,He,w,s,w);{let C=w;w=s,s=C}ce(M,ge,w,s,w);{let C=w;w=s,s=C}Re(M,w,ht,bt,st,256,6,12,12),Re(M,w,wt,Et,Fe,256,108,12,12);{let C=ne(new Uint32Array([1,256,12,12,24,24])),J=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:xt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),K=M.beginComputePass();K.setPipeline(N),K.setBindGroup(0,J),K.dispatchWorkgroups(F(24,8),F(24,8),256),K.end()}{let C=w;w=s,s=C}dt(M,w,nt,Ie,Je,s,256,128,24,24);{let C=w;w=s,s=C}{let C=ne(new Uint32Array([1,128,24,24,24,24])),J=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:Ke}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),K=M.beginComputePass();K.setPipeline(N),K.setBindGroup(0,J),K.dispatchWorkgroups(F(24,8),F(24,8),128),K.end()}{let C=w;w=s,s=C}ce(M,Ft,w,s,w);{let C=w;w=s,s=C}ce(M,zt,w,s,w);{let C=w;w=s,s=C}Re(M,w,gt,Ht,ot,128,2,24,24),Re(M,w,Wt,We,Q,128,36,24,24),t.queue.submit([M.finish()]);let I=t.createCommandEncoder();I.copyBufferToBuffer(st,0,fe,0,864*4),I.copyBufferToBuffer(Fe,0,Le,0,15552*4),I.copyBufferToBuffer(ot,0,ut,0,576*2*4),I.copyBufferToBuffer(Q,0,pt,0,576*36*4),t.queue.submit([I.finish()]),await Promise.all([fe.mapAsync(GPUMapMode.READ),Le.mapAsync(GPUMapMode.READ),ut.mapAsync(GPUMapMode.READ),pt.mapAsync(GPUMapMode.READ)]);let ie=new Float32Array(fe.getMappedRange()).slice(),m=new Float32Array(Le.getMappedRange()).slice(),D=new Float32Array(ut.getMappedRange()).slice(),v=new Float32Array(pt.getMappedRange()).slice();fe.unmap(),Le.unmap(),ut.unmap(),pt.unmap();let q=2016,W=new Float32Array(q),ye=new Float32Array(q*18),le=0;for(let C=0;C<12;C++)for(let J=0;J<12;J++)for(let K=0;K<6;K++){W[le]=ie[K*144+C*12+J];for(let Se=0;Se<18;Se++){let vt=K*18+Se;ye[le*18+Se]=m[vt*144+C*12+J]}le++}for(let C=0;C<24;C++)for(let J=0;J<24;J++)for(let K=0;K<2;K++){W[le]=D[K*576+C*24+J];for(let Se=0;Se<18;Se++){let vt=K*18+Se;ye[le*18+Se]=v[vt*576+C*24+J]}le++}return{scores:W,regressors:ye}}async function X(d,i){let x=t.createBuffer({size:i*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),M=t.createCommandEncoder();M.copyBufferToBuffer(d,0,x,0,i*4),t.queue.submit([M.finish()]),await x.mapAsync(GPUMapMode.READ);let w=new Float32Array(x.getMappedRange()).slice();return x.unmap(),x.destroy(),w}async function Yt(d){t.queue.copyExternalImageToTexture({source:d},{texture:ct},[192,192]);function i(v,q=1e3){let W=v.slice(0,q);return{min:Math.min(...W),max:Math.max(...W),mean:W.reduce((ye,le)=>ye+le,0)/W.length,nonZero:W.filter(ye=>ye!==0).length,sample:Array.from(W.slice(0,10))}}let x={},M=ne(new Uint32Array([192,192,192])),w=t.createBindGroup({layout:Pe,entries:[{binding:0,resource:ct.createView()},{binding:1,resource:{buffer:Qe}},{binding:2,resource:{buffer:M}}]}),s=t.createCommandEncoder(),I=s.beginComputePass();I.setPipeline(pe),I.setBindGroup(0,w),I.dispatchWorkgroups(F(192,16),F(192,16),1),I.end(),t.queue.submit([s.finish()]),x.input=i(await X(Qe,36864*3)),s=t.createCommandEncoder();let ie=t.createBindGroup({layout:O,entries:[{binding:0,resource:{buffer:Qe}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:Ge}},{binding:3,resource:{buffer:Oe}},{binding:4,resource:{buffer:ze}},{binding:5,resource:{buffer:ae}}]});I=s.beginComputePass(),I.setPipeline(mt),I.setBindGroup(0,ie),I.dispatchWorkgroups(F(96,8),F(96,8),32),I.end(),t.queue.submit([s.finish()]),x.initConv=i(await X(ze,9216*32));let m=ze,D=qt;for(let v=0;v<Me.length;v++){let q=Me[v];s=t.createCommandEncoder(),ce(s,q,m,D,m),t.queue.submit([s.finish()]);let W=m;if(m=D,D=W,v===0||v===3||v===7||v===11||v===14||v===15||v===18){let ye=q.stride===2?q.inH/2:q.inH,le=ye*ye*q.outCh;x[`block${v}`]=i(await X(m,le))}v===10&&(s=t.createCommandEncoder(),s.copyBufferToBuffer(m,0,Ke,0,576*128*4),t.queue.submit([s.finish()])),v===14&&(s=t.createCommandEncoder(),s.copyBufferToBuffer(m,0,rt,0,144*256*4),t.queue.submit([s.finish()]))}s=t.createCommandEncoder();{let v=ne(new Uint32Array([1,256,6,6,12,12])),q=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:xt}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline(N),W.setBindGroup(0,q),W.dispatchWorkgroups(F(12,8),F(12,8),256),W.end()}t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpnUpsample6to12=i(await X(m,144*256)),s=t.createCommandEncoder(),dt(s,m,Ae,xe,Ee,D,256,256,12,12),t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpn6to12Conv=i(await X(m,144*256)),x.backbone12Skip=i(await X(rt,144*256)),s=t.createCommandEncoder();{let v=ne(new Uint32Array([1,256,12,12,12,12])),q=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:rt}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline(N),W.setBindGroup(0,q),W.dispatchWorkgroups(F(12,8),F(12,8),256),W.end()}t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpnAdd12=i(await X(m,144*256)),s=t.createCommandEncoder(),ce(s,He,m,D,m),t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpn12Block1=i(await X(m,144*256)),s=t.createCommandEncoder(),ce(s,ge,m,D,m),t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpn12Block2=i(await X(m,144*256)),s=t.createCommandEncoder(),Re(s,m,ht,bt,st,256,6,12,12),t.queue.submit([s.finish()]),x.cls16=i(await X(st,864)),s=t.createCommandEncoder(),Re(s,m,wt,Et,Fe,256,108,12,12),t.queue.submit([s.finish()]),x.reg16=i(await X(Fe,15552),500),s=t.createCommandEncoder();{let v=ne(new Uint32Array([1,256,12,12,24,24])),q=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:xt}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline(N),W.setBindGroup(0,q),W.dispatchWorkgroups(F(24,8),F(24,8),256),W.end()}t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpnUpsample12to24=i(await X(m,576*256)),s=t.createCommandEncoder(),dt(s,m,nt,Ie,Je,D,256,128,24,24),t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpn12to24Conv=i(await X(m,576*128)),x.backbone24Skip=i(await X(Ke,576*128)),s=t.createCommandEncoder();{let v=ne(new Uint32Array([1,128,24,24,24,24])),q=t.createBindGroup({layout:me,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Ke}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline(N),W.setBindGroup(0,q),W.dispatchWorkgroups(F(24,8),F(24,8),128),W.end()}t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpnAdd24=i(await X(m,576*128)),s=t.createCommandEncoder(),ce(s,Ft,m,D,m),t.queue.submit([s.finish()]);{let v=m;m=D,D=v}x.fpn24Block1=i(await X(m,576*128)),s=t.createCommandEncoder(),ce(s,zt,m,D,m),t.queue.submit([s.finish()]);{let v=m;m=D,D=v}return x.fpn24Block2=i(await X(m,576*128)),s=t.createCommandEncoder(),Re(s,m,gt,Ht,ot,128,2,24,24),t.queue.submit([s.finish()]),x.cls8=i(await X(ot,576*2)),s=t.createCommandEncoder(),Re(s,m,Wt,We,Q,128,36,24,24),t.queue.submit([s.finish()]),x.reg8=i(await X(Q,576*36)),x.initWeights=i(await X(be,100),100),x.initBias=i(await X(Ge,32),32),x.cls16Weights=i(await X(ht,100),100),x.cls16Bias=i(await X(bt,6),6),x.cls8Weights=i(await X(gt,100),100),x.cls8Bias=i(await X(Ht,2),2),x.fpn6to12Weights=i(await X(Ae,100),100),x}return{device:t,run:$t,debugRun:Yt}}function Zn(){let n=[];for(let l=0;l<12;l++)for(let t=0;t<12;t++){let P=(t+.5)/12,h=(l+.5)/12;for(let e=0;e<6;e++)n.push({x:P,y:h})}for(let l=0;l<24;l++)for(let t=0;t<24;t++){let P=(t+.5)/24,h=(l+.5)/24;for(let e=0;e<2;e++)n.push({x:P,y:h})}return n}var bn=Zn();function jn(n){return 1/(1+Math.exp(-n))}function wn(n,l){let t=[],{scores:P,regressors:h}=n,e=192;for(let y=0;y<bn.length;y++){let E=jn(P[y]);if(E<l)continue;let A=bn[y],_=y*18,b=A.x+h[_+0]/e,B=A.y+h[_+1]/e,S=h[_+2]/e,z=h[_+3]/e,f=[];for(let V=0;V<7;V++){let G=A.x+h[_+4+V*2]/e,de=A.y+h[_+4+V*2+1]/e;f.push([G,de])}t.push({score:E,box:[b,B,S,z],keypoints:f})}return t}function gn(n,l){if(n.length===0)return[];let t=[...n].sort((e,y)=>y.score-e.score),P=[],h=new Set;for(let e=0;e<t.length;e++)if(!h.has(e)){P.push(t[e]);for(let y=e+1;y<t.length;y++)h.has(y)||Jn(t[e],t[y])>l&&h.add(y)}return P}function Jn(n,l){let t=n.box[0]-n.box[2]/2,P=n.box[1]-n.box[3]/2,h=n.box[0]+n.box[2]/2,e=n.box[1]+n.box[3]/2,y=l.box[0]-l.box[2]/2,E=l.box[1]-l.box[3]/2,A=l.box[0]+l.box[2]/2,_=l.box[1]+l.box[3]/2,b=Math.max(t,y),B=Math.max(P,E),S=Math.min(h,A),z=Math.min(e,_),f=Math.max(0,S-b),V=Math.max(0,z-B),G=f*V,de=(h-t)*(e-P),re=(A-y)*(_-E),ve=de+re-G;return ve>0?G/ve:0}function Qn(n){let[l,t,P,h]=n.box,e=n.keypoints[0],y=n.keypoints[2],E=y[0]-e[0],A=y[1]-e[1],_=Math.atan2(A,E),B=-Math.PI/2-_,S=Math.max(P,h),f=S*2.6,V=-.5*S,G=Math.cos(B),de=Math.sin(B),re=-V*de,ve=V*G;return{centerX:l+re,centerY:t+ve,width:f,height:f,rotation:B}}function ga(n,l={}){let{scoreThreshold:t=.5,nmsThreshold:P=.3,maxHands:h=2}=l;async function e(E){let A=await n.run(E),_=wn(A,t);return gn(_,P).slice(0,h).map(Qn)}async function y(E){let A=await n.run(E),_=wn(A,t);return gn(_,P).slice(0,h)}return{detect:e,detectRaw:y,model:n}}function yn(n,l=256){let t=Math.cos(n.rotation),P=Math.sin(n.rotation),h=n.width/l,e=n.height/l,y=h*t,E=-e*P,A=h*P,_=e*t,b=n.centerX-(y*l/2+E*l/2),B=n.centerY-(A*l/2+_*l/2),S=y*_-E*A,z=_/S,f=-E/S,V=-A/S,G=y/S,de=-(z*b+f*B),re=-(V*b+G*B);return{forward:[y,E,b,A,_,B],inverse:[z,f,de,V,G,re]}}function ya(n,l){let{forward:t}=yn(l,1),[P,h,e,y,E,A]=t;return n.map(_=>({x:P*_.x+h*_.y+e,y:y*_.x+E*_.y+A,z:_.z}))}var xa=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function na(n){let l={};for(let t=0;t<xa.length;t++)l[xa[t]]=n[t];return l}function xn(n,l,t){return n.initialized?(n.value=t*l+(1-t)*n.value,n.value):(n.value=l,n.initialized=!0,l)}function vn(n,l){let t=2*Math.PI*l*n;return t/(t+1)}function ei(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function va(n,l,t,P,h,e){let y=n.lastTime<0?.03333333333333333:t-n.lastTime;n.lastTime=t;let E=vn(y,e),A=n.x.initialized?(l-n.x.value)/y:0,_=xn(n.dx,A,E),b=P+h*Math.abs(_),B=vn(y,b);return xn(n.x,l,B)}function Pa(n={}){let{minCutoff:l=1,beta:t=10,dCutoff:P=1}=n,h=[];function e(A){h.length!==A&&(h=Array.from({length:A},()=>ei()))}function y(A,_){let b=_??performance.now()/1e3,B=A.length*3;return e(B),A.map((S,z)=>({x:va(h[z*3],S.x,b,l,t,P),y:va(h[z*3+1],S.y,b,l,t,P),z:va(h[z*3+2],S.z,b,l,t,P)}))}function E(){h=[]}return{apply:y,reset:E}}var Pn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ti(n={}){let{weightsUrl:l,scoreThreshold:t=.5,forceF32:P=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let h=l??Pn,e=h.endsWith("/")?h:`${h}/`,y=`${e}weights_f16.json`,E=`${e}weights_f16.bin`,[A,_]=await Promise.all([fetch(y),fetch(E)]);if(!A.ok)throw new Error(`Failed to fetch weights metadata: ${A.status}`);if(!_.ok)throw new Error(`Failed to fetch weights binary: ${_.status}`);let b=await A.json(),B=await _.arrayBuffer(),S=Ot(b,B),z=await It(S,{forceF32:P});if(!P){let O=new OffscreenCanvas(256,256),Y=O.getContext("2d");Y.fillStyle="#886644",Y.fillRect(0,0,256,256),Y.fillStyle="#cc9966",Y.fillRect(50,50,156,156);let Z=await z.runFromCanvas(O);Z.landmarks.every(ke=>ke===0)&&Z.handflag.every(ke=>ke===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),z.device.destroy(),z=await It(S,{forceF32:!0}))}let f=null;function V(){return f||(f=new OffscreenCanvas(256,256)),f}async function G(O){if(O instanceof HTMLCanvasElement||O instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&O instanceof ImageBitmap)return O;let Y=V();Y.width=256,Y.height=256;let Z=Y.getContext("2d");return O instanceof ImageData?Z.putImageData(O,0,0):Z.drawImage(O,0,0,256,256),Y}function de(O,Y,Z){let Ce=O[0];if(Ce<t)return null;let ke=Y[0]>.5,me=[];for(let Pe=0;Pe<21;Pe++)me.push({x:Z[Pe*3],y:Z[Pe*3+1],z:Z[Pe*3+2]});return{score:Ce,handedness:ke?"right":"left",landmarks:me,keypoints:na(me)}}async function re(O){let Y=await G(O),Z=await z.runFromCanvas(Y);return de(Z.handflag,Z.handedness,Z.landmarks)}async function ve(O){let Y=await G(O),Z=await z.runFromCanvasPipelined(Y);return Z?de(Z.handflag,Z.handedness,Z.landmarks):null}async function ue(){let O=await z.flushPipelined();return O?de(O.handflag,O.handedness,O.landmarks):null}function T(){z.device.destroy(),f=null}async function ee(O){let Y=await G(O);return z.benchmarkDiagnostic(Y)}async function se(O){let Y=await G(O);return z.debugLayerOutputs(Y)}return{detect:re,detectPipelined:ve,flushPipelined:ue,dispose:T,benchmarkDiagnostic:ee,debugLayerOutputs:se}}async function ai(n={}){let{weightsUrl:l,palmWeightsUrl:t,scoreThreshold:P=.5,palmScoreThreshold:h=.5,maxHands:e=3,forceF32:y=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let E=l??Pn,A=E.endsWith("/")?E:`${E}/`;if(!t)throw new Error("palmWeightsUrl is required for createFullHandpose");let _=t.endsWith("/")?t:`${t}/`,[b,B,S,z]=await Promise.all([fetch(`${A}weights_f16.json`),fetch(`${A}weights_f16.bin`),fetch(`${_}palm_detection_weights.json`),fetch(`${_}palm_detection_weights.bin`)]);if(!b.ok)throw new Error(`Failed to fetch landmark weights metadata: ${b.status}`);if(!B.ok)throw new Error(`Failed to fetch landmark weights binary: ${B.status}`);if(!S.ok)throw new Error(`Failed to fetch palm weights metadata: ${S.status}`);if(!z.ok)throw new Error(`Failed to fetch palm weights binary: ${z.status}`);let[f,V,G,de]=await Promise.all([b.json(),B.arrayBuffer(),S.json(),z.arrayBuffer()]),re=Ot(f,V),ve=Ot(G,de),ue=await It(re,{forceF32:y});if(!y){let k=new OffscreenCanvas(256,256),j=k.getContext("2d");j.fillStyle="#886644",j.fillRect(0,0,256,256),j.fillStyle="#cc9966",j.fillRect(50,50,156,156);let he=await ue.runFromCanvas(k);he.landmarks.every(pe=>pe===0)&&he.handflag.every(pe=>pe===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),ue.device.destroy(),ue=await It(re,{forceF32:!0}))}let T=await wa(ve),ee=ga(T,{scoreThreshold:h,maxHands:e}),se=[];for(let k=0;k<e;k++)se.push(Pa());let O=0,Y=null,Z=null;function Ce(){return Y||(Y=new OffscreenCanvas(192,192)),Y}function ke(){return Z||(Z=new OffscreenCanvas(256,256)),Z}async function me(k){if(k instanceof HTMLCanvasElement||k instanceof OffscreenCanvas){if(k.width===192&&k.height===192)return k;let N=Ce();return N.width=192,N.height=192,N.getContext("2d").drawImage(k,0,0,192,192),N}if(typeof ImageBitmap<"u"&&k instanceof ImageBitmap){if(k.width===192&&k.height===192)return k;let N=Ce();return N.width=192,N.height=192,N.getContext("2d").drawImage(k,0,0,192,192),N}let j=Ce();j.width=192,j.height=192;let he=j.getContext("2d");if(k instanceof ImageData){let N=new OffscreenCanvas(k.width,k.height);N.getContext("2d").putImageData(k,0,0),he.drawImage(N,0,0,192,192)}else he.drawImage(k,0,0,192,192);return j}function Pe(k,j,he,N){let pe=ke();pe.width=256,pe.height=256;let te=pe.getContext("2d"),Ue=Math.cos(-j.rotation),Ze=Math.sin(-j.rotation);te.clearRect(0,0,256,256),te.save(),te.translate(128,128),te.scale(j.width*he/256,j.height*N/256),te.rotate(-j.rotation),te.translate(-128,-128);let be=j.centerX*he,Ge=j.centerY*N;te.restore();let Oe=Math.min(he,N),je=256/(j.width*Oe),Me=256/(j.height*Oe),Be=Math.cos(j.rotation),Ae=Math.sin(j.rotation),xe=Be*je,Ee=Ae*je,He=-Ae*Me,ge=Be*Me,ft=-be*xe-Ge*He+128,nt=-be*Ee-Ge*ge+128;if(te.setTransform(xe,Ee,He,ge,ft,nt),k instanceof ImageData){let Ie=new OffscreenCanvas(k.width,k.height);Ie.getContext("2d").putImageData(k,0,0),te.drawImage(Ie,0,0)}else te.drawImage(k,0,0);return te.setTransform(1,0,0,1,0,0),pe}function De(k){return k instanceof HTMLCanvasElement||k instanceof OffscreenCanvas?[k.width,k.height]:typeof ImageBitmap<"u"&&k instanceof ImageBitmap?[k.width,k.height]:k instanceof ImageData?[k.width,k.height]:k instanceof HTMLVideoElement?[k.videoWidth,k.videoHeight]:k instanceof HTMLImageElement?[k.naturalWidth,k.naturalHeight]:[256,256]}async function mt(k){let j=await me(k),he=await ee.detect(j);if(he.length===0)return[];let[N,pe]=De(k),te=[];for(let Ue of he){let Ze=Pe(k,Ue,N,pe),be=await ue.runFromCanvas(Ze),Ge=be.handflag[0];if(Ge<P)continue;let Oe=be.handedness[0]>.5,je=[];for(let ge=0;ge<21;ge++)je.push({x:be.landmarks[ge*3],y:be.landmarks[ge*3+1],z:be.landmarks[ge*3+2]});let Me=Math.min(N,pe),Be=Ue.width*Me,Ae={...Ue,width:Be/N,height:Be/pe},xe=ya(je,Ae),Ee=te.length,He=Ee<se.length?se[Ee].apply(xe):xe;te.push({score:Ge,handedness:Oe?"right":"left",landmarks:He,keypoints:na(He),palmScore:0})}if(te.length<O)for(let Ue=te.length;Ue<O;Ue++)Ue<se.length&&se[Ue].reset();return O=te.length,te}function Dt(){ue.device.destroy(),T.device.destroy(),Y=null,Z=null}return{detect:mt,dispose:Dt}}function ni(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ii=ni(`
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
`);function ri(n){let l=n.createShaderModule({code:ii}),t=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),P=n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:l,entryPoint:"main"}});function h(e,y,E,A,_,b,B){let S=new Uint32Array([_,b,B,0]),z=n.createBuffer({size:S.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(z,0,S);let f=new Float32Array(A),V=new Float32Array(8);V.set(f);let G=n.createBuffer({size:V.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(G,0,V);let de=n.createBindGroup({layout:t,entries:[{binding:0,resource:y.createView()},{binding:1,resource:{buffer:E}},{binding:2,resource:{buffer:z}},{binding:3,resource:{buffer:G}}]}),re=e.beginComputePass();re.setPipeline(P),re.setBindGroup(0,de),re.dispatchWorkgroups(Math.ceil(B/16),Math.ceil(B/16),1),re.end()}return{crop:h}}export{xa as LANDMARK_NAMES,wa as compilePalmModel,yn as computeCropTransform,ri as createCropPipeline,ai as createFullHandpose,ti as createHandpose,Pa as createLandmarkSmoother,ga as createPalmDetector,Ot as loadWeightsFromBuffer,ya as projectLandmarksToOriginal,na as toKeypoints};
