function we(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ia(n){let l=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],t="enable f16;"+n;for(let P of l)for(;t.includes(`${P}:array<f32>`);)t=t.replace(`${P}:array<f32>`,`${P}:array<f16>`);return t}var fa=we(`
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
`),ha=we(`
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
`),wa=we(`
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
`);function Fa(n,l){return ha.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function za(n,l){return fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Ka(n,l){return ba.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function Na(n,l){return wa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${l},1)`)}function qa(n,l){return[8,8]}var $a=we(`
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
`);function It(n,l){let t=new Map,P=n.dtype??"float32";for(let h=0;h<n.keys.length;h++){let e=n.keys[h],y=n.shapes[h],E=n.offsets[h],A=y.reduce((B,S)=>B*S,1),_,b;if(P==="float32")_=new Float32Array(l,E,A);else{let B=new DataView(l);_=new Float32Array(A);for(let S=0;S<A;S++)_[S]=Nn(B.getUint16(E+S*2,!0));b=l.slice(E,E+A*2)}t.set(e,{data:_,shape:y,rawF16:b})}return t}function Nn(n){let l=n>>15&1,t=n>>10&31,P=n&1023;if(t===0){if(P===0)return l?-0:0;let y=-14,E=P/1024;return(l?-1:1)*Math.pow(2,y)*E}if(t===31)return P===0?l?-1/0:1/0:NaN;let h=t-15,e=1+P/1024;return(l?-1:1)*Math.pow(2,h)*e}var qn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],pn=qn.map(([n,l,t,P,h])=>({type:"resmodule",inCh:n,outCh:l,h:t,w:t,stride:P,prefix:h})),$n=2,Yn=5,Xn=8,Vn=11;async function Ft(n,l){if(!navigator.gpu)throw new Error("WebGPU not supported");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No GPU adapter found");let P=t.features.has("shader-f16"),h=P?["shader-f16"]:[],e=await t.requestDevice({requiredFeatures:h,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(t.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(t.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(t.limits.maxComputeInvocationsPerWorkgroup,288)}}),y=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(P)try{let r=`enable f16;
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
}`,u=e.createShaderModule({code:r}),o=e.createShaderModule({code:c}),a=await u.getCompilationInfo(),L=await o.getCompilationInfo();if(a.messages.some(H=>H.type==="error")||L.messages.some(H=>H.type==="error"))y=!1;else{let H=new Float32Array(2400);H.fill(1);let R=new Uint16Array(2400);R.fill(10516);let U=new Uint16Array(96);U.fill(14336);let g=new Uint16Array(9216);g.fill(8478);let p=new Uint16Array(96);p.fill(12288);let X=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ue=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,_e=e.createBuffer({size:H.byteLength,usage:X}),kt=e.createBuffer({size:R.byteLength,usage:X}),Ut=e.createBuffer({size:U.byteLength,usage:X}),Gt=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),At=e.createBuffer({size:g.byteLength,usage:X}),St=e.createBuffer({size:p.byteLength,usage:X}),Dt=e.createBuffer({size:384,usage:ue}),nt=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(_e,0,H),e.queue.writeBuffer(kt,0,R),e.queue.writeBuffer(Ut,0,U),e.queue.writeBuffer(At,0,g),e.queue.writeBuffer(St,0,p);let Ze="read-only-storage",Lt="storage",Rt=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Lt}}]}),Ra=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ze}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Lt}}]}),Rn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Rt]}),compute:{module:u,entryPoint:"main"}}),On=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Ra]}),compute:{module:o,entryPoint:"main"}}),In=e.createBindGroup({layout:Rt,entries:[{binding:0,resource:{buffer:_e}},{binding:1,resource:{buffer:kt}},{binding:2,resource:{buffer:Ut}},{binding:3,resource:{buffer:Gt}}]}),Fn=e.createBindGroup({layout:Ra,entries:[{binding:0,resource:{buffer:Gt}},{binding:1,resource:{buffer:At}},{binding:2,resource:{buffer:St}},{binding:3,resource:{buffer:Dt}}]}),ta=e.createCommandEncoder(),aa=ta.beginComputePass();aa.setPipeline(Rn),aa.setBindGroup(0,In),aa.dispatchWorkgroups(2),aa.end();let na=ta.beginComputePass();na.setPipeline(On),na.setBindGroup(0,Fn),na.dispatchWorkgroups(2),na.end(),ta.copyBufferToBuffer(Dt,0,nt,0,384),e.queue.submit([ta.finish()]),await e.queue.onSubmittedWorkDone(),await nt.mapAsync(GPUMapMode.READ);let Ot=new Float32Array(nt.getMappedRange()),Oa=1.5*.0104*96+.25,zn=Ot[0]!==0&&Ot[47]!==0&&Ot[95]!==0,Kn=Math.abs(Ot[0]-Oa)<1;y=zn&&Kn,nt.unmap(),_e.destroy(),kt.destroy(),Ut.destroy(),Gt.destroy(),At.destroy(),St.destroy(),Dt.destroy(),nt.destroy(),y||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Ot[0]}, expected ~${Oa.toFixed(2)}) \u2014 falling back to f32`)}}catch{y=!1}let A=n.values().next().value,_=y&&!!A?.rawF16&&!l?.forceF32;console.log(_?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${P}, f16 validated: ${y}, f16 data: ${!!A?.rawF16})`);function b(r){if(_&&r.rawF16){let c=new Uint16Array(r.rawF16);if(c.length%2!==0){let u=new Uint16Array(c.length+1);return u.set(c),u}return c}return r.data}function B(r){if(_&&r.rawF16){let c=r.rawF16.byteLength;return Math.ceil(c/4)*4}return r.data.byteLength}function S(r){return _?Ia(r):r}let N={r:"read-only-storage",s:"storage",u:"uniform"};function m(r){return e.createBindGroupLayout({entries:r.map((c,u)=>({binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:N[c]}}))})}function O(r){return e.createBindGroupLayout({entries:r.map((c,u)=>c==="t"?{binding:u,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:N[c]}})})}let G=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,me=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,ee=GPUBufferUsage.STORAGE,ge=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,pe=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function T(r,c){return e.createBuffer({size:r,usage:c})}function ae(r,c){return e.createBindGroup({layout:r,entries:c.map((u,o)=>({binding:o,resource:"size"in u?{buffer:u}:u}))})}function oe(r,c){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[r]}),compute:{module:c,entryPoint:"main"}})}let De=e.createShaderModule({code:$a}),Me=e.createShaderModule({code:un}),I=e.createShaderModule({code:S(an)}),z=e.createShaderModule({code:S(ha)}),Q=e.createShaderModule({code:S(fa)}),V=e.createShaderModule({code:S(ba)}),Be=e.createShaderModule({code:S(wa)}),ye=e.createShaderModule({code:S(Ya)}),ke=e.createShaderModule({code:Va}),Mt=e.createShaderModule({code:ja}),k=e.createShaderModule({code:Za}),j=e.createShaderModule({code:S(Ja)}),he=e.createShaderModule({code:S(en)}),$=e.createShaderModule({code:S(tn)}),ce=e.createShaderModule({code:S(nn)}),ne=new Map;function Ue(r,c){let u=`${r}_${c}`,o=ne.get(u);return o||(o=e.createShaderModule({code:S(rn(r,c))}),ne.set(u,o)),o}let je=m(["r","r","r","s","u"]),be=m(["r","r","r","r","s","u"]),Ge=m(["r","s","u"]),Ie=m(["r","r","r","s","u"]),Je=m(["r","s","u"]),Ee=m(["r","r","s","u"]),Ce=m(["r","r","s","u"]),Ae=m(["r","r","r","s","u"]),Pe=m(["r","r","r","s","u"]),He=O(["t","s","u"]),We=m(["r","r","r","r","r","r","r","s"]),xe=m(["r","r","r","r","r","s","u"]),ht=e.createPipelineLayout({bindGroupLayouts:[je]}),it=e.createPipelineLayout({bindGroupLayouts:[be]}),Fe=r=>e.createComputePipeline({layout:ht,compute:{module:r,entryPoint:"main"}}),Qe=r=>e.createComputePipeline({layout:it,compute:{module:r,entryPoint:"main"}}),zt=Fe(z),Kt=Fe(Q),Nt=Qe(V),bt=Qe(Be),wt=new Map,Et=new Map,gt=new Map,Ht=new Map;wt.set("8,8",zt),Et.set("8,8",Kt),gt.set("8,8",Nt),Ht.set("8,8",bt);function rt(r,c,u,o,a){let L=`${c},${u}`,H=r.get(L);return H||(H=a(e.createShaderModule({code:S(o(c,u))})),r.set(L,H)),H}let yt=(r,c)=>rt(wt,r,c,Fa,Fe),Wt=(r,c)=>rt(Et,r,c,za,Fe),qt=(r,c)=>rt(gt,r,c,Ka,Qe),Tt=(r,c)=>rt(Ht,r,c,Na,Qe),Te=pn.map(r=>{let c=r.stride===2?r.h/2:r.h,u=r.stride===2?r.w/2:r.w,[o,a]=qa(r.inCh,c),L=r.h>=64,H=c>=16&&r.inCh>=288&&r.outCh>=288&&r.outCh%2===0;return{dwPipeline:L?Wt(o,a):yt(o,a),pwPipeline:H?Tt(o,a):qt(o,a),dwDispatchX:Math.ceil(u/o),dwDispatchY:Math.ceil(c/a),dwDispatchZ:r.inCh,pwDispatchX:Math.ceil(u/o),pwDispatchY:Math.ceil(c/a),pwDispatchZ:H?r.outCh/2:r.outCh}}),xt=oe(Ge,De),et=oe(Ie,ye);oe(Je,ke),oe(Ee,Mt);let Ke=oe(Ce,k),$t=oe(Ae,j);oe(Pe,he),oe(Pe,$);let Le=oe(He,Me),vt=oe(We,I),st=oe(xe,ce),Ne=1*288*128*128*4,ot=T(3*256*256*4,G),ze=T(3*257*257*4,ee),ut=T(12,pe);e.queue.writeBuffer(ut,0,new Uint32Array([3,256,257]));let te=T(Ne,me),fe=T(Ne,ge),Re=T(Ne,ee),pt=T(3072*64*4,G),ct=T(3072*32*4,G),dt=T(1536*16*4,G),K=T(6144*64*4,ee),re=T(260,ge),ie=T(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);T(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let de=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Oe=T(8,pe);e.queue.writeBuffer(Oe,0,new Uint32Array([256,257]));let lt=n.get("backbone1.1.weight"),Yt=n.get("backbone1.1.bias");if(!lt||!Yt)throw new Error("Missing input conv weights");let Z=b(lt),Xt=b(Yt),d=T(Z.byteLength,G),i=T(Xt.byteLength,G),x=T(28,pe);e.queue.writeBuffer(d,0,Z),e.queue.writeBuffer(i,0,Xt),e.queue.writeBuffer(x,0,new Uint32Array([1,3,24,257,257,128,128]));let M=n.get("backbone6.1.weight"),w=n.get("backbone6.1.bias");if(!M||!w)throw new Error("Missing backbone6.1 conv1x1 weights");let s=b(M),F=b(w),se=T(s.byteLength,G),f=T(F.byteLength,G),D=T(20,pe);e.queue.writeBuffer(se,0,s),e.queue.writeBuffer(f,0,F),e.queue.writeBuffer(D,0,new Uint32Array([1,96,48,32,32]));let v=n.get("handflag.weight"),Y=n.get("handflag.bias");if(!v||!Y)throw new Error("Missing handflag weights");let W=b(v),ve=b(Y),le=T(W.byteLength,G),C=T(ve.byteLength,G),J=T(12,pe);e.queue.writeBuffer(le,0,W),e.queue.writeBuffer(C,0,ve),e.queue.writeBuffer(J,0,new Uint32Array([1,288,1]));let q=n.get("handedness.weight"),Se=n.get("handedness.bias");if(!q||!Se)throw new Error("Missing handedness weights");let Pt=b(q),Ba=b(Se),ra=T(Pt.byteLength,G),sa=T(Ba.byteLength,G),Ca=T(12,pe);e.queue.writeBuffer(ra,0,Pt),e.queue.writeBuffer(sa,0,Ba),e.queue.writeBuffer(Ca,0,new Uint32Array([1,288,1]));let ka=n.get("reg_3d.weight"),Ua=n.get("reg_3d.bias");if(!ka||!Ua)throw new Error("Missing reg_3d weights");let Ga=b(ka),Aa=b(Ua),oa=T(Ga.byteLength,G),ua=T(Aa.byteLength,G),Sa=T(12,pe);e.queue.writeBuffer(oa,0,Ga),e.queue.writeBuffer(ua,0,Aa),e.queue.writeBuffer(Sa,0,new Uint32Array([1,288,63]));let _t=pn.map(r=>{let{inCh:c,outCh:u,h:o,w:a,stride:L,prefix:H}=r,R=L===2?o/2:o,U=L===2?a/2:a,g=L===2?1:2,p=n.get(`${H}convs.0.weight`),X=n.get(`${H}convs.0.bias`),ue=n.get(`${H}convs.1.weight`),_e=n.get(`${H}convs.1.bias`);if(!p||!X||!ue||!_e)throw new Error(`Missing weights for ${H}`);let kt=b(p),Ut=b(X),Gt=b(ue),At=b(_e),St=T(kt.byteLength,G),Dt=T(Ut.byteLength,G),nt=T(Gt.byteLength,G),Ze=T(At.byteLength,G),Lt=T(32,pe),Rt=T(36,pe);return e.queue.writeBuffer(St,0,kt),e.queue.writeBuffer(Dt,0,Ut),e.queue.writeBuffer(nt,0,Gt),e.queue.writeBuffer(Ze,0,At),e.queue.writeBuffer(Lt,0,new Uint32Array([1,c,o,a,R,U,L,g])),e.queue.writeBuffer(Rt,0,new Uint32Array([1,c,u,R,U,Math.max(0,u-c),L,o,a])),{dwWeight:St,dwBias:Dt,pwWeight:nt,pwBias:Ze,dwUniform:Lt,pwUniform:Rt,spec:r,outH:R,outW:U}});function Bt(r){let c=T(r.length*4,pe);return e.queue.writeBuffer(c,0,new Uint32Array(r)),c}let Bn=Bt([1,96,8,8,16,16]),Cn=Bt([1,96,16,16,32,32]),kn=Bt([1,48,32,32,64,64]);Bt([1536*16]),Bt([3072*32]),Bt([3072*64]);let Da=ae(Ge,[ot,ze,ut]),Ma=ae(Ie,[ze,d,i,te,x]),qe=[],$e=[],Ye=[],Xe=[];for(let r of _t)qe.push(ae(je,[te,r.dwWeight,r.dwBias,Re,r.dwUniform])),$e.push(ae(be,[Re,te,r.pwWeight,r.pwBias,fe,r.pwUniform])),Ye.push(ae(je,[fe,r.dwWeight,r.dwBias,Re,r.dwUniform])),Xe.push(ae(be,[Re,fe,r.pwWeight,r.pwBias,te,r.pwUniform]));let Un=ae(Ce,[te,dt,fe,Bn]),Gn=ae(Ce,[te,ct,fe,Cn]),An=ae(Ae,[te,se,f,K,D]),Sn=ae(Ce,[K,pt,fe,kn]);ae(Pe,[te,le,C,re,J]),ae(Pe,[te,ra,sa,re,Ca]),ae(Pe,[te,oa,ua,re,Sa]);let tt=ae(He,[de.createView(),ze,Oe]),Dn=ae(We,[te,le,C,ra,sa,oa,ua,re]),pa=24,Ea=[],Ha=[];for(let r=pa;r<_t.length;r++){let c=_t[r];Ea.push(ae(xe,[te,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,fe,c.dwUniform])),Ha.push(ae(xe,[fe,c.dwWeight,c.dwBias,c.pwWeight,c.pwBias,te,c.dwUniform]))}let ca=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});ca.globalCompositeOperation="copy";let Wa=new OffscreenCanvas(9,8),Vt=Wa.getContext("webgpu"),Zt=null,da=null;if(Vt){Vt.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let r=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),c=e.createShaderModule({code:sn}),u=e.createShaderModule({code:on});Zt=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[r]}),vertex:{module:c,entryPoint:"vs"},fragment:{module:u,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),da=e.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:re}}]})}let jt=new Float32Array(1),Jt=new Float32Array(1),Qt=new Float32Array(63);function Ve(r,c){let u=!0,o=0,a=r.beginComputePass();for(a.setPipeline(et),a.setBindGroup(0,Ma),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);o<=$n;o++){let U=u?qe[o]:Ye[o],g=u?$e[o]:Xe[o],p=Te[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let L=u?te:fe;for(r.copyBufferToBuffer(L,0,pt,0,3072*64*4),a=r.beginComputePass();o<=Yn;o++){let U=u?qe[o]:Ye[o],g=u?$e[o]:Xe[o],p=Te[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let H=u?te:fe;for(r.copyBufferToBuffer(H,0,ct,0,3072*32*4),a=r.beginComputePass();o<=Xn;o++){let U=u?qe[o]:Ye[o],g=u?$e[o]:Xe[o],p=Te[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.end();let R=u?te:fe;for(r.copyBufferToBuffer(R,0,dt,0,1536*16*4),a=r.beginComputePass();o<=Vn;o++){let U=u?qe[o]:Ye[o],g=u?$e[o]:Xe[o],p=Te[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}a.setPipeline(Ke),a.setBindGroup(0,Un),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),u=!1,a=r.beginComputePass();{let U=u?qe[o]:Ye[o],g=u?$e[o]:Xe[o],p=Te[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}a.setPipeline(Ke),a.setBindGroup(0,Gn),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),u=!1,a=r.beginComputePass();{let U=u?qe[o]:Ye[o],g=u?$e[o]:Xe[o],p=Te[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u,o++}for(a.setPipeline($t),a.setBindGroup(0,An),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(Ke),a.setBindGroup(0,Sn),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),u=!1,a=r.beginComputePass();o<pa;o++){let U=u?qe[o]:Ye[o],g=u?$e[o]:Xe[o],p=Te[o];a.setPipeline(p.dwPipeline),a.setBindGroup(0,U),a.dispatchWorkgroups(p.dwDispatchX,p.dwDispatchY,p.dwDispatchZ),a.setPipeline(p.pwPipeline),a.setBindGroup(0,g),a.dispatchWorkgroups(p.pwDispatchX,p.pwDispatchY,p.pwDispatchZ),u=!u}for(;o<_t.length;o++){let U=o-pa,g=u?Ea[U]:Ha[U],p=_t[o];a.setPipeline(st),a.setBindGroup(0,g),a.dispatchWorkgroups(p.outW,p.outH,1),u=!u}a.setPipeline(vt),a.setBindGroup(0,Dn),a.dispatchWorkgroups(1),a.end(),c&&r.copyBufferToBuffer(re,0,c,0,260)}async function ea(r){e.queue.writeBuffer(ot,0,r);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(xt),a.setBindGroup(0,Da),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),a.end()}Ve(c,ie),e.queue.submit([c.finish()]);let u=ie.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(ie.getMappedRange());return jt[0]=o[0],Jt[0]=o[1],Qt.set(o.subarray(2,65)),ie.unmap(),{handflag:new Float32Array(jt),handedness:new Float32Array(Jt),landmarks:new Float32Array(Qt)}}async function la(r){e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);let c=e.createCommandEncoder();{let a=c.beginComputePass();a.setPipeline(Le),a.setBindGroup(0,tt),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}Ve(c,ie),e.queue.submit([c.finish()]);let u=ie.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await u;let o=new Float32Array(ie.getMappedRange());return jt[0]=o[0],Jt[0]=o[1],Qt.set(o.subarray(2,65)),ie.unmap(),{handflag:new Float32Array(jt),handedness:new Float32Array(Jt),landmarks:new Float32Array(Qt)}}async function Ta(r){if(!Zt||!da||!Vt)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);let c=e.createCommandEncoder();{let U=c.beginComputePass();U.setPipeline(Le),U.setBindGroup(0,tt),U.dispatchWorkgroups(16,16,1),U.end()}Ve(c,null);let u=Vt.getCurrentTexture(),o=c.beginRenderPass({colorAttachments:[{view:u.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});o.setPipeline(Zt),o.setBindGroup(0,da),o.draw(3),o.end(),e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),ca.drawImage(Wa,0,0);let L=ca.getImageData(0,0,9,8).data,H=new Float32Array(65),R=new DataView(new ArrayBuffer(4));for(let U=0;U<65;U++){let g=U*4;R.setUint8(0,L[g]),R.setUint8(1,L[g+1]),R.setUint8(2,L[g+2]),R.setUint8(3,L[g+3]),H[U]=R.getFloat32(0)}return{handflag:new Float32Array([H[0]]),handedness:new Float32Array([H[1]]),landmarks:new Float32Array(H.subarray(2,65))}}let Mn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),_a=0,En=[ie,Mn],Ct=null,at=null;async function ma(r){let c=En[_a];_a=1-_a,e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);let u=e.createCommandEncoder();{let a=u.beginComputePass();a.setPipeline(Le),a.setBindGroup(0,tt),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}Ve(u,c),e.queue.submit([u.finish()]);let o=null;if(Ct!==null&&at!==null){await Ct;let a=new Float32Array(at.getMappedRange());o={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))},at.unmap()}return at=c,Ct=c.mapAsync(GPUMapMode.READ),o}async function La(){if(!Ct||!at)return null;await Ct;let r=new Float32Array(at.getMappedRange()),c={handflag:new Float32Array([r[0]]),handedness:new Float32Array([r[1]]),landmarks:new Float32Array(r.subarray(2,65))};return at.unmap(),Ct=null,at=null,c}async function Hn(r=50){let c=new Float32Array(196608);for(let a=0;a<5;a++)await ea(c);let u=[];for(let a=0;a<r;a++){let L=performance.now();await ea(c),u.push(performance.now()-L)}let o=u.reduce((a,L)=>a+L,0)/u.length;return{avgMs:o,fps:1e3/o}}async function Wn(r=50){let c=new Float32Array(196608);for(let H=0;H<5;H++)await ea(c);let u=[];for(let H=0;H<r;H++){let R=e.createCommandEncoder();{let g=R.beginComputePass();g.setPipeline(xt),g.setBindGroup(0,Da),g.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),g.end()}Ve(R,ie);let U=performance.now();e.queue.submit([R.finish()]),await e.queue.onSubmittedWorkDone(),u.push(performance.now()-U)}u.sort((H,R)=>H-R);let o=u.reduce((H,R)=>H+R,0)/u.length,a=u[Math.floor(u.length/2)],L=u[0];return{avgMs:o,fps:1e3/o,medianMs:a,minMs:L}}function oi(r){e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);let c=e.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(Le),u.setBindGroup(0,tt),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),u.end()}Ve(c,ie),e.queue.submit([c.finish()])}async function Tn(r,c=50){function u(g){let p=[...g].sort((X,ue)=>X-ue);return{median:p[Math.floor(p.length/2)],min:p[0]}}for(let g=0;g<10;g++)await la(r);let o=[];for(let g=0;g<c;g++){e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);let p=e.createCommandEncoder();{let ue=p.beginComputePass();ue.setPipeline(Le),ue.setBindGroup(0,tt),ue.dispatchWorkgroups(16,16,1),ue.end()}Ve(p,ie);let X=performance.now();e.queue.submit([p.finish()]),await e.queue.onSubmittedWorkDone(),o.push(performance.now()-X)}let a=[];for(let g=0;g<c;g++){e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);let p=e.createCommandEncoder();{let _e=p.beginComputePass();_e.setPipeline(Le),_e.setBindGroup(0,tt),_e.dispatchWorkgroups(16,16,1),_e.end()}Ve(p,ie),e.queue.submit([p.finish()]);let X=ie.mapAsync(GPUMapMode.READ),ue=performance.now();await e.queue.onSubmittedWorkDone(),await X,ie.getMappedRange(),ie.unmap(),a.push(performance.now()-ue)}let L=[];for(let g=0;g<c;g++){e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);let p=e.createCommandEncoder();{let ue=p.beginComputePass();ue.setPipeline(Le),ue.setBindGroup(0,tt),ue.dispatchWorkgroups(16,16,1),ue.end()}Ve(p,ie),e.queue.submit([p.finish()]);let X=performance.now();await ie.mapAsync(GPUMapMode.READ),ie.getMappedRange(),ie.unmap(),L.push(performance.now()-X)}let H=[];for(let g=0;g<c;g++){let p=performance.now();await la(r),H.push(performance.now()-p)}await ma(r);let R=[];for(let g=0;g<c;g++){let p=performance.now();await ma(r),R.push(performance.now()-p)}await La();let U=null;if(Zt){let g=[];for(let p=0;p<c;p++){let X=performance.now();await Ta(r),g.push(performance.now()-X)}U=u(g)}return{gpuOnly:u(o),mapAsyncOnly:u(a),mapAsyncNoWait:u(L),total:u(H),pipelined:u(R),renderReadback:U}}async function Ln(r){let c=[];async function u(a,L,H){let R=e.createBuffer({size:L,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),U=e.createCommandEncoder();U.copyBufferToBuffer(a,0,R,0,L),e.queue.submit([U.finish()]),await e.queue.onSubmittedWorkDone(),await R.mapAsync(GPUMapMode.READ);let g=new Float32Array(R.getMappedRange()),p=1/0,X=-1/0,ue=0;for(let _e=0;_e<g.length;_e++)g[_e]<p&&(p=g[_e]),g[_e]>X&&(X=g[_e]),g[_e]!==0&&ue++;R.unmap(),R.destroy(),c.push({layer:H,stats:{min:p,max:X,nonZero:ue,total:g.length}})}e.queue.copyExternalImageToTexture({source:r},{texture:de},[256,256]);{let a=e.createCommandEncoder(),L=a.beginComputePass();L.setPipeline(Le),L.setBindGroup(0,tt),L.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),L.end(),e.queue.submit([a.finish()])}await u(ze,Math.min(ze.size,3*257*257*4),"canvas\u2192bufInput");{let a=e.createCommandEncoder(),L=a.beginComputePass();L.setPipeline(et),L.setBindGroup(0,Ma),L.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),L.end(),e.queue.submit([a.finish()])}await u(te,Math.min(te.size,3072*128*4),"inputConv\u2192bufA");let o=!0;for(let a=0;a<Math.min(_t.length,6);a++){let L=o?qe[a]:Ye[a],H=o?$e[a]:Xe[a],R=Te[a],U=_t[a];{let p=e.createCommandEncoder(),X=p.beginComputePass();X.setPipeline(R.dwPipeline),X.setBindGroup(0,L),X.dispatchWorkgroups(R.dwDispatchX,R.dwDispatchY,R.dwDispatchZ),X.end(),e.queue.submit([p.finish()])}await u(Re,Math.min(Re.size,U.spec.inCh*U.outH*U.outW*4),`layer${a}.DW\u2192bufDW (${U.spec.prefix})`);{let p=e.createCommandEncoder(),X=p.beginComputePass();X.setPipeline(R.pwPipeline),X.setBindGroup(0,H),X.dispatchWorkgroups(R.pwDispatchX,R.pwDispatchY,R.pwDispatchZ),X.end(),e.queue.submit([p.finish()])}let g=o?fe:te;await u(g,Math.min(g.size,U.spec.outCh*U.outH*U.outW*4),`layer${a}.PW\u2192buf${o?"B":"A"} (${U.spec.prefix})`),o=!o}return c}return{device:e,run:ea,runFromCanvas:la,runFromCanvasViaRender:Ta,runFromCanvasPipelined:ma,flushPipelined:La,benchmark:Hn,benchmarkGPU:Wn,benchmarkDiagnostic:Tn,debugLayerOutputs:Ln}}function mt(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var cn=mt(`
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
`),dn=mt(`
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
`),ln=mt(`
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
`),_n=mt(`
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
`),mn=mt(`
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
`),fn=mt(`
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
`),hn=mt(`
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
`);async function ga(n,l){let t;if(l)t=l;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");t=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8)}})}let P={r:"read-only-storage",s:"storage",u:"uniform"};function h(d){return t.createBindGroupLayout({entries:d.map((i,x)=>i==="t"?{binding:x,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:x,visibility:GPUShaderStage.COMPUTE,buffer:{type:P[i]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,E=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function _(d,i){return t.createBuffer({size:Math.max(d,4),usage:i})}function b(d,i,x){t.queue.writeBuffer(d,i,x)}function B(d){let i=_(d.data.byteLength,e);return b(i,0,d.data),i}let S=Array.from(n.keys());function N(d){let i=n.get(d);if(!i)throw new Error(`Weight not found: ${d}`);return i}function m(...d){let i=S.find(x=>d.every(M=>x.includes(M)));if(!i)throw new Error(`Weight not found for: ${d.join(", ")}`);return N(i)}function O(d){let[,i,x,M]=d.shape,w=new Float32Array(M*25);for(let s=0;s<M;s++)for(let F=0;F<i;F++)for(let se=0;se<x;se++)w[s*25+F*5+se]=d.data[F*x*M+se*M+s];return w}function G(d){let[i,,,x]=d.shape,M=new Float32Array(i*x);for(let w=0;w<i;w++)for(let s=0;s<x;s++)M[w*x+s]=d.data[w*x+s];return M}let me=t.createShaderModule({code:cn}),ee=t.createShaderModule({code:dn}),ge=t.createShaderModule({code:ln}),pe=t.createShaderModule({code:_n}),T=t.createShaderModule({code:fn}),ae=t.createShaderModule({code:mn}),oe=t.createShaderModule({code:hn}),De=h(["r","r","r","r","s","u"]),Me=h(["r","r","r","s","u"]),I=h(["r","r","r","r","r","s","u"]),z=h(["r","r","r","s","u"]),Q=h(["r","r","r","r","s","u"]),V=h(["r","r","s","u"]),Be=h(["t","s","u"]);function ye(d,i){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:i,entryPoint:"main"}})}let ke=ye(De,me),Mt=ye(Me,ee),k=ye(I,ge),j=ye(z,pe),he=ye(Q,T),$=ye(V,ae),ce=ye(Be,oe),ne=m("conv2d/Conv2D"),Ue=m("batch_normalization/","conv2d/Conv2D"),je=m("p_re_lu/"),be=B(ne),Ge=B(Ue),Ie=B(je),Ee=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(d=>{let i=m(d.dwKey),x=m(d.pwKey),M=m(d.bnKey),w=m(d.preluKey),s=O(i),F=_(s.byteLength,e);b(F,0,s);let se=new Float32Array(d.inCh),f=_(se.byteLength,e);b(f,0,se);let D=G(x),v=_(D.byteLength,e);b(v,0,D);let Y=B(M),W=B(w);return{dwWeightBuf:F,dwBiasBuf:f,pwWeightBuf:v,pwBiasBuf:Y,alphaBuf:W,inCh:d.inCh,outCh:d.outCh,stride:d.stride,inH:d.inH}}),Ce=G(m("conv2d_20/Conv2D")),Ae=_(Ce.byteLength,e);b(Ae,0,Ce);let Pe=B(m("batch_normalization_20/")),He=B(m("p_re_lu_20/")),We={dwWeightBuf:(()=>{let d=O(m("depthwise_conv2d_19/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(256),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(m("conv2d_21/")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(m("batch_normalization_21/")),alphaBuf:B(m("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},xe={dwWeightBuf:(()=>{let d=O(m("depthwise_conv2d_20/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(256),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(m("conv2d_22/Conv2D1")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(m("batch_normalization_22/")),alphaBuf:B(m("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},ht=G(m("conv2d_23/Conv2D")),it=_(ht.byteLength,e);b(it,0,ht);let Fe=B(m("batch_normalization_23/")),Qe=B(m("p_re_lu_23/")),zt={dwWeightBuf:(()=>{let d=O(m("depthwise_conv2d_21/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(128),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(m("conv2d_24/")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(m("batch_normalization_24/")),alphaBuf:B(m("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},Kt={dwWeightBuf:(()=>{let d=O(m("depthwise_conv2d_22/")),i=_(d.byteLength,e);return b(i,0,d),i})(),dwBiasBuf:(()=>{let d=new Float32Array(128),i=_(d.byteLength,e);return b(i,0,d),i})(),pwWeightBuf:(()=>{let d=G(m("conv2d_25/Conv2D1")),i=_(d.byteLength,e);return b(i,0,d),i})(),pwBiasBuf:B(m("batch_normalization_25/")),alphaBuf:B(m("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},Nt=G(m("classifier_palm_16_NO_PRUNING/Conv2D")),bt=_(Nt.byteLength,e);b(bt,0,Nt);let wt=B(m("classifier_palm_16_NO_PRUNING/BiasAdd")),Et=G(m("regressor_palm_16_NO_PRUNING/Conv2D")),gt=_(Et.byteLength,e);b(gt,0,Et);let Ht=B(m("regressor_palm_16_NO_PRUNING/BiasAdd")),rt=G(m("classifier_palm_8_NO_PRUNING/Conv2D")),yt=_(rt.byteLength,e);b(yt,0,rt);let Wt=B(m("classifier_palm_8_NO_PRUNING/BiasAdd")),qt=G(m("regressor_palm_8_NO_PRUNING/Conv2D")),Tt=_(qt.byteLength,e);b(Tt,0,qt);let Te=B(m("regressor_palm_8_NO_PRUNING/BiasAdd")),xt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,et=_(36864*3*4,e),Ke=_(xt,y),$t=_(xt,y),Le=_(xt,y),vt=_(576*256*4,y),st=_(144*256*4,y|GPUBufferUsage.COPY_DST),Ne=_(576*128*4,y|GPUBufferUsage.COPY_DST),ot=_(864*4,E),ze=_(15552*4,E),ut=_(576*2*4,E),te=_(576*36*4,E),fe=_(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Re=_(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pt=_(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=_(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),dt=t.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function K(d,i){return Math.ceil(d/i)}function re(d){let i=_(d.byteLength,A);return b(i,0,d),i}let ie=re(new Uint32Array([1,3,32,192,192,96,96]));function de(d,i,x,M,w){let s=i.stride===2?i.inH/2:i.inH,F=s,se=i.stride===2?1:2,f=re(new Uint32Array([1,i.inCh,i.inH,i.inH,s,F,i.stride,se])),D=t.createBindGroup({layout:Me,entries:[{binding:0,resource:{buffer:x}},{binding:1,resource:{buffer:i.dwWeightBuf}},{binding:2,resource:{buffer:i.dwBiasBuf}},{binding:3,resource:{buffer:Le}},{binding:4,resource:{buffer:f}}]}),v=d.beginComputePass();v.setPipeline(Mt),v.setBindGroup(0,D),v.dispatchWorkgroups(K(F,8),K(s,8),i.inCh),v.end();let Y=i.inCh,W=re(new Uint32Array([1,i.inCh,i.outCh,s,F,Y,i.stride,i.inH,i.inH])),ve=t.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:Le}},{binding:1,resource:{buffer:w}},{binding:2,resource:{buffer:i.pwWeightBuf}},{binding:3,resource:{buffer:i.pwBiasBuf}},{binding:4,resource:{buffer:i.alphaBuf}},{binding:5,resource:{buffer:M}},{binding:6,resource:{buffer:W}}]}),le=d.beginComputePass();le.setPipeline(k),le.setBindGroup(0,ve),le.dispatchWorkgroups(K(F,8),K(s,8),i.outCh),le.end()}function Oe(d,i,x,M,w,s,F,se,f){let D=re(new Uint32Array([1,s,F,se,f])),v=t.createBindGroup({layout:z,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:D}}]}),Y=d.beginComputePass();Y.setPipeline(j),Y.setBindGroup(0,v),Y.dispatchWorkgroups(K(f,8),K(se,8),F),Y.end()}function lt(d,i,x,M,w,s,F,se,f,D){let v=re(new Uint32Array([1,F,se,f,D])),Y=t.createBindGroup({layout:Q,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:s}},{binding:5,resource:{buffer:v}}]}),W=d.beginComputePass();W.setPipeline(he),W.setBindGroup(0,Y),W.dispatchWorkgroups(K(D,8),K(f,8),se),W.end()}async function Yt(d){t.queue.copyExternalImageToTexture({source:d},{texture:dt},[192,192]);let i=re(new Uint32Array([192,192,192])),x=t.createBindGroup({layout:Be,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:et}},{binding:2,resource:{buffer:i}}]}),M=t.createCommandEncoder();{let C=M.beginComputePass();C.setPipeline(ce),C.setBindGroup(0,x),C.dispatchWorkgroups(K(192,16),K(192,16),1),C.end()}{let C=t.createBindGroup({layout:De,entries:[{binding:0,resource:{buffer:et}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:Ge}},{binding:3,resource:{buffer:Ie}},{binding:4,resource:{buffer:Ke}},{binding:5,resource:{buffer:ie}}]}),J=M.beginComputePass();J.setPipeline(ke),J.setBindGroup(0,C),J.dispatchWorkgroups(K(96,8),K(96,8),32),J.end()}let w=Ke,s=$t;for(let C=0;C<Ee.length;C++){let J=Ee[C];de(M,J,w,s,w);let q=w;w=s,s=q,C===10&&M.copyBufferToBuffer(w,0,Ne,0,576*128*4),C===14&&M.copyBufferToBuffer(w,0,st,0,144*256*4)}{let C=re(new Uint32Array([1,256,6,6,12,12])),J=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),q=M.beginComputePass();q.setPipeline($),q.setBindGroup(0,J),q.dispatchWorkgroups(K(12,8),K(12,8),256),q.end()}{let C=w;w=s,s=C}lt(M,w,Ae,Pe,He,s,256,256,12,12);{let C=w;w=s,s=C}{let C=re(new Uint32Array([1,256,12,12,12,12])),J=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),q=M.beginComputePass();q.setPipeline($),q.setBindGroup(0,J),q.dispatchWorkgroups(K(12,8),K(12,8),256),q.end()}{let C=w;w=s,s=C}de(M,We,w,s,w);{let C=w;w=s,s=C}de(M,xe,w,s,w);{let C=w;w=s,s=C}Oe(M,w,bt,wt,ot,256,6,12,12),Oe(M,w,gt,Ht,ze,256,108,12,12);{let C=re(new Uint32Array([1,256,12,12,24,24])),J=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),q=M.beginComputePass();q.setPipeline($),q.setBindGroup(0,J),q.dispatchWorkgroups(K(24,8),K(24,8),256),q.end()}{let C=w;w=s,s=C}lt(M,w,it,Fe,Qe,s,256,128,24,24);{let C=w;w=s,s=C}{let C=re(new Uint32Array([1,128,24,24,24,24])),J=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:w}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:C}}]}),q=M.beginComputePass();q.setPipeline($),q.setBindGroup(0,J),q.dispatchWorkgroups(K(24,8),K(24,8),128),q.end()}{let C=w;w=s,s=C}de(M,zt,w,s,w);{let C=w;w=s,s=C}de(M,Kt,w,s,w);{let C=w;w=s,s=C}Oe(M,w,yt,Wt,ut,128,2,24,24),Oe(M,w,Tt,Te,te,128,36,24,24),t.queue.submit([M.finish()]);let F=t.createCommandEncoder();F.copyBufferToBuffer(ot,0,fe,0,864*4),F.copyBufferToBuffer(ze,0,Re,0,15552*4),F.copyBufferToBuffer(ut,0,pt,0,576*2*4),F.copyBufferToBuffer(te,0,ct,0,576*36*4),t.queue.submit([F.finish()]),await Promise.all([fe.mapAsync(GPUMapMode.READ),Re.mapAsync(GPUMapMode.READ),pt.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ)]);let se=new Float32Array(fe.getMappedRange()).slice(),f=new Float32Array(Re.getMappedRange()).slice(),D=new Float32Array(pt.getMappedRange()).slice(),v=new Float32Array(ct.getMappedRange()).slice();fe.unmap(),Re.unmap(),pt.unmap(),ct.unmap();let Y=2016,W=new Float32Array(Y),ve=new Float32Array(Y*18),le=0;for(let C=0;C<12;C++)for(let J=0;J<12;J++)for(let q=0;q<6;q++){W[le]=se[q*144+C*12+J];for(let Se=0;Se<18;Se++){let Pt=q*18+Se;ve[le*18+Se]=f[Pt*144+C*12+J]}le++}for(let C=0;C<24;C++)for(let J=0;J<24;J++)for(let q=0;q<2;q++){W[le]=D[q*576+C*24+J];for(let Se=0;Se<18;Se++){let Pt=q*18+Se;ve[le*18+Se]=v[Pt*576+C*24+J]}le++}return{scores:W,regressors:ve}}async function Z(d,i){let x=t.createBuffer({size:i*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),M=t.createCommandEncoder();M.copyBufferToBuffer(d,0,x,0,i*4),t.queue.submit([M.finish()]),await x.mapAsync(GPUMapMode.READ);let w=new Float32Array(x.getMappedRange()).slice();return x.unmap(),x.destroy(),w}async function Xt(d){t.queue.copyExternalImageToTexture({source:d},{texture:dt},[192,192]);function i(v,Y=1e3){let W=v.slice(0,Y);return{min:Math.min(...W),max:Math.max(...W),mean:W.reduce((ve,le)=>ve+le,0)/W.length,nonZero:W.filter(ve=>ve!==0).length,sample:Array.from(W.slice(0,10))}}let x={},M=re(new Uint32Array([192,192,192])),w=t.createBindGroup({layout:Be,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:et}},{binding:2,resource:{buffer:M}}]}),s=t.createCommandEncoder(),F=s.beginComputePass();F.setPipeline(ce),F.setBindGroup(0,w),F.dispatchWorkgroups(K(192,16),K(192,16),1),F.end(),t.queue.submit([s.finish()]),x.input=i(await Z(et,36864*3)),s=t.createCommandEncoder();let se=t.createBindGroup({layout:De,entries:[{binding:0,resource:{buffer:et}},{binding:1,resource:{buffer:be}},{binding:2,resource:{buffer:Ge}},{binding:3,resource:{buffer:Ie}},{binding:4,resource:{buffer:Ke}},{binding:5,resource:{buffer:ie}}]});F=s.beginComputePass(),F.setPipeline(ke),F.setBindGroup(0,se),F.dispatchWorkgroups(K(96,8),K(96,8),32),F.end(),t.queue.submit([s.finish()]),x.initConv=i(await Z(Ke,9216*32));let f=Ke,D=$t;for(let v=0;v<Ee.length;v++){let Y=Ee[v];s=t.createCommandEncoder(),de(s,Y,f,D,f),t.queue.submit([s.finish()]);let W=f;if(f=D,D=W,v===0||v===3||v===7||v===11||v===14||v===15||v===18){let ve=Y.stride===2?Y.inH/2:Y.inH,le=ve*ve*Y.outCh;x[`block${v}`]=i(await Z(f,le))}v===10&&(s=t.createCommandEncoder(),s.copyBufferToBuffer(f,0,Ne,0,576*128*4),t.queue.submit([s.finish()])),v===14&&(s=t.createCommandEncoder(),s.copyBufferToBuffer(f,0,st,0,144*256*4),t.queue.submit([s.finish()]))}s=t.createCommandEncoder();{let v=re(new Uint32Array([1,256,6,6,12,12])),Y=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline($),W.setBindGroup(0,Y),W.dispatchWorkgroups(K(12,8),K(12,8),256),W.end()}t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpnUpsample6to12=i(await Z(f,144*256)),s=t.createCommandEncoder(),lt(s,f,Ae,Pe,He,D,256,256,12,12),t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpn6to12Conv=i(await Z(f,144*256)),x.backbone12Skip=i(await Z(st,144*256)),s=t.createCommandEncoder();{let v=re(new Uint32Array([1,256,12,12,12,12])),Y=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline($),W.setBindGroup(0,Y),W.dispatchWorkgroups(K(12,8),K(12,8),256),W.end()}t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpnAdd12=i(await Z(f,144*256)),s=t.createCommandEncoder(),de(s,We,f,D,f),t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpn12Block1=i(await Z(f,144*256)),s=t.createCommandEncoder(),de(s,xe,f,D,f),t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpn12Block2=i(await Z(f,144*256)),s=t.createCommandEncoder(),Oe(s,f,bt,wt,ot,256,6,12,12),t.queue.submit([s.finish()]),x.cls16=i(await Z(ot,864)),s=t.createCommandEncoder(),Oe(s,f,gt,Ht,ze,256,108,12,12),t.queue.submit([s.finish()]),x.reg16=i(await Z(ze,15552),500),s=t.createCommandEncoder();{let v=re(new Uint32Array([1,256,12,12,24,24])),Y=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:vt}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline($),W.setBindGroup(0,Y),W.dispatchWorkgroups(K(24,8),K(24,8),256),W.end()}t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpnUpsample12to24=i(await Z(f,576*256)),s=t.createCommandEncoder(),lt(s,f,it,Fe,Qe,D,256,128,24,24),t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpn12to24Conv=i(await Z(f,576*128)),x.backbone24Skip=i(await Z(Ne,576*128)),s=t.createCommandEncoder();{let v=re(new Uint32Array([1,128,24,24,24,24])),Y=t.createBindGroup({layout:V,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:D}},{binding:3,resource:{buffer:v}}]}),W=s.beginComputePass();W.setPipeline($),W.setBindGroup(0,Y),W.dispatchWorkgroups(K(24,8),K(24,8),128),W.end()}t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpnAdd24=i(await Z(f,576*128)),s=t.createCommandEncoder(),de(s,zt,f,D,f),t.queue.submit([s.finish()]);{let v=f;f=D,D=v}x.fpn24Block1=i(await Z(f,576*128)),s=t.createCommandEncoder(),de(s,Kt,f,D,f),t.queue.submit([s.finish()]);{let v=f;f=D,D=v}return x.fpn24Block2=i(await Z(f,576*128)),s=t.createCommandEncoder(),Oe(s,f,yt,Wt,ut,128,2,24,24),t.queue.submit([s.finish()]),x.cls8=i(await Z(ut,576*2)),s=t.createCommandEncoder(),Oe(s,f,Tt,Te,te,128,36,24,24),t.queue.submit([s.finish()]),x.reg8=i(await Z(te,576*36)),x.initWeights=i(await Z(be,100),100),x.initBias=i(await Z(Ge,32),32),x.cls16Weights=i(await Z(bt,100),100),x.cls16Bias=i(await Z(wt,6),6),x.cls8Weights=i(await Z(yt,100),100),x.cls8Bias=i(await Z(Wt,2),2),x.fpn6to12Weights=i(await Z(Ae,100),100),x}return{device:t,run:Yt,debugRun:Xt}}function Zn(){let n=[];for(let l=0;l<12;l++)for(let t=0;t<12;t++){let P=(t+.5)/12,h=(l+.5)/12;for(let e=0;e<6;e++)n.push({x:P,y:h})}for(let l=0;l<24;l++)for(let t=0;t<24;t++){let P=(t+.5)/24,h=(l+.5)/24;for(let e=0;e<2;e++)n.push({x:P,y:h})}return n}var bn=Zn();function jn(n){return 1/(1+Math.exp(-n))}function wn(n,l){let t=[],{scores:P,regressors:h}=n,e=192;for(let y=0;y<bn.length;y++){let E=jn(P[y]);if(E<l)continue;let A=bn[y],_=y*18,b=A.x+h[_+0]/e,B=A.y+h[_+1]/e,S=h[_+2]/e,N=h[_+3]/e,m=[];for(let O=0;O<7;O++){let G=A.x+h[_+4+O*2]/e,me=A.y+h[_+4+O*2+1]/e;m.push([G,me])}t.push({score:E,box:[b,B,S,N],keypoints:m})}return t}function gn(n,l){if(n.length===0)return[];let t=[...n].sort((e,y)=>y.score-e.score),P=[],h=new Set;for(let e=0;e<t.length;e++)if(!h.has(e)){P.push(t[e]);for(let y=e+1;y<t.length;y++)h.has(y)||Jn(t[e],t[y])>l&&h.add(y)}return P}function Jn(n,l){let t=n.box[0]-n.box[2]/2,P=n.box[1]-n.box[3]/2,h=n.box[0]+n.box[2]/2,e=n.box[1]+n.box[3]/2,y=l.box[0]-l.box[2]/2,E=l.box[1]-l.box[3]/2,A=l.box[0]+l.box[2]/2,_=l.box[1]+l.box[3]/2,b=Math.max(t,y),B=Math.max(P,E),S=Math.min(h,A),N=Math.min(e,_),m=Math.max(0,S-b),O=Math.max(0,N-B),G=m*O,me=(h-t)*(e-P),ee=(A-y)*(_-E),ge=me+ee-G;return ge>0?G/ge:0}function Qn(n){let[l,t,P,h]=n.box,e=n.keypoints[0],y=n.keypoints[2],E=y[0]-e[0],A=y[1]-e[1],_=Math.atan2(A,E),B=-Math.PI/2-_,S=Math.max(P,h),m=S*2.6,O=-.5*S,G=Math.cos(B),me=Math.sin(B),ee=-O*me,ge=O*G;return{centerX:l+ee,centerY:t+ge,width:m,height:m,rotation:B}}function ya(n,l={}){let{scoreThreshold:t=.5,nmsThreshold:P=.3,maxHands:h=2}=l;async function e(E){let A=await n.run(E),_=wn(A,t);return gn(_,P).slice(0,h).map(Qn)}async function y(E){let A=await n.run(E),_=wn(A,t);return gn(_,P).slice(0,h)}return{detect:e,detectRaw:y,model:n}}function yn(n,l=256){let t=Math.cos(n.rotation),P=Math.sin(n.rotation),h=n.width/l,e=n.height/l,y=h*t,E=-e*P,A=h*P,_=e*t,b=n.centerX-(y*l/2+E*l/2),B=n.centerY-(A*l/2+_*l/2),S=y*_-E*A,N=_/S,m=-E/S,O=-A/S,G=y/S,me=-(N*b+m*B),ee=-(O*b+G*B);return{forward:[y,E,b,A,_,B],inverse:[N,m,me,O,G,ee]}}function xa(n,l){let{forward:t}=yn(l,1),[P,h,e,y,E,A]=t;return n.map(_=>({x:P*_.x+h*_.y+e,y:y*_.x+E*_.y+A,z:_.z}))}var va=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function ft(n){let l={};for(let t=0;t<va.length;t++)l[va[t]]=n[t];return l}function xn(n,l,t){return n.initialized?(n.value=t*l+(1-t)*n.value,n.value):(n.value=l,n.initialized=!0,l)}function vn(n,l){let t=2*Math.PI*l*n;return t/(t+1)}function ei(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function Pa(n,l,t,P,h,e){let y=n.lastTime<0?.03333333333333333:t-n.lastTime;n.lastTime=t;let E=vn(y,e),A=n.x.initialized?(l-n.x.value)/y:0,_=xn(n.dx,A,E),b=P+h*Math.abs(_),B=vn(y,b);return xn(n.x,l,B)}function ia(n={}){let{minCutoff:l=1,beta:t=10,dCutoff:P=1}=n,h=[];function e(A){h.length!==A&&(h=Array.from({length:A},()=>ei()))}function y(A,_){let b=_??performance.now()/1e3,B=A.length*3;return e(B),A.map((S,N)=>({x:Pa(h[N*3],S.x,b,l,t,P),y:Pa(h[N*3+1],S.y,b,l,t,P),z:Pa(h[N*3+2],S.z,b,l,t,P)}))}function E(){h=[]}return{apply:y,reset:E}}var Pn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ti(n={}){let{weightsUrl:l,scoreThreshold:t=.5,forceF32:P=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let h=l??Pn,e=h.endsWith("/")?h:`${h}/`,y=`${e}weights_f16.json`,E=`${e}weights_f16.bin`,[A,_]=await Promise.all([fetch(y),fetch(E)]);if(!A.ok)throw new Error(`Failed to fetch weights metadata: ${A.status}`);if(!_.ok)throw new Error(`Failed to fetch weights binary: ${_.status}`);let b=await A.json(),B=await _.arrayBuffer(),S=It(b,B),N=await Ft(S,{forceF32:P});if(!P){let I=new OffscreenCanvas(256,256),z=I.getContext("2d");z.fillStyle="#886644",z.fillRect(0,0,256,256),z.fillStyle="#cc9966",z.fillRect(50,50,156,156);let Q=await N.runFromCanvas(I);Q.landmarks.every(Be=>Be===0)&&Q.handflag.every(Be=>Be===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),N.device.destroy(),N=await Ft(S,{forceF32:!0}))}let m=ia(),O=!1,G=null;function me(){return G||(G=new OffscreenCanvas(256,256)),G}async function ee(I){if(I instanceof HTMLCanvasElement||I instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&I instanceof ImageBitmap)return I;let z=me();z.width=256,z.height=256;let Q=z.getContext("2d");return I instanceof ImageData?Q.putImageData(I,0,0):Q.drawImage(I,0,0,256,256),z}function ge(I,z,Q){let V=I[0];if(V<t)return null;let Be=z[0]>.5,ye=[];for(let ke=0;ke<21;ke++)ye.push({x:Q[ke*3],y:Q[ke*3+1],z:Q[ke*3+2]});return{score:V,handedness:Be?"right":"left",landmarks:ye,keypoints:ft(ye)}}async function pe(I){let z=await ee(I),Q=await N.runFromCanvas(z),V=ge(Q.handflag,Q.handedness,Q.landmarks);return V?(O=!0,V.landmarks=m.apply(V.landmarks),V.keypoints=ft(V.landmarks),V):(O&&(m.reset(),O=!1),null)}async function T(I){let z=await ee(I),Q=await N.runFromCanvasPipelined(z);if(!Q)return null;let V=ge(Q.handflag,Q.handedness,Q.landmarks);return V?(O=!0,V.landmarks=m.apply(V.landmarks),V.keypoints=ft(V.landmarks),V):(O&&(m.reset(),O=!1),null)}async function ae(){let I=await N.flushPipelined();if(!I)return null;let z=ge(I.handflag,I.handedness,I.landmarks);return z?(O=!0,z.landmarks=m.apply(z.landmarks),z.keypoints=ft(z.landmarks),z):(O&&(m.reset(),O=!1),null)}function oe(){N.device.destroy(),G=null}async function De(I){let z=await ee(I);return N.benchmarkDiagnostic(z)}async function Me(I){let z=await ee(I);return N.debugLayerOutputs(z)}return{detect:pe,detectPipelined:T,flushPipelined:ae,dispose:oe,benchmarkDiagnostic:De,debugLayerOutputs:Me}}async function ai(n={}){let{weightsUrl:l,palmWeightsUrl:t,scoreThreshold:P=.5,palmScoreThreshold:h=.5,maxHands:e=3,forceF32:y=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let E=l??Pn,A=E.endsWith("/")?E:`${E}/`;if(!t)throw new Error("palmWeightsUrl is required for createFullHandpose");let _=t.endsWith("/")?t:`${t}/`,[b,B,S,N]=await Promise.all([fetch(`${A}weights_f16.json`),fetch(`${A}weights_f16.bin`),fetch(`${_}palm_detection_weights.json`),fetch(`${_}palm_detection_weights.bin`)]);if(!b.ok)throw new Error(`Failed to fetch landmark weights metadata: ${b.status}`);if(!B.ok)throw new Error(`Failed to fetch landmark weights binary: ${B.status}`);if(!S.ok)throw new Error(`Failed to fetch palm weights metadata: ${S.status}`);if(!N.ok)throw new Error(`Failed to fetch palm weights binary: ${N.status}`);let[m,O,G,me]=await Promise.all([b.json(),B.arrayBuffer(),S.json(),N.arrayBuffer()]),ee=It(m,O),ge=It(G,me),pe=await Ft(ee,{forceF32:y});if(!y){let k=new OffscreenCanvas(256,256),j=k.getContext("2d");j.fillStyle="#886644",j.fillRect(0,0,256,256),j.fillStyle="#cc9966",j.fillRect(50,50,156,156);let he=await pe.runFromCanvas(k);he.landmarks.every(ce=>ce===0)&&he.handflag.every(ce=>ce===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),pe.device.destroy(),pe=await Ft(ee,{forceF32:!0}))}let T=await ga(ge),ae=ya(T,{scoreThreshold:h,maxHands:e}),oe=[];for(let k=0;k<e;k++)oe.push(ia());let De=0,Me=null,I=null;function z(){return Me||(Me=new OffscreenCanvas(192,192)),Me}function Q(){return I||(I=new OffscreenCanvas(256,256)),I}async function V(k){if(k instanceof HTMLCanvasElement||k instanceof OffscreenCanvas){if(k.width===192&&k.height===192)return k;let $=z();return $.width=192,$.height=192,$.getContext("2d").drawImage(k,0,0,192,192),$}if(typeof ImageBitmap<"u"&&k instanceof ImageBitmap){if(k.width===192&&k.height===192)return k;let $=z();return $.width=192,$.height=192,$.getContext("2d").drawImage(k,0,0,192,192),$}let j=z();j.width=192,j.height=192;let he=j.getContext("2d");if(k instanceof ImageData){let $=new OffscreenCanvas(k.width,k.height);$.getContext("2d").putImageData(k,0,0),he.drawImage($,0,0,192,192)}else he.drawImage(k,0,0,192,192);return j}function Be(k,j,he,$){let ce=Q();ce.width=256,ce.height=256;let ne=ce.getContext("2d"),Ue=Math.cos(-j.rotation),je=Math.sin(-j.rotation);ne.clearRect(0,0,256,256),ne.save(),ne.translate(128,128),ne.scale(j.width*he/256,j.height*$/256),ne.rotate(-j.rotation),ne.translate(-128,-128);let be=j.centerX*he,Ge=j.centerY*$;ne.restore();let Ie=Math.min(he,$),Je=256/(j.width*Ie),Ee=256/(j.height*Ie),Ce=Math.cos(j.rotation),Ae=Math.sin(j.rotation),Pe=Ce*Je,He=Ae*Je,We=-Ae*Ee,xe=Ce*Ee,ht=-be*Pe-Ge*We+128,it=-be*He-Ge*xe+128;if(ne.setTransform(Pe,He,We,xe,ht,it),k instanceof ImageData){let Fe=new OffscreenCanvas(k.width,k.height);Fe.getContext("2d").putImageData(k,0,0),ne.drawImage(Fe,0,0)}else ne.drawImage(k,0,0);return ne.setTransform(1,0,0,1,0,0),ce}function ye(k){return k instanceof HTMLCanvasElement||k instanceof OffscreenCanvas?[k.width,k.height]:typeof ImageBitmap<"u"&&k instanceof ImageBitmap?[k.width,k.height]:k instanceof ImageData?[k.width,k.height]:k instanceof HTMLVideoElement?[k.videoWidth,k.videoHeight]:k instanceof HTMLImageElement?[k.naturalWidth,k.naturalHeight]:[256,256]}async function ke(k){let j=await V(k),he=await ae.detect(j);if(he.length===0)return[];let[$,ce]=ye(k),ne=[];for(let Ue of he){let je=Be(k,Ue,$,ce),be=await pe.runFromCanvas(je),Ge=be.handflag[0];if(Ge<P)continue;let Ie=be.handedness[0]>.5,Je=[];for(let xe=0;xe<21;xe++)Je.push({x:be.landmarks[xe*3],y:be.landmarks[xe*3+1],z:be.landmarks[xe*3+2]});let Ee=Math.min($,ce),Ce=Ue.width*Ee,Ae={...Ue,width:Ce/$,height:Ce/ce},Pe=xa(Je,Ae),He=ne.length,We=He<oe.length?oe[He].apply(Pe):Pe;ne.push({score:Ge,handedness:Ie?"right":"left",landmarks:We,keypoints:ft(We),palmScore:0})}if(ne.length<De)for(let Ue=ne.length;Ue<De;Ue++)Ue<oe.length&&oe[Ue].reset();return De=ne.length,ne}function Mt(){pe.device.destroy(),T.device.destroy(),Me=null,I=null}return{detect:ke,dispose:Mt}}function ni(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ii=ni(`
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
`);function ri(n){let l=n.createShaderModule({code:ii}),t=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),P=n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:l,entryPoint:"main"}});function h(e,y,E,A,_,b,B){let S=new Uint32Array([_,b,B,0]),N=n.createBuffer({size:S.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(N,0,S);let m=new Float32Array(A),O=new Float32Array(8);O.set(m);let G=n.createBuffer({size:O.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(G,0,O);let me=n.createBindGroup({layout:t,entries:[{binding:0,resource:y.createView()},{binding:1,resource:{buffer:E}},{binding:2,resource:{buffer:N}},{binding:3,resource:{buffer:G}}]}),ee=e.beginComputePass();ee.setPipeline(P),ee.setBindGroup(0,me),ee.dispatchWorkgroups(Math.ceil(B/16),Math.ceil(B/16),1),ee.end()}return{crop:h}}export{va as LANDMARK_NAMES,ga as compilePalmModel,yn as computeCropTransform,ri as createCropPipeline,ai as createFullHandpose,ti as createHandpose,ia as createLandmarkSmoother,ya as createPalmDetector,It as loadWeightsFromBuffer,xa as projectLandmarksToOriginal,ft as toKeypoints};
