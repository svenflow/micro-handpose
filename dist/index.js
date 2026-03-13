function _e(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function $n(o){let m=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+o;for(let U of m)for(;a.includes(`${U}:array<f32>`);)a=a.replace(`${U}:array<f32>`,`${U}:array<f16>`);return a}var Un=_e(`
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
`),An=_e(`
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
`),Gn=_e(`
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
`),Sn=_e(`
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
`);function Vn(o,m){return An.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function Zn(o,m){return Un.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function jn(o,m){return Gn.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function Jn(o,m){return Sn.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function Qn(o,m){return[8,8]}var ea=_e(`
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
`),ta=_e(`
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
`);function na(o){return _e(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${o?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${o?"+skip[out_idx]":""};
}
`)}var aa=na(!1),ia=na(!0),ra=_e(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),sa=_e(`
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
`);function oa(o){return _e(`
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
  ${o==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var ua=oa("sigmoid"),pa=oa("div256"),ca=_e(`
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
`),da=_e(`
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
`);function la(o,m){let U=Math.min(m,256),x=m>U,P=o%4===0?`var ic:u32=0u;
    while(ic<${o}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${o}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,I=`var skip_val:f32=0.0;
    if(c<${o}u){
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
    }`,W=o===m?"":`if(c<${o}u){`,f=o===m?"":"}",g=x?`for(var c:u32=lid.x;c<${o}u;c+=${U}u){`:`let c=lid.x;
  ${W}`,A=x?"}":f,L=x?`for(var c:u32=lid.x;c<${m}u;c+=${U}u){`:"{let c=lid.x;";return _e(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${o}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${U},1,1)
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
  ${A}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${L}
    let pw_base=c*${o}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${P}
    // Skip connection (only for c < inCh)
    ${I}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var _a=_e(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),ma=_e(`
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
`),fa=_e(`
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
`);function Mn(o,m){let a=new Map,U=o.dtype??"float32";for(let x=0;x<o.keys.length;x++){let t=o.keys[x],P=o.shapes[x],I=o.offsets[x],W=P.reduce((A,L)=>A*L,1),f,g;if(U==="float32")f=new Float32Array(m,I,W);else{let A=new DataView(m);f=new Float32Array(W);for(let L=0;L<W;L++)f[L]=Fa(A.getUint16(I+L*2,!0));g=m.slice(I,I+W*2)}a.set(t,{data:f,shape:P,rawF16:g})}return a}function Fa(o){let m=o>>15&1,a=o>>10&31,U=o&1023;if(a===0){if(U===0)return m?-0:0;let P=-14,I=U/1024;return(m?-1:1)*Math.pow(2,P)*I}if(a===31)return U===0?m?-1/0:1/0:NaN;let x=a-15,t=1+U/1024;return(m?-1:1)*Math.pow(2,x)*t}var Ka=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],ha=Ka.map(([o,m,a,U,x])=>({type:"resmodule",inCh:o,outCh:m,h:a,w:a,stride:U,prefix:x})),Na=2,qa=5,Ya=8,Xa=11;async function Dn(o,m){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let U=a.features.has("shader-f16"),x=U?["shader-f16"]:[],t=await a.requestDevice({requiredFeatures:x,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(a.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(a.limits.maxComputeInvocationsPerWorkgroup,288)}}),P=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(U)try{let r=`enable f16;
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
}`,d=`enable f16;
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
}`,p=t.createShaderModule({code:r}),u=t.createShaderModule({code:d}),i=await p.getCompilationInfo(),E=await u.getCompilationInfo();if(i.messages.some(S=>S.type==="error")||E.messages.some(S=>S.type==="error"))P=!1;else{let S=new Float32Array(2400);S.fill(1);let T=new Uint16Array(2400);T.fill(10516);let k=new Uint16Array(96);k.fill(14336);let w=new Uint16Array(9216);w.fill(8478);let c=new Uint16Array(96);c.fill(12288);let z=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,oe=t.createBuffer({size:S.byteLength,usage:z}),xt=t.createBuffer({size:T.byteLength,usage:z}),Pt=t.createBuffer({size:k.byteLength,usage:z}),vt=t.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),Bt=t.createBuffer({size:w.byteLength,usage:z}),Ct=t.createBuffer({size:c.byteLength,usage:z}),kt=t.createBuffer({size:384,usage:ae}),Ze=t.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});t.queue.writeBuffer(oe,0,S),t.queue.writeBuffer(xt,0,T),t.queue.writeBuffer(Pt,0,k),t.queue.writeBuffer(Bt,0,w),t.queue.writeBuffer(Ct,0,c);let Ne="read-only-storage",qt="storage",Yt=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ne}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ne}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ne}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:qt}}]}),Yn=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ne}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ne}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Ne}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:qt}}]}),Wa=t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[Yt]}),compute:{module:p,entryPoint:"main"}}),La=t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[Yn]}),compute:{module:u,entryPoint:"main"}}),Ra=t.createBindGroup({layout:Yt,entries:[{binding:0,resource:{buffer:oe}},{binding:1,resource:{buffer:xt}},{binding:2,resource:{buffer:Pt}},{binding:3,resource:{buffer:vt}}]}),Oa=t.createBindGroup({layout:Yn,entries:[{binding:0,resource:{buffer:vt}},{binding:1,resource:{buffer:Bt}},{binding:2,resource:{buffer:Ct}},{binding:3,resource:{buffer:kt}}]}),wn=t.createCommandEncoder(),gn=wn.beginComputePass();gn.setPipeline(Wa),gn.setBindGroup(0,Ra),gn.dispatchWorkgroups(2),gn.end();let yn=wn.beginComputePass();yn.setPipeline(La),yn.setBindGroup(0,Oa),yn.dispatchWorkgroups(2),yn.end(),wn.copyBufferToBuffer(kt,0,Ze,0,384),t.queue.submit([wn.finish()]),await t.queue.onSubmittedWorkDone(),await Ze.mapAsync(GPUMapMode.READ);let Xt=new Float32Array(Ze.getMappedRange()),Xn=1.5*.0104*96+.25,Ia=Xt[0]!==0&&Xt[47]!==0&&Xt[95]!==0,za=Math.abs(Xt[0]-Xn)<1;P=Ia&&za,Ze.unmap(),oe.destroy(),xt.destroy(),Pt.destroy(),vt.destroy(),Bt.destroy(),Ct.destroy(),kt.destroy(),Ze.destroy(),P||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Xt[0]}, expected ~${Xn.toFixed(2)}) \u2014 falling back to f32`)}}catch{P=!1}let W=o.values().next().value,f=P&&!!W?.rawF16&&!m?.forceF32;console.log(f?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${U}, f16 validated: ${P}, f16 data: ${!!W?.rawF16})`);function g(r){if(f&&r.rawF16){let d=new Uint16Array(r.rawF16);if(d.length%2!==0){let p=new Uint16Array(d.length+1);return p.set(d),p}return d}return r.data}function A(r){if(f&&r.rawF16){let d=r.rawF16.byteLength;return Math.ceil(d/4)*4}return r.data.byteLength}function L(r){return f?$n(r):r}let ue={r:"read-only-storage",s:"storage",u:"uniform"};function y(r){return t.createBindGroupLayout({entries:r.map((d,p)=>({binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:ue[d]}}))})}function ie(r){return t.createBindGroupLayout({entries:r.map((d,p)=>d==="t"?{binding:p,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:ue[d]}})})}let R=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,me=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,Ce=GPUBufferUsage.STORAGE,Me=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,re=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function G(r,d){return t.createBuffer({size:r,usage:d})}function Z(r,d){return t.createBindGroup({layout:r,entries:d.map((p,u)=>({binding:u,resource:"size"in p?{buffer:p}:p}))})}function pe(r,d){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[r]}),compute:{module:d,entryPoint:"main"}})}let Ut=t.createShaderModule({code:ea}),Je=t.createShaderModule({code:fa}),We=t.createShaderModule({code:L(ca)}),Le=t.createShaderModule({code:L(An)}),At=t.createShaderModule({code:L(Un)}),dt=t.createShaderModule({code:L(Gn)}),ge=t.createShaderModule({code:L(Sn)}),Qe=t.createShaderModule({code:L(ta)}),lt=t.createShaderModule({code:aa}),ke=t.createShaderModule({code:ra}),_t=t.createShaderModule({code:ia}),v=t.createShaderModule({code:L(sa)}),F=t.createShaderModule({code:L(ua)}),X=t.createShaderModule({code:L(pa)}),J=t.createShaderModule({code:L(da)}),K=new Map;function ye(r,d){let p=`${r}_${d}`,u=K.get(p);return u||(u=t.createShaderModule({code:L(la(r,d))}),K.set(p,u)),u}let ne=y(["r","r","r","s","u"]),$=y(["r","r","r","r","s","u"]),be=y(["r","s","u"]),Q=y(["r","r","r","s","u"]),fe=y(["r","s","u"]),de=y(["r","r","s","u"]),we=y(["r","r","s","u"]),Re=y(["r","r","r","s","u"]),se=y(["r","r","r","s","u"]),De=ie(["t","s","u"]),xe=y(["r","r","r","r","r","r","r","s"]),Ue=y(["r","r","r","r","r","s","u"]),Oe=t.createPipelineLayout({bindGroupLayouts:[ne]}),Pe=t.createPipelineLayout({bindGroupLayouts:[$]}),Ae=r=>t.createComputePipeline({layout:Oe,compute:{module:r,entryPoint:"main"}}),Ie=r=>t.createComputePipeline({layout:Pe,compute:{module:r,entryPoint:"main"}}),Fe=Ae(Le),et=Ae(At),tt=Ie(dt),nt=Ie(ge),qe=new Map,Gt=new Map,at=new Map,mt=new Map;qe.set("8,8",Fe),Gt.set("8,8",et),at.set("8,8",tt),mt.set("8,8",nt);function it(r,d,p,u,i){let E=`${d},${p}`,S=r.get(E);return S||(S=i(t.createShaderModule({code:L(u(d,p))})),r.set(E,S)),S}let St=(r,d)=>it(qe,r,d,Vn,Ae),$t=(r,d)=>it(Gt,r,d,Zn,Ae),Vt=(r,d)=>it(at,r,d,jn,Ie),ft=(r,d)=>it(mt,r,d,Jn,Ie),Ge=ha.map(r=>{let d=r.stride===2?r.h/2:r.h,p=r.stride===2?r.w/2:r.w,[u,i]=Qn(r.inCh,d),E=r.h>=64,S=d>=16&&r.inCh>=288&&r.outCh>=288&&r.outCh%2===0;return{dwPipeline:E?$t(u,i):St(u,i),pwPipeline:S?ft(u,i):Vt(u,i),dwDispatchX:Math.ceil(p/u),dwDispatchY:Math.ceil(d/i),dwDispatchZ:r.inCh,pwDispatchX:Math.ceil(p/u),pwDispatchY:Math.ceil(d/i),pwDispatchZ:S?r.outCh/2:r.outCh}}),Mt=pe(be,Ut),ht=pe(Q,Qe);pe(fe,lt),pe(de,ke);let bt=pe(we,_t),Dt=pe(Re,v);pe(se,F),pe(se,X);let he=pe(De,Je),rt=pe(xe,We),Zt=pe(Ue,J),wt=1*288*128*128*4,st=G(3*256*256*4,R),Ee=G(3*257*257*4,Ce),Ye=G(12,re);t.queue.writeBuffer(Ye,0,new Uint32Array([3,256,257]));let j=G(wt,me),ce=G(wt,Me),He=G(wt,Ce),ot=G(3072*64*4,R),ut=G(3072*32*4,R),pt=G(1536*16*4,R),ct=G(6144*64*4,Ce),Te=G(260,Me),V=G(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);G(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let M=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),q=G(8,re);t.queue.writeBuffer(q,0,new Uint32Array([256,257]));let Et=o.get("backbone1.1.weight"),Ht=o.get("backbone1.1.bias");if(!Et||!Ht)throw new Error("Missing input conv weights");let Tt=g(Et),Wt=g(Ht),Lt=G(Tt.byteLength,R),Rt=G(Wt.byteLength,R),jt=G(28,re);t.queue.writeBuffer(Lt,0,Tt),t.queue.writeBuffer(Rt,0,Wt),t.queue.writeBuffer(jt,0,new Uint32Array([1,3,24,257,257,128,128]));let Jt=o.get("backbone6.1.weight"),Qt=o.get("backbone6.1.bias");if(!Jt||!Qt)throw new Error("Missing backbone6.1 conv1x1 weights");let en=g(Jt),Ot=g(Qt),tn=G(en.byteLength,R),It=G(Ot.byteLength,R),zt=G(20,re);t.queue.writeBuffer(tn,0,en),t.queue.writeBuffer(It,0,Ot),t.queue.writeBuffer(zt,0,new Uint32Array([1,96,48,32,32]));let nn=o.get("handflag.weight"),an=o.get("handflag.bias");if(!nn||!an)throw new Error("Missing handflag weights");let rn=g(nn),sn=g(an),ze=G(rn.byteLength,R),gt=G(sn.byteLength,R),Ft=G(12,re);t.queue.writeBuffer(ze,0,rn),t.queue.writeBuffer(gt,0,sn),t.queue.writeBuffer(Ft,0,new Uint32Array([1,288,1]));let Kt=o.get("handedness.weight"),on=o.get("handedness.bias");if(!Kt||!on)throw new Error("Missing handedness weights");let un=g(Kt),ve=g(on),Se=G(un.byteLength,R),Xe=G(ve.byteLength,R),Nt=G(12,re);t.queue.writeBuffer(Se,0,un),t.queue.writeBuffer(Xe,0,ve),t.queue.writeBuffer(Nt,0,new Uint32Array([1,288,1]));let pn=o.get("reg_3d.weight"),cn=o.get("reg_3d.bias");if(!pn||!cn)throw new Error("Missing reg_3d weights");let Y=g(pn),dn=g(cn),n=G(Y.byteLength,R),e=G(dn.byteLength,R),s=G(12,re);t.queue.writeBuffer(n,0,Y),t.queue.writeBuffer(e,0,dn),t.queue.writeBuffer(s,0,new Uint32Array([1,288,63]));let H=ha.map(r=>{let{inCh:d,outCh:p,h:u,w:i,stride:E,prefix:S}=r,T=E===2?u/2:u,k=E===2?i/2:i,w=E===2?1:2,c=o.get(`${S}convs.0.weight`),z=o.get(`${S}convs.0.bias`),ae=o.get(`${S}convs.1.weight`),oe=o.get(`${S}convs.1.bias`);if(!c||!z||!ae||!oe)throw new Error(`Missing weights for ${S}`);let xt=g(c),Pt=g(z),vt=g(ae),Bt=g(oe),Ct=G(xt.byteLength,R),kt=G(Pt.byteLength,R),Ze=G(vt.byteLength,R),Ne=G(Bt.byteLength,R),qt=G(32,re),Yt=G(36,re);return t.queue.writeBuffer(Ct,0,xt),t.queue.writeBuffer(kt,0,Pt),t.queue.writeBuffer(Ze,0,vt),t.queue.writeBuffer(Ne,0,Bt),t.queue.writeBuffer(qt,0,new Uint32Array([1,d,u,i,T,k,E,w])),t.queue.writeBuffer(Yt,0,new Uint32Array([1,d,p,T,k,Math.max(0,p-d),E,u,i])),{dwWeight:Ct,dwBias:kt,pwWeight:Ze,pwBias:Ne,dwUniform:qt,pwUniform:Yt,spec:r,outH:T,outW:k}});function O(r){let d=G(r.length*4,re);return t.queue.writeBuffer(d,0,new Uint32Array(r)),d}let _=O([1,96,8,8,16,16]),N=O([1,96,16,16,32,32]),ee=O([1,48,32,32,64,64]);O([1536*16]),O([3072*32]),O([3072*64]);let b=Z(be,[st,Ee,Ye]),C=Z(Q,[Ee,Lt,Rt,j,jt]),h=[],D=[],l=[],B=[];for(let r of H)h.push(Z(ne,[j,r.dwWeight,r.dwBias,He,r.dwUniform])),D.push(Z($,[He,j,r.pwWeight,r.pwBias,ce,r.pwUniform])),l.push(Z(ne,[ce,r.dwWeight,r.dwBias,He,r.dwUniform])),B.push(Z($,[He,ce,r.pwWeight,r.pwBias,j,r.pwUniform]));let te=Z(we,[j,pt,ce,_]),Be=Z(we,[j,ut,ce,N]),$e=Z(Re,[j,tn,It,ct,zt]),On=Z(we,[ct,ot,ce,ee]);Z(se,[j,ze,gt,Te,Ft]),Z(se,[j,Se,Xe,Te,Nt]),Z(se,[j,n,e,Te,s]);let le=Z(De,[M.createView(),Ee,q]),In=Z(xe,[j,ze,gt,Se,Xe,n,e,Te]),xn=24,zn=[],Fn=[];for(let r=xn;r<H.length;r++){let d=H[r];zn.push(Z(Ue,[j,d.dwWeight,d.dwBias,d.pwWeight,d.pwBias,ce,d.dwUniform])),Fn.push(Z(Ue,[ce,d.dwWeight,d.dwBias,d.pwWeight,d.pwBias,j,d.dwUniform]))}let Pn=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Pn.globalCompositeOperation="copy";let Kn=new OffscreenCanvas(9,8),ln=Kn.getContext("webgpu"),_n=null,vn=null;if(ln){ln.configure({device:t,format:"rgba8unorm",alphaMode:"premultiplied"});let r=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),d=t.createShaderModule({code:_a}),p=t.createShaderModule({code:ma});_n=t.createRenderPipeline({layout:t.createPipelineLayout({bindGroupLayouts:[r]}),vertex:{module:d,entryPoint:"vs"},fragment:{module:p,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),vn=t.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:Te}}]})}let mn=new Float32Array(1),fn=new Float32Array(1),hn=new Float32Array(63);function Ke(r,d){let p=!0,u=0,i=r.beginComputePass();for(i.setPipeline(ht),i.setBindGroup(0,C),i.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);u<=Na;u++){let k=p?h[u]:l[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let E=p?j:ce;for(r.copyBufferToBuffer(E,0,ot,0,3072*64*4),i=r.beginComputePass();u<=qa;u++){let k=p?h[u]:l[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let S=p?j:ce;for(r.copyBufferToBuffer(S,0,ut,0,3072*32*4),i=r.beginComputePass();u<=Ya;u++){let k=p?h[u]:l[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let T=p?j:ce;for(r.copyBufferToBuffer(T,0,pt,0,1536*16*4),i=r.beginComputePass();u<=Xa;u++){let k=p?h[u]:l[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.setPipeline(bt),i.setBindGroup(0,te),i.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),i.end(),p=!1,i=r.beginComputePass();{let k=p?h[u]:l[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p,u++}i.setPipeline(bt),i.setBindGroup(0,Be),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),i.end(),p=!1,i=r.beginComputePass();{let k=p?h[u]:l[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p,u++}for(i.setPipeline(Dt),i.setBindGroup(0,$e),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),i.setPipeline(bt),i.setBindGroup(0,On),i.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),i.end(),p=!1,i=r.beginComputePass();u<xn;u++){let k=p?h[u]:l[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}for(;u<H.length;u++){let k=u-xn,w=p?zn[k]:Fn[k],c=H[u];i.setPipeline(Zt),i.setBindGroup(0,w),i.dispatchWorkgroups(c.outW,c.outH,1),p=!p}i.setPipeline(rt),i.setBindGroup(0,In),i.dispatchWorkgroups(1),i.end(),d&&r.copyBufferToBuffer(Te,0,d,0,260)}async function bn(r){t.queue.writeBuffer(st,0,r);let d=t.createCommandEncoder();{let i=d.beginComputePass();i.setPipeline(Mt),i.setBindGroup(0,b),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),i.end()}Ke(d,V),t.queue.submit([d.finish()]);let p=V.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await p;let u=new Float32Array(V.getMappedRange());return mn[0]=u[0],fn[0]=u[1],hn.set(u.subarray(2,65)),V.unmap(),{handflag:new Float32Array(mn),handedness:new Float32Array(fn),landmarks:new Float32Array(hn)}}async function Bn(r){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let d=t.createCommandEncoder();{let i=d.beginComputePass();i.setPipeline(he),i.setBindGroup(0,le),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}Ke(d,V),t.queue.submit([d.finish()]);let p=V.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await p;let u=new Float32Array(V.getMappedRange());return mn[0]=u[0],fn[0]=u[1],hn.set(u.subarray(2,65)),V.unmap(),{handflag:new Float32Array(mn),handedness:new Float32Array(fn),landmarks:new Float32Array(hn)}}async function Nn(r){if(!_n||!vn||!ln)throw new Error("Render-based readback not available (no WebGPU canvas context)");t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let d=t.createCommandEncoder();{let k=d.beginComputePass();k.setPipeline(he),k.setBindGroup(0,le),k.dispatchWorkgroups(16,16,1),k.end()}Ke(d,null);let p=ln.getCurrentTexture(),u=d.beginRenderPass({colorAttachments:[{view:p.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});u.setPipeline(_n),u.setBindGroup(0,vn),u.draw(3),u.end(),t.queue.submit([d.finish()]),await t.queue.onSubmittedWorkDone(),Pn.drawImage(Kn,0,0);let E=Pn.getImageData(0,0,9,8).data,S=new Float32Array(65),T=new DataView(new ArrayBuffer(4));for(let k=0;k<65;k++){let w=k*4;T.setUint8(0,E[w]),T.setUint8(1,E[w+1]),T.setUint8(2,E[w+2]),T.setUint8(3,E[w+3]),S[k]=T.getFloat32(0)}return{handflag:new Float32Array([S[0]]),handedness:new Float32Array([S[1]]),landmarks:new Float32Array(S.subarray(2,65))}}let Sa=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Cn=0,Ma=[V,Sa],yt=null,Ve=null;async function kn(r){let d=Ma[Cn];Cn=1-Cn,t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let p=t.createCommandEncoder();{let i=p.beginComputePass();i.setPipeline(he),i.setBindGroup(0,le),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}Ke(p,d),t.queue.submit([p.finish()]);let u=null;if(yt!==null&&Ve!==null){await yt;let i=new Float32Array(Ve.getMappedRange());u={handflag:new Float32Array([i[0]]),handedness:new Float32Array([i[1]]),landmarks:new Float32Array(i.subarray(2,65))},Ve.unmap()}return Ve=d,yt=d.mapAsync(GPUMapMode.READ),u}async function qn(){if(!yt||!Ve)return null;await yt;let r=new Float32Array(Ve.getMappedRange()),d={handflag:new Float32Array([r[0]]),handedness:new Float32Array([r[1]]),landmarks:new Float32Array(r.subarray(2,65))};return Ve.unmap(),yt=null,Ve=null,d}async function Da(r=50){let d=new Float32Array(196608);for(let i=0;i<5;i++)await bn(d);let p=[];for(let i=0;i<r;i++){let E=performance.now();await bn(d),p.push(performance.now()-E)}let u=p.reduce((i,E)=>i+E,0)/p.length;return{avgMs:u,fps:1e3/u}}async function Ea(r=50){let d=new Float32Array(196608);for(let S=0;S<5;S++)await bn(d);let p=[];for(let S=0;S<r;S++){let T=t.createCommandEncoder();{let w=T.beginComputePass();w.setPipeline(Mt),w.setBindGroup(0,b),w.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),w.end()}Ke(T,V);let k=performance.now();t.queue.submit([T.finish()]),await t.queue.onSubmittedWorkDone(),p.push(performance.now()-k)}p.sort((S,T)=>S-T);let u=p.reduce((S,T)=>S+T,0)/p.length,i=p[Math.floor(p.length/2)],E=p[0];return{avgMs:u,fps:1e3/u,medianMs:i,minMs:E}}function ni(r){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let d=t.createCommandEncoder();{let p=d.beginComputePass();p.setPipeline(he),p.setBindGroup(0,le),p.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),p.end()}Ke(d,V),t.queue.submit([d.finish()])}async function Ha(r,d=50){function p(w){let c=[...w].sort((z,ae)=>z-ae);return{median:c[Math.floor(c.length/2)],min:c[0]}}for(let w=0;w<10;w++)await Bn(r);let u=[];for(let w=0;w<d;w++){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let c=t.createCommandEncoder();{let ae=c.beginComputePass();ae.setPipeline(he),ae.setBindGroup(0,le),ae.dispatchWorkgroups(16,16,1),ae.end()}Ke(c,V);let z=performance.now();t.queue.submit([c.finish()]),await t.queue.onSubmittedWorkDone(),u.push(performance.now()-z)}let i=[];for(let w=0;w<d;w++){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let c=t.createCommandEncoder();{let oe=c.beginComputePass();oe.setPipeline(he),oe.setBindGroup(0,le),oe.dispatchWorkgroups(16,16,1),oe.end()}Ke(c,V),t.queue.submit([c.finish()]);let z=V.mapAsync(GPUMapMode.READ),ae=performance.now();await t.queue.onSubmittedWorkDone(),await z,V.getMappedRange(),V.unmap(),i.push(performance.now()-ae)}let E=[];for(let w=0;w<d;w++){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let c=t.createCommandEncoder();{let ae=c.beginComputePass();ae.setPipeline(he),ae.setBindGroup(0,le),ae.dispatchWorkgroups(16,16,1),ae.end()}Ke(c,V),t.queue.submit([c.finish()]);let z=performance.now();await V.mapAsync(GPUMapMode.READ),V.getMappedRange(),V.unmap(),E.push(performance.now()-z)}let S=[];for(let w=0;w<d;w++){let c=performance.now();await Bn(r),S.push(performance.now()-c)}await kn(r);let T=[];for(let w=0;w<d;w++){let c=performance.now();await kn(r),T.push(performance.now()-c)}await qn();let k=null;if(_n){let w=[];for(let c=0;c<d;c++){let z=performance.now();await Nn(r),w.push(performance.now()-z)}k=p(w)}return{gpuOnly:p(u),mapAsyncOnly:p(i),mapAsyncNoWait:p(E),total:p(S),pipelined:p(T),renderReadback:k}}async function Ta(r){let d=[];async function p(i,E,S){let T=t.createBuffer({size:E,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),k=t.createCommandEncoder();k.copyBufferToBuffer(i,0,T,0,E),t.queue.submit([k.finish()]),await t.queue.onSubmittedWorkDone(),await T.mapAsync(GPUMapMode.READ);let w=new Float32Array(T.getMappedRange()),c=1/0,z=-1/0,ae=0;for(let oe=0;oe<w.length;oe++)w[oe]<c&&(c=w[oe]),w[oe]>z&&(z=w[oe]),w[oe]!==0&&ae++;T.unmap(),T.destroy(),d.push({layer:S,stats:{min:c,max:z,nonZero:ae,total:w.length}})}t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);{let i=t.createCommandEncoder(),E=i.beginComputePass();E.setPipeline(he),E.setBindGroup(0,le),E.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),E.end(),t.queue.submit([i.finish()])}await p(Ee,Math.min(Ee.size,3*257*257*4),"canvas\u2192bufInput");{let i=t.createCommandEncoder(),E=i.beginComputePass();E.setPipeline(ht),E.setBindGroup(0,C),E.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),E.end(),t.queue.submit([i.finish()])}await p(j,Math.min(j.size,3072*128*4),"inputConv\u2192bufA");let u=!0;for(let i=0;i<Math.min(H.length,6);i++){let E=u?h[i]:l[i],S=u?D[i]:B[i],T=Ge[i],k=H[i];{let c=t.createCommandEncoder(),z=c.beginComputePass();z.setPipeline(T.dwPipeline),z.setBindGroup(0,E),z.dispatchWorkgroups(T.dwDispatchX,T.dwDispatchY,T.dwDispatchZ),z.end(),t.queue.submit([c.finish()])}await p(He,Math.min(He.size,k.spec.inCh*k.outH*k.outW*4),`layer${i}.DW\u2192bufDW (${k.spec.prefix})`);{let c=t.createCommandEncoder(),z=c.beginComputePass();z.setPipeline(T.pwPipeline),z.setBindGroup(0,S),z.dispatchWorkgroups(T.pwDispatchX,T.pwDispatchY,T.pwDispatchZ),z.end(),t.queue.submit([c.finish()])}let w=u?ce:j;await p(w,Math.min(w.size,k.spec.outCh*k.outH*k.outW*4),`layer${i}.PW\u2192buf${u?"B":"A"} (${k.spec.prefix})`),u=!u}return d}return{device:t,run:bn,runFromCanvas:Bn,runFromCanvasViaRender:Nn,runFromCanvasPipelined:kn,flushPipelined:qn,benchmark:Da,benchmarkGPU:Ea,benchmarkDiagnostic:Ha,debugLayerOutputs:Ta}}function je(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ba=je(`
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
`),wa=je(`
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
`),ga=je(`
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
`),ya=je(`
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
`),xa=je(`
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
`),Pa=je(`
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
`),va=je(`
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
`),Ba=je(`
struct LBParams {
  src_w:u32, src_h:u32, dst_size:u32, _pad:u32,
  scale_x:f32, scale_y:f32, offset_x:f32, offset_y:f32,
}
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:LBParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let dx=gid.x; let dy=gid.y;
  if(dx>=params.dst_size||dy>=params.dst_size){return;}

  let out_stride=params.dst_size*params.dst_size;

  // Map dst pixel to src pixel using MediaPipe's convention:
  // dst pixel center at (dx + 0.5), offset by letterbox padding, then scale to src
  let src_x = (f32(dx) - params.offset_x + 0.5) * params.scale_x - 0.5;
  let src_y = (f32(dy) - params.offset_y + 0.5) * params.scale_y - 0.5;

  // Check if we're in the letterbox padding region
  let in_region = src_x >= -0.5 && src_x < f32(params.src_w) - 0.5
               && src_y >= -0.5 && src_y < f32(params.src_h) - 0.5;

  if(!in_region){
    // Zero padding (letterbox)
    output[0u*out_stride+dy*params.dst_size+dx]=0.0;
    output[1u*out_stride+dy*params.dst_size+dx]=0.0;
    output[2u*out_stride+dy*params.dst_size+dx]=0.0;
    return;
  }

  // Bilinear interpolation matching GL_LINEAR
  let x0=i32(floor(src_x));
  let y0=i32(floor(src_y));
  let x1=x0+1;
  let y1=y0+1;
  let fx=src_x-f32(x0);
  let fy=src_y-f32(y0);

  let sw=i32(params.src_w);
  let sh=i32(params.src_h);

  // Clamp coordinates to valid range
  let cx0=clamp(x0,0,sw-1);
  let cx1=clamp(x1,0,sw-1);
  let cy0=clamp(y0,0,sh-1);
  let cy1=clamp(y1,0,sh-1);

  let p00=textureLoad(input_tex,vec2<u32>(u32(cx0),u32(cy0)),0);
  let p10=textureLoad(input_tex,vec2<u32>(u32(cx1),u32(cy0)),0);
  let p01=textureLoad(input_tex,vec2<u32>(u32(cx0),u32(cy1)),0);
  let p11=textureLoad(input_tex,vec2<u32>(u32(cx1),u32(cy1)),0);

  let pixel=p00*(1.0-fx)*(1.0-fy) + p10*fx*(1.0-fy) + p01*(1.0-fx)*fy + p11*fx*fy;

  output[0u*out_stride+dy*params.dst_size+dx]=pixel.r;
  output[1u*out_stride+dy*params.dst_size+dx]=pixel.g;
  output[2u*out_stride+dy*params.dst_size+dx]=pixel.b;
}
`);async function Ca(o,m){let a;if(m)a=m;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");a=await n.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}})}let U={r:"read-only-storage",s:"storage",u:"uniform"};function x(n){return a.createBindGroupLayout({entries:n.map((e,s)=>e==="t"?{binding:s,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:s,visibility:GPUShaderStage.COMPUTE,buffer:{type:U[e]}})})}let t=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,P=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,I=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,W=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function f(n,e){return a.createBuffer({size:Math.max(n,4),usage:e})}function g(n,e,s){a.queue.writeBuffer(n,e,s)}function A(n){let e=f(n.data.byteLength,t);return g(e,0,n.data),e}let L=Array.from(o.keys());function ue(n){let e=o.get(n);if(!e)throw new Error(`Weight not found: ${n}`);return e}function y(...n){let e=L.find(s=>n.every(H=>s.includes(H)));if(!e)throw new Error(`Weight not found for: ${n.join(", ")}`);return ue(e)}function ie(n){let[,e,s,H]=n.shape,O=new Float32Array(H*25);for(let _=0;_<H;_++)for(let N=0;N<e;N++)for(let ee=0;ee<s;ee++)O[_*25+N*5+ee]=n.data[N*s*H+ee*H+_];return O}function R(n){let[e,,,s]=n.shape,H=new Float32Array(e*s);for(let O=0;O<e;O++)for(let _=0;_<s;_++)H[O*s+_]=n.data[O*s+_];return H}let me=a.createShaderModule({code:ba}),Ce=a.createShaderModule({code:wa}),Me=a.createShaderModule({code:ga}),re=a.createShaderModule({code:ya}),G=a.createShaderModule({code:Pa}),Z=a.createShaderModule({code:xa}),pe=a.createShaderModule({code:va}),Ut=a.createShaderModule({code:Ba}),Je=x(["r","r","r","r","s","u"]),We=x(["r","r","r","s","u"]),Le=x(["r","r","r","r","r","s","u"]),At=x(["r","r","r","s","u"]),dt=x(["r","r","r","r","s","u"]),ge=x(["r","r","s","u"]),Qe=x(["t","s","u"]),lt=x(["t","s","u"]);function ke(n,e){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:e,entryPoint:"main"}})}let _t=ke(Je,me),v=ke(We,Ce),F=ke(Le,Me),X=ke(At,re),J=ke(dt,G),K=ke(ge,Z),ye=ke(Qe,pe),ne=ke(lt,Ut),$=y("conv2d/Conv2D"),be=y("batch_normalization/","conv2d/Conv2D"),Q=y("p_re_lu/"),fe=A($),de=A(be),we=A(Q),se=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:6}].map(n=>{let e=y(n.dwKey),s=y(n.pwKey),H=y(n.bnKey),O=y(n.preluKey),_=ie(e),N=f(_.byteLength,t);g(N,0,_);let ee=new Float32Array(n.inCh),b=f(ee.byteLength,t);g(b,0,ee);let C=R(s),h=f(C.byteLength,t);g(h,0,C);let D=A(H),l=A(O);return{dwWeightBuf:N,dwBiasBuf:b,pwWeightBuf:h,pwBiasBuf:D,alphaBuf:l,inCh:n.inCh,outCh:n.outCh,stride:n.stride,inH:n.inH}}),De=R(y("conv2d_20/Conv2D")),xe=f(De.byteLength,t);g(xe,0,De);let Ue=A(y("batch_normalization_20/")),Oe=A(y("p_re_lu_20/")),Pe={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_19/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(256),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_21/")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_21/")),alphaBuf:A(y("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},Ae={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_20/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(256),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_22/Conv2D1")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_22/")),alphaBuf:A(y("p_re_lu_22/")),inCh:256,outCh:256,stride:1,inH:12},Ie=R(y("conv2d_23/Conv2D")),Fe=f(Ie.byteLength,t);g(Fe,0,Ie);let et=A(y("batch_normalization_23/")),tt=A(y("p_re_lu_23/")),nt={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_21/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(128),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_24/")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_24/")),alphaBuf:A(y("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},qe={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_22/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(128),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_25/Conv2D1")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_25/")),alphaBuf:A(y("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},Gt=R(y("classifier_palm_16_NO_PRUNING/Conv2D")),at=f(Gt.byteLength,t);g(at,0,Gt);let mt=A(y("classifier_palm_16_NO_PRUNING/BiasAdd")),it=R(y("regressor_palm_16_NO_PRUNING/Conv2D")),St=f(it.byteLength,t);g(St,0,it);let $t=A(y("regressor_palm_16_NO_PRUNING/BiasAdd")),Vt=R(y("classifier_palm_8_NO_PRUNING/Conv2D")),ft=f(Vt.byteLength,t);g(ft,0,Vt);let Ge=A(y("classifier_palm_8_NO_PRUNING/BiasAdd")),Mt=R(y("regressor_palm_8_NO_PRUNING/Conv2D")),ht=f(Mt.byteLength,t);g(ht,0,Mt);let bt=A(y("regressor_palm_8_NO_PRUNING/BiasAdd")),Dt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,he=f(36864*3*4,t),rt=f(Dt,P),Zt=f(Dt,P),wt=f(Dt,P),st=f(576*256*4,P),Ee=f(144*256*4,P|GPUBufferUsage.COPY_DST),Ye=f(576*128*4,P|GPUBufferUsage.COPY_DST),j=f(864*4,I),ce=f(15552*4,I),He=f(576*2*4,I),ot=f(576*36*4,I),ut=f(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pt=f(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=f(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Te=f(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),V=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function M(n,e){return Math.ceil(n/e)}function q(n){let e=f(n.byteLength,W);return g(e,0,n),e}let Et=q(new Uint32Array([1,3,32,192,192,96,96])),Ht=se.map(n=>{let e=n.stride===2?n.inH/2:n.inH,s=e,H=n.stride===2?1:2,O=n.inCh;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,s,n.stride,H])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,s,O,n.stride,n.inH,n.inH])),outH:e,outW:s}}),Tt=(()=>{let n=Pe,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),Wt=(()=>{let n=Ae,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),Lt=(()=>{let n=nt,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),Rt=(()=>{let n=qe,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),jt=q(new Uint32Array([1,256,6,6,12,12])),Jt=q(new Uint32Array([1,256,12,12,12,12])),Qt=q(new Uint32Array([1,256,12,12,24,24])),en=q(new Uint32Array([1,128,24,24,24,24])),Ot=q(new Uint32Array([1,256,256,12,12])),tn=q(new Uint32Array([1,256,128,24,24])),It=q(new Uint32Array([1,256,6,12,12])),zt=q(new Uint32Array([1,256,108,12,12])),nn=q(new Uint32Array([1,128,2,24,24])),an=q(new Uint32Array([1,128,36,24,24])),rn=q(new Uint32Array([192,192,192])),sn=a.createBindGroup({layout:Qe,entries:[{binding:0,resource:V.createView()},{binding:1,resource:{buffer:he}},{binding:2,resource:{buffer:rn}}]}),ze=null,gt=0,Ft=0,Kt=f(32,W);function on(n,e){return ze&&gt===n&&Ft===e||(ze&&ze.destroy(),ze=a.createTexture({size:[n,e,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),gt=n,Ft=e),ze}let un=a.createBindGroup({layout:Je,entries:[{binding:0,resource:{buffer:he}},{binding:1,resource:{buffer:fe}},{binding:2,resource:{buffer:de}},{binding:3,resource:{buffer:we}},{binding:4,resource:{buffer:rt}},{binding:5,resource:{buffer:Et}}]});function ve(n,e,s,H,O,_){let N=_.outH,ee=a.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:e.dwWeightBuf}},{binding:2,resource:{buffer:e.dwBiasBuf}},{binding:3,resource:{buffer:wt}},{binding:4,resource:{buffer:_.dw}}]}),b=n.beginComputePass();b.setPipeline(v),b.setBindGroup(0,ee),b.dispatchWorkgroups(M(N,8),M(_.outH,8),e.inCh),b.end();let C=a.createBindGroup({layout:Le,entries:[{binding:0,resource:{buffer:wt}},{binding:1,resource:{buffer:O}},{binding:2,resource:{buffer:e.pwWeightBuf}},{binding:3,resource:{buffer:e.pwBiasBuf}},{binding:4,resource:{buffer:e.alphaBuf}},{binding:5,resource:{buffer:H}},{binding:6,resource:{buffer:_.pw}}]}),h=n.beginComputePass();h.setPipeline(F),h.setBindGroup(0,C),h.dispatchWorkgroups(M(N,8),M(_.outH,8),e.outCh),h.end()}function Se(n,e,s,H,O,_,N,ee,b){let C=a.createBindGroup({layout:At,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:H}},{binding:3,resource:{buffer:O}},{binding:4,resource:{buffer:_}}]}),h=n.beginComputePass();h.setPipeline(X),h.setBindGroup(0,C),h.dispatchWorkgroups(M(b,8),M(ee,8),N),h.end()}function Xe(n,e,s,H,O,_,N,ee,b,C){let h=a.createBindGroup({layout:dt,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:H}},{binding:3,resource:{buffer:O}},{binding:4,resource:{buffer:_}},{binding:5,resource:{buffer:N}}]}),D=n.beginComputePass();D.setPipeline(J),D.setBindGroup(0,h),D.dispatchWorkgroups(M(C,8),M(b,8),ee),D.end()}async function Nt(n){{let l=n.beginComputePass();l.setPipeline(_t),l.setBindGroup(0,un),l.dispatchWorkgroups(M(96,8),M(96,8),32),l.end()}let e=rt,s=Zt;for(let l=0;l<se.length;l++){let B=se[l];ve(n,B,e,s,e,Ht[l]);let te=e;e=s,s=te,l===10&&n.copyBufferToBuffer(e,0,Ye,0,576*128*4),l===14&&n.copyBufferToBuffer(e,0,Ee,0,144*256*4)}{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:jt}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,l),B.dispatchWorkgroups(M(12,8),M(12,8),256),B.end()}{let l=e;e=s,s=l}Xe(n,e,xe,Ue,Oe,s,Ot,256,12,12);{let l=e;e=s,s=l}{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Ee}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Jt}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,l),B.dispatchWorkgroups(M(12,8),M(12,8),256),B.end()}{let l=e;e=s,s=l}ve(n,Pe,e,s,e,Tt);{let l=e;e=s,s=l}ve(n,Ae,e,s,e,Wt);{let l=e;e=s,s=l}Se(n,e,at,mt,j,It,6,12,12),Se(n,e,St,$t,ce,zt,108,12,12);{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Qt}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,l),B.dispatchWorkgroups(M(24,8),M(24,8),256),B.end()}{let l=e;e=s,s=l}Xe(n,e,Fe,et,tt,s,tn,128,24,24);{let l=e;e=s,s=l}{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:en}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,l),B.dispatchWorkgroups(M(24,8),M(24,8),128),B.end()}{let l=e;e=s,s=l}ve(n,nt,e,s,e,Lt);{let l=e;e=s,s=l}ve(n,qe,e,s,e,Rt);{let l=e;e=s,s=l}Se(n,e,ft,Ge,He,nn,2,24,24),Se(n,e,ht,bt,ot,an,36,24,24),a.queue.submit([n.finish()]);let H=a.createCommandEncoder();H.copyBufferToBuffer(j,0,ut,0,864*4),H.copyBufferToBuffer(ce,0,pt,0,15552*4),H.copyBufferToBuffer(He,0,ct,0,576*2*4),H.copyBufferToBuffer(ot,0,Te,0,576*36*4),a.queue.submit([H.finish()]),await Promise.all([ut.mapAsync(GPUMapMode.READ),pt.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ),Te.mapAsync(GPUMapMode.READ)]);let O=new Float32Array(ut.getMappedRange()).slice(),_=new Float32Array(pt.getMappedRange()).slice(),N=new Float32Array(ct.getMappedRange()).slice(),ee=new Float32Array(Te.getMappedRange()).slice();ut.unmap(),pt.unmap(),ct.unmap(),Te.unmap();let b=2016,C=new Float32Array(b),h=new Float32Array(b*18),D=0;for(let l=0;l<12;l++)for(let B=0;B<12;B++)for(let te=0;te<6;te++){C[D]=O[te*144+l*12+B];for(let Be=0;Be<18;Be++){let $e=te*18+Be;h[D*18+Be]=_[$e*144+l*12+B]}D++}for(let l=0;l<24;l++)for(let B=0;B<24;B++)for(let te=0;te<2;te++){C[D]=N[te*576+l*24+B];for(let Be=0;Be<18;Be++){let $e=te*18+Be;h[D*18+Be]=ee[$e*576+l*24+B]}D++}return{scores:C,regressors:h}}async function pn(n){a.queue.copyExternalImageToTexture({source:n},{texture:V},[192,192]);let e=a.createCommandEncoder();{let s=e.beginComputePass();s.setPipeline(ye),s.setBindGroup(0,sn),s.dispatchWorkgroups(M(192,16),M(192,16),1),s.end()}return Nt(e)}async function cn(n,e,s){let H=Math.min(192/e,192/s),O=Math.round(e*H),_=Math.round(s*H),N=(192-O)/2,ee=(192-_)/2,b=N/192,C=ee/192,h=on(e,s),D;if(n instanceof HTMLVideoElement||n instanceof HTMLImageElement){let le=new OffscreenCanvas(e,s);le.getContext("2d").drawImage(n,0,0),D=le}else D=n;a.queue.copyExternalImageToTexture({source:D},{texture:h},[e,s]);let l=new ArrayBuffer(32),B=new Uint32Array(l),te=new Float32Array(l);B[0]=e,B[1]=s,B[2]=192,B[3]=0,te[4]=e/O,te[5]=s/_,te[6]=N,te[7]=ee,a.queue.writeBuffer(Kt,0,l);let Be=a.createBindGroup({layout:lt,entries:[{binding:0,resource:h.createView()},{binding:1,resource:{buffer:he}},{binding:2,resource:{buffer:Kt}}]}),$e=a.createCommandEncoder();{let le=$e.beginComputePass();le.setPipeline(ne),le.setBindGroup(0,Be),le.dispatchWorkgroups(M(192,16),M(192,16),1),le.end()}return{output:await Nt($e),lbPadX:b,lbPadY:C}}async function Y(n,e){let s=a.createBuffer({size:e*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),H=a.createCommandEncoder();H.copyBufferToBuffer(n,0,s,0,e*4),a.queue.submit([H.finish()]),await s.mapAsync(GPUMapMode.READ);let O=new Float32Array(s.getMappedRange()).slice();return s.unmap(),s.destroy(),O}async function dn(n){a.queue.copyExternalImageToTexture({source:n},{texture:V},[192,192]);function e(h,D=1e3){let l=h.slice(0,D);return{min:Math.min(...l),max:Math.max(...l),mean:l.reduce((B,te)=>B+te,0)/l.length,nonZero:l.filter(B=>B!==0).length,sample:Array.from(l.slice(0,10))}}let s={},H=q(new Uint32Array([192,192,192])),O=a.createBindGroup({layout:Qe,entries:[{binding:0,resource:V.createView()},{binding:1,resource:{buffer:he}},{binding:2,resource:{buffer:H}}]}),_=a.createCommandEncoder(),N=_.beginComputePass();N.setPipeline(ye),N.setBindGroup(0,O),N.dispatchWorkgroups(M(192,16),M(192,16),1),N.end(),a.queue.submit([_.finish()]),s.input=e(await Y(he,36864*3)),_=a.createCommandEncoder();let ee=a.createBindGroup({layout:Je,entries:[{binding:0,resource:{buffer:he}},{binding:1,resource:{buffer:fe}},{binding:2,resource:{buffer:de}},{binding:3,resource:{buffer:we}},{binding:4,resource:{buffer:rt}},{binding:5,resource:{buffer:Et}}]});N=_.beginComputePass(),N.setPipeline(_t),N.setBindGroup(0,ee),N.dispatchWorkgroups(M(96,8),M(96,8),32),N.end(),a.queue.submit([_.finish()]),s.initConv=e(await Y(rt,9216*32));let b=rt,C=Zt;for(let h=0;h<se.length;h++){let D=se[h];_=a.createCommandEncoder(),ve(_,D,b,C,b,Ht[h]),a.queue.submit([_.finish()]);let l=b;if(b=C,C=l,h===0||h===3||h===7||h===11||h===14||h===15||h===18){let B=D.stride===2?D.inH/2:D.inH,te=B*B*D.outCh;s[`block${h}`]=e(await Y(b,te))}h===10&&(_=a.createCommandEncoder(),_.copyBufferToBuffer(b,0,Ye,0,576*128*4),a.queue.submit([_.finish()])),h===14&&(_=a.createCommandEncoder(),_.copyBufferToBuffer(b,0,Ee,0,144*256*4),a.queue.submit([_.finish()]))}_=a.createCommandEncoder();{let h=q(new Uint32Array([1,256,6,6,12,12])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),l=_.beginComputePass();l.setPipeline(K),l.setBindGroup(0,D),l.dispatchWorkgroups(M(12,8),M(12,8),256),l.end()}a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpnUpsample6to12=e(await Y(b,144*256)),_=a.createCommandEncoder(),Xe(_,b,xe,Ue,Oe,C,Ot,256,12,12),a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpn6to12Conv=e(await Y(b,144*256)),s.backbone12Skip=e(await Y(Ee,144*256)),_=a.createCommandEncoder();{let h=q(new Uint32Array([1,256,12,12,12,12])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:Ee}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),l=_.beginComputePass();l.setPipeline(K),l.setBindGroup(0,D),l.dispatchWorkgroups(M(12,8),M(12,8),256),l.end()}a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpnAdd12=e(await Y(b,144*256)),_=a.createCommandEncoder(),ve(_,Pe,b,C,b,Tt),a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpn12Block1=e(await Y(b,144*256)),_=a.createCommandEncoder(),ve(_,Ae,b,C,b,Wt),a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpn12Block2=e(await Y(b,144*256)),_=a.createCommandEncoder(),Se(_,b,at,mt,j,It,6,12,12),a.queue.submit([_.finish()]),s.cls16=e(await Y(j,864)),_=a.createCommandEncoder(),Se(_,b,St,$t,ce,zt,108,12,12),a.queue.submit([_.finish()]),s.reg16=e(await Y(ce,15552),500),_=a.createCommandEncoder();{let h=q(new Uint32Array([1,256,12,12,24,24])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),l=_.beginComputePass();l.setPipeline(K),l.setBindGroup(0,D),l.dispatchWorkgroups(M(24,8),M(24,8),256),l.end()}a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpnUpsample12to24=e(await Y(b,576*256)),_=a.createCommandEncoder(),Xe(_,b,Fe,et,tt,C,256,128,24,24),a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpn12to24Conv=e(await Y(b,576*128)),s.backbone24Skip=e(await Y(Ye,576*128)),_=a.createCommandEncoder();{let h=q(new Uint32Array([1,128,24,24,24,24])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),l=_.beginComputePass();l.setPipeline(K),l.setBindGroup(0,D),l.dispatchWorkgroups(M(24,8),M(24,8),128),l.end()}a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpnAdd24=e(await Y(b,576*128)),_=a.createCommandEncoder(),ve(_,nt,b,C,b,Lt),a.queue.submit([_.finish()]);{let h=b;b=C,C=h}s.fpn24Block1=e(await Y(b,576*128)),_=a.createCommandEncoder(),ve(_,qe,b,C,b,Rt),a.queue.submit([_.finish()]);{let h=b;b=C,C=h}return s.fpn24Block2=e(await Y(b,576*128)),_=a.createCommandEncoder(),Se(_,b,ft,Ge,He,128,2,24,24),a.queue.submit([_.finish()]),s.cls8=e(await Y(He,576*2)),_=a.createCommandEncoder(),Se(_,b,ht,bt,ot,128,36,24,24),a.queue.submit([_.finish()]),s.reg8=e(await Y(ot,576*36)),s.initWeights=e(await Y(fe,100),100),s.initBias=e(await Y(de,32),32),s.cls16Weights=e(await Y(at,100),100),s.cls16Bias=e(await Y(mt,6),6),s.cls8Weights=e(await Y(ft,100),100),s.cls8Bias=e(await Y(Ge,2),2),s.fpn6to12Weights=e(await Y(xe,100),100),s}return{device:a,run:pn,runWithResize:cn,debugRun:dn}}function $a(){let o=[];for(let m=0;m<12;m++)for(let a=0;a<12;a++){let U=(a+.5)/12,x=(m+.5)/12;for(let t=0;t<6;t++)o.push({x:U,y:x})}for(let m=0;m<24;m++)for(let a=0;a<24;a++){let U=(a+.5)/24,x=(m+.5)/24;for(let t=0;t<2;t++)o.push({x:U,y:x})}return o}var ka=$a();function Va(o){return 1/(1+Math.exp(-o))}function En(o,m){let a=[],{scores:U,regressors:x}=o,t=192;for(let P=0;P<ka.length;P++){let I=Va(U[P]);if(I<m)continue;let W=ka[P],f=P*18,g=W.x+x[f+0]/t,A=W.y+x[f+1]/t,L=x[f+2]/t,ue=x[f+3]/t,y=[];for(let ie=0;ie<7;ie++){let R=W.x+x[f+4+ie*2]/t,me=W.y+x[f+4+ie*2+1]/t;y.push([R,me])}a.push({score:I,box:[g,A,L,ue],keypoints:y})}return a}function Hn(o,m){if(o.length===0)return[];let a=[...o].sort((t,P)=>P.score-t.score),U=[],x=new Set;for(let t=0;t<a.length;t++)if(!x.has(t)){U.push(a[t]);for(let P=t+1;P<a.length;P++)x.has(P)||Za(a[t],a[P])>m&&x.add(P)}return U}function Za(o,m){let a=o.box[0]-o.box[2]/2,U=o.box[1]-o.box[3]/2,x=o.box[0]+o.box[2]/2,t=o.box[1]+o.box[3]/2,P=m.box[0]-m.box[2]/2,I=m.box[1]-m.box[3]/2,W=m.box[0]+m.box[2]/2,f=m.box[1]+m.box[3]/2,g=Math.max(a,P),A=Math.max(U,I),L=Math.min(x,W),ue=Math.min(t,f),y=Math.max(0,L-g),ie=Math.max(0,ue-A),R=y*ie,me=(x-a)*(t-U),Ce=(W-P)*(f-I),Me=me+Ce-R;return Me>0?R/Me:0}function ja(o){let[m,a,U,x]=o.box,t=o.keypoints[0],P=o.keypoints[2],I=P[0]-t[0],W=P[1]-t[1],f=Math.atan2(W,I),A=-Math.PI/2-f,L=Math.max(U,x),y=L*2.6,ie=-.5*L,R=Math.cos(A),me=Math.sin(A),Ce=ie*me,Me=ie*R;return{centerX:m+Ce,centerY:a+Me,width:y,height:y,rotation:A}}function Ua(o,m={}){let{scoreThreshold:a=.5,nmsThreshold:U=.3,maxHands:x=2}=m;async function t(W){let f=await o.run(W),g=En(f,a);return Hn(g,U).slice(0,x).map(ja)}async function P(W){let f=await o.run(W),g=En(f,a);return Hn(g,U).slice(0,x)}async function I(W,f,g){let{output:A,lbPadX:L,lbPadY:ue}=await o.runWithResize(W,f,g),y=En(A,a);return{detections:Hn(y,U).slice(0,x),lbPadX:L,lbPadY:ue}}return{detect:t,detectRaw:P,detectRawWithResize:I,model:o}}var Tn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Wn(o){let m={};for(let a=0;a<Tn.length;a++)m[Tn[a]]=o[a];return m}function Aa(o,m,a){return o.initialized?(o.value=a*m+(1-a)*o.value,o.value):(o.value=m,o.initialized=!0,m)}function Ga(o,m){let a=2*Math.PI*m*o;return a/(a+1)}function Ja(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function Ln(o,m,a,U,x,t){let P=o.lastTime<0?.03333333333333333:a-o.lastTime;o.lastTime=a;let I=Ga(P,t),W=o.x.initialized?(m-o.x.value)/P:0,f=Aa(o.dx,W,I),g=U+x*Math.abs(f),A=Ga(P,g);return Aa(o.x,m,A)}function Rn(o={}){let{minCutoff:m=.05,beta:a=80,dCutoff:U=1}=o,x=[];function t(W){x.length!==W&&(x=Array.from({length:W},()=>Ja()))}function P(W,f){let g=f??performance.now()/1e3,A=W.length*3;return t(A),W.map((L,ue)=>({x:Ln(x[ue*3],L.x,g,m,a,U),y:Ln(x[ue*3+1],L.y,g,m,a,U),z:Ln(x[ue*3+2],L.z,g,m,a,U)}))}function I(){x=[]}return{apply:P,reset:I}}var Qa="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ei(o={}){let{weightsUrl:m,scoreThreshold:a=.5,palmScoreThreshold:U=.5,maxHands:x=3,forceF32:t=!1}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let P=(m??Qa).replace(/\/$/,"")+"/",[I,W,f,g]=await Promise.all([fetch(`${P}weights_f16.json`),fetch(`${P}weights_f16.bin`),fetch(`${P}palm_detection_weights.json`),fetch(`${P}palm_detection_weights.bin`)]);if(!I.ok)throw new Error(`Failed to fetch landmark weights: ${I.status}`);if(!W.ok)throw new Error(`Failed to fetch landmark weights: ${W.status}`);if(!f.ok)throw new Error(`Failed to fetch palm detection weights: ${f.status}`);if(!g.ok)throw new Error(`Failed to fetch palm detection weights: ${g.status}`);let[A,L,ue,y]=await Promise.all([I.json(),W.arrayBuffer(),f.json(),g.arrayBuffer()]),ie=Mn(A,L),R=Mn(ue,y),me=await Dn(ie,{forceF32:t});if(!t){let v=new OffscreenCanvas(256,256),F=v.getContext("2d");F.fillStyle="#886644",F.fillRect(0,0,256,256),F.fillStyle="#cc9966",F.fillRect(50,50,156,156);let X=await me.runFromCanvas(v);X.landmarks.every(K=>K===0)&&X.handflag.every(K=>K===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),me.device.destroy(),me=await Dn(ie,{forceF32:!0}))}let Ce=await Ca(R),Me=Ua(Ce,{scoreThreshold:U,maxHands:x}),re=[];for(let v=0;v<x;v++)re.push(Rn());let G=0,Z=null,pe=null;function Ut(){return Z||(Z=new OffscreenCanvas(192,192)),Z}function Je(){return pe||(pe=new OffscreenCanvas(256,256)),pe}let We=0,Le=0;function At(v,F,X){let J=Ut();J.width=192,J.height=192;let K=J.getContext("2d");K.clearRect(0,0,192,192);let ye=Math.min(192/F,192/X),ne=Math.round(F*ye),$=Math.round(X*ye),be=(192-ne)/2,Q=(192-$)/2;if(We=be/192,Le=Q/192,v instanceof ImageData){let fe=new OffscreenCanvas(v.width,v.height);fe.getContext("2d").putImageData(v,0,0),K.drawImage(fe,be,Q,ne,$)}else K.drawImage(v,0,0,F,X,be,Q,ne,$);return J}function dt(v){let F=1/(1-2*We),X=1/(1-2*Le);return{score:v.score,box:[(v.box[0]-We)*F,(v.box[1]-Le)*X,v.box[2]*F,v.box[3]*X],keypoints:v.keypoints.map(([J,K])=>[(J-We)*F,(K-Le)*X])}}function ge(v,F,X){let J=v.keypoints[0],K=v.keypoints[2],ye=K[0]-J[0],ne=K[1]-J[1],$=Math.atan2(-ne,ye),Q=Math.PI/2-$,[fe,de,we,Re]=v.box,se=Math.max(we*F,Re*X),De=0,xe=-.5*se/X,Ue=Math.cos(Q),Oe=Math.sin(Q),Pe=fe+(De*Ue-xe*Oe),Ae=de+(De*Oe+xe*Ue),Fe=se*2.6;return{centerXpx:Pe*F,centerYpx:Ae*X,sizePx:Fe,rotation:Q}}function Qe(v,F){let X=Je();X.width=256,X.height=256;let J=X.getContext("2d");J.clearRect(0,0,256,256);let K=256/F.sizePx,ye=Math.cos(F.rotation),ne=Math.sin(F.rotation),$=ye*K,be=-ne*K,Q=ne*K,fe=ye*K,de=-F.centerXpx*$-F.centerYpx*Q+128,we=-F.centerXpx*be-F.centerYpx*fe+128;if(J.setTransform($,be,Q,fe,de,we),v instanceof ImageData){let Re=new OffscreenCanvas(v.width,v.height);Re.getContext("2d").putImageData(v,0,0),J.drawImage(Re,0,0)}else J.drawImage(v,0,0);return J.setTransform(1,0,0,1,0,0),X}function lt(v){return v instanceof HTMLCanvasElement||v instanceof OffscreenCanvas?[v.width,v.height]:typeof ImageBitmap<"u"&&v instanceof ImageBitmap?[v.width,v.height]:v instanceof ImageData?[v.width,v.height]:v instanceof HTMLVideoElement?[v.videoWidth,v.videoHeight]:v instanceof HTMLImageElement?[v.naturalWidth,v.naturalHeight]:[256,256]}async function ke(v){let[F,X]=lt(v),{detections:J,lbPadX:K,lbPadY:ye}=await Me.detectRawWithResize(v,F,X);if(We=K,Le=ye,J.length===0){if(G>0)for(let $=0;$<G&&$<re.length;$++)re[$].reset();return G=0,[]}let ne=[];for(let $ of J){let be=dt($),Q=ge(be,F,X),fe=Qe(v,Q),de=await me.runFromCanvas(fe),we=de.handflag[0];if(we<a)continue;let Re=de.handedness[0]>.5,se=[],De=Math.cos(Q.rotation),xe=Math.sin(Q.rotation);for(let Pe=0;Pe<21;Pe++){let Ae=de.landmarks[Pe*3],Ie=de.landmarks[Pe*3+1],Fe=de.landmarks[Pe*3+2],et=(Ae-.5)*Q.sizePx,tt=(Ie-.5)*Q.sizePx,nt=De*et-xe*tt+Q.centerXpx,qe=xe*et+De*tt+Q.centerYpx;se.push({x:nt/F,y:qe/X,z:Fe})}let Ue=ne.length,Oe=Ue<re.length?re[Ue].apply(se):se;ne.push({score:we,handedness:Re?"right":"left",landmarks:Oe,keypoints:Wn(Oe)})}if(ne.length<G)for(let $=ne.length;$<G;$++)$<re.length&&re[$].reset();return G=ne.length,ne}function _t(){me.device.destroy(),Ce.device.destroy(),Z=null,pe=null}return{detect:ke,dispose:_t}}export{Tn as LANDMARK_NAMES,ei as createHandpose,Rn as createLandmarkSmoother,Wn as toKeypoints};
