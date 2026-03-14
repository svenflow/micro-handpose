function _e(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Yt(r){let p=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+r;for(let v of p)for(;a.includes(`${v}:array<f32>`);)a=a.replace(`${v}:array<f32>`,`${v}:array<f16>`);return a}var Wa=_e(`
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
`),Ha=_e(`
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
`),La=_e(`
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
`),Ra=_e(`
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
`);var za=_e(`
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
`),Oa=_e(`
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
`);function Vt(r){return _e(`
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
`)}var Ka=Vt(!1),Fa=Vt(!0),Ia=_e(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Na=_e(`
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
`);function Xt(r){return _e(`
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
`)}var qa=Xt("sigmoid"),ja=Xt("div256"),Ya=_e(`
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
`),Va=_e(`
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
`);var Xa=_e(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),$a=_e(`
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
`),Za=_e(`
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
`);function zt(r,p){let a=new Map,v=r.dtype??"float32";for(let h=0;h<r.keys.length;h++){let n=r.keys[h],y=r.shapes[h],M=r.offsets[h],w=y.reduce((B,D)=>B*D,1),s,b;if(v==="float32")s=new Float32Array(p,M,w);else{let B=new DataView(p);s=new Float32Array(w);for(let D=0;D<w;D++)s[D]=da(B.getUint16(M+D*2,!0));b=p.slice(M,M+w*2)}a.set(n,{data:s,shape:y,rawF16:b})}return a}function da(r){let p=r>>15&1,a=r>>10&31,v=r&1023;if(a===0){if(v===0)return p?-0:0;let y=-14,M=v/1024;return(p?-1:1)*Math.pow(2,y)*M}if(a===31)return v===0?p?-1/0:1/0:NaN;let h=a-15,n=1+v/1024;return(p?-1:1)*Math.pow(2,h)*n}var la=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Qa=la.map(([r,p,a,v,h])=>({type:"resmodule",inCh:r,outCh:p,h:a,w:a,stride:v,prefix:h}));function $t(r,p){let a=new Map,v=r.dtype??"float32",h=new Map;for(let n=0;n<r.keys.length;n++){let y=r.keys[n],M=r.shapes[n],w=r.offsets[n],s=M.reduce((g,O)=>g*O,1),b,B;if(v==="float32")b=new Float32Array(p,w,s);else{let g=new DataView(p);b=new Float32Array(s);for(let O=0;O<s;O++)b[O]=_a(g.getUint16(w+O*2,!0));B=p.slice(w,w+s*2)}let D=h.get(y)??0;h.set(y,D+1);let S=D===0?y:`${y}__${D}`;a.set(S,{data:b,shape:M,rawF16:B})}return a}function _a(r){let p=r>>15&1,a=r>>10&31,v=r&1023;return a===0?v===0?p?-0:0:(p?-1:1)*Math.pow(2,-14)*(v/1024):a===31?v===0?p?-1/0:1/0:NaN:(p?-1:1)*Math.pow(2,a-15)*(1+v/1024)}function Ye(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ma=Ye(`
struct CanvasParams { in_size:u32, }
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CanvasParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y;
  if(x>=params.in_size||y>=params.in_size){return;}
  let pixel=textureLoad(input_tex,vec2<u32>(x,y),0);
  let stride=params.in_size*params.in_size;
  output[0u*stride+y*params.in_size+x]=pixel.r;
  output[1u*stride+y*params.in_size+x]=pixel.g;
  output[2u*stride+y*params.in_size+x]=pixel.b;
}
`),fa=Ye(`
struct Params { in_channels:u32, out_channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    for(var ky:u32=0u;ky<3u;ky++){
      for(var kx:u32=0u;kx<3u;kx++){
        let iy=i32(out_y*2u+ky); let ix=i32(out_x*2u+kx);
        if(iy>=0 && iy<in_h && ix>=0 && ix<in_w){
          let in_idx=ic*params.in_h*params.in_w+u32(iy)*params.in_w+u32(ix);
          let w_idx=oc*params.in_channels*9u+ic*9u+ky*3u+kx;
          sum+=input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum+=bias[oc];
  sum=min(max(sum,0.0),6.0);
  let out_idx=oc*params.out_h*params.out_w+out_y*params.out_w+out_x;
  output[out_idx]=sum;
}
`),ha=Ye(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  sum=min(max(sum,0.0),6.0);
  output[oc*spatial+pix]=sum;
}
`),ba=Ye(`
struct Params { channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, stride:u32, pad:u32, kernel:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||c>=params.channels){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  let kk=params.kernel*params.kernel;
  for(var ky:u32=0u;ky<params.kernel;ky++){
    for(var kx:u32=0u;kx<params.kernel;kx++){
      let iy=i32(out_y*params.stride+ky)-i32(params.pad);
      let ix=i32(out_x*params.stride+kx)-i32(params.pad);
      if(iy>=0 && iy<in_h && ix>=0 && ix<in_w){
        sum+=input[c*params.in_h*params.in_w+u32(iy)*params.in_w+u32(ix)]*weight[c*kk+ky*params.kernel+kx];
      }
    }
  }
  sum+=bias[c];
  sum=min(max(sum,0.0),6.0);
  output[c*params.out_h*params.out_w+out_y*params.out_w+out_x]=sum;
}
`),ga=Ye(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  output[oc*spatial+pix]=sum;
}
`),wa=Ye(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),ya=Ye(`
struct Params { channels:u32, spatial:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let c=gid.x;
  if(c>=params.channels){return;}
  var sum:f32=0.0;
  let base=c*params.spatial;
  for(var i:u32=0u;i<params.spatial;i++){
    sum+=input[base+i];
  }
  output[c]=sum/f32(params.spatial);
}
`),xa=Ye(`
struct Params { in_features:u32, out_features:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  if(oc>=params.out_features){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_features;ic++){
    sum+=input[ic]*weight[oc*params.in_features+ic];
  }
  output[oc]=sum+bias[oc];
}
`),va=Ye(`
struct Params { in_features:u32, out_features:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  if(oc>=params.out_features){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_features;ic++){
    sum+=input[ic]*weight[oc*params.in_features+ic];
  }
  sum+=bias[oc];
  output[oc]=1.0/(1.0+exp(-sum));
}
`),dt=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],Pa=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function Ot(r,p){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let v=a.features.has("shader-f16"),h=v?["shader-f16"]:[],n=await a.requestDevice({requiredFeatures:h,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8)}}),y=r.values().next().value,M=v&&!!y?.rawF16&&!p?.forceF32;function w(_){if(M&&_.rawF16){let f=new Uint16Array(_.rawF16);if(f.length%2!==0){let c=new Uint16Array(f.length+1);return c.set(f),c}return f}return _.data}function s(_){return M&&_.rawF16?Math.ceil(_.rawF16.byteLength/4)*4:_.data.byteLength}function b(_){return M?Yt(_):_}let B={r:"read-only-storage",s:"storage",u:"uniform"};function D(_){return n.createBindGroupLayout({entries:_.map((f,c)=>f==="t"?{binding:c,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:c,visibility:GPUShaderStage.COMPUTE,buffer:{type:B[f]}})})}let S=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,g=GPUBufferUsage.STORAGE,O=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,K=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,A=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function x(_,f){return n.createBuffer({size:Math.max(_,4),usage:f})}function Q(_,f){return n.createBindGroup({layout:_,entries:f.map((c,L)=>({binding:L,resource:"size"in c?{buffer:c}:c}))})}function me(_,f){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[_]}),compute:{module:f,entryPoint:"main"}})}function j(_){let f=r.get(_);if(!f)throw new Error(`Missing weight: ${_}`);return f}let Ee=n.createShaderModule({code:ma}),Ve=n.createShaderModule({code:b(fa)}),Xe=n.createShaderModule({code:b(ha)}),tt=n.createShaderModule({code:b(ba)}),lt=n.createShaderModule({code:b(ga)}),ve=n.createShaderModule({code:wa}),Te=n.createShaderModule({code:ya}),Ce=n.createShaderModule({code:b(xa)}),ee=n.createShaderModule({code:b(va)}),ce=D(["r","r","r","s","u"]),$e=D(["r","r","s","u"]),be=D(["r","s","u"]),De=D(["r","r","r","s","u"]),_t=D(["t","s","u"]),We=me(_t,Ee),He=me(ce,Ve),Gt=me(ce,Xe),ge=me(ce,tt),at=me(ce,lt),xt=me($e,ve),vt=me(be,Te),Pt=me(De,Ce),mt=me(De,ee),l=1152*112*112*4,C=x(l,K),W=x(l,K),q=x(l,g),Y=x(l,g),oe=x(l,S),$=x(672*224*4,K),ne=x(1152*4,O),se=x(252,O),ie=x(252,O),I=x(4,O),pe=x(4,O),J=x(260,K),X=x(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ue=n.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),de=x(4,A);n.queue.writeBuffer(de,0,new Uint32Array([224]));let Ue=j("conv2d"),we=j("batch_normalization"),ke=w(Ue),Ze=w(we),Ge=x(s(Ue),S),Le=x(s(we),S),Re=x(24,A);n.queue.writeBuffer(Ge,0,ke),n.queue.writeBuffer(Le,0,Ze),n.queue.writeBuffer(Re,0,new Uint32Array([3,24,224,224,112,112]));let ye=112,le=112,ze=[];for(let _=0;_<dt.length;_++){let f=dt[_],c=Pa[_],L=ye,T=le,Z=f.stride===2?Math.floor(ye/2):ye,H=f.stride===2?Math.floor(le/2):le,P={spec:f,inH:L,inW:T,outH:Z,outW:H,dwW:x(4,S),dwB:x(4,S),dwU:x(32,A)};if(c.expandConvKey){let re=j(c.expandConvKey),G=j(c.expandBNKey);P.expandW=x(s(re),S),P.expandB=x(s(G),S),P.expandU=x(16,A),n.queue.writeBuffer(P.expandW,0,w(re)),n.queue.writeBuffer(P.expandB,0,w(G)),n.queue.writeBuffer(P.expandU,0,new Uint32Array([f.inCh,f.expandCh,L,T]))}let Be=j(c.dwWeightKey),xe=j(c.dwBNKey);P.dwW=x(s(Be),S),P.dwB=x(s(xe),S),n.queue.writeBuffer(P.dwW,0,w(Be)),n.queue.writeBuffer(P.dwB,0,w(xe));let he=Math.floor((f.dwKernel-f.stride)/2);if(n.queue.writeBuffer(P.dwU,0,new Uint32Array([f.expandCh,L,T,Z,H,f.stride,he,f.dwKernel])),f.hasProject&&c.projectConvKey){let re=j(c.projectConvKey),G=j(c.projectBNKey);P.projectW=x(s(re),S),P.projectB=x(s(G),S),P.projectU=x(16,A),n.queue.writeBuffer(P.projectW,0,w(re)),n.queue.writeBuffer(P.projectB,0,w(G)),n.queue.writeBuffer(P.projectU,0,new Uint32Array([f.expandCh,f.outCh,Z,H]))}ze.push(P),ye=Z,le=H}function ft(_,f){let c=r.get(_);if(!c)throw new Error(`Missing weight: ${_}`);if(c.shape.length!==f)throw new Error(`Weight ${_} has rank ${c.shape.length}, expected ${f}`);return c}let Oe=j("conv_landmarks__1"),Ke=j("conv_world_landmarks__1"),Pe=j("conv_handflag__1"),fe=j("conv_handedness__1"),nt=j("Identity"),it=j("Identity_1"),Fe=j("Identity_2"),Ae=j("Identity_3"),Me=x(s(Oe),S),Ie=x(s(nt),S),Je=x(s(Ke),S),ht=x(s(Ae),S),rt=x(s(Pe),S),st=x(s(it),S),ot=x(s(fe),S),ut=x(s(Fe),S);n.queue.writeBuffer(Me,0,w(Oe)),n.queue.writeBuffer(Ie,0,w(nt)),n.queue.writeBuffer(Je,0,w(Ke)),n.queue.writeBuffer(ht,0,w(Ae)),n.queue.writeBuffer(rt,0,w(Pe)),n.queue.writeBuffer(st,0,w(it)),n.queue.writeBuffer(ot,0,w(fe)),n.queue.writeBuffer(ut,0,w(Fe));let ct=x(8,A),bt=x(8,A),E=x(8,A),F=x(8,A);n.queue.writeBuffer(ct,0,new Uint32Array([1152,63])),n.queue.writeBuffer(bt,0,new Uint32Array([1152,63])),n.queue.writeBuffer(E,0,new Uint32Array([1152,1])),n.queue.writeBuffer(F,0,new Uint32Array([1152,1]));let Ct=x(8,A);n.queue.writeBuffer(Ct,0,new Uint32Array([1152,ye*le]));let Ut=new Map;for(let _=0;_<dt.length;_++)if(dt[_].hasResidual){let f=ze[_],c=x(4,A);n.queue.writeBuffer(c,0,new Uint32Array([dt[_].outCh*f.outH*f.outW])),Ut.set(_,c)}let At=Q(_t,[ue.createView(),C,de]),Mt=Q(ce,[C,Ge,Le,W,Re]),Ne=new Float32Array(1),qe=new Float32Array(1),Qe=new Float32Array(63);function gt(_,f){let c=_.beginComputePass();c.setPipeline(He),c.setBindGroup(0,Mt),c.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),c.end();let L=W,T=C;for(let Z=0;Z<dt.length;Z++){let H=dt[Z],P=ze[Z];if(H.hasResidual){let he=H.inCh*P.inH*P.inW*4;_.copyBufferToBuffer(L,0,oe,0,he)}if(c=_.beginComputePass(),P.expandW){let he=Q(ce,[L,P.expandW,P.expandB,q,P.expandU]);c.setPipeline(Gt),c.setBindGroup(0,he),c.dispatchWorkgroups(Math.ceil(P.inW/8),Math.ceil(P.inH/8),H.expandCh)}let Be=P.expandW?q:L,xe=Q(ce,[Be,P.dwW,P.dwB,Y,P.dwU]);if(c.setPipeline(ge),c.setBindGroup(0,xe),c.dispatchWorkgroups(Math.ceil(P.outW/8),Math.ceil(P.outH/8),H.expandCh),H.hasProject&&P.projectW){let he=(H.hasResidual,T),re=Q(ce,[Y,P.projectW,P.projectB,he,P.projectU]);if(c.setPipeline(at),c.setBindGroup(0,re),c.dispatchWorkgroups(Math.ceil(P.outW/8),Math.ceil(P.outH/8),H.outCh),H.hasResidual){let G=Ut.get(Z),je=Q($e,[T,oe,L,G]);c.setPipeline(xt),c.setBindGroup(0,je),c.dispatchWorkgroups(Math.ceil(H.outCh*P.outH*P.outW/256))}else{let G=L;L=T,T=G}}if(c.end(),!H.hasProject){c=_.beginComputePass();let he=Q(be,[Y,ne,Ct]);c.setPipeline(vt),c.setBindGroup(0,he),c.dispatchWorkgroups(Math.ceil(1152/256));let re=Q(De,[ne,Me,Ie,se,ct]);c.setPipeline(Pt),c.setBindGroup(0,re),c.dispatchWorkgroups(1);let G=Q(De,[ne,rt,st,I,E]);c.setPipeline(mt),c.setBindGroup(0,G),c.dispatchWorkgroups(1);let je=Q(De,[ne,ot,ut,pe,F]);c.setPipeline(mt),c.setBindGroup(0,je),c.dispatchWorkgroups(1),c.end(),_.copyBufferToBuffer(I,0,J,0,4),_.copyBufferToBuffer(pe,0,J,4,4),_.copyBufferToBuffer(se,0,J,8,252),f&&_.copyBufferToBuffer(J,0,f,0,260);return}}}async function Ht(_){n.queue.writeBuffer($,0,_);let f=n.createCommandEncoder();f.copyBufferToBuffer($,0,C,0,672*224*4),gt(f,X),n.queue.submit([f.finish()]);let c=X.mapAsync(GPUMapMode.READ);await n.queue.onSubmittedWorkDone(),await c;let L=new Float32Array(X.getMappedRange());Ne[0]=L[0],qe[0]=L[1];for(let T=0;T<63;T++)Qe[T]=L[2+T]/224;return X.unmap(),{handflag:new Float32Array(Ne),handedness:new Float32Array(qe),landmarks:new Float32Array(Qe)}}async function wt(_){n.queue.copyExternalImageToTexture({source:_},{texture:ue},[224,224]);let f=n.createCommandEncoder();{let T=f.beginComputePass();T.setPipeline(We),T.setBindGroup(0,At),T.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),T.end()}gt(f,X),n.queue.submit([f.finish()]);let c=X.mapAsync(GPUMapMode.READ);await n.queue.onSubmittedWorkDone(),await c;let L=new Float32Array(X.getMappedRange());Ne[0]=L[0],qe[0]=L[1];for(let T=0;T<63;T++)Qe[T]=L[2+T]/224;return X.unmap(),{handflag:new Float32Array(Ne),handedness:new Float32Array(qe),landmarks:new Float32Array(Qe)}}async function St(_){let f=n.createCommandEncoder();f.copyBufferToBuffer(_,0,C,0,672*224*4),gt(f,X),n.queue.submit([f.finish()]);let c=X.mapAsync(GPUMapMode.READ);await n.queue.onSubmittedWorkDone(),await c;let L=new Float32Array(X.getMappedRange());Ne[0]=L[0],qe[0]=L[1];for(let T=0;T<63;T++)Qe[T]=L[2+T]/224;return X.unmap(),{handflag:new Float32Array(Ne),handedness:new Float32Array(qe),landmarks:new Float32Array(Qe)}}async function Et(){return null}async function Dt(){return null}async function kt(_=100){let f=new OffscreenCanvas(224,224),c=f.getContext("2d");c.fillStyle="#886644",c.fillRect(0,0,224,224);for(let H=0;H<5;H++)await wt(f);let L=performance.now();for(let H=0;H<_;H++)await wt(f);let Z=(performance.now()-L)/_;return{avgMs:Z,fps:1e3/Z}}async function Tt(_=100){let f=await kt(_);return{...f,medianMs:f.avgMs,minMs:f.avgMs}}async function Wt(_){return wt(_)}async function Lt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Rt(_){let f={};async function c(Z,H,P){let Be=H*4,xe=n.createBuffer({size:Be,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),he=n.createCommandEncoder();he.copyBufferToBuffer(Z,0,xe,0,Be),n.queue.submit([he.finish()]),await n.queue.onSubmittedWorkDone(),await xe.mapAsync(GPUMapMode.READ);let re=new Float32Array(xe.getMappedRange()),G=1/0,je=-1/0,t=0;for(let i=0;i<re.length;i++)re[i]<G&&(G=re[i]),re[i]>je&&(je=re[i]),re[i]!==0&&t++;let e=Array.from(re.slice(0,5));xe.unmap(),xe.destroy(),f[P]={min:G,max:je,nonZero:t,total:H,sample:e}}let L=new Float32Array(672*224);for(let Z=0;Z<50176;Z++)L[Z]=.5,L[50176+Z]=.3,L[448*224+Z]=.7;n.queue.writeBuffer($,0,L);let T=n.createCommandEncoder();return T.copyBufferToBuffer($,0,C,0,672*224*4),gt(T,X),n.queue.submit([T.finish()]),await n.queue.onSubmittedWorkDone(),await c(C,672*224,"inputBufA"),await c(W,2688*112,"afterInitConvBufB"),await c(ne,1152,"gapOutput"),await c(se,63,"landmarks"),await c(I,1,"handflag"),await c(J,65,"unifiedOutput"),f}return{device:n,run:Ht,runFromCanvas:wt,runFromGPUBuffer:St,runFromCanvasPipelined:Et,flushPipelined:Dt,benchmark:kt,benchmarkGPU:Tt,runFromCanvasViaRender:Wt,benchmarkDiagnostic:Lt,debugLayerOutputs:Rt}}function et(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Zt=et(`
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
  let in_h=i32(params.in_height); let in_w=i32(params.in_width);
  let in_stride=params.in_height*params.in_width;
  let in_batch_base=batch*params.in_channels*in_stride;
  for(var ky:u32=0u;ky<5u;ky=ky+1u){
    let in_y=i32(out_y*2u+ky)-1;
    if(in_y<0 || in_y>=in_h){continue;}
    for(var kx:u32=0u;kx<5u;kx=kx+1u){
      let in_x=i32(out_x*2u+kx)-1;
      if(in_x<0 || in_x>=in_w){continue;}
      let pix_off=u32(in_y)*params.in_width+u32(in_x);
      // Load all 3 input channels for this pixel into vec3, dot with 3 weights
      let inp=vec3<f32>(
        input[in_batch_base+pix_off],
        input[in_batch_base+in_stride+pix_off],
        input[in_batch_base+2u*in_stride+pix_off]
      );
      let w_off=oc*75u+ky*15u+kx*3u;
      let w=vec3<f32>(weight[w_off],weight[w_off+1u],weight[w_off+2u]);
      sum+=dot(inp,w);
    }
  }
  sum=sum+bias[oc];
  // PReLU
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.out_height*params.out_width+oc*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=result;
}
`),Jt=et(`
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
`),Qt=et(`
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
  // vec4 dot product accumulation: 4x fewer iterations, deterministic hardware dot()
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      dw_output[dw_base+ic*spatial_stride],
      dw_output[dw_base+(ic+1u)*spatial_stride],
      dw_output[dw_base+(ic+2u)*spatial_stride],
      dw_output[dw_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      pw_weight[w_base+ic],
      pw_weight[w_base+ic+1u],
      pw_weight[w_base+ic+2u],
      pw_weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
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
`),ea=et(`
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
  let in_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels;
  let spatial_stride=params.height*params.width;
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      input[in_base+ic*spatial_stride],
      input[in_base+(ic+1u)*spatial_stride],
      input[in_base+(ic+2u)*spatial_stride],
      input[in_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      weight[w_base+ic],
      weight[w_base+ic+1u],
      weight[w_base+ic+2u],
      weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
  }
  sum=sum+bias[oc];
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=sum;
}
`),ta=et(`
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
`),aa=et(`
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
  let in_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels;
  let spatial_stride=params.height*params.width;
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      input[in_base+ic*spatial_stride],
      input[in_base+(ic+1u)*spatial_stride],
      input[in_base+(ic+2u)*spatial_stride],
      input[in_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      weight[w_base+ic],
      weight[w_base+ic+1u],
      weight[w_base+ic+2u],
      weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
  }
  sum=sum+bias[oc];
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),na=et(`
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
`),ia=et(`
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
`);async function ra(r,p){let a;if(p)a=p;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No GPU adapter found");a=await t.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(t.limits.maxStorageBuffersPerShaderStage,8)}})}let v={r:"read-only-storage",s:"storage",u:"uniform"};function h(t){return a.createBindGroupLayout({entries:t.map((e,i)=>e==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:v[e]}})})}let n=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,M=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,w=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function s(t,e){return a.createBuffer({size:Math.max(t,4),usage:e})}function b(t,e,i){a.queue.writeBuffer(t,e,i)}function B(t){let e=s(t.data.byteLength,n);return b(e,0,t.data),e}let D=Array.from(r.keys());function S(t){let e=r.get(t);if(!e)throw new Error(`Weight not found: ${t}`);return e}function g(...t){let e=D.find(i=>t.every(R=>i.includes(R)));if(!e)throw new Error(`Weight not found for: ${t.join(", ")}`);return S(e)}function O(t){let[,e,i,R]=t.shape,V=new Float32Array(R*25);for(let o=0;o<R;o++)for(let N=0;N<e;N++)for(let te=0;te<i;te++)V[o*25+N*5+te]=t.data[N*i*R+te*R+o];return V}function K(t){let[e,,,i]=t.shape,R=new Float32Array(e*i);for(let V=0;V<e;V++)for(let o=0;o<i;o++)R[V*i+o]=t.data[V*i+o];return R}let A=a.createShaderModule({code:Zt}),x=a.createShaderModule({code:Jt}),Q=a.createShaderModule({code:Qt}),me=a.createShaderModule({code:ea}),j=a.createShaderModule({code:aa}),Ee=a.createShaderModule({code:ta}),Ve=a.createShaderModule({code:na}),Xe=a.createShaderModule({code:ia}),tt=h(["r","r","r","r","s","u"]),lt=h(["r","r","r","s","u"]),ve=h(["r","r","r","r","r","s","u"]),Te=h(["r","r","r","s","u"]),Ce=h(["r","r","r","r","s","u"]),ee=h(["r","r","s","u"]),ce=h(["t","s","u"]),$e=h(["t","s","u"]);function be(t,e){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:e,entryPoint:"main"}})}let De=be(tt,A),_t=be(lt,x),We=be(ve,Q),He=be(Te,me),Gt=be(Ce,j),ge=be(ee,Ee),at=be(ce,Ve),xt=be($e,Xe),vt=g("conv2d/Conv2D"),Pt=g("batch_normalization/","conv2d/Conv2D"),mt=g("p_re_lu/"),Bt=B(vt),l=B(Pt),C=B(mt),q=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(t=>{let e=g(t.dwKey),i=g(t.pwKey),R=g(t.bnKey),V=g(t.preluKey),o=O(e),N=s(o.byteLength,n);b(N,0,o);let te=new Float32Array(t.inCh),d=s(te.byteLength,n);b(d,0,te);let U=K(i),m=s(U.byteLength,n);b(m,0,U);let z=B(R),u=B(V);return{dwWeightBuf:N,dwBiasBuf:d,pwWeightBuf:m,pwBiasBuf:z,alphaBuf:u,inCh:t.inCh,outCh:t.outCh,stride:t.stride,inH:t.inH}}),Y=K(g("conv2d_25/Conv2D")),oe=s(Y.byteLength,n);b(oe,0,Y);let $=B(g("batch_normalization_25/")),ne=B(g("p_re_lu_25/")),se={dwWeightBuf:(()=>{let t=O(g("depthwise_conv2d_24/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(256),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=K(g("conv2d_26/")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(g("batch_normalization_26/")),alphaBuf:B(g("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},ie={dwWeightBuf:(()=>{let t=O(g("depthwise_conv2d_25/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(256),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=K(g("conv2d_27/Conv2D1")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(g("batch_normalization_27/")),alphaBuf:B(g("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},I=K(g("conv2d_28/Conv2D")),pe=s(I.byteLength,n);b(pe,0,I);let J=B(g("batch_normalization_28/")),X=B(g("p_re_lu_28/")),ue={dwWeightBuf:(()=>{let t=O(g("depthwise_conv2d_26/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(128),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=K(g("conv2d_29/")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(g("batch_normalization_29/")),alphaBuf:B(g("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},de={dwWeightBuf:(()=>{let t=O(g("depthwise_conv2d_27/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(128),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=K(g("conv2d_30/Conv2D1")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(g("batch_normalization_30/")),alphaBuf:B(g("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},Ue=K(g("classifier_palm_16_NO_PRUNING/Conv2D")),we=s(Ue.byteLength,n);b(we,0,Ue);let ke=B(g("classifier_palm_16_NO_PRUNING/BiasAdd")),Ze=K(g("regressor_palm_16_NO_PRUNING/Conv2D")),Ge=s(Ze.byteLength,n);b(Ge,0,Ze);let Le=B(g("regressor_palm_16_NO_PRUNING/BiasAdd")),Re=K(g("classifier_palm_8_NO_PRUNING/Conv2D")),ye=s(Re.byteLength,n);b(ye,0,Re);let le=B(g("classifier_palm_8_NO_PRUNING/BiasAdd")),ze=K(g("regressor_palm_8_NO_PRUNING/Conv2D")),ft=s(ze.byteLength,n);b(ft,0,ze);let Oe=B(g("regressor_palm_8_NO_PRUNING/BiasAdd")),Ke=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Pe=s(36864*3*4,n),fe=s(Ke,y),nt=s(Ke,y),it=s(Ke,y),Fe=s(576*256*4,y),Ae=s(144*256*4,y|GPUBufferUsage.COPY_DST),Me=s(576*128*4,y|GPUBufferUsage.COPY_DST),Ie=s(864*4,M),Je=s(15552*4,M),ht=s(576*2*4,M),rt=s(576*36*4,M),st=s(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ot=s(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ut=s(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=s(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),bt=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function E(t,e){return Math.ceil(t/e)}function F(t){let e=s(t.byteLength,w);return b(e,0,t),e}let Ct=F(new Uint32Array([1,3,32,192,192,96,96])),Ut=q.map(t=>{let e=t.stride===2?t.inH/2:t.inH,i=e,R=t.stride===2?1:2,V=t.inCh;return{dw:F(new Uint32Array([1,t.inCh,t.inH,t.inH,e,i,t.stride,R])),pw:F(new Uint32Array([1,t.inCh,t.outCh,e,i,V,t.stride,t.inH,t.inH])),outH:e,outW:i}}),At=(()=>{let t=se,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:F(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:F(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),Mt=(()=>{let t=ie,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:F(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:F(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),Ne=(()=>{let t=ue,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:F(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:F(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),qe=(()=>{let t=de,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:F(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:F(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),Qe=F(new Uint32Array([1,256,6,6,12,12])),gt=F(new Uint32Array([1,256,12,12,12,12])),Ht=F(new Uint32Array([1,256,12,12,24,24])),wt=F(new Uint32Array([1,128,24,24,24,24])),St=F(new Uint32Array([1,256,256,12,12])),Et=F(new Uint32Array([1,256,128,24,24])),Dt=F(new Uint32Array([1,256,6,12,12])),kt=F(new Uint32Array([1,256,108,12,12])),Tt=F(new Uint32Array([1,128,2,24,24])),Wt=F(new Uint32Array([1,128,36,24,24])),Lt=F(new Uint32Array([192,192,192])),Rt=a.createBindGroup({layout:ce,entries:[{binding:0,resource:bt.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:Lt}}]}),_=null,f=0,c=0,L=s(32,w);function T(t,e){return _&&f===t&&c===e||(_&&_.destroy(),_=a.createTexture({size:[t,e,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),f=t,c=e),_}let Z=a.createBindGroup({layout:tt,entries:[{binding:0,resource:{buffer:Pe}},{binding:1,resource:{buffer:Bt}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:C}},{binding:4,resource:{buffer:fe}},{binding:5,resource:{buffer:Ct}}]});function H(t,e,i,R,V,o){let N=o.outH,te=a.createBindGroup({layout:lt,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:e.dwWeightBuf}},{binding:2,resource:{buffer:e.dwBiasBuf}},{binding:3,resource:{buffer:it}},{binding:4,resource:{buffer:o.dw}}]}),d=t.beginComputePass();d.setPipeline(_t),d.setBindGroup(0,te),d.dispatchWorkgroups(E(N,8),E(o.outH,8),e.inCh),d.end();let U=a.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:it}},{binding:1,resource:{buffer:V}},{binding:2,resource:{buffer:e.pwWeightBuf}},{binding:3,resource:{buffer:e.pwBiasBuf}},{binding:4,resource:{buffer:e.alphaBuf}},{binding:5,resource:{buffer:R}},{binding:6,resource:{buffer:o.pw}}]}),m=t.beginComputePass();m.setPipeline(We),m.setBindGroup(0,U),m.dispatchWorkgroups(E(N,8),E(o.outH,8),e.outCh),m.end()}function P(t,e,i,R,V,o,N,te,d){let U=a.createBindGroup({layout:Te,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:R}},{binding:3,resource:{buffer:V}},{binding:4,resource:{buffer:o}}]}),m=t.beginComputePass();m.setPipeline(He),m.setBindGroup(0,U),m.dispatchWorkgroups(E(d,8),E(te,8),N),m.end()}function Be(t,e,i,R,V,o,N,te,d,U){let m=a.createBindGroup({layout:Ce,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:R}},{binding:3,resource:{buffer:V}},{binding:4,resource:{buffer:o}},{binding:5,resource:{buffer:N}}]}),z=t.beginComputePass();z.setPipeline(Gt),z.setBindGroup(0,m),z.dispatchWorkgroups(E(U,8),E(d,8),te),z.end()}async function xe(t){{let u=t.beginComputePass();u.setPipeline(De),u.setBindGroup(0,Z),u.dispatchWorkgroups(E(96,8),E(96,8),32),u.end()}let e=fe,i=nt;for(let u=0;u<q.length;u++){let k=q[u];H(t,k,e,i,e,Ut[u]);let ae=e;e=i,i=ae,u===13&&t.copyBufferToBuffer(e,0,Me,0,576*128*4),u===18&&t.copyBufferToBuffer(e,0,Ae,0,144*256*4)}{let u=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Qe}}]}),k=t.beginComputePass();k.setPipeline(ge),k.setBindGroup(0,u),k.dispatchWorkgroups(E(12,8),E(12,8),256),k.end()}{let u=e;e=i,i=u}Be(t,e,oe,$,ne,i,St,256,12,12);{let u=e;e=i,i=u}{let u=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Ae}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:gt}}]}),k=t.beginComputePass();k.setPipeline(ge),k.setBindGroup(0,u),k.dispatchWorkgroups(E(12,8),E(12,8),256),k.end()}{let u=e;e=i,i=u}H(t,se,e,i,e,At);{let u=e;e=i,i=u}H(t,ie,e,i,e,Mt);{let u=e;e=i,i=u}P(t,e,we,ke,Ie,Dt,6,12,12),P(t,e,Ge,Le,Je,kt,108,12,12);{let u=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Ht}}]}),k=t.beginComputePass();k.setPipeline(ge),k.setBindGroup(0,u),k.dispatchWorkgroups(E(24,8),E(24,8),256),k.end()}{let u=e;e=i,i=u}Be(t,e,pe,J,X,i,Et,128,24,24);{let u=e;e=i,i=u}{let u=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Me}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:wt}}]}),k=t.beginComputePass();k.setPipeline(ge),k.setBindGroup(0,u),k.dispatchWorkgroups(E(24,8),E(24,8),128),k.end()}{let u=e;e=i,i=u}H(t,ue,e,i,e,Ne);{let u=e;e=i,i=u}H(t,de,e,i,e,qe);{let u=e;e=i,i=u}P(t,e,ye,le,ht,Tt,2,24,24),P(t,e,ft,Oe,rt,Wt,36,24,24),a.queue.submit([t.finish()]);let R=a.createCommandEncoder();R.copyBufferToBuffer(Ie,0,st,0,864*4),R.copyBufferToBuffer(Je,0,ot,0,15552*4),R.copyBufferToBuffer(ht,0,ut,0,576*2*4),R.copyBufferToBuffer(rt,0,ct,0,576*36*4),a.queue.submit([R.finish()]),await Promise.all([st.mapAsync(GPUMapMode.READ),ot.mapAsync(GPUMapMode.READ),ut.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ)]);let V=new Float32Array(st.getMappedRange()).slice(),o=new Float32Array(ot.getMappedRange()).slice(),N=new Float32Array(ut.getMappedRange()).slice(),te=new Float32Array(ct.getMappedRange()).slice();st.unmap(),ot.unmap(),ut.unmap(),ct.unmap();let d=2016,U=new Float32Array(d),m=new Float32Array(d*18),z=0;for(let u=0;u<12;u++)for(let k=0;k<12;k++)for(let ae=0;ae<6;ae++){U[z]=V[ae*144+u*12+k];for(let Se=0;Se<18;Se++){let yt=ae*18+Se;m[z*18+Se]=o[yt*144+u*12+k]}z++}for(let u=0;u<24;u++)for(let k=0;k<24;k++)for(let ae=0;ae<2;ae++){U[z]=N[ae*576+u*24+k];for(let Se=0;Se<18;Se++){let yt=ae*18+Se;m[z*18+Se]=te[yt*576+u*24+k]}z++}return{scores:U,regressors:m}}async function he(t){a.queue.copyExternalImageToTexture({source:t},{texture:bt},[192,192]);let e=a.createCommandEncoder();{let i=e.beginComputePass();i.setPipeline(at),i.setBindGroup(0,Rt),i.dispatchWorkgroups(E(192,16),E(192,16),1),i.end()}return xe(e)}async function re(t,e,i){let R=Math.min(192/e,192/i),V=Math.round(e*R),o=Math.round(i*R),N=(192-V)/2,te=(192-o)/2,d=N/192,U=te/192,m=T(e,i),z;if(t instanceof HTMLVideoElement||t instanceof HTMLImageElement){let pt=new OffscreenCanvas(e,i);pt.getContext("2d").drawImage(t,0,0),z=pt}else z=t;a.queue.copyExternalImageToTexture({source:z},{texture:m},[e,i]);let u=new ArrayBuffer(32),k=new Uint32Array(u),ae=new Float32Array(u);k[0]=e,k[1]=i,k[2]=192,k[3]=0,ae[4]=e/V,ae[5]=i/o,ae[6]=N,ae[7]=te,a.queue.writeBuffer(L,0,u);let Se=a.createBindGroup({layout:$e,entries:[{binding:0,resource:m.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:L}}]}),yt=a.createCommandEncoder();{let pt=yt.beginComputePass();pt.setPipeline(xt),pt.setBindGroup(0,Se),pt.dispatchWorkgroups(E(192,16),E(192,16),1),pt.end()}return{output:await xe(yt),lbPadX:d,lbPadY:U}}async function G(t,e){let i=a.createBuffer({size:e*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),R=a.createCommandEncoder();R.copyBufferToBuffer(t,0,i,0,e*4),a.queue.submit([R.finish()]),await i.mapAsync(GPUMapMode.READ);let V=new Float32Array(i.getMappedRange()).slice();return i.unmap(),i.destroy(),V}async function je(t){a.queue.copyExternalImageToTexture({source:t},{texture:bt},[192,192]);function e(m,z=1e3){let u=m.slice(0,z);return{min:Math.min(...u),max:Math.max(...u),mean:u.reduce((k,ae)=>k+ae,0)/u.length,nonZero:u.filter(k=>k!==0).length,sample:Array.from(u.slice(0,10))}}let i={},R=F(new Uint32Array([192,192,192])),V=a.createBindGroup({layout:ce,entries:[{binding:0,resource:bt.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:R}}]}),o=a.createCommandEncoder(),N=o.beginComputePass();N.setPipeline(at),N.setBindGroup(0,V),N.dispatchWorkgroups(E(192,16),E(192,16),1),N.end(),a.queue.submit([o.finish()]),i.input=e(await G(Pe,36864*3)),o=a.createCommandEncoder();let te=a.createBindGroup({layout:tt,entries:[{binding:0,resource:{buffer:Pe}},{binding:1,resource:{buffer:Bt}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:C}},{binding:4,resource:{buffer:fe}},{binding:5,resource:{buffer:Ct}}]});N=o.beginComputePass(),N.setPipeline(De),N.setBindGroup(0,te),N.dispatchWorkgroups(E(96,8),E(96,8),32),N.end(),a.queue.submit([o.finish()]),i.initConv=e(await G(fe,9216*32));let d=fe,U=nt;for(let m=0;m<q.length;m++){let z=q[m];o=a.createCommandEncoder(),H(o,z,d,U,d,Ut[m]),a.queue.submit([o.finish()]);let u=d;if(d=U,U=u,m===0||m===4||m===9||m===14||m===18||m===19||m===23){let k=z.stride===2?z.inH/2:z.inH,ae=k*k*z.outCh;i[`block${m}`]=e(await G(d,ae))}m===13&&(o=a.createCommandEncoder(),o.copyBufferToBuffer(d,0,Me,0,576*128*4),a.queue.submit([o.finish()])),m===18&&(o=a.createCommandEncoder(),o.copyBufferToBuffer(d,0,Ae,0,144*256*4),a.queue.submit([o.finish()]))}o=a.createCommandEncoder();{let m=F(new Uint32Array([1,256,6,6,12,12])),z=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:d}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(ge),u.setBindGroup(0,z),u.dispatchWorkgroups(E(12,8),E(12,8),256),u.end()}a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpnUpsample6to12=e(await G(d,144*256)),o=a.createCommandEncoder(),Be(o,d,oe,$,ne,U,St,256,12,12),a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpn6to12Conv=e(await G(d,144*256)),i.backbone12Skip=e(await G(Ae,144*256)),o=a.createCommandEncoder();{let m=F(new Uint32Array([1,256,12,12,12,12])),z=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:d}},{binding:1,resource:{buffer:Ae}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(ge),u.setBindGroup(0,z),u.dispatchWorkgroups(E(12,8),E(12,8),256),u.end()}a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpnAdd12=e(await G(d,144*256)),o=a.createCommandEncoder(),H(o,se,d,U,d,At),a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpn12Block1=e(await G(d,144*256)),o=a.createCommandEncoder(),H(o,ie,d,U,d,Mt),a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpn12Block2=e(await G(d,144*256)),o=a.createCommandEncoder(),P(o,d,we,ke,Ie,Dt,6,12,12),a.queue.submit([o.finish()]),i.cls16=e(await G(Ie,864)),o=a.createCommandEncoder(),P(o,d,Ge,Le,Je,kt,108,12,12),a.queue.submit([o.finish()]),i.reg16=e(await G(Je,15552),500),o=a.createCommandEncoder();{let m=F(new Uint32Array([1,256,12,12,24,24])),z=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:d}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(ge),u.setBindGroup(0,z),u.dispatchWorkgroups(E(24,8),E(24,8),256),u.end()}a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpnUpsample12to24=e(await G(d,576*256)),o=a.createCommandEncoder(),Be(o,d,pe,J,X,U,Et,128,24,24),a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpn12to24Conv=e(await G(d,576*128)),i.backbone24Skip=e(await G(Me,576*128)),o=a.createCommandEncoder();{let m=F(new Uint32Array([1,128,24,24,24,24])),z=a.createBindGroup({layout:ee,entries:[{binding:0,resource:{buffer:d}},{binding:1,resource:{buffer:Me}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(ge),u.setBindGroup(0,z),u.dispatchWorkgroups(E(24,8),E(24,8),128),u.end()}a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpnAdd24=e(await G(d,576*128)),o=a.createCommandEncoder(),H(o,ue,d,U,d,Ne),a.queue.submit([o.finish()]);{let m=d;d=U,U=m}i.fpn24Block1=e(await G(d,576*128)),o=a.createCommandEncoder(),H(o,de,d,U,d,qe),a.queue.submit([o.finish()]);{let m=d;d=U,U=m}return i.fpn24Block2=e(await G(d,576*128)),o=a.createCommandEncoder(),P(o,d,ye,le,ht,Tt,2,24,24),a.queue.submit([o.finish()]),i.cls8=e(await G(ht,576*2)),o=a.createCommandEncoder(),P(o,d,ft,Oe,rt,Wt,36,24,24),a.queue.submit([o.finish()]),i.reg8=e(await G(rt,576*36)),i.initWeights=e(await G(Bt,100),100),i.initBias=e(await G(l,32),32),i.cls16Weights=e(await G(we,100),100),i.cls16Bias=e(await G(ke,6),6),i.cls8Weights=e(await G(ye,100),100),i.cls8Bias=e(await G(le,2),2),i.fpn6to12Weights=e(await G(oe,100),100),i}return{device:a,run:he,runWithResize:re,debugRun:je}}function Ba(){let r=[];for(let p=0;p<12;p++)for(let a=0;a<12;a++){let v=(a+.5)/12,h=(p+.5)/12;for(let n=0;n<6;n++)r.push({x:v,y:h})}for(let p=0;p<24;p++)for(let a=0;a<24;a++){let v=(a+.5)/24,h=(p+.5)/24;for(let n=0;n<2;n++)r.push({x:v,y:h})}return r}var sa=Ba();function Ca(r){return 1/(1+Math.exp(-r))}function Kt(r,p){let a=[],{scores:v,regressors:h}=r,n=192;for(let y=0;y<sa.length;y++){let M=Ca(v[y]);if(M<p)continue;let w=sa[y],s=y*18,b=w.x+h[s+0]/n,B=w.y+h[s+1]/n,D=h[s+2]/n,S=h[s+3]/n,g=[];for(let O=0;O<7;O++){let K=w.x+h[s+4+O*2]/n,A=w.y+h[s+4+O*2+1]/n;g.push([K,A])}a.push({score:M,box:[b,B,D,S],keypoints:g})}return a}function Ft(r,p){if(r.length===0)return[];let a=[...r].sort((n,y)=>y.score-n.score),v=[],h=new Set;for(let n=0;n<a.length;n++)if(!h.has(n)){v.push(a[n]);for(let y=n+1;y<a.length;y++)h.has(y)||Ua(a[n],a[y])>p&&h.add(y)}return v}function Ua(r,p){let a=r.box[0]-r.box[2]/2,v=r.box[1]-r.box[3]/2,h=r.box[0]+r.box[2]/2,n=r.box[1]+r.box[3]/2,y=p.box[0]-p.box[2]/2,M=p.box[1]-p.box[3]/2,w=p.box[0]+p.box[2]/2,s=p.box[1]+p.box[3]/2,b=Math.max(a,y),B=Math.max(v,M),D=Math.min(h,w),S=Math.min(n,s),g=Math.max(0,D-b),O=Math.max(0,S-B),K=g*O,A=(h-a)*(n-v),x=(w-y)*(s-M),Q=A+x-K;return Q>0?K/Q:0}function ka(r){let[p,a,v,h]=r.box,n=r.keypoints[0],y=r.keypoints[2],M=y[0]-n[0],w=y[1]-n[1],s=Math.atan2(w,M),B=-Math.PI/2-s,D=Math.max(v,h),g=D*2.6,O=-.5*D,K=Math.cos(B),A=Math.sin(B),x=O*A,Q=O*K;return{centerX:p+x,centerY:a+Q,width:g,height:g,rotation:B}}function oa(r,p={}){let{scoreThreshold:a=.5,nmsThreshold:v=.3,maxHands:h=2}=p;async function n(w){let s=await r.run(w),b=Kt(s,a);return Ft(b,v).slice(0,h).map(ka)}async function y(w){let s=await r.run(w),b=Kt(s,a);return Ft(b,v).slice(0,h)}async function M(w,s,b){let{output:B,lbPadX:D,lbPadY:S}=await r.runWithResize(w,s,b),g=Kt(B,a);return{detections:Ft(g,v).slice(0,h),lbPadX:D,lbPadY:S}}return{detect:n,detectRaw:y,detectRawWithResize:M,model:r}}var It=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Nt(r){let p={};for(let a=0;a<It.length;a++)p[It[a]]=r[a];return p}function ua(r,p,a){return r.initialized?(r.value=a*p+(1-a)*r.value,r.value):(r.value=p,r.initialized=!0,p)}function ca(r,p){let a=2*Math.PI*p*r;return a/(a+1)}function Ga(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function qt(r,p,a,v,h,n){let y=r.lastTime<0?.03333333333333333:a-r.lastTime;r.lastTime=a;let M=ca(y,n),w=r.x.initialized?(p-r.x.value)/y:0,s=ua(r.dx,w,M),b=v+h*Math.abs(s),B=ca(y,b);return ua(r.x,p,B)}function jt(r={}){let{minCutoff:p=.05,beta:a=80,dCutoff:v=1}=r,h=[];function n(w){h.length!==w&&(h=Array.from({length:w},()=>Ga()))}function y(w,s){let b=s??performance.now()/1e3,B=w.length*3;return n(B),w.map((D,S)=>({x:qt(h[S*3],D.x,b,p,a,v),y:qt(h[S*3+1],D.y,b,p,a,v),z:qt(h[S*3+2],D.z,b,p,a,v)}))}function M(){h=[]}return{apply:y,reset:M}}function Aa(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Ma=Aa(`
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

  // Zero for out-of-bounds (matches MediaPipe's kZero border mode)
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
`);function pa(r){let p=r.createShaderModule({code:Ma}),a=r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),v=r.createComputePipeline({layout:r.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:p,entryPoint:"main"}}),h=r.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),n=r.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),y=new Float32Array(8);function M(w,s,b,B,D,S,g){r.queue.writeBuffer(h,0,new Uint32Array([D,S,g,0])),y.set(B),r.queue.writeBuffer(n,0,y);let O=r.createBindGroup({layout:a,entries:[{binding:0,resource:s.createView()},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:h}},{binding:3,resource:{buffer:n}}]}),K=w.beginComputePass();K.setPipeline(v),K.setBindGroup(0,O),K.dispatchWorkgroups(Math.ceil(g/16),Math.ceil(g/16),1),K.end()}return{crop:M}}var Sa="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Ea(r={}){let{weightsUrl:p,scoreThreshold:a=.5,palmScoreThreshold:v=.5,maxHands:h=3,forceF32:n=!1}=r;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let y=(p??Sa).replace(/\/$/,"")+"/",[M,w,s,b]=await Promise.all([fetch(`${y}weights_f16_full.json`),fetch(`${y}weights_f16_full.bin`),fetch(`${y}palm_detection_weights.json`),fetch(`${y}palm_detection_weights.bin`)]);if(!M.ok)throw new Error(`Failed to fetch landmark weights: ${M.status}`);if(!w.ok)throw new Error(`Failed to fetch landmark weights: ${w.status}`);if(!s.ok)throw new Error(`Failed to fetch palm detection weights: ${s.status}`);if(!b.ok)throw new Error(`Failed to fetch palm detection weights: ${b.status}`);let[B,D,S,g]=await Promise.all([M.json(),w.arrayBuffer(),s.json(),b.arrayBuffer()]),O=$t(B,D),K=zt(S,g),A=224,x=await Ot(O,{forceF32:!0});{let l=new OffscreenCanvas(A,A),C=l.getContext("2d");C.fillStyle="#886644",C.fillRect(0,0,A,A),C.fillStyle="#cc9966",C.fillRect(50,50,124,124);let W=await x.runFromCanvas(l);W.landmarks.every(Y=>Y===0)&&W.handflag.every(Y=>Y===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let Q=await ra(K),me=oa(Q,{scoreThreshold:v,maxHands:h}),j=[];for(let l=0;l<h;l++)j.push(jt());let Ee=0,Ve=null,Xe=null;function tt(){return Ve||(Ve=new OffscreenCanvas(192,192)),Ve}function lt(){return Xe||(Xe=new OffscreenCanvas(A,A)),Xe}let ve=x.device,Te=null,Ce=null,ee=null,ce=0,$e=0;function be(){return Te||(Te=pa(ve)),Te}function De(){return Ce||(Ce=ve.createBuffer({size:3*A*A*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),Ce}function _t(l,C){return(!ee||ce!==l||$e!==C)&&(ee&&ee.destroy(),ee=ve.createTexture({size:[l,C],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ce=l,$e=C),ee}let We=0,He=0;function Gt(l,C,W){let q=tt();q.width=192,q.height=192;let Y=q.getContext("2d");Y.clearRect(0,0,192,192);let oe=Math.min(192/C,192/W),$=Math.round(C*oe),ne=Math.round(W*oe),se=(192-$)/2,ie=(192-ne)/2;if(We=se/192,He=ie/192,l instanceof ImageData){let I=new OffscreenCanvas(l.width,l.height);I.getContext("2d").putImageData(l,0,0),Y.drawImage(I,se,ie,$,ne)}else Y.drawImage(l,0,0,C,W,se,ie,$,ne);return q}function ge(l){let C=1/(1-2*We),W=1/(1-2*He);return{score:l.score,box:[(l.box[0]-We)*C,(l.box[1]-He)*W,l.box[2]*C,l.box[3]*W],keypoints:l.keypoints.map(([q,Y])=>[(q-We)*C,(Y-He)*W])}}function at(l,C,W){let q=l.keypoints[0],Y=l.keypoints[2],oe=Y[0]-q[0],$=Y[1]-q[1],ne=Math.atan2(-$,oe),ie=Math.PI/2-ne,[I,pe,J,X]=l.box,ue=Math.max(J*C,X*W),de=0,Ue=-.5*ue/W,we=Math.cos(ie),ke=Math.sin(ie),Ze=I+(de*we-Ue*ke),Ge=pe+(de*ke+Ue*we),Re=ue*2.6;return{centerXpx:Ze*C,centerYpx:Ge*W,sizePx:Re,rotation:ie}}function xt(l,C){let W=lt();W.width=A,W.height=A;let q=W.getContext("2d");q.clearRect(0,0,A,A);let Y=A/C.sizePx,oe=Math.cos(C.rotation),$=Math.sin(C.rotation),ne=oe*Y,se=-$*Y,ie=$*Y,I=oe*Y,pe=A/2,J=-C.centerXpx*ne-C.centerYpx*ie+pe,X=-C.centerXpx*se-C.centerYpx*I+pe;if(q.setTransform(ne,se,ie,I,J,X),l instanceof ImageData){let ue=new OffscreenCanvas(l.width,l.height);ue.getContext("2d").putImageData(l,0,0),q.drawImage(ue,0,0)}else q.drawImage(l,0,0);return q.setTransform(1,0,0,1,0,0),W}function vt(l){return l instanceof HTMLCanvasElement||l instanceof OffscreenCanvas?[l.width,l.height]:typeof ImageBitmap<"u"&&l instanceof ImageBitmap?[l.width,l.height]:l instanceof ImageData?[l.width,l.height]:l instanceof HTMLVideoElement?[l.videoWidth,l.videoHeight]:l instanceof HTMLImageElement?[l.naturalWidth,l.naturalHeight]:[A,A]}async function Pt(l){let[C,W]=vt(l),{detections:q,lbPadX:Y,lbPadY:oe}=await me.detectRawWithResize(l,C,W);if(We=Y,He=oe,q.length===0){if(Ee>0)for(let I=0;I<Ee&&I<j.length;I++)j[I].reset();return Ee=0,[]}let $=[],ne=be(),se=De(),ie=_t(C,W);if(l instanceof ImageData){let I=new OffscreenCanvas(l.width,l.height);I.getContext("2d").putImageData(l,0,0),ve.queue.copyExternalImageToTexture({source:I},{texture:ie},[C,W])}else ve.queue.copyExternalImageToTexture({source:l},{texture:ie},[C,W]);for(let I of q){let pe=ge(I),J=at(pe,C,W),X=Math.cos(J.rotation),ue=Math.sin(J.rotation),de=J.sizePx/A,Ue=A/2,we=X*de/C,ke=-ue*de/C,Ze=J.centerXpx/C-Ue*(we+ke),Ge=ue*de/W,Le=X*de/W,Re=J.centerYpx/W-Ue*(Ge+Le),ye=ve.createCommandEncoder();ne.crop(ye,ie,se,[we,ke,Ze,Ge,Le,Re],C,W,A),ve.queue.submit([ye.finish()]);let le=await x.runFromGPUBuffer(se),ze=le.handflag[0];if(ze<a)continue;let ft=le.handedness[0]>.5,Oe=[];for(let fe=0;fe<21;fe++){let nt=le.landmarks[fe*3],it=le.landmarks[fe*3+1],Fe=le.landmarks[fe*3+2],Ae=(nt-.5)*J.sizePx,Me=(it-.5)*J.sizePx,Ie=X*Ae-ue*Me+J.centerXpx,Je=ue*Ae+X*Me+J.centerYpx;Oe.push({x:Ie/C,y:Je/W,z:Fe})}let Ke=$.length,Pe=Ke<j.length?j[Ke].apply(Oe):Oe;$.push({score:ze,handedness:ft?"right":"left",landmarks:Pe,keypoints:Nt(Pe)})}if($.length<Ee)for(let I=$.length;I<Ee;I++)I<j.length&&j[I].reset();return Ee=$.length,$}function mt(){ee&&ee.destroy(),Ce&&Ce.destroy(),ee=null,Ce=null,Te=null,x.device.destroy(),Q.device.destroy(),Ve=null,Xe=null}return{detect:Pt,dispose:mt,_debug:{palmDetector:me,palmModel:Q,landmarkModel:x,removeLetterbox:ge,detectionToPixelROI:at,cropHandRegion:xt}}}export{It as LANDMARK_NAMES,Ot as compileFullModel,Ea as createHandpose,jt as createLandmarkSmoother,zt as loadWeightsFromBuffer,Nt as toKeypoints};
