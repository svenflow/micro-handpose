function pe(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function jt(r){let c=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+r;for(let v of c)for(;a.includes(`${v}:array<f32>`);)a=a.replace(`${v}:array<f32>`,`${v}:array<f16>`);return a}var Wa=pe(`
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
`),Ha=pe(`
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
`),La=pe(`
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
`),Ra=pe(`
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
`);var za=pe(`
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
`),Oa=pe(`
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
`);function Vt(r){return pe(`
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
`)}var Ka=Vt(!1),Fa=Vt(!0),Na=pe(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Ia=pe(`
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
`);function Yt(r){return pe(`
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
`)}var qa=Yt("sigmoid"),ja=Yt("div256"),Va=pe(`
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
`),Ya=pe(`
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
`);var Xa=pe(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),$a=pe(`
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
`),Za=pe(`
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
`);function Rt(r,c){let a=new Map,v=r.dtype??"float32";for(let h=0;h<r.keys.length;h++){let n=r.keys[h],x=r.shapes[h],A=r.offsets[h],y=x.reduce((B,E)=>B*E,1),s,b;if(v==="float32")s=new Float32Array(c,A,y);else{let B=new DataView(c);s=new Float32Array(y);for(let E=0;E<y;E++)s[E]=da(B.getUint16(A+E*2,!0));b=c.slice(A,A+y*2)}a.set(n,{data:s,shape:x,rawF16:b})}return a}function da(r){let c=r>>15&1,a=r>>10&31,v=r&1023;if(a===0){if(v===0)return c?-0:0;let x=-14,A=v/1024;return(c?-1:1)*Math.pow(2,x)*A}if(a===31)return v===0?c?-1/0:1/0:NaN;let h=a-15,n=1+v/1024;return(c?-1:1)*Math.pow(2,h)*n}var la=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Qa=la.map(([r,c,a,v,h])=>({type:"resmodule",inCh:r,outCh:c,h:a,w:a,stride:v,prefix:h}));function Xt(r,c){let a=new Map,v=r.dtype??"float32",h=new Map;for(let n=0;n<r.keys.length;n++){let x=r.keys[n],A=r.shapes[n],y=r.offsets[n],s=A.reduce((w,D)=>w*D,1),b,B;if(v==="float32")b=new Float32Array(c,y,s);else{let w=new DataView(c);b=new Float32Array(s);for(let D=0;D<s;D++)b[D]=_a(w.getUint16(y+D*2,!0));B=c.slice(y,y+s*2)}let E=h.get(x)??0;h.set(x,E+1);let G=E===0?x:`${x}__${E}`;a.set(G,{data:b,shape:A,rawF16:B})}return a}function _a(r){let c=r>>15&1,a=r>>10&31,v=r&1023;return a===0?v===0?c?-0:0:(c?-1:1)*Math.pow(2,-14)*(v/1024):a===31?v===0?c?-1/0:1/0:NaN:(c?-1:1)*Math.pow(2,a-15)*(1+v/1024)}function Ve(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ma=Ve(`
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
`),fa=Ve(`
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
        let iy=i32(out_y*2u+ky)-1; let ix=i32(out_x*2u+kx)-1;
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
`),ha=Ve(`
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
`),ba=Ve(`
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
`),ga=Ve(`
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
`),wa=Ve(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),ya=Ve(`
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
`),xa=Ve(`
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
`),va=Ve(`
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
`),_t=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],Pa=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function zt(r,c){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let v=a.features.has("shader-f16"),h=v?["shader-f16"]:[],n=await a.requestDevice({requiredFeatures:h,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8)}}),x=r.values().next().value,A=v&&!!x?.rawF16&&!c?.forceF32;function y(g){if(A&&g.rawF16){let f=new Uint16Array(g.rawF16);if(f.length%2!==0){let d=new Uint16Array(f.length+1);return d.set(f),d}return f}return g.data}function s(g){return A&&g.rawF16?Math.ceil(g.rawF16.byteLength/4)*4:g.data.byteLength}function b(g){return A?jt(g):g}let B={r:"read-only-storage",s:"storage",u:"uniform"};function E(g){return n.createBindGroupLayout({entries:g.map((f,d)=>f==="t"?{binding:d,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:d,visibility:GPUShaderStage.COMPUTE,buffer:{type:B[f]}})})}let G=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,w=GPUBufferUsage.STORAGE,D=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,M=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(g,f){return n.createBuffer({size:Math.max(g,4),usage:f})}function X(g,f){return n.createBindGroup({layout:g,entries:f.map((d,K)=>({binding:K,resource:"size"in d?{buffer:d}:d}))})}function J(g,f){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[g]}),compute:{module:f,entryPoint:"main"}})}function $(g){let f=r.get(g);if(!f)throw new Error(`Missing weight: ${g}`);return f}let Pe=n.createShaderModule({code:ma}),Ee=n.createShaderModule({code:b(fa)}),Ye=n.createShaderModule({code:b(ha)}),Xe=n.createShaderModule({code:b(ba)}),tt=n.createShaderModule({code:b(ga)}),mt=n.createShaderModule({code:wa}),ge=n.createShaderModule({code:ya}),Te=n.createShaderModule({code:b(xa)}),Be=n.createShaderModule({code:b(va)}),I=E(["r","r","r","s","u"]),We=E(["r","r","s","u"]),$e=E(["r","s","u"]),re=E(["r","r","r","s","u"]),at=E(["t","s","u"]),yt=J(at,Pe),He=J(I,Ee),Le=J(I,Ye),Ut=J(I,Xe),_e=J(I,tt),nt=J(We,mt),xt=J($e,ge),vt=J(re,Te),ft=J(re,Be),De=1152*112*112*4,_=l(De,D),U=l(De,D),T=l(De,w),z=l(De,w),j=l(De,G),ae=l(672*224*4,G),V=l(1152*4,D),se=l(252,D),oe=l(252,D),Z=l(4,D),N=l(4,D),ne=l(260,D),O=l(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),de=n.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ie=l(4,M);n.queue.writeBuffer(ie,0,new Uint32Array([224]));let ue=$("conv2d"),Ce=$("batch_normalization"),we=y(ue),Ue=y(Ce),Re=l(s(ue),G),ke=l(s(Ce),G),ze=l(24,M);n.queue.writeBuffer(Re,0,we),n.queue.writeBuffer(ke,0,Ue),n.queue.writeBuffer(ze,0,new Uint32Array([3,24,224,224,112,112]));let ye=112,me=112,fe=[];for(let g=0;g<_t.length;g++){let f=_t[g],d=Pa[g],K=ye,te=me,ce=f.stride===2?Math.floor(ye/2):ye,Y=f.stride===2?Math.floor(me/2):me,P={spec:f,inH:K,inW:te,outH:ce,outW:Y,dwW:l(4,G),dwB:l(4,G),dwU:l(32,M)};if(d.expandConvKey){let Me=$(d.expandConvKey),be=$(d.expandBNKey);P.expandW=l(s(Me),G),P.expandB=l(s(be),G),P.expandU=l(16,M),n.queue.writeBuffer(P.expandW,0,y(Me)),n.queue.writeBuffer(P.expandB,0,y(be)),n.queue.writeBuffer(P.expandU,0,new Uint32Array([f.inCh,f.expandCh,K,te]))}let he=$(d.dwWeightKey),je=$(d.dwBNKey);P.dwW=l(s(he),G),P.dwB=l(s(je),G),n.queue.writeBuffer(P.dwW,0,y(he)),n.queue.writeBuffer(P.dwB,0,y(je));let ve=Math.floor(f.dwKernel/2);if(n.queue.writeBuffer(P.dwU,0,new Uint32Array([f.expandCh,K,te,ce,Y,f.stride,ve,f.dwKernel])),f.hasProject&&d.projectConvKey){let Me=$(d.projectConvKey),be=$(d.projectBNKey);P.projectW=l(s(Me),G),P.projectB=l(s(be),G),P.projectU=l(16,M),n.queue.writeBuffer(P.projectW,0,y(Me)),n.queue.writeBuffer(P.projectB,0,y(be)),n.queue.writeBuffer(P.projectU,0,new Uint32Array([f.expandCh,f.outCh,ce,Y]))}fe.push(P),ye=ce,me=Y}function ht(g,f){let d=r.get(g);if(!d)throw new Error(`Missing weight: ${g}`);if(d.shape.length!==f)throw new Error(`Weight ${g} has rank ${d.shape.length}, expected ${f}`);return d}let Ze=$("conv_landmarks"),Oe=$("conv_world_landmarks"),Ke=$("conv_handflag"),xe=$("conv_handedness"),le=$("Identity"),it=$("Identity_1"),rt=$("Identity_2"),Fe=$("Identity_3"),Ge=l(s(Ze),G),Ae=l(s(le),G),Je=l(s(Oe),G),Qe=l(s(Fe),G),st=l(s(Ke),G),ot=l(s(it),G),ut=l(s(xe),G),ct=l(s(rt),G);n.queue.writeBuffer(Ge,0,y(Ze)),n.queue.writeBuffer(Ae,0,y(le)),n.queue.writeBuffer(Je,0,y(Oe)),n.queue.writeBuffer(Qe,0,y(Fe)),n.queue.writeBuffer(st,0,y(Ke)),n.queue.writeBuffer(ot,0,y(it)),n.queue.writeBuffer(ut,0,y(xe)),n.queue.writeBuffer(ct,0,y(rt));let pt=l(8,M),bt=l(8,M),dt=l(8,M),S=l(8,M);n.queue.writeBuffer(pt,0,new Uint32Array([1152,63])),n.queue.writeBuffer(bt,0,new Uint32Array([1152,63])),n.queue.writeBuffer(dt,0,new Uint32Array([1152,1])),n.queue.writeBuffer(S,0,new Uint32Array([1152,1]));let L=l(8,M);n.queue.writeBuffer(L,0,new Uint32Array([1152,ye*me]));let Pt=new Map;for(let g=0;g<_t.length;g++)if(_t[g].hasResidual){let f=fe[g],d=l(4,M);n.queue.writeBuffer(d,0,new Uint32Array([_t[g].outCh*f.outH*f.outW])),Pt.set(g,d)}let Gt=X(at,[de.createView(),_,ie]),At=X(I,[_,Re,ke,U,ze]),Ne=new Float32Array(1),Ie=new Float32Array(1),qe=new Float32Array(63);function Bt(g,f){let d=g.beginComputePass();d.setPipeline(He),d.setBindGroup(0,At),d.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),d.end();let K=U,te=_;for(let ce=0;ce<_t.length;ce++){let Y=_t[ce],P=fe[ce];if(Y.hasResidual){let ve=Y.inCh*P.inH*P.inW*4;g.copyBufferToBuffer(K,0,j,0,ve)}if(d=g.beginComputePass(),P.expandW){let ve=X(I,[K,P.expandW,P.expandB,T,P.expandU]);d.setPipeline(Le),d.setBindGroup(0,ve),d.dispatchWorkgroups(Math.ceil(P.inW/8),Math.ceil(P.inH/8),Y.expandCh)}let he=P.expandW?T:K,je=X(I,[he,P.dwW,P.dwB,z,P.dwU]);if(d.setPipeline(Ut),d.setBindGroup(0,je),d.dispatchWorkgroups(Math.ceil(P.outW/8),Math.ceil(P.outH/8),Y.expandCh),Y.hasProject&&P.projectW){let ve=(Y.hasResidual,te),Me=X(I,[z,P.projectW,P.projectB,ve,P.projectU]);if(d.setPipeline(_e),d.setBindGroup(0,Me),d.dispatchWorkgroups(Math.ceil(P.outW/8),Math.ceil(P.outH/8),Y.outCh),Y.hasResidual){let be=Pt.get(ce),R=X(We,[te,j,K,be]);d.setPipeline(nt),d.setBindGroup(0,R),d.dispatchWorkgroups(Math.ceil(Y.outCh*P.outH*P.outW/256))}else{let be=K;K=te,te=be}}if(d.end(),!Y.hasProject){d=g.beginComputePass();let ve=X($e,[z,V,L]);d.setPipeline(xt),d.setBindGroup(0,ve),d.dispatchWorkgroups(Math.ceil(1152/256));let Me=X(re,[V,Ge,Ae,se,pt]);d.setPipeline(vt),d.setBindGroup(0,Me),d.dispatchWorkgroups(1);let be=X(re,[V,st,ot,Z,dt]);d.setPipeline(ft),d.setBindGroup(0,be),d.dispatchWorkgroups(1);let R=X(re,[V,ut,ct,N,S]);d.setPipeline(ft),d.setBindGroup(0,R),d.dispatchWorkgroups(1),d.end(),g.copyBufferToBuffer(Z,0,ne,0,4),g.copyBufferToBuffer(N,0,ne,4,4),g.copyBufferToBuffer(se,0,ne,8,252),f&&g.copyBufferToBuffer(ne,0,f,0,260);return}}}async function Wt(g){n.queue.writeBuffer(ae,0,g);let f=n.createCommandEncoder();f.copyBufferToBuffer(ae,0,_,0,672*224*4),Bt(f,O),n.queue.submit([f.finish()]);let d=O.mapAsync(GPUMapMode.READ);await n.queue.onSubmittedWorkDone(),await d;let K=new Float32Array(O.getMappedRange());return Ne[0]=K[0],Ie[0]=K[1],qe.set(K.subarray(2,65)),O.unmap(),{handflag:new Float32Array(Ne),handedness:new Float32Array(Ie),landmarks:new Float32Array(qe)}}async function gt(g){n.queue.copyExternalImageToTexture({source:g},{texture:de},[224,224]);let f=n.createCommandEncoder();{let te=f.beginComputePass();te.setPipeline(yt),te.setBindGroup(0,Gt),te.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),te.end()}Bt(f,O),n.queue.submit([f.finish()]);let d=O.mapAsync(GPUMapMode.READ);await n.queue.onSubmittedWorkDone(),await d;let K=new Float32Array(O.getMappedRange());return Ne[0]=K[0],Ie[0]=K[1],qe.set(K.subarray(2,65)),O.unmap(),{handflag:new Float32Array(Ne),handedness:new Float32Array(Ie),landmarks:new Float32Array(qe)}}async function Ht(g){let f=n.createCommandEncoder();f.copyBufferToBuffer(g,0,_,0,672*224*4),Bt(f,O),n.queue.submit([f.finish()]);let d=O.mapAsync(GPUMapMode.READ);await n.queue.onSubmittedWorkDone(),await d;let K=new Float32Array(O.getMappedRange());return Ne[0]=K[0],Ie[0]=K[1],qe.set(K.subarray(2,65)),O.unmap(),{handflag:new Float32Array(Ne),handedness:new Float32Array(Ie),landmarks:new Float32Array(qe)}}async function Mt(){return null}async function St(){return null}async function Ct(g=100){let f=new OffscreenCanvas(224,224),d=f.getContext("2d");d.fillStyle="#886644",d.fillRect(0,0,224,224);for(let Y=0;Y<5;Y++)await gt(f);let K=performance.now();for(let Y=0;Y<g;Y++)await gt(f);let ce=(performance.now()-K)/g;return{avgMs:ce,fps:1e3/ce}}async function Et(g=100){let f=await Ct(g);return{...f,medianMs:f.avgMs,minMs:f.avgMs}}async function Dt(g){return gt(g)}async function Tt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Lt(){return{}}return{device:n,run:Wt,runFromCanvas:gt,runFromGPUBuffer:Ht,runFromCanvasPipelined:Mt,flushPipelined:St,benchmark:Ct,benchmarkGPU:Et,runFromCanvasViaRender:Dt,benchmarkDiagnostic:Tt,debugLayerOutputs:Lt}}function et(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var $t=et(`
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
`),Zt=et(`
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
`),Jt=et(`
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
`),Qt=et(`
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
`),ea=et(`
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
`),ta=et(`
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
`),aa=et(`
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
`),na=et(`
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
`);async function ia(r,c){let a;if(c)a=c;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No GPU adapter found");a=await t.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(t.limits.maxStorageBuffersPerShaderStage,8)}})}let v={r:"read-only-storage",s:"storage",u:"uniform"};function h(t){return a.createBindGroupLayout({entries:t.map((e,i)=>e==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:v[e]}})})}let n=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,x=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function s(t,e){return a.createBuffer({size:Math.max(t,4),usage:e})}function b(t,e,i){a.queue.writeBuffer(t,e,i)}function B(t){let e=s(t.data.byteLength,n);return b(e,0,t.data),e}let E=Array.from(r.keys());function G(t){let e=r.get(t);if(!e)throw new Error(`Weight not found: ${t}`);return e}function w(...t){let e=E.find(i=>t.every(W=>i.includes(W)));if(!e)throw new Error(`Weight not found for: ${t.join(", ")}`);return G(e)}function D(t){let[,e,i,W]=t.shape,q=new Float32Array(W*25);for(let o=0;o<W;o++)for(let F=0;F<e;F++)for(let Q=0;Q<i;Q++)q[o*25+F*5+Q]=t.data[F*i*W+Q*W+o];return q}function M(t){let[e,,,i]=t.shape,W=new Float32Array(e*i);for(let q=0;q<e;q++)for(let o=0;o<i;o++)W[q*i+o]=t.data[q*i+o];return W}let l=a.createShaderModule({code:$t}),X=a.createShaderModule({code:Zt}),J=a.createShaderModule({code:Jt}),$=a.createShaderModule({code:Qt}),Pe=a.createShaderModule({code:ta}),Ee=a.createShaderModule({code:ea}),Ye=a.createShaderModule({code:aa}),Xe=a.createShaderModule({code:na}),tt=h(["r","r","r","r","s","u"]),mt=h(["r","r","r","s","u"]),ge=h(["r","r","r","r","r","s","u"]),Te=h(["r","r","r","s","u"]),Be=h(["r","r","r","r","s","u"]),I=h(["r","r","s","u"]),We=h(["t","s","u"]),$e=h(["t","s","u"]);function re(t,e){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:e,entryPoint:"main"}})}let at=re(tt,l),yt=re(mt,X),He=re(ge,J),Le=re(Te,$),Ut=re(Be,Pe),_e=re(I,Ee),nt=re(We,Ye),xt=re($e,Xe),vt=w("conv2d/Conv2D"),ft=w("batch_normalization/","conv2d/Conv2D"),kt=w("p_re_lu/"),De=B(vt),_=B(ft),U=B(kt),z=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(t=>{let e=w(t.dwKey),i=w(t.pwKey),W=w(t.bnKey),q=w(t.preluKey),o=D(e),F=s(o.byteLength,n);b(F,0,o);let Q=new Float32Array(t.inCh),p=s(Q.byteLength,n);b(p,0,Q);let C=M(i),m=s(C.byteLength,n);b(m,0,C);let H=B(W),u=B(q);return{dwWeightBuf:F,dwBiasBuf:p,pwWeightBuf:m,pwBiasBuf:H,alphaBuf:u,inCh:t.inCh,outCh:t.outCh,stride:t.stride,inH:t.inH}}),j=M(w("conv2d_25/Conv2D")),ae=s(j.byteLength,n);b(ae,0,j);let V=B(w("batch_normalization_25/")),se=B(w("p_re_lu_25/")),oe={dwWeightBuf:(()=>{let t=D(w("depthwise_conv2d_24/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(256),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=M(w("conv2d_26/")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(w("batch_normalization_26/")),alphaBuf:B(w("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},Z={dwWeightBuf:(()=>{let t=D(w("depthwise_conv2d_25/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(256),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=M(w("conv2d_27/Conv2D1")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(w("batch_normalization_27/")),alphaBuf:B(w("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},N=M(w("conv2d_28/Conv2D")),ne=s(N.byteLength,n);b(ne,0,N);let O=B(w("batch_normalization_28/")),de=B(w("p_re_lu_28/")),ie={dwWeightBuf:(()=>{let t=D(w("depthwise_conv2d_26/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(128),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=M(w("conv2d_29/")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(w("batch_normalization_29/")),alphaBuf:B(w("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},ue={dwWeightBuf:(()=>{let t=D(w("depthwise_conv2d_27/")),e=s(t.byteLength,n);return b(e,0,t),e})(),dwBiasBuf:(()=>{let t=new Float32Array(128),e=s(t.byteLength,n);return b(e,0,t),e})(),pwWeightBuf:(()=>{let t=M(w("conv2d_30/Conv2D1")),e=s(t.byteLength,n);return b(e,0,t),e})(),pwBiasBuf:B(w("batch_normalization_30/")),alphaBuf:B(w("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},Ce=M(w("classifier_palm_16_NO_PRUNING/Conv2D")),we=s(Ce.byteLength,n);b(we,0,Ce);let Ue=B(w("classifier_palm_16_NO_PRUNING/BiasAdd")),Re=M(w("regressor_palm_16_NO_PRUNING/Conv2D")),ke=s(Re.byteLength,n);b(ke,0,Re);let ze=B(w("regressor_palm_16_NO_PRUNING/BiasAdd")),ye=M(w("classifier_palm_8_NO_PRUNING/Conv2D")),me=s(ye.byteLength,n);b(me,0,ye);let fe=B(w("classifier_palm_8_NO_PRUNING/BiasAdd")),ht=M(w("regressor_palm_8_NO_PRUNING/Conv2D")),Ze=s(ht.byteLength,n);b(Ze,0,ht);let Oe=B(w("regressor_palm_8_NO_PRUNING/BiasAdd")),Ke=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,xe=s(36864*3*4,n),le=s(Ke,x),it=s(Ke,x),rt=s(Ke,x),Fe=s(576*256*4,x),Ge=s(144*256*4,x|GPUBufferUsage.COPY_DST),Ae=s(576*128*4,x|GPUBufferUsage.COPY_DST),Je=s(864*4,A),Qe=s(15552*4,A),st=s(576*2*4,A),ot=s(576*36*4,A),ut=s(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=s(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),pt=s(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),bt=s(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),dt=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function S(t,e){return Math.ceil(t/e)}function L(t){let e=s(t.byteLength,y);return b(e,0,t),e}let Pt=L(new Uint32Array([1,3,32,192,192,96,96])),Gt=z.map(t=>{let e=t.stride===2?t.inH/2:t.inH,i=e,W=t.stride===2?1:2,q=t.inCh;return{dw:L(new Uint32Array([1,t.inCh,t.inH,t.inH,e,i,t.stride,W])),pw:L(new Uint32Array([1,t.inCh,t.outCh,e,i,q,t.stride,t.inH,t.inH])),outH:e,outW:i}}),At=(()=>{let t=oe,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:L(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:L(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),Ne=(()=>{let t=Z,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:L(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:L(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),Ie=(()=>{let t=ie,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:L(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:L(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),qe=(()=>{let t=ue,e=t.stride===2?t.inH/2:t.inH,i=t.stride===2?1:2;return{dw:L(new Uint32Array([1,t.inCh,t.inH,t.inH,e,e,t.stride,i])),pw:L(new Uint32Array([1,t.inCh,t.outCh,e,e,t.inCh,t.stride,t.inH,t.inH])),outH:e}})(),Bt=L(new Uint32Array([1,256,6,6,12,12])),Wt=L(new Uint32Array([1,256,12,12,12,12])),gt=L(new Uint32Array([1,256,12,12,24,24])),Ht=L(new Uint32Array([1,128,24,24,24,24])),Mt=L(new Uint32Array([1,256,256,12,12])),St=L(new Uint32Array([1,256,128,24,24])),Ct=L(new Uint32Array([1,256,6,12,12])),Et=L(new Uint32Array([1,256,108,12,12])),Dt=L(new Uint32Array([1,128,2,24,24])),Tt=L(new Uint32Array([1,128,36,24,24])),Lt=L(new Uint32Array([192,192,192])),g=a.createBindGroup({layout:We,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:xe}},{binding:2,resource:{buffer:Lt}}]}),f=null,d=0,K=0,te=s(32,y);function ce(t,e){return f&&d===t&&K===e||(f&&f.destroy(),f=a.createTexture({size:[t,e,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),d=t,K=e),f}let Y=a.createBindGroup({layout:tt,entries:[{binding:0,resource:{buffer:xe}},{binding:1,resource:{buffer:De}},{binding:2,resource:{buffer:_}},{binding:3,resource:{buffer:U}},{binding:4,resource:{buffer:le}},{binding:5,resource:{buffer:Pt}}]});function P(t,e,i,W,q,o){let F=o.outH,Q=a.createBindGroup({layout:mt,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:e.dwWeightBuf}},{binding:2,resource:{buffer:e.dwBiasBuf}},{binding:3,resource:{buffer:rt}},{binding:4,resource:{buffer:o.dw}}]}),p=t.beginComputePass();p.setPipeline(yt),p.setBindGroup(0,Q),p.dispatchWorkgroups(S(F,8),S(o.outH,8),e.inCh),p.end();let C=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:rt}},{binding:1,resource:{buffer:q}},{binding:2,resource:{buffer:e.pwWeightBuf}},{binding:3,resource:{buffer:e.pwBiasBuf}},{binding:4,resource:{buffer:e.alphaBuf}},{binding:5,resource:{buffer:W}},{binding:6,resource:{buffer:o.pw}}]}),m=t.beginComputePass();m.setPipeline(He),m.setBindGroup(0,C),m.dispatchWorkgroups(S(F,8),S(o.outH,8),e.outCh),m.end()}function he(t,e,i,W,q,o,F,Q,p){let C=a.createBindGroup({layout:Te,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:W}},{binding:3,resource:{buffer:q}},{binding:4,resource:{buffer:o}}]}),m=t.beginComputePass();m.setPipeline(Le),m.setBindGroup(0,C),m.dispatchWorkgroups(S(p,8),S(Q,8),F),m.end()}function je(t,e,i,W,q,o,F,Q,p,C){let m=a.createBindGroup({layout:Be,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:W}},{binding:3,resource:{buffer:q}},{binding:4,resource:{buffer:o}},{binding:5,resource:{buffer:F}}]}),H=t.beginComputePass();H.setPipeline(Ut),H.setBindGroup(0,m),H.dispatchWorkgroups(S(C,8),S(p,8),Q),H.end()}async function ve(t){{let u=t.beginComputePass();u.setPipeline(at),u.setBindGroup(0,Y),u.dispatchWorkgroups(S(96,8),S(96,8),32),u.end()}let e=le,i=it;for(let u=0;u<z.length;u++){let k=z[u];P(t,k,e,i,e,Gt[u]);let ee=e;e=i,i=ee,u===13&&t.copyBufferToBuffer(e,0,Ae,0,576*128*4),u===18&&t.copyBufferToBuffer(e,0,Ge,0,144*256*4)}{let u=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Bt}}]}),k=t.beginComputePass();k.setPipeline(_e),k.setBindGroup(0,u),k.dispatchWorkgroups(S(12,8),S(12,8),256),k.end()}{let u=e;e=i,i=u}je(t,e,ae,V,se,i,Mt,256,12,12);{let u=e;e=i,i=u}{let u=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Ge}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Wt}}]}),k=t.beginComputePass();k.setPipeline(_e),k.setBindGroup(0,u),k.dispatchWorkgroups(S(12,8),S(12,8),256),k.end()}{let u=e;e=i,i=u}P(t,oe,e,i,e,At);{let u=e;e=i,i=u}P(t,Z,e,i,e,Ne);{let u=e;e=i,i=u}he(t,e,we,Ue,Je,Ct,6,12,12),he(t,e,ke,ze,Qe,Et,108,12,12);{let u=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:gt}}]}),k=t.beginComputePass();k.setPipeline(_e),k.setBindGroup(0,u),k.dispatchWorkgroups(S(24,8),S(24,8),256),k.end()}{let u=e;e=i,i=u}je(t,e,ne,O,de,i,St,128,24,24);{let u=e;e=i,i=u}{let u=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Ae}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Ht}}]}),k=t.beginComputePass();k.setPipeline(_e),k.setBindGroup(0,u),k.dispatchWorkgroups(S(24,8),S(24,8),128),k.end()}{let u=e;e=i,i=u}P(t,ie,e,i,e,Ie);{let u=e;e=i,i=u}P(t,ue,e,i,e,qe);{let u=e;e=i,i=u}he(t,e,me,fe,st,Dt,2,24,24),he(t,e,Ze,Oe,ot,Tt,36,24,24),a.queue.submit([t.finish()]);let W=a.createCommandEncoder();W.copyBufferToBuffer(Je,0,ut,0,864*4),W.copyBufferToBuffer(Qe,0,ct,0,15552*4),W.copyBufferToBuffer(st,0,pt,0,576*2*4),W.copyBufferToBuffer(ot,0,bt,0,576*36*4),a.queue.submit([W.finish()]),await Promise.all([ut.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ),pt.mapAsync(GPUMapMode.READ),bt.mapAsync(GPUMapMode.READ)]);let q=new Float32Array(ut.getMappedRange()).slice(),o=new Float32Array(ct.getMappedRange()).slice(),F=new Float32Array(pt.getMappedRange()).slice(),Q=new Float32Array(bt.getMappedRange()).slice();ut.unmap(),ct.unmap(),pt.unmap(),bt.unmap();let p=2016,C=new Float32Array(p),m=new Float32Array(p*18),H=0;for(let u=0;u<12;u++)for(let k=0;k<12;k++)for(let ee=0;ee<6;ee++){C[H]=q[ee*144+u*12+k];for(let Se=0;Se<18;Se++){let wt=ee*18+Se;m[H*18+Se]=o[wt*144+u*12+k]}H++}for(let u=0;u<24;u++)for(let k=0;k<24;k++)for(let ee=0;ee<2;ee++){C[H]=F[ee*576+u*24+k];for(let Se=0;Se<18;Se++){let wt=ee*18+Se;m[H*18+Se]=Q[wt*576+u*24+k]}H++}return{scores:C,regressors:m}}async function Me(t){a.queue.copyExternalImageToTexture({source:t},{texture:dt},[192,192]);let e=a.createCommandEncoder();{let i=e.beginComputePass();i.setPipeline(nt),i.setBindGroup(0,g),i.dispatchWorkgroups(S(192,16),S(192,16),1),i.end()}return ve(e)}async function be(t,e,i){let W=Math.min(192/e,192/i),q=Math.round(e*W),o=Math.round(i*W),F=(192-q)/2,Q=(192-o)/2,p=F/192,C=Q/192,m=ce(e,i),H;if(t instanceof HTMLVideoElement||t instanceof HTMLImageElement){let lt=new OffscreenCanvas(e,i);lt.getContext("2d").drawImage(t,0,0),H=lt}else H=t;a.queue.copyExternalImageToTexture({source:H},{texture:m},[e,i]);let u=new ArrayBuffer(32),k=new Uint32Array(u),ee=new Float32Array(u);k[0]=e,k[1]=i,k[2]=192,k[3]=0,ee[4]=e/q,ee[5]=i/o,ee[6]=F,ee[7]=Q,a.queue.writeBuffer(te,0,u);let Se=a.createBindGroup({layout:$e,entries:[{binding:0,resource:m.createView()},{binding:1,resource:{buffer:xe}},{binding:2,resource:{buffer:te}}]}),wt=a.createCommandEncoder();{let lt=wt.beginComputePass();lt.setPipeline(xt),lt.setBindGroup(0,Se),lt.dispatchWorkgroups(S(192,16),S(192,16),1),lt.end()}return{output:await ve(wt),lbPadX:p,lbPadY:C}}async function R(t,e){let i=a.createBuffer({size:e*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),W=a.createCommandEncoder();W.copyBufferToBuffer(t,0,i,0,e*4),a.queue.submit([W.finish()]),await i.mapAsync(GPUMapMode.READ);let q=new Float32Array(i.getMappedRange()).slice();return i.unmap(),i.destroy(),q}async function pa(t){a.queue.copyExternalImageToTexture({source:t},{texture:dt},[192,192]);function e(m,H=1e3){let u=m.slice(0,H);return{min:Math.min(...u),max:Math.max(...u),mean:u.reduce((k,ee)=>k+ee,0)/u.length,nonZero:u.filter(k=>k!==0).length,sample:Array.from(u.slice(0,10))}}let i={},W=L(new Uint32Array([192,192,192])),q=a.createBindGroup({layout:We,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:xe}},{binding:2,resource:{buffer:W}}]}),o=a.createCommandEncoder(),F=o.beginComputePass();F.setPipeline(nt),F.setBindGroup(0,q),F.dispatchWorkgroups(S(192,16),S(192,16),1),F.end(),a.queue.submit([o.finish()]),i.input=e(await R(xe,36864*3)),o=a.createCommandEncoder();let Q=a.createBindGroup({layout:tt,entries:[{binding:0,resource:{buffer:xe}},{binding:1,resource:{buffer:De}},{binding:2,resource:{buffer:_}},{binding:3,resource:{buffer:U}},{binding:4,resource:{buffer:le}},{binding:5,resource:{buffer:Pt}}]});F=o.beginComputePass(),F.setPipeline(at),F.setBindGroup(0,Q),F.dispatchWorkgroups(S(96,8),S(96,8),32),F.end(),a.queue.submit([o.finish()]),i.initConv=e(await R(le,9216*32));let p=le,C=it;for(let m=0;m<z.length;m++){let H=z[m];o=a.createCommandEncoder(),P(o,H,p,C,p,Gt[m]),a.queue.submit([o.finish()]);let u=p;if(p=C,C=u,m===0||m===4||m===9||m===14||m===18||m===19||m===23){let k=H.stride===2?H.inH/2:H.inH,ee=k*k*H.outCh;i[`block${m}`]=e(await R(p,ee))}m===13&&(o=a.createCommandEncoder(),o.copyBufferToBuffer(p,0,Ae,0,576*128*4),a.queue.submit([o.finish()])),m===18&&(o=a.createCommandEncoder(),o.copyBufferToBuffer(p,0,Ge,0,144*256*4),a.queue.submit([o.finish()]))}o=a.createCommandEncoder();{let m=L(new Uint32Array([1,256,6,6,12,12])),H=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(_e),u.setBindGroup(0,H),u.dispatchWorkgroups(S(12,8),S(12,8),256),u.end()}a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpnUpsample6to12=e(await R(p,144*256)),o=a.createCommandEncoder(),je(o,p,ae,V,se,C,Mt,256,12,12),a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpn6to12Conv=e(await R(p,144*256)),i.backbone12Skip=e(await R(Ge,144*256)),o=a.createCommandEncoder();{let m=L(new Uint32Array([1,256,12,12,12,12])),H=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:Ge}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(_e),u.setBindGroup(0,H),u.dispatchWorkgroups(S(12,8),S(12,8),256),u.end()}a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpnAdd12=e(await R(p,144*256)),o=a.createCommandEncoder(),P(o,oe,p,C,p,At),a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpn12Block1=e(await R(p,144*256)),o=a.createCommandEncoder(),P(o,Z,p,C,p,Ne),a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpn12Block2=e(await R(p,144*256)),o=a.createCommandEncoder(),he(o,p,we,Ue,Je,Ct,6,12,12),a.queue.submit([o.finish()]),i.cls16=e(await R(Je,864)),o=a.createCommandEncoder(),he(o,p,ke,ze,Qe,Et,108,12,12),a.queue.submit([o.finish()]),i.reg16=e(await R(Qe,15552),500),o=a.createCommandEncoder();{let m=L(new Uint32Array([1,256,12,12,24,24])),H=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:Fe}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(_e),u.setBindGroup(0,H),u.dispatchWorkgroups(S(24,8),S(24,8),256),u.end()}a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpnUpsample12to24=e(await R(p,576*256)),o=a.createCommandEncoder(),je(o,p,ne,O,de,C,St,128,24,24),a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpn12to24Conv=e(await R(p,576*128)),i.backbone24Skip=e(await R(Ae,576*128)),o=a.createCommandEncoder();{let m=L(new Uint32Array([1,128,24,24,24,24])),H=a.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:Ae}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:m}}]}),u=o.beginComputePass();u.setPipeline(_e),u.setBindGroup(0,H),u.dispatchWorkgroups(S(24,8),S(24,8),128),u.end()}a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpnAdd24=e(await R(p,576*128)),o=a.createCommandEncoder(),P(o,ie,p,C,p,Ie),a.queue.submit([o.finish()]);{let m=p;p=C,C=m}i.fpn24Block1=e(await R(p,576*128)),o=a.createCommandEncoder(),P(o,ue,p,C,p,qe),a.queue.submit([o.finish()]);{let m=p;p=C,C=m}return i.fpn24Block2=e(await R(p,576*128)),o=a.createCommandEncoder(),he(o,p,me,fe,st,Dt,2,24,24),a.queue.submit([o.finish()]),i.cls8=e(await R(st,576*2)),o=a.createCommandEncoder(),he(o,p,Ze,Oe,ot,Tt,36,24,24),a.queue.submit([o.finish()]),i.reg8=e(await R(ot,576*36)),i.initWeights=e(await R(De,100),100),i.initBias=e(await R(_,32),32),i.cls16Weights=e(await R(we,100),100),i.cls16Bias=e(await R(Ue,6),6),i.cls8Weights=e(await R(me,100),100),i.cls8Bias=e(await R(fe,2),2),i.fpn6to12Weights=e(await R(ae,100),100),i}return{device:a,run:Me,runWithResize:be,debugRun:pa}}function Ba(){let r=[];for(let c=0;c<12;c++)for(let a=0;a<12;a++){let v=(a+.5)/12,h=(c+.5)/12;for(let n=0;n<6;n++)r.push({x:v,y:h})}for(let c=0;c<24;c++)for(let a=0;a<24;a++){let v=(a+.5)/24,h=(c+.5)/24;for(let n=0;n<2;n++)r.push({x:v,y:h})}return r}var ra=Ba();function Ca(r){return 1/(1+Math.exp(-r))}function Ot(r,c){let a=[],{scores:v,regressors:h}=r,n=192;for(let x=0;x<ra.length;x++){let A=Ca(v[x]);if(A<c)continue;let y=ra[x],s=x*18,b=y.x+h[s+0]/n,B=y.y+h[s+1]/n,E=h[s+2]/n,G=h[s+3]/n,w=[];for(let D=0;D<7;D++){let M=y.x+h[s+4+D*2]/n,l=y.y+h[s+4+D*2+1]/n;w.push([M,l])}a.push({score:A,box:[b,B,E,G],keypoints:w})}return a}function Kt(r,c){if(r.length===0)return[];let a=[...r].sort((n,x)=>x.score-n.score),v=[],h=new Set;for(let n=0;n<a.length;n++)if(!h.has(n)){v.push(a[n]);for(let x=n+1;x<a.length;x++)h.has(x)||Ua(a[n],a[x])>c&&h.add(x)}return v}function Ua(r,c){let a=r.box[0]-r.box[2]/2,v=r.box[1]-r.box[3]/2,h=r.box[0]+r.box[2]/2,n=r.box[1]+r.box[3]/2,x=c.box[0]-c.box[2]/2,A=c.box[1]-c.box[3]/2,y=c.box[0]+c.box[2]/2,s=c.box[1]+c.box[3]/2,b=Math.max(a,x),B=Math.max(v,A),E=Math.min(h,y),G=Math.min(n,s),w=Math.max(0,E-b),D=Math.max(0,G-B),M=w*D,l=(h-a)*(n-v),X=(y-x)*(s-A),J=l+X-M;return J>0?M/J:0}function ka(r){let[c,a,v,h]=r.box,n=r.keypoints[0],x=r.keypoints[2],A=x[0]-n[0],y=x[1]-n[1],s=Math.atan2(y,A),B=-Math.PI/2-s,E=Math.max(v,h),w=E*2.6,D=-.5*E,M=Math.cos(B),l=Math.sin(B),X=D*l,J=D*M;return{centerX:c+X,centerY:a+J,width:w,height:w,rotation:B}}function sa(r,c={}){let{scoreThreshold:a=.5,nmsThreshold:v=.3,maxHands:h=2}=c;async function n(y){let s=await r.run(y),b=Ot(s,a);return Kt(b,v).slice(0,h).map(ka)}async function x(y){let s=await r.run(y),b=Ot(s,a);return Kt(b,v).slice(0,h)}async function A(y,s,b){let{output:B,lbPadX:E,lbPadY:G}=await r.runWithResize(y,s,b),w=Ot(B,a);return{detections:Kt(w,v).slice(0,h),lbPadX:E,lbPadY:G}}return{detect:n,detectRaw:x,detectRawWithResize:A,model:r}}var Ft=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Nt(r){let c={};for(let a=0;a<Ft.length;a++)c[Ft[a]]=r[a];return c}function oa(r,c,a){return r.initialized?(r.value=a*c+(1-a)*r.value,r.value):(r.value=c,r.initialized=!0,c)}function ua(r,c){let a=2*Math.PI*c*r;return a/(a+1)}function Ga(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function It(r,c,a,v,h,n){let x=r.lastTime<0?.03333333333333333:a-r.lastTime;r.lastTime=a;let A=ua(x,n),y=r.x.initialized?(c-r.x.value)/x:0,s=oa(r.dx,y,A),b=v+h*Math.abs(s),B=ua(x,b);return oa(r.x,c,B)}function qt(r={}){let{minCutoff:c=.05,beta:a=80,dCutoff:v=1}=r,h=[];function n(y){h.length!==y&&(h=Array.from({length:y},()=>Ga()))}function x(y,s){let b=s??performance.now()/1e3,B=y.length*3;return n(B),y.map((E,G)=>({x:It(h[G*3],E.x,b,c,a,v),y:It(h[G*3+1],E.y,b,c,a,v),z:It(h[G*3+2],E.z,b,c,a,v)}))}function A(){h=[]}return{apply:x,reset:A}}function Aa(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Ma=Aa(`
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
`);function ca(r){let c=r.createShaderModule({code:Ma}),a=r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),v=r.createComputePipeline({layout:r.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:c,entryPoint:"main"}}),h=r.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),n=r.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),x=new Float32Array(8);function A(y,s,b,B,E,G,w){r.queue.writeBuffer(h,0,new Uint32Array([E,G,w,0])),x.set(B),r.queue.writeBuffer(n,0,x);let D=r.createBindGroup({layout:a,entries:[{binding:0,resource:s.createView()},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:h}},{binding:3,resource:{buffer:n}}]}),M=y.beginComputePass();M.setPipeline(v),M.setBindGroup(0,D),M.dispatchWorkgroups(Math.ceil(w/16),Math.ceil(w/16),1),M.end()}return{crop:A}}var Sa="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Ea(r={}){let{weightsUrl:c,scoreThreshold:a=.5,palmScoreThreshold:v=.5,maxHands:h=3,forceF32:n=!1}=r;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let x=(c??Sa).replace(/\/$/,"")+"/",[A,y,s,b]=await Promise.all([fetch(`${x}weights_f16_full.json`),fetch(`${x}weights_f16_full.bin`),fetch(`${x}palm_detection_weights.json`),fetch(`${x}palm_detection_weights.bin`)]);if(!A.ok)throw new Error(`Failed to fetch landmark weights: ${A.status}`);if(!y.ok)throw new Error(`Failed to fetch landmark weights: ${y.status}`);if(!s.ok)throw new Error(`Failed to fetch palm detection weights: ${s.status}`);if(!b.ok)throw new Error(`Failed to fetch palm detection weights: ${b.status}`);let[B,E,G,w]=await Promise.all([A.json(),y.arrayBuffer(),s.json(),b.arrayBuffer()]),D=Xt(B,E),M=Rt(G,w),l=224,X=await zt(D);{let _=new OffscreenCanvas(l,l),U=_.getContext("2d");U.fillStyle="#886644",U.fillRect(0,0,l,l),U.fillStyle="#cc9966",U.fillRect(50,50,124,124);let T=await X.runFromCanvas(_);T.landmarks.every(j=>j===0)&&T.handflag.every(j=>j===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let J=await ia(M),$=sa(J,{scoreThreshold:v,maxHands:h}),Pe=[];for(let _=0;_<h;_++)Pe.push(qt());let Ee=0,Ye=null,Xe=null;function tt(){return Ye||(Ye=new OffscreenCanvas(192,192)),Ye}function mt(){return Xe||(Xe=new OffscreenCanvas(l,l)),Xe}let ge=X.device,Te=null,Be=null,I=null,We=0,$e=0;function re(){return Te||(Te=ca(ge)),Te}function at(){return Be||(Be=ge.createBuffer({size:3*l*l*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),Be}function yt(_,U){return(!I||We!==_||$e!==U)&&(I&&I.destroy(),I=ge.createTexture({size:[_,U],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),We=_,$e=U),I}let He=0,Le=0;function Ut(_,U,T){let z=tt();z.width=192,z.height=192;let j=z.getContext("2d");j.clearRect(0,0,192,192);let ae=Math.min(192/U,192/T),V=Math.round(U*ae),se=Math.round(T*ae),oe=(192-V)/2,Z=(192-se)/2;if(He=oe/192,Le=Z/192,_ instanceof ImageData){let N=new OffscreenCanvas(_.width,_.height);N.getContext("2d").putImageData(_,0,0),j.drawImage(N,oe,Z,V,se)}else j.drawImage(_,0,0,U,T,oe,Z,V,se);return z}function _e(_){let U=1/(1-2*He),T=1/(1-2*Le);return{score:_.score,box:[(_.box[0]-He)*U,(_.box[1]-Le)*T,_.box[2]*U,_.box[3]*T],keypoints:_.keypoints.map(([z,j])=>[(z-He)*U,(j-Le)*T])}}function nt(_,U,T){let z=_.keypoints[0],j=_.keypoints[2],ae=j[0]-z[0],V=j[1]-z[1],se=Math.atan2(-V,ae),Z=Math.PI/2-se,[N,ne,O,de]=_.box,ie=Math.max(O*U,de*T),ue=0,Ce=-.5*ie/T,we=Math.cos(Z),Ue=Math.sin(Z),Re=N+(ue*we-Ce*Ue),ke=ne+(ue*Ue+Ce*we),ye=ie*2.6;return{centerXpx:Re*U,centerYpx:ke*T,sizePx:ye,rotation:Z}}function xt(_,U){let T=mt();T.width=l,T.height=l;let z=T.getContext("2d");z.clearRect(0,0,l,l);let j=l/U.sizePx,ae=Math.cos(U.rotation),V=Math.sin(U.rotation),se=ae*j,oe=-V*j,Z=V*j,N=ae*j,ne=l/2,O=-U.centerXpx*se-U.centerYpx*Z+ne,de=-U.centerXpx*oe-U.centerYpx*N+ne;if(z.setTransform(se,oe,Z,N,O,de),_ instanceof ImageData){let ie=new OffscreenCanvas(_.width,_.height);ie.getContext("2d").putImageData(_,0,0),z.drawImage(ie,0,0)}else z.drawImage(_,0,0);return z.setTransform(1,0,0,1,0,0),T}function vt(_){return _ instanceof HTMLCanvasElement||_ instanceof OffscreenCanvas?[_.width,_.height]:typeof ImageBitmap<"u"&&_ instanceof ImageBitmap?[_.width,_.height]:_ instanceof ImageData?[_.width,_.height]:_ instanceof HTMLVideoElement?[_.videoWidth,_.videoHeight]:_ instanceof HTMLImageElement?[_.naturalWidth,_.naturalHeight]:[l,l]}async function ft(_){let[U,T]=vt(_),{detections:z,lbPadX:j,lbPadY:ae}=await $.detectRawWithResize(_,U,T);if(He=j,Le=ae,z.length===0){if(Ee>0)for(let N=0;N<Ee&&N<Pe.length;N++)Pe[N].reset();return Ee=0,[]}let V=[],se=re(),oe=at(),Z=yt(U,T);if(_ instanceof ImageData){let N=new OffscreenCanvas(_.width,_.height);N.getContext("2d").putImageData(_,0,0),ge.queue.copyExternalImageToTexture({source:N},{texture:Z},[U,T])}else ge.queue.copyExternalImageToTexture({source:_},{texture:Z},[U,T]);for(let N of z){let ne=_e(N),O=nt(ne,U,T),de=Math.cos(O.rotation),ie=Math.sin(O.rotation),ue=O.sizePx/l,Ce=l/2,we=de*ue/U,Ue=-ie*ue/U,Re=O.centerXpx/U-Ce*(we+Ue),ke=ie*ue/T,ze=de*ue/T,ye=O.centerYpx/T-Ce*(ke+ze),me=ge.createCommandEncoder();se.crop(me,Z,oe,[we,Ue,Re,ke,ze,ye],U,T,l),ge.queue.submit([me.finish()]);let fe=await X.runFromGPUBuffer(oe),ht=fe.handflag[0];if(ht<a)continue;let Ze=fe.handedness[0]>.5,Oe=[];for(let le=0;le<21;le++){let it=fe.landmarks[le*3],rt=fe.landmarks[le*3+1],Fe=fe.landmarks[le*3+2],Ge=(it-.5)*O.sizePx,Ae=(rt-.5)*O.sizePx,Je=de*Ge-ie*Ae+O.centerXpx,Qe=ie*Ge+de*Ae+O.centerYpx;Oe.push({x:Je/U,y:Qe/T,z:Fe})}let Ke=V.length,xe=Ke<Pe.length?Pe[Ke].apply(Oe):Oe;V.push({score:ht,handedness:Ze?"right":"left",landmarks:xe,keypoints:Nt(xe)})}if(V.length<Ee)for(let N=V.length;N<Ee;N++)N<Pe.length&&Pe[N].reset();return Ee=V.length,V}function kt(){I&&I.destroy(),Be&&Be.destroy(),I=null,Be=null,Te=null,X.device.destroy(),J.device.destroy(),Ye=null,Xe=null}return{detect:ft,dispose:kt,_debug:{palmDetector:$,palmModel:J,landmarkModel:X,removeLetterbox:_e,detectionToPixelROI:nt,cropHandRegion:xt}}}export{Ft as LANDMARK_NAMES,zt as compileFullModel,Ea as createHandpose,qt as createLandmarkSmoother,Rt as loadWeightsFromBuffer,Nt as toKeypoints};
