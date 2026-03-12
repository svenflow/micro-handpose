function S(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ka(u){let d=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],b="enable f16;"+u;for(let y of d)for(;b.includes(`${y}:array<f32>`);)b=b.replace(`${y}:array<f32>`,`${y}:array<f16>`);return b}var Le=S(`
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
`),Re=S(`
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
`),Oe=S(`
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
`),Fe=S(`
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
`);function ja(u,d){return Re.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Ja(u,d){return Le.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Qa(u,d){return Oe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function et(u,d){return Fe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function at(u,d){return[8,8]}var tt=S(`
struct PadParams { channels:u32, in_size:u32, out_size:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:PadParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y; let c=gid.z;
  if(x>=params.in_size||y>=params.in_size||c>=params.channels){return;}
  let in_idx=c*params.in_size*params.in_size+y*params.in_size+x;
  let out_idx=c*params.out_size*params.out_size+y*params.out_size+x;
  output[out_idx]=input[in_idx];
}
`),it=S(`
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
`);function nt(u){return S(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${u?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${u?"+skip[out_idx]":""};
}
`)}var rt=nt(!1),st=nt(!0),ut=S(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),ot=S(`
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
`);function pt(u){return S(`
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
  ${u==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var dt=pt("sigmoid"),_t=pt("div256"),lt=S(`
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
`),ct=S(`
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
@compute @workgroup_size(288,1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let c=lid.x;
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
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
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
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
`);function wt(u,d){let y=u%4===0?`var ic:u32=0u;
  while(ic<${u}u){
    sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
    sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
    sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
    sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
    ic+=4u;
  }
  var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
  while(ic<${u}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
  var pw_sum=sum0+pw_bias[c];`,t=`var skip_val:f32=0.0;
  if(c<${u}u){
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
  }`,E=u===d?"":`if(c<${u}u){`;return S(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${u}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${d},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let c=lid.x;
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution (only threads 0..inCh-1)
  ${E}
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
  ${u===d?"":"}"}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  let pw_base=c*${u}u;
  var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
  ${y}
  // Skip connection (only for c < inCh)
  ${t}
  let result=max(0.0,pw_sum+skip_val);
  output[c*outH*outW+out_y*outW+out_x]=result;
}
`)}var mt=S(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),ht=S(`
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
`),bt=S(`
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
  output[0u*out_stride+y*params.out_size+x]=pixel.r;
  output[1u*out_stride+y*params.out_size+x]=pixel.g;
  output[2u*out_stride+y*params.out_size+x]=pixel.b;
}
`);function gt(u,d){let b=new Map,y=u.dtype??"float32";for(let t=0;t<u.keys.length;t++){let E=u.keys[t],C=u.shapes[t],m=u.offsets[t],g=C.reduce((q,c)=>q*c,1),M,k;if(y==="float32")M=new Float32Array(d,m,g);else{let q=new DataView(d);M=new Float32Array(g);for(let c=0;c<g;c++)M[c]=ni(q.getUint16(m+c*2,!0));k=d.slice(m,m+g*2)}b.set(E,{data:M,shape:C,rawF16:k})}return b}function ni(u){let d=u>>15&1,b=u>>10&31,y=u&1023;if(b===0){if(y===0)return d?-0:0;let C=-14,m=y/1024;return(d?-1:1)*Math.pow(2,C)*m}if(b===31)return y===0?d?-1/0:1/0:NaN;let t=b-15,E=1+y/1024;return(d?-1:1)*Math.pow(2,t)*E}var ri=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],ft=ri.map(([u,d,b,y,t])=>({type:"resmodule",inCh:u,outCh:d,h:b,w:b,stride:y,prefix:t})),si=2,ui=5,oi=8,pi=11;async function yt(u){if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");let b=d.features.has("shader-f16"),y=b?["shader-f16"]:[],t=await d.requestDevice({requiredFeatures:y,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(d.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(d.limits.maxComputeInvocationsPerWorkgroup,288)}}),E=u.values().next().value,C=b&&!!E?.rawF16;console.log(C?"[micro-handpose] Using f16 weight storage (shader-f16 enabled)":`[micro-handpose] Using f32 weights (shader-f16: ${b}, f16 data: ${!!E?.rawF16})`);function m(a){return C&&a.rawF16?new Uint16Array(a.rawF16):a.data}function g(a){return C?Ka(a):a}let M={r:"read-only-storage",s:"storage",u:"uniform"};function k(a){return t.createBindGroupLayout({entries:a.map((s,n)=>({binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:M[s]}}))})}function q(a){return t.createBindGroupLayout({entries:a.map((s,n)=>s==="t"?{binding:n,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:M[s]}})})}let c=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,Z=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,ee=GPUBufferUsage.STORAGE,K=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,D=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function p(a,s){return t.createBuffer({size:a,usage:s})}function v(a,s){return t.createBindGroup({layout:a,entries:s.map((n,i)=>({binding:i,resource:"size"in n?{buffer:n}:n}))})}function W(a,s){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:s,entryPoint:"main"}})}let he=t.createShaderModule({code:tt}),be=t.createShaderModule({code:bt}),h=t.createShaderModule({code:g(lt)}),G=t.createShaderModule({code:g(Re)}),P=t.createShaderModule({code:g(Le)}),ie=t.createShaderModule({code:g(Oe)}),fe=t.createShaderModule({code:g(Fe)}),ne=t.createShaderModule({code:g(it)}),N=t.createShaderModule({code:rt}),xt=t.createShaderModule({code:ut}),vt=t.createShaderModule({code:st}),Pt=t.createShaderModule({code:g(ot)}),kt=t.createShaderModule({code:g(dt)}),Bt=t.createShaderModule({code:g(_t)}),Ut=t.createShaderModule({code:g(ct)}),Ie=new Map;function li(a,s){let n=`${a}_${s}`,i=Ie.get(n);return i||(i=t.createShaderModule({code:g(wt(a,s))}),Ie.set(n,i)),i}let ge=k(["r","r","r","s","u"]),ye=k(["r","r","r","r","s","u"]),ze=k(["r","s","u"]),qe=k(["r","r","r","s","u"]),St=k(["r","s","u"]),Dt=k(["r","r","s","u"]),re=k(["r","r","s","u"]),Ne=k(["r","r","r","s","u"]),ae=k(["r","r","r","s","u"]),$e=q(["t","s","u"]),Xe=k(["r","r","r","r","r","r","r","s"]),xe=k(["r","r","r","r","r","s","u"]),Gt=t.createPipelineLayout({bindGroupLayouts:[ge]}),Wt=t.createPipelineLayout({bindGroupLayouts:[ye]}),se=a=>t.createComputePipeline({layout:Gt,compute:{module:a,entryPoint:"main"}}),ue=a=>t.createComputePipeline({layout:Wt,compute:{module:a,entryPoint:"main"}}),At=se(G),Ct=se(P),Ht=ue(ie),Et=ue(fe),Ye=new Map,Ve=new Map,Ze=new Map,Ke=new Map;Ye.set("8,8",At),Ve.set("8,8",Ct),Ze.set("8,8",Ht),Ke.set("8,8",Et);function oe(a,s,n,i,e){let w=`${s},${n}`,l=a.get(w);return l||(l=e(t.createShaderModule({code:g(i(s,n))})),a.set(w,l)),l}let Mt=(a,s)=>oe(Ye,a,s,ja,se),Tt=(a,s)=>oe(Ve,a,s,Ja,se),Lt=(a,s)=>oe(Ze,a,s,Qa,ue),Rt=(a,s)=>oe(Ke,a,s,et,ue),$=ft.map(a=>{let s=a.stride===2?a.h/2:a.h,n=a.stride===2?a.w/2:a.w,[i,e]=at(a.inCh,s),w=a.h>=64,l=s>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:w?Tt(i,e):Mt(i,e),pwPipeline:l?Rt(i,e):Lt(i,e),dwDispatchX:Math.ceil(n/i),dwDispatchY:Math.ceil(s/e),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(n/i),pwDispatchY:Math.ceil(s/e),pwDispatchZ:l?a.outCh/2:a.outCh}}),je=W(ze,he),Ot=W(qe,ne);W(St,N),W(Dt,xt);let ve=W(re,vt),Ft=W(Ne,Pt);W(ae,kt),W(ae,Bt);let X=W($e,be),It=W(Xe,h),zt=W(xe,Ut),Pe=1*288*128*128*4,Je=p(3*256*256*4,c),ke=p(3*257*257*4,ee),Qe=p(12,D);t.queue.writeBuffer(Qe,0,new Uint32Array([3,256,257]));let B=p(Pe,Z),H=p(Pe,K),pe=p(Pe,ee),ea=p(3072*64*4,c),aa=p(3072*32*4,c),ta=p(1536*16*4,c),ia=p(6144*64*4,ee),j=p(260,K),x=p(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);p(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let L=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),na=p(8,D);t.queue.writeBuffer(na,0,new Uint32Array([256,257]));let ra=u.get("backbone1.1.weight"),sa=u.get("backbone1.1.bias");if(!ra||!sa)throw new Error("Missing input conv weights");let ua=m(ra),oa=m(sa),pa=p(ua.byteLength,c),da=p(oa.byteLength,c),_a=p(28,D);t.queue.writeBuffer(pa,0,ua),t.queue.writeBuffer(da,0,oa),t.queue.writeBuffer(_a,0,new Uint32Array([1,3,24,257,257,128,128]));let la=u.get("backbone6.1.weight"),ca=u.get("backbone6.1.bias");if(!la||!ca)throw new Error("Missing backbone6.1 conv1x1 weights");let wa=m(la),ma=m(ca),ha=p(wa.byteLength,c),ba=p(ma.byteLength,c),fa=p(20,D);t.queue.writeBuffer(ha,0,wa),t.queue.writeBuffer(ba,0,ma),t.queue.writeBuffer(fa,0,new Uint32Array([1,96,48,32,32]));let ga=u.get("handflag.weight"),ya=u.get("handflag.bias");if(!ga||!ya)throw new Error("Missing handflag weights");let xa=m(ga),va=m(ya),Be=p(xa.byteLength,c),Ue=p(va.byteLength,c),Pa=p(12,D);t.queue.writeBuffer(Be,0,xa),t.queue.writeBuffer(Ue,0,va),t.queue.writeBuffer(Pa,0,new Uint32Array([1,288,1]));let ka=u.get("handedness.weight"),Ba=u.get("handedness.bias");if(!ka||!Ba)throw new Error("Missing handedness weights");let Ua=m(ka),Sa=m(Ba),Se=p(Ua.byteLength,c),De=p(Sa.byteLength,c),Da=p(12,D);t.queue.writeBuffer(Se,0,Ua),t.queue.writeBuffer(De,0,Sa),t.queue.writeBuffer(Da,0,new Uint32Array([1,288,1]));let Ga=u.get("reg_3d.weight"),Wa=u.get("reg_3d.bias");if(!Ga||!Wa)throw new Error("Missing reg_3d weights");let Aa=m(Ga),Ca=m(Wa),Ge=p(Aa.byteLength,c),We=p(Ca.byteLength,c),Ha=p(12,D);t.queue.writeBuffer(Ge,0,Aa),t.queue.writeBuffer(We,0,Ca),t.queue.writeBuffer(Ha,0,new Uint32Array([1,288,63]));let te=ft.map(a=>{let{inCh:s,outCh:n,h:i,w:e,stride:w,prefix:l}=a,f=w===2?i/2:i,_=w===2?e/2:e,o=w===1?2:1,r=u.get(`${l}convs.0.weight`),A=u.get(`${l}convs.0.bias`),U=u.get(`${l}convs.1.weight`),V=u.get(`${l}convs.1.bias`);if(!r||!A||!U||!V)throw new Error(`Missing weights for ${l}`);let Fa=m(r),Ia=m(A),za=m(U),qa=m(V),Na=p(Fa.byteLength,c),$a=p(Ia.byteLength,c),Xa=p(za.byteLength,c),Ya=p(qa.byteLength,c),Va=p(32,D),Za=p(36,D);return t.queue.writeBuffer(Na,0,Fa),t.queue.writeBuffer($a,0,Ia),t.queue.writeBuffer(Xa,0,za),t.queue.writeBuffer(Ya,0,qa),t.queue.writeBuffer(Va,0,new Uint32Array([1,s,i,e,f,_,w,o])),t.queue.writeBuffer(Za,0,new Uint32Array([1,s,n,f,_,Math.max(0,n-s),w,i,e])),{dwWeight:Na,dwBias:$a,pwWeight:Xa,pwBias:Ya,dwUniform:Va,pwUniform:Za,spec:a,outH:f,outW:_}});function J(a){let s=p(a.length*4,D);return t.queue.writeBuffer(s,0,new Uint32Array(a)),s}let qt=J([1,96,8,8,16,16]),Nt=J([1,96,16,16,32,32]),$t=J([1,48,32,32,64,64]);J([1536*16]),J([3072*32]),J([3072*64]);let Ea=v(ze,[Je,ke,Qe]),Xt=v(qe,[ke,pa,da,B,_a]),R=[],O=[],F=[],I=[];for(let a of te)R.push(v(ge,[B,a.dwWeight,a.dwBias,pe,a.dwUniform])),O.push(v(ye,[pe,B,a.pwWeight,a.pwBias,H,a.pwUniform])),F.push(v(ge,[H,a.dwWeight,a.dwBias,pe,a.dwUniform])),I.push(v(ye,[pe,H,a.pwWeight,a.pwBias,B,a.pwUniform]));let Yt=v(re,[B,ta,H,qt]),Vt=v(re,[B,aa,H,Nt]),Zt=v(Ne,[B,ha,ba,ia,fa]),Kt=v(re,[ia,ea,H,$t]);v(ae,[B,Be,Ue,j,Pa]),v(ae,[B,Se,De,j,Da]),v(ae,[B,Ge,We,j,Ha]);let Y=v($e,[L.createView(),ke,na]),jt=v(Xe,[B,Be,Ue,Se,De,Ge,We,j]),Ae=24,Ma=[],Ta=[];for(let a=Ae;a<te.length;a++){let s=te[a];Ma.push(v(xe,[B,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,H,s.dwUniform])),Ta.push(v(xe,[H,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,B,s.dwUniform]))}let Ce=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Ce.globalCompositeOperation="copy";let La=new OffscreenCanvas(9,8),de=La.getContext("webgpu"),_e=null,He=null;if(de){de.configure({device:t,format:"rgba8unorm",alphaMode:"premultiplied"});let a=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),s=t.createShaderModule({code:mt}),n=t.createShaderModule({code:ht});_e=t.createRenderPipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:s,entryPoint:"vs"},fragment:{module:n,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),He=t.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:j}}]})}let le=new Float32Array(1),ce=new Float32Array(1),we=new Float32Array(63);function T(a,s){let n=!0,i=0,e=a.beginComputePass();for(e.setPipeline(Ot),e.setBindGroup(0,Xt),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=si;i++){let _=n?R[i]:F[i],o=n?O[i]:I[i],r=$[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let w=n?B:H;for(a.copyBufferToBuffer(w,0,ea,0,3072*64*4),e=a.beginComputePass();i<=ui;i++){let _=n?R[i]:F[i],o=n?O[i]:I[i],r=$[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let l=n?B:H;for(a.copyBufferToBuffer(l,0,aa,0,3072*32*4),e=a.beginComputePass();i<=oi;i++){let _=n?R[i]:F[i],o=n?O[i]:I[i],r=$[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let f=n?B:H;for(a.copyBufferToBuffer(f,0,ta,0,1536*16*4),e=a.beginComputePass();i<=pi;i++){let _=n?R[i]:F[i],o=n?O[i]:I[i],r=$[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.setPipeline(ve),e.setBindGroup(0,Yt),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),n=!1,e=a.beginComputePass();{let _=n?R[i]:F[i],o=n?O[i]:I[i],r=$[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,i++}e.setPipeline(ve),e.setBindGroup(0,Vt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),n=!1,e=a.beginComputePass();{let _=n?R[i]:F[i],o=n?O[i]:I[i],r=$[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,i++}for(e.setPipeline(Ft),e.setBindGroup(0,Zt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(ve),e.setBindGroup(0,Kt),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),n=!1,e=a.beginComputePass();i<Ae;i++){let _=n?R[i]:F[i],o=n?O[i]:I[i],r=$[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}for(;i<te.length;i++){let _=i-Ae,o=n?Ma[_]:Ta[_],r=te[i];e.setPipeline(zt),e.setBindGroup(0,o),e.dispatchWorkgroups(r.outW,r.outH,1),n=!n}e.setPipeline(It),e.setBindGroup(0,jt),e.dispatchWorkgroups(1),e.end(),s&&a.copyBufferToBuffer(j,0,s,0,260)}async function me(a){t.queue.writeBuffer(Je,0,a);let s=t.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(je),e.setBindGroup(0,Ea),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}T(s,x),t.queue.submit([s.finish()]);let n=x.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(x.getMappedRange());return le[0]=i[0],ce[0]=i[1],we.set(i.subarray(2,65)),x.unmap(),{handflag:new Float32Array(le),handedness:new Float32Array(ce),landmarks:new Float32Array(we)}}async function Ee(a){t.queue.copyExternalImageToTexture({source:a},{texture:L},[256,256]);let s=t.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(X),e.setBindGroup(0,Y),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}T(s,x),t.queue.submit([s.finish()]);let n=x.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(x.getMappedRange());return le[0]=i[0],ce[0]=i[1],we.set(i.subarray(2,65)),x.unmap(),{handflag:new Float32Array(le),handedness:new Float32Array(ce),landmarks:new Float32Array(we)}}async function Ra(a){if(!_e||!He||!de)throw new Error("Render-based readback not available (no WebGPU canvas context)");t.queue.copyExternalImageToTexture({source:a},{texture:L},[256,256]);let s=t.createCommandEncoder();{let _=s.beginComputePass();_.setPipeline(X),_.setBindGroup(0,Y),_.dispatchWorkgroups(16,16,1),_.end()}T(s,null);let n=de.getCurrentTexture(),i=s.beginRenderPass({colorAttachments:[{view:n.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});i.setPipeline(_e),i.setBindGroup(0,He),i.draw(3),i.end(),t.queue.submit([s.finish()]),await t.queue.onSubmittedWorkDone(),Ce.drawImage(La,0,0);let w=Ce.getImageData(0,0,9,8).data,l=new Float32Array(65),f=new DataView(new ArrayBuffer(4));for(let _=0;_<65;_++){let o=_*4;f.setUint8(0,w[o]),f.setUint8(1,w[o+1]),f.setUint8(2,w[o+2]),f.setUint8(3,w[o+3]),l[_]=f.getFloat32(0)}return{handflag:new Float32Array([l[0]]),handedness:new Float32Array([l[1]]),landmarks:new Float32Array(l.subarray(2,65))}}let Jt=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Me=0,Qt=[x,Jt],Q=null,z=null;async function Te(a){let s=Qt[Me];Me=1-Me,t.queue.copyExternalImageToTexture({source:a},{texture:L},[256,256]);let n=t.createCommandEncoder();{let e=n.beginComputePass();e.setPipeline(X),e.setBindGroup(0,Y),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}T(n,s),t.queue.submit([n.finish()]);let i=null;if(Q!==null&&z!==null){await Q;let e=new Float32Array(z.getMappedRange());i={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},z.unmap()}return z=s,Q=s.mapAsync(GPUMapMode.READ),i}async function Oa(){if(!Q||!z)return null;await Q;let a=new Float32Array(z.getMappedRange()),s={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return z.unmap(),Q=null,z=null,s}async function ei(a=50){let s=new Float32Array(196608);for(let e=0;e<5;e++)await me(s);let n=[];for(let e=0;e<a;e++){let w=performance.now();await me(s),n.push(performance.now()-w)}let i=n.reduce((e,w)=>e+w,0)/n.length;return{avgMs:i,fps:1e3/i}}async function ai(a=50){let s=new Float32Array(196608);for(let l=0;l<5;l++)await me(s);let n=[];for(let l=0;l<a;l++){let f=t.createCommandEncoder();{let o=f.beginComputePass();o.setPipeline(je),o.setBindGroup(0,Ea),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),o.end()}T(f,x);let _=performance.now();t.queue.submit([f.finish()]),await t.queue.onSubmittedWorkDone(),n.push(performance.now()-_)}n.sort((l,f)=>l-f);let i=n.reduce((l,f)=>l+f,0)/n.length,e=n[Math.floor(n.length/2)],w=n[0];return{avgMs:i,fps:1e3/i,medianMs:e,minMs:w}}function ti(a){t.queue.copyExternalImageToTexture({source:a},{texture:L},[256,256]);let s=t.createCommandEncoder();{let n=s.beginComputePass();n.setPipeline(X),n.setBindGroup(0,Y),n.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),n.end()}T(s,x),t.queue.submit([s.finish()])}async function ii(a,s=50){function n(o){let r=[...o].sort((A,U)=>A-U);return{median:r[Math.floor(r.length/2)],min:r[0]}}for(let o=0;o<10;o++)await Ee(a);let i=[];for(let o=0;o<s;o++){t.queue.copyExternalImageToTexture({source:a},{texture:L},[256,256]);let r=t.createCommandEncoder();{let U=r.beginComputePass();U.setPipeline(X),U.setBindGroup(0,Y),U.dispatchWorkgroups(16,16,1),U.end()}T(r,x);let A=performance.now();t.queue.submit([r.finish()]),await t.queue.onSubmittedWorkDone(),i.push(performance.now()-A)}let e=[];for(let o=0;o<s;o++){t.queue.copyExternalImageToTexture({source:a},{texture:L},[256,256]);let r=t.createCommandEncoder();{let V=r.beginComputePass();V.setPipeline(X),V.setBindGroup(0,Y),V.dispatchWorkgroups(16,16,1),V.end()}T(r,x),t.queue.submit([r.finish()]);let A=x.mapAsync(GPUMapMode.READ),U=performance.now();await t.queue.onSubmittedWorkDone(),await A,x.getMappedRange(),x.unmap(),e.push(performance.now()-U)}let w=[];for(let o=0;o<s;o++){t.queue.copyExternalImageToTexture({source:a},{texture:L},[256,256]);let r=t.createCommandEncoder();{let U=r.beginComputePass();U.setPipeline(X),U.setBindGroup(0,Y),U.dispatchWorkgroups(16,16,1),U.end()}T(r,x),t.queue.submit([r.finish()]);let A=performance.now();await x.mapAsync(GPUMapMode.READ),x.getMappedRange(),x.unmap(),w.push(performance.now()-A)}let l=[];for(let o=0;o<s;o++){let r=performance.now();await Ee(a),l.push(performance.now()-r)}await Te(a);let f=[];for(let o=0;o<s;o++){let r=performance.now();await Te(a),f.push(performance.now()-r)}await Oa();let _=null;if(_e){let o=[];for(let r=0;r<s;r++){let A=performance.now();await Ra(a),o.push(performance.now()-A)}_=n(o)}return{gpuOnly:n(i),mapAsyncOnly:n(e),mapAsyncNoWait:n(w),total:n(l),pipelined:n(f),renderReadback:_}}return{device:t,run:me,runFromCanvas:Ee,runFromCanvasViaRender:Ra,runFromCanvasPipelined:Te,flushPipelined:Oa,benchmark:ei,benchmarkGPU:ai,benchmarkDiagnostic:ii,_device:t,_benchmarkSubmitOnly:ti}}var di="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function _i(u={}){let{weightsUrl:d,scoreThreshold:b=.5}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let y=d??di,t=y.endsWith("/")?y:`${y}/`,E=`${t}weights_f16.json`,C=`${t}weights_f16.bin`,[m,g]=await Promise.all([fetch(E),fetch(C)]);if(!m.ok)throw new Error(`Failed to fetch weights metadata: ${m.status}`);if(!g.ok)throw new Error(`Failed to fetch weights binary: ${g.status}`);let M=await m.json(),k=await g.arrayBuffer(),q=gt(M,k),c=await yt(q),Z=null;function ee(){return Z||(Z=new OffscreenCanvas(256,256)),Z}async function K(h){if(h instanceof HTMLCanvasElement||h instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&h instanceof ImageBitmap)return h;let G=ee();G.width=256,G.height=256;let P=G.getContext("2d");return h instanceof ImageData?P.putImageData(h,0,0):P.drawImage(h,0,0,256,256),G}function D(h,G,P){let ie=h[0];if(ie<b)return null;let fe=G[0]>.5,ne=[];for(let N=0;N<21;N++)ne.push({x:P[N*3],y:P[N*3+1],z:P[N*3+2]});return{score:ie,handedness:fe?"right":"left",landmarks:ne}}async function p(h){let G=await K(h),P=await c.runFromCanvas(G);return D(P.handflag,P.handedness,P.landmarks)}async function v(h){let G=await K(h),P=await c.runFromCanvasPipelined(G);return P?D(P.handflag,P.handedness,P.landmarks):null}async function W(){let h=await c.flushPipelined();return h?D(h.handflag,h.handedness,h.landmarks):null}function he(){c.device.destroy(),Z=null}async function be(h){let G=await K(h);return c.benchmarkDiagnostic(G)}return{detect:p,detectPipelined:v,flushPipelined:W,dispose:he,benchmarkDiagnostic:be}}export{_i as createHandpose};
