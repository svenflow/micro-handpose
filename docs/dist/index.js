function S(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ka(u){let d=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],b="enable f16;"+u;for(let g of d)for(;b.includes(`${g}:array<f32>`);)b=b.replace(`${g}:array<f32>`,`${g}:array<f16>`);return b}var Le=S(`
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
`);function wt(u,d){let g=u%4===0?`var ic:u32=0u;
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
  }`,M=u===d?"":`if(c<${u}u){`;return S(`
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
  ${M}
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
  ${g}
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
`);function gt(u,d){let b=new Map,g=u.dtype??"float32";for(let t=0;t<u.keys.length;t++){let M=u.keys[t],A=u.shapes[t],w=u.offsets[t],T=A.reduce((P,D)=>P*D,1),x,N;if(g==="float32")x=new Float32Array(d,w,T);else{let P=new DataView(d);x=new Float32Array(T);for(let D=0;D<T;D++)x[D]=ri(P.getUint16(w+D*2,!0));N=d.slice(w,w+T*2)}b.set(M,{data:x,shape:A,rawF16:N})}return b}function ri(u){let d=u>>15&1,b=u>>10&31,g=u&1023;if(b===0){if(g===0)return d?-0:0;let A=-14,w=g/1024;return(d?-1:1)*Math.pow(2,A)*w}if(b===31)return g===0?d?-1/0:1/0:NaN;let t=b-15,M=1+g/1024;return(d?-1:1)*Math.pow(2,t)*M}var si=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],ft=si.map(([u,d,b,g,t])=>({type:"resmodule",inCh:u,outCh:d,h:b,w:b,stride:g,prefix:t})),ui=2,oi=5,pi=8,di=11;async function yt(u){if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");let b=d.features.has("shader-f16"),g=b?["shader-f16"]:[],t=await d.requestDevice({requiredFeatures:g,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(d.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(d.limits.maxComputeInvocationsPerWorkgroup,288)}}),M=u.values().next().value,A=b&&!!M?.rawF16;console.log(A?"[micro-handpose] Using f16 weight storage (shader-f16 enabled)":`[micro-handpose] Using f32 weights (shader-f16: ${b}, f16 data: ${!!M?.rawF16})`);function w(a){if(A&&a.rawF16){let r=new Uint16Array(a.rawF16);if(r.length%2!==0){let i=new Uint16Array(r.length+1);return i.set(r),i}return r}return a.data}function T(a){if(A&&a.rawF16){let r=a.rawF16.byteLength;return Math.ceil(r/4)*4}return a.data.byteLength}function x(a){return A?Ka(a):a}let N={r:"read-only-storage",s:"storage",u:"uniform"};function P(a){return t.createBindGroupLayout({entries:a.map((r,i)=>({binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:N[r]}}))})}function D(a){return t.createBindGroupLayout({entries:a.map((r,i)=>r==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:N[r]}})})}let m=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,he=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,$=GPUBufferUsage.STORAGE,j=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,C=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function p(a,r){return t.createBuffer({size:a,usage:r})}function v(a,r){return t.createBindGroup({layout:a,entries:r.map((i,n)=>({binding:n,resource:"size"in i?{buffer:i}:i}))})}function W(a,r){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:r,entryPoint:"main"}})}let be=t.createShaderModule({code:tt}),h=t.createShaderModule({code:bt}),G=t.createShaderModule({code:x(lt)}),k=t.createShaderModule({code:x(Re)}),ie=t.createShaderModule({code:x(Le)}),fe=t.createShaderModule({code:x(Oe)}),ne=t.createShaderModule({code:x(Fe)}),X=t.createShaderModule({code:x(it)}),xt=t.createShaderModule({code:rt}),vt=t.createShaderModule({code:ut}),Pt=t.createShaderModule({code:st}),kt=t.createShaderModule({code:x(ot)}),Bt=t.createShaderModule({code:x(dt)}),Ut=t.createShaderModule({code:x(_t)}),St=t.createShaderModule({code:x(ct)}),Ie=new Map;function ci(a,r){let i=`${a}_${r}`,n=Ie.get(i);return n||(n=t.createShaderModule({code:x(wt(a,r))}),Ie.set(i,n)),n}let ge=P(["r","r","r","s","u"]),ye=P(["r","r","r","r","s","u"]),ze=P(["r","s","u"]),qe=P(["r","r","r","s","u"]),Dt=P(["r","s","u"]),Gt=P(["r","r","s","u"]),re=P(["r","r","s","u"]),Ne=P(["r","r","r","s","u"]),ae=P(["r","r","r","s","u"]),$e=D(["t","s","u"]),Xe=P(["r","r","r","r","r","r","r","s"]),xe=P(["r","r","r","r","r","s","u"]),Wt=t.createPipelineLayout({bindGroupLayouts:[ge]}),At=t.createPipelineLayout({bindGroupLayouts:[ye]}),se=a=>t.createComputePipeline({layout:Wt,compute:{module:a,entryPoint:"main"}}),ue=a=>t.createComputePipeline({layout:At,compute:{module:a,entryPoint:"main"}}),Ct=se(k),Ht=se(ie),Et=ue(fe),Mt=ue(ne),Ye=new Map,Ve=new Map,Ze=new Map,Ke=new Map;Ye.set("8,8",Ct),Ve.set("8,8",Ht),Ze.set("8,8",Et),Ke.set("8,8",Mt);function oe(a,r,i,n,e){let c=`${r},${i}`,l=a.get(c);return l||(l=e(t.createShaderModule({code:x(n(r,i))})),a.set(c,l)),l}let Tt=(a,r)=>oe(Ye,a,r,ja,se),Lt=(a,r)=>oe(Ve,a,r,Ja,se),Rt=(a,r)=>oe(Ze,a,r,Qa,ue),Ot=(a,r)=>oe(Ke,a,r,et,ue),Y=ft.map(a=>{let r=a.stride===2?a.h/2:a.h,i=a.stride===2?a.w/2:a.w,[n,e]=at(a.inCh,r),c=a.h>=64,l=r>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:c?Lt(n,e):Tt(n,e),pwPipeline:l?Ot(n,e):Rt(n,e),dwDispatchX:Math.ceil(i/n),dwDispatchY:Math.ceil(r/e),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(i/n),pwDispatchY:Math.ceil(r/e),pwDispatchZ:l?a.outCh/2:a.outCh}}),je=W(ze,be),Ft=W(qe,X);W(Dt,xt),W(Gt,vt);let ve=W(re,Pt),It=W(Ne,kt);W(ae,Bt),W(ae,Ut);let V=W($e,h),zt=W(Xe,G),qt=W(xe,St),Pe=1*288*128*128*4,Je=p(3*256*256*4,m),ke=p(3*257*257*4,$),Qe=p(12,C);t.queue.writeBuffer(Qe,0,new Uint32Array([3,256,257]));let B=p(Pe,he),E=p(Pe,j),pe=p(Pe,$),ea=p(3072*64*4,m),aa=p(3072*32*4,m),ta=p(1536*16*4,m),ia=p(6144*64*4,$),J=p(260,j),y=p(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);p(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let R=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),na=p(8,C);t.queue.writeBuffer(na,0,new Uint32Array([256,257]));let ra=u.get("backbone1.1.weight"),sa=u.get("backbone1.1.bias");if(!ra||!sa)throw new Error("Missing input conv weights");let ua=w(ra),oa=w(sa),pa=p(ua.byteLength,m),da=p(oa.byteLength,m),_a=p(28,C);t.queue.writeBuffer(pa,0,ua),t.queue.writeBuffer(da,0,oa),t.queue.writeBuffer(_a,0,new Uint32Array([1,3,24,257,257,128,128]));let la=u.get("backbone6.1.weight"),ca=u.get("backbone6.1.bias");if(!la||!ca)throw new Error("Missing backbone6.1 conv1x1 weights");let wa=w(la),ma=w(ca),ha=p(wa.byteLength,m),ba=p(ma.byteLength,m),fa=p(20,C);t.queue.writeBuffer(ha,0,wa),t.queue.writeBuffer(ba,0,ma),t.queue.writeBuffer(fa,0,new Uint32Array([1,96,48,32,32]));let ga=u.get("handflag.weight"),ya=u.get("handflag.bias");if(!ga||!ya)throw new Error("Missing handflag weights");let xa=w(ga),va=w(ya),Be=p(xa.byteLength,m),Ue=p(va.byteLength,m),Pa=p(12,C);t.queue.writeBuffer(Be,0,xa),t.queue.writeBuffer(Ue,0,va),t.queue.writeBuffer(Pa,0,new Uint32Array([1,288,1]));let ka=u.get("handedness.weight"),Ba=u.get("handedness.bias");if(!ka||!Ba)throw new Error("Missing handedness weights");let Ua=w(ka),Sa=w(Ba),Se=p(Ua.byteLength,m),De=p(Sa.byteLength,m),Da=p(12,C);t.queue.writeBuffer(Se,0,Ua),t.queue.writeBuffer(De,0,Sa),t.queue.writeBuffer(Da,0,new Uint32Array([1,288,1]));let Ga=u.get("reg_3d.weight"),Wa=u.get("reg_3d.bias");if(!Ga||!Wa)throw new Error("Missing reg_3d weights");let Aa=w(Ga),Ca=w(Wa),Ge=p(Aa.byteLength,m),We=p(Ca.byteLength,m),Ha=p(12,C);t.queue.writeBuffer(Ge,0,Aa),t.queue.writeBuffer(We,0,Ca),t.queue.writeBuffer(Ha,0,new Uint32Array([1,288,63]));let te=ft.map(a=>{let{inCh:r,outCh:i,h:n,w:e,stride:c,prefix:l}=a,f=c===2?n/2:n,_=c===2?e/2:e,o=c===1?2:1,s=u.get(`${l}convs.0.weight`),H=u.get(`${l}convs.0.bias`),U=u.get(`${l}convs.1.weight`),K=u.get(`${l}convs.1.bias`);if(!s||!H||!U||!K)throw new Error(`Missing weights for ${l}`);let Fa=w(s),Ia=w(H),za=w(U),qa=w(K),Na=p(Fa.byteLength,m),$a=p(Ia.byteLength,m),Xa=p(za.byteLength,m),Ya=p(qa.byteLength,m),Va=p(32,C),Za=p(36,C);return t.queue.writeBuffer(Na,0,Fa),t.queue.writeBuffer($a,0,Ia),t.queue.writeBuffer(Xa,0,za),t.queue.writeBuffer(Ya,0,qa),t.queue.writeBuffer(Va,0,new Uint32Array([1,r,n,e,f,_,c,o])),t.queue.writeBuffer(Za,0,new Uint32Array([1,r,i,f,_,Math.max(0,i-r),c,n,e])),{dwWeight:Na,dwBias:$a,pwWeight:Xa,pwBias:Ya,dwUniform:Va,pwUniform:Za,spec:a,outH:f,outW:_}});function Q(a){let r=p(a.length*4,C);return t.queue.writeBuffer(r,0,new Uint32Array(a)),r}let Nt=Q([1,96,8,8,16,16]),$t=Q([1,96,16,16,32,32]),Xt=Q([1,48,32,32,64,64]);Q([1536*16]),Q([3072*32]),Q([3072*64]);let Ea=v(ze,[Je,ke,Qe]),Yt=v(qe,[ke,pa,da,B,_a]),O=[],F=[],I=[],z=[];for(let a of te)O.push(v(ge,[B,a.dwWeight,a.dwBias,pe,a.dwUniform])),F.push(v(ye,[pe,B,a.pwWeight,a.pwBias,E,a.pwUniform])),I.push(v(ge,[E,a.dwWeight,a.dwBias,pe,a.dwUniform])),z.push(v(ye,[pe,E,a.pwWeight,a.pwBias,B,a.pwUniform]));let Vt=v(re,[B,ta,E,Nt]),Zt=v(re,[B,aa,E,$t]),Kt=v(Ne,[B,ha,ba,ia,fa]),jt=v(re,[ia,ea,E,Xt]);v(ae,[B,Be,Ue,J,Pa]),v(ae,[B,Se,De,J,Da]),v(ae,[B,Ge,We,J,Ha]);let Z=v($e,[R.createView(),ke,na]),Jt=v(Xe,[B,Be,Ue,Se,De,Ge,We,J]),Ae=24,Ma=[],Ta=[];for(let a=Ae;a<te.length;a++){let r=te[a];Ma.push(v(xe,[B,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,E,r.dwUniform])),Ta.push(v(xe,[E,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,B,r.dwUniform]))}let Ce=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Ce.globalCompositeOperation="copy";let La=new OffscreenCanvas(9,8),de=La.getContext("webgpu"),_e=null,He=null;if(de){de.configure({device:t,format:"rgba8unorm",alphaMode:"premultiplied"});let a=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),r=t.createShaderModule({code:mt}),i=t.createShaderModule({code:ht});_e=t.createRenderPipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:r,entryPoint:"vs"},fragment:{module:i,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),He=t.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:J}}]})}let le=new Float32Array(1),ce=new Float32Array(1),we=new Float32Array(63);function L(a,r){let i=!0,n=0,e=a.beginComputePass();for(e.setPipeline(Ft),e.setBindGroup(0,Yt),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);n<=ui;n++){let _=i?O[n]:I[n],o=i?F[n]:z[n],s=Y[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.end();let c=i?B:E;for(a.copyBufferToBuffer(c,0,ea,0,3072*64*4),e=a.beginComputePass();n<=oi;n++){let _=i?O[n]:I[n],o=i?F[n]:z[n],s=Y[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.end();let l=i?B:E;for(a.copyBufferToBuffer(l,0,aa,0,3072*32*4),e=a.beginComputePass();n<=pi;n++){let _=i?O[n]:I[n],o=i?F[n]:z[n],s=Y[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.end();let f=i?B:E;for(a.copyBufferToBuffer(f,0,ta,0,1536*16*4),e=a.beginComputePass();n<=di;n++){let _=i?O[n]:I[n],o=i?F[n]:z[n],s=Y[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.setPipeline(ve),e.setBindGroup(0,Vt),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),i=!1,e=a.beginComputePass();{let _=i?O[n]:I[n],o=i?F[n]:z[n],s=Y[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i,n++}e.setPipeline(ve),e.setBindGroup(0,Zt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),i=!1,e=a.beginComputePass();{let _=i?O[n]:I[n],o=i?F[n]:z[n],s=Y[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i,n++}for(e.setPipeline(It),e.setBindGroup(0,Kt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(ve),e.setBindGroup(0,jt),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),i=!1,e=a.beginComputePass();n<Ae;n++){let _=i?O[n]:I[n],o=i?F[n]:z[n],s=Y[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}for(;n<te.length;n++){let _=n-Ae,o=i?Ma[_]:Ta[_],s=te[n];e.setPipeline(qt),e.setBindGroup(0,o),e.dispatchWorkgroups(s.outW,s.outH,1),i=!i}e.setPipeline(zt),e.setBindGroup(0,Jt),e.dispatchWorkgroups(1),e.end(),r&&a.copyBufferToBuffer(J,0,r,0,260)}async function me(a){t.queue.writeBuffer(Je,0,a);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(je),e.setBindGroup(0,Ea),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}L(r,y),t.queue.submit([r.finish()]);let i=y.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await i;let n=new Float32Array(y.getMappedRange());return le[0]=n[0],ce[0]=n[1],we.set(n.subarray(2,65)),y.unmap(),{handflag:new Float32Array(le),handedness:new Float32Array(ce),landmarks:new Float32Array(we)}}async function Ee(a){t.queue.copyExternalImageToTexture({source:a},{texture:R},[256,256]);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(V),e.setBindGroup(0,Z),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}L(r,y),t.queue.submit([r.finish()]);let i=y.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await i;let n=new Float32Array(y.getMappedRange());return le[0]=n[0],ce[0]=n[1],we.set(n.subarray(2,65)),y.unmap(),{handflag:new Float32Array(le),handedness:new Float32Array(ce),landmarks:new Float32Array(we)}}async function Ra(a){if(!_e||!He||!de)throw new Error("Render-based readback not available (no WebGPU canvas context)");t.queue.copyExternalImageToTexture({source:a},{texture:R},[256,256]);let r=t.createCommandEncoder();{let _=r.beginComputePass();_.setPipeline(V),_.setBindGroup(0,Z),_.dispatchWorkgroups(16,16,1),_.end()}L(r,null);let i=de.getCurrentTexture(),n=r.beginRenderPass({colorAttachments:[{view:i.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});n.setPipeline(_e),n.setBindGroup(0,He),n.draw(3),n.end(),t.queue.submit([r.finish()]),await t.queue.onSubmittedWorkDone(),Ce.drawImage(La,0,0);let c=Ce.getImageData(0,0,9,8).data,l=new Float32Array(65),f=new DataView(new ArrayBuffer(4));for(let _=0;_<65;_++){let o=_*4;f.setUint8(0,c[o]),f.setUint8(1,c[o+1]),f.setUint8(2,c[o+2]),f.setUint8(3,c[o+3]),l[_]=f.getFloat32(0)}return{handflag:new Float32Array([l[0]]),handedness:new Float32Array([l[1]]),landmarks:new Float32Array(l.subarray(2,65))}}let Qt=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Me=0,ei=[y,Qt],ee=null,q=null;async function Te(a){let r=ei[Me];Me=1-Me,t.queue.copyExternalImageToTexture({source:a},{texture:R},[256,256]);let i=t.createCommandEncoder();{let e=i.beginComputePass();e.setPipeline(V),e.setBindGroup(0,Z),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}L(i,r),t.queue.submit([i.finish()]);let n=null;if(ee!==null&&q!==null){await ee;let e=new Float32Array(q.getMappedRange());n={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},q.unmap()}return q=r,ee=r.mapAsync(GPUMapMode.READ),n}async function Oa(){if(!ee||!q)return null;await ee;let a=new Float32Array(q.getMappedRange()),r={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return q.unmap(),ee=null,q=null,r}async function ai(a=50){let r=new Float32Array(196608);for(let e=0;e<5;e++)await me(r);let i=[];for(let e=0;e<a;e++){let c=performance.now();await me(r),i.push(performance.now()-c)}let n=i.reduce((e,c)=>e+c,0)/i.length;return{avgMs:n,fps:1e3/n}}async function ti(a=50){let r=new Float32Array(196608);for(let l=0;l<5;l++)await me(r);let i=[];for(let l=0;l<a;l++){let f=t.createCommandEncoder();{let o=f.beginComputePass();o.setPipeline(je),o.setBindGroup(0,Ea),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),o.end()}L(f,y);let _=performance.now();t.queue.submit([f.finish()]),await t.queue.onSubmittedWorkDone(),i.push(performance.now()-_)}i.sort((l,f)=>l-f);let n=i.reduce((l,f)=>l+f,0)/i.length,e=i[Math.floor(i.length/2)],c=i[0];return{avgMs:n,fps:1e3/n,medianMs:e,minMs:c}}function ii(a){t.queue.copyExternalImageToTexture({source:a},{texture:R},[256,256]);let r=t.createCommandEncoder();{let i=r.beginComputePass();i.setPipeline(V),i.setBindGroup(0,Z),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}L(r,y),t.queue.submit([r.finish()])}async function ni(a,r=50){function i(o){let s=[...o].sort((H,U)=>H-U);return{median:s[Math.floor(s.length/2)],min:s[0]}}for(let o=0;o<10;o++)await Ee(a);let n=[];for(let o=0;o<r;o++){t.queue.copyExternalImageToTexture({source:a},{texture:R},[256,256]);let s=t.createCommandEncoder();{let U=s.beginComputePass();U.setPipeline(V),U.setBindGroup(0,Z),U.dispatchWorkgroups(16,16,1),U.end()}L(s,y);let H=performance.now();t.queue.submit([s.finish()]),await t.queue.onSubmittedWorkDone(),n.push(performance.now()-H)}let e=[];for(let o=0;o<r;o++){t.queue.copyExternalImageToTexture({source:a},{texture:R},[256,256]);let s=t.createCommandEncoder();{let K=s.beginComputePass();K.setPipeline(V),K.setBindGroup(0,Z),K.dispatchWorkgroups(16,16,1),K.end()}L(s,y),t.queue.submit([s.finish()]);let H=y.mapAsync(GPUMapMode.READ),U=performance.now();await t.queue.onSubmittedWorkDone(),await H,y.getMappedRange(),y.unmap(),e.push(performance.now()-U)}let c=[];for(let o=0;o<r;o++){t.queue.copyExternalImageToTexture({source:a},{texture:R},[256,256]);let s=t.createCommandEncoder();{let U=s.beginComputePass();U.setPipeline(V),U.setBindGroup(0,Z),U.dispatchWorkgroups(16,16,1),U.end()}L(s,y),t.queue.submit([s.finish()]);let H=performance.now();await y.mapAsync(GPUMapMode.READ),y.getMappedRange(),y.unmap(),c.push(performance.now()-H)}let l=[];for(let o=0;o<r;o++){let s=performance.now();await Ee(a),l.push(performance.now()-s)}await Te(a);let f=[];for(let o=0;o<r;o++){let s=performance.now();await Te(a),f.push(performance.now()-s)}await Oa();let _=null;if(_e){let o=[];for(let s=0;s<r;s++){let H=performance.now();await Ra(a),o.push(performance.now()-H)}_=i(o)}return{gpuOnly:i(n),mapAsyncOnly:i(e),mapAsyncNoWait:i(c),total:i(l),pipelined:i(f),renderReadback:_}}return{device:t,run:me,runFromCanvas:Ee,runFromCanvasViaRender:Ra,runFromCanvasPipelined:Te,flushPipelined:Oa,benchmark:ai,benchmarkGPU:ti,benchmarkDiagnostic:ni,_device:t,_benchmarkSubmitOnly:ii}}var _i="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function li(u={}){let{weightsUrl:d,scoreThreshold:b=.5}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let g=d??_i,t=g.endsWith("/")?g:`${g}/`,M=`${t}weights_f16.json`,A=`${t}weights_f16.bin`,[w,T]=await Promise.all([fetch(M),fetch(A)]);if(!w.ok)throw new Error(`Failed to fetch weights metadata: ${w.status}`);if(!T.ok)throw new Error(`Failed to fetch weights binary: ${T.status}`);let x=await w.json(),N=await T.arrayBuffer(),P=gt(x,N),D=await yt(P),m=null;function he(){return m||(m=new OffscreenCanvas(256,256)),m}async function $(h){if(h instanceof HTMLCanvasElement||h instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&h instanceof ImageBitmap)return h;let G=he();G.width=256,G.height=256;let k=G.getContext("2d");return h instanceof ImageData?k.putImageData(h,0,0):k.drawImage(h,0,0,256,256),G}function j(h,G,k){let ie=h[0];if(ie<b)return null;let fe=G[0]>.5,ne=[];for(let X=0;X<21;X++)ne.push({x:k[X*3],y:k[X*3+1],z:k[X*3+2]});return{score:ie,handedness:fe?"right":"left",landmarks:ne}}async function C(h){let G=await $(h),k=await D.runFromCanvas(G);return j(k.handflag,k.handedness,k.landmarks)}async function p(h){let G=await $(h),k=await D.runFromCanvasPipelined(G);return k?j(k.handflag,k.handedness,k.landmarks):null}async function v(){let h=await D.flushPipelined();return h?j(h.handflag,h.handedness,h.landmarks):null}function W(){D.device.destroy(),m=null}async function be(h){let G=await $(h);return D.benchmarkDiagnostic(G)}return{detect:C,detectPipelined:p,flushPipelined:v,dispose:W,benchmarkDiagnostic:be}}export{li as createHandpose};
