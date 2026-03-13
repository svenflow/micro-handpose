function D(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ka(u){let p=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],f="enable f16;"+u;for(let h of p)for(;f.includes(`${h}:array<f32>`);)f=f.replace(`${h}:array<f32>`,`${h}:array<f16>`);return f}var Le=D(`
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
`),Re=D(`
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
`),Oe=D(`
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
`),Fe=D(`
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
`);function ja(u,p){return Re.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function Ja(u,p){return Le.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function Qa(u,p){return Oe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function et(u,p){return Fe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function at(u,p){return[8,8]}var tt=D(`
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
`),it=D(`
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
`);function nt(u){return D(`
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
`)}var rt=nt(!1),st=nt(!0),ut=D(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),ot=D(`
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
`);function pt(u){return D(`
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
`)}var dt=pt("sigmoid"),_t=pt("div256"),ct=D(`
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
`),lt=D(`
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
`);function wt(u,p){let h=Math.min(p,256),t=p>h,M=u%4===0?`var ic:u32=0u;
    while(ic<${u}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${u}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,k=`var skip_val:f32=0.0;
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
    }`,m=u===p?"":`if(c<${u}u){`,L=u===p?"":"}",v=t?`for(var c:u32=lid.x;c<${u}u;c+=${h}u){`:`let c=lid.x;
  ${m}`,T=t?"}":L,l=t?`for(var c:u32=lid.x;c<${p}u;c+=${h}u){`:"{let c=lid.x;";return D(`
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
@compute @workgroup_size(${h},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${v}
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
  ${T}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${l}
    let pw_base=c*${u}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${M}
    // Skip connection (only for c < inCh)
    ${k}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var mt=D(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),ht=D(`
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
`),bt=D(`
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
`);function gt(u,p){let f=new Map,h=u.dtype??"float32";for(let t=0;t<u.keys.length;t++){let H=u.keys[t],M=u.shapes[t],k=u.offsets[t],m=M.reduce((T,l)=>T*l,1),L,v;if(h==="float32")L=new Float32Array(p,k,m);else{let T=new DataView(p);L=new Float32Array(m);for(let l=0;l<m;l++)L[l]=si(T.getUint16(k+l*2,!0));v=p.slice(k,k+m*2)}f.set(H,{data:L,shape:M,rawF16:v})}return f}function si(u){let p=u>>15&1,f=u>>10&31,h=u&1023;if(f===0){if(h===0)return p?-0:0;let M=-14,k=h/1024;return(p?-1:1)*Math.pow(2,M)*k}if(f===31)return h===0?p?-1/0:1/0:NaN;let t=f-15,H=1+h/1024;return(p?-1:1)*Math.pow(2,t)*H}var ui=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],ft=ui.map(([u,p,f,h,t])=>({type:"resmodule",inCh:u,outCh:p,h:f,w:f,stride:h,prefix:t})),oi=2,pi=5,di=8,_i=11;async function yt(u){if(!navigator.gpu)throw new Error("WebGPU not supported");let p=await navigator.gpu.requestAdapter();if(!p)throw new Error("No GPU adapter found");let f=p.features.has("shader-f16"),h=f?["shader-f16"]:[],t=await p.requestDevice({requiredFeatures:h,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(p.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(p.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(p.limits.maxComputeInvocationsPerWorkgroup,288)}}),H=!1;if(f)try{H=!(await t.createShaderModule({code:`enable f16;
@compute @workgroup_size(1)
fn main() { var x: f16 = f16(1.0); _ = x; }`}).getCompilationInfo()).messages.some(i=>i.type==="error")}catch{H=!1}let M=u.values().next().value,k=H&&!!M?.rawF16;console.log(k?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${f}, f16 validated: ${H}, f16 data: ${!!M?.rawF16})`);function m(a){if(k&&a.rawF16){let r=new Uint16Array(a.rawF16);if(r.length%2!==0){let i=new Uint16Array(r.length+1);return i.set(r),i}return r}return a.data}function L(a){if(k&&a.rawF16){let r=a.rawF16.byteLength;return Math.ceil(r/4)*4}return a.data.byteLength}function v(a){return k?Ka(a):a}let T={r:"read-only-storage",s:"storage",u:"uniform"};function l(a){return t.createBindGroupLayout({entries:a.map((r,i)=>({binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[r]}}))})}function $(a){return t.createBindGroupLayout({entries:a.map((r,i)=>r==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[r]}})})}let y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,X=GPUBufferUsage.STORAGE,ne=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(a,r){return t.createBuffer({size:a,usage:r})}function P(a,r){return t.createBindGroup({layout:a,entries:r.map((i,n)=>({binding:n,resource:"size"in i?{buffer:i}:i}))})}function W(a,r){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:r,entryPoint:"main"}})}let b=t.createShaderModule({code:tt}),G=t.createShaderModule({code:bt}),B=t.createShaderModule({code:v(ct)}),re=t.createShaderModule({code:v(Re)}),fe=t.createShaderModule({code:v(Le)}),se=t.createShaderModule({code:v(Oe)}),Y=t.createShaderModule({code:v(Fe)}),xt=t.createShaderModule({code:v(it)}),vt=t.createShaderModule({code:rt}),Pt=t.createShaderModule({code:ut}),kt=t.createShaderModule({code:st}),Bt=t.createShaderModule({code:v(ot)}),Ut=t.createShaderModule({code:v(dt)}),St=t.createShaderModule({code:v(_t)}),Dt=t.createShaderModule({code:v(lt)}),Ie=new Map;function wi(a,r){let i=`${a}_${r}`,n=Ie.get(i);return n||(n=t.createShaderModule({code:v(wt(a,r))}),Ie.set(i,n)),n}let ge=l(["r","r","r","s","u"]),ye=l(["r","r","r","r","s","u"]),ze=l(["r","s","u"]),qe=l(["r","r","r","s","u"]),Gt=l(["r","s","u"]),Wt=l(["r","r","s","u"]),ue=l(["r","r","s","u"]),Ne=l(["r","r","r","s","u"]),te=l(["r","r","r","s","u"]),$e=$(["t","s","u"]),Xe=l(["r","r","r","r","r","r","r","s"]),xe=l(["r","r","r","r","r","s","u"]),At=t.createPipelineLayout({bindGroupLayouts:[ge]}),Ct=t.createPipelineLayout({bindGroupLayouts:[ye]}),oe=a=>t.createComputePipeline({layout:At,compute:{module:a,entryPoint:"main"}}),pe=a=>t.createComputePipeline({layout:Ct,compute:{module:a,entryPoint:"main"}}),Ht=oe(re),Mt=oe(fe),Et=pe(se),Tt=pe(Y),Ye=new Map,Ve=new Map,Ze=new Map,Ke=new Map;Ye.set("8,8",Ht),Ve.set("8,8",Mt),Ze.set("8,8",Et),Ke.set("8,8",Tt);function de(a,r,i,n,e){let w=`${r},${i}`,c=a.get(w);return c||(c=e(t.createShaderModule({code:v(n(r,i))})),a.set(w,c)),c}let Lt=(a,r)=>de(Ye,a,r,ja,oe),Rt=(a,r)=>de(Ve,a,r,Ja,oe),Ot=(a,r)=>de(Ze,a,r,Qa,pe),Ft=(a,r)=>de(Ke,a,r,et,pe),V=ft.map(a=>{let r=a.stride===2?a.h/2:a.h,i=a.stride===2?a.w/2:a.w,[n,e]=at(a.inCh,r),w=a.h>=64,c=r>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:w?Rt(n,e):Lt(n,e),pwPipeline:c?Ft(n,e):Ot(n,e),dwDispatchX:Math.ceil(i/n),dwDispatchY:Math.ceil(r/e),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(i/n),pwDispatchY:Math.ceil(r/e),pwDispatchZ:c?a.outCh/2:a.outCh}}),je=W(ze,b),It=W(qe,xt);W(Gt,vt),W(Wt,Pt);let ve=W(ue,kt),zt=W(Ne,Bt);W(te,Ut),W(te,St);let Z=W($e,G),qt=W(Xe,B),Nt=W(xe,Dt),Pe=1*288*128*128*4,Je=d(3*256*256*4,y),ke=d(3*257*257*4,X),Qe=d(12,A);t.queue.writeBuffer(Qe,0,new Uint32Array([3,256,257]));let U=d(Pe,ae),E=d(Pe,ne),_e=d(Pe,X),ea=d(3072*64*4,y),aa=d(3072*32*4,y),ta=d(1536*16*4,y),ia=d(6144*64*4,X),J=d(260,ne),x=d(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);d(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let O=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),na=d(8,A);t.queue.writeBuffer(na,0,new Uint32Array([256,257]));let ra=u.get("backbone1.1.weight"),sa=u.get("backbone1.1.bias");if(!ra||!sa)throw new Error("Missing input conv weights");let ua=m(ra),oa=m(sa),pa=d(ua.byteLength,y),da=d(oa.byteLength,y),_a=d(28,A);t.queue.writeBuffer(pa,0,ua),t.queue.writeBuffer(da,0,oa),t.queue.writeBuffer(_a,0,new Uint32Array([1,3,24,257,257,128,128]));let ca=u.get("backbone6.1.weight"),la=u.get("backbone6.1.bias");if(!ca||!la)throw new Error("Missing backbone6.1 conv1x1 weights");let wa=m(ca),ma=m(la),ha=d(wa.byteLength,y),ba=d(ma.byteLength,y),fa=d(20,A);t.queue.writeBuffer(ha,0,wa),t.queue.writeBuffer(ba,0,ma),t.queue.writeBuffer(fa,0,new Uint32Array([1,96,48,32,32]));let ga=u.get("handflag.weight"),ya=u.get("handflag.bias");if(!ga||!ya)throw new Error("Missing handflag weights");let xa=m(ga),va=m(ya),Be=d(xa.byteLength,y),Ue=d(va.byteLength,y),Pa=d(12,A);t.queue.writeBuffer(Be,0,xa),t.queue.writeBuffer(Ue,0,va),t.queue.writeBuffer(Pa,0,new Uint32Array([1,288,1]));let ka=u.get("handedness.weight"),Ba=u.get("handedness.bias");if(!ka||!Ba)throw new Error("Missing handedness weights");let Ua=m(ka),Sa=m(Ba),Se=d(Ua.byteLength,y),De=d(Sa.byteLength,y),Da=d(12,A);t.queue.writeBuffer(Se,0,Ua),t.queue.writeBuffer(De,0,Sa),t.queue.writeBuffer(Da,0,new Uint32Array([1,288,1]));let Ga=u.get("reg_3d.weight"),Wa=u.get("reg_3d.bias");if(!Ga||!Wa)throw new Error("Missing reg_3d weights");let Aa=m(Ga),Ca=m(Wa),Ge=d(Aa.byteLength,y),We=d(Ca.byteLength,y),Ha=d(12,A);t.queue.writeBuffer(Ge,0,Aa),t.queue.writeBuffer(We,0,Ca),t.queue.writeBuffer(Ha,0,new Uint32Array([1,288,63]));let ie=ft.map(a=>{let{inCh:r,outCh:i,h:n,w:e,stride:w,prefix:c}=a,g=w===2?n/2:n,_=w===2?e/2:e,o=w===1?2:1,s=u.get(`${c}convs.0.weight`),C=u.get(`${c}convs.0.bias`),S=u.get(`${c}convs.1.weight`),j=u.get(`${c}convs.1.bias`);if(!s||!C||!S||!j)throw new Error(`Missing weights for ${c}`);let Fa=m(s),Ia=m(C),za=m(S),qa=m(j),Na=d(Fa.byteLength,y),$a=d(Ia.byteLength,y),Xa=d(za.byteLength,y),Ya=d(qa.byteLength,y),Va=d(32,A),Za=d(36,A);return t.queue.writeBuffer(Na,0,Fa),t.queue.writeBuffer($a,0,Ia),t.queue.writeBuffer(Xa,0,za),t.queue.writeBuffer(Ya,0,qa),t.queue.writeBuffer(Va,0,new Uint32Array([1,r,n,e,g,_,w,o])),t.queue.writeBuffer(Za,0,new Uint32Array([1,r,i,g,_,Math.max(0,i-r),w,n,e])),{dwWeight:Na,dwBias:$a,pwWeight:Xa,pwBias:Ya,dwUniform:Va,pwUniform:Za,spec:a,outH:g,outW:_}});function Q(a){let r=d(a.length*4,A);return t.queue.writeBuffer(r,0,new Uint32Array(a)),r}let $t=Q([1,96,8,8,16,16]),Xt=Q([1,96,16,16,32,32]),Yt=Q([1,48,32,32,64,64]);Q([1536*16]),Q([3072*32]),Q([3072*64]);let Ma=P(ze,[Je,ke,Qe]),Vt=P(qe,[ke,pa,da,U,_a]),F=[],I=[],z=[],q=[];for(let a of ie)F.push(P(ge,[U,a.dwWeight,a.dwBias,_e,a.dwUniform])),I.push(P(ye,[_e,U,a.pwWeight,a.pwBias,E,a.pwUniform])),z.push(P(ge,[E,a.dwWeight,a.dwBias,_e,a.dwUniform])),q.push(P(ye,[_e,E,a.pwWeight,a.pwBias,U,a.pwUniform]));let Zt=P(ue,[U,ta,E,$t]),Kt=P(ue,[U,aa,E,Xt]),jt=P(Ne,[U,ha,ba,ia,fa]),Jt=P(ue,[ia,ea,E,Yt]);P(te,[U,Be,Ue,J,Pa]),P(te,[U,Se,De,J,Da]),P(te,[U,Ge,We,J,Ha]);let K=P($e,[O.createView(),ke,na]),Qt=P(Xe,[U,Be,Ue,Se,De,Ge,We,J]),Ae=24,Ea=[],Ta=[];for(let a=Ae;a<ie.length;a++){let r=ie[a];Ea.push(P(xe,[U,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,E,r.dwUniform])),Ta.push(P(xe,[E,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,U,r.dwUniform]))}let Ce=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Ce.globalCompositeOperation="copy";let La=new OffscreenCanvas(9,8),ce=La.getContext("webgpu"),le=null,He=null;if(ce){ce.configure({device:t,format:"rgba8unorm",alphaMode:"premultiplied"});let a=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),r=t.createShaderModule({code:mt}),i=t.createShaderModule({code:ht});le=t.createRenderPipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:r,entryPoint:"vs"},fragment:{module:i,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),He=t.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:J}}]})}let we=new Float32Array(1),me=new Float32Array(1),he=new Float32Array(63);function R(a,r){let i=!0,n=0,e=a.beginComputePass();for(e.setPipeline(It),e.setBindGroup(0,Vt),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);n<=oi;n++){let _=i?F[n]:z[n],o=i?I[n]:q[n],s=V[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.end();let w=i?U:E;for(a.copyBufferToBuffer(w,0,ea,0,3072*64*4),e=a.beginComputePass();n<=pi;n++){let _=i?F[n]:z[n],o=i?I[n]:q[n],s=V[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.end();let c=i?U:E;for(a.copyBufferToBuffer(c,0,aa,0,3072*32*4),e=a.beginComputePass();n<=di;n++){let _=i?F[n]:z[n],o=i?I[n]:q[n],s=V[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.end();let g=i?U:E;for(a.copyBufferToBuffer(g,0,ta,0,1536*16*4),e=a.beginComputePass();n<=_i;n++){let _=i?F[n]:z[n],o=i?I[n]:q[n],s=V[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}e.setPipeline(ve),e.setBindGroup(0,Zt),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),i=!1,e=a.beginComputePass();{let _=i?F[n]:z[n],o=i?I[n]:q[n],s=V[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i,n++}e.setPipeline(ve),e.setBindGroup(0,Kt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),i=!1,e=a.beginComputePass();{let _=i?F[n]:z[n],o=i?I[n]:q[n],s=V[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i,n++}for(e.setPipeline(zt),e.setBindGroup(0,jt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(ve),e.setBindGroup(0,Jt),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),i=!1,e=a.beginComputePass();n<Ae;n++){let _=i?F[n]:z[n],o=i?I[n]:q[n],s=V[n];e.setPipeline(s.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}for(;n<ie.length;n++){let _=n-Ae,o=i?Ea[_]:Ta[_],s=ie[n];e.setPipeline(Nt),e.setBindGroup(0,o),e.dispatchWorkgroups(s.outW,s.outH,1),i=!i}e.setPipeline(qt),e.setBindGroup(0,Qt),e.dispatchWorkgroups(1),e.end(),r&&a.copyBufferToBuffer(J,0,r,0,260)}async function be(a){t.queue.writeBuffer(Je,0,a);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(je),e.setBindGroup(0,Ma),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}R(r,x),t.queue.submit([r.finish()]);let i=x.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await i;let n=new Float32Array(x.getMappedRange());return we[0]=n[0],me[0]=n[1],he.set(n.subarray(2,65)),x.unmap(),{handflag:new Float32Array(we),handedness:new Float32Array(me),landmarks:new Float32Array(he)}}async function Me(a){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(Z),e.setBindGroup(0,K),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}R(r,x),t.queue.submit([r.finish()]);let i=x.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await i;let n=new Float32Array(x.getMappedRange());return we[0]=n[0],me[0]=n[1],he.set(n.subarray(2,65)),x.unmap(),{handflag:new Float32Array(we),handedness:new Float32Array(me),landmarks:new Float32Array(he)}}async function Ra(a){if(!le||!He||!ce)throw new Error("Render-based readback not available (no WebGPU canvas context)");t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let r=t.createCommandEncoder();{let _=r.beginComputePass();_.setPipeline(Z),_.setBindGroup(0,K),_.dispatchWorkgroups(16,16,1),_.end()}R(r,null);let i=ce.getCurrentTexture(),n=r.beginRenderPass({colorAttachments:[{view:i.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});n.setPipeline(le),n.setBindGroup(0,He),n.draw(3),n.end(),t.queue.submit([r.finish()]),await t.queue.onSubmittedWorkDone(),Ce.drawImage(La,0,0);let w=Ce.getImageData(0,0,9,8).data,c=new Float32Array(65),g=new DataView(new ArrayBuffer(4));for(let _=0;_<65;_++){let o=_*4;g.setUint8(0,w[o]),g.setUint8(1,w[o+1]),g.setUint8(2,w[o+2]),g.setUint8(3,w[o+3]),c[_]=g.getFloat32(0)}return{handflag:new Float32Array([c[0]]),handedness:new Float32Array([c[1]]),landmarks:new Float32Array(c.subarray(2,65))}}let ei=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Ee=0,ai=[x,ei],ee=null,N=null;async function Te(a){let r=ai[Ee];Ee=1-Ee,t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let i=t.createCommandEncoder();{let e=i.beginComputePass();e.setPipeline(Z),e.setBindGroup(0,K),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}R(i,r),t.queue.submit([i.finish()]);let n=null;if(ee!==null&&N!==null){await ee;let e=new Float32Array(N.getMappedRange());n={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},N.unmap()}return N=r,ee=r.mapAsync(GPUMapMode.READ),n}async function Oa(){if(!ee||!N)return null;await ee;let a=new Float32Array(N.getMappedRange()),r={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return N.unmap(),ee=null,N=null,r}async function ti(a=50){let r=new Float32Array(196608);for(let e=0;e<5;e++)await be(r);let i=[];for(let e=0;e<a;e++){let w=performance.now();await be(r),i.push(performance.now()-w)}let n=i.reduce((e,w)=>e+w,0)/i.length;return{avgMs:n,fps:1e3/n}}async function ii(a=50){let r=new Float32Array(196608);for(let c=0;c<5;c++)await be(r);let i=[];for(let c=0;c<a;c++){let g=t.createCommandEncoder();{let o=g.beginComputePass();o.setPipeline(je),o.setBindGroup(0,Ma),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),o.end()}R(g,x);let _=performance.now();t.queue.submit([g.finish()]),await t.queue.onSubmittedWorkDone(),i.push(performance.now()-_)}i.sort((c,g)=>c-g);let n=i.reduce((c,g)=>c+g,0)/i.length,e=i[Math.floor(i.length/2)],w=i[0];return{avgMs:n,fps:1e3/n,medianMs:e,minMs:w}}function ni(a){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let r=t.createCommandEncoder();{let i=r.beginComputePass();i.setPipeline(Z),i.setBindGroup(0,K),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}R(r,x),t.queue.submit([r.finish()])}async function ri(a,r=50){function i(o){let s=[...o].sort((C,S)=>C-S);return{median:s[Math.floor(s.length/2)],min:s[0]}}for(let o=0;o<10;o++)await Me(a);let n=[];for(let o=0;o<r;o++){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let s=t.createCommandEncoder();{let S=s.beginComputePass();S.setPipeline(Z),S.setBindGroup(0,K),S.dispatchWorkgroups(16,16,1),S.end()}R(s,x);let C=performance.now();t.queue.submit([s.finish()]),await t.queue.onSubmittedWorkDone(),n.push(performance.now()-C)}let e=[];for(let o=0;o<r;o++){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let s=t.createCommandEncoder();{let j=s.beginComputePass();j.setPipeline(Z),j.setBindGroup(0,K),j.dispatchWorkgroups(16,16,1),j.end()}R(s,x),t.queue.submit([s.finish()]);let C=x.mapAsync(GPUMapMode.READ),S=performance.now();await t.queue.onSubmittedWorkDone(),await C,x.getMappedRange(),x.unmap(),e.push(performance.now()-S)}let w=[];for(let o=0;o<r;o++){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let s=t.createCommandEncoder();{let S=s.beginComputePass();S.setPipeline(Z),S.setBindGroup(0,K),S.dispatchWorkgroups(16,16,1),S.end()}R(s,x),t.queue.submit([s.finish()]);let C=performance.now();await x.mapAsync(GPUMapMode.READ),x.getMappedRange(),x.unmap(),w.push(performance.now()-C)}let c=[];for(let o=0;o<r;o++){let s=performance.now();await Me(a),c.push(performance.now()-s)}await Te(a);let g=[];for(let o=0;o<r;o++){let s=performance.now();await Te(a),g.push(performance.now()-s)}await Oa();let _=null;if(le){let o=[];for(let s=0;s<r;s++){let C=performance.now();await Ra(a),o.push(performance.now()-C)}_=i(o)}return{gpuOnly:i(n),mapAsyncOnly:i(e),mapAsyncNoWait:i(w),total:i(c),pipelined:i(g),renderReadback:_}}return{device:t,run:be,runFromCanvas:Me,runFromCanvasViaRender:Ra,runFromCanvasPipelined:Te,flushPipelined:Oa,benchmark:ti,benchmarkGPU:ii,benchmarkDiagnostic:ri,_device:t,_benchmarkSubmitOnly:ni}}var ci="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function li(u={}){let{weightsUrl:p,scoreThreshold:f=.5}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let h=p??ci,t=h.endsWith("/")?h:`${h}/`,H=`${t}weights_f16.json`,M=`${t}weights_f16.bin`,[k,m]=await Promise.all([fetch(H),fetch(M)]);if(!k.ok)throw new Error(`Failed to fetch weights metadata: ${k.status}`);if(!m.ok)throw new Error(`Failed to fetch weights binary: ${m.status}`);let L=await k.json(),v=await m.arrayBuffer(),T=gt(L,v),l=await yt(T),$=null;function y(){return $||($=new OffscreenCanvas(256,256)),$}async function ae(b){if(b instanceof HTMLCanvasElement||b instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&b instanceof ImageBitmap)return b;let G=y();G.width=256,G.height=256;let B=G.getContext("2d");return b instanceof ImageData?B.putImageData(b,0,0):B.drawImage(b,0,0,256,256),G}function X(b,G,B){let re=b[0];if(re<f)return null;let fe=G[0]>.5,se=[];for(let Y=0;Y<21;Y++)se.push({x:B[Y*3],y:B[Y*3+1],z:B[Y*3+2]});return{score:re,handedness:fe?"right":"left",landmarks:se}}async function ne(b){let G=await ae(b),B=await l.runFromCanvas(G);return X(B.handflag,B.handedness,B.landmarks)}async function A(b){let G=await ae(b),B=await l.runFromCanvasPipelined(G);return B?X(B.handflag,B.handedness,B.landmarks):null}async function d(){let b=await l.flushPipelined();return b?X(b.handflag,b.handedness,b.landmarks):null}function P(){l.device.destroy(),$=null}async function W(b){let G=await ae(b);return l.benchmarkDiagnostic(G)}return{detect:ne,detectPipelined:A,flushPipelined:d,dispose:P,benchmarkDiagnostic:W}}export{li as createHandpose};
