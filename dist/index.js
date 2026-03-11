function B(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Xe=B(`
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
`),Ye=B(`
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
`),Ve=B(`
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
`),Ze=B(`
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
`);function Ma(u,d){return Ye.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function La(u,d){return Xe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Ra(u,d){return Ve.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Ta(u,d){return Ze.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Oa(u,d){return[8,8]}var Fa=B(`
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
`),Ia=B(`
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
`);function za(u){return B(`
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
`)}var qa=za(!1),Na=za(!0),$a=B(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Xa=B(`
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
`);function Ya(u){return B(`
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
`)}var Va=Ya("sigmoid"),Za=Ya("div256"),Ka=B(`
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
`),ja=B(`
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
`);function Ja(u,d){let g=u%4===0?`var ic:u32=0u;
  while(ic<${u}u){
    sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
    sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
    sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
    sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
    ic+=4u;
  }
  var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
  while(ic<${u}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
  var pw_sum=sum0+pw_bias[c];`,m=`var skip_val:f32=0.0;
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
  }`,C=u===d?"":`if(c<${u}u){`;return B(`
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
  ${C}
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
  ${m}
  let result=max(0.0,pw_sum+skip_val);
  output[c*outH*outW+out_y*outW+out_x]=result;
}
`)}var Qa=B(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),et=B(`
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
`),at=B(`
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
`);function it(u,d){let a=new Map,g=u.dtype??"float32";for(let m=0;m<u.keys.length;m++){let C=u.keys[m],c=u.shapes[m],A=u.offsets[m],W=c.reduce((P,p)=>P*p,1),H;if(g==="float32")H=new Float32Array(d,A,W);else{let P=new DataView(d);H=new Float32Array(W);for(let p=0;p<W;p++)H[p]=Rt(P.getUint16(A+p*2,!0))}a.set(C,{data:H,shape:c})}return a}function Rt(u){let d=u>>15&1,a=u>>10&31,g=u&1023;if(a===0){if(g===0)return d?-0:0;let c=-14,A=g/1024;return(d?-1:1)*Math.pow(2,c)*A}if(a===31)return g===0?d?-1/0:1/0:NaN;let m=a-15,C=1+g/1024;return(d?-1:1)*Math.pow(2,m)*C}var Tt=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],tt=Tt.map(([u,d,a,g,m])=>({type:"resmodule",inCh:u,outCh:d,h:a,w:a,stride:g,prefix:m})),Ot=2,Ft=5,It=8,zt=11;async function nt(u){if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");let a=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(d.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(d.limits.maxComputeInvocationsPerWorkgroup,288)}}),g={r:"read-only-storage",s:"storage",u:"uniform"};function m(t){return a.createBindGroupLayout({entries:t.map((s,n)=>({binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:g[s]}}))})}function C(t){return a.createBindGroupLayout({entries:t.map((s,n)=>s==="t"?{binding:n,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:g[s]}})})}let c=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,A=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,W=GPUBufferUsage.STORAGE,H=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,P=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function p(t,s){return a.createBuffer({size:t,usage:s})}function b(t,s){return a.createBindGroup({layout:t,entries:s.map((n,i)=>({binding:i,resource:"size"in n?{buffer:n}:n}))})}function S(t,s){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:s,entryPoint:"main"}})}let Z=a.createShaderModule({code:Fa}),K=a.createShaderModule({code:at}),le=a.createShaderModule({code:Ka}),ce=a.createShaderModule({code:Ye}),we=a.createShaderModule({code:Xe}),me=a.createShaderModule({code:Ve}),he=a.createShaderModule({code:Ze}),h=a.createShaderModule({code:Ia}),U=a.createShaderModule({code:qa}),x=a.createShaderModule({code:$a}),Q=a.createShaderModule({code:Na}),be=a.createShaderModule({code:Xa}),ee=a.createShaderModule({code:Va}),z=a.createShaderModule({code:Za}),rt=a.createShaderModule({code:ja}),Ke=new Map;function $t(t,s){let n=`${t}_${s}`,i=Ke.get(n);return i||(i=a.createShaderModule({code:Ja(t,s)}),Ke.set(n,i)),i}let fe=m(["r","r","r","s","u"]),ge=m(["r","r","r","r","s","u"]),je=m(["r","s","u"]),Je=m(["r","r","r","s","u"]),st=m(["r","s","u"]),ut=m(["r","r","s","u"]),ae=m(["r","r","s","u"]),Qe=m(["r","r","r","s","u"]),j=m(["r","r","r","s","u"]),ea=C(["t","s","u"]),aa=m(["r","r","r","r","r","r","r","s"]),ye=m(["r","r","r","r","r","s","u"]),ot=a.createPipelineLayout({bindGroupLayouts:[fe]}),pt=a.createPipelineLayout({bindGroupLayouts:[ge]}),te=t=>a.createComputePipeline({layout:ot,compute:{module:t,entryPoint:"main"}}),ie=t=>a.createComputePipeline({layout:pt,compute:{module:t,entryPoint:"main"}}),_t=te(ce),dt=te(we),lt=ie(me),ct=ie(he),ta=new Map,ia=new Map,na=new Map,ra=new Map;ta.set("8,8",_t),ia.set("8,8",dt),na.set("8,8",lt),ra.set("8,8",ct);function ne(t,s,n,i,e){let w=`${s},${n}`,l=t.get(w);return l||(l=e(a.createShaderModule({code:i(s,n)})),t.set(w,l)),l}let wt=(t,s)=>ne(ta,t,s,Ma,te),mt=(t,s)=>ne(ia,t,s,La,te),ht=(t,s)=>ne(na,t,s,Ra,ie),bt=(t,s)=>ne(ra,t,s,Ta,ie),q=tt.map(t=>{let s=t.stride===2?t.h/2:t.h,n=t.stride===2?t.w/2:t.w,[i,e]=Oa(t.inCh,s),w=t.h>=64,l=s>=16&&t.inCh>=288&&t.outCh>=288&&t.outCh%2===0;return{dwPipeline:w?mt(i,e):wt(i,e),pwPipeline:l?bt(i,e):ht(i,e),dwDispatchX:Math.ceil(n/i),dwDispatchY:Math.ceil(s/e),dwDispatchZ:t.inCh,pwDispatchX:Math.ceil(n/i),pwDispatchY:Math.ceil(s/e),pwDispatchZ:l?t.outCh/2:t.outCh}}),sa=S(je,Z),ft=S(Je,h);S(st,U),S(ut,x);let xe=S(ae,Q),gt=S(Qe,be);S(j,ee),S(j,z);let N=S(ea,K),yt=S(aa,le),xt=S(ye,rt),ve=1*288*128*128*4,ua=p(3*256*256*4,c),Pe=p(3*257*257*4,W),oa=p(12,P);a.queue.writeBuffer(oa,0,new Uint32Array([3,256,257]));let k=p(ve,A),G=p(ve,H),re=p(ve,W),pa=p(3072*64*4,c),_a=p(3072*32*4,c),da=p(1536*16*4,c),la=p(6144*64*4,W),X=p(260,H),y=p(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);p(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let M=a.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ca=p(8,P);a.queue.writeBuffer(ca,0,new Uint32Array([256,257]));let ke=u.get("backbone1.1.weight")?.data,Be=u.get("backbone1.1.bias")?.data;if(!ke||!Be)throw new Error("Missing input conv weights");let wa=p(ke.byteLength,c),ma=p(Be.byteLength,c),ha=p(28,P);a.queue.writeBuffer(wa,0,ke),a.queue.writeBuffer(ma,0,Be),a.queue.writeBuffer(ha,0,new Uint32Array([1,3,24,257,257,128,128]));let Ue=u.get("backbone6.1.weight")?.data,Se=u.get("backbone6.1.bias")?.data;if(!Ue||!Se)throw new Error("Missing backbone6.1 conv1x1 weights");let ba=p(Ue.byteLength,c),fa=p(Se.byteLength,c),ga=p(20,P);a.queue.writeBuffer(ba,0,Ue),a.queue.writeBuffer(fa,0,Se),a.queue.writeBuffer(ga,0,new Uint32Array([1,96,48,32,32]));let De=u.get("handflag.weight")?.data,Ge=u.get("handflag.bias")?.data;if(!De||!Ge)throw new Error("Missing handflag weights");let Ae=p(De.byteLength,c),Ce=p(Ge.byteLength,c),ya=p(12,P);a.queue.writeBuffer(Ae,0,De),a.queue.writeBuffer(Ce,0,Ge),a.queue.writeBuffer(ya,0,new Uint32Array([1,288,1]));let We=u.get("handedness.weight")?.data,He=u.get("handedness.bias")?.data;if(!We||!He)throw new Error("Missing handedness weights");let Ee=p(We.byteLength,c),Me=p(He.byteLength,c),xa=p(12,P);a.queue.writeBuffer(Ee,0,We),a.queue.writeBuffer(Me,0,He),a.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let Le=u.get("reg_3d.weight")?.data,Re=u.get("reg_3d.bias")?.data;if(!Le||!Re)throw new Error("Missing reg_3d weights");let Te=p(Le.byteLength,c),Oe=p(Re.byteLength,c),va=p(12,P);a.queue.writeBuffer(Te,0,Le),a.queue.writeBuffer(Oe,0,Re),a.queue.writeBuffer(va,0,new Uint32Array([1,288,63]));let J=tt.map(t=>{let{inCh:s,outCh:n,h:i,w:e,stride:w,prefix:l}=t,f=w===2?i/2:i,_=w===2?e/2:e,o=w===1?2:1,r=u.get(`${l}convs.0.weight`)?.data,D=u.get(`${l}convs.0.bias`)?.data,v=u.get(`${l}convs.1.weight`)?.data,I=u.get(`${l}convs.1.bias`)?.data;if(!r||!D||!v||!I)throw new Error(`Missing weights for ${l}`);let Ga=p(r.byteLength,c),Aa=p(D.byteLength,c),Ca=p(v.byteLength,c),Wa=p(I.byteLength,c),Ha=p(32,P),Ea=p(36,P);return a.queue.writeBuffer(Ga,0,r),a.queue.writeBuffer(Aa,0,D),a.queue.writeBuffer(Ca,0,v),a.queue.writeBuffer(Wa,0,I),a.queue.writeBuffer(Ha,0,new Uint32Array([1,s,i,e,f,_,w,o])),a.queue.writeBuffer(Ea,0,new Uint32Array([1,s,n,f,_,Math.max(0,n-s),w,i,e])),{dwWeight:Ga,dwBias:Aa,pwWeight:Ca,pwBias:Wa,dwUniform:Ha,pwUniform:Ea,spec:t,outH:f,outW:_}});function Y(t){let s=p(t.length*4,P);return a.queue.writeBuffer(s,0,new Uint32Array(t)),s}let vt=Y([1,96,8,8,16,16]),Pt=Y([1,96,16,16,32,32]),kt=Y([1,48,32,32,64,64]);Y([1536*16]),Y([3072*32]),Y([3072*64]);let Pa=b(je,[ua,Pe,oa]),Bt=b(Je,[Pe,wa,ma,k,ha]),L=[],R=[],T=[],O=[];for(let t of J)L.push(b(fe,[k,t.dwWeight,t.dwBias,re,t.dwUniform])),R.push(b(ge,[re,k,t.pwWeight,t.pwBias,G,t.pwUniform])),T.push(b(fe,[G,t.dwWeight,t.dwBias,re,t.dwUniform])),O.push(b(ge,[re,G,t.pwWeight,t.pwBias,k,t.pwUniform]));let Ut=b(ae,[k,da,G,vt]),St=b(ae,[k,_a,G,Pt]),Dt=b(Qe,[k,ba,fa,la,ga]),Gt=b(ae,[la,pa,G,kt]);b(j,[k,Ae,Ce,X,ya]),b(j,[k,Ee,Me,X,xa]),b(j,[k,Te,Oe,X,va]);let $=b(ea,[M.createView(),Pe,ca]),At=b(aa,[k,Ae,Ce,Ee,Me,Te,Oe,X]),Fe=24,ka=[],Ba=[];for(let t=Fe;t<J.length;t++){let s=J[t];ka.push(b(ye,[k,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,G,s.dwUniform])),Ba.push(b(ye,[G,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,k,s.dwUniform]))}let Ie=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Ie.globalCompositeOperation="copy";let Ua=new OffscreenCanvas(9,8),se=Ua.getContext("webgpu"),ue=null,ze=null;if(se){se.configure({device:a,format:"rgba8unorm",alphaMode:"premultiplied"});let t=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),s=a.createShaderModule({code:Qa}),n=a.createShaderModule({code:et});ue=a.createRenderPipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),vertex:{module:s,entryPoint:"vs"},fragment:{module:n,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),ze=a.createBindGroup({layout:t,entries:[{binding:0,resource:{buffer:X}}]})}let oe=new Float32Array(1),pe=new Float32Array(1),_e=new Float32Array(63);function E(t,s){let n=!0,i=0,e=t.beginComputePass();for(e.setPipeline(ft),e.setBindGroup(0,Bt),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=Ot;i++){let _=n?L[i]:T[i],o=n?R[i]:O[i],r=q[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let w=n?k:G;for(t.copyBufferToBuffer(w,0,pa,0,3072*64*4),e=t.beginComputePass();i<=Ft;i++){let _=n?L[i]:T[i],o=n?R[i]:O[i],r=q[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let l=n?k:G;for(t.copyBufferToBuffer(l,0,_a,0,3072*32*4),e=t.beginComputePass();i<=It;i++){let _=n?L[i]:T[i],o=n?R[i]:O[i],r=q[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.end();let f=n?k:G;for(t.copyBufferToBuffer(f,0,da,0,1536*16*4),e=t.beginComputePass();i<=zt;i++){let _=n?L[i]:T[i],o=n?R[i]:O[i],r=q[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}e.setPipeline(xe),e.setBindGroup(0,Ut),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),n=!1,e=t.beginComputePass();{let _=n?L[i]:T[i],o=n?R[i]:O[i],r=q[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,i++}e.setPipeline(xe),e.setBindGroup(0,St),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),n=!1,e=t.beginComputePass();{let _=n?L[i]:T[i],o=n?R[i]:O[i],r=q[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,i++}for(e.setPipeline(gt),e.setBindGroup(0,Dt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(xe),e.setBindGroup(0,Gt),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),n=!1,e=t.beginComputePass();i<Fe;i++){let _=n?L[i]:T[i],o=n?R[i]:O[i],r=q[i];e.setPipeline(r.dwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,o),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}for(;i<J.length;i++){let _=i-Fe,o=n?ka[_]:Ba[_],r=J[i];e.setPipeline(xt),e.setBindGroup(0,o),e.dispatchWorkgroups(r.outW,r.outH,1),n=!n}e.setPipeline(yt),e.setBindGroup(0,At),e.dispatchWorkgroups(1),e.end(),s&&t.copyBufferToBuffer(X,0,s,0,260)}async function de(t){a.queue.writeBuffer(ua,0,t);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(sa),e.setBindGroup(0,Pa),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}E(s,y),a.queue.submit([s.finish()]);let n=y.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(y.getMappedRange());return oe[0]=i[0],pe[0]=i[1],_e.set(i.subarray(2,65)),y.unmap(),{handflag:new Float32Array(oe),handedness:new Float32Array(pe),landmarks:new Float32Array(_e)}}async function qe(t){a.queue.copyExternalImageToTexture({source:t},{texture:M},[256,256]);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(N),e.setBindGroup(0,$),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}E(s,y),a.queue.submit([s.finish()]);let n=y.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(y.getMappedRange());return oe[0]=i[0],pe[0]=i[1],_e.set(i.subarray(2,65)),y.unmap(),{handflag:new Float32Array(oe),handedness:new Float32Array(pe),landmarks:new Float32Array(_e)}}async function Sa(t){if(!ue||!ze||!se)throw new Error("Render-based readback not available (no WebGPU canvas context)");a.queue.copyExternalImageToTexture({source:t},{texture:M},[256,256]);let s=a.createCommandEncoder();{let _=s.beginComputePass();_.setPipeline(N),_.setBindGroup(0,$),_.dispatchWorkgroups(16,16,1),_.end()}E(s,null);let n=se.getCurrentTexture(),i=s.beginRenderPass({colorAttachments:[{view:n.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});i.setPipeline(ue),i.setBindGroup(0,ze),i.draw(3),i.end(),a.queue.submit([s.finish()]),await a.queue.onSubmittedWorkDone(),Ie.drawImage(Ua,0,0);let w=Ie.getImageData(0,0,9,8).data,l=new Float32Array(65),f=new DataView(new ArrayBuffer(4));for(let _=0;_<65;_++){let o=_*4;f.setUint8(0,w[o]),f.setUint8(1,w[o+1]),f.setUint8(2,w[o+2]),f.setUint8(3,w[o+3]),l[_]=f.getFloat32(0)}return{handflag:new Float32Array([l[0]]),handedness:new Float32Array([l[1]]),landmarks:new Float32Array(l.subarray(2,65))}}let Ct=a.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Ne=0,Wt=[y,Ct],V=null,F=null;async function $e(t){let s=Wt[Ne];Ne=1-Ne,a.queue.copyExternalImageToTexture({source:t},{texture:M},[256,256]);let n=a.createCommandEncoder();{let e=n.beginComputePass();e.setPipeline(N),e.setBindGroup(0,$),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}E(n,s),a.queue.submit([n.finish()]);let i=null;if(V!==null&&F!==null){await V;let e=new Float32Array(F.getMappedRange());i={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},F.unmap()}return F=s,V=s.mapAsync(GPUMapMode.READ),i}async function Da(){if(!V||!F)return null;await V;let t=new Float32Array(F.getMappedRange()),s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))};return F.unmap(),V=null,F=null,s}async function Ht(t=50){let s=new Float32Array(196608);for(let e=0;e<5;e++)await de(s);let n=[];for(let e=0;e<t;e++){let w=performance.now();await de(s),n.push(performance.now()-w)}let i=n.reduce((e,w)=>e+w,0)/n.length;return{avgMs:i,fps:1e3/i}}async function Et(t=50){let s=new Float32Array(196608);for(let l=0;l<5;l++)await de(s);let n=[];for(let l=0;l<t;l++){let f=a.createCommandEncoder();{let o=f.beginComputePass();o.setPipeline(sa),o.setBindGroup(0,Pa),o.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),o.end()}E(f,y);let _=performance.now();a.queue.submit([f.finish()]),await a.queue.onSubmittedWorkDone(),n.push(performance.now()-_)}n.sort((l,f)=>l-f);let i=n.reduce((l,f)=>l+f,0)/n.length,e=n[Math.floor(n.length/2)],w=n[0];return{avgMs:i,fps:1e3/i,medianMs:e,minMs:w}}function Mt(t){a.queue.copyExternalImageToTexture({source:t},{texture:M},[256,256]);let s=a.createCommandEncoder();{let n=s.beginComputePass();n.setPipeline(N),n.setBindGroup(0,$),n.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),n.end()}E(s,y),a.queue.submit([s.finish()])}async function Lt(t,s=50){function n(o){let r=[...o].sort((D,v)=>D-v);return{median:r[Math.floor(r.length/2)],min:r[0]}}for(let o=0;o<10;o++)await qe(t);let i=[];for(let o=0;o<s;o++){a.queue.copyExternalImageToTexture({source:t},{texture:M},[256,256]);let r=a.createCommandEncoder();{let v=r.beginComputePass();v.setPipeline(N),v.setBindGroup(0,$),v.dispatchWorkgroups(16,16,1),v.end()}E(r,y);let D=performance.now();a.queue.submit([r.finish()]),await a.queue.onSubmittedWorkDone(),i.push(performance.now()-D)}let e=[];for(let o=0;o<s;o++){a.queue.copyExternalImageToTexture({source:t},{texture:M},[256,256]);let r=a.createCommandEncoder();{let I=r.beginComputePass();I.setPipeline(N),I.setBindGroup(0,$),I.dispatchWorkgroups(16,16,1),I.end()}E(r,y),a.queue.submit([r.finish()]);let D=y.mapAsync(GPUMapMode.READ),v=performance.now();await a.queue.onSubmittedWorkDone(),await D,y.getMappedRange(),y.unmap(),e.push(performance.now()-v)}let w=[];for(let o=0;o<s;o++){a.queue.copyExternalImageToTexture({source:t},{texture:M},[256,256]);let r=a.createCommandEncoder();{let v=r.beginComputePass();v.setPipeline(N),v.setBindGroup(0,$),v.dispatchWorkgroups(16,16,1),v.end()}E(r,y),a.queue.submit([r.finish()]);let D=performance.now();await y.mapAsync(GPUMapMode.READ),y.getMappedRange(),y.unmap(),w.push(performance.now()-D)}let l=[];for(let o=0;o<s;o++){let r=performance.now();await qe(t),l.push(performance.now()-r)}await $e(t);let f=[];for(let o=0;o<s;o++){let r=performance.now();await $e(t),f.push(performance.now()-r)}await Da();let _=null;if(ue){let o=[];for(let r=0;r<s;r++){let D=performance.now();await Sa(t),o.push(performance.now()-D)}_=n(o)}return{gpuOnly:n(i),mapAsyncOnly:n(e),mapAsyncNoWait:n(w),total:n(l),pipelined:n(f),renderReadback:_}}return{device:a,run:de,runFromCanvas:qe,runFromCanvasViaRender:Sa,runFromCanvasPipelined:$e,flushPipelined:Da,benchmark:Ht,benchmarkGPU:Et,benchmarkDiagnostic:Lt,_device:a,_benchmarkSubmitOnly:Mt}}var qt="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Nt(u={}){let{weightsUrl:d,scoreThreshold:a=.5}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let g=d??qt,m=g.endsWith("/")?`${g}weights.json`:`${g}/weights.json`,C=g.endsWith("/")?`${g}weights.bin`:`${g}/weights.bin`,[c,A]=await Promise.all([fetch(m),fetch(C)]);if(!c.ok)throw new Error(`Failed to fetch weights metadata: ${c.status}`);if(!A.ok)throw new Error(`Failed to fetch weights binary: ${A.status}`);let W=await c.json(),H=await A.arrayBuffer(),P=it(W,H),p=await nt(P),b=null;function S(){return b||(b=new OffscreenCanvas(256,256)),b}async function Z(h){if(h instanceof HTMLCanvasElement||h instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&h instanceof ImageBitmap)return h;let U=S();U.width=256,U.height=256;let x=U.getContext("2d");return h instanceof ImageData?x.putImageData(h,0,0):x.drawImage(h,0,0,256,256),U}function K(h,U,x){let Q=h[0];if(Q<a)return null;let be=U[0]>.5,ee=[];for(let z=0;z<21;z++)ee.push({x:x[z*3],y:x[z*3+1],z:x[z*3+2]});return{score:Q,handedness:be?"right":"left",landmarks:ee}}async function le(h){let U=await Z(h),x=await p.runFromCanvas(U);return K(x.handflag,x.handedness,x.landmarks)}async function ce(h){let U=await Z(h),x=await p.runFromCanvasPipelined(U);return x?K(x.handflag,x.handedness,x.landmarks):null}async function we(){let h=await p.flushPipelined();return h?K(h.handflag,h.handedness,h.landmarks):null}function me(){p.device.destroy(),b=null}async function he(h){let U=await Z(h);return p.benchmarkDiagnostic(U)}return{detect:le,detectPipelined:ce,flushPipelined:we,dispose:me,benchmarkDiagnostic:he}}export{Nt as createHandpose};
