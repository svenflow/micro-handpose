function P(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Ha=P(`
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
`),Ma=P(`
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
`),Wa=P(`
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
`),La=P(`
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
`);function ge(o,_){return Ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${_},1)`)}function ye(o,_){return Ha.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${_},1)`)}function xe(o,_){return Wa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${_},1)`)}function ve(o,_){return La.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${_},1)`)}function Pe(o,_){return[8,8]}var ke=P(`
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
`),Be=P(`
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
`);function Ue(o){return P(`
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
`)}var De=Ue(!1),Se=Ue(!0),Ge=P(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Ae=P(`
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
`);function Ce(o){return P(`
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
`)}var Ee=Ce("sigmoid"),He=Ce("div256"),Me=P(`
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> handflag_w:array<f32>;
@group(0)@binding(2) var<storage,read> handflag_b:array<f32>;
@group(0)@binding(3) var<storage,read> handedness_w:array<f32>;
@group(0)@binding(4) var<storage,read> handedness_b:array<f32>;
@group(0)@binding(5) var<storage,read> landmarks_w:array<f32>;
@group(0)@binding(6) var<storage,read> landmarks_b:array<f32>;
@group(0)@binding(7) var<storage,read_write> handflag:array<f32>;
@group(0)@binding(8) var<storage,read_write> handedness:array<f32>;
@group(0)@binding(9) var<storage,read_write> landmarks:array<f32>;
const IN_CHANNELS:u32=288u;
@compute @workgroup_size(65)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  var sum:f32=0.0;
  var w_base:u32; var bias_val:f32;
  if(oc==0u){
    for(var ic:u32=0u;ic<IN_CHANNELS;ic++){ for(var y:u32=0u;y<2u;y++){ for(var x:u32=0u;x<2u;x++){
      let in_idx=ic*4u+y*2u+x; sum+=input[in_idx]*handflag_w[in_idx];
    } } }
    sum+=handflag_b[0]; handflag[0]=1.0/(1.0+exp(-sum));
  } else if(oc==1u){
    for(var ic:u32=0u;ic<IN_CHANNELS;ic++){ for(var y:u32=0u;y<2u;y++){ for(var x:u32=0u;x<2u;x++){
      let in_idx=ic*4u+y*2u+x; sum+=input[in_idx]*handedness_w[in_idx];
    } } }
    sum+=handedness_b[0]; handedness[0]=1.0/(1.0+exp(-sum));
  } else {
    let landmark_oc=oc-2u;
    for(var ic:u32=0u;ic<IN_CHANNELS;ic++){ for(var y:u32=0u;y<2u;y++){ for(var x:u32=0u;x<2u;x++){
      let in_idx=ic*4u+y*2u+x; let w_idx=landmark_oc*IN_CHANNELS*4u+ic*4u+y*2u+x;
      sum+=input[in_idx]*landmarks_w[w_idx];
    } } }
    sum+=landmarks_b[landmark_oc]; landmarks[landmark_oc]=sum/256.0;
  }
}
`),We=P(`
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
`);var Le=P(`
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
`);function Re(o,_){let n=new Map,m=o.dtype??"float32";for(let c=0;c<o.keys.length;c++){let W=o.keys[c],p=o.shapes[c],D=o.offsets[c],G=p.reduce((b,u)=>b*u,1),B;if(m==="float32")B=new Float32Array(_,D,G);else{let b=new DataView(_);B=new Float32Array(G);for(let u=0;u<G;u++)B[u]=yt(b.getUint16(D+u*2,!0))}n.set(W,{data:B,shape:p})}return n}function yt(o){let _=o>>15&1,n=o>>10&31,m=o&1023;if(n===0){if(m===0)return _?-0:0;let p=-14,D=m/1024;return(_?-1:1)*Math.pow(2,p)*D}if(n===31)return m===0?_?-1/0:1/0:NaN;let c=n-15,W=1+m/1024;return(_?-1:1)*Math.pow(2,c)*W}var xt=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Te=xt.map(([o,_,n,m,c])=>({type:"resmodule",inCh:o,outCh:_,h:n,w:n,stride:m,prefix:c})),vt=2,Pt=5,kt=8,Bt=11;async function Oe(o){if(!navigator.gpu)throw new Error("WebGPU not supported");let _=await navigator.gpu.requestAdapter();if(!_)throw new Error("No GPU adapter found");let n=await _.requestDevice(),m={r:"read-only-storage",s:"storage",u:"uniform"};function c(e){return n.createBindGroupLayout({entries:e.map((s,i)=>({binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:m[s]}}))})}function W(e){return n.createBindGroupLayout({entries:e.map((s,i)=>s==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:m[s]}})})}let p=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,D=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,G=GPUBufferUsage.STORAGE,B=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,b=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function u(e,s){return n.createBuffer({size:e,usage:s})}function w(e,s){return n.createBindGroup({layout:e,entries:s.map((i,t)=>({binding:t,resource:"size"in i?{buffer:i}:i}))})}function x(e,s){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:s,entryPoint:"main"}})}let J=n.createShaderModule({code:ke}),Q=n.createShaderModule({code:Le}),aa=n.createShaderModule({code:Me}),ea=n.createShaderModule({code:Ma}),y=n.createShaderModule({code:Ha}),S=n.createShaderModule({code:Wa}),k=n.createShaderModule({code:La}),z=n.createShaderModule({code:Be}),ta=n.createShaderModule({code:De}),F=n.createShaderModule({code:Ge}),L=n.createShaderModule({code:Se}),Ie=n.createShaderModule({code:Ae}),ze=n.createShaderModule({code:Ee}),Fe=n.createShaderModule({code:He}),Ne=n.createShaderModule({code:We}),ia=c(["r","r","r","s","u"]),na=c(["r","r","r","r","s","u"]),Ta=c(["r","s","u"]),Ra=c(["r","r","r","s","u"]),qe=c(["r","s","u"]),Ye=c(["r","r","s","u"]),N=c(["r","r","s","u"]),Oa=c(["r","r","r","s","u"]),O=c(["r","r","r","s","u"]),Ia=W(["t","s","u"]),za=c(["r","r","r","r","r","r","r","s","s","s"]),ra=c(["r","r","r","r","r","s","u"]),Xe=n.createPipelineLayout({bindGroupLayouts:[ia]}),$e=n.createPipelineLayout({bindGroupLayouts:[na]}),q=e=>n.createComputePipeline({layout:Xe,compute:{module:e,entryPoint:"main"}}),Y=e=>n.createComputePipeline({layout:$e,compute:{module:e,entryPoint:"main"}}),Ze=q(ea),Ve=q(y),je=Y(S),Ke=Y(k),Fa=new Map,Na=new Map,qa=new Map,Ya=new Map;Fa.set("8,8",Ze),Na.set("8,8",Ve),qa.set("8,8",je),Ya.set("8,8",Ke);function X(e,s,i,t,a){let f=`${s},${i}`,d=e.get(f);return d||(d=a(n.createShaderModule({code:t(s,i)})),e.set(f,d)),d}let Je=(e,s)=>X(Fa,e,s,ge,q),Qe=(e,s)=>X(Na,e,s,ye,q),at=(e,s)=>X(qa,e,s,xe,Y),et=(e,s)=>X(Ya,e,s,ve,Y),T=Te.map(e=>{let s=e.stride===2?e.h/2:e.h,i=e.stride===2?e.w/2:e.w,[t,a]=Pe(e.inCh,s),f=e.h>=64,d=s>=16&&e.inCh>=288&&e.outCh>=288&&e.outCh%2===0;return{dwPipeline:f?Qe(t,a):Je(t,a),pwPipeline:d?et(t,a):at(t,a),dwDispatchX:Math.ceil(i/t),dwDispatchY:Math.ceil(s/a),dwDispatchZ:e.inCh,pwDispatchX:Math.ceil(i/t),pwDispatchY:Math.ceil(s/a),pwDispatchZ:d?e.outCh/2:e.outCh}}),Xa=x(Ta,J),tt=x(Ra,z);x(qe,ta),x(Ye,F);let sa=x(N,L),it=x(Oa,Ie);x(O,ze),x(O,Fe);let nt=x(Ia,Q),rt=x(za,aa),st=x(ra,Ne),ua=1*288*128*128*4,$a=u(3*256*256*4,p),oa=u(3*257*257*4,G),Za=u(12,b);n.queue.writeBuffer(Za,0,new Uint32Array([3,256,257]));let g=u(ua,D),U=u(ua,B),$=u(ua,G),Va=u(3072*64*4,p),ja=u(3072*32*4,p),Ka=u(1536*16*4,p),Ja=u(6144*64*4,G),pa=u(4,B),_a=u(4,B),da=u(252,B),A=u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let Qa=n.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ae=u(8,b);n.queue.writeBuffer(ae,0,new Uint32Array([256,257]));let ca=o.get("backbone1.1.weight")?.data,wa=o.get("backbone1.1.bias")?.data;if(!ca||!wa)throw new Error("Missing input conv weights");let ee=u(ca.byteLength,p),te=u(wa.byteLength,p),ie=u(28,b);n.queue.writeBuffer(ee,0,ca),n.queue.writeBuffer(te,0,wa),n.queue.writeBuffer(ie,0,new Uint32Array([1,3,24,257,257,128,128]));let la=o.get("backbone6.1.weight")?.data,ha=o.get("backbone6.1.bias")?.data;if(!la||!ha)throw new Error("Missing backbone6.1 conv1x1 weights");let ne=u(la.byteLength,p),re=u(ha.byteLength,p),se=u(20,b);n.queue.writeBuffer(ne,0,la),n.queue.writeBuffer(re,0,ha),n.queue.writeBuffer(se,0,new Uint32Array([1,96,48,32,32]));let ma=o.get("handflag.weight")?.data,fa=o.get("handflag.bias")?.data;if(!ma||!fa)throw new Error("Missing handflag weights");let ba=u(ma.byteLength,p),ga=u(fa.byteLength,p),ue=u(12,b);n.queue.writeBuffer(ba,0,ma),n.queue.writeBuffer(ga,0,fa),n.queue.writeBuffer(ue,0,new Uint32Array([1,288,1]));let ya=o.get("handedness.weight")?.data,xa=o.get("handedness.bias")?.data;if(!ya||!xa)throw new Error("Missing handedness weights");let va=u(ya.byteLength,p),Pa=u(xa.byteLength,p),oe=u(12,b);n.queue.writeBuffer(va,0,ya),n.queue.writeBuffer(Pa,0,xa),n.queue.writeBuffer(oe,0,new Uint32Array([1,288,1]));let ka=o.get("reg_3d.weight")?.data,Ba=o.get("reg_3d.bias")?.data;if(!ka||!Ba)throw new Error("Missing reg_3d weights");let Ua=u(ka.byteLength,p),Da=u(Ba.byteLength,p),pe=u(12,b);n.queue.writeBuffer(Ua,0,ka),n.queue.writeBuffer(Da,0,Ba),n.queue.writeBuffer(pe,0,new Uint32Array([1,288,63]));let I=Te.map(e=>{let{inCh:s,outCh:i,h:t,w:a,stride:f,prefix:d}=e,v=f===2?t/2:t,l=f===2?a/2:a,h=f===1?2:1,r=o.get(`${d}convs.0.weight`)?.data,Aa=o.get(`${d}convs.0.bias`)?.data,Ca=o.get(`${d}convs.1.weight`)?.data,Ea=o.get(`${d}convs.1.bias`)?.data;if(!r||!Aa||!Ca||!Ea)throw new Error(`Missing weights for ${d}`);let we=u(r.byteLength,p),le=u(Aa.byteLength,p),he=u(Ca.byteLength,p),me=u(Ea.byteLength,p),fe=u(32,b),be=u(36,b);return n.queue.writeBuffer(we,0,r),n.queue.writeBuffer(le,0,Aa),n.queue.writeBuffer(he,0,Ca),n.queue.writeBuffer(me,0,Ea),n.queue.writeBuffer(fe,0,new Uint32Array([1,s,t,a,v,l,f,h])),n.queue.writeBuffer(be,0,new Uint32Array([1,s,i,v,l,Math.max(0,i-s),f,t,a])),{dwWeight:we,dwBias:le,pwWeight:he,pwBias:me,dwUniform:fe,pwUniform:be,spec:e,outH:v,outW:l}});function R(e){let s=u(e.length*4,b);return n.queue.writeBuffer(s,0,new Uint32Array(e)),s}let ut=R([1,96,8,8,16,16]),ot=R([1,96,16,16,32,32]),pt=R([1,48,32,32,64,64]);R([1536*16]),R([3072*32]),R([3072*64]);let _e=w(Ta,[$a,oa,Za]),_t=w(Ra,[oa,ee,te,g,ie]),C=[],E=[],H=[],M=[];for(let e of I)C.push(w(ia,[g,e.dwWeight,e.dwBias,$,e.dwUniform])),E.push(w(na,[$,g,e.pwWeight,e.pwBias,U,e.pwUniform])),H.push(w(ia,[U,e.dwWeight,e.dwBias,$,e.dwUniform])),M.push(w(na,[$,U,e.pwWeight,e.pwBias,g,e.pwUniform]));let dt=w(N,[g,Ka,U,ut]),ct=w(N,[g,ja,U,ot]),wt=w(Oa,[g,ne,re,Ja,se]),lt=w(N,[Ja,Va,U,pt]);w(O,[g,ba,ga,pa,ue]),w(O,[g,va,Pa,_a,oe]),w(O,[g,Ua,Da,da,pe]);let ht=w(Ia,[Qa.createView(),oa,ae]),mt=w(za,[g,ba,ga,va,Pa,Ua,Da,pa,_a,da]),Sa=24,de=[],ce=[];for(let e=Sa;e<I.length;e++){let s=I[e];de.push(w(ra,[g,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,U,s.dwUniform])),ce.push(w(ra,[U,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,g,s.dwUniform]))}let Z=new Float32Array(1),V=new Float32Array(1),j=new Float32Array(63);function Ga(e,s){let i=!0,t=0,a=e.beginComputePass();for(a.setPipeline(tt),a.setBindGroup(0,_t),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);t<=vt;t++){let l=i?C[t]:H[t],h=i?E[t]:M[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.end();let f=i?g:U;for(e.copyBufferToBuffer(f,0,Va,0,3072*64*4),a=e.beginComputePass();t<=Pt;t++){let l=i?C[t]:H[t],h=i?E[t]:M[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.end();let d=i?g:U;for(e.copyBufferToBuffer(d,0,ja,0,3072*32*4),a=e.beginComputePass();t<=kt;t++){let l=i?C[t]:H[t],h=i?E[t]:M[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.end();let v=i?g:U;for(e.copyBufferToBuffer(v,0,Ka,0,1536*16*4),a=e.beginComputePass();t<=Bt;t++){let l=i?C[t]:H[t],h=i?E[t]:M[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.setPipeline(sa),a.setBindGroup(0,dt),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),i=!1,a=e.beginComputePass();{let l=i?C[t]:H[t],h=i?E[t]:M[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i,t++}a.setPipeline(sa),a.setBindGroup(0,ct),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),i=!1,a=e.beginComputePass();{let l=i?C[t]:H[t],h=i?E[t]:M[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i,t++}for(a.setPipeline(it),a.setBindGroup(0,wt),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(sa),a.setBindGroup(0,lt),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),i=!1,a=e.beginComputePass();t<Sa;t++){let l=i?C[t]:H[t],h=i?E[t]:M[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}for(;t<I.length;t++){let l=t-Sa,h=i?de[l]:ce[l],r=I[t];a.setPipeline(st),a.setBindGroup(0,h),a.dispatchWorkgroups(r.outW,r.outH,1),i=!i}a.setPipeline(rt),a.setBindGroup(0,mt),a.dispatchWorkgroups(1),a.end(),e.copyBufferToBuffer(pa,0,s,0,4),e.copyBufferToBuffer(_a,0,s,4,4),e.copyBufferToBuffer(da,0,s,8,252)}async function K(e){n.queue.writeBuffer($a,0,e);let s=n.createCommandEncoder();{let t=s.beginComputePass();t.setPipeline(Xa),t.setBindGroup(0,_e),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}Ga(s,A),n.queue.submit([s.finish()]),await A.mapAsync(GPUMapMode.READ);let i=new Float32Array(A.getMappedRange());return Z[0]=i[0],V[0]=i[1],j.set(i.subarray(2,65)),A.unmap(),{handflag:new Float32Array(Z),handedness:new Float32Array(V),landmarks:new Float32Array(j)}}async function ft(e){n.queue.copyExternalImageToTexture({source:e},{texture:Qa},[256,256]);let s=n.createCommandEncoder();{let t=s.beginComputePass();t.setPipeline(nt),t.setBindGroup(0,ht),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Ga(s,A),n.queue.submit([s.finish()]),await A.mapAsync(GPUMapMode.READ);let i=new Float32Array(A.getMappedRange());return Z[0]=i[0],V[0]=i[1],j.set(i.subarray(2,65)),A.unmap(),{handflag:new Float32Array(Z),handedness:new Float32Array(V),landmarks:new Float32Array(j)}}async function bt(e=50){let s=new Float32Array(196608);for(let a=0;a<5;a++)await K(s);let i=[];for(let a=0;a<e;a++){let f=performance.now();await K(s),i.push(performance.now()-f)}let t=i.reduce((a,f)=>a+f,0)/i.length;return{avgMs:t,fps:1e3/t}}async function gt(e=50){let s=new Float32Array(196608);for(let d=0;d<5;d++)await K(s);let i=[];for(let d=0;d<e;d++){let v=n.createCommandEncoder();{let h=v.beginComputePass();h.setPipeline(Xa),h.setBindGroup(0,_e),h.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),h.end()}Ga(v,A);let l=performance.now();n.queue.submit([v.finish()]),await n.queue.onSubmittedWorkDone(),i.push(performance.now()-l)}i.sort((d,v)=>d-v);let t=i.reduce((d,v)=>d+v,0)/i.length,a=i[Math.floor(i.length/2)],f=i[0];return{avgMs:t,fps:1e3/t,medianMs:a,minMs:f}}return{device:n,run:K,runFromCanvas:ft,benchmark:bt,benchmarkGPU:gt}}var Ut="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Dt(o={}){let{weightsUrl:_,scoreThreshold:n=.5}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let m=_??Ut,c=m.endsWith("/")?`${m}weights.json`:`${m}/weights.json`,W=m.endsWith("/")?`${m}weights.bin`:`${m}/weights.bin`,[p,D]=await Promise.all([fetch(c),fetch(W)]);if(!p.ok)throw new Error(`Failed to fetch weights metadata: ${p.status}`);if(!D.ok)throw new Error(`Failed to fetch weights binary: ${D.status}`);let G=await p.json(),B=await D.arrayBuffer(),b=Re(G,B),u=await Oe(b),w=null;function x(){return w||(w=new OffscreenCanvas(256,256)),w}async function J(y){if(y instanceof HTMLCanvasElement||y instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&y instanceof ImageBitmap)return y;let S=x();S.width=256,S.height=256;let k=S.getContext("2d");return y instanceof ImageData?k.putImageData(y,0,0):k.drawImage(y,0,0,256,256),S}function Q(y,S,k){let z=y[0];if(z<n)return null;let ta=S[0]>.5,F=[];for(let L=0;L<21;L++)F.push({x:k[L*3],y:k[L*3+1],z:k[L*3+2]});return{score:z,handedness:ta?"right":"left",landmarks:F}}async function aa(y){let S=await J(y),k=await u.runFromCanvas(S);return Q(k.handflag,k.handedness,k.landmarks)}function ea(){u.device.destroy(),w=null}return{detect:aa,dispose:ea}}export{Dt as createHandpose};
