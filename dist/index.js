function P(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Wa=P(`
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
`),Ha=P(`
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
`),Ta=P(`
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
`);function ye(o,p){return Ha.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function ve(o,p){return Wa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function Pe(o,p){return La.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function ke(o,p){return Ta.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function Be(o,p){return[8,8]}var Ue=P(`
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
`),Se=P(`
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
`);function De(o){return P(`
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
`)}var Ge=De(!1),Ae=De(!0),Ce=P(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Ee=P(`
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
`);function Me(o){return P(`
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
`)}var We=Me("sigmoid"),He=Me("div256"),Le=P(`
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
`),Te=P(`
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
`);var Re=P(`
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
`);function Ie(o,p){let n=new Map,m=o.dtype??"float32";for(let c=0;c<o.keys.length;c++){let H=o.keys[c],_=o.shapes[c],S=o.offsets[c],A=_.reduce((b,u)=>b*u,1),B;if(m==="float32")B=new Float32Array(p,S,A);else{let b=new DataView(p);B=new Float32Array(A);for(let u=0;u<A;u++)B[u]=yt(b.getUint16(S+u*2,!0))}n.set(H,{data:B,shape:_})}return n}function yt(o){let p=o>>15&1,n=o>>10&31,m=o&1023;if(n===0){if(m===0)return p?-0:0;let _=-14,S=m/1024;return(p?-1:1)*Math.pow(2,_)*S}if(n===31)return m===0?p?-1/0:1/0:NaN;let c=n-15,H=1+m/1024;return(p?-1:1)*Math.pow(2,c)*H}var vt=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Oe=vt.map(([o,p,n,m,c])=>({type:"resmodule",inCh:o,outCh:p,h:n,w:n,stride:m,prefix:c})),Pt=2,kt=5,Bt=8,Ut=11;async function ze(o){if(!navigator.gpu)throw new Error("WebGPU not supported");let p=await navigator.gpu.requestAdapter();if(!p)throw new Error("No GPU adapter found");let n=await p.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(p.limits.maxStorageBuffersPerShaderStage,10),maxComputeWorkgroupSizeX:Math.min(p.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(p.limits.maxComputeInvocationsPerWorkgroup,288)}}),m={r:"read-only-storage",s:"storage",u:"uniform"};function c(e){return n.createBindGroupLayout({entries:e.map((s,i)=>({binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:m[s]}}))})}function H(e){return n.createBindGroupLayout({entries:e.map((s,i)=>s==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:m[s]}})})}let _=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,S=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.STORAGE,B=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,b=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function u(e,s){return n.createBuffer({size:e,usage:s})}function w(e,s){return n.createBindGroup({layout:e,entries:s.map((i,t)=>({binding:t,resource:"size"in i?{buffer:i}:i}))})}function y(e,s){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:s,entryPoint:"main"}})}let Q=n.createShaderModule({code:Ue}),aa=n.createShaderModule({code:Re}),ea=n.createShaderModule({code:Le}),ta=n.createShaderModule({code:Ha}),x=n.createShaderModule({code:Wa}),D=n.createShaderModule({code:La}),k=n.createShaderModule({code:Ta}),z=n.createShaderModule({code:Se}),ia=n.createShaderModule({code:Ge}),F=n.createShaderModule({code:Ce}),L=n.createShaderModule({code:Ae}),Fe=n.createShaderModule({code:Ee}),Ne=n.createShaderModule({code:We}),qe=n.createShaderModule({code:He}),Xe=n.createShaderModule({code:Te}),na=c(["r","r","r","s","u"]),ra=c(["r","r","r","r","s","u"]),Ra=c(["r","s","u"]),Oa=c(["r","r","r","s","u"]),Ye=c(["r","s","u"]),$e=c(["r","r","s","u"]),N=c(["r","r","s","u"]),Ia=c(["r","r","r","s","u"]),O=c(["r","r","r","s","u"]),za=H(["t","s","u"]),Fa=c(["r","r","r","r","r","r","r","s","s","s"]),sa=c(["r","r","r","r","r","s","u"]),Ze=n.createPipelineLayout({bindGroupLayouts:[na]}),Ve=n.createPipelineLayout({bindGroupLayouts:[ra]}),q=e=>n.createComputePipeline({layout:Ze,compute:{module:e,entryPoint:"main"}}),X=e=>n.createComputePipeline({layout:Ve,compute:{module:e,entryPoint:"main"}}),je=q(ta),Ke=q(x),Je=X(D),Qe=X(k),Na=new Map,qa=new Map,Xa=new Map,Ya=new Map;Na.set("8,8",je),qa.set("8,8",Ke),Xa.set("8,8",Je),Ya.set("8,8",Qe);function Y(e,s,i,t,a){let f=`${s},${i}`,d=e.get(f);return d||(d=a(n.createShaderModule({code:t(s,i)})),e.set(f,d)),d}let at=(e,s)=>Y(Na,e,s,ye,q),et=(e,s)=>Y(qa,e,s,ve,q),tt=(e,s)=>Y(Xa,e,s,Pe,X),it=(e,s)=>Y(Ya,e,s,ke,X),T=Oe.map(e=>{let s=e.stride===2?e.h/2:e.h,i=e.stride===2?e.w/2:e.w,[t,a]=Be(e.inCh,s),f=e.h>=64,d=s>=16&&e.inCh>=288&&e.outCh>=288&&e.outCh%2===0;return{dwPipeline:f?et(t,a):at(t,a),pwPipeline:d?it(t,a):tt(t,a),dwDispatchX:Math.ceil(i/t),dwDispatchY:Math.ceil(s/a),dwDispatchZ:e.inCh,pwDispatchX:Math.ceil(i/t),pwDispatchY:Math.ceil(s/a),pwDispatchZ:d?e.outCh/2:e.outCh}}),$a=y(Ra,Q),nt=y(Oa,z);y(Ye,ia),y($e,F);let ua=y(N,L),rt=y(Ia,Fe);y(O,Ne),y(O,qe);let Za=y(za,aa),st=y(Fa,ea),ut=y(sa,Xe),oa=1*288*128*128*4,Va=u(3*256*256*4,_),pa=u(3*257*257*4,A),ja=u(12,b);n.queue.writeBuffer(ja,0,new Uint32Array([3,256,257]));let g=u(oa,S),U=u(oa,B),$=u(oa,A),Ka=u(3072*64*4,_),Ja=u(3072*32*4,_),Qa=u(1536*16*4,_),ae=u(6144*64*4,A),_a=u(4,B),da=u(4,B),ca=u(252,B),G=u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let wa=n.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ee=u(8,b);n.queue.writeBuffer(ee,0,new Uint32Array([256,257]));let la=o.get("backbone1.1.weight")?.data,ha=o.get("backbone1.1.bias")?.data;if(!la||!ha)throw new Error("Missing input conv weights");let te=u(la.byteLength,_),ie=u(ha.byteLength,_),ne=u(28,b);n.queue.writeBuffer(te,0,la),n.queue.writeBuffer(ie,0,ha),n.queue.writeBuffer(ne,0,new Uint32Array([1,3,24,257,257,128,128]));let ma=o.get("backbone6.1.weight")?.data,fa=o.get("backbone6.1.bias")?.data;if(!ma||!fa)throw new Error("Missing backbone6.1 conv1x1 weights");let re=u(ma.byteLength,_),se=u(fa.byteLength,_),ue=u(20,b);n.queue.writeBuffer(re,0,ma),n.queue.writeBuffer(se,0,fa),n.queue.writeBuffer(ue,0,new Uint32Array([1,96,48,32,32]));let ba=o.get("handflag.weight")?.data,ga=o.get("handflag.bias")?.data;if(!ba||!ga)throw new Error("Missing handflag weights");let xa=u(ba.byteLength,_),ya=u(ga.byteLength,_),oe=u(12,b);n.queue.writeBuffer(xa,0,ba),n.queue.writeBuffer(ya,0,ga),n.queue.writeBuffer(oe,0,new Uint32Array([1,288,1]));let va=o.get("handedness.weight")?.data,Pa=o.get("handedness.bias")?.data;if(!va||!Pa)throw new Error("Missing handedness weights");let ka=u(va.byteLength,_),Ba=u(Pa.byteLength,_),pe=u(12,b);n.queue.writeBuffer(ka,0,va),n.queue.writeBuffer(Ba,0,Pa),n.queue.writeBuffer(pe,0,new Uint32Array([1,288,1]));let Ua=o.get("reg_3d.weight")?.data,Sa=o.get("reg_3d.bias")?.data;if(!Ua||!Sa)throw new Error("Missing reg_3d weights");let Da=u(Ua.byteLength,_),Ga=u(Sa.byteLength,_),_e=u(12,b);n.queue.writeBuffer(Da,0,Ua),n.queue.writeBuffer(Ga,0,Sa),n.queue.writeBuffer(_e,0,new Uint32Array([1,288,63]));let I=Oe.map(e=>{let{inCh:s,outCh:i,h:t,w:a,stride:f,prefix:d}=e,v=f===2?t/2:t,l=f===2?a/2:a,h=f===1?2:1,r=o.get(`${d}convs.0.weight`)?.data,Ca=o.get(`${d}convs.0.bias`)?.data,Ea=o.get(`${d}convs.1.weight`)?.data,Ma=o.get(`${d}convs.1.bias`)?.data;if(!r||!Ca||!Ea||!Ma)throw new Error(`Missing weights for ${d}`);let he=u(r.byteLength,_),me=u(Ca.byteLength,_),fe=u(Ea.byteLength,_),be=u(Ma.byteLength,_),ge=u(32,b),xe=u(36,b);return n.queue.writeBuffer(he,0,r),n.queue.writeBuffer(me,0,Ca),n.queue.writeBuffer(fe,0,Ea),n.queue.writeBuffer(be,0,Ma),n.queue.writeBuffer(ge,0,new Uint32Array([1,s,t,a,v,l,f,h])),n.queue.writeBuffer(xe,0,new Uint32Array([1,s,i,v,l,Math.max(0,i-s),f,t,a])),{dwWeight:he,dwBias:me,pwWeight:fe,pwBias:be,dwUniform:ge,pwUniform:xe,spec:e,outH:v,outW:l}});function R(e){let s=u(e.length*4,b);return n.queue.writeBuffer(s,0,new Uint32Array(e)),s}let ot=R([1,96,8,8,16,16]),pt=R([1,96,16,16,32,32]),_t=R([1,48,32,32,64,64]);R([1536*16]),R([3072*32]),R([3072*64]);let de=w(Ra,[Va,pa,ja]),dt=w(Oa,[pa,te,ie,g,ne]),C=[],E=[],M=[],W=[];for(let e of I)C.push(w(na,[g,e.dwWeight,e.dwBias,$,e.dwUniform])),E.push(w(ra,[$,g,e.pwWeight,e.pwBias,U,e.pwUniform])),M.push(w(na,[U,e.dwWeight,e.dwBias,$,e.dwUniform])),W.push(w(ra,[$,U,e.pwWeight,e.pwBias,g,e.pwUniform]));let ct=w(N,[g,Qa,U,ot]),wt=w(N,[g,Ja,U,pt]),lt=w(Ia,[g,re,se,ae,ue]),ht=w(N,[ae,Ka,U,_t]);w(O,[g,xa,ya,_a,oe]),w(O,[g,ka,Ba,da,pe]),w(O,[g,Da,Ga,ca,_e]);let ce=w(za,[wa.createView(),pa,ee]),mt=w(Fa,[g,xa,ya,ka,Ba,Da,Ga,_a,da,ca]),Aa=24,we=[],le=[];for(let e=Aa;e<I.length;e++){let s=I[e];we.push(w(sa,[g,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,U,s.dwUniform])),le.push(w(sa,[U,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,g,s.dwUniform]))}let Z=new Float32Array(1),V=new Float32Array(1),j=new Float32Array(63);function K(e,s){let i=!0,t=0,a=e.beginComputePass();for(a.setPipeline(nt),a.setBindGroup(0,dt),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);t<=Pt;t++){let l=i?C[t]:M[t],h=i?E[t]:W[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.end();let f=i?g:U;for(e.copyBufferToBuffer(f,0,Ka,0,3072*64*4),a=e.beginComputePass();t<=kt;t++){let l=i?C[t]:M[t],h=i?E[t]:W[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.end();let d=i?g:U;for(e.copyBufferToBuffer(d,0,Ja,0,3072*32*4),a=e.beginComputePass();t<=Bt;t++){let l=i?C[t]:M[t],h=i?E[t]:W[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.end();let v=i?g:U;for(e.copyBufferToBuffer(v,0,Qa,0,1536*16*4),a=e.beginComputePass();t<=Ut;t++){let l=i?C[t]:M[t],h=i?E[t]:W[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}a.setPipeline(ua),a.setBindGroup(0,ct),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),i=!1,a=e.beginComputePass();{let l=i?C[t]:M[t],h=i?E[t]:W[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i,t++}a.setPipeline(ua),a.setBindGroup(0,wt),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),i=!1,a=e.beginComputePass();{let l=i?C[t]:M[t],h=i?E[t]:W[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i,t++}for(a.setPipeline(rt),a.setBindGroup(0,lt),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(ua),a.setBindGroup(0,ht),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),i=!1,a=e.beginComputePass();t<Aa;t++){let l=i?C[t]:M[t],h=i?E[t]:W[t],r=T[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}for(;t<I.length;t++){let l=t-Aa,h=i?we[l]:le[l],r=I[t];a.setPipeline(ut),a.setBindGroup(0,h),a.dispatchWorkgroups(r.outW,r.outH,1),i=!i}a.setPipeline(st),a.setBindGroup(0,mt),a.dispatchWorkgroups(1),a.end(),e.copyBufferToBuffer(_a,0,s,0,4),e.copyBufferToBuffer(da,0,s,4,4),e.copyBufferToBuffer(ca,0,s,8,252)}async function J(e){n.queue.writeBuffer(Va,0,e);let s=n.createCommandEncoder();{let t=s.beginComputePass();t.setPipeline($a),t.setBindGroup(0,de),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}K(s,G),n.queue.submit([s.finish()]),await G.mapAsync(GPUMapMode.READ);let i=new Float32Array(G.getMappedRange());return Z[0]=i[0],V[0]=i[1],j.set(i.subarray(2,65)),G.unmap(),{handflag:new Float32Array(Z),handedness:new Float32Array(V),landmarks:new Float32Array(j)}}async function ft(e){n.queue.copyExternalImageToTexture({source:e},{texture:wa},[256,256]);let s=n.createCommandEncoder();{let t=s.beginComputePass();t.setPipeline(Za),t.setBindGroup(0,ce),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}K(s,G),n.queue.submit([s.finish()]),await G.mapAsync(GPUMapMode.READ);let i=new Float32Array(G.getMappedRange());return Z[0]=i[0],V[0]=i[1],j.set(i.subarray(2,65)),G.unmap(),{handflag:new Float32Array(Z),handedness:new Float32Array(V),landmarks:new Float32Array(j)}}async function bt(e=50){let s=new Float32Array(196608);for(let a=0;a<5;a++)await J(s);let i=[];for(let a=0;a<e;a++){let f=performance.now();await J(s),i.push(performance.now()-f)}let t=i.reduce((a,f)=>a+f,0)/i.length;return{avgMs:t,fps:1e3/t}}async function gt(e=50){let s=new Float32Array(196608);for(let d=0;d<5;d++)await J(s);let i=[];for(let d=0;d<e;d++){let v=n.createCommandEncoder();{let h=v.beginComputePass();h.setPipeline($a),h.setBindGroup(0,de),h.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),h.end()}K(v,G);let l=performance.now();n.queue.submit([v.finish()]),await n.queue.onSubmittedWorkDone(),i.push(performance.now()-l)}i.sort((d,v)=>d-v);let t=i.reduce((d,v)=>d+v,0)/i.length,a=i[Math.floor(i.length/2)],f=i[0];return{avgMs:t,fps:1e3/t,medianMs:a,minMs:f}}function xt(e){n.queue.copyExternalImageToTexture({source:e},{texture:wa},[256,256]);let s=n.createCommandEncoder();{let i=s.beginComputePass();i.setPipeline(Za),i.setBindGroup(0,ce),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}K(s,G),n.queue.submit([s.finish()])}return{device:n,run:J,runFromCanvas:ft,benchmark:bt,benchmarkGPU:gt,_device:n,_benchmarkSubmitOnly:xt}}var St="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Dt(o={}){let{weightsUrl:p,scoreThreshold:n=.5}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let m=p??St,c=m.endsWith("/")?`${m}weights.json`:`${m}/weights.json`,H=m.endsWith("/")?`${m}weights.bin`:`${m}/weights.bin`,[_,S]=await Promise.all([fetch(c),fetch(H)]);if(!_.ok)throw new Error(`Failed to fetch weights metadata: ${_.status}`);if(!S.ok)throw new Error(`Failed to fetch weights binary: ${S.status}`);let A=await _.json(),B=await S.arrayBuffer(),b=Ie(A,B),u=await ze(b),w=null;function y(){return w||(w=new OffscreenCanvas(256,256)),w}async function Q(x){if(x instanceof HTMLCanvasElement||x instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&x instanceof ImageBitmap)return x;let D=y();D.width=256,D.height=256;let k=D.getContext("2d");return x instanceof ImageData?k.putImageData(x,0,0):k.drawImage(x,0,0,256,256),D}function aa(x,D,k){let z=x[0];if(z<n)return null;let ia=D[0]>.5,F=[];for(let L=0;L<21;L++)F.push({x:k[L*3],y:k[L*3+1],z:k[L*3+2]});return{score:z,handedness:ia?"right":"left",landmarks:F}}async function ea(x){let D=await Q(x),k=await u.runFromCanvas(D);return aa(k.handflag,k.handedness,k.landmarks)}function ta(){u.device.destroy(),w=null}return{detect:ea,dispose:ta}}export{Dt as createHandpose};
