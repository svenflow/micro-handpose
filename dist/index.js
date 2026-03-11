function k(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Aa=k(`
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
`),Ca=k(`
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
`),Ea=k(`
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
`),Ma=k(`
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
`);function me(o,d){return Ca.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${d},1)`)}function fe(o,d){return Aa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${d},1)`)}function ge(o,d){return Ea.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${d},1)`)}function be(o,d){return Ma.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${d},1)`)}function ye(o,d){return[8,8]}var xe=k(`
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
`),ve=k(`
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
`);function Pe(o){return k(`
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
`)}var ke=Pe(!1),Be=Pe(!0),Ue=k(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),De=k(`
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
`);function Se(o){return k(`
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
`)}var Ge=Se("sigmoid"),Ae=Se("div256"),Ce=k(`
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
`),Ee=k(`
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
`);function Le(o,d){let i=new Map,m=o.dtype??"float32";for(let c=0;c<o.keys.length;c++){let H=o.keys[c],p=o.shapes[c],U=o.offsets[c],S=p.reduce((g,u)=>g*u,1),B;if(m==="float32")B=new Float32Array(d,U,S);else{let g=new DataView(d);B=new Float32Array(S);for(let u=0;u<S;u++)B[u]=mt(g.getUint16(U+u*2,!0))}i.set(H,{data:B,shape:p})}return i}function mt(o){let d=o>>15&1,i=o>>10&31,m=o&1023;if(i===0){if(m===0)return d?-0:0;let p=-14,U=m/1024;return(d?-1:1)*Math.pow(2,p)*U}if(i===31)return m===0?d?-1/0:1/0:NaN;let c=i-15,H=1+m/1024;return(d?-1:1)*Math.pow(2,c)*H}var ft=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Me=ft.map(([o,d,i,m,c])=>({type:"resmodule",inCh:o,outCh:d,h:i,w:i,stride:m,prefix:c})),gt=2,bt=5,yt=8,xt=11;async function He(o){if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");let i=await d.requestDevice(),m={r:"read-only-storage",s:"storage",u:"uniform"};function c(e){return i.createBindGroupLayout({entries:e.map((s,n)=>({binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:m[s]}}))})}function H(e){return i.createBindGroupLayout({entries:e.map((s,n)=>s==="t"?{binding:n,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:m[s]}})})}let p=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,U=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,S=GPUBufferUsage.STORAGE,B=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,g=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function u(e,s){return i.createBuffer({size:e,usage:s})}function h(e,s){return i.createBindGroup({layout:e,entries:s.map((n,t)=>({binding:t,resource:"size"in n?{buffer:n}:n}))})}function v(e,s){return i.createComputePipeline({layout:i.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:s,entryPoint:"main"}})}let K=i.createShaderModule({code:xe}),J=i.createShaderModule({code:Ee}),Q=i.createShaderModule({code:Ce}),aa=i.createShaderModule({code:Ca}),b=i.createShaderModule({code:Aa}),D=i.createShaderModule({code:Ea}),P=i.createShaderModule({code:Ma}),I=i.createShaderModule({code:ve}),ea=i.createShaderModule({code:ke}),z=i.createShaderModule({code:Ue}),T=i.createShaderModule({code:Be}),Te=i.createShaderModule({code:De}),Re=i.createShaderModule({code:Ge}),Oe=i.createShaderModule({code:Ae}),ta=c(["r","r","r","s","u"]),ia=c(["r","r","r","r","s","u"]),La=c(["r","s","u"]),Ha=c(["r","r","r","s","u"]),We=c(["r","s","u"]),Ie=c(["r","r","s","u"]),F=c(["r","r","s","u"]),Ta=c(["r","r","r","s","u"]),W=c(["r","r","r","s","u"]),Ra=H(["t","s","u"]),Oa=c(["r","r","r","r","r","r","r","s","s","s"]),ze=i.createPipelineLayout({bindGroupLayouts:[ta]}),Fe=i.createPipelineLayout({bindGroupLayouts:[ia]}),N=e=>i.createComputePipeline({layout:ze,compute:{module:e,entryPoint:"main"}}),q=e=>i.createComputePipeline({layout:Fe,compute:{module:e,entryPoint:"main"}}),Ne=N(aa),qe=N(b),Ye=q(D),Xe=q(P),Wa=new Map,Ia=new Map,za=new Map,Fa=new Map;Wa.set("8,8",Ne),Ia.set("8,8",qe),za.set("8,8",Ye),Fa.set("8,8",Xe);function Y(e,s,n,t,a){let f=`${s},${n}`,_=e.get(f);return _||(_=a(i.createShaderModule({code:t(s,n)})),e.set(f,_)),_}let $e=(e,s)=>Y(Wa,e,s,me,N),Ze=(e,s)=>Y(Ia,e,s,fe,N),Ve=(e,s)=>Y(za,e,s,ge,q),je=(e,s)=>Y(Fa,e,s,be,q),R=Me.map(e=>{let s=e.stride===2?e.h/2:e.h,n=e.stride===2?e.w/2:e.w,[t,a]=ye(e.inCh,s),f=e.h>=64,_=s>=16&&e.inCh>=288&&e.outCh>=288&&e.outCh%2===0;return{dwPipeline:f?Ze(t,a):$e(t,a),pwPipeline:_?je(t,a):Ve(t,a),dwDispatchX:Math.ceil(n/t),dwDispatchY:Math.ceil(s/a),dwDispatchZ:e.inCh,pwDispatchX:Math.ceil(n/t),pwDispatchY:Math.ceil(s/a),pwDispatchZ:_?e.outCh/2:e.outCh}}),Na=v(La,K),Ke=v(Ha,I);v(We,ea),v(Ie,z);let na=v(F,T),Je=v(Ta,Te),qa=v(W,Re),Qe=v(W,Oe),at=v(Ra,J);v(Oa,Q);let ra=1*288*128*128*4,Ya=u(3*256*256*4,p),sa=u(3*257*257*4,S),Xa=u(12,g);i.queue.writeBuffer(Xa,0,new Uint32Array([3,256,257]));let y=u(ra,U),G=u(ra,B),X=u(ra,S),$a=u(3072*64*4,p),Za=u(3072*32*4,p),Va=u(1536*16*4,p),ja=u(6144*64*4,S),ua=u(4,B),oa=u(4,B),pa=u(252,B),A=u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let Ka=i.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ja=u(8,g);i.queue.writeBuffer(Ja,0,new Uint32Array([256,257]));let da=o.get("backbone1.1.weight")?.data,_a=o.get("backbone1.1.bias")?.data;if(!da||!_a)throw new Error("Missing input conv weights");let Qa=u(da.byteLength,p),ae=u(_a.byteLength,p),ee=u(28,g);i.queue.writeBuffer(Qa,0,da),i.queue.writeBuffer(ae,0,_a),i.queue.writeBuffer(ee,0,new Uint32Array([1,3,24,257,257,128,128]));let ca=o.get("backbone6.1.weight")?.data,ha=o.get("backbone6.1.bias")?.data;if(!ca||!ha)throw new Error("Missing backbone6.1 conv1x1 weights");let te=u(ca.byteLength,p),ie=u(ha.byteLength,p),ne=u(20,g);i.queue.writeBuffer(te,0,ca),i.queue.writeBuffer(ie,0,ha),i.queue.writeBuffer(ne,0,new Uint32Array([1,96,48,32,32]));let la=o.get("handflag.weight")?.data,wa=o.get("handflag.bias")?.data;if(!la||!wa)throw new Error("Missing handflag weights");let ma=u(la.byteLength,p),fa=u(wa.byteLength,p),re=u(12,g);i.queue.writeBuffer(ma,0,la),i.queue.writeBuffer(fa,0,wa),i.queue.writeBuffer(re,0,new Uint32Array([1,288,1]));let ga=o.get("handedness.weight")?.data,ba=o.get("handedness.bias")?.data;if(!ga||!ba)throw new Error("Missing handedness weights");let ya=u(ga.byteLength,p),xa=u(ba.byteLength,p),se=u(12,g);i.queue.writeBuffer(ya,0,ga),i.queue.writeBuffer(xa,0,ba),i.queue.writeBuffer(se,0,new Uint32Array([1,288,1]));let va=o.get("reg_3d.weight")?.data,Pa=o.get("reg_3d.bias")?.data;if(!va||!Pa)throw new Error("Missing reg_3d weights");let ka=u(va.byteLength,p),Ba=u(Pa.byteLength,p),ue=u(12,g);i.queue.writeBuffer(ka,0,va),i.queue.writeBuffer(Ba,0,Pa),i.queue.writeBuffer(ue,0,new Uint32Array([1,288,63]));let oe=Me.map(e=>{let{inCh:s,outCh:n,h:t,w:a,stride:f,prefix:_}=e,x=f===2?t/2:t,w=f===2?a/2:a,l=f===1?2:1,r=o.get(`${_}convs.0.weight`)?.data,Da=o.get(`${_}convs.0.bias`)?.data,Sa=o.get(`${_}convs.1.weight`)?.data,Ga=o.get(`${_}convs.1.bias`)?.data;if(!r||!Da||!Sa||!Ga)throw new Error(`Missing weights for ${_}`);let de=u(r.byteLength,p),_e=u(Da.byteLength,p),ce=u(Sa.byteLength,p),he=u(Ga.byteLength,p),le=u(32,g),we=u(36,g);return i.queue.writeBuffer(de,0,r),i.queue.writeBuffer(_e,0,Da),i.queue.writeBuffer(ce,0,Sa),i.queue.writeBuffer(he,0,Ga),i.queue.writeBuffer(le,0,new Uint32Array([1,s,t,a,x,w,f,l])),i.queue.writeBuffer(we,0,new Uint32Array([1,s,n,x,w,Math.max(0,n-s),f,t,a])),{dwWeight:de,dwBias:_e,pwWeight:ce,pwBias:he,dwUniform:le,pwUniform:we,spec:e,outH:x,outW:w}});function O(e){let s=u(e.length*4,g);return i.queue.writeBuffer(s,0,new Uint32Array(e)),s}let et=O([1,96,8,8,16,16]),tt=O([1,96,16,16,32,32]),it=O([1,48,32,32,64,64]);O([1536*16]),O([3072*32]),O([3072*64]);let pe=h(La,[Ya,sa,Xa]),nt=h(Ha,[sa,Qa,ae,y,ee]),C=[],E=[],M=[],L=[];for(let e of oe)C.push(h(ta,[y,e.dwWeight,e.dwBias,X,e.dwUniform])),E.push(h(ia,[X,y,e.pwWeight,e.pwBias,G,e.pwUniform])),M.push(h(ta,[G,e.dwWeight,e.dwBias,X,e.dwUniform])),L.push(h(ia,[X,G,e.pwWeight,e.pwBias,y,e.pwUniform]));let rt=h(F,[y,Va,G,et]),st=h(F,[y,Za,G,tt]),ut=h(Ta,[y,te,ie,ja,ne]),ot=h(F,[ja,$a,G,it]),pt=h(W,[y,ma,fa,ua,re]),dt=h(W,[y,ya,xa,oa,se]),_t=h(W,[y,ka,Ba,pa,ue]),ct=h(Ra,[Ka.createView(),sa,Ja]);h(Oa,[y,ma,fa,ya,xa,ka,Ba,ua,oa,pa]);let $=new Float32Array(1),Z=new Float32Array(1),V=new Float32Array(63);function Ua(e,s){let n=!0,t=0,a=e.beginComputePass();for(a.setPipeline(Ke),a.setBindGroup(0,nt),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);t<=gt;t++){let w=n?C[t]:M[t],l=n?E[t]:L[t],r=R[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}a.end();let f=n?y:G;for(e.copyBufferToBuffer(f,0,$a,0,3072*64*4),a=e.beginComputePass();t<=bt;t++){let w=n?C[t]:M[t],l=n?E[t]:L[t],r=R[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}a.end();let _=n?y:G;for(e.copyBufferToBuffer(_,0,Za,0,3072*32*4),a=e.beginComputePass();t<=yt;t++){let w=n?C[t]:M[t],l=n?E[t]:L[t],r=R[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}a.end();let x=n?y:G;for(e.copyBufferToBuffer(x,0,Va,0,1536*16*4),a=e.beginComputePass();t<=xt;t++){let w=n?C[t]:M[t],l=n?E[t]:L[t],r=R[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}a.setPipeline(na),a.setBindGroup(0,rt),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),n=!1,a=e.beginComputePass();{let w=n?C[t]:M[t],l=n?E[t]:L[t],r=R[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,t++}a.setPipeline(na),a.setBindGroup(0,st),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),n=!1,a=e.beginComputePass();{let w=n?C[t]:M[t],l=n?E[t]:L[t],r=R[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n,t++}for(a.setPipeline(Je),a.setBindGroup(0,ut),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(na),a.setBindGroup(0,ot),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),n=!1,a=e.beginComputePass();t<oe.length;t++){let w=n?C[t]:M[t],l=n?E[t]:L[t],r=R[t];a.setPipeline(r.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),a.setPipeline(r.pwPipeline),a.setBindGroup(0,l),a.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),n=!n}a.setPipeline(qa),a.setBindGroup(0,pt),a.dispatchWorkgroups(1),a.setPipeline(qa),a.setBindGroup(0,dt),a.dispatchWorkgroups(1),a.setPipeline(Qe),a.setBindGroup(0,_t),a.dispatchWorkgroups(1),a.end(),e.copyBufferToBuffer(ua,0,s,0,4),e.copyBufferToBuffer(oa,0,s,4,4),e.copyBufferToBuffer(pa,0,s,8,252)}async function j(e){i.queue.writeBuffer(Ya,0,e);let s=i.createCommandEncoder();{let t=s.beginComputePass();t.setPipeline(Na),t.setBindGroup(0,pe),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}Ua(s,A),i.queue.submit([s.finish()]),await A.mapAsync(GPUMapMode.READ);let n=new Float32Array(A.getMappedRange());return $[0]=n[0],Z[0]=n[1],V.set(n.subarray(2,65)),A.unmap(),{handflag:new Float32Array($),handedness:new Float32Array(Z),landmarks:new Float32Array(V)}}async function ht(e){i.queue.copyExternalImageToTexture({source:e},{texture:Ka},[256,256]);let s=i.createCommandEncoder();{let t=s.beginComputePass();t.setPipeline(at),t.setBindGroup(0,ct),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Ua(s,A),i.queue.submit([s.finish()]),await A.mapAsync(GPUMapMode.READ);let n=new Float32Array(A.getMappedRange());return $[0]=n[0],Z[0]=n[1],V.set(n.subarray(2,65)),A.unmap(),{handflag:new Float32Array($),handedness:new Float32Array(Z),landmarks:new Float32Array(V)}}async function lt(e=50){let s=new Float32Array(196608);for(let a=0;a<5;a++)await j(s);let n=[];for(let a=0;a<e;a++){let f=performance.now();await j(s),n.push(performance.now()-f)}let t=n.reduce((a,f)=>a+f,0)/n.length;return{avgMs:t,fps:1e3/t}}async function wt(e=50){let s=new Float32Array(196608);for(let _=0;_<5;_++)await j(s);let n=[];for(let _=0;_<e;_++){let x=i.createCommandEncoder();{let l=x.beginComputePass();l.setPipeline(Na),l.setBindGroup(0,pe),l.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),l.end()}Ua(x,A);let w=performance.now();i.queue.submit([x.finish()]),await i.queue.onSubmittedWorkDone(),n.push(performance.now()-w)}n.sort((_,x)=>_-x);let t=n.reduce((_,x)=>_+x,0)/n.length,a=n[Math.floor(n.length/2)],f=n[0];return{avgMs:t,fps:1e3/t,medianMs:a,minMs:f}}return{device:i,run:j,runFromCanvas:ht,benchmark:lt,benchmarkGPU:wt}}var vt="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Pt(o={}){let{weightsUrl:d,scoreThreshold:i=.5}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let m=d??vt,c=m.endsWith("/")?`${m}weights.json`:`${m}/weights.json`,H=m.endsWith("/")?`${m}weights.bin`:`${m}/weights.bin`,[p,U]=await Promise.all([fetch(c),fetch(H)]);if(!p.ok)throw new Error(`Failed to fetch weights metadata: ${p.status}`);if(!U.ok)throw new Error(`Failed to fetch weights binary: ${U.status}`);let S=await p.json(),B=await U.arrayBuffer(),g=Le(S,B),u=await He(g),h=null;function v(){return h||(h=new OffscreenCanvas(256,256)),h}async function K(b){if(b instanceof HTMLCanvasElement||b instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&b instanceof ImageBitmap)return b;let D=v();D.width=256,D.height=256;let P=D.getContext("2d");return b instanceof ImageData?P.putImageData(b,0,0):P.drawImage(b,0,0,256,256),D}function J(b,D,P){let I=b[0];if(I<i)return null;let ea=D[0]>.5,z=[];for(let T=0;T<21;T++)z.push({x:P[T*3],y:P[T*3+1],z:P[T*3+2]});return{score:I,handedness:ea?"right":"left",landmarks:z}}async function Q(b){let D=await K(b),P=await u.runFromCanvas(D);return J(P.handflag,P.handedness,P.landmarks)}function aa(){u.device.destroy(),h=null}return{detect:Q,dispose:aa}}export{Pt as createHandpose};
