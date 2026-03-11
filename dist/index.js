function U(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ze=U(`
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
`),qe=U(`
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
`),Ne=U(`
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
`),$e=U(`
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
`);function Wa(u,d){return qe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Ga(u,d){return ze.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Aa(u,d){return Ne.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Ha(u,d){return $e.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${d},1)`)}function Ca(u,d){return[8,8]}var Ma=U(`
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
`),Ea=U(`
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
`);function La(u){return U(`
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
`)}var Ta=La(!1),Oa=La(!0),Ra=U(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Ia=U(`
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
`);function Fa(u){return U(`
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
`)}var za=Fa("sigmoid"),qa=Fa("div256"),Na=U(`
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
`),$a=U(`
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
`);function Xa(u,d){let f=u%4===0?`var ic:u32=0u;
  while(ic<${u}u){
    sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
    sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
    sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
    sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
    ic+=4u;
  }
  var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
  while(ic<${u}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
  var pw_sum=sum0+pw_bias[c];`,c=`var skip_val:f32=0.0;
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
  }`,G=u===d?"":`if(c<${u}u){`;return U(`
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
  ${G}
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
  ${f}
  // Skip connection (only for c < inCh)
  ${c}
  let result=max(0.0,pw_sum+skip_val);
  output[c*outH*outW+out_y*outW+out_x]=result;
}
`)}var Ya=U(`
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
`);function Va(u,d){let t=new Map,f=u.dtype??"float32";for(let c=0;c<u.keys.length;c++){let G=u.keys[c],w=u.shapes[c],W=u.offsets[c],A=w.reduce((v,o)=>v*o,1),H;if(f==="float32")H=new Float32Array(d,W,A);else{let v=new DataView(d);H=new Float32Array(A);for(let o=0;o<A;o++)H[o]=Wt(v.getUint16(W+o*2,!0))}t.set(G,{data:H,shape:w})}return t}function Wt(u){let d=u>>15&1,t=u>>10&31,f=u&1023;if(t===0){if(f===0)return d?-0:0;let w=-14,W=f/1024;return(d?-1:1)*Math.pow(2,w)*W}if(t===31)return f===0?d?-1/0:1/0:NaN;let c=t-15,G=1+f/1024;return(d?-1:1)*Math.pow(2,c)*G}var Gt=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Za=Gt.map(([u,d,t,f,c])=>({type:"resmodule",inCh:u,outCh:d,h:t,w:t,stride:f,prefix:c})),At=2,Ht=5,Ct=8,Mt=11;async function ja(u){if(!navigator.gpu)throw new Error("WebGPU not supported");let d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No GPU adapter found");let t=await d.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(d.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(d.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(d.limits.maxComputeInvocationsPerWorkgroup,288)}}),f={r:"read-only-storage",s:"storage",u:"uniform"};function c(a){return t.createBindGroupLayout({entries:a.map((r,n)=>({binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:f[r]}}))})}function G(a){return t.createBindGroupLayout({entries:a.map((r,n)=>r==="t"?{binding:n,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:f[r]}})})}let w=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,W=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.STORAGE,H=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,v=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function o(a,r){return t.createBuffer({size:a,usage:r})}function h(a,r){return t.createBindGroup({layout:a,entries:r.map((n,i)=>({binding:i,resource:"size"in n?{buffer:n}:n}))})}function S(a,r){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:r,entryPoint:"main"}})}let Y=t.createShaderModule({code:Ma}),Z=t.createShaderModule({code:Ya}),pe=t.createShaderModule({code:Na}),_e=t.createShaderModule({code:qe}),de=t.createShaderModule({code:ze}),we=t.createShaderModule({code:Ne}),le=t.createShaderModule({code:$e}),m=t.createShaderModule({code:Ea}),B=t.createShaderModule({code:Ta}),y=t.createShaderModule({code:Ra}),J=t.createShaderModule({code:Oa}),ce=t.createShaderModule({code:Ia}),Q=t.createShaderModule({code:za}),I=t.createShaderModule({code:qa}),Ka=t.createShaderModule({code:$a}),Xe=new Map;function Tt(a,r){let n=`${a}_${r}`,i=Xe.get(n);return i||(i=t.createShaderModule({code:Xa(a,r)}),Xe.set(n,i)),i}let me=c(["r","r","r","s","u"]),he=c(["r","r","r","r","s","u"]),Ye=c(["r","s","u"]),Ze=c(["r","r","r","s","u"]),Ja=c(["r","s","u"]),Qa=c(["r","r","s","u"]),ee=c(["r","r","s","u"]),Ve=c(["r","r","r","s","u"]),V=c(["r","r","r","s","u"]),je=G(["t","s","u"]),Ke=c(["r","r","r","r","r","r","r","s"]),be=c(["r","r","r","r","r","s","u"]),et=t.createPipelineLayout({bindGroupLayouts:[me]}),at=t.createPipelineLayout({bindGroupLayouts:[he]}),ae=a=>t.createComputePipeline({layout:et,compute:{module:a,entryPoint:"main"}}),te=a=>t.createComputePipeline({layout:at,compute:{module:a,entryPoint:"main"}}),tt=ae(_e),it=ae(de),nt=te(we),rt=te(le),Je=new Map,Qe=new Map,ea=new Map,aa=new Map;Je.set("8,8",tt),Qe.set("8,8",it),ea.set("8,8",nt),aa.set("8,8",rt);function ie(a,r,n,i,e){let b=`${r},${n}`,l=a.get(b);return l||(l=e(t.createShaderModule({code:i(r,n)})),a.set(b,l)),l}let st=(a,r)=>ie(Je,a,r,Wa,ae),ut=(a,r)=>ie(Qe,a,r,Ga,ae),ot=(a,r)=>ie(ea,a,r,Aa,te),pt=(a,r)=>ie(aa,a,r,Ha,te),F=Za.map(a=>{let r=a.stride===2?a.h/2:a.h,n=a.stride===2?a.w/2:a.w,[i,e]=Ca(a.inCh,r),b=a.h>=64,l=r>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:b?ut(i,e):st(i,e),pwPipeline:l?pt(i,e):ot(i,e),dwDispatchX:Math.ceil(n/i),dwDispatchY:Math.ceil(r/e),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(n/i),pwDispatchY:Math.ceil(r/e),pwDispatchZ:l?a.outCh/2:a.outCh}}),ta=S(Ye,Y),_t=S(Ze,m);S(Ja,B),S(Qa,y);let fe=S(ee,J),dt=S(Ve,ce);S(V,Q),S(V,I);let q=S(je,Z),wt=S(Ke,pe),lt=S(be,Ka),ge=1*288*128*128*4,ia=o(3*256*256*4,w),ye=o(3*257*257*4,A),na=o(12,v);t.queue.writeBuffer(na,0,new Uint32Array([3,256,257]));let P=o(ge,W),D=o(ge,H),ne=o(ge,A),ra=o(3072*64*4,w),sa=o(3072*32*4,w),ua=o(1536*16*4,w),oa=o(6144*64*4,A),j=o(260,H),g=o(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);o(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let z=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),pa=o(8,v);t.queue.writeBuffer(pa,0,new Uint32Array([256,257]));let xe=u.get("backbone1.1.weight")?.data,ve=u.get("backbone1.1.bias")?.data;if(!xe||!ve)throw new Error("Missing input conv weights");let _a=o(xe.byteLength,w),da=o(ve.byteLength,w),wa=o(28,v);t.queue.writeBuffer(_a,0,xe),t.queue.writeBuffer(da,0,ve),t.queue.writeBuffer(wa,0,new Uint32Array([1,3,24,257,257,128,128]));let Pe=u.get("backbone6.1.weight")?.data,ke=u.get("backbone6.1.bias")?.data;if(!Pe||!ke)throw new Error("Missing backbone6.1 conv1x1 weights");let la=o(Pe.byteLength,w),ca=o(ke.byteLength,w),ma=o(20,v);t.queue.writeBuffer(la,0,Pe),t.queue.writeBuffer(ca,0,ke),t.queue.writeBuffer(ma,0,new Uint32Array([1,96,48,32,32]));let Be=u.get("handflag.weight")?.data,Ue=u.get("handflag.bias")?.data;if(!Be||!Ue)throw new Error("Missing handflag weights");let Se=o(Be.byteLength,w),De=o(Ue.byteLength,w),ha=o(12,v);t.queue.writeBuffer(Se,0,Be),t.queue.writeBuffer(De,0,Ue),t.queue.writeBuffer(ha,0,new Uint32Array([1,288,1]));let We=u.get("handedness.weight")?.data,Ge=u.get("handedness.bias")?.data;if(!We||!Ge)throw new Error("Missing handedness weights");let Ae=o(We.byteLength,w),He=o(Ge.byteLength,w),ba=o(12,v);t.queue.writeBuffer(Ae,0,We),t.queue.writeBuffer(He,0,Ge),t.queue.writeBuffer(ba,0,new Uint32Array([1,288,1]));let Ce=u.get("reg_3d.weight")?.data,Me=u.get("reg_3d.bias")?.data;if(!Ce||!Me)throw new Error("Missing reg_3d weights");let Ee=o(Ce.byteLength,w),Le=o(Me.byteLength,w),fa=o(12,v);t.queue.writeBuffer(Ee,0,Ce),t.queue.writeBuffer(Le,0,Me),t.queue.writeBuffer(fa,0,new Uint32Array([1,288,63]));let K=Za.map(a=>{let{inCh:r,outCh:n,h:i,w:e,stride:b,prefix:l}=a,k=b===2?i/2:i,p=b===2?e/2:e,_=b===1?2:1,s=u.get(`${l}convs.0.weight`)?.data,x=u.get(`${l}convs.0.bias`)?.data,R=u.get(`${l}convs.1.weight`)?.data,Fe=u.get(`${l}convs.1.bias`)?.data;if(!s||!x||!R||!Fe)throw new Error(`Missing weights for ${l}`);let Pa=o(s.byteLength,w),ka=o(x.byteLength,w),Ba=o(R.byteLength,w),Ua=o(Fe.byteLength,w),Sa=o(32,v),Da=o(36,v);return t.queue.writeBuffer(Pa,0,s),t.queue.writeBuffer(ka,0,x),t.queue.writeBuffer(Ba,0,R),t.queue.writeBuffer(Ua,0,Fe),t.queue.writeBuffer(Sa,0,new Uint32Array([1,r,i,e,k,p,b,_])),t.queue.writeBuffer(Da,0,new Uint32Array([1,r,n,k,p,Math.max(0,n-r),b,i,e])),{dwWeight:Pa,dwBias:ka,pwWeight:Ba,pwBias:Ua,dwUniform:Sa,pwUniform:Da,spec:a,outH:k,outW:p}});function N(a){let r=o(a.length*4,v);return t.queue.writeBuffer(r,0,new Uint32Array(a)),r}let ct=N([1,96,8,8,16,16]),mt=N([1,96,16,16,32,32]),ht=N([1,48,32,32,64,64]);N([1536*16]),N([3072*32]),N([3072*64]);let ga=h(Ye,[ia,ye,na]),bt=h(Ze,[ye,_a,da,P,wa]),C=[],M=[],E=[],L=[];for(let a of K)C.push(h(me,[P,a.dwWeight,a.dwBias,ne,a.dwUniform])),M.push(h(he,[ne,P,a.pwWeight,a.pwBias,D,a.pwUniform])),E.push(h(me,[D,a.dwWeight,a.dwBias,ne,a.dwUniform])),L.push(h(he,[ne,D,a.pwWeight,a.pwBias,P,a.pwUniform]));let ft=h(ee,[P,ua,D,ct]),gt=h(ee,[P,sa,D,mt]),yt=h(Ve,[P,la,ca,oa,ma]),xt=h(ee,[oa,ra,D,ht]);h(V,[P,Se,De,j,ha]),h(V,[P,Ae,He,j,ba]),h(V,[P,Ee,Le,j,fa]);let $=h(je,[z.createView(),ye,pa]),vt=h(Ke,[P,Se,De,Ae,He,Ee,Le,j]),Te=24,ya=[],xa=[];for(let a=Te;a<K.length;a++){let r=K[a];ya.push(h(be,[P,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,D,r.dwUniform])),xa.push(h(be,[D,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,P,r.dwUniform]))}let re=new Float32Array(1),se=new Float32Array(1),ue=new Float32Array(63);function T(a,r){let n=!0,i=0,e=a.beginComputePass();for(e.setPipeline(_t),e.setBindGroup(0,bt),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=At;i++){let p=n?C[i]:E[i],_=n?M[i]:L[i],s=F[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.end();let b=n?P:D;for(a.copyBufferToBuffer(b,0,ra,0,3072*64*4),e=a.beginComputePass();i<=Ht;i++){let p=n?C[i]:E[i],_=n?M[i]:L[i],s=F[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.end();let l=n?P:D;for(a.copyBufferToBuffer(l,0,sa,0,3072*32*4),e=a.beginComputePass();i<=Ct;i++){let p=n?C[i]:E[i],_=n?M[i]:L[i],s=F[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.end();let k=n?P:D;for(a.copyBufferToBuffer(k,0,ua,0,1536*16*4),e=a.beginComputePass();i<=Mt;i++){let p=n?C[i]:E[i],_=n?M[i]:L[i],s=F[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.setPipeline(fe),e.setBindGroup(0,ft),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),n=!1,e=a.beginComputePass();{let p=n?C[i]:E[i],_=n?M[i]:L[i],s=F[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n,i++}e.setPipeline(fe),e.setBindGroup(0,gt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),n=!1,e=a.beginComputePass();{let p=n?C[i]:E[i],_=n?M[i]:L[i],s=F[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n,i++}for(e.setPipeline(dt),e.setBindGroup(0,yt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(fe),e.setBindGroup(0,xt),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),n=!1,e=a.beginComputePass();i<Te;i++){let p=n?C[i]:E[i],_=n?M[i]:L[i],s=F[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,_),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}for(;i<K.length;i++){let p=i-Te,_=n?ya[p]:xa[p],s=K[i];e.setPipeline(lt),e.setBindGroup(0,_),e.dispatchWorkgroups(s.outW,s.outH,1),n=!n}e.setPipeline(wt),e.setBindGroup(0,vt),e.dispatchWorkgroups(1),e.end(),a.copyBufferToBuffer(j,0,r,0,260)}async function oe(a){t.queue.writeBuffer(ia,0,a);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(ta),e.setBindGroup(0,ga),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}T(r,g),t.queue.submit([r.finish()]);let n=g.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(g.getMappedRange());return re[0]=i[0],se[0]=i[1],ue.set(i.subarray(2,65)),g.unmap(),{handflag:new Float32Array(re),handedness:new Float32Array(se),landmarks:new Float32Array(ue)}}async function Oe(a){t.queue.copyExternalImageToTexture({source:a},{texture:z},[256,256]);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(q),e.setBindGroup(0,$),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}T(r,g),t.queue.submit([r.finish()]);let n=g.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(g.getMappedRange());return re[0]=i[0],se[0]=i[1],ue.set(i.subarray(2,65)),g.unmap(),{handflag:new Float32Array(re),handedness:new Float32Array(se),landmarks:new Float32Array(ue)}}let Pt=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Re=0,kt=[g,Pt],X=null,O=null;async function Ie(a){let r=kt[Re];Re=1-Re,t.queue.copyExternalImageToTexture({source:a},{texture:z},[256,256]);let n=t.createCommandEncoder();{let e=n.beginComputePass();e.setPipeline(q),e.setBindGroup(0,$),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}T(n,r),t.queue.submit([n.finish()]);let i=null;if(X!==null&&O!==null){await X;let e=new Float32Array(O.getMappedRange());i={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},O.unmap()}return O=r,X=r.mapAsync(GPUMapMode.READ),i}async function va(){if(!X||!O)return null;await X;let a=new Float32Array(O.getMappedRange()),r={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return O.unmap(),X=null,O=null,r}async function Bt(a=50){let r=new Float32Array(196608);for(let e=0;e<5;e++)await oe(r);let n=[];for(let e=0;e<a;e++){let b=performance.now();await oe(r),n.push(performance.now()-b)}let i=n.reduce((e,b)=>e+b,0)/n.length;return{avgMs:i,fps:1e3/i}}async function Ut(a=50){let r=new Float32Array(196608);for(let l=0;l<5;l++)await oe(r);let n=[];for(let l=0;l<a;l++){let k=t.createCommandEncoder();{let _=k.beginComputePass();_.setPipeline(ta),_.setBindGroup(0,ga),_.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),_.end()}T(k,g);let p=performance.now();t.queue.submit([k.finish()]),await t.queue.onSubmittedWorkDone(),n.push(performance.now()-p)}n.sort((l,k)=>l-k);let i=n.reduce((l,k)=>l+k,0)/n.length,e=n[Math.floor(n.length/2)],b=n[0];return{avgMs:i,fps:1e3/i,medianMs:e,minMs:b}}function St(a){t.queue.copyExternalImageToTexture({source:a},{texture:z},[256,256]);let r=t.createCommandEncoder();{let n=r.beginComputePass();n.setPipeline(q),n.setBindGroup(0,$),n.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),n.end()}T(r,g),t.queue.submit([r.finish()])}async function Dt(a,r=50){function n(p){let _=[...p].sort((s,x)=>s-x);return{median:_[Math.floor(_.length/2)],min:_[0]}}for(let p=0;p<10;p++)await Oe(a);let i=[];for(let p=0;p<r;p++){t.queue.copyExternalImageToTexture({source:a},{texture:z},[256,256]);let _=t.createCommandEncoder();{let x=_.beginComputePass();x.setPipeline(q),x.setBindGroup(0,$),x.dispatchWorkgroups(16,16,1),x.end()}T(_,g);let s=performance.now();t.queue.submit([_.finish()]),await t.queue.onSubmittedWorkDone(),i.push(performance.now()-s)}let e=[];for(let p=0;p<r;p++){t.queue.copyExternalImageToTexture({source:a},{texture:z},[256,256]);let _=t.createCommandEncoder();{let R=_.beginComputePass();R.setPipeline(q),R.setBindGroup(0,$),R.dispatchWorkgroups(16,16,1),R.end()}T(_,g),t.queue.submit([_.finish()]);let s=g.mapAsync(GPUMapMode.READ),x=performance.now();await t.queue.onSubmittedWorkDone(),await s,g.getMappedRange(),g.unmap(),e.push(performance.now()-x)}let b=[];for(let p=0;p<r;p++){t.queue.copyExternalImageToTexture({source:a},{texture:z},[256,256]);let _=t.createCommandEncoder();{let x=_.beginComputePass();x.setPipeline(q),x.setBindGroup(0,$),x.dispatchWorkgroups(16,16,1),x.end()}T(_,g),t.queue.submit([_.finish()]);let s=performance.now();await g.mapAsync(GPUMapMode.READ),g.getMappedRange(),g.unmap(),b.push(performance.now()-s)}let l=[];for(let p=0;p<r;p++){let _=performance.now();await Oe(a),l.push(performance.now()-_)}await Ie(a);let k=[];for(let p=0;p<r;p++){let _=performance.now();await Ie(a),k.push(performance.now()-_)}return await va(),{gpuOnly:n(i),mapAsyncOnly:n(e),mapAsyncNoWait:n(b),total:n(l),pipelined:n(k)}}return{device:t,run:oe,runFromCanvas:Oe,runFromCanvasPipelined:Ie,flushPipelined:va,benchmark:Bt,benchmarkGPU:Ut,benchmarkDiagnostic:Dt,_device:t,_benchmarkSubmitOnly:St}}var Et="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Lt(u={}){let{weightsUrl:d,scoreThreshold:t=.5}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let f=d??Et,c=f.endsWith("/")?`${f}weights.json`:`${f}/weights.json`,G=f.endsWith("/")?`${f}weights.bin`:`${f}/weights.bin`,[w,W]=await Promise.all([fetch(c),fetch(G)]);if(!w.ok)throw new Error(`Failed to fetch weights metadata: ${w.status}`);if(!W.ok)throw new Error(`Failed to fetch weights binary: ${W.status}`);let A=await w.json(),H=await W.arrayBuffer(),v=Va(A,H),o=await ja(v),h=null;function S(){return h||(h=new OffscreenCanvas(256,256)),h}async function Y(m){if(m instanceof HTMLCanvasElement||m instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&m instanceof ImageBitmap)return m;let B=S();B.width=256,B.height=256;let y=B.getContext("2d");return m instanceof ImageData?y.putImageData(m,0,0):y.drawImage(m,0,0,256,256),B}function Z(m,B,y){let J=m[0];if(J<t)return null;let ce=B[0]>.5,Q=[];for(let I=0;I<21;I++)Q.push({x:y[I*3],y:y[I*3+1],z:y[I*3+2]});return{score:J,handedness:ce?"right":"left",landmarks:Q}}async function pe(m){let B=await Y(m),y=await o.runFromCanvas(B);return Z(y.handflag,y.handedness,y.landmarks)}async function _e(m){let B=await Y(m),y=await o.runFromCanvasPipelined(B);return y?Z(y.handflag,y.handedness,y.landmarks):null}async function de(){let m=await o.flushPipelined();return m?Z(m.handflag,m.handedness,m.landmarks):null}function we(){o.device.destroy(),h=null}async function le(m){let B=await Y(m);return o.benchmarkDiagnostic(B)}return{detect:pe,detectPipelined:_e,flushPipelined:de,dispose:we,benchmarkDiagnostic:le}}export{Lt as createHandpose};
