function v(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Oe=v(`
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
`),Ie=v(`
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
`),Fe=v(`
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
`),ze=v(`
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
`);function Ba(u,p){return Ie.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function Ua(u,p){return Oe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function Sa(u,p){return Fe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function Da(u,p){return ze.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${u},${p},1)`)}function Wa(u,p){return[8,8]}var Ga=v(`
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
`),Ha=v(`
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
`);function Aa(u){return v(`
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
`)}var Ca=Aa(!1),Ma=Aa(!0),Ea=v(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),La=v(`
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
`);function Ta(u){return v(`
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
`)}var Ra=Ta("sigmoid"),Oa=Ta("div256"),Ia=v(`
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
`),Fa=v(`
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
`);function za(u,p){let b=u%4===0?`var ic:u32=0u;
  while(ic<${u}u){
    sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
    sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
    sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
    sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
    ic+=4u;
  }
  var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
  while(ic<${u}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
  var pw_sum=sum0+pw_bias[c];`,d=`var skip_val:f32=0.0;
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
  }`,W=u===p?"":`if(c<${u}u){`;return v(`
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
@compute @workgroup_size(${p},1,1)
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
  ${W}
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
  ${u===p?"":"}"}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  let pw_base=c*${u}u;
  var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
  ${b}
  // Skip connection (only for c < inCh)
  ${d}
  let result=max(0.0,pw_sum+skip_val);
  output[c*outH*outW+out_y*outW+out_x]=result;
}
`)}var Na=v(`
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
`);function $a(u,p){let t=new Map,b=u.dtype??"float32";for(let d=0;d<u.keys.length;d++){let W=u.keys[d],_=u.shapes[d],D=u.offsets[d],G=_.reduce((y,o)=>y*o,1),H;if(b==="float32")H=new Float32Array(p,D,G);else{let y=new DataView(p);H=new Float32Array(G);for(let o=0;o<G;o++)H[o]=Di(y.getUint16(D+o*2,!0))}t.set(W,{data:H,shape:_})}return t}function Di(u){let p=u>>15&1,t=u>>10&31,b=u&1023;if(t===0){if(b===0)return p?-0:0;let _=-14,D=b/1024;return(p?-1:1)*Math.pow(2,_)*D}if(t===31)return b===0?p?-1/0:1/0:NaN;let d=t-15,W=1+b/1024;return(p?-1:1)*Math.pow(2,d)*W}var Wi=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],qa=Wi.map(([u,p,t,b,d])=>({type:"resmodule",inCh:u,outCh:p,h:t,w:t,stride:b,prefix:d})),Gi=2,Hi=5,Ai=8,Ci=11;async function Xa(u){if(!navigator.gpu)throw new Error("WebGPU not supported");let p=await navigator.gpu.requestAdapter();if(!p)throw new Error("No GPU adapter found");let t=await p.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(p.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(p.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(p.limits.maxComputeInvocationsPerWorkgroup,288)}}),b={r:"read-only-storage",s:"storage",u:"uniform"};function d(a){return t.createBindGroupLayout({entries:a.map((r,n)=>({binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:b[r]}}))})}function W(a){return t.createBindGroupLayout({entries:a.map((r,n)=>r==="t"?{binding:n,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:n,visibility:GPUShaderStage.COMPUTE,buffer:{type:b[r]}})})}let _=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,D=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,G=GPUBufferUsage.STORAGE,H=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function o(a,r){return t.createBuffer({size:a,usage:r})}function l(a,r){return t.createBindGroup({layout:a,entries:r.map((n,i)=>({binding:i,resource:"size"in n?{buffer:n}:n}))})}function P(a,r){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:r,entryPoint:"main"}})}let X=t.createShaderModule({code:Ga}),F=t.createShaderModule({code:Na}),re=t.createShaderModule({code:Ia}),se=t.createShaderModule({code:Ie}),ue=t.createShaderModule({code:Oe}),oe=t.createShaderModule({code:Fe}),m=t.createShaderModule({code:ze}),B=t.createShaderModule({code:Ha}),g=t.createShaderModule({code:Ca}),Y=t.createShaderModule({code:Ea}),pe=t.createShaderModule({code:Ma}),Z=t.createShaderModule({code:La}),T=t.createShaderModule({code:Ra}),Ya=t.createShaderModule({code:Oa}),Za=t.createShaderModule({code:Fa}),Ne=new Map;function Li(a,r){let n=`${a}_${r}`,i=Ne.get(n);return i||(i=t.createShaderModule({code:za(a,r)}),Ne.set(n,i)),i}let _e=d(["r","r","r","s","u"]),de=d(["r","r","r","r","s","u"]),qe=d(["r","s","u"]),$e=d(["r","r","r","s","u"]),Va=d(["r","s","u"]),ja=d(["r","r","s","u"]),V=d(["r","r","s","u"]),Xe=d(["r","r","r","s","u"]),z=d(["r","r","r","s","u"]),Ye=W(["t","s","u"]),Ze=d(["r","r","r","r","r","r","r","s"]),we=d(["r","r","r","r","r","s","u"]),Ka=t.createPipelineLayout({bindGroupLayouts:[_e]}),Ja=t.createPipelineLayout({bindGroupLayouts:[de]}),j=a=>t.createComputePipeline({layout:Ka,compute:{module:a,entryPoint:"main"}}),K=a=>t.createComputePipeline({layout:Ja,compute:{module:a,entryPoint:"main"}}),Qa=j(se),ei=j(ue),ai=K(oe),ii=K(m),Ve=new Map,je=new Map,Ke=new Map,Je=new Map;Ve.set("8,8",Qa),je.set("8,8",ei),Ke.set("8,8",ai),Je.set("8,8",ii);function J(a,r,n,i,e){let f=`${r},${n}`,w=a.get(f);return w||(w=e(t.createShaderModule({code:i(r,n)})),a.set(f,w)),w}let ti=(a,r)=>J(Ve,a,r,Ba,j),ni=(a,r)=>J(je,a,r,Ua,j),ri=(a,r)=>J(Ke,a,r,Sa,K),si=(a,r)=>J(Je,a,r,Da,K),R=qa.map(a=>{let r=a.stride===2?a.h/2:a.h,n=a.stride===2?a.w/2:a.w,[i,e]=Wa(a.inCh,r),f=a.h>=64,w=r>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:f?ni(i,e):ti(i,e),pwPipeline:w?si(i,e):ri(i,e),dwDispatchX:Math.ceil(n/i),dwDispatchY:Math.ceil(r/e),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(n/i),pwDispatchY:Math.ceil(r/e),pwDispatchZ:w?a.outCh/2:a.outCh}}),Qe=P(qe,X),ui=P($e,B);P(Va,g),P(ja,Y);let le=P(V,pe),oi=P(Xe,Z);P(z,T),P(z,Ya);let ce=P(Ye,F),pi=P(Ze,re),_i=P(we,Za),he=1*288*128*128*4,ea=o(3*256*256*4,_),me=o(3*257*257*4,G),aa=o(12,y);t.queue.writeBuffer(aa,0,new Uint32Array([3,256,257]));let x=o(he,D),U=o(he,H),Q=o(he,G),ia=o(3072*64*4,_),ta=o(3072*32*4,_),na=o(1536*16*4,_),ra=o(6144*64*4,G),N=o(260,H),S=o(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);o(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let ee=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),sa=o(8,y);t.queue.writeBuffer(sa,0,new Uint32Array([256,257]));let be=u.get("backbone1.1.weight")?.data,fe=u.get("backbone1.1.bias")?.data;if(!be||!fe)throw new Error("Missing input conv weights");let ua=o(be.byteLength,_),oa=o(fe.byteLength,_),pa=o(28,y);t.queue.writeBuffer(ua,0,be),t.queue.writeBuffer(oa,0,fe),t.queue.writeBuffer(pa,0,new Uint32Array([1,3,24,257,257,128,128]));let ge=u.get("backbone6.1.weight")?.data,ye=u.get("backbone6.1.bias")?.data;if(!ge||!ye)throw new Error("Missing backbone6.1 conv1x1 weights");let _a=o(ge.byteLength,_),da=o(ye.byteLength,_),wa=o(20,y);t.queue.writeBuffer(_a,0,ge),t.queue.writeBuffer(da,0,ye),t.queue.writeBuffer(wa,0,new Uint32Array([1,96,48,32,32]));let xe=u.get("handflag.weight")?.data,ve=u.get("handflag.bias")?.data;if(!xe||!ve)throw new Error("Missing handflag weights");let Pe=o(xe.byteLength,_),ke=o(ve.byteLength,_),la=o(12,y);t.queue.writeBuffer(Pe,0,xe),t.queue.writeBuffer(ke,0,ve),t.queue.writeBuffer(la,0,new Uint32Array([1,288,1]));let Be=u.get("handedness.weight")?.data,Ue=u.get("handedness.bias")?.data;if(!Be||!Ue)throw new Error("Missing handedness weights");let Se=o(Be.byteLength,_),De=o(Ue.byteLength,_),ca=o(12,y);t.queue.writeBuffer(Se,0,Be),t.queue.writeBuffer(De,0,Ue),t.queue.writeBuffer(ca,0,new Uint32Array([1,288,1]));let We=u.get("reg_3d.weight")?.data,Ge=u.get("reg_3d.bias")?.data;if(!We||!Ge)throw new Error("Missing reg_3d weights");let He=o(We.byteLength,_),Ae=o(Ge.byteLength,_),ha=o(12,y);t.queue.writeBuffer(He,0,We),t.queue.writeBuffer(Ae,0,Ge),t.queue.writeBuffer(ha,0,new Uint32Array([1,288,63]));let q=qa.map(a=>{let{inCh:r,outCh:n,h:i,w:e,stride:f,prefix:w}=a,k=f===2?i/2:i,c=f===2?e/2:e,h=f===1?2:1,s=u.get(`${w}convs.0.weight`)?.data,Le=u.get(`${w}convs.0.bias`)?.data,Te=u.get(`${w}convs.1.weight`)?.data,Re=u.get(`${w}convs.1.bias`)?.data;if(!s||!Le||!Te||!Re)throw new Error(`Missing weights for ${w}`);let ga=o(s.byteLength,_),ya=o(Le.byteLength,_),xa=o(Te.byteLength,_),va=o(Re.byteLength,_),Pa=o(32,y),ka=o(36,y);return t.queue.writeBuffer(ga,0,s),t.queue.writeBuffer(ya,0,Le),t.queue.writeBuffer(xa,0,Te),t.queue.writeBuffer(va,0,Re),t.queue.writeBuffer(Pa,0,new Uint32Array([1,r,i,e,k,c,f,h])),t.queue.writeBuffer(ka,0,new Uint32Array([1,r,n,k,c,Math.max(0,n-r),f,i,e])),{dwWeight:ga,dwBias:ya,pwWeight:xa,pwBias:va,dwUniform:Pa,pwUniform:ka,spec:a,outH:k,outW:c}});function O(a){let r=o(a.length*4,y);return t.queue.writeBuffer(r,0,new Uint32Array(a)),r}let di=O([1,96,8,8,16,16]),wi=O([1,96,16,16,32,32]),li=O([1,48,32,32,64,64]);O([1536*16]),O([3072*32]),O([3072*64]);let ma=l(qe,[ea,me,aa]),ci=l($e,[me,ua,oa,x,pa]),A=[],C=[],M=[],E=[];for(let a of q)A.push(l(_e,[x,a.dwWeight,a.dwBias,Q,a.dwUniform])),C.push(l(de,[Q,x,a.pwWeight,a.pwBias,U,a.pwUniform])),M.push(l(_e,[U,a.dwWeight,a.dwBias,Q,a.dwUniform])),E.push(l(de,[Q,U,a.pwWeight,a.pwBias,x,a.pwUniform]));let hi=l(V,[x,na,U,di]),mi=l(V,[x,ta,U,wi]),bi=l(Xe,[x,_a,da,ra,wa]),fi=l(V,[ra,ia,U,li]);l(z,[x,Pe,ke,N,la]),l(z,[x,Se,De,N,ca]),l(z,[x,He,Ae,N,ha]);let Ce=l(Ye,[ee.createView(),me,sa]),gi=l(Ze,[x,Pe,ke,Se,De,He,Ae,N]),Me=24,ba=[],fa=[];for(let a=Me;a<q.length;a++){let r=q[a];ba.push(l(we,[x,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,U,r.dwUniform])),fa.push(l(we,[U,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,x,r.dwUniform]))}let ae=new Float32Array(1),ie=new Float32Array(1),te=new Float32Array(63);function $(a,r){let n=!0,i=0,e=a.beginComputePass();for(e.setPipeline(ui),e.setBindGroup(0,ci),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=Gi;i++){let c=n?A[i]:M[i],h=n?C[i]:E[i],s=R[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,c),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,h),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.end();let f=n?x:U;for(a.copyBufferToBuffer(f,0,ia,0,3072*64*4),e=a.beginComputePass();i<=Hi;i++){let c=n?A[i]:M[i],h=n?C[i]:E[i],s=R[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,c),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,h),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.end();let w=n?x:U;for(a.copyBufferToBuffer(w,0,ta,0,3072*32*4),e=a.beginComputePass();i<=Ai;i++){let c=n?A[i]:M[i],h=n?C[i]:E[i],s=R[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,c),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,h),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.end();let k=n?x:U;for(a.copyBufferToBuffer(k,0,na,0,1536*16*4),e=a.beginComputePass();i<=Ci;i++){let c=n?A[i]:M[i],h=n?C[i]:E[i],s=R[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,c),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,h),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}e.setPipeline(le),e.setBindGroup(0,hi),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),n=!1,e=a.beginComputePass();{let c=n?A[i]:M[i],h=n?C[i]:E[i],s=R[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,c),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,h),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n,i++}e.setPipeline(le),e.setBindGroup(0,mi),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),n=!1,e=a.beginComputePass();{let c=n?A[i]:M[i],h=n?C[i]:E[i],s=R[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,c),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,h),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n,i++}for(e.setPipeline(oi),e.setBindGroup(0,bi),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(le),e.setBindGroup(0,fi),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),n=!1,e=a.beginComputePass();i<Me;i++){let c=n?A[i]:M[i],h=n?C[i]:E[i],s=R[i];e.setPipeline(s.dwPipeline),e.setBindGroup(0,c),e.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),e.setPipeline(s.pwPipeline),e.setBindGroup(0,h),e.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),n=!n}for(;i<q.length;i++){let c=i-Me,h=n?ba[c]:fa[c],s=q[i];e.setPipeline(_i),e.setBindGroup(0,h),e.dispatchWorkgroups(s.outW,s.outH,1),n=!n}e.setPipeline(pi),e.setBindGroup(0,gi),e.dispatchWorkgroups(1),e.end(),a.copyBufferToBuffer(N,0,r,0,260)}async function ne(a){t.queue.writeBuffer(ea,0,a);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(Qe),e.setBindGroup(0,ma),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}$(r,S),t.queue.submit([r.finish()]);let n=S.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(S.getMappedRange());return ae[0]=i[0],ie[0]=i[1],te.set(i.subarray(2,65)),S.unmap(),{handflag:new Float32Array(ae),handedness:new Float32Array(ie),landmarks:new Float32Array(te)}}async function yi(a){t.queue.copyExternalImageToTexture({source:a},{texture:ee},[256,256]);let r=t.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(ce),e.setBindGroup(0,Ce),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}$(r,S),t.queue.submit([r.finish()]);let n=S.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await n;let i=new Float32Array(S.getMappedRange());return ae[0]=i[0],ie[0]=i[1],te.set(i.subarray(2,65)),S.unmap(),{handflag:new Float32Array(ae),handedness:new Float32Array(ie),landmarks:new Float32Array(te)}}let xi=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Ee=0,vi=[S,xi],I=null,L=null;async function Pi(a){let r=vi[Ee];Ee=1-Ee,t.queue.copyExternalImageToTexture({source:a},{texture:ee},[256,256]);let n=t.createCommandEncoder();{let e=n.beginComputePass();e.setPipeline(ce),e.setBindGroup(0,Ce),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}$(n,r),t.queue.submit([n.finish()]);let i=null;if(I!==null&&L!==null){await I;let e=new Float32Array(L.getMappedRange());i={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},L.unmap()}return L=r,I=r.mapAsync(GPUMapMode.READ),i}async function ki(){if(!I||!L)return null;await I;let a=new Float32Array(L.getMappedRange()),r={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return L.unmap(),I=null,L=null,r}async function Bi(a=50){let r=new Float32Array(196608);for(let e=0;e<5;e++)await ne(r);let n=[];for(let e=0;e<a;e++){let f=performance.now();await ne(r),n.push(performance.now()-f)}let i=n.reduce((e,f)=>e+f,0)/n.length;return{avgMs:i,fps:1e3/i}}async function Ui(a=50){let r=new Float32Array(196608);for(let w=0;w<5;w++)await ne(r);let n=[];for(let w=0;w<a;w++){let k=t.createCommandEncoder();{let h=k.beginComputePass();h.setPipeline(Qe),h.setBindGroup(0,ma),h.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),h.end()}$(k,S);let c=performance.now();t.queue.submit([k.finish()]),await t.queue.onSubmittedWorkDone(),n.push(performance.now()-c)}n.sort((w,k)=>w-k);let i=n.reduce((w,k)=>w+k,0)/n.length,e=n[Math.floor(n.length/2)],f=n[0];return{avgMs:i,fps:1e3/i,medianMs:e,minMs:f}}function Si(a){t.queue.copyExternalImageToTexture({source:a},{texture:ee},[256,256]);let r=t.createCommandEncoder();{let n=r.beginComputePass();n.setPipeline(ce),n.setBindGroup(0,Ce),n.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),n.end()}$(r,S),t.queue.submit([r.finish()])}return{device:t,run:ne,runFromCanvas:yi,runFromCanvasPipelined:Pi,flushPipelined:ki,benchmark:Bi,benchmarkGPU:Ui,_device:t,_benchmarkSubmitOnly:Si}}var Mi="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Ei(u={}){let{weightsUrl:p,scoreThreshold:t=.5}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let b=p??Mi,d=b.endsWith("/")?`${b}weights.json`:`${b}/weights.json`,W=b.endsWith("/")?`${b}weights.bin`:`${b}/weights.bin`,[_,D]=await Promise.all([fetch(d),fetch(W)]);if(!_.ok)throw new Error(`Failed to fetch weights metadata: ${_.status}`);if(!D.ok)throw new Error(`Failed to fetch weights binary: ${D.status}`);let G=await _.json(),H=await D.arrayBuffer(),y=$a(G,H),o=await Xa(y),l=null;function P(){return l||(l=new OffscreenCanvas(256,256)),l}async function X(m){if(m instanceof HTMLCanvasElement||m instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&m instanceof ImageBitmap)return m;let B=P();B.width=256,B.height=256;let g=B.getContext("2d");return m instanceof ImageData?g.putImageData(m,0,0):g.drawImage(m,0,0,256,256),B}function F(m,B,g){let Y=m[0];if(Y<t)return null;let pe=B[0]>.5,Z=[];for(let T=0;T<21;T++)Z.push({x:g[T*3],y:g[T*3+1],z:g[T*3+2]});return{score:Y,handedness:pe?"right":"left",landmarks:Z}}async function re(m){let B=await X(m),g=await o.runFromCanvas(B);return F(g.handflag,g.handedness,g.landmarks)}async function se(m){let B=await X(m),g=await o.runFromCanvasPipelined(B);return g?F(g.handflag,g.handedness,g.landmarks):null}async function ue(){let m=await o.flushPipelined();return m?F(m.handflag,m.handedness,m.landmarks):null}function oe(){o.device.destroy(),l=null}return{detect:re,detectPipelined:se,flushPipelined:ue,dispose:oe}}export{Ei as createHandpose};
