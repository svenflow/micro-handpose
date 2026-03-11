function k(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Fa=k(`
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
`),za=k(`
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
`),Na=k(`
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
`),qa=k(`
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
`);function Ue(o,p){return za.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function Se(o,p){return Fa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function De(o,p){return Na.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function Ge(o,p){return qa.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${p},1)`)}function Ae(o,p){return[8,8]}var Ce=k(`
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
`),Ee=k(`
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
`);function Me(o){return k(`
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
`)}var He=Me(!1),We=Me(!0),Le=k(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Te=k(`
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
`);function Re(o){return k(`
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
`)}var Oe=Re("sigmoid"),Ie=Re("div256"),Fe=k(`
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
`),ze=k(`
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
`);var Ne=k(`
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
`);function Xe(o,p){let n=new Map,f=o.dtype??"float32";for(let c=0;c<o.keys.length;c++){let L=o.keys[c],_=o.shapes[c],G=o.offsets[c],A=_.reduce((y,u)=>y*u,1),U;if(f==="float32")U=new Float32Array(p,G,A);else{let y=new DataView(p);U=new Float32Array(A);for(let u=0;u<A;u++)U[u]=Dt(y.getUint16(G+u*2,!0))}n.set(L,{data:U,shape:_})}return n}function Dt(o){let p=o>>15&1,n=o>>10&31,f=o&1023;if(n===0){if(f===0)return p?-0:0;let _=-14,G=f/1024;return(p?-1:1)*Math.pow(2,_)*G}if(n===31)return f===0?p?-1/0:1/0:NaN;let c=n-15,L=1+f/1024;return(p?-1:1)*Math.pow(2,c)*L}var Gt=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],qe=Gt.map(([o,p,n,f,c])=>({type:"resmodule",inCh:o,outCh:p,h:n,w:n,stride:f,prefix:c})),At=2,Ct=5,Et=8,Mt=11;async function Ye(o){if(!navigator.gpu)throw new Error("WebGPU not supported");let p=await navigator.gpu.requestAdapter();if(!p)throw new Error("No GPU adapter found");let n=await p.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(p.limits.maxStorageBuffersPerShaderStage,10),maxComputeWorkgroupSizeX:Math.min(p.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(p.limits.maxComputeInvocationsPerWorkgroup,288)}}),f={r:"read-only-storage",s:"storage",u:"uniform"};function c(e){return n.createBindGroupLayout({entries:e.map((r,i)=>({binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:f[r]}}))})}function L(e){return n.createBindGroupLayout({entries:e.map((r,i)=>r==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:f[r]}})})}let _=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,G=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.STORAGE,U=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function u(e,r){return n.createBuffer({size:e,usage:r})}function l(e,r){return n.createBindGroup({layout:e,entries:r.map((i,t)=>({binding:t,resource:"size"in i?{buffer:i}:i}))})}function v(e,r){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:r,entryPoint:"main"}})}let X=n.createShaderModule({code:Ce}),F=n.createShaderModule({code:Ne}),na=n.createShaderModule({code:Fe}),ra=n.createShaderModule({code:za}),sa=n.createShaderModule({code:Fa}),ua=n.createShaderModule({code:Na}),m=n.createShaderModule({code:qa}),B=n.createShaderModule({code:Ee}),g=n.createShaderModule({code:He}),Y=n.createShaderModule({code:Le}),oa=n.createShaderModule({code:We}),$=n.createShaderModule({code:Te}),T=n.createShaderModule({code:Oe}),$e=n.createShaderModule({code:Ie}),Ze=n.createShaderModule({code:ze}),pa=c(["r","r","r","s","u"]),_a=c(["r","r","r","r","s","u"]),Xa=c(["r","s","u"]),Ya=c(["r","r","r","s","u"]),Ve=c(["r","s","u"]),je=c(["r","r","s","u"]),Z=c(["r","r","s","u"]),$a=c(["r","r","r","s","u"]),z=c(["r","r","r","s","u"]),Za=L(["t","s","u"]),Va=c(["r","r","r","r","r","r","r","s","s","s"]),da=c(["r","r","r","r","r","s","u"]),Ke=n.createPipelineLayout({bindGroupLayouts:[pa]}),Je=n.createPipelineLayout({bindGroupLayouts:[_a]}),V=e=>n.createComputePipeline({layout:Ke,compute:{module:e,entryPoint:"main"}}),j=e=>n.createComputePipeline({layout:Je,compute:{module:e,entryPoint:"main"}}),Qe=V(ra),at=V(sa),et=j(ua),tt=j(m),ja=new Map,Ka=new Map,Ja=new Map,Qa=new Map;ja.set("8,8",Qe),Ka.set("8,8",at),Ja.set("8,8",et),Qa.set("8,8",tt);function K(e,r,i,t,a){let b=`${r},${i}`,d=e.get(b);return d||(d=a(n.createShaderModule({code:t(r,i)})),e.set(b,d)),d}let it=(e,r)=>K(ja,e,r,Ue,V),nt=(e,r)=>K(Ka,e,r,Se,V),rt=(e,r)=>K(Ja,e,r,De,j),st=(e,r)=>K(Qa,e,r,Ge,j),R=qe.map(e=>{let r=e.stride===2?e.h/2:e.h,i=e.stride===2?e.w/2:e.w,[t,a]=Ae(e.inCh,r),b=e.h>=64,d=r>=16&&e.inCh>=288&&e.outCh>=288&&e.outCh%2===0;return{dwPipeline:b?nt(t,a):it(t,a),pwPipeline:d?st(t,a):rt(t,a),dwDispatchX:Math.ceil(i/t),dwDispatchY:Math.ceil(r/a),dwDispatchZ:e.inCh,pwDispatchX:Math.ceil(i/t),pwDispatchY:Math.ceil(r/a),pwDispatchZ:d?e.outCh/2:e.outCh}}),ae=v(Xa,X),ut=v(Ya,B);v(Ve,g),v(je,Y);let ca=v(Z,oa),ot=v($a,$);v(z,T),v(z,$e);let la=v(Za,F),pt=v(Va,na),_t=v(da,Ze),wa=1*288*128*128*4,ee=u(3*256*256*4,_),ha=u(3*257*257*4,A),te=u(12,y);n.queue.writeBuffer(te,0,new Uint32Array([3,256,257]));let x=u(wa,G),S=u(wa,U),J=u(wa,A),ie=u(3072*64*4,_),ne=u(3072*32*4,_),re=u(1536*16*4,_),se=u(6144*64*4,A),ma=u(4,U),fa=u(4,U),ba=u(252,U),D=u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);u(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let Q=n.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ue=u(8,y);n.queue.writeBuffer(ue,0,new Uint32Array([256,257]));let ga=o.get("backbone1.1.weight")?.data,ya=o.get("backbone1.1.bias")?.data;if(!ga||!ya)throw new Error("Missing input conv weights");let oe=u(ga.byteLength,_),pe=u(ya.byteLength,_),_e=u(28,y);n.queue.writeBuffer(oe,0,ga),n.queue.writeBuffer(pe,0,ya),n.queue.writeBuffer(_e,0,new Uint32Array([1,3,24,257,257,128,128]));let xa=o.get("backbone6.1.weight")?.data,va=o.get("backbone6.1.bias")?.data;if(!xa||!va)throw new Error("Missing backbone6.1 conv1x1 weights");let de=u(xa.byteLength,_),ce=u(va.byteLength,_),le=u(20,y);n.queue.writeBuffer(de,0,xa),n.queue.writeBuffer(ce,0,va),n.queue.writeBuffer(le,0,new Uint32Array([1,96,48,32,32]));let Pa=o.get("handflag.weight")?.data,ka=o.get("handflag.bias")?.data;if(!Pa||!ka)throw new Error("Missing handflag weights");let Ba=u(Pa.byteLength,_),Ua=u(ka.byteLength,_),we=u(12,y);n.queue.writeBuffer(Ba,0,Pa),n.queue.writeBuffer(Ua,0,ka),n.queue.writeBuffer(we,0,new Uint32Array([1,288,1]));let Sa=o.get("handedness.weight")?.data,Da=o.get("handedness.bias")?.data;if(!Sa||!Da)throw new Error("Missing handedness weights");let Ga=u(Sa.byteLength,_),Aa=u(Da.byteLength,_),he=u(12,y);n.queue.writeBuffer(Ga,0,Sa),n.queue.writeBuffer(Aa,0,Da),n.queue.writeBuffer(he,0,new Uint32Array([1,288,1]));let Ca=o.get("reg_3d.weight")?.data,Ea=o.get("reg_3d.bias")?.data;if(!Ca||!Ea)throw new Error("Missing reg_3d weights");let Ma=u(Ca.byteLength,_),Ha=u(Ea.byteLength,_),me=u(12,y);n.queue.writeBuffer(Ma,0,Ca),n.queue.writeBuffer(Ha,0,Ea),n.queue.writeBuffer(me,0,new Uint32Array([1,288,63]));let N=qe.map(e=>{let{inCh:r,outCh:i,h:t,w:a,stride:b,prefix:d}=e,P=b===2?t/2:t,w=b===2?a/2:a,h=b===1?2:1,s=o.get(`${d}convs.0.weight`)?.data,Ra=o.get(`${d}convs.0.bias`)?.data,Oa=o.get(`${d}convs.1.weight`)?.data,Ia=o.get(`${d}convs.1.bias`)?.data;if(!s||!Ra||!Oa||!Ia)throw new Error(`Missing weights for ${d}`);let ye=u(s.byteLength,_),xe=u(Ra.byteLength,_),ve=u(Oa.byteLength,_),Pe=u(Ia.byteLength,_),ke=u(32,y),Be=u(36,y);return n.queue.writeBuffer(ye,0,s),n.queue.writeBuffer(xe,0,Ra),n.queue.writeBuffer(ve,0,Oa),n.queue.writeBuffer(Pe,0,Ia),n.queue.writeBuffer(ke,0,new Uint32Array([1,r,t,a,P,w,b,h])),n.queue.writeBuffer(Be,0,new Uint32Array([1,r,i,P,w,Math.max(0,i-r),b,t,a])),{dwWeight:ye,dwBias:xe,pwWeight:ve,pwBias:Pe,dwUniform:ke,pwUniform:Be,spec:e,outH:P,outW:w}});function O(e){let r=u(e.length*4,y);return n.queue.writeBuffer(r,0,new Uint32Array(e)),r}let dt=O([1,96,8,8,16,16]),ct=O([1,96,16,16,32,32]),lt=O([1,48,32,32,64,64]);O([1536*16]),O([3072*32]),O([3072*64]);let fe=l(Xa,[ee,ha,te]),wt=l(Ya,[ha,oe,pe,x,_e]),C=[],E=[],M=[],H=[];for(let e of N)C.push(l(pa,[x,e.dwWeight,e.dwBias,J,e.dwUniform])),E.push(l(_a,[J,x,e.pwWeight,e.pwBias,S,e.pwUniform])),M.push(l(pa,[S,e.dwWeight,e.dwBias,J,e.dwUniform])),H.push(l(_a,[J,S,e.pwWeight,e.pwBias,x,e.pwUniform]));let ht=l(Z,[x,re,S,dt]),mt=l(Z,[x,ne,S,ct]),ft=l($a,[x,de,ce,se,le]),bt=l(Z,[se,ie,S,lt]);l(z,[x,Ba,Ua,ma,we]),l(z,[x,Ga,Aa,fa,he]),l(z,[x,Ma,Ha,ba,me]);let Wa=l(Za,[Q.createView(),ha,ue]),gt=l(Va,[x,Ba,Ua,Ga,Aa,Ma,Ha,ma,fa,ba]),La=24,be=[],ge=[];for(let e=La;e<N.length;e++){let r=N[e];be.push(l(da,[x,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,S,r.dwUniform])),ge.push(l(da,[S,r.dwWeight,r.dwBias,r.pwWeight,r.pwBias,x,r.dwUniform]))}let aa=new Float32Array(1),ea=new Float32Array(1),ta=new Float32Array(63);function q(e,r){let i=!0,t=0,a=e.beginComputePass();for(a.setPipeline(ut),a.setBindGroup(0,wt),a.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);t<=At;t++){let w=i?C[t]:M[t],h=i?E[t]:H[t],s=R[t];a.setPipeline(s.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),a.setPipeline(s.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}a.end();let b=i?x:S;for(e.copyBufferToBuffer(b,0,ie,0,3072*64*4),a=e.beginComputePass();t<=Ct;t++){let w=i?C[t]:M[t],h=i?E[t]:H[t],s=R[t];a.setPipeline(s.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),a.setPipeline(s.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}a.end();let d=i?x:S;for(e.copyBufferToBuffer(d,0,ne,0,3072*32*4),a=e.beginComputePass();t<=Et;t++){let w=i?C[t]:M[t],h=i?E[t]:H[t],s=R[t];a.setPipeline(s.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),a.setPipeline(s.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}a.end();let P=i?x:S;for(e.copyBufferToBuffer(P,0,re,0,1536*16*4),a=e.beginComputePass();t<=Mt;t++){let w=i?C[t]:M[t],h=i?E[t]:H[t],s=R[t];a.setPipeline(s.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),a.setPipeline(s.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}a.setPipeline(ca),a.setBindGroup(0,ht),a.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),a.end(),i=!1,a=e.beginComputePass();{let w=i?C[t]:M[t],h=i?E[t]:H[t],s=R[t];a.setPipeline(s.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),a.setPipeline(s.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i,t++}a.setPipeline(ca),a.setBindGroup(0,mt),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),a.end(),i=!1,a=e.beginComputePass();{let w=i?C[t]:M[t],h=i?E[t]:H[t],s=R[t];a.setPipeline(s.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),a.setPipeline(s.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i,t++}for(a.setPipeline(ot),a.setBindGroup(0,ft),a.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),a.setPipeline(ca),a.setBindGroup(0,bt),a.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),a.end(),i=!1,a=e.beginComputePass();t<La;t++){let w=i?C[t]:M[t],h=i?E[t]:H[t],s=R[t];a.setPipeline(s.dwPipeline),a.setBindGroup(0,w),a.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),a.setPipeline(s.pwPipeline),a.setBindGroup(0,h),a.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),i=!i}for(;t<N.length;t++){let w=t-La,h=i?be[w]:ge[w],s=N[t];a.setPipeline(_t),a.setBindGroup(0,h),a.dispatchWorkgroups(s.outW,s.outH,1),i=!i}a.setPipeline(pt),a.setBindGroup(0,gt),a.dispatchWorkgroups(1),a.end(),e.copyBufferToBuffer(ma,0,r,0,4),e.copyBufferToBuffer(fa,0,r,4,4),e.copyBufferToBuffer(ba,0,r,8,252)}async function ia(e){n.queue.writeBuffer(ee,0,e);let r=n.createCommandEncoder();{let t=r.beginComputePass();t.setPipeline(ae),t.setBindGroup(0,fe),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}q(r,D),n.queue.submit([r.finish()]),await D.mapAsync(GPUMapMode.READ);let i=new Float32Array(D.getMappedRange());return aa[0]=i[0],ea[0]=i[1],ta.set(i.subarray(2,65)),D.unmap(),{handflag:new Float32Array(aa),handedness:new Float32Array(ea),landmarks:new Float32Array(ta)}}async function yt(e){n.queue.copyExternalImageToTexture({source:e},{texture:Q},[256,256]);let r=n.createCommandEncoder();{let t=r.beginComputePass();t.setPipeline(la),t.setBindGroup(0,Wa),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}q(r,D),n.queue.submit([r.finish()]),await D.mapAsync(GPUMapMode.READ);let i=new Float32Array(D.getMappedRange());return aa[0]=i[0],ea[0]=i[1],ta.set(i.subarray(2,65)),D.unmap(),{handflag:new Float32Array(aa),handedness:new Float32Array(ea),landmarks:new Float32Array(ta)}}let xt=n.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Ta=0,vt=[D,xt],I=null,W=null;async function Pt(e){let r=vt[Ta];Ta=1-Ta,n.queue.copyExternalImageToTexture({source:e},{texture:Q},[256,256]);let i=n.createCommandEncoder();{let a=i.beginComputePass();a.setPipeline(la),a.setBindGroup(0,Wa),a.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),a.end()}q(i,r),n.queue.submit([i.finish()]);let t=null;if(I!==null&&W!==null){await I;let a=new Float32Array(W.getMappedRange());t={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))},W.unmap()}return W=r,I=r.mapAsync(GPUMapMode.READ),t}async function kt(){if(!I||!W)return null;await I;let e=new Float32Array(W.getMappedRange()),r={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))};return W.unmap(),I=null,W=null,r}async function Bt(e=50){let r=new Float32Array(196608);for(let a=0;a<5;a++)await ia(r);let i=[];for(let a=0;a<e;a++){let b=performance.now();await ia(r),i.push(performance.now()-b)}let t=i.reduce((a,b)=>a+b,0)/i.length;return{avgMs:t,fps:1e3/t}}async function Ut(e=50){let r=new Float32Array(196608);for(let d=0;d<5;d++)await ia(r);let i=[];for(let d=0;d<e;d++){let P=n.createCommandEncoder();{let h=P.beginComputePass();h.setPipeline(ae),h.setBindGroup(0,fe),h.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),h.end()}q(P,D);let w=performance.now();n.queue.submit([P.finish()]),await n.queue.onSubmittedWorkDone(),i.push(performance.now()-w)}i.sort((d,P)=>d-P);let t=i.reduce((d,P)=>d+P,0)/i.length,a=i[Math.floor(i.length/2)],b=i[0];return{avgMs:t,fps:1e3/t,medianMs:a,minMs:b}}function St(e){n.queue.copyExternalImageToTexture({source:e},{texture:Q},[256,256]);let r=n.createCommandEncoder();{let i=r.beginComputePass();i.setPipeline(la),i.setBindGroup(0,Wa),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}q(r,D),n.queue.submit([r.finish()])}return{device:n,run:ia,runFromCanvas:yt,runFromCanvasPipelined:Pt,flushPipelined:kt,benchmark:Bt,benchmarkGPU:Ut,_device:n,_benchmarkSubmitOnly:St}}var Ht="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Wt(o={}){let{weightsUrl:p,scoreThreshold:n=.5}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let f=p??Ht,c=f.endsWith("/")?`${f}weights.json`:`${f}/weights.json`,L=f.endsWith("/")?`${f}weights.bin`:`${f}/weights.bin`,[_,G]=await Promise.all([fetch(c),fetch(L)]);if(!_.ok)throw new Error(`Failed to fetch weights metadata: ${_.status}`);if(!G.ok)throw new Error(`Failed to fetch weights binary: ${G.status}`);let A=await _.json(),U=await G.arrayBuffer(),y=Xe(A,U),u=await Ye(y),l=null;function v(){return l||(l=new OffscreenCanvas(256,256)),l}async function X(m){if(m instanceof HTMLCanvasElement||m instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&m instanceof ImageBitmap)return m;let B=v();B.width=256,B.height=256;let g=B.getContext("2d");return m instanceof ImageData?g.putImageData(m,0,0):g.drawImage(m,0,0,256,256),B}function F(m,B,g){let Y=m[0];if(Y<n)return null;let oa=B[0]>.5,$=[];for(let T=0;T<21;T++)$.push({x:g[T*3],y:g[T*3+1],z:g[T*3+2]});return{score:Y,handedness:oa?"right":"left",landmarks:$}}async function na(m){let B=await X(m),g=await u.runFromCanvas(B);return F(g.handflag,g.handedness,g.landmarks)}async function ra(m){let B=await X(m),g=await u.runFromCanvasPipelined(B);return g?F(g.handflag,g.handedness,g.landmarks):null}async function sa(){let m=await u.flushPipelined();return m?F(m.handflag,m.handedness,m.landmarks):null}function ua(){u.device.destroy(),l=null}return{detect:na,detectPipelined:ra,flushPipelined:sa,dispose:ua}}export{Wt as createHandpose};
