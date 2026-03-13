function fe(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function $n(o){let m=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+o;for(let U of m)for(;a.includes(`${U}:array<f32>`);)a=a.replace(`${U}:array<f32>`,`${U}:array<f16>`);return a}var Un=fe(`
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
`),An=fe(`
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
`),Gn=fe(`
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
`),Sn=fe(`
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
`);function Vn(o,m){return An.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function Zn(o,m){return Un.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function jn(o,m){return Gn.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function Jn(o,m){return Sn.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${m},1)`)}function Qn(o,m){return[8,8]}var ea=fe(`
struct PadParams { channels:u32, in_size:u32, out_size:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:PadParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y; let c=gid.z;
  if(x>=params.in_size||y>=params.in_size||c>=params.channels){return;}
  let in_idx=c*params.in_size*params.in_size+y*params.in_size+x;
  // PyTorch uses ConstantPad2d((0,1,0,1)): pad right+bottom, image stays at (0,0)
  let out_idx=c*params.out_size*params.out_size+y*params.out_size+x;
  output[out_idx]=input[in_idx];
}
`),ta=fe(`
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
`);function na(o){return fe(`
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
`)}var aa=na(!1),ia=na(!0),ra=fe(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),sa=fe(`
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
`);function oa(o){return fe(`
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
`)}var ua=oa("sigmoid"),pa=oa("div256"),ca=fe(`
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
`),da=fe(`
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
`);function _a(o,m){let U=Math.min(m,256),x=m>U,v=o%4===0?`var ic:u32=0u;
    while(ic<${o}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${o}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,z=`var skip_val:f32=0.0;
    if(c<${o}u){
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
    }`,W=o===m?"":`if(c<${o}u){`,f=o===m?"":"}",g=x?`for(var c:u32=lid.x;c<${o}u;c+=${U}u){`:`let c=lid.x;
  ${W}`,A=x?"}":f,L=x?`for(var c:u32=lid.x;c<${m}u;c+=${U}u){`:"{let c=lid.x;";return fe(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${o}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${U},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${g}
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
  ${A}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${L}
    let pw_base=c*${o}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${v}
    // Skip connection (only for c < inCh)
    ${z}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var la=fe(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),ma=fe(`
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
`),fa=fe(`
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
  // PyTorch uses ConstantPad2d((0,1,0,1)): pad right+bottom, image stays at (0,0)
  output[0u*out_stride+y*params.out_size+x]=pixel.r;
  output[1u*out_stride+y*params.out_size+x]=pixel.g;
  output[2u*out_stride+y*params.out_size+x]=pixel.b;
}
`);function Mn(o,m){let a=new Map,U=o.dtype??"float32";for(let x=0;x<o.keys.length;x++){let t=o.keys[x],v=o.shapes[x],z=o.offsets[x],W=v.reduce((A,L)=>A*L,1),f,g;if(U==="float32")f=new Float32Array(m,z,W);else{let A=new DataView(m);f=new Float32Array(W);for(let L=0;L<W;L++)f[L]=Fa(A.getUint16(z+L*2,!0));g=m.slice(z,z+W*2)}a.set(t,{data:f,shape:v,rawF16:g})}return a}function Fa(o){let m=o>>15&1,a=o>>10&31,U=o&1023;if(a===0){if(U===0)return m?-0:0;let v=-14,z=U/1024;return(m?-1:1)*Math.pow(2,v)*z}if(a===31)return U===0?m?-1/0:1/0:NaN;let x=a-15,t=1+U/1024;return(m?-1:1)*Math.pow(2,x)*t}var Ka=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],ha=Ka.map(([o,m,a,U,x])=>({type:"resmodule",inCh:o,outCh:m,h:a,w:a,stride:U,prefix:x})),Na=2,qa=5,Ya=8,Xa=11;async function Dn(o,m){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let U=a.features.has("shader-f16"),x=U?["shader-f16"]:[],t=await a.requestDevice({requiredFeatures:x,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(a.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(a.limits.maxComputeInvocationsPerWorkgroup,288)}}),v=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(U)try{let r=`enable f16;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> dw_weight: array<f16>;
@group(0) @binding(2) var<storage, read> dw_bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> intermediate: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= 96u) { return; }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < 25u; i = i + 1u) {
    sum = sum + input[ch * 25u + i] * f32(dw_weight[ch * 25u + i]);
  }
  intermediate[ch] = max(0.0, sum + f32(dw_bias[ch]));
}`,d=`enable f16;
@group(0) @binding(0) var<storage, read> intermediate: array<f32>;
@group(0) @binding(1) var<storage, read> pw_weight: array<f16>;
@group(0) @binding(2) var<storage, read> pw_bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let oc = gid.x;
  if (oc >= 96u) { return; }
  var sum: f32 = 0.0;
  for (var ic: u32 = 0u; ic < 96u; ic = ic + 1u) {
    sum = sum + intermediate[ic] * f32(pw_weight[oc * 96u + ic]);
  }
  output[oc] = sum + f32(pw_bias[oc]);
}`,p=t.createShaderModule({code:r}),u=t.createShaderModule({code:d}),i=await p.getCompilationInfo(),E=await u.getCompilationInfo();if(i.messages.some(S=>S.type==="error")||E.messages.some(S=>S.type==="error"))v=!1;else{let S=new Float32Array(2400);S.fill(1);let T=new Uint16Array(2400);T.fill(10516);let k=new Uint16Array(96);k.fill(14336);let w=new Uint16Array(9216);w.fill(8478);let c=new Uint16Array(96);c.fill(12288);let I=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,se=t.createBuffer({size:S.byteLength,usage:I}),xt=t.createBuffer({size:T.byteLength,usage:I}),vt=t.createBuffer({size:k.byteLength,usage:I}),Pt=t.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),Bt=t.createBuffer({size:w.byteLength,usage:I}),Ct=t.createBuffer({size:c.byteLength,usage:I}),kt=t.createBuffer({size:384,usage:ae}),Je=t.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});t.queue.writeBuffer(se,0,S),t.queue.writeBuffer(xt,0,T),t.queue.writeBuffer(vt,0,k),t.queue.writeBuffer(Bt,0,w),t.queue.writeBuffer(Ct,0,c);let Fe="read-only-storage",Xt="storage",$t=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xt}}]}),Yn=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Fe}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Xt}}]}),Wa=t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[$t]}),compute:{module:p,entryPoint:"main"}}),La=t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[Yn]}),compute:{module:u,entryPoint:"main"}}),Ra=t.createBindGroup({layout:$t,entries:[{binding:0,resource:{buffer:se}},{binding:1,resource:{buffer:xt}},{binding:2,resource:{buffer:vt}},{binding:3,resource:{buffer:Pt}}]}),Oa=t.createBindGroup({layout:Yn,entries:[{binding:0,resource:{buffer:Pt}},{binding:1,resource:{buffer:Bt}},{binding:2,resource:{buffer:Ct}},{binding:3,resource:{buffer:kt}}]}),wn=t.createCommandEncoder(),gn=wn.beginComputePass();gn.setPipeline(Wa),gn.setBindGroup(0,Ra),gn.dispatchWorkgroups(2),gn.end();let yn=wn.beginComputePass();yn.setPipeline(La),yn.setBindGroup(0,Oa),yn.dispatchWorkgroups(2),yn.end(),wn.copyBufferToBuffer(kt,0,Je,0,384),t.queue.submit([wn.finish()]),await t.queue.onSubmittedWorkDone(),await Je.mapAsync(GPUMapMode.READ);let Vt=new Float32Array(Je.getMappedRange()),Xn=1.5*.0104*96+.25,za=Vt[0]!==0&&Vt[47]!==0&&Vt[95]!==0,Ia=Math.abs(Vt[0]-Xn)<1;v=za&&Ia,Je.unmap(),se.destroy(),xt.destroy(),vt.destroy(),Pt.destroy(),Bt.destroy(),Ct.destroy(),kt.destroy(),Je.destroy(),v||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Vt[0]}, expected ~${Xn.toFixed(2)}) \u2014 falling back to f32`)}}catch{v=!1}let W=o.values().next().value,f=v&&!!W?.rawF16&&!m?.forceF32;console.log(f?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${U}, f16 validated: ${v}, f16 data: ${!!W?.rawF16})`);function g(r){if(f&&r.rawF16){let d=new Uint16Array(r.rawF16);if(d.length%2!==0){let p=new Uint16Array(d.length+1);return p.set(d),p}return d}return r.data}function A(r){if(f&&r.rawF16){let d=r.rawF16.byteLength;return Math.ceil(d/4)*4}return r.data.byteLength}function L(r){return f?$n(r):r}let oe={r:"read-only-storage",s:"storage",u:"uniform"};function y(r){return t.createBindGroupLayout({entries:r.map((d,p)=>({binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:oe[d]}}))})}function ie(r){return t.createBindGroupLayout({entries:r.map((d,p)=>d==="t"?{binding:p,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:oe[d]}})})}let R=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,_e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,xe=GPUBufferUsage.STORAGE,ke=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,re=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function G(r,d){return t.createBuffer({size:r,usage:d})}function Z(r,d){return t.createBindGroup({layout:r,entries:d.map((p,u)=>({binding:u,resource:"size"in p?{buffer:p}:p}))})}function ue(r,d){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[r]}),compute:{module:d,entryPoint:"main"}})}let Ut=t.createShaderModule({code:ea}),et=t.createShaderModule({code:fa}),We=t.createShaderModule({code:L(ca)}),Le=t.createShaderModule({code:L(An)}),At=t.createShaderModule({code:L(Un)}),tt=t.createShaderModule({code:L(Gn)}),ge=t.createShaderModule({code:L(Sn)}),Ke=t.createShaderModule({code:L(ta)}),_t=t.createShaderModule({code:aa}),Ue=t.createShaderModule({code:ra}),lt=t.createShaderModule({code:ia}),Zt=t.createShaderModule({code:L(sa)}),P=t.createShaderModule({code:L(ua)}),F=t.createShaderModule({code:L(pa)}),X=t.createShaderModule({code:L(da)}),K=new Map;function Q(r,d){let p=`${r}_${d}`,u=K.get(p);return u||(u=t.createShaderModule({code:L(_a(r,d))}),K.set(p,u)),u}let he=y(["r","r","r","s","u"]),ne=y(["r","r","r","r","s","u"]),$=y(["r","s","u"]),ye=y(["r","r","r","s","u"]),j=y(["r","s","u"]),be=y(["r","r","s","u"]),pe=y(["r","r","s","u"]),Re=y(["r","r","r","s","u"]),ce=y(["r","r","r","s","u"]),ve=ie(["t","s","u"]),Pe=y(["r","r","r","r","r","r","r","s"]),Ae=y(["r","r","r","r","r","s","u"]),Oe=t.createPipelineLayout({bindGroupLayouts:[he]}),Me=t.createPipelineLayout({bindGroupLayouts:[ne]}),le=r=>t.createComputePipeline({layout:Oe,compute:{module:r,entryPoint:"main"}}),De=r=>t.createComputePipeline({layout:Me,compute:{module:r,entryPoint:"main"}}),Ne=le(Le),qe=le(At),nt=De(tt),Ye=De(ge),Xe=new Map,at=new Map,it=new Map,mt=new Map;Xe.set("8,8",Ne),at.set("8,8",qe),it.set("8,8",nt),mt.set("8,8",Ye);function rt(r,d,p,u,i){let E=`${d},${p}`,S=r.get(E);return S||(S=i(t.createShaderModule({code:L(u(d,p))})),r.set(E,S)),S}let Gt=(r,d)=>rt(Xe,r,d,Vn,le),jt=(r,d)=>rt(at,r,d,Zn,le),Jt=(r,d)=>rt(it,r,d,jn,De),ft=(r,d)=>rt(mt,r,d,Jn,De),Ge=ha.map(r=>{let d=r.stride===2?r.h/2:r.h,p=r.stride===2?r.w/2:r.w,[u,i]=Qn(r.inCh,d),E=r.h>=64,S=d>=16&&r.inCh>=288&&r.outCh>=288&&r.outCh%2===0;return{dwPipeline:E?jt(u,i):Gt(u,i),pwPipeline:S?ft(u,i):Jt(u,i),dwDispatchX:Math.ceil(p/u),dwDispatchY:Math.ceil(d/i),dwDispatchZ:r.inCh,pwDispatchX:Math.ceil(p/u),pwDispatchY:Math.ceil(d/i),pwDispatchZ:S?r.outCh/2:r.outCh}}),St=ue($,Ut),ht=ue(ye,Ke);ue(j,_t),ue(be,Ue);let bt=ue(pe,lt),Mt=ue(Re,Zt);ue(ce,P),ue(ce,F);let we=ue(ve,et),st=ue(Pe,We),Qt=ue(Ae,X),wt=1*288*128*128*4,ot=G(3*256*256*4,R),Ee=G(3*257*257*4,xe),$e=G(12,re);t.queue.writeBuffer($e,0,new Uint32Array([3,256,257]));let J=G(wt,_e),de=G(wt,ke),He=G(wt,xe),ut=G(3072*64*4,R),pt=G(3072*32*4,R),ct=G(1536*16*4,R),dt=G(6144*64*4,xe),Te=G(260,ke),V=G(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);G(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let M=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),q=G(8,re);t.queue.writeBuffer(q,0,new Uint32Array([256,257]));let Dt=o.get("backbone1.1.weight"),Et=o.get("backbone1.1.bias");if(!Dt||!Et)throw new Error("Missing input conv weights");let Ht=g(Dt),Tt=g(Et),Wt=G(Ht.byteLength,R),Lt=G(Tt.byteLength,R),en=G(28,re);t.queue.writeBuffer(Wt,0,Ht),t.queue.writeBuffer(Lt,0,Tt),t.queue.writeBuffer(en,0,new Uint32Array([1,3,24,257,257,128,128]));let tn=o.get("backbone6.1.weight"),nn=o.get("backbone6.1.bias");if(!tn||!nn)throw new Error("Missing backbone6.1 conv1x1 weights");let an=g(tn),Rt=g(nn),Ot=G(an.byteLength,R),zt=G(Rt.byteLength,R),It=G(20,re);t.queue.writeBuffer(Ot,0,an),t.queue.writeBuffer(zt,0,Rt),t.queue.writeBuffer(It,0,new Uint32Array([1,96,48,32,32]));let Ft=o.get("handflag.weight"),Kt=o.get("handflag.bias");if(!Ft||!Kt)throw new Error("Missing handflag weights");let rn=g(Ft),sn=g(Kt),ze=G(rn.byteLength,R),gt=G(sn.byteLength,R),Nt=G(12,re);t.queue.writeBuffer(ze,0,rn),t.queue.writeBuffer(gt,0,sn),t.queue.writeBuffer(Nt,0,new Uint32Array([1,288,1]));let qt=o.get("handedness.weight"),on=o.get("handedness.bias");if(!qt||!on)throw new Error("Missing handedness weights");let un=g(qt),Be=g(on),Se=G(un.byteLength,R),Ve=G(Be.byteLength,R),Yt=G(12,re);t.queue.writeBuffer(Se,0,un),t.queue.writeBuffer(Ve,0,Be),t.queue.writeBuffer(Yt,0,new Uint32Array([1,288,1]));let pn=o.get("reg_3d.weight"),cn=o.get("reg_3d.bias");if(!pn||!cn)throw new Error("Missing reg_3d weights");let Y=g(pn),dn=g(cn),n=G(Y.byteLength,R),e=G(dn.byteLength,R),s=G(12,re);t.queue.writeBuffer(n,0,Y),t.queue.writeBuffer(e,0,dn),t.queue.writeBuffer(s,0,new Uint32Array([1,288,63]));let H=ha.map(r=>{let{inCh:d,outCh:p,h:u,w:i,stride:E,prefix:S}=r,T=E===2?u/2:u,k=E===2?i/2:i,w=E===2?1:2,c=o.get(`${S}convs.0.weight`),I=o.get(`${S}convs.0.bias`),ae=o.get(`${S}convs.1.weight`),se=o.get(`${S}convs.1.bias`);if(!c||!I||!ae||!se)throw new Error(`Missing weights for ${S}`);let xt=g(c),vt=g(I),Pt=g(ae),Bt=g(se),Ct=G(xt.byteLength,R),kt=G(vt.byteLength,R),Je=G(Pt.byteLength,R),Fe=G(Bt.byteLength,R),Xt=G(32,re),$t=G(36,re);return t.queue.writeBuffer(Ct,0,xt),t.queue.writeBuffer(kt,0,vt),t.queue.writeBuffer(Je,0,Pt),t.queue.writeBuffer(Fe,0,Bt),t.queue.writeBuffer(Xt,0,new Uint32Array([1,d,u,i,T,k,E,w])),t.queue.writeBuffer($t,0,new Uint32Array([1,d,p,T,k,Math.max(0,p-d),E,u,i])),{dwWeight:Ct,dwBias:kt,pwWeight:Je,pwBias:Fe,dwUniform:Xt,pwUniform:$t,spec:r,outH:T,outW:k}});function O(r){let d=G(r.length*4,re);return t.queue.writeBuffer(d,0,new Uint32Array(r)),d}let l=O([1,96,8,8,16,16]),N=O([1,96,16,16,32,32]),ee=O([1,48,32,32,64,64]);O([1536*16]),O([3072*32]),O([3072*64]);let b=Z($,[ot,Ee,$e]),C=Z(ye,[Ee,Wt,Lt,J,en]),h=[],D=[],_=[],B=[];for(let r of H)h.push(Z(he,[J,r.dwWeight,r.dwBias,He,r.dwUniform])),D.push(Z(ne,[He,J,r.pwWeight,r.pwBias,de,r.pwUniform])),_.push(Z(he,[de,r.dwWeight,r.dwBias,He,r.dwUniform])),B.push(Z(ne,[He,de,r.pwWeight,r.pwBias,J,r.pwUniform]));let te=Z(pe,[J,ct,de,l]),Ce=Z(pe,[J,pt,de,N]),Ze=Z(Re,[J,Ot,zt,dt,It]),On=Z(pe,[dt,ut,de,ee]);Z(ce,[J,ze,gt,Te,Nt]),Z(ce,[J,Se,Ve,Te,Yt]),Z(ce,[J,n,e,Te,s]);let me=Z(ve,[M.createView(),Ee,q]),zn=Z(Pe,[J,ze,gt,Se,Ve,n,e,Te]),xn=24,In=[],Fn=[];for(let r=xn;r<H.length;r++){let d=H[r];In.push(Z(Ae,[J,d.dwWeight,d.dwBias,d.pwWeight,d.pwBias,de,d.dwUniform])),Fn.push(Z(Ae,[de,d.dwWeight,d.dwBias,d.pwWeight,d.pwBias,J,d.dwUniform]))}let vn=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});vn.globalCompositeOperation="copy";let Kn=new OffscreenCanvas(9,8),_n=Kn.getContext("webgpu"),ln=null,Pn=null;if(_n){_n.configure({device:t,format:"rgba8unorm",alphaMode:"premultiplied"});let r=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),d=t.createShaderModule({code:la}),p=t.createShaderModule({code:ma});ln=t.createRenderPipeline({layout:t.createPipelineLayout({bindGroupLayouts:[r]}),vertex:{module:d,entryPoint:"vs"},fragment:{module:p,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),Pn=t.createBindGroup({layout:r,entries:[{binding:0,resource:{buffer:Te}}]})}let mn=new Float32Array(1),fn=new Float32Array(1),hn=new Float32Array(63);function Ie(r,d){let p=!0,u=0,i=r.beginComputePass();for(i.setPipeline(ht),i.setBindGroup(0,C),i.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);u<=Na;u++){let k=p?h[u]:_[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let E=p?J:de;for(r.copyBufferToBuffer(E,0,ut,0,3072*64*4),i=r.beginComputePass();u<=qa;u++){let k=p?h[u]:_[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let S=p?J:de;for(r.copyBufferToBuffer(S,0,pt,0,3072*32*4),i=r.beginComputePass();u<=Ya;u++){let k=p?h[u]:_[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let T=p?J:de;for(r.copyBufferToBuffer(T,0,ct,0,1536*16*4),i=r.beginComputePass();u<=Xa;u++){let k=p?h[u]:_[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.setPipeline(bt),i.setBindGroup(0,te),i.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),i.end(),p=!1,i=r.beginComputePass();{let k=p?h[u]:_[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p,u++}i.setPipeline(bt),i.setBindGroup(0,Ce),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),i.end(),p=!1,i=r.beginComputePass();{let k=p?h[u]:_[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p,u++}for(i.setPipeline(Mt),i.setBindGroup(0,Ze),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),i.setPipeline(bt),i.setBindGroup(0,On),i.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),i.end(),p=!1,i=r.beginComputePass();u<xn;u++){let k=p?h[u]:_[u],w=p?D[u]:B[u],c=Ge[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,w),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}for(;u<H.length;u++){let k=u-xn,w=p?In[k]:Fn[k],c=H[u];i.setPipeline(Qt),i.setBindGroup(0,w),i.dispatchWorkgroups(c.outW,c.outH,1),p=!p}i.setPipeline(st),i.setBindGroup(0,zn),i.dispatchWorkgroups(1),i.end(),d&&r.copyBufferToBuffer(Te,0,d,0,260)}async function bn(r){t.queue.writeBuffer(ot,0,r);let d=t.createCommandEncoder();{let i=d.beginComputePass();i.setPipeline(St),i.setBindGroup(0,b),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),i.end()}Ie(d,V),t.queue.submit([d.finish()]);let p=V.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await p;let u=new Float32Array(V.getMappedRange());return mn[0]=u[0],fn[0]=u[1],hn.set(u.subarray(2,65)),V.unmap(),{handflag:new Float32Array(mn),handedness:new Float32Array(fn),landmarks:new Float32Array(hn)}}async function Bn(r){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let d=t.createCommandEncoder();{let i=d.beginComputePass();i.setPipeline(we),i.setBindGroup(0,me),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}Ie(d,V),t.queue.submit([d.finish()]);let p=V.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await p;let u=new Float32Array(V.getMappedRange());return mn[0]=u[0],fn[0]=u[1],hn.set(u.subarray(2,65)),V.unmap(),{handflag:new Float32Array(mn),handedness:new Float32Array(fn),landmarks:new Float32Array(hn)}}async function Nn(r){if(!ln||!Pn||!_n)throw new Error("Render-based readback not available (no WebGPU canvas context)");t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let d=t.createCommandEncoder();{let k=d.beginComputePass();k.setPipeline(we),k.setBindGroup(0,me),k.dispatchWorkgroups(16,16,1),k.end()}Ie(d,null);let p=_n.getCurrentTexture(),u=d.beginRenderPass({colorAttachments:[{view:p.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});u.setPipeline(ln),u.setBindGroup(0,Pn),u.draw(3),u.end(),t.queue.submit([d.finish()]),await t.queue.onSubmittedWorkDone(),vn.drawImage(Kn,0,0);let E=vn.getImageData(0,0,9,8).data,S=new Float32Array(65),T=new DataView(new ArrayBuffer(4));for(let k=0;k<65;k++){let w=k*4;T.setUint8(0,E[w]),T.setUint8(1,E[w+1]),T.setUint8(2,E[w+2]),T.setUint8(3,E[w+3]),S[k]=T.getFloat32(0)}return{handflag:new Float32Array([S[0]]),handedness:new Float32Array([S[1]]),landmarks:new Float32Array(S.subarray(2,65))}}let Sa=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Cn=0,Ma=[V,Sa],yt=null,je=null;async function kn(r){let d=Ma[Cn];Cn=1-Cn,t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let p=t.createCommandEncoder();{let i=p.beginComputePass();i.setPipeline(we),i.setBindGroup(0,me),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}Ie(p,d),t.queue.submit([p.finish()]);let u=null;if(yt!==null&&je!==null){await yt;let i=new Float32Array(je.getMappedRange());u={handflag:new Float32Array([i[0]]),handedness:new Float32Array([i[1]]),landmarks:new Float32Array(i.subarray(2,65))},je.unmap()}return je=d,yt=d.mapAsync(GPUMapMode.READ),u}async function qn(){if(!yt||!je)return null;await yt;let r=new Float32Array(je.getMappedRange()),d={handflag:new Float32Array([r[0]]),handedness:new Float32Array([r[1]]),landmarks:new Float32Array(r.subarray(2,65))};return je.unmap(),yt=null,je=null,d}async function Da(r=50){let d=new Float32Array(196608);for(let i=0;i<5;i++)await bn(d);let p=[];for(let i=0;i<r;i++){let E=performance.now();await bn(d),p.push(performance.now()-E)}let u=p.reduce((i,E)=>i+E,0)/p.length;return{avgMs:u,fps:1e3/u}}async function Ea(r=50){let d=new Float32Array(196608);for(let S=0;S<5;S++)await bn(d);let p=[];for(let S=0;S<r;S++){let T=t.createCommandEncoder();{let w=T.beginComputePass();w.setPipeline(St),w.setBindGroup(0,b),w.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),w.end()}Ie(T,V);let k=performance.now();t.queue.submit([T.finish()]),await t.queue.onSubmittedWorkDone(),p.push(performance.now()-k)}p.sort((S,T)=>S-T);let u=p.reduce((S,T)=>S+T,0)/p.length,i=p[Math.floor(p.length/2)],E=p[0];return{avgMs:u,fps:1e3/u,medianMs:i,minMs:E}}function ni(r){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let d=t.createCommandEncoder();{let p=d.beginComputePass();p.setPipeline(we),p.setBindGroup(0,me),p.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),p.end()}Ie(d,V),t.queue.submit([d.finish()])}async function Ha(r,d=50){function p(w){let c=[...w].sort((I,ae)=>I-ae);return{median:c[Math.floor(c.length/2)],min:c[0]}}for(let w=0;w<10;w++)await Bn(r);let u=[];for(let w=0;w<d;w++){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let c=t.createCommandEncoder();{let ae=c.beginComputePass();ae.setPipeline(we),ae.setBindGroup(0,me),ae.dispatchWorkgroups(16,16,1),ae.end()}Ie(c,V);let I=performance.now();t.queue.submit([c.finish()]),await t.queue.onSubmittedWorkDone(),u.push(performance.now()-I)}let i=[];for(let w=0;w<d;w++){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let c=t.createCommandEncoder();{let se=c.beginComputePass();se.setPipeline(we),se.setBindGroup(0,me),se.dispatchWorkgroups(16,16,1),se.end()}Ie(c,V),t.queue.submit([c.finish()]);let I=V.mapAsync(GPUMapMode.READ),ae=performance.now();await t.queue.onSubmittedWorkDone(),await I,V.getMappedRange(),V.unmap(),i.push(performance.now()-ae)}let E=[];for(let w=0;w<d;w++){t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);let c=t.createCommandEncoder();{let ae=c.beginComputePass();ae.setPipeline(we),ae.setBindGroup(0,me),ae.dispatchWorkgroups(16,16,1),ae.end()}Ie(c,V),t.queue.submit([c.finish()]);let I=performance.now();await V.mapAsync(GPUMapMode.READ),V.getMappedRange(),V.unmap(),E.push(performance.now()-I)}let S=[];for(let w=0;w<d;w++){let c=performance.now();await Bn(r),S.push(performance.now()-c)}await kn(r);let T=[];for(let w=0;w<d;w++){let c=performance.now();await kn(r),T.push(performance.now()-c)}await qn();let k=null;if(ln){let w=[];for(let c=0;c<d;c++){let I=performance.now();await Nn(r),w.push(performance.now()-I)}k=p(w)}return{gpuOnly:p(u),mapAsyncOnly:p(i),mapAsyncNoWait:p(E),total:p(S),pipelined:p(T),renderReadback:k}}async function Ta(r){let d=[];async function p(i,E,S){let T=t.createBuffer({size:E,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),k=t.createCommandEncoder();k.copyBufferToBuffer(i,0,T,0,E),t.queue.submit([k.finish()]),await t.queue.onSubmittedWorkDone(),await T.mapAsync(GPUMapMode.READ);let w=new Float32Array(T.getMappedRange()),c=1/0,I=-1/0,ae=0;for(let se=0;se<w.length;se++)w[se]<c&&(c=w[se]),w[se]>I&&(I=w[se]),w[se]!==0&&ae++;T.unmap(),T.destroy(),d.push({layer:S,stats:{min:c,max:I,nonZero:ae,total:w.length}})}t.queue.copyExternalImageToTexture({source:r},{texture:M},[256,256]);{let i=t.createCommandEncoder(),E=i.beginComputePass();E.setPipeline(we),E.setBindGroup(0,me),E.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),E.end(),t.queue.submit([i.finish()])}await p(Ee,Math.min(Ee.size,3*257*257*4),"canvas\u2192bufInput");{let i=t.createCommandEncoder(),E=i.beginComputePass();E.setPipeline(ht),E.setBindGroup(0,C),E.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),E.end(),t.queue.submit([i.finish()])}await p(J,Math.min(J.size,3072*128*4),"inputConv\u2192bufA");let u=!0;for(let i=0;i<Math.min(H.length,6);i++){let E=u?h[i]:_[i],S=u?D[i]:B[i],T=Ge[i],k=H[i];{let c=t.createCommandEncoder(),I=c.beginComputePass();I.setPipeline(T.dwPipeline),I.setBindGroup(0,E),I.dispatchWorkgroups(T.dwDispatchX,T.dwDispatchY,T.dwDispatchZ),I.end(),t.queue.submit([c.finish()])}await p(He,Math.min(He.size,k.spec.inCh*k.outH*k.outW*4),`layer${i}.DW\u2192bufDW (${k.spec.prefix})`);{let c=t.createCommandEncoder(),I=c.beginComputePass();I.setPipeline(T.pwPipeline),I.setBindGroup(0,S),I.dispatchWorkgroups(T.pwDispatchX,T.pwDispatchY,T.pwDispatchZ),I.end(),t.queue.submit([c.finish()])}let w=u?de:J;await p(w,Math.min(w.size,k.spec.outCh*k.outH*k.outW*4),`layer${i}.PW\u2192buf${u?"B":"A"} (${k.spec.prefix})`),u=!u}return d}return{device:t,run:bn,runFromCanvas:Bn,runFromCanvasViaRender:Nn,runFromCanvasPipelined:kn,flushPipelined:qn,benchmark:Da,benchmarkGPU:Ea,benchmarkDiagnostic:Ha,debugLayerOutputs:Ta}}function Qe(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ba=Qe(`
struct ConvParams { batch:u32, in_channels:u32, out_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read> alpha:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:ConvParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.out_width||out_y>=params.out_height||batch>=params.batch){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_height); let in_w=i32(params.in_width);
  let in_stride=params.in_height*params.in_width;
  let in_batch_base=batch*params.in_channels*in_stride;
  for(var ky:u32=0u;ky<5u;ky=ky+1u){
    let in_y=i32(out_y*2u+ky)-1;
    if(in_y<0 || in_y>=in_h){continue;}
    for(var kx:u32=0u;kx<5u;kx=kx+1u){
      let in_x=i32(out_x*2u+kx)-1;
      if(in_x<0 || in_x>=in_w){continue;}
      let pix_off=u32(in_y)*params.in_width+u32(in_x);
      // Load all 3 input channels for this pixel into vec3, dot with 3 weights
      let inp=vec3<f32>(
        input[in_batch_base+pix_off],
        input[in_batch_base+in_stride+pix_off],
        input[in_batch_base+2u*in_stride+pix_off]
      );
      let w_off=oc*75u+ky*15u+kx*3u;
      let w=vec3<f32>(weight[w_off],weight[w_off+1u],weight[w_off+2u]);
      sum+=dot(inp,w);
    }
  }
  sum=sum+bias[oc];
  // PReLU
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.out_height*params.out_width+oc*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=result;
}
`),wa=Qe(`
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
  var sum:f32=0.0;
  for(var ky:u32=0u;ky<5u;ky=ky+1u){
    let in_y=base_in_y+i32(ky);
    if(in_y>=0 && in_y<in_h){
      let row_base=in_base+u32(in_y)*params.in_width;
      for(var kx:u32=0u;kx<5u;kx=kx+1u){
        let in_x=base_in_x+i32(kx);
        if(in_x>=0 && in_x<in_w){
          sum+=input[row_base+u32(in_x)]*weight[w_base+ky*5u+kx];
        }
      }
    }
  }
  sum+=bias[c];
  let out_idx=batch*params.channels*params.out_height*params.out_width+c*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=sum;
}
`),ga=Qe(`
struct PointwiseParams { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, channel_pad:u32, stride:u32, in_height:u32, in_width:u32, }
@group(0)@binding(0) var<storage,read> dw_output:array<f32>;
@group(0)@binding(1) var<storage,read> skip_input:array<f32>;
@group(0)@binding(2) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(3) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(4) var<storage,read> alpha:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:PointwiseParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  var sum:f32=0.0;
  let dw_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels; let spatial_stride=params.height*params.width;
  // vec4 dot product accumulation: 4x fewer iterations, deterministic hardware dot()
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      dw_output[dw_base+ic*spatial_stride],
      dw_output[dw_base+(ic+1u)*spatial_stride],
      dw_output[dw_base+(ic+2u)*spatial_stride],
      dw_output[dw_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      pw_weight[w_base+ic],
      pw_weight[w_base+ic+1u],
      pw_weight[w_base+ic+2u],
      pw_weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
  }
  sum+=pw_bias[oc];
  // Skip connection: zero-pad channels
  var skip_val:f32=0.0;
  if(oc<params.channel_pad){
    if(params.stride==2u){
      var max_val:f32=-1e38;
      for(var py:u32=0u;py<2u;py=py+1u){
        for(var px:u32=0u;px<2u;px=px+1u){
          let skip_y=out_y*2u+py; let skip_x=out_x*2u+px;
          if(skip_y<params.in_height && skip_x<params.in_width){
            let skip_idx=batch*params.channel_pad*params.in_height*params.in_width+oc*params.in_height*params.in_width+skip_y*params.in_width+skip_x;
            max_val=max(max_val,skip_input[skip_idx]);
          }
        }
      }
      skip_val=max_val;
    } else {
      let skip_idx=batch*params.channel_pad*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
      skip_val=skip_input[skip_idx];
    }
  }
  let v=sum+skip_val;
  let a=alpha[oc];
  let result=max(0.0,v)+a*min(0.0,v);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),ya=Qe(`
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
  let in_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels;
  let spatial_stride=params.height*params.width;
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      input[in_base+ic*spatial_stride],
      input[in_base+(ic+1u)*spatial_stride],
      input[in_base+(ic+2u)*spatial_stride],
      input[in_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      weight[w_base+ic],
      weight[w_base+ic+1u],
      weight[w_base+ic+2u],
      weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
  }
  sum=sum+bias[oc];
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=sum;
}
`),xa=Qe(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> skip:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> params:UpsampleParams;
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
  output[out_idx]=val+skip[out_idx];
}
`),va=Qe(`
struct Conv1x1Params { batch:u32, in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read> alpha:array<f32>;
@group(0)@binding(4) var<storage,read_write> output:array<f32>;
@group(0)@binding(5) var<uniform> params:Conv1x1Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc_batch=gid.z;
  let oc=oc_batch%params.out_channels; let batch=oc_batch/params.out_channels;
  if(out_x>=params.width||out_y>=params.height||batch>=params.batch){return;}
  var sum:f32=0.0;
  let in_base=batch*params.in_channels*params.height*params.width+out_y*params.width+out_x;
  let w_base=oc*params.in_channels;
  let spatial_stride=params.height*params.width;
  let ic4=params.in_channels/4u;
  for(var i:u32=0u;i<ic4;i=i+1u){
    let ic=i*4u;
    let inp=vec4<f32>(
      input[in_base+ic*spatial_stride],
      input[in_base+(ic+1u)*spatial_stride],
      input[in_base+(ic+2u)*spatial_stride],
      input[in_base+(ic+3u)*spatial_stride]
    );
    let w=vec4<f32>(
      weight[w_base+ic],
      weight[w_base+ic+1u],
      weight[w_base+ic+2u],
      weight[w_base+ic+3u]
    );
    sum+=dot(inp,w);
  }
  sum=sum+bias[oc];
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),Pa=Qe(`
struct CanvasParams { in_width:u32, in_height:u32, out_size:u32, }
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CanvasParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y;
  if(x>=params.in_width||y>=params.in_height){return;}
  let pixel=textureLoad(input_tex,vec2<u32>(x,y),0);
  let out_stride=params.out_size*params.out_size;
  output[0u*out_stride+y*params.out_size+x]=pixel.r;
  output[1u*out_stride+y*params.out_size+x]=pixel.g;
  output[2u*out_stride+y*params.out_size+x]=pixel.b;
}
`),Ba=Qe(`
struct LBParams {
  src_w:u32, src_h:u32, dst_size:u32, _pad:u32,
  scale_x:f32, scale_y:f32, offset_x:f32, offset_y:f32,
}
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:LBParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let dx=gid.x; let dy=gid.y;
  if(dx>=params.dst_size||dy>=params.dst_size){return;}

  let out_stride=params.dst_size*params.dst_size;

  // Map dst pixel to src pixel using MediaPipe's convention:
  // dst pixel center at (dx + 0.5), offset by letterbox padding, then scale to src
  let src_x = (f32(dx) - params.offset_x + 0.5) * params.scale_x - 0.5;
  let src_y = (f32(dy) - params.offset_y + 0.5) * params.scale_y - 0.5;

  // Check if we're in the letterbox padding region
  let in_region = src_x >= -0.5 && src_x < f32(params.src_w) - 0.5
               && src_y >= -0.5 && src_y < f32(params.src_h) - 0.5;

  if(!in_region){
    // Zero padding (letterbox)
    output[0u*out_stride+dy*params.dst_size+dx]=0.0;
    output[1u*out_stride+dy*params.dst_size+dx]=0.0;
    output[2u*out_stride+dy*params.dst_size+dx]=0.0;
    return;
  }

  // Bilinear interpolation matching GL_LINEAR
  let x0=i32(floor(src_x));
  let y0=i32(floor(src_y));
  let x1=x0+1;
  let y1=y0+1;
  let fx=src_x-f32(x0);
  let fy=src_y-f32(y0);

  let sw=i32(params.src_w);
  let sh=i32(params.src_h);

  // Clamp coordinates to valid range
  let cx0=clamp(x0,0,sw-1);
  let cx1=clamp(x1,0,sw-1);
  let cy0=clamp(y0,0,sh-1);
  let cy1=clamp(y1,0,sh-1);

  let p00=textureLoad(input_tex,vec2<u32>(u32(cx0),u32(cy0)),0);
  let p10=textureLoad(input_tex,vec2<u32>(u32(cx1),u32(cy0)),0);
  let p01=textureLoad(input_tex,vec2<u32>(u32(cx0),u32(cy1)),0);
  let p11=textureLoad(input_tex,vec2<u32>(u32(cx1),u32(cy1)),0);

  let pixel=p00*(1.0-fx)*(1.0-fy) + p10*fx*(1.0-fy) + p01*(1.0-fx)*fy + p11*fx*fy;

  output[0u*out_stride+dy*params.dst_size+dx]=pixel.r;
  output[1u*out_stride+dy*params.dst_size+dx]=pixel.g;
  output[2u*out_stride+dy*params.dst_size+dx]=pixel.b;
}
`);async function Ca(o,m){let a;if(m)a=m;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");a=await n.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}})}let U={r:"read-only-storage",s:"storage",u:"uniform"};function x(n){return a.createBindGroupLayout({entries:n.map((e,s)=>e==="t"?{binding:s,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:s,visibility:GPUShaderStage.COMPUTE,buffer:{type:U[e]}})})}let t=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,z=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,W=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function f(n,e){return a.createBuffer({size:Math.max(n,4),usage:e})}function g(n,e,s){a.queue.writeBuffer(n,e,s)}function A(n){let e=f(n.data.byteLength,t);return g(e,0,n.data),e}let L=Array.from(o.keys());function oe(n){let e=o.get(n);if(!e)throw new Error(`Weight not found: ${n}`);return e}function y(...n){let e=L.find(s=>n.every(H=>s.includes(H)));if(!e)throw new Error(`Weight not found for: ${n.join(", ")}`);return oe(e)}function ie(n){let[,e,s,H]=n.shape,O=new Float32Array(H*25);for(let l=0;l<H;l++)for(let N=0;N<e;N++)for(let ee=0;ee<s;ee++)O[l*25+N*5+ee]=n.data[N*s*H+ee*H+l];return O}function R(n){let[e,,,s]=n.shape,H=new Float32Array(e*s);for(let O=0;O<e;O++)for(let l=0;l<s;l++)H[O*s+l]=n.data[O*s+l];return H}let _e=a.createShaderModule({code:ba}),xe=a.createShaderModule({code:wa}),ke=a.createShaderModule({code:ga}),re=a.createShaderModule({code:ya}),G=a.createShaderModule({code:va}),Z=a.createShaderModule({code:xa}),ue=a.createShaderModule({code:Pa}),Ut=a.createShaderModule({code:Ba}),et=x(["r","r","r","r","s","u"]),We=x(["r","r","r","s","u"]),Le=x(["r","r","r","r","r","s","u"]),At=x(["r","r","r","s","u"]),tt=x(["r","r","r","r","s","u"]),ge=x(["r","r","s","u"]),Ke=x(["t","s","u"]),_t=x(["t","s","u"]);function Ue(n,e){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:e,entryPoint:"main"}})}let lt=Ue(et,_e),Zt=Ue(We,xe),P=Ue(Le,ke),F=Ue(At,re),X=Ue(tt,G),K=Ue(ge,Z),Q=Ue(Ke,ue),he=Ue(_t,Ut),ne=y("conv2d/Conv2D"),$=y("batch_normalization/","conv2d/Conv2D"),ye=y("p_re_lu/"),j=A(ne),be=A($),pe=A(ye),ce=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(n=>{let e=y(n.dwKey),s=y(n.pwKey),H=y(n.bnKey),O=y(n.preluKey),l=ie(e),N=f(l.byteLength,t);g(N,0,l);let ee=new Float32Array(n.inCh),b=f(ee.byteLength,t);g(b,0,ee);let C=R(s),h=f(C.byteLength,t);g(h,0,C);let D=A(H),_=A(O);return{dwWeightBuf:N,dwBiasBuf:b,pwWeightBuf:h,pwBiasBuf:D,alphaBuf:_,inCh:n.inCh,outCh:n.outCh,stride:n.stride,inH:n.inH}}),ve=R(y("conv2d_25/Conv2D")),Pe=f(ve.byteLength,t);g(Pe,0,ve);let Ae=A(y("batch_normalization_25/")),Oe=A(y("p_re_lu_25/")),Me={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_24/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(256),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_26/")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_26/")),alphaBuf:A(y("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},le={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_25/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(256),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_27/Conv2D1")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_27/")),alphaBuf:A(y("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},De=R(y("conv2d_28/Conv2D")),Ne=f(De.byteLength,t);g(Ne,0,De);let qe=A(y("batch_normalization_28/")),nt=A(y("p_re_lu_28/")),Ye={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_26/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(128),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_29/")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_29/")),alphaBuf:A(y("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},Xe={dwWeightBuf:(()=>{let n=ie(y("depthwise_conv2d_27/")),e=f(n.byteLength,t);return g(e,0,n),e})(),dwBiasBuf:(()=>{let n=new Float32Array(128),e=f(n.byteLength,t);return g(e,0,n),e})(),pwWeightBuf:(()=>{let n=R(y("conv2d_30/Conv2D1")),e=f(n.byteLength,t);return g(e,0,n),e})(),pwBiasBuf:A(y("batch_normalization_30/")),alphaBuf:A(y("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},at=R(y("classifier_palm_16_NO_PRUNING/Conv2D")),it=f(at.byteLength,t);g(it,0,at);let mt=A(y("classifier_palm_16_NO_PRUNING/BiasAdd")),rt=R(y("regressor_palm_16_NO_PRUNING/Conv2D")),Gt=f(rt.byteLength,t);g(Gt,0,rt);let jt=A(y("regressor_palm_16_NO_PRUNING/BiasAdd")),Jt=R(y("classifier_palm_8_NO_PRUNING/Conv2D")),ft=f(Jt.byteLength,t);g(ft,0,Jt);let Ge=A(y("classifier_palm_8_NO_PRUNING/BiasAdd")),St=R(y("regressor_palm_8_NO_PRUNING/Conv2D")),ht=f(St.byteLength,t);g(ht,0,St);let bt=A(y("regressor_palm_8_NO_PRUNING/BiasAdd")),Mt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,we=f(36864*3*4,t),st=f(Mt,v),Qt=f(Mt,v),wt=f(Mt,v),ot=f(576*256*4,v),Ee=f(144*256*4,v|GPUBufferUsage.COPY_DST),$e=f(576*128*4,v|GPUBufferUsage.COPY_DST),J=f(864*4,z),de=f(15552*4,z),He=f(576*2*4,z),ut=f(576*36*4,z),pt=f(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ct=f(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),dt=f(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Te=f(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),V=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function M(n,e){return Math.ceil(n/e)}function q(n){let e=f(n.byteLength,W);return g(e,0,n),e}let Dt=q(new Uint32Array([1,3,32,192,192,96,96])),Et=ce.map(n=>{let e=n.stride===2?n.inH/2:n.inH,s=e,H=n.stride===2?1:2,O=n.inCh;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,s,n.stride,H])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,s,O,n.stride,n.inH,n.inH])),outH:e,outW:s}}),Ht=(()=>{let n=Me,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),Tt=(()=>{let n=le,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),Wt=(()=>{let n=Ye,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),Lt=(()=>{let n=Xe,e=n.stride===2?n.inH/2:n.inH,s=n.stride===2?1:2;return{dw:q(new Uint32Array([1,n.inCh,n.inH,n.inH,e,e,n.stride,s])),pw:q(new Uint32Array([1,n.inCh,n.outCh,e,e,n.inCh,n.stride,n.inH,n.inH])),outH:e}})(),en=q(new Uint32Array([1,256,6,6,12,12])),tn=q(new Uint32Array([1,256,12,12,12,12])),nn=q(new Uint32Array([1,256,12,12,24,24])),an=q(new Uint32Array([1,128,24,24,24,24])),Rt=q(new Uint32Array([1,256,256,12,12])),Ot=q(new Uint32Array([1,256,128,24,24])),zt=q(new Uint32Array([1,256,6,12,12])),It=q(new Uint32Array([1,256,108,12,12])),Ft=q(new Uint32Array([1,128,2,24,24])),Kt=q(new Uint32Array([1,128,36,24,24])),rn=q(new Uint32Array([192,192,192])),sn=a.createBindGroup({layout:Ke,entries:[{binding:0,resource:V.createView()},{binding:1,resource:{buffer:we}},{binding:2,resource:{buffer:rn}}]}),ze=null,gt=0,Nt=0,qt=f(32,W);function on(n,e){return ze&&gt===n&&Nt===e||(ze&&ze.destroy(),ze=a.createTexture({size:[n,e,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),gt=n,Nt=e),ze}let un=a.createBindGroup({layout:et,entries:[{binding:0,resource:{buffer:we}},{binding:1,resource:{buffer:j}},{binding:2,resource:{buffer:be}},{binding:3,resource:{buffer:pe}},{binding:4,resource:{buffer:st}},{binding:5,resource:{buffer:Dt}}]});function Be(n,e,s,H,O,l){let N=l.outH,ee=a.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:e.dwWeightBuf}},{binding:2,resource:{buffer:e.dwBiasBuf}},{binding:3,resource:{buffer:wt}},{binding:4,resource:{buffer:l.dw}}]}),b=n.beginComputePass();b.setPipeline(Zt),b.setBindGroup(0,ee),b.dispatchWorkgroups(M(N,8),M(l.outH,8),e.inCh),b.end();let C=a.createBindGroup({layout:Le,entries:[{binding:0,resource:{buffer:wt}},{binding:1,resource:{buffer:O}},{binding:2,resource:{buffer:e.pwWeightBuf}},{binding:3,resource:{buffer:e.pwBiasBuf}},{binding:4,resource:{buffer:e.alphaBuf}},{binding:5,resource:{buffer:H}},{binding:6,resource:{buffer:l.pw}}]}),h=n.beginComputePass();h.setPipeline(P),h.setBindGroup(0,C),h.dispatchWorkgroups(M(N,8),M(l.outH,8),e.outCh),h.end()}function Se(n,e,s,H,O,l,N,ee,b){let C=a.createBindGroup({layout:At,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:H}},{binding:3,resource:{buffer:O}},{binding:4,resource:{buffer:l}}]}),h=n.beginComputePass();h.setPipeline(F),h.setBindGroup(0,C),h.dispatchWorkgroups(M(b,8),M(ee,8),N),h.end()}function Ve(n,e,s,H,O,l,N,ee,b,C){let h=a.createBindGroup({layout:tt,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:H}},{binding:3,resource:{buffer:O}},{binding:4,resource:{buffer:l}},{binding:5,resource:{buffer:N}}]}),D=n.beginComputePass();D.setPipeline(X),D.setBindGroup(0,h),D.dispatchWorkgroups(M(C,8),M(b,8),ee),D.end()}async function Yt(n){{let _=n.beginComputePass();_.setPipeline(lt),_.setBindGroup(0,un),_.dispatchWorkgroups(M(96,8),M(96,8),32),_.end()}let e=st,s=Qt;for(let _=0;_<ce.length;_++){let B=ce[_];Be(n,B,e,s,e,Et[_]);let te=e;e=s,s=te,_===13&&n.copyBufferToBuffer(e,0,$e,0,576*128*4),_===18&&n.copyBufferToBuffer(e,0,Ee,0,144*256*4)}{let _=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:ot}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:en}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,_),B.dispatchWorkgroups(M(12,8),M(12,8),256),B.end()}{let _=e;e=s,s=_}Ve(n,e,Pe,Ae,Oe,s,Rt,256,12,12);{let _=e;e=s,s=_}{let _=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:Ee}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:tn}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,_),B.dispatchWorkgroups(M(12,8),M(12,8),256),B.end()}{let _=e;e=s,s=_}Be(n,Me,e,s,e,Ht);{let _=e;e=s,s=_}Be(n,le,e,s,e,Tt);{let _=e;e=s,s=_}Se(n,e,it,mt,J,zt,6,12,12),Se(n,e,Gt,jt,de,It,108,12,12);{let _=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:ot}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:nn}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,_),B.dispatchWorkgroups(M(24,8),M(24,8),256),B.end()}{let _=e;e=s,s=_}Ve(n,e,Ne,qe,nt,s,Ot,128,24,24);{let _=e;e=s,s=_}{let _=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:$e}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:an}}]}),B=n.beginComputePass();B.setPipeline(K),B.setBindGroup(0,_),B.dispatchWorkgroups(M(24,8),M(24,8),128),B.end()}{let _=e;e=s,s=_}Be(n,Ye,e,s,e,Wt);{let _=e;e=s,s=_}Be(n,Xe,e,s,e,Lt);{let _=e;e=s,s=_}Se(n,e,ft,Ge,He,Ft,2,24,24),Se(n,e,ht,bt,ut,Kt,36,24,24),a.queue.submit([n.finish()]);let H=a.createCommandEncoder();H.copyBufferToBuffer(J,0,pt,0,864*4),H.copyBufferToBuffer(de,0,ct,0,15552*4),H.copyBufferToBuffer(He,0,dt,0,576*2*4),H.copyBufferToBuffer(ut,0,Te,0,576*36*4),a.queue.submit([H.finish()]),await Promise.all([pt.mapAsync(GPUMapMode.READ),ct.mapAsync(GPUMapMode.READ),dt.mapAsync(GPUMapMode.READ),Te.mapAsync(GPUMapMode.READ)]);let O=new Float32Array(pt.getMappedRange()).slice(),l=new Float32Array(ct.getMappedRange()).slice(),N=new Float32Array(dt.getMappedRange()).slice(),ee=new Float32Array(Te.getMappedRange()).slice();pt.unmap(),ct.unmap(),dt.unmap(),Te.unmap();let b=2016,C=new Float32Array(b),h=new Float32Array(b*18),D=0;for(let _=0;_<12;_++)for(let B=0;B<12;B++)for(let te=0;te<6;te++){C[D]=O[te*144+_*12+B];for(let Ce=0;Ce<18;Ce++){let Ze=te*18+Ce;h[D*18+Ce]=l[Ze*144+_*12+B]}D++}for(let _=0;_<24;_++)for(let B=0;B<24;B++)for(let te=0;te<2;te++){C[D]=N[te*576+_*24+B];for(let Ce=0;Ce<18;Ce++){let Ze=te*18+Ce;h[D*18+Ce]=ee[Ze*576+_*24+B]}D++}return{scores:C,regressors:h}}async function pn(n){a.queue.copyExternalImageToTexture({source:n},{texture:V},[192,192]);let e=a.createCommandEncoder();{let s=e.beginComputePass();s.setPipeline(Q),s.setBindGroup(0,sn),s.dispatchWorkgroups(M(192,16),M(192,16),1),s.end()}return Yt(e)}async function cn(n,e,s){let H=Math.min(192/e,192/s),O=Math.round(e*H),l=Math.round(s*H),N=(192-O)/2,ee=(192-l)/2,b=N/192,C=ee/192,h=on(e,s),D;if(n instanceof HTMLVideoElement||n instanceof HTMLImageElement){let me=new OffscreenCanvas(e,s);me.getContext("2d").drawImage(n,0,0),D=me}else D=n;a.queue.copyExternalImageToTexture({source:D},{texture:h},[e,s]);let _=new ArrayBuffer(32),B=new Uint32Array(_),te=new Float32Array(_);B[0]=e,B[1]=s,B[2]=192,B[3]=0,te[4]=e/O,te[5]=s/l,te[6]=N,te[7]=ee,a.queue.writeBuffer(qt,0,_);let Ce=a.createBindGroup({layout:_t,entries:[{binding:0,resource:h.createView()},{binding:1,resource:{buffer:we}},{binding:2,resource:{buffer:qt}}]}),Ze=a.createCommandEncoder();{let me=Ze.beginComputePass();me.setPipeline(he),me.setBindGroup(0,Ce),me.dispatchWorkgroups(M(192,16),M(192,16),1),me.end()}return{output:await Yt(Ze),lbPadX:b,lbPadY:C}}async function Y(n,e){let s=a.createBuffer({size:e*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),H=a.createCommandEncoder();H.copyBufferToBuffer(n,0,s,0,e*4),a.queue.submit([H.finish()]),await s.mapAsync(GPUMapMode.READ);let O=new Float32Array(s.getMappedRange()).slice();return s.unmap(),s.destroy(),O}async function dn(n){a.queue.copyExternalImageToTexture({source:n},{texture:V},[192,192]);function e(h,D=1e3){let _=h.slice(0,D);return{min:Math.min(..._),max:Math.max(..._),mean:_.reduce((B,te)=>B+te,0)/_.length,nonZero:_.filter(B=>B!==0).length,sample:Array.from(_.slice(0,10))}}let s={},H=q(new Uint32Array([192,192,192])),O=a.createBindGroup({layout:Ke,entries:[{binding:0,resource:V.createView()},{binding:1,resource:{buffer:we}},{binding:2,resource:{buffer:H}}]}),l=a.createCommandEncoder(),N=l.beginComputePass();N.setPipeline(Q),N.setBindGroup(0,O),N.dispatchWorkgroups(M(192,16),M(192,16),1),N.end(),a.queue.submit([l.finish()]),s.input=e(await Y(we,36864*3)),l=a.createCommandEncoder();let ee=a.createBindGroup({layout:et,entries:[{binding:0,resource:{buffer:we}},{binding:1,resource:{buffer:j}},{binding:2,resource:{buffer:be}},{binding:3,resource:{buffer:pe}},{binding:4,resource:{buffer:st}},{binding:5,resource:{buffer:Dt}}]});N=l.beginComputePass(),N.setPipeline(lt),N.setBindGroup(0,ee),N.dispatchWorkgroups(M(96,8),M(96,8),32),N.end(),a.queue.submit([l.finish()]),s.initConv=e(await Y(st,9216*32));let b=st,C=Qt;for(let h=0;h<ce.length;h++){let D=ce[h];l=a.createCommandEncoder(),Be(l,D,b,C,b,Et[h]),a.queue.submit([l.finish()]);let _=b;if(b=C,C=_,h===0||h===4||h===9||h===14||h===18||h===19||h===23){let B=D.stride===2?D.inH/2:D.inH,te=B*B*D.outCh;s[`block${h}`]=e(await Y(b,te))}h===13&&(l=a.createCommandEncoder(),l.copyBufferToBuffer(b,0,$e,0,576*128*4),a.queue.submit([l.finish()])),h===18&&(l=a.createCommandEncoder(),l.copyBufferToBuffer(b,0,Ee,0,144*256*4),a.queue.submit([l.finish()]))}l=a.createCommandEncoder();{let h=q(new Uint32Array([1,256,6,6,12,12])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:ot}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),_=l.beginComputePass();_.setPipeline(K),_.setBindGroup(0,D),_.dispatchWorkgroups(M(12,8),M(12,8),256),_.end()}a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpnUpsample6to12=e(await Y(b,144*256)),l=a.createCommandEncoder(),Ve(l,b,Pe,Ae,Oe,C,Rt,256,12,12),a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpn6to12Conv=e(await Y(b,144*256)),s.backbone12Skip=e(await Y(Ee,144*256)),l=a.createCommandEncoder();{let h=q(new Uint32Array([1,256,12,12,12,12])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:Ee}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),_=l.beginComputePass();_.setPipeline(K),_.setBindGroup(0,D),_.dispatchWorkgroups(M(12,8),M(12,8),256),_.end()}a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpnAdd12=e(await Y(b,144*256)),l=a.createCommandEncoder(),Be(l,Me,b,C,b,Ht),a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpn12Block1=e(await Y(b,144*256)),l=a.createCommandEncoder(),Be(l,le,b,C,b,Tt),a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpn12Block2=e(await Y(b,144*256)),l=a.createCommandEncoder(),Se(l,b,it,mt,J,zt,6,12,12),a.queue.submit([l.finish()]),s.cls16=e(await Y(J,864)),l=a.createCommandEncoder(),Se(l,b,Gt,jt,de,It,108,12,12),a.queue.submit([l.finish()]),s.reg16=e(await Y(de,15552),500),l=a.createCommandEncoder();{let h=q(new Uint32Array([1,256,12,12,24,24])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:ot}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),_=l.beginComputePass();_.setPipeline(K),_.setBindGroup(0,D),_.dispatchWorkgroups(M(24,8),M(24,8),256),_.end()}a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpnUpsample12to24=e(await Y(b,576*256)),l=a.createCommandEncoder(),Ve(l,b,Ne,qe,nt,C,Ot,128,24,24),a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpn12to24Conv=e(await Y(b,576*128)),s.backbone24Skip=e(await Y($e,576*128)),l=a.createCommandEncoder();{let h=q(new Uint32Array([1,128,24,24,24,24])),D=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:$e}},{binding:2,resource:{buffer:C}},{binding:3,resource:{buffer:h}}]}),_=l.beginComputePass();_.setPipeline(K),_.setBindGroup(0,D),_.dispatchWorkgroups(M(24,8),M(24,8),128),_.end()}a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpnAdd24=e(await Y(b,576*128)),l=a.createCommandEncoder(),Be(l,Ye,b,C,b,Wt),a.queue.submit([l.finish()]);{let h=b;b=C,C=h}s.fpn24Block1=e(await Y(b,576*128)),l=a.createCommandEncoder(),Be(l,Xe,b,C,b,Lt),a.queue.submit([l.finish()]);{let h=b;b=C,C=h}return s.fpn24Block2=e(await Y(b,576*128)),l=a.createCommandEncoder(),Se(l,b,ft,Ge,He,Ft,2,24,24),a.queue.submit([l.finish()]),s.cls8=e(await Y(He,576*2)),l=a.createCommandEncoder(),Se(l,b,ht,bt,ut,Kt,36,24,24),a.queue.submit([l.finish()]),s.reg8=e(await Y(ut,576*36)),s.initWeights=e(await Y(j,100),100),s.initBias=e(await Y(be,32),32),s.cls16Weights=e(await Y(it,100),100),s.cls16Bias=e(await Y(mt,6),6),s.cls8Weights=e(await Y(ft,100),100),s.cls8Bias=e(await Y(Ge,2),2),s.fpn6to12Weights=e(await Y(Pe,100),100),s}return{device:a,run:pn,runWithResize:cn,debugRun:dn}}function $a(){let o=[];for(let m=0;m<12;m++)for(let a=0;a<12;a++){let U=(a+.5)/12,x=(m+.5)/12;for(let t=0;t<6;t++)o.push({x:U,y:x})}for(let m=0;m<24;m++)for(let a=0;a<24;a++){let U=(a+.5)/24,x=(m+.5)/24;for(let t=0;t<2;t++)o.push({x:U,y:x})}return o}var ka=$a();function Va(o){return 1/(1+Math.exp(-o))}function En(o,m){let a=[],{scores:U,regressors:x}=o,t=192;for(let v=0;v<ka.length;v++){let z=Va(U[v]);if(z<m)continue;let W=ka[v],f=v*18,g=W.x+x[f+0]/t,A=W.y+x[f+1]/t,L=x[f+2]/t,oe=x[f+3]/t,y=[];for(let ie=0;ie<7;ie++){let R=W.x+x[f+4+ie*2]/t,_e=W.y+x[f+4+ie*2+1]/t;y.push([R,_e])}a.push({score:z,box:[g,A,L,oe],keypoints:y})}return a}function Hn(o,m){if(o.length===0)return[];let a=[...o].sort((t,v)=>v.score-t.score),U=[],x=new Set;for(let t=0;t<a.length;t++)if(!x.has(t)){U.push(a[t]);for(let v=t+1;v<a.length;v++)x.has(v)||Za(a[t],a[v])>m&&x.add(v)}return U}function Za(o,m){let a=o.box[0]-o.box[2]/2,U=o.box[1]-o.box[3]/2,x=o.box[0]+o.box[2]/2,t=o.box[1]+o.box[3]/2,v=m.box[0]-m.box[2]/2,z=m.box[1]-m.box[3]/2,W=m.box[0]+m.box[2]/2,f=m.box[1]+m.box[3]/2,g=Math.max(a,v),A=Math.max(U,z),L=Math.min(x,W),oe=Math.min(t,f),y=Math.max(0,L-g),ie=Math.max(0,oe-A),R=y*ie,_e=(x-a)*(t-U),xe=(W-v)*(f-z),ke=_e+xe-R;return ke>0?R/ke:0}function ja(o){let[m,a,U,x]=o.box,t=o.keypoints[0],v=o.keypoints[2],z=v[0]-t[0],W=v[1]-t[1],f=Math.atan2(W,z),A=-Math.PI/2-f,L=Math.max(U,x),y=L*2.6,ie=-.5*L,R=Math.cos(A),_e=Math.sin(A),xe=ie*_e,ke=ie*R;return{centerX:m+xe,centerY:a+ke,width:y,height:y,rotation:A}}function Ua(o,m={}){let{scoreThreshold:a=.5,nmsThreshold:U=.3,maxHands:x=2}=m;async function t(W){let f=await o.run(W),g=En(f,a);return Hn(g,U).slice(0,x).map(ja)}async function v(W){let f=await o.run(W),g=En(f,a);return Hn(g,U).slice(0,x)}async function z(W,f,g){let{output:A,lbPadX:L,lbPadY:oe}=await o.runWithResize(W,f,g),y=En(A,a);return{detections:Hn(y,U).slice(0,x),lbPadX:L,lbPadY:oe}}return{detect:t,detectRaw:v,detectRawWithResize:z,model:o}}var Tn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Wn(o){let m={};for(let a=0;a<Tn.length;a++)m[Tn[a]]=o[a];return m}function Aa(o,m,a){return o.initialized?(o.value=a*m+(1-a)*o.value,o.value):(o.value=m,o.initialized=!0,m)}function Ga(o,m){let a=2*Math.PI*m*o;return a/(a+1)}function Ja(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function Ln(o,m,a,U,x,t){let v=o.lastTime<0?.03333333333333333:a-o.lastTime;o.lastTime=a;let z=Ga(v,t),W=o.x.initialized?(m-o.x.value)/v:0,f=Aa(o.dx,W,z),g=U+x*Math.abs(f),A=Ga(v,g);return Aa(o.x,m,A)}function Rn(o={}){let{minCutoff:m=.05,beta:a=80,dCutoff:U=1}=o,x=[];function t(W){x.length!==W&&(x=Array.from({length:W},()=>Ja()))}function v(W,f){let g=f??performance.now()/1e3,A=W.length*3;return t(A),W.map((L,oe)=>({x:Ln(x[oe*3],L.x,g,m,a,U),y:Ln(x[oe*3+1],L.y,g,m,a,U),z:Ln(x[oe*3+2],L.z,g,m,a,U)}))}function z(){x=[]}return{apply:v,reset:z}}var Qa="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ei(o={}){let{weightsUrl:m,scoreThreshold:a=.5,palmScoreThreshold:U=.5,maxHands:x=3,forceF32:t=!1}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let v=(m??Qa).replace(/\/$/,"")+"/",[z,W,f,g]=await Promise.all([fetch(`${v}weights_f16.json`),fetch(`${v}weights_f16.bin`),fetch(`${v}palm_detection_weights.json`),fetch(`${v}palm_detection_weights.bin`)]);if(!z.ok)throw new Error(`Failed to fetch landmark weights: ${z.status}`);if(!W.ok)throw new Error(`Failed to fetch landmark weights: ${W.status}`);if(!f.ok)throw new Error(`Failed to fetch palm detection weights: ${f.status}`);if(!g.ok)throw new Error(`Failed to fetch palm detection weights: ${g.status}`);let[A,L,oe,y]=await Promise.all([z.json(),W.arrayBuffer(),f.json(),g.arrayBuffer()]),ie=Mn(A,L),R=Mn(oe,y),_e=await Dn(ie,{forceF32:t});if(!t){let P=new OffscreenCanvas(256,256),F=P.getContext("2d");F.fillStyle="#886644",F.fillRect(0,0,256,256),F.fillStyle="#cc9966",F.fillRect(50,50,156,156);let X=await _e.runFromCanvas(P);X.landmarks.every(Q=>Q===0)&&X.handflag.every(Q=>Q===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),_e.device.destroy(),_e=await Dn(ie,{forceF32:!0}))}let xe=await Ca(R),ke=Ua(xe,{scoreThreshold:U,maxHands:x}),re=[];for(let P=0;P<x;P++)re.push(Rn());let G=0,Z=null,ue=null;function Ut(){return Z||(Z=new OffscreenCanvas(192,192)),Z}function et(){return ue||(ue=new OffscreenCanvas(256,256)),ue}let We=0,Le=0;function At(P,F,X){let K=Ut();K.width=192,K.height=192;let Q=K.getContext("2d");Q.clearRect(0,0,192,192);let he=Math.min(192/F,192/X),ne=Math.round(F*he),$=Math.round(X*he),ye=(192-ne)/2,j=(192-$)/2;if(We=ye/192,Le=j/192,P instanceof ImageData){let be=new OffscreenCanvas(P.width,P.height);be.getContext("2d").putImageData(P,0,0),Q.drawImage(be,ye,j,ne,$)}else Q.drawImage(P,0,0,F,X,ye,j,ne,$);return K}function tt(P){let F=1/(1-2*We),X=1/(1-2*Le);return{score:P.score,box:[(P.box[0]-We)*F,(P.box[1]-Le)*X,P.box[2]*F,P.box[3]*X],keypoints:P.keypoints.map(([K,Q])=>[(K-We)*F,(Q-Le)*X])}}function ge(P,F,X){let K=P.keypoints[0],Q=P.keypoints[2],he=Q[0]-K[0],ne=Q[1]-K[1],$=Math.atan2(-ne,he),j=Math.PI/2-$,[be,pe,Re,ce]=P.box,ve=Math.max(Re*F,ce*X),Pe=0,Ae=-.5*ve/X,Oe=Math.cos(j),Me=Math.sin(j),le=be+(Pe*Oe-Ae*Me),De=pe+(Pe*Me+Ae*Oe),qe=ve*2.6;return{centerXpx:le*F,centerYpx:De*X,sizePx:qe,rotation:j}}function Ke(P,F){let X=et();X.width=256,X.height=256;let K=X.getContext("2d");K.clearRect(0,0,256,256);let Q=256/F.sizePx,he=Math.cos(F.rotation),ne=Math.sin(F.rotation),$=he*Q,ye=-ne*Q,j=ne*Q,be=he*Q,pe=-F.centerXpx*$-F.centerYpx*j+128,Re=-F.centerXpx*ye-F.centerYpx*be+128;if(K.setTransform($,ye,j,be,pe,Re),P instanceof ImageData){let ce=new OffscreenCanvas(P.width,P.height);ce.getContext("2d").putImageData(P,0,0),K.drawImage(ce,0,0)}else K.drawImage(P,0,0);return K.setTransform(1,0,0,1,0,0),X}function _t(P){return P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas?[P.width,P.height]:typeof ImageBitmap<"u"&&P instanceof ImageBitmap?[P.width,P.height]:P instanceof ImageData?[P.width,P.height]:P instanceof HTMLVideoElement?[P.videoWidth,P.videoHeight]:P instanceof HTMLImageElement?[P.naturalWidth,P.naturalHeight]:[256,256]}async function Ue(P){let[F,X]=_t(P),{detections:K,lbPadX:Q,lbPadY:he}=await ke.detectRawWithResize(P,F,X);if(We=Q,Le=he,K.length===0){if(G>0)for(let $=0;$<G&&$<re.length;$++)re[$].reset();return G=0,[]}let ne=[];for(let $ of K){let ye=tt($),j=ge(ye,F,X),be=Ke(P,j),pe=await _e.runFromCanvas(be),Re=pe.handflag[0];if(Re<a)continue;let ce=pe.handedness[0]>.5,ve=[],Pe=Math.cos(j.rotation),Ae=Math.sin(j.rotation);for(let le=0;le<21;le++){let De=pe.landmarks[le*3],Ne=pe.landmarks[le*3+1],qe=pe.landmarks[le*3+2],nt=(De-.5)*j.sizePx,Ye=(Ne-.5)*j.sizePx,Xe=Pe*nt-Ae*Ye+j.centerXpx,at=Ae*nt+Pe*Ye+j.centerYpx;ve.push({x:Xe/F,y:at/X,z:qe})}let Oe=ne.length,Me=Oe<re.length?re[Oe].apply(ve):ve;ne.push({score:Re,handedness:ce?"right":"left",landmarks:Me,keypoints:Wn(Me)})}if(ne.length<G)for(let $=ne.length;$<G;$++)$<re.length&&re[$].reset();return G=ne.length,ne}function lt(){_e.device.destroy(),xe.device.destroy(),Z=null,ue=null}return{detect:Ue,dispose:lt,_debug:{palmDetector:ke,palmModel:xe,landmarkModel:_e,removeLetterbox:tt,detectionToPixelROI:ge,cropHandRegion:Ke}}}export{Tn as LANDMARK_NAMES,ei as createHandpose,Rn as createLandmarkSmoother,Wn as toKeypoints};
