function fe(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function $n(r){let m=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],a="enable f16;"+r;for(let C of m)for(;a.includes(`${C}:array<f32>`);)a=a.replace(`${C}:array<f32>`,`${C}:array<f16>`);return a}var kn=fe(`
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
`),Gn=fe(`
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
`),An=fe(`
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
`);function Vn(r,m){return Gn.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${m},1)`)}function Zn(r,m){return kn.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${m},1)`)}function jn(r,m){return An.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${m},1)`)}function Jn(r,m){return Sn.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${r},${m},1)`)}function Qn(r,m){return[8,8]}var ea=fe(`
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
`);function na(r){return fe(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${r?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${r?"+skip[out_idx]":""};
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
`);function oa(r){return fe(`
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
  ${r==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
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
`);function la(r,m){let C=Math.min(m,256),x=m>C,v=r%4===0?`var ic:u32=0u;
    while(ic<${r}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${r}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,I=`var skip_val:f32=0.0;
    if(c<${r}u){
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
    }`,L=r===m?"":`if(c<${r}u){`,f=r===m?"":"}",w=x?`for(var c:u32=lid.x;c<${r}u;c+=${C}u){`:`let c=lid.x;
  ${L}`,G=x?"}":f,W=x?`for(var c:u32=lid.x;c<${m}u;c+=${C}u){`:"{let c=lid.x;";return fe(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${r}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${C},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${w}
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
  ${G}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${W}
    let pw_base=c*${r}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${v}
    // Skip connection (only for c < inCh)
    ${I}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var _a=fe(`
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
`);function Mn(r,m){let a=new Map,C=r.dtype??"float32";for(let x=0;x<r.keys.length;x++){let e=r.keys[x],v=r.shapes[x],I=r.offsets[x],L=v.reduce((G,W)=>G*W,1),f,w;if(C==="float32")f=new Float32Array(m,I,L);else{let G=new DataView(m);f=new Float32Array(L);for(let W=0;W<L;W++)f[W]=Na(G.getUint16(I+W*2,!0));w=m.slice(I,I+L*2)}a.set(e,{data:f,shape:v,rawF16:w})}return a}function Na(r){let m=r>>15&1,a=r>>10&31,C=r&1023;if(a===0){if(C===0)return m?-0:0;let v=-14,I=C/1024;return(m?-1:1)*Math.pow(2,v)*I}if(a===31)return C===0?m?-1/0:1/0:NaN;let x=a-15,e=1+C/1024;return(m?-1:1)*Math.pow(2,x)*e}var qa=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],ha=qa.map(([r,m,a,C,x])=>({type:"resmodule",inCh:r,outCh:m,h:a,w:a,stride:C,prefix:x})),Ya=2,Xa=5,$a=8,Va=11;async function En(r,m){if(!navigator.gpu)throw new Error("WebGPU not supported");let a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No GPU adapter found");let C=a.features.has("shader-f16"),x=C?["shader-f16"]:[],e=await a.requestDevice({requiredFeatures:x,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(a.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(a.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(a.limits.maxComputeInvocationsPerWorkgroup,288)}}),v=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(C)try{let s=`enable f16;
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
}`,p=e.createShaderModule({code:s}),u=e.createShaderModule({code:d}),i=await p.getCompilationInfo(),H=await u.getCompilationInfo();if(i.messages.some(S=>S.type==="error")||H.messages.some(S=>S.type==="error"))v=!1;else{let S=new Float32Array(2400);S.fill(1);let O=new Uint16Array(2400);O.fill(10516);let k=new Uint16Array(96);k.fill(14336);let y=new Uint16Array(9216);y.fill(8478);let c=new Uint16Array(96);c.fill(12288);let K=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ue=e.createBuffer({size:S.byteLength,usage:K}),At=e.createBuffer({size:O.byteLength,usage:K}),St=e.createBuffer({size:k.byteLength,usage:K}),Mt=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),Et=e.createBuffer({size:y.byteLength,usage:K}),Dt=e.createBuffer({size:c.byteLength,usage:K}),Tt=e.createBuffer({size:384,usage:ae}),ct=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(ue,0,S),e.queue.writeBuffer(At,0,O),e.queue.writeBuffer(St,0,k),e.queue.writeBuffer(Et,0,y),e.queue.writeBuffer(Dt,0,c);let it="read-only-storage",Qt="storage",en=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:it}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:it}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:it}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Qt}}]}),Yn=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:it}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:it}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:it}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:Qt}}]}),Ra=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[en]}),compute:{module:p,entryPoint:"main"}}),Oa=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Yn]}),compute:{module:u,entryPoint:"main"}}),za=e.createBindGroup({layout:en,entries:[{binding:0,resource:{buffer:ue}},{binding:1,resource:{buffer:At}},{binding:2,resource:{buffer:St}},{binding:3,resource:{buffer:Mt}}]}),Ia=e.createBindGroup({layout:Yn,entries:[{binding:0,resource:{buffer:Mt}},{binding:1,resource:{buffer:Et}},{binding:2,resource:{buffer:Dt}},{binding:3,resource:{buffer:Tt}}]}),wn=e.createCommandEncoder(),gn=wn.beginComputePass();gn.setPipeline(Ra),gn.setBindGroup(0,za),gn.dispatchWorkgroups(2),gn.end();let yn=wn.beginComputePass();yn.setPipeline(Oa),yn.setBindGroup(0,Ia),yn.dispatchWorkgroups(2),yn.end(),wn.copyBufferToBuffer(Tt,0,ct,0,384),e.queue.submit([wn.finish()]),await e.queue.onSubmittedWorkDone(),await ct.mapAsync(GPUMapMode.READ);let tn=new Float32Array(ct.getMappedRange()),Xn=1.5*.0104*96+.25,Fa=tn[0]!==0&&tn[47]!==0&&tn[95]!==0,Ka=Math.abs(tn[0]-Xn)<1;v=Fa&&Ka,ct.unmap(),ue.destroy(),At.destroy(),St.destroy(),Mt.destroy(),Et.destroy(),Dt.destroy(),Tt.destroy(),ct.destroy(),v||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${tn[0]}, expected ~${Xn.toFixed(2)}) \u2014 falling back to f32`)}}catch{v=!1}let L=r.values().next().value,f=v&&!!L?.rawF16&&!m?.forceF32;console.log(f?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${C}, f16 validated: ${v}, f16 data: ${!!L?.rawF16})`);function w(s){if(f&&s.rawF16){let d=new Uint16Array(s.rawF16);if(d.length%2!==0){let p=new Uint16Array(d.length+1);return p.set(d),p}return d}return s.data}function G(s){if(f&&s.rawF16){let d=s.rawF16.byteLength;return Math.ceil(d/4)*4}return s.data.byteLength}function W(s){return f?$n(s):s}let ie={r:"read-only-storage",s:"storage",u:"uniform"};function g(s){return e.createBindGroupLayout({entries:s.map((d,p)=>({binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:ie[d]}}))})}function ee(s){return e.createBindGroupLayout({entries:s.map((d,p)=>d==="t"?{binding:p,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:ie[d]}})})}let M=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ce=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,ve=GPUBufferUsage.STORAGE,Se=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,re=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function A(s,d){return e.createBuffer({size:s,usage:d})}function Z(s,d){return e.createBindGroup({layout:s,entries:d.map((p,u)=>({binding:u,resource:"size"in p?{buffer:p}:p}))})}function pe(s,d){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[s]}),compute:{module:d,entryPoint:"main"}})}let Ht=e.createShaderModule({code:ea}),lt=e.createShaderModule({code:fa}),Me=e.createShaderModule({code:W(ca)}),Ve=e.createShaderModule({code:W(Gn)}),We=e.createShaderModule({code:W(kn)}),Be=e.createShaderModule({code:W(An)}),ge=e.createShaderModule({code:W(Sn)}),rt=e.createShaderModule({code:W(ta)}),gt=e.createShaderModule({code:aa}),Ee=e.createShaderModule({code:ra}),yt=e.createShaderModule({code:ia}),Ze=e.createShaderModule({code:W(sa)}),je=e.createShaderModule({code:W(ua)}),nn=e.createShaderModule({code:W(pa)}),xt=e.createShaderModule({code:W(da)}),ye=new Map;function Lt(s,d){let p=`${s}_${d}`,u=ye.get(p);return u||(u=e.createShaderModule({code:W(la(s,d))}),ye.set(p,u)),u}let _t=g(["r","r","r","s","u"]),mt=g(["r","r","r","r","s","u"]),Pt=g(["r","s","u"]),Wt=g(["r","r","r","s","u"]),P=g(["r","s","u"]),E=g(["r","r","s","u"]),z=g(["r","r","s","u"]),J=g(["r","r","r","s","u"]),N=g(["r","r","r","s","u"]),he=ee(["t","s","u"]),Q=g(["r","r","r","r","r","r","r","s"]),de=g(["r","r","r","r","r","s","u"]),be=e.createPipelineLayout({bindGroupLayouts:[_t]}),se=e.createPipelineLayout({bindGroupLayouts:[mt]}),q=s=>e.createComputePipeline({layout:be,compute:{module:s,entryPoint:"main"}}),we=s=>e.createComputePipeline({layout:se,compute:{module:s,entryPoint:"main"}}),oe=q(Ve),xe=q(We),Ce=we(Be),Ue=we(ge),De=new Map,Re=new Map,Te=new Map,Oe=new Map;De.set("8,8",oe),Re.set("8,8",xe),Te.set("8,8",Ce),Oe.set("8,8",Ue);function He(s,d,p,u,i){let H=`${d},${p}`,S=s.get(H);return S||(S=i(e.createShaderModule({code:W(u(d,p))})),s.set(H,S)),S}let st=(s,d)=>He(De,s,d,Vn,q),Je=(s,d)=>He(Re,s,d,Zn,q),Ne=(s,d)=>He(Te,s,d,jn,we),Qe=(s,d)=>He(Oe,s,d,Jn,we),Pe=ha.map(s=>{let d=s.stride===2?s.h/2:s.h,p=s.stride===2?s.w/2:s.w,[u,i]=Qn(s.inCh,d),H=s.h>=64,S=d>=16&&s.inCh>=288&&s.outCh>=288&&s.outCh%2===0;return{dwPipeline:H?Je(u,i):st(u,i),pwPipeline:S?Qe(u,i):Ne(u,i),dwDispatchX:Math.ceil(p/u),dwDispatchY:Math.ceil(d/i),dwDispatchZ:s.inCh,pwDispatchX:Math.ceil(p/u),pwDispatchY:Math.ceil(d/i),pwDispatchZ:S?s.outCh/2:s.outCh}}),qe=pe(Pt,Ht),et=pe(Wt,rt);pe(P,gt),pe(E,Ee);let tt=pe(z,yt),ze=pe(J,Ze);pe(N,je),pe(N,nn);let le=pe(he,lt),nt=pe(Q,Me),vt=pe(de,xt),at=1*288*128*128*4,Ie=A(3*256*256*4,M),ke=A(3*257*257*4,ve),Ye=A(12,re);e.queue.writeBuffer(Ye,0,new Uint32Array([3,256,257]));let j=A(at,ce),_e=A(at,Se),Fe=A(at,ve),ft=A(3072*64*4,M),ht=A(3072*32*4,M),bt=A(1536*16*4,M),wt=A(6144*64*4,ve),Ke=A(260,Se),$=A(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);A(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let D=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),X=A(8,re);e.queue.writeBuffer(X,0,new Uint32Array([256,257]));let Rt=r.get("backbone1.1.weight"),Ot=r.get("backbone1.1.bias");if(!Rt||!Ot)throw new Error("Missing input conv weights");let zt=w(Rt),It=w(Ot),Ft=A(zt.byteLength,M),Kt=A(It.byteLength,M),an=A(28,re);e.queue.writeBuffer(Ft,0,zt),e.queue.writeBuffer(Kt,0,It),e.queue.writeBuffer(an,0,new Uint32Array([1,3,24,257,257,128,128]));let rn=r.get("backbone6.1.weight"),sn=r.get("backbone6.1.bias");if(!rn||!sn)throw new Error("Missing backbone6.1 conv1x1 weights");let on=w(rn),Nt=w(sn),qt=A(on.byteLength,M),Yt=A(Nt.byteLength,M),Xt=A(20,re);e.queue.writeBuffer(qt,0,on),e.queue.writeBuffer(Yt,0,Nt),e.queue.writeBuffer(Xt,0,new Uint32Array([1,96,48,32,32]));let $t=r.get("handflag.weight"),Vt=r.get("handflag.bias");if(!$t||!Vt)throw new Error("Missing handflag weights");let un=w($t),pn=w(Vt),Xe=A(un.byteLength,M),Bt=A(pn.byteLength,M),Zt=A(12,re);e.queue.writeBuffer(Xe,0,un),e.queue.writeBuffer(Bt,0,pn),e.queue.writeBuffer(Zt,0,new Uint32Array([1,288,1]));let jt=r.get("handedness.weight"),cn=r.get("handedness.bias");if(!jt||!cn)throw new Error("Missing handedness weights");let dn=w(jt),Ge=w(cn),Le=A(dn.byteLength,M),ot=A(Ge.byteLength,M),Jt=A(12,re);e.queue.writeBuffer(Le,0,dn),e.queue.writeBuffer(ot,0,Ge),e.queue.writeBuffer(Jt,0,new Uint32Array([1,288,1]));let ln=r.get("reg_3d.weight"),_n=r.get("reg_3d.bias");if(!ln||!_n)throw new Error("Missing reg_3d weights");let V=w(ln),mn=w(_n),n=A(V.byteLength,M),t=A(mn.byteLength,M),o=A(12,re);e.queue.writeBuffer(n,0,V),e.queue.writeBuffer(t,0,mn),e.queue.writeBuffer(o,0,new Uint32Array([1,288,63]));let R=ha.map(s=>{let{inCh:d,outCh:p,h:u,w:i,stride:H,prefix:S}=s,O=H===2?u/2:u,k=H===2?i/2:i,y=H===2?1:2,c=r.get(`${S}convs.0.weight`),K=r.get(`${S}convs.0.bias`),ae=r.get(`${S}convs.1.weight`),ue=r.get(`${S}convs.1.bias`);if(!c||!K||!ae||!ue)throw new Error(`Missing weights for ${S}`);let At=w(c),St=w(K),Mt=w(ae),Et=w(ue),Dt=A(At.byteLength,M),Tt=A(St.byteLength,M),ct=A(Mt.byteLength,M),it=A(Et.byteLength,M),Qt=A(32,re),en=A(36,re);return e.queue.writeBuffer(Dt,0,At),e.queue.writeBuffer(Tt,0,St),e.queue.writeBuffer(ct,0,Mt),e.queue.writeBuffer(it,0,Et),e.queue.writeBuffer(Qt,0,new Uint32Array([1,d,u,i,O,k,H,y])),e.queue.writeBuffer(en,0,new Uint32Array([1,d,p,O,k,Math.max(0,p-d),H,u,i])),{dwWeight:Dt,dwBias:Tt,pwWeight:ct,pwBias:it,dwUniform:Qt,pwUniform:en,spec:s,outH:O,outW:k}});function F(s){let d=A(s.length*4,re);return e.queue.writeBuffer(d,0,new Uint32Array(s)),d}let _=F([1,96,8,8,16,16]),Y=F([1,96,16,16,32,32]),te=F([1,48,32,32,64,64]);F([1536*16]),F([3072*32]),F([3072*64]);let h=Z(Pt,[Ie,ke,Ye]),U=Z(Wt,[ke,Ft,Kt,j,an]),b=[],T=[],l=[],B=[];for(let s of R)b.push(Z(_t,[j,s.dwWeight,s.dwBias,Fe,s.dwUniform])),T.push(Z(mt,[Fe,j,s.pwWeight,s.pwBias,_e,s.pwUniform])),l.push(Z(_t,[_e,s.dwWeight,s.dwBias,Fe,s.dwUniform])),B.push(Z(mt,[Fe,_e,s.pwWeight,s.pwBias,j,s.pwUniform]));let ne=Z(z,[j,bt,_e,_]),Ae=Z(z,[j,ht,_e,Y]),ut=Z(J,[j,qt,Yt,wt,Xt]),On=Z(z,[wt,ft,_e,te]);Z(N,[j,Xe,Bt,Ke,Zt]),Z(N,[j,Le,ot,Ke,Jt]),Z(N,[j,n,t,Ke,o]);let me=Z(he,[D.createView(),ke,X]),zn=Z(Q,[j,Xe,Bt,Le,ot,n,t,Ke]),xn=24,In=[],Fn=[];for(let s=xn;s<R.length;s++){let d=R[s];In.push(Z(de,[j,d.dwWeight,d.dwBias,d.pwWeight,d.pwBias,_e,d.dwUniform])),Fn.push(Z(de,[_e,d.dwWeight,d.dwBias,d.pwWeight,d.pwBias,j,d.dwUniform]))}let Pn=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Pn.globalCompositeOperation="copy";let Kn=new OffscreenCanvas(9,8),fn=Kn.getContext("webgpu"),hn=null,vn=null;if(fn){fn.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let s=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),d=e.createShaderModule({code:_a}),p=e.createShaderModule({code:ma});hn=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[s]}),vertex:{module:d,entryPoint:"vs"},fragment:{module:p,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),vn=e.createBindGroup({layout:s,entries:[{binding:0,resource:{buffer:Ke}}]})}let Ct=new Float32Array(1),Ut=new Float32Array(1),kt=new Float32Array(63);function $e(s,d){let p=!0,u=0,i=s.beginComputePass();for(i.setPipeline(et),i.setBindGroup(0,U),i.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);u<=Ya;u++){let k=p?b[u]:l[u],y=p?T[u]:B[u],c=Pe[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,y),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let H=p?j:_e;for(s.copyBufferToBuffer(H,0,ft,0,3072*64*4),i=s.beginComputePass();u<=Xa;u++){let k=p?b[u]:l[u],y=p?T[u]:B[u],c=Pe[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,y),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let S=p?j:_e;for(s.copyBufferToBuffer(S,0,ht,0,3072*32*4),i=s.beginComputePass();u<=$a;u++){let k=p?b[u]:l[u],y=p?T[u]:B[u],c=Pe[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,y),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.end();let O=p?j:_e;for(s.copyBufferToBuffer(O,0,bt,0,1536*16*4),i=s.beginComputePass();u<=Va;u++){let k=p?b[u]:l[u],y=p?T[u]:B[u],c=Pe[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,y),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}i.setPipeline(tt),i.setBindGroup(0,ne),i.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),i.end(),p=!1,i=s.beginComputePass();{let k=p?b[u]:l[u],y=p?T[u]:B[u],c=Pe[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,y),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p,u++}i.setPipeline(tt),i.setBindGroup(0,Ae),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),i.end(),p=!1,i=s.beginComputePass();{let k=p?b[u]:l[u],y=p?T[u]:B[u],c=Pe[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,y),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p,u++}for(i.setPipeline(ze),i.setBindGroup(0,ut),i.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),i.setPipeline(tt),i.setBindGroup(0,On),i.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),i.end(),p=!1,i=s.beginComputePass();u<xn;u++){let k=p?b[u]:l[u],y=p?T[u]:B[u],c=Pe[u];i.setPipeline(c.dwPipeline),i.setBindGroup(0,k),i.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),i.setPipeline(c.pwPipeline),i.setBindGroup(0,y),i.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),p=!p}for(;u<R.length;u++){let k=u-xn,y=p?In[k]:Fn[k],c=R[u];i.setPipeline(vt),i.setBindGroup(0,y),i.dispatchWorkgroups(c.outW,c.outH,1),p=!p}i.setPipeline(nt),i.setBindGroup(0,zn),i.dispatchWorkgroups(1),i.end(),d&&s.copyBufferToBuffer(Ke,0,d,0,260)}async function bn(s){e.queue.writeBuffer(Ie,0,s);let d=e.createCommandEncoder();{let i=d.beginComputePass();i.setPipeline(qe),i.setBindGroup(0,h),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),i.end()}$e(d,$),e.queue.submit([d.finish()]);let p=$.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await p;let u=new Float32Array($.getMappedRange());return Ct[0]=u[0],Ut[0]=u[1],kt.set(u.subarray(2,65)),$.unmap(),{handflag:new Float32Array(Ct),handedness:new Float32Array(Ut),landmarks:new Float32Array(kt)}}async function Bn(s){e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);let d=e.createCommandEncoder();{let i=d.beginComputePass();i.setPipeline(le),i.setBindGroup(0,me),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}$e(d,$),e.queue.submit([d.finish()]);let p=$.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await p;let u=new Float32Array($.getMappedRange());return Ct[0]=u[0],Ut[0]=u[1],kt.set(u.subarray(2,65)),$.unmap(),{handflag:new Float32Array(Ct),handedness:new Float32Array(Ut),landmarks:new Float32Array(kt)}}async function Ma(s){let d=e.createCommandEncoder();d.copyBufferToBuffer(s,0,Ie,0,3*256*256*4);{let i=d.beginComputePass();i.setPipeline(qe),i.setBindGroup(0,h),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),i.end()}$e(d,$),e.queue.submit([d.finish()]);let p=$.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await p;let u=new Float32Array($.getMappedRange());return Ct[0]=u[0],Ut[0]=u[1],kt.set(u.subarray(2,65)),$.unmap(),{handflag:new Float32Array(Ct),handedness:new Float32Array(Ut),landmarks:new Float32Array(kt)}}async function Nn(s){if(!hn||!vn||!fn)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);let d=e.createCommandEncoder();{let k=d.beginComputePass();k.setPipeline(le),k.setBindGroup(0,me),k.dispatchWorkgroups(16,16,1),k.end()}$e(d,null);let p=fn.getCurrentTexture(),u=d.beginRenderPass({colorAttachments:[{view:p.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});u.setPipeline(hn),u.setBindGroup(0,vn),u.draw(3),u.end(),e.queue.submit([d.finish()]),await e.queue.onSubmittedWorkDone(),Pn.drawImage(Kn,0,0);let H=Pn.getImageData(0,0,9,8).data,S=new Float32Array(65),O=new DataView(new ArrayBuffer(4));for(let k=0;k<65;k++){let y=k*4;O.setUint8(0,H[y]),O.setUint8(1,H[y+1]),O.setUint8(2,H[y+2]),O.setUint8(3,H[y+3]),S[k]=O.getFloat32(0)}return{handflag:new Float32Array([S[0]]),handedness:new Float32Array([S[1]]),landmarks:new Float32Array(S.subarray(2,65))}}let Ea=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Cn=0,Da=[$,Ea],Gt=null,pt=null;async function Un(s){let d=Da[Cn];Cn=1-Cn,e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);let p=e.createCommandEncoder();{let i=p.beginComputePass();i.setPipeline(le),i.setBindGroup(0,me),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}$e(p,d),e.queue.submit([p.finish()]);let u=null;if(Gt!==null&&pt!==null){await Gt;let i=new Float32Array(pt.getMappedRange());u={handflag:new Float32Array([i[0]]),handedness:new Float32Array([i[1]]),landmarks:new Float32Array(i.subarray(2,65))},pt.unmap()}return pt=d,Gt=d.mapAsync(GPUMapMode.READ),u}async function qn(){if(!Gt||!pt)return null;await Gt;let s=new Float32Array(pt.getMappedRange()),d={handflag:new Float32Array([s[0]]),handedness:new Float32Array([s[1]]),landmarks:new Float32Array(s.subarray(2,65))};return pt.unmap(),Gt=null,pt=null,d}async function Ta(s=50){let d=new Float32Array(196608);for(let i=0;i<5;i++)await bn(d);let p=[];for(let i=0;i<s;i++){let H=performance.now();await bn(d),p.push(performance.now()-H)}let u=p.reduce((i,H)=>i+H,0)/p.length;return{avgMs:u,fps:1e3/u}}async function Ha(s=50){let d=new Float32Array(196608);for(let S=0;S<5;S++)await bn(d);let p=[];for(let S=0;S<s;S++){let O=e.createCommandEncoder();{let y=O.beginComputePass();y.setPipeline(qe),y.setBindGroup(0,h),y.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),y.end()}$e(O,$);let k=performance.now();e.queue.submit([O.finish()]),await e.queue.onSubmittedWorkDone(),p.push(performance.now()-k)}p.sort((S,O)=>S-O);let u=p.reduce((S,O)=>S+O,0)/p.length,i=p[Math.floor(p.length/2)],H=p[0];return{avgMs:u,fps:1e3/u,medianMs:i,minMs:H}}function si(s){e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);let d=e.createCommandEncoder();{let p=d.beginComputePass();p.setPipeline(le),p.setBindGroup(0,me),p.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),p.end()}$e(d,$),e.queue.submit([d.finish()])}async function La(s,d=50){function p(y){let c=[...y].sort((K,ae)=>K-ae);return{median:c[Math.floor(c.length/2)],min:c[0]}}for(let y=0;y<10;y++)await Bn(s);let u=[];for(let y=0;y<d;y++){e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);let c=e.createCommandEncoder();{let ae=c.beginComputePass();ae.setPipeline(le),ae.setBindGroup(0,me),ae.dispatchWorkgroups(16,16,1),ae.end()}$e(c,$);let K=performance.now();e.queue.submit([c.finish()]),await e.queue.onSubmittedWorkDone(),u.push(performance.now()-K)}let i=[];for(let y=0;y<d;y++){e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);let c=e.createCommandEncoder();{let ue=c.beginComputePass();ue.setPipeline(le),ue.setBindGroup(0,me),ue.dispatchWorkgroups(16,16,1),ue.end()}$e(c,$),e.queue.submit([c.finish()]);let K=$.mapAsync(GPUMapMode.READ),ae=performance.now();await e.queue.onSubmittedWorkDone(),await K,$.getMappedRange(),$.unmap(),i.push(performance.now()-ae)}let H=[];for(let y=0;y<d;y++){e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);let c=e.createCommandEncoder();{let ae=c.beginComputePass();ae.setPipeline(le),ae.setBindGroup(0,me),ae.dispatchWorkgroups(16,16,1),ae.end()}$e(c,$),e.queue.submit([c.finish()]);let K=performance.now();await $.mapAsync(GPUMapMode.READ),$.getMappedRange(),$.unmap(),H.push(performance.now()-K)}let S=[];for(let y=0;y<d;y++){let c=performance.now();await Bn(s),S.push(performance.now()-c)}await Un(s);let O=[];for(let y=0;y<d;y++){let c=performance.now();await Un(s),O.push(performance.now()-c)}await qn();let k=null;if(hn){let y=[];for(let c=0;c<d;c++){let K=performance.now();await Nn(s),y.push(performance.now()-K)}k=p(y)}return{gpuOnly:p(u),mapAsyncOnly:p(i),mapAsyncNoWait:p(H),total:p(S),pipelined:p(O),renderReadback:k}}async function Wa(s){let d=[];async function p(i,H,S){let O=e.createBuffer({size:H,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),k=e.createCommandEncoder();k.copyBufferToBuffer(i,0,O,0,H),e.queue.submit([k.finish()]),await e.queue.onSubmittedWorkDone(),await O.mapAsync(GPUMapMode.READ);let y=new Float32Array(O.getMappedRange()),c=1/0,K=-1/0,ae=0;for(let ue=0;ue<y.length;ue++)y[ue]<c&&(c=y[ue]),y[ue]>K&&(K=y[ue]),y[ue]!==0&&ae++;O.unmap(),O.destroy(),d.push({layer:S,stats:{min:c,max:K,nonZero:ae,total:y.length}})}e.queue.copyExternalImageToTexture({source:s},{texture:D},[256,256]);{let i=e.createCommandEncoder(),H=i.beginComputePass();H.setPipeline(le),H.setBindGroup(0,me),H.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),H.end(),e.queue.submit([i.finish()])}await p(ke,Math.min(ke.size,3*257*257*4),"canvas\u2192bufInput");{let i=e.createCommandEncoder(),H=i.beginComputePass();H.setPipeline(et),H.setBindGroup(0,U),H.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),H.end(),e.queue.submit([i.finish()])}await p(j,Math.min(j.size,3072*128*4),"inputConv\u2192bufA");let u=!0;for(let i=0;i<Math.min(R.length,6);i++){let H=u?b[i]:l[i],S=u?T[i]:B[i],O=Pe[i],k=R[i];{let c=e.createCommandEncoder(),K=c.beginComputePass();K.setPipeline(O.dwPipeline),K.setBindGroup(0,H),K.dispatchWorkgroups(O.dwDispatchX,O.dwDispatchY,O.dwDispatchZ),K.end(),e.queue.submit([c.finish()])}await p(Fe,Math.min(Fe.size,k.spec.inCh*k.outH*k.outW*4),`layer${i}.DW\u2192bufDW (${k.spec.prefix})`);{let c=e.createCommandEncoder(),K=c.beginComputePass();K.setPipeline(O.pwPipeline),K.setBindGroup(0,S),K.dispatchWorkgroups(O.pwDispatchX,O.pwDispatchY,O.pwDispatchZ),K.end(),e.queue.submit([c.finish()])}let y=u?_e:j;await p(y,Math.min(y.size,k.spec.outCh*k.outH*k.outW*4),`layer${i}.PW\u2192buf${u?"B":"A"} (${k.spec.prefix})`),u=!u}return d}return{device:e,run:bn,runFromCanvas:Bn,runFromGPUBuffer:Ma,runFromCanvasViaRender:Nn,runFromCanvasPipelined:Un,flushPipelined:qn,benchmark:Ta,benchmarkGPU:Ha,benchmarkDiagnostic:La,debugLayerOutputs:Wa}}function dt(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ba=dt(`
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
`),wa=dt(`
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
`),ga=dt(`
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
`),ya=dt(`
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
`),xa=dt(`
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
`),Pa=dt(`
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
`),va=dt(`
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
`),Ba=dt(`
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
`);async function Ca(r,m){let a;if(m)a=m;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");a=await n.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}})}let C={r:"read-only-storage",s:"storage",u:"uniform"};function x(n){return a.createBindGroupLayout({entries:n.map((t,o)=>t==="t"?{binding:o,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:o,visibility:GPUShaderStage.COMPUTE,buffer:{type:C[t]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,I=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,L=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function f(n,t){return a.createBuffer({size:Math.max(n,4),usage:t})}function w(n,t,o){a.queue.writeBuffer(n,t,o)}function G(n){let t=f(n.data.byteLength,e);return w(t,0,n.data),t}let W=Array.from(r.keys());function ie(n){let t=r.get(n);if(!t)throw new Error(`Weight not found: ${n}`);return t}function g(...n){let t=W.find(o=>n.every(R=>o.includes(R)));if(!t)throw new Error(`Weight not found for: ${n.join(", ")}`);return ie(t)}function ee(n){let[,t,o,R]=n.shape,F=new Float32Array(R*25);for(let _=0;_<R;_++)for(let Y=0;Y<t;Y++)for(let te=0;te<o;te++)F[_*25+Y*5+te]=n.data[Y*o*R+te*R+_];return F}function M(n){let[t,,,o]=n.shape,R=new Float32Array(t*o);for(let F=0;F<t;F++)for(let _=0;_<o;_++)R[F*o+_]=n.data[F*o+_];return R}let ce=a.createShaderModule({code:ba}),ve=a.createShaderModule({code:wa}),Se=a.createShaderModule({code:ga}),re=a.createShaderModule({code:ya}),A=a.createShaderModule({code:Pa}),Z=a.createShaderModule({code:xa}),pe=a.createShaderModule({code:va}),Ht=a.createShaderModule({code:Ba}),lt=x(["r","r","r","r","s","u"]),Me=x(["r","r","r","s","u"]),Ve=x(["r","r","r","r","r","s","u"]),We=x(["r","r","r","s","u"]),Be=x(["r","r","r","r","s","u"]),ge=x(["r","r","s","u"]),rt=x(["t","s","u"]),gt=x(["t","s","u"]);function Ee(n,t){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:t,entryPoint:"main"}})}let yt=Ee(lt,ce),Ze=Ee(Me,ve),je=Ee(Ve,Se),nn=Ee(We,re),xt=Ee(Be,A),ye=Ee(ge,Z),Lt=Ee(rt,pe),_t=Ee(gt,Ht),mt=g("conv2d/Conv2D"),Pt=g("batch_normalization/","conv2d/Conv2D"),Wt=g("p_re_lu/"),P=G(mt),E=G(Pt),z=G(Wt),N=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(n=>{let t=g(n.dwKey),o=g(n.pwKey),R=g(n.bnKey),F=g(n.preluKey),_=ee(t),Y=f(_.byteLength,e);w(Y,0,_);let te=new Float32Array(n.inCh),h=f(te.byteLength,e);w(h,0,te);let U=M(o),b=f(U.byteLength,e);w(b,0,U);let T=G(R),l=G(F);return{dwWeightBuf:Y,dwBiasBuf:h,pwWeightBuf:b,pwBiasBuf:T,alphaBuf:l,inCh:n.inCh,outCh:n.outCh,stride:n.stride,inH:n.inH}}),he=M(g("conv2d_25/Conv2D")),Q=f(he.byteLength,e);w(Q,0,he);let de=G(g("batch_normalization_25/")),be=G(g("p_re_lu_25/")),se={dwWeightBuf:(()=>{let n=ee(g("depthwise_conv2d_24/")),t=f(n.byteLength,e);return w(t,0,n),t})(),dwBiasBuf:(()=>{let n=new Float32Array(256),t=f(n.byteLength,e);return w(t,0,n),t})(),pwWeightBuf:(()=>{let n=M(g("conv2d_26/")),t=f(n.byteLength,e);return w(t,0,n),t})(),pwBiasBuf:G(g("batch_normalization_26/")),alphaBuf:G(g("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},q={dwWeightBuf:(()=>{let n=ee(g("depthwise_conv2d_25/")),t=f(n.byteLength,e);return w(t,0,n),t})(),dwBiasBuf:(()=>{let n=new Float32Array(256),t=f(n.byteLength,e);return w(t,0,n),t})(),pwWeightBuf:(()=>{let n=M(g("conv2d_27/Conv2D1")),t=f(n.byteLength,e);return w(t,0,n),t})(),pwBiasBuf:G(g("batch_normalization_27/")),alphaBuf:G(g("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},we=M(g("conv2d_28/Conv2D")),oe=f(we.byteLength,e);w(oe,0,we);let xe=G(g("batch_normalization_28/")),Ce=G(g("p_re_lu_28/")),Ue={dwWeightBuf:(()=>{let n=ee(g("depthwise_conv2d_26/")),t=f(n.byteLength,e);return w(t,0,n),t})(),dwBiasBuf:(()=>{let n=new Float32Array(128),t=f(n.byteLength,e);return w(t,0,n),t})(),pwWeightBuf:(()=>{let n=M(g("conv2d_29/")),t=f(n.byteLength,e);return w(t,0,n),t})(),pwBiasBuf:G(g("batch_normalization_29/")),alphaBuf:G(g("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},De={dwWeightBuf:(()=>{let n=ee(g("depthwise_conv2d_27/")),t=f(n.byteLength,e);return w(t,0,n),t})(),dwBiasBuf:(()=>{let n=new Float32Array(128),t=f(n.byteLength,e);return w(t,0,n),t})(),pwWeightBuf:(()=>{let n=M(g("conv2d_30/Conv2D1")),t=f(n.byteLength,e);return w(t,0,n),t})(),pwBiasBuf:G(g("batch_normalization_30/")),alphaBuf:G(g("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},Re=M(g("classifier_palm_16_NO_PRUNING/Conv2D")),Te=f(Re.byteLength,e);w(Te,0,Re);let Oe=G(g("classifier_palm_16_NO_PRUNING/BiasAdd")),He=M(g("regressor_palm_16_NO_PRUNING/Conv2D")),st=f(He.byteLength,e);w(st,0,He);let Je=G(g("regressor_palm_16_NO_PRUNING/BiasAdd")),Ne=M(g("classifier_palm_8_NO_PRUNING/Conv2D")),Qe=f(Ne.byteLength,e);w(Qe,0,Ne);let Pe=G(g("classifier_palm_8_NO_PRUNING/BiasAdd")),qe=M(g("regressor_palm_8_NO_PRUNING/Conv2D")),et=f(qe.byteLength,e);w(et,0,qe);let tt=G(g("regressor_palm_8_NO_PRUNING/BiasAdd")),ze=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,le=f(36864*3*4,e),nt=f(ze,v),vt=f(ze,v),at=f(ze,v),Ie=f(576*256*4,v),ke=f(144*256*4,v|GPUBufferUsage.COPY_DST),Ye=f(576*128*4,v|GPUBufferUsage.COPY_DST),j=f(864*4,I),_e=f(15552*4,I),Fe=f(576*2*4,I),ft=f(576*36*4,I),ht=f(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),bt=f(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),wt=f(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ke=f(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),$=a.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function D(n,t){return Math.ceil(n/t)}function X(n){let t=f(n.byteLength,L);return w(t,0,n),t}let Rt=X(new Uint32Array([1,3,32,192,192,96,96])),Ot=N.map(n=>{let t=n.stride===2?n.inH/2:n.inH,o=t,R=n.stride===2?1:2,F=n.inCh;return{dw:X(new Uint32Array([1,n.inCh,n.inH,n.inH,t,o,n.stride,R])),pw:X(new Uint32Array([1,n.inCh,n.outCh,t,o,F,n.stride,n.inH,n.inH])),outH:t,outW:o}}),zt=(()=>{let n=se,t=n.stride===2?n.inH/2:n.inH,o=n.stride===2?1:2;return{dw:X(new Uint32Array([1,n.inCh,n.inH,n.inH,t,t,n.stride,o])),pw:X(new Uint32Array([1,n.inCh,n.outCh,t,t,n.inCh,n.stride,n.inH,n.inH])),outH:t}})(),It=(()=>{let n=q,t=n.stride===2?n.inH/2:n.inH,o=n.stride===2?1:2;return{dw:X(new Uint32Array([1,n.inCh,n.inH,n.inH,t,t,n.stride,o])),pw:X(new Uint32Array([1,n.inCh,n.outCh,t,t,n.inCh,n.stride,n.inH,n.inH])),outH:t}})(),Ft=(()=>{let n=Ue,t=n.stride===2?n.inH/2:n.inH,o=n.stride===2?1:2;return{dw:X(new Uint32Array([1,n.inCh,n.inH,n.inH,t,t,n.stride,o])),pw:X(new Uint32Array([1,n.inCh,n.outCh,t,t,n.inCh,n.stride,n.inH,n.inH])),outH:t}})(),Kt=(()=>{let n=De,t=n.stride===2?n.inH/2:n.inH,o=n.stride===2?1:2;return{dw:X(new Uint32Array([1,n.inCh,n.inH,n.inH,t,t,n.stride,o])),pw:X(new Uint32Array([1,n.inCh,n.outCh,t,t,n.inCh,n.stride,n.inH,n.inH])),outH:t}})(),an=X(new Uint32Array([1,256,6,6,12,12])),rn=X(new Uint32Array([1,256,12,12,12,12])),sn=X(new Uint32Array([1,256,12,12,24,24])),on=X(new Uint32Array([1,128,24,24,24,24])),Nt=X(new Uint32Array([1,256,256,12,12])),qt=X(new Uint32Array([1,256,128,24,24])),Yt=X(new Uint32Array([1,256,6,12,12])),Xt=X(new Uint32Array([1,256,108,12,12])),$t=X(new Uint32Array([1,128,2,24,24])),Vt=X(new Uint32Array([1,128,36,24,24])),un=X(new Uint32Array([192,192,192])),pn=a.createBindGroup({layout:rt,entries:[{binding:0,resource:$.createView()},{binding:1,resource:{buffer:le}},{binding:2,resource:{buffer:un}}]}),Xe=null,Bt=0,Zt=0,jt=f(32,L);function cn(n,t){return Xe&&Bt===n&&Zt===t||(Xe&&Xe.destroy(),Xe=a.createTexture({size:[n,t,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Bt=n,Zt=t),Xe}let dn=a.createBindGroup({layout:lt,entries:[{binding:0,resource:{buffer:le}},{binding:1,resource:{buffer:P}},{binding:2,resource:{buffer:E}},{binding:3,resource:{buffer:z}},{binding:4,resource:{buffer:nt}},{binding:5,resource:{buffer:Rt}}]});function Ge(n,t,o,R,F,_){let Y=_.outH,te=a.createBindGroup({layout:Me,entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:t.dwWeightBuf}},{binding:2,resource:{buffer:t.dwBiasBuf}},{binding:3,resource:{buffer:at}},{binding:4,resource:{buffer:_.dw}}]}),h=n.beginComputePass();h.setPipeline(Ze),h.setBindGroup(0,te),h.dispatchWorkgroups(D(Y,8),D(_.outH,8),t.inCh),h.end();let U=a.createBindGroup({layout:Ve,entries:[{binding:0,resource:{buffer:at}},{binding:1,resource:{buffer:F}},{binding:2,resource:{buffer:t.pwWeightBuf}},{binding:3,resource:{buffer:t.pwBiasBuf}},{binding:4,resource:{buffer:t.alphaBuf}},{binding:5,resource:{buffer:R}},{binding:6,resource:{buffer:_.pw}}]}),b=n.beginComputePass();b.setPipeline(je),b.setBindGroup(0,U),b.dispatchWorkgroups(D(Y,8),D(_.outH,8),t.outCh),b.end()}function Le(n,t,o,R,F,_,Y,te,h){let U=a.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:R}},{binding:3,resource:{buffer:F}},{binding:4,resource:{buffer:_}}]}),b=n.beginComputePass();b.setPipeline(nn),b.setBindGroup(0,U),b.dispatchWorkgroups(D(h,8),D(te,8),Y),b.end()}function ot(n,t,o,R,F,_,Y,te,h,U){let b=a.createBindGroup({layout:Be,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:R}},{binding:3,resource:{buffer:F}},{binding:4,resource:{buffer:_}},{binding:5,resource:{buffer:Y}}]}),T=n.beginComputePass();T.setPipeline(xt),T.setBindGroup(0,b),T.dispatchWorkgroups(D(U,8),D(h,8),te),T.end()}async function Jt(n){{let l=n.beginComputePass();l.setPipeline(yt),l.setBindGroup(0,dn),l.dispatchWorkgroups(D(96,8),D(96,8),32),l.end()}let t=nt,o=vt;for(let l=0;l<N.length;l++){let B=N[l];Ge(n,B,t,o,t,Ot[l]);let ne=t;t=o,o=ne,l===13&&n.copyBufferToBuffer(t,0,Ye,0,576*128*4),l===18&&n.copyBufferToBuffer(t,0,ke,0,144*256*4)}{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Ie}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:an}}]}),B=n.beginComputePass();B.setPipeline(ye),B.setBindGroup(0,l),B.dispatchWorkgroups(D(12,8),D(12,8),256),B.end()}{let l=t;t=o,o=l}ot(n,t,Q,de,be,o,Nt,256,12,12);{let l=t;t=o,o=l}{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:ke}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:rn}}]}),B=n.beginComputePass();B.setPipeline(ye),B.setBindGroup(0,l),B.dispatchWorkgroups(D(12,8),D(12,8),256),B.end()}{let l=t;t=o,o=l}Ge(n,se,t,o,t,zt);{let l=t;t=o,o=l}Ge(n,q,t,o,t,It);{let l=t;t=o,o=l}Le(n,t,Te,Oe,j,Yt,6,12,12),Le(n,t,st,Je,_e,Xt,108,12,12);{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Ie}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:sn}}]}),B=n.beginComputePass();B.setPipeline(ye),B.setBindGroup(0,l),B.dispatchWorkgroups(D(24,8),D(24,8),256),B.end()}{let l=t;t=o,o=l}ot(n,t,oe,xe,Ce,o,qt,128,24,24);{let l=t;t=o,o=l}{let l=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:on}}]}),B=n.beginComputePass();B.setPipeline(ye),B.setBindGroup(0,l),B.dispatchWorkgroups(D(24,8),D(24,8),128),B.end()}{let l=t;t=o,o=l}Ge(n,Ue,t,o,t,Ft);{let l=t;t=o,o=l}Ge(n,De,t,o,t,Kt);{let l=t;t=o,o=l}Le(n,t,Qe,Pe,Fe,$t,2,24,24),Le(n,t,et,tt,ft,Vt,36,24,24),a.queue.submit([n.finish()]);let R=a.createCommandEncoder();R.copyBufferToBuffer(j,0,ht,0,864*4),R.copyBufferToBuffer(_e,0,bt,0,15552*4),R.copyBufferToBuffer(Fe,0,wt,0,576*2*4),R.copyBufferToBuffer(ft,0,Ke,0,576*36*4),a.queue.submit([R.finish()]),await Promise.all([ht.mapAsync(GPUMapMode.READ),bt.mapAsync(GPUMapMode.READ),wt.mapAsync(GPUMapMode.READ),Ke.mapAsync(GPUMapMode.READ)]);let F=new Float32Array(ht.getMappedRange()).slice(),_=new Float32Array(bt.getMappedRange()).slice(),Y=new Float32Array(wt.getMappedRange()).slice(),te=new Float32Array(Ke.getMappedRange()).slice();ht.unmap(),bt.unmap(),wt.unmap(),Ke.unmap();let h=2016,U=new Float32Array(h),b=new Float32Array(h*18),T=0;for(let l=0;l<12;l++)for(let B=0;B<12;B++)for(let ne=0;ne<6;ne++){U[T]=F[ne*144+l*12+B];for(let Ae=0;Ae<18;Ae++){let ut=ne*18+Ae;b[T*18+Ae]=_[ut*144+l*12+B]}T++}for(let l=0;l<24;l++)for(let B=0;B<24;B++)for(let ne=0;ne<2;ne++){U[T]=Y[ne*576+l*24+B];for(let Ae=0;Ae<18;Ae++){let ut=ne*18+Ae;b[T*18+Ae]=te[ut*576+l*24+B]}T++}return{scores:U,regressors:b}}async function ln(n){a.queue.copyExternalImageToTexture({source:n},{texture:$},[192,192]);let t=a.createCommandEncoder();{let o=t.beginComputePass();o.setPipeline(Lt),o.setBindGroup(0,pn),o.dispatchWorkgroups(D(192,16),D(192,16),1),o.end()}return Jt(t)}async function _n(n,t,o){let R=Math.min(192/t,192/o),F=Math.round(t*R),_=Math.round(o*R),Y=(192-F)/2,te=(192-_)/2,h=Y/192,U=te/192,b=cn(t,o),T;if(n instanceof HTMLVideoElement||n instanceof HTMLImageElement){let me=new OffscreenCanvas(t,o);me.getContext("2d").drawImage(n,0,0),T=me}else T=n;a.queue.copyExternalImageToTexture({source:T},{texture:b},[t,o]);let l=new ArrayBuffer(32),B=new Uint32Array(l),ne=new Float32Array(l);B[0]=t,B[1]=o,B[2]=192,B[3]=0,ne[4]=t/F,ne[5]=o/_,ne[6]=Y,ne[7]=te,a.queue.writeBuffer(jt,0,l);let Ae=a.createBindGroup({layout:gt,entries:[{binding:0,resource:b.createView()},{binding:1,resource:{buffer:le}},{binding:2,resource:{buffer:jt}}]}),ut=a.createCommandEncoder();{let me=ut.beginComputePass();me.setPipeline(_t),me.setBindGroup(0,Ae),me.dispatchWorkgroups(D(192,16),D(192,16),1),me.end()}return{output:await Jt(ut),lbPadX:h,lbPadY:U}}async function V(n,t){let o=a.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),R=a.createCommandEncoder();R.copyBufferToBuffer(n,0,o,0,t*4),a.queue.submit([R.finish()]),await o.mapAsync(GPUMapMode.READ);let F=new Float32Array(o.getMappedRange()).slice();return o.unmap(),o.destroy(),F}async function mn(n){a.queue.copyExternalImageToTexture({source:n},{texture:$},[192,192]);function t(b,T=1e3){let l=b.slice(0,T);return{min:Math.min(...l),max:Math.max(...l),mean:l.reduce((B,ne)=>B+ne,0)/l.length,nonZero:l.filter(B=>B!==0).length,sample:Array.from(l.slice(0,10))}}let o={},R=X(new Uint32Array([192,192,192])),F=a.createBindGroup({layout:rt,entries:[{binding:0,resource:$.createView()},{binding:1,resource:{buffer:le}},{binding:2,resource:{buffer:R}}]}),_=a.createCommandEncoder(),Y=_.beginComputePass();Y.setPipeline(Lt),Y.setBindGroup(0,F),Y.dispatchWorkgroups(D(192,16),D(192,16),1),Y.end(),a.queue.submit([_.finish()]),o.input=t(await V(le,36864*3)),_=a.createCommandEncoder();let te=a.createBindGroup({layout:lt,entries:[{binding:0,resource:{buffer:le}},{binding:1,resource:{buffer:P}},{binding:2,resource:{buffer:E}},{binding:3,resource:{buffer:z}},{binding:4,resource:{buffer:nt}},{binding:5,resource:{buffer:Rt}}]});Y=_.beginComputePass(),Y.setPipeline(yt),Y.setBindGroup(0,te),Y.dispatchWorkgroups(D(96,8),D(96,8),32),Y.end(),a.queue.submit([_.finish()]),o.initConv=t(await V(nt,9216*32));let h=nt,U=vt;for(let b=0;b<N.length;b++){let T=N[b];_=a.createCommandEncoder(),Ge(_,T,h,U,h,Ot[b]),a.queue.submit([_.finish()]);let l=h;if(h=U,U=l,b===0||b===4||b===9||b===14||b===18||b===19||b===23){let B=T.stride===2?T.inH/2:T.inH,ne=B*B*T.outCh;o[`block${b}`]=t(await V(h,ne))}b===13&&(_=a.createCommandEncoder(),_.copyBufferToBuffer(h,0,Ye,0,576*128*4),a.queue.submit([_.finish()])),b===18&&(_=a.createCommandEncoder(),_.copyBufferToBuffer(h,0,ke,0,144*256*4),a.queue.submit([_.finish()]))}_=a.createCommandEncoder();{let b=X(new Uint32Array([1,256,6,6,12,12])),T=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:Ie}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:b}}]}),l=_.beginComputePass();l.setPipeline(ye),l.setBindGroup(0,T),l.dispatchWorkgroups(D(12,8),D(12,8),256),l.end()}a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpnUpsample6to12=t(await V(h,144*256)),_=a.createCommandEncoder(),ot(_,h,Q,de,be,U,Nt,256,12,12),a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpn6to12Conv=t(await V(h,144*256)),o.backbone12Skip=t(await V(ke,144*256)),_=a.createCommandEncoder();{let b=X(new Uint32Array([1,256,12,12,12,12])),T=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:ke}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:b}}]}),l=_.beginComputePass();l.setPipeline(ye),l.setBindGroup(0,T),l.dispatchWorkgroups(D(12,8),D(12,8),256),l.end()}a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpnAdd12=t(await V(h,144*256)),_=a.createCommandEncoder(),Ge(_,se,h,U,h,zt),a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpn12Block1=t(await V(h,144*256)),_=a.createCommandEncoder(),Ge(_,q,h,U,h,It),a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpn12Block2=t(await V(h,144*256)),_=a.createCommandEncoder(),Le(_,h,Te,Oe,j,Yt,6,12,12),a.queue.submit([_.finish()]),o.cls16=t(await V(j,864)),_=a.createCommandEncoder(),Le(_,h,st,Je,_e,Xt,108,12,12),a.queue.submit([_.finish()]),o.reg16=t(await V(_e,15552),500),_=a.createCommandEncoder();{let b=X(new Uint32Array([1,256,12,12,24,24])),T=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:Ie}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:b}}]}),l=_.beginComputePass();l.setPipeline(ye),l.setBindGroup(0,T),l.dispatchWorkgroups(D(24,8),D(24,8),256),l.end()}a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpnUpsample12to24=t(await V(h,576*256)),_=a.createCommandEncoder(),ot(_,h,oe,xe,Ce,U,qt,128,24,24),a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpn12to24Conv=t(await V(h,576*128)),o.backbone24Skip=t(await V(Ye,576*128)),_=a.createCommandEncoder();{let b=X(new Uint32Array([1,128,24,24,24,24])),T=a.createBindGroup({layout:ge,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:U}},{binding:3,resource:{buffer:b}}]}),l=_.beginComputePass();l.setPipeline(ye),l.setBindGroup(0,T),l.dispatchWorkgroups(D(24,8),D(24,8),128),l.end()}a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpnAdd24=t(await V(h,576*128)),_=a.createCommandEncoder(),Ge(_,Ue,h,U,h,Ft),a.queue.submit([_.finish()]);{let b=h;h=U,U=b}o.fpn24Block1=t(await V(h,576*128)),_=a.createCommandEncoder(),Ge(_,De,h,U,h,Kt),a.queue.submit([_.finish()]);{let b=h;h=U,U=b}return o.fpn24Block2=t(await V(h,576*128)),_=a.createCommandEncoder(),Le(_,h,Qe,Pe,Fe,$t,2,24,24),a.queue.submit([_.finish()]),o.cls8=t(await V(Fe,576*2)),_=a.createCommandEncoder(),Le(_,h,et,tt,ft,Vt,36,24,24),a.queue.submit([_.finish()]),o.reg8=t(await V(ft,576*36)),o.initWeights=t(await V(P,100),100),o.initBias=t(await V(E,32),32),o.cls16Weights=t(await V(Te,100),100),o.cls16Bias=t(await V(Oe,6),6),o.cls8Weights=t(await V(Qe,100),100),o.cls8Bias=t(await V(Pe,2),2),o.fpn6to12Weights=t(await V(Q,100),100),o}return{device:a,run:ln,runWithResize:_n,debugRun:mn}}function Za(){let r=[];for(let m=0;m<12;m++)for(let a=0;a<12;a++){let C=(a+.5)/12,x=(m+.5)/12;for(let e=0;e<6;e++)r.push({x:C,y:x})}for(let m=0;m<24;m++)for(let a=0;a<24;a++){let C=(a+.5)/24,x=(m+.5)/24;for(let e=0;e<2;e++)r.push({x:C,y:x})}return r}var Ua=Za();function ja(r){return 1/(1+Math.exp(-r))}function Dn(r,m){let a=[],{scores:C,regressors:x}=r,e=192;for(let v=0;v<Ua.length;v++){let I=ja(C[v]);if(I<m)continue;let L=Ua[v],f=v*18,w=L.x+x[f+0]/e,G=L.y+x[f+1]/e,W=x[f+2]/e,ie=x[f+3]/e,g=[];for(let ee=0;ee<7;ee++){let M=L.x+x[f+4+ee*2]/e,ce=L.y+x[f+4+ee*2+1]/e;g.push([M,ce])}a.push({score:I,box:[w,G,W,ie],keypoints:g})}return a}function Tn(r,m){if(r.length===0)return[];let a=[...r].sort((e,v)=>v.score-e.score),C=[],x=new Set;for(let e=0;e<a.length;e++)if(!x.has(e)){C.push(a[e]);for(let v=e+1;v<a.length;v++)x.has(v)||Ja(a[e],a[v])>m&&x.add(v)}return C}function Ja(r,m){let a=r.box[0]-r.box[2]/2,C=r.box[1]-r.box[3]/2,x=r.box[0]+r.box[2]/2,e=r.box[1]+r.box[3]/2,v=m.box[0]-m.box[2]/2,I=m.box[1]-m.box[3]/2,L=m.box[0]+m.box[2]/2,f=m.box[1]+m.box[3]/2,w=Math.max(a,v),G=Math.max(C,I),W=Math.min(x,L),ie=Math.min(e,f),g=Math.max(0,W-w),ee=Math.max(0,ie-G),M=g*ee,ce=(x-a)*(e-C),ve=(L-v)*(f-I),Se=ce+ve-M;return Se>0?M/Se:0}function Qa(r){let[m,a,C,x]=r.box,e=r.keypoints[0],v=r.keypoints[2],I=v[0]-e[0],L=v[1]-e[1],f=Math.atan2(L,I),G=-Math.PI/2-f,W=Math.max(C,x),g=W*2.6,ee=-.5*W,M=Math.cos(G),ce=Math.sin(G),ve=ee*ce,Se=ee*M;return{centerX:m+ve,centerY:a+Se,width:g,height:g,rotation:G}}function ka(r,m={}){let{scoreThreshold:a=.5,nmsThreshold:C=.3,maxHands:x=2}=m;async function e(L){let f=await r.run(L),w=Dn(f,a);return Tn(w,C).slice(0,x).map(Qa)}async function v(L){let f=await r.run(L),w=Dn(f,a);return Tn(w,C).slice(0,x)}async function I(L,f,w){let{output:G,lbPadX:W,lbPadY:ie}=await r.runWithResize(L,f,w),g=Dn(G,a);return{detections:Tn(g,C).slice(0,x),lbPadX:W,lbPadY:ie}}return{detect:e,detectRaw:v,detectRawWithResize:I,model:r}}var Hn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Ln(r){let m={};for(let a=0;a<Hn.length;a++)m[Hn[a]]=r[a];return m}function Ga(r,m,a){return r.initialized?(r.value=a*m+(1-a)*r.value,r.value):(r.value=m,r.initialized=!0,m)}function Aa(r,m){let a=2*Math.PI*m*r;return a/(a+1)}function ei(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function Wn(r,m,a,C,x,e){let v=r.lastTime<0?.03333333333333333:a-r.lastTime;r.lastTime=a;let I=Aa(v,e),L=r.x.initialized?(m-r.x.value)/v:0,f=Ga(r.dx,L,I),w=C+x*Math.abs(f),G=Aa(v,w);return Ga(r.x,m,G)}function Rn(r={}){let{minCutoff:m=.05,beta:a=80,dCutoff:C=1}=r,x=[];function e(L){x.length!==L&&(x=Array.from({length:L},()=>ei()))}function v(L,f){let w=f??performance.now()/1e3,G=L.length*3;return e(G),L.map((W,ie)=>({x:Wn(x[ie*3],W.x,w,m,a,C),y:Wn(x[ie*3+1],W.y,w,m,a,C),z:Wn(x[ie*3+2],W.z,w,m,a,C)}))}function I(){x=[]}return{apply:v,reset:I}}function ti(r){return r.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var ni=ti(`
struct CropParams { src_width:u32, src_height:u32, dst_size:u32, _pad:u32, }
struct AffineTransform { a:f32, b:f32, tx:f32, c:f32, d:f32, ty:f32, }

@group(0)@binding(0) var src_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CropParams;
@group(0)@binding(3) var<uniform> transform:AffineTransform;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let dst_x=gid.x; let dst_y=gid.y;
  if(dst_x>=params.dst_size||dst_y>=params.dst_size){return;}

  // Map crop pixel to source normalized coordinates
  let fx=f32(dst_x)+0.5;
  let fy=f32(dst_y)+0.5;
  let src_nx=transform.a*fx+transform.b*fy+transform.tx;
  let src_ny=transform.c*fx+transform.d*fy+transform.ty;

  // Convert to pixel coordinates
  let src_px=src_nx*f32(params.src_width)-0.5;
  let src_py=src_ny*f32(params.src_height)-0.5;

  // Bilinear interpolation
  let x0=i32(floor(src_px)); let y0=i32(floor(src_py));
  let x1=x0+1; let y1=y0+1;
  let lx=src_px-f32(x0); let ly=src_py-f32(y0);

  let sw=i32(params.src_width); let sh=i32(params.src_height);

  // Zero for out-of-bounds (matches MediaPipe's kZero border mode)
  var p00=vec4<f32>(0.0); var p01=vec4<f32>(0.0);
  var p10=vec4<f32>(0.0); var p11=vec4<f32>(0.0);

  if(x0>=0 && x0<sw && y0>=0 && y0<sh){ p00=textureLoad(src_tex,vec2<u32>(u32(x0),u32(y0)),0); }
  if(x1>=0 && x1<sw && y0>=0 && y0<sh){ p01=textureLoad(src_tex,vec2<u32>(u32(x1),u32(y0)),0); }
  if(x0>=0 && x0<sw && y1>=0 && y1<sh){ p10=textureLoad(src_tex,vec2<u32>(u32(x0),u32(y1)),0); }
  if(x1>=0 && x1<sw && y1>=0 && y1<sh){ p11=textureLoad(src_tex,vec2<u32>(u32(x1),u32(y1)),0); }

  let pixel=p00*(1.0-lx)*(1.0-ly)+p01*lx*(1.0-ly)+p10*(1.0-lx)*ly+p11*lx*ly;

  // Write CHW format
  let out_stride=params.dst_size*params.dst_size;
  output[0u*out_stride+dst_y*params.dst_size+dst_x]=pixel.r;
  output[1u*out_stride+dst_y*params.dst_size+dst_x]=pixel.g;
  output[2u*out_stride+dst_y*params.dst_size+dst_x]=pixel.b;
}
`);function Sa(r){let m=r.createShaderModule({code:ni}),a=r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),C=r.createComputePipeline({layout:r.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:m,entryPoint:"main"}}),x=r.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e=r.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),v=new Float32Array(8);function I(L,f,w,G,W,ie,g){r.queue.writeBuffer(x,0,new Uint32Array([W,ie,g,0])),v.set(G),r.queue.writeBuffer(e,0,v);let ee=r.createBindGroup({layout:a,entries:[{binding:0,resource:f.createView()},{binding:1,resource:{buffer:w}},{binding:2,resource:{buffer:x}},{binding:3,resource:{buffer:e}}]}),M=L.beginComputePass();M.setPipeline(C),M.setBindGroup(0,ee),M.dispatchWorkgroups(Math.ceil(g/16),Math.ceil(g/16),1),M.end()}return{crop:I}}var ai="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function ii(r={}){let{weightsUrl:m,scoreThreshold:a=.5,palmScoreThreshold:C=.5,maxHands:x=3,forceF32:e=!1}=r;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let v=(m??ai).replace(/\/$/,"")+"/",[I,L,f,w]=await Promise.all([fetch(`${v}weights_f16.json`),fetch(`${v}weights_f16.bin`),fetch(`${v}palm_detection_weights.json`),fetch(`${v}palm_detection_weights.bin`)]);if(!I.ok)throw new Error(`Failed to fetch landmark weights: ${I.status}`);if(!L.ok)throw new Error(`Failed to fetch landmark weights: ${L.status}`);if(!f.ok)throw new Error(`Failed to fetch palm detection weights: ${f.status}`);if(!w.ok)throw new Error(`Failed to fetch palm detection weights: ${w.status}`);let[G,W,ie,g]=await Promise.all([I.json(),L.arrayBuffer(),f.json(),w.arrayBuffer()]),ee=Mn(G,W),M=Mn(ie,g),ce=await En(ee,{forceF32:e});if(!e){let P=new OffscreenCanvas(256,256),E=P.getContext("2d");E.fillStyle="#886644",E.fillRect(0,0,256,256),E.fillStyle="#cc9966",E.fillRect(50,50,156,156);let z=await ce.runFromCanvas(P);z.landmarks.every(N=>N===0)&&z.handflag.every(N=>N===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),ce.device.destroy(),ce=await En(ee,{forceF32:!0}))}let ve=await Ca(M),Se=ka(ve,{scoreThreshold:C,maxHands:x}),re=[];for(let P=0;P<x;P++)re.push(Rn());let A=0,Z=null,pe=null;function Ht(){return Z||(Z=new OffscreenCanvas(192,192)),Z}function lt(){return pe||(pe=new OffscreenCanvas(256,256)),pe}let Me=ce.device,Ve=null,We=null,Be=null,ge=0,rt=0;function gt(){return Ve||(Ve=Sa(Me)),Ve}function Ee(){return We||(We=Me.createBuffer({size:3*256*256*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),We}function yt(P,E){return(!Be||ge!==P||rt!==E)&&(Be&&Be.destroy(),Be=Me.createTexture({size:[P,E],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ge=P,rt=E),Be}let Ze=0,je=0;function nn(P,E,z){let J=Ht();J.width=192,J.height=192;let N=J.getContext("2d");N.clearRect(0,0,192,192);let he=Math.min(192/E,192/z),Q=Math.round(E*he),de=Math.round(z*he),be=(192-Q)/2,se=(192-de)/2;if(Ze=be/192,je=se/192,P instanceof ImageData){let q=new OffscreenCanvas(P.width,P.height);q.getContext("2d").putImageData(P,0,0),N.drawImage(q,be,se,Q,de)}else N.drawImage(P,0,0,E,z,be,se,Q,de);return J}function xt(P){let E=1/(1-2*Ze),z=1/(1-2*je);return{score:P.score,box:[(P.box[0]-Ze)*E,(P.box[1]-je)*z,P.box[2]*E,P.box[3]*z],keypoints:P.keypoints.map(([J,N])=>[(J-Ze)*E,(N-je)*z])}}function ye(P,E,z){let J=P.keypoints[0],N=P.keypoints[2],he=N[0]-J[0],Q=N[1]-J[1],de=Math.atan2(-Q,he),se=Math.PI/2-de,[q,we,oe,xe]=P.box,Ce=Math.max(oe*E,xe*z),Ue=0,De=-.5*Ce/z,Re=Math.cos(se),Te=Math.sin(se),Oe=q+(Ue*Re-De*Te),He=we+(Ue*Te+De*Re),Je=Ce*2.6;return{centerXpx:Oe*E,centerYpx:He*z,sizePx:Je,rotation:se}}function Lt(P,E){let z=lt();z.width=256,z.height=256;let J=z.getContext("2d");J.clearRect(0,0,256,256);let N=256/E.sizePx,he=Math.cos(E.rotation),Q=Math.sin(E.rotation),de=he*N,be=-Q*N,se=Q*N,q=he*N,we=-E.centerXpx*de-E.centerYpx*se+128,oe=-E.centerXpx*be-E.centerYpx*q+128;if(J.setTransform(de,be,se,q,we,oe),P instanceof ImageData){let xe=new OffscreenCanvas(P.width,P.height);xe.getContext("2d").putImageData(P,0,0),J.drawImage(xe,0,0)}else J.drawImage(P,0,0);return J.setTransform(1,0,0,1,0,0),z}function _t(P){return P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas?[P.width,P.height]:typeof ImageBitmap<"u"&&P instanceof ImageBitmap?[P.width,P.height]:P instanceof ImageData?[P.width,P.height]:P instanceof HTMLVideoElement?[P.videoWidth,P.videoHeight]:P instanceof HTMLImageElement?[P.naturalWidth,P.naturalHeight]:[256,256]}async function mt(P){let[E,z]=_t(P),{detections:J,lbPadX:N,lbPadY:he}=await Se.detectRawWithResize(P,E,z);if(Ze=N,je=he,J.length===0){if(A>0)for(let q=0;q<A&&q<re.length;q++)re[q].reset();return A=0,[]}let Q=[],de=gt(),be=Ee(),se=yt(E,z);if(P instanceof ImageData){let q=new OffscreenCanvas(P.width,P.height);q.getContext("2d").putImageData(P,0,0),Me.queue.copyExternalImageToTexture({source:q},{texture:se},[E,z])}else Me.queue.copyExternalImageToTexture({source:P},{texture:se},[E,z]);for(let q of J){let we=xt(q),oe=ye(we,E,z),xe=Math.cos(oe.rotation),Ce=Math.sin(oe.rotation),Ue=oe.sizePx/256,De=xe*Ue/E,Re=Ce*Ue/E,Te=oe.centerXpx/E-128*(De+Re),Oe=-Ce*Ue/z,He=xe*Ue/z,st=oe.centerYpx/z-128*(Oe+He),Je=Me.createCommandEncoder();de.crop(Je,se,be,[De,Re,Te,Oe,He,st],E,z,256),Me.queue.submit([Je.finish()]);let Ne=await ce.runFromGPUBuffer(be),Qe=Ne.handflag[0];if(Qe<a)continue;let Pe=Ne.handedness[0]>.5,qe=[];for(let ze=0;ze<21;ze++){let le=Ne.landmarks[ze*3],nt=Ne.landmarks[ze*3+1],vt=Ne.landmarks[ze*3+2],at=(le-.5)*oe.sizePx,Ie=(nt-.5)*oe.sizePx,ke=xe*at-Ce*Ie+oe.centerXpx,Ye=Ce*at+xe*Ie+oe.centerYpx;qe.push({x:ke/E,y:Ye/z,z:vt})}let et=Q.length,tt=et<re.length?re[et].apply(qe):qe;Q.push({score:Qe,handedness:Pe?"right":"left",landmarks:tt,keypoints:Ln(tt)})}if(Q.length<A)for(let q=Q.length;q<A;q++)q<re.length&&re[q].reset();return A=Q.length,Q}function Pt(){Be&&Be.destroy(),We&&We.destroy(),Be=null,We=null,Ve=null,ce.device.destroy(),ve.device.destroy(),Z=null,pe=null}return{detect:mt,dispose:Pt,_debug:{palmDetector:Se,palmModel:ve,landmarkModel:ce,removeLetterbox:xt,detectionToPixelROI:ye,cropHandRegion:Lt}}}export{Hn as LANDMARK_NAMES,ii as createHandpose,Rn as createLandmarkSmoother,Ln as toKeypoints};
