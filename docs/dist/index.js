function oe(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function Ta(n){let _=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],p="enable f16;"+n;for(let g of _)for(;p.includes(`${g}:array<f32>`);)p=p.replace(`${g}:array<f32>`,`${g}:array<f16>`);return p}var ca=oe(`
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
`),da=oe(`
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
`),_a=oe(`
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
`),la=oe(`
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
`);function La(n,_){return da.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Oa(n,_){return ca.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Ra(n,_){return _a.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Ia(n,_){return la.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${n},${_},1)`)}function Fa(n,_){return[8,8]}var za=oe(`
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
`),Na=oe(`
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
`);function Ka(n){return oe(`
struct UpsampleParams { batch:u32, channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, }
${n?`@group(0)@binding(0) var<storage,read> input:array<f32>;
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
  output[out_idx]=val${n?"+skip[out_idx]":""};
}
`)}var qa=Ka(!1),$a=Ka(!0),Ya=oe(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Xa=oe(`
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
`);function Va(n){return oe(`
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
  ${n==="sigmoid"?"let r=1.0/(1.0+exp(-sum));":"let r=sum/256.0;"}
  output[batch*params.out_channels+oc]=r;
}
`)}var ja=Va("sigmoid"),Za=Va("div256"),Ja=oe(`
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
`),Qa=oe(`
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
`);function en(n,_){let g=Math.min(_,256),h=_>g,w=n%4===0?`var ic:u32=0u;
    while(ic<${n}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${n}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,U=`var skip_val:f32=0.0;
    if(c<${n}u){
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
    }`,G=n===_?"":`if(c<${n}u){`,d=n===_?"":"}",f=h?`for(var c:u32=lid.x;c<${n}u;c+=${g}u){`:`let c=lid.x;
  ${G}`,y=h?"}":d,C=h?`for(var c:u32=lid.x;c<${_}u;c+=${g}u){`:"{let c=lid.x;";return oe(`
struct FusedParams { batch:u32, in_channels:u32, in_height:u32, in_width:u32, out_height:u32, out_width:u32, stride:u32, pad:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> dw_weight:array<f32>;
@group(0)@binding(2) var<storage,read> dw_bias:array<f32>;
@group(0)@binding(3) var<storage,read> pw_weight:array<f32>;
@group(0)@binding(4) var<storage,read> pw_bias:array<f32>;
@group(0)@binding(5) var<storage,read_write> output:array<f32>;
@group(0)@binding(6) var<uniform> params:FusedParams;
var<workgroup> shared_dw:array<f32,${n}>;
fn load_input_f(base:u32, y:i32, x:i32, in_h:i32, in_w:i32)->f32 {
  if(y>=0 && y<in_h && x>=0 && x<in_w){ return input[base+u32(y)*u32(in_w)+u32(x)]; }
  return 0.0;
}
@compute @workgroup_size(${g},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${f}
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
  ${y}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${C}
    let pw_base=c*${n}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${w}
    // Skip connection (only for c < inCh)
    ${U}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var tn=oe(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),an=oe(`
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
`),nn=oe(`
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
`);function Xt(n,_){let p=new Map,g=n.dtype??"float32";for(let h=0;h<n.keys.length;h++){let e=n.keys[h],w=n.shapes[h],U=n.offsets[h],G=w.reduce((y,C)=>y*C,1),d,f;if(g==="float32")d=new Float32Array(_,U,G);else{let y=new DataView(_);d=new Float32Array(G);for(let C=0;C<G;C++)d[C]=On(y.getUint16(U+C*2,!0));f=_.slice(U,U+G*2)}p.set(e,{data:d,shape:w,rawF16:f})}return p}function On(n){let _=n>>15&1,p=n>>10&31,g=n&1023;if(p===0){if(g===0)return _?-0:0;let w=-14,U=g/1024;return(_?-1:1)*Math.pow(2,w)*U}if(p===31)return g===0?_?-1/0:1/0:NaN;let h=p-15,e=1+g/1024;return(_?-1:1)*Math.pow(2,h)*e}var Rn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],rn=Rn.map(([n,_,p,g,h])=>({type:"resmodule",inCh:n,outCh:_,h:p,w:p,stride:g,prefix:h})),In=2,Fn=5,zn=8,Nn=11;async function Vt(n,_){if(!navigator.gpu)throw new Error("WebGPU not supported");let p=await navigator.gpu.requestAdapter();if(!p)throw new Error("No GPU adapter found");let g=p.features.has("shader-f16"),h=g?["shader-f16"]:[],e=await p.requestDevice({requiredFeatures:h,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(p.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(p.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(p.limits.maxComputeInvocationsPerWorkgroup,288)}}),w=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(g)try{let a=`enable f16;
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
}`,o=`enable f16;
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
}`,r=e.createShaderModule({code:a}),i=e.createShaderModule({code:o}),t=await r.getCompilationInfo(),D=await i.getCompilationInfo();if(t.messages.some(k=>k.type==="error")||D.messages.some(k=>k.type==="error"))w=!1;else{let k=new Float32Array(2400);k.fill(1);let M=new Uint16Array(2400);M.fill(10516);let b=new Uint16Array(96);b.fill(14336);let m=new Uint16Array(9216);m.fill(8478);let s=new Uint16Array(96);s.fill(12288);let T=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,X=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,J=e.createBuffer({size:k.byteLength,usage:T}),ct=e.createBuffer({size:M.byteLength,usage:T}),dt=e.createBuffer({size:b.byteLength,usage:T}),_t=e.createBuffer({size:384,usage:GPUBufferUsage.STORAGE}),lt=e.createBuffer({size:m.byteLength,usage:T}),mt=e.createBuffer({size:s.byteLength,usage:T}),ft=e.createBuffer({size:384,usage:X}),qe=e.createBuffer({size:384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(J,0,k),e.queue.writeBuffer(ct,0,M),e.queue.writeBuffer(dt,0,b),e.queue.writeBuffer(lt,0,m),e.queue.writeBuffer(mt,0,s);let Re="read-only-storage",vt="storage",Pt=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Re}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Re}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Re}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:vt}}]}),Wa=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:Re}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:Re}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:Re}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:vt}}]}),Mn=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Pt]}),compute:{module:r,entryPoint:"main"}}),En=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[Wa]}),compute:{module:i,entryPoint:"main"}}),Wn=e.createBindGroup({layout:Pt,entries:[{binding:0,resource:{buffer:J}},{binding:1,resource:{buffer:ct}},{binding:2,resource:{buffer:dt}},{binding:3,resource:{buffer:_t}}]}),Hn=e.createBindGroup({layout:Wa,entries:[{binding:0,resource:{buffer:_t}},{binding:1,resource:{buffer:lt}},{binding:2,resource:{buffer:mt}},{binding:3,resource:{buffer:ft}}]}),qt=e.createCommandEncoder(),$t=qt.beginComputePass();$t.setPipeline(Mn),$t.setBindGroup(0,Wn),$t.dispatchWorkgroups(2),$t.end();let Yt=qt.beginComputePass();Yt.setPipeline(En),Yt.setBindGroup(0,Hn),Yt.dispatchWorkgroups(2),Yt.end(),qt.copyBufferToBuffer(ft,0,qe,0,384),e.queue.submit([qt.finish()]),await e.queue.onSubmittedWorkDone(),await qe.mapAsync(GPUMapMode.READ);let Bt=new Float32Array(qe.getMappedRange()),Ha=1.5*.0104*96+.25,Tn=Bt[0]!==0&&Bt[47]!==0&&Bt[95]!==0,Ln=Math.abs(Bt[0]-Ha)<1;w=Tn&&Ln,qe.unmap(),J.destroy(),ct.destroy(),dt.destroy(),_t.destroy(),lt.destroy(),mt.destroy(),ft.destroy(),qe.destroy(),w||console.warn(`[micro-handpose] f16 multi-pass validation FAILED (got ${Bt[0]}, expected ~${Ha.toFixed(2)}) \u2014 falling back to f32`)}}catch{w=!1}let G=n.values().next().value,d=w&&!!G?.rawF16&&!_?.forceF32;console.log(d?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${g}, f16 validated: ${w}, f16 data: ${!!G?.rawF16})`);function f(a){if(d&&a.rawF16){let o=new Uint16Array(a.rawF16);if(o.length%2!==0){let r=new Uint16Array(o.length+1);return r.set(o),r}return o}return a.data}function y(a){if(d&&a.rawF16){let o=a.rawF16.byteLength;return Math.ceil(o/4)*4}return a.data.byteLength}function C(a){return d?Ta(a):a}let H={r:"read-only-storage",s:"storage",u:"uniform"};function l(a){return e.createBindGroupLayout({entries:a.map((o,r)=>({binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[o]}}))})}function O(a){return e.createBindGroupLayout({entries:a.map((o,r)=>o==="t"?{binding:r,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[o]}})})}let v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,te=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,Q=GPUBufferUsage.STORAGE,ge=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,ae=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function A(a,o){return e.createBuffer({size:a,usage:o})}function F(a,o){return e.createBindGroup({layout:a,entries:o.map((r,i)=>({binding:i,resource:"size"in r?{buffer:r}:r}))})}function V(a,o){return e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:o,entryPoint:"main"}})}let W=e.createShaderModule({code:za}),L=e.createShaderModule({code:nn}),I=e.createShaderModule({code:C(Ja)}),Ce=e.createShaderModule({code:C(da)}),he=e.createShaderModule({code:C(ca)}),Me=e.createShaderModule({code:C(_a)}),we=e.createShaderModule({code:C(la)}),be=e.createShaderModule({code:C(Na)}),P=e.createShaderModule({code:qa}),z=e.createShaderModule({code:Ya}),le=e.createShaderModule({code:$a}),N=e.createShaderModule({code:C(Xa)}),me=e.createShaderModule({code:C(ja)}),ee=e.createShaderModule({code:C(Za)}),$e=e.createShaderModule({code:C(Qa)}),Ye=new Map;function ye(a,o){let r=`${a}_${o}`,i=Ye.get(r);return i||(i=e.createShaderModule({code:C(en(a,o))}),Ye.set(r,i)),i}let xe=l(["r","r","r","s","u"]),ke=l(["r","r","r","r","s","u"]),Ue=l(["r","s","u"]),Ee=l(["r","r","r","s","u"]),ve=l(["r","s","u"]),Xe=l(["r","r","s","u"]),Ge=l(["r","r","s","u"]),Ie=l(["r","r","r","s","u"]),Ae=l(["r","r","r","s","u"]),tt=O(["t","s","u"]),at=l(["r","r","r","r","r","r","r","s"]),Fe=l(["r","r","r","r","r","s","u"]),ht=e.createPipelineLayout({bindGroupLayouts:[xe]}),Ct=e.createPipelineLayout({bindGroupLayouts:[ke]}),nt=a=>e.createComputePipeline({layout:ht,compute:{module:a,entryPoint:"main"}}),it=a=>e.createComputePipeline({layout:Ct,compute:{module:a,entryPoint:"main"}}),jt=nt(Ce),Zt=nt(he),kt=it(Me),Ut=it(we),Gt=new Map,wt=new Map,gt=new Map,At=new Map;Gt.set("8,8",jt),wt.set("8,8",Zt),gt.set("8,8",kt),At.set("8,8",Ut);function Ve(a,o,r,i,t){let D=`${o},${r}`,k=a.get(D);return k||(k=t(e.createShaderModule({code:C(i(o,r))})),a.set(D,k)),k}let St=(a,o)=>Ve(Gt,a,o,La,nt),Jt=(a,o)=>Ve(wt,a,o,Oa,nt),Dt=(a,o)=>Ve(gt,a,o,Ra,it),Mt=(a,o)=>Ve(At,a,o,Ia,it),Se=rn.map(a=>{let o=a.stride===2?a.h/2:a.h,r=a.stride===2?a.w/2:a.w,[i,t]=Fa(a.inCh,o),D=a.h>=64,k=o>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:D?Jt(i,t):St(i,t),pwPipeline:k?Mt(i,t):Dt(i,t),dwDispatchX:Math.ceil(r/i),dwDispatchY:Math.ceil(o/t),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(r/i),pwDispatchY:Math.ceil(o/t),pwDispatchZ:k?a.outCh/2:a.outCh}}),Qt=V(Ue,W),rt=V(Ee,be);V(ve,P),V(Xe,z);let st=V(Ge,le),Et=V(Ie,N);V(Ae,me),V(Ae,ee);let De=V(tt,L),Wt=V(at,I),Ht=V(Fe,$e),Tt=1*288*128*128*4,bt=A(3*256*256*4,v),ze=A(3*257*257*4,Q),yt=A(12,ae);e.queue.writeBuffer(yt,0,new Uint32Array([3,256,257]));let K=A(Tt,te),ne=A(Tt,ge),Pe=A(Tt,Q),je=A(3072*64*4,v),Ze=A(3072*32*4,v),xt=A(1536*16*4,v),ie=A(6144*64*4,Q),de=A(260,ge),q=A(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);A(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let ue=e.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Je=A(8,ae);e.queue.writeBuffer(Je,0,new Uint32Array([256,257]));let Lt=n.get("backbone1.1.weight"),Ot=n.get("backbone1.1.bias");if(!Lt||!Ot)throw new Error("Missing input conv weights");let c=f(Lt),u=f(Ot),R=A(c.byteLength,v),E=A(u.byteLength,v),x=A(28,ae);e.queue.writeBuffer(R,0,c),e.queue.writeBuffer(E,0,u),e.queue.writeBuffer(x,0,new Uint32Array([1,3,24,257,257,128,128]));let B=n.get("backbone6.1.weight"),$=n.get("backbone6.1.bias");if(!B||!$)throw new Error("Missing backbone6.1 conv1x1 weights");let j=f(B),pe=f($),_e=A(j.byteLength,v),re=A(pe.byteLength,v),se=A(20,ae);e.queue.writeBuffer(_e,0,j),e.queue.writeBuffer(re,0,pe),e.queue.writeBuffer(se,0,new Uint32Array([1,96,48,32,32]));let ce=n.get("handflag.weight"),Be=n.get("handflag.bias");if(!ce||!Be)throw new Error("Missing handflag weights");let S=f(ce),Y=f(Be),Z=A(S.byteLength,v),fe=A(Y.byteLength,v),ot=A(12,ae);e.queue.writeBuffer(Z,0,S),e.queue.writeBuffer(fe,0,Y),e.queue.writeBuffer(ot,0,new Uint32Array([1,288,1]));let wa=n.get("handedness.weight"),ga=n.get("handedness.bias");if(!wa||!ga)throw new Error("Missing handedness weights");let ba=f(wa),ya=f(ga),ea=A(ba.byteLength,v),ta=A(ya.byteLength,v),xa=A(12,ae);e.queue.writeBuffer(ea,0,ba),e.queue.writeBuffer(ta,0,ya),e.queue.writeBuffer(xa,0,new Uint32Array([1,288,1]));let va=n.get("reg_3d.weight"),Pa=n.get("reg_3d.bias");if(!va||!Pa)throw new Error("Missing reg_3d weights");let Ba=f(va),Ca=f(Pa),aa=A(Ba.byteLength,v),na=A(Ca.byteLength,v),ka=A(12,ae);e.queue.writeBuffer(aa,0,Ba),e.queue.writeBuffer(na,0,Ca),e.queue.writeBuffer(ka,0,new Uint32Array([1,288,63]));let Qe=rn.map(a=>{let{inCh:o,outCh:r,h:i,w:t,stride:D,prefix:k}=a,M=D===2?i/2:i,b=D===2?t/2:t,m=D===2?1:2,s=n.get(`${k}convs.0.weight`),T=n.get(`${k}convs.0.bias`),X=n.get(`${k}convs.1.weight`),J=n.get(`${k}convs.1.bias`);if(!s||!T||!X||!J)throw new Error(`Missing weights for ${k}`);let ct=f(s),dt=f(T),_t=f(X),lt=f(J),mt=A(ct.byteLength,v),ft=A(dt.byteLength,v),qe=A(_t.byteLength,v),Re=A(lt.byteLength,v),vt=A(32,ae),Pt=A(36,ae);return e.queue.writeBuffer(mt,0,ct),e.queue.writeBuffer(ft,0,dt),e.queue.writeBuffer(qe,0,_t),e.queue.writeBuffer(Re,0,lt),e.queue.writeBuffer(vt,0,new Uint32Array([1,o,i,t,M,b,D,m])),e.queue.writeBuffer(Pt,0,new Uint32Array([1,o,r,M,b,Math.max(0,r-o),D,i,t])),{dwWeight:mt,dwBias:ft,pwWeight:qe,pwBias:Re,dwUniform:vt,pwUniform:Pt,spec:a,outH:M,outW:b}});function ut(a){let o=A(a.length*4,ae);return e.queue.writeBuffer(o,0,new Uint32Array(a)),o}let gn=ut([1,96,8,8,16,16]),bn=ut([1,96,16,16,32,32]),yn=ut([1,48,32,32,64,64]);ut([1536*16]),ut([3072*32]),ut([3072*64]);let Ua=F(Ue,[bt,ze,yt]),Ga=F(Ee,[ze,R,E,K,x]),We=[],He=[],Te=[],Le=[];for(let a of Qe)We.push(F(xe,[K,a.dwWeight,a.dwBias,Pe,a.dwUniform])),He.push(F(ke,[Pe,K,a.pwWeight,a.pwBias,ne,a.pwUniform])),Te.push(F(xe,[ne,a.dwWeight,a.dwBias,Pe,a.dwUniform])),Le.push(F(ke,[Pe,ne,a.pwWeight,a.pwBias,K,a.pwUniform]));let xn=F(Ge,[K,xt,ne,gn]),vn=F(Ge,[K,Ze,ne,bn]),Pn=F(Ie,[K,_e,re,ie,se]),Bn=F(Ge,[ie,je,ne,yn]);F(Ae,[K,Z,fe,de,ot]),F(Ae,[K,ea,ta,de,xa]),F(Ae,[K,aa,na,de,ka]);let Ne=F(tt,[ue.createView(),ze,Je]),Cn=F(at,[K,Z,fe,ea,ta,aa,na,de]),ia=24,Aa=[],Sa=[];for(let a=ia;a<Qe.length;a++){let o=Qe[a];Aa.push(F(Fe,[K,o.dwWeight,o.dwBias,o.pwWeight,o.pwBias,ne,o.dwUniform])),Sa.push(F(Fe,[ne,o.dwWeight,o.dwBias,o.pwWeight,o.pwBias,K,o.dwUniform]))}let ra=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});ra.globalCompositeOperation="copy";let Da=new OffscreenCanvas(9,8),Rt=Da.getContext("webgpu"),It=null,sa=null;if(Rt){Rt.configure({device:e,format:"rgba8unorm",alphaMode:"premultiplied"});let a=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),o=e.createShaderModule({code:tn}),r=e.createShaderModule({code:an});It=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:o,entryPoint:"vs"},fragment:{module:r,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),sa=e.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:de}}]})}let Ft=new Float32Array(1),zt=new Float32Array(1),Nt=new Float32Array(63);function Oe(a,o){let r=!0,i=0,t=a.beginComputePass();for(t.setPipeline(rt),t.setBindGroup(0,Ga),t.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=In;i++){let b=r?We[i]:Te[i],m=r?He[i]:Le[i],s=Se[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,b),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.end();let D=r?K:ne;for(a.copyBufferToBuffer(D,0,je,0,3072*64*4),t=a.beginComputePass();i<=Fn;i++){let b=r?We[i]:Te[i],m=r?He[i]:Le[i],s=Se[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,b),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.end();let k=r?K:ne;for(a.copyBufferToBuffer(k,0,Ze,0,3072*32*4),t=a.beginComputePass();i<=zn;i++){let b=r?We[i]:Te[i],m=r?He[i]:Le[i],s=Se[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,b),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.end();let M=r?K:ne;for(a.copyBufferToBuffer(M,0,xt,0,1536*16*4),t=a.beginComputePass();i<=Nn;i++){let b=r?We[i]:Te[i],m=r?He[i]:Le[i],s=Se[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,b),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}t.setPipeline(st),t.setBindGroup(0,xn),t.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),t.end(),r=!1,t=a.beginComputePass();{let b=r?We[i]:Te[i],m=r?He[i]:Le[i],s=Se[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,b),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r,i++}t.setPipeline(st),t.setBindGroup(0,vn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),t.end(),r=!1,t=a.beginComputePass();{let b=r?We[i]:Te[i],m=r?He[i]:Le[i],s=Se[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,b),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r,i++}for(t.setPipeline(Et),t.setBindGroup(0,Pn),t.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),t.setPipeline(st),t.setBindGroup(0,Bn),t.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),t.end(),r=!1,t=a.beginComputePass();i<ia;i++){let b=r?We[i]:Te[i],m=r?He[i]:Le[i],s=Se[i];t.setPipeline(s.dwPipeline),t.setBindGroup(0,b),t.dispatchWorkgroups(s.dwDispatchX,s.dwDispatchY,s.dwDispatchZ),t.setPipeline(s.pwPipeline),t.setBindGroup(0,m),t.dispatchWorkgroups(s.pwDispatchX,s.pwDispatchY,s.pwDispatchZ),r=!r}for(;i<Qe.length;i++){let b=i-ia,m=r?Aa[b]:Sa[b],s=Qe[i];t.setPipeline(Ht),t.setBindGroup(0,m),t.dispatchWorkgroups(s.outW,s.outH,1),r=!r}t.setPipeline(Wt),t.setBindGroup(0,Cn),t.dispatchWorkgroups(1),t.end(),o&&a.copyBufferToBuffer(de,0,o,0,260)}async function Kt(a){e.queue.writeBuffer(bt,0,a);let o=e.createCommandEncoder();{let t=o.beginComputePass();t.setPipeline(Qt),t.setBindGroup(0,Ua),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),t.end()}Oe(o,q),e.queue.submit([o.finish()]);let r=q.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(q.getMappedRange());return Ft[0]=i[0],zt[0]=i[1],Nt.set(i.subarray(2,65)),q.unmap(),{handflag:new Float32Array(Ft),handedness:new Float32Array(zt),landmarks:new Float32Array(Nt)}}async function oa(a){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let o=e.createCommandEncoder();{let t=o.beginComputePass();t.setPipeline(De),t.setBindGroup(0,Ne),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Oe(o,q),e.queue.submit([o.finish()]);let r=q.mapAsync(GPUMapMode.READ);await e.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(q.getMappedRange());return Ft[0]=i[0],zt[0]=i[1],Nt.set(i.subarray(2,65)),q.unmap(),{handflag:new Float32Array(Ft),handedness:new Float32Array(zt),landmarks:new Float32Array(Nt)}}async function Ma(a){if(!It||!sa||!Rt)throw new Error("Render-based readback not available (no WebGPU canvas context)");e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let o=e.createCommandEncoder();{let b=o.beginComputePass();b.setPipeline(De),b.setBindGroup(0,Ne),b.dispatchWorkgroups(16,16,1),b.end()}Oe(o,null);let r=Rt.getCurrentTexture(),i=o.beginRenderPass({colorAttachments:[{view:r.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});i.setPipeline(It),i.setBindGroup(0,sa),i.draw(3),i.end(),e.queue.submit([o.finish()]),await e.queue.onSubmittedWorkDone(),ra.drawImage(Da,0,0);let D=ra.getImageData(0,0,9,8).data,k=new Float32Array(65),M=new DataView(new ArrayBuffer(4));for(let b=0;b<65;b++){let m=b*4;M.setUint8(0,D[m]),M.setUint8(1,D[m+1]),M.setUint8(2,D[m+2]),M.setUint8(3,D[m+3]),k[b]=M.getFloat32(0)}return{handflag:new Float32Array([k[0]]),handedness:new Float32Array([k[1]]),landmarks:new Float32Array(k.subarray(2,65))}}let kn=e.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),ua=0,Un=[q,kn],pt=null,Ke=null;async function pa(a){let o=Un[ua];ua=1-ua,e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let r=e.createCommandEncoder();{let t=r.beginComputePass();t.setPipeline(De),t.setBindGroup(0,Ne),t.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),t.end()}Oe(r,o),e.queue.submit([r.finish()]);let i=null;if(pt!==null&&Ke!==null){await pt;let t=new Float32Array(Ke.getMappedRange());i={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))},Ke.unmap()}return Ke=o,pt=o.mapAsync(GPUMapMode.READ),i}async function Ea(){if(!pt||!Ke)return null;await pt;let a=new Float32Array(Ke.getMappedRange()),o={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return Ke.unmap(),pt=null,Ke=null,o}async function Gn(a=50){let o=new Float32Array(196608);for(let t=0;t<5;t++)await Kt(o);let r=[];for(let t=0;t<a;t++){let D=performance.now();await Kt(o),r.push(performance.now()-D)}let i=r.reduce((t,D)=>t+D,0)/r.length;return{avgMs:i,fps:1e3/i}}async function An(a=50){let o=new Float32Array(196608);for(let k=0;k<5;k++)await Kt(o);let r=[];for(let k=0;k<a;k++){let M=e.createCommandEncoder();{let m=M.beginComputePass();m.setPipeline(Qt),m.setBindGroup(0,Ua),m.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),m.end()}Oe(M,q);let b=performance.now();e.queue.submit([M.finish()]),await e.queue.onSubmittedWorkDone(),r.push(performance.now()-b)}r.sort((k,M)=>k-M);let i=r.reduce((k,M)=>k+M,0)/r.length,t=r[Math.floor(r.length/2)],D=r[0];return{avgMs:i,fps:1e3/i,medianMs:t,minMs:D}}function ei(a){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let o=e.createCommandEncoder();{let r=o.beginComputePass();r.setPipeline(De),r.setBindGroup(0,Ne),r.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),r.end()}Oe(o,q),e.queue.submit([o.finish()])}async function Sn(a,o=50){function r(m){let s=[...m].sort((T,X)=>T-X);return{median:s[Math.floor(s.length/2)],min:s[0]}}for(let m=0;m<10;m++)await oa(a);let i=[];for(let m=0;m<o;m++){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let s=e.createCommandEncoder();{let X=s.beginComputePass();X.setPipeline(De),X.setBindGroup(0,Ne),X.dispatchWorkgroups(16,16,1),X.end()}Oe(s,q);let T=performance.now();e.queue.submit([s.finish()]),await e.queue.onSubmittedWorkDone(),i.push(performance.now()-T)}let t=[];for(let m=0;m<o;m++){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let s=e.createCommandEncoder();{let J=s.beginComputePass();J.setPipeline(De),J.setBindGroup(0,Ne),J.dispatchWorkgroups(16,16,1),J.end()}Oe(s,q),e.queue.submit([s.finish()]);let T=q.mapAsync(GPUMapMode.READ),X=performance.now();await e.queue.onSubmittedWorkDone(),await T,q.getMappedRange(),q.unmap(),t.push(performance.now()-X)}let D=[];for(let m=0;m<o;m++){e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);let s=e.createCommandEncoder();{let X=s.beginComputePass();X.setPipeline(De),X.setBindGroup(0,Ne),X.dispatchWorkgroups(16,16,1),X.end()}Oe(s,q),e.queue.submit([s.finish()]);let T=performance.now();await q.mapAsync(GPUMapMode.READ),q.getMappedRange(),q.unmap(),D.push(performance.now()-T)}let k=[];for(let m=0;m<o;m++){let s=performance.now();await oa(a),k.push(performance.now()-s)}await pa(a);let M=[];for(let m=0;m<o;m++){let s=performance.now();await pa(a),M.push(performance.now()-s)}await Ea();let b=null;if(It){let m=[];for(let s=0;s<o;s++){let T=performance.now();await Ma(a),m.push(performance.now()-T)}b=r(m)}return{gpuOnly:r(i),mapAsyncOnly:r(t),mapAsyncNoWait:r(D),total:r(k),pipelined:r(M),renderReadback:b}}async function Dn(a){let o=[];async function r(t,D,k){let M=e.createBuffer({size:D,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),b=e.createCommandEncoder();b.copyBufferToBuffer(t,0,M,0,D),e.queue.submit([b.finish()]),await e.queue.onSubmittedWorkDone(),await M.mapAsync(GPUMapMode.READ);let m=new Float32Array(M.getMappedRange()),s=1/0,T=-1/0,X=0;for(let J=0;J<m.length;J++)m[J]<s&&(s=m[J]),m[J]>T&&(T=m[J]),m[J]!==0&&X++;M.unmap(),M.destroy(),o.push({layer:k,stats:{min:s,max:T,nonZero:X,total:m.length}})}e.queue.copyExternalImageToTexture({source:a},{texture:ue},[256,256]);{let t=e.createCommandEncoder(),D=t.beginComputePass();D.setPipeline(De),D.setBindGroup(0,Ne),D.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),D.end(),e.queue.submit([t.finish()])}await r(ze,Math.min(ze.size,3*257*257*4),"canvas\u2192bufInput");{let t=e.createCommandEncoder(),D=t.beginComputePass();D.setPipeline(rt),D.setBindGroup(0,Ga),D.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),D.end(),e.queue.submit([t.finish()])}await r(K,Math.min(K.size,3072*128*4),"inputConv\u2192bufA");let i=!0;for(let t=0;t<Math.min(Qe.length,6);t++){let D=i?We[t]:Te[t],k=i?He[t]:Le[t],M=Se[t],b=Qe[t];{let s=e.createCommandEncoder(),T=s.beginComputePass();T.setPipeline(M.dwPipeline),T.setBindGroup(0,D),T.dispatchWorkgroups(M.dwDispatchX,M.dwDispatchY,M.dwDispatchZ),T.end(),e.queue.submit([s.finish()])}await r(Pe,Math.min(Pe.size,b.spec.inCh*b.outH*b.outW*4),`layer${t}.DW\u2192bufDW (${b.spec.prefix})`);{let s=e.createCommandEncoder(),T=s.beginComputePass();T.setPipeline(M.pwPipeline),T.setBindGroup(0,k),T.dispatchWorkgroups(M.pwDispatchX,M.pwDispatchY,M.pwDispatchZ),T.end(),e.queue.submit([s.finish()])}let m=i?ne:K;await r(m,Math.min(m.size,b.spec.outCh*b.outH*b.outW*4),`layer${t}.PW\u2192buf${i?"B":"A"} (${b.spec.prefix})`),i=!i}return o}return{device:e,run:Kt,runFromCanvas:oa,runFromCanvasViaRender:Ma,runFromCanvasPipelined:pa,flushPipelined:Ea,benchmark:Gn,benchmarkGPU:An,benchmarkDiagnostic:Sn,debugLayerOutputs:Dn}}function et(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var sn=et(`
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
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){
    for(var ky:u32=0u;ky<5u;ky=ky+1u){
      for(var kx:u32=0u;kx<5u;kx=kx+1u){
        let in_y=out_y*2u+ky;
        let in_x=out_x*2u+kx;
        if(in_y<params.in_height && in_x<params.in_width){
          let in_idx=batch*params.in_channels*params.in_height*params.in_width+ic*params.in_height*params.in_width+in_y*params.in_width+in_x;
          let w_idx=oc*5u*5u*params.in_channels+ky*5u*params.in_channels+kx*params.in_channels+ic;
          sum=sum+input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum=sum+bias[oc];
  // PReLU
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.out_height*params.out_width+oc*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=result;
}
`),on=et(`
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
`),un=et(`
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
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){
    sum+=dw_output[dw_base+ic*spatial_stride]*pw_weight[w_base+ic];
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
`),pn=et(`
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
`),cn=et(`
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
`),dn=et(`
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
  for(var ic:u32=0u;ic<params.in_channels;ic=ic+1u){
    let in_idx=batch*params.in_channels*params.height*params.width+ic*params.height*params.width+out_y*params.width+out_x;
    let w_idx=oc*params.in_channels+ic;
    sum=sum+input[in_idx]*weight[w_idx];
  }
  sum=sum+bias[oc];
  let a=alpha[oc];
  let result=max(0.0,sum)+a*min(0.0,sum);
  let out_idx=batch*params.out_channels*params.height*params.width+oc*params.height*params.width+out_y*params.width+out_x;
  output[out_idx]=result;
}
`),_n=et(`
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
`);async function ma(n,_){let p;if(_)p=_;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let c=await navigator.gpu.requestAdapter();if(!c)throw new Error("No GPU adapter found");p=await c.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(c.limits.maxStorageBuffersPerShaderStage,8)}})}let g={r:"read-only-storage",s:"storage",u:"uniform"};function h(c){return p.createBindGroupLayout({entries:c.map((u,R)=>u==="t"?{binding:R,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:R,visibility:GPUShaderStage.COMPUTE,buffer:{type:g[u]}})})}let e=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,w=GPUBufferUsage.STORAGE,U=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,G=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(c,u){return p.createBuffer({size:Math.max(c,4),usage:u})}function f(c,u,R){p.queue.writeBuffer(c,u,R)}function y(c){let u=d(c.data.byteLength,e);return f(u,0,c.data),u}let C=Array.from(n.keys());function H(c){let u=n.get(c);if(!u)throw new Error(`Weight not found: ${c}`);return u}function l(...c){let u=C.find(R=>c.every(E=>R.includes(E)));if(!u)throw new Error(`Weight not found for: ${c.join(", ")}`);return H(u)}function O(c){let[,u,R,E]=c.shape,x=new Float32Array(E*25);for(let B=0;B<E;B++)for(let $=0;$<u;$++)for(let j=0;j<R;j++)x[B*25+$*5+j]=c.data[$*R*E+j*E+B];return x}function v(c){let[u,,,R]=c.shape,E=new Float32Array(u*R);for(let x=0;x<u;x++)for(let B=0;B<R;B++)E[x*R+B]=c.data[x*R+B];return E}let te=p.createShaderModule({code:sn}),Q=p.createShaderModule({code:on}),ge=p.createShaderModule({code:un}),ae=p.createShaderModule({code:pn}),A=p.createShaderModule({code:dn}),F=p.createShaderModule({code:cn}),V=p.createShaderModule({code:_n}),W=h(["r","r","r","r","s","u"]),L=h(["r","r","r","s","u"]),I=h(["r","r","r","r","r","s","u"]),Ce=h(["r","r","r","s","u"]),he=h(["r","r","r","r","s","u"]),Me=h(["r","r","s","u"]),we=h(["t","s","u"]);function be(c,u){return p.createComputePipeline({layout:p.createPipelineLayout({bindGroupLayouts:[c]}),compute:{module:u,entryPoint:"main"}})}let P=be(W,te),z=be(L,Q),le=be(I,ge),N=be(Ce,ae),me=be(he,A),ee=be(Me,F),$e=be(we,V),Ye=l("conv2d/Conv2D"),ye=l("batch_normalization/","conv2d/Conv2D"),xe=l("p_re_lu/"),ke=y(Ye),Ue=y(ye),Ee=y(xe),Xe=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12}].map(c=>{let u=l(c.dwKey),R=l(c.pwKey),E=l(c.bnKey),x=l(c.preluKey),B=O(u),$=d(B.byteLength,e);f($,0,B);let j=new Float32Array(c.inCh),pe=d(j.byteLength,e);f(pe,0,j);let _e=v(R),re=d(_e.byteLength,e);f(re,0,_e);let se=y(E),ce=y(x);return{dwWeightBuf:$,dwBiasBuf:pe,pwWeightBuf:re,pwBiasBuf:se,alphaBuf:ce,inCh:c.inCh,outCh:c.outCh,stride:c.stride,inH:c.inH}}),Ge=v(l("conv2d_20/Conv2D")),Ie=d(Ge.byteLength,e);f(Ie,0,Ge);let Ae=y(l("batch_normalization_20/")),tt=y(l("p_re_lu_20/")),at={dwWeightBuf:(()=>{let c=O(l("depthwise_conv2d_19/")),u=d(c.byteLength,e);return f(u,0,c),u})(),dwBiasBuf:(()=>{let c=new Float32Array(256),u=d(c.byteLength,e);return f(u,0,c),u})(),pwWeightBuf:Ie,pwBiasBuf:Ae,alphaBuf:tt,inCh:256,outCh:256,stride:1,inH:12},Fe={dwWeightBuf:(()=>{let c=O(l("depthwise_conv2d_20/")),u=d(c.byteLength,e);return f(u,0,c),u})(),dwBiasBuf:(()=>{let c=new Float32Array(256),u=d(c.byteLength,e);return f(u,0,c),u})(),pwWeightBuf:(()=>{let c=v(l("conv2d_21/")),u=d(c.byteLength,e);return f(u,0,c),u})(),pwBiasBuf:y(l("batch_normalization_21/")),alphaBuf:y(l("p_re_lu_21/")),inCh:256,outCh:256,stride:1,inH:12},ht=v(l("conv2d_23/Conv2D")),Ct=d(ht.byteLength,e);f(Ct,0,ht);let nt=y(l("batch_normalization_23/")),it=y(l("p_re_lu_23/")),jt={dwWeightBuf:(()=>{let c=O(l("depthwise_conv2d_21/")),u=d(c.byteLength,e);return f(u,0,c),u})(),dwBiasBuf:(()=>{let c=new Float32Array(128),u=d(c.byteLength,e);return f(u,0,c),u})(),pwWeightBuf:(()=>{let c=v(l("conv2d_24/")),u=d(c.byteLength,e);return f(u,0,c),u})(),pwBiasBuf:y(l("batch_normalization_24/")),alphaBuf:y(l("p_re_lu_24/")),inCh:128,outCh:128,stride:1,inH:24},Zt={dwWeightBuf:(()=>{let c=O(l("depthwise_conv2d_22/")),u=d(c.byteLength,e);return f(u,0,c),u})(),dwBiasBuf:(()=>{let c=new Float32Array(128),u=d(c.byteLength,e);return f(u,0,c),u})(),pwWeightBuf:(()=>{let c=v(l("conv2d_25/Conv2D1")),u=d(c.byteLength,e);return f(u,0,c),u})(),pwBiasBuf:y(l("batch_normalization_25/")),alphaBuf:y(l("p_re_lu_25/")),inCh:128,outCh:128,stride:1,inH:24},kt=v(l("classifier_palm_16_NO_PRUNING/Conv2D")),Ut=d(kt.byteLength,e);f(Ut,0,kt);let Gt=y(l("classifier_palm_16_NO_PRUNING/BiasAdd")),wt=v(l("regressor_palm_16_NO_PRUNING/Conv2D")),gt=d(wt.byteLength,e);f(gt,0,wt);let At=y(l("regressor_palm_16_NO_PRUNING/BiasAdd")),Ve=v(l("classifier_palm_8_NO_PRUNING/Conv2D")),St=d(Ve.byteLength,e);f(St,0,Ve);let Jt=y(l("classifier_palm_8_NO_PRUNING/BiasAdd")),Dt=v(l("regressor_palm_8_NO_PRUNING/Conv2D")),Mt=d(Dt.byteLength,e);f(Mt,0,Dt);let Se=y(l("regressor_palm_8_NO_PRUNING/BiasAdd")),Qt=36864*3,rt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,st=d(36864*3*4,e),Et=d(rt,w),De=d(rt,w),Wt=d(rt,w),Ht=d(576*128*4,w),Tt=d(576*128*4,w),bt=d(864*4,U),ze=d(15552*4,U),yt=d(576*2*4,U),K=d(576*36*4,U),ne=d(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Pe=d(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),je=d(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ze=d(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),xt=p.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function ie(c,u){return Math.ceil(c/u)}function de(c){let u=d(c.byteLength,G);return f(u,0,c),u}let q=de(new Uint32Array([1,3,32,192,192,96,96]));function ue(c,u,R,E,x){let B=u.stride===2?u.inH/2:u.inH,$=B,j=u.stride===2?1:2,pe=de(new Uint32Array([1,u.inCh,u.inH,u.inH,B,$,u.stride,j])),_e=p.createBindGroup({layout:L,entries:[{binding:0,resource:{buffer:R}},{binding:1,resource:{buffer:u.dwWeightBuf}},{binding:2,resource:{buffer:u.dwBiasBuf}},{binding:3,resource:{buffer:Wt}},{binding:4,resource:{buffer:pe}}]}),re=c.beginComputePass();re.setPipeline(z),re.setBindGroup(0,_e),re.dispatchWorkgroups(ie($,8),ie(B,8),u.inCh),re.end();let se=u.inCh,ce=de(new Uint32Array([1,u.inCh,u.outCh,B,$,se,u.stride,u.inH,u.inH])),Be=p.createBindGroup({layout:I,entries:[{binding:0,resource:{buffer:Wt}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:u.pwWeightBuf}},{binding:3,resource:{buffer:u.pwBiasBuf}},{binding:4,resource:{buffer:u.alphaBuf}},{binding:5,resource:{buffer:E}},{binding:6,resource:{buffer:ce}}]}),S=c.beginComputePass();S.setPipeline(le),S.setBindGroup(0,Be),S.dispatchWorkgroups(ie($,8),ie(B,8),u.outCh),S.end()}function Je(c,u,R,E,x,B,$,j,pe){let _e=de(new Uint32Array([1,B,$,j,pe])),re=p.createBindGroup({layout:Ce,entries:[{binding:0,resource:{buffer:u}},{binding:1,resource:{buffer:R}},{binding:2,resource:{buffer:E}},{binding:3,resource:{buffer:x}},{binding:4,resource:{buffer:_e}}]}),se=c.beginComputePass();se.setPipeline(N),se.setBindGroup(0,re),se.dispatchWorkgroups(ie(pe,8),ie(j,8),$),se.end()}function Lt(c,u,R,E,x,B,$,j,pe,_e){let re=de(new Uint32Array([1,$,j,pe,_e])),se=p.createBindGroup({layout:he,entries:[{binding:0,resource:{buffer:u}},{binding:1,resource:{buffer:R}},{binding:2,resource:{buffer:E}},{binding:3,resource:{buffer:x}},{binding:4,resource:{buffer:B}},{binding:5,resource:{buffer:re}}]}),ce=c.beginComputePass();ce.setPipeline(me),ce.setBindGroup(0,se),ce.dispatchWorkgroups(ie(_e,8),ie(pe,8),j),ce.end()}async function Ot(c){p.queue.copyExternalImageToTexture({source:c},{texture:xt},[192,192]);let u=de(new Uint32Array([192,192,192])),R=p.createBindGroup({layout:we,entries:[{binding:0,resource:xt.createView()},{binding:1,resource:{buffer:st}},{binding:2,resource:{buffer:u}}]}),E=p.createCommandEncoder();{let S=E.beginComputePass();S.setPipeline($e),S.setBindGroup(0,R),S.dispatchWorkgroups(ie(192,16),ie(192,16),1),S.end()}{let S=p.createBindGroup({layout:W,entries:[{binding:0,resource:{buffer:st}},{binding:1,resource:{buffer:ke}},{binding:2,resource:{buffer:Ue}},{binding:3,resource:{buffer:Ee}},{binding:4,resource:{buffer:Et}},{binding:5,resource:{buffer:q}}]}),Y=E.beginComputePass();Y.setPipeline(P),Y.setBindGroup(0,S),Y.dispatchWorkgroups(ie(96,8),ie(96,8),32),Y.end()}let x=Et,B=De;for(let S=0;S<Xe.length;S++){let Y=Xe[S];ue(E,Y,x,B,x);let Z=x;x=B,B=Z,S===10&&E.copyBufferToBuffer(x,0,Ht,0,576*128*4)}ue(E,at,x,B,x);{let S=x;x=B,B=S}ue(E,Fe,x,B,x);{let S=x;x=B,B=S}Je(E,x,Ut,Gt,bt,256,6,12,12),Je(E,x,gt,At,ze,256,108,12,12),Lt(E,x,Ct,nt,it,B,256,128,12,12);{let S=x;x=B,B=S}{let S=de(new Uint32Array([1,128,12,12,24,24])),Y=p.createBindGroup({layout:Me,entries:[{binding:0,resource:{buffer:x}},{binding:1,resource:{buffer:Ht}},{binding:2,resource:{buffer:B}},{binding:3,resource:{buffer:S}}]}),Z=E.beginComputePass();Z.setPipeline(ee),Z.setBindGroup(0,Y),Z.dispatchWorkgroups(ie(24,8),ie(24,8),128),Z.end()}{let S=x;x=B,B=S}ue(E,jt,x,B,x);{let S=x;x=B,B=S}ue(E,Zt,x,B,x);{let S=x;x=B,B=S}Je(E,x,St,Jt,yt,128,2,24,24),Je(E,x,Mt,Se,K,128,36,24,24),E.copyBufferToBuffer(bt,0,ne,0,864*4),E.copyBufferToBuffer(ze,0,Pe,0,15552*4),E.copyBufferToBuffer(yt,0,je,0,576*2*4),E.copyBufferToBuffer(K,0,Ze,0,576*36*4),p.queue.submit([E.finish()]),await Promise.all([ne.mapAsync(GPUMapMode.READ),Pe.mapAsync(GPUMapMode.READ),je.mapAsync(GPUMapMode.READ),Ze.mapAsync(GPUMapMode.READ)]);let $=new Float32Array(ne.getMappedRange()).slice(),j=new Float32Array(Pe.getMappedRange()).slice(),pe=new Float32Array(je.getMappedRange()).slice(),_e=new Float32Array(Ze.getMappedRange()).slice();ne.unmap(),Pe.unmap(),je.unmap(),Ze.unmap();let re=2016,se=new Float32Array(re),ce=new Float32Array(re*18),Be=0;for(let S=0;S<12;S++)for(let Y=0;Y<12;Y++)for(let Z=0;Z<6;Z++){se[Be]=$[Z*144+S*12+Y];for(let fe=0;fe<18;fe++){let ot=Z*18+fe;ce[Be*18+fe]=j[ot*144+S*12+Y]}Be++}for(let S=0;S<24;S++)for(let Y=0;Y<24;Y++)for(let Z=0;Z<2;Z++){se[Be]=pe[Z*576+S*24+Y];for(let fe=0;fe<18;fe++){let ot=Z*18+fe;ce[Be*18+fe]=_e[ot*576+S*24+Y]}Be++}return{scores:se,regressors:ce}}return{device:p,run:Ot}}function Kn(){let n=[];for(let _=0;_<12;_++)for(let p=0;p<12;p++){let g=(p+.5)/12,h=(_+.5)/12;for(let e=0;e<6;e++)n.push({x:g,y:h})}for(let _=0;_<24;_++)for(let p=0;p<24;p++){let g=(p+.5)/24,h=(_+.5)/24;for(let e=0;e<2;e++)n.push({x:g,y:h})}return n}var ln=Kn();function qn(n){return 1/(1+Math.exp(-n))}function mn(n,_){let p=[],{scores:g,regressors:h}=n,e=192;for(let w=0;w<ln.length;w++){let U=qn(g[w]);if(U<_)continue;let G=ln[w],d=w*18,f=G.x+h[d+0]/e,y=G.y+h[d+1]/e,C=h[d+2]/e,H=h[d+3]/e,l=[];for(let O=0;O<7;O++){let v=G.x+h[d+4+O*2]/e,te=G.y+h[d+4+O*2+1]/e;l.push([v,te])}p.push({score:U,box:[f,y,C,H],keypoints:l})}return p}function fn(n,_){if(n.length===0)return[];let p=[...n].sort((e,w)=>w.score-e.score),g=[],h=new Set;for(let e=0;e<p.length;e++)if(!h.has(e)){g.push(p[e]);for(let w=e+1;w<p.length;w++)h.has(w)||$n(p[e],p[w])>_&&h.add(w)}return g}function $n(n,_){let p=n.box[0]-n.box[2]/2,g=n.box[1]-n.box[3]/2,h=n.box[0]+n.box[2]/2,e=n.box[1]+n.box[3]/2,w=_.box[0]-_.box[2]/2,U=_.box[1]-_.box[3]/2,G=_.box[0]+_.box[2]/2,d=_.box[1]+_.box[3]/2,f=Math.max(p,w),y=Math.max(g,U),C=Math.min(h,G),H=Math.min(e,d),l=Math.max(0,C-f),O=Math.max(0,H-y),v=l*O,te=(h-p)*(e-g),Q=(G-w)*(d-U),ge=te+Q-v;return ge>0?v/ge:0}function Yn(n){let[_,p,g,h]=n.box,e=n.keypoints[0],w=n.keypoints[2],U=w[0]-e[0],G=w[1]-e[1],d=Math.atan2(U,G),y=Math.max(g,h)*2.6,C=.5,H=Math.sqrt(U*U+G*G),l=H>0?U/H*y*C*.5:0,O=H>0?G/H*y*C*.5:0;return{centerX:_+l,centerY:p+O,width:y,height:y,rotation:d}}function fa(n,_={}){let{scoreThreshold:p=.5,nmsThreshold:g=.3,maxHands:h=2}=_;async function e(U){let G=await n.run(U),d=mn(G,p);return fn(d,g).slice(0,h).map(Yn)}async function w(U){let G=await n.run(U),d=mn(G,p);return fn(d,g).slice(0,h)}return{detect:e,detectRaw:w,model:n}}function hn(n,_=256){let p=Math.cos(n.rotation),g=Math.sin(n.rotation),h=n.width/_,e=n.height/_,w=h*p,U=-e*g,G=h*g,d=e*p,f=n.centerX-(w*_/2+U*_/2),y=n.centerY-(G*_/2+d*_/2),C=w*d-U*G,H=d/C,l=-U/C,O=-G/C,v=w/C,te=-(H*f+l*y),Q=-(O*f+v*y);return{forward:[w,U,f,G,d,y],inverse:[H,l,te,O,v,Q]}}function ha(n,_){let{forward:p}=hn(_,1),[g,h,e,w,U,G]=p;return n.map(d=>({x:g*d.x+h*d.y+e,y:w*d.x+U*d.y+G,z:d.z}))}var wn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Xn(n={}){let{weightsUrl:_,scoreThreshold:p=.5,forceF32:g=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let h=_??wn,e=h.endsWith("/")?h:`${h}/`,w=`${e}weights_f16.json`,U=`${e}weights_f16.bin`,[G,d]=await Promise.all([fetch(w),fetch(U)]);if(!G.ok)throw new Error(`Failed to fetch weights metadata: ${G.status}`);if(!d.ok)throw new Error(`Failed to fetch weights binary: ${d.status}`);let f=await G.json(),y=await d.arrayBuffer(),C=Xt(f,y),H=await Vt(C,{forceF32:g});if(!g){let W=new OffscreenCanvas(256,256),L=W.getContext("2d");L.fillStyle="#886644",L.fillRect(0,0,256,256),L.fillStyle="#cc9966",L.fillRect(50,50,156,156);let I=await H.runFromCanvas(W);I.landmarks.every(he=>he===0)&&I.handflag.every(he=>he===0)&&(console.warn("[micro-handpose] f16 model produced all-zero output \u2014 recompiling with f32"),H.device.destroy(),H=await Vt(C,{forceF32:!0}))}let l=null;function O(){return l||(l=new OffscreenCanvas(256,256)),l}async function v(W){if(W instanceof HTMLCanvasElement||W instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&W instanceof ImageBitmap)return W;let L=O();L.width=256,L.height=256;let I=L.getContext("2d");return W instanceof ImageData?I.putImageData(W,0,0):I.drawImage(W,0,0,256,256),L}function te(W,L,I){let Ce=W[0];if(Ce<p)return null;let he=L[0]>.5,Me=[];for(let we=0;we<21;we++)Me.push({x:I[we*3],y:I[we*3+1],z:I[we*3+2]});return{score:Ce,handedness:he?"right":"left",landmarks:Me}}async function Q(W){let L=await v(W),I=await H.runFromCanvas(L);return te(I.handflag,I.handedness,I.landmarks)}async function ge(W){let L=await v(W),I=await H.runFromCanvasPipelined(L);return I?te(I.handflag,I.handedness,I.landmarks):null}async function ae(){let W=await H.flushPipelined();return W?te(W.handflag,W.handedness,W.landmarks):null}function A(){H.device.destroy(),l=null}async function F(W){let L=await v(W);return H.benchmarkDiagnostic(L)}async function V(W){let L=await v(W);return H.debugLayerOutputs(L)}return{detect:Q,detectPipelined:ge,flushPipelined:ae,dispose:A,benchmarkDiagnostic:F,debugLayerOutputs:V}}async function Vn(n={}){let{weightsUrl:_,palmWeightsUrl:p,scoreThreshold:g=.5,palmScoreThreshold:h=.5,maxHands:e=2,forceF32:w=!1}=n;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let U=_??wn,G=U.endsWith("/")?U:`${U}/`;if(!p)throw new Error("palmWeightsUrl is required for createFullHandpose");let d=p.endsWith("/")?p:`${p}/`,[f,y,C,H]=await Promise.all([fetch(`${G}weights_f16.json`),fetch(`${G}weights_f16.bin`),fetch(`${d}palm_detection_weights.json`),fetch(`${d}palm_detection_weights.bin`)]);if(!f.ok)throw new Error(`Failed to fetch landmark weights metadata: ${f.status}`);if(!y.ok)throw new Error(`Failed to fetch landmark weights binary: ${y.status}`);if(!C.ok)throw new Error(`Failed to fetch palm weights metadata: ${C.status}`);if(!H.ok)throw new Error(`Failed to fetch palm weights binary: ${H.status}`);let[l,O,v,te]=await Promise.all([f.json(),y.arrayBuffer(),C.json(),H.arrayBuffer()]),Q=Xt(l,O),ge=Xt(v,te),ae=await Vt(Q,{forceF32:w}),A=await ma(ge),F=fa(A,{scoreThreshold:h,maxHands:e}),V=null,W=null;function L(){return V||(V=new OffscreenCanvas(192,192)),V}function I(){return W||(W=new OffscreenCanvas(256,256)),W}async function Ce(P){if(P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas){if(P.width===192&&P.height===192)return P;let N=L();return N.width=192,N.height=192,N.getContext("2d").drawImage(P,0,0,192,192),N}if(typeof ImageBitmap<"u"&&P instanceof ImageBitmap){if(P.width===192&&P.height===192)return P;let N=L();return N.width=192,N.height=192,N.getContext("2d").drawImage(P,0,0,192,192),N}let z=L();z.width=192,z.height=192;let le=z.getContext("2d");if(P instanceof ImageData){let N=new OffscreenCanvas(P.width,P.height);N.getContext("2d").putImageData(P,0,0),le.drawImage(N,0,0,192,192)}else le.drawImage(P,0,0,192,192);return z}function he(P,z,le,N){let me=I();me.width=256,me.height=256;let ee=me.getContext("2d"),$e=Math.cos(-z.rotation),Ye=Math.sin(-z.rotation);ee.clearRect(0,0,256,256),ee.save(),ee.translate(128,128),ee.scale(z.width*le/256,z.height*N/256),ee.rotate(-z.rotation),ee.translate(-128,-128);let ye=z.centerX*le,xe=z.centerY*N;ee.restore();let ke=256/(z.width*le),Ue=256/(z.height*N),Ee=Math.cos(z.rotation),ve=Math.sin(z.rotation),Xe=Ee*ke,Ge=ve*ke,Ie=-ve*Ue,Ae=Ee*Ue,tt=-ye*Xe-xe*Ie+128,at=-ye*Ge-xe*Ae+128;if(ee.setTransform(Xe,Ge,Ie,Ae,tt,at),P instanceof ImageData){let Fe=new OffscreenCanvas(P.width,P.height);Fe.getContext("2d").putImageData(P,0,0),ee.drawImage(Fe,0,0)}else ee.drawImage(P,0,0);return ee.setTransform(1,0,0,1,0,0),me}function Me(P){return P instanceof HTMLCanvasElement||P instanceof OffscreenCanvas?[P.width,P.height]:typeof ImageBitmap<"u"&&P instanceof ImageBitmap?[P.width,P.height]:P instanceof ImageData?[P.width,P.height]:P instanceof HTMLVideoElement?[P.videoWidth,P.videoHeight]:P instanceof HTMLImageElement?[P.naturalWidth,P.naturalHeight]:[256,256]}async function we(P){let z=await Ce(P),le=await F.detect(z);if(le.length===0)return[];let[N,me]=Me(P),ee=[];for(let $e of le){let Ye=he(P,$e,N,me),ye=await ae.runFromCanvas(Ye),xe=ye.handflag[0];if(xe<g)continue;let ke=ye.handedness[0]>.5,Ue=[];for(let ve=0;ve<21;ve++)Ue.push({x:ye.landmarks[ve*3],y:ye.landmarks[ve*3+1],z:ye.landmarks[ve*3+2]});let Ee=ha(Ue,$e);ee.push({score:xe,handedness:ke?"right":"left",landmarks:Ee,palmScore:0})}return ee}function be(){ae.device.destroy(),A.device.destroy(),V=null,W=null}return{detect:we,dispose:be}}function jn(n){return n.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Zn=jn(`
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

  // Clamp to bounds (or return 0 for out-of-bounds)
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
`);function Jn(n){let _=n.createShaderModule({code:Zn}),p=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),g=n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[p]}),compute:{module:_,entryPoint:"main"}});function h(e,w,U,G,d,f,y){let C=new Uint32Array([d,f,y,0]),H=n.createBuffer({size:C.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(H,0,C);let l=new Float32Array(G),O=new Float32Array(8);O.set(l);let v=n.createBuffer({size:O.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(v,0,O);let te=n.createBindGroup({layout:p,entries:[{binding:0,resource:w.createView()},{binding:1,resource:{buffer:U}},{binding:2,resource:{buffer:H}},{binding:3,resource:{buffer:v}}]}),Q=e.beginComputePass();Q.setPipeline(g),Q.setBindGroup(0,te),Q.dispatchWorkgroups(Math.ceil(y/16),Math.ceil(y/16),1),Q.end()}return{crop:h}}export{ma as compilePalmModel,hn as computeCropTransform,Jn as createCropPipeline,Vn as createFullHandpose,Xn as createHandpose,fa as createPalmDetector,ha as projectLandmarksToOriginal};
