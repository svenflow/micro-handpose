function W(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function et(o){let w=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],h="enable f16;"+o;for(let g of w)for(;h.includes(`${g}:array<f32>`);)h=h.replace(`${g}:array<f32>`,`${g}:array<f16>`);return h}var Oe=W(`
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
`),Fe=W(`
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
`),Ie=W(`
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
`),ze=W(`
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
`);function at(o,w){return Fe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function tt(o,w){return Oe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function it(o,w){return Ie.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function nt(o,w){return ze.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function rt(o,w){return[8,8]}var st=W(`
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
`),ut=W(`
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
`);function ot(o){return W(`
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
`)}var pt=ot(!1),dt=ot(!0),_t=W(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),ct=W(`
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
`);function lt(o){return W(`
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
`)}var wt=lt("sigmoid"),mt=lt("div256"),ht=W(`
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
`),bt=W(`
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
`);function ft(o,w){let g=Math.min(w,256),P=w>g,M=o%4===0?`var ic:u32=0u;
    while(ic<${o}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${o}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,E=`var skip_val:f32=0.0;
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
    }`,A=o===w?"":`if(c<${o}u){`,b=o===w?"":"}",V=P?`for(var c:u32=lid.x;c<${o}u;c+=${g}u){`:`let c=lid.x;
  ${A}`,y=P?"}":b,H=P?`for(var c:u32=lid.x;c<${w}u;c+=${g}u){`:"{let c=lid.x;";return W(`
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
  ${V}
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
  ${H}
    let pw_base=c*${o}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${M}
    // Skip connection (only for c < inCh)
    ${E}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var gt=W(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),yt=W(`
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
`),xt=W(`
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
`);function Pt(o,w){let h=new Map,g=o.dtype??"float32";for(let P=0;P<o.keys.length;P++){let t=o.keys[P],M=o.shapes[P],E=o.offsets[P],A=M.reduce((y,H)=>y*H,1),b,V;if(g==="float32")b=new Float32Array(w,E,A);else{let y=new DataView(w);b=new Float32Array(A);for(let H=0;H<A;H++)b[H]=oi(y.getUint16(E+H*2,!0));V=w.slice(E,E+A*2)}h.set(t,{data:b,shape:M,rawF16:V})}return h}function oi(o){let w=o>>15&1,h=o>>10&31,g=o&1023;if(h===0){if(g===0)return w?-0:0;let M=-14,E=g/1024;return(w?-1:1)*Math.pow(2,M)*E}if(h===31)return g===0?w?-1/0:1/0:NaN;let P=h-15,t=1+g/1024;return(w?-1:1)*Math.pow(2,P)*t}var pi=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],vt=pi.map(([o,w,h,g,P])=>({type:"resmodule",inCh:o,outCh:w,h,w:h,stride:g,prefix:P})),di=2,_i=5,ci=8,li=11;async function kt(o,w){if(!navigator.gpu)throw new Error("WebGPU not supported");let h=await navigator.gpu.requestAdapter();if(!h)throw new Error("No GPU adapter found");let g=h.features.has("shader-f16"),P=g?["shader-f16"]:[],t=await h.requestDevice({requiredFeatures:P,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(h.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(h.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(h.limits.maxComputeInvocationsPerWorkgroup,288)}}),M=!1;if(g)try{M=!(await t.createShaderModule({code:`enable f16;
@compute @workgroup_size(1)
fn main() { var x: f16 = f16(1.0); _ = x; }`}).getCompilationInfo()).messages.some(i=>i.type==="error")}catch{M=!1}let E=o.values().next().value,A=M&&!!E?.rawF16&&!w?.forceF32;console.log(A?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${g}, f16 validated: ${M}, f16 data: ${!!E?.rawF16})`);function b(a){if(A&&a.rawF16){let s=new Uint16Array(a.rawF16);if(s.length%2!==0){let i=new Uint16Array(s.length+1);return i.set(s),i}return s}return a.data}function V(a){if(A&&a.rawF16){let s=a.rawF16.byteLength;return Math.ceil(s/4)*4}return a.data.byteLength}function y(a){return A?et(a):a}let H={r:"read-only-storage",s:"storage",u:"uniform"};function k(a){return t.createBindGroupLayout({entries:a.map((s,i)=>({binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[s]}}))})}function Q(a){return t.createBindGroupLayout({entries:a.map((s,i)=>s==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[s]}})})}let x=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ee=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,K=GPUBufferUsage.STORAGE,ue=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,L=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function _(a,s){return t.createBuffer({size:a,usage:s})}function B(a,s){return t.createBindGroup({layout:a,entries:s.map((i,n)=>({binding:n,resource:"size"in i?{buffer:i}:i}))})}function T(a,s){return t.createComputePipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),compute:{module:s,entryPoint:"main"}})}let ye=t.createShaderModule({code:st}),f=t.createShaderModule({code:xt}),G=t.createShaderModule({code:y(ht)}),S=t.createShaderModule({code:y(Fe)}),oe=t.createShaderModule({code:y(Oe)}),xe=t.createShaderModule({code:y(Ie)}),pe=t.createShaderModule({code:y(ze)}),j=t.createShaderModule({code:y(ut)}),Bt=t.createShaderModule({code:pt}),Ut=t.createShaderModule({code:_t}),Dt=t.createShaderModule({code:dt}),St=t.createShaderModule({code:y(ct)}),Gt=t.createShaderModule({code:y(wt)}),Wt=t.createShaderModule({code:y(mt)}),At=t.createShaderModule({code:y(bt)}),qe=new Map;function hi(a,s){let i=`${a}_${s}`,n=qe.get(i);return n||(n=t.createShaderModule({code:y(ft(a,s))}),qe.set(i,n)),n}let ve=k(["r","r","r","s","u"]),Pe=k(["r","r","r","r","s","u"]),$e=k(["r","s","u"]),Ne=k(["r","r","r","s","u"]),Ct=k(["r","s","u"]),Mt=k(["r","r","s","u"]),de=k(["r","r","s","u"]),Xe=k(["r","r","r","s","u"]),re=k(["r","r","r","s","u"]),Ye=Q(["t","s","u"]),Ze=k(["r","r","r","r","r","r","r","s"]),ke=k(["r","r","r","r","r","s","u"]),Et=t.createPipelineLayout({bindGroupLayouts:[ve]}),Ht=t.createPipelineLayout({bindGroupLayouts:[Pe]}),_e=a=>t.createComputePipeline({layout:Et,compute:{module:a,entryPoint:"main"}}),ce=a=>t.createComputePipeline({layout:Ht,compute:{module:a,entryPoint:"main"}}),Tt=_e(S),Lt=_e(oe),Rt=ce(xe),Ot=ce(pe),Ve=new Map,Ke=new Map,je=new Map,Je=new Map;Ve.set("8,8",Tt),Ke.set("8,8",Lt),je.set("8,8",Rt),Je.set("8,8",Ot);function le(a,s,i,n,e){let d=`${s},${i}`,c=a.get(d);return c||(c=e(t.createShaderModule({code:y(n(s,i))})),a.set(d,c)),c}let Ft=(a,s)=>le(Ve,a,s,at,_e),It=(a,s)=>le(Ke,a,s,tt,_e),zt=(a,s)=>le(je,a,s,it,ce),qt=(a,s)=>le(Je,a,s,nt,ce),N=vt.map(a=>{let s=a.stride===2?a.h/2:a.h,i=a.stride===2?a.w/2:a.w,[n,e]=rt(a.inCh,s),d=a.h>=64,c=s>=16&&a.inCh>=288&&a.outCh>=288&&a.outCh%2===0;return{dwPipeline:d?It(n,e):Ft(n,e),pwPipeline:c?qt(n,e):zt(n,e),dwDispatchX:Math.ceil(i/n),dwDispatchY:Math.ceil(s/e),dwDispatchZ:a.inCh,pwDispatchX:Math.ceil(i/n),pwDispatchY:Math.ceil(s/e),pwDispatchZ:c?a.outCh/2:a.outCh}}),Qe=T($e,ye),ea=T(Ne,j);T(Ct,Bt),T(Mt,Ut);let Be=T(de,Dt),$t=T(Xe,St);T(re,Gt),T(re,Wt);let X=T(Ye,f),Nt=T(Ze,G),Xt=T(ke,At),Ue=1*288*128*128*4,aa=_(3*256*256*4,x),se=_(3*257*257*4,K),ta=_(12,L);t.queue.writeBuffer(ta,0,new Uint32Array([3,256,257]));let U=_(Ue,ee),R=_(Ue,ue),ae=_(Ue,K),ia=_(3072*64*4,x),na=_(3072*32*4,x),ra=_(1536*16*4,x),sa=_(6144*64*4,K),te=_(260,ue),v=_(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);_(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let O=t.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ua=_(8,L);t.queue.writeBuffer(ua,0,new Uint32Array([256,257]));let oa=o.get("backbone1.1.weight"),pa=o.get("backbone1.1.bias");if(!oa||!pa)throw new Error("Missing input conv weights");let da=b(oa),_a=b(pa),ca=_(da.byteLength,x),la=_(_a.byteLength,x),wa=_(28,L);t.queue.writeBuffer(ca,0,da),t.queue.writeBuffer(la,0,_a),t.queue.writeBuffer(wa,0,new Uint32Array([1,3,24,257,257,128,128]));let ma=o.get("backbone6.1.weight"),ha=o.get("backbone6.1.bias");if(!ma||!ha)throw new Error("Missing backbone6.1 conv1x1 weights");let ba=b(ma),fa=b(ha),ga=_(ba.byteLength,x),ya=_(fa.byteLength,x),xa=_(20,L);t.queue.writeBuffer(ga,0,ba),t.queue.writeBuffer(ya,0,fa),t.queue.writeBuffer(xa,0,new Uint32Array([1,96,48,32,32]));let va=o.get("handflag.weight"),Pa=o.get("handflag.bias");if(!va||!Pa)throw new Error("Missing handflag weights");let ka=b(va),Ba=b(Pa),De=_(ka.byteLength,x),Se=_(Ba.byteLength,x),Ua=_(12,L);t.queue.writeBuffer(De,0,ka),t.queue.writeBuffer(Se,0,Ba),t.queue.writeBuffer(Ua,0,new Uint32Array([1,288,1]));let Da=o.get("handedness.weight"),Sa=o.get("handedness.bias");if(!Da||!Sa)throw new Error("Missing handedness weights");let Ga=b(Da),Wa=b(Sa),Ge=_(Ga.byteLength,x),We=_(Wa.byteLength,x),Aa=_(12,L);t.queue.writeBuffer(Ge,0,Ga),t.queue.writeBuffer(We,0,Wa),t.queue.writeBuffer(Aa,0,new Uint32Array([1,288,1]));let Ca=o.get("reg_3d.weight"),Ma=o.get("reg_3d.bias");if(!Ca||!Ma)throw new Error("Missing reg_3d weights");let Ea=b(Ca),Ha=b(Ma),Ae=_(Ea.byteLength,x),Ce=_(Ha.byteLength,x),Ta=_(12,L);t.queue.writeBuffer(Ae,0,Ea),t.queue.writeBuffer(Ce,0,Ha),t.queue.writeBuffer(Ta,0,new Uint32Array([1,288,63]));let J=vt.map(a=>{let{inCh:s,outCh:i,h:n,w:e,stride:d,prefix:c}=a,l=d===2?n/2:n,p=d===2?e/2:e,u=d===1?2:1,r=o.get(`${c}convs.0.weight`),m=o.get(`${c}convs.0.bias`),D=o.get(`${c}convs.1.weight`),C=o.get(`${c}convs.1.bias`);if(!r||!m||!D||!C)throw new Error(`Missing weights for ${c}`);let $a=b(r),Na=b(m),Xa=b(D),Ya=b(C),Za=_($a.byteLength,x),Va=_(Na.byteLength,x),Ka=_(Xa.byteLength,x),ja=_(Ya.byteLength,x),Ja=_(32,L),Qa=_(36,L);return t.queue.writeBuffer(Za,0,$a),t.queue.writeBuffer(Va,0,Na),t.queue.writeBuffer(Ka,0,Xa),t.queue.writeBuffer(ja,0,Ya),t.queue.writeBuffer(Ja,0,new Uint32Array([1,s,n,e,l,p,d,u])),t.queue.writeBuffer(Qa,0,new Uint32Array([1,s,i,l,p,Math.max(0,i-s),d,n,e])),{dwWeight:Za,dwBias:Va,pwWeight:Ka,pwBias:ja,dwUniform:Ja,pwUniform:Qa,spec:a,outH:l,outW:p}});function ie(a){let s=_(a.length*4,L);return t.queue.writeBuffer(s,0,new Uint32Array(a)),s}let Yt=ie([1,96,8,8,16,16]),Zt=ie([1,96,16,16,32,32]),Vt=ie([1,48,32,32,64,64]);ie([1536*16]),ie([3072*32]),ie([3072*64]);let La=B($e,[aa,se,ta]),Ra=B(Ne,[se,ca,la,U,wa]),F=[],I=[],z=[],q=[];for(let a of J)F.push(B(ve,[U,a.dwWeight,a.dwBias,ae,a.dwUniform])),I.push(B(Pe,[ae,U,a.pwWeight,a.pwBias,R,a.pwUniform])),z.push(B(ve,[R,a.dwWeight,a.dwBias,ae,a.dwUniform])),q.push(B(Pe,[ae,R,a.pwWeight,a.pwBias,U,a.pwUniform]));let Kt=B(de,[U,ra,R,Yt]),jt=B(de,[U,na,R,Zt]),Jt=B(Xe,[U,ga,ya,sa,xa]),Qt=B(de,[sa,ia,R,Vt]);B(re,[U,De,Se,te,Ua]),B(re,[U,Ge,We,te,Aa]),B(re,[U,Ae,Ce,te,Ta]);let Y=B(Ye,[O.createView(),se,ua]),ei=B(Ze,[U,De,Se,Ge,We,Ae,Ce,te]),Me=24,Oa=[],Fa=[];for(let a=Me;a<J.length;a++){let s=J[a];Oa.push(B(ke,[U,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,R,s.dwUniform])),Fa.push(B(ke,[R,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,U,s.dwUniform]))}let Ee=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Ee.globalCompositeOperation="copy";let Ia=new OffscreenCanvas(9,8),we=Ia.getContext("webgpu"),me=null,He=null;if(we){we.configure({device:t,format:"rgba8unorm",alphaMode:"premultiplied"});let a=t.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),s=t.createShaderModule({code:gt}),i=t.createShaderModule({code:yt});me=t.createRenderPipeline({layout:t.createPipelineLayout({bindGroupLayouts:[a]}),vertex:{module:s,entryPoint:"vs"},fragment:{module:i,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),He=t.createBindGroup({layout:a,entries:[{binding:0,resource:{buffer:te}}]})}let he=new Float32Array(1),be=new Float32Array(1),fe=new Float32Array(63);function $(a,s){let i=!0,n=0,e=a.beginComputePass();for(e.setPipeline(ea),e.setBindGroup(0,Ra),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);n<=di;n++){let p=i?F[n]:z[n],u=i?I[n]:q[n],r=N[n];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}e.end();let d=i?U:R;for(a.copyBufferToBuffer(d,0,ia,0,3072*64*4),e=a.beginComputePass();n<=_i;n++){let p=i?F[n]:z[n],u=i?I[n]:q[n],r=N[n];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}e.end();let c=i?U:R;for(a.copyBufferToBuffer(c,0,na,0,3072*32*4),e=a.beginComputePass();n<=ci;n++){let p=i?F[n]:z[n],u=i?I[n]:q[n],r=N[n];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}e.end();let l=i?U:R;for(a.copyBufferToBuffer(l,0,ra,0,1536*16*4),e=a.beginComputePass();n<=li;n++){let p=i?F[n]:z[n],u=i?I[n]:q[n],r=N[n];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}e.setPipeline(Be),e.setBindGroup(0,Kt),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),i=!1,e=a.beginComputePass();{let p=i?F[n]:z[n],u=i?I[n]:q[n],r=N[n];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i,n++}e.setPipeline(Be),e.setBindGroup(0,jt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),i=!1,e=a.beginComputePass();{let p=i?F[n]:z[n],u=i?I[n]:q[n],r=N[n];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i,n++}for(e.setPipeline($t),e.setBindGroup(0,Jt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(Be),e.setBindGroup(0,Qt),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),i=!1,e=a.beginComputePass();n<Me;n++){let p=i?F[n]:z[n],u=i?I[n]:q[n],r=N[n];e.setPipeline(r.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(r.dwDispatchX,r.dwDispatchY,r.dwDispatchZ),e.setPipeline(r.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(r.pwDispatchX,r.pwDispatchY,r.pwDispatchZ),i=!i}for(;n<J.length;n++){let p=n-Me,u=i?Oa[p]:Fa[p],r=J[n];e.setPipeline(Xt),e.setBindGroup(0,u),e.dispatchWorkgroups(r.outW,r.outH,1),i=!i}e.setPipeline(Nt),e.setBindGroup(0,ei),e.dispatchWorkgroups(1),e.end(),s&&a.copyBufferToBuffer(te,0,s,0,260)}async function ge(a){t.queue.writeBuffer(aa,0,a);let s=t.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(Qe),e.setBindGroup(0,La),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}$(s,v),t.queue.submit([s.finish()]);let i=v.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await i;let n=new Float32Array(v.getMappedRange());return he[0]=n[0],be[0]=n[1],fe.set(n.subarray(2,65)),v.unmap(),{handflag:new Float32Array(he),handedness:new Float32Array(be),landmarks:new Float32Array(fe)}}async function Te(a){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let s=t.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(X),e.setBindGroup(0,Y),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}$(s,v),t.queue.submit([s.finish()]);let i=v.mapAsync(GPUMapMode.READ);await t.queue.onSubmittedWorkDone(),await i;let n=new Float32Array(v.getMappedRange());return he[0]=n[0],be[0]=n[1],fe.set(n.subarray(2,65)),v.unmap(),{handflag:new Float32Array(he),handedness:new Float32Array(be),landmarks:new Float32Array(fe)}}async function za(a){if(!me||!He||!we)throw new Error("Render-based readback not available (no WebGPU canvas context)");t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let s=t.createCommandEncoder();{let p=s.beginComputePass();p.setPipeline(X),p.setBindGroup(0,Y),p.dispatchWorkgroups(16,16,1),p.end()}$(s,null);let i=we.getCurrentTexture(),n=s.beginRenderPass({colorAttachments:[{view:i.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});n.setPipeline(me),n.setBindGroup(0,He),n.draw(3),n.end(),t.queue.submit([s.finish()]),await t.queue.onSubmittedWorkDone(),Ee.drawImage(Ia,0,0);let d=Ee.getImageData(0,0,9,8).data,c=new Float32Array(65),l=new DataView(new ArrayBuffer(4));for(let p=0;p<65;p++){let u=p*4;l.setUint8(0,d[u]),l.setUint8(1,d[u+1]),l.setUint8(2,d[u+2]),l.setUint8(3,d[u+3]),c[p]=l.getFloat32(0)}return{handflag:new Float32Array([c[0]]),handedness:new Float32Array([c[1]]),landmarks:new Float32Array(c.subarray(2,65))}}let ai=t.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Le=0,ti=[v,ai],ne=null,Z=null;async function Re(a){let s=ti[Le];Le=1-Le,t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let i=t.createCommandEncoder();{let e=i.beginComputePass();e.setPipeline(X),e.setBindGroup(0,Y),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}$(i,s),t.queue.submit([i.finish()]);let n=null;if(ne!==null&&Z!==null){await ne;let e=new Float32Array(Z.getMappedRange());n={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},Z.unmap()}return Z=s,ne=s.mapAsync(GPUMapMode.READ),n}async function qa(){if(!ne||!Z)return null;await ne;let a=new Float32Array(Z.getMappedRange()),s={handflag:new Float32Array([a[0]]),handedness:new Float32Array([a[1]]),landmarks:new Float32Array(a.subarray(2,65))};return Z.unmap(),ne=null,Z=null,s}async function ii(a=50){let s=new Float32Array(196608);for(let e=0;e<5;e++)await ge(s);let i=[];for(let e=0;e<a;e++){let d=performance.now();await ge(s),i.push(performance.now()-d)}let n=i.reduce((e,d)=>e+d,0)/i.length;return{avgMs:n,fps:1e3/n}}async function ni(a=50){let s=new Float32Array(196608);for(let c=0;c<5;c++)await ge(s);let i=[];for(let c=0;c<a;c++){let l=t.createCommandEncoder();{let u=l.beginComputePass();u.setPipeline(Qe),u.setBindGroup(0,La),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),u.end()}$(l,v);let p=performance.now();t.queue.submit([l.finish()]),await t.queue.onSubmittedWorkDone(),i.push(performance.now()-p)}i.sort((c,l)=>c-l);let n=i.reduce((c,l)=>c+l,0)/i.length,e=i[Math.floor(i.length/2)],d=i[0];return{avgMs:n,fps:1e3/n,medianMs:e,minMs:d}}function ri(a){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let s=t.createCommandEncoder();{let i=s.beginComputePass();i.setPipeline(X),i.setBindGroup(0,Y),i.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),i.end()}$(s,v),t.queue.submit([s.finish()])}async function si(a,s=50){function i(u){let r=[...u].sort((m,D)=>m-D);return{median:r[Math.floor(r.length/2)],min:r[0]}}for(let u=0;u<10;u++)await Te(a);let n=[];for(let u=0;u<s;u++){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let r=t.createCommandEncoder();{let D=r.beginComputePass();D.setPipeline(X),D.setBindGroup(0,Y),D.dispatchWorkgroups(16,16,1),D.end()}$(r,v);let m=performance.now();t.queue.submit([r.finish()]),await t.queue.onSubmittedWorkDone(),n.push(performance.now()-m)}let e=[];for(let u=0;u<s;u++){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let r=t.createCommandEncoder();{let C=r.beginComputePass();C.setPipeline(X),C.setBindGroup(0,Y),C.dispatchWorkgroups(16,16,1),C.end()}$(r,v),t.queue.submit([r.finish()]);let m=v.mapAsync(GPUMapMode.READ),D=performance.now();await t.queue.onSubmittedWorkDone(),await m,v.getMappedRange(),v.unmap(),e.push(performance.now()-D)}let d=[];for(let u=0;u<s;u++){t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);let r=t.createCommandEncoder();{let D=r.beginComputePass();D.setPipeline(X),D.setBindGroup(0,Y),D.dispatchWorkgroups(16,16,1),D.end()}$(r,v),t.queue.submit([r.finish()]);let m=performance.now();await v.mapAsync(GPUMapMode.READ),v.getMappedRange(),v.unmap(),d.push(performance.now()-m)}let c=[];for(let u=0;u<s;u++){let r=performance.now();await Te(a),c.push(performance.now()-r)}await Re(a);let l=[];for(let u=0;u<s;u++){let r=performance.now();await Re(a),l.push(performance.now()-r)}await qa();let p=null;if(me){let u=[];for(let r=0;r<s;r++){let m=performance.now();await za(a),u.push(performance.now()-m)}p=i(u)}return{gpuOnly:i(n),mapAsyncOnly:i(e),mapAsyncNoWait:i(d),total:i(c),pipelined:i(l),renderReadback:p}}async function ui(a){let s=[];async function i(e,d,c){let l=t.createBuffer({size:d,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),p=t.createCommandEncoder();p.copyBufferToBuffer(e,0,l,0,d),t.queue.submit([p.finish()]),await t.queue.onSubmittedWorkDone(),await l.mapAsync(GPUMapMode.READ);let u=new Float32Array(l.getMappedRange()),r=1/0,m=-1/0,D=0;for(let C=0;C<u.length;C++)u[C]<r&&(r=u[C]),u[C]>m&&(m=u[C]),u[C]!==0&&D++;l.unmap(),l.destroy(),s.push({layer:c,stats:{min:r,max:m,nonZero:D,total:u.length}})}t.queue.copyExternalImageToTexture({source:a},{texture:O},[256,256]);{let e=t.createCommandEncoder(),d=e.beginComputePass();d.setPipeline(X),d.setBindGroup(0,Y),d.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),d.end(),t.queue.submit([e.finish()])}await i(se,Math.min(se.size,3*257*257*4),"canvas\u2192bufInput");{let e=t.createCommandEncoder(),d=e.beginComputePass();d.setPipeline(ea),d.setBindGroup(0,Ra),d.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),d.end(),t.queue.submit([e.finish()])}await i(U,Math.min(U.size,3072*128*4),"inputConv\u2192bufA");let n=!0;for(let e=0;e<Math.min(J.length,6);e++){let d=n?F[e]:z[e],c=n?I[e]:q[e],l=N[e],p=J[e];{let r=t.createCommandEncoder(),m=r.beginComputePass();m.setPipeline(l.dwPipeline),m.setBindGroup(0,d),m.dispatchWorkgroups(l.dwDispatchX,l.dwDispatchY,l.dwDispatchZ),m.end(),t.queue.submit([r.finish()])}await i(ae,Math.min(ae.size,p.spec.inCh*p.outH*p.outW*4),`layer${e}.DW\u2192bufDW (${p.spec.prefix})`);{let r=t.createCommandEncoder(),m=r.beginComputePass();m.setPipeline(l.pwPipeline),m.setBindGroup(0,c),m.dispatchWorkgroups(l.pwDispatchX,l.pwDispatchY,l.pwDispatchZ),m.end(),t.queue.submit([r.finish()])}let u=n?R:U;await i(u,Math.min(u.size,p.spec.outCh*p.outH*p.outW*4),`layer${e}.PW\u2192buf${n?"B":"A"} (${p.spec.prefix})`),n=!n}return s}return{device:t,run:ge,runFromCanvas:Te,runFromCanvasViaRender:za,runFromCanvasPipelined:Re,flushPipelined:qa,benchmark:ii,benchmarkGPU:ni,benchmarkDiagnostic:si,debugLayerOutputs:ui,_device:t,_benchmarkSubmitOnly:ri}}var wi="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function mi(o={}){let{weightsUrl:w,scoreThreshold:h=.5,forceF32:g=!1}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let P=w??wi,t=P.endsWith("/")?P:`${P}/`,M=`${t}weights_f16.json`,E=`${t}weights_f16.bin`,[A,b]=await Promise.all([fetch(M),fetch(E)]);if(!A.ok)throw new Error(`Failed to fetch weights metadata: ${A.status}`);if(!b.ok)throw new Error(`Failed to fetch weights binary: ${b.status}`);let V=await A.json(),y=await b.arrayBuffer(),H=Pt(V,y),k=await kt(H,{forceF32:g}),Q=null;function x(){return Q||(Q=new OffscreenCanvas(256,256)),Q}async function ee(f){if(f instanceof HTMLCanvasElement||f instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&f instanceof ImageBitmap)return f;let G=x();G.width=256,G.height=256;let S=G.getContext("2d");return f instanceof ImageData?S.putImageData(f,0,0):S.drawImage(f,0,0,256,256),G}function K(f,G,S){let oe=f[0];if(oe<h)return null;let xe=G[0]>.5,pe=[];for(let j=0;j<21;j++)pe.push({x:S[j*3],y:S[j*3+1],z:S[j*3+2]});return{score:oe,handedness:xe?"right":"left",landmarks:pe}}async function ue(f){let G=await ee(f),S=await k.runFromCanvas(G);return K(S.handflag,S.handedness,S.landmarks)}async function L(f){let G=await ee(f),S=await k.runFromCanvasPipelined(G);return S?K(S.handflag,S.handedness,S.landmarks):null}async function _(){let f=await k.flushPipelined();return f?K(f.handflag,f.handedness,f.landmarks):null}function B(){k.device.destroy(),Q=null}async function T(f){let G=await ee(f);return k.benchmarkDiagnostic(G)}async function ye(f){let G=await ee(f);return k.debugLayerOutputs(G)}return{detect:ue,detectPipelined:L,flushPipelined:_,dispose:B,benchmarkDiagnostic:T,debugLayerOutputs:ye}}export{mi as createHandpose};
