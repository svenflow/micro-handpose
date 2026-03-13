function A(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function et(o){let w=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],f="enable f16;"+o;for(let b of w)for(;f.includes(`${b}:array<f32>`);)f=f.replace(`${b}:array<f32>`,`${b}:array<f16>`);return f}var ze=A(`
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
`),qe=A(`
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
`),$e=A(`
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
`),Ne=A(`
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
`);function at(o,w){return qe.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function tt(o,w){return ze.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function it(o,w){return $e.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function nt(o,w){return Ne.replace("@compute @workgroup_size(8,8,1)",`@compute @workgroup_size(${o},${w},1)`)}function rt(o,w){return[8,8]}var st=A(`
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
`),ut=A(`
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
`);function ot(o){return A(`
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
`)}var pt=ot(!1),dt=ot(!0),_t=A(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),ct=A(`
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
`);function lt(o){return A(`
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
`)}var wt=lt("sigmoid"),mt=lt("div256"),ft=A(`
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
`),ht=A(`
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
`);function bt(o,w){let b=Math.min(w,256),k=w>b,W=o%4===0?`var ic:u32=0u;
    while(ic<${o}u){
      sum0+=shared_dw[ic]*pw_weight[pw_base+ic];
      sum1+=shared_dw[ic+1u]*pw_weight[pw_base+ic+1u];
      sum2+=shared_dw[ic+2u]*pw_weight[pw_base+ic+2u];
      sum3+=shared_dw[ic+3u]*pw_weight[pw_base+ic+3u];
      ic+=4u;
    }
    var pw_sum=sum0+sum1+sum2+sum3+pw_bias[c];`:`var ic:u32=0u;
    while(ic<${o}u){ sum0+=shared_dw[ic]*pw_weight[pw_base+ic]; ic+=1u; }
    var pw_sum=sum0+pw_bias[c];`,O=`var skip_val:f32=0.0;
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
    }`,E=o===w?"":`if(c<${o}u){`,M=o===w?"":"}",x=k?`for(var c:u32=lid.x;c<${o}u;c+=${b}u){`:`let c=lid.x;
  ${E}`,Y=k?"}":M,g=k?`for(var c:u32=lid.x;c<${w}u;c+=${b}u){`:"{let c=lid.x;";return A(`
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
@compute @workgroup_size(${b},1,1)
fn main(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>){
  let out_x=wid.x;
  let out_y=wid.y;
  let outH=params.out_height;
  let outW=params.out_width;
  if(out_x>=outW||out_y>=outH){return;}
  let inH=i32(params.in_height);
  let inW=i32(params.in_width);
  // Step 1: DW 5x5 convolution
  ${x}
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
  ${Y}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${g}
    let pw_base=c*${o}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${W}
    // Skip connection (only for c < inCh)
    ${O}
    let result=max(0.0,pw_sum+skip_val);
    output[c*outH*outW+out_y*outW+out_x]=result;
  }
}
`)}var gt=A(`
@vertex fn vs(@builtin(vertex_index) vid:u32)->@builtin(position) vec4<f32>{
  var pos=array<vec2<f32>,3>(vec2(-1,-1),vec2(3,-1),vec2(-1,3));
  return vec4(pos[vid],0,1);
}
`),yt=A(`
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
`),xt=A(`
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
`);function Pt(o,w){let f=new Map,b=o.dtype??"float32";for(let k=0;k<o.keys.length;k++){let a=o.keys[k],W=o.shapes[k],O=o.offsets[k],E=W.reduce((Y,g)=>Y*g,1),M,x;if(b==="float32")M=new Float32Array(w,O,E);else{let Y=new DataView(w);M=new Float32Array(E);for(let g=0;g<E;g++)M[g]=pi(Y.getUint16(O+g*2,!0));x=w.slice(O,O+E*2)}f.set(a,{data:M,shape:W,rawF16:x})}return f}function pi(o){let w=o>>15&1,f=o>>10&31,b=o&1023;if(f===0){if(b===0)return w?-0:0;let W=-14,O=b/1024;return(w?-1:1)*Math.pow(2,W)*O}if(f===31)return b===0?w?-1/0:1/0:NaN;let k=f-15,a=1+b/1024;return(w?-1:1)*Math.pow(2,k)*a}var di=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],vt=di.map(([o,w,f,b,k])=>({type:"resmodule",inCh:o,outCh:w,h:f,w:f,stride:b,prefix:k})),_i=2,ci=5,li=8,wi=11;async function kt(o,w){if(!navigator.gpu)throw new Error("WebGPU not supported");let f=await navigator.gpu.requestAdapter();if(!f)throw new Error("No GPU adapter found");let b=f.features.has("shader-f16"),k=b?["shader-f16"]:[],a=await f.requestDevice({requiredFeatures:k,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(f.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(f.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(f.limits.maxComputeInvocationsPerWorkgroup,288)}}),W=!1;if(typeof navigator<"u"&&/iPhone|iPad/.test(navigator.userAgent)&&/Safari/.test(navigator.userAgent))console.warn("[micro-handpose] iOS Safari detected \u2014 disabling f16 due to WebGPU bug");else if(b)try{let s=a.createShaderModule({code:`enable f16;
@group(0) @binding(0) var<storage, read> weights: array<f16>;
@group(0) @binding(1) var<storage, read> bias: array<f16>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= 64u) { return; }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < 64u; i = i + 1u) {
    sum = sum + f32(weights[gid.x * 64u + i]) * input[i];
  }
  output[gid.x] = sum + f32(bias[gid.x]);
}`});if((await s.getCompilationInfo()).messages.some(i=>i.type==="error"))W=!1;else{let i=new Uint16Array(4096);i.fill(15360);let e=new Uint16Array(64);e.fill(14336);let d=new Float32Array(64);d.fill(1);let _=a.createBuffer({size:i.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),c=a.createBuffer({size:e.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),p=a.createBuffer({size:d.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),u=a.createBuffer({size:256,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),n=a.createBuffer({size:256,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});a.queue.writeBuffer(_,0,i),a.queue.writeBuffer(c,0,e),a.queue.writeBuffer(p,0,d);let m=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),v=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[m]}),compute:{module:s,entryPoint:"main"}}),G=a.createBindGroup({layout:m,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:p}},{binding:3,resource:{buffer:u}}]}),ne=a.createCommandEncoder(),Q=ne.beginComputePass();Q.setPipeline(v),Q.setBindGroup(0,G),Q.dispatchWorkgroups(1),Q.end(),ne.copyBufferToBuffer(u,0,n,0,256),a.queue.submit([ne.finish()]),await a.queue.onSubmittedWorkDone(),await n.mapAsync(GPUMapMode.READ);let pe=new Float32Array(n.getMappedRange());W=Math.abs(pe[0]-64.5)<.5&&Math.abs(pe[63]-64.5)<.5,n.unmap(),_.destroy(),c.destroy(),p.destroy(),u.destroy(),n.destroy(),W||console.warn("[micro-handpose] f16 storage validation FAILED (realistic test) \u2014 falling back to f32")}}catch{W=!1}let E=o.values().next().value,M=W&&!!E?.rawF16&&!w?.forceF32;console.log(M?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${b}, f16 validated: ${W}, f16 data: ${!!E?.rawF16})`);function x(t){if(M&&t.rawF16){let s=new Uint16Array(t.rawF16);if(s.length%2!==0){let r=new Uint16Array(s.length+1);return r.set(s),r}return s}return t.data}function Y(t){if(M&&t.rawF16){let s=t.rawF16.byteLength;return Math.ceil(s/4)*4}return t.data.byteLength}function g(t){return M?et(t):t}let R={r:"read-only-storage",s:"storage",u:"uniform"};function D(t){return a.createBindGroupLayout({entries:t.map((s,r)=>({binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:R[s]}}))})}function Pe(t){return a.createBindGroupLayout({entries:t.map((s,r)=>s==="t"?{binding:r,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:R[s]}})})}let y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,re=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,se=GPUBufferUsage.STORAGE,de=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,T=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(t,s){return a.createBuffer({size:t,usage:s})}function B(t,s){return a.createBindGroup({layout:t,entries:s.map((r,i)=>({binding:i,resource:"size"in r?{buffer:r}:r}))})}function H(t,s){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:s,entryPoint:"main"}})}let h=a.createShaderModule({code:st}),C=a.createShaderModule({code:xt}),S=a.createShaderModule({code:g(ft)}),_e=a.createShaderModule({code:g(qe)}),ke=a.createShaderModule({code:g(ze)}),ce=a.createShaderModule({code:g($e)}),j=a.createShaderModule({code:g(Ne)}),Bt=a.createShaderModule({code:g(ut)}),Ut=a.createShaderModule({code:pt}),St=a.createShaderModule({code:_t}),Gt=a.createShaderModule({code:dt}),Dt=a.createShaderModule({code:g(ct)}),Ct=a.createShaderModule({code:g(wt)}),At=a.createShaderModule({code:g(mt)}),Wt=a.createShaderModule({code:g(ht)}),Ye=new Map;function hi(t,s){let r=`${t}_${s}`,i=Ye.get(r);return i||(i=a.createShaderModule({code:g(bt(t,s))}),Ye.set(r,i)),i}let Be=D(["r","r","r","s","u"]),Ue=D(["r","r","r","r","s","u"]),Xe=D(["r","s","u"]),Ze=D(["r","r","r","s","u"]),Mt=D(["r","s","u"]),Et=D(["r","r","s","u"]),le=D(["r","r","s","u"]),Ve=D(["r","r","r","s","u"]),ue=D(["r","r","r","s","u"]),Ke=Pe(["t","s","u"]),je=D(["r","r","r","r","r","r","r","s"]),Se=D(["r","r","r","r","r","s","u"]),Ht=a.createPipelineLayout({bindGroupLayouts:[Be]}),Tt=a.createPipelineLayout({bindGroupLayouts:[Ue]}),we=t=>a.createComputePipeline({layout:Ht,compute:{module:t,entryPoint:"main"}}),me=t=>a.createComputePipeline({layout:Tt,compute:{module:t,entryPoint:"main"}}),Lt=we(_e),Ot=we(ke),Rt=me(ce),Ft=me(j),Je=new Map,Qe=new Map,ea=new Map,aa=new Map;Je.set("8,8",Lt),Qe.set("8,8",Ot),ea.set("8,8",Rt),aa.set("8,8",Ft);function fe(t,s,r,i,e){let d=`${s},${r}`,_=t.get(d);return _||(_=e(a.createShaderModule({code:g(i(s,r))})),t.set(d,_)),_}let It=(t,s)=>fe(Je,t,s,at,we),zt=(t,s)=>fe(Qe,t,s,tt,we),qt=(t,s)=>fe(ea,t,s,it,me),$t=(t,s)=>fe(aa,t,s,nt,me),X=vt.map(t=>{let s=t.stride===2?t.h/2:t.h,r=t.stride===2?t.w/2:t.w,[i,e]=rt(t.inCh,s),d=t.h>=64,_=s>=16&&t.inCh>=288&&t.outCh>=288&&t.outCh%2===0;return{dwPipeline:d?zt(i,e):It(i,e),pwPipeline:_?$t(i,e):qt(i,e),dwDispatchX:Math.ceil(r/i),dwDispatchY:Math.ceil(s/e),dwDispatchZ:t.inCh,pwDispatchX:Math.ceil(r/i),pwDispatchY:Math.ceil(s/e),pwDispatchZ:_?t.outCh/2:t.outCh}}),ta=H(Xe,h),ia=H(Ze,Bt);H(Mt,Ut),H(Et,St);let Ge=H(le,Gt),Nt=H(Ve,Dt);H(ue,Ct),H(ue,At);let Z=H(Ke,C),Yt=H(je,S),Xt=H(Se,Wt),De=1*288*128*128*4,na=l(3*256*256*4,y),oe=l(3*257*257*4,se),ra=l(12,T);a.queue.writeBuffer(ra,0,new Uint32Array([3,256,257]));let U=l(De,re),L=l(De,de),ee=l(De,se),sa=l(3072*64*4,y),ua=l(3072*32*4,y),oa=l(1536*16*4,y),pa=l(6144*64*4,se),ae=l(260,de),P=l(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);l(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let F=a.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),da=l(8,T);a.queue.writeBuffer(da,0,new Uint32Array([256,257]));let _a=o.get("backbone1.1.weight"),ca=o.get("backbone1.1.bias");if(!_a||!ca)throw new Error("Missing input conv weights");let la=x(_a),wa=x(ca),ma=l(la.byteLength,y),fa=l(wa.byteLength,y),ha=l(28,T);a.queue.writeBuffer(ma,0,la),a.queue.writeBuffer(fa,0,wa),a.queue.writeBuffer(ha,0,new Uint32Array([1,3,24,257,257,128,128]));let ba=o.get("backbone6.1.weight"),ga=o.get("backbone6.1.bias");if(!ba||!ga)throw new Error("Missing backbone6.1 conv1x1 weights");let ya=x(ba),xa=x(ga),va=l(ya.byteLength,y),Pa=l(xa.byteLength,y),ka=l(20,T);a.queue.writeBuffer(va,0,ya),a.queue.writeBuffer(Pa,0,xa),a.queue.writeBuffer(ka,0,new Uint32Array([1,96,48,32,32]));let Ba=o.get("handflag.weight"),Ua=o.get("handflag.bias");if(!Ba||!Ua)throw new Error("Missing handflag weights");let Sa=x(Ba),Ga=x(Ua),Ce=l(Sa.byteLength,y),Ae=l(Ga.byteLength,y),Da=l(12,T);a.queue.writeBuffer(Ce,0,Sa),a.queue.writeBuffer(Ae,0,Ga),a.queue.writeBuffer(Da,0,new Uint32Array([1,288,1]));let Ca=o.get("handedness.weight"),Aa=o.get("handedness.bias");if(!Ca||!Aa)throw new Error("Missing handedness weights");let Wa=x(Ca),Ma=x(Aa),We=l(Wa.byteLength,y),Me=l(Ma.byteLength,y),Ea=l(12,T);a.queue.writeBuffer(We,0,Wa),a.queue.writeBuffer(Me,0,Ma),a.queue.writeBuffer(Ea,0,new Uint32Array([1,288,1]));let Ha=o.get("reg_3d.weight"),Ta=o.get("reg_3d.bias");if(!Ha||!Ta)throw new Error("Missing reg_3d weights");let La=x(Ha),Oa=x(Ta),Ee=l(La.byteLength,y),He=l(Oa.byteLength,y),Ra=l(12,T);a.queue.writeBuffer(Ee,0,La),a.queue.writeBuffer(He,0,Oa),a.queue.writeBuffer(Ra,0,new Uint32Array([1,288,63]));let J=vt.map(t=>{let{inCh:s,outCh:r,h:i,w:e,stride:d,prefix:_}=t,c=d===2?i/2:i,p=d===2?e/2:e,u=d===1?2:1,n=o.get(`${_}convs.0.weight`),m=o.get(`${_}convs.0.bias`),v=o.get(`${_}convs.1.weight`),G=o.get(`${_}convs.1.bias`);if(!n||!m||!v||!G)throw new Error(`Missing weights for ${_}`);let ne=x(n),Q=x(m),pe=x(v),Xa=x(G),Za=l(ne.byteLength,y),Va=l(Q.byteLength,y),Ka=l(pe.byteLength,y),ja=l(Xa.byteLength,y),Ja=l(32,T),Qa=l(36,T);return a.queue.writeBuffer(Za,0,ne),a.queue.writeBuffer(Va,0,Q),a.queue.writeBuffer(Ka,0,pe),a.queue.writeBuffer(ja,0,Xa),a.queue.writeBuffer(Ja,0,new Uint32Array([1,s,i,e,c,p,d,u])),a.queue.writeBuffer(Qa,0,new Uint32Array([1,s,r,c,p,Math.max(0,r-s),d,i,e])),{dwWeight:Za,dwBias:Va,pwWeight:Ka,pwBias:ja,dwUniform:Ja,pwUniform:Qa,spec:t,outH:c,outW:p}});function te(t){let s=l(t.length*4,T);return a.queue.writeBuffer(s,0,new Uint32Array(t)),s}let Zt=te([1,96,8,8,16,16]),Vt=te([1,96,16,16,32,32]),Kt=te([1,48,32,32,64,64]);te([1536*16]),te([3072*32]),te([3072*64]);let Fa=B(Xe,[na,oe,ra]),Ia=B(Ze,[oe,ma,fa,U,ha]),I=[],z=[],q=[],$=[];for(let t of J)I.push(B(Be,[U,t.dwWeight,t.dwBias,ee,t.dwUniform])),z.push(B(Ue,[ee,U,t.pwWeight,t.pwBias,L,t.pwUniform])),q.push(B(Be,[L,t.dwWeight,t.dwBias,ee,t.dwUniform])),$.push(B(Ue,[ee,L,t.pwWeight,t.pwBias,U,t.pwUniform]));let jt=B(le,[U,oa,L,Zt]),Jt=B(le,[U,ua,L,Vt]),Qt=B(Ve,[U,va,Pa,pa,ka]),ei=B(le,[pa,sa,L,Kt]);B(ue,[U,Ce,Ae,ae,Da]),B(ue,[U,We,Me,ae,Ea]),B(ue,[U,Ee,He,ae,Ra]);let V=B(Ke,[F.createView(),oe,da]),ai=B(je,[U,Ce,Ae,We,Me,Ee,He,ae]),Te=24,za=[],qa=[];for(let t=Te;t<J.length;t++){let s=J[t];za.push(B(Se,[U,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,L,s.dwUniform])),qa.push(B(Se,[L,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,U,s.dwUniform]))}let Le=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Le.globalCompositeOperation="copy";let $a=new OffscreenCanvas(9,8),he=$a.getContext("webgpu"),be=null,Oe=null;if(he){he.configure({device:a,format:"rgba8unorm",alphaMode:"premultiplied"});let t=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),s=a.createShaderModule({code:gt}),r=a.createShaderModule({code:yt});be=a.createRenderPipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),vertex:{module:s,entryPoint:"vs"},fragment:{module:r,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),Oe=a.createBindGroup({layout:t,entries:[{binding:0,resource:{buffer:ae}}]})}let ge=new Float32Array(1),ye=new Float32Array(1),xe=new Float32Array(63);function N(t,s){let r=!0,i=0,e=t.beginComputePass();for(e.setPipeline(ia),e.setBindGroup(0,Ia),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=_i;i++){let p=r?I[i]:q[i],u=r?z[i]:$[i],n=X[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.end();let d=r?U:L;for(t.copyBufferToBuffer(d,0,sa,0,3072*64*4),e=t.beginComputePass();i<=ci;i++){let p=r?I[i]:q[i],u=r?z[i]:$[i],n=X[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.end();let _=r?U:L;for(t.copyBufferToBuffer(_,0,ua,0,3072*32*4),e=t.beginComputePass();i<=li;i++){let p=r?I[i]:q[i],u=r?z[i]:$[i],n=X[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.end();let c=r?U:L;for(t.copyBufferToBuffer(c,0,oa,0,1536*16*4),e=t.beginComputePass();i<=wi;i++){let p=r?I[i]:q[i],u=r?z[i]:$[i],n=X[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.setPipeline(Ge),e.setBindGroup(0,jt),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),r=!1,e=t.beginComputePass();{let p=r?I[i]:q[i],u=r?z[i]:$[i],n=X[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r,i++}e.setPipeline(Ge),e.setBindGroup(0,Jt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),r=!1,e=t.beginComputePass();{let p=r?I[i]:q[i],u=r?z[i]:$[i],n=X[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r,i++}for(e.setPipeline(Nt),e.setBindGroup(0,Qt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(Ge),e.setBindGroup(0,ei),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),r=!1,e=t.beginComputePass();i<Te;i++){let p=r?I[i]:q[i],u=r?z[i]:$[i],n=X[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}for(;i<J.length;i++){let p=i-Te,u=r?za[p]:qa[p],n=J[i];e.setPipeline(Xt),e.setBindGroup(0,u),e.dispatchWorkgroups(n.outW,n.outH,1),r=!r}e.setPipeline(Yt),e.setBindGroup(0,ai),e.dispatchWorkgroups(1),e.end(),s&&t.copyBufferToBuffer(ae,0,s,0,260)}async function ve(t){a.queue.writeBuffer(na,0,t);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(ta),e.setBindGroup(0,Fa),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}N(s,P),a.queue.submit([s.finish()]);let r=P.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(P.getMappedRange());return ge[0]=i[0],ye[0]=i[1],xe.set(i.subarray(2,65)),P.unmap(),{handflag:new Float32Array(ge),handedness:new Float32Array(ye),landmarks:new Float32Array(xe)}}async function Re(t){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(Z),e.setBindGroup(0,V),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}N(s,P),a.queue.submit([s.finish()]);let r=P.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(P.getMappedRange());return ge[0]=i[0],ye[0]=i[1],xe.set(i.subarray(2,65)),P.unmap(),{handflag:new Float32Array(ge),handedness:new Float32Array(ye),landmarks:new Float32Array(xe)}}async function Na(t){if(!be||!Oe||!he)throw new Error("Render-based readback not available (no WebGPU canvas context)");a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let s=a.createCommandEncoder();{let p=s.beginComputePass();p.setPipeline(Z),p.setBindGroup(0,V),p.dispatchWorkgroups(16,16,1),p.end()}N(s,null);let r=he.getCurrentTexture(),i=s.beginRenderPass({colorAttachments:[{view:r.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});i.setPipeline(be),i.setBindGroup(0,Oe),i.draw(3),i.end(),a.queue.submit([s.finish()]),await a.queue.onSubmittedWorkDone(),Le.drawImage($a,0,0);let d=Le.getImageData(0,0,9,8).data,_=new Float32Array(65),c=new DataView(new ArrayBuffer(4));for(let p=0;p<65;p++){let u=p*4;c.setUint8(0,d[u]),c.setUint8(1,d[u+1]),c.setUint8(2,d[u+2]),c.setUint8(3,d[u+3]),_[p]=c.getFloat32(0)}return{handflag:new Float32Array([_[0]]),handedness:new Float32Array([_[1]]),landmarks:new Float32Array(_.subarray(2,65))}}let ti=a.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Fe=0,ii=[P,ti],ie=null,K=null;async function Ie(t){let s=ii[Fe];Fe=1-Fe,a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let r=a.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(Z),e.setBindGroup(0,V),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}N(r,s),a.queue.submit([r.finish()]);let i=null;if(ie!==null&&K!==null){await ie;let e=new Float32Array(K.getMappedRange());i={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},K.unmap()}return K=s,ie=s.mapAsync(GPUMapMode.READ),i}async function Ya(){if(!ie||!K)return null;await ie;let t=new Float32Array(K.getMappedRange()),s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))};return K.unmap(),ie=null,K=null,s}async function ni(t=50){let s=new Float32Array(196608);for(let e=0;e<5;e++)await ve(s);let r=[];for(let e=0;e<t;e++){let d=performance.now();await ve(s),r.push(performance.now()-d)}let i=r.reduce((e,d)=>e+d,0)/r.length;return{avgMs:i,fps:1e3/i}}async function ri(t=50){let s=new Float32Array(196608);for(let _=0;_<5;_++)await ve(s);let r=[];for(let _=0;_<t;_++){let c=a.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(ta),u.setBindGroup(0,Fa),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),u.end()}N(c,P);let p=performance.now();a.queue.submit([c.finish()]),await a.queue.onSubmittedWorkDone(),r.push(performance.now()-p)}r.sort((_,c)=>_-c);let i=r.reduce((_,c)=>_+c,0)/r.length,e=r[Math.floor(r.length/2)],d=r[0];return{avgMs:i,fps:1e3/i,medianMs:e,minMs:d}}function si(t){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let s=a.createCommandEncoder();{let r=s.beginComputePass();r.setPipeline(Z),r.setBindGroup(0,V),r.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),r.end()}N(s,P),a.queue.submit([s.finish()])}async function ui(t,s=50){function r(u){let n=[...u].sort((m,v)=>m-v);return{median:n[Math.floor(n.length/2)],min:n[0]}}for(let u=0;u<10;u++)await Re(t);let i=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let n=a.createCommandEncoder();{let v=n.beginComputePass();v.setPipeline(Z),v.setBindGroup(0,V),v.dispatchWorkgroups(16,16,1),v.end()}N(n,P);let m=performance.now();a.queue.submit([n.finish()]),await a.queue.onSubmittedWorkDone(),i.push(performance.now()-m)}let e=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let n=a.createCommandEncoder();{let G=n.beginComputePass();G.setPipeline(Z),G.setBindGroup(0,V),G.dispatchWorkgroups(16,16,1),G.end()}N(n,P),a.queue.submit([n.finish()]);let m=P.mapAsync(GPUMapMode.READ),v=performance.now();await a.queue.onSubmittedWorkDone(),await m,P.getMappedRange(),P.unmap(),e.push(performance.now()-v)}let d=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);let n=a.createCommandEncoder();{let v=n.beginComputePass();v.setPipeline(Z),v.setBindGroup(0,V),v.dispatchWorkgroups(16,16,1),v.end()}N(n,P),a.queue.submit([n.finish()]);let m=performance.now();await P.mapAsync(GPUMapMode.READ),P.getMappedRange(),P.unmap(),d.push(performance.now()-m)}let _=[];for(let u=0;u<s;u++){let n=performance.now();await Re(t),_.push(performance.now()-n)}await Ie(t);let c=[];for(let u=0;u<s;u++){let n=performance.now();await Ie(t),c.push(performance.now()-n)}await Ya();let p=null;if(be){let u=[];for(let n=0;n<s;n++){let m=performance.now();await Na(t),u.push(performance.now()-m)}p=r(u)}return{gpuOnly:r(i),mapAsyncOnly:r(e),mapAsyncNoWait:r(d),total:r(_),pipelined:r(c),renderReadback:p}}async function oi(t){let s=[];async function r(e,d,_){let c=a.createBuffer({size:d,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),p=a.createCommandEncoder();p.copyBufferToBuffer(e,0,c,0,d),a.queue.submit([p.finish()]),await a.queue.onSubmittedWorkDone(),await c.mapAsync(GPUMapMode.READ);let u=new Float32Array(c.getMappedRange()),n=1/0,m=-1/0,v=0;for(let G=0;G<u.length;G++)u[G]<n&&(n=u[G]),u[G]>m&&(m=u[G]),u[G]!==0&&v++;c.unmap(),c.destroy(),s.push({layer:_,stats:{min:n,max:m,nonZero:v,total:u.length}})}a.queue.copyExternalImageToTexture({source:t},{texture:F},[256,256]);{let e=a.createCommandEncoder(),d=e.beginComputePass();d.setPipeline(Z),d.setBindGroup(0,V),d.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),d.end(),a.queue.submit([e.finish()])}await r(oe,Math.min(oe.size,3*257*257*4),"canvas\u2192bufInput");{let e=a.createCommandEncoder(),d=e.beginComputePass();d.setPipeline(ia),d.setBindGroup(0,Ia),d.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),d.end(),a.queue.submit([e.finish()])}await r(U,Math.min(U.size,3072*128*4),"inputConv\u2192bufA");let i=!0;for(let e=0;e<Math.min(J.length,6);e++){let d=i?I[e]:q[e],_=i?z[e]:$[e],c=X[e],p=J[e];{let n=a.createCommandEncoder(),m=n.beginComputePass();m.setPipeline(c.dwPipeline),m.setBindGroup(0,d),m.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),m.end(),a.queue.submit([n.finish()])}await r(ee,Math.min(ee.size,p.spec.inCh*p.outH*p.outW*4),`layer${e}.DW\u2192bufDW (${p.spec.prefix})`);{let n=a.createCommandEncoder(),m=n.beginComputePass();m.setPipeline(c.pwPipeline),m.setBindGroup(0,_),m.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),m.end(),a.queue.submit([n.finish()])}let u=i?L:U;await r(u,Math.min(u.size,p.spec.outCh*p.outH*p.outW*4),`layer${e}.PW\u2192buf${i?"B":"A"} (${p.spec.prefix})`),i=!i}return s}return{device:a,run:ve,runFromCanvas:Re,runFromCanvasViaRender:Na,runFromCanvasPipelined:Ie,flushPipelined:Ya,benchmark:ni,benchmarkGPU:ri,benchmarkDiagnostic:ui,debugLayerOutputs:oi,_device:a,_benchmarkSubmitOnly:si}}var mi="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function fi(o={}){let{weightsUrl:w,scoreThreshold:f=.5,forceF32:b=!1}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let k=w??mi,a=k.endsWith("/")?k:`${k}/`,W=`${a}weights_f16.json`,O=`${a}weights_f16.bin`,[E,M]=await Promise.all([fetch(W),fetch(O)]);if(!E.ok)throw new Error(`Failed to fetch weights metadata: ${E.status}`);if(!M.ok)throw new Error(`Failed to fetch weights binary: ${M.status}`);let x=await E.json(),Y=await M.arrayBuffer(),g=Pt(x,Y),R=await kt(g,{forceF32:b}),D=null;function Pe(){return D||(D=new OffscreenCanvas(256,256)),D}async function y(h){if(h instanceof HTMLCanvasElement||h instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&h instanceof ImageBitmap)return h;let C=Pe();C.width=256,C.height=256;let S=C.getContext("2d");return h instanceof ImageData?S.putImageData(h,0,0):S.drawImage(h,0,0,256,256),C}function re(h,C,S){let _e=h[0];if(_e<f)return null;let ke=C[0]>.5,ce=[];for(let j=0;j<21;j++)ce.push({x:S[j*3],y:S[j*3+1],z:S[j*3+2]});return{score:_e,handedness:ke?"right":"left",landmarks:ce}}async function se(h){let C=await y(h),S=await R.runFromCanvas(C);return re(S.handflag,S.handedness,S.landmarks)}async function de(h){let C=await y(h),S=await R.runFromCanvasPipelined(C);return S?re(S.handflag,S.handedness,S.landmarks):null}async function T(){let h=await R.flushPipelined();return h?re(h.handflag,h.handedness,h.landmarks):null}function l(){R.device.destroy(),D=null}async function B(h){let C=await y(h);return R.benchmarkDiagnostic(C)}async function H(h){let C=await y(h);return R.debugLayerOutputs(C)}return{detect:se,detectPipelined:de,flushPipelined:T,dispose:l,benchmarkDiagnostic:B,debugLayerOutputs:H}}export{fi as createHandpose};
