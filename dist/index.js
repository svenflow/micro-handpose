function A(o){return o.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}function et(o){let w=["weight","bias","pw_weight","pw_bias","dw_weight","dw_bias","handflag_w","handflag_b","handedness_w","handedness_b","landmarks_w","landmarks_b"],h="enable f16;"+o;for(let g of w)for(;h.includes(`${g}:array<f32>`);)h=h.replace(`${g}:array<f32>`,`${g}:array<f16>`);return h}var ze=A(`
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
`)}var wt=lt("sigmoid"),mt=lt("div256"),ht=A(`
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
`),ft=A(`
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
`);function bt(o,w){let g=Math.min(w,256),k=w>g,W=o%4===0?`var ic:u32=0u;
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
    }`,M=o===w?"":`if(c<${o}u){`,f=o===w?"":"}",V=k?`for(var c:u32=lid.x;c<${o}u;c+=${g}u){`:`let c=lid.x;
  ${M}`,x=k?"}":f,H=k?`for(var c:u32=lid.x;c<${w}u;c+=${g}u){`:"{let c=lid.x;";return A(`
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
  ${x}
  // Step 2: barrier
  workgroupBarrier();
  // Step 3: PW 1x1 + skip + ReLU
  ${H}
    let pw_base=c*${o}u;
    var sum0:f32=0.0; var sum1:f32=0.0; var sum2:f32=0.0; var sum3:f32=0.0;
    ${W}
    // Skip connection (only for c < inCh)
    ${E}
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
`);function Pt(o,w){let h=new Map,g=o.dtype??"float32";for(let k=0;k<o.keys.length;k++){let a=o.keys[k],W=o.shapes[k],E=o.offsets[k],M=W.reduce((x,H)=>x*H,1),f,V;if(g==="float32")f=new Float32Array(w,E,M);else{let x=new DataView(w);f=new Float32Array(M);for(let H=0;H<M;H++)f[H]=oi(x.getUint16(E+H*2,!0));V=w.slice(E,E+M*2)}h.set(a,{data:f,shape:W,rawF16:V})}return h}function oi(o){let w=o>>15&1,h=o>>10&31,g=o&1023;if(h===0){if(g===0)return w?-0:0;let W=-14,E=g/1024;return(w?-1:1)*Math.pow(2,W)*E}if(h===31)return g===0?w?-1/0:1/0:NaN;let k=h-15,a=1+g/1024;return(w?-1:1)*Math.pow(2,k)*a}var pi=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],vt=pi.map(([o,w,h,g,k])=>({type:"resmodule",inCh:o,outCh:w,h,w:h,stride:g,prefix:k})),di=2,_i=5,ci=8,li=11;async function kt(o,w){if(!navigator.gpu)throw new Error("WebGPU not supported");let h=await navigator.gpu.requestAdapter();if(!h)throw new Error("No GPU adapter found");let g=h.features.has("shader-f16"),k=g?["shader-f16"]:[],a=await h.requestDevice({requiredFeatures:k,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(h.limits.maxStorageBuffersPerShaderStage,8),maxComputeWorkgroupSizeX:Math.min(h.limits.maxComputeWorkgroupSizeX,288),maxComputeInvocationsPerWorkgroup:Math.min(h.limits.maxComputeInvocationsPerWorkgroup,288)}}),W=!1;if(g)try{let s=a.createShaderModule({code:`enable f16;
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
}`});if((await s.getCompilationInfo()).messages.some(i=>i.type==="error"))W=!1;else{let i=new Uint16Array(4096);i.fill(15360);let e=new Uint16Array(64);e.fill(14336);let d=new Float32Array(64);d.fill(1);let _=a.createBuffer({size:i.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),c=a.createBuffer({size:e.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),p=a.createBuffer({size:d.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),u=a.createBuffer({size:256,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),n=a.createBuffer({size:256,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});a.queue.writeBuffer(_,0,i),a.queue.writeBuffer(c,0,e),a.queue.writeBuffer(p,0,d);let m=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),y=a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[m]}),compute:{module:s,entryPoint:"main"}}),D=a.createBindGroup({layout:m,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:p}},{binding:3,resource:{buffer:u}}]}),se=a.createCommandEncoder(),Q=se.beginComputePass();Q.setPipeline(y),Q.setBindGroup(0,D),Q.dispatchWorkgroups(1),Q.end(),se.copyBufferToBuffer(u,0,n,0,256),a.queue.submit([se.finish()]),await a.queue.onSubmittedWorkDone(),await n.mapAsync(GPUMapMode.READ);let pe=new Float32Array(n.getMappedRange());W=Math.abs(pe[0]-64.5)<.5&&Math.abs(pe[63]-64.5)<.5,n.unmap(),_.destroy(),c.destroy(),p.destroy(),u.destroy(),n.destroy(),W||console.warn("[micro-handpose] f16 storage validation FAILED (realistic test) \u2014 falling back to f32")}}catch{W=!1}let E=o.values().next().value,M=W&&!!E?.rawF16&&!w?.forceF32;console.log(M?"[micro-handpose] Using f16 weight storage (shader-f16 validated)":`[micro-handpose] Using f32 weights (shader-f16 feature: ${g}, f16 validated: ${W}, f16 data: ${!!E?.rawF16})`);function f(t){if(M&&t.rawF16){let s=new Uint16Array(t.rawF16);if(s.length%2!==0){let r=new Uint16Array(s.length+1);return r.set(s),r}return s}return t.data}function V(t){if(M&&t.rawF16){let s=t.rawF16.byteLength;return Math.ceil(s/4)*4}return t.data.byteLength}function x(t){return M?et(t):t}let H={r:"read-only-storage",s:"storage",u:"uniform"};function B(t){return a.createBindGroupLayout({entries:t.map((s,r)=>({binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[s]}}))})}function ee(t){return a.createBindGroupLayout({entries:t.map((s,r)=>s==="t"?{binding:r,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[s]}})})}let v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,ae=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,K=GPUBufferUsage.STORAGE,de=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,L=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(t,s){return a.createBuffer({size:t,usage:s})}function U(t,s){return a.createBindGroup({layout:t,entries:s.map((r,i)=>({binding:i,resource:"size"in r?{buffer:r}:r}))})}function T(t,s){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),compute:{module:s,entryPoint:"main"}})}let Pe=a.createShaderModule({code:st}),b=a.createShaderModule({code:xt}),C=a.createShaderModule({code:x(ht)}),S=a.createShaderModule({code:x(qe)}),_e=a.createShaderModule({code:x(ze)}),ke=a.createShaderModule({code:x($e)}),ce=a.createShaderModule({code:x(Ne)}),j=a.createShaderModule({code:x(ut)}),Bt=a.createShaderModule({code:pt}),Ut=a.createShaderModule({code:_t}),Gt=a.createShaderModule({code:dt}),St=a.createShaderModule({code:x(ct)}),Dt=a.createShaderModule({code:x(wt)}),Ct=a.createShaderModule({code:x(mt)}),At=a.createShaderModule({code:x(ft)}),Ye=new Map;function hi(t,s){let r=`${t}_${s}`,i=Ye.get(r);return i||(i=a.createShaderModule({code:x(bt(t,s))}),Ye.set(r,i)),i}let Be=B(["r","r","r","s","u"]),Ue=B(["r","r","r","r","s","u"]),Xe=B(["r","s","u"]),Ze=B(["r","r","r","s","u"]),Wt=B(["r","s","u"]),Mt=B(["r","r","s","u"]),le=B(["r","r","s","u"]),Ve=B(["r","r","r","s","u"]),ue=B(["r","r","r","s","u"]),Ke=ee(["t","s","u"]),je=B(["r","r","r","r","r","r","r","s"]),Ge=B(["r","r","r","r","r","s","u"]),Et=a.createPipelineLayout({bindGroupLayouts:[Be]}),Ht=a.createPipelineLayout({bindGroupLayouts:[Ue]}),we=t=>a.createComputePipeline({layout:Et,compute:{module:t,entryPoint:"main"}}),me=t=>a.createComputePipeline({layout:Ht,compute:{module:t,entryPoint:"main"}}),Tt=we(S),Lt=we(_e),Ot=me(ke),Rt=me(ce),Je=new Map,Qe=new Map,ea=new Map,aa=new Map;Je.set("8,8",Tt),Qe.set("8,8",Lt),ea.set("8,8",Ot),aa.set("8,8",Rt);function he(t,s,r,i,e){let d=`${s},${r}`,_=t.get(d);return _||(_=e(a.createShaderModule({code:x(i(s,r))})),t.set(d,_)),_}let Ft=(t,s)=>he(Je,t,s,at,we),It=(t,s)=>he(Qe,t,s,tt,we),zt=(t,s)=>he(ea,t,s,it,me),qt=(t,s)=>he(aa,t,s,nt,me),N=vt.map(t=>{let s=t.stride===2?t.h/2:t.h,r=t.stride===2?t.w/2:t.w,[i,e]=rt(t.inCh,s),d=t.h>=64,_=s>=16&&t.inCh>=288&&t.outCh>=288&&t.outCh%2===0;return{dwPipeline:d?It(i,e):Ft(i,e),pwPipeline:_?qt(i,e):zt(i,e),dwDispatchX:Math.ceil(r/i),dwDispatchY:Math.ceil(s/e),dwDispatchZ:t.inCh,pwDispatchX:Math.ceil(r/i),pwDispatchY:Math.ceil(s/e),pwDispatchZ:_?t.outCh/2:t.outCh}}),ta=T(Xe,Pe),ia=T(Ze,j);T(Wt,Bt),T(Mt,Ut);let Se=T(le,Gt),$t=T(Ve,St);T(ue,Dt),T(ue,Ct);let Y=T(Ke,b),Nt=T(je,C),Yt=T(Ge,At),De=1*288*128*128*4,na=l(3*256*256*4,v),oe=l(3*257*257*4,K),ra=l(12,L);a.queue.writeBuffer(ra,0,new Uint32Array([3,256,257]));let G=l(De,ae),O=l(De,de),te=l(De,K),sa=l(3072*64*4,v),ua=l(3072*32*4,v),oa=l(1536*16*4,v),pa=l(6144*64*4,K),ie=l(260,de),P=l(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);l(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);let R=a.createTexture({size:[256,256],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),da=l(8,L);a.queue.writeBuffer(da,0,new Uint32Array([256,257]));let _a=o.get("backbone1.1.weight"),ca=o.get("backbone1.1.bias");if(!_a||!ca)throw new Error("Missing input conv weights");let la=f(_a),wa=f(ca),ma=l(la.byteLength,v),ha=l(wa.byteLength,v),fa=l(28,L);a.queue.writeBuffer(ma,0,la),a.queue.writeBuffer(ha,0,wa),a.queue.writeBuffer(fa,0,new Uint32Array([1,3,24,257,257,128,128]));let ba=o.get("backbone6.1.weight"),ga=o.get("backbone6.1.bias");if(!ba||!ga)throw new Error("Missing backbone6.1 conv1x1 weights");let ya=f(ba),xa=f(ga),va=l(ya.byteLength,v),Pa=l(xa.byteLength,v),ka=l(20,L);a.queue.writeBuffer(va,0,ya),a.queue.writeBuffer(Pa,0,xa),a.queue.writeBuffer(ka,0,new Uint32Array([1,96,48,32,32]));let Ba=o.get("handflag.weight"),Ua=o.get("handflag.bias");if(!Ba||!Ua)throw new Error("Missing handflag weights");let Ga=f(Ba),Sa=f(Ua),Ce=l(Ga.byteLength,v),Ae=l(Sa.byteLength,v),Da=l(12,L);a.queue.writeBuffer(Ce,0,Ga),a.queue.writeBuffer(Ae,0,Sa),a.queue.writeBuffer(Da,0,new Uint32Array([1,288,1]));let Ca=o.get("handedness.weight"),Aa=o.get("handedness.bias");if(!Ca||!Aa)throw new Error("Missing handedness weights");let Wa=f(Ca),Ma=f(Aa),We=l(Wa.byteLength,v),Me=l(Ma.byteLength,v),Ea=l(12,L);a.queue.writeBuffer(We,0,Wa),a.queue.writeBuffer(Me,0,Ma),a.queue.writeBuffer(Ea,0,new Uint32Array([1,288,1]));let Ha=o.get("reg_3d.weight"),Ta=o.get("reg_3d.bias");if(!Ha||!Ta)throw new Error("Missing reg_3d weights");let La=f(Ha),Oa=f(Ta),Ee=l(La.byteLength,v),He=l(Oa.byteLength,v),Ra=l(12,L);a.queue.writeBuffer(Ee,0,La),a.queue.writeBuffer(He,0,Oa),a.queue.writeBuffer(Ra,0,new Uint32Array([1,288,63]));let J=vt.map(t=>{let{inCh:s,outCh:r,h:i,w:e,stride:d,prefix:_}=t,c=d===2?i/2:i,p=d===2?e/2:e,u=d===1?2:1,n=o.get(`${_}convs.0.weight`),m=o.get(`${_}convs.0.bias`),y=o.get(`${_}convs.1.weight`),D=o.get(`${_}convs.1.bias`);if(!n||!m||!y||!D)throw new Error(`Missing weights for ${_}`);let se=f(n),Q=f(m),pe=f(y),Xa=f(D),Za=l(se.byteLength,v),Va=l(Q.byteLength,v),Ka=l(pe.byteLength,v),ja=l(Xa.byteLength,v),Ja=l(32,L),Qa=l(36,L);return a.queue.writeBuffer(Za,0,se),a.queue.writeBuffer(Va,0,Q),a.queue.writeBuffer(Ka,0,pe),a.queue.writeBuffer(ja,0,Xa),a.queue.writeBuffer(Ja,0,new Uint32Array([1,s,i,e,c,p,d,u])),a.queue.writeBuffer(Qa,0,new Uint32Array([1,s,r,c,p,Math.max(0,r-s),d,i,e])),{dwWeight:Za,dwBias:Va,pwWeight:Ka,pwBias:ja,dwUniform:Ja,pwUniform:Qa,spec:t,outH:c,outW:p}});function ne(t){let s=l(t.length*4,L);return a.queue.writeBuffer(s,0,new Uint32Array(t)),s}let Xt=ne([1,96,8,8,16,16]),Zt=ne([1,96,16,16,32,32]),Vt=ne([1,48,32,32,64,64]);ne([1536*16]),ne([3072*32]),ne([3072*64]);let Fa=U(Xe,[na,oe,ra]),Ia=U(Ze,[oe,ma,ha,G,fa]),F=[],I=[],z=[],q=[];for(let t of J)F.push(U(Be,[G,t.dwWeight,t.dwBias,te,t.dwUniform])),I.push(U(Ue,[te,G,t.pwWeight,t.pwBias,O,t.pwUniform])),z.push(U(Be,[O,t.dwWeight,t.dwBias,te,t.dwUniform])),q.push(U(Ue,[te,O,t.pwWeight,t.pwBias,G,t.pwUniform]));let Kt=U(le,[G,oa,O,Xt]),jt=U(le,[G,ua,O,Zt]),Jt=U(Ve,[G,va,Pa,pa,ka]),Qt=U(le,[pa,sa,O,Vt]);U(ue,[G,Ce,Ae,ie,Da]),U(ue,[G,We,Me,ie,Ea]),U(ue,[G,Ee,He,ie,Ra]);let X=U(Ke,[R.createView(),oe,da]),ei=U(je,[G,Ce,Ae,We,Me,Ee,He,ie]),Te=24,za=[],qa=[];for(let t=Te;t<J.length;t++){let s=J[t];za.push(U(Ge,[G,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,O,s.dwUniform])),qa.push(U(Ge,[O,s.dwWeight,s.dwBias,s.pwWeight,s.pwBias,G,s.dwUniform]))}let Le=new OffscreenCanvas(9,8).getContext("2d",{willReadFrequently:!0});Le.globalCompositeOperation="copy";let $a=new OffscreenCanvas(9,8),fe=$a.getContext("webgpu"),be=null,Oe=null;if(fe){fe.configure({device:a,format:"rgba8unorm",alphaMode:"premultiplied"});let t=a.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]}),s=a.createShaderModule({code:gt}),r=a.createShaderModule({code:yt});be=a.createRenderPipeline({layout:a.createPipelineLayout({bindGroupLayouts:[t]}),vertex:{module:s,entryPoint:"vs"},fragment:{module:r,entryPoint:"fs",targets:[{format:"rgba8unorm"}]}}),Oe=a.createBindGroup({layout:t,entries:[{binding:0,resource:{buffer:ie}}]})}let ge=new Float32Array(1),ye=new Float32Array(1),xe=new Float32Array(63);function $(t,s){let r=!0,i=0,e=t.beginComputePass();for(e.setPipeline(ia),e.setBindGroup(0,Ia),e.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24);i<=di;i++){let p=r?F[i]:z[i],u=r?I[i]:q[i],n=N[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.end();let d=r?G:O;for(t.copyBufferToBuffer(d,0,sa,0,3072*64*4),e=t.beginComputePass();i<=_i;i++){let p=r?F[i]:z[i],u=r?I[i]:q[i],n=N[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.end();let _=r?G:O;for(t.copyBufferToBuffer(_,0,ua,0,3072*32*4),e=t.beginComputePass();i<=ci;i++){let p=r?F[i]:z[i],u=r?I[i]:q[i],n=N[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.end();let c=r?G:O;for(t.copyBufferToBuffer(c,0,oa,0,1536*16*4),e=t.beginComputePass();i<=li;i++){let p=r?F[i]:z[i],u=r?I[i]:q[i],n=N[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}e.setPipeline(Se),e.setBindGroup(0,Kt),e.dispatchWorkgroups(Math.ceil(16/8),Math.ceil(16/8),96),e.end(),r=!1,e=t.beginComputePass();{let p=r?F[i]:z[i],u=r?I[i]:q[i],n=N[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r,i++}e.setPipeline(Se),e.setBindGroup(0,jt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),96),e.end(),r=!1,e=t.beginComputePass();{let p=r?F[i]:z[i],u=r?I[i]:q[i],n=N[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r,i++}for(e.setPipeline($t),e.setBindGroup(0,Jt),e.dispatchWorkgroups(Math.ceil(32/8),Math.ceil(32/8),48),e.setPipeline(Se),e.setBindGroup(0,Qt),e.dispatchWorkgroups(Math.ceil(64/8),Math.ceil(64/8),48),e.end(),r=!1,e=t.beginComputePass();i<Te;i++){let p=r?F[i]:z[i],u=r?I[i]:q[i],n=N[i];e.setPipeline(n.dwPipeline),e.setBindGroup(0,p),e.dispatchWorkgroups(n.dwDispatchX,n.dwDispatchY,n.dwDispatchZ),e.setPipeline(n.pwPipeline),e.setBindGroup(0,u),e.dispatchWorkgroups(n.pwDispatchX,n.pwDispatchY,n.pwDispatchZ),r=!r}for(;i<J.length;i++){let p=i-Te,u=r?za[p]:qa[p],n=J[i];e.setPipeline(Yt),e.setBindGroup(0,u),e.dispatchWorkgroups(n.outW,n.outH,1),r=!r}e.setPipeline(Nt),e.setBindGroup(0,ei),e.dispatchWorkgroups(1),e.end(),s&&t.copyBufferToBuffer(ie,0,s,0,260)}async function ve(t){a.queue.writeBuffer(na,0,t);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(ta),e.setBindGroup(0,Fa),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),e.end()}$(s,P),a.queue.submit([s.finish()]);let r=P.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(P.getMappedRange());return ge[0]=i[0],ye[0]=i[1],xe.set(i.subarray(2,65)),P.unmap(),{handflag:new Float32Array(ge),handedness:new Float32Array(ye),landmarks:new Float32Array(xe)}}async function Re(t){a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);let s=a.createCommandEncoder();{let e=s.beginComputePass();e.setPipeline(Y),e.setBindGroup(0,X),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}$(s,P),a.queue.submit([s.finish()]);let r=P.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await r;let i=new Float32Array(P.getMappedRange());return ge[0]=i[0],ye[0]=i[1],xe.set(i.subarray(2,65)),P.unmap(),{handflag:new Float32Array(ge),handedness:new Float32Array(ye),landmarks:new Float32Array(xe)}}async function Na(t){if(!be||!Oe||!fe)throw new Error("Render-based readback not available (no WebGPU canvas context)");a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);let s=a.createCommandEncoder();{let p=s.beginComputePass();p.setPipeline(Y),p.setBindGroup(0,X),p.dispatchWorkgroups(16,16,1),p.end()}$(s,null);let r=fe.getCurrentTexture(),i=s.beginRenderPass({colorAttachments:[{view:r.createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});i.setPipeline(be),i.setBindGroup(0,Oe),i.draw(3),i.end(),a.queue.submit([s.finish()]),await a.queue.onSubmittedWorkDone(),Le.drawImage($a,0,0);let d=Le.getImageData(0,0,9,8).data,_=new Float32Array(65),c=new DataView(new ArrayBuffer(4));for(let p=0;p<65;p++){let u=p*4;c.setUint8(0,d[u]),c.setUint8(1,d[u+1]),c.setUint8(2,d[u+2]),c.setUint8(3,d[u+3]),_[p]=c.getFloat32(0)}return{handflag:new Float32Array([_[0]]),handedness:new Float32Array([_[1]]),landmarks:new Float32Array(_.subarray(2,65))}}let ai=a.createBuffer({size:260,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),Fe=0,ti=[P,ai],re=null,Z=null;async function Ie(t){let s=ti[Fe];Fe=1-Fe,a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);let r=a.createCommandEncoder();{let e=r.beginComputePass();e.setPipeline(Y),e.setBindGroup(0,X),e.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),e.end()}$(r,s),a.queue.submit([r.finish()]);let i=null;if(re!==null&&Z!==null){await re;let e=new Float32Array(Z.getMappedRange());i={handflag:new Float32Array([e[0]]),handedness:new Float32Array([e[1]]),landmarks:new Float32Array(e.subarray(2,65))},Z.unmap()}return Z=s,re=s.mapAsync(GPUMapMode.READ),i}async function Ya(){if(!re||!Z)return null;await re;let t=new Float32Array(Z.getMappedRange()),s={handflag:new Float32Array([t[0]]),handedness:new Float32Array([t[1]]),landmarks:new Float32Array(t.subarray(2,65))};return Z.unmap(),re=null,Z=null,s}async function ii(t=50){let s=new Float32Array(196608);for(let e=0;e<5;e++)await ve(s);let r=[];for(let e=0;e<t;e++){let d=performance.now();await ve(s),r.push(performance.now()-d)}let i=r.reduce((e,d)=>e+d,0)/r.length;return{avgMs:i,fps:1e3/i}}async function ni(t=50){let s=new Float32Array(196608);for(let _=0;_<5;_++)await ve(s);let r=[];for(let _=0;_<t;_++){let c=a.createCommandEncoder();{let u=c.beginComputePass();u.setPipeline(ta),u.setBindGroup(0,Fa),u.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),3),u.end()}$(c,P);let p=performance.now();a.queue.submit([c.finish()]),await a.queue.onSubmittedWorkDone(),r.push(performance.now()-p)}r.sort((_,c)=>_-c);let i=r.reduce((_,c)=>_+c,0)/r.length,e=r[Math.floor(r.length/2)],d=r[0];return{avgMs:i,fps:1e3/i,medianMs:e,minMs:d}}function ri(t){a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);let s=a.createCommandEncoder();{let r=s.beginComputePass();r.setPipeline(Y),r.setBindGroup(0,X),r.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),r.end()}$(s,P),a.queue.submit([s.finish()])}async function si(t,s=50){function r(u){let n=[...u].sort((m,y)=>m-y);return{median:n[Math.floor(n.length/2)],min:n[0]}}for(let u=0;u<10;u++)await Re(t);let i=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);let n=a.createCommandEncoder();{let y=n.beginComputePass();y.setPipeline(Y),y.setBindGroup(0,X),y.dispatchWorkgroups(16,16,1),y.end()}$(n,P);let m=performance.now();a.queue.submit([n.finish()]),await a.queue.onSubmittedWorkDone(),i.push(performance.now()-m)}let e=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);let n=a.createCommandEncoder();{let D=n.beginComputePass();D.setPipeline(Y),D.setBindGroup(0,X),D.dispatchWorkgroups(16,16,1),D.end()}$(n,P),a.queue.submit([n.finish()]);let m=P.mapAsync(GPUMapMode.READ),y=performance.now();await a.queue.onSubmittedWorkDone(),await m,P.getMappedRange(),P.unmap(),e.push(performance.now()-y)}let d=[];for(let u=0;u<s;u++){a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);let n=a.createCommandEncoder();{let y=n.beginComputePass();y.setPipeline(Y),y.setBindGroup(0,X),y.dispatchWorkgroups(16,16,1),y.end()}$(n,P),a.queue.submit([n.finish()]);let m=performance.now();await P.mapAsync(GPUMapMode.READ),P.getMappedRange(),P.unmap(),d.push(performance.now()-m)}let _=[];for(let u=0;u<s;u++){let n=performance.now();await Re(t),_.push(performance.now()-n)}await Ie(t);let c=[];for(let u=0;u<s;u++){let n=performance.now();await Ie(t),c.push(performance.now()-n)}await Ya();let p=null;if(be){let u=[];for(let n=0;n<s;n++){let m=performance.now();await Na(t),u.push(performance.now()-m)}p=r(u)}return{gpuOnly:r(i),mapAsyncOnly:r(e),mapAsyncNoWait:r(d),total:r(_),pipelined:r(c),renderReadback:p}}async function ui(t){let s=[];async function r(e,d,_){let c=a.createBuffer({size:d,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),p=a.createCommandEncoder();p.copyBufferToBuffer(e,0,c,0,d),a.queue.submit([p.finish()]),await a.queue.onSubmittedWorkDone(),await c.mapAsync(GPUMapMode.READ);let u=new Float32Array(c.getMappedRange()),n=1/0,m=-1/0,y=0;for(let D=0;D<u.length;D++)u[D]<n&&(n=u[D]),u[D]>m&&(m=u[D]),u[D]!==0&&y++;c.unmap(),c.destroy(),s.push({layer:_,stats:{min:n,max:m,nonZero:y,total:u.length}})}a.queue.copyExternalImageToTexture({source:t},{texture:R},[256,256]);{let e=a.createCommandEncoder(),d=e.beginComputePass();d.setPipeline(Y),d.setBindGroup(0,X),d.dispatchWorkgroups(Math.ceil(256/16),Math.ceil(256/16),1),d.end(),a.queue.submit([e.finish()])}await r(oe,Math.min(oe.size,3*257*257*4),"canvas\u2192bufInput");{let e=a.createCommandEncoder(),d=e.beginComputePass();d.setPipeline(ia),d.setBindGroup(0,Ia),d.dispatchWorkgroups(Math.ceil(128/8),Math.ceil(128/8),24),d.end(),a.queue.submit([e.finish()])}await r(G,Math.min(G.size,3072*128*4),"inputConv\u2192bufA");let i=!0;for(let e=0;e<Math.min(J.length,6);e++){let d=i?F[e]:z[e],_=i?I[e]:q[e],c=N[e],p=J[e];{let n=a.createCommandEncoder(),m=n.beginComputePass();m.setPipeline(c.dwPipeline),m.setBindGroup(0,d),m.dispatchWorkgroups(c.dwDispatchX,c.dwDispatchY,c.dwDispatchZ),m.end(),a.queue.submit([n.finish()])}await r(te,Math.min(te.size,p.spec.inCh*p.outH*p.outW*4),`layer${e}.DW\u2192bufDW (${p.spec.prefix})`);{let n=a.createCommandEncoder(),m=n.beginComputePass();m.setPipeline(c.pwPipeline),m.setBindGroup(0,_),m.dispatchWorkgroups(c.pwDispatchX,c.pwDispatchY,c.pwDispatchZ),m.end(),a.queue.submit([n.finish()])}let u=i?O:G;await r(u,Math.min(u.size,p.spec.outCh*p.outH*p.outW*4),`layer${e}.PW\u2192buf${i?"B":"A"} (${p.spec.prefix})`),i=!i}return s}return{device:a,run:ve,runFromCanvas:Re,runFromCanvasViaRender:Na,runFromCanvasPipelined:Ie,flushPipelined:Ya,benchmark:ii,benchmarkGPU:ni,benchmarkDiagnostic:si,debugLayerOutputs:ui,_device:a,_benchmarkSubmitOnly:ri}}var wi="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function mi(o={}){let{weightsUrl:w,scoreThreshold:h=.5,forceF32:g=!1}=o;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let k=w??wi,a=k.endsWith("/")?k:`${k}/`,W=`${a}weights_f16.json`,E=`${a}weights_f16.bin`,[M,f]=await Promise.all([fetch(W),fetch(E)]);if(!M.ok)throw new Error(`Failed to fetch weights metadata: ${M.status}`);if(!f.ok)throw new Error(`Failed to fetch weights binary: ${f.status}`);let V=await M.json(),x=await f.arrayBuffer(),H=Pt(V,x),B=await kt(H,{forceF32:g}),ee=null;function v(){return ee||(ee=new OffscreenCanvas(256,256)),ee}async function ae(b){if(b instanceof HTMLCanvasElement||b instanceof OffscreenCanvas||typeof ImageBitmap<"u"&&b instanceof ImageBitmap)return b;let C=v();C.width=256,C.height=256;let S=C.getContext("2d");return b instanceof ImageData?S.putImageData(b,0,0):S.drawImage(b,0,0,256,256),C}function K(b,C,S){let _e=b[0];if(_e<h)return null;let ke=C[0]>.5,ce=[];for(let j=0;j<21;j++)ce.push({x:S[j*3],y:S[j*3+1],z:S[j*3+2]});return{score:_e,handedness:ke?"right":"left",landmarks:ce}}async function de(b){let C=await ae(b),S=await B.runFromCanvas(C);return K(S.handflag,S.handedness,S.landmarks)}async function L(b){let C=await ae(b),S=await B.runFromCanvasPipelined(C);return S?K(S.handflag,S.handedness,S.landmarks):null}async function l(){let b=await B.flushPipelined();return b?K(b.handflag,b.handedness,b.landmarks):null}function U(){B.device.destroy(),ee=null}async function T(b){let C=await ae(b);return B.benchmarkDiagnostic(C)}async function Pe(b){let C=await ae(b);return B.debugLayerOutputs(C)}return{detect:de,detectPipelined:L,flushPipelined:l,dispose:U,benchmarkDiagnostic:T,debugLayerOutputs:Pe}}export{mi as createHandpose};
