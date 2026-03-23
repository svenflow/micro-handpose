function un(u,B){let n=new Map,T=u.dtype??"float32";for(let v=0;v<u.keys.length;v++){let a=u.keys[v],b=u.shapes[v],S=u.offsets[v],U=b.reduce((_,x)=>_*x,1),w,f;if(T==="float32")w=new Float32Array(B,S,U);else{let _=new DataView(B);w=new Float32Array(U);for(let x=0;x<U;x++)w[x]=vn(_.getUint16(S+x*2,!0));f=B.slice(S,S+U*2)}n.set(a,{data:w,shape:b,rawF16:f})}return n}function vn(u){let B=u>>15&1,n=u>>10&31,T=u&1023;if(n===0){if(T===0)return B?-0:0;let b=-14,S=T/1024;return(B?-1:1)*Math.pow(2,b)*S}if(n===31)return T===0?B?-1/0:1/0:NaN;let v=n-15,a=1+T/1024;return(B?-1:1)*Math.pow(2,v)*a}var Un=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],jn=Un.map(([u,B,n,T,v])=>({type:"resmodule",inCh:u,outCh:B,h:n,w:n,stride:T,prefix:v}));function cn(u,B){let n=new Map,T=u.dtype??"float32",v=new Map;for(let a=0;a<u.keys.length;a++){let b=u.keys[a],S=u.shapes[a],U=u.offsets[a],w=S.reduce((H,c)=>H*c,1),f,_;if(T==="float32")f=new Float32Array(B,U,w);else{let H=new DataView(B);f=new Float32Array(w);for(let c=0;c<w;c++)f[c]=Gn(H.getUint16(U+c*2,!0));_=B.slice(U,U+w*2)}let x=v.get(b)??0;v.set(b,x+1);let q=x===0?b:`${b}__${x}`;n.set(q,{data:f,shape:S,rawF16:_})}return n}function Gn(u){let B=u>>15&1,n=u>>10&31,T=u&1023;return n===0?T===0?B?-0:0:(B?-1:1)*Math.pow(2,-14)*(T/1024):n===31?T===0?B?-1/0:1/0:NaN:(B?-1:1)*Math.pow(2,n-15)*(1+T/1024)}function nt(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Mn=nt(`
struct CanvasParams { in_size:u32, }
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CanvasParams;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let x=gid.x; let y=gid.y;
  if(x>=params.in_size||y>=params.in_size){return;}
  let pixel=textureLoad(input_tex,vec2<u32>(x,y),0);
  let stride=params.in_size*params.in_size;
  output[0u*stride+y*params.in_size+x]=pixel.r;
  output[1u*stride+y*params.in_size+x]=pixel.g;
  output[2u*stride+y*params.in_size+x]=pixel.b;
}
`),An=nt(`
struct Params { in_channels:u32, out_channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    for(var ky:u32=0u;ky<3u;ky++){
      for(var kx:u32=0u;kx<3u;kx++){
        let iy=i32(out_y*2u+ky); let ix=i32(out_x*2u+kx);
        if(iy>=0 && iy<in_h && ix>=0 && ix<in_w){
          let in_idx=ic*params.in_h*params.in_w+u32(iy)*params.in_w+u32(ix);
          let w_idx=oc*params.in_channels*9u+ic*9u+ky*3u+kx;
          sum+=input[in_idx]*weight[w_idx];
        }
      }
    }
  }
  sum+=bias[oc];
  sum=min(max(sum,0.0),6.0);
  let out_idx=oc*params.out_h*params.out_w+out_y*params.out_w+out_x;
  output[out_idx]=sum;
}
`),kn=nt(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  sum=min(max(sum,0.0),6.0);
  output[oc*spatial+pix]=sum;
}
`),Sn=nt(`
struct Params { channels:u32, in_h:u32, in_w:u32, out_h:u32, out_w:u32, stride:u32, pad:u32, kernel:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let c=gid.z;
  if(out_x>=params.out_w||out_y>=params.out_h||c>=params.channels){return;}
  var sum:f32=0.0;
  let in_h=i32(params.in_h); let in_w=i32(params.in_w);
  let kk=params.kernel*params.kernel;
  for(var ky:u32=0u;ky<params.kernel;ky++){
    for(var kx:u32=0u;kx<params.kernel;kx++){
      let iy=i32(out_y*params.stride+ky)-i32(params.pad);
      let ix=i32(out_x*params.stride+kx)-i32(params.pad);
      if(iy>=0 && iy<in_h && ix>=0 && ix<in_w){
        sum+=input[c*params.in_h*params.in_w+u32(iy)*params.in_w+u32(ix)]*weight[c*kk+ky*params.kernel+kx];
      }
    }
  }
  sum+=bias[c];
  sum=min(max(sum,0.0),6.0);
  output[c*params.out_h*params.out_w+out_y*params.out_w+out_x]=sum;
}
`),Tn=nt(`
struct Params { in_channels:u32, out_channels:u32, height:u32, width:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let out_x=gid.x; let out_y=gid.y; let oc=gid.z;
  if(out_x>=params.width||out_y>=params.height||oc>=params.out_channels){return;}
  var sum:f32=0.0;
  let spatial=params.height*params.width;
  let pix=out_y*params.width+out_x;
  for(var ic:u32=0u;ic<params.in_channels;ic++){
    sum+=input[ic*spatial+pix]*weight[oc*params.in_channels+ic];
  }
  sum+=bias[oc];
  output[oc*spatial+pix]=sum;
}
`),En=nt(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Dn=nt(`
struct Params { channels:u32, spatial:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let c=gid.x;
  if(c>=params.channels){return;}
  var sum:f32=0.0;
  let base=c*params.spatial;
  for(var i:u32=0u;i<params.spatial;i++){
    sum+=input[base+i];
  }
  output[c]=sum/f32(params.spatial);
}
`),Rn=nt(`
struct Params { in_features:u32, out_features:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  if(oc>=params.out_features){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_features;ic++){
    sum+=input[ic]*weight[oc*params.in_features+ic];
  }
  output[oc]=sum+bias[oc];
}
`),Ln=nt(`
struct Params { in_features:u32, out_features:u32, }
@group(0)@binding(0) var<storage,read> input:array<f32>;
@group(0)@binding(1) var<storage,read> weight:array<f32>;
@group(0)@binding(2) var<storage,read> bias:array<f32>;
@group(0)@binding(3) var<storage,read_write> output:array<f32>;
@group(0)@binding(4) var<uniform> params:Params;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let oc=gid.x;
  if(oc>=params.out_features){return;}
  var sum:f32=0.0;
  for(var ic:u32=0u;ic<params.in_features;ic++){
    sum+=input[ic]*weight[oc*params.in_features+ic];
  }
  sum+=bias[oc];
  output[oc]=1.0/(1.0+exp(-sum));
}
`),xt=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],Wn=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function dn(u,B){if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");let T=n.features.has("shader-f16"),v=T?["shader-f16"]:[],a=await n.requestDevice({requiredFeatures:v,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}}),b=u.values().next().value,S=T&&!!b?.rawF16&&!B?.forceF32;function U(m){if(S&&m.rawF16){let o=new Uint16Array(m.rawF16);if(o.length%2!==0){let p=new Uint16Array(o.length+1);return p.set(o),p}return o}return m.data}function w(m){return S&&m.rawF16?Math.ceil(m.rawF16.byteLength/4)*4:m.data.byteLength}let f=S?2:4;function _(m){if(!S)return m;let o=m;return o=o.replace(/array<f32>/g,"array<f16>"),o=o.replace(/array<f32,/g,"array<f16,"),o=o.replace(/var sum:f32=0\.0/g,"var sum:f16=0.0h"),o=o.replace(/var sum0:f32=0\.0/g,"var sum0:f16=0.0h"),o=o.replace(/var sum1:f32=0\.0/g,"var sum1:f16=0.0h"),o=o.replace(/var sum2:f32=0\.0/g,"var sum2:f16=0.0h"),o=o.replace(/var sum3:f32=0\.0/g,"var sum3:f16=0.0h"),o=o.replace(/\/f32\(params/g,"/f16(params"),o=o.replace(/,0\.0\),6\.0\)/g,",0.0h),6.0h)"),o=o.replace(/->f32\{/g,"->f16{"),o=o.replace(/->f32 \{/g,"->f16 {"),o=o.replace(/return 0\.0;/g,"return 0.0h;"),"enable f16;"+o}function x(m){if(!S)return m;let o=_(m);return o=o.replace("read>input:array<f16>","read>input:array<f32>"),o=o.replace(/input\[in_idx\]/g,"f16(input[in_idx])"),o}function q(m){if(!S)return m;let o=m;return o=o.replace("read>input:array<f32>","read>input:array<f16>"),o=o.replace("read>weight:array<f32>","read>weight:array<f16>"),o=o.replace("read>bias:array<f32>","read>bias:array<f16>"),o=o.replace(/input\[ic\]/g,"f32(input[ic])"),o=o.replace(/weight\[oc\*params\.in_features\+ic\]/g,"f32(weight[oc*params.in_features+ic])"),o=o.replace(/bias\[oc\]/g,"f32(bias[oc])"),"enable f16;"+o}let H={r:"read-only-storage",s:"storage",u:"uniform"};function c(m){return a.createBindGroupLayout({entries:m.map((o,p)=>o==="t"?{binding:p,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:H[o]}})})}let C=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,K=GPUBufferUsage.STORAGE,Ce=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,Ge=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,ie=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function M(m,o){return a.createBuffer({size:Math.max(m,4),usage:o})}function de(m,o){return a.createBindGroup({layout:m,entries:o.map((p,L)=>({binding:L,resource:"size"in p?{buffer:p}:p}))})}function ye(m,o){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[m]}),compute:{module:o,entryPoint:"main"}})}function X(m){let o=u.get(m);if(!o)throw new Error(`Missing weight: ${m}`);return o}let Te=a.createShaderModule({code:Mn}),Ct=a.createShaderModule({code:x(An)}),rt=a.createShaderModule({code:_(kn)}),vt=a.createShaderModule({code:_(Sn)}),Ut=a.createShaderModule({code:_(Tn)}),Gt=a.createShaderModule({code:_(En)}),qe=a.createShaderModule({code:_(Dn)}),we=a.createShaderModule({code:q(Rn)}),lt=a.createShaderModule({code:q(Ln)}),ve=c(["r","r","r","s","u"]),Mt=c(["r","r","s","u"]),Be=c(["r","s","u"]),ze=c(["r","r","r","s","u"]),At=c(["t","s","u"]),Wt=ye(At,Te),y=ye(ve,Ct),R=ye(ve,rt),G=ye(ve,vt),F=ye(ve,Ut),se=ye(Mt,Gt),Me=ye(Be,qe),be=ye(ze,we),_e=ye(ze,lt),ue=1152*112*112*4,Z=M(ue,Ge),me=M(ue,Ge),$=M(ue,K),ee=M(ue,K),oe=M(ue,C),z=M(672*224*4,Ge),he=M(1152*4,Ce),pe=M(252,Ce),Ue=M(252,Ce),Pe=M(4,Ce),Ee=M(4,Ce),fe=M(260,Ge),te=M(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),De=a.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),We=M(4,ie);a.queue.writeBuffer(We,0,new Uint32Array([224]));let Re=X("conv2d"),He=X("batch_normalization"),je=U(Re),ot=U(He),mt=M(w(Re),C),it=M(w(He),C),st=M(24,ie);a.queue.writeBuffer(mt,0,je),a.queue.writeBuffer(it,0,ot),a.queue.writeBuffer(st,0,new Uint32Array([3,24,224,224,112,112]));let Ke=112,ke=112,Fe=[];for(let m=0;m<xt.length;m++){let o=xt[m],p=Wn[m],L=Ke,O=ke,Q=o.stride===2?Math.floor(Ke/2):Ke,N=o.stride===2?Math.floor(ke/2):ke,A={spec:o,inH:L,inW:O,outH:Q,outW:N,dwW:M(4,C),dwB:M(4,C),dwU:M(32,ie)};if(p.expandConvKey){let j=X(p.expandConvKey),ce=X(p.expandBNKey);A.expandW=M(w(j),C),A.expandB=M(w(ce),C),A.expandU=M(16,ie),a.queue.writeBuffer(A.expandW,0,U(j)),a.queue.writeBuffer(A.expandB,0,U(ce)),a.queue.writeBuffer(A.expandU,0,new Uint32Array([o.inCh,o.expandCh,L,O]))}let tt=X(p.dwWeightKey),Oe=X(p.dwBNKey);A.dwW=M(w(tt),C),A.dwB=M(w(Oe),C),a.queue.writeBuffer(A.dwW,0,U(tt)),a.queue.writeBuffer(A.dwB,0,U(Oe));let ne=Math.floor((o.dwKernel-o.stride)/2);if(a.queue.writeBuffer(A.dwU,0,new Uint32Array([o.expandCh,L,O,Q,N,o.stride,ne,o.dwKernel])),o.hasProject&&p.projectConvKey){let j=X(p.projectConvKey),ce=X(p.projectBNKey);A.projectW=M(w(j),C),A.projectB=M(w(ce),C),A.projectU=M(16,ie),a.queue.writeBuffer(A.projectW,0,U(j)),a.queue.writeBuffer(A.projectB,0,U(ce)),a.queue.writeBuffer(A.projectU,0,new Uint32Array([o.expandCh,o.outCh,Q,N]))}Fe.push(A),Ke=Q,ke=N}function kt(m,o){let p=u.get(m);if(!p)throw new Error(`Missing weight: ${m}`);if(p.shape.length!==o)throw new Error(`Weight ${m} has rank ${p.shape.length}, expected ${o}`);return p}let St=X("conv_landmarks__1"),Ht=X("conv_world_landmarks__1"),Tt=X("conv_handflag__1"),Le=X("conv_handedness__1"),Ye=X("Identity"),Kt=X("Identity_1"),Et=X("Identity_2"),ht=X("Identity_3"),Ot=M(w(St),C),$t=M(w(Ye),C),gt=M(w(Ht),C),bt=M(w(ht),C),Ve=M(w(Tt),C),Xe=M(w(Kt),C),$e=M(w(Le),C),Ze=M(w(Et),C);a.queue.writeBuffer(Ot,0,U(St)),a.queue.writeBuffer($t,0,U(Ye)),a.queue.writeBuffer(gt,0,U(Ht)),a.queue.writeBuffer(bt,0,U(ht)),a.queue.writeBuffer(Ve,0,U(Tt)),a.queue.writeBuffer(Xe,0,U(Kt)),a.queue.writeBuffer($e,0,U(Le)),a.queue.writeBuffer(Ze,0,U(Et));let _t=M(8,ie),Dt=M(8,ie),yt=M(8,ie),wt=M(8,ie);a.queue.writeBuffer(_t,0,new Uint32Array([1152,63])),a.queue.writeBuffer(Dt,0,new Uint32Array([1152,63])),a.queue.writeBuffer(yt,0,new Uint32Array([1152,1])),a.queue.writeBuffer(wt,0,new Uint32Array([1152,1]));let Pt=M(8,ie);a.queue.writeBuffer(Pt,0,new Uint32Array([1152,Ke*ke]));let E=new Map;for(let m=0;m<xt.length;m++)if(xt[m].hasResidual){let o=Fe[m],p=M(4,ie);a.queue.writeBuffer(p,0,new Uint32Array([xt[m].outCh*o.outH*o.outW])),E.set(m,p)}let I=de(At,[De.createView(),Z,We]),It=de(ve,[Z,mt,it,me,st]),Qe=new Float32Array(1),Je=new Float32Array(1),et=new Float32Array(63);function Bt(m,o){let p=m.beginComputePass();p.setPipeline(y),p.setBindGroup(0,It),p.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),p.end();let L=me,O=Z;for(let Q=0;Q<xt.length;Q++){let N=xt[Q],A=Fe[Q];if(N.hasResidual){let ne=N.inCh*A.inH*A.inW*f;m.copyBufferToBuffer(L,0,oe,0,ne)}if(p=m.beginComputePass(),A.expandW){let ne=de(ve,[L,A.expandW,A.expandB,$,A.expandU]);p.setPipeline(R),p.setBindGroup(0,ne),p.dispatchWorkgroups(Math.ceil(A.inW/8),Math.ceil(A.inH/8),N.expandCh)}let tt=A.expandW?$:L,Oe=de(ve,[tt,A.dwW,A.dwB,ee,A.dwU]);if(p.setPipeline(G),p.setBindGroup(0,Oe),p.dispatchWorkgroups(Math.ceil(A.outW/8),Math.ceil(A.outH/8),N.expandCh),N.hasProject&&A.projectW){let ne=(N.hasResidual,O),j=de(ve,[ee,A.projectW,A.projectB,ne,A.projectU]);if(p.setPipeline(F),p.setBindGroup(0,j),p.dispatchWorkgroups(Math.ceil(A.outW/8),Math.ceil(A.outH/8),N.outCh),N.hasResidual){let ce=E.get(Q),Ie=de(Mt,[O,oe,L,ce]);p.setPipeline(se),p.setBindGroup(0,Ie),p.dispatchWorkgroups(Math.ceil(N.outCh*A.outH*A.outW/256))}else{let ce=L;L=O,O=ce}}if(p.end(),!N.hasProject){p=m.beginComputePass();let ne=de(Be,[ee,he,Pt]);p.setPipeline(Me),p.setBindGroup(0,ne),p.dispatchWorkgroups(Math.ceil(1152/256));let j=de(ze,[he,Ot,$t,pe,_t]);p.setPipeline(be),p.setBindGroup(0,j),p.dispatchWorkgroups(1);let ce=de(ze,[he,Ve,Xe,Pe,yt]);p.setPipeline(_e),p.setBindGroup(0,ce),p.dispatchWorkgroups(1);let Ie=de(ze,[he,$e,Ze,Ee,wt]);p.setPipeline(_e),p.setBindGroup(0,Ie),p.dispatchWorkgroups(1),p.end(),m.copyBufferToBuffer(Pe,0,fe,0,4),m.copyBufferToBuffer(Ee,0,fe,4,4),m.copyBufferToBuffer(pe,0,fe,8,252),o&&m.copyBufferToBuffer(fe,0,o,0,260);return}}}async function Nt(m){a.queue.writeBuffer(z,0,m);let o=a.createCommandEncoder();o.copyBufferToBuffer(z,0,Z,0,672*224*4),Bt(o,te),a.queue.submit([o.finish()]);let p=te.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await p;let L=new Float32Array(te.getMappedRange());Qe[0]=L[0],Je[0]=L[1];for(let O=0;O<63;O++)et[O]=L[2+O]/224;return te.unmap(),{handflag:new Float32Array(Qe),handedness:new Float32Array(Je),landmarks:new Float32Array(et)}}async function Rt(m){a.queue.copyExternalImageToTexture({source:m},{texture:De},[224,224]);let o=a.createCommandEncoder();{let O=o.beginComputePass();O.setPipeline(Wt),O.setBindGroup(0,I),O.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),O.end()}Bt(o,te),a.queue.submit([o.finish()]);let p=te.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await p;let L=new Float32Array(te.getMappedRange());Qe[0]=L[0],Je[0]=L[1];for(let O=0;O<63;O++)et[O]=L[2+O]/224;return te.unmap(),{handflag:new Float32Array(Qe),handedness:new Float32Array(Je),landmarks:new Float32Array(et)}}async function Zt(m){let o=a.createCommandEncoder();o.copyBufferToBuffer(m,0,Z,0,672*224*4),Bt(o,te),a.queue.submit([o.finish()]);let p=te.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await p;let L=new Float32Array(te.getMappedRange());Qe[0]=L[0],Je[0]=L[1];for(let O=0;O<63;O++)et[O]=L[2+O]/224;return te.unmap(),{handflag:new Float32Array(Qe),handedness:new Float32Array(Je),landmarks:new Float32Array(et)}}async function Qt(){return null}async function Jt(){return null}async function zt(m=100){let o=new OffscreenCanvas(224,224),p=o.getContext("2d");p.fillStyle="#886644",p.fillRect(0,0,224,224);for(let N=0;N<5;N++)await Rt(o);let L=performance.now();for(let N=0;N<m;N++)await Rt(o);let Q=(performance.now()-L)/m;return{avgMs:Q,fps:1e3/Q}}async function qt(m=100){let o=await zt(m);return{...o,medianMs:o.avgMs,minMs:o.avgMs}}async function jt(m){return Rt(m)}async function Yt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Vt(m){let o={};async function p(Q,N,A){let tt=N*4,Oe=a.createBuffer({size:tt,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),ne=a.createCommandEncoder();ne.copyBufferToBuffer(Q,0,Oe,0,tt),a.queue.submit([ne.finish()]),await a.queue.onSubmittedWorkDone(),await Oe.mapAsync(GPUMapMode.READ);let j=new Float32Array(Oe.getMappedRange()),ce=1/0,Ie=-1/0,Xt=0;for(let D=0;D<j.length;D++)j[D]<ce&&(ce=j[D]),j[D]>Ie&&(Ie=j[D]),j[D]!==0&&Xt++;let en=Array.from(j.slice(0,5));Oe.unmap(),Oe.destroy(),o[A]={min:ce,max:Ie,nonZero:Xt,total:N,sample:en}}let L=new Float32Array(672*224);for(let Q=0;Q<50176;Q++)L[Q]=.5,L[50176+Q]=.3,L[448*224+Q]=.7;a.queue.writeBuffer(z,0,L);let O=a.createCommandEncoder();return O.copyBufferToBuffer(z,0,Z,0,672*224*4),Bt(O,te),a.queue.submit([O.finish()]),await a.queue.onSubmittedWorkDone(),await p(Z,672*224,"inputBufA"),await p(me,2688*112,"afterInitConvBufB"),await p(he,1152,"gapOutput"),await p(pe,63,"landmarks"),await p(Pe,1,"handflag"),await p(fe,65,"unifiedOutput"),o}return{device:a,run:Nt,runFromCanvas:Rt,runFromGPUBuffer:Zt,runFromCanvasPipelined:Qt,flushPipelined:Jt,benchmark:zt,benchmarkGPU:qt,runFromCanvasViaRender:jt,benchmarkDiagnostic:Yt,debugLayerOutputs:Vt}}function at(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var pn=at(`
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
`),fn=at(`
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
`),ln=at(`
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
`),mn=at(`
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
`),hn=at(`
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
  // TFLite ResizeBilinear: half_pixel_centers=true, align_corners=false
  // src = (dst + 0.5) * scale - 0.5
  let src_y=(f32(out_y)+0.5)*scale_y-0.5; let src_x=(f32(out_x)+0.5)*scale_x-0.5;
  let y0=u32(max(0.0,floor(src_y))); let x0=u32(max(0.0,floor(src_x)));
  let y1=min(y0+1u,params.in_height-1u); let x1=min(x0+1u,params.in_width-1u);
  let ly=max(0.0,src_y-f32(y0)); let lx=max(0.0,src_x-f32(x0));
  let base=batch*params.channels*params.in_height*params.in_width+c*params.in_height*params.in_width;
  let v00=input[base+y0*params.in_width+x0]; let v01=input[base+y0*params.in_width+x1];
  let v10=input[base+y1*params.in_width+x0]; let v11=input[base+y1*params.in_width+x1];
  let val=v00*(1.0-ly)*(1.0-lx)+v01*(1.0-ly)*lx+v10*ly*(1.0-lx)+v11*ly*lx;
  let out_idx=batch*params.channels*params.out_height*params.out_width+c*params.out_height*params.out_width+out_y*params.out_width+out_x;
  output[out_idx]=val+skip[out_idx];
}
`),gn=at(`
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
`),bn=at(`
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
`),_n=at(`
struct LBParams {
  src_w:u32, src_h:u32, dst_size:u32, _pad:u32,
  scale_x:f32, scale_y:f32, offset_x:f32, offset_y:f32,
}
@group(0)@binding(0) var input_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:LBParams;
@group(0)@binding(3) var input_sampler:sampler;
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

  // Hardware bilinear sampling via textureSampleLevel
  // Matches MediaPipe's OpenGL GL_LINEAR + GL_CLAMP_TO_EDGE exactly
  // (uses same GPU texture filtering hardware)
  let u = (src_x + 0.5) / f32(params.src_w);
  let v = (src_y + 0.5) / f32(params.src_h);
  let pixel = textureSampleLevel(input_tex, input_sampler, vec2<f32>(u, v), 0.0);

  output[0u*out_stride+dy*params.dst_size+dx]=pixel.r;
  output[1u*out_stride+dy*params.dst_size+dx]=pixel.g;
  output[2u*out_stride+dy*params.dst_size+dx]=pixel.b;
}
`),yn=at(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function wn(u,B){let n;if(B)n=B;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");n=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let T={r:"read-only-storage",s:"storage",u:"uniform"};function v(e){return n.createBindGroupLayout({entries:e.map((t,s)=>t==="t"?{binding:s,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:t==="sm"?{binding:s,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:s,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[t]}})})}let a=n.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),b=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,S=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,U=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,w=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function f(e,t){return n.createBuffer({size:Math.max(e,4),usage:t})}function _(e,t,s){n.queue.writeBuffer(e,t,s)}function x(e){let t=f(e.data.byteLength,b);return _(t,0,e.data),t}let q=Array.from(u.keys());function H(e){let t=u.get(e);if(!t)throw new Error(`Weight not found: ${e}`);return t}function c(...e){let t=q.find(s=>e.every(P=>s.includes(P)));if(!t)throw new Error(`Weight not found for: ${e.join(", ")}`);return H(t)}function C(e){let[,t,s,P]=e.shape,k=new Float32Array(P*25);for(let g=0;g<P;g++)for(let Y=0;Y<t;Y++)for(let V=0;V<s;V++)k[g*25+Y*5+V]=e.data[Y*s*P+V*P+g];return k}function K(e){let[t,,,s]=e.shape,P=new Float32Array(t*s);for(let k=0;k<t;k++)for(let g=0;g<s;g++)P[k*s+g]=e.data[k*s+g];return P}let Ce=n.createShaderModule({code:pn}),Ge=n.createShaderModule({code:fn}),ie=n.createShaderModule({code:ln}),M=n.createShaderModule({code:mn}),de=n.createShaderModule({code:gn}),ye=n.createShaderModule({code:hn}),X=n.createShaderModule({code:bn}),Te=n.createShaderModule({code:_n}),Ct=n.createShaderModule({code:yn}),rt=v(["r","r","r","r","s","u"]),vt=v(["r","r","r","s","u"]),Ut=v(["r","r","r","r","r","s","u"]),Gt=v(["r","r","r","s","u"]),qe=v(["r","r","r","r","s","u"]),we=v(["r","r","s","u"]),lt=v(["t","s","u"]),ve=v(["t","s","u","sm"]),Mt=v(["s","u"]);function Be(e,t){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:t,entryPoint:"main"}})}let ze=Be(rt,Ce),At=Be(vt,Ge),Wt=Be(Ut,ie),y=Be(Gt,M),R=Be(qe,de),G=Be(we,ye),F=Be(lt,X),se=Be(ve,Te),Me=Be(Mt,Ct),be=c("conv2d/Conv2D"),_e=c("batch_normalization/","conv2d/Conv2D"),Ae=c("p_re_lu/"),ue=x(be),Z=x(_e),me=x(Ae),ee=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(e=>{let t=c(e.dwKey),s=c(e.pwKey),P=c(e.bnKey),k=c(e.preluKey),g=C(t),Y=f(g.byteLength,b);_(Y,0,g);let V=new Float32Array(e.inCh),J=f(V.byteLength,b);_(J,0,V);let h=K(s),ae=f(h.byteLength,b);_(ae,0,h);let re=x(P),r=x(k);return{dwWeightBuf:Y,dwBiasBuf:J,pwWeightBuf:ae,pwBiasBuf:re,alphaBuf:r,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),oe=K(c("conv2d_25/Conv2D")),z=f(oe.byteLength,b);_(z,0,oe);let he=x(c("batch_normalization_25/")),pe=x(c("p_re_lu_25/")),Ue={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_24/")),t=f(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=f(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=K(c("conv2d_26/")),t=f(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_26/")),alphaBuf:x(c("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},Pe={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_25/")),t=f(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=f(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=K(c("conv2d_27/Conv2D1")),t=f(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_27/")),alphaBuf:x(c("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},Ee=K(c("conv2d_28/Conv2D")),fe=f(Ee.byteLength,b);_(fe,0,Ee);let te=x(c("batch_normalization_28/")),De=x(c("p_re_lu_28/")),We={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_26/")),t=f(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=f(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=K(c("conv2d_29/")),t=f(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_29/")),alphaBuf:x(c("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},Re={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_27/")),t=f(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=f(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=K(c("conv2d_30/Conv2D1")),t=f(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_30/")),alphaBuf:x(c("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},He=K(c("classifier_palm_16_NO_PRUNING/Conv2D")),je=f(He.byteLength,b);_(je,0,He);let ot=x(c("classifier_palm_16_NO_PRUNING/BiasAdd")),mt=K(c("regressor_palm_16_NO_PRUNING/Conv2D")),it=f(mt.byteLength,b);_(it,0,mt);let st=x(c("regressor_palm_16_NO_PRUNING/BiasAdd")),Ke=K(c("classifier_palm_8_NO_PRUNING/Conv2D")),ke=f(Ke.byteLength,b);_(ke,0,Ke);let Fe=x(c("classifier_palm_8_NO_PRUNING/BiasAdd")),kt=K(c("regressor_palm_8_NO_PRUNING/Conv2D")),St=f(kt.byteLength,b);_(St,0,kt);let Ht=x(c("regressor_palm_8_NO_PRUNING/BiasAdd")),Tt=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Le=f(36864*3*4,b),Ye=f(Tt,S),Kt=f(Tt,S),Et=f(Tt,S),ht=f(576*256*4,S),Ot=new Map;function $t(e){let t=Ot.get(e);return t||(t=f(4,w),_(t,0,new Uint32Array([e])),Ot.set(e,t)),t}let gt=f(144*256*4,S|GPUBufferUsage.COPY_DST),bt=f(576*128*4,S|GPUBufferUsage.COPY_DST),Ve=f(864*4,U),Xe=f(15552*4,U),$e=f(576*2*4,U),Ze=f(576*36*4,U),_t=f(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Dt=f(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),yt=f(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),wt=f(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Pt=n.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function E(e,t){return Math.ceil(e/t)}function I(e){let t=f(e.byteLength,w);return _(t,0,e),t}let It=I(new Uint32Array([1,3,32,192,192,96,96])),Qe=ee.map(e=>{let t=e.stride===2?e.inH/2:e.inH,s=t,P=e.stride===2?1:2,k=e.inCh;return{dw:I(new Uint32Array([1,e.inCh,e.inH,e.inH,t,s,e.stride,P])),pw:I(new Uint32Array([1,e.inCh,e.outCh,t,s,k,e.stride,e.inH,e.inH])),outH:t,outW:s}}),Je=(()=>{let e=Ue;return{dw:I(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:I(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),et=(()=>{let e=Pe;return{dw:I(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:I(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),Bt=(()=>{let e=We;return{dw:I(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:I(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),Nt=(()=>{let e=Re;return{dw:I(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:I(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),Rt=I(new Uint32Array([1,256,6,6,12,12])),Zt=I(new Uint32Array([1,256,12,12,12,12])),Qt=I(new Uint32Array([1,256,12,12,24,24])),Jt=I(new Uint32Array([1,128,24,24,24,24])),zt=I(new Uint32Array([1,256,256,12,12])),qt=I(new Uint32Array([1,256,128,24,24])),jt=I(new Uint32Array([1,256,6,12,12])),Yt=I(new Uint32Array([1,256,108,12,12])),Vt=I(new Uint32Array([1,128,2,24,24])),m=I(new Uint32Array([1,128,36,24,24])),o=I(new Uint32Array([192,192,192])),p=n.createBindGroup({layout:lt,entries:[{binding:0,resource:Pt.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:o}}]}),L=null,O=0,Q=0,N=f(32,w);function A(e,t){return L&&O===e&&Q===t||(L&&L.destroy(),L=n.createTexture({size:[e,t,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),O=e,Q=t),L}let tt=n.createBindGroup({layout:rt,entries:[{binding:0,resource:{buffer:Le}},{binding:1,resource:{buffer:ue}},{binding:2,resource:{buffer:Z}},{binding:3,resource:{buffer:me}},{binding:4,resource:{buffer:Ye}},{binding:5,resource:{buffer:It}}]});function Oe(e,t,s){}function ne(e,t,s,P,k,g){let Y=g.outH,V=n.createBindGroup({layout:vt,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:t.dwWeightBuf}},{binding:2,resource:{buffer:t.dwBiasBuf}},{binding:3,resource:{buffer:Et}},{binding:4,resource:{buffer:g.dw}}]}),J=e.beginComputePass();J.setPipeline(At),J.setBindGroup(0,V),J.dispatchWorkgroups(E(Y,8),E(g.outH,8),t.inCh),J.end(),t.inCh*g.outH*Y;let h=n.createBindGroup({layout:Ut,entries:[{binding:0,resource:{buffer:Et}},{binding:1,resource:{buffer:k}},{binding:2,resource:{buffer:t.pwWeightBuf}},{binding:3,resource:{buffer:t.pwBiasBuf}},{binding:4,resource:{buffer:t.alphaBuf}},{binding:5,resource:{buffer:P}},{binding:6,resource:{buffer:g.pw}}]}),ae=e.beginComputePass();ae.setPipeline(Wt),ae.setBindGroup(0,h),ae.dispatchWorkgroups(E(Y,8),E(g.outH,8),t.outCh),ae.end(),t.outCh*g.outH*Y}function j(e,t,s,P,k,g,Y,V,J){let h=n.createBindGroup({layout:Gt,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:P}},{binding:3,resource:{buffer:k}},{binding:4,resource:{buffer:g}}]}),ae=e.beginComputePass();ae.setPipeline(y),ae.setBindGroup(0,h),ae.dispatchWorkgroups(E(J,8),E(V,8),Y),ae.end()}function ce(e,t,s,P,k,g,Y,V,J,h){let ae=n.createBindGroup({layout:qe,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:P}},{binding:3,resource:{buffer:k}},{binding:4,resource:{buffer:g}},{binding:5,resource:{buffer:Y}}]}),re=e.beginComputePass();re.setPipeline(R),re.setBindGroup(0,ae),re.dispatchWorkgroups(E(h,8),E(J,8),V),re.end()}async function Ie(e){36864*3;{let r=e.beginComputePass();r.setPipeline(ze),r.setBindGroup(0,tt),r.dispatchWorkgroups(E(96,8),E(96,8),32),r.end()}9216*32;let t=Ye,s=Kt;for(let r=0;r<ee.length;r++){let l=ee[r];ne(e,l,t,s,t,Qe[r]);let ge=t;t=s,s=ge,r===13&&e.copyBufferToBuffer(t,0,bt,0,576*128*4),r===18&&e.copyBufferToBuffer(t,0,gt,0,144*256*4)}{let r=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:ht}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Rt}}]}),l=e.beginComputePass();l.setPipeline(G),l.setBindGroup(0,r),l.dispatchWorkgroups(E(12,8),E(12,8),256),l.end()}{let r=t;t=s,s=r}144*256,ce(e,t,z,he,pe,s,zt,256,12,12);{let r=t;t=s,s=r}144*256;{let r=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:gt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Zt}}]}),l=e.beginComputePass();l.setPipeline(G),l.setBindGroup(0,r),l.dispatchWorkgroups(E(12,8),E(12,8),256),l.end()}{let r=t;t=s,s=r}144*256,ne(e,Ue,t,s,t,Je);{let r=t;t=s,s=r}ne(e,Pe,t,s,t,et);{let r=t;t=s,s=r}j(e,t,je,ot,Ve,jt,6,12,12),j(e,t,it,st,Xe,Yt,108,12,12);{let r=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:ht}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Qt}}]}),l=e.beginComputePass();l.setPipeline(G),l.setBindGroup(0,r),l.dispatchWorkgroups(E(24,8),E(24,8),256),l.end()}{let r=t;t=s,s=r}576*256,ce(e,t,fe,te,De,s,qt,128,24,24);{let r=t;t=s,s=r}576*128;{let r=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:bt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Jt}}]}),l=e.beginComputePass();l.setPipeline(G),l.setBindGroup(0,r),l.dispatchWorkgroups(E(24,8),E(24,8),128),l.end()}{let r=t;t=s,s=r}576*128,ne(e,We,t,s,t,Bt);{let r=t;t=s,s=r}ne(e,Re,t,s,t,Nt);{let r=t;t=s,s=r}j(e,t,ke,Fe,$e,Vt,2,24,24),576*2,j(e,t,St,Ht,Ze,m,36,24,24),576*36,n.queue.submit([e.finish()]);let P=n.createCommandEncoder();P.copyBufferToBuffer(Ve,0,_t,0,864*4),P.copyBufferToBuffer(Xe,0,Dt,0,15552*4),P.copyBufferToBuffer($e,0,yt,0,576*2*4),P.copyBufferToBuffer(Ze,0,wt,0,576*36*4),n.queue.submit([P.finish()]),await Promise.all([_t.mapAsync(GPUMapMode.READ),Dt.mapAsync(GPUMapMode.READ),yt.mapAsync(GPUMapMode.READ),wt.mapAsync(GPUMapMode.READ)]);let k=new Float32Array(_t.getMappedRange()).slice(),g=new Float32Array(Dt.getMappedRange()).slice(),Y=new Float32Array(yt.getMappedRange()).slice(),V=new Float32Array(wt.getMappedRange()).slice();_t.unmap(),Dt.unmap(),yt.unmap(),wt.unmap();let J=2016,h=new Float32Array(J),ae=new Float32Array(J*18),re=0;for(let r=0;r<12;r++)for(let l=0;l<12;l++)for(let ge=0;ge<6;ge++){h[re]=k[ge*144+r*12+l];for(let Se=0;Se<18;Se++){let ut=ge*18+Se;ae[re*18+Se]=g[ut*144+r*12+l]}re++}for(let r=0;r<24;r++)for(let l=0;l<24;l++)for(let ge=0;ge<2;ge++){h[re]=Y[ge*576+r*24+l];for(let Se=0;Se<18;Se++){let ut=ge*18+Se;ae[re*18+Se]=V[ut*576+r*24+l]}re++}return{scores:h,regressors:ae}}async function Xt(e){n.queue.copyExternalImageToTexture({source:e},{texture:Pt},[192,192]);let t=n.createCommandEncoder();{let s=t.beginComputePass();s.setPipeline(F),s.setBindGroup(0,p),s.dispatchWorkgroups(E(192,16),E(192,16),1),s.end()}return Ie(t)}async function en(e,t,s){let P=Math.min(192/t,192/s),k=Math.round(t*P),g=Math.round(s*P),Y=Math.floor((192-k)/2),V=Math.floor((192-g)/2),J=Y/192,h=V/192,ae=A(t,s),re;e instanceof HTMLVideoElement?re=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?re=await createImageBitmap(e,{colorSpaceConversion:"none"}):re=e,n.queue.copyExternalImageToTexture({source:re},{texture:ae},[t,s]);let r=new ArrayBuffer(32),l=new Uint32Array(r),ge=new Float32Array(r);l[0]=t,l[1]=s,l[2]=192,l[3]=0,ge[4]=t/k,ge[5]=s/g,ge[6]=Y,ge[7]=V,n.queue.writeBuffer(N,0,r);let Se=n.createBindGroup({layout:ve,entries:[{binding:0,resource:ae.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:N}},{binding:3,resource:a}]}),ut=n.createCommandEncoder();{let i=ut.beginComputePass();i.setPipeline(se),i.setBindGroup(0,Se),i.dispatchWorkgroups(E(192,16),E(192,16),1),i.end()}return{output:await Ie(ut),lbPadX:J,lbPadY:h}}async function D(e,t){let s=n.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),P=n.createCommandEncoder();P.copyBufferToBuffer(e,0,s,0,t*4),n.queue.submit([P.finish()]),await s.mapAsync(GPUMapMode.READ);let k=new Float32Array(s.getMappedRange()).slice();return s.unmap(),s.destroy(),k}async function Cn(e,t,s){function P(i,d=1e3){let W=i.slice(0,d),le=Math.max(0,Math.floor(i.length/2)-250);return{min:Math.min(...W),max:Math.max(...W),mean:W.reduce((xe,ct)=>xe+ct,0)/W.length,nonZero:W.filter(xe=>xe!==0).length,sample:Array.from(W.slice(0,10)),data500:Array.from(i.slice(0,500)),dataMid500:Array.from(i.slice(le,le+500)),totalLength:i.length}}function k(i,d,W,le){let xe=[],ct=Math.floor(W/2),dt=Math.floor(le/2),Ne=W*le;for(let pt=0;pt<d&&xe.length<500;pt++)for(let Ft=-1;Ft<=1&&xe.length<500;Ft++)for(let Lt=-1;Lt<=1&&xe.length<500;Lt++){let ft=ct+Ft,tn=dt+Lt;ft>=0&&ft<W&&tn>=0&&tn<le&&xe.push(i[pt*Ne+ft*le+tn])}return xe}let g={},Y;e instanceof HTMLImageElement?(t=t??e.naturalWidth,s=s??e.naturalHeight,Y=await createImageBitmap(e,{colorSpaceConversion:"none"})):(t=t??e.width??192,s=s??e.height??192,Y=e);let V=t,J=s;if(V!==192||J!==192){let i=Math.min(192/V,192/J),d=Math.round(V*i),W=Math.round(J*i),le=Math.floor((192-d)/2),xe=Math.floor((192-W)/2),ct=A(V,J);n.queue.copyExternalImageToTexture({source:Y},{texture:ct},[V,J]);let dt=new ArrayBuffer(32),Ne=new Uint32Array(dt),pt=new Float32Array(dt);Ne[0]=V,Ne[1]=J,Ne[2]=192,Ne[3]=0,pt[4]=V/d,pt[5]=J/W,pt[6]=le,pt[7]=xe,n.queue.writeBuffer(N,0,dt);let Ft=n.createBindGroup({layout:ve,entries:[{binding:0,resource:ct.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:N}},{binding:3,resource:a}]});{let Lt=n.createCommandEncoder(),ft=Lt.beginComputePass();ft.setPipeline(se),ft.setBindGroup(0,Ft),ft.dispatchWorkgroups(E(192,16),E(192,16),1),ft.end(),n.queue.submit([Lt.finish()])}}else{n.queue.copyExternalImageToTexture({source:Y},{texture:Pt},[192,192]);let i=I(new Uint32Array([192,192,192])),d=n.createBindGroup({layout:lt,entries:[{binding:0,resource:Pt.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:i}}]});{let W=n.createCommandEncoder(),le=W.beginComputePass();le.setPipeline(F),le.setBindGroup(0,d),le.dispatchWorkgroups(E(192,16),E(192,16),1),le.end(),n.queue.submit([W.finish()])}}{let i=await D(Le,110592),d=P(i);d.dataCenter500=k(i,3,192,192),g.input=d}let h=n.createCommandEncoder(),ae=n.createBindGroup({layout:rt,entries:[{binding:0,resource:{buffer:Le}},{binding:1,resource:{buffer:ue}},{binding:2,resource:{buffer:Z}},{binding:3,resource:{buffer:me}},{binding:4,resource:{buffer:Ye}},{binding:5,resource:{buffer:It}}]}),re=h.beginComputePass();re.setPipeline(ze),re.setBindGroup(0,ae),re.dispatchWorkgroups(E(96,8),E(96,8),32),re.end(),n.queue.submit([h.finish()]);{let i=await D(Ye,294912),d=P(i);d.dataCenter500=k(i,32,96,96),g.initConv=d}let r=Ye,l=Kt;for(let i=0;i<ee.length;i++){let d=ee[i];h=n.createCommandEncoder(),ne(h,d,r,l,r,Qe[i]),n.queue.submit([h.finish()]);let W=r;r=l,l=W;{let le=d.stride===2?d.inH/2:d.inH,xe=le,ct=le*xe*d.outCh,dt=await D(r,ct),Ne=P(dt);Ne.dataCenter500=k(dt,d.outCh,le,xe),Ne.spatialShape=[d.outCh,le,xe],g[`block${i}`]=Ne}i===13&&(h=n.createCommandEncoder(),h.copyBufferToBuffer(r,0,bt,0,576*128*4),n.queue.submit([h.finish()])),i===18&&(h=n.createCommandEncoder(),h.copyBufferToBuffer(r,0,gt,0,144*256*4),n.queue.submit([h.finish()]))}h=n.createCommandEncoder();{let i=I(new Uint32Array([1,256,6,6,12,12])),d=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:ht}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:i}}]}),W=h.beginComputePass();W.setPipeline(G),W.setBindGroup(0,d),W.dispatchWorkgroups(E(12,8),E(12,8),256),W.end()}n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,36864),d=P(i);d.dataCenter500=k(i,256,12,12),g.fpnUpsample6to12=d}h=n.createCommandEncoder(),ce(h,r,z,he,pe,l,zt,256,12,12),n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,36864),d=P(i);d.dataCenter500=k(i,256,12,12),g.fpn6to12Conv=d}{let i=await D(gt,36864),d=P(i);d.dataCenter500=k(i,256,12,12),g.backbone12Skip=d}h=n.createCommandEncoder();{let i=I(new Uint32Array([1,256,12,12,12,12])),d=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:gt}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:i}}]}),W=h.beginComputePass();W.setPipeline(G),W.setBindGroup(0,d),W.dispatchWorkgroups(E(12,8),E(12,8),256),W.end()}n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,36864),d=P(i);d.dataCenter500=k(i,256,12,12),g.fpnAdd12=d}h=n.createCommandEncoder(),ne(h,Ue,r,l,r,Je),n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,36864),d=P(i);d.dataCenter500=k(i,256,12,12),g.fpn12Block1=d}h=n.createCommandEncoder(),ne(h,Pe,r,l,r,et),n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,36864),d=P(i);d.dataCenter500=k(i,256,12,12),g.fpn12Block2=d}h=n.createCommandEncoder(),j(h,r,je,ot,Ve,jt,6,12,12),n.queue.submit([h.finish()]);{let i=await D(Ve,864),d=P(i);d.dataCenter500=k(i,6,12,12),g.cls16=d}h=n.createCommandEncoder(),j(h,r,it,st,Xe,Yt,108,12,12),n.queue.submit([h.finish()]);{let i=await D(Xe,15552),d=P(i,500);d.dataCenter500=k(i,108,12,12),g.reg16=d}h=n.createCommandEncoder();{let i=I(new Uint32Array([1,256,12,12,24,24])),d=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:ht}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:i}}]}),W=h.beginComputePass();W.setPipeline(G),W.setBindGroup(0,d),W.dispatchWorkgroups(E(24,8),E(24,8),256),W.end()}n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,147456),d=P(i);d.dataCenter500=k(i,256,24,24),g.fpnUpsample12to24=d}h=n.createCommandEncoder(),ce(h,r,fe,te,De,l,qt,128,24,24),n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,73728),d=P(i);d.dataCenter500=k(i,128,24,24),g.fpn12to24Conv=d}{let i=await D(bt,73728),d=P(i);d.dataCenter500=k(i,128,24,24),g.backbone24Skip=d}h=n.createCommandEncoder();{let i=I(new Uint32Array([1,128,24,24,24,24])),d=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:bt}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:i}}]}),W=h.beginComputePass();W.setPipeline(G),W.setBindGroup(0,d),W.dispatchWorkgroups(E(24,8),E(24,8),128),W.end()}n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,73728),d=P(i);d.dataCenter500=k(i,128,24,24),g.fpnAdd24=d}h=n.createCommandEncoder(),ne(h,We,r,l,r,Bt),n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,73728),d=P(i);d.dataCenter500=k(i,128,24,24),g.fpn24Block1=d}h=n.createCommandEncoder(),ne(h,Re,r,l,r,Nt),n.queue.submit([h.finish()]);{let i=r;r=l,l=i}{let i=await D(r,73728),d=P(i);d.dataCenter500=k(i,128,24,24),g.fpn24Block2=d}h=n.createCommandEncoder(),j(h,r,ke,Fe,$e,Vt,2,24,24),n.queue.submit([h.finish()]);{let i=await D($e,1152),d=P(i);d.dataCenter500=k(i,2,24,24),g.cls8=d}h=n.createCommandEncoder(),j(h,r,St,Ht,Ze,m,36,24,24),n.queue.submit([h.finish()]);{let i=await D(Ze,20736),d=P(i);d.dataCenter500=k(i,36,24,24),g.reg8=d}g.initWeights=P(await D(ue,100),100),g.initBias=P(await D(Z,32),32),g.cls16Weights=P(await D(je,100),100),g.cls16Bias=P(await D(ot,6),6),g.cls8Weights=P(await D(ke,100),100),g.cls8Bias=P(await D(Fe,2),2),g.fpn6to12Weights=P(await D(z,100),100);let ge=await D(Ve,864),Se=await D($e,576*2);g.rawScores=new Float32Array(2016),g.rawScores.set(ge,0),g.rawScores.set(Se,864);let ut=await D(Xe,15552),sn=await D(Ze,576*36);return g.rawRegressors=new Float32Array(36288),g.rawRegressors.set(ut,0),g.rawRegressors.set(sn,15552),g.rawInput=await D(Le,36864*3),g}return{device:n,run:Xt,runWithResize:en,debugRun:Cn}}function Hn(){let u=[];for(let B=0;B<12;B++)for(let n=0;n<12;n++){let T=(n+.5)/12,v=(B+.5)/12;for(let a=0;a<6;a++)u.push({x:T,y:v})}for(let B=0;B<24;B++)for(let n=0;n<24;n++){let T=(n+.5)/24,v=(B+.5)/24;for(let a=0;a<2;a++)u.push({x:T,y:v})}return u}var Pn=Hn();function Kn(u){return 1/(1+Math.exp(-u))}function nn(u,B){let n=[],{scores:T,regressors:v}=u,a=192;for(let b=0;b<Pn.length;b++){let S=Kn(T[b]);if(S<B)continue;let U=Pn[b],w=b*18,f=U.x+v[w+0]/a,_=U.y+v[w+1]/a,x=v[w+2]/a,q=v[w+3]/a,H=[];for(let c=0;c<7;c++){let C=U.x+v[w+4+c*2]/a,K=U.y+v[w+4+c*2+1]/a;H.push([C,K])}n.push({score:S,box:[f,_,x,q],keypoints:H})}return n}function an(u,B){if(u.length===0)return[];let n=[...u].sort((a,b)=>b.score-a.score),T=[],v=new Set;for(let a=0;a<n.length;a++){if(v.has(a))continue;let b=[a];for(let H=a+1;H<n.length;H++)v.has(H)||On(n[a],n[H])>B&&(b.push(H),v.add(H));let S=0,U=0,w=0,f=0,_=0,x=[];for(let H=0;H<7;H++)x.push([0,0]);for(let H of b){let c=n[H],C=c.score;S+=C,U+=c.box[0]*C,w+=c.box[1]*C,f+=c.box[2]*C,_+=c.box[3]*C;for(let K=0;K<7;K++)x[K][0]+=c.keypoints[K][0]*C,x[K][1]+=c.keypoints[K][1]*C}let q=1/S;T.push({score:n[a].score,box:[U*q,w*q,f*q,_*q],keypoints:x.map(([H,c])=>[H*q,c*q])})}return T}function On(u,B){let n=u.box[0]-u.box[2]/2,T=u.box[1]-u.box[3]/2,v=u.box[0]+u.box[2]/2,a=u.box[1]+u.box[3]/2,b=B.box[0]-B.box[2]/2,S=B.box[1]-B.box[3]/2,U=B.box[0]+B.box[2]/2,w=B.box[1]+B.box[3]/2,f=Math.max(n,b),_=Math.max(T,S),x=Math.min(v,U),q=Math.min(a,w),H=Math.max(0,x-f),c=Math.max(0,q-_),C=H*c,K=(v-n)*(a-T),Ce=(U-b)*(w-S),Ge=K+Ce-C;return Ge>0?C/Ge:0}function zn(u){let[B,n,T,v]=u.box,a=u.keypoints[0],b=u.keypoints[2],S=b[0]-a[0],U=b[1]-a[1],w=Math.atan2(U,S),_=-Math.PI/2-w,x=Math.max(T,v),H=x*2.6,c=-.5*x,C=Math.cos(_),K=Math.sin(_),Ce=c*K,Ge=c*C;return{centerX:B+Ce,centerY:n+Ge,width:H,height:H,rotation:_}}function Bn(u,B={}){let{scoreThreshold:n=.5,nmsThreshold:T=.3,maxHands:v=2}=B;async function a(w){let f=await u.run(w),_=nn(f,n);return an(_,T).slice(0,v).map(zn)}async function b(w){let f=await u.run(w),_=nn(f,n);return an(_,T).slice(0,v)}async function S(w,f,_){let{output:x,lbPadX:q,lbPadY:H}=await u.runWithResize(w,f,_),c=nn(x,n);return{detections:an(c,T).slice(0,v),lbPadX:q,lbPadY:H}}async function U(w,f,_){let{output:x,lbPadX:q,lbPadY:H}=await u.runWithResize(w,f,_);return{scores:x.scores,regressors:x.regressors,lbPadX:q,lbPadY:H}}return{detect:a,detectRaw:b,detectRawWithResize:S,detectRawSSD:U,model:u}}var rn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function on(u){let B={};for(let n=0;n<rn.length;n++)B[rn[n]]=u[n];return B}function Fn(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var In=Fn(`
struct CropParams { src_width:u32, src_height:u32, dst_size:u32, _pad:u32, }
struct AffineTransform { a:f32, b:f32, tx:f32, c:f32, d:f32, ty:f32, }

@group(0)@binding(0) var src_tex:texture_2d<f32>;
@group(0)@binding(1) var<storage,read_write> output:array<f32>;
@group(0)@binding(2) var<uniform> params:CropParams;
@group(0)@binding(3) var<uniform> transform:AffineTransform;
@group(0)@binding(4) var src_sampler:sampler;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let dst_x=gid.x; let dst_y=gid.y;
  if(dst_x>=params.dst_size||dst_y>=params.dst_size){return;}

  // Map crop pixel to source normalized coordinates [0,1]
  let fx=f32(dst_x)+0.5;
  let fy=f32(dst_y)+0.5;
  let src_nx=transform.a*fx+transform.b*fy+transform.tx;
  let src_ny=transform.c*fx+transform.d*fy+transform.ty;

  let out_stride=params.dst_size*params.dst_size;

  // Hardware bilinear sampling via textureSampleLevel with clamp-to-edge sampler.
  // Clamp-to-edge matches MediaPipe's BORDER_REPLICATE default
  // (ImageToTensorCalculatorOptions proto: "BORDER_REPLICATE is used by default").
  let pixel = textureSampleLevel(src_tex, src_sampler, vec2<f32>(src_nx, src_ny), 0.0);

  // Write CHW format
  output[0u*out_stride+dst_y*params.dst_size+dst_x]=pixel.r;
  output[1u*out_stride+dst_y*params.dst_size+dst_x]=pixel.g;
  output[2u*out_stride+dst_y*params.dst_size+dst_x]=pixel.b;
}
`);function xn(u){let B=u.createShaderModule({code:In}),n=u.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),T=u.createComputePipeline({layout:u.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:B,entryPoint:"main"}}),v=u.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),a=u.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),b=u.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),S=new Float32Array(8);function U(w,f,_,x,q,H,c){u.queue.writeBuffer(a,0,new Uint32Array([q,H,c,0])),S.set(x),u.queue.writeBuffer(b,0,S);let C=u.createBindGroup({layout:n,entries:[{binding:0,resource:f.createView()},{binding:1,resource:{buffer:_}},{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:b}},{binding:4,resource:v}]}),K=w.beginComputePass();K.setPipeline(T),K.setBindGroup(0,C),K.dispatchWorkgroups(Math.ceil(c/16),Math.ceil(c/16),1),K.end()}return{crop:U}}var Nn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@0.3.0/weights";async function qn(u={}){let{weightsUrl:B,scoreThreshold:n=.5,palmScoreThreshold:T=.5,maxHands:v=3}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let a=(B??Nn).replace(/\/$/,"")+"/",[b,S,U,w]=await Promise.all([fetch(`${a}weights_f16_full.json`),fetch(`${a}weights_f16_full.bin`),fetch(`${a}palm_detection_weights.json`),fetch(`${a}palm_detection_weights.bin`)]);if(!b.ok)throw new Error(`Failed to fetch landmark weights: ${b.status}`);if(!S.ok)throw new Error(`Failed to fetch landmark weights: ${S.status}`);if(!U.ok)throw new Error(`Failed to fetch palm detection weights: ${U.status}`);if(!w.ok)throw new Error(`Failed to fetch palm detection weights: ${w.status}`);let[f,_,x,q]=await Promise.all([b.json(),S.arrayBuffer(),U.json(),w.arrayBuffer()]),H=cn(f,_),c=un(x,q),C=224,K=await dn(H);{let y=new OffscreenCanvas(C,C),R=y.getContext("2d");R.fillStyle="#886644",R.fillRect(0,0,C,C),R.fillStyle="#cc9966",R.fillRect(50,50,124,124);let G=await K.runFromCanvas(y);G.landmarks.every(se=>se===0)&&G.handflag.every(se=>se===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let Ce=await wn(c),Ge=Bn(Ce,{scoreThreshold:T,maxHands:v}),ie=[];function M(y,R,G){let F=y[0],se=y[5],Me=y[9],be=y[13],_e=F.x*R,Ae=F.y*G,ue=(se.x+be.x)/2,Z=(se.y+be.y)/2;ue=(ue+Me.x)/2*R,Z=(Z+Me.y)/2*G;let me=Math.PI/2-Math.atan2(-(Z-Ae),ue-_e),$=me-2*Math.PI*Math.floor((me+Math.PI)/(2*Math.PI)),ee=[0,1,2,3,5,6,9,10,13,14,17,18],oe=Math.cos($),z=Math.sin($),he=1/0,pe=-1/0,Ue=1/0,Pe=-1/0;for(let it of ee){let st=y[it],Ke=st.x*R,ke=st.y*G,Fe=oe*Ke+z*ke,kt=-z*Ke+oe*ke;he=Math.min(he,Fe),pe=Math.max(pe,Fe),Ue=Math.min(Ue,kt),Pe=Math.max(Pe,kt)}let Ee=(he+pe)/2,fe=(Ue+Pe)/2,te=pe-he,De=Pe-Ue,We=(oe*Ee-z*fe)/R,Re=(z*Ee+oe*fe)/G;te/=R,De/=G;let He=-.1;We+=-G*De*He*z/R,Re+=De*He*oe;let mt=Math.max(te*R,De*G)*2;return{centerXpx:We*R,centerYpx:Re*G,sizePx:mt,rotation:$}}let de=K.device,ye=null,X=null,Te=null,Ct=0,rt=0;function vt(){return ye||(ye=xn(de)),ye}function Ut(){return X||(X=de.createBuffer({size:3*C*C*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),X}function Gt(y,R){return(!Te||Ct!==y||rt!==R)&&(Te&&Te.destroy(),Te=de.createTexture({size:[y,R],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Ct=y,rt=R),Te}let qe=0,we=0;function lt(y){let R=1/(1-2*qe),G=1/(1-2*we);return{score:y.score,box:[(y.box[0]-qe)*R,(y.box[1]-we)*G,y.box[2]*R,y.box[3]*G],keypoints:y.keypoints.map(([F,se])=>[(F-qe)*R,(se-we)*G])}}function ve(y,R,G){let F=y.keypoints[0],se=y.keypoints[2],Me=(se[0]-F[0])*R,be=(se[1]-F[1])*G,_e=Math.atan2(-be,Me),ue=Math.PI/2-_e,Z=ue-2*Math.PI*Math.floor((ue+Math.PI)/(2*Math.PI)),[me,$,ee,oe]=y.box,z=Math.cos(Z),he=Math.sin(Z),pe=oe*G,Ue=me+.5*pe*he/R,Pe=$+-.5*oe*z,te=Math.max(ee*R,oe*G)*2.6;return{centerXpx:Ue*R,centerYpx:Pe*G,sizePx:te,rotation:Z}}function Mt(y){return y instanceof HTMLCanvasElement||y instanceof OffscreenCanvas?[y.width,y.height]:typeof ImageBitmap<"u"&&y instanceof ImageBitmap?[y.width,y.height]:y instanceof ImageData?[y.width,y.height]:y instanceof HTMLVideoElement?[y.videoWidth,y.videoHeight]:y instanceof HTMLImageElement?[y.naturalWidth,y.naturalHeight]:[C,C]}async function Be(y,R,G,F,se,Me){let be=Math.cos(y.rotation),_e=Math.sin(y.rotation),Ae=y.sizePx/C,ue=C/2,Z=be*Ae/G,me=-_e*Ae/G,$=y.centerXpx/G-ue*(Z+me),ee=_e*Ae/F,oe=be*Ae/F,z=y.centerYpx/F-ue*(ee+oe),he=de.createCommandEncoder();se.crop(he,R,Me,[Z,me,$,ee,oe,z],G,F,C),de.queue.submit([he.finish()]);let pe=await K.runFromGPUBuffer(Me),Ue=pe.handflag[0];if(Ue<n)return null;let Pe=pe.handedness[0]>.5,Ee=[];for(let fe=0;fe<21;fe++){let te=pe.landmarks[fe*3],De=pe.landmarks[fe*3+1],We=pe.landmarks[fe*3+2],Re=(te-.5)*y.sizePx,He=(De-.5)*y.sizePx,je=be*Re-_e*He+y.centerXpx,ot=_e*Re+be*He+y.centerYpx;Ee.push({x:je/G,y:ot/F,z:We})}return{landmarks:Ee,score:Ue,handedness:Pe?"right":"left"}}async function ze(y){let R=y,G,F;if(y instanceof HTMLVideoElement||y instanceof HTMLImageElement){let $=await createImageBitmap(y,{colorSpaceConversion:"none"});R=$,G=$.width,F=$.height}else[G,F]=Mt(y);let se=vt(),Me=Ut(),be=Gt(G,F),_e;if(R instanceof ImageData?_e=await createImageBitmap(R,{colorSpaceConversion:"none"}):_e=R,de.queue.copyExternalImageToTexture({source:_e},{texture:be},[G,F]),ie.length>0){let $=[];for(let ee of ie){let oe=M(ee.landmarks,G,F),z=await Be(oe,be,G,F,se,Me);z&&$.push({score:z.score,handedness:z.handedness,landmarks:z.landmarks,keypoints:on(z.landmarks)})}if($.length>0)return ie=$.map(ee=>({landmarks:ee.landmarks,handedness:ee.handedness})),$;ie=[]}let{detections:Ae,lbPadX:ue,lbPadY:Z}=await Ge.detectRawWithResize(R,G,F);if(qe=ue,we=Z,Ae.length===0)return ie=[],[];let me=[];for(let $ of Ae){let ee=lt($),oe=ve(ee,G,F),z=await Be(oe,be,G,F,se,Me);z&&me.push({score:z.score,handedness:z.handedness,landmarks:z.landmarks,keypoints:on(z.landmarks)})}return ie=me.map($=>({landmarks:$.landmarks,handedness:$.handedness})),me}function At(){Te&&Te.destroy(),X&&X.destroy(),Te=null,X=null,ye=null,K.device.destroy(),Ce.device.destroy()}function Wt(){ie=[]}return{detect:ze,dispose:At,reset:Wt}}export{rn as LANDMARK_NAMES,qn as createHandpose};
