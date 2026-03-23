function on(s,B){let n=new Map,D=s.dtype??"float32";for(let v=0;v<s.keys.length;v++){let a=s.keys[v],g=s.shapes[v],T=s.offsets[v],U=g.reduce((b,x)=>b*x,1),y,l;if(D==="float32")y=new Float32Array(B,T,U);else{let b=new DataView(B);y=new Float32Array(U);for(let x=0;x<U;x++)y[x]=xn(b.getUint16(T+x*2,!0));l=B.slice(T,T+U*2)}n.set(a,{data:y,shape:g,rawF16:l})}return n}function xn(s){let B=s>>15&1,n=s>>10&31,D=s&1023;if(n===0){if(D===0)return B?-0:0;let g=-14,T=D/1024;return(B?-1:1)*Math.pow(2,g)*T}if(n===31)return D===0?B?-1/0:1/0:NaN;let v=n-15,a=1+D/1024;return(B?-1:1)*Math.pow(2,v)*a}var Cn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Nn=Cn.map(([s,B,n,D,v])=>({type:"resmodule",inCh:s,outCh:B,h:n,w:n,stride:D,prefix:v}));function sn(s,B){let n=new Map,D=s.dtype??"float32",v=new Map;for(let a=0;a<s.keys.length;a++){let g=s.keys[a],T=s.shapes[a],U=s.offsets[a],y=T.reduce((O,c)=>O*c,1),l,b;if(D==="float32")l=new Float32Array(B,U,y);else{let O=new DataView(B);l=new Float32Array(y);for(let c=0;c<y;c++)l[c]=vn(O.getUint16(U+c*2,!0));b=B.slice(U,U+y*2)}let x=v.get(g)??0;v.set(g,x+1);let Y=x===0?g:`${g}__${x}`;n.set(Y,{data:l,shape:T,rawF16:b})}return n}function vn(s){let B=s>>15&1,n=s>>10&31,D=s&1023;return n===0?D===0?B?-0:0:(B?-1:1)*Math.pow(2,-14)*(D/1024):n===31?D===0?B?-1/0:1/0:NaN:(B?-1:1)*Math.pow(2,n-15)*(1+D/1024)}function nt(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Un=nt(`
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
`),Gn=nt(`
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
`),Mn=nt(`
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
`),An=nt(`
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
  output[oc*spatial+pix]=sum;
}
`),Sn=nt(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Tn=nt(`
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
`),En=nt(`
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
`),Dn=nt(`
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
`),Bt=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],Rn=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function un(s,B){if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");let D=n.features.has("shader-f16"),v=D?["shader-f16"]:[],a=await n.requestDevice({requiredFeatures:v,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}}),g=s.values().next().value,T=D&&!!g?.rawF16&&!B?.forceF32;function U(m){if(T&&m.rawF16){let r=new Uint16Array(m.rawF16);if(r.length%2!==0){let d=new Uint16Array(r.length+1);return d.set(r),d}return r}return m.data}function y(m){return T&&m.rawF16?Math.ceil(m.rawF16.byteLength/4)*4:m.data.byteLength}let l=T?2:4;function b(m){if(!T)return m;let r=m;return r=r.replace(/array<f32>/g,"array<f16>"),r=r.replace(/array<f32,/g,"array<f16,"),r=r.replace(/var sum:f32=0\.0/g,"var sum:f16=0.0h"),r=r.replace(/var sum0:f32=0\.0/g,"var sum0:f16=0.0h"),r=r.replace(/var sum1:f32=0\.0/g,"var sum1:f16=0.0h"),r=r.replace(/var sum2:f32=0\.0/g,"var sum2:f16=0.0h"),r=r.replace(/var sum3:f32=0\.0/g,"var sum3:f16=0.0h"),r=r.replace(/\/f32\(params/g,"/f16(params"),r=r.replace(/,0\.0\),6\.0\)/g,",0.0h),6.0h)"),r=r.replace(/->f32\{/g,"->f16{"),r=r.replace(/->f32 \{/g,"->f16 {"),r=r.replace(/return 0\.0;/g,"return 0.0h;"),"enable f16;"+r}function x(m){if(!T)return m;let r=b(m);return r=r.replace("read>input:array<f16>","read>input:array<f32>"),r=r.replace(/input\[in_idx\]/g,"f16(input[in_idx])"),r}function Y(m){if(!T)return m;let r=m;return r=r.replace("read>input:array<f32>","read>input:array<f16>"),r=r.replace("read>weight:array<f32>","read>weight:array<f16>"),r=r.replace("read>bias:array<f32>","read>bias:array<f16>"),r=r.replace(/input\[ic\]/g,"f32(input[ic])"),r=r.replace(/weight\[oc\*params\.in_features\+ic\]/g,"f32(weight[oc*params.in_features+ic])"),r=r.replace(/bias\[oc\]/g,"f32(bias[oc])"),"enable f16;"+r}let O={r:"read-only-storage",s:"storage",u:"uniform"};function c(m){return a.createBindGroupLayout({entries:m.map((r,d)=>r==="t"?{binding:d,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:d,visibility:GPUShaderStage.COMPUTE,buffer:{type:O[r]}})})}let C=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,z=GPUBufferUsage.STORAGE,Ce=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,Ge=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,re=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function A(m,r){return a.createBuffer({size:Math.max(m,4),usage:r})}function ue(m,r){return a.createBindGroup({layout:m,entries:r.map((d,H)=>({binding:H,resource:"size"in d?{buffer:d}:d}))})}function ye(m,r){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[m]}),compute:{module:r,entryPoint:"main"}})}function $(m){let r=s.get(m);if(!r)throw new Error(`Missing weight: ${m}`);return r}let Te=a.createShaderModule({code:Un}),xt=a.createShaderModule({code:x(Gn)}),rt=a.createShaderModule({code:b(Mn)}),Ct=a.createShaderModule({code:b(An)}),vt=a.createShaderModule({code:b(kn)}),Ut=a.createShaderModule({code:b(Sn)}),qe=a.createShaderModule({code:b(Tn)}),we=a.createShaderModule({code:Y(En)}),ft=a.createShaderModule({code:Y(Dn)}),ve=c(["r","r","r","s","u"]),Gt=c(["r","r","s","u"]),Be=c(["r","s","u"]),ze=c(["r","r","r","s","u"]),Mt=c(["t","s","u"]),Lt=ye(Mt,Te),_=ye(ve,xt),W=ye(ve,rt),M=ye(ve,Ct),N=ye(ve,vt),oe=ye(Gt,Ut),Me=ye(Be,qe),be=ye(ze,we),_e=ye(ze,ft),ie=1152*112*112*4,Q=A(ie,Ge),le=A(ie,Ge),Z=A(ie,z),ee=A(ie,z),ae=A(ie,C),I=A(672*224*4,Ge),me=A(1152*4,Ce),ce=A(252,Ce),Ue=A(252,Ce),Pe=A(4,Ce),Ee=A(4,Ce),de=A(260,Ge),te=A(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),De=a.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),We=A(4,re);a.queue.writeBuffer(We,0,new Uint32Array([224]));let Re=$("conv2d"),He=$("batch_normalization"),je=U(Re),ot=U(He),lt=A(y(Re),C),it=A(y(He),C),st=A(24,re);a.queue.writeBuffer(lt,0,je),a.queue.writeBuffer(it,0,ot),a.queue.writeBuffer(st,0,new Uint32Array([3,24,224,224,112,112]));let Ke=112,ke=112,Fe=[];for(let m=0;m<Bt.length;m++){let r=Bt[m],d=Rn[m],H=Ke,F=ke,J=r.stride===2?Math.floor(Ke/2):Ke,j=r.stride===2?Math.floor(ke/2):ke,k={spec:r,inH:H,inW:F,outH:J,outW:j,dwW:A(4,C),dwB:A(4,C),dwU:A(32,re)};if(d.expandConvKey){let V=$(d.expandConvKey),se=$(d.expandBNKey);k.expandW=A(y(V),C),k.expandB=A(y(se),C),k.expandU=A(16,re),a.queue.writeBuffer(k.expandW,0,U(V)),a.queue.writeBuffer(k.expandB,0,U(se)),a.queue.writeBuffer(k.expandU,0,new Uint32Array([r.inCh,r.expandCh,H,F]))}let tt=$(d.dwWeightKey),Oe=$(d.dwBNKey);k.dwW=A(y(tt),C),k.dwB=A(y(Oe),C),a.queue.writeBuffer(k.dwW,0,U(tt)),a.queue.writeBuffer(k.dwB,0,U(Oe));let ne=Math.floor((r.dwKernel-r.stride)/2);if(a.queue.writeBuffer(k.dwU,0,new Uint32Array([r.expandCh,H,F,J,j,r.stride,ne,r.dwKernel])),r.hasProject&&d.projectConvKey){let V=$(d.projectConvKey),se=$(d.projectBNKey);k.projectW=A(y(V),C),k.projectB=A(y(se),C),k.projectU=A(16,re),a.queue.writeBuffer(k.projectW,0,U(V)),a.queue.writeBuffer(k.projectB,0,U(se)),a.queue.writeBuffer(k.projectU,0,new Uint32Array([r.expandCh,r.outCh,J,j]))}Fe.push(k),Ke=J,ke=j}function At(m,r){let d=s.get(m);if(!d)throw new Error(`Missing weight: ${m}`);if(d.shape.length!==r)throw new Error(`Weight ${m} has rank ${d.shape.length}, expected ${r}`);return d}let kt=$("conv_landmarks__1"),Wt=$("conv_world_landmarks__1"),St=$("conv_handflag__1"),Le=$("conv_handedness__1"),Ye=$("Identity"),Ht=$("Identity_1"),Tt=$("Identity_2"),mt=$("Identity_3"),Kt=A(y(kt),C),Xt=A(y(Ye),C),ht=A(y(Wt),C),gt=A(y(mt),C),Ve=A(y(St),C),Xe=A(y(Ht),C),$e=A(y(Le),C),Ze=A(y(Tt),C);a.queue.writeBuffer(Kt,0,U(kt)),a.queue.writeBuffer(Xt,0,U(Ye)),a.queue.writeBuffer(ht,0,U(Wt)),a.queue.writeBuffer(gt,0,U(mt)),a.queue.writeBuffer(Ve,0,U(St)),a.queue.writeBuffer(Xe,0,U(Ht)),a.queue.writeBuffer($e,0,U(Le)),a.queue.writeBuffer(Ze,0,U(Tt));let bt=A(8,re),Et=A(8,re),_t=A(8,re),yt=A(8,re);a.queue.writeBuffer(bt,0,new Uint32Array([1152,63])),a.queue.writeBuffer(Et,0,new Uint32Array([1152,63])),a.queue.writeBuffer(_t,0,new Uint32Array([1152,1])),a.queue.writeBuffer(yt,0,new Uint32Array([1152,1]));let wt=A(8,re);a.queue.writeBuffer(wt,0,new Uint32Array([1152,Ke*ke]));let R=new Map;for(let m=0;m<Bt.length;m++)if(Bt[m].hasResidual){let r=Fe[m],d=A(4,re);a.queue.writeBuffer(d,0,new Uint32Array([Bt[m].outCh*r.outH*r.outW])),R.set(m,d)}let q=ue(Mt,[De.createView(),Q,We]),Ft=ue(ve,[Q,lt,it,le,st]),Qe=new Float32Array(1),Je=new Float32Array(1),et=new Float32Array(63);function Pt(m,r){let d=m.beginComputePass();d.setPipeline(_),d.setBindGroup(0,Ft),d.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),d.end();let H=le,F=Q;for(let J=0;J<Bt.length;J++){let j=Bt[J],k=Fe[J];if(j.hasResidual){let ne=j.inCh*k.inH*k.inW*l;m.copyBufferToBuffer(H,0,ae,0,ne)}if(d=m.beginComputePass(),k.expandW){let ne=ue(ve,[H,k.expandW,k.expandB,Z,k.expandU]);d.setPipeline(W),d.setBindGroup(0,ne),d.dispatchWorkgroups(Math.ceil(k.inW/8),Math.ceil(k.inH/8),j.expandCh)}let tt=k.expandW?Z:H,Oe=ue(ve,[tt,k.dwW,k.dwB,ee,k.dwU]);if(d.setPipeline(M),d.setBindGroup(0,Oe),d.dispatchWorkgroups(Math.ceil(k.outW/8),Math.ceil(k.outH/8),j.expandCh),j.hasProject&&k.projectW){let ne=(j.hasResidual,F),V=ue(ve,[ee,k.projectW,k.projectB,ne,k.projectU]);if(d.setPipeline(N),d.setBindGroup(0,V),d.dispatchWorkgroups(Math.ceil(k.outW/8),Math.ceil(k.outH/8),j.outCh),j.hasResidual){let se=R.get(J),Ie=ue(Gt,[F,ae,H,se]);d.setPipeline(oe),d.setBindGroup(0,Ie),d.dispatchWorkgroups(Math.ceil(j.outCh*k.outH*k.outW/256))}else{let se=H;H=F,F=se}}if(d.end(),!j.hasProject){d=m.beginComputePass();let ne=ue(Be,[ee,me,wt]);d.setPipeline(Me),d.setBindGroup(0,ne),d.dispatchWorkgroups(Math.ceil(1152/256));let V=ue(ze,[me,Kt,Xt,ce,bt]);d.setPipeline(be),d.setBindGroup(0,V),d.dispatchWorkgroups(1);let se=ue(ze,[me,Ve,Xe,Pe,_t]);d.setPipeline(_e),d.setBindGroup(0,se),d.dispatchWorkgroups(1);let Ie=ue(ze,[me,$e,Ze,Ee,yt]);d.setPipeline(_e),d.setBindGroup(0,Ie),d.dispatchWorkgroups(1),d.end(),m.copyBufferToBuffer(Pe,0,de,0,4),m.copyBufferToBuffer(Ee,0,de,4,4),m.copyBufferToBuffer(ce,0,de,8,252),r&&m.copyBufferToBuffer(de,0,r,0,260);return}}}async function It(m){a.queue.writeBuffer(I,0,m);let r=a.createCommandEncoder();r.copyBufferToBuffer(I,0,Q,0,672*224*4),Pt(r,te),a.queue.submit([r.finish()]);let d=te.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await d;let H=new Float32Array(te.getMappedRange());Qe[0]=H[0],Je[0]=H[1];for(let F=0;F<63;F++)et[F]=H[2+F]/224;return te.unmap(),{handflag:new Float32Array(Qe),handedness:new Float32Array(Je),landmarks:new Float32Array(et)}}async function Dt(m){a.queue.copyExternalImageToTexture({source:m},{texture:De},[224,224]);let r=a.createCommandEncoder();{let F=r.beginComputePass();F.setPipeline(Lt),F.setBindGroup(0,q),F.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),F.end()}Pt(r,te),a.queue.submit([r.finish()]);let d=te.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await d;let H=new Float32Array(te.getMappedRange());Qe[0]=H[0],Je[0]=H[1];for(let F=0;F<63;F++)et[F]=H[2+F]/224;return te.unmap(),{handflag:new Float32Array(Qe),handedness:new Float32Array(Je),landmarks:new Float32Array(et)}}async function $t(m){let r=a.createCommandEncoder();r.copyBufferToBuffer(m,0,Q,0,672*224*4),Pt(r,te),a.queue.submit([r.finish()]);let d=te.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await d;let H=new Float32Array(te.getMappedRange());Qe[0]=H[0],Je[0]=H[1];for(let F=0;F<63;F++)et[F]=H[2+F]/224;return te.unmap(),{handflag:new Float32Array(Qe),handedness:new Float32Array(Je),landmarks:new Float32Array(et)}}async function Zt(){return null}async function Qt(){return null}async function Ot(m=100){let r=new OffscreenCanvas(224,224),d=r.getContext("2d");d.fillStyle="#886644",d.fillRect(0,0,224,224);for(let j=0;j<5;j++)await Dt(r);let H=performance.now();for(let j=0;j<m;j++)await Dt(r);let J=(performance.now()-H)/m;return{avgMs:J,fps:1e3/J}}async function Nt(m=100){let r=await Ot(m);return{...r,medianMs:r.avgMs,minMs:r.avgMs}}async function qt(m){return Dt(m)}async function jt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Yt(m){let r={};async function d(J,j,k){let tt=j*4,Oe=a.createBuffer({size:tt,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),ne=a.createCommandEncoder();ne.copyBufferToBuffer(J,0,Oe,0,tt),a.queue.submit([ne.finish()]),await a.queue.onSubmittedWorkDone(),await Oe.mapAsync(GPUMapMode.READ);let V=new Float32Array(Oe.getMappedRange()),se=1/0,Ie=-1/0,Vt=0;for(let L=0;L<V.length;L++)V[L]<se&&(se=V[L]),V[L]>Ie&&(Ie=V[L]),V[L]!==0&&Vt++;let Jt=Array.from(V.slice(0,5));Oe.unmap(),Oe.destroy(),r[k]={min:se,max:Ie,nonZero:Vt,total:j,sample:Jt}}let H=new Float32Array(672*224);for(let J=0;J<50176;J++)H[J]=.5,H[50176+J]=.3,H[448*224+J]=.7;a.queue.writeBuffer(I,0,H);let F=a.createCommandEncoder();return F.copyBufferToBuffer(I,0,Q,0,672*224*4),Pt(F,te),a.queue.submit([F.finish()]),await a.queue.onSubmittedWorkDone(),await d(Q,672*224,"inputBufA"),await d(le,2688*112,"afterInitConvBufB"),await d(me,1152,"gapOutput"),await d(ce,63,"landmarks"),await d(Pe,1,"handflag"),await d(de,65,"unifiedOutput"),r}return{device:a,run:It,runFromCanvas:Dt,runFromGPUBuffer:$t,runFromCanvasPipelined:Zt,flushPipelined:Qt,benchmark:Ot,benchmarkGPU:Nt,runFromCanvasViaRender:qt,benchmarkDiagnostic:jt,debugLayerOutputs:Yt}}function at(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var cn=at(`
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
`),dn=at(`
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
`),pn=at(`
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
`),fn=at(`
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
`),ln=at(`
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
`),mn=at(`
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
`),hn=at(`
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
`),gn=at(`
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
`),bn=at(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function _n(s,B){let n;if(B)n=B;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");n=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let D={r:"read-only-storage",s:"storage",u:"uniform"};function v(e){return n.createBindGroupLayout({entries:e.map((t,i)=>t==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:t==="sm"?{binding:i,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:D[t]}})})}let a=n.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),g=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,T=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,U=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,y=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(e,t){return n.createBuffer({size:Math.max(e,4),usage:t})}function b(e,t,i){n.queue.writeBuffer(e,t,i)}function x(e){let t=l(e.data.byteLength,g);return b(t,0,e.data),t}let Y=Array.from(s.keys());function O(e){let t=s.get(e);if(!t)throw new Error(`Weight not found: ${e}`);return t}function c(...e){let t=Y.find(i=>e.every(w=>i.includes(w)));if(!t)throw new Error(`Weight not found for: ${e.join(", ")}`);return O(t)}function C(e){let[,t,i,w]=e.shape,S=new Float32Array(w*25);for(let h=0;h<w;h++)for(let X=0;X<t;X++)for(let f=0;f<i;f++)S[h*25+X*5+f]=e.data[X*i*w+f*w+h];return S}function z(e){let[t,,,i]=e.shape,w=new Float32Array(t*i);for(let S=0;S<t;S++)for(let h=0;h<i;h++)w[S*i+h]=e.data[S*i+h];return w}let Ce=n.createShaderModule({code:cn}),Ge=n.createShaderModule({code:dn}),re=n.createShaderModule({code:pn}),A=n.createShaderModule({code:fn}),ue=n.createShaderModule({code:mn}),ye=n.createShaderModule({code:ln}),$=n.createShaderModule({code:hn}),Te=n.createShaderModule({code:gn}),xt=n.createShaderModule({code:bn}),rt=v(["r","r","r","r","s","u"]),Ct=v(["r","r","r","s","u"]),vt=v(["r","r","r","r","r","s","u"]),Ut=v(["r","r","r","s","u"]),qe=v(["r","r","r","r","s","u"]),we=v(["r","r","s","u"]),ft=v(["t","s","u"]),ve=v(["t","s","u","sm"]),Gt=v(["s","u"]);function Be(e,t){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:t,entryPoint:"main"}})}let ze=Be(rt,Ce),Mt=Be(Ct,Ge),Lt=Be(vt,re),_=Be(Ut,A),W=Be(qe,ue),M=Be(we,ye),N=Be(ft,$),oe=Be(ve,Te),Me=Be(Gt,xt),be=c("conv2d/Conv2D"),_e=c("batch_normalization/","conv2d/Conv2D"),Ae=c("p_re_lu/"),ie=x(be),Q=x(_e),le=x(Ae),ee=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(e=>{let t=c(e.dwKey),i=c(e.pwKey),w=c(e.bnKey),S=c(e.preluKey),h=C(t),X=l(h.byteLength,g);b(X,0,h);let f=new Float32Array(e.inCh),he=l(f.byteLength,g);b(he,0,f);let pe=z(i),p=l(pe.byteLength,g);b(p,0,pe);let G=x(w),P=x(S);return{dwWeightBuf:X,dwBiasBuf:he,pwWeightBuf:p,pwBiasBuf:G,alphaBuf:P,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),ae=z(c("conv2d_25/Conv2D")),I=l(ae.byteLength,g);b(I,0,ae);let me=x(c("batch_normalization_25/")),ce=x(c("p_re_lu_25/")),Ue={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_24/")),t=l(e.byteLength,g);return b(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=l(e.byteLength,g);return b(t,0,e),t})(),pwWeightBuf:(()=>{let e=z(c("conv2d_26/")),t=l(e.byteLength,g);return b(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_26/")),alphaBuf:x(c("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},Pe={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_25/")),t=l(e.byteLength,g);return b(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=l(e.byteLength,g);return b(t,0,e),t})(),pwWeightBuf:(()=>{let e=z(c("conv2d_27/Conv2D1")),t=l(e.byteLength,g);return b(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_27/")),alphaBuf:x(c("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},Ee=z(c("conv2d_28/Conv2D")),de=l(Ee.byteLength,g);b(de,0,Ee);let te=x(c("batch_normalization_28/")),De=x(c("p_re_lu_28/")),We={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_26/")),t=l(e.byteLength,g);return b(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=l(e.byteLength,g);return b(t,0,e),t})(),pwWeightBuf:(()=>{let e=z(c("conv2d_29/")),t=l(e.byteLength,g);return b(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_29/")),alphaBuf:x(c("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},Re={dwWeightBuf:(()=>{let e=C(c("depthwise_conv2d_27/")),t=l(e.byteLength,g);return b(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=l(e.byteLength,g);return b(t,0,e),t})(),pwWeightBuf:(()=>{let e=z(c("conv2d_30/Conv2D1")),t=l(e.byteLength,g);return b(t,0,e),t})(),pwBiasBuf:x(c("batch_normalization_30/")),alphaBuf:x(c("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},He=z(c("classifier_palm_16_NO_PRUNING/Conv2D")),je=l(He.byteLength,g);b(je,0,He);let ot=x(c("classifier_palm_16_NO_PRUNING/BiasAdd")),lt=z(c("regressor_palm_16_NO_PRUNING/Conv2D")),it=l(lt.byteLength,g);b(it,0,lt);let st=x(c("regressor_palm_16_NO_PRUNING/BiasAdd")),Ke=z(c("classifier_palm_8_NO_PRUNING/Conv2D")),ke=l(Ke.byteLength,g);b(ke,0,Ke);let Fe=x(c("classifier_palm_8_NO_PRUNING/BiasAdd")),At=z(c("regressor_palm_8_NO_PRUNING/Conv2D")),kt=l(At.byteLength,g);b(kt,0,At);let Wt=x(c("regressor_palm_8_NO_PRUNING/BiasAdd")),St=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Le=l(36864*3*4,g),Ye=l(St,T),Ht=l(St,T),Tt=l(St,T),mt=l(576*256*4,T),Kt=new Map;function Xt(e){let t=Kt.get(e);return t||(t=l(4,y),b(t,0,new Uint32Array([e])),Kt.set(e,t)),t}let ht=l(144*256*4,T|GPUBufferUsage.COPY_DST),gt=l(576*128*4,T|GPUBufferUsage.COPY_DST),Ve=l(864*4,U),Xe=l(15552*4,U),$e=l(576*2*4,U),Ze=l(576*36*4,U),bt=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Et=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),_t=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),yt=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),wt=n.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function R(e,t){return Math.ceil(e/t)}function q(e){let t=l(e.byteLength,y);return b(t,0,e),t}let Ft=q(new Uint32Array([1,3,32,192,192,96,96])),Qe=ee.map(e=>{let t=e.stride===2?e.inH/2:e.inH,i=t,w=e.stride===2?1:2,S=e.inCh;return{dw:q(new Uint32Array([1,e.inCh,e.inH,e.inH,t,i,e.stride,w])),pw:q(new Uint32Array([1,e.inCh,e.outCh,t,i,S,e.stride,e.inH,e.inH])),outH:t,outW:i}}),Je=(()=>{let e=Ue,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:q(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:q(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),et=(()=>{let e=Pe,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:q(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:q(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Pt=(()=>{let e=We,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:q(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:q(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),It=(()=>{let e=Re,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:q(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:q(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Dt=q(new Uint32Array([1,256,6,6,12,12])),$t=q(new Uint32Array([1,256,12,12,12,12])),Zt=q(new Uint32Array([1,256,12,12,24,24])),Qt=q(new Uint32Array([1,128,24,24,24,24])),Ot=q(new Uint32Array([1,256,256,12,12])),Nt=q(new Uint32Array([1,256,128,24,24])),qt=q(new Uint32Array([1,256,6,12,12])),jt=q(new Uint32Array([1,256,108,12,12])),Yt=q(new Uint32Array([1,128,2,24,24])),m=q(new Uint32Array([1,128,36,24,24])),r=q(new Uint32Array([192,192,192])),d=n.createBindGroup({layout:ft,entries:[{binding:0,resource:wt.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:r}}]}),H=null,F=0,J=0,j=l(32,y);function k(e,t){return H&&F===e&&J===t||(H&&H.destroy(),H=n.createTexture({size:[e,t,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),F=e,J=t),H}let tt=n.createBindGroup({layout:rt,entries:[{binding:0,resource:{buffer:Le}},{binding:1,resource:{buffer:ie}},{binding:2,resource:{buffer:Q}},{binding:3,resource:{buffer:le}},{binding:4,resource:{buffer:Ye}},{binding:5,resource:{buffer:Ft}}]});function Oe(e,t,i){}function ne(e,t,i,w,S,h){let X=h.outH,f=n.createBindGroup({layout:Ct,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:t.dwWeightBuf}},{binding:2,resource:{buffer:t.dwBiasBuf}},{binding:3,resource:{buffer:Tt}},{binding:4,resource:{buffer:h.dw}}]}),he=e.beginComputePass();he.setPipeline(Mt),he.setBindGroup(0,f),he.dispatchWorkgroups(R(X,8),R(h.outH,8),t.inCh),he.end(),t.inCh*h.outH*X;let pe=n.createBindGroup({layout:vt,entries:[{binding:0,resource:{buffer:Tt}},{binding:1,resource:{buffer:S}},{binding:2,resource:{buffer:t.pwWeightBuf}},{binding:3,resource:{buffer:t.pwBiasBuf}},{binding:4,resource:{buffer:t.alphaBuf}},{binding:5,resource:{buffer:w}},{binding:6,resource:{buffer:h.pw}}]}),p=e.beginComputePass();p.setPipeline(Lt),p.setBindGroup(0,pe),p.dispatchWorkgroups(R(X,8),R(h.outH,8),t.outCh),p.end(),t.outCh*h.outH*X}function V(e,t,i,w,S,h,X,f,he){let pe=n.createBindGroup({layout:Ut,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:w}},{binding:3,resource:{buffer:S}},{binding:4,resource:{buffer:h}}]}),p=e.beginComputePass();p.setPipeline(_),p.setBindGroup(0,pe),p.dispatchWorkgroups(R(he,8),R(f,8),X),p.end()}function se(e,t,i,w,S,h,X,f,he,pe){let p=n.createBindGroup({layout:qe,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:w}},{binding:3,resource:{buffer:S}},{binding:4,resource:{buffer:h}},{binding:5,resource:{buffer:X}}]}),G=e.beginComputePass();G.setPipeline(W),G.setBindGroup(0,p),G.dispatchWorkgroups(R(pe,8),R(he,8),f),G.end()}async function Ie(e){36864*3;{let P=e.beginComputePass();P.setPipeline(ze),P.setBindGroup(0,tt),P.dispatchWorkgroups(R(96,8),R(96,8),32),P.end()}9216*32;let t=Ye,i=Ht;for(let P=0;P<ee.length;P++){let K=ee[P];ne(e,K,t,i,t,Qe[P]);let ge=t;t=i,i=ge,P===13&&e.copyBufferToBuffer(t,0,gt,0,576*128*4),P===18&&e.copyBufferToBuffer(t,0,ht,0,144*256*4)}{let P=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:mt}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Dt}}]}),K=e.beginComputePass();K.setPipeline(M),K.setBindGroup(0,P),K.dispatchWorkgroups(R(12,8),R(12,8),256),K.end()}{let P=t;t=i,i=P}144*256,se(e,t,I,me,ce,i,Ot,256,12,12);{let P=t;t=i,i=P}144*256;{let P=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:ht}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:$t}}]}),K=e.beginComputePass();K.setPipeline(M),K.setBindGroup(0,P),K.dispatchWorkgroups(R(12,8),R(12,8),256),K.end()}{let P=t;t=i,i=P}144*256,ne(e,Ue,t,i,t,Je);{let P=t;t=i,i=P}ne(e,Pe,t,i,t,et);{let P=t;t=i,i=P}V(e,t,je,ot,Ve,qt,6,12,12),V(e,t,it,st,Xe,jt,108,12,12);{let P=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:mt}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Zt}}]}),K=e.beginComputePass();K.setPipeline(M),K.setBindGroup(0,P),K.dispatchWorkgroups(R(24,8),R(24,8),256),K.end()}{let P=t;t=i,i=P}576*256,se(e,t,de,te,De,i,Nt,128,24,24);{let P=t;t=i,i=P}576*128;{let P=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:gt}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Qt}}]}),K=e.beginComputePass();K.setPipeline(M),K.setBindGroup(0,P),K.dispatchWorkgroups(R(24,8),R(24,8),128),K.end()}{let P=t;t=i,i=P}576*128,ne(e,We,t,i,t,Pt);{let P=t;t=i,i=P}ne(e,Re,t,i,t,It);{let P=t;t=i,i=P}V(e,t,ke,Fe,$e,Yt,2,24,24),576*2,V(e,t,kt,Wt,Ze,m,36,24,24),576*36,n.queue.submit([e.finish()]);let w=n.createCommandEncoder();w.copyBufferToBuffer(Ve,0,bt,0,864*4),w.copyBufferToBuffer(Xe,0,Et,0,15552*4),w.copyBufferToBuffer($e,0,_t,0,576*2*4),w.copyBufferToBuffer(Ze,0,yt,0,576*36*4),n.queue.submit([w.finish()]),await Promise.all([bt.mapAsync(GPUMapMode.READ),Et.mapAsync(GPUMapMode.READ),_t.mapAsync(GPUMapMode.READ),yt.mapAsync(GPUMapMode.READ)]);let S=new Float32Array(bt.getMappedRange()).slice(),h=new Float32Array(Et.getMappedRange()).slice(),X=new Float32Array(_t.getMappedRange()).slice(),f=new Float32Array(yt.getMappedRange()).slice();bt.unmap(),Et.unmap(),_t.unmap(),yt.unmap();let he=2016,pe=new Float32Array(he),p=new Float32Array(he*18),G=0;for(let P=0;P<12;P++)for(let K=0;K<12;K++)for(let ge=0;ge<6;ge++){pe[G]=S[ge*144+P*12+K];for(let Se=0;Se<18;Se++){let o=ge*18+Se;p[G*18+Se]=h[o*144+P*12+K]}G++}for(let P=0;P<24;P++)for(let K=0;K<24;K++)for(let ge=0;ge<2;ge++){pe[G]=X[ge*576+P*24+K];for(let Se=0;Se<18;Se++){let o=ge*18+Se;p[G*18+Se]=f[o*576+P*24+K]}G++}return{scores:pe,regressors:p}}async function Vt(e){n.queue.copyExternalImageToTexture({source:e},{texture:wt},[192,192]);let t=n.createCommandEncoder();{let i=t.beginComputePass();i.setPipeline(N),i.setBindGroup(0,d),i.dispatchWorkgroups(R(192,16),R(192,16),1),i.end()}return Ie(t)}async function Jt(e,t,i){let w=Math.min(192/t,192/i),S=Math.round(t*w),h=Math.round(i*w),X=Math.floor((192-S)/2),f=Math.floor((192-h)/2),he=X/192,pe=f/192,p=k(t,i),G;e instanceof HTMLVideoElement?G=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?G=await createImageBitmap(e,{colorSpaceConversion:"none"}):G=e,n.queue.copyExternalImageToTexture({source:G},{texture:p},[t,i]);let P=new ArrayBuffer(32),K=new Uint32Array(P),ge=new Float32Array(P);K[0]=t,K[1]=i,K[2]=192,K[3]=0,ge[4]=t/S,ge[5]=i/h,ge[6]=X,ge[7]=f,n.queue.writeBuffer(j,0,P);let Se=n.createBindGroup({layout:ve,entries:[{binding:0,resource:p.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:j}},{binding:3,resource:a}]}),o=n.createCommandEncoder();{let E=o.beginComputePass();E.setPipeline(oe),E.setBindGroup(0,Se),E.dispatchWorkgroups(R(192,16),R(192,16),1),E.end()}return{output:await Ie(o),lbPadX:he,lbPadY:pe}}async function L(e,t){let i=n.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),w=n.createCommandEncoder();w.copyBufferToBuffer(e,0,i,0,t*4),n.queue.submit([w.finish()]),await i.mapAsync(GPUMapMode.READ);let S=new Float32Array(i.getMappedRange()).slice();return i.unmap(),i.destroy(),S}async function Bn(e,t,i){function w(o,u=1e3){let E=o.slice(0,u),fe=Math.max(0,Math.floor(o.length/2)-250);return{min:Math.min(...E),max:Math.max(...E),mean:E.reduce((xe,ut)=>xe+ut,0)/E.length,nonZero:E.filter(xe=>xe!==0).length,sample:Array.from(E.slice(0,10)),data500:Array.from(o.slice(0,500)),dataMid500:Array.from(o.slice(fe,fe+500)),totalLength:o.length}}function S(o,u,E,fe){let xe=[],ut=Math.floor(E/2),ct=Math.floor(fe/2),Ne=E*fe;for(let dt=0;dt<u&&xe.length<500;dt++)for(let zt=-1;zt<=1&&xe.length<500;zt++)for(let Rt=-1;Rt<=1&&xe.length<500;Rt++){let pt=ut+zt,en=ct+Rt;pt>=0&&pt<E&&en>=0&&en<fe&&xe.push(o[dt*Ne+pt*fe+en])}return xe}let h={},X;if(e instanceof HTMLImageElement?(t=t??e.naturalWidth,i=i??e.naturalHeight,X=await createImageBitmap(e,{colorSpaceConversion:"none"})):(t=t??e.width??192,i=i??e.height??192,X=e),t!==192||i!==192){let o=Math.min(192/t,192/i),u=Math.round(t*o),E=Math.round(i*o),fe=Math.floor((192-u)/2),xe=Math.floor((192-E)/2),ut=k(t,i);n.queue.copyExternalImageToTexture({source:X},{texture:ut},[t,i]);let ct=new ArrayBuffer(32),Ne=new Uint32Array(ct),dt=new Float32Array(ct);Ne[0]=t,Ne[1]=i,Ne[2]=192,Ne[3]=0,dt[4]=t/u,dt[5]=i/E,dt[6]=fe,dt[7]=xe,n.queue.writeBuffer(j,0,ct);let zt=n.createBindGroup({layout:ve,entries:[{binding:0,resource:ut.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:j}},{binding:3,resource:a}]});{let Rt=n.createCommandEncoder(),pt=Rt.beginComputePass();pt.setPipeline(oe),pt.setBindGroup(0,zt),pt.dispatchWorkgroups(R(192,16),R(192,16),1),pt.end(),n.queue.submit([Rt.finish()])}}else{n.queue.copyExternalImageToTexture({source:X},{texture:wt},[192,192]);let o=q(new Uint32Array([192,192,192])),u=n.createBindGroup({layout:ft,entries:[{binding:0,resource:wt.createView()},{binding:1,resource:{buffer:Le}},{binding:2,resource:{buffer:o}}]});{let E=n.createCommandEncoder(),fe=E.beginComputePass();fe.setPipeline(N),fe.setBindGroup(0,u),fe.dispatchWorkgroups(R(192,16),R(192,16),1),fe.end(),n.queue.submit([E.finish()])}}{let o=await L(Le,110592),u=w(o);u.dataCenter500=S(o,3,192,192),h.input=u}let f=n.createCommandEncoder(),he=n.createBindGroup({layout:rt,entries:[{binding:0,resource:{buffer:Le}},{binding:1,resource:{buffer:ie}},{binding:2,resource:{buffer:Q}},{binding:3,resource:{buffer:le}},{binding:4,resource:{buffer:Ye}},{binding:5,resource:{buffer:Ft}}]}),pe=f.beginComputePass();pe.setPipeline(ze),pe.setBindGroup(0,he),pe.dispatchWorkgroups(R(96,8),R(96,8),32),pe.end(),n.queue.submit([f.finish()]);{let o=await L(Ye,294912),u=w(o);u.dataCenter500=S(o,32,96,96),h.initConv=u}let p=Ye,G=Ht;for(let o=0;o<ee.length;o++){let u=ee[o];f=n.createCommandEncoder(),ne(f,u,p,G,p,Qe[o]),n.queue.submit([f.finish()]);let E=p;p=G,G=E;{let fe=u.stride===2?u.inH/2:u.inH,xe=fe,ut=fe*xe*u.outCh,ct=await L(p,ut),Ne=w(ct);Ne.dataCenter500=S(ct,u.outCh,fe,xe),Ne.spatialShape=[u.outCh,fe,xe],h[`block${o}`]=Ne}o===13&&(f=n.createCommandEncoder(),f.copyBufferToBuffer(p,0,gt,0,576*128*4),n.queue.submit([f.finish()])),o===18&&(f=n.createCommandEncoder(),f.copyBufferToBuffer(p,0,ht,0,144*256*4),n.queue.submit([f.finish()]))}f=n.createCommandEncoder();{let o=q(new Uint32Array([1,256,6,6,12,12])),u=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:mt}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:o}}]}),E=f.beginComputePass();E.setPipeline(M),E.setBindGroup(0,u),E.dispatchWorkgroups(R(12,8),R(12,8),256),E.end()}n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,36864),u=w(o);u.dataCenter500=S(o,256,12,12),h.fpnUpsample6to12=u}f=n.createCommandEncoder(),se(f,p,I,me,ce,G,Ot,256,12,12),n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,36864),u=w(o);u.dataCenter500=S(o,256,12,12),h.fpn6to12Conv=u}{let o=await L(ht,36864),u=w(o);u.dataCenter500=S(o,256,12,12),h.backbone12Skip=u}f=n.createCommandEncoder();{let o=q(new Uint32Array([1,256,12,12,12,12])),u=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:ht}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:o}}]}),E=f.beginComputePass();E.setPipeline(M),E.setBindGroup(0,u),E.dispatchWorkgroups(R(12,8),R(12,8),256),E.end()}n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,36864),u=w(o);u.dataCenter500=S(o,256,12,12),h.fpnAdd12=u}f=n.createCommandEncoder(),ne(f,Ue,p,G,p,Je),n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,36864),u=w(o);u.dataCenter500=S(o,256,12,12),h.fpn12Block1=u}f=n.createCommandEncoder(),ne(f,Pe,p,G,p,et),n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,36864),u=w(o);u.dataCenter500=S(o,256,12,12),h.fpn12Block2=u}f=n.createCommandEncoder(),V(f,p,je,ot,Ve,qt,6,12,12),n.queue.submit([f.finish()]);{let o=await L(Ve,864),u=w(o);u.dataCenter500=S(o,6,12,12),h.cls16=u}f=n.createCommandEncoder(),V(f,p,it,st,Xe,jt,108,12,12),n.queue.submit([f.finish()]);{let o=await L(Xe,15552),u=w(o,500);u.dataCenter500=S(o,108,12,12),h.reg16=u}f=n.createCommandEncoder();{let o=q(new Uint32Array([1,256,12,12,24,24])),u=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:mt}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:o}}]}),E=f.beginComputePass();E.setPipeline(M),E.setBindGroup(0,u),E.dispatchWorkgroups(R(24,8),R(24,8),256),E.end()}n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,147456),u=w(o);u.dataCenter500=S(o,256,24,24),h.fpnUpsample12to24=u}f=n.createCommandEncoder(),se(f,p,de,te,De,G,Nt,128,24,24),n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,73728),u=w(o);u.dataCenter500=S(o,128,24,24),h.fpn12to24Conv=u}{let o=await L(gt,73728),u=w(o);u.dataCenter500=S(o,128,24,24),h.backbone24Skip=u}f=n.createCommandEncoder();{let o=q(new Uint32Array([1,128,24,24,24,24])),u=n.createBindGroup({layout:we,entries:[{binding:0,resource:{buffer:p}},{binding:1,resource:{buffer:gt}},{binding:2,resource:{buffer:G}},{binding:3,resource:{buffer:o}}]}),E=f.beginComputePass();E.setPipeline(M),E.setBindGroup(0,u),E.dispatchWorkgroups(R(24,8),R(24,8),128),E.end()}n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,73728),u=w(o);u.dataCenter500=S(o,128,24,24),h.fpnAdd24=u}f=n.createCommandEncoder(),ne(f,We,p,G,p,Pt),n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,73728),u=w(o);u.dataCenter500=S(o,128,24,24),h.fpn24Block1=u}f=n.createCommandEncoder(),ne(f,Re,p,G,p,It),n.queue.submit([f.finish()]);{let o=p;p=G,G=o}{let o=await L(p,73728),u=w(o);u.dataCenter500=S(o,128,24,24),h.fpn24Block2=u}f=n.createCommandEncoder(),V(f,p,ke,Fe,$e,Yt,2,24,24),n.queue.submit([f.finish()]);{let o=await L($e,1152),u=w(o);u.dataCenter500=S(o,2,24,24),h.cls8=u}f=n.createCommandEncoder(),V(f,p,kt,Wt,Ze,m,36,24,24),n.queue.submit([f.finish()]);{let o=await L(Ze,20736),u=w(o);u.dataCenter500=S(o,36,24,24),h.reg8=u}h.initWeights=w(await L(ie,100),100),h.initBias=w(await L(Q,32),32),h.cls16Weights=w(await L(je,100),100),h.cls16Bias=w(await L(ot,6),6),h.cls8Weights=w(await L(ke,100),100),h.cls8Bias=w(await L(Fe,2),2),h.fpn6to12Weights=w(await L(I,100),100);let P=await L(Ve,864),K=await L($e,576*2);h.rawScores=new Float32Array(2016),h.rawScores.set(P,0),h.rawScores.set(K,864);let ge=await L(Xe,15552),Se=await L(Ze,576*36);return h.rawRegressors=new Float32Array(36288),h.rawRegressors.set(ge,0),h.rawRegressors.set(Se,15552),h.rawInput=await L(Le,36864*3),h}return{device:n,run:Vt,runWithResize:Jt,debugRun:Bn}}function Ln(){let s=[];for(let B=0;B<12;B++)for(let n=0;n<12;n++){let D=(n+.5)/12,v=(B+.5)/12;for(let a=0;a<6;a++)s.push({x:D,y:v})}for(let B=0;B<24;B++)for(let n=0;n<24;n++){let D=(n+.5)/24,v=(B+.5)/24;for(let a=0;a<2;a++)s.push({x:D,y:v})}return s}var yn=Ln();function Wn(s){return 1/(1+Math.exp(-s))}function tn(s,B){let n=[],{scores:D,regressors:v}=s,a=192;for(let g=0;g<yn.length;g++){let T=Wn(D[g]);if(T<B)continue;let U=yn[g],y=g*18,l=U.x+v[y+0]/a,b=U.y+v[y+1]/a,x=v[y+2]/a,Y=v[y+3]/a,O=[];for(let c=0;c<7;c++){let C=U.x+v[y+4+c*2]/a,z=U.y+v[y+4+c*2+1]/a;O.push([C,z])}n.push({score:T,box:[l,b,x,Y],keypoints:O})}return n}function nn(s,B){if(s.length===0)return[];let n=[...s].sort((a,g)=>g.score-a.score),D=[],v=new Set;for(let a=0;a<n.length;a++){if(v.has(a))continue;let g=[a];for(let O=a+1;O<n.length;O++)v.has(O)||Hn(n[a],n[O])>B&&(g.push(O),v.add(O));let T=0,U=0,y=0,l=0,b=0,x=[];for(let O=0;O<7;O++)x.push([0,0]);for(let O of g){let c=n[O],C=c.score;T+=C,U+=c.box[0]*C,y+=c.box[1]*C,l+=c.box[2]*C,b+=c.box[3]*C;for(let z=0;z<7;z++)x[z][0]+=c.keypoints[z][0]*C,x[z][1]+=c.keypoints[z][1]*C}let Y=1/T;D.push({score:n[a].score,box:[U*Y,y*Y,l*Y,b*Y],keypoints:x.map(([O,c])=>[O*Y,c*Y])})}return D}function Hn(s,B){let n=s.box[0]-s.box[2]/2,D=s.box[1]-s.box[3]/2,v=s.box[0]+s.box[2]/2,a=s.box[1]+s.box[3]/2,g=B.box[0]-B.box[2]/2,T=B.box[1]-B.box[3]/2,U=B.box[0]+B.box[2]/2,y=B.box[1]+B.box[3]/2,l=Math.max(n,g),b=Math.max(D,T),x=Math.min(v,U),Y=Math.min(a,y),O=Math.max(0,x-l),c=Math.max(0,Y-b),C=O*c,z=(v-n)*(a-D),Ce=(U-g)*(y-T),Ge=z+Ce-C;return Ge>0?C/Ge:0}function Kn(s){let[B,n,D,v]=s.box,a=s.keypoints[0],g=s.keypoints[2],T=g[0]-a[0],U=g[1]-a[1],y=Math.atan2(U,T),b=-Math.PI/2-y,x=Math.max(D,v),O=x*2.6,c=-.5*x,C=Math.cos(b),z=Math.sin(b),Ce=c*z,Ge=c*C;return{centerX:B+Ce,centerY:n+Ge,width:O,height:O,rotation:b}}function wn(s,B={}){let{scoreThreshold:n=.5,nmsThreshold:D=.3,maxHands:v=2}=B;async function a(y){let l=await s.run(y),b=tn(l,n);return nn(b,D).slice(0,v).map(Kn)}async function g(y){let l=await s.run(y),b=tn(l,n);return nn(b,D).slice(0,v)}async function T(y,l,b){let{output:x,lbPadX:Y,lbPadY:O}=await s.runWithResize(y,l,b),c=tn(x,n);return{detections:nn(c,D).slice(0,v),lbPadX:Y,lbPadY:O}}async function U(y,l,b){let{output:x,lbPadX:Y,lbPadY:O}=await s.runWithResize(y,l,b);return{scores:x.scores,regressors:x.regressors,lbPadX:Y,lbPadY:O}}return{detect:a,detectRaw:g,detectRawWithResize:T,detectRawSSD:U,model:s}}var an=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function rn(s){let B={};for(let n=0;n<an.length;n++)B[an[n]]=s[n];return B}function On(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var zn=On(`
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
`);function Pn(s){let B=s.createShaderModule({code:zn}),n=s.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),D=s.createComputePipeline({layout:s.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:B,entryPoint:"main"}}),v=s.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),a=s.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),g=s.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),T=new Float32Array(8);function U(y,l,b,x,Y,O,c){s.queue.writeBuffer(a,0,new Uint32Array([Y,O,c,0])),T.set(x),s.queue.writeBuffer(g,0,T);let C=s.createBindGroup({layout:n,entries:[{binding:0,resource:l.createView()},{binding:1,resource:{buffer:b}},{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:g}},{binding:4,resource:v}]}),z=y.beginComputePass();z.setPipeline(D),z.setBindGroup(0,C),z.dispatchWorkgroups(Math.ceil(c/16),Math.ceil(c/16),1),z.end()}return{crop:U}}var Fn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@0.3.0/weights";async function In(s={}){let{weightsUrl:B,scoreThreshold:n=.5,palmScoreThreshold:D=.5,maxHands:v=3}=s;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let a=(B??Fn).replace(/\/$/,"")+"/",[g,T,U,y]=await Promise.all([fetch(`${a}weights_f16_full.json`),fetch(`${a}weights_f16_full.bin`),fetch(`${a}palm_detection_weights.json`),fetch(`${a}palm_detection_weights.bin`)]);if(!g.ok)throw new Error(`Failed to fetch landmark weights: ${g.status}`);if(!T.ok)throw new Error(`Failed to fetch landmark weights: ${T.status}`);if(!U.ok)throw new Error(`Failed to fetch palm detection weights: ${U.status}`);if(!y.ok)throw new Error(`Failed to fetch palm detection weights: ${y.status}`);let[l,b,x,Y]=await Promise.all([g.json(),T.arrayBuffer(),U.json(),y.arrayBuffer()]),O=sn(l,b),c=on(x,Y),C=224,z=await un(O);{let _=new OffscreenCanvas(C,C),W=_.getContext("2d");W.fillStyle="#886644",W.fillRect(0,0,C,C),W.fillStyle="#cc9966",W.fillRect(50,50,124,124);let M=await z.runFromCanvas(_);M.landmarks.every(oe=>oe===0)&&M.handflag.every(oe=>oe===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let Ce=await _n(c),Ge=wn(Ce,{scoreThreshold:D,maxHands:v}),re=[];function A(_,W,M){let N=_[0],oe=_[5],Me=_[9],be=_[13],_e=N.x*W,Ae=N.y*M,ie=(oe.x+be.x)/2,Q=(oe.y+be.y)/2;ie=(ie+Me.x)/2*W,Q=(Q+Me.y)/2*M;let le=Math.PI/2-Math.atan2(-(Q-Ae),ie-_e),Z=le-2*Math.PI*Math.floor((le+Math.PI)/(2*Math.PI)),ee=[0,1,2,3,5,6,9,10,13,14,17,18],ae=Math.cos(Z),I=Math.sin(Z),me=1/0,ce=-1/0,Ue=1/0,Pe=-1/0;for(let it of ee){let st=_[it],Ke=st.x*W,ke=st.y*M,Fe=ae*Ke+I*ke,At=-I*Ke+ae*ke;me=Math.min(me,Fe),ce=Math.max(ce,Fe),Ue=Math.min(Ue,At),Pe=Math.max(Pe,At)}let Ee=(me+ce)/2,de=(Ue+Pe)/2,te=ce-me,De=Pe-Ue,We=(ae*Ee-I*de)/W,Re=(I*Ee+ae*de)/M;te/=W,De/=M;let He=-.1;We+=-M*De*He*I/W,Re+=De*He*ae;let lt=Math.max(te*W,De*M)*2;return{centerXpx:We*W,centerYpx:Re*M,sizePx:lt,rotation:Z}}let ue=z.device,ye=null,$=null,Te=null,xt=0,rt=0;function Ct(){return ye||(ye=Pn(ue)),ye}function vt(){return $||($=ue.createBuffer({size:3*C*C*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),$}function Ut(_,W){return(!Te||xt!==_||rt!==W)&&(Te&&Te.destroy(),Te=ue.createTexture({size:[_,W],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),xt=_,rt=W),Te}let qe=0,we=0;function ft(_){let W=1/(1-2*qe),M=1/(1-2*we);return{score:_.score,box:[(_.box[0]-qe)*W,(_.box[1]-we)*M,_.box[2]*W,_.box[3]*M],keypoints:_.keypoints.map(([N,oe])=>[(N-qe)*W,(oe-we)*M])}}function ve(_,W,M){let N=_.keypoints[0],oe=_.keypoints[2],Me=(oe[0]-N[0])*W,be=(oe[1]-N[1])*M,_e=Math.atan2(-be,Me),ie=Math.PI/2-_e,Q=ie-2*Math.PI*Math.floor((ie+Math.PI)/(2*Math.PI)),[le,Z,ee,ae]=_.box,I=Math.cos(Q),me=Math.sin(Q),ce=ae*M,Ue=le+.5*ce*me/W,Pe=Z+-.5*ae*I,te=Math.max(ee*W,ae*M)*2.6;return{centerXpx:Ue*W,centerYpx:Pe*M,sizePx:te,rotation:Q}}function Gt(_){return _ instanceof HTMLCanvasElement||_ instanceof OffscreenCanvas?[_.width,_.height]:typeof ImageBitmap<"u"&&_ instanceof ImageBitmap?[_.width,_.height]:_ instanceof ImageData?[_.width,_.height]:_ instanceof HTMLVideoElement?[_.videoWidth,_.videoHeight]:_ instanceof HTMLImageElement?[_.naturalWidth,_.naturalHeight]:[C,C]}async function Be(_,W,M,N,oe,Me){let be=Math.cos(_.rotation),_e=Math.sin(_.rotation),Ae=_.sizePx/C,ie=C/2,Q=be*Ae/M,le=-_e*Ae/M,Z=_.centerXpx/M-ie*(Q+le),ee=_e*Ae/N,ae=be*Ae/N,I=_.centerYpx/N-ie*(ee+ae),me=ue.createCommandEncoder();oe.crop(me,W,Me,[Q,le,Z,ee,ae,I],M,N,C),ue.queue.submit([me.finish()]);let ce=await z.runFromGPUBuffer(Me),Ue=ce.handflag[0];if(Ue<n)return null;let Pe=ce.handedness[0]>.5,Ee=[];for(let de=0;de<21;de++){let te=ce.landmarks[de*3],De=ce.landmarks[de*3+1],We=ce.landmarks[de*3+2],Re=(te-.5)*_.sizePx,He=(De-.5)*_.sizePx,je=be*Re-_e*He+_.centerXpx,ot=_e*Re+be*He+_.centerYpx;Ee.push({x:je/M,y:ot/N,z:We})}return{landmarks:Ee,score:Ue,handedness:Pe?"right":"left"}}async function ze(_){let W=_,M,N;if(_ instanceof HTMLVideoElement||_ instanceof HTMLImageElement){let Z=await createImageBitmap(_,{colorSpaceConversion:"none"});W=Z,M=Z.width,N=Z.height}else[M,N]=Gt(_);let oe=Ct(),Me=vt(),be=Ut(M,N),_e;if(W instanceof ImageData?_e=await createImageBitmap(W,{colorSpaceConversion:"none"}):_e=W,ue.queue.copyExternalImageToTexture({source:_e},{texture:be},[M,N]),re.length>0){let Z=[];for(let ee of re){let ae=A(ee.landmarks,M,N),I=await Be(ae,be,M,N,oe,Me);I&&Z.push({score:I.score,handedness:I.handedness,landmarks:I.landmarks,keypoints:rn(I.landmarks)})}if(Z.length>0)return re=Z.map(ee=>({landmarks:ee.landmarks,handedness:ee.handedness})),Z;re=[]}let{detections:Ae,lbPadX:ie,lbPadY:Q}=await Ge.detectRawWithResize(W,M,N);if(qe=ie,we=Q,Ae.length===0)return re=[],[];let le=[];for(let Z of Ae){let ee=ft(Z),ae=ve(ee,M,N),I=await Be(ae,be,M,N,oe,Me);I&&le.push({score:I.score,handedness:I.handedness,landmarks:I.landmarks,keypoints:rn(I.landmarks)})}return re=le.map(Z=>({landmarks:Z.landmarks,handedness:Z.handedness})),le}function Mt(){Te&&Te.destroy(),$&&$.destroy(),Te=null,$=null,ye=null,z.device.destroy(),Ce.device.destroy()}function Lt(){re=[]}return{detect:ze,dispose:Mt,reset:Lt}}export{an as LANDMARK_NAMES,In as createHandpose};
