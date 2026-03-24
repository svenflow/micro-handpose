function cn(c,B){let n=new Map,R=c.dtype??"float32";for(let G=0;G<c.keys.length;G++){let r=c.keys[G],y=c.shapes[G],D=c.offsets[G],M=y.reduce((w,x)=>w*x,1),P,h;if(R==="float32")P=new Float32Array(B,D,M);else{let w=new DataView(B);P=new Float32Array(M);for(let x=0;x<M;x++)P[x]=vn(w.getUint16(D+x*2,!0));h=B.slice(D,D+M*2)}n.set(r,{data:P,shape:y,rawF16:h})}return n}function vn(c){let B=c>>15&1,n=c>>10&31,R=c&1023;if(n===0){if(R===0)return B?-0:0;let y=-14,D=R/1024;return(B?-1:1)*Math.pow(2,y)*D}if(n===31)return R===0?B?-1/0:1/0:NaN;let G=n-15,r=1+R/1024;return(B?-1:1)*Math.pow(2,G)*r}var Un=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Yn=Un.map(([c,B,n,R,G])=>({type:"resmodule",inCh:c,outCh:B,h:n,w:n,stride:R,prefix:G}));function dn(c,B){let n=new Map,R=c.dtype??"float32",G=new Map;for(let r=0;r<c.keys.length;r++){let y=c.keys[r],D=c.shapes[r],M=c.offsets[r],P=D.reduce((K,d)=>K*d,1),h,w;if(R==="float32")h=new Float32Array(B,M,P);else{let K=new DataView(B);h=new Float32Array(P);for(let d=0;d<P;d++)h[d]=Gn(K.getUint16(M+d*2,!0));w=B.slice(M,M+P*2)}let x=G.get(y)??0;G.set(y,x+1);let Q=x===0?y:`${y}__${x}`;n.set(Q,{data:h,shape:D,rawF16:w})}return n}function Gn(c){let B=c>>15&1,n=c>>10&31,R=c&1023;return n===0?R===0?B?-0:0:(B?-1:1)*Math.pow(2,-14)*(R/1024):n===31?R===0?B?-1/0:1/0:NaN:(B?-1:1)*Math.pow(2,n-15)*(1+R/1024)}function tt(c){return c.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Mn=tt(`
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
`),An=tt(`
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
`),kn=tt(`
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
`),Sn=tt(`
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
`),Tn=tt(`
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
`),En=tt(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Dn=tt(`
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
`),Rn=tt(`
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
`),Ln=tt(`
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
`),Ct=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],Hn=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function pn(c,B){if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");let R=n.features.has("shader-f16"),G=R?["shader-f16"]:[],r=await n.requestDevice({requiredFeatures:G,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}}),y=c.values().next().value,D=R&&!!y?.rawF16&&!B?.forceF32;function M(f){if(D&&f.rawF16){let a=new Uint16Array(f.rawF16);if(a.length%2!==0){let u=new Uint16Array(a.length+1);return u.set(a),u}return a}return f.data}function P(f){return D&&f.rawF16?Math.ceil(f.rawF16.byteLength/4)*4:f.data.byteLength}let h=D?2:4;function w(f){if(!D)return f;let a=f;return a=a.replace(/array<f32>/g,"array<f16>"),a=a.replace(/array<f32,/g,"array<f16,"),a=a.replace(/var sum:f32=0\.0/g,"var sum:f16=0.0h"),a=a.replace(/var sum0:f32=0\.0/g,"var sum0:f16=0.0h"),a=a.replace(/var sum1:f32=0\.0/g,"var sum1:f16=0.0h"),a=a.replace(/var sum2:f32=0\.0/g,"var sum2:f16=0.0h"),a=a.replace(/var sum3:f32=0\.0/g,"var sum3:f16=0.0h"),a=a.replace(/\/f32\(params/g,"/f16(params"),a=a.replace(/,0\.0\),6\.0\)/g,",0.0h),6.0h)"),a=a.replace(/->f32\{/g,"->f16{"),a=a.replace(/->f32 \{/g,"->f16 {"),a=a.replace(/return 0\.0;/g,"return 0.0h;"),"enable f16;"+a}function x(f){if(!D)return f;let a=w(f);return a=a.replace("read>input:array<f16>","read>input:array<f32>"),a=a.replace(/input\[in_idx\]/g,"f16(input[in_idx])"),a}function Q(f){if(!D)return f;let a=f;return a=a.replace("read>input:array<f32>","read>input:array<f16>"),a=a.replace("read>weight:array<f32>","read>weight:array<f16>"),a=a.replace("read>bias:array<f32>","read>bias:array<f16>"),a=a.replace(/input\[ic\]/g,"f32(input[ic])"),a=a.replace(/weight\[oc\*params\.in_features\+ic\]/g,"f32(weight[oc*params.in_features+ic])"),a=a.replace(/bias\[oc\]/g,"f32(bias[oc])"),"enable f16;"+a}let K={r:"read-only-storage",s:"storage",u:"uniform"};function d(f){return r.createBindGroupLayout({entries:f.map((a,u)=>a==="t"?{binding:u,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:u,visibility:GPUShaderStage.COMPUTE,buffer:{type:K[a]}})})}let U=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,H=GPUBufferUsage.STORAGE,Pe=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,Ue=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,J=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function k(f,a){return r.createBuffer({size:Math.max(f,4),usage:a})}function pe(f,a){return r.createBindGroup({layout:f,entries:a.map((u,v)=>({binding:v,resource:"size"in u?{buffer:u}:u}))})}function ge(f,a){return r.createComputePipeline({layout:r.createPipelineLayout({bindGroupLayouts:[f]}),compute:{module:a,entryPoint:"main"}})}function ae(f){let a=c.get(f);if(!a)throw new Error(`Missing weight: ${f}`);return a}let Me=r.createShaderModule({code:Mn}),vt=r.createShaderModule({code:x(An)}),at=r.createShaderModule({code:w(kn)}),Ut=r.createShaderModule({code:w(Sn)}),Gt=r.createShaderModule({code:w(Tn)}),Mt=r.createShaderModule({code:w(En)}),Fe=r.createShaderModule({code:w(Dn)}),be=r.createShaderModule({code:Q(Rn)}),mt=r.createShaderModule({code:Q(Ln)}),Be=d(["r","r","r","s","u"]),At=d(["r","r","s","u"]),_e=d(["r","s","u"]),Te=d(["r","r","r","s","u"]),ht=d(["t","s","u"]),Re=ge(ht,Me),Ht=ge(Be,vt),Wt=ge(Be,at),Ae=ge(Be,Ut),l=ge(Be,Gt),C=ge(At,Mt),A=ge(_e,Fe),I=ge(Te,be),ee=ge(Te,mt),X=1152*112*112*4,$=k(X,Ue),ce=k(X,Ue),de=k(X,H),N=k(X,H),z=k(X,U),q=k(672*224*4,Ue),Y=k(1152*4,Pe),T=k(252,Pe),fe=k(252,Pe),le=k(4,Pe),xe=k(4,Pe),ye=k(260,Ue),re=k(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ze=r.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),Le=k(4,J);r.queue.writeBuffer(Le,0,new Uint32Array([224]));let Ee=ae("conv2d"),He=ae("batch_normalization"),Ie=M(Ee),rt=M(He),ot=k(P(Ee),U),gt=k(P(He),U),kt=k(24,J);r.queue.writeBuffer(ot,0,Ie),r.queue.writeBuffer(gt,0,rt),r.queue.writeBuffer(kt,0,new Uint32Array([3,24,224,224,112,112]));let We=112,ke=112,Oe=[];for(let f=0;f<Ct.length;f++){let a=Ct[f],u=Hn[f],v=We,F=ke,Z=a.stride===2?Math.floor(We/2):We,V=a.stride===2?Math.floor(ke/2):ke,S={spec:a,inH:v,inW:F,outH:Z,outW:V,dwW:k(4,U),dwB:k(4,U),dwU:k(32,J)};if(u.expandConvKey){let e=ae(u.expandConvKey),t=ae(u.expandBNKey);S.expandW=k(P(e),U),S.expandB=k(P(t),U),S.expandU=k(16,J),r.queue.writeBuffer(S.expandW,0,M(e)),r.queue.writeBuffer(S.expandB,0,M(t)),r.queue.writeBuffer(S.expandU,0,new Uint32Array([a.inCh,a.expandCh,v,F]))}let Ce=ae(u.dwWeightKey),L=ae(u.dwBNKey);S.dwW=k(P(Ce),U),S.dwB=k(P(L),U),r.queue.writeBuffer(S.dwW,0,M(Ce)),r.queue.writeBuffer(S.dwB,0,M(L));let ve=Math.floor((a.dwKernel-a.stride)/2);if(r.queue.writeBuffer(S.dwU,0,new Uint32Array([a.expandCh,v,F,Z,V,a.stride,ve,a.dwKernel])),a.hasProject&&u.projectConvKey){let e=ae(u.projectConvKey),t=ae(u.projectBNKey);S.projectW=k(P(e),U),S.projectB=k(P(t),U),S.projectU=k(16,J),r.queue.writeBuffer(S.projectW,0,M(e)),r.queue.writeBuffer(S.projectB,0,M(t)),r.queue.writeBuffer(S.projectU,0,new Uint32Array([a.expandCh,a.outCh,Z,V]))}Oe.push(S),We=Z,ke=V}function St(f,a){let u=c.get(f);if(!u)throw new Error(`Missing weight: ${f}`);if(u.shape.length!==a)throw new Error(`Weight ${f} has rank ${u.shape.length}, expected ${a}`);return u}let Ne=ae("conv_landmarks__1"),it=ae("conv_world_landmarks__1"),qe=ae("conv_handflag__1"),Se=ae("conv_handedness__1"),Ye=ae("Identity"),Ot=ae("Identity_1"),Tt=ae("Identity_2"),bt=ae("Identity_3"),Kt=k(P(Ne),U),Jt=k(P(Ye),U),_t=k(P(it),U),yt=k(P(bt),U),je=k(P(qe),U),Ve=k(P(Ot),U),Xe=k(P(Se),U),$e=k(P(Tt),U);r.queue.writeBuffer(Kt,0,M(Ne)),r.queue.writeBuffer(Jt,0,M(Ye)),r.queue.writeBuffer(_t,0,M(it)),r.queue.writeBuffer(yt,0,M(bt)),r.queue.writeBuffer(je,0,M(qe)),r.queue.writeBuffer(Ve,0,M(Ot)),r.queue.writeBuffer(Xe,0,M(Se)),r.queue.writeBuffer($e,0,M(Tt));let wt=k(8,J),Et=k(8,J),Pt=k(8,J),Bt=k(8,J);r.queue.writeBuffer(wt,0,new Uint32Array([1152,63])),r.queue.writeBuffer(Et,0,new Uint32Array([1152,63])),r.queue.writeBuffer(Pt,0,new Uint32Array([1152,1])),r.queue.writeBuffer(Bt,0,new Uint32Array([1152,1]));let xt=k(8,J);r.queue.writeBuffer(xt,0,new Uint32Array([1152,We*ke]));let W=new Map;for(let f=0;f<Ct.length;f++)if(Ct[f].hasResidual){let a=Oe[f],u=k(4,J);r.queue.writeBuffer(u,0,new Uint32Array([Ct[f].outCh*a.outH*a.outW])),W.set(f,u)}let j=pe(ht,[ze.createView(),$,Le]),It=pe(Be,[$,ot,gt,ce,kt]),Ze=new Float32Array(1),Qe=new Float32Array(1),Je=new Float32Array(63);function st(f,a){let u=f.beginComputePass();u.setPipeline(Ht),u.setBindGroup(0,It),u.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),u.end();let v=ce,F=$;for(let Z=0;Z<Ct.length;Z++){let V=Ct[Z],S=Oe[Z];if(V.hasResidual){let ve=V.inCh*S.inH*S.inW*h;f.copyBufferToBuffer(v,0,z,0,ve)}if(u=f.beginComputePass(),S.expandW){let ve=pe(Be,[v,S.expandW,S.expandB,de,S.expandU]);u.setPipeline(Wt),u.setBindGroup(0,ve),u.dispatchWorkgroups(Math.ceil(S.inW/8),Math.ceil(S.inH/8),V.expandCh)}let Ce=S.expandW?de:v,L=pe(Be,[Ce,S.dwW,S.dwB,N,S.dwU]);if(u.setPipeline(Ae),u.setBindGroup(0,L),u.dispatchWorkgroups(Math.ceil(S.outW/8),Math.ceil(S.outH/8),V.expandCh),V.hasProject&&S.projectW){let ve=(V.hasResidual,F),e=pe(Be,[N,S.projectW,S.projectB,ve,S.projectU]);if(u.setPipeline(l),u.setBindGroup(0,e),u.dispatchWorkgroups(Math.ceil(S.outW/8),Math.ceil(S.outH/8),V.outCh),V.hasResidual){let t=W.get(Z),s=pe(At,[F,z,v,t]);u.setPipeline(C),u.setBindGroup(0,s),u.dispatchWorkgroups(Math.ceil(V.outCh*S.outH*S.outW/256))}else{let t=v;v=F,F=t}}if(u.end(),!V.hasProject){u=f.beginComputePass();let ve=pe(_e,[N,Y,xt]);u.setPipeline(A),u.setBindGroup(0,ve),u.dispatchWorkgroups(Math.ceil(1152/256));let e=pe(Te,[Y,Kt,Jt,T,wt]);u.setPipeline(I),u.setBindGroup(0,e),u.dispatchWorkgroups(1);let t=pe(Te,[Y,je,Ve,le,Pt]);u.setPipeline(ee),u.setBindGroup(0,t),u.dispatchWorkgroups(1);let s=pe(Te,[Y,Xe,$e,xe,Bt]);u.setPipeline(ee),u.setBindGroup(0,s),u.dispatchWorkgroups(1),u.end(),f.copyBufferToBuffer(le,0,ye,0,4),f.copyBufferToBuffer(xe,0,ye,4,4),f.copyBufferToBuffer(T,0,ye,8,252),a&&f.copyBufferToBuffer(ye,0,a,0,260);return}}}async function Nt(f){r.queue.writeBuffer(q,0,f);let a=r.createCommandEncoder();a.copyBufferToBuffer(q,0,$,0,672*224*4),st(a,re),r.queue.submit([a.finish()]);let u=re.mapAsync(GPUMapMode.READ);await r.queue.onSubmittedWorkDone(),await u;let v=new Float32Array(re.getMappedRange());Ze[0]=v[0],Qe[0]=v[1];for(let F=0;F<63;F++)Je[F]=v[2+F]/224;return re.unmap(),{handflag:new Float32Array(Ze),handedness:new Float32Array(Qe),landmarks:new Float32Array(Je)}}async function Dt(f){r.queue.copyExternalImageToTexture({source:f},{texture:ze},[224,224]);let a=r.createCommandEncoder();{let v=a.beginComputePass();v.setPipeline(Re),v.setBindGroup(0,j),v.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),v.end()}st(a,re),r.queue.submit([a.finish()]),await re.mapAsync(GPUMapMode.READ);let u=new Float32Array(re.getMappedRange());Ze[0]=u[0],Qe[0]=u[1];for(let v=0;v<63;v++)Je[v]=u[2+v]/224;return re.unmap(),{handflag:new Float32Array(Ze),handedness:new Float32Array(Qe),landmarks:new Float32Array(Je)}}async function en(f){let a=r.createCommandEncoder();a.copyBufferToBuffer(f,0,$,0,672*224*4),st(a,re),r.queue.submit([a.finish()]),await re.mapAsync(GPUMapMode.READ);let u=new Float32Array(re.getMappedRange());Ze[0]=u[0],Qe[0]=u[1];for(let v=0;v<63;v++)Je[v]=u[2+v]/224;return re.unmap(),{handflag:new Float32Array(Ze),handedness:new Float32Array(Qe),landmarks:new Float32Array(Je)}}let tn=k(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ft=0,qt=[re,tn],et=null,De=null;async function Yt(f){let a=qt[Ft];Ft=1-Ft;let u=r.createCommandEncoder();u.copyBufferToBuffer(f,0,$,0,672*224*4),st(u,a),r.queue.submit([u.finish()]);let v=null;if(et!==null&&De!==null){await et;let F=new Float32Array(De.getMappedRange()),Z=F[0],V=F[1],S=new Float32Array(63);for(let Ce=0;Ce<63;Ce++)S[Ce]=F[2+Ce]/224;De.unmap(),v={handflag:new Float32Array([Z]),handedness:new Float32Array([V]),landmarks:S}}return De=a,et=a.mapAsync(GPUMapMode.READ),v}async function jt(){if(!et||!De)return null;await et;let f=new Float32Array(De.getMappedRange()),a=f[0],u=f[1],v=new Float32Array(63);for(let F=0;F<63;F++)v[F]=f[2+F]/224;return De.unmap(),et=null,De=null,{handflag:new Float32Array([a]),handedness:new Float32Array([u]),landmarks:v}}async function Vt(){return null}async function nn(){return null}async function Xt(f=100){let a=new OffscreenCanvas(224,224),u=a.getContext("2d");u.fillStyle="#886644",u.fillRect(0,0,224,224);for(let V=0;V<5;V++)await Dt(a);let v=performance.now();for(let V=0;V<f;V++)await Dt(a);let Z=(performance.now()-v)/f;return{avgMs:Z,fps:1e3/Z}}async function ut(f=100){let a=await Xt(f);return{...a,medianMs:a.avgMs,minMs:a.avgMs}}async function $t(f){return Dt(f)}async function Zt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Rt(f){let a={};async function u(Z,V,S){let Ce=V*4,L=r.createBuffer({size:Ce,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),ve=r.createCommandEncoder();ve.copyBufferToBuffer(Z,0,L,0,Ce),r.queue.submit([ve.finish()]),await r.queue.onSubmittedWorkDone(),await L.mapAsync(GPUMapMode.READ);let e=new Float32Array(L.getMappedRange()),t=1/0,s=-1/0,_=0;for(let m=0;m<e.length;m++)e[m]<t&&(t=e[m]),e[m]>s&&(s=e[m]),e[m]!==0&&_++;let E=Array.from(e.slice(0,5));L.unmap(),L.destroy(),a[S]={min:t,max:s,nonZero:_,total:V,sample:E}}let v=new Float32Array(672*224);for(let Z=0;Z<50176;Z++)v[Z]=.5,v[50176+Z]=.3,v[448*224+Z]=.7;r.queue.writeBuffer(q,0,v);let F=r.createCommandEncoder();return F.copyBufferToBuffer(q,0,$,0,672*224*4),st(F,re),r.queue.submit([F.finish()]),await r.queue.onSubmittedWorkDone(),await u($,672*224,"inputBufA"),await u(ce,2688*112,"afterInitConvBufB"),await u(Y,1152,"gapOutput"),await u(T,63,"landmarks"),await u(le,1,"handflag"),await u(ye,65,"unifiedOutput"),a}return{device:r,run:Nt,runFromCanvas:Dt,runFromGPUBuffer:en,runFromGPUBufferPipelined:Yt,flushGPUBufferPipelined:jt,runFromCanvasPipelined:Vt,flushPipelined:nn,benchmark:Xt,benchmarkGPU:ut,runFromCanvasViaRender:$t,benchmarkDiagnostic:Zt,debugLayerOutputs:Rt}}function nt(c){return c.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var fn=nt(`
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
`),ln=nt(`
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
`),mn=nt(`
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
`),hn=nt(`
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
`),gn=nt(`
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
`),bn=nt(`
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
`),_n=nt(`
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
`),yn=nt(`
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
`),wn=nt(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function Pn(c,B){let n;if(B)n=B;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");n=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let R={r:"read-only-storage",s:"storage",u:"uniform"};function G(e){return n.createBindGroupLayout({entries:e.map((t,s)=>t==="t"?{binding:s,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:t==="sm"?{binding:s,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:s,visibility:GPUShaderStage.COMPUTE,buffer:{type:R[t]}})})}let r=n.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,D=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,M=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,P=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function h(e,t){return n.createBuffer({size:Math.max(e,4),usage:t})}function w(e,t,s){n.queue.writeBuffer(e,t,s)}function x(e){let t=h(e.data.byteLength,y);return w(t,0,e.data),t}let Q=Array.from(c.keys());function K(e){let t=c.get(e);if(!t)throw new Error(`Weight not found: ${e}`);return t}function d(...e){let t=Q.find(s=>e.every(_=>s.includes(_)));if(!t)throw new Error(`Weight not found for: ${e.join(", ")}`);return K(t)}function U(e){let[,t,s,_]=e.shape,E=new Float32Array(_*25);for(let m=0;m<_;m++)for(let te=0;te<t;te++)for(let ne=0;ne<s;ne++)E[m*25+te*5+ne]=e.data[te*s*_+ne*_+m];return E}function H(e){let[t,,,s]=e.shape,_=new Float32Array(t*s);for(let E=0;E<t;E++)for(let m=0;m<s;m++)_[E*s+m]=e.data[E*s+m];return _}let Pe=n.createShaderModule({code:fn}),Ue=n.createShaderModule({code:ln}),J=n.createShaderModule({code:mn}),k=n.createShaderModule({code:hn}),pe=n.createShaderModule({code:bn}),ge=n.createShaderModule({code:gn}),ae=n.createShaderModule({code:_n}),Me=n.createShaderModule({code:yn}),vt=n.createShaderModule({code:wn}),at=G(["r","r","r","r","s","u"]),Ut=G(["r","r","r","s","u"]),Gt=G(["r","r","r","r","r","s","u"]),Mt=G(["r","r","r","s","u"]),Fe=G(["r","r","r","r","s","u"]),be=G(["r","r","s","u"]),mt=G(["t","s","u"]),Be=G(["t","s","u","sm"]),At=G(["s","u"]);function _e(e,t){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:t,entryPoint:"main"}})}let Te=_e(at,Pe),ht=_e(Ut,Ue),Re=_e(Gt,J),Ht=_e(Mt,k),Wt=_e(Fe,pe),Ae=_e(be,ge),l=_e(mt,ae),C=_e(Be,Me),A=_e(At,vt),I=d("conv2d/Conv2D"),ee=d("batch_normalization/","conv2d/Conv2D"),ue=d("p_re_lu/"),X=x(I),$=x(ee),ce=x(ue),N=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(e=>{let t=d(e.dwKey),s=d(e.pwKey),_=d(e.bnKey),E=d(e.preluKey),m=U(t),te=h(m.byteLength,y);w(te,0,m);let ne=new Float32Array(e.inCh),oe=h(ne.byteLength,y);w(oe,0,ne);let b=H(s),ie=h(b.byteLength,y);w(ie,0,b);let se=x(_),o=x(E);return{dwWeightBuf:te,dwBiasBuf:oe,pwWeightBuf:ie,pwBiasBuf:se,alphaBuf:o,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),z=H(d("conv2d_25/Conv2D")),q=h(z.byteLength,y);w(q,0,z);let Y=x(d("batch_normalization_25/")),T=x(d("p_re_lu_25/")),fe={dwWeightBuf:(()=>{let e=U(d("depthwise_conv2d_24/")),t=h(e.byteLength,y);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=h(e.byteLength,y);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=H(d("conv2d_26/")),t=h(e.byteLength,y);return w(t,0,e),t})(),pwBiasBuf:x(d("batch_normalization_26/")),alphaBuf:x(d("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},le={dwWeightBuf:(()=>{let e=U(d("depthwise_conv2d_25/")),t=h(e.byteLength,y);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=h(e.byteLength,y);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=H(d("conv2d_27/Conv2D1")),t=h(e.byteLength,y);return w(t,0,e),t})(),pwBiasBuf:x(d("batch_normalization_27/")),alphaBuf:x(d("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},xe=H(d("conv2d_28/Conv2D")),ye=h(xe.byteLength,y);w(ye,0,xe);let re=x(d("batch_normalization_28/")),ze=x(d("p_re_lu_28/")),Le={dwWeightBuf:(()=>{let e=U(d("depthwise_conv2d_26/")),t=h(e.byteLength,y);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=h(e.byteLength,y);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=H(d("conv2d_29/")),t=h(e.byteLength,y);return w(t,0,e),t})(),pwBiasBuf:x(d("batch_normalization_29/")),alphaBuf:x(d("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},Ee={dwWeightBuf:(()=>{let e=U(d("depthwise_conv2d_27/")),t=h(e.byteLength,y);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=h(e.byteLength,y);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=H(d("conv2d_30/Conv2D1")),t=h(e.byteLength,y);return w(t,0,e),t})(),pwBiasBuf:x(d("batch_normalization_30/")),alphaBuf:x(d("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},He=H(d("classifier_palm_16_NO_PRUNING/Conv2D")),Ie=h(He.byteLength,y);w(Ie,0,He);let rt=x(d("classifier_palm_16_NO_PRUNING/BiasAdd")),ot=H(d("regressor_palm_16_NO_PRUNING/Conv2D")),gt=h(ot.byteLength,y);w(gt,0,ot);let kt=x(d("regressor_palm_16_NO_PRUNING/BiasAdd")),We=H(d("classifier_palm_8_NO_PRUNING/Conv2D")),ke=h(We.byteLength,y);w(ke,0,We);let Oe=x(d("classifier_palm_8_NO_PRUNING/BiasAdd")),St=H(d("regressor_palm_8_NO_PRUNING/Conv2D")),Ne=h(St.byteLength,y);w(Ne,0,St);let it=x(d("regressor_palm_8_NO_PRUNING/BiasAdd")),qe=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Se=h(36864*3*4,y),Ye=h(qe,D),Ot=h(qe,D),Tt=h(qe,D),bt=h(576*256*4,D),Kt=new Map;function Jt(e){let t=Kt.get(e);return t||(t=h(4,P),w(t,0,new Uint32Array([e])),Kt.set(e,t)),t}let _t=h(144*256*4,D|GPUBufferUsage.COPY_DST),yt=h(576*128*4,D|GPUBufferUsage.COPY_DST),je=h(864*4,M),Ve=h(15552*4,M),Xe=h(576*2*4,M),$e=h(576*36*4,M),wt=h(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Et=h(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Pt=h(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Bt=h(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),xt=n.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function W(e,t){return Math.ceil(e/t)}function j(e){let t=h(e.byteLength,P);return w(t,0,e),t}let It=j(new Uint32Array([1,3,32,192,192,96,96])),Ze=N.map(e=>{let t=e.stride===2?e.inH/2:e.inH,s=t,_=e.stride===2?1:2,E=e.inCh;return{dw:j(new Uint32Array([1,e.inCh,e.inH,e.inH,t,s,e.stride,_])),pw:j(new Uint32Array([1,e.inCh,e.outCh,t,s,E,e.stride,e.inH,e.inH])),outH:t,outW:s}}),Qe=(()=>{let e=fe;return{dw:j(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:j(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),Je=(()=>{let e=le;return{dw:j(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:j(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),st=(()=>{let e=Le;return{dw:j(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:j(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),Nt=(()=>{let e=Ee;return{dw:j(new Uint32Array([1,e.inCh,e.inH,e.inH,e.inH,e.inH,e.stride,2])),pw:j(new Uint32Array([1,e.inCh,e.outCh,e.inH,e.inH,e.inCh,e.stride,e.inH,e.inH])),outH:e.inH}})(),Dt=j(new Uint32Array([1,256,6,6,12,12])),en=j(new Uint32Array([1,256,12,12,12,12])),tn=j(new Uint32Array([1,256,12,12,24,24])),Ft=j(new Uint32Array([1,128,24,24,24,24])),qt=j(new Uint32Array([1,256,256,12,12])),et=j(new Uint32Array([1,256,128,24,24])),De=j(new Uint32Array([1,256,6,12,12])),Yt=j(new Uint32Array([1,256,108,12,12])),jt=j(new Uint32Array([1,128,2,24,24])),Vt=j(new Uint32Array([1,128,36,24,24])),nn=j(new Uint32Array([192,192,192])),Xt=n.createBindGroup({layout:mt,entries:[{binding:0,resource:xt.createView()},{binding:1,resource:{buffer:Se}},{binding:2,resource:{buffer:nn}}]}),ut=null,$t=0,Zt=0,Rt=h(32,P);function f(e,t){return ut&&$t===e&&Zt===t||(ut&&ut.destroy(),ut=n.createTexture({size:[e,t,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),$t=e,Zt=t),ut}let a=n.createBindGroup({layout:at,entries:[{binding:0,resource:{buffer:Se}},{binding:1,resource:{buffer:X}},{binding:2,resource:{buffer:$}},{binding:3,resource:{buffer:ce}},{binding:4,resource:{buffer:Ye}},{binding:5,resource:{buffer:It}}]});function u(e,t,s){}function v(e,t,s,_,E,m){let te=m.outH,ne=n.createBindGroup({layout:Ut,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:t.dwWeightBuf}},{binding:2,resource:{buffer:t.dwBiasBuf}},{binding:3,resource:{buffer:Tt}},{binding:4,resource:{buffer:m.dw}}]}),oe=e.beginComputePass();oe.setPipeline(ht),oe.setBindGroup(0,ne),oe.dispatchWorkgroups(W(te,8),W(m.outH,8),t.inCh),oe.end(),t.inCh*m.outH*te;let b=n.createBindGroup({layout:Gt,entries:[{binding:0,resource:{buffer:Tt}},{binding:1,resource:{buffer:E}},{binding:2,resource:{buffer:t.pwWeightBuf}},{binding:3,resource:{buffer:t.pwBiasBuf}},{binding:4,resource:{buffer:t.alphaBuf}},{binding:5,resource:{buffer:_}},{binding:6,resource:{buffer:m.pw}}]}),ie=e.beginComputePass();ie.setPipeline(Re),ie.setBindGroup(0,b),ie.dispatchWorkgroups(W(te,8),W(m.outH,8),t.outCh),ie.end(),t.outCh*m.outH*te}function F(e,t,s,_,E,m,te,ne,oe){let b=n.createBindGroup({layout:Mt,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:_}},{binding:3,resource:{buffer:E}},{binding:4,resource:{buffer:m}}]}),ie=e.beginComputePass();ie.setPipeline(Ht),ie.setBindGroup(0,b),ie.dispatchWorkgroups(W(oe,8),W(ne,8),te),ie.end()}function Z(e,t,s,_,E,m,te,ne,oe,b){let ie=n.createBindGroup({layout:Fe,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:_}},{binding:3,resource:{buffer:E}},{binding:4,resource:{buffer:m}},{binding:5,resource:{buffer:te}}]}),se=e.beginComputePass();se.setPipeline(Wt),se.setBindGroup(0,ie),se.dispatchWorkgroups(W(b,8),W(oe,8),ne),se.end()}async function V(e){36864*3;{let o=e.beginComputePass();o.setPipeline(Te),o.setBindGroup(0,a),o.dispatchWorkgroups(W(96,8),W(96,8),32),o.end()}9216*32;let t=Ye,s=Ot;for(let o=0;o<N.length;o++){let g=N[o];v(e,g,t,s,t,Ze[o]);let he=t;t=s,s=he,o===13&&e.copyBufferToBuffer(t,0,yt,0,576*128*4),o===18&&e.copyBufferToBuffer(t,0,_t,0,144*256*4)}{let o=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:bt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Dt}}]}),g=e.beginComputePass();g.setPipeline(Ae),g.setBindGroup(0,o),g.dispatchWorkgroups(W(12,8),W(12,8),256),g.end()}{let o=t;t=s,s=o}144*256,Z(e,t,q,Y,T,s,qt,256,12,12);{let o=t;t=s,s=o}144*256;{let o=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:_t}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:en}}]}),g=e.beginComputePass();g.setPipeline(Ae),g.setBindGroup(0,o),g.dispatchWorkgroups(W(12,8),W(12,8),256),g.end()}{let o=t;t=s,s=o}144*256,v(e,fe,t,s,t,Qe);{let o=t;t=s,s=o}v(e,le,t,s,t,Je);{let o=t;t=s,s=o}F(e,t,Ie,rt,je,De,6,12,12),F(e,t,gt,kt,Ve,Yt,108,12,12);{let o=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:bt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:tn}}]}),g=e.beginComputePass();g.setPipeline(Ae),g.setBindGroup(0,o),g.dispatchWorkgroups(W(24,8),W(24,8),256),g.end()}{let o=t;t=s,s=o}576*256,Z(e,t,ye,re,ze,s,et,128,24,24);{let o=t;t=s,s=o}576*128;{let o=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:Ft}}]}),g=e.beginComputePass();g.setPipeline(Ae),g.setBindGroup(0,o),g.dispatchWorkgroups(W(24,8),W(24,8),128),g.end()}{let o=t;t=s,s=o}576*128,v(e,Le,t,s,t,st);{let o=t;t=s,s=o}v(e,Ee,t,s,t,Nt);{let o=t;t=s,s=o}F(e,t,ke,Oe,Xe,jt,2,24,24),576*2,F(e,t,Ne,it,$e,Vt,36,24,24),576*36,n.queue.submit([e.finish()]);let _=n.createCommandEncoder();_.copyBufferToBuffer(je,0,wt,0,864*4),_.copyBufferToBuffer(Ve,0,Et,0,15552*4),_.copyBufferToBuffer(Xe,0,Pt,0,576*2*4),_.copyBufferToBuffer($e,0,Bt,0,576*36*4),n.queue.submit([_.finish()]),await Promise.all([wt.mapAsync(GPUMapMode.READ),Et.mapAsync(GPUMapMode.READ),Pt.mapAsync(GPUMapMode.READ),Bt.mapAsync(GPUMapMode.READ)]);let E=new Float32Array(wt.getMappedRange()).slice(),m=new Float32Array(Et.getMappedRange()).slice(),te=new Float32Array(Pt.getMappedRange()).slice(),ne=new Float32Array(Bt.getMappedRange()).slice();wt.unmap(),Et.unmap(),Pt.unmap(),Bt.unmap();let oe=2016,b=new Float32Array(oe),ie=new Float32Array(oe*18),se=0;for(let o=0;o<12;o++)for(let g=0;g<12;g++)for(let he=0;he<6;he++){b[se]=E[he*144+o*12+g];for(let Ge=0;Ge<18;Ge++){let ct=he*18+Ge;ie[se*18+Ge]=m[ct*144+o*12+g]}se++}for(let o=0;o<24;o++)for(let g=0;g<24;g++)for(let he=0;he<2;he++){b[se]=te[he*576+o*24+g];for(let Ge=0;Ge<18;Ge++){let ct=he*18+Ge;ie[se*18+Ge]=ne[ct*576+o*24+g]}se++}return{scores:b,regressors:ie}}async function S(e){n.queue.copyExternalImageToTexture({source:e},{texture:xt},[192,192]);let t=n.createCommandEncoder();{let s=t.beginComputePass();s.setPipeline(l),s.setBindGroup(0,Xt),s.dispatchWorkgroups(W(192,16),W(192,16),1),s.end()}return V(t)}async function Ce(e,t,s){let _=Math.min(192/t,192/s),E=Math.round(t*_),m=Math.round(s*_),te=Math.floor((192-E)/2),ne=Math.floor((192-m)/2),oe=te/192,b=ne/192,ie=f(t,s),se;e instanceof HTMLVideoElement?se=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?se=await createImageBitmap(e,{colorSpaceConversion:"none"}):se=e,n.queue.copyExternalImageToTexture({source:se},{texture:ie},[t,s]);let o=new ArrayBuffer(32),g=new Uint32Array(o),he=new Float32Array(o);g[0]=t,g[1]=s,g[2]=192,g[3]=0,he[4]=t/E,he[5]=s/m,he[6]=te,he[7]=ne,n.queue.writeBuffer(Rt,0,o);let Ge=n.createBindGroup({layout:Be,entries:[{binding:0,resource:ie.createView()},{binding:1,resource:{buffer:Se}},{binding:2,resource:{buffer:Rt}},{binding:3,resource:r}]}),ct=n.createCommandEncoder();{let i=ct.beginComputePass();i.setPipeline(C),i.setBindGroup(0,Ge),i.dispatchWorkgroups(W(192,16),W(192,16),1),i.end()}return{output:await V(ct),lbPadX:oe,lbPadY:b}}async function L(e,t){let s=n.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),_=n.createCommandEncoder();_.copyBufferToBuffer(e,0,s,0,t*4),n.queue.submit([_.finish()]),await s.mapAsync(GPUMapMode.READ);let E=new Float32Array(s.getMappedRange()).slice();return s.unmap(),s.destroy(),E}async function ve(e,t,s){function _(i,p=1e3){let O=i.slice(0,p),me=Math.max(0,Math.floor(i.length/2)-250);return{min:Math.min(...O),max:Math.max(...O),mean:O.reduce((we,dt)=>we+dt,0)/O.length,nonZero:O.filter(we=>we!==0).length,sample:Array.from(O.slice(0,10)),data500:Array.from(i.slice(0,500)),dataMid500:Array.from(i.slice(me,me+500)),totalLength:i.length}}function E(i,p,O,me){let we=[],dt=Math.floor(O/2),pt=Math.floor(me/2),Ke=O*me;for(let ft=0;ft<p&&we.length<500;ft++)for(let zt=-1;zt<=1&&we.length<500;zt++)for(let Lt=-1;Lt<=1&&we.length<500;Lt++){let lt=dt+zt,an=pt+Lt;lt>=0&&lt<O&&an>=0&&an<me&&we.push(i[ft*Ke+lt*me+an])}return we}let m={},te;e instanceof HTMLImageElement?(t=t??e.naturalWidth,s=s??e.naturalHeight,te=await createImageBitmap(e,{colorSpaceConversion:"none"})):(t=t??e.width??192,s=s??e.height??192,te=e);let ne=t,oe=s;if(ne!==192||oe!==192){let i=Math.min(192/ne,192/oe),p=Math.round(ne*i),O=Math.round(oe*i),me=Math.floor((192-p)/2),we=Math.floor((192-O)/2),dt=f(ne,oe);n.queue.copyExternalImageToTexture({source:te},{texture:dt},[ne,oe]);let pt=new ArrayBuffer(32),Ke=new Uint32Array(pt),ft=new Float32Array(pt);Ke[0]=ne,Ke[1]=oe,Ke[2]=192,Ke[3]=0,ft[4]=ne/p,ft[5]=oe/O,ft[6]=me,ft[7]=we,n.queue.writeBuffer(Rt,0,pt);let zt=n.createBindGroup({layout:Be,entries:[{binding:0,resource:dt.createView()},{binding:1,resource:{buffer:Se}},{binding:2,resource:{buffer:Rt}},{binding:3,resource:r}]});{let Lt=n.createCommandEncoder(),lt=Lt.beginComputePass();lt.setPipeline(C),lt.setBindGroup(0,zt),lt.dispatchWorkgroups(W(192,16),W(192,16),1),lt.end(),n.queue.submit([Lt.finish()])}}else{n.queue.copyExternalImageToTexture({source:te},{texture:xt},[192,192]);let i=j(new Uint32Array([192,192,192])),p=n.createBindGroup({layout:mt,entries:[{binding:0,resource:xt.createView()},{binding:1,resource:{buffer:Se}},{binding:2,resource:{buffer:i}}]});{let O=n.createCommandEncoder(),me=O.beginComputePass();me.setPipeline(l),me.setBindGroup(0,p),me.dispatchWorkgroups(W(192,16),W(192,16),1),me.end(),n.queue.submit([O.finish()])}}{let i=await L(Se,110592),p=_(i);p.dataCenter500=E(i,3,192,192),m.input=p}let b=n.createCommandEncoder(),ie=n.createBindGroup({layout:at,entries:[{binding:0,resource:{buffer:Se}},{binding:1,resource:{buffer:X}},{binding:2,resource:{buffer:$}},{binding:3,resource:{buffer:ce}},{binding:4,resource:{buffer:Ye}},{binding:5,resource:{buffer:It}}]}),se=b.beginComputePass();se.setPipeline(Te),se.setBindGroup(0,ie),se.dispatchWorkgroups(W(96,8),W(96,8),32),se.end(),n.queue.submit([b.finish()]);{let i=await L(Ye,294912),p=_(i);p.dataCenter500=E(i,32,96,96),m.initConv=p}let o=Ye,g=Ot;for(let i=0;i<N.length;i++){let p=N[i];b=n.createCommandEncoder(),v(b,p,o,g,o,Ze[i]),n.queue.submit([b.finish()]);let O=o;o=g,g=O;{let me=p.stride===2?p.inH/2:p.inH,we=me,dt=me*we*p.outCh,pt=await L(o,dt),Ke=_(pt);Ke.dataCenter500=E(pt,p.outCh,me,we),Ke.spatialShape=[p.outCh,me,we],m[`block${i}`]=Ke}i===13&&(b=n.createCommandEncoder(),b.copyBufferToBuffer(o,0,yt,0,576*128*4),n.queue.submit([b.finish()])),i===18&&(b=n.createCommandEncoder(),b.copyBufferToBuffer(o,0,_t,0,144*256*4),n.queue.submit([b.finish()]))}b=n.createCommandEncoder();{let i=j(new Uint32Array([1,256,6,6,12,12])),p=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:bt}},{binding:2,resource:{buffer:g}},{binding:3,resource:{buffer:i}}]}),O=b.beginComputePass();O.setPipeline(Ae),O.setBindGroup(0,p),O.dispatchWorkgroups(W(12,8),W(12,8),256),O.end()}n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,36864),p=_(i);p.dataCenter500=E(i,256,12,12),m.fpnUpsample6to12=p}b=n.createCommandEncoder(),Z(b,o,q,Y,T,g,qt,256,12,12),n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,36864),p=_(i);p.dataCenter500=E(i,256,12,12),m.fpn6to12Conv=p}{let i=await L(_t,36864),p=_(i);p.dataCenter500=E(i,256,12,12),m.backbone12Skip=p}b=n.createCommandEncoder();{let i=j(new Uint32Array([1,256,12,12,12,12])),p=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:_t}},{binding:2,resource:{buffer:g}},{binding:3,resource:{buffer:i}}]}),O=b.beginComputePass();O.setPipeline(Ae),O.setBindGroup(0,p),O.dispatchWorkgroups(W(12,8),W(12,8),256),O.end()}n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,36864),p=_(i);p.dataCenter500=E(i,256,12,12),m.fpnAdd12=p}b=n.createCommandEncoder(),v(b,fe,o,g,o,Qe),n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,36864),p=_(i);p.dataCenter500=E(i,256,12,12),m.fpn12Block1=p}b=n.createCommandEncoder(),v(b,le,o,g,o,Je),n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,36864),p=_(i);p.dataCenter500=E(i,256,12,12),m.fpn12Block2=p}b=n.createCommandEncoder(),F(b,o,Ie,rt,je,De,6,12,12),n.queue.submit([b.finish()]);{let i=await L(je,864),p=_(i);p.dataCenter500=E(i,6,12,12),m.cls16=p}b=n.createCommandEncoder(),F(b,o,gt,kt,Ve,Yt,108,12,12),n.queue.submit([b.finish()]);{let i=await L(Ve,15552),p=_(i,500);p.dataCenter500=E(i,108,12,12),m.reg16=p}b=n.createCommandEncoder();{let i=j(new Uint32Array([1,256,12,12,24,24])),p=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:bt}},{binding:2,resource:{buffer:g}},{binding:3,resource:{buffer:i}}]}),O=b.beginComputePass();O.setPipeline(Ae),O.setBindGroup(0,p),O.dispatchWorkgroups(W(24,8),W(24,8),256),O.end()}n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,147456),p=_(i);p.dataCenter500=E(i,256,24,24),m.fpnUpsample12to24=p}b=n.createCommandEncoder(),Z(b,o,ye,re,ze,g,et,128,24,24),n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,73728),p=_(i);p.dataCenter500=E(i,128,24,24),m.fpn12to24Conv=p}{let i=await L(yt,73728),p=_(i);p.dataCenter500=E(i,128,24,24),m.backbone24Skip=p}b=n.createCommandEncoder();{let i=j(new Uint32Array([1,128,24,24,24,24])),p=n.createBindGroup({layout:be,entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:yt}},{binding:2,resource:{buffer:g}},{binding:3,resource:{buffer:i}}]}),O=b.beginComputePass();O.setPipeline(Ae),O.setBindGroup(0,p),O.dispatchWorkgroups(W(24,8),W(24,8),128),O.end()}n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,73728),p=_(i);p.dataCenter500=E(i,128,24,24),m.fpnAdd24=p}b=n.createCommandEncoder(),v(b,Le,o,g,o,st),n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,73728),p=_(i);p.dataCenter500=E(i,128,24,24),m.fpn24Block1=p}b=n.createCommandEncoder(),v(b,Ee,o,g,o,Nt),n.queue.submit([b.finish()]);{let i=o;o=g,g=i}{let i=await L(o,73728),p=_(i);p.dataCenter500=E(i,128,24,24),m.fpn24Block2=p}b=n.createCommandEncoder(),F(b,o,ke,Oe,Xe,jt,2,24,24),n.queue.submit([b.finish()]);{let i=await L(Xe,1152),p=_(i);p.dataCenter500=E(i,2,24,24),m.cls8=p}b=n.createCommandEncoder(),F(b,o,Ne,it,$e,Vt,36,24,24),n.queue.submit([b.finish()]);{let i=await L($e,20736),p=_(i);p.dataCenter500=E(i,36,24,24),m.reg8=p}m.initWeights=_(await L(X,100),100),m.initBias=_(await L($,32),32),m.cls16Weights=_(await L(Ie,100),100),m.cls16Bias=_(await L(rt,6),6),m.cls8Weights=_(await L(ke,100),100),m.cls8Bias=_(await L(Oe,2),2),m.fpn6to12Weights=_(await L(q,100),100);let he=await L(je,864),Ge=await L(Xe,576*2);m.rawScores=new Float32Array(2016),m.rawScores.set(he,0),m.rawScores.set(Ge,864);let ct=await L(Ve,15552),un=await L($e,576*36);return m.rawRegressors=new Float32Array(36288),m.rawRegressors.set(ct,0),m.rawRegressors.set(un,15552),m.rawInput=await L(Se,36864*3),m}return{device:n,run:S,runWithResize:Ce,debugRun:ve}}function Wn(){let c=[];for(let B=0;B<12;B++)for(let n=0;n<12;n++){let R=(n+.5)/12,G=(B+.5)/12;for(let r=0;r<6;r++)c.push({x:R,y:G})}for(let B=0;B<24;B++)for(let n=0;n<24;n++){let R=(n+.5)/24,G=(B+.5)/24;for(let r=0;r<2;r++)c.push({x:R,y:G})}return c}var Bn=Wn();function On(c){return 1/(1+Math.exp(-c))}function rn(c,B){let n=[],{scores:R,regressors:G}=c,r=192;for(let y=0;y<Bn.length;y++){let D=On(R[y]);if(D<B)continue;let M=Bn[y],P=y*18,h=M.x+G[P+0]/r,w=M.y+G[P+1]/r,x=G[P+2]/r,Q=G[P+3]/r,K=[];for(let d=0;d<7;d++){let U=M.x+G[P+4+d*2]/r,H=M.y+G[P+4+d*2+1]/r;K.push([U,H])}n.push({score:D,box:[h,w,x,Q],keypoints:K})}return n}function on(c,B){if(c.length===0)return[];let n=[...c].sort((r,y)=>y.score-r.score),R=[],G=new Set;for(let r=0;r<n.length;r++){if(G.has(r))continue;let y=[r];for(let K=r+1;K<n.length;K++)G.has(K)||Kn(n[r],n[K])>B&&(y.push(K),G.add(K));let D=0,M=0,P=0,h=0,w=0,x=[];for(let K=0;K<7;K++)x.push([0,0]);for(let K of y){let d=n[K],U=d.score;D+=U,M+=d.box[0]*U,P+=d.box[1]*U,h+=d.box[2]*U,w+=d.box[3]*U;for(let H=0;H<7;H++)x[H][0]+=d.keypoints[H][0]*U,x[H][1]+=d.keypoints[H][1]*U}let Q=1/D;R.push({score:n[r].score,box:[M*Q,P*Q,h*Q,w*Q],keypoints:x.map(([K,d])=>[K*Q,d*Q])})}return R}function Kn(c,B){let n=c.box[0]-c.box[2]/2,R=c.box[1]-c.box[3]/2,G=c.box[0]+c.box[2]/2,r=c.box[1]+c.box[3]/2,y=B.box[0]-B.box[2]/2,D=B.box[1]-B.box[3]/2,M=B.box[0]+B.box[2]/2,P=B.box[1]+B.box[3]/2,h=Math.max(n,y),w=Math.max(R,D),x=Math.min(G,M),Q=Math.min(r,P),K=Math.max(0,x-h),d=Math.max(0,Q-w),U=K*d,H=(G-n)*(r-R),Pe=(M-y)*(P-D),Ue=H+Pe-U;return Ue>0?U/Ue:0}function Fn(c){let[B,n,R,G]=c.box,r=c.keypoints[0],y=c.keypoints[2],D=y[0]-r[0],M=y[1]-r[1],P=Math.atan2(M,D),w=-Math.PI/2-P,x=Math.max(R,G),K=x*2.6,d=-.5*x,U=Math.cos(w),H=Math.sin(w),Pe=d*H,Ue=d*U;return{centerX:B+Pe,centerY:n+Ue,width:K,height:K,rotation:w}}function xn(c,B={}){let{scoreThreshold:n=.5,nmsThreshold:R=.3,maxHands:G=2}=B;async function r(P){let h=await c.run(P),w=rn(h,n);return on(w,R).slice(0,G).map(Fn)}async function y(P){let h=await c.run(P),w=rn(h,n);return on(w,R).slice(0,G)}async function D(P,h,w){let{output:x,lbPadX:Q,lbPadY:K}=await c.runWithResize(P,h,w),d=rn(x,n);return{detections:on(d,R).slice(0,G),lbPadX:Q,lbPadY:K}}async function M(P,h,w){let{output:x,lbPadX:Q,lbPadY:K}=await c.runWithResize(P,h,w);return{scores:x.scores,regressors:x.regressors,lbPadX:Q,lbPadY:K}}return{detect:r,detectRaw:y,detectRawWithResize:D,detectRawSSD:M,model:c}}var sn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Qt(c){let B={};for(let n=0;n<sn.length;n++)B[sn[n]]=c[n];return B}function zn(c){return c.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var In=zn(`
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
`);function Cn(c){let B=c.createShaderModule({code:In}),n=c.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),R=c.createComputePipeline({layout:c.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:B,entryPoint:"main"}}),G=c.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),r=c.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),y=c.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),D=new Float32Array(8);function M(P,h,w,x,Q,K,d){c.queue.writeBuffer(r,0,new Uint32Array([Q,K,d,0])),D.set(x),c.queue.writeBuffer(y,0,D);let U=c.createBindGroup({layout:n,entries:[{binding:0,resource:h.createView()},{binding:1,resource:{buffer:w}},{binding:2,resource:{buffer:r}},{binding:3,resource:{buffer:y}},{binding:4,resource:G}]}),H=P.beginComputePass();H.setPipeline(R),H.setBindGroup(0,U),H.dispatchWorkgroups(Math.ceil(d/16),Math.ceil(d/16),1),H.end()}return{crop:M}}var Nn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@0.3.0/weights";async function qn(c={}){let{weightsUrl:B,scoreThreshold:n=.5,palmScoreThreshold:R=.5,maxHands:G=3}=c;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let r=(B??Nn).replace(/\/$/,"")+"/",[y,D,M,P]=await Promise.all([fetch(`${r}weights_f16_full.json`),fetch(`${r}weights_f16_full.bin`),fetch(`${r}palm_detection_weights.json`),fetch(`${r}palm_detection_weights.bin`)]);if(!y.ok)throw new Error(`Failed to fetch landmark weights: ${y.status}`);if(!D.ok)throw new Error(`Failed to fetch landmark weights: ${D.status}`);if(!M.ok)throw new Error(`Failed to fetch palm detection weights: ${M.status}`);if(!P.ok)throw new Error(`Failed to fetch palm detection weights: ${P.status}`);let[h,w,x,Q]=await Promise.all([y.json(),D.arrayBuffer(),M.json(),P.arrayBuffer()]),K=dn(h,w),d=cn(x,Q),U=224,H=await pn(K);{let l=new OffscreenCanvas(U,U),C=l.getContext("2d");C.fillStyle="#886644",C.fillRect(0,0,U,U),C.fillStyle="#cc9966",C.fillRect(50,50,124,124);let A=await H.runFromCanvas(l);A.landmarks.every(ee=>ee===0)&&A.handflag.every(ee=>ee===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let Pe=await Pn(d),Ue=xn(Pe,{scoreThreshold:R,maxHands:G}),J=[];function k(l,C,A){let I=l[0],ee=l[5],ue=l[9],X=l[13],$=I.x*C,ce=I.y*A,de=(ee.x+X.x)/2,N=(ee.y+X.y)/2;de=(de+ue.x)/2*C,N=(N+ue.y)/2*A;let z=Math.PI/2-Math.atan2(-(N-ce),de-$),q=z-2*Math.PI*Math.floor((z+Math.PI)/(2*Math.PI)),Y=[0,1,2,3,5,6,9,10,13,14,17,18],T=Math.cos(q),fe=Math.sin(q),le=1/0,xe=-1/0,ye=1/0,re=-1/0;for(let ke of Y){let Oe=l[ke],St=Oe.x*C,Ne=Oe.y*A,it=T*St+fe*Ne,qe=-fe*St+T*Ne;le=Math.min(le,it),xe=Math.max(xe,it),ye=Math.min(ye,qe),re=Math.max(re,qe)}let ze=(le+xe)/2,Le=(ye+re)/2,Ee=xe-le,He=re-ye,Ie=(T*ze-fe*Le)/C,rt=(fe*ze+T*Le)/A;Ee/=C,He/=A;let ot=-.1;Ie+=-A*He*ot*fe/C,rt+=He*ot*T;let We=Math.max(Ee*C,He*A)*2;return{centerXpx:Ie*C,centerYpx:rt*A,sizePx:We,rotation:q}}let pe=H.device,ge=null,ae=null,Me=null,vt=0,at=0;function Ut(){return ge||(ge=Cn(pe)),ge}function Gt(){return ae||(ae=pe.createBuffer({size:3*U*U*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),ae}function Mt(l,C){return(!Me||vt!==l||at!==C)&&(Me&&Me.destroy(),Me=pe.createTexture({size:[l,C],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),vt=l,at=C),Me}let Fe=0,be=0;function mt(l){let C=1/(1-2*Fe),A=1/(1-2*be);return{score:l.score,box:[(l.box[0]-Fe)*C,(l.box[1]-be)*A,l.box[2]*C,l.box[3]*A],keypoints:l.keypoints.map(([I,ee])=>[(I-Fe)*C,(ee-be)*A])}}function Be(l,C,A){let I=l.keypoints[0],ee=l.keypoints[2],ue=(ee[0]-I[0])*C,X=(ee[1]-I[1])*A,$=Math.atan2(-X,ue),de=Math.PI/2-$,N=de-2*Math.PI*Math.floor((de+Math.PI)/(2*Math.PI)),[z,q,Y,T]=l.box,fe=Math.cos(N),le=Math.sin(N),xe=T*A,ye=z+.5*xe*le/C,re=q+-.5*T*fe,Ee=Math.max(Y*C,T*A)*2.6;return{centerXpx:ye*C,centerYpx:re*A,sizePx:Ee,rotation:N}}function At(l){return l instanceof HTMLCanvasElement||l instanceof OffscreenCanvas?[l.width,l.height]:typeof ImageBitmap<"u"&&l instanceof ImageBitmap?[l.width,l.height]:l instanceof ImageData?[l.width,l.height]:l instanceof HTMLVideoElement?[l.videoWidth,l.videoHeight]:l instanceof HTMLImageElement?[l.naturalWidth,l.naturalHeight]:[U,U]}function _e(l,C,A,I,ee,ue){let X=Math.cos(l.rotation),$=Math.sin(l.rotation),ce=l.sizePx/U,de=U/2,N=X*ce/A,z=-$*ce/A,q=l.centerXpx/A-de*(N+z),Y=$*ce/I,T=X*ce/I,fe=l.centerYpx/I-de*(Y+T),le=pe.createCommandEncoder();return ee.crop(le,C,ue,[N,z,q,Y,T,fe],A,I,U),pe.queue.submit([le.finish()]),{cosR:X,sinR:$,pxROI:l}}function Te(l,C,A,I,ee,ue,X){let $=l.handflag[0];if($<X)return null;let ce=l.handedness[0]>.5,de=[];for(let N=0;N<21;N++){let z=l.landmarks[N*3],q=l.landmarks[N*3+1],Y=l.landmarks[N*3+2],T=(z-.5)*I.sizePx,fe=(q-.5)*I.sizePx,le=C*T-A*fe+I.centerXpx,xe=A*T+C*fe+I.centerYpx;de.push({x:le/ee,y:xe/ue,z:Y})}return{landmarks:de,score:$,handedness:ce?"right":"left"}}async function ht(l,C,A,I,ee,ue,X=!1){let{cosR:$,sinR:ce}=_e(l,C,A,I,ee,ue),de=await H.runFromGPUBuffer(ue),N=X?Math.min(n,.1):n;return Te(de,$,ce,l,A,I,N)}let Re=null;async function Ht(l){let C,A,I;if(l instanceof HTMLVideoElement)C=l.videoWidth,A=l.videoHeight,I=await createImageBitmap(l,{colorSpaceConversion:"none"});else if(l instanceof HTMLImageElement)C=l.naturalWidth,A=l.naturalHeight,I=await createImageBitmap(l,{colorSpaceConversion:"none"});else if(l instanceof ImageData){let z=await createImageBitmap(l,{colorSpaceConversion:"none"});[C,A]=[z.width,z.height],I=z}else[C,A]=At(l),I=l;let ee=Ut(),ue=Gt(),X=Mt(C,A);if(pe.queue.copyExternalImageToTexture({source:I},{texture:X},[C,A]),J.length===1&&Re!==null){let z=await H.flushGPUBufferPipelined(),q=[];if(z){let Y=Re,T=Te(z,Y.cosR,Y.sinR,Y.pxROI,Y.srcWidth,Y.srcHeight,Math.min(n,.1));T?(q=[{score:T.score,handedness:T.handedness,landmarks:T.landmarks,keypoints:Qt(T.landmarks)}],J=[{landmarks:T.landmarks,handedness:T.handedness}]):(J=[],Re=null)}if(J.length===1){let Y=J[0],T=k(Y.landmarks,C,A),fe=_e(T,X,C,A,ee,ue);return await H.runFromGPUBufferPipelined(ue),Re={...fe,srcWidth:C,srcHeight:A},q}}if(J.length>0){let z=[];for(let q of J){let Y=k(q.landmarks,C,A),T=await ht(Y,X,C,A,ee,ue,!0);T&&z.push({score:T.score,handedness:T.handedness,landmarks:T.landmarks,keypoints:Qt(T.landmarks)})}if(z.length>0){if(J=z.map(q=>({landmarks:q.landmarks,handedness:q.handedness})),z.length===1){let q=J[0],Y=k(q.landmarks,C,A),T=_e(Y,X,C,A,ee,ue);await H.runFromGPUBufferPipelined(ue),Re={...T,srcWidth:C,srcHeight:A}}return z}J=[],Re=null}let{detections:$,lbPadX:ce,lbPadY:de}=await Ue.detectRawWithResize(I,C,A);if(Fe=ce,be=de,$.length===0)return J=[],[];let N=[];for(let z of $){let q=mt(z),Y=Be(q,C,A),T=await ht(Y,X,C,A,ee,ue,!0);T&&N.push({score:T.score,handedness:T.handedness,landmarks:T.landmarks,keypoints:Qt(T.landmarks)})}return J=N.map(z=>({landmarks:z.landmarks,handedness:z.handedness})),N}function Wt(){Me&&Me.destroy(),ae&&ae.destroy(),Me=null,ae=null,ge=null,H.device.destroy(),Pe.device.destroy()}function Ae(){J=[]}return{detect:Ht,dispose:Wt,reset:Ae}}export{sn as LANDMARK_NAMES,qn as createHandpose};
