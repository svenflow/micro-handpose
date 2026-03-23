function nn(s,h){let n=new Map,T=s.dtype??"float32";for(let P=0;P<s.keys.length;P++){let a=s.keys[P],b=s.shapes[P],E=s.offsets[P],B=b.reduce((_,x)=>_*x,1),w,d;if(T==="float32")w=new Float32Array(h,E,B);else{let _=new DataView(h);w=new Float32Array(B);for(let x=0;x<B;x++)w[x]=Gn(_.getUint16(E+x*2,!0));d=h.slice(E,E+B*2)}n.set(a,{data:w,shape:b,rawF16:d})}return n}function Gn(s){let h=s>>15&1,n=s>>10&31,T=s&1023;if(n===0){if(T===0)return h?-0:0;let b=-14,E=T/1024;return(h?-1:1)*Math.pow(2,b)*E}if(n===31)return T===0?h?-1/0:1/0:NaN;let P=n-15,a=1+T/1024;return(h?-1:1)*Math.pow(2,P)*a}var Mn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Xn=Mn.map(([s,h,n,T,P])=>({type:"resmodule",inCh:s,outCh:h,h:n,w:n,stride:T,prefix:P}));function dn(s,h){let n=new Map,T=s.dtype??"float32",P=new Map;for(let a=0;a<s.keys.length;a++){let b=s.keys[a],E=s.shapes[a],B=s.offsets[a],w=E.reduce((F,p)=>F*p,1),d,_;if(T==="float32")d=new Float32Array(h,B,w);else{let F=new DataView(h);d=new Float32Array(w);for(let p=0;p<w;p++)d[p]=An(F.getUint16(B+p*2,!0));_=h.slice(B,B+w*2)}let x=P.get(b)??0;P.set(b,x+1);let X=x===0?b:`${b}__${x}`;n.set(X,{data:d,shape:E,rawF16:_})}return n}function An(s){let h=s>>15&1,n=s>>10&31,T=s&1023;return n===0?T===0?h?-0:0:(h?-1:1)*Math.pow(2,-14)*(T/1024):n===31?T===0?h?-1/0:1/0:NaN:(h?-1:1)*Math.pow(2,n-15)*(1+T/1024)}function ft(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Sn=ft(`
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
`),kn=ft(`
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
`),Tn=ft(`
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
`),En=ft(`
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
`),Dn=ft(`
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
`),Ln=ft(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Rn=ft(`
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
`),Hn=ft(`
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
`),Wn=ft(`
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
`),Gt=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],On=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function an(s,h){if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");let T=n.features.has("shader-f16"),P=T?["shader-f16"]:[],a=await n.requestDevice({requiredFeatures:P,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}}),b=s.values().next().value,E=T&&!!b?.rawF16&&!h?.forceF32;function B(g){if(E&&g.rawF16){let r=new Uint16Array(g.rawF16);if(r.length%2!==0){let f=new Uint16Array(r.length+1);return f.set(r),f}return r}return g.data}function w(g){return E&&g.rawF16?Math.ceil(g.rawF16.byteLength/4)*4:g.data.byteLength}let d=E?2:4;function _(g){if(!E)return g;let r=g;return r=r.replace(/array<f32>/g,"array<f16>"),r=r.replace(/array<f32,/g,"array<f16,"),r=r.replace(/var sum:f32=0\.0/g,"var sum:f16=0.0h"),r=r.replace(/var sum0:f32=0\.0/g,"var sum0:f16=0.0h"),r=r.replace(/var sum1:f32=0\.0/g,"var sum1:f16=0.0h"),r=r.replace(/var sum2:f32=0\.0/g,"var sum2:f16=0.0h"),r=r.replace(/var sum3:f32=0\.0/g,"var sum3:f16=0.0h"),r=r.replace(/\/f32\(params/g,"/f16(params"),r=r.replace(/,0\.0\),6\.0\)/g,",0.0h),6.0h)"),r=r.replace(/->f32\{/g,"->f16{"),r=r.replace(/->f32 \{/g,"->f16 {"),r=r.replace(/return 0\.0;/g,"return 0.0h;"),"enable f16;"+r}function x(g){if(!E)return g;let r=_(g);return r=r.replace("read>input:array<f16>","read>input:array<f32>"),r=r.replace(/input\[in_idx\]/g,"f16(input[in_idx])"),r}function X(g){if(!E)return g;let r=g;return r=r.replace("read>input:array<f32>","read>input:array<f16>"),r=r.replace("read>weight:array<f32>","read>weight:array<f16>"),r=r.replace("read>bias:array<f32>","read>bias:array<f16>"),r=r.replace(/input\[ic\]/g,"f32(input[ic])"),r=r.replace(/weight\[oc\*params\.in_features\+ic\]/g,"f32(weight[oc*params.in_features+ic])"),r=r.replace(/bias\[oc\]/g,"f32(bias[oc])"),"enable f16;"+r}let F={r:"read-only-storage",s:"storage",u:"uniform"};function p(g){return a.createBindGroupLayout({entries:g.map((r,f)=>r==="t"?{binding:f,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:f,visibility:GPUShaderStage.COMPUTE,buffer:{type:F[r]}})})}let L=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,U=GPUBufferUsage.STORAGE,he=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,De=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,Ue=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function A(g,r){return a.createBuffer({size:Math.max(g,4),usage:r})}function me(g,r){return a.createBindGroup({layout:g,entries:r.map((f,K)=>({binding:K,resource:"size"in f?{buffer:f}:f}))})}function Se(g,r){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[g]}),compute:{module:r,entryPoint:"main"}})}function re(g){let r=s.get(g);if(!r)throw new Error(`Missing weight: ${g}`);return r}let Dt=a.createShaderModule({code:Sn}),Lt=a.createShaderModule({code:x(kn)}),Be=a.createShaderModule({code:_(Tn)}),nt=a.createShaderModule({code:_(En)}),Ve=a.createShaderModule({code:_(Dn)}),Fe=a.createShaderModule({code:_(Ln)}),wt=a.createShaderModule({code:_(Rn)}),We=a.createShaderModule({code:X(Hn)}),at=a.createShaderModule({code:X(Wn)}),ke=p(["r","r","r","s","u"]),lt=p(["r","r","s","u"]),ge=p(["r","s","u"]),Le=p(["r","r","r","s","u"]),Rt=p(["t","s","u"]),Pt=Se(Rt,Dt),ht=Se(ke,Lt),Ht=Se(ke,Be),Oe=Se(ke,nt),Mt=Se(ke,Ve),At=Se(lt,Fe),It=Se(ge,wt),Wt=Se(Le,We),St=Se(Le,at),u=1152*112*112*4,M=A(u,De),C=A(u,De),S=A(u,U),q=A(u,U),be=A(u,L),te=A(672*224*4,De),$=A(1152*4,he),oe=A(252,he),ue=A(252,he),j=A(4,he),ne=A(4,he),W=A(260,De),V=A(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Y=a.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),ce=A(4,Ue);a.queue.writeBuffer(ce,0,new Uint32Array([224]));let pe=re("conv2d"),_e=re("batch_normalization"),Ge=B(pe),Te=B(_e),ze=A(w(pe),L),Ke=A(w(_e),L),ye=A(24,Ue);a.queue.writeBuffer(ze,0,Ge),a.queue.writeBuffer(Ke,0,Te),a.queue.writeBuffer(ye,0,new Uint32Array([3,24,224,224,112,112]));let Me=112,Ee=112,Ae=[];for(let g=0;g<Gt.length;g++){let r=Gt[g],f=On[g],K=Me,N=Ee,ae=r.stride===2?Math.floor(Me/2):Me,Q=r.stride===2?Math.floor(Ee/2):Ee,D={spec:r,inH:K,inW:N,outH:ae,outW:Q,dwW:A(4,L),dwB:A(4,L),dwU:A(32,Ue)};if(f.expandConvKey){let J=re(f.expandConvKey),le=re(f.expandBNKey);D.expandW=A(w(J),L),D.expandB=A(w(le),L),D.expandU=A(16,Ue),a.queue.writeBuffer(D.expandW,0,B(J)),a.queue.writeBuffer(D.expandB,0,B(le)),a.queue.writeBuffer(D.expandU,0,new Uint32Array([r.inCh,r.expandCh,K,N]))}let pt=re(f.dwWeightKey),Qe=re(f.dwBNKey);D.dwW=A(w(pt),L),D.dwB=A(w(Qe),L),a.queue.writeBuffer(D.dwW,0,B(pt)),a.queue.writeBuffer(D.dwB,0,B(Qe));let se=Math.floor((r.dwKernel-r.stride)/2);if(a.queue.writeBuffer(D.dwU,0,new Uint32Array([r.expandCh,K,N,ae,Q,r.stride,se,r.dwKernel])),r.hasProject&&f.projectConvKey){let J=re(f.projectConvKey),le=re(f.projectBNKey);D.projectW=A(w(J),L),D.projectB=A(w(le),L),D.projectU=A(16,Ue),a.queue.writeBuffer(D.projectW,0,B(J)),a.queue.writeBuffer(D.projectB,0,B(le)),a.queue.writeBuffer(D.projectU,0,new Uint32Array([r.expandCh,r.outCh,ae,Q]))}Ae.push(D),Me=ae,Ee=Q}function fe(g,r){let f=s.get(g);if(!f)throw new Error(`Missing weight: ${g}`);if(f.shape.length!==r)throw new Error(`Weight ${g} has rank ${f.shape.length}, expected ${r}`);return f}let Ie=re("conv_landmarks__1"),Xe=re("conv_world_landmarks__1"),Re=re("conv_handflag__1"),de=re("conv_handedness__1"),ie=re("Identity"),xe=re("Identity_1"),Ne=re("Identity_2"),Ye=re("Identity_3"),Je=A(w(Ie),L),rt=A(w(ie),L),je=A(w(Xe),L),$e=A(w(Ye),L),Ze=A(w(Re),L),ot=A(w(xe),L),it=A(w(de),L),st=A(w(Ne),L);a.queue.writeBuffer(Je,0,B(Ie)),a.queue.writeBuffer(rt,0,B(ie)),a.queue.writeBuffer(je,0,B(Xe)),a.queue.writeBuffer($e,0,B(Ye)),a.queue.writeBuffer(Ze,0,B(Re)),a.queue.writeBuffer(ot,0,B(xe)),a.queue.writeBuffer(it,0,B(de)),a.queue.writeBuffer(st,0,B(Ne));let Bt=A(8,Ue),kt=A(8,Ue),xt=A(8,Ue),Ct=A(8,Ue);a.queue.writeBuffer(Bt,0,new Uint32Array([1152,63])),a.queue.writeBuffer(kt,0,new Uint32Array([1152,63])),a.queue.writeBuffer(xt,0,new Uint32Array([1152,1])),a.queue.writeBuffer(Ct,0,new Uint32Array([1152,1]));let vt=A(8,Ue);a.queue.writeBuffer(vt,0,new Uint32Array([1152,Me*Ee]));let O=new Map;for(let g=0;g<Gt.length;g++)if(Gt[g].hasResidual){let r=Ae[g],f=A(4,Ue);a.queue.writeBuffer(f,0,new Uint32Array([Gt[g].outCh*r.outH*r.outW])),O.set(g,f)}let Z=me(Rt,[Y.createView(),M,ce]),Nt=me(ke,[M,ze,Ke,C,ye]),ut=new Float32Array(1),ct=new Float32Array(1),dt=new Float32Array(63);function Ut(g,r){let f=g.beginComputePass();f.setPipeline(ht),f.setBindGroup(0,Nt),f.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),f.end();let K=C,N=M;for(let ae=0;ae<Gt.length;ae++){let Q=Gt[ae],D=Ae[ae];if(Q.hasResidual){let se=Q.inCh*D.inH*D.inW*d;g.copyBufferToBuffer(K,0,be,0,se)}if(f=g.beginComputePass(),D.expandW){let se=me(ke,[K,D.expandW,D.expandB,S,D.expandU]);f.setPipeline(Ht),f.setBindGroup(0,se),f.dispatchWorkgroups(Math.ceil(D.inW/8),Math.ceil(D.inH/8),Q.expandCh)}let pt=D.expandW?S:K,Qe=me(ke,[pt,D.dwW,D.dwB,q,D.dwU]);if(f.setPipeline(Oe),f.setBindGroup(0,Qe),f.dispatchWorkgroups(Math.ceil(D.outW/8),Math.ceil(D.outH/8),Q.expandCh),Q.hasProject&&D.projectW){let se=(Q.hasResidual,N),J=me(ke,[q,D.projectW,D.projectB,se,D.projectU]);if(f.setPipeline(Mt),f.setBindGroup(0,J),f.dispatchWorkgroups(Math.ceil(D.outW/8),Math.ceil(D.outH/8),Q.outCh),Q.hasResidual){let le=O.get(ae),et=me(lt,[N,be,K,le]);f.setPipeline(At),f.setBindGroup(0,et),f.dispatchWorkgroups(Math.ceil(Q.outCh*D.outH*D.outW/256))}else{let le=K;K=N,N=le}}if(f.end(),!Q.hasProject){f=g.beginComputePass();let se=me(ge,[q,$,vt]);f.setPipeline(It),f.setBindGroup(0,se),f.dispatchWorkgroups(Math.ceil(1152/256));let J=me(Le,[$,Je,rt,oe,Bt]);f.setPipeline(Wt),f.setBindGroup(0,J),f.dispatchWorkgroups(1);let le=me(Le,[$,Ze,ot,j,xt]);f.setPipeline(St),f.setBindGroup(0,le),f.dispatchWorkgroups(1);let et=me(Le,[$,it,st,ne,Ct]);f.setPipeline(St),f.setBindGroup(0,et),f.dispatchWorkgroups(1),f.end(),g.copyBufferToBuffer(j,0,W,0,4),g.copyBufferToBuffer(ne,0,W,4,4),g.copyBufferToBuffer(oe,0,W,8,252),r&&g.copyBufferToBuffer(W,0,r,0,260);return}}}async function qt(g){a.queue.writeBuffer(te,0,g);let r=a.createCommandEncoder();r.copyBufferToBuffer(te,0,M,0,672*224*4),Ut(r,V),a.queue.submit([r.finish()]);let f=V.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await f;let K=new Float32Array(V.getMappedRange());ut[0]=K[0],ct[0]=K[1];for(let N=0;N<63;N++)dt[N]=K[2+N]/224;return V.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Tt(g){a.queue.copyExternalImageToTexture({source:g},{texture:Y},[224,224]);let r=a.createCommandEncoder();{let N=r.beginComputePass();N.setPipeline(Pt),N.setBindGroup(0,Z),N.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),N.end()}Ut(r,V),a.queue.submit([r.finish()]);let f=V.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await f;let K=new Float32Array(V.getMappedRange());ut[0]=K[0],ct[0]=K[1];for(let N=0;N<63;N++)dt[N]=K[2+N]/224;return V.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Zt(g){let r=a.createCommandEncoder();r.copyBufferToBuffer(g,0,M,0,672*224*4),Ut(r,V),a.queue.submit([r.finish()]);let f=V.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await f;let K=new Float32Array(V.getMappedRange());ut[0]=K[0],ct[0]=K[1];for(let N=0;N<63;N++)dt[N]=K[2+N]/224;return V.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Qt(){return null}async function Jt(){return null}async function Ot(g=100){let r=new OffscreenCanvas(224,224),f=r.getContext("2d");f.fillStyle="#886644",f.fillRect(0,0,224,224);for(let Q=0;Q<5;Q++)await Tt(r);let K=performance.now();for(let Q=0;Q<g;Q++)await Tt(r);let ae=(performance.now()-K)/g;return{avgMs:ae,fps:1e3/ae}}async function Yt(g=100){let r=await Ot(g);return{...r,medianMs:r.avgMs,minMs:r.avgMs}}async function jt(g){return Tt(g)}async function Vt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Xt(g){let r={};async function f(ae,Q,D){let pt=Q*4,Qe=a.createBuffer({size:pt,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),se=a.createCommandEncoder();se.copyBufferToBuffer(ae,0,Qe,0,pt),a.queue.submit([se.finish()]),await a.queue.onSubmittedWorkDone(),await Qe.mapAsync(GPUMapMode.READ);let J=new Float32Array(Qe.getMappedRange()),le=1/0,et=-1/0,$t=0;for(let z=0;z<J.length;z++)J[z]<le&&(le=J[z]),J[z]>et&&(et=J[z]),J[z]!==0&&$t++;let en=Array.from(J.slice(0,5));Qe.unmap(),Qe.destroy(),r[D]={min:le,max:et,nonZero:$t,total:Q,sample:en}}let K=new Float32Array(672*224);for(let ae=0;ae<50176;ae++)K[ae]=.5,K[50176+ae]=.3,K[448*224+ae]=.7;a.queue.writeBuffer(te,0,K);let N=a.createCommandEncoder();return N.copyBufferToBuffer(te,0,M,0,672*224*4),Ut(N,V),a.queue.submit([N.finish()]),await a.queue.onSubmittedWorkDone(),await f(M,672*224,"inputBufA"),await f(C,2688*112,"afterInitConvBufB"),await f($,1152,"gapOutput"),await f(oe,63,"landmarks"),await f(j,1,"handflag"),await f(W,65,"unifiedOutput"),r}return{device:a,run:qt,runFromCanvas:Tt,runFromGPUBuffer:Zt,runFromCanvasPipelined:Qt,flushPipelined:Jt,benchmark:Ot,benchmarkGPU:Yt,runFromCanvasViaRender:jt,benchmarkDiagnostic:Vt,debugLayerOutputs:Xt}}function mt(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var pn=mt(`
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
`),fn=mt(`
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
`),mn=mt(`
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
`),ln=mt(`
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
`),hn=mt(`
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
`),gn=mt(`
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
`),bn=mt(`
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
`),_n=mt(`
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
`),yn=mt(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function wn(s,h){let n;if(h)n=h;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");n=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let T={r:"read-only-storage",s:"storage",u:"uniform"};function P(e){return n.createBindGroupLayout({entries:e.map((t,i)=>t==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:t==="sm"?{binding:i,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:T[t]}})})}let a=n.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),b=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,E=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,B=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,w=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(e,t){return n.createBuffer({size:Math.max(e,4),usage:t})}function _(e,t,i){n.queue.writeBuffer(e,t,i)}function x(e){let t=d(e.data.byteLength,b);return _(t,0,e.data),t}let X=Array.from(s.keys());function F(e){let t=s.get(e);if(!t)throw new Error(`Weight not found: ${e}`);return t}function p(...e){let t=X.find(i=>e.every(v=>i.includes(v)));if(!t)throw new Error(`Weight not found for: ${e.join(", ")}`);return F(t)}function L(e){let[,t,i,v]=e.shape,R=new Float32Array(v*25);for(let y=0;y<v;y++)for(let ee=0;ee<t;ee++)for(let l=0;l<i;l++)R[y*25+ee*5+l]=e.data[ee*i*v+l*v+y];return R}function U(e){let[t,,,i]=e.shape,v=new Float32Array(t*i);for(let R=0;R<t;R++)for(let y=0;y<i;y++)v[R*i+y]=e.data[R*i+y];return v}let he=n.createShaderModule({code:pn}),De=n.createShaderModule({code:fn}),Ue=n.createShaderModule({code:mn}),A=n.createShaderModule({code:ln}),me=n.createShaderModule({code:gn}),Se=n.createShaderModule({code:hn}),re=n.createShaderModule({code:bn}),Dt=n.createShaderModule({code:_n}),Lt=n.createShaderModule({code:yn}),Be=P(["r","r","r","r","s","u"]),nt=P(["r","r","r","s","u"]),Ve=P(["r","r","r","r","r","s","u"]),Fe=P(["r","r","r","s","u"]),wt=P(["r","r","r","r","s","u"]),We=P(["r","r","s","u"]),at=P(["t","s","u"]),ke=P(["t","s","u","sm"]),lt=P(["s","u"]);function ge(e,t){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:t,entryPoint:"main"}})}let Le=ge(Be,he),Rt=ge(nt,De),Pt=ge(Ve,Ue),ht=ge(Fe,A),Ht=ge(wt,me),Oe=ge(We,Se),Mt=ge(at,re),At=ge(ke,Dt),It=ge(lt,Lt),Wt=p("conv2d/Conv2D"),St=p("batch_normalization/","conv2d/Conv2D"),Ft=p("p_re_lu/"),u=x(Wt),M=x(St),C=x(Ft),q=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(e=>{let t=p(e.dwKey),i=p(e.pwKey),v=p(e.bnKey),R=p(e.preluKey),y=L(t),ee=d(y.byteLength,b);_(ee,0,y);let l=new Float32Array(e.inCh),Ce=d(l.byteLength,b);_(Ce,0,l);let we=U(i),m=d(we.byteLength,b);_(m,0,we);let k=x(v),G=x(R);return{dwWeightBuf:ee,dwBiasBuf:Ce,pwWeightBuf:m,pwBiasBuf:k,alphaBuf:G,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),be=U(p("conv2d_25/Conv2D")),te=d(be.byteLength,b);_(te,0,be);let $=x(p("batch_normalization_25/")),oe=x(p("p_re_lu_25/")),ue={dwWeightBuf:(()=>{let e=L(p("depthwise_conv2d_24/")),t=d(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=d(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=U(p("conv2d_26/")),t=d(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(p("batch_normalization_26/")),alphaBuf:x(p("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},j={dwWeightBuf:(()=>{let e=L(p("depthwise_conv2d_25/")),t=d(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=d(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=U(p("conv2d_27/Conv2D1")),t=d(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(p("batch_normalization_27/")),alphaBuf:x(p("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},ne=U(p("conv2d_28/Conv2D")),W=d(ne.byteLength,b);_(W,0,ne);let V=x(p("batch_normalization_28/")),Y=x(p("p_re_lu_28/")),ce={dwWeightBuf:(()=>{let e=L(p("depthwise_conv2d_26/")),t=d(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=d(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=U(p("conv2d_29/")),t=d(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(p("batch_normalization_29/")),alphaBuf:x(p("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},pe={dwWeightBuf:(()=>{let e=L(p("depthwise_conv2d_27/")),t=d(e.byteLength,b);return _(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=d(e.byteLength,b);return _(t,0,e),t})(),pwWeightBuf:(()=>{let e=U(p("conv2d_30/Conv2D1")),t=d(e.byteLength,b);return _(t,0,e),t})(),pwBiasBuf:x(p("batch_normalization_30/")),alphaBuf:x(p("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},_e=U(p("classifier_palm_16_NO_PRUNING/Conv2D")),Ge=d(_e.byteLength,b);_(Ge,0,_e);let Te=x(p("classifier_palm_16_NO_PRUNING/BiasAdd")),ze=U(p("regressor_palm_16_NO_PRUNING/Conv2D")),Ke=d(ze.byteLength,b);_(Ke,0,ze);let ye=x(p("regressor_palm_16_NO_PRUNING/BiasAdd")),Me=U(p("classifier_palm_8_NO_PRUNING/Conv2D")),Ee=d(Me.byteLength,b);_(Ee,0,Me);let Ae=x(p("classifier_palm_8_NO_PRUNING/BiasAdd")),fe=U(p("regressor_palm_8_NO_PRUNING/Conv2D")),Ie=d(fe.byteLength,b);_(Ie,0,fe);let Xe=x(p("regressor_palm_8_NO_PRUNING/BiasAdd")),Re=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,de=d(36864*3*4,b),ie=d(Re,E),xe=d(Re,E),Ne=d(Re,E),Ye=d(576*256*4,E),Je=new Map;function rt(e){let t=Je.get(e);return t||(t=d(4,w),_(t,0,new Uint32Array([e])),Je.set(e,t)),t}let je=d(144*256*4,E|GPUBufferUsage.COPY_DST),$e=d(576*128*4,E|GPUBufferUsage.COPY_DST),Ze=d(864*4,B),ot=d(15552*4,B),it=d(576*2*4,B),st=d(576*36*4,B),Bt=d(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),kt=d(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),xt=d(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ct=d(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),vt=n.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function O(e,t){return Math.ceil(e/t)}function Z(e){let t=d(e.byteLength,w);return _(t,0,e),t}let Nt=Z(new Uint32Array([1,3,32,192,192,96,96])),ut=q.map(e=>{let t=e.stride===2?e.inH/2:e.inH,i=t,v=e.stride===2?1:2,R=e.inCh;return{dw:Z(new Uint32Array([1,e.inCh,e.inH,e.inH,t,i,e.stride,v])),pw:Z(new Uint32Array([1,e.inCh,e.outCh,t,i,R,e.stride,e.inH,e.inH])),outH:t,outW:i}}),ct=(()=>{let e=ue,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:Z(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:Z(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),dt=(()=>{let e=j,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:Z(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:Z(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Ut=(()=>{let e=ce,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:Z(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:Z(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),qt=(()=>{let e=pe,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:Z(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:Z(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Tt=Z(new Uint32Array([1,256,6,6,12,12])),Zt=Z(new Uint32Array([1,256,12,12,12,12])),Qt=Z(new Uint32Array([1,256,12,12,24,24])),Jt=Z(new Uint32Array([1,128,24,24,24,24])),Ot=Z(new Uint32Array([1,256,256,12,12])),Yt=Z(new Uint32Array([1,256,128,24,24])),jt=Z(new Uint32Array([1,256,6,12,12])),Vt=Z(new Uint32Array([1,256,108,12,12])),Xt=Z(new Uint32Array([1,128,2,24,24])),g=Z(new Uint32Array([1,128,36,24,24])),r=Z(new Uint32Array([192,192,192])),f=n.createBindGroup({layout:at,entries:[{binding:0,resource:vt.createView()},{binding:1,resource:{buffer:de}},{binding:2,resource:{buffer:r}}]}),K=null,N=0,ae=0,Q=d(32,w);function D(e,t){return K&&N===e&&ae===t||(K&&K.destroy(),K=n.createTexture({size:[e,t,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),N=e,ae=t),K}let pt=n.createBindGroup({layout:Be,entries:[{binding:0,resource:{buffer:de}},{binding:1,resource:{buffer:u}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:C}},{binding:4,resource:{buffer:ie}},{binding:5,resource:{buffer:Nt}}]});function Qe(e,t,i){}function se(e,t,i,v,R,y){let ee=y.outH,l=n.createBindGroup({layout:nt,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:t.dwWeightBuf}},{binding:2,resource:{buffer:t.dwBiasBuf}},{binding:3,resource:{buffer:Ne}},{binding:4,resource:{buffer:y.dw}}]}),Ce=e.beginComputePass();Ce.setPipeline(Rt),Ce.setBindGroup(0,l),Ce.dispatchWorkgroups(O(ee,8),O(y.outH,8),t.inCh),Ce.end(),t.inCh*y.outH*ee;let we=n.createBindGroup({layout:Ve,entries:[{binding:0,resource:{buffer:Ne}},{binding:1,resource:{buffer:R}},{binding:2,resource:{buffer:t.pwWeightBuf}},{binding:3,resource:{buffer:t.pwBiasBuf}},{binding:4,resource:{buffer:t.alphaBuf}},{binding:5,resource:{buffer:v}},{binding:6,resource:{buffer:y.pw}}]}),m=e.beginComputePass();m.setPipeline(Pt),m.setBindGroup(0,we),m.dispatchWorkgroups(O(ee,8),O(y.outH,8),t.outCh),m.end(),t.outCh*y.outH*ee}function J(e,t,i,v,R,y,ee,l,Ce){let we=n.createBindGroup({layout:Fe,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:v}},{binding:3,resource:{buffer:R}},{binding:4,resource:{buffer:y}}]}),m=e.beginComputePass();m.setPipeline(ht),m.setBindGroup(0,we),m.dispatchWorkgroups(O(Ce,8),O(l,8),ee),m.end()}function le(e,t,i,v,R,y,ee,l,Ce,we){let m=n.createBindGroup({layout:wt,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:v}},{binding:3,resource:{buffer:R}},{binding:4,resource:{buffer:y}},{binding:5,resource:{buffer:ee}}]}),k=e.beginComputePass();k.setPipeline(Ht),k.setBindGroup(0,m),k.dispatchWorkgroups(O(we,8),O(Ce,8),l),k.end()}async function et(e){36864*3;{let G=e.beginComputePass();G.setPipeline(Le),G.setBindGroup(0,pt),G.dispatchWorkgroups(O(96,8),O(96,8),32),G.end()}9216*32;let t=ie,i=xe;for(let G=0;G<q.length;G++){let I=q[G];se(e,I,t,i,t,ut[G]);let ve=t;t=i,i=ve,G===13&&e.copyBufferToBuffer(t,0,$e,0,576*128*4),G===18&&e.copyBufferToBuffer(t,0,je,0,144*256*4)}{let G=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Tt}}]}),I=e.beginComputePass();I.setPipeline(Oe),I.setBindGroup(0,G),I.dispatchWorkgroups(O(12,8),O(12,8),256),I.end()}{let G=t;t=i,i=G}144*256,le(e,t,te,$,oe,i,Ot,256,12,12);{let G=t;t=i,i=G}144*256;{let G=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:je}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Zt}}]}),I=e.beginComputePass();I.setPipeline(Oe),I.setBindGroup(0,G),I.dispatchWorkgroups(O(12,8),O(12,8),256),I.end()}{let G=t;t=i,i=G}144*256,se(e,ue,t,i,t,ct);{let G=t;t=i,i=G}se(e,j,t,i,t,dt);{let G=t;t=i,i=G}J(e,t,Ge,Te,Ze,jt,6,12,12),J(e,t,Ke,ye,ot,Vt,108,12,12);{let G=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Qt}}]}),I=e.beginComputePass();I.setPipeline(Oe),I.setBindGroup(0,G),I.dispatchWorkgroups(O(24,8),O(24,8),256),I.end()}{let G=t;t=i,i=G}576*256,le(e,t,W,V,Y,i,Yt,128,24,24);{let G=t;t=i,i=G}576*128;{let G=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:$e}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Jt}}]}),I=e.beginComputePass();I.setPipeline(Oe),I.setBindGroup(0,G),I.dispatchWorkgroups(O(24,8),O(24,8),128),I.end()}{let G=t;t=i,i=G}576*128,se(e,ce,t,i,t,Ut);{let G=t;t=i,i=G}se(e,pe,t,i,t,qt);{let G=t;t=i,i=G}J(e,t,Ee,Ae,it,Xt,2,24,24),576*2,J(e,t,Ie,Xe,st,g,36,24,24),576*36,n.queue.submit([e.finish()]);let v=n.createCommandEncoder();v.copyBufferToBuffer(Ze,0,Bt,0,864*4),v.copyBufferToBuffer(ot,0,kt,0,15552*4),v.copyBufferToBuffer(it,0,xt,0,576*2*4),v.copyBufferToBuffer(st,0,Ct,0,576*36*4),n.queue.submit([v.finish()]),await Promise.all([Bt.mapAsync(GPUMapMode.READ),kt.mapAsync(GPUMapMode.READ),xt.mapAsync(GPUMapMode.READ),Ct.mapAsync(GPUMapMode.READ)]);let R=new Float32Array(Bt.getMappedRange()).slice(),y=new Float32Array(kt.getMappedRange()).slice(),ee=new Float32Array(xt.getMappedRange()).slice(),l=new Float32Array(Ct.getMappedRange()).slice();Bt.unmap(),kt.unmap(),xt.unmap(),Ct.unmap();let Ce=2016,we=new Float32Array(Ce),m=new Float32Array(Ce*18),k=0;for(let G=0;G<12;G++)for(let I=0;I<12;I++)for(let ve=0;ve<6;ve++){we[k]=R[ve*144+G*12+I];for(let qe=0;qe<18;qe++){let o=ve*18+qe;m[k*18+qe]=y[o*144+G*12+I]}k++}for(let G=0;G<24;G++)for(let I=0;I<24;I++)for(let ve=0;ve<2;ve++){we[k]=ee[ve*576+G*24+I];for(let qe=0;qe<18;qe++){let o=ve*18+qe;m[k*18+qe]=l[o*576+G*24+I]}k++}return{scores:we,regressors:m}}async function $t(e){n.queue.copyExternalImageToTexture({source:e},{texture:vt},[192,192]);let t=n.createCommandEncoder();{let i=t.beginComputePass();i.setPipeline(Mt),i.setBindGroup(0,f),i.dispatchWorkgroups(O(192,16),O(192,16),1),i.end()}return et(t)}async function en(e,t,i){let v=Math.min(192/t,192/i),R=Math.round(t*v),y=Math.round(i*v),ee=Math.floor((192-R)/2),l=Math.floor((192-y)/2),Ce=ee/192,we=l/192,m=D(t,i),k;e instanceof HTMLVideoElement?k=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?k=await createImageBitmap(e,{colorSpaceConversion:"none"}):k=e,n.queue.copyExternalImageToTexture({source:k},{texture:m},[t,i]);let G=new ArrayBuffer(32),I=new Uint32Array(G),ve=new Float32Array(G);I[0]=t,I[1]=i,I[2]=192,I[3]=0,ve[4]=t/R,ve[5]=i/y,ve[6]=ee,ve[7]=l,n.queue.writeBuffer(Q,0,G);let qe=n.createBindGroup({layout:ke,entries:[{binding:0,resource:m.createView()},{binding:1,resource:{buffer:de}},{binding:2,resource:{buffer:Q}},{binding:3,resource:a}]}),o=n.createCommandEncoder();{let H=o.beginComputePass();H.setPipeline(At),H.setBindGroup(0,qe),H.dispatchWorkgroups(O(192,16),O(192,16),1),H.end()}return{output:await et(o),lbPadX:Ce,lbPadY:we}}async function z(e,t){let i=n.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),v=n.createCommandEncoder();v.copyBufferToBuffer(e,0,i,0,t*4),n.queue.submit([v.finish()]),await i.mapAsync(GPUMapMode.READ);let R=new Float32Array(i.getMappedRange()).slice();return i.unmap(),i.destroy(),R}async function Un(e,t,i){function v(o,c=1e3){let H=o.slice(0,c),Pe=Math.max(0,Math.floor(o.length/2)-250);return{min:Math.min(...H),max:Math.max(...H),mean:H.reduce((He,gt)=>He+gt,0)/H.length,nonZero:H.filter(He=>He!==0).length,sample:Array.from(H.slice(0,10)),data500:Array.from(o.slice(0,500)),dataMid500:Array.from(o.slice(Pe,Pe+500)),totalLength:o.length}}function R(o,c,H,Pe){let He=[],gt=Math.floor(H/2),bt=Math.floor(Pe/2),tt=H*Pe;for(let _t=0;_t<c&&He.length<500;_t++)for(let zt=-1;zt<=1&&He.length<500;zt++)for(let Et=-1;Et<=1&&He.length<500;Et++){let yt=gt+zt,tn=bt+Et;yt>=0&&yt<H&&tn>=0&&tn<Pe&&He.push(o[_t*tt+yt*Pe+tn])}return He}let y={},ee;if(e instanceof HTMLImageElement?(t=t??e.naturalWidth,i=i??e.naturalHeight,ee=await createImageBitmap(e,{colorSpaceConversion:"none"})):(t=t??e.width??192,i=i??e.height??192,ee=e),t!==192||i!==192){let o=Math.min(192/t,192/i),c=Math.round(t*o),H=Math.round(i*o),Pe=Math.floor((192-c)/2),He=Math.floor((192-H)/2),gt=D(t,i);n.queue.copyExternalImageToTexture({source:ee},{texture:gt},[t,i]);let bt=new ArrayBuffer(32),tt=new Uint32Array(bt),_t=new Float32Array(bt);tt[0]=t,tt[1]=i,tt[2]=192,tt[3]=0,_t[4]=t/c,_t[5]=i/H,_t[6]=Pe,_t[7]=He,n.queue.writeBuffer(Q,0,bt);let zt=n.createBindGroup({layout:ke,entries:[{binding:0,resource:gt.createView()},{binding:1,resource:{buffer:de}},{binding:2,resource:{buffer:Q}},{binding:3,resource:a}]});{let Et=n.createCommandEncoder(),yt=Et.beginComputePass();yt.setPipeline(At),yt.setBindGroup(0,zt),yt.dispatchWorkgroups(O(192,16),O(192,16),1),yt.end(),n.queue.submit([Et.finish()])}}else{n.queue.copyExternalImageToTexture({source:ee},{texture:vt},[192,192]);let o=Z(new Uint32Array([192,192,192])),c=n.createBindGroup({layout:at,entries:[{binding:0,resource:vt.createView()},{binding:1,resource:{buffer:de}},{binding:2,resource:{buffer:o}}]});{let H=n.createCommandEncoder(),Pe=H.beginComputePass();Pe.setPipeline(Mt),Pe.setBindGroup(0,c),Pe.dispatchWorkgroups(O(192,16),O(192,16),1),Pe.end(),n.queue.submit([H.finish()])}}{let o=await z(de,110592),c=v(o);c.dataCenter500=R(o,3,192,192),y.input=c}let l=n.createCommandEncoder(),Ce=n.createBindGroup({layout:Be,entries:[{binding:0,resource:{buffer:de}},{binding:1,resource:{buffer:u}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:C}},{binding:4,resource:{buffer:ie}},{binding:5,resource:{buffer:Nt}}]}),we=l.beginComputePass();we.setPipeline(Le),we.setBindGroup(0,Ce),we.dispatchWorkgroups(O(96,8),O(96,8),32),we.end(),n.queue.submit([l.finish()]);{let o=await z(ie,294912),c=v(o);c.dataCenter500=R(o,32,96,96),y.initConv=c}let m=ie,k=xe;for(let o=0;o<q.length;o++){let c=q[o];l=n.createCommandEncoder(),se(l,c,m,k,m,ut[o]),n.queue.submit([l.finish()]);let H=m;m=k,k=H;{let Pe=c.stride===2?c.inH/2:c.inH,He=Pe,gt=Pe*He*c.outCh,bt=await z(m,gt),tt=v(bt);tt.dataCenter500=R(bt,c.outCh,Pe,He),tt.spatialShape=[c.outCh,Pe,He],y[`block${o}`]=tt}o===13&&(l=n.createCommandEncoder(),l.copyBufferToBuffer(m,0,$e,0,576*128*4),n.queue.submit([l.finish()])),o===18&&(l=n.createCommandEncoder(),l.copyBufferToBuffer(m,0,je,0,144*256*4),n.queue.submit([l.finish()]))}l=n.createCommandEncoder();{let o=Z(new Uint32Array([1,256,6,6,12,12])),c=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:k}},{binding:3,resource:{buffer:o}}]}),H=l.beginComputePass();H.setPipeline(Oe),H.setBindGroup(0,c),H.dispatchWorkgroups(O(12,8),O(12,8),256),H.end()}n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,36864),c=v(o);c.dataCenter500=R(o,256,12,12),y.fpnUpsample6to12=c}l=n.createCommandEncoder(),le(l,m,te,$,oe,k,Ot,256,12,12),n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,36864),c=v(o);c.dataCenter500=R(o,256,12,12),y.fpn6to12Conv=c}{let o=await z(je,36864),c=v(o);c.dataCenter500=R(o,256,12,12),y.backbone12Skip=c}l=n.createCommandEncoder();{let o=Z(new Uint32Array([1,256,12,12,12,12])),c=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:je}},{binding:2,resource:{buffer:k}},{binding:3,resource:{buffer:o}}]}),H=l.beginComputePass();H.setPipeline(Oe),H.setBindGroup(0,c),H.dispatchWorkgroups(O(12,8),O(12,8),256),H.end()}n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,36864),c=v(o);c.dataCenter500=R(o,256,12,12),y.fpnAdd12=c}l=n.createCommandEncoder(),se(l,ue,m,k,m,ct),n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,36864),c=v(o);c.dataCenter500=R(o,256,12,12),y.fpn12Block1=c}l=n.createCommandEncoder(),se(l,j,m,k,m,dt),n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,36864),c=v(o);c.dataCenter500=R(o,256,12,12),y.fpn12Block2=c}l=n.createCommandEncoder(),J(l,m,Ge,Te,Ze,jt,6,12,12),n.queue.submit([l.finish()]);{let o=await z(Ze,864),c=v(o);c.dataCenter500=R(o,6,12,12),y.cls16=c}l=n.createCommandEncoder(),J(l,m,Ke,ye,ot,Vt,108,12,12),n.queue.submit([l.finish()]);{let o=await z(ot,15552),c=v(o,500);c.dataCenter500=R(o,108,12,12),y.reg16=c}l=n.createCommandEncoder();{let o=Z(new Uint32Array([1,256,12,12,24,24])),c=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Ye}},{binding:2,resource:{buffer:k}},{binding:3,resource:{buffer:o}}]}),H=l.beginComputePass();H.setPipeline(Oe),H.setBindGroup(0,c),H.dispatchWorkgroups(O(24,8),O(24,8),256),H.end()}n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,147456),c=v(o);c.dataCenter500=R(o,256,24,24),y.fpnUpsample12to24=c}l=n.createCommandEncoder(),le(l,m,W,V,Y,k,Yt,128,24,24),n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,73728),c=v(o);c.dataCenter500=R(o,128,24,24),y.fpn12to24Conv=c}{let o=await z($e,73728),c=v(o);c.dataCenter500=R(o,128,24,24),y.backbone24Skip=c}l=n.createCommandEncoder();{let o=Z(new Uint32Array([1,128,24,24,24,24])),c=n.createBindGroup({layout:We,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:$e}},{binding:2,resource:{buffer:k}},{binding:3,resource:{buffer:o}}]}),H=l.beginComputePass();H.setPipeline(Oe),H.setBindGroup(0,c),H.dispatchWorkgroups(O(24,8),O(24,8),128),H.end()}n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,73728),c=v(o);c.dataCenter500=R(o,128,24,24),y.fpnAdd24=c}l=n.createCommandEncoder(),se(l,ce,m,k,m,Ut),n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,73728),c=v(o);c.dataCenter500=R(o,128,24,24),y.fpn24Block1=c}l=n.createCommandEncoder(),se(l,pe,m,k,m,qt),n.queue.submit([l.finish()]);{let o=m;m=k,k=o}{let o=await z(m,73728),c=v(o);c.dataCenter500=R(o,128,24,24),y.fpn24Block2=c}l=n.createCommandEncoder(),J(l,m,Ee,Ae,it,Xt,2,24,24),n.queue.submit([l.finish()]);{let o=await z(it,1152),c=v(o);c.dataCenter500=R(o,2,24,24),y.cls8=c}l=n.createCommandEncoder(),J(l,m,Ie,Xe,st,g,36,24,24),n.queue.submit([l.finish()]);{let o=await z(st,20736),c=v(o);c.dataCenter500=R(o,36,24,24),y.reg8=c}y.initWeights=v(await z(u,100),100),y.initBias=v(await z(M,32),32),y.cls16Weights=v(await z(Ge,100),100),y.cls16Bias=v(await z(Te,6),6),y.cls8Weights=v(await z(Ee,100),100),y.cls8Bias=v(await z(Ae,2),2),y.fpn6to12Weights=v(await z(te,100),100);let G=await z(Ze,864),I=await z(it,576*2);y.rawScores=new Float32Array(2016),y.rawScores.set(G,0),y.rawScores.set(I,864);let ve=await z(ot,15552),qe=await z(st,576*36);return y.rawRegressors=new Float32Array(36288),y.rawRegressors.set(ve,0),y.rawRegressors.set(qe,15552),y.rawInput=await z(de,36864*3),y}return{device:n,run:$t,runWithResize:en,debugRun:Un}}function zn(){let s=[];for(let h=0;h<12;h++)for(let n=0;n<12;n++){let T=(n+.5)/12,P=(h+.5)/12;for(let a=0;a<6;a++)s.push({x:T,y:P})}for(let h=0;h<24;h++)for(let n=0;n<24;n++){let T=(n+.5)/24,P=(h+.5)/24;for(let a=0;a<2;a++)s.push({x:T,y:P})}return s}var Pn=zn();function Kn(s){return 1/(1+Math.exp(-s))}function rn(s,h){let n=[],{scores:T,regressors:P}=s,a=192;for(let b=0;b<Pn.length;b++){let E=Kn(T[b]);if(E<h)continue;let B=Pn[b],w=b*18,d=B.x+P[w+0]/a,_=B.y+P[w+1]/a,x=P[w+2]/a,X=P[w+3]/a,F=[];for(let p=0;p<7;p++){let L=B.x+P[w+4+p*2]/a,U=B.y+P[w+4+p*2+1]/a;F.push([L,U])}n.push({score:E,box:[d,_,x,X],keypoints:F})}return n}function on(s,h){if(s.length===0)return[];let n=[...s].sort((a,b)=>b.score-a.score),T=[],P=new Set;for(let a=0;a<n.length;a++){if(P.has(a))continue;let b=[a];for(let F=a+1;F<n.length;F++)P.has(F)||In(n[a],n[F])>h&&(b.push(F),P.add(F));let E=0,B=0,w=0,d=0,_=0,x=[];for(let F=0;F<7;F++)x.push([0,0]);for(let F of b){let p=n[F],L=p.score;E+=L,B+=p.box[0]*L,w+=p.box[1]*L,d+=p.box[2]*L,_+=p.box[3]*L;for(let U=0;U<7;U++)x[U][0]+=p.keypoints[U][0]*L,x[U][1]+=p.keypoints[U][1]*L}let X=1/E;T.push({score:n[a].score,box:[B*X,w*X,d*X,_*X],keypoints:x.map(([F,p])=>[F*X,p*X])})}return T}function In(s,h){let n=s.box[0]-s.box[2]/2,T=s.box[1]-s.box[3]/2,P=s.box[0]+s.box[2]/2,a=s.box[1]+s.box[3]/2,b=h.box[0]-h.box[2]/2,E=h.box[1]-h.box[3]/2,B=h.box[0]+h.box[2]/2,w=h.box[1]+h.box[3]/2,d=Math.max(n,b),_=Math.max(T,E),x=Math.min(P,B),X=Math.min(a,w),F=Math.max(0,x-d),p=Math.max(0,X-_),L=F*p,U=(P-n)*(a-T),he=(B-b)*(w-E),De=U+he-L;return De>0?L/De:0}function Fn(s){let[h,n,T,P]=s.box,a=s.keypoints[0],b=s.keypoints[2],E=b[0]-a[0],B=b[1]-a[1],w=Math.atan2(B,E),_=-Math.PI/2-w,x=Math.max(T,P),F=x*2.6,p=-.5*x,L=Math.cos(_),U=Math.sin(_),he=p*U,De=p*L;return{centerX:h+he,centerY:n+De,width:F,height:F,rotation:_}}function Bn(s,h={}){let{scoreThreshold:n=.5,nmsThreshold:T=.3,maxHands:P=2}=h;async function a(w){let d=await s.run(w),_=rn(d,n);return on(_,T).slice(0,P).map(Fn)}async function b(w){let d=await s.run(w),_=rn(d,n);return on(_,T).slice(0,P)}async function E(w,d,_){let{output:x,lbPadX:X,lbPadY:F}=await s.runWithResize(w,d,_),p=rn(x,n);return{detections:on(p,T).slice(0,P),lbPadX:X,lbPadY:F}}async function B(w,d,_){let{output:x,lbPadX:X,lbPadY:F}=await s.runWithResize(w,d,_);return{scores:x.scores,regressors:x.regressors,lbPadX:X,lbPadY:F}}return{detect:a,detectRaw:b,detectRawWithResize:E,detectRawSSD:B,model:s}}var sn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Kt(s){let h={};for(let n=0;n<sn.length;n++)h[sn[n]]=s[n];return h}function xn(s,h,n){return s.initialized?(s.value=n*h+(1-n)*s.value,s.value):(s.value=h,s.initialized=!0,h)}function Cn(s,h){let n=2*Math.PI*h*s;return n/(n+1)}function Nn(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function un(s,h,n,T,P,a){let b=s.lastTime<0?.03333333333333333:n-s.lastTime;s.lastTime=n;let E=Cn(b,a),B=s.x.initialized?(h-s.x.value)/b:0,w=xn(s.dx,B,E),d=T+P*Math.abs(w),_=Cn(b,d);return xn(s.x,h,_)}function cn(s={}){let{minCutoff:h=.05,beta:n=80,dCutoff:T=1}=s,P=[];function a(B){P.length!==B&&(P=Array.from({length:B},()=>Nn()))}function b(B,w){let d=w??performance.now()/1e3,_=B.length*3;return a(_),B.map((x,X)=>({x:un(P[X*3],x.x,d,h,n,T),y:un(P[X*3+1],x.y,d,h,n,T),z:un(P[X*3+2],x.z,d,h,n,T)}))}function E(){P=[]}return{apply:b,reset:E}}function qn(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Yn=qn(`
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
`);function vn(s){let h=s.createShaderModule({code:Yn}),n=s.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),T=s.createComputePipeline({layout:s.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:h,entryPoint:"main"}}),P=s.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),a=s.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),b=s.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),E=new Float32Array(8);function B(w,d,_,x,X,F,p){s.queue.writeBuffer(a,0,new Uint32Array([X,F,p,0])),E.set(x),s.queue.writeBuffer(b,0,E);let L=s.createBindGroup({layout:n,entries:[{binding:0,resource:d.createView()},{binding:1,resource:{buffer:_}},{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:b}},{binding:4,resource:P}]}),U=w.beginComputePass();U.setPipeline(T),U.setBindGroup(0,L),U.dispatchWorkgroups(Math.ceil(p/16),Math.ceil(p/16),1),U.end()}return{crop:B}}var jn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Vn(s={}){let{weightsUrl:h,scoreThreshold:n=.5,palmScoreThreshold:T=.5,maxHands:P=3,forceF32:a=!1}=s;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let b=(h??jn).replace(/\/$/,"")+"/",[E,B,w,d]=await Promise.all([fetch(`${b}weights_f16_full.json`),fetch(`${b}weights_f16_full.bin`),fetch(`${b}palm_detection_weights.json`),fetch(`${b}palm_detection_weights.bin`)]);if(!E.ok)throw new Error(`Failed to fetch landmark weights: ${E.status}`);if(!B.ok)throw new Error(`Failed to fetch landmark weights: ${B.status}`);if(!w.ok)throw new Error(`Failed to fetch palm detection weights: ${w.status}`);if(!d.ok)throw new Error(`Failed to fetch palm detection weights: ${d.status}`);let[_,x,X,F]=await Promise.all([E.json(),B.arrayBuffer(),w.json(),d.arrayBuffer()]),p=dn(_,x),L=nn(X,F),U=224,he=await an(p,{forceF32:a});{let u=new OffscreenCanvas(U,U),M=u.getContext("2d");M.fillStyle="#886644",M.fillRect(0,0,U,U),M.fillStyle="#cc9966",M.fillRect(50,50,124,124);let C=await he.runFromCanvas(u);C.landmarks.every(q=>q===0)&&C.handflag.every(q=>q===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let De=await wn(L),Ue=Bn(De,{scoreThreshold:T,maxHands:P}),A=[];for(let u=0;u<P;u++)A.push(cn());let me=0,Se=null,re=null;function Dt(){return Se||(Se=new OffscreenCanvas(192,192)),Se}function Lt(){return re||(re=new OffscreenCanvas(U,U)),re}let Be=he.device,nt=null,Ve=null,Fe=null,wt=0,We=0;function at(){return nt||(nt=vn(Be)),nt}function ke(){return Ve||(Ve=Be.createBuffer({size:3*U*U*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),Ve}function lt(u,M){return(!Fe||wt!==u||We!==M)&&(Fe&&Fe.destroy(),Fe=Be.createTexture({size:[u,M],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),wt=u,We=M),Fe}let ge=0,Le=0;function Rt(u,M,C){let S=Dt();S.width=192,S.height=192;let q=S.getContext("2d");q.clearRect(0,0,192,192);let be=Math.min(192/M,192/C),te=Math.round(M*be),$=Math.round(C*be),oe=(192-te)/2,ue=(192-$)/2;if(ge=oe/192,Le=ue/192,u instanceof ImageData){let j=new OffscreenCanvas(u.width,u.height);j.getContext("2d").putImageData(u,0,0),q.drawImage(j,oe,ue,te,$)}else q.drawImage(u,0,0,M,C,oe,ue,te,$);return S}function Pt(u){let M=1/(1-2*ge),C=1/(1-2*Le);return{score:u.score,box:[(u.box[0]-ge)*M,(u.box[1]-Le)*C,u.box[2]*M,u.box[3]*C],keypoints:u.keypoints.map(([S,q])=>[(S-ge)*M,(q-Le)*C])}}function ht(u,M,C){let S=u.keypoints[0],q=u.keypoints[2],be=(q[0]-S[0])*M,te=(q[1]-S[1])*C,$=Math.atan2(-te,be),ue=Math.PI/2-$,j=ue-2*Math.PI*Math.floor((ue+Math.PI)/(2*Math.PI)),[ne,W,V,Y]=u.box,ce=Math.cos(j),pe=Math.sin(j),_e=Y*C,Ge=ne+.5*_e*pe/M,Te=W+-.5*Y*ce,ye=Math.max(V*M,Y*C)*2.6;return{centerXpx:Ge*M,centerYpx:Te*C,sizePx:ye,rotation:j}}function Ht(u,M){let C=Lt();C.width=U,C.height=U;let S=C.getContext("2d");S.clearRect(0,0,U,U);let q=U/M.sizePx,be=Math.cos(M.rotation),te=Math.sin(M.rotation),$=be*q,oe=-te*q,ue=te*q,j=be*q,ne=U/2,W=-M.centerXpx*$-M.centerYpx*ue+ne,V=-M.centerXpx*oe-M.centerYpx*j+ne;if(S.setTransform($,oe,ue,j,W,V),u instanceof ImageData){let Y=new OffscreenCanvas(u.width,u.height);Y.getContext("2d").putImageData(u,0,0),S.drawImage(Y,0,0)}else S.drawImage(u,0,0);return S.setTransform(1,0,0,1,0,0),C}function Oe(u){return u instanceof HTMLCanvasElement||u instanceof OffscreenCanvas?[u.width,u.height]:typeof ImageBitmap<"u"&&u instanceof ImageBitmap?[u.width,u.height]:u instanceof ImageData?[u.width,u.height]:u instanceof HTMLVideoElement?[u.videoWidth,u.videoHeight]:u instanceof HTMLImageElement?[u.naturalWidth,u.naturalHeight]:[U,U]}async function Mt(u){let M=u,C,S;if(u instanceof HTMLVideoElement||u instanceof HTMLImageElement){let W=await createImageBitmap(u,{colorSpaceConversion:"none"});M=W,C=W.width,S=W.height}else[C,S]=Oe(u);let{detections:q,lbPadX:be,lbPadY:te}=await Ue.detectRawWithResize(M,C,S);if(ge=be,Le=te,q.length===0){if(me>0)for(let W=0;W<me&&W<A.length;W++)A[W].reset();return me=0,[]}let $=[],oe=at(),ue=ke(),j=lt(C,S),ne;M instanceof ImageData?ne=await createImageBitmap(M,{colorSpaceConversion:"none"}):ne=M,Be.queue.copyExternalImageToTexture({source:ne},{texture:j},[C,S]);for(let W of q){let V=Pt(W),Y=ht(V,C,S),ce=Math.cos(Y.rotation),pe=Math.sin(Y.rotation),_e=Y.sizePx/U,Ge=U/2,Te=ce*_e/C,ze=-pe*_e/C,Ke=Y.centerXpx/C-Ge*(Te+ze),ye=pe*_e/S,Me=ce*_e/S,Ee=Y.centerYpx/S-Ge*(ye+Me),Ae=Be.createCommandEncoder();oe.crop(Ae,j,ue,[Te,ze,Ke,ye,Me,Ee],C,S,U),Be.queue.submit([Ae.finish()]);let fe=await he.runFromGPUBuffer(ue),Ie=fe.handflag[0];if(Ie<n)continue;let Xe=fe.handedness[0]>.5,Re=[];for(let xe=0;xe<21;xe++){let Ne=fe.landmarks[xe*3],Ye=fe.landmarks[xe*3+1],Je=fe.landmarks[xe*3+2],rt=(Ne-.5)*Y.sizePx,je=(Ye-.5)*Y.sizePx,$e=ce*rt-pe*je+Y.centerXpx,Ze=pe*rt+ce*je+Y.centerYpx;Re.push({x:$e/C,y:Ze/S,z:Je})}let de=$.length,ie=de<A.length?A[de].apply(Re):Re;$.push({score:Ie,handedness:Xe?"right":"left",landmarks:ie,keypoints:Kt(ie)})}if($.length<me)for(let W=$.length;W<me;W++)W<A.length&&A[W].reset();return me=$.length,$}async function At(u){let M=u,C,S;if(u instanceof HTMLVideoElement||u instanceof HTMLImageElement){let W=await createImageBitmap(u,{colorSpaceConversion:"none"});M=W,C=W.width,S=W.height}else[C,S]=Oe(u);let{detections:q,lbPadX:be,lbPadY:te}=await Ue.detectRawWithResize(M,C,S);if(ge=be,Le=te,q.length===0)return[];let $=[],oe=at(),ue=ke(),j=lt(C,S),ne;M instanceof ImageData?ne=await createImageBitmap(M,{colorSpaceConversion:"none"}):ne=M,Be.queue.copyExternalImageToTexture({source:ne},{texture:j},[C,S]);for(let W of q){let V=Pt(W),Y=ht(V,C,S),ce=Math.cos(Y.rotation),pe=Math.sin(Y.rotation),_e=Y.sizePx/U,Ge=U/2,Te=ce*_e/C,ze=-pe*_e/C,Ke=Y.centerXpx/C-Ge*(Te+ze),ye=pe*_e/S,Me=ce*_e/S,Ee=Y.centerYpx/S-Ge*(ye+Me),Ae=Be.createCommandEncoder();oe.crop(Ae,j,ue,[Te,ze,Ke,ye,Me,Ee],C,S,U),Be.queue.submit([Ae.finish()]);let fe=await he.runFromGPUBuffer(ue),Ie=fe.handflag[0];if(Ie<n)continue;let Xe=fe.handedness[0]>.5,Re=[],de=[];for(let ie=0;ie<21;ie++){let xe=fe.landmarks[ie*3],Ne=fe.landmarks[ie*3+1],Ye=fe.landmarks[ie*3+2];Re.push({x:xe,y:Ne,z:Ye});let Je=(xe-.5)*Y.sizePx,rt=(Ne-.5)*Y.sizePx,je=ce*Je-pe*rt+Y.centerXpx,$e=pe*Je+ce*rt+Y.centerYpx;de.push({x:je/C,y:$e/S,z:Ye})}$.push({score:Ie,handedness:Xe?"right":"left",landmarks:de,keypoints:Kt(de),cropLandmarks:Re,roi:{...Y},palmDetection:{score:V.score,box:[...V.box],keypoints:V.keypoints.map(([ie,xe])=>[ie,xe])}})}return $}async function It(u,M){let[C,S]=Oe(u);if(M.length===0)return[];let q=[],be=at(),te=ke(),$=lt(C,S),oe;u instanceof ImageData?oe=await createImageBitmap(u,{colorSpaceConversion:"none"}):u instanceof HTMLImageElement?oe=await createImageBitmap(u,{colorSpaceConversion:"none"}):oe=u,Be.queue.copyExternalImageToTexture({source:oe},{texture:$},[C,S]);for(let ue of M){let j=ht(ue,C,S),ne=Math.cos(j.rotation),W=Math.sin(j.rotation),V=j.sizePx/U,Y=U/2,ce=ne*V/C,pe=-W*V/C,_e=j.centerXpx/C-Y*(ce+pe),Ge=W*V/S,Te=ne*V/S,ze=j.centerYpx/S-Y*(Ge+Te),Ke=Be.createCommandEncoder();be.crop(Ke,$,te,[ce,pe,_e,Ge,Te,ze],C,S,U),Be.queue.submit([Ke.finish()]);let ye=await he.runFromGPUBuffer(te),Me=ye.handflag[0];if(Me<n)continue;let Ee=ye.handedness[0]>.5,Ae=[];for(let fe=0;fe<21;fe++){let Ie=ye.landmarks[fe*3],Xe=ye.landmarks[fe*3+1],Re=ye.landmarks[fe*3+2],de=(Ie-.5)*j.sizePx,ie=(Xe-.5)*j.sizePx,xe=ne*de-W*ie+j.centerXpx,Ne=W*de+ne*ie+j.centerYpx;Ae.push({x:xe/C,y:Ne/S,z:Re})}q.push({score:Me,handedness:Ee?"right":"left",landmarks:Ae,keypoints:Kt(Ae)})}return q}function Wt(){Fe&&Fe.destroy(),Ve&&Ve.destroy(),Fe=null,Ve=null,nt=null,he.device.destroy(),De.device.destroy(),Se=null,re=null}let St={palmDetector:Ue,palmModel:De,landmarkModel:he,removeLetterbox:Pt,detectionToPixelROI:ht,cropHandRegion:Ht};function Ft(){for(let u of A)u.reset();me=0}return{detect:Mt,detectWithDebug:At,detectFromDetections:It,dispose:Wt,reset:Ft,_debug:St}}export{sn as LANDMARK_NAMES,an as compileFullModel,Vn as createHandpose,cn as createLandmarkSmoother,nn as loadWeightsFromBuffer,Kt as toKeypoints};
