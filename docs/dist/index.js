function nn(s,g){let n=new Map,E=s.dtype??"float32";for(let C=0;C<s.keys.length;C++){let a=s.keys[C],_=s.shapes[C],D=s.offsets[C],v=_.reduce((w,U)=>w*U,1),B,d;if(E==="float32")B=new Float32Array(g,D,v);else{let w=new DataView(g);B=new Float32Array(v);for(let U=0;U<v;U++)B[U]=Un(w.getUint16(D+U*2,!0));d=g.slice(D,D+v*2)}n.set(a,{data:B,shape:_,rawF16:d})}return n}function Un(s){let g=s>>15&1,n=s>>10&31,E=s&1023;if(n===0){if(E===0)return g?-0:0;let _=-14,D=E/1024;return(g?-1:1)*Math.pow(2,_)*D}if(n===31)return E===0?g?-1/0:1/0:NaN;let C=n-15,a=1+E/1024;return(g?-1:1)*Math.pow(2,C)*a}var Mn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Xn=Mn.map(([s,g,n,E,C])=>({type:"resmodule",inCh:s,outCh:g,h:n,w:n,stride:E,prefix:C}));function cn(s,g){let n=new Map,E=s.dtype??"float32",C=new Map;for(let a=0;a<s.keys.length;a++){let _=s.keys[a],D=s.shapes[a],v=s.offsets[a],B=D.reduce((q,p)=>q*p,1),d,w;if(E==="float32")d=new Float32Array(g,v,B);else{let q=new DataView(g);d=new Float32Array(B);for(let p=0;p<B;p++)d[p]=Gn(q.getUint16(v+p*2,!0));w=g.slice(v,v+B*2)}let U=C.get(_)??0;C.set(_,U+1);let Q=U===0?_:`${_}__${U}`;n.set(Q,{data:d,shape:D,rawF16:w})}return n}function Gn(s){let g=s>>15&1,n=s>>10&31,E=s&1023;return n===0?E===0?g?-0:0:(g?-1:1)*Math.pow(2,-14)*(E/1024):n===31?E===0?g?-1/0:1/0:NaN:(g?-1:1)*Math.pow(2,n-15)*(1+E/1024)}function ft(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var An=ft(`
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
`),Sn=ft(`
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
`),Tn=ft(`
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
`),En=ft(`
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
`),Dn=ft(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Ln=ft(`
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
`),Rn=ft(`
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
`),Hn=ft(`
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
`),Gt=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],Wn=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function an(s,g){if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");let E=n.features.has("shader-f16"),C=E?["shader-f16"]:[],a=await n.requestDevice({requiredFeatures:C,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}}),_=s.values().next().value,D=E&&!!_?.rawF16&&!g?.forceF32;function v(b){if(D&&b.rawF16){let r=new Uint16Array(b.rawF16);if(r.length%2!==0){let f=new Uint16Array(r.length+1);return f.set(r),f}return r}return b.data}function B(b){return D&&b.rawF16?Math.ceil(b.rawF16.byteLength/4)*4:b.data.byteLength}let d=D?2:4;function w(b){if(!D)return b;let r=b;return r=r.replace(/array<f32>/g,"array<f16>"),r=r.replace(/array<f32,/g,"array<f16,"),r=r.replace(/var sum:f32=0\.0/g,"var sum:f16=0.0h"),r=r.replace(/var sum0:f32=0\.0/g,"var sum0:f16=0.0h"),r=r.replace(/var sum1:f32=0\.0/g,"var sum1:f16=0.0h"),r=r.replace(/var sum2:f32=0\.0/g,"var sum2:f16=0.0h"),r=r.replace(/var sum3:f32=0\.0/g,"var sum3:f16=0.0h"),r=r.replace(/\/f32\(params/g,"/f16(params"),r=r.replace(/,0\.0\),6\.0\)/g,",0.0h),6.0h)"),r=r.replace(/->f32\{/g,"->f16{"),r=r.replace(/->f32 \{/g,"->f16 {"),r=r.replace(/return 0\.0;/g,"return 0.0h;"),"enable f16;"+r}function U(b){if(!D)return b;let r=w(b);return r=r.replace("read>input:array<f16>","read>input:array<f32>"),r=r.replace(/input\[in_idx\]/g,"f16(input[in_idx])"),r}function Q(b){if(!D)return b;let r=b;return r=r.replace("read>input:array<f32>","read>input:array<f16>"),r=r.replace("read>weight:array<f32>","read>weight:array<f16>"),r=r.replace("read>bias:array<f32>","read>bias:array<f16>"),r=r.replace(/input\[ic\]/g,"f32(input[ic])"),r=r.replace(/weight\[oc\*params\.in_features\+ic\]/g,"f32(weight[oc*params.in_features+ic])"),r=r.replace(/bias\[oc\]/g,"f32(bias[oc])"),"enable f16;"+r}let q={r:"read-only-storage",s:"storage",u:"uniform"};function p(b){return a.createBindGroupLayout({entries:b.map((r,f)=>r==="t"?{binding:f,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:f,visibility:GPUShaderStage.COMPUTE,buffer:{type:q[r]}})})}let R=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,G=GPUBufferUsage.STORAGE,ye=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,Le=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,Ge=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function k(b,r){return a.createBuffer({size:Math.max(b,4),usage:r})}function Oe(b,r){return a.createBindGroup({layout:b,entries:r.map((f,F)=>({binding:F,resource:"size"in f?{buffer:f}:f}))})}function Te(b,r){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[b]}),compute:{module:r,entryPoint:"main"}})}function fe(b){let r=s.get(b);if(!r)throw new Error(`Missing weight: ${b}`);return r}let Lt=a.createShaderModule({code:An}),Rt=a.createShaderModule({code:U(kn)}),Ce=a.createShaderModule({code:w(Sn)}),nt=a.createShaderModule({code:w(Tn)}),$e=a.createShaderModule({code:w(En)}),Ye=a.createShaderModule({code:w(Dn)}),wt=a.createShaderModule({code:w(Ln)}),ze=a.createShaderModule({code:Q(Rn)}),at=a.createShaderModule({code:Q(Hn)}),Ee=p(["r","r","r","s","u"]),lt=p(["r","r","s","u"]),we=p(["r","s","u"]),Re=p(["r","r","r","s","u"]),Ht=p(["t","s","u"]),Pt=Te(Ht,Lt),ht=Te(Ee,Rt),Wt=Te(Ee,Ce),Ke=Te(Ee,nt),Bt=Te(Ee,$e),At=Te(lt,Ye),It=Te(we,wt),Ot=Te(Re,ze),kt=Te(Re,at),Ze=1152*112*112*4,u=k(Ze,Le),x=k(Ze,Le),l=k(Ze,G),y=k(Ze,G),Y=k(Ze,R),te=k(672*224*4,Le),X=k(1152*4,ye),Z=k(252,ye),ne=k(252,ye),J=k(4,ye),j=k(4,ye),$=k(260,Le),T=k(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),ee=a.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),O=k(4,Ge);a.queue.writeBuffer(O,0,new Uint32Array([224]));let I=fe("conv2d"),me=fe("batch_normalization"),oe=v(I),he=v(me),ge=k(B(I),R),ve=k(B(me),R),be=k(24,Ge);a.queue.writeBuffer(ge,0,oe),a.queue.writeBuffer(ve,0,he),a.queue.writeBuffer(be,0,new Uint32Array([3,24,224,224,112,112]));let ce=112,de=112,Ae=[];for(let b=0;b<Gt.length;b++){let r=Gt[b],f=Wn[b],F=ce,V=de,pe=r.stride===2?Math.floor(ce/2):ce,re=r.stride===2?Math.floor(de/2):de,L={spec:r,inH:F,inW:V,outH:pe,outW:re,dwW:k(4,R),dwB:k(4,R),dwU:k(32,Ge)};if(f.expandConvKey){let ie=fe(f.expandConvKey),_e=fe(f.expandBNKey);L.expandW=k(B(ie),R),L.expandB=k(B(_e),R),L.expandU=k(16,Ge),a.queue.writeBuffer(L.expandW,0,v(ie)),a.queue.writeBuffer(L.expandB,0,v(_e)),a.queue.writeBuffer(L.expandU,0,new Uint32Array([r.inCh,r.expandCh,F,V]))}let pt=fe(f.dwWeightKey),Je=fe(f.dwBNKey);L.dwW=k(B(pt),R),L.dwB=k(B(Je),R),a.queue.writeBuffer(L.dwW,0,v(pt)),a.queue.writeBuffer(L.dwB,0,v(Je));let le=Math.floor((r.dwKernel-r.stride)/2);if(a.queue.writeBuffer(L.dwU,0,new Uint32Array([r.expandCh,F,V,pe,re,r.stride,le,r.dwKernel])),r.hasProject&&f.projectConvKey){let ie=fe(f.projectConvKey),_e=fe(f.projectBNKey);L.projectW=k(B(ie),R),L.projectB=k(B(_e),R),L.projectU=k(16,Ge),a.queue.writeBuffer(L.projectW,0,v(ie)),a.queue.writeBuffer(L.projectB,0,v(_e)),a.queue.writeBuffer(L.projectU,0,new Uint32Array([r.expandCh,r.outCh,pe,re]))}Ae.push(L),ce=pe,de=re}function De(b,r){let f=s.get(b);if(!f)throw new Error(`Missing weight: ${b}`);if(f.shape.length!==r)throw new Error(`Weight ${b} has rank ${f.shape.length}, expected ${r}`);return f}let ue=fe("conv_landmarks__1"),Ie=fe("conv_world_landmarks__1"),Fe=fe("conv_handflag__1"),Pe=fe("conv_handedness__1"),ke=fe("Identity"),Se=fe("Identity_1"),He=fe("Identity_2"),Ne=fe("Identity_3"),Qe=k(B(ue),R),rt=k(B(ke),R),je=k(B(Ie),R),Xe=k(B(Ne),R),qe=k(B(Fe),R),ot=k(B(Se),R),it=k(B(Pe),R),st=k(B(He),R);a.queue.writeBuffer(Qe,0,v(ue)),a.queue.writeBuffer(rt,0,v(ke)),a.queue.writeBuffer(je,0,v(Ie)),a.queue.writeBuffer(Xe,0,v(Ne)),a.queue.writeBuffer(qe,0,v(Fe)),a.queue.writeBuffer(ot,0,v(Se)),a.queue.writeBuffer(it,0,v(Pe)),a.queue.writeBuffer(st,0,v(He));let xt=k(8,Ge),St=k(8,Ge),Ct=k(8,Ge),vt=k(8,Ge);a.queue.writeBuffer(xt,0,new Uint32Array([1152,63])),a.queue.writeBuffer(St,0,new Uint32Array([1152,63])),a.queue.writeBuffer(Ct,0,new Uint32Array([1152,1])),a.queue.writeBuffer(vt,0,new Uint32Array([1152,1]));let Ut=k(8,Ge);a.queue.writeBuffer(Ut,0,new Uint32Array([1152,ce*de]));let z=new Map;for(let b=0;b<Gt.length;b++)if(Gt[b].hasResidual){let r=Ae[b],f=k(4,Ge);a.queue.writeBuffer(f,0,new Uint32Array([Gt[b].outCh*r.outH*r.outW])),z.set(b,f)}let ae=Oe(Ht,[ee.createView(),u,O]),Nt=Oe(Ee,[u,ge,ve,x,be]),ut=new Float32Array(1),ct=new Float32Array(1),dt=new Float32Array(63);function Mt(b,r){let f=b.beginComputePass();f.setPipeline(ht),f.setBindGroup(0,Nt),f.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),f.end();let F=x,V=u;for(let pe=0;pe<Gt.length;pe++){let re=Gt[pe],L=Ae[pe];if(re.hasResidual){let le=re.inCh*L.inH*L.inW*d;b.copyBufferToBuffer(F,0,Y,0,le)}if(f=b.beginComputePass(),L.expandW){let le=Oe(Ee,[F,L.expandW,L.expandB,l,L.expandU]);f.setPipeline(Wt),f.setBindGroup(0,le),f.dispatchWorkgroups(Math.ceil(L.inW/8),Math.ceil(L.inH/8),re.expandCh)}let pt=L.expandW?l:F,Je=Oe(Ee,[pt,L.dwW,L.dwB,y,L.dwU]);if(f.setPipeline(Ke),f.setBindGroup(0,Je),f.dispatchWorkgroups(Math.ceil(L.outW/8),Math.ceil(L.outH/8),re.expandCh),re.hasProject&&L.projectW){let le=(re.hasResidual,V),ie=Oe(Ee,[y,L.projectW,L.projectB,le,L.projectU]);if(f.setPipeline(Bt),f.setBindGroup(0,ie),f.dispatchWorkgroups(Math.ceil(L.outW/8),Math.ceil(L.outH/8),re.outCh),re.hasResidual){let _e=z.get(pe),et=Oe(lt,[V,Y,F,_e]);f.setPipeline(At),f.setBindGroup(0,et),f.dispatchWorkgroups(Math.ceil(re.outCh*L.outH*L.outW/256))}else{let _e=F;F=V,V=_e}}if(f.end(),!re.hasProject){f=b.beginComputePass();let le=Oe(we,[y,X,Ut]);f.setPipeline(It),f.setBindGroup(0,le),f.dispatchWorkgroups(Math.ceil(1152/256));let ie=Oe(Re,[X,Qe,rt,Z,xt]);f.setPipeline(Ot),f.setBindGroup(0,ie),f.dispatchWorkgroups(1);let _e=Oe(Re,[X,qe,ot,J,Ct]);f.setPipeline(kt),f.setBindGroup(0,_e),f.dispatchWorkgroups(1);let et=Oe(Re,[X,it,st,j,vt]);f.setPipeline(kt),f.setBindGroup(0,et),f.dispatchWorkgroups(1),f.end(),b.copyBufferToBuffer(J,0,$,0,4),b.copyBufferToBuffer(j,0,$,4,4),b.copyBufferToBuffer(Z,0,$,8,252),r&&b.copyBufferToBuffer($,0,r,0,260);return}}}async function qt(b){a.queue.writeBuffer(te,0,b);let r=a.createCommandEncoder();r.copyBufferToBuffer(te,0,u,0,672*224*4),Mt(r,T),a.queue.submit([r.finish()]);let f=T.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await f;let F=new Float32Array(T.getMappedRange());ut[0]=F[0],ct[0]=F[1];for(let V=0;V<63;V++)dt[V]=F[2+V]/224;return T.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Tt(b){a.queue.copyExternalImageToTexture({source:b},{texture:ee},[224,224]);let r=a.createCommandEncoder();{let V=r.beginComputePass();V.setPipeline(Pt),V.setBindGroup(0,ae),V.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),V.end()}Mt(r,T),a.queue.submit([r.finish()]);let f=T.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await f;let F=new Float32Array(T.getMappedRange());ut[0]=F[0],ct[0]=F[1];for(let V=0;V<63;V++)dt[V]=F[2+V]/224;return T.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Zt(b){let r=a.createCommandEncoder();r.copyBufferToBuffer(b,0,u,0,672*224*4),Mt(r,T),a.queue.submit([r.finish()]);let f=T.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await f;let F=new Float32Array(T.getMappedRange());ut[0]=F[0],ct[0]=F[1];for(let V=0;V<63;V++)dt[V]=F[2+V]/224;return T.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Qt(){return null}async function Jt(){return null}async function zt(b=100){let r=new OffscreenCanvas(224,224),f=r.getContext("2d");f.fillStyle="#886644",f.fillRect(0,0,224,224);for(let re=0;re<5;re++)await Tt(r);let F=performance.now();for(let re=0;re<b;re++)await Tt(r);let pe=(performance.now()-F)/b;return{avgMs:pe,fps:1e3/pe}}async function Yt(b=100){let r=await zt(b);return{...r,medianMs:r.avgMs,minMs:r.avgMs}}async function jt(b){return Tt(b)}async function Vt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Xt(b){let r={};async function f(pe,re,L){let pt=re*4,Je=a.createBuffer({size:pt,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),le=a.createCommandEncoder();le.copyBufferToBuffer(pe,0,Je,0,pt),a.queue.submit([le.finish()]),await a.queue.onSubmittedWorkDone(),await Je.mapAsync(GPUMapMode.READ);let ie=new Float32Array(Je.getMappedRange()),_e=1/0,et=-1/0,$t=0;for(let K=0;K<ie.length;K++)ie[K]<_e&&(_e=ie[K]),ie[K]>et&&(et=ie[K]),ie[K]!==0&&$t++;let en=Array.from(ie.slice(0,5));Je.unmap(),Je.destroy(),r[L]={min:_e,max:et,nonZero:$t,total:re,sample:en}}let F=new Float32Array(672*224);for(let pe=0;pe<50176;pe++)F[pe]=.5,F[50176+pe]=.3,F[448*224+pe]=.7;a.queue.writeBuffer(te,0,F);let V=a.createCommandEncoder();return V.copyBufferToBuffer(te,0,u,0,672*224*4),Mt(V,T),a.queue.submit([V.finish()]),await a.queue.onSubmittedWorkDone(),await f(u,672*224,"inputBufA"),await f(x,2688*112,"afterInitConvBufB"),await f(X,1152,"gapOutput"),await f(Z,63,"landmarks"),await f(J,1,"handflag"),await f($,65,"unifiedOutput"),r}return{device:a,run:qt,runFromCanvas:Tt,runFromGPUBuffer:Zt,runFromCanvasPipelined:Qt,flushPipelined:Jt,benchmark:zt,benchmarkGPU:Yt,runFromCanvasViaRender:jt,benchmarkDiagnostic:Vt,debugLayerOutputs:Xt}}function mt(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var dn=mt(`
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
`),pn=mt(`
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
`),fn=mt(`
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
`),mn=mt(`
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
`),ln=mt(`
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
`),hn=mt(`
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
`),gn=mt(`
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
`),bn=mt(`
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
`),_n=mt(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function yn(s,g){let n;if(g)n=g;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");n=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let E={r:"read-only-storage",s:"storage",u:"uniform"};function C(e){return n.createBindGroupLayout({entries:e.map((t,i)=>t==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:t==="sm"?{binding:i,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:E[t]}})})}let a=n.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),_=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,D=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,v=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,B=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function d(e,t){return n.createBuffer({size:Math.max(e,4),usage:t})}function w(e,t,i){n.queue.writeBuffer(e,t,i)}function U(e){let t=d(e.data.byteLength,_);return w(t,0,e.data),t}let Q=Array.from(s.keys());function q(e){let t=s.get(e);if(!t)throw new Error(`Weight not found: ${e}`);return t}function p(...e){let t=Q.find(i=>e.every(M=>i.includes(M)));if(!t)throw new Error(`Weight not found for: ${e.join(", ")}`);return q(t)}function R(e){let[,t,i,M]=e.shape,H=new Float32Array(M*25);for(let P=0;P<M;P++)for(let se=0;se<t;se++)for(let h=0;h<i;h++)H[P*25+se*5+h]=e.data[se*i*M+h*M+P];return H}function G(e){let[t,,,i]=e.shape,M=new Float32Array(t*i);for(let H=0;H<t;H++)for(let P=0;P<i;P++)M[H*i+P]=e.data[H*i+P];return M}let ye=n.createShaderModule({code:dn}),Le=n.createShaderModule({code:pn}),Ge=n.createShaderModule({code:fn}),k=n.createShaderModule({code:mn}),Oe=n.createShaderModule({code:hn}),Te=n.createShaderModule({code:ln}),fe=n.createShaderModule({code:gn}),Lt=n.createShaderModule({code:bn}),Rt=n.createShaderModule({code:_n}),Ce=C(["r","r","r","r","s","u"]),nt=C(["r","r","r","s","u"]),$e=C(["r","r","r","r","r","s","u"]),Ye=C(["r","r","r","s","u"]),wt=C(["r","r","r","r","s","u"]),ze=C(["r","r","s","u"]),at=C(["t","s","u"]),Ee=C(["t","s","u","sm"]),lt=C(["s","u"]);function we(e,t){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:t,entryPoint:"main"}})}let Re=we(Ce,ye),Ht=we(nt,Le),Pt=we($e,Ge),ht=we(Ye,k),Wt=we(wt,Oe),Ke=we(ze,Te),Bt=we(at,fe),At=we(Ee,Lt),It=we(lt,Rt),Ot=p("conv2d/Conv2D"),kt=p("batch_normalization/","conv2d/Conv2D"),Ft=p("p_re_lu/"),Ze=U(Ot),u=U(kt),x=U(Ft),y=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(e=>{let t=p(e.dwKey),i=p(e.pwKey),M=p(e.bnKey),H=p(e.preluKey),P=R(t),se=d(P.byteLength,_);w(se,0,P);let h=new Float32Array(e.inCh),Ue=d(h.byteLength,_);w(Ue,0,h);let Be=G(i),m=d(Be.byteLength,_);w(m,0,Be);let S=U(M),A=U(H);return{dwWeightBuf:se,dwBiasBuf:Ue,pwWeightBuf:m,pwBiasBuf:S,alphaBuf:A,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),Y=G(p("conv2d_25/Conv2D")),te=d(Y.byteLength,_);w(te,0,Y);let X=U(p("batch_normalization_25/")),Z=U(p("p_re_lu_25/")),ne={dwWeightBuf:(()=>{let e=R(p("depthwise_conv2d_24/")),t=d(e.byteLength,_);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=d(e.byteLength,_);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=G(p("conv2d_26/")),t=d(e.byteLength,_);return w(t,0,e),t})(),pwBiasBuf:U(p("batch_normalization_26/")),alphaBuf:U(p("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},J={dwWeightBuf:(()=>{let e=R(p("depthwise_conv2d_25/")),t=d(e.byteLength,_);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=d(e.byteLength,_);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=G(p("conv2d_27/Conv2D1")),t=d(e.byteLength,_);return w(t,0,e),t})(),pwBiasBuf:U(p("batch_normalization_27/")),alphaBuf:U(p("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},j=G(p("conv2d_28/Conv2D")),$=d(j.byteLength,_);w($,0,j);let T=U(p("batch_normalization_28/")),ee=U(p("p_re_lu_28/")),O={dwWeightBuf:(()=>{let e=R(p("depthwise_conv2d_26/")),t=d(e.byteLength,_);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=d(e.byteLength,_);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=G(p("conv2d_29/")),t=d(e.byteLength,_);return w(t,0,e),t})(),pwBiasBuf:U(p("batch_normalization_29/")),alphaBuf:U(p("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},I={dwWeightBuf:(()=>{let e=R(p("depthwise_conv2d_27/")),t=d(e.byteLength,_);return w(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=d(e.byteLength,_);return w(t,0,e),t})(),pwWeightBuf:(()=>{let e=G(p("conv2d_30/Conv2D1")),t=d(e.byteLength,_);return w(t,0,e),t})(),pwBiasBuf:U(p("batch_normalization_30/")),alphaBuf:U(p("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},me=G(p("classifier_palm_16_NO_PRUNING/Conv2D")),oe=d(me.byteLength,_);w(oe,0,me);let he=U(p("classifier_palm_16_NO_PRUNING/BiasAdd")),ge=G(p("regressor_palm_16_NO_PRUNING/Conv2D")),ve=d(ge.byteLength,_);w(ve,0,ge);let be=U(p("regressor_palm_16_NO_PRUNING/BiasAdd")),ce=G(p("classifier_palm_8_NO_PRUNING/Conv2D")),de=d(ce.byteLength,_);w(de,0,ce);let Ae=U(p("classifier_palm_8_NO_PRUNING/BiasAdd")),De=G(p("regressor_palm_8_NO_PRUNING/Conv2D")),ue=d(De.byteLength,_);w(ue,0,De);let Ie=U(p("regressor_palm_8_NO_PRUNING/BiasAdd")),Fe=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Pe=d(36864*3*4,_),ke=d(Fe,D),Se=d(Fe,D),He=d(Fe,D),Ne=d(576*256*4,D),Qe=new Map;function rt(e){let t=Qe.get(e);return t||(t=d(4,B),w(t,0,new Uint32Array([e])),Qe.set(e,t)),t}let je=d(144*256*4,D|GPUBufferUsage.COPY_DST),Xe=d(576*128*4,D|GPUBufferUsage.COPY_DST),qe=d(864*4,v),ot=d(15552*4,v),it=d(576*2*4,v),st=d(576*36*4,v),xt=d(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),St=d(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ct=d(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),vt=d(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ut=n.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function z(e,t){return Math.ceil(e/t)}function ae(e){let t=d(e.byteLength,B);return w(t,0,e),t}let Nt=ae(new Uint32Array([1,3,32,192,192,96,96])),ut=y.map(e=>{let t=e.stride===2?e.inH/2:e.inH,i=t,M=e.stride===2?1:2,H=e.inCh;return{dw:ae(new Uint32Array([1,e.inCh,e.inH,e.inH,t,i,e.stride,M])),pw:ae(new Uint32Array([1,e.inCh,e.outCh,t,i,H,e.stride,e.inH,e.inH])),outH:t,outW:i}}),ct=(()=>{let e=ne,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ae(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ae(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),dt=(()=>{let e=J,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ae(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ae(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Mt=(()=>{let e=O,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ae(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ae(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),qt=(()=>{let e=I,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ae(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ae(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Tt=ae(new Uint32Array([1,256,6,6,12,12])),Zt=ae(new Uint32Array([1,256,12,12,12,12])),Qt=ae(new Uint32Array([1,256,12,12,24,24])),Jt=ae(new Uint32Array([1,128,24,24,24,24])),zt=ae(new Uint32Array([1,256,256,12,12])),Yt=ae(new Uint32Array([1,256,128,24,24])),jt=ae(new Uint32Array([1,256,6,12,12])),Vt=ae(new Uint32Array([1,256,108,12,12])),Xt=ae(new Uint32Array([1,128,2,24,24])),b=ae(new Uint32Array([1,128,36,24,24])),r=ae(new Uint32Array([192,192,192])),f=n.createBindGroup({layout:at,entries:[{binding:0,resource:Ut.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:r}}]}),F=null,V=0,pe=0,re=d(32,B);function L(e,t){return F&&V===e&&pe===t||(F&&F.destroy(),F=n.createTexture({size:[e,t,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),V=e,pe=t),F}let pt=n.createBindGroup({layout:Ce,entries:[{binding:0,resource:{buffer:Pe}},{binding:1,resource:{buffer:Ze}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:x}},{binding:4,resource:{buffer:ke}},{binding:5,resource:{buffer:Nt}}]});function Je(e,t,i){}function le(e,t,i,M,H,P){let se=P.outH,h=n.createBindGroup({layout:nt,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:t.dwWeightBuf}},{binding:2,resource:{buffer:t.dwBiasBuf}},{binding:3,resource:{buffer:He}},{binding:4,resource:{buffer:P.dw}}]}),Ue=e.beginComputePass();Ue.setPipeline(Ht),Ue.setBindGroup(0,h),Ue.dispatchWorkgroups(z(se,8),z(P.outH,8),t.inCh),Ue.end(),t.inCh*P.outH*se;let Be=n.createBindGroup({layout:$e,entries:[{binding:0,resource:{buffer:He}},{binding:1,resource:{buffer:H}},{binding:2,resource:{buffer:t.pwWeightBuf}},{binding:3,resource:{buffer:t.pwBiasBuf}},{binding:4,resource:{buffer:t.alphaBuf}},{binding:5,resource:{buffer:M}},{binding:6,resource:{buffer:P.pw}}]}),m=e.beginComputePass();m.setPipeline(Pt),m.setBindGroup(0,Be),m.dispatchWorkgroups(z(se,8),z(P.outH,8),t.outCh),m.end(),t.outCh*P.outH*se}function ie(e,t,i,M,H,P,se,h,Ue){let Be=n.createBindGroup({layout:Ye,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:H}},{binding:4,resource:{buffer:P}}]}),m=e.beginComputePass();m.setPipeline(ht),m.setBindGroup(0,Be),m.dispatchWorkgroups(z(Ue,8),z(h,8),se),m.end()}function _e(e,t,i,M,H,P,se,h,Ue,Be){let m=n.createBindGroup({layout:wt,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:M}},{binding:3,resource:{buffer:H}},{binding:4,resource:{buffer:P}},{binding:5,resource:{buffer:se}}]}),S=e.beginComputePass();S.setPipeline(Wt),S.setBindGroup(0,m),S.dispatchWorkgroups(z(Be,8),z(Ue,8),h),S.end()}async function et(e){36864*3;{let A=e.beginComputePass();A.setPipeline(Re),A.setBindGroup(0,pt),A.dispatchWorkgroups(z(96,8),z(96,8),32),A.end()}9216*32;let t=ke,i=Se;for(let A=0;A<y.length;A++){let N=y[A];le(e,N,t,i,t,ut[A]);let Me=t;t=i,i=Me,A===13&&e.copyBufferToBuffer(t,0,Xe,0,576*128*4),A===18&&e.copyBufferToBuffer(t,0,je,0,144*256*4)}{let A=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Tt}}]}),N=e.beginComputePass();N.setPipeline(Ke),N.setBindGroup(0,A),N.dispatchWorkgroups(z(12,8),z(12,8),256),N.end()}{let A=t;t=i,i=A}144*256,_e(e,t,te,X,Z,i,zt,256,12,12);{let A=t;t=i,i=A}144*256;{let A=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:je}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Zt}}]}),N=e.beginComputePass();N.setPipeline(Ke),N.setBindGroup(0,A),N.dispatchWorkgroups(z(12,8),z(12,8),256),N.end()}{let A=t;t=i,i=A}144*256,le(e,ne,t,i,t,ct);{let A=t;t=i,i=A}le(e,J,t,i,t,dt);{let A=t;t=i,i=A}ie(e,t,oe,he,qe,jt,6,12,12),ie(e,t,ve,be,ot,Vt,108,12,12);{let A=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Qt}}]}),N=e.beginComputePass();N.setPipeline(Ke),N.setBindGroup(0,A),N.dispatchWorkgroups(z(24,8),z(24,8),256),N.end()}{let A=t;t=i,i=A}576*256,_e(e,t,$,T,ee,i,Yt,128,24,24);{let A=t;t=i,i=A}576*128;{let A=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Xe}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Jt}}]}),N=e.beginComputePass();N.setPipeline(Ke),N.setBindGroup(0,A),N.dispatchWorkgroups(z(24,8),z(24,8),128),N.end()}{let A=t;t=i,i=A}576*128,le(e,O,t,i,t,Mt);{let A=t;t=i,i=A}le(e,I,t,i,t,qt);{let A=t;t=i,i=A}ie(e,t,de,Ae,it,Xt,2,24,24),576*2,ie(e,t,ue,Ie,st,b,36,24,24),576*36,n.queue.submit([e.finish()]);let M=n.createCommandEncoder();M.copyBufferToBuffer(qe,0,xt,0,864*4),M.copyBufferToBuffer(ot,0,St,0,15552*4),M.copyBufferToBuffer(it,0,Ct,0,576*2*4),M.copyBufferToBuffer(st,0,vt,0,576*36*4),n.queue.submit([M.finish()]),await Promise.all([xt.mapAsync(GPUMapMode.READ),St.mapAsync(GPUMapMode.READ),Ct.mapAsync(GPUMapMode.READ),vt.mapAsync(GPUMapMode.READ)]);let H=new Float32Array(xt.getMappedRange()).slice(),P=new Float32Array(St.getMappedRange()).slice(),se=new Float32Array(Ct.getMappedRange()).slice(),h=new Float32Array(vt.getMappedRange()).slice();xt.unmap(),St.unmap(),Ct.unmap(),vt.unmap();let Ue=2016,Be=new Float32Array(Ue),m=new Float32Array(Ue*18),S=0;for(let A=0;A<12;A++)for(let N=0;N<12;N++)for(let Me=0;Me<6;Me++){Be[S]=H[Me*144+A*12+N];for(let Ve=0;Ve<18;Ve++){let o=Me*18+Ve;m[S*18+Ve]=P[o*144+A*12+N]}S++}for(let A=0;A<24;A++)for(let N=0;N<24;N++)for(let Me=0;Me<2;Me++){Be[S]=se[Me*576+A*24+N];for(let Ve=0;Ve<18;Ve++){let o=Me*18+Ve;m[S*18+Ve]=h[o*576+A*24+N]}S++}return{scores:Be,regressors:m}}async function $t(e){n.queue.copyExternalImageToTexture({source:e},{texture:Ut},[192,192]);let t=n.createCommandEncoder();{let i=t.beginComputePass();i.setPipeline(Bt),i.setBindGroup(0,f),i.dispatchWorkgroups(z(192,16),z(192,16),1),i.end()}return et(t)}async function en(e,t,i){let M=Math.min(192/t,192/i),H=Math.round(t*M),P=Math.round(i*M),se=Math.floor((192-H)/2),h=Math.floor((192-P)/2),Ue=se/192,Be=h/192,m=L(t,i),S;e instanceof HTMLVideoElement?S=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?S=await createImageBitmap(e,{colorSpaceConversion:"none"}):S=e,n.queue.copyExternalImageToTexture({source:S},{texture:m},[t,i]);let A=new ArrayBuffer(32),N=new Uint32Array(A),Me=new Float32Array(A);N[0]=t,N[1]=i,N[2]=192,N[3]=0,Me[4]=t/H,Me[5]=i/P,Me[6]=se,Me[7]=h,n.queue.writeBuffer(re,0,A);let Ve=n.createBindGroup({layout:Ee,entries:[{binding:0,resource:m.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:re}},{binding:3,resource:a}]}),o=n.createCommandEncoder();{let W=o.beginComputePass();W.setPipeline(At),W.setBindGroup(0,Ve),W.dispatchWorkgroups(z(192,16),z(192,16),1),W.end()}return{output:await et(o),lbPadX:Ue,lbPadY:Be}}async function K(e,t){let i=n.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),M=n.createCommandEncoder();M.copyBufferToBuffer(e,0,i,0,t*4),n.queue.submit([M.finish()]),await i.mapAsync(GPUMapMode.READ);let H=new Float32Array(i.getMappedRange()).slice();return i.unmap(),i.destroy(),H}async function vn(e,t,i){function M(o,c=1e3){let W=o.slice(0,c),xe=Math.max(0,Math.floor(o.length/2)-250);return{min:Math.min(...W),max:Math.max(...W),mean:W.reduce((We,gt)=>We+gt,0)/W.length,nonZero:W.filter(We=>We!==0).length,sample:Array.from(W.slice(0,10)),data500:Array.from(o.slice(0,500)),dataMid500:Array.from(o.slice(xe,xe+500)),totalLength:o.length}}function H(o,c,W,xe){let We=[],gt=Math.floor(W/2),bt=Math.floor(xe/2),tt=W*xe;for(let _t=0;_t<c&&We.length<500;_t++)for(let Kt=-1;Kt<=1&&We.length<500;Kt++)for(let Et=-1;Et<=1&&We.length<500;Et++){let yt=gt+Kt,tn=bt+Et;yt>=0&&yt<W&&tn>=0&&tn<xe&&We.push(o[_t*tt+yt*xe+tn])}return We}let P={},se;if(e instanceof HTMLImageElement?(t=t??e.naturalWidth,i=i??e.naturalHeight,se=await createImageBitmap(e,{colorSpaceConversion:"none"})):(t=t??e.width??192,i=i??e.height??192,se=e),t!==192||i!==192){let o=Math.min(192/t,192/i),c=Math.round(t*o),W=Math.round(i*o),xe=Math.floor((192-c)/2),We=Math.floor((192-W)/2),gt=L(t,i);n.queue.copyExternalImageToTexture({source:se},{texture:gt},[t,i]);let bt=new ArrayBuffer(32),tt=new Uint32Array(bt),_t=new Float32Array(bt);tt[0]=t,tt[1]=i,tt[2]=192,tt[3]=0,_t[4]=t/c,_t[5]=i/W,_t[6]=xe,_t[7]=We,n.queue.writeBuffer(re,0,bt);let Kt=n.createBindGroup({layout:Ee,entries:[{binding:0,resource:gt.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:re}},{binding:3,resource:a}]});{let Et=n.createCommandEncoder(),yt=Et.beginComputePass();yt.setPipeline(At),yt.setBindGroup(0,Kt),yt.dispatchWorkgroups(z(192,16),z(192,16),1),yt.end(),n.queue.submit([Et.finish()])}}else{n.queue.copyExternalImageToTexture({source:se},{texture:Ut},[192,192]);let o=ae(new Uint32Array([192,192,192])),c=n.createBindGroup({layout:at,entries:[{binding:0,resource:Ut.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:o}}]});{let W=n.createCommandEncoder(),xe=W.beginComputePass();xe.setPipeline(Bt),xe.setBindGroup(0,c),xe.dispatchWorkgroups(z(192,16),z(192,16),1),xe.end(),n.queue.submit([W.finish()])}}{let o=await K(Pe,110592),c=M(o);c.dataCenter500=H(o,3,192,192),P.input=c}let h=n.createCommandEncoder(),Ue=n.createBindGroup({layout:Ce,entries:[{binding:0,resource:{buffer:Pe}},{binding:1,resource:{buffer:Ze}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:x}},{binding:4,resource:{buffer:ke}},{binding:5,resource:{buffer:Nt}}]}),Be=h.beginComputePass();Be.setPipeline(Re),Be.setBindGroup(0,Ue),Be.dispatchWorkgroups(z(96,8),z(96,8),32),Be.end(),n.queue.submit([h.finish()]);{let o=await K(ke,294912),c=M(o);c.dataCenter500=H(o,32,96,96),P.initConv=c}let m=ke,S=Se;for(let o=0;o<y.length;o++){let c=y[o];h=n.createCommandEncoder(),le(h,c,m,S,m,ut[o]),n.queue.submit([h.finish()]);let W=m;m=S,S=W;{let xe=c.stride===2?c.inH/2:c.inH,We=xe,gt=xe*We*c.outCh,bt=await K(m,gt),tt=M(bt);tt.dataCenter500=H(bt,c.outCh,xe,We),tt.spatialShape=[c.outCh,xe,We],P[`block${o}`]=tt}o===13&&(h=n.createCommandEncoder(),h.copyBufferToBuffer(m,0,Xe,0,576*128*4),n.queue.submit([h.finish()])),o===18&&(h=n.createCommandEncoder(),h.copyBufferToBuffer(m,0,je,0,144*256*4),n.queue.submit([h.finish()]))}h=n.createCommandEncoder();{let o=ae(new Uint32Array([1,256,6,6,12,12])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),W=h.beginComputePass();W.setPipeline(Ke),W.setBindGroup(0,c),W.dispatchWorkgroups(z(12,8),z(12,8),256),W.end()}n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,36864),c=M(o);c.dataCenter500=H(o,256,12,12),P.fpnUpsample6to12=c}h=n.createCommandEncoder(),_e(h,m,te,X,Z,S,zt,256,12,12),n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,36864),c=M(o);c.dataCenter500=H(o,256,12,12),P.fpn6to12Conv=c}{let o=await K(je,36864),c=M(o);c.dataCenter500=H(o,256,12,12),P.backbone12Skip=c}h=n.createCommandEncoder();{let o=ae(new Uint32Array([1,256,12,12,12,12])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:je}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),W=h.beginComputePass();W.setPipeline(Ke),W.setBindGroup(0,c),W.dispatchWorkgroups(z(12,8),z(12,8),256),W.end()}n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,36864),c=M(o);c.dataCenter500=H(o,256,12,12),P.fpnAdd12=c}h=n.createCommandEncoder(),le(h,ne,m,S,m,ct),n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,36864),c=M(o);c.dataCenter500=H(o,256,12,12),P.fpn12Block1=c}h=n.createCommandEncoder(),le(h,J,m,S,m,dt),n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,36864),c=M(o);c.dataCenter500=H(o,256,12,12),P.fpn12Block2=c}h=n.createCommandEncoder(),ie(h,m,oe,he,qe,jt,6,12,12),n.queue.submit([h.finish()]);{let o=await K(qe,864),c=M(o);c.dataCenter500=H(o,6,12,12),P.cls16=c}h=n.createCommandEncoder(),ie(h,m,ve,be,ot,Vt,108,12,12),n.queue.submit([h.finish()]);{let o=await K(ot,15552),c=M(o,500);c.dataCenter500=H(o,108,12,12),P.reg16=c}h=n.createCommandEncoder();{let o=ae(new Uint32Array([1,256,12,12,24,24])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Ne}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),W=h.beginComputePass();W.setPipeline(Ke),W.setBindGroup(0,c),W.dispatchWorkgroups(z(24,8),z(24,8),256),W.end()}n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,147456),c=M(o);c.dataCenter500=H(o,256,24,24),P.fpnUpsample12to24=c}h=n.createCommandEncoder(),_e(h,m,$,T,ee,S,Yt,128,24,24),n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,73728),c=M(o);c.dataCenter500=H(o,128,24,24),P.fpn12to24Conv=c}{let o=await K(Xe,73728),c=M(o);c.dataCenter500=H(o,128,24,24),P.backbone24Skip=c}h=n.createCommandEncoder();{let o=ae(new Uint32Array([1,128,24,24,24,24])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:Xe}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),W=h.beginComputePass();W.setPipeline(Ke),W.setBindGroup(0,c),W.dispatchWorkgroups(z(24,8),z(24,8),128),W.end()}n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,73728),c=M(o);c.dataCenter500=H(o,128,24,24),P.fpnAdd24=c}h=n.createCommandEncoder(),le(h,O,m,S,m,Mt),n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,73728),c=M(o);c.dataCenter500=H(o,128,24,24),P.fpn24Block1=c}h=n.createCommandEncoder(),le(h,I,m,S,m,qt),n.queue.submit([h.finish()]);{let o=m;m=S,S=o}{let o=await K(m,73728),c=M(o);c.dataCenter500=H(o,128,24,24),P.fpn24Block2=c}h=n.createCommandEncoder(),ie(h,m,de,Ae,it,Xt,2,24,24),n.queue.submit([h.finish()]);{let o=await K(it,1152),c=M(o);c.dataCenter500=H(o,2,24,24),P.cls8=c}h=n.createCommandEncoder(),ie(h,m,ue,Ie,st,b,36,24,24),n.queue.submit([h.finish()]);{let o=await K(st,20736),c=M(o);c.dataCenter500=H(o,36,24,24),P.reg8=c}P.initWeights=M(await K(Ze,100),100),P.initBias=M(await K(u,32),32),P.cls16Weights=M(await K(oe,100),100),P.cls16Bias=M(await K(he,6),6),P.cls8Weights=M(await K(de,100),100),P.cls8Bias=M(await K(Ae,2),2),P.fpn6to12Weights=M(await K(te,100),100);let A=await K(qe,864),N=await K(it,576*2);P.rawScores=new Float32Array(2016),P.rawScores.set(A,0),P.rawScores.set(N,864);let Me=await K(ot,15552),Ve=await K(st,576*36);return P.rawRegressors=new Float32Array(36288),P.rawRegressors.set(Me,0),P.rawRegressors.set(Ve,15552),P.rawInput=await K(Pe,36864*3),P}return{device:n,run:$t,runWithResize:en,debugRun:vn}}function On(){let s=[];for(let g=0;g<12;g++)for(let n=0;n<12;n++){let E=(n+.5)/12,C=(g+.5)/12;for(let a=0;a<6;a++)s.push({x:E,y:C})}for(let g=0;g<24;g++)for(let n=0;n<24;n++){let E=(n+.5)/24,C=(g+.5)/24;for(let a=0;a<2;a++)s.push({x:E,y:C})}return s}var wn=On();function zn(s){return 1/(1+Math.exp(-s))}function rn(s,g){let n=[],{scores:E,regressors:C}=s,a=192;for(let _=0;_<wn.length;_++){let D=zn(E[_]);if(D<g)continue;let v=wn[_],B=_*18,d=v.x+C[B+0]/a,w=v.y+C[B+1]/a,U=C[B+2]/a,Q=C[B+3]/a,q=[];for(let p=0;p<7;p++){let R=v.x+C[B+4+p*2]/a,G=v.y+C[B+4+p*2+1]/a;q.push([R,G])}n.push({score:D,box:[d,w,U,Q],keypoints:q})}return n}function on(s,g){if(s.length===0)return[];let n=[...s].sort((a,_)=>_.score-a.score),E=[],C=new Set;for(let a=0;a<n.length;a++){if(C.has(a))continue;let _=[a];for(let q=a+1;q<n.length;q++)C.has(q)||Kn(n[a],n[q])>g&&(_.push(q),C.add(q));let D=0,v=0,B=0,d=0,w=0,U=[];for(let q=0;q<7;q++)U.push([0,0]);for(let q of _){let p=n[q],R=p.score;D+=R,v+=p.box[0]*R,B+=p.box[1]*R,d+=p.box[2]*R,w+=p.box[3]*R;for(let G=0;G<7;G++)U[G][0]+=p.keypoints[G][0]*R,U[G][1]+=p.keypoints[G][1]*R}let Q=1/D;E.push({score:n[a].score,box:[v*Q,B*Q,d*Q,w*Q],keypoints:U.map(([q,p])=>[q*Q,p*Q])})}return E}function Kn(s,g){let n=s.box[0]-s.box[2]/2,E=s.box[1]-s.box[3]/2,C=s.box[0]+s.box[2]/2,a=s.box[1]+s.box[3]/2,_=g.box[0]-g.box[2]/2,D=g.box[1]-g.box[3]/2,v=g.box[0]+g.box[2]/2,B=g.box[1]+g.box[3]/2,d=Math.max(n,_),w=Math.max(E,D),U=Math.min(C,v),Q=Math.min(a,B),q=Math.max(0,U-d),p=Math.max(0,Q-w),R=q*p,G=(C-n)*(a-E),ye=(v-_)*(B-D),Le=G+ye-R;return Le>0?R/Le:0}function In(s){let[g,n,E,C]=s.box,a=s.keypoints[0],_=s.keypoints[2],D=_[0]-a[0],v=_[1]-a[1],B=Math.atan2(v,D),w=-Math.PI/2-B,U=Math.max(E,C),q=U*2.6,p=-.5*U,R=Math.cos(w),G=Math.sin(w),ye=p*G,Le=p*R;return{centerX:g+ye,centerY:n+Le,width:q,height:q,rotation:w}}function Pn(s,g={}){let{scoreThreshold:n=.5,nmsThreshold:E=.3,maxHands:C=2}=g;async function a(B){let d=await s.run(B),w=rn(d,n);return on(w,E).slice(0,C).map(In)}async function _(B){let d=await s.run(B),w=rn(d,n);return on(w,E).slice(0,C)}async function D(B,d,w){let{output:U,lbPadX:Q,lbPadY:q}=await s.runWithResize(B,d,w),p=rn(U,n);return{detections:on(p,E).slice(0,C),lbPadX:Q,lbPadY:q}}async function v(B,d,w){let{output:U,lbPadX:Q,lbPadY:q}=await s.runWithResize(B,d,w);return{scores:U.scores,regressors:U.regressors,lbPadX:Q,lbPadY:q}}return{detect:a,detectRaw:_,detectRawWithResize:D,detectRawSSD:v,model:s}}var sn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function Dt(s){let g={};for(let n=0;n<sn.length;n++)g[sn[n]]=s[n];return g}function Fn(s){return s.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Nn=Fn(`
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
`);function Bn(s){let g=s.createShaderModule({code:Nn}),n=s.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),E=s.createComputePipeline({layout:s.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:g,entryPoint:"main"}}),C=s.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),a=s.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),_=s.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),D=new Float32Array(8);function v(B,d,w,U,Q,q,p){s.queue.writeBuffer(a,0,new Uint32Array([Q,q,p,0])),D.set(U),s.queue.writeBuffer(_,0,D);let R=s.createBindGroup({layout:n,entries:[{binding:0,resource:d.createView()},{binding:1,resource:{buffer:w}},{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:_}},{binding:4,resource:C}]}),G=B.beginComputePass();G.setPipeline(E),G.setBindGroup(0,R),G.dispatchWorkgroups(Math.ceil(p/16),Math.ceil(p/16),1),G.end()}return{crop:v}}var qn="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Yn(s={}){let{weightsUrl:g,scoreThreshold:n=.5,palmScoreThreshold:E=.5,maxHands:C=3,forceF32:a=!1}=s;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let _=(g??qn).replace(/\/$/,"")+"/",[D,v,B,d]=await Promise.all([fetch(`${_}weights_f16_full.json`),fetch(`${_}weights_f16_full.bin`),fetch(`${_}palm_detection_weights.json`),fetch(`${_}palm_detection_weights.bin`)]);if(!D.ok)throw new Error(`Failed to fetch landmark weights: ${D.status}`);if(!v.ok)throw new Error(`Failed to fetch landmark weights: ${v.status}`);if(!B.ok)throw new Error(`Failed to fetch palm detection weights: ${B.status}`);if(!d.ok)throw new Error(`Failed to fetch palm detection weights: ${d.status}`);let[w,U,Q,q]=await Promise.all([D.json(),v.arrayBuffer(),B.json(),d.arrayBuffer()]),p=cn(w,U),R=nn(Q,q),G=224,ye=await an(p,{forceF32:a});{let u=new OffscreenCanvas(G,G),x=u.getContext("2d");x.fillStyle="#886644",x.fillRect(0,0,G,G),x.fillStyle="#cc9966",x.fillRect(50,50,124,124);let l=await ye.runFromCanvas(u);l.landmarks.every(Y=>Y===0)&&l.handflag.every(Y=>Y===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let Le=await yn(R),Ge=Pn(Le,{scoreThreshold:E,maxHands:C}),k=[];function Oe(u,x,l){let y=u[0],Y=u[5],te=u[9],X=u[13],Z=y.x*x,ne=y.y*l,J=(Y.x+X.x)/2,j=(Y.y+X.y)/2;J=(J+te.x)/2*x,j=(j+te.y)/2*l;let $=Math.PI/2-Math.atan2(-(j-ne),J-Z),T=$-2*Math.PI*Math.floor(($+Math.PI)/(2*Math.PI)),ee=[0,1,2,3,5,6,9,10,13,14,17,18],O=Math.cos(T),I=Math.sin(T),me=1/0,oe=-1/0,he=1/0,ge=-1/0;for(let Ne of ee){let Qe=u[Ne],rt=Qe.x*x,je=Qe.y*l,Xe=O*rt+I*je,qe=-I*rt+O*je;me=Math.min(me,Xe),oe=Math.max(oe,Xe),he=Math.min(he,qe),ge=Math.max(ge,qe)}let ve=(me+oe)/2,be=(he+ge)/2,ce=oe-me,de=ge-he,Ae=(O*ve-I*be)/x,De=(I*ve+O*be)/l;ce/=x,de/=l;let ue=-.1,Ie=de*l;Ae+=.5*Ie*ue*I/x;let Fe=-l*de*ue*I/x,Pe=l*de*ue*O/l;Ae+=Fe,De+=Pe;let He=Math.max(ce*x,de*l)*2;return{centerXpx:Ae*x,centerYpx:De*l,sizePx:He,rotation:T}}let Te=null,fe=null;function Lt(){return Te||(Te=new OffscreenCanvas(192,192)),Te}function Rt(){return fe||(fe=new OffscreenCanvas(G,G)),fe}let Ce=ye.device,nt=null,$e=null,Ye=null,wt=0,ze=0;function at(){return nt||(nt=Bn(Ce)),nt}function Ee(){return $e||($e=Ce.createBuffer({size:3*G*G*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),$e}function lt(u,x){return(!Ye||wt!==u||ze!==x)&&(Ye&&Ye.destroy(),Ye=Ce.createTexture({size:[u,x],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),wt=u,ze=x),Ye}let we=0,Re=0;function Ht(u,x,l){let y=Lt();y.width=192,y.height=192;let Y=y.getContext("2d");Y.clearRect(0,0,192,192);let te=Math.min(192/x,192/l),X=Math.round(x*te),Z=Math.round(l*te),ne=(192-X)/2,J=(192-Z)/2;if(we=ne/192,Re=J/192,u instanceof ImageData){let j=new OffscreenCanvas(u.width,u.height);j.getContext("2d").putImageData(u,0,0),Y.drawImage(j,ne,J,X,Z)}else Y.drawImage(u,0,0,x,l,ne,J,X,Z);return y}function Pt(u){let x=1/(1-2*we),l=1/(1-2*Re);return{score:u.score,box:[(u.box[0]-we)*x,(u.box[1]-Re)*l,u.box[2]*x,u.box[3]*l],keypoints:u.keypoints.map(([y,Y])=>[(y-we)*x,(Y-Re)*l])}}function ht(u,x,l){let y=u.keypoints[0],Y=u.keypoints[2],te=(Y[0]-y[0])*x,X=(Y[1]-y[1])*l,Z=Math.atan2(-X,te),J=Math.PI/2-Z,j=J-2*Math.PI*Math.floor((J+Math.PI)/(2*Math.PI)),[$,T,ee,O]=u.box,I=Math.cos(j),me=Math.sin(j),oe=O*l,he=$+.5*oe*me/x,ge=T+-.5*O*I,ce=Math.max(ee*x,O*l)*2.6;return{centerXpx:he*x,centerYpx:ge*l,sizePx:ce,rotation:j}}function Wt(u,x){let l=Rt();l.width=G,l.height=G;let y=l.getContext("2d");y.clearRect(0,0,G,G);let Y=G/x.sizePx,te=Math.cos(x.rotation),X=Math.sin(x.rotation),Z=te*Y,ne=-X*Y,J=X*Y,j=te*Y,$=G/2,T=-x.centerXpx*Z-x.centerYpx*J+$,ee=-x.centerXpx*ne-x.centerYpx*j+$;if(y.setTransform(Z,ne,J,j,T,ee),u instanceof ImageData){let O=new OffscreenCanvas(u.width,u.height);O.getContext("2d").putImageData(u,0,0),y.drawImage(O,0,0)}else y.drawImage(u,0,0);return y.setTransform(1,0,0,1,0,0),l}function Ke(u){return u instanceof HTMLCanvasElement||u instanceof OffscreenCanvas?[u.width,u.height]:typeof ImageBitmap<"u"&&u instanceof ImageBitmap?[u.width,u.height]:u instanceof ImageData?[u.width,u.height]:u instanceof HTMLVideoElement?[u.videoWidth,u.videoHeight]:u instanceof HTMLImageElement?[u.naturalWidth,u.naturalHeight]:[G,G]}async function Bt(u,x,l,y,Y,te){let X=Math.cos(u.rotation),Z=Math.sin(u.rotation),ne=u.sizePx/G,J=G/2,j=X*ne/l,$=-Z*ne/l,T=u.centerXpx/l-J*(j+$),ee=Z*ne/y,O=X*ne/y,I=u.centerYpx/y-J*(ee+O),me=Ce.createCommandEncoder();Y.crop(me,x,te,[j,$,T,ee,O,I],l,y,G),Ce.queue.submit([me.finish()]);let oe=await ye.runFromGPUBuffer(te),he=oe.handflag[0];if(he<n)return null;let ge=oe.handedness[0]>.5,ve=[];for(let be=0;be<21;be++){let ce=oe.landmarks[be*3],de=oe.landmarks[be*3+1],Ae=oe.landmarks[be*3+2],De=(ce-.5)*u.sizePx,ue=(de-.5)*u.sizePx,Ie=X*De-Z*ue+u.centerXpx,Fe=Z*De+X*ue+u.centerYpx;ve.push({x:Ie/l,y:Fe/y,z:Ae})}return{landmarks:ve,score:he,handedness:ge?"right":"left"}}async function At(u){let x=u,l,y;if(u instanceof HTMLVideoElement||u instanceof HTMLImageElement){let T=await createImageBitmap(u,{colorSpaceConversion:"none"});x=T,l=T.width,y=T.height}else[l,y]=Ke(u);let Y=at(),te=Ee(),X=lt(l,y),Z;if(x instanceof ImageData?Z=await createImageBitmap(x,{colorSpaceConversion:"none"}):Z=x,Ce.queue.copyExternalImageToTexture({source:Z},{texture:X},[l,y]),k.length>0){let T=[];for(let ee of k){let O=Oe(ee.landmarks,l,y),I=await Bt(O,X,l,y,Y,te);I&&T.push({score:I.score,handedness:I.handedness,landmarks:I.landmarks,keypoints:Dt(I.landmarks)})}if(T.length>0)return k=T.map(ee=>({landmarks:ee.landmarks,handedness:ee.handedness})),T;k=[]}let{detections:ne,lbPadX:J,lbPadY:j}=await Ge.detectRawWithResize(x,l,y);if(we=J,Re=j,ne.length===0)return k=[],[];let $=[];for(let T of ne){let ee=Pt(T),O=ht(ee,l,y),I=await Bt(O,X,l,y,Y,te);I&&$.push({score:I.score,handedness:I.handedness,landmarks:I.landmarks,keypoints:Dt(I.landmarks)})}return k=$.map(T=>({landmarks:T.landmarks,handedness:T.handedness})),$}async function It(u){let x=u,l,y;if(u instanceof HTMLVideoElement||u instanceof HTMLImageElement){let T=await createImageBitmap(u,{colorSpaceConversion:"none"});x=T,l=T.width,y=T.height}else[l,y]=Ke(u);let{detections:Y,lbPadX:te,lbPadY:X}=await Ge.detectRawWithResize(x,l,y);if(we=te,Re=X,Y.length===0)return[];let Z=[],ne=at(),J=Ee(),j=lt(l,y),$;x instanceof ImageData?$=await createImageBitmap(x,{colorSpaceConversion:"none"}):$=x,Ce.queue.copyExternalImageToTexture({source:$},{texture:j},[l,y]);for(let T of Y){let ee=Pt(T),O=ht(ee,l,y),I=Math.cos(O.rotation),me=Math.sin(O.rotation),oe=O.sizePx/G,he=G/2,ge=I*oe/l,ve=-me*oe/l,be=O.centerXpx/l-he*(ge+ve),ce=me*oe/y,de=I*oe/y,Ae=O.centerYpx/y-he*(ce+de),De=Ce.createCommandEncoder();ne.crop(De,j,J,[ge,ve,be,ce,de,Ae],l,y,G),Ce.queue.submit([De.finish()]);let ue=await ye.runFromGPUBuffer(J),Ie=ue.handflag[0];if(Ie<n)continue;let Fe=ue.handedness[0]>.5,Pe=[],ke=[];for(let Se=0;Se<21;Se++){let He=ue.landmarks[Se*3],Ne=ue.landmarks[Se*3+1],Qe=ue.landmarks[Se*3+2];Pe.push({x:He,y:Ne,z:Qe});let rt=(He-.5)*O.sizePx,je=(Ne-.5)*O.sizePx,Xe=I*rt-me*je+O.centerXpx,qe=me*rt+I*je+O.centerYpx;ke.push({x:Xe/l,y:qe/y,z:Qe})}Z.push({score:Ie,handedness:Fe?"right":"left",landmarks:ke,keypoints:Dt(ke),cropLandmarks:Pe,roi:{...O},palmDetection:{score:ee.score,box:[...ee.box],keypoints:ee.keypoints.map(([Se,He])=>[Se,He])}})}return Z}async function Ot(u,x){let[l,y]=Ke(u);if(x.length===0)return[];let Y=[],te=at(),X=Ee(),Z=lt(l,y),ne;u instanceof ImageData?ne=await createImageBitmap(u,{colorSpaceConversion:"none"}):u instanceof HTMLImageElement?ne=await createImageBitmap(u,{colorSpaceConversion:"none"}):ne=u,Ce.queue.copyExternalImageToTexture({source:ne},{texture:Z},[l,y]);for(let J of x){let j=ht(J,l,y),$=Math.cos(j.rotation),T=Math.sin(j.rotation),ee=j.sizePx/G,O=G/2,I=$*ee/l,me=-T*ee/l,oe=j.centerXpx/l-O*(I+me),he=T*ee/y,ge=$*ee/y,ve=j.centerYpx/y-O*(he+ge),be=Ce.createCommandEncoder();te.crop(be,Z,X,[I,me,oe,he,ge,ve],l,y,G),Ce.queue.submit([be.finish()]);let ce=await ye.runFromGPUBuffer(X),de=ce.handflag[0];if(de<n)continue;let Ae=ce.handedness[0]>.5,De=[];for(let ue=0;ue<21;ue++){let Ie=ce.landmarks[ue*3],Fe=ce.landmarks[ue*3+1],Pe=ce.landmarks[ue*3+2],ke=(Ie-.5)*j.sizePx,Se=(Fe-.5)*j.sizePx,He=$*ke-T*Se+j.centerXpx,Ne=T*ke+$*Se+j.centerYpx;De.push({x:He/l,y:Ne/y,z:Pe})}Y.push({score:de,handedness:Ae?"right":"left",landmarks:De,keypoints:Dt(De)})}return Y}function kt(){Ye&&Ye.destroy(),$e&&$e.destroy(),Ye=null,$e=null,nt=null,ye.device.destroy(),Le.device.destroy(),Te=null,fe=null}let Ft={palmDetector:Ge,palmModel:Le,landmarkModel:ye,removeLetterbox:Pt,detectionToPixelROI:ht,cropHandRegion:Wt};function Ze(){k=[]}return{detect:At,detectWithDebug:It,detectFromDetections:Ot,dispose:kt,reset:Ze,_debug:Ft}}function xn(s,g,n){return s.initialized?(s.value=n*g+(1-n)*s.value,s.value):(s.value=g,s.initialized=!0,g)}function Cn(s,g){let n=2*Math.PI*g*s;return n/(n+1)}function jn(){return{x:{value:0,initialized:!1},dx:{value:0,initialized:!1},lastTime:-1}}function un(s,g,n,E,C,a){let _=s.lastTime<0?.03333333333333333:n-s.lastTime;s.lastTime=n;let D=Cn(_,a),v=s.x.initialized?(g-s.x.value)/_:0,B=xn(s.dx,v,D),d=E+C*Math.abs(B),w=Cn(_,d);return xn(s.x,g,w)}function Vn(s={}){let{minCutoff:g=.05,beta:n=80,dCutoff:E=1}=s,C=[];function a(v){C.length!==v&&(C=Array.from({length:v},()=>jn()))}function _(v,B){let d=B??performance.now()/1e3,w=v.length*3;return a(w),v.map((U,Q)=>({x:un(C[Q*3],U.x,d,g,n,E),y:un(C[Q*3+1],U.y,d,g,n,E),z:un(C[Q*3+2],U.z,d,g,n,E)}))}function D(){C=[]}return{apply:_,reset:D}}export{sn as LANDMARK_NAMES,an as compileFullModel,Yn as createHandpose,Vn as createLandmarkSmoother,nn as loadWeightsFromBuffer,Dt as toKeypoints};
