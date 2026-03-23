function on(u,U){let n=new Map,W=u.dtype??"float32";for(let M=0;M<u.keys.length;M++){let a=u.keys[M],y=u.shapes[M],L=u.offsets[M],A=y.reduce((P,G)=>P*G,1),B,l;if(W==="float32")B=new Float32Array(U,L,A);else{let P=new DataView(U);B=new Float32Array(A);for(let G=0;G<A;G++)B[G]=xn(P.getUint16(L+G*2,!0));l=U.slice(L,L+A*2)}n.set(a,{data:B,shape:y,rawF16:l})}return n}function xn(u){let U=u>>15&1,n=u>>10&31,W=u&1023;if(n===0){if(W===0)return U?-0:0;let y=-14,L=W/1024;return(U?-1:1)*Math.pow(2,y)*L}if(n===31)return W===0?U?-1/0:1/0:NaN;let M=n-15,a=1+W/1024;return(U?-1:1)*Math.pow(2,M)*a}var Cn=[[24,24,128,1,"backbone1.3.f.0."],[24,24,128,1,"backbone1.3.f.1."],[24,48,128,2,"backbone1.4."],[48,48,64,1,"backbone2.0.f.0."],[48,48,64,1,"backbone2.0.f.1."],[48,96,64,2,"backbone2.1."],[96,96,32,1,"backbone3.0.f.0."],[96,96,32,1,"backbone3.0.f.1."],[96,96,32,2,"backbone3.1."],[96,96,16,1,"backbone4.0.f.0."],[96,96,16,1,"backbone4.0.f.1."],[96,96,16,2,"backbone4.1."],[96,96,16,1,"backbone5.0."],[96,96,32,1,"backbone6.0."],[48,48,64,1,"ff.0.f.0."],[48,48,64,1,"ff.0.f.1."],[48,48,64,1,"ff.0.f.2."],[48,48,64,1,"ff.0.f.3."],[48,96,64,2,"ff.1."],[96,96,32,1,"ff.2.f.0."],[96,96,32,1,"ff.2.f.1."],[96,96,32,1,"ff.2.f.2."],[96,96,32,1,"ff.2.f.3."],[96,288,32,2,"ff.3."],[288,288,16,1,"ff.4.f.0."],[288,288,16,1,"ff.4.f.1."],[288,288,16,1,"ff.4.f.2."],[288,288,16,1,"ff.4.f.3."],[288,288,16,2,"ff.5."],[288,288,8,1,"ff.6.f.0."],[288,288,8,1,"ff.6.f.1."],[288,288,8,1,"ff.6.f.2."],[288,288,8,1,"ff.6.f.3."],[288,288,8,2,"ff.7."],[288,288,4,1,"ff.8.f.0."],[288,288,4,1,"ff.8.f.1."],[288,288,4,1,"ff.8.f.2."],[288,288,4,1,"ff.8.f.3."],[288,288,4,2,"ff.9."],[288,288,2,1,"ff.10.f.0."],[288,288,2,1,"ff.10.f.1."],[288,288,2,1,"ff.10.f.2."],[288,288,2,1,"ff.10.f.3."]],Nn=Cn.map(([u,U,n,W,M])=>({type:"resmodule",inCh:u,outCh:U,h:n,w:n,stride:W,prefix:M}));function sn(u,U){let n=new Map,W=u.dtype??"float32",M=new Map;for(let a=0;a<u.keys.length;a++){let y=u.keys[a],L=u.shapes[a],A=u.offsets[a],B=L.reduce((N,d)=>N*d,1),l,P;if(W==="float32")l=new Float32Array(U,A,B);else{let N=new DataView(U);l=new Float32Array(B);for(let d=0;d<B;d++)l[d]=vn(N.getUint16(A+d*2,!0));P=U.slice(A,A+B*2)}let G=M.get(y)??0;M.set(y,G+1);let re=G===0?y:`${y}__${G}`;n.set(re,{data:l,shape:L,rawF16:P})}return n}function vn(u){let U=u>>15&1,n=u>>10&31,W=u&1023;return n===0?W===0?U?-0:0:(U?-1:1)*Math.pow(2,-14)*(W/1024):n===31?W===0?U?-1/0:1/0:NaN:(U?-1:1)*Math.pow(2,n-15)*(1+W/1024)}function ft(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var Un=ft(`
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
`),Gn=ft(`
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
`),Mn=ft(`
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
`),An=ft(`
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
`),kn=ft(`
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
`),Sn=ft(`
@group(0)@binding(0) var<storage,read> a:array<f32>;
@group(0)@binding(1) var<storage,read> b:array<f32>;
@group(0)@binding(2) var<storage,read_write> output:array<f32>;
@group(0)@binding(3) var<uniform> size:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x; if(idx>=size){return;} output[idx]=a[idx]+b[idx];
}
`),Tn=ft(`
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
`),En=ft(`
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
`),Dn=ft(`
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
`),Mt=[{inCh:24,expandCh:24,dwKernel:3,stride:1,outCh:16,hasResidual:!1,hasProject:!0},{inCh:16,expandCh:64,dwKernel:3,stride:2,outCh:24,hasResidual:!1,hasProject:!0},{inCh:24,expandCh:144,dwKernel:3,stride:1,outCh:24,hasResidual:!0,hasProject:!0},{inCh:24,expandCh:144,dwKernel:5,stride:2,outCh:40,hasResidual:!1,hasProject:!0},{inCh:40,expandCh:240,dwKernel:5,stride:1,outCh:40,hasResidual:!0,hasProject:!0},{inCh:40,expandCh:240,dwKernel:3,stride:2,outCh:80,hasResidual:!1,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:3,stride:1,outCh:80,hasResidual:!0,hasProject:!0},{inCh:80,expandCh:480,dwKernel:5,stride:1,outCh:112,hasResidual:!1,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:1,outCh:112,hasResidual:!0,hasProject:!0},{inCh:112,expandCh:672,dwKernel:5,stride:2,outCh:192,hasResidual:!1,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:5,stride:1,outCh:192,hasResidual:!0,hasProject:!0},{inCh:192,expandCh:1152,dwKernel:3,stride:1,outCh:1152,hasResidual:!1,hasProject:!1}],Rn=[{dwWeightKey:"batch_normalization_1/FusedBatchNormV3",dwBNKey:"batch_normalization_1",projectConvKey:"conv2d_1",projectBNKey:"batch_normalization_2/FusedBatchNormV3"},{expandConvKey:"conv2d_2",expandBNKey:"batch_normalization_3",dwWeightKey:"batch_normalization_4/FusedBatchNormV3",dwBNKey:"batch_normalization_4",projectConvKey:"conv2d_3",projectBNKey:"batch_normalization_5/FusedBatchNormV3"},{expandConvKey:"conv2d_4",expandBNKey:"batch_normalization_6",dwWeightKey:"batch_normalization_7/FusedBatchNormV3",dwBNKey:"batch_normalization_7",projectConvKey:"conv2d_5",projectBNKey:"batch_normalization_8/FusedBatchNormV3"},{expandConvKey:"conv2d_6",expandBNKey:"batch_normalization_9",dwWeightKey:"batch_normalization_10/FusedBatchNormV3",dwBNKey:"batch_normalization_10",projectConvKey:"conv2d_7",projectBNKey:"batch_normalization_11/FusedBatchNormV3"},{expandConvKey:"conv2d_8",expandBNKey:"batch_normalization_12",dwWeightKey:"batch_normalization_13/FusedBatchNormV3",dwBNKey:"batch_normalization_13",projectConvKey:"conv2d_9",projectBNKey:"batch_normalization_14/FusedBatchNormV3"},{expandConvKey:"conv2d_10",expandBNKey:"batch_normalization_15",dwWeightKey:"batch_normalization_16/FusedBatchNormV3",dwBNKey:"batch_normalization_16",projectConvKey:"conv2d_11",projectBNKey:"batch_normalization_17/FusedBatchNormV3"},{expandConvKey:"conv2d_12",expandBNKey:"batch_normalization_18",dwWeightKey:"batch_normalization_19/FusedBatchNormV3",dwBNKey:"batch_normalization_19",projectConvKey:"conv2d_13",projectBNKey:"batch_normalization_20/FusedBatchNormV3"},{expandConvKey:"conv2d_14",expandBNKey:"batch_normalization_21",dwWeightKey:"batch_normalization_22/FusedBatchNormV3",dwBNKey:"batch_normalization_22",projectConvKey:"conv2d_15",projectBNKey:"batch_normalization_23/FusedBatchNormV3"},{expandConvKey:"conv2d_16",expandBNKey:"batch_normalization_24",dwWeightKey:"batch_normalization_25/FusedBatchNormV3",dwBNKey:"batch_normalization_25",projectConvKey:"conv2d_17",projectBNKey:"batch_normalization_26/FusedBatchNormV3"},{expandConvKey:"conv2d_18",expandBNKey:"batch_normalization_27",dwWeightKey:"batch_normalization_28/FusedBatchNormV3",dwBNKey:"batch_normalization_28",projectConvKey:"conv2d_19",projectBNKey:"batch_normalization_29/FusedBatchNormV3"},{expandConvKey:"conv2d_20",expandBNKey:"batch_normalization_30",dwWeightKey:"batch_normalization_31/FusedBatchNormV3",dwBNKey:"batch_normalization_31",projectConvKey:"conv2d_21",projectBNKey:"batch_normalization_32/FusedBatchNormV3"},{expandConvKey:"conv2d_22",expandBNKey:"batch_normalization_33",dwWeightKey:"batch_normalization_34/FusedBatchNormV3",dwBNKey:"batch_normalization_34",projectConvKey:"conv2d_23",projectBNKey:"batch_normalization_35/FusedBatchNormV3"},{expandConvKey:"conv2d_24",expandBNKey:"batch_normalization_36",dwWeightKey:"batch_normalization_37/FusedBatchNormV3",dwBNKey:"batch_normalization_37",projectConvKey:"conv2d_25",projectBNKey:"batch_normalization_38/FusedBatchNormV3"},{expandConvKey:"conv2d_26",expandBNKey:"batch_normalization_39",dwWeightKey:"batch_normalization_40/FusedBatchNormV3",dwBNKey:"batch_normalization_40",projectConvKey:"conv2d_27",projectBNKey:"batch_normalization_41/FusedBatchNormV3"},{expandConvKey:"conv2d_28",expandBNKey:"batch_normalization_42",dwWeightKey:"batch_normalization_43/FusedBatchNormV3",dwBNKey:"batch_normalization_43",projectConvKey:"conv2d_29",projectBNKey:"batch_normalization_44/FusedBatchNormV3"},{expandConvKey:"conv2d_30",expandBNKey:"batch_normalization_45",dwWeightKey:"batch_normalization_46/FusedBatchNormV3",dwBNKey:"batch_normalization_46"}];async function un(u,U){if(!navigator.gpu)throw new Error("WebGPU not supported");let n=await navigator.gpu.requestAdapter();if(!n)throw new Error("No GPU adapter found");let W=n.features.has("shader-f16"),M=W?["shader-f16"]:[],a=await n.requestDevice({requiredFeatures:M,requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(n.limits.maxStorageBuffersPerShaderStage,8)}}),y=u.values().next().value,L=W&&!!y?.rawF16&&!U?.forceF32;function A(g){if(L&&g.rawF16){let r=new Uint16Array(g.rawF16);if(r.length%2!==0){let p=new Uint16Array(r.length+1);return p.set(r),p}return r}return g.data}function B(g){return L&&g.rawF16?Math.ceil(g.rawF16.byteLength/4)*4:g.data.byteLength}let l=L?2:4;function P(g){if(!L)return g;let r=g;return r=r.replace(/array<f32>/g,"array<f16>"),r=r.replace(/array<f32,/g,"array<f16,"),r=r.replace(/var sum:f32=0\.0/g,"var sum:f16=0.0h"),r=r.replace(/var sum0:f32=0\.0/g,"var sum0:f16=0.0h"),r=r.replace(/var sum1:f32=0\.0/g,"var sum1:f16=0.0h"),r=r.replace(/var sum2:f32=0\.0/g,"var sum2:f16=0.0h"),r=r.replace(/var sum3:f32=0\.0/g,"var sum3:f16=0.0h"),r=r.replace(/\/f32\(params/g,"/f16(params"),r=r.replace(/,0\.0\),6\.0\)/g,",0.0h),6.0h)"),r=r.replace(/->f32\{/g,"->f16{"),r=r.replace(/->f32 \{/g,"->f16 {"),r=r.replace(/return 0\.0;/g,"return 0.0h;"),"enable f16;"+r}function G(g){if(!L)return g;let r=P(g);return r=r.replace("read>input:array<f16>","read>input:array<f32>"),r=r.replace(/input\[in_idx\]/g,"f16(input[in_idx])"),r}function re(g){if(!L)return g;let r=g;return r=r.replace("read>input:array<f32>","read>input:array<f16>"),r=r.replace("read>weight:array<f32>","read>weight:array<f16>"),r=r.replace("read>bias:array<f32>","read>bias:array<f16>"),r=r.replace(/input\[ic\]/g,"f32(input[ic])"),r=r.replace(/weight\[oc\*params\.in_features\+ic\]/g,"f32(weight[oc*params.in_features+ic])"),r=r.replace(/bias\[oc\]/g,"f32(bias[oc])"),"enable f16;"+r}let N={r:"read-only-storage",s:"storage",u:"uniform"};function d(g){return a.createBindGroupLayout({entries:g.map((r,p)=>r==="t"?{binding:p,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:{binding:p,visibility:GPUShaderStage.COMPUTE,buffer:{type:N[r]}})})}let D=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,C=GPUBufferUsage.STORAGE,ye=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,Le=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,ke=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function k(g,r){return a.createBuffer({size:Math.max(g,4),usage:r})}function Ke(g,r){return a.createBindGroup({layout:g,entries:r.map((p,I)=>({binding:I,resource:"size"in p?{buffer:p}:p}))})}function Se(g,r){return a.createComputePipeline({layout:a.createPipelineLayout({bindGroupLayouts:[g]}),compute:{module:r,entryPoint:"main"}})}function pe(g){let r=u.get(g);if(!r)throw new Error(`Missing weight: ${g}`);return r}let Dt=a.createShaderModule({code:Un}),Rt=a.createShaderModule({code:G(Gn)}),ve=a.createShaderModule({code:P(Mn)}),tt=a.createShaderModule({code:P(An)}),je=a.createShaderModule({code:P(kn)}),Fe=a.createShaderModule({code:P(Sn)}),wt=a.createShaderModule({code:P(Tn)}),ze=a.createShaderModule({code:re(En)}),nt=a.createShaderModule({code:re(Dn)}),Te=d(["r","r","r","s","u"]),mt=d(["r","r","s","u"]),we=d(["r","s","u"]),He=d(["r","r","r","s","u"]),Lt=d(["t","s","u"]),Pt=Se(Lt,Dt),ht=Se(Te,Rt),Ht=Se(Te,ve),Ie=Se(Te,tt),Bt=Se(Te,je),At=Se(mt,Fe),It=Se(we,wt),Wt=Se(He,ze),kt=Se(He,nt),Xe=1152*112*112*4,s=k(Xe,Le),w=k(Xe,Le),h=k(Xe,C),b=k(Xe,C),q=k(Xe,D),ee=k(672*224*4,Le),V=k(1152*4,ye),Z=k(252,ye),te=k(252,ye),Q=k(4,ye),Y=k(4,ye),$=k(260,Le),T=k(260,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),J=a.createTexture({size:[224,224],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),O=k(4,ke);a.queue.writeBuffer(O,0,new Uint32Array([224]));let j=pe("conv2d"),fe=pe("batch_normalization"),oe=A(j),he=A(fe),ge=k(B(j),D),Ue=k(B(fe),D),be=k(24,ke);a.queue.writeBuffer(ge,0,oe),a.queue.writeBuffer(Ue,0,he),a.queue.writeBuffer(be,0,new Uint32Array([3,24,224,224,112,112]));let ue=112,le=112,Ee=[];for(let g=0;g<Mt.length;g++){let r=Mt[g],p=Rn[g],I=ue,X=le,de=r.stride===2?Math.floor(ue/2):ue,ae=r.stride===2?Math.floor(le/2):le,E={spec:r,inH:I,inW:X,outH:de,outW:ae,dwW:k(4,D),dwB:k(4,D),dwU:k(32,ke)};if(p.expandConvKey){let ie=pe(p.expandConvKey),_e=pe(p.expandBNKey);E.expandW=k(B(ie),D),E.expandB=k(B(_e),D),E.expandU=k(16,ke),a.queue.writeBuffer(E.expandW,0,A(ie)),a.queue.writeBuffer(E.expandB,0,A(_e)),a.queue.writeBuffer(E.expandU,0,new Uint32Array([r.inCh,r.expandCh,I,X]))}let pt=pe(p.dwWeightKey),Ze=pe(p.dwBNKey);E.dwW=k(B(pt),D),E.dwB=k(B(Ze),D),a.queue.writeBuffer(E.dwW,0,A(pt)),a.queue.writeBuffer(E.dwB,0,A(Ze));let me=Math.floor((r.dwKernel-r.stride)/2);if(a.queue.writeBuffer(E.dwU,0,new Uint32Array([r.expandCh,I,X,de,ae,r.stride,me,r.dwKernel])),r.hasProject&&p.projectConvKey){let ie=pe(p.projectConvKey),_e=pe(p.projectBNKey);E.projectW=k(B(ie),D),E.projectB=k(B(_e),D),E.projectU=k(16,ke),a.queue.writeBuffer(E.projectW,0,A(ie)),a.queue.writeBuffer(E.projectB,0,A(_e)),a.queue.writeBuffer(E.projectU,0,new Uint32Array([r.expandCh,r.outCh,de,ae]))}Ee.push(E),ue=de,le=ae}function De(g,r){let p=u.get(g);if(!p)throw new Error(`Missing weight: ${g}`);if(p.shape.length!==r)throw new Error(`Weight ${g} has rank ${p.shape.length}, expected ${r}`);return p}let ce=pe("conv_landmarks__1"),Ne=pe("conv_world_landmarks__1"),qe=pe("conv_handflag__1"),Pe=pe("conv_handedness__1"),Ge=pe("Identity"),Be=pe("Identity_1"),Re=pe("Identity_2"),We=pe("Identity_3"),Ve=k(B(ce),D),at=k(B(Ge),D),Qe=k(B(Ne),D),rt=k(B(We),D),$e=k(B(qe),D),ot=k(B(Be),D),it=k(B(Pe),D),st=k(B(Re),D);a.queue.writeBuffer(Ve,0,A(ce)),a.queue.writeBuffer(at,0,A(Ge)),a.queue.writeBuffer(Qe,0,A(Ne)),a.queue.writeBuffer(rt,0,A(We)),a.queue.writeBuffer($e,0,A(qe)),a.queue.writeBuffer(ot,0,A(Be)),a.queue.writeBuffer(it,0,A(Pe)),a.queue.writeBuffer(st,0,A(Re));let xt=k(8,ke),St=k(8,ke),Ct=k(8,ke),vt=k(8,ke);a.queue.writeBuffer(xt,0,new Uint32Array([1152,63])),a.queue.writeBuffer(St,0,new Uint32Array([1152,63])),a.queue.writeBuffer(Ct,0,new Uint32Array([1152,1])),a.queue.writeBuffer(vt,0,new Uint32Array([1152,1]));let Ut=k(8,ke);a.queue.writeBuffer(Ut,0,new Uint32Array([1152,ue*le]));let K=new Map;for(let g=0;g<Mt.length;g++)if(Mt[g].hasResidual){let r=Ee[g],p=k(4,ke);a.queue.writeBuffer(p,0,new Uint32Array([Mt[g].outCh*r.outH*r.outW])),K.set(g,p)}let ne=Ke(Lt,[J.createView(),s,O]),Nt=Ke(Te,[s,ge,Ue,w,be]),ut=new Float32Array(1),ct=new Float32Array(1),dt=new Float32Array(63);function Gt(g,r){let p=g.beginComputePass();p.setPipeline(ht),p.setBindGroup(0,Nt),p.dispatchWorkgroups(Math.ceil(112/8),Math.ceil(112/8),24),p.end();let I=w,X=s;for(let de=0;de<Mt.length;de++){let ae=Mt[de],E=Ee[de];if(ae.hasResidual){let me=ae.inCh*E.inH*E.inW*l;g.copyBufferToBuffer(I,0,q,0,me)}if(p=g.beginComputePass(),E.expandW){let me=Ke(Te,[I,E.expandW,E.expandB,h,E.expandU]);p.setPipeline(Ht),p.setBindGroup(0,me),p.dispatchWorkgroups(Math.ceil(E.inW/8),Math.ceil(E.inH/8),ae.expandCh)}let pt=E.expandW?h:I,Ze=Ke(Te,[pt,E.dwW,E.dwB,b,E.dwU]);if(p.setPipeline(Ie),p.setBindGroup(0,Ze),p.dispatchWorkgroups(Math.ceil(E.outW/8),Math.ceil(E.outH/8),ae.expandCh),ae.hasProject&&E.projectW){let me=(ae.hasResidual,X),ie=Ke(Te,[b,E.projectW,E.projectB,me,E.projectU]);if(p.setPipeline(Bt),p.setBindGroup(0,ie),p.dispatchWorkgroups(Math.ceil(E.outW/8),Math.ceil(E.outH/8),ae.outCh),ae.hasResidual){let _e=K.get(de),Je=Ke(mt,[X,q,I,_e]);p.setPipeline(At),p.setBindGroup(0,Je),p.dispatchWorkgroups(Math.ceil(ae.outCh*E.outH*E.outW/256))}else{let _e=I;I=X,X=_e}}if(p.end(),!ae.hasProject){p=g.beginComputePass();let me=Ke(we,[b,V,Ut]);p.setPipeline(It),p.setBindGroup(0,me),p.dispatchWorkgroups(Math.ceil(1152/256));let ie=Ke(He,[V,Ve,at,Z,xt]);p.setPipeline(Wt),p.setBindGroup(0,ie),p.dispatchWorkgroups(1);let _e=Ke(He,[V,$e,ot,Q,Ct]);p.setPipeline(kt),p.setBindGroup(0,_e),p.dispatchWorkgroups(1);let Je=Ke(He,[V,it,st,Y,vt]);p.setPipeline(kt),p.setBindGroup(0,Je),p.dispatchWorkgroups(1),p.end(),g.copyBufferToBuffer(Q,0,$,0,4),g.copyBufferToBuffer(Y,0,$,4,4),g.copyBufferToBuffer(Z,0,$,8,252),r&&g.copyBufferToBuffer($,0,r,0,260);return}}}async function qt(g){a.queue.writeBuffer(ee,0,g);let r=a.createCommandEncoder();r.copyBufferToBuffer(ee,0,s,0,672*224*4),Gt(r,T),a.queue.submit([r.finish()]);let p=T.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await p;let I=new Float32Array(T.getMappedRange());ut[0]=I[0],ct[0]=I[1];for(let X=0;X<63;X++)dt[X]=I[2+X]/224;return T.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Tt(g){a.queue.copyExternalImageToTexture({source:g},{texture:J},[224,224]);let r=a.createCommandEncoder();{let X=r.beginComputePass();X.setPipeline(Pt),X.setBindGroup(0,ne),X.dispatchWorkgroups(Math.ceil(224/16),Math.ceil(224/16),1),X.end()}Gt(r,T),a.queue.submit([r.finish()]);let p=T.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await p;let I=new Float32Array(T.getMappedRange());ut[0]=I[0],ct[0]=I[1];for(let X=0;X<63;X++)dt[X]=I[2+X]/224;return T.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Zt(g){let r=a.createCommandEncoder();r.copyBufferToBuffer(g,0,s,0,672*224*4),Gt(r,T),a.queue.submit([r.finish()]);let p=T.mapAsync(GPUMapMode.READ);await a.queue.onSubmittedWorkDone(),await p;let I=new Float32Array(T.getMappedRange());ut[0]=I[0],ct[0]=I[1];for(let X=0;X<63;X++)dt[X]=I[2+X]/224;return T.unmap(),{handflag:new Float32Array(ut),handedness:new Float32Array(ct),landmarks:new Float32Array(dt)}}async function Qt(){return null}async function Jt(){return null}async function Ot(g=100){let r=new OffscreenCanvas(224,224),p=r.getContext("2d");p.fillStyle="#886644",p.fillRect(0,0,224,224);for(let ae=0;ae<5;ae++)await Tt(r);let I=performance.now();for(let ae=0;ae<g;ae++)await Tt(r);let de=(performance.now()-I)/g;return{avgMs:de,fps:1e3/de}}async function Yt(g=100){let r=await Ot(g);return{...r,medianMs:r.avgMs,minMs:r.avgMs}}async function jt(g){return Tt(g)}async function Xt(){return{gpuOnly:{median:0,min:0},mapAsyncOnly:{median:0,min:0},mapAsyncNoWait:{median:0,min:0},total:{median:0,min:0},pipelined:{median:0,min:0},renderReadback:null}}async function Vt(g){let r={};async function p(de,ae,E){let pt=ae*4,Ze=a.createBuffer({size:pt,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),me=a.createCommandEncoder();me.copyBufferToBuffer(de,0,Ze,0,pt),a.queue.submit([me.finish()]),await a.queue.onSubmittedWorkDone(),await Ze.mapAsync(GPUMapMode.READ);let ie=new Float32Array(Ze.getMappedRange()),_e=1/0,Je=-1/0,$t=0;for(let z=0;z<ie.length;z++)ie[z]<_e&&(_e=ie[z]),ie[z]>Je&&(Je=ie[z]),ie[z]!==0&&$t++;let en=Array.from(ie.slice(0,5));Ze.unmap(),Ze.destroy(),r[E]={min:_e,max:Je,nonZero:$t,total:ae,sample:en}}let I=new Float32Array(672*224);for(let de=0;de<50176;de++)I[de]=.5,I[50176+de]=.3,I[448*224+de]=.7;a.queue.writeBuffer(ee,0,I);let X=a.createCommandEncoder();return X.copyBufferToBuffer(ee,0,s,0,672*224*4),Gt(X,T),a.queue.submit([X.finish()]),await a.queue.onSubmittedWorkDone(),await p(s,672*224,"inputBufA"),await p(w,2688*112,"afterInitConvBufB"),await p(V,1152,"gapOutput"),await p(Z,63,"landmarks"),await p(Q,1,"handflag"),await p($,65,"unifiedOutput"),r}return{device:a,run:qt,runFromCanvas:Tt,runFromGPUBuffer:Zt,runFromCanvasPipelined:Qt,flushPipelined:Jt,benchmark:Ot,benchmarkGPU:Yt,runFromCanvasViaRender:jt,benchmarkDiagnostic:Xt,debugLayerOutputs:Vt}}function lt(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var cn=lt(`
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
`),dn=lt(`
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
`),pn=lt(`
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
`),fn=lt(`
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
`),ln=lt(`
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
`),mn=lt(`
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
`),hn=lt(`
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
`),gn=lt(`
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
`),bn=lt(`
@group(0)@binding(0) var<storage,read_write> buf:array<f32>;
@group(0)@binding(1) var<uniform> count:u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let idx=gid.x;
  if(idx>=count){return;}
  let v=buf[idx];
  buf[idx]=unpack2x16float(pack2x16float(vec2(v,0.0))).x;
}
`);async function _n(u,U){let n;if(U)n=U;else{if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");n=await e.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,8)}})}let W={r:"read-only-storage",s:"storage",u:"uniform"};function M(e){return n.createBindGroupLayout({entries:e.map((t,i)=>t==="t"?{binding:i,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}}:t==="sm"?{binding:i,visibility:GPUShaderStage.COMPUTE,sampler:{}}:{binding:i,visibility:GPUShaderStage.COMPUTE,buffer:{type:W[t]}})})}let a=n.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),y=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,L=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,A=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,B=GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST;function l(e,t){return n.createBuffer({size:Math.max(e,4),usage:t})}function P(e,t,i){n.queue.writeBuffer(e,t,i)}function G(e){let t=l(e.data.byteLength,y);return P(t,0,e.data),t}let re=Array.from(u.keys());function N(e){let t=u.get(e);if(!t)throw new Error(`Weight not found: ${e}`);return t}function d(...e){let t=re.find(i=>e.every(x=>i.includes(x)));if(!t)throw new Error(`Weight not found for: ${e.join(", ")}`);return N(t)}function D(e){let[,t,i,x]=e.shape,R=new Float32Array(x*25);for(let _=0;_<x;_++)for(let se=0;se<t;se++)for(let m=0;m<i;m++)R[_*25+se*5+m]=e.data[se*i*x+m*x+_];return R}function C(e){let[t,,,i]=e.shape,x=new Float32Array(t*i);for(let R=0;R<t;R++)for(let _=0;_<i;_++)x[R*i+_]=e.data[R*i+_];return x}let ye=n.createShaderModule({code:cn}),Le=n.createShaderModule({code:dn}),ke=n.createShaderModule({code:pn}),k=n.createShaderModule({code:fn}),Ke=n.createShaderModule({code:mn}),Se=n.createShaderModule({code:ln}),pe=n.createShaderModule({code:hn}),Dt=n.createShaderModule({code:gn}),Rt=n.createShaderModule({code:bn}),ve=M(["r","r","r","r","s","u"]),tt=M(["r","r","r","s","u"]),je=M(["r","r","r","r","r","s","u"]),Fe=M(["r","r","r","s","u"]),wt=M(["r","r","r","r","s","u"]),ze=M(["r","r","s","u"]),nt=M(["t","s","u"]),Te=M(["t","s","u","sm"]),mt=M(["s","u"]);function we(e,t){return n.createComputePipeline({layout:n.createPipelineLayout({bindGroupLayouts:[e]}),compute:{module:t,entryPoint:"main"}})}let He=we(ve,ye),Lt=we(tt,Le),Pt=we(je,ke),ht=we(Fe,k),Ht=we(wt,Ke),Ie=we(ze,Se),Bt=we(nt,pe),At=we(Te,Dt),It=we(mt,Rt),Wt=d("conv2d/Conv2D"),kt=d("batch_normalization/","conv2d/Conv2D"),Ft=d("p_re_lu/"),Xe=G(Wt),s=G(kt),w=G(Ft),b=[{dwKey:"depthwise_conv2d/",pwKey:"conv2d_1/",bnKey:"batch_normalization_1/",preluKey:"p_re_lu_1/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_1/",pwKey:"conv2d_2/",bnKey:"batch_normalization_2/",preluKey:"p_re_lu_2/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_2/",pwKey:"conv2d_3/",bnKey:"batch_normalization_3/",preluKey:"p_re_lu_3/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_3/",pwKey:"conv2d_4/",bnKey:"batch_normalization_4/",preluKey:"p_re_lu_4/",inCh:32,outCh:32,stride:1,inH:96},{dwKey:"depthwise_conv2d_4/",pwKey:"conv2d_5/",bnKey:"batch_normalization_5/",preluKey:"p_re_lu_5/",inCh:32,outCh:64,stride:2,inH:96},{dwKey:"depthwise_conv2d_5/",pwKey:"conv2d_6/",bnKey:"batch_normalization_6/",preluKey:"p_re_lu_6/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_6/",pwKey:"conv2d_7/",bnKey:"batch_normalization_7/",preluKey:"p_re_lu_7/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_7/",pwKey:"conv2d_8/",bnKey:"batch_normalization_8/",preluKey:"p_re_lu_8/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_8/",pwKey:"conv2d_9/",bnKey:"batch_normalization_9/",preluKey:"p_re_lu_9/",inCh:64,outCh:64,stride:1,inH:48},{dwKey:"depthwise_conv2d_9/",pwKey:"conv2d_10/",bnKey:"batch_normalization_10/",preluKey:"p_re_lu_10/",inCh:64,outCh:128,stride:2,inH:48},{dwKey:"depthwise_conv2d_10/",pwKey:"conv2d_11/",bnKey:"batch_normalization_11/",preluKey:"p_re_lu_11/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_11/",pwKey:"conv2d_12/",bnKey:"batch_normalization_12/",preluKey:"p_re_lu_12/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_12/",pwKey:"conv2d_13/",bnKey:"batch_normalization_13/",preluKey:"p_re_lu_13/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_13/",pwKey:"conv2d_14/",bnKey:"batch_normalization_14/",preluKey:"p_re_lu_14/",inCh:128,outCh:128,stride:1,inH:24},{dwKey:"depthwise_conv2d_14/",pwKey:"conv2d_15/",bnKey:"batch_normalization_15/",preluKey:"p_re_lu_15/",inCh:128,outCh:256,stride:2,inH:24},{dwKey:"depthwise_conv2d_15/",pwKey:"conv2d_16/",bnKey:"batch_normalization_16/",preluKey:"p_re_lu_16/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_16/",pwKey:"conv2d_17/",bnKey:"batch_normalization_17/",preluKey:"p_re_lu_17/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_17/",pwKey:"conv2d_18/",bnKey:"batch_normalization_18/",preluKey:"p_re_lu_18/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_18/",pwKey:"conv2d_19/",bnKey:"batch_normalization_19/",preluKey:"p_re_lu_19/",inCh:256,outCh:256,stride:1,inH:12},{dwKey:"depthwise_conv2d_19/",pwKey:"conv2d_20/",bnKey:"batch_normalization_20/",preluKey:"p_re_lu_20/",inCh:256,outCh:256,stride:2,inH:12},{dwKey:"depthwise_conv2d_20/",pwKey:"conv2d_21/",bnKey:"batch_normalization_21/",preluKey:"p_re_lu_21/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_21/",pwKey:"conv2d_22/",bnKey:"batch_normalization_22/",preluKey:"p_re_lu_22/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_22/",pwKey:"conv2d_23/",bnKey:"batch_normalization_23/",preluKey:"p_re_lu_23/",inCh:256,outCh:256,stride:1,inH:6},{dwKey:"depthwise_conv2d_23/",pwKey:"conv2d_24/",bnKey:"batch_normalization_24/",preluKey:"p_re_lu_24/",inCh:256,outCh:256,stride:1,inH:6}].map(e=>{let t=d(e.dwKey),i=d(e.pwKey),x=d(e.bnKey),R=d(e.preluKey),_=D(t),se=l(_.byteLength,y);P(se,0,_);let m=new Float32Array(e.inCh),Me=l(m.byteLength,y);P(Me,0,m);let xe=C(i),f=l(xe.byteLength,y);P(f,0,xe);let S=G(x),v=G(R);return{dwWeightBuf:se,dwBiasBuf:Me,pwWeightBuf:f,pwBiasBuf:S,alphaBuf:v,inCh:e.inCh,outCh:e.outCh,stride:e.stride,inH:e.inH}}),q=C(d("conv2d_25/Conv2D")),ee=l(q.byteLength,y);P(ee,0,q);let V=G(d("batch_normalization_25/")),Z=G(d("p_re_lu_25/")),te={dwWeightBuf:(()=>{let e=D(d("depthwise_conv2d_24/")),t=l(e.byteLength,y);return P(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=l(e.byteLength,y);return P(t,0,e),t})(),pwWeightBuf:(()=>{let e=C(d("conv2d_26/")),t=l(e.byteLength,y);return P(t,0,e),t})(),pwBiasBuf:G(d("batch_normalization_26/")),alphaBuf:G(d("p_re_lu_26/")),inCh:256,outCh:256,stride:1,inH:12},Q={dwWeightBuf:(()=>{let e=D(d("depthwise_conv2d_25/")),t=l(e.byteLength,y);return P(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(256),t=l(e.byteLength,y);return P(t,0,e),t})(),pwWeightBuf:(()=>{let e=C(d("conv2d_27/Conv2D1")),t=l(e.byteLength,y);return P(t,0,e),t})(),pwBiasBuf:G(d("batch_normalization_27/")),alphaBuf:G(d("p_re_lu_27/")),inCh:256,outCh:256,stride:1,inH:12},Y=C(d("conv2d_28/Conv2D")),$=l(Y.byteLength,y);P($,0,Y);let T=G(d("batch_normalization_28/")),J=G(d("p_re_lu_28/")),O={dwWeightBuf:(()=>{let e=D(d("depthwise_conv2d_26/")),t=l(e.byteLength,y);return P(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=l(e.byteLength,y);return P(t,0,e),t})(),pwWeightBuf:(()=>{let e=C(d("conv2d_29/")),t=l(e.byteLength,y);return P(t,0,e),t})(),pwBiasBuf:G(d("batch_normalization_29/")),alphaBuf:G(d("p_re_lu_29/")),inCh:128,outCh:128,stride:1,inH:24},j={dwWeightBuf:(()=>{let e=D(d("depthwise_conv2d_27/")),t=l(e.byteLength,y);return P(t,0,e),t})(),dwBiasBuf:(()=>{let e=new Float32Array(128),t=l(e.byteLength,y);return P(t,0,e),t})(),pwWeightBuf:(()=>{let e=C(d("conv2d_30/Conv2D1")),t=l(e.byteLength,y);return P(t,0,e),t})(),pwBiasBuf:G(d("batch_normalization_30/")),alphaBuf:G(d("p_re_lu_30/")),inCh:128,outCh:128,stride:1,inH:24},fe=C(d("classifier_palm_16_NO_PRUNING/Conv2D")),oe=l(fe.byteLength,y);P(oe,0,fe);let he=G(d("classifier_palm_16_NO_PRUNING/BiasAdd")),ge=C(d("regressor_palm_16_NO_PRUNING/Conv2D")),Ue=l(ge.byteLength,y);P(Ue,0,ge);let be=G(d("regressor_palm_16_NO_PRUNING/BiasAdd")),ue=C(d("classifier_palm_8_NO_PRUNING/Conv2D")),le=l(ue.byteLength,y);P(le,0,ue);let Ee=G(d("classifier_palm_8_NO_PRUNING/BiasAdd")),De=C(d("regressor_palm_8_NO_PRUNING/Conv2D")),ce=l(De.byteLength,y);P(ce,0,De);let Ne=G(d("regressor_palm_8_NO_PRUNING/BiasAdd")),qe=Math.max(36864*3,9216*64,2304*128,576*256,144*256)*4,Pe=l(36864*3*4,y),Ge=l(qe,L),Be=l(qe,L),Re=l(qe,L),We=l(576*256*4,L),Ve=new Map;function at(e){let t=Ve.get(e);return t||(t=l(4,B),P(t,0,new Uint32Array([e])),Ve.set(e,t)),t}let Qe=l(144*256*4,L|GPUBufferUsage.COPY_DST),rt=l(576*128*4,L|GPUBufferUsage.COPY_DST),$e=l(864*4,A),ot=l(15552*4,A),it=l(576*2*4,A),st=l(576*36*4,A),xt=l(864*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),St=l(15552*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ct=l(576*2*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),vt=l(576*36*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST),Ut=n.createTexture({size:[192,192,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});function K(e,t){return Math.ceil(e/t)}function ne(e){let t=l(e.byteLength,B);return P(t,0,e),t}let Nt=ne(new Uint32Array([1,3,32,192,192,96,96])),ut=b.map(e=>{let t=e.stride===2?e.inH/2:e.inH,i=t,x=e.stride===2?1:2,R=e.inCh;return{dw:ne(new Uint32Array([1,e.inCh,e.inH,e.inH,t,i,e.stride,x])),pw:ne(new Uint32Array([1,e.inCh,e.outCh,t,i,R,e.stride,e.inH,e.inH])),outH:t,outW:i}}),ct=(()=>{let e=te,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ne(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ne(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),dt=(()=>{let e=Q,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ne(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ne(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Gt=(()=>{let e=O,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ne(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ne(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),qt=(()=>{let e=j,t=e.stride===2?e.inH/2:e.inH,i=e.stride===2?1:2;return{dw:ne(new Uint32Array([1,e.inCh,e.inH,e.inH,t,t,e.stride,i])),pw:ne(new Uint32Array([1,e.inCh,e.outCh,t,t,e.inCh,e.stride,e.inH,e.inH])),outH:t}})(),Tt=ne(new Uint32Array([1,256,6,6,12,12])),Zt=ne(new Uint32Array([1,256,12,12,12,12])),Qt=ne(new Uint32Array([1,256,12,12,24,24])),Jt=ne(new Uint32Array([1,128,24,24,24,24])),Ot=ne(new Uint32Array([1,256,256,12,12])),Yt=ne(new Uint32Array([1,256,128,24,24])),jt=ne(new Uint32Array([1,256,6,12,12])),Xt=ne(new Uint32Array([1,256,108,12,12])),Vt=ne(new Uint32Array([1,128,2,24,24])),g=ne(new Uint32Array([1,128,36,24,24])),r=ne(new Uint32Array([192,192,192])),p=n.createBindGroup({layout:nt,entries:[{binding:0,resource:Ut.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:r}}]}),I=null,X=0,de=0,ae=l(32,B);function E(e,t){return I&&X===e&&de===t||(I&&I.destroy(),I=n.createTexture({size:[e,t,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),X=e,de=t),I}let pt=n.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:Pe}},{binding:1,resource:{buffer:Xe}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:Ge}},{binding:5,resource:{buffer:Nt}}]});function Ze(e,t,i){}function me(e,t,i,x,R,_){let se=_.outH,m=n.createBindGroup({layout:tt,entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:t.dwWeightBuf}},{binding:2,resource:{buffer:t.dwBiasBuf}},{binding:3,resource:{buffer:Re}},{binding:4,resource:{buffer:_.dw}}]}),Me=e.beginComputePass();Me.setPipeline(Lt),Me.setBindGroup(0,m),Me.dispatchWorkgroups(K(se,8),K(_.outH,8),t.inCh),Me.end(),t.inCh*_.outH*se;let xe=n.createBindGroup({layout:je,entries:[{binding:0,resource:{buffer:Re}},{binding:1,resource:{buffer:R}},{binding:2,resource:{buffer:t.pwWeightBuf}},{binding:3,resource:{buffer:t.pwBiasBuf}},{binding:4,resource:{buffer:t.alphaBuf}},{binding:5,resource:{buffer:x}},{binding:6,resource:{buffer:_.pw}}]}),f=e.beginComputePass();f.setPipeline(Pt),f.setBindGroup(0,xe),f.dispatchWorkgroups(K(se,8),K(_.outH,8),t.outCh),f.end(),t.outCh*_.outH*se}function ie(e,t,i,x,R,_,se,m,Me){let xe=n.createBindGroup({layout:Fe,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:x}},{binding:3,resource:{buffer:R}},{binding:4,resource:{buffer:_}}]}),f=e.beginComputePass();f.setPipeline(ht),f.setBindGroup(0,xe),f.dispatchWorkgroups(K(Me,8),K(m,8),se),f.end()}function _e(e,t,i,x,R,_,se,m,Me,xe){let f=n.createBindGroup({layout:wt,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:x}},{binding:3,resource:{buffer:R}},{binding:4,resource:{buffer:_}},{binding:5,resource:{buffer:se}}]}),S=e.beginComputePass();S.setPipeline(Ht),S.setBindGroup(0,f),S.dispatchWorkgroups(K(xe,8),K(Me,8),m),S.end()}async function Je(e){36864*3;{let v=e.beginComputePass();v.setPipeline(He),v.setBindGroup(0,pt),v.dispatchWorkgroups(K(96,8),K(96,8),32),v.end()}9216*32;let t=Ge,i=Be;for(let v=0;v<b.length;v++){let F=b[v];me(e,F,t,i,t,ut[v]);let Ae=t;t=i,i=Ae,v===13&&e.copyBufferToBuffer(t,0,rt,0,576*128*4),v===18&&e.copyBufferToBuffer(t,0,Qe,0,144*256*4)}{let v=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:We}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Tt}}]}),F=e.beginComputePass();F.setPipeline(Ie),F.setBindGroup(0,v),F.dispatchWorkgroups(K(12,8),K(12,8),256),F.end()}{let v=t;t=i,i=v}144*256,_e(e,t,ee,V,Z,i,Ot,256,12,12);{let v=t;t=i,i=v}144*256;{let v=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:Qe}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Zt}}]}),F=e.beginComputePass();F.setPipeline(Ie),F.setBindGroup(0,v),F.dispatchWorkgroups(K(12,8),K(12,8),256),F.end()}{let v=t;t=i,i=v}144*256,me(e,te,t,i,t,ct);{let v=t;t=i,i=v}me(e,Q,t,i,t,dt);{let v=t;t=i,i=v}ie(e,t,oe,he,$e,jt,6,12,12),ie(e,t,Ue,be,ot,Xt,108,12,12);{let v=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:We}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Qt}}]}),F=e.beginComputePass();F.setPipeline(Ie),F.setBindGroup(0,v),F.dispatchWorkgroups(K(24,8),K(24,8),256),F.end()}{let v=t;t=i,i=v}576*256,_e(e,t,$,T,J,i,Yt,128,24,24);{let v=t;t=i,i=v}576*128;{let v=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:t}},{binding:1,resource:{buffer:rt}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:Jt}}]}),F=e.beginComputePass();F.setPipeline(Ie),F.setBindGroup(0,v),F.dispatchWorkgroups(K(24,8),K(24,8),128),F.end()}{let v=t;t=i,i=v}576*128,me(e,O,t,i,t,Gt);{let v=t;t=i,i=v}me(e,j,t,i,t,qt);{let v=t;t=i,i=v}ie(e,t,le,Ee,it,Vt,2,24,24),576*2,ie(e,t,ce,Ne,st,g,36,24,24),576*36,n.queue.submit([e.finish()]);let x=n.createCommandEncoder();x.copyBufferToBuffer($e,0,xt,0,864*4),x.copyBufferToBuffer(ot,0,St,0,15552*4),x.copyBufferToBuffer(it,0,Ct,0,576*2*4),x.copyBufferToBuffer(st,0,vt,0,576*36*4),n.queue.submit([x.finish()]),await Promise.all([xt.mapAsync(GPUMapMode.READ),St.mapAsync(GPUMapMode.READ),Ct.mapAsync(GPUMapMode.READ),vt.mapAsync(GPUMapMode.READ)]);let R=new Float32Array(xt.getMappedRange()).slice(),_=new Float32Array(St.getMappedRange()).slice(),se=new Float32Array(Ct.getMappedRange()).slice(),m=new Float32Array(vt.getMappedRange()).slice();xt.unmap(),St.unmap(),Ct.unmap(),vt.unmap();let Me=2016,xe=new Float32Array(Me),f=new Float32Array(Me*18),S=0;for(let v=0;v<12;v++)for(let F=0;F<12;F++)for(let Ae=0;Ae<6;Ae++){xe[S]=R[Ae*144+v*12+F];for(let Ye=0;Ye<18;Ye++){let o=Ae*18+Ye;f[S*18+Ye]=_[o*144+v*12+F]}S++}for(let v=0;v<24;v++)for(let F=0;F<24;F++)for(let Ae=0;Ae<2;Ae++){xe[S]=se[Ae*576+v*24+F];for(let Ye=0;Ye<18;Ye++){let o=Ae*18+Ye;f[S*18+Ye]=m[o*576+v*24+F]}S++}return{scores:xe,regressors:f}}async function $t(e){n.queue.copyExternalImageToTexture({source:e},{texture:Ut},[192,192]);let t=n.createCommandEncoder();{let i=t.beginComputePass();i.setPipeline(Bt),i.setBindGroup(0,p),i.dispatchWorkgroups(K(192,16),K(192,16),1),i.end()}return Je(t)}async function en(e,t,i){let x=Math.min(192/t,192/i),R=Math.round(t*x),_=Math.round(i*x),se=Math.floor((192-R)/2),m=Math.floor((192-_)/2),Me=se/192,xe=m/192,f=E(t,i),S;e instanceof HTMLVideoElement?S=await createImageBitmap(e,{colorSpaceConversion:"none"}):e instanceof HTMLImageElement?S=await createImageBitmap(e,{colorSpaceConversion:"none"}):S=e,n.queue.copyExternalImageToTexture({source:S},{texture:f},[t,i]);let v=new ArrayBuffer(32),F=new Uint32Array(v),Ae=new Float32Array(v);F[0]=t,F[1]=i,F[2]=192,F[3]=0,Ae[4]=t/R,Ae[5]=i/_,Ae[6]=se,Ae[7]=m,n.queue.writeBuffer(ae,0,v);let Ye=n.createBindGroup({layout:Te,entries:[{binding:0,resource:f.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:ae}},{binding:3,resource:a}]}),o=n.createCommandEncoder();{let H=o.beginComputePass();H.setPipeline(At),H.setBindGroup(0,Ye),H.dispatchWorkgroups(K(192,16),K(192,16),1),H.end()}return{output:await Je(o),lbPadX:Me,lbPadY:xe}}async function z(e,t){let i=n.createBuffer({size:t*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),x=n.createCommandEncoder();x.copyBufferToBuffer(e,0,i,0,t*4),n.queue.submit([x.finish()]),await i.mapAsync(GPUMapMode.READ);let R=new Float32Array(i.getMappedRange()).slice();return i.unmap(),i.destroy(),R}async function Bn(e,t,i){function x(o,c=1e3){let H=o.slice(0,c),Ce=Math.max(0,Math.floor(o.length/2)-250);return{min:Math.min(...H),max:Math.max(...H),mean:H.reduce((Oe,gt)=>Oe+gt,0)/H.length,nonZero:H.filter(Oe=>Oe!==0).length,sample:Array.from(H.slice(0,10)),data500:Array.from(o.slice(0,500)),dataMid500:Array.from(o.slice(Ce,Ce+500)),totalLength:o.length}}function R(o,c,H,Ce){let Oe=[],gt=Math.floor(H/2),bt=Math.floor(Ce/2),et=H*Ce;for(let _t=0;_t<c&&Oe.length<500;_t++)for(let Kt=-1;Kt<=1&&Oe.length<500;Kt++)for(let Et=-1;Et<=1&&Oe.length<500;Et++){let yt=gt+Kt,tn=bt+Et;yt>=0&&yt<H&&tn>=0&&tn<Ce&&Oe.push(o[_t*et+yt*Ce+tn])}return Oe}let _={},se;if(e instanceof HTMLImageElement?(t=t??e.naturalWidth,i=i??e.naturalHeight,se=await createImageBitmap(e,{colorSpaceConversion:"none"})):(t=t??e.width??192,i=i??e.height??192,se=e),t!==192||i!==192){let o=Math.min(192/t,192/i),c=Math.round(t*o),H=Math.round(i*o),Ce=Math.floor((192-c)/2),Oe=Math.floor((192-H)/2),gt=E(t,i);n.queue.copyExternalImageToTexture({source:se},{texture:gt},[t,i]);let bt=new ArrayBuffer(32),et=new Uint32Array(bt),_t=new Float32Array(bt);et[0]=t,et[1]=i,et[2]=192,et[3]=0,_t[4]=t/c,_t[5]=i/H,_t[6]=Ce,_t[7]=Oe,n.queue.writeBuffer(ae,0,bt);let Kt=n.createBindGroup({layout:Te,entries:[{binding:0,resource:gt.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:ae}},{binding:3,resource:a}]});{let Et=n.createCommandEncoder(),yt=Et.beginComputePass();yt.setPipeline(At),yt.setBindGroup(0,Kt),yt.dispatchWorkgroups(K(192,16),K(192,16),1),yt.end(),n.queue.submit([Et.finish()])}}else{n.queue.copyExternalImageToTexture({source:se},{texture:Ut},[192,192]);let o=ne(new Uint32Array([192,192,192])),c=n.createBindGroup({layout:nt,entries:[{binding:0,resource:Ut.createView()},{binding:1,resource:{buffer:Pe}},{binding:2,resource:{buffer:o}}]});{let H=n.createCommandEncoder(),Ce=H.beginComputePass();Ce.setPipeline(Bt),Ce.setBindGroup(0,c),Ce.dispatchWorkgroups(K(192,16),K(192,16),1),Ce.end(),n.queue.submit([H.finish()])}}{let o=await z(Pe,110592),c=x(o);c.dataCenter500=R(o,3,192,192),_.input=c}let m=n.createCommandEncoder(),Me=n.createBindGroup({layout:ve,entries:[{binding:0,resource:{buffer:Pe}},{binding:1,resource:{buffer:Xe}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:w}},{binding:4,resource:{buffer:Ge}},{binding:5,resource:{buffer:Nt}}]}),xe=m.beginComputePass();xe.setPipeline(He),xe.setBindGroup(0,Me),xe.dispatchWorkgroups(K(96,8),K(96,8),32),xe.end(),n.queue.submit([m.finish()]);{let o=await z(Ge,294912),c=x(o);c.dataCenter500=R(o,32,96,96),_.initConv=c}let f=Ge,S=Be;for(let o=0;o<b.length;o++){let c=b[o];m=n.createCommandEncoder(),me(m,c,f,S,f,ut[o]),n.queue.submit([m.finish()]);let H=f;f=S,S=H;{let Ce=c.stride===2?c.inH/2:c.inH,Oe=Ce,gt=Ce*Oe*c.outCh,bt=await z(f,gt),et=x(bt);et.dataCenter500=R(bt,c.outCh,Ce,Oe),et.spatialShape=[c.outCh,Ce,Oe],_[`block${o}`]=et}o===13&&(m=n.createCommandEncoder(),m.copyBufferToBuffer(f,0,rt,0,576*128*4),n.queue.submit([m.finish()])),o===18&&(m=n.createCommandEncoder(),m.copyBufferToBuffer(f,0,Qe,0,144*256*4),n.queue.submit([m.finish()]))}m=n.createCommandEncoder();{let o=ne(new Uint32Array([1,256,6,6,12,12])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:We}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),H=m.beginComputePass();H.setPipeline(Ie),H.setBindGroup(0,c),H.dispatchWorkgroups(K(12,8),K(12,8),256),H.end()}n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,36864),c=x(o);c.dataCenter500=R(o,256,12,12),_.fpnUpsample6to12=c}m=n.createCommandEncoder(),_e(m,f,ee,V,Z,S,Ot,256,12,12),n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,36864),c=x(o);c.dataCenter500=R(o,256,12,12),_.fpn6to12Conv=c}{let o=await z(Qe,36864),c=x(o);c.dataCenter500=R(o,256,12,12),_.backbone12Skip=c}m=n.createCommandEncoder();{let o=ne(new Uint32Array([1,256,12,12,12,12])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:Qe}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),H=m.beginComputePass();H.setPipeline(Ie),H.setBindGroup(0,c),H.dispatchWorkgroups(K(12,8),K(12,8),256),H.end()}n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,36864),c=x(o);c.dataCenter500=R(o,256,12,12),_.fpnAdd12=c}m=n.createCommandEncoder(),me(m,te,f,S,f,ct),n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,36864),c=x(o);c.dataCenter500=R(o,256,12,12),_.fpn12Block1=c}m=n.createCommandEncoder(),me(m,Q,f,S,f,dt),n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,36864),c=x(o);c.dataCenter500=R(o,256,12,12),_.fpn12Block2=c}m=n.createCommandEncoder(),ie(m,f,oe,he,$e,jt,6,12,12),n.queue.submit([m.finish()]);{let o=await z($e,864),c=x(o);c.dataCenter500=R(o,6,12,12),_.cls16=c}m=n.createCommandEncoder(),ie(m,f,Ue,be,ot,Xt,108,12,12),n.queue.submit([m.finish()]);{let o=await z(ot,15552),c=x(o,500);c.dataCenter500=R(o,108,12,12),_.reg16=c}m=n.createCommandEncoder();{let o=ne(new Uint32Array([1,256,12,12,24,24])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:We}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),H=m.beginComputePass();H.setPipeline(Ie),H.setBindGroup(0,c),H.dispatchWorkgroups(K(24,8),K(24,8),256),H.end()}n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,147456),c=x(o);c.dataCenter500=R(o,256,24,24),_.fpnUpsample12to24=c}m=n.createCommandEncoder(),_e(m,f,$,T,J,S,Yt,128,24,24),n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,73728),c=x(o);c.dataCenter500=R(o,128,24,24),_.fpn12to24Conv=c}{let o=await z(rt,73728),c=x(o);c.dataCenter500=R(o,128,24,24),_.backbone24Skip=c}m=n.createCommandEncoder();{let o=ne(new Uint32Array([1,128,24,24,24,24])),c=n.createBindGroup({layout:ze,entries:[{binding:0,resource:{buffer:f}},{binding:1,resource:{buffer:rt}},{binding:2,resource:{buffer:S}},{binding:3,resource:{buffer:o}}]}),H=m.beginComputePass();H.setPipeline(Ie),H.setBindGroup(0,c),H.dispatchWorkgroups(K(24,8),K(24,8),128),H.end()}n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,73728),c=x(o);c.dataCenter500=R(o,128,24,24),_.fpnAdd24=c}m=n.createCommandEncoder(),me(m,O,f,S,f,Gt),n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,73728),c=x(o);c.dataCenter500=R(o,128,24,24),_.fpn24Block1=c}m=n.createCommandEncoder(),me(m,j,f,S,f,qt),n.queue.submit([m.finish()]);{let o=f;f=S,S=o}{let o=await z(f,73728),c=x(o);c.dataCenter500=R(o,128,24,24),_.fpn24Block2=c}m=n.createCommandEncoder(),ie(m,f,le,Ee,it,Vt,2,24,24),n.queue.submit([m.finish()]);{let o=await z(it,1152),c=x(o);c.dataCenter500=R(o,2,24,24),_.cls8=c}m=n.createCommandEncoder(),ie(m,f,ce,Ne,st,g,36,24,24),n.queue.submit([m.finish()]);{let o=await z(st,20736),c=x(o);c.dataCenter500=R(o,36,24,24),_.reg8=c}_.initWeights=x(await z(Xe,100),100),_.initBias=x(await z(s,32),32),_.cls16Weights=x(await z(oe,100),100),_.cls16Bias=x(await z(he,6),6),_.cls8Weights=x(await z(le,100),100),_.cls8Bias=x(await z(Ee,2),2),_.fpn6to12Weights=x(await z(ee,100),100);let v=await z($e,864),F=await z(it,576*2);_.rawScores=new Float32Array(2016),_.rawScores.set(v,0),_.rawScores.set(F,864);let Ae=await z(ot,15552),Ye=await z(st,576*36);return _.rawRegressors=new Float32Array(36288),_.rawRegressors.set(Ae,0),_.rawRegressors.set(Ye,15552),_.rawInput=await z(Pe,36864*3),_}return{device:n,run:$t,runWithResize:en,debugRun:Bn}}function Ln(){let u=[];for(let U=0;U<12;U++)for(let n=0;n<12;n++){let W=(n+.5)/12,M=(U+.5)/12;for(let a=0;a<6;a++)u.push({x:W,y:M})}for(let U=0;U<24;U++)for(let n=0;n<24;n++){let W=(n+.5)/24,M=(U+.5)/24;for(let a=0;a<2;a++)u.push({x:W,y:M})}return u}var yn=Ln();function Hn(u){return 1/(1+Math.exp(-u))}function nn(u,U){let n=[],{scores:W,regressors:M}=u,a=192;for(let y=0;y<yn.length;y++){let L=Hn(W[y]);if(L<U)continue;let A=yn[y],B=y*18,l=A.x+M[B+0]/a,P=A.y+M[B+1]/a,G=M[B+2]/a,re=M[B+3]/a,N=[];for(let d=0;d<7;d++){let D=A.x+M[B+4+d*2]/a,C=A.y+M[B+4+d*2+1]/a;N.push([D,C])}n.push({score:L,box:[l,P,G,re],keypoints:N})}return n}function an(u,U){if(u.length===0)return[];let n=[...u].sort((a,y)=>y.score-a.score),W=[],M=new Set;for(let a=0;a<n.length;a++){if(M.has(a))continue;let y=[a];for(let N=a+1;N<n.length;N++)M.has(N)||Wn(n[a],n[N])>U&&(y.push(N),M.add(N));let L=0,A=0,B=0,l=0,P=0,G=[];for(let N=0;N<7;N++)G.push([0,0]);for(let N of y){let d=n[N],D=d.score;L+=D,A+=d.box[0]*D,B+=d.box[1]*D,l+=d.box[2]*D,P+=d.box[3]*D;for(let C=0;C<7;C++)G[C][0]+=d.keypoints[C][0]*D,G[C][1]+=d.keypoints[C][1]*D}let re=1/L;W.push({score:n[a].score,box:[A*re,B*re,l*re,P*re],keypoints:G.map(([N,d])=>[N*re,d*re])})}return W}function Wn(u,U){let n=u.box[0]-u.box[2]/2,W=u.box[1]-u.box[3]/2,M=u.box[0]+u.box[2]/2,a=u.box[1]+u.box[3]/2,y=U.box[0]-U.box[2]/2,L=U.box[1]-U.box[3]/2,A=U.box[0]+U.box[2]/2,B=U.box[1]+U.box[3]/2,l=Math.max(n,y),P=Math.max(W,L),G=Math.min(M,A),re=Math.min(a,B),N=Math.max(0,G-l),d=Math.max(0,re-P),D=N*d,C=(M-n)*(a-W),ye=(A-y)*(B-L),Le=C+ye-D;return Le>0?D/Le:0}function On(u){let[U,n,W,M]=u.box,a=u.keypoints[0],y=u.keypoints[2],L=y[0]-a[0],A=y[1]-a[1],B=Math.atan2(A,L),P=-Math.PI/2-B,G=Math.max(W,M),N=G*2.6,d=-.5*G,D=Math.cos(P),C=Math.sin(P),ye=d*C,Le=d*D;return{centerX:U+ye,centerY:n+Le,width:N,height:N,rotation:P}}function wn(u,U={}){let{scoreThreshold:n=.5,nmsThreshold:W=.3,maxHands:M=2}=U;async function a(B){let l=await u.run(B),P=nn(l,n);return an(P,W).slice(0,M).map(On)}async function y(B){let l=await u.run(B),P=nn(l,n);return an(P,W).slice(0,M)}async function L(B,l,P){let{output:G,lbPadX:re,lbPadY:N}=await u.runWithResize(B,l,P),d=nn(G,n);return{detections:an(d,W).slice(0,M),lbPadX:re,lbPadY:N}}async function A(B,l,P){let{output:G,lbPadX:re,lbPadY:N}=await u.runWithResize(B,l,P);return{scores:G.scores,regressors:G.regressors,lbPadX:re,lbPadY:N}}return{detect:a,detectRaw:y,detectRawWithResize:L,detectRawSSD:A,model:u}}var rn=["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip"];function zt(u){let U={};for(let n=0;n<rn.length;n++)U[rn[n]]=u[n];return U}function Kn(u){return u.replace(/\/\/[^\n]*/g,"").replace(/\s+/g," ").replace(/\s*([{}();,=+\-*/<>!&|@])\s*/g,"$1").trim()}var zn=Kn(`
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
`);function Pn(u){let U=u.createShaderModule({code:zn}),n=u.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,sampler:{}}]}),W=u.createComputePipeline({layout:u.createPipelineLayout({bindGroupLayouts:[n]}),compute:{module:U,entryPoint:"main"}}),M=u.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),a=u.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),y=u.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),L=new Float32Array(8);function A(B,l,P,G,re,N,d){u.queue.writeBuffer(a,0,new Uint32Array([re,N,d,0])),L.set(G),u.queue.writeBuffer(y,0,L);let D=u.createBindGroup({layout:n,entries:[{binding:0,resource:l.createView()},{binding:1,resource:{buffer:P}},{binding:2,resource:{buffer:a}},{binding:3,resource:{buffer:y}},{binding:4,resource:M}]}),C=B.beginComputePass();C.setPipeline(W),C.setBindGroup(0,D),C.dispatchWorkgroups(Math.ceil(d/16),Math.ceil(d/16),1),C.end()}return{crop:A}}var In="https://cdn.jsdelivr.net/npm/@svenflow/micro-handpose@latest/weights";async function Fn(u={}){let{weightsUrl:U,scoreThreshold:n=.5,palmScoreThreshold:W=.5,maxHands:M=3,forceF32:a=!1}=u;if(typeof navigator>"u"||!navigator.gpu)throw new Error("micro-handpose requires WebGPU. Check browser support at https://webgpureport.org");let y=(U??In).replace(/\/$/,"")+"/",[L,A,B,l]=await Promise.all([fetch(`${y}weights_f16_full.json`),fetch(`${y}weights_f16_full.bin`),fetch(`${y}palm_detection_weights.json`),fetch(`${y}palm_detection_weights.bin`)]);if(!L.ok)throw new Error(`Failed to fetch landmark weights: ${L.status}`);if(!A.ok)throw new Error(`Failed to fetch landmark weights: ${A.status}`);if(!B.ok)throw new Error(`Failed to fetch palm detection weights: ${B.status}`);if(!l.ok)throw new Error(`Failed to fetch palm detection weights: ${l.status}`);let[P,G,re,N]=await Promise.all([L.json(),A.arrayBuffer(),B.json(),l.arrayBuffer()]),d=sn(P,G),D=on(re,N),C=224,ye=await un(d,{forceF32:a});{let s=new OffscreenCanvas(C,C),w=s.getContext("2d");w.fillStyle="#886644",w.fillRect(0,0,C,C),w.fillStyle="#cc9966",w.fillRect(50,50,124,124);let h=await ye.runFromCanvas(s);h.landmarks.every(q=>q===0)&&h.handflag.every(q=>q===0)&&console.warn("[micro-handpose] FULL model produced all-zero output on self-test")}let Le=await _n(D),ke=wn(Le,{scoreThreshold:W,maxHands:M}),k=[];function Ke(s,w,h){let b=s[0],q=s[5],ee=s[9],V=s[13],Z=b.x*w,te=b.y*h,Q=(q.x+V.x)/2,Y=(q.y+V.y)/2;Q=(Q+ee.x)/2*w,Y=(Y+ee.y)/2*h;let $=Math.PI/2-Math.atan2(-(Y-te),Q-Z),T=$-2*Math.PI*Math.floor(($+Math.PI)/(2*Math.PI)),J=[0,1,2,3,5,6,9,10,13,14,17,18],O=Math.cos(T),j=Math.sin(T),fe=1/0,oe=-1/0,he=1/0,ge=-1/0;for(let Ge of J){let Be=s[Ge],Re=Be.x*w,We=Be.y*h,Ve=O*Re+j*We,at=-j*Re+O*We;fe=Math.min(fe,Ve),oe=Math.max(oe,Ve),he=Math.min(he,at),ge=Math.max(ge,at)}let Ue=(fe+oe)/2,be=(he+ge)/2,ue=oe-fe,le=ge-he,Ee=(O*Ue-j*be)/w,De=(j*Ue+O*be)/h;ue/=w,le/=h;let ce=-.1;Ee+=-h*le*ce*j/w,De+=le*ce*O;let Pe=Math.max(ue*w,le*h)*2;return{centerXpx:Ee*w,centerYpx:De*h,sizePx:Pe,rotation:T}}let Se=null,pe=null;function Dt(){return Se||(Se=new OffscreenCanvas(192,192)),Se}function Rt(){return pe||(pe=new OffscreenCanvas(C,C)),pe}let ve=ye.device,tt=null,je=null,Fe=null,wt=0,ze=0;function nt(){return tt||(tt=Pn(ve)),tt}function Te(){return je||(je=ve.createBuffer({size:3*C*C*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),je}function mt(s,w){return(!Fe||wt!==s||ze!==w)&&(Fe&&Fe.destroy(),Fe=ve.createTexture({size:[s,w],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),wt=s,ze=w),Fe}let we=0,He=0;function Lt(s,w,h){let b=Dt();b.width=192,b.height=192;let q=b.getContext("2d");q.clearRect(0,0,192,192);let ee=Math.min(192/w,192/h),V=Math.round(w*ee),Z=Math.round(h*ee),te=(192-V)/2,Q=(192-Z)/2;if(we=te/192,He=Q/192,s instanceof ImageData){let Y=new OffscreenCanvas(s.width,s.height);Y.getContext("2d").putImageData(s,0,0),q.drawImage(Y,te,Q,V,Z)}else q.drawImage(s,0,0,w,h,te,Q,V,Z);return b}function Pt(s){let w=1/(1-2*we),h=1/(1-2*He);return{score:s.score,box:[(s.box[0]-we)*w,(s.box[1]-He)*h,s.box[2]*w,s.box[3]*h],keypoints:s.keypoints.map(([b,q])=>[(b-we)*w,(q-He)*h])}}function ht(s,w,h){let b=s.keypoints[0],q=s.keypoints[2],ee=(q[0]-b[0])*w,V=(q[1]-b[1])*h,Z=Math.atan2(-V,ee),Q=Math.PI/2-Z,Y=Q-2*Math.PI*Math.floor((Q+Math.PI)/(2*Math.PI)),[$,T,J,O]=s.box,j=Math.cos(Y),fe=Math.sin(Y),oe=O*h,he=$+.5*oe*fe/w,ge=T+-.5*O*j,ue=Math.max(J*w,O*h)*2.6;return{centerXpx:he*w,centerYpx:ge*h,sizePx:ue,rotation:Y}}function Ht(s,w){let h=Rt();h.width=C,h.height=C;let b=h.getContext("2d");b.clearRect(0,0,C,C);let q=C/w.sizePx,ee=Math.cos(w.rotation),V=Math.sin(w.rotation),Z=ee*q,te=-V*q,Q=V*q,Y=ee*q,$=C/2,T=-w.centerXpx*Z-w.centerYpx*Q+$,J=-w.centerXpx*te-w.centerYpx*Y+$;if(b.setTransform(Z,te,Q,Y,T,J),s instanceof ImageData){let O=new OffscreenCanvas(s.width,s.height);O.getContext("2d").putImageData(s,0,0),b.drawImage(O,0,0)}else b.drawImage(s,0,0);return b.setTransform(1,0,0,1,0,0),h}function Ie(s){return s instanceof HTMLCanvasElement||s instanceof OffscreenCanvas?[s.width,s.height]:typeof ImageBitmap<"u"&&s instanceof ImageBitmap?[s.width,s.height]:s instanceof ImageData?[s.width,s.height]:s instanceof HTMLVideoElement?[s.videoWidth,s.videoHeight]:s instanceof HTMLImageElement?[s.naturalWidth,s.naturalHeight]:[C,C]}async function Bt(s,w,h,b,q,ee){let V=Math.cos(s.rotation),Z=Math.sin(s.rotation),te=s.sizePx/C,Q=C/2,Y=V*te/h,$=-Z*te/h,T=s.centerXpx/h-Q*(Y+$),J=Z*te/b,O=V*te/b,j=s.centerYpx/b-Q*(J+O),fe=ve.createCommandEncoder();q.crop(fe,w,ee,[Y,$,T,J,O,j],h,b,C),ve.queue.submit([fe.finish()]);let oe=await ye.runFromGPUBuffer(ee),he=oe.handflag[0];if(he<n)return null;let ge=oe.handedness[0]>.5,Ue=[];for(let be=0;be<21;be++){let ue=oe.landmarks[be*3],le=oe.landmarks[be*3+1],Ee=oe.landmarks[be*3+2],De=(ue-.5)*s.sizePx,ce=(le-.5)*s.sizePx,Ne=V*De-Z*ce+s.centerXpx,qe=Z*De+V*ce+s.centerYpx;Ue.push({x:Ne/h,y:qe/b,z:Ee})}return{landmarks:Ue,score:he,handedness:ge?"right":"left"}}async function At(s){let w=s,h,b;if(s instanceof HTMLVideoElement||s instanceof HTMLImageElement){let T=await createImageBitmap(s,{colorSpaceConversion:"none"});w=T,h=T.width,b=T.height}else[h,b]=Ie(s);let q=nt(),ee=Te(),V=mt(h,b),Z;if(w instanceof ImageData?Z=await createImageBitmap(w,{colorSpaceConversion:"none"}):Z=w,ve.queue.copyExternalImageToTexture({source:Z},{texture:V},[h,b]),k.length>0){let T=[];for(let J of k){let O=Ke(J.landmarks,h,b),j=await Bt(O,V,h,b,q,ee);j&&T.push({score:j.score,handedness:j.handedness,landmarks:j.landmarks,keypoints:zt(j.landmarks)})}if(T.length>0)return k=T.map(J=>({landmarks:J.landmarks,handedness:J.handedness})),T;k=[]}let{detections:te,lbPadX:Q,lbPadY:Y}=await ke.detectRawWithResize(w,h,b);if(we=Q,He=Y,te.length===0)return k=[],[];let $=[];for(let T of te){let J=Pt(T),O=ht(J,h,b),j=await Bt(O,V,h,b,q,ee);j&&$.push({score:j.score,handedness:j.handedness,landmarks:j.landmarks,keypoints:zt(j.landmarks)})}return k=$.map(T=>({landmarks:T.landmarks,handedness:T.handedness})),$}async function It(s){let w=s,h,b;if(s instanceof HTMLVideoElement||s instanceof HTMLImageElement){let T=await createImageBitmap(s,{colorSpaceConversion:"none"});w=T,h=T.width,b=T.height}else[h,b]=Ie(s);let{detections:q,lbPadX:ee,lbPadY:V}=await ke.detectRawWithResize(w,h,b);if(we=ee,He=V,q.length===0)return[];let Z=[],te=nt(),Q=Te(),Y=mt(h,b),$;w instanceof ImageData?$=await createImageBitmap(w,{colorSpaceConversion:"none"}):$=w,ve.queue.copyExternalImageToTexture({source:$},{texture:Y},[h,b]);for(let T of q){let J=Pt(T),O=ht(J,h,b),j=Math.cos(O.rotation),fe=Math.sin(O.rotation),oe=O.sizePx/C,he=C/2,ge=j*oe/h,Ue=-fe*oe/h,be=O.centerXpx/h-he*(ge+Ue),ue=fe*oe/b,le=j*oe/b,Ee=O.centerYpx/b-he*(ue+le),De=ve.createCommandEncoder();te.crop(De,Y,Q,[ge,Ue,be,ue,le,Ee],h,b,C),ve.queue.submit([De.finish()]);let ce=await ye.runFromGPUBuffer(Q),Ne=ce.handflag[0];if(Ne<n)continue;let qe=ce.handedness[0]>.5,Pe=[],Ge=[];for(let Be=0;Be<21;Be++){let Re=ce.landmarks[Be*3],We=ce.landmarks[Be*3+1],Ve=ce.landmarks[Be*3+2];Pe.push({x:Re,y:We,z:Ve});let at=(Re-.5)*O.sizePx,Qe=(We-.5)*O.sizePx,rt=j*at-fe*Qe+O.centerXpx,$e=fe*at+j*Qe+O.centerYpx;Ge.push({x:rt/h,y:$e/b,z:Ve})}Z.push({score:Ne,handedness:qe?"right":"left",landmarks:Ge,keypoints:zt(Ge),cropLandmarks:Pe,roi:{...O},palmDetection:{score:J.score,box:[...J.box],keypoints:J.keypoints.map(([Be,Re])=>[Be,Re])}})}return Z}async function Wt(s,w){let[h,b]=Ie(s);if(w.length===0)return[];let q=[],ee=nt(),V=Te(),Z=mt(h,b),te;s instanceof ImageData?te=await createImageBitmap(s,{colorSpaceConversion:"none"}):s instanceof HTMLImageElement?te=await createImageBitmap(s,{colorSpaceConversion:"none"}):te=s,ve.queue.copyExternalImageToTexture({source:te},{texture:Z},[h,b]);for(let Q of w){let Y=ht(Q,h,b),$=Math.cos(Y.rotation),T=Math.sin(Y.rotation),J=Y.sizePx/C,O=C/2,j=$*J/h,fe=-T*J/h,oe=Y.centerXpx/h-O*(j+fe),he=T*J/b,ge=$*J/b,Ue=Y.centerYpx/b-O*(he+ge),be=ve.createCommandEncoder();ee.crop(be,Z,V,[j,fe,oe,he,ge,Ue],h,b,C),ve.queue.submit([be.finish()]);let ue=await ye.runFromGPUBuffer(V),le=ue.handflag[0];if(le<n)continue;let Ee=ue.handedness[0]>.5,De=[];for(let ce=0;ce<21;ce++){let Ne=ue.landmarks[ce*3],qe=ue.landmarks[ce*3+1],Pe=ue.landmarks[ce*3+2],Ge=(Ne-.5)*Y.sizePx,Be=(qe-.5)*Y.sizePx,Re=$*Ge-T*Be+Y.centerXpx,We=T*Ge+$*Be+Y.centerYpx;De.push({x:Re/h,y:We/b,z:Pe})}q.push({score:le,handedness:Ee?"right":"left",landmarks:De,keypoints:zt(De)})}return q}function kt(){Fe&&Fe.destroy(),je&&je.destroy(),Fe=null,je=null,tt=null,ye.device.destroy(),Le.device.destroy(),Se=null,pe=null}let Ft={palmDetector:ke,palmModel:Le,landmarkModel:ye,removeLetterbox:Pt,detectionToPixelROI:ht,cropHandRegion:Ht};function Xe(){k=[]}return{detect:At,detectWithDebug:It,detectFromDetections:Wt,dispose:kt,reset:Xe,_debug:Ft}}export{rn as LANDMARK_NAMES,Fn as createHandpose};
