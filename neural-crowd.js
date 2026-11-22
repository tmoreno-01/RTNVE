import {gpu,loadPlayable} from "./vendor/ganlive/player.mjs";
import {screen} from "./vendor/ganlive/runner.mjs";
const $=s=>document.getElementById(s);
const MAX=15, BASE="https://huggingface.co/gustavecortal/ganlive-ffhq/resolve/main/engine/";
const S={device:null,entry:null,context:null,atlas:null,pipeline:null,bind:null,buffer:null,slots:[],
 running:false,busy:false,pause:false,visible:true,cols:4,rows:3,count:12,next:0,last:0,
 time:0,lastGen:0,genMs:80,hz:12,frames:0,fpsStart:performance.now(),renderFps:0};
const C={count:$("count"),speed:$("speed"),chaos:$("chaos"),morph:$("morph"),seed:$("seed"),
 pause:$("pause"),save:$("save"),status:$("status"),fps:$("fps"),details:$("details"),stage:$("stage"),art:$("art")};
const values=()=>({speed:+C.speed.value,chaos:+C.chaos.value,morph:+C.morph.value});
const gaussian=()=>Math.sqrt(-2*Math.log(Math.max(Math.random(),1e-8)))*Math.cos(Math.random()*Math.PI*2);
function makeLatent(n){let v=new Float32Array(n);for(let i=0;i<n;i++)v[i]=gaussian();return v}
function freshSlots(){return Array.from({length:MAX},(_,id)=>({
 id,z:makeLatent(S.entry.nz),ready:0,phase:1,at:0,duration:1,seed:Math.random()*1000,steps:0
}))}
function setMessage(text){C.status.textContent=text}
function selectCount(){let v=+C.count.value;S.cols=v===6||v===9?3:v===15?5:4;S.rows=v/S.cols;S.count=v;S.next=0}
function resize(){
 if(!S.context)return;
 const box=C.stage.getBoundingClientRect();
 const dpr=Math.min(1.6,devicePixelRatio||1);
 const w=Math.max(2,Math.min(1900,Math.round(box.width*dpr)));
 const h=Math.max(2,Math.min(1200,Math.round(box.height*dpr)));
 if(C.art.width!==w||C.art.height!==h){C.art.width=w;C.art.height=h}
}
async function renderer(){
 const d=S.device;
 S.context=C.art.getContext("webgpu");
 S.context.configure({device:d,format:navigator.gpu.getPreferredCanvasFormat(),alphaMode:"opaque"});
 S.atlas=d.createTexture({size:[S.entry.width,S.entry.height,MAX*2],format:"rgba8unorm",
   usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT});
 S.buffer=d.createBuffer({size:16*(2+MAX),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const module=d.createShaderModule({code:"\nstruct Params {\n  layoutInfo: vec4f,\n  fx: vec4f,\n  tiles: array<vec4f, 15>\n};\n@group(0) @binding(0) var faces: texture_2d_array<f32>;\n@group(0) @binding(1) var faceSampler: sampler;\n@group(0) @binding(2) var<uniform> cfg: Params;\nstruct Output { @builtin(position) pos: vec4f, @location(0) uv: vec2f };\n@vertex fn vertex(@builtin(vertex_index) i:u32) -> Output {\n  var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));\n  var o:Output;\n  o.pos=vec4f(p[i],0,1);\n  o.uv=vec2f((p[i].x+1)*0.5, 1-(p[i].y+1)*0.5);\n  return o;\n}\nfn hash2(p:vec2f)->f32{return fract(sin(dot(p,vec2f(127.1,311.7)))*43758.5453);}\nfn noise(p:vec2f)->f32{\n  let a=floor(p);let f=fract(p);let v=f*f*(3-2*f);\n  return mix(mix(hash2(a),hash2(a+vec2f(1,0)),v.x),\n             mix(hash2(a+vec2f(0,1)),hash2(a+vec2f(1,1)),v.x),v.y);\n}\n@fragment fn fragment(f:Output)->@location(0) vec4f {\n  let cols=u32(cfg.layoutInfo.x);let rows=u32(cfg.layoutInfo.y);\n  let global=clamp(f.uv,vec2f(0),vec2f(0.99999));\n  let col=min(cols-1,u32(floor(global.x*f32(cols))));\n  let row=min(rows-1,u32(floor(global.y*f32(rows))));\n  let id=min(14u,row*cols+col);\n  let info=cfg.tiles[id]; // morph, isGenerated, unique seed, unused\n  if(info.y<0.5){return vec4f(0.978,0.978,0.973,1);}\n  let t=cfg.fx.x;let chaos=cfg.fx.y;let organic=cfg.fx.z;\n  let p=fract(global*vec2f(f32(cols),f32(rows)));\n  let n=noise(p*3.8+vec2f(cos(t*.22+info.z),sin(t*.16+info.z))*1.3);\n  let n2=noise(p*11.0+vec2f(t*.10,-t*.13)+info.z);\n  let displacement=(.002+chaos*.027);\n  let flow=vec2f(sin(p.y*13.0+n*7.0+t*.65+info.z),\n                 cos(p.x*12.0+n2*8.0-t*.47+info.z))*displacement;\n  let phase=clamp(info.x,0.,1.);\n  let aUV=clamp(p+flow*(1-phase),vec2f(.002),vec2f(.998));\n  let bUV=clamp(p-flow*phase,vec2f(.002),vec2f(.998));\n  let a=textureSampleLevel(faces,faceSampler,aUV,i32(id*2u),0.0).rgb;\n  let b=textureSampleLevel(faces,faceSampler,bUV,i32(id*2u+1u),0.0).rgb;\n  let region=(n*.7+n2*.3-.5)*organic*.75;\n  let factor=smoothstep(0.,1.,clamp(phase+region,0.,1.));\n  return vec4f(mix(a,b,factor),1);\n}\n"});
 const compilation=await module.getCompilationInfo();
 const shaderErrors=compilation.messages.filter(m=>m.type==="error");
 if(shaderErrors.length)throw new Error("Crowd shader: "+shaderErrors.map(m=>m.message).join("; ").slice(0,500));
 S.pipeline=await d.createRenderPipelineAsync({layout:"auto",vertex:{module,entryPoint:"vertex"},
   fragment:{module,entryPoint:"fragment",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},
   primitive:{topology:"triangle-list"}});
 S.bind=d.createBindGroup({layout:S.pipeline.getBindGroupLayout(0),entries:[
   {binding:0,resource:S.atlas.createView({dimension:"2d-array"})},
   {binding:1,resource:d.createSampler({magFilter:"linear",minFilter:"linear"})},
   {binding:2,resource:{buffer:S.buffer}}
 ]});
 resize();
}
function evolve(slot){
 const chaos=values().chaos;
 const leap=Math.random()<(.025+chaos*.13);
 const angle=leap?(.65+chaos*.5):(.10+chaos*.35);
 const a=Math.cos(angle),b=Math.sin(angle);
 for(let i=0;i<slot.z.length;i++)slot.z[i]=slot.z[i]*a+gaussian()*b;
 slot.steps++;
}
async function generate(now){
 if(S.busy||!S.running||S.pause||document.hidden||!S.visible)return;
 S.busy=true;
 const beginning=performance.now();
 const slot=S.slots[S.next%S.count];S.next=(S.next+1)%S.count;
 try{
   const encoder=S.device.createCommandEncoder();
   if(slot.ready){
      encoder.copyTextureToTexture({texture:S.atlas,origin:{z:slot.id*2+1}},
       {texture:S.atlas,origin:{z:slot.id*2}},
       {width:S.entry.width,height:S.entry.height,depthOrArrayLayers:1});
      evolve(slot);
   }
   S.entry.model.setLatent(slot.z);
   S.entry.model.encode(encoder);
   const layers=slot.ready?[slot.id*2+1]:[slot.id*2,slot.id*2+1];
   for(const layer of layers){
      // The generator's output buffer is an internal neural format.
      // Reuse ganlive's own proven screen shader to convert it to RGB,
      // rendering directly into the correct array layer.
      S.facePresenter.draw(encoder,{
        width:S.entry.width,
        height:S.entry.height,
        createView:()=>S.atlas.createView({
          dimension:"2d",
          baseArrayLayer:layer,
          arrayLayerCount:1
        })
      });
   }
   S.device.queue.submit([encoder.finish()]);
   if(slot.ready){
     slot.phase=0;slot.at=now;
     slot.duration=Math.min(4,Math.max(.35,(S.count/Math.max(2,S.hz))*(.80+Math.random()*.4)));
   } else {slot.ready=1;slot.phase=1}
   await S.device.queue.onSubmittedWorkDone();
   const ms=performance.now()-beginning;
   S.genMs=S.genMs*.84+ms*.16;
   S.hz=Math.max(2,Math.min(18,800/Math.max(20,S.genMs)));
 }catch(e){console.error(e);S.running=false;setMessage("Neural generation error: "+(e?.message||String(e)))}
 finally{S.busy=false}
}
function draw(now){
 requestAnimationFrame(draw);
 const dt=S.last?Math.min(.07,(now-S.last)/1000):0;S.last=now;
 if(!S.running||S.pause||document.hidden||!S.visible)return;
 S.time+=dt;
 const {speed,chaos,morph}=values();
 const packed=new Float32Array(4*(2+MAX));
 packed.set([S.cols,S.rows,C.art.width,C.art.height],0);
 packed.set([S.time,chaos,morph,0],4);
 for(let i=0;i<MAX;i++){
   let s=S.slots[i];
   if(s.ready&&s.phase<1)s.phase=Math.min(1,(now-s.at)/1000/s.duration);
   packed.set([s.phase,s.ready,s.seed,0],8+i*4);
 }
 S.device.queue.writeBuffer(S.buffer,0,packed);
 const encoder=S.device.createCommandEncoder();
 const pass=encoder.beginRenderPass({colorAttachments:[{view:S.context.getCurrentTexture().createView(),
  loadOp:"clear",storeOp:"store",clearValue:{r:1,g:1,b:1,a:1}}]});
 pass.setPipeline(S.pipeline);pass.setBindGroup(0,S.bind);pass.draw(3);pass.end();
 S.device.queue.submit([encoder.finish()]);
 const rate=Math.max(60,(1000/S.hz)/(0.3+speed*1.3));
 if(now-S.lastGen>=rate){S.lastGen=now;void generate(now)}
 S.frames++;
 if(now-S.fpsStart>1100){
   S.renderFps=(S.frames*1000/(now-S.fpsStart));
   S.frames=0;S.fpsStart=now;
   C.fps.textContent=Math.round(S.renderFps)+" fps";
   C.details.textContent=S.count+" faces / "+Math.max(1,Math.round(S.hz*(.3+speed*1.3)))+" neural updates/sec shared";
 }
}
async function start(){
 try{
  if(!isSecureContext||!navigator.gpu)throw new Error("WebGPU needs HTTPS and a supported browser/GPU.");
  setMessage("Loading neural faces…");
  S.device=await gpu();
  S.device.lost.then(info=>{S.running=false;setMessage("GPU device lost: "+(info?.message||"reload"))});
  S.entry=await loadPlayable(S.device,BASE,"FFHQ",{
   onStage:step=>setMessage({download:"Downloading neural model weights…",compile:"Compiling GPU neural generator…",draw:"Rendering first face…"}[step]||step),
   onProgress:bytes=>setMessage("Loading neural model: "+Math.round(bytes/1048576)+" MB")
  });
  if(S.device.limits.maxTextureArrayLayers<MAX*2)throw new Error("GPU cannot store the required face layers.");
  // The atlas stores rgba8unorm, which may differ from the browser canvas format.
  S.facePresenter=screen(S.device,S.entry.model,"rgba8unorm");
  await renderer();
  S.slots=freshSlots();selectCount();
  S.running=true;C.seed.disabled=false;C.pause.disabled=false;C.save.disabled=false;
  setMessage("Neural crowd running locally");
  requestAnimationFrame(draw);
 }catch(e){console.error(e);setMessage("Cannot start: "+(e?.message||String(e)))}
}
C.count.onchange=()=>{selectCount();S.slots=freshSlots();setMessage("Creating "+S.count+" new neural faces…")};
for(let id of ["speed","chaos","morph"])C[id].addEventListener("input",()=>{$(id+"Value").textContent=(+C[id].value).toFixed(2)});
C.seed.onclick=()=>{if(!S.entry)return;S.slots=freshSlots();S.next=0;S.time=0;setMessage("New random identities generating…")};
C.pause.onclick=()=>{S.pause=!S.pause;C.pause.textContent=S.pause?"Resume":"Pause";S.last=0};
C.save.onclick=()=>C.art.toBlob(blob=>{
 if(!blob){setMessage("Could not capture a frame");return}
 const url=URL.createObjectURL(blob),a=document.createElement("a");
 a.href=url;a.download="lab-crowd-"+Date.now()+".png";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000)
},"image/png");
window.addEventListener("resize",resize);
new ResizeObserver(resize).observe(C.stage);
new IntersectionObserver(([entry])=>{S.visible=entry?.isIntersecting??true;S.last=0},{threshold:.01}).observe(C.stage);
document.addEventListener("visibilitychange",()=>{S.last=0});
start();
