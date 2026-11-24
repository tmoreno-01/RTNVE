"""Looping independent spatial conditioning for RTNVE (Diffusers AnimateDiff).
Pure schedule functions are importable without torch.
"""
import math
import random

REGIONAL_LOOP_MODE = "R-P"

GLOBAL = ("high angle fixed CCTV view, crowded New York intersection, pedestrians, taxis, "
          "illuminated LED screens, fractured glass, overlapping architecture, "
          "documentary photograph, crisp fine detail, deep focus")
NEGATIVE = ("nsfw, blurry, out of focus, low resolution, low detail, oversmoothed, "
            "fog, flat lighting, washed out, empty background, indistinct shapes, "
            "cartoon, CGI, watermark, logo, vignette, shallow depth of field")
REGIONS = (
    ("faces", (
        "high angle CCTV, hundreds of overlapping faces and bodies, diverse commuters, heads emerging from coats, crisp documentary detail",
        "high angle CCTV, people climbing out of suitcases, intertwined strangers, faces inside faces, detailed clothing",
        "high angle CCTV, human faces forming apartment windows, crowds folding into each other, torn paper and glass",
    ), (0, 0.31, 0.67), (0.35,0.80)),
    ("traffic", (
        "high angle CCTV, dense traffic of taxis and buses, people pushing tiny cars, sharp headlights and windscreens",
        "high angle CCTV, vehicles carrying smaller cars, people merging with windshields, shattered concrete, detailed street",
        "high angle CCTV, cars unfolding into miniature rooms and crowds, overlapping roads, broken glass and reflections",
    ), (0.12, 0.48, 0.78), (0.30,0.75)),
    ("technology", (
        "high angle CCTV, dense LED panels and CRT televisions, tiny faces inside monitors, illuminated circuitry and reflective glass",
        "high angle CCTV, computer screens opening into crowded rooms, illuminated wires and electronic displays, sharp detail",
        "high angle CCTV, LED signs and surveillance monitors growing through buildings and human figures, tangled cables, glass",
    ), (0.23, 0.55, 0.90), (0.25,0.70)),
)

def prompt_at(t, knots):
    """Periodic eased interpolation, with a small hold near each anchor."""
    t = t % 1.0
    for i in range(len(knots)):
        lo = knots[i]
        hi = knots[i+1] if i+1<len(knots) else knots[0]+1
        x = t if t>=knots[0] else t+1
        if lo<=x<hi:
            u=(x-lo)/(hi-lo)
            u=max(0.,min(1.,(u-.18)/.82))
            return i,(i+1)%len(knots),u*u*(3-2*u)
    raise AssertionError("Periodic prompt schedule failed")

def build_plan(frames=60, seed=9191):
    if frames<16:
        raise ValueError("At least 16 frames required")
    r=random.Random(seed)
    regions=[]
    for i,(name,prompts,knots,bounds) in enumerate(REGIONS):
        regions.append(dict(name=name,prompts=list(prompts),knots=list(knots),
                            strength_range=list(bounds),phase=i*2*math.pi/3+r.uniform(-.22,.22),
                            distortion_phase=r.uniform(0,2*math.pi)))
    plan=dict(version=2,frames=frames,seed=seed,loop=REGIONAL_LOOP_MODE,global_prompt=GLOBAL,negative=NEGATIVE,
              global_weight=.38,regions=regions,mask_method="periodic warped soft ellipses; normalized overlapping attention weights")
    plan["schedule"]=[frame_params(plan,f) for f in range(frames)]
    return plan

def frame_params(plan, f):
    t=(f%plan["frames"])/plan["frames"]
    a=2*math.pi*t
    records=[]
    for reg in plan["regions"]:
        p=reg["phase"];q=reg["distortion_phase"]
        lo,hi=reg["strength_range"]
        left,right,alpha=prompt_at(t,reg["knots"])
        records.append(dict(name=reg["name"],left=left,right=right,alpha=alpha,
                            cx=.50+.29*math.cos(a+p)+.055*math.sin(2*a+q),
                            cy=.50+.25*math.sin(a+p)+.045*math.cos(2*a+q),
                            rx=.30+.095*math.sin(a+p+.6),
                            ry=.34+.10*math.cos(a+p-1.),
                            angle=.53*math.sin(a+p),
                            warp=.15+.055*math.sin(2*a+q),
                            wave=a+q,
                            strength=lo+(hi-lo)*(.5+.5*math.cos(a+p+.4))))
    return dict(frame=f%plan["frames"],regions=records)

def make_masks(plan, height, width, device="cpu"):
    """[frames, 4, H, W] float32. Region strength is unnormalized influence.
    Persistent global field, smooth masks, partition of unity at each pixel.
    """
    import torch
    y,x=torch.meshgrid(torch.linspace(0,1,height,device=device),
                       torch.linspace(0,1,width,device=device),indexing="ij")
    frames=[]
    for rec in plan["schedule"]:
        fields=[torch.full_like(x,plan["global_weight"])]
        for m in rec["regions"]:
            dx=x-m["cx"];dy=y-m["cy"]
            c=math.cos(m["angle"]);s=math.sin(m["angle"])
            xx=(c*dx-s*dy)/m["rx"];yy=(s*dx+c*dy)/m["ry"]
            theta=torch.atan2(yy,xx)
            edge=1+m["warp"]*(torch.sin(3*theta+m["wave"])+.5*torch.cos(5*theta-2*m["wave"]))
            dist=torch.sqrt(xx.square()+yy.square())/edge
            v=((1.45-dist)/.75).clamp(0,1)
            fields.append(m["strength"]*v.square()*(3-2*v))
        z=torch.stack(fields)
        frames.append(z/z.sum(dim=0,keepdim=True))
    return torch.stack(frames)

def encode_regions(pipe,plan):
    """Positive text states only; the ordinary global CFG pair stays intact."""
    import torch
    texts=[plan["global_prompt"],plan["negative"]]
    texts.extend(p for r in plan["regions"] for p in r["prompts"])
    for p in texts:
        n=len(pipe.tokenizer(p,truncation=False,add_special_tokens=True)["input_ids"])
        if n>pipe.tokenizer.model_max_length:
            raise ValueError(f"CLIP token overflow ({n}): {p}")
    device=next(pipe.unet.parameters()).device
    with torch.inference_mode():
        bank=[]
        for i,reg in enumerate(plan["regions"]):
            positive,_=pipe.encode_prompt(prompt=reg["prompts"],device=device,
                        num_images_per_prompt=1,do_classifier_free_guidance=False)
            bank.append(torch.stack([
                positive[row["regions"][i]["left"]].lerp(
                    positive[row["regions"][i]["right"]],row["regions"][i]["alpha"])
                for row in plan["schedule"]]))
    return bank

class RegionalState:
    def __init__(self,masks,embeddings):
        self.masks=masks
        self.embeddings=embeddings
        self.indices=None
        self.cache={}
        self.calls=0

    def select(self,indices):
        import torch
        self.indices=indices
        ids=torch.tensor(indices,device=self.masks.device,dtype=torch.long)
        self.local_masks=self.masks.index_select(0,ids)
        self.local_embeddings=[e.index_select(0,ids.to(e.device)) for e in self.embeddings]
        self.cache.clear()

    def weights(self,seq,device,dtype):
        import torch.nn.functional as F
        key=(seq,device,dtype)
        if key not in self.cache:
            h,w=self.local_masks.shape[-2:]
            # UNet stride-2 convolutions use ceil division (90 -> 45 -> 23).
            for _ in range(8):
                if h*w==seq:
                    v=F.interpolate(self.local_masks,size=(h,w),mode="bilinear",align_corners=False)
                    self.cache[key]=v.flatten(2).transpose(1,2).to(device=device,dtype=dtype)
                    break
                h=(h+1)//2;w=(w+1)//2
            else:
                raise RuntimeError(f"Unknown spatial attention grid {seq}")
        return self.cache[key]

class SpatialProcessor:
    """Mix the outputs of the actual spatial cross-attention processor.
    Negative CFG is left unchanged; global and regional positives are blended.
    No patching of temporal self-attention or separate latent clips.
    """
    def __init__(self,original,state):
        self.original=original
        self.state=state

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,
                 attention_mask=None,temb=None,*args,**kwargs):
        if encoder_hidden_states is None or self.state.indices is None:
            return self.original(attn,hidden_states,encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask,temb=temb,*args,**kwargs)
        n=len(self.state.indices)
        if hidden_states.ndim!=3 or hidden_states.shape[0]!=2*n or encoder_hidden_states.shape[0]!=2*n:
            raise RuntimeError(f"Regional CFG shape mismatch: {tuple(hidden_states.shape)} vs {2*n} frames")
        if attention_mask is not None or temb is not None:
            raise RuntimeError("Unexpected masked/temb spatial attention; inspect Diffusers API")
        out=self.original(attn,hidden_states,encoder_hidden_states=encoder_hidden_states,
                          attention_mask=None,temb=None,*args,**kwargs)
        weights=self.state.weights(hidden_states.shape[1],out.device,out.dtype)
        pos=out[n:]
        pos.mul_(weights[:,:,0:1])
        for i,embedding in enumerate(self.state.local_embeddings):
            extra=self.original(attn,hidden_states[n:],encoder_hidden_states=embedding,
                                attention_mask=None,temb=None,*args,**kwargs)
            pos.addcmul_(extra,weights[:,:,i+1:i+2])
            del extra
        self.state.calls+=1
        return out

def install_processors(unet,state):
    original=dict(unet.attn_processors)
    new=dict(original)
    names=[k for k in original if k.endswith(".attn2.processor") and ".motion_modules." not in k]
    if not names:
        raise RuntimeError("No spatial cross-attention layers found")
    for name in names:
        new[name]=SpatialProcessor(original[name],state)
    unet.set_attn_processor(new)
    return original,names

class CircularContext:
    """One global video with wrapped contexts and tapered prediction consensus."""
    def __init__(self,unet,state,frames,window=16,overlap=8):
        self.original=unet.forward
        self.state=state
        self.frames=frames
        self.window=window
        self.overlap=overlap
        self.loop=REGIONAL_LOOP_MODE
        self.step=0
        self.total_unet_calls=0

    def forward(self,sample,timestep,*args,**kwargs):
        import torch
        from animatediff_long_context import uniform_contexts, frame_condition_indices
        if args or sample.ndim!=5 or sample.shape[0]!=2 or sample.shape[2]!=self.frames:
            raise RuntimeError(f"Unexpected circular latent shape {tuple(sample.shape)}")
        emb=kwargs["encoder_hidden_states"]
        if emb.ndim!=3 or emb.shape[0]!=2*self.frames:
            raise RuntimeError(f"Unexpected global conditioning shape {tuple(emb.shape)}")
        windows=uniform_contexts(self.step,self.frames,self.window,1,self.overlap,self.loop)
        accum=torch.zeros_like(sample,dtype=torch.float32)
        denominator=torch.zeros(self.frames,device=sample.device,dtype=torch.float32)
        taper=.15+.85*torch.sin(math.pi*(torch.arange(self.window,device=sample.device)+.5)/self.window).square()
        for indices in windows:
            idx=torch.tensor(indices,device=sample.device,dtype=torch.long)
            self.state.select(indices)
            local=dict(kwargs)
            local["encoder_hidden_states"]=emb.index_select(0,torch.tensor(
                frame_condition_indices(2,self.frames,indices),device=emb.device))
            result=self.original(sample.index_select(2,idx),timestep,**local)
            accum.index_add_(2,idx,result.sample.float()*taper.view(1,1,-1,1,1))
            denominator.index_add_(0,idx,taper)
            self.total_unet_calls+=1
        if not bool((denominator>0).all()):
            raise RuntimeError("Circular context uncovered frames")
        self.step+=1
        self.state.indices=None
        self.state.cache.clear()
        return result.__class__(sample=(accum/denominator.view(1,1,-1,1,1)).to(sample.dtype))
