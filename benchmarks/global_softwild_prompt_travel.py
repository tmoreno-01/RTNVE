"""Whole-frame Softwild prompt travel, no spatial masks. CPU-only planner."""
import random

BASE_PROMPTS = (
    "high angle CCTV, huge crowd walking through city traffic, taxis, buses, strange faces, overlapping bodies, reflections, broken glass, giant screens, dreamlike realism, soft motion blur",
    "high angle view, faces and bodies squeezed together in a station, commuters, duplicated heads, blurred coats and bags, faces inside faces, glass reflections, surreal photography",
    "high angle CCTV, traffic and crowds mixed together, cars unfolding into people, windscreens showing human faces, buses opening into crowded interiors, LED screens, soft motion blur",
    "high angle view, crowded public space, people, faces, cars, buses, electronic glow, torn paper, shattering reflections, bodies dissolving into clothing, photographic realism",
    "high angle CCTV, dense crowd with hundreds of faces, traffic, duplicate people, giant heads appearing across windows and buses, smeared identities, broken reflections, soft blur",
)
MODIFIERS = (
    "shattering glass", "ripping paper", "tearing apart", "broken reflections",
    "soft motion blur", "smeared faces", "duplicated bodies", "crowded visual noise",
    "identity breaking apart", "electronic glow", "faces inside screens",
    "cars turning into crowds", "crowds turning into cars", "giant faces",
    "tiny vehicles", "soft blur and haze",
)
NEGATIVE = "nsfw, lowres, text, watermark, logo, frame, border, cartoon, CGI, empty background, simple composition"

def ease(x):
    x=max(0.,min(1.,x))
    return x*x*(3.-2.*x)

def make_plan(frames=150, seed=9191):
    if frames!=150:
        raise ValueError("This experimental preset is specifically 150 frames")
    rng=random.Random(seed)
    # Five fully held states cannot fit the agreed minimum dwell and morph
    # lengths in 150 frames. Pick four of the five, always start with crowds.
    bases=[BASE_PROMPTS[0]]+rng.sample(list(BASE_PROMPTS[1:]),3)
    dwells=[20,20,20,21]
    transitions=[22,24,23]
    assert sum(dwells)+sum(transitions)==frames
    base_schedule=[]
    segments=[]
    start=0
    for i,dwell in enumerate(dwells):
        fade=transitions[i] if i<len(transitions) else 0
        end=start+dwell+fade
        segments.append(dict(start=start,end_exclusive=end,prompt_id=i,
                             dwell_frames=dwell,transition_frames=fade))
        for f in range(start,end):
            alpha=ease((f-start-dwell+1)/fade) if fade and f>=start+dwell else 0.
            base_schedule.append((i,min(i+1,len(bases)-1),alpha))
        start=end
    # Independent modifier timing: two phrases per anchor; a crossfade can
    # activate four phrases, with no discontinuous text swaps.
    mod_words=rng.sample(list(MODIFIERS),12)
    bundles=[mod_words[i:i+2] for i in range(0,12,2)]
    knots=[0,28,58,86,117,149]
    prompts=[base+", "+", ".join(bundle) for base in bases for bundle in bundles]
    schedule=[]
    for f,(b0,b1,ba) in enumerate(base_schedule):
        m0=next((i for i in range(len(knots)-1) if knots[i]<=f<knots[i+1]),len(knots)-2)
        m1=m0+1
        ma=ease((f-knots[m0])/(knots[m1]-knots[m0]))
        weights={}
        for b,bw in ((b0,1-ba),(b1,ba)):
            for m,mw in ((m0,1-ma),(m1,ma)):
                weight=bw*mw
                if weight>0:
                    k=b*len(bundles)+m
                    weights[k]=weights.get(k,0.)+weight
        schedule.append(dict(frame=f,base_from=b0,base_to=b1,base_alpha=ba,
                             modifier_from=m0,modifier_to=m1,modifier_alpha=ma,
                             weights=[dict(prompt_id=k,weight=w) for k,w in weights.items()]))
    return dict(version=1,profile="global-softwild",loop="N",frames=frames,seed=seed,
                regional_masks=False,negative=NEGATIVE,base_prompts=bases,modifier_pool=list(MODIFIERS),
                modifier_bundles=bundles,modifier_knots=knots,prompts=prompts,segments=segments,
                frame_schedule=schedule,method="bilinear interpolation of whole-frame CLIP prompt embeddings")

def encode_global_conditioning(pipe, plan):
    import torch
    texts=plan["prompts"]
    for p in texts+[plan["negative"]]:
        n=len(pipe.tokenizer(p,truncation=False,add_special_tokens=True)["input_ids"])
        if n>pipe.tokenizer.model_max_length:
            raise ValueError(f"CLIP token overflow ({n}): {p}")
    device=next(pipe.unet.parameters()).device
    with torch.inference_mode():
        chunks=[]
        for start in range(0,len(texts),4):
            pos,_=pipe.encode_prompt(prompt=texts[start:start+4],device=device,
                    num_images_per_prompt=1,do_classifier_free_guidance=False)
            chunks.append(pos)
        bank=torch.cat(chunks,dim=0)
        _,neg=pipe.encode_prompt(prompt=texts[0],negative_prompt=plan["negative"],
                device=device,num_images_per_prompt=1,do_classifier_free_guidance=True)
        frames=[]
        for row in plan["frame_schedule"]:
            acc=torch.zeros_like(bank[0],dtype=torch.float32)
            for w in row["weights"]:
                acc.add_(bank[w["prompt_id"]].float(),alpha=w["weight"])
            frames.append(acc.to(bank.dtype))
        positives=torch.stack(frames)
        return torch.cat([neg.expand_as(positives),positives],dim=0).contiguous()
