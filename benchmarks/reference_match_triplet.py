#!/usr/bin/env python3
"""Three reference-matched AnimateDiff V3 renders from the original MP4 metadata.
Native 1280x720, original per-clip settings, R-P, no LoRAs or prompt travel.
"""
from __future__ import annotations
import argparse
import gc
import json
import re
import shutil
import time
from pathlib import Path

REFERENCES = (
    dict(name="01_1223", source="00000-1223.mp4",
         prompt="people walking, cars driving,( medium angle view of people walking around new york city, cctv:1.4),  shattering, ripping apart, glass shattering",
         negative="nsfw", seed=1223, frames=100, steps=20, guidance=6., overlap=6,
         original_size=[720,1280], original_lora_tags=["shatterAnimatediff_v10_NEW:1"]),
    dict(name="02_highercar", source="highercar.mp4",
         prompt="strange looking people, cars, planes, boats, buses, motorbikes, faces, ( High angle view of people walking around new york city, cctv:1.2),  shattering, ripping apart,vines, tearing",
         negative="nsfw, naked, duplicates, zoom, pan", seed=144, frames=100, steps=20, guidance=5., overlap=4,
         original_size=[720,1280], original_lora_tags=["shatterAnimatediff_v10:0.65","explosiveAnimatediff_v10:0.3"]),
    dict(name="03_motionnoise", source="higheffectmotionnoisetwo.mp4",
         prompt="(shattering, ripping apart, tearing paper:1.1), strange looking people , cars driving,(High angle view of people walking around new york city, cctv:1.4),",
         negative="nsfw, naked, duplicates, zoom, pan", seed=21, frames=80, steps=22, guidance=5., overlap=4,
         original_size=[648,1152], original_lora_tags=["shatterAnimatediff_v10:0.7"]),
)
LOOP = "R-P"
WIDTH, HEIGHT, FPS, WINDOW, STRIDE = 1280,720,10,16,1

def emphasis_parts(prompt):
    """Parse the explicit (phrase:weight) groups used in these three references.
    No generic nested A1111 syntax or multi-chunk text support is implied.
    """
    parts=[]
    start=0
    for m in re.finditer(r"\(([^()]+):([0-9]+(?:\.[0-9]+)?)\)",prompt):
        if m.start()>start:
            parts.append((prompt[start:m.start()],1.))
        parts.append((m.group(1),float(m.group(2))))
        start=m.end()
    if start<len(prompt):
        parts.append((prompt[start:],1.))
    return parts

def encode_static(pipe, ref):
    """Approximate A1111 Original emphasis: weighted CLIP outputs, mean preserved.
    One static positive/negative pair repeated for all frames.
    """
    import torch
    parts=emphasis_parts(ref["prompt"])
    plain="".join(s for s,w in parts)
    tokenizer=pipe.tokenizer
    ids=[]
    weights=[]
    for s,w in parts:
        tokens=tokenizer(s,add_special_tokens=False,truncation=False)["input_ids"]
        ids.extend(tokens)
        weights.extend([w]*len(tokens))
    whole=tokenizer(plain,add_special_tokens=False,truncation=False)["input_ids"]
    if ids!=whole:
        raise RuntimeError("Weighted prompt token boundaries differ; refusing silent emphasis mismatch")
    maxlen=tokenizer.model_max_length
    if len(ids)+2>maxlen:
        raise RuntimeError("Reference exceeds one CLIP context")
    if len(tokenizer(ref["negative"],truncation=False)["input_ids"])>maxlen:
        raise RuntimeError("Negative exceeds one CLIP context")
    with torch.inference_mode():
        pos,neg=pipe.encode_prompt(prompt=plain,negative_prompt=ref["negative"],
            device=next(pipe.unet.parameters()).device,num_images_per_prompt=1,
            do_classifier_free_guidance=True)
        if pos.shape[1]!=maxlen:
            raise RuntimeError("Unexpected CLIP sequence length")
        w=torch.tensor([1.]+weights+[1.]*(maxlen-len(weights)-1),
                       device=pos.device,dtype=torch.float32).view(1,-1,1)
        original=pos.float()
        weighted=original*w
        denom=weighted.mean()
        if not bool(torch.isfinite(denom)) or abs(float(denom))<1e-8:
            raise RuntimeError("Invalid emphasis normalization")
        pos=(weighted*(original.mean()/denom)).to(pos.dtype)
    return plain,pos,neg

def write_json(path, payload):
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    tmp.replace(path)

def main():
    from animatediff_v3 import prepare_model
    from animatediff_long_context import SlidingContextController, uniform_contexts
    from overnight_1000 import progress_status, decode_to_mp4
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--motion",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/reference_triplet_1280x720"))
    a=p.parse_args()
    for path in (a.checkpoint,a.motion):
        if not path.is_file():
            raise FileNotFoundError(path)
    a.out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out).free<2*1024**3:
        raise RuntimeError("At least 2 GiB free disk required")
    import torch
    import diffusers
    a.vae_slicing=True
    a.width=WIDTH;a.height=HEIGHT;a.fps=FPS
    # Validate every temporal window schedule before model loading.
    for ref in REFERENCES:
        for step in range(ref["steps"]):
            uniform_contexts(step,ref["frames"],WINDOW,STRIDE,ref["overlap"],LOOP)
    manifest=dict(width=WIDTH,height=HEIGHT,fps=FPS,loop=LOOP,window=WINDOW,stride=STRIDE,
                  checkpoint=str(a.checkpoint),motion=str(a.motion),torch=torch.__version__,
                  diffusers=diffusers.__version__,lora=False,prompt_travel=False,regional_masks=False,
                  sampler="Diffusers DPM++ 2M SDE exponential approximation",
                  emphasis="Explicit numeric groups, weighted CLIP outputs with mean normalization",
                  notes="Original MP4 metadata except landscape resolution and omitted LoRAs. No Topaz enhancement. Not exact A1111 parity.",
                  references=list(REFERENCES),results=[])
    write_json(a.out/"triplet_manifest.json",manifest)
    progress_status(a.out,"loading")
    pipe=prepare_model(a)
    original=pipe.unet.forward

    # Exercise actual weighted text conditioning and full-size GPU attention before all three runs.
    ref=REFERENCES[0]
    plain,pos,neg=encode_static(pipe,ref)
    probe=SlidingContextController(pipe.unet,WINDOW,WINDOW,ref["overlap"],STRIDE,LOOP,
        frame_conditioning=torch.cat([neg.repeat(WINDOW,1,1),pos.repeat(WINDOW,1,1)]))
    progress_status(a.out,"preflight")
    try:
        pipe.unet.forward=probe.forward
        with torch.inference_mode():
            result=pipe(prompt_embeds=pos,negative_prompt_embeds=neg,num_frames=WINDOW,
                        width=WIDTH,height=HEIGHT,num_inference_steps=1,guidance_scale=ref["guidance"],
                        generator=torch.Generator("cuda").manual_seed(ref["seed"]),output_type="latent")
        del result
        torch.cuda.synchronize()
    finally:
        pipe.unet.forward=original
        del probe,pos,neg
        gc.collect();torch.cuda.empty_cache()
    print("PREFLIGHT PASS: native 720p, R-P, weighted static prompt",flush=True)

    for index,ref in enumerate(REFERENCES,1):
        out=a.out/ref["name"]
        out.mkdir(exist_ok=True)
        name=f"RTNVE_refmatch_{ref['name']}_{ref['frames']}f_1280x720_loopRP.mp4"
        destination=out/name
        print(f"\nREFERENCE {index}/3: {ref['source']} | {ref['frames']} frames | CFG {ref['guidance']} | {ref['steps']} steps | 16/{ref['overlap']} | R-P",flush=True)
        print("PROMPT:",ref["prompt"],"\nNEGATIVE:",ref["negative"],flush=True)
        plain,pos,neg=encode_static(pipe,ref)
        n=ref["frames"]
        conditioning=torch.cat([neg.repeat(n,1,1),pos.repeat(n,1,1)]).contiguous()
        control=SlidingContextController(pipe.unet,n,WINDOW,ref["overlap"],STRIDE,LOOP,
                                         frame_conditioning=conditioning)
        planned=sum(len(uniform_contexts(s,n,WINDOW,STRIDE,ref["overlap"],LOOP)) for s in range(ref["steps"]))
        # Reset the scheduler for each independent clip.
        pipe.scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config)
        row=dict(ref,width=WIDTH,height=HEIGHT,fps=FPS,loop=control.loop,context=WINDOW,stride=STRIDE,
                 lora=False,regional_masks=False,prompt_travel=False,planned_unet_calls=planned)
        write_json(out/"run_manifest.json",row)
        start=time.monotonic()
        def callback(pipeline,step,timestep,kwargs):
            elapsed=time.monotonic()-start
            print(f"REFERENCE {index}/3 | DENOISE {step+1}/{ref['steps']} | {elapsed/60:.1f} min",flush=True)
            progress_status(out,"denoising",step=step+1,total=ref["steps"],info={"elapsed_seconds":elapsed})
            progress_status(a.out,"denoising",step=step+1,total=ref["steps"],info={"clip":index,"clips":3})
            return kwargs
        try:
            progress_status(out,"denoising",step=0,total=ref["steps"])
            pipe.unet.forward=control.forward
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                generated=pipe(prompt_embeds=pos,negative_prompt_embeds=neg,num_frames=n,
                    width=WIDTH,height=HEIGHT,num_inference_steps=ref["steps"],guidance_scale=ref["guidance"],
                    generator=torch.Generator("cuda").manual_seed(ref["seed"]),output_type="latent",
                    callback_on_step_end=callback,callback_on_step_end_tensor_inputs=["latents"])
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
            peak=torch.cuda.max_memory_allocated()/2**30
            if control.total_unet_calls!=planned:
                raise RuntimeError("R-P window count mismatch")
            latent=generated.frames
            del generated
        finally:
            pipe.unet.forward=original
            control.frame_conditioning=None
            del conditioning,pos,neg
        torch.save(latent.detach().cpu(),out/"completed_latents.pt")
        progress_status(out,"encoding",step=0,total=n)
        decode_to_mp4(pipe,latent,destination,FPS,
            lambda i,total:progress_status(out,"encoding",step=i,total=total))
        del latent
        gc.collect();torch.cuda.empty_cache()
        row.update(video=str(destination),latent_denoising_seconds=seconds,total_seconds=time.monotonic()-start,
                   peak_allocated_gib=peak,unet_forward_calls=control.total_unet_calls)
        write_json(out/"completed_results.json",row)
        manifest["results"].append(row)
        write_json(a.out/"triplet_manifest.json",manifest)
        progress_status(out,"complete",step=n,total=n,info={"video":str(destination)})
        print("COMPLETE:",destination,flush=True)
    progress_status(a.out,"complete",step=3,total=3)
    print("ALL THREE REFERENCES COMPLETE:",a.out,flush=True)

if __name__=="__main__":
    main()
