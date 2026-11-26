#!/usr/bin/env python3
"""Three 300-frame Highercar variants at native 1280x720, R-P, no LoRAs or prompt travel.
"""
from __future__ import annotations
import argparse
import gc
import json
import re
import shutil
import time
from pathlib import Path

from reference_match_triplet import REFERENCES as ORIGINAL_REFERENCES, encode_static, write_json, LOOP, WIDTH, HEIGHT, FPS, WINDOW, STRIDE

BASE = ORIGINAL_REFERENCES[1]
REFERENCES = (
    dict(BASE, name="01_baseline", frames=300,
         filename="RTNVE_highercar_300f_1280x720_loopRP_cfg5_seed144.mp4"),
    dict(BASE, name="02_dark", frames=300,
         prompt=BASE["prompt"]+", deep shadows, darker tones, moody contrast, black shadows, underlit city",
         filename="RTNVE_highercar_dark_300f_1280x720_loopRP_cfg5_seed144.mp4"),
    dict(BASE, name="03_soft", frames=300, guidance=4.5, overlap=8,
         prompt=BASE["prompt"]+", motion blur, smeared reflections, fractured glass",
         filename="RTNVE_highercar_soft_300f_1280x720_loopRP_cfg45_seed144_ctx16-8.mp4"),
)

def main():
    from animatediff_v3 import prepare_model
    from animatediff_long_context import SlidingContextController, uniform_contexts
    from overnight_1000 import progress_status, decode_to_mp4
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--motion",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/highercar_300_triplet"))
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
                  notes="Baseline preserves the successful 100-frame Highercar test except length. Dark changes prompt only; soft changes prompt, CFG and overlap. No post-grade or Topaz enhancement.",
                  references=list(REFERENCES),results=[])
    write_json(a.out/"triplet_manifest.json",manifest)
    progress_status(a.out,"loading")
    pipe=prepare_model(a)
    original=pipe.unet.forward

    # Validate every weighted prompt before spending time on the first full render.
    for preset in REFERENCES:
        _, validation_pos, validation_neg=encode_static(pipe,preset)
        del validation_pos,validation_neg
    print("All three weighted prompts validated.",flush=True)

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
        name=ref["filename"]
        destination=out/name
        print(f"\nHIGHERCAR {index}/3: {ref['source']} | {ref['frames']} frames | CFG {ref['guidance']} | {ref['steps']} steps | 16/{ref['overlap']} | R-P",flush=True)
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
            print(f"HIGHERCAR {index}/3 | DENOISE {step+1}/{ref['steps']} | {elapsed/60:.1f} min",flush=True)
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
    print("ALL THREE HIGHERCAR VARIANTS COMPLETE:",a.out,flush=True)

if __name__=="__main__":
    main()
