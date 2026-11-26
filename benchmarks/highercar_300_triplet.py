#!/usr/bin/env python3
"""Highercar 720p and 900p experiments, R-P, no LoRAs or prompt travel.
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
    dict(BASE, name="04_highres", frames=50, width=1600, height=900,
         native_width=1600, native_height=904,
         filename="RTNVE_highercar_50f_1600x900_loopRP_cfg5_seed144.mp4"),
)

# Keep the proven 900p path and original batch untouched; only conditioning,
# seed and the third preset's CFG/overlap differ in this new experiment.
HIGHRES_REFERENCES = (
    dict(REFERENCES[-1], name="01_original", seed=1223,
         filename="RTNVE_highercar_original_50f_1600x900_loopRP_cfg5_seed1223.mp4"),
    dict(REFERENCES[-1], name="02_subtle_dark", seed=731,
         prompt=BASE["prompt"]+", deep shadows, moody contrast",
         filename="RTNVE_highercar_subtledark_50f_1600x900_loopRP_cfg5_seed731.mp4"),
    dict(REFERENCES[-1], name="03_hybrid", seed=2849, guidance=4.8, overlap=6,
         prompt=BASE["prompt"]+", moody contrast, smeared reflections, fractured glass",
         filename="RTNVE_highercar_hybrid_50f_1600x900_loopRP_cfg48_seed2849_ctx16-6.mp4"),
)

# Same three seed/prompt presets, with only clip length changed.
HIGHRES_70_REFERENCES = tuple(
    dict(ref, frames=70, filename=ref["filename"].replace("_50f_", "_70f_"))
    for ref in HIGHRES_REFERENCES
)

def render_size(ref):
    return ref.get("native_width", WIDTH), ref.get("native_height", HEIGHT)

def output_size(ref):
    return ref.get("width", WIDTH), ref.get("height", HEIGHT)

def main():
    from animatediff_v3 import prepare_model
    from animatediff_long_context import SlidingContextController, uniform_contexts
    from overnight_1000 import progress_status, decode_to_mp4
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--motion",required=True,type=Path)
    p.add_argument("--out",type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--only-highres",action="store_true",help="Run only the original seed-144 test 04.")
    mode.add_argument("--highres-triplet",action="store_true",help="Run three new 50-frame 900p seed variants.")
    mode.add_argument("--highres-70-triplet",action="store_true",help="Run the same three 900p seed variants at 70 frames.")
    a=p.parse_args()
    refs=HIGHRES_70_REFERENCES if a.highres_70_triplet else HIGHRES_REFERENCES if a.highres_triplet else REFERENCES[-1:] if a.only_highres else REFERENCES
    if a.out is None:
        folder="highercar_900p_70f_triplet" if a.highres_70_triplet else "highercar_900p_triplet" if a.highres_triplet else "highercar_50_highres" if a.only_highres else "highercar_300_triplet"
        a.out=Path("benchmarks/results")/folder
    count=len(refs)
    for path in (a.checkpoint,a.motion):
        if not path.is_file():
            raise FileNotFoundError(path)
    a.out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out).free<2*1024**3:
        raise RuntimeError("At least 2 GiB free disk required")
    import torch
    import diffusers
    a.vae_slicing=True
    a.width,a.height=render_size(refs[0]);a.fps=FPS
    # Validate every temporal window schedule before model loading.
    for ref in refs:
        w,h=render_size(ref)
        ow,oh=output_size(ref)
        if w%8 or h%8 or ow%2 or oh%2 or ow>w or oh>h:
            raise ValueError("Invalid native/output dimensions")
        for step in range(ref["steps"]):
            uniform_contexts(step,ref["frames"],WINDOW,STRIDE,ref["overlap"],LOOP)
    manifest=dict(batch="highres_70_triplet" if a.highres_70_triplet else "highres_triplet" if a.highres_triplet else "original_highres" if a.only_highres else "original_batch",width=output_size(refs[0])[0],height=output_size(refs[0])[1],fps=FPS,loop=LOOP,window=WINDOW,stride=STRIDE,
                  checkpoint=str(a.checkpoint),motion=str(a.motion),torch=torch.__version__,
                  diffusers=diffusers.__version__,lora=False,prompt_travel=False,regional_masks=False,
                  sampler="Diffusers DPM++ 2M SDE exponential approximation",
                  emphasis="Explicit numeric groups, weighted CLIP outputs with mean normalization",
                  notes=("Same three 900p seed and conditioning presets as the 50-frame triplet, extended to 70 frames. Native 1600x904 cropped to 1600x900; no scaling or post-grade." if a.highres_70_triplet else "Three new seeds, original baseline prompt or restrained dark/hybrid variations. All native 1600x904 cropped to 1600x900; no scaling or post-grade." if a.highres_triplet else "Baseline preserves the successful 100-frame Highercar test except length. Dark changes prompt only; soft changes prompt, CFG and overlap. Test 04 uses baseline conditioning, 50 frames, native 1600x904 cropped to 1600x900 (no scaling). No post-grade or Topaz enhancement."),
                  references=[dict(r,width=output_size(r)[0],height=output_size(r)[1],native_width=render_size(r)[0],native_height=render_size(r)[1]) for r in refs],results=[])
    write_json(a.out/"triplet_manifest.json",manifest)
    progress_status(a.out,"loading")
    pipe=prepare_model(a)
    original=pipe.unet.forward

    # Validate every weighted prompt before spending time on the first full render.
    for preset in refs:
        _, validation_pos, validation_neg=encode_static(pipe,preset)
        del validation_pos,validation_neg
    print(f"All {count} weighted prompts validated.",flush=True)

    # Exercise actual weighted text conditioning and full-size GPU attention for the first run.
    ref=refs[0]
    probe_w,probe_h=render_size(ref)
    plain,pos,neg=encode_static(pipe,ref)
    probe=SlidingContextController(pipe.unet,WINDOW,WINDOW,ref["overlap"],STRIDE,LOOP,
        frame_conditioning=torch.cat([neg.repeat(WINDOW,1,1),pos.repeat(WINDOW,1,1)]))
    progress_status(a.out,"preflight")
    try:
        pipe.unet.forward=probe.forward
        with torch.inference_mode():
            result=pipe(prompt_embeds=pos,negative_prompt_embeds=neg,num_frames=WINDOW,
                        width=probe_w,height=probe_h,num_inference_steps=1,guidance_scale=ref["guidance"],
                        generator=torch.Generator("cuda").manual_seed(ref["seed"]),output_type="latent")
        del result
        torch.cuda.synchronize()
    finally:
        pipe.unet.forward=original
        del probe,pos,neg
        gc.collect();torch.cuda.empty_cache()
    print(f"PREFLIGHT PASS: native {probe_w}x{probe_h}, R-P, weighted static prompt",flush=True)

    for index,ref in enumerate(refs,1):
        native_w,native_h=render_size(ref)
        output_w,output_h=output_size(ref)
        out=a.out/ref["name"]
        out.mkdir(exist_ok=True)
        name=ref["filename"]
        destination=out/name
        print(f"\nHIGHERCAR {index}/{count}: {ref['source']} | {ref['frames']} frames | native {native_w}x{native_h} -> {output_w}x{output_h} | seed {ref['seed']} | CFG {ref['guidance']} | {ref['steps']} steps | 16/{ref['overlap']} | R-P",flush=True)
        print("PROMPT:",ref["prompt"],"\nNEGATIVE:",ref["negative"],flush=True)
        plain,pos,neg=encode_static(pipe,ref)
        n=ref["frames"]
        conditioning=torch.cat([neg.repeat(n,1,1),pos.repeat(n,1,1)]).contiguous()
        control=SlidingContextController(pipe.unet,n,WINDOW,ref["overlap"],STRIDE,LOOP,
                                         frame_conditioning=conditioning)
        planned=sum(len(uniform_contexts(s,n,WINDOW,STRIDE,ref["overlap"],LOOP)) for s in range(ref["steps"]))
        # Reset the scheduler for each independent clip.
        pipe.scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config)
        row=dict(ref,width=output_w,height=output_h,native_width=native_w,native_height=native_h,fps=FPS,loop=control.loop,context=WINDOW,stride=STRIDE,
                 lora=False,regional_masks=False,prompt_travel=False,planned_unet_calls=planned)
        write_json(out/"run_manifest.json",row)
        start=time.monotonic()
        def callback(pipeline,step,timestep,kwargs):
            elapsed=time.monotonic()-start
            print(f"HIGHERCAR {index}/{count} | DENOISE {step+1}/{ref['steps']} | {elapsed/60:.1f} min",flush=True)
            progress_status(out,"denoising",step=step+1,total=ref["steps"],info={"elapsed_seconds":elapsed})
            progress_status(a.out,"denoising",step=step+1,total=ref["steps"],info={"clip":index,"clips":count})
            return kwargs
        try:
            progress_status(out,"denoising",step=0,total=ref["steps"])
            pipe.unet.forward=control.forward
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                generated=pipe(prompt_embeds=pos,negative_prompt_embeds=neg,num_frames=n,
                    width=native_w,height=native_h,num_inference_steps=ref["steps"],guidance_scale=ref["guidance"],
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
            lambda i,total:progress_status(out,"encoding",step=i,total=total),
            crop_size=(output_w,output_h) if (native_w,native_h)!=(output_w,output_h) else None)
        del latent
        gc.collect();torch.cuda.empty_cache()
        row.update(video=str(destination),latent_denoising_seconds=seconds,total_seconds=time.monotonic()-start,
                   peak_allocated_gib=peak,unet_forward_calls=control.total_unet_calls)
        write_json(out/"completed_results.json",row)
        manifest["results"].append(row)
        write_json(a.out/"triplet_manifest.json",manifest)
        progress_status(out,"complete",step=n,total=n,info={"video":str(destination)})
        print("COMPLETE:",destination,flush=True)
    progress_status(a.out,"complete",step=count,total=count)
    print(f"ALL {count} HIGHERCAR TESTS COMPLETE:",a.out,flush=True)

if __name__=="__main__":
    main()
