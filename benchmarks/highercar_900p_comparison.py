#!/usr/bin/env python3
"""Six 70-frame tests: native 900p versus two-pass latent diffusion.
Uses the existing R-P temporal controller and native 900p decoder.
"""
from __future__ import annotations

import argparse
import gc
import math
import shutil
import time
from pathlib import Path

from reference_match_triplet import encode_static, write_json
from animatediff_long_context import SlidingContextController, uniform_contexts

PROMPT = (
    "strange looking people, faces, cars, buses, dense crowds and traffic, "
    "high angle CCTV view of New York streets, deep focus, "
    "detailed buildings and pedestrians filling the whole frame, "
    "documentary realism, shattering, ripping apart, torn surfaces, "
    "fractured glass, smeared motion, layered reflections"
)
NEGATIVE = (
    "nsfw, naked, zoom, pan, empty sky, empty background, abstract blobs, "
    "large flat colour areas, smooth gradients, simple geometric shapes, "
    "blank areas, minimal scene, isolated subject"
)
FRAMES, STEPS, FPS, WINDOW, OVERLAP, STRIDE = 70, 20, 10, 16, 4, 1
CFG, LOOP = 5.4, "R-P"
BASE_SIZE, NATIVE_SIZE, OUTPUT_SIZE = (1280, 720), (1600, 904), (1600, 900)

def preset(name, seed, strength=None, refine_steps=None):
    return dict(name=name, seed=seed, mode="native" if strength is None else "multires",
                frames=FRAMES, steps=STEPS, guidance=CFG, fps=FPS, context=WINDOW,
                overlap=OVERLAP, stride=STRIDE, loop=LOOP, width=OUTPUT_SIZE[0],
                height=OUTPUT_SIZE[1], native_width=NATIVE_SIZE[0], native_height=NATIVE_SIZE[1],
                prompt=PROMPT, negative=NEGATIVE, refine_strength=strength,
                refine_steps=refine_steps, lora=False, regional_masks=False,
                prompt_travel=False, filename=f"RTNVE_{name}_70f_1600x900_loopRP_cfg54.mp4")

SUITE = (
    preset("A1_native_seed1223", 1223),
    preset("A2_native_seed731", 731),
    preset("A3_native_seed1889", 1889),
    preset("B1_multires_seed1223_ref35", 1223, .35, 10),
    preset("B2_multires_seed1223_ref50", 1223, .50, 12),
    preset("B3_multires_seed731_ref35", 731, .35, 10),
)

def refinement_schedule(strength, steps):
    """Img2img-style suffix of an exponential DPM schedule.
    Strength denotes fraction of the schedule, NOT linear noise amplitude.
    Retain requested actual denoising steps and record the starting sigma.
    """
    if not 0 < strength < 1 or steps < 2:
        raise ValueError("Invalid refinement schedule")
    total = math.ceil(steps / strength)
    return total, total - steps

def calls_for(steps, frames=FRAMES):
    return sum(len(uniform_contexts(i,frames,WINDOW,STRIDE,OVERLAP,LOOP))
               for i in range(steps))

def clean():
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def fresh_scheduler(pipe):
    return type(pipe.scheduler).from_config(pipe.scheduler.config)

def native_pass(pipe, pos, neg, seed, size, frames, steps, report):
    import torch
    width,height=size
    pipe.scheduler=fresh_scheduler(pipe)
    original=pipe.unet.forward
    embeddings=torch.cat([neg.repeat(frames,1,1),pos.repeat(frames,1,1)]).contiguous()
    control=SlidingContextController(pipe.unet,frames,WINDOW,OVERLAP,STRIDE,LOOP,
                                     frame_conditioning=embeddings)
    generator=torch.Generator(device=pos.device).manual_seed(seed)
    def callback(pipeline, step, timestep, kwargs):
        report(step+1, steps)
        return kwargs
    try:
        pipe.unet.forward=control.forward
        with torch.inference_mode():
            result=pipe(prompt_embeds=pos,negative_prompt_embeds=neg,num_frames=frames,
                        width=width,height=height,num_inference_steps=steps,guidance_scale=CFG,
                        generator=generator,output_type="latent",
                        callback_on_step_end=callback,callback_on_step_end_tensor_inputs=["latents"])
            latent=result.frames
        torch.cuda.synchronize()
        if control.total_unet_calls != calls_for(steps,frames):
            raise RuntimeError("Unexpected temporal window count")
        if tuple(latent.shape)!=(1,4,frames,height//8,width//8):
            raise RuntimeError(f"Unexpected latent shape: {tuple(latent.shape)}")
        return latent.detach(),control.total_unet_calls
    finally:
        pipe.unet.forward=original
        control.frame_conditioning=None
        pipe.scheduler=fresh_scheduler(pipe)

def enlarge_latents(latents, target_size):
    """Spatial resize only: each frame retains its own latent; no temporal interpolation."""
    import torch.nn.functional as F
    w,h=target_size
    b,c,n,lh,lw=latents.shape
    x=latents.permute(0,2,1,3,4).reshape(b*n,c,lh,lw)
    x=F.interpolate(x.float(),size=(h//8,w//8),mode="bicubic",align_corners=False)
    return x.to(latents.dtype).reshape(b,n,c,h//8,w//8).permute(0,2,1,3,4).contiguous()

def refine_pass(pipe, base, pos, neg, ref, report):
    import torch
    strength,steps=ref["refine_strength"],ref["refine_steps"]
    total,start=refinement_schedule(strength,steps)
    scheduler=fresh_scheduler(pipe)
    device=pos.device
    scheduler.set_timesteps(total,device=device)
    timesteps=scheduler.timesteps[start:]
    if len(timesteps)!=steps:
        raise RuntimeError("Scheduler did not create expected refinement timesteps")
    scheduler.set_begin_index(start)
    # Same refinement RNG seed in B1/B2, isolating the schedule change.
    noise_seed=ref["seed"]+100000
    generator=torch.Generator(device=device).manual_seed(noise_seed)
    embeddings=torch.cat([neg.repeat(FRAMES,1,1),pos.repeat(FRAMES,1,1)]).contiguous()
    control=SlidingContextController(pipe.unet,FRAMES,WINDOW,OVERLAP,STRIDE,LOOP,
                                     frame_conditioning=embeddings)
    sigma=float(scheduler.sigmas[start])
    with torch.inference_mode():
        latent=enlarge_latents(base.to(device=device,dtype=pos.dtype),NATIVE_SIZE)
        noise=torch.randn(latent.shape,device=device,dtype=latent.dtype,generator=generator)
        latent=scheduler.add_noise(latent,noise,timesteps[:1])
        del noise
        for i,t in enumerate(timesteps):
            # Match the proven AnimateDiff pipeline's CFG ordering and UNet inputs.
            model_input=scheduler.scale_model_input(torch.cat([latent,latent]),t)
            predicted=control.forward(model_input,t,encoder_hidden_states=embeddings).sample
            uncond,cond=predicted.chunk(2)
            guided=uncond+CFG*(cond-uncond)
            latent=scheduler.step(guided,t,latent,generator=generator).prev_sample
            del model_input,predicted,uncond,cond,guided
            report(i+1,steps)
        if not bool(torch.isfinite(latent).all()):
            raise RuntimeError("Non-finite refined latents")
    torch.cuda.synchronize()
    if control.total_unet_calls!=calls_for(steps):
        raise RuntimeError("Refinement context count mismatch")
    control.frame_conditioning=None
    return latent.detach(),dict(refine_schedule_steps=total,refine_start_index=start,
        refine_schedule_fraction=steps/total,refine_start_sigma=sigma,
        refine_noise_seed=noise_seed,refine_unet_calls=control.total_unet_calls)

def main():
    import torch
    import diffusers
    from animatediff_v3 import prepare_model
    from overnight_1000 import decode_to_mp4, progress_status
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--motion",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/highercar_900p_comparison_suite"))
    a=p.parse_args()
    for path in (a.checkpoint,a.motion):
        if not path.is_file():
            raise FileNotFoundError(path)
    a.out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out).free<3*1024**3:
        raise RuntimeError("At least 3 GiB free disk required")
    a.width,a.height=NATIVE_SIZE
    a.fps=FPS
    a.vae_slicing=True
    manifest=dict(tests=list(SUITE),results=[],checkpoint=str(a.checkpoint),motion=str(a.motion),
                  torch=torch.__version__,diffusers=diffusers.__version__,
                  native_size=NATIVE_SIZE,base_size=BASE_SIZE,output_size=OUTPUT_SIZE,
                  notes="Native 900p versus spatial latent resize and scheduler-consistent second denoise. B1/B2 share the exact base and refinement seed. No frame blending or pixel upscaling.")
    path=a.out/"suite_manifest.json"
    write_json(path,manifest)
    progress_status(a.out,"loading")
    pipe=prepare_model(a)
    plain,pos,neg=encode_static(pipe,SUITE[0])
    print("PROMPT:",plain,"\nNEGATIVE:",NEGATIVE,flush=True)
    print("CLIP prompt validation passed; 6 sequential tests.",flush=True)
    for steps in (1,20,10,12):
        calls_for(steps)
    progress_status(a.out,"preflight")
    probe,_=native_pass(pipe,pos,neg,1223,NATIVE_SIZE,WINDOW,1,lambda i,n:None)
    del probe
    clean()
    print("Native 900p GPU preflight passed.",flush=True)

    # Reuse the exact same clean 720p base for B1 and B2, cached on CPU.
    bases={}
    failures=0
    for index,ref in enumerate(SUITE,1):
        out=a.out/ref["name"]
        out.mkdir(exist_ok=True)
        row=dict(ref)
        write_json(out/"run_manifest.json",row)
        print(f"\nTEST {index}/6: {ref['name']}",flush=True)
        started=time.monotonic()
        latent=None
        def report(stage):
            def update(i,n):
                elapsed=time.monotonic()-started
                print(f"{ref['name']} | {stage} {i}/{n} | {elapsed/60:.1f} min",flush=True)
                progress_status(out,stage,step=i,total=n)
                progress_status(a.out,stage,step=i,total=n,info={"clip":index,"clips":6})
            return update
        try:
            torch.cuda.reset_peak_memory_stats()
            if ref["mode"]=="native":
                latent,count=native_pass(pipe,pos,neg,ref["seed"],NATIVE_SIZE,FRAMES,STEPS,report("native_denoising"))
                row["unet_forward_calls"]=count
            else:
                seed=ref["seed"]
                reused=seed in bases
                if not reused:
                    base_start=time.monotonic()
                    base,base_calls=native_pass(pipe,pos,neg,seed,BASE_SIZE,FRAMES,STEPS,report("base_denoising"))
                    base_cpu=base.cpu()
                    del base
                    base_seconds=time.monotonic()-base_start
                    bases[seed]=(base_cpu,base_calls,base_seconds)
                    torch.save(base_cpu,out/"base_latents.pt")
                    clean()
                base_cpu,base_calls,base_seconds=bases[seed]
                row.update(base_width=BASE_SIZE[0],base_height=BASE_SIZE[1],
                           base_reused=reused,base_unet_calls=base_calls,base_seconds=base_seconds,
                           base_steps=STEPS)
                refine_start=time.monotonic()
                latent,details=refine_pass(pipe,base_cpu,pos,neg,ref,report("refine_denoising"))
                row.update(details,refine_seconds=time.monotonic()-refine_start,
                           unet_forward_calls=details["refine_unet_calls"]+(0 if reused else base_calls))
            row["latent_denoising_seconds"]=time.monotonic()-started
            if tuple(latent.shape)!=(1,4,FRAMES,NATIVE_SIZE[1]//8,NATIVE_SIZE[0]//8):
                raise RuntimeError("Wrong final latent resolution")
            torch.save(latent.detach().cpu(),out/"completed_latents.pt")
            destination=out/ref["filename"]
            decode_to_mp4(pipe,latent,destination,FPS,report("encoding"),crop_size=OUTPUT_SIZE)
            torch.cuda.synchronize()
            row.update(video=str(destination),total_seconds=time.monotonic()-started,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,status="complete")
            write_json(out/"completed_results.json",row)
            progress_status(out,"complete",step=FRAMES,total=FRAMES,info={"video":str(destination)})
            print("COMPLETE:",destination,flush=True)
        except Exception as exc:
            failures+=1
            row.update(status="failed",error=f"{type(exc).__name__}: {exc}",total_seconds=time.monotonic()-started)
            write_json(out/"failed_results.json",row)
            progress_status(out,"failed",info={"error":row["error"]})
            print("FAILED:",row["error"],flush=True)
        finally:
            del latent
            pipe.scheduler=fresh_scheduler(pipe)
            clean()
        manifest["results"].append(row)
        write_json(path,manifest)
    progress_status(a.out,"complete" if not failures else "completed_with_errors",step=6,total=6,
                    info={"failures":failures})
    if failures:
        raise SystemExit(f"{failures} test(s) failed; other completed videos retained. See suite_manifest.json.")
    print("ALL SIX TESTS COMPLETE:",a.out,flush=True)

if __name__=="__main__":
    main()
