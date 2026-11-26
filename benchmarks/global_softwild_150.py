#!/usr/bin/env python3
"""150-frame global Softwild experiment, no spatial masks or resolution fallback."""
from __future__ import annotations
import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from animatediff_long_context import SlidingContextController, uniform_contexts
from animatediff_v3 import prepare_model
from overnight_1000 import decode_to_mp4, progress_status
from global_softwild_prompt_travel import make_plan, encode_global_conditioning, LOOP_MODE

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--motion",required=True,type=Path)
    p.add_argument("--seed",type=int,default=611)
    p.add_argument("--travel-seed",type=int,default=9191)
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/global150_A_cfg65_softwild"))
    a=p.parse_args()
    a.frames=150;a.width=1280;a.height=720;a.fps=10
    a.steps=20;a.guidance=6.5;a.window=16;a.overlap=8;a.loop=LOOP_MODE
    return a

def main():
    a=cli()
    for name in ("checkpoint","motion"):
        path=getattr(a,name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if any(x<1 for x in (a.frames,a.width,a.height,a.steps,a.fps)):
        raise ValueError("Invalid dimensions, steps or frames")
    if a.frames>10000:
        raise ValueError("10,000-frame job limit (the old 200-frame cap is NOT used)")
    if a.width%8 or a.height%8:
        raise ValueError("Dimensions must be divisible by 8")
    if a.window!=16 or a.overlap!=8:
        raise ValueError("Overnight teacher is intentionally fixed at 16/8, proven quality setup")
    # Import GPU deps only after CLI/path errors have been checked.
    import torch
    import diffusers
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RTX3090 not available")
    a.out.mkdir(parents=True,exist_ok=True)
    free_disk=shutil.disk_usage(a.out).free
    if free_disk<2*1024**3:
        raise RuntimeError("At least 2 GiB free disk requested before overnight encoding")
    plan=make_plan(a.frames,a.travel_seed)
    negative=plan["negative"]
    first_prompt=plan["prompts"][0]
    label="global_softwild_reference_loop"
    if plan["loop"]!=a.loop:
        raise RuntimeError("Prompt and temporal loop modes differ")
    planned_calls=sum(len(uniform_contexts(step,a.frames,16,1,8,a.loop)) for step in range(a.steps))
    plan_path=a.out/"global_prompt_plan.json"
    plan_path.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding="utf-8")

    print("="*68)
    print(f"RTNVE {label.upper()}: one connected latent video")
    print(f"FRAMES: {a.frames}; FPS: {a.fps}; PLAYBACK: {a.frames/a.fps:.0f} sec")
    print(f"TARGET: {a.width}x{a.height}, {a.steps} steps, CFG {a.guidance}")
    print(f"CONTEXT: 16/8, stride 1, {a.loop} (circular)")
    print("Four circular whole-frame scenes; independently drifting modifier pairs. No regional masks.")
    print("NO LoRAs. No server inference. This is SLOW offline CUDA synthesis.")
    print("PROGRESS file: ",a.out/"status.json")
    print("WARN: global denoising does not yet support resuming")
    print("      halfway through if power is lost. Completed MP4 is streamed safely.")
    print("="*68,flush=True)
    for i,prompt in enumerate(plan["prompts"]):
        print(f"PROMPT {i+1:02d}: {prompt}",flush=True)

    a.vae_slicing=True
    progress_status(a.out,"loading")
    pipe=prepare_model(a)
    # Preserve original SD1.5 VAE visual detail; no spatial tiling until
    # experience indicates a decoder OOM at a single 720p frame.
    progress_status(a.out,"preflight")
    original=pipe.unet.forward
    probe=SlidingContextController(pipe.unet,16,16,8,1,a.loop)
    try:
        pipe.unet.forward=probe.forward
        with torch.inference_mode():
            result=pipe(prompt=first_prompt,negative_prompt=negative,num_frames=16,
                        width=a.width,height=a.height,num_inference_steps=1,
                        guidance_scale=a.guidance,generator=torch.Generator("cuda").manual_seed(a.seed),
                        output_type="latent")
        del result
        torch.cuda.synchronize()
        print(f"PREFLIGHT PASS: {a.width}x{a.height}, loop {probe.loop}",flush=True)
    finally:
        pipe.unet.forward=original
        gc.collect()
        torch.cuda.empty_cache()

    # Report and preserve actual resolution, never silently pretend fallback=720p.
    manifest={
        "run_type":"full-global-latent-v3-not-causal-not-realtime",
        "created":datetime.now(timezone.utc).isoformat(),
        "frames":a.frames,"width":a.width,"height":a.height,"fps":a.fps,
        "steps":a.steps,"guidance":a.guidance,"seed":a.seed,
        "context_window":16,"context_overlap":8,"stride":1,"loop":a.loop,
        "checkpoint":str(a.checkpoint),"motion":str(a.motion),
        "prompt_plan":str(plan_path),
        "prompt_mode":label,"negative_prompt":negative,
        "regional_masks":False,"travel_seed":a.travel_seed,
        "travel_prompts":len(plan["base_prompts"]),"modifier_bundles":plan["modifier_bundles"],
        "individual_transition_lengths": [seg["transition_frames"] for seg in plan["segments"]],
        "torch":torch.__version__,"diffusers":diffusers.__version__,
        "fallback_used":(a.width,a.height)!=(1280,720),
        "note":"Circular temporal windows and periodic conditioning encourage a loop; visual seam not guaranteed. No midstep-resume."
    }
    (a.out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Encoding {label} prompt CLIP bank.",flush=True)
    frame_conditioning=encode_global_conditioning(pipe,plan)
    if frame_conditioning.shape[0]!=2*a.frames:
        raise RuntimeError("Prompt bank didn't align with global frame CFG conditioning")
    del plan
    torch.cuda.empty_cache()

    controller=SlidingContextController(
        pipe.unet,a.frames,16,8,1,a.loop,frame_conditioning=frame_conditioning
    )
    windows=len(uniform_contexts(0,a.frames,16,1,8,a.loop))
    print(f"Verified live controller loop: {controller.loop}",flush=True)
    print(f"GLOBAL VIDEO: {windows} UNet windows per diffusion step (not clip segments).")
    print("STARTING FULL DIFFUSION (no VAE decoding until ALL steps finish)",flush=True)
    original_forward=pipe.unet.forward
    generated=None
    started=time.monotonic()

    def callback(pipe_obj, step, timestep, kwargs):
        if (step+1)%1==0:
            allocated=torch.cuda.max_memory_allocated()/1024**3
            print(f"DENOISE STEP {step+1}/{a.steps}; peak CUDA allocated {allocated:.2f} GiB",flush=True)
            progress_status(a.out,"denoising",step=step+1,total=a.steps,
                            info={"peak_cuda_gib":round(allocated,3)})
        return kwargs

    try:
        pipe.unet.forward=controller.forward
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.inference_mode():
            generated=pipe(
                prompt=first_prompt,
                negative_prompt=negative,
                num_frames=a.frames,
                width=a.width,height=a.height,
                num_inference_steps=a.steps,
                guidance_scale=a.guidance,
                generator=torch.Generator("cuda").manual_seed(a.seed),
                output_type="latent",
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        torch.cuda.synchronize()
        seconds=time.monotonic()-started
        peak=torch.cuda.max_memory_allocated()/1024**3
        print(f"DENOISING COMPLETE: {seconds:.1f}s, peak {peak:.2f} GiB.",flush=True)
    finally:
        pipe.unet.forward=original_forward
        del frame_conditioning
        controller.frame_conditioning=None

    if generated is None:
        raise RuntimeError("Denoising returned no latent video")
    if controller.total_unet_calls!=planned_calls:
        raise RuntimeError("Unexpected circular context call count")
    latent=generated.frames
    del generated;gc.collect()
    torch.save(latent.detach().cpu(),a.out/"completed_latents.pt")
    name=f"RTNVE_global_150f_1280x720_loopA_cfg65_seed{a.seed}.mp4"
    output=a.out/name
    progress_status(a.out,"encoding",info={"latent_seconds":round(seconds,2)})

    def on_encoded(frame,n):
        progress_status(a.out,"encoding",step=frame,total=n)
    decode_to_mp4(pipe,latent,output,a.fps,on_encoded)
    del latent;gc.collect();torch.cuda.empty_cache()

    manifest.update({
        "completed":datetime.now(timezone.utc).isoformat(),
        "latent_denoising_seconds":round(seconds,3),
        "full_elapsed_seconds":round(time.monotonic()-started,3),
        "batch_generated_fps":round(a.frames/(time.monotonic()-started),4),
        "max_allocated_cuda_gib":round(peak,3),
        "unet_forward_calls":controller.total_unet_calls,
        "output_mp4":str(output),
    })
    (a.out/"completed_results.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    progress_status(a.out,"complete",step=a.frames,total=a.frames,
                    info={"video":str(output)})
    print("COMPLETE:",output,flush=True)
    print("RESULTS:",a.out/"completed_results.json",flush=True)
    print("PROMPTS:",plan_path,flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        sys.exit(f"RTNVE GLOBAL RENDER FAILED: {exc}")
