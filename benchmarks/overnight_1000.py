#!/usr/bin/env python3
"""RTNVE - 1000 FRAME overnight ONE-LATENT-VIDEO teacher.

IMPORTANT:
* One connected 1000-frame latent video, NOT separate clips or crossfades.
* Every denoising step runs a non-looping overlapping 16/8 AnimateDiff context.
* 20 steps CFG6 with wildly random prompts and per-change 20-60-frame CLIP morphs.
* 1280x720 HIGH RISK on an RTX3090. A 16-frame 1-step GPU preflight checks
  whether the largest UNet activation fits before the expensive 1000f pass.
* output_type='latent': no 1000-frame PIL/CPU image stack.
* Stream VAE-decoded frames directly into an MP4 writer after denoising.
* No mid-denoising-step resume yet: interruption during the 20-step global
  diffusion loses that attempt. Status is logged, not falsely 'resumable'.
* Optional explicit 896x504 fallback on preflight CUDA OOM.
* Doesn't touch A1111, models or any unrelated user files.
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from animatediff_long_context import SlidingContextController, uniform_contexts
from animatediff_v3 import prepare_model
from prompt_travel import encode_frame_conditioning
from wild_prompt_travel import make_plan


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--motion",required=True,type=Path)
    parser.add_argument("--frames",type=int,default=1000)
    parser.add_argument("--width",type=int,default=1280)
    parser.add_argument("--height",type=int,default=720)
    parser.add_argument("--fps",type=int,default=10)
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--guidance",type=float,default=6.0)
    parser.add_argument("--seed",type=int,default=611)
    parser.add_argument("--travel-seed",type=int,default=9191)
    parser.add_argument("--travel-min",type=int,default=60)
    parser.add_argument("--travel-max",type=int,default=140)
    parser.add_argument("--travel-transition-min",type=int,default=20)
    parser.add_argument("--travel-transition-max",type=int,default=60)
    parser.add_argument("--window",type=int,default=16)
    parser.add_argument("--overlap",type=int,default=8)
    parser.add_argument("--out",type=Path,default=Path("benchmarks/results/overnight1000"))
    parser.add_argument("--fallback-896",action="store_true",help="Only if 1280x720 GPU preflight OOM, automatically use 896x504 instead of ending overnight")
    parser.add_argument("--skip-preflight",action="store_true",help="Not recommended")
    return parser.parse_args()


def progress_status(folder:Path, phase:str, *, step:int|None=None, total:int|None=None, info:dict|None=None):
    payload={"utc":datetime.now(timezone.utc).isoformat(),"phase":phase,
             "step":step,"total_steps":total}
    if info:payload.update(info)
    pending=folder/"status.json.tmp"
    pending.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    os.replace(pending,folder/"status.json")


def gpu_preflight(pipe,args):
    """Single UNet step with 16 video frames at the chosen full resolution.

    It detects severe per-window OOM before launching 1000 frames.
    It is not a guarantee: full-length global buffers need extra memory.
    """
    import torch
    sample_len=max(16,args.window)
    controller=SlidingContextController(
        pipe.unet,sample_len,args.window,args.overlap,1,"N",
        frame_conditioning=None
    )
    original=pipe.unet.forward
    print("GPU PREFLIGHT: one step at",args.width,"x",args.height,
          "with",sample_len,"frames; this can still OOM.",flush=True)
    try:
        pipe.unet.forward=controller.forward
        with torch.inference_mode():
            result=pipe(
                prompt="high angle view, busy detailed street, crowded pedestrians, broken glass",
                negative_prompt="nsfw, vignette, shallow depth of field",
                num_frames=sample_len,
                width=args.width,height=args.height,
                num_inference_steps=1,guidance_scale=args.guidance,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                output_type="latent",
            )
        del result
        torch.cuda.synchronize()
        print("GPU PREFLIGHT PASS for",args.width,"x",args.height,flush=True)
    finally:
        pipe.unet.forward=original
        gc.collect()
        torch.cuda.empty_cache()


def decode_to_mp4(pipe, latents, destination:Path, fps:int, log_callback=None):
    """Stream one neural VAE frame at a time; never create a 1000-frame PIL list."""
    import torch
    import numpy as np
    import imageio.v2 as imageio

    if latents.ndim!=5 or latents.shape[0]!=1 or latents.shape[1]!=4:
        raise RuntimeError("Expected one [1,4,frames,h,w] SD1.5 latent video")
    n=latents.shape[2]
    temporary=destination.with_name(destination.stem+".incomplete.mp4")
    # .incomplete.mp4 is not silently mistaken for a complete rendered MP4.
    writer=imageio.get_writer(
        str(temporary),format="FFMPEG",mode="I",fps=fps,
        codec="libx264",quality=8,macro_block_size=1,
    )
    try:
        scale=pipe.vae.config.scaling_factor
        with torch.inference_mode():
            for i in range(n):
                mini=latents[:,:,i] / scale
                reconstructed=pipe.vae.decode(mini).sample
                pixels=((reconstructed[0].float().clamp(-1,1)+1.0)*127.5)
                frame=(pixels.permute(1,2,0).round().byte().cpu().numpy())
                if frame.dtype != np.uint8 or frame.shape[-1]!=3:
                    raise RuntimeError("Unexpected VAE RGB frame")
                writer.append_data(frame)
                del mini,reconstructed,pixels,frame
                if (i+1)%25==0 or i==n-1:
                    print(f"VIDEO ENCODE: {i+1}/{n} frames",flush=True)
                    if log_callback:log_callback(i+1,n)
    finally:
        writer.close()
    os.replace(temporary,destination)


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
    if not 1<=a.travel_transition_min<=a.travel_transition_max:
        raise ValueError("Prompt transition range invalid")
    if a.travel_transition_min>=a.travel_min or a.travel_min>a.travel_max:
        raise ValueError("Prompt transitions must fit within dwell lengths")
    if a.frames<a.travel_min:
        raise ValueError("Frame count less than minimum prompt dwell")

    # Import GPU deps only after CLI/path errors have been checked.
    import torch
    import diffusers
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RTX3090 not available")
    a.out.mkdir(parents=True,exist_ok=True)
    free_disk=shutil.disk_usage(a.out).free
    if free_disk<2*1024**3:
        raise RuntimeError("At least 2 GiB free disk requested before overnight encoding")
    plan=make_plan(a.frames, a.travel_seed,a.travel_min,a.travel_max,
                   a.travel_transition_min,a.travel_transition_max)
    plan_path=a.out/"wild_prompt_plan.json"
    plan_path.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding="utf-8")

    print("="*68)
    print("RTNVE OVERNIGHT: one connected latent video, WITHOUT the old 200f cap")
    print(f"FRAMES: {a.frames}; FPS: {a.fps}; PLAYBACK: {a.frames/a.fps:.0f} sec")
    print(f"TARGET: {a.width}x{a.height}, {a.steps} steps, CFG {a.guidance}")
    print(f"CONTEXT: 16/8, stride 1, N (absolutely no loop)")
    print(f"SEMANTICS: {len(plan['prompts'])} separate wild prompts,")
    print(f"{a.travel_min}-{a.travel_max} frame random dwell, independently varied "
          f"{a.travel_transition_min}-{a.travel_transition_max} frame CLIP morphs.")
    print("NO LoRAs. No server inference. This is SLOW offline CUDA synthesis.")
    print("PROGRESS file: ",a.out/"status.json")
    print("WARN: 1000-frame global denoising does not yet support resuming")
    print("      halfway through if power is lost. Completed MP4 is streamed safely.")
    print("="*68,flush=True)
    for i,prompt in enumerate(plan["prompts"]):
        print(f"PROMPT {i+1:02d}: {prompt}",flush=True)

    a.vae_slicing=True
    progress_status(a.out,"loading")
    pipe=prepare_model(a)
    # Preserve original SD1.5 VAE visual detail; no spatial tiling until
    # experience indicates a decoder OOM at a single 720p frame.
    if not a.skip_preflight:
        try:
            gpu_preflight(pipe,a)
        except torch.OutOfMemoryError:
            if not a.fallback_896 or (a.width,a.height)!=(1280,720):
                raise RuntimeError(
                    "1280x720 context16 GPU preflight OOM. Try --fallback-896 "
                    "or rerun with --width 896 --height 504."
                )
            print("WARNING: 1280x720 context OOM; automatically trying 896x504.",flush=True)
            a.width=896;a.height=504
            torch.cuda.empty_cache();gc.collect()
            gpu_preflight(pipe,a)

    # Report and preserve actual resolution, never silently pretend fallback=720p.
    manifest={
        "run_type":"full-global-latent-v3-not-causal-not-realtime",
        "created":datetime.now(timezone.utc).isoformat(),
        "frames":a.frames,"width":a.width,"height":a.height,"fps":a.fps,
        "steps":a.steps,"guidance":a.guidance,"seed":a.seed,
        "context_window":16,"context_overlap":8,"stride":1,"loop":"N",
        "checkpoint":str(a.checkpoint),"motion":str(a.motion),
        "prompt_plan":str(plan_path),
        "travel_prompts":len(plan["prompts"]),
        "travel_seed":a.travel_seed,"travel_dwell_min":a.travel_min,
        "travel_dwell_max":a.travel_max,
        "travel_transition_min":a.travel_transition_min,
        "travel_transition_max":a.travel_transition_max,
        "individual_transition_lengths": [seg["transition_frames"] for seg in plan["segments"]],
        "torch":torch.__version__,"diffusers":diffusers.__version__,
        "fallback_used":(a.width,a.height)!=(1280,720),
        "note":"No midstep-resume; all 1000 latents jointly denoised, then stream-decoded."
    }
    (a.out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Encoding wild prompt CLIP bank. Some long text may truncate at 77 tokens.",flush=True)
    # Unlike old 48-frame test, initial text is the first wild prompt.
    frame_conditioning=encode_frame_conditioning(
        pipe,plan,
        "nsfw, vignette, shallow depth of field, blank background, empty scene"
    )
    if frame_conditioning.shape[0]!=2*a.frames:
        raise RuntimeError("Prompt bank didn't align with 1000 frame CFG conditioning")
    del plan
    torch.cuda.empty_cache()

    controller=SlidingContextController(
        pipe.unet,a.frames,16,8,1,"N",frame_conditioning=frame_conditioning
    )
    windows=len(uniform_contexts(0,a.frames,16,1,8,"N"))
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
                prompt="high-angle crowded street transformed into impossible architecture",
                negative_prompt="nsfw, vignette, shallow depth of field, blank background, empty scene",
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
    latent=generated.frames
    del generated;gc.collect()
    name=f"RTNVE_wild_{a.frames}f_{a.width}x{a.height}_N_ctx16-8_seed{a.seed}.mp4"
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
        sys.exit(f"RTNVE 1000-FRAME OVERNIGHT RENDER FAILED: {exc}")
