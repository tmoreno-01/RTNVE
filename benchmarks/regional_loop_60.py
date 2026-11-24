#!/usr/bin/env python3
"""60-frame 720p circular regional prompt experiment; no resolution fallback."""
import argparse
import gc
import json
import time
from pathlib import Path

from animatediff_v3 import prepare_model
from animatediff_long_context import uniform_contexts
from overnight_1000 import decode_to_mp4, progress_status
from regional_prompt_travel import (build_plan, make_masks, encode_regions,
                                    RegionalState, CircularContext, install_processors,
                                    REGIONAL_LOOP_MODE)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--motion",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/regional60_rp_cfg7"))
    p.add_argument("--seed",type=int,default=611)
    p.add_argument("--travel-seed",type=int,default=9191)
    a=p.parse_args()
    a.frames=60;a.width=1280;a.height=720;a.fps=10
    a.steps=20;a.guidance=7.;a.vae_slicing=True
    for path in (a.checkpoint,a.motion):
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    import diffusers
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    a.out.mkdir(parents=True,exist_ok=True)
    plan=build_plan(a.frames,a.travel_seed)
    (a.out/"regional_prompt_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    print("RTNVE REGIONAL LOOP: 60 frames, 1280x720, 10 FPS, CFG7, 20 steps",flush=True)
    print(f"Actual context mode: {REGIONAL_LOOP_MODE}, 16/8; three moving, overlapping regional prompts.",flush=True)
    print("Global:",plan["global_prompt"],flush=True)
    for r in plan["regions"]:
        print(r["name"]+": "+" | ".join(r["prompts"]),flush=True)
    print("No LoRAs, no RGB blends, no resolution fallback.",flush=True)
    progress_status(a.out,"loading")
    pipe=prepare_model(a)
    bank=encode_regions(pipe,plan)
    masks=make_masks(plan,a.height//8,a.width//8,"cuda")
    # Save a visual diagnostic of the actual influence fields, NOT final artwork.
    from PIL import Image
    palette=torch.tensor([[.20,.20,.20],[1.,.22,.25],[.18,.70,1.],[.45,1.,.32]],device=masks.device)
    rgb=torch.einsum("fkhw,kc->fhwc",masks,palette)
    sheet=Image.new("RGB",(640,360))
    for i,f in enumerate(range(0,a.frames,5)):
        im=Image.fromarray((rgb[f]*255).round().byte().cpu().numpy()).resize((160,120))
        sheet.paste(im,((i%4)*160,(i//4)*120))
    sheet.save(a.out/"regional_masks.jpg")
    del rgb
    state=RegionalState(masks,bank)
    base_forward=pipe.unet.forward
    original_processors,names=install_processors(pipe.unet,state)
    print(f"Patched {len(names)} spatial cross-attention layers; temporal layers unchanged.",flush=True)
    total_start=time.monotonic()
    def render(frames,steps,control):
        pipe.unet.forward=control.forward
        def callback(pipe_obj,step,timestep,kw):
            if frames==a.frames:
                elapsed=time.monotonic()-start
                print(f"DENOISE {step+1}/{steps} | elapsed {elapsed/60:.1f} min | "
                      f"peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",flush=True)
                progress_status(a.out,"denoising",step=step+1,total=steps,
                    info={"elapsed_seconds":elapsed,"estimated_remaining_seconds":elapsed/(step+1)*(steps-step-1)})
            return kw
        with torch.inference_mode():
            return pipe(prompt=plan["global_prompt"],negative_prompt=plan["negative"],
                        num_frames=frames,width=a.width,height=a.height,
                        num_inference_steps=steps,guidance_scale=a.guidance,
                        generator=torch.Generator("cuda").manual_seed(a.seed),
                        output_type="latent",callback_on_step_end=callback,
                        callback_on_step_end_tensor_inputs=["latents"]).frames
    try:
        progress_status(a.out,"preflight")
        # Test the actual regional attention at native resolution, on one full window.
        print("Regional GPU preflight: 16 frames / 1 diffusion step at 720p",flush=True)
        probe=CircularContext(pipe.unet,state,16)
        t=time.monotonic()
        test=render(16,1,probe)
        torch.cuda.synchronize()
        probe_sec=time.monotonic()-t
        if state.calls==0:
            raise RuntimeError("Regional attention did not execute")
        del test
        pipe.unet.forward=base_forward
        gc.collect();torch.cuda.empty_cache()
        windows=[len(uniform_contexts(s,a.frames,16,1,8,REGIONAL_LOOP_MODE)) for s in range(20)]
        print(f"PREFLIGHT PASS: {probe_sec:.1f}s; {sum(windows)} full UNet windows planned.",flush=True)
        print(f"Very rough GPU-time estimate from preflight: {probe_sec*sum(windows)/60:.0f} min; actual may vary considerably.",flush=True)
        print(f"Starting full {REGIONAL_LOOP_MODE} diffusion.",flush=True)
        control=CircularContext(pipe.unet,state,a.frames)
        if control.loop != REGIONAL_LOOP_MODE or probe.loop != REGIONAL_LOOP_MODE:
            raise RuntimeError("Regional loop configuration mismatch")
        print(f"Verified live controller loop: {control.loop}",flush=True)
        state.calls=0
        torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        progress_status(a.out,"denoising",step=0,total=a.steps)
        latent=render(a.frames,a.steps,control)
        torch.cuda.synchronize()
        seconds=time.monotonic()-start
        peak=torch.cuda.max_memory_allocated()/2**30
        calls=state.calls
        if control.total_unet_calls != sum(windows):
            raise RuntimeError("Actual context call count differs from R-P schedule")
    finally:
        pipe.unet.forward=base_forward
        pipe.unet.set_attn_processor(dict(original_processors))
    # Preserve the completed latents so a subsequent decoder failure does not
    # destroy the expensive result. Mid-denoising resume is not implemented.
    torch.save(latent.detach().cpu(),a.out/"completed_latents.pt")
    path=a.out/f"RTNVE_regional_60f_1280x720_loopRP_cfg7_seed{a.seed}.mp4"
    progress_status(a.out,"encoding",step=0,total=a.frames)
    decode_to_mp4(pipe,latent,path,a.fps,
                  lambda i,n:progress_status(a.out,"encoding",step=i,total=n))
    result=dict(frames=a.frames,width=a.width,height=a.height,fps=a.fps,steps=a.steps,
                guidance=a.guidance,seed=a.seed,travel_seed=a.travel_seed,
                checkpoint=str(a.checkpoint),motion=str(a.motion),
                loop=control.loop,context_window=16,context_overlap=8,stride=1,
                window_merge="tapered noise prediction consensus",
                regional_attention_layers=names,regional_attention_calls=calls,
                unet_forward_calls=control.total_unet_calls,
                preflight_seconds=probe_sec,latent_denoising_seconds=seconds,
                total_seconds=time.monotonic()-total_start,peak_allocated_gib=peak,
                torch=torch.__version__,diffusers=diffusers.__version__,video=str(path),
                warning="R-P wrapped contexts and periodic conditioning encourage closure; visual seam not guaranteed. Not exact A1111 parity.")
    (a.out/"completed_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    progress_status(a.out,"complete",step=a.frames,total=a.frames,info={"video":str(path)})
    print("COMPLETE:",path,flush=True)
    print("Review the visual loop seam and send MP4 + completed_results.json + regional_masks.jpg",flush=True)

if __name__=="__main__":
    main()
