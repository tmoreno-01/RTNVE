#!/usr/bin/env python3
"""RTNVE long temporal-context experiment — research only, NOT real time.

Based on the overlapping-context selection in continue-revolution/sd-webui-animatediff:
https://github.com/continue-revolution/sd-webui-animatediff/blob/master/scripts/animatediff_infv2v.py

During EACH diffusion denoising step, run the AnimateDiff UNet on overlapping
13-frame temporal groups instead of the entire 100-frame sequence at once.
Merge predictions across the SAME global latent video. This is NOT joining
independent short MP4s, nor does it provide causal streaming.

Fidelity caveats: the A1111 extension's original context selection/overwrite
method is approximated, but its prompt emphasis encoder, scheduler exactness,
A1111 seed/noise details, optional LoRAs and long-video prompt scheduling are
NOT reproduced identically. Refer to screenshots/original clips when judging.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RTNVE_CONTEXT_PATCH = "2026-11-23-cfg-window-v2"

from animatediff_v3 import prepare_model
from prompt_travel import build_prompt_plan, encode_frame_conditioning

CITY_PROMPT = (
    "strange looking people, cars driving,( High angle view of people walking "
    "around new york city, cctv:1.2),  shattering, ripping apart, destruction, "
    "leaves, plants growing, tearing"
)
WINDOWS_PROMPT = (
    "(A massive infinite grid of tiny rectangular apartment windows:1.1), "
    "extreme distant view. (Each window is an illuminated room showing tiny people, "
    "small crowds, and miniature furniture, living room:1.3), shattering, ripping"
)
DEFAULT_NEGATIVE = "nsfw, naked, duplicates, zoom, pan"


def ordered_halving(value: int) -> float:
    """A1111 AnimateDiff's 64-bit bit-reversed fractional shift."""
    if value < 0 or value >= 2**64:
        raise ValueError("step must be a nonnegative 64-bit integer")
    return int(f"{value:064b}"[::-1], 2) / (2**64)


def uniform_contexts(
    step: int,
    frame_count: int,
    batch_size: int,
    stride: int,
    overlap: int,
    closed_loop: str = "R+P",
) -> list[list[int]]:
    """Replicate A1111's uniform context indices for R-P, R+P or A.

    R+P in the extension primarily affects prompt continuity; its context
    positions use the same end-boundary rule as R-P. Closed-loop A differs.
    'N' has special behavior and is intentionally unsupported here.
    """
    if frame_count < 1 or batch_size < 1 or batch_size > frame_count:
        raise ValueError("frame_count must be >= batch_size >= 1")
    if not (0 <= overlap < batch_size):
        raise ValueError("overlap must be in [0, batch_size)")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if closed_loop not in ("R-P", "R+P", "A"):
        raise ValueError("Use R-P, R+P or A; A1111 'N' uses a separate window rule")

    if frame_count <= batch_size:
        return [list(range(frame_count))]

    max_stride = min(stride, math.ceil(math.log2(frame_count / batch_size)) + 1)
    phase = ordered_halving(step)
    windows = []
    for stride_level in range(max_stride):
        context_step = 1 << stride_level
        pad = int(round(frame_count * phase))
        begin = int(phase * context_step) + pad
        end = frame_count + pad + (0 if closed_loop == "A" else -overlap)
        increment = batch_size * context_step - overlap
        if increment <= 0:
            raise ValueError("context step minus overlap must be positive")
        for j in range(begin, end, increment):
            windows.append(
                [(j + k * context_step) % frame_count for k in range(batch_size)]
            )

    # Detect mistakes in the window schedule before a costly GPU generation.
    seen = set(i for window in windows for i in window)
    if len(seen) != frame_count:
        missing = sorted(set(range(frame_count)) - seen)
        raise RuntimeError(f"Context scheduling failed: {len(missing)} frames uncovered: {missing[:15]}")
    return windows


def frame_condition_indices(batch_size: int, num_frames: int, context: list[int]) -> list[int]:
    """Flat positions for Diffusers embeddings repeated per frame.

    Diffusers 0.39's AnimateDiffPipeline uses
    prompt_embeds.repeat_interleave(repeats=num_frames, dim=0).
    For classifier-free guidance, those embeddings have shape
    [2*batch_size*num_frames, tokens, channels], with negative and
    positive prompts in separate blocks. We must take each temporal
    window from EVERY block, in the matching order.
    """
    if batch_size < 1 or num_frames < 1:
        raise ValueError("batch_size and num_frames must be positive")
    if not context or any(f < 0 or f >= num_frames for f in context):
        raise ValueError("context indices must be valid for the full video")
    return [batch * num_frames + frame
            for batch in range(batch_size) for frame in context]


class SlidingContextController:
    """Replace only UNet.forward; keep the original pipeline denoising algorithm.

    A1111 extension uses context-ordered assignment of UNet predictions;
    overlapping windows overwrite earlier predictions for shared frames.
    We match that assignment here rather than averaging (which would be
    a different algorithm).
    """

    def __init__(self, unet, frames: int, window: int, overlap: int, stride: int, loop: str, frame_conditioning=None):
        self.original_forward = unet.forward
        self.frame_conditioning = frame_conditioning
        self.frames = frames
        self.window = window
        self.overlap = overlap
        self.stride = stride
        self.loop = loop
        self.step = 0
        self.total_unet_calls = 0
        self.first_step_windows = len(uniform_contexts(0, frames, window, stride, overlap, loop))

    def forward(self, sample, timestep, *args, **kwargs):
        # sample: [batch (incl. CFG), channels, frames, latent_h, latent_w]
        import torch

        if sample.ndim != 5 or sample.shape[2] != self.frames:
            raise ValueError(
                f"Unexpected UNet sample shape {tuple(sample.shape)}. "
                f"Expected [batch,channels,{self.frames},height,width]."
            )

        windows = uniform_contexts(
            self.step, self.frames, self.window, self.stride, self.overlap, self.loop
        )
        full_noise = torch.empty_like(sample)
        last_output = None
        # In Diffusers 0.39, prompt embeddings are repeated across ALL
        # frames before UNet.forward is called. Without slicing them too,
        # AttnProcessor2_0 reshapes the 13-frame query using the 48-frame
        # conditioning batch; this caused the 975 vs 3600 tensor error.
        if args:
            raise RuntimeError(
                "Expected conditioning as keyword args. Diffusers pipeline API changed; "
                "update the context wrapper before retrying."
            )
        embeddings = (
            self.frame_conditioning
            if self.frame_conditioning is not None
            else kwargs.get("encoder_hidden_states")
        )
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            raise RuntimeError("Expected a 3D encoder_hidden_states tensor from Diffusers AnimateDiff.")
        expected = sample.shape[0] * self.frames
        if embeddings.shape[0] != expected:
            raise RuntimeError(
                f"Text-conditioning batch mismatch: received {embeddings.shape[0]}, "
                f"expected {expected} ({sample.shape[0]} CFG batches * {self.frames} frames)."
            )

        for indices in windows:
            idx = torch.tensor(indices, device=sample.device, dtype=torch.long)
            small_sample = sample.index_select(2, idx)
            condition_indices = torch.tensor(
                frame_condition_indices(sample.shape[0], self.frames, indices),
                device=embeddings.device, dtype=torch.long,
            )
            small_kwargs = kwargs.copy()
            small_kwargs["encoder_hidden_states"] = embeddings.index_select(
                0, condition_indices
            )
            output = self.original_forward(small_sample, timestep, **small_kwargs)
            if not hasattr(output, "sample"):
                raise RuntimeError("Unexpected UNet output; expected .sample")
            if output.sample.shape != small_sample.shape:
                raise RuntimeError(
                    f"UNet temporal output has shape {tuple(output.sample.shape)} "
                    f"but expected {tuple(small_sample.shape)}."
                )
            # A1111-like overwrite, in window order. Not a frame-blend.
            full_noise.index_copy_(2, idx, output.sample)
            last_output = output
            self.total_unet_calls += 1
        self.step += 1
        if self.step == 1:
            print(
                f"Context pass: {len(windows)} overlapping windows per denoising step, "
                f"{self.window} frames each, overlap {self.overlap}; all {self.frames} frames covered.",
                flush=True,
            )
        return last_output.__class__(sample=full_noise)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--motion", required=True, type=Path)
    p.add_argument("--preset", choices=["city", "windows"], default="city")
    p.add_argument("--prompt", default=None, help="Override preset prompt. A1111 weighting is not exact.")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--window", type=int, default=13)
    p.add_argument("--overlap", type=int, default=6)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--loop", choices=["R-P", "R+P", "A"], default="R+P")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=611)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--prompt-travel", action="store_true", help="Experimental changing CLIP conditioning per frame, no image crossfade.")
    p.add_argument("--travel-seed", type=int, default=9191)
    p.add_argument("--travel-min", type=int, default=50, help="Minimum frames between semantic prompt changes.")
    p.add_argument("--travel-max", type=int, default=200, help="Maximum frames between semantic prompt changes.")
    p.add_argument("--travel-transition", type=int, default=16, help="Frames blending between CLIP prompt embeddings.")
    p.add_argument("--out", type=Path, default=Path("benchmarks/results/animatediff-context"))
    return p.parse_args()


def main():
    a = parse_args()
    if not a.checkpoint.is_file() or not a.motion.is_file():
        raise FileNotFoundError("Choose the original CyberRealistic checkpoint and V3 motion file.")
    if any(x < 1 for x in (a.frames, a.width, a.height, a.steps, a.fps)):
        raise ValueError("Frames, width, height, steps and FPS must be positive")
    if a.width % 8 or a.height % 8:
        raise ValueError("Width and height must be divisible by 8")
    if a.frames > 200:
        raise ValueError("Limit this initial validation to 200 frames")
    if a.window < 2 or a.window > a.frames:
        raise ValueError("Window size must be in [2, frames]")
    if a.guidance < 0:
        raise ValueError("Guidance must be nonnegative")
    windows = uniform_contexts(0, a.frames, a.window, a.stride, a.overlap, a.loop)
    import torch
    import imageio.v2 as imageio
    import diffusers

    a.out.mkdir(parents=True, exist_ok=True)
    prompt = a.prompt if a.prompt is not None else (CITY_PROMPT if a.preset == "city" else WINDOWS_PROMPT)
    print(f"REFERENCE PRESET: {a.preset}")
    print(f"PROMPT: {prompt}")
    print(f"NEGATIVE: {a.negative}")
    print(
        f"TEST: {a.frames} frames, {a.width}x{a.height}, {a.steps} steps, "
        f"CFG {a.guidance}, seed {a.seed}, {a.window} context, "
        f"{a.overlap} overlap, stride {a.stride}, loop {a.loop}"
    )
    print(
        f"CONTEXT: {len(windows)} windows per denoising step "
        f"(approximately {len(windows) * a.steps} UNet calls total)."
    )
    print("Not real time. This first test is about temporal fidelity.")
    print("NOTE: A1111 weighted prompt syntax and exact seed/sampler implementation may differ.")

    # Shared model loader supports a.vae_slicing; set it explicitly here.
    a.vae_slicing = True
    pipe = prepare_model(a)
    travel_plan = None
    frame_conditioning = None
    if a.prompt_travel:
        # A deterministic plan changes text embeddings but NEVER blends image frames.
        # In the causal student this small state machine will run continuously,
        # with encoded prompts cached on GPU. This offline teacher is just a test.
        travel_plan = build_prompt_plan(
            frames=a.frames, seed=a.travel_seed, first_prompt=prompt,
            min_interval=a.travel_min, max_interval=a.travel_max,
            transition_frames=a.travel_transition,
        )
        for seg in travel_plan["segments"]:
            print(
                f"PROMPT TRAVEL frames {seg['start']}-{seg['end_exclusive'] - 1}: "
                f"{travel_plan['prompts'][seg['prompt_id']]}"
            )
        frame_conditioning = encode_frame_conditioning(pipe, travel_plan, a.negative)
        expected = 2 * a.frames
        if frame_conditioning.shape[0] != expected:
            raise RuntimeError(
                f"Prompt travel produced {frame_conditioning.shape[0]} conditionings "
                f"but CFG expects {expected}"
            )
        print(
            f"TRAVEL ACTIVE: {len(travel_plan['prompts'])} prompt states; "
            f"changing semantic text conditioning in the SAME evolving latent video."
        )
    control = SlidingContextController(
        pipe.unet, a.frames, a.window, a.overlap, a.stride, a.loop,
        frame_conditioning=frame_conditioning,
    )
    orig = pipe.unet.forward
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        pipe.unet.forward = control.forward
        generator = torch.Generator("cuda").manual_seed(a.seed)
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=a.negative or None,
                width=a.width,
                height=a.height,
                num_frames=a.frames,
                num_inference_steps=a.steps,
                guidance_scale=a.guidance,
                generator=generator,
                decode_chunk_size=4,
                output_type="pil",
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        memory = torch.cuda.max_memory_allocated() / 1024**3
    finally:
        pipe.unet.forward = orig

    stem = (
        f"context_{a.preset}_{a.width}x{a.height}_{a.frames}f_"
        f"{a.steps}s_seed{a.seed}_w{a.window}_o{a.overlap}"
        + (f"_travel_seed{a.travel_seed}" if a.prompt_travel else "_static")
    )
    vid = a.out / f"{stem}.mp4"
    if travel_plan is not None:
        plan_file = a.out / f"{stem}_plan.json"
        with plan_file.open("w", encoding="utf-8") as f:
            json.dump(travel_plan, f, indent=2)
        print("Saved exact deterministic prompt travel plan:", plan_file)
    print("Encoding MP4:", vid)
    imageio.mimsave(
        str(vid), output.frames[0], fps=a.fps, codec="libx264",
        macro_block_size=1, quality=8,
    )
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "single-global-latent-video with per-step overlapping temporal windows",
        "checkpoint": str(a.checkpoint),
        "motion": str(a.motion),
        "prompt": prompt,
        "negative": a.negative,
        "frames": a.frames,
        "width": a.width,
        "height": a.height,
        "steps": a.steps,
        "cfg": a.guidance,
        "seed": a.seed,
        "context_window": a.window,
        "context_overlap": a.overlap,
        "context_stride": a.stride,
        "context_loop": a.loop,
        "prompt_travel": a.prompt_travel,
        "prompt_travel_seed": a.travel_seed if a.prompt_travel else None,
        "prompt_travel_plan": str(plan_file) if travel_plan else None,
        "prompt_travel_prompts": travel_plan["prompts"] if travel_plan else None,
        "windows_per_step": len(windows),
        "unet_forward_calls": control.total_unet_calls,
        "full_pipeline_seconds": round(seconds, 3),
        "batch_generated_fps": round(a.frames / seconds, 4),
        "peak_allocated_gib": round(memory, 3),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "video": str(vid),
        "warning": "Not a real-time/streaming result; approximate A1111 parity only",
    }
    with (a.out / f"{stem}_results.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with (a.out / f"{stem}_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(
        f"RESULT: {a.frames} frames, {seconds:.2f}s end-to-end generation+decode, "
        f"{row['batch_generated_fps']:.3f} generated FPS, peak {memory:.2f} GiB,"
        f" {control.total_unet_calls} total context forward calls."
    )
    print("Output:", vid)
    print("Results:", a.out / f"{stem}_results.json")
    print("Review changes throughout whole clip. This is a fidelity test, not real-time playback.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        sys.exit(f"RTNVE long-context test failed: {exc}")
