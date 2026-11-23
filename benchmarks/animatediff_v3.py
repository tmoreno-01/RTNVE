#!/usr/bin/env python3
"""LifeAfterLogic AI Lab — actual AnimateDiff V3 GPU benchmark.

Research tool, NOT a realtime browser engine.

Accepts the user's original SD1.5 checkpoint and v3_sd15_mm.ckpt from A1111.
Builds a standalone Diffusers AnimateDiffPipeline so the working A1111
installations remain untouched. Measures full generation + VAE decoding,
peak allocated VRAM and generated frames per second for multiple step counts.

The Diffusers scheduler approximately matches A1111 DPM++ 2M SDE/exponential,
but output is NOT pixel-identical. A1111's sliding temporal context settings
are not replicated. This first isolated benchmark uses a single short
temporal window (default 16 frames), so it does not demonstrate streaming.

Install in an isolated environment, see benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path, help="Original SD1.5 checkpoint, .safetensors or .ckpt")
    p.add_argument("--motion", required=True, type=Path, help="AnimateDiff V3 motion file (v3_sd15_mm.ckpt)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--steps", nargs="+", type=int, default=[20, 12, 8, 4])
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--fps", type=int, default=10, help="Playback FPS for saved preview, not measured generation FPS")
    p.add_argument("--prompt", default="strange looking people, cars driving,( High angle view of people walking around new york city, cctv:1.2),  shattering, ripping apart, destruction, leaves, plants growing, tearing")
    p.add_argument("--negative", default="")
    p.add_argument("--out", type=Path, default=Path("benchmarks/results/animatediff-v3"))
    p.add_argument("--warmup", action="store_true", help="Run an additional short warmup before timed measurements.")
    p.add_argument("--vae-slicing", action="store_true", help="Reduce decoding memory when supported.")
    return p.parse_args()


def validate(a: argparse.Namespace) -> None:
    for label in ("checkpoint", "motion"):
        path = getattr(a, label)
        if not path.is_file():
            raise FileNotFoundError(f"Missing local {label}: {path}")
    if any(v <= 0 for v in (a.width, a.height, a.frames, a.fps)):
        raise ValueError("Width, height, frames and FPS must be positive.")
    if a.width % 8 or a.height % 8:
        raise ValueError("Width and height must be divisible by 8.")
    if a.frames > 32:
        raise ValueError("This baseline is for small single-window tests (<=32 frames), not endless playback.")
    if any(v < 1 or v > 100 for v in a.steps):
        raise ValueError("Every step count must be between 1 and 100.")


def prepare_model(a):
    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Run on your RTX 3090, not CPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    print("Reading motion adapter:", a.motion)
    adapter = MotionAdapter.from_single_file(str(a.motion), torch_dtype=torch.float16)

    print("Reading SD1.5 checkpoint:", a.checkpoint)
    # Pipeline single-file loading is different from StableDiffusionPipeline.
    # Keep the motion adapter explicitly attached while converting the checkpoint.
    pipe = AnimateDiffPipeline.from_single_file(
        str(a.checkpoint), motion_adapter=adapter, torch_dtype=torch.float16,
        safety_checker=None, local_files_only=False,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
        use_exponential_sigmas=True,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    if a.vae_slicing:
        pipe.enable_vae_slicing()

    print("Motion module:", type(adapter).__name__)
    print("Scheduler:", type(pipe.scheduler).__name__, "SDE DPM++ / exponential sigmas")
    return pipe


def run_one(pipe, a, steps: int, *, save: bool) -> dict:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator("cuda").manual_seed(a.seed)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = pipe(
            prompt=a.prompt,
            negative_prompt=a.negative or None,
            num_frames=a.frames,
            width=a.width,
            height=a.height,
            num_inference_steps=steps,
            guidance_scale=a.guidance,
            generator=gen,
            output_type="pil",
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    max_gb = torch.cuda.max_memory_allocated() / (1024**3)
    max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

    row = {
        "steps": steps, "frames": a.frames, "width": a.width, "height": a.height,
        "seed": a.seed, "guidance": a.guidance,
        "seconds_full_pipeline": round(seconds, 3),
        "generated_fps": round(a.frames / max(seconds, 0.001), 4),
        "peak_allocated_gib": round(max_gb, 3),
        "peak_reserved_gib": round(max_reserved_gb, 3),
        "video_path": "",
        "notes": "One clip, no sliding window, no streaming; full generation+VAE decode.",
    }
    if save:
        import imageio.v2 as imageio
        frames = outputs.frames[0]
        path = a.out / f"v3_{a.width}x{a.height}_{a.frames}f_{steps}steps_seed{a.seed}.mp4"
        imageio.mimsave(str(path), frames, fps=a.fps, codec="libx264", macro_block_size=1,
                        quality=8, output_params=["-pix_fmt", "yuv420p"])
        row["video_path"] = str(path)
    print(f"RESULT {steps:>2} steps: {seconds:.1f}s, {row['generated_fps']:.3f} generated FPS, "
          f"peak allocated {max_gb:.2f} GiB")
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    a = parse_args()
    validate(a)
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import diffusers
        import imageio
    except ImportError as error:
        raise RuntimeError("Missing Python dependencies; see benchmarks/README.md") from error

    print("PROMPT:", a.prompt)
    print("NOTE: A1111 emphasis syntax such as (:1.2) is not interpreted identically by the plain Diffusers prompt encoder.")
    pipe = prepare_model(a)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(a.checkpoint.resolve()),
        "motion": str(a.motion.resolve()),
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "prompt": a.prompt, "negative": a.negative, "seed": a.seed,
        "width": a.width, "height": a.height, "frames": a.frames,
        "step_counts": a.steps,
        "warning": "NOT an exact A1111 reproduction or a realtime/browser benchmark",
    }
    with (a.out / "run.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if a.warmup:
        print("Warming GPU pipeline at 4 denoising steps (not timed)...")
        run_one(pipe, a, 4, save=False)

    results = []
    for steps in a.steps:
        try:
            results.append(run_one(pipe, a, steps, save=True))
        except torch.cuda.OutOfMemoryError as error:
            print(f"OOM at {steps} steps: {error}", file=sys.stderr)
            results.append({"steps": steps, "error": "CUDA OOM"})
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
            print("Unexpected failure; results produced so far will be saved.", file=sys.stderr)
            break

    with (a.out / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if any("generated_fps" in row for row in results):
        keys = ["steps", "frames", "width", "height", "seed", "guidance",
                "seconds_full_pipeline", "generated_fps", "peak_allocated_gib",
                "peak_reserved_gib", "video_path", "notes"]
        with (a.out / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows([{k: row.get(k, "") for k in keys} for row in results])
    print("Saved real benchmark results and preview videos to:", a.out)
    print("IMPORTANT: A1111 V3 uses temporal sliding-context logic not reproduced here.")
    print("A fast result from 4 steps is not sufficient unless its VISUAL QUALITY is comparable.")
    print("Nothing in this script implies the same speed on WebGPU or mobile browsers.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        sys.exit(f"Benchmark error: {error}")
