"""RTNVE experimental prompt-travel schedule (CPU-only; no PyTorch dependency).

Creates a seeded, reproducible set of semantic changes for a long neural
sequence. Changes are interpolated in TEXT-CONDITIONING embedding space by
the AnimateDiff teacher adapter, never by crossfading rendered video frames.

This is a research controller, not a trained causal video model.
"""
from __future__ import annotations

import random
from typing import Any

# Curated semantic components intentionally shift the world instead of just
# repainting one street. No text-generation API or recurring server required.
SCENES = (
    "a crowded underground railway platform at dawn",
    "an endless corridor of worn graffiti-covered doors",
    "a high angle view of a sprawling city junction",
    "an abandoned department store with enormous escalators",
    "a concert hall filled with hundreds of empty chairs",
    "a motorway threading through a dense botanical greenhouse",
    "an old public swimming pool under a glass ceiling",
    "an airport terminal flooded by shallow reflective water",
    "a ruined amusement park in midday sunlight",
    "a museum of enormous torn paintings and misplaced furniture",
    "a tower block filled with miniature illuminated rooms",
    "a library whose bookshelves extend into the street",
    "a railway station built inside a giant hollow statue",
    "a concrete plaza below thousands of suspended windows",
    "a vast public square covered in newspaper",
    "an impossible apartment building without a roof",
    "a cinema overtaken by staircases and houseplants",
    "a procession of cars travelling through a cathedral",
    "a city seen through hundreds of surveillance cameras",
    "an open-air market inside a decaying office building",
    "a harbour where buildings rise directly from the water",
    "a row of suburban houses merging into an enormous theatre",
)
ACTORS = (
    "strange pedestrians and cars", "tiny figures pulling enormous objects",
    "crowds of people walking backwards", "people made from scraps of paper",
    "unfamiliar animals wandering through the architecture",
    "empty vehicles moving with no drivers", "people carrying oversized doors",
    "houses furnished with human-sized mechanical parts",
    "miniature crowds reflected in shattered glass",
    "impossible machines and forgotten furniture",
)
TRANSFORMATIONS = (
    "shattering into living foliage and forming again",
    "ripping apart as hidden rooms grow out of the surfaces",
    "folding through one another into an impossible new structure",
    "melting into clear liquid glass then becoming solid",
    "peeling open to reveal other places underneath",
    "turning into architectural drawings that come to life",
    "breaking into thousands of connected moving fragments",
    "growing twisting vines through moving mechanical parts",
    "turning into piles of handwritten letters and dust",
    "swapping places in unpredictable physical rearrangements",
    "transforming into doorways that open into other scenes",
    "splintering into furniture and reforming as architecture",
)
MATERIALS = (
    "rain on cracked concrete", "rough plywood and faded paint",
    "dirty glass and green moss", "old carved stone and wet paper",
    "rusted steel and translucent plastic", "weathered brick and leaves",
    "torn posters and rough plaster", "reflective water and broken ceramic",
)
CAMERAS = (
    "elevated documentary camera", "fixed high angle CCTV camera",
    "wide architectural view", "unusual distant observational viewpoint",
)


def invent_prompt(rng: random.Random, last: str, base_style: str) -> str:
    """Choose a nonrepeated compositional prompt from many combinations."""
    for _ in range(10):
        p = (
            f"{rng.choice(CAMERAS)}, {rng.choice(SCENES)}, "
            f"{rng.choice(ACTORS)}, {rng.choice(TRANSFORMATIONS)}, "
            f"{rng.choice(MATERIALS)}, {base_style}"
        )
        if p != last:
            return p
    raise RuntimeError("Failed to generate a distinct prompt")


def transition_weight(position: int, count: int) -> float:
    """Continuous cosine-like ease without importing numpy."""
    if count <= 0:
        return 1.0
    x = max(0.0, min(1.0, (position + 1) / count))
    # smoothstep; meaningful values near endpoints, easy to test deterministically
    return x * x * (3 - 2 * x)


def build_prompt_plan(
    frames: int,
    seed: int,
    first_prompt: str,
    min_interval: int = 50,
    max_interval: int = 200,
    transition_frames: int = 28,
    base_style: str = "surreal documentary footage, detailed, cohesive scene",
) -> dict[str, Any]:
    """Return prompt bank plus a per-frame semantic-conditioning schedule.

    Segments are randomly 50, 100 or 200 frames when allowed, otherwise
    randomly chosen within a custom interval (for short validation tests).
    Transitions occupy the end of each segment, *not* a hard image cut.
    The plan can extend to arbitrary frame counts, and seed ensures replay.
    """
    if not 1 <= frames <= 1_000_000:
        raise ValueError("frames must be between 1 and 1,000,000")
    if not first_prompt.strip():
        raise ValueError("first_prompt cannot be empty")
    if min_interval < 2 or max_interval < min_interval:
        raise ValueError("invalid prompt interval")
    if transition_frames < 0 or transition_frames >= min_interval:
        raise ValueError("transition_frames must be smaller than min_interval")

    rng = random.Random(seed)
    prompts = [first_prompt]
    records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    current_id = 0
    start = 0
    while start < frames:
        options = [n for n in (50, 100, 200) if min_interval <= n <= max_interval]
        duration = rng.choice(options) if options else rng.randint(min_interval, max_interval)
        end = min(start + duration, frames)
        next_id: int | None = None
        if end < frames:
            prompts.append(invent_prompt(rng, prompts[current_id], base_style))
            next_id = len(prompts) - 1
        fade = min(transition_frames, duration - 1) if next_id is not None else 0
        segments.append({
            "start": start, "end_exclusive": end, "prompt_id": current_id,
            "next_prompt_id": next_id, "transition_start": end - fade,
        })
        for frame in range(start, end):
            local_transition = frame - (end - fade)
            if next_id is not None and fade > 0 and local_transition >= 0:
                alpha = transition_weight(local_transition, fade)
            else:
                alpha = 0.0
            records.append({
                "frame": frame,
                "from_prompt": current_id,
                "to_prompt": next_id if next_id is not None else current_id,
                "alpha": round(alpha, 6),
            })
        start = end
        if next_id is not None:
            current_id = next_id

    if len(records) != frames:
        raise AssertionError("Prompt plan did not cover every frame")
    return {
        "version": 1,
        "generator": "seeded local semantic grammar + smooth CLIP embedding mix",
        "seed": seed,
        "frames": frames,
        "min_interval": min_interval,
        "max_interval": max_interval,
        "transition_frames": transition_frames,
        "prompts": prompts,
        "segments": segments,
        "frame_schedule": records,
        "warning": "Embedding interpolation does not guarantee visual morphing or causal continuity.",
    }


def encode_frame_conditioning(pipe, plan: dict, negative: str):
    """Encode the plan once, constructing a [2*frames,77,768] CFG bank.

    Important: with the v0.39.0 AnimateDiff pipeline, the normal call uses
    one prompt and repeats it once per frame before invoking the UNet.
    SlidingContextController swaps that default conditioning for this bank
    and extracts the same frame indices in every temporal window.

    CLIP is called once per unique prompt, not every generated frame.
    """
    import torch

    prompts = plan["prompts"]
    if not prompts:
        raise ValueError("Empty prompt bank")
    device = next(pipe.unet.parameters()).device
    with torch.inference_mode():
        positive, negative_bank = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[negative] * len(prompts),
        )
        if positive.ndim != 3 or negative_bank.ndim != 3:
            raise ValueError("Unexpected CLIP conditioning dimensions")
        if positive.shape != negative_bank.shape or positive.shape[0] != len(prompts):
            raise ValueError("Unexpected CLIP prompt bank size")

        conditioning = []
        for f in plan["frame_schedule"]:
            a = f["alpha"]
            left = positive[f["from_prompt"]]
            right = positive[f["to_prompt"]]
            conditioning.append(left.lerp(right, a) if a else left)
        positive_frames = torch.stack(conditioning, dim=0)
        negative_frames = negative_bank[0].unsqueeze(0).expand_as(positive_frames)
        # Must match the main Diffusers CFG order [unconditional, conditional].
        return torch.cat([negative_frames, positive_frames], dim=0).contiguous()
