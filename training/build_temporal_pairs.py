#!/usr/bin/env python3
"""Build paired consecutive-image training data for one-step neural feedback.

Inputs: dataset/sequence/<video>/frame_000001.jpg ... from prepare_video_frames.py

Outputs: dataset/paired/train_A,train_B,test_A,test_B and two *_prompts.json
This is a supervised teacher->student temporal learning experiment, NOT a
working real-time art model or a guarantee against blurry autoregressive drift.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def list_clips(root: Path) -> list[tuple[str, list[Path]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input sequence directory not found: {root}")
    clips = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(p for p in folder.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if len(imgs) >= 3:
            clips.append((folder.name, imgs))
    if not clips:
        raise ValueError("No clip folders with at least 3 frames found.")
    return clips


def choose_split(clips: list[tuple[str, list[Path]]], holdout: float) -> dict[str, int]:
    """Return index where the test sequence starts for each clip.

    With 3+ clips, hold out one entire clip to reduce temporal leakage.
    With fewer, hold out a contiguous suffix, not interleaved adjacent frames.
    """
    if len(clips) >= 3:
        return {name: (0 if i == len(clips) - 1 else len(images))
                for i, (name, images) in enumerate(clips)}
    return {name: min(len(images) - 2,
                      max(2, int(len(images) * (1 - holdout))))
            for name, images in clips}


def build(args: argparse.Namespace) -> None:
    source = args.sequence.resolve()
    root = args.out.resolve()
    clips = list_clips(source)
    divisions = choose_split(clips, args.holdout)
    if root.is_relative_to(source):
        raise ValueError("Output folder must be outside sequence input directory.")
    if root.exists() and not args.overwrite:
        if any(root.iterdir()):
            raise FileExistsError(f"Output {root} is not empty; pass --overwrite to replace its pair files.")
    if args.overwrite and root.exists():
        # Only clean generated paired folders, never input sequences.
        for name in ("train_A", "train_B", "test_A", "test_B"):
            shutil.rmtree(root / name, ignore_errors=True)
        for name in ("train_prompts.json", "test_prompts.json", "pair_manifest.json"):
            (root / name).unlink(missing_ok=True)
    for split in ("train", "test"):
        for suffix in ("A", "B"):
            (root / f"{split}_{suffix}").mkdir(parents=True, exist_ok=True)

    pairs = {"train": {}, "test": {}}
    manifest = {"clips": [], "offset": args.offset, "prompts": args.prompt,
                "note": "A = earlier neural teacher frame; B = later frame; sequence source preserved."}
    rng = random.Random(args.seed)
    for clip_id, (name, frames) in enumerate(clips):
        cut = divisions[name]
        possibilities = []
        for i in range(len(frames) - args.offset):
            j = i + args.offset
            # One whole clip is exclusively holdout if 3+ clips.
            # When splitting a single/two clip, don't use pairs crossing boundary.
            split = "train" if j < cut else ("test" if i >= cut else None)
            if split:
                possibilities.append((i, j, split))
        manifest["clips"].append({"name": name, "frames": len(frames), "cut_at": cut,
                                   "candidate_pairs": len(possibilities)})
        rng.shuffle(possibilities)
        for i, j, split in possibilities:
            # Clip/name prefix prevents collisions; training pairs stay within a clip.
            fname = f"clip{clip_id:03d}_{i:06d}.jpg"
            shutil.copy2(frames[i], root / f"{split}_A" / fname)
            shutil.copy2(frames[j], root / f"{split}_B" / fname)
            pairs[split][fname] = args.prompt

    for split in ("train", "test"):
        with (root / f"{split}_prompts.json").open("w", encoding="utf-8") as f:
            json.dump(pairs[split], f, indent=2, ensure_ascii=False)
        print(f"{split}: {len(pairs[split])} paired transitions")
    with (root / "pair_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if not pairs["train"] or not pairs["test"]:
        raise RuntimeError("Need both train and test pairs; add clips or lower --offset.")
    print("Prepared consecutive-frame pairs. Next: train an image-conditioned neural model.")
    print("Warning: repeated recursive outputs may freeze or collapse; test long rollouts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, default=Path("dataset/sequence"))
    parser.add_argument("--out", type=Path, default=Path("dataset/paired"))
    parser.add_argument("--offset", type=int, default=1,
                        help="Target frame index minus input frame index (>=1).")
    parser.add_argument("--holdout", type=float, default=0.18)
    parser.add_argument("--prompt", default="a dreamlike scene continually transforming, surreal organic metamorphosis, intricate new details")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.offset < 1 or not 0 < args.holdout < 0.5:
        parser.error("--offset must be >=1 and --holdout must be between 0 and 0.5")
    build(args)


if __name__ == "__main__":
    main()
