#!/usr/bin/env python3
"""Extract full-scene AnimateDiff frames without breaking the scene into portrait tiles.

Requires Python 3.10+ and FFmpeg/ffprobe installed on PATH.
Usage:
  python prepare_video_frames.py --input "Z:\\ART\\renders\\long_underground.mp4" --out dataset
  python prepare_video_frames.py --input clip1.mp4 --input clip2.mp4 --out dataset --still-fps 1 --motion-fps 5
  python prepare_video_frames.py --input https://media.tyronemoreno.com/motion/upresunderground-web.mp4 --out dataset

Produces:
  dataset/stills/<clip>/frame_000001.jpg ...    for image-GAN training
  dataset/sequence/<clip>/frame_000001.jpg ...  ordered for future temporal training
  dataset/manifest.json                         input provenance and extraction settings

Important: source videos are never changed. Consecutive frames are NOT treated
as independent evidence of new visual diversity in quality evaluation.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone


def tool(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Missing {name}. Install FFmpeg and make {name} available on PATH.")
    return found


def slug(path):
    source = urllib.parse.urlparse(str(path)).path if str(path).startswith(("http://", "https://")) else str(path)
    return re.sub(r"[^a-z0-9_-]", "_", Path(source).stem.lower())[:72] or "source"


def run(cmd):
    print("Running:", " ".join(str(p) for p in cmd), flush=True)
    subprocess.run([str(p) for p in cmd], check=True)


def probe(ffprobe, source):
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(source)],
        capture_output=True, text=True, check=True)
    result = json.loads(proc.stdout)
    if not result.get("streams"):
        raise RuntimeError(f"No video stream: {source}")
    return result


def acquire(source, cache):
    if not source.startswith(("http://", "https://")):
        p = Path(source).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Video not found: {p}")
        return p
    # HTTPS download once, not once per frame. Prefer your local originals
    # for quality; published web encodes are secondary training sources.
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS remote videos are supported.")
    filename = slug(source) + Path(parsed.path).suffix.lower()
    dest = cache / filename
    if dest.is_file() and dest.stat().st_size > 1024:
        print("Using cached download:", dest)
        return dest
    print("Downloading:", source)
    request = urllib.request.Request(source, headers={"User-Agent": "AI-Lab-dataset-prep/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        if tmp.stat().st_size < 1024:
            raise RuntimeError("Download appears empty.")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def extract(ffmpeg, source, dest, fps, w, h, quality):
    dest.mkdir(parents=True, exist_ok=True)
    # Image dimensions must be divisible by 32 for typical GAN generators.
    # One entire source frame becomes ONE training image — never a tile.
    graph = (f"fps={fps},"
             f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
             f"crop={w}:{h},setsar=1")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", source, "-map", "0:v:0", "-an", "-vf", graph,
         "-q:v", quality, str(dest / "frame_%06d.jpg")])
    return len(list(dest.glob("frame_*.jpg")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="Local MP4 path or HTTPS URL; repeatable.")
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--still-fps", type=float, default=1.0)
    parser.add_argument("--motion-fps", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=2, help="FFmpeg JPEG quality (2 is high).")
    a = parser.parse_args()
    if a.width % 32 or a.height % 32 or a.width <= 0 or a.height <= 0:
        parser.error("width and height must be positive multiples of 32.")
    if a.still_fps <= 0 or a.motion_fps <= 0:
        parser.error("frame rates must be positive.")
    if not 2 <= a.quality <= 31:
        parser.error("quality must be 2..31.")
    ffmpeg, ffprobe = tool("ffmpeg"), tool("ffprobe")
    out = a.out.expanduser().resolve()
    cache = out / "video_cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "target": {"width": a.width, "height": a.height},
                "still_fps": a.still_fps, "motion_fps": a.motion_fps,
                "clips": []}
    used = set()
    for item in a.input:
        name = slug(item)
        if name in used:
            raise ValueError(f"Duplicate filename stem {name}; rename the clips.")
        used.add(name)
        path = acquire(item, cache)
        info = probe(ffprobe, path)
        stills = out / "stills" / name
        sequence = out / "sequence" / name
        n_stills = extract(ffmpeg, path, stills, a.still_fps, a.width, a.height, a.quality)
        n_motion = extract(ffmpeg, path, sequence, a.motion_fps, a.width, a.height, a.quality)
        manifest["clips"].append({"source": item, "name": name, "duration_sec": info.get("format", {}).get("duration"),
                                  "original": info["streams"][0], "stills": n_stills, "motion_frames": n_motion})
    with (out / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Done. Wrote {out / 'manifest.json'}.")
    print("WARNING: A small or repetitive frame set can be memorised by the model.")
    print("Do not mistake GAN latent interpolation for learned temporal dynamics.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        sys.exit(f"Error: {exc}")
