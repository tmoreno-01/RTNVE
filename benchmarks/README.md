# LifeAfterLogic — AnimateDiff V3 acceleration benchmark

This is a real neural inference benchmark, **not a browser or realtime-streaming engine**.

It uses your original SD1.5 checkpoint plus V3 motion-module .ckpt
in an independent Diffusers environment, without changing the A1111 setup.
It measures end-to-end time, actual generated FPS, and peak GPU memory
for 24, 12, 8 and 4 denoising steps. It saves real clips for comparison.

Important:
- A1111's sliding context isn't reproduced; the pipeline may not render
  pixel-identically to your originals.
- The approximated DPM++ 2M SDE exponential scheduler isn't guaranteed
  identical to A1111.
- Four-step inference without distillation is likely to degrade quality.
  This is an honest speed/quality test, not proof of realtime generation.
- Browser inference may be substantially slower than CUDA inference.

## Install in a separate Python environment (Windows, RTX 3090)

From a local clone of tmoreno-01/experimentation:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip

Install CUDA-enabled PyTorch using https://pytorch.org/get-started/locally
for your current driver and OS, then:

    pip install "diffusers>=0.30" "transformers<5" accelerate safetensors imageio imageio-ffmpeg pillow
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

CUDA must be True. Do NOT install this environment in A1111's venv.

## Run on your original files

Replace the placeholders with your actual local checkpoint and motion file:

    python benchmarks/animatediff_v3.py --checkpoint "Z:\path\to\CyberRealistic_V9_FP32.safetensors" --motion "Z:\path\to\v3_sd15_mm.ckpt" --width 512 --height 288 --frames 16 --steps 24 12 8 4 --warmup

To try higher resolution, use width 768 height 432. For 720p, try width
1280 height 720 AFTER measuring VRAM. This may OOM on a 24GB RTX3090,
especially with CFG. The first run may also download SD model
configuration from Hugging Face.

Saved results: benchmarks/results/animatediff-v3/results.csv,
results.json, and generated MP4 previews.

Note: the 10 FPS setting for exported video PLAYBACK is not the actual
generated FPS. The CSV clearly separates them.

## Product target and next step

- Browser WebGPU, no recurring cloud GPU bill.
- Single evolving image with genuinely learned temporal generation.
- Milestone 1: aim for 10 genuinely generated FPS at useful quality on
  a high-end visitor GPU; scale to 1280x720 if achievable.
- Milestone 2: strive toward 24–30 FPS and lower-spec browsers.
- A student model distilled from AnimateDiff-like teacher sequences
  must be vastly cheaper than full AnimateDiff, retain temporal state,
  and generate new content rather than replay video or shader morphs.

This benchmark is the first empirical step in finding out how much
compute we need to remove. We have not run it on the 3090 yet.
