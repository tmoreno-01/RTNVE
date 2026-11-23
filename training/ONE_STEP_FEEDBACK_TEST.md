# Experiment: one-step image-conditioned neural feedback

This is the most direct research route to the **hallucinatory metamorphosis**
of AnimateDiff, without requiring the final system to reproduce a specific
train station, character, or environment.

The goal is a neural update function roughly like:

    next_frame, next_state = network(previous_frame,
                                      hidden_temporal_state,
                                      noise,
                                      optional_prompt)

The *network*, not a crossfade shader, must make the transformations.
An off-the-shelf one-step image-to-image network only supplies the first
step. We must test and train for recursive rollouts to avoid collapse.

## Upstream project

https://github.com/GaParmar/img2img-turbo

The authors release a one-step conditional diffusion/translation model,
**pix2pix-Turbo**, and code to train it on paired images.

The published benchmarks are native Python GPU inference (not browser):
about 0.29 seconds per 512x512 frame on A6000 and 0.11 on A100. Our
RTX 3090 and WebGPU browser performance must be measured independently.

## Make a small pilot dataset (offline; don't touch A1111)

Windows PowerShell in the `experimentation` checkout, with FFmpeg installed:

```powershell
python training/prepare_video_frames.py --input "C:\path\to\original_video.mp4" --out dataset --width 512 --height 512 --still-fps 1 --motion-fps 10
python training/build_temporal_pairs.py --sequence dataset/sequence --out dataset/paired --offset 1 --overwrite
```

The first script creates ordered *whole-image* source frames; the second
makes `train_A` (frame t), `train_B` (frame t+1) and matching validation
pairs. Training never consumes a grid of unrelated faces.

For a serious model, include multiple videos, initial conditions and
compositions. A single clip is only a functional pipeline test.

## Run the upstream trainer (first proof only)

Use a separate Python/Conda environment from your Stable Diffusion
installations, following the official project requirements. Example commands:

```bash
git clone https://github.com/GaParmar/img2img-turbo.git
cd img2img-turbo
conda env create -f environment.yaml
conda activate img2img-turbo
accelerate config
accelerate launch src/train_pix2pix_turbo.py \
  --pretrained_model_name_or_path="stabilityai/sd-turbo" \
  --output_dir="output/ai_lab" \
  --dataset_folder="/path/to/experimentation/dataset/paired" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=500 \
  --mixed_precision=fp16 \
  --enable_xformers_memory_efficient_attention \
  --report_to tensorboard
```

This is a small **exploratory** training run, not guaranteed to fit or
produce good images on 3090 without dependency/performance tuning.
Use the upstream documentation for the actual environment requirements.

The base trainer uses supervised per-pair prediction. It doesn't yet have
a *self-forcing rollout objective*: that is the next code change if
short sequences look promising. Simply feeding its predictions back for
hundreds of frames is likely to accumulate errors or converge to a static
image. Test for this, don't hide it.

## Why browser portability is not done yet

The browser can already run neural models in WebGPU. But the upstream
Pix2Pix-Turbo network is PyTorch, not a WebGPU bundle. We need to:

- Export and validate ONNX or translate operations to WGSL.
- Ensure model weights are small enough for consumer GPUs.
- Keep previous hidden state on the GPU across frames.
- Benchmark *complete* encode→predict→decode time.
- Handle unsupported/mobile hardware honestly.
- Distil the model further if browser throughput is not real time.

This is a **development pipeline**, not a claim of a finished neural art engine.
We won't publish it as a live generative artwork until it actually works.
