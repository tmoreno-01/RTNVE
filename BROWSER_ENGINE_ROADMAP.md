# RTNVE — from AnimateDiff to a real-time browser neural video engine

## The outcome

LifeAfterLogic should show one continually evolving hallucinated world. Not a video loop, an interpolation between static images, separate face tiles, shader-morphed pictures, or a cloud GPU stream hidden behind the website.

The target is **landscape 16:9**; artist-approved quality is more important than nominal resolution. Aim for **10 genuinely generated frames/s** first, **24–30 generated frames/s** later, with up to **1280×720 display output** where feasible. A lower native neural render resolution upscaled to 720p is acceptable *only if explicitly reported* and visually convincing.

The browser version runs on the *visitor's* GPU using WebGPU, not the artist's RTX 3090. The performance/installation version can use a local dedicated GPU. There may need to be separate weight sizes and quality presets.

**We have not achieved 10 FPS or 720p browser neural generation yet.**

## Why the A1111 test matters but is not the browser engine

- Your existing A1111 + CyberRealistic + AnimateDiff V3 produces the strange visuals worth pursuing.
- The 16-frame standalone CUDA benchmark verified that the checkpoint and V3 motion module run outside A1111.
- The result was not visually comparable with your 100-frame original clips. A1111's overlapping temporal context, prompt weighting, noise/scheduler and different settings could explain the gap.
- The new `benchmarks/animatediff_long_context.py` attempts a *100-frame unified latent denoising process*, with overlapping temporal windows rather than isolated short clips.
- This provides a **teacher/reference generator**. It does not turn into a fast browser engine simply by reducing sampling steps or converting the heavy model to ONNX.

We need to stop treating faster, low-quality 4-step inference as progress toward the finished artwork.

## The five actual technical stages

| Stage | What we build | What proves it works | Who/where |
|---|---|---|---|
| 1 — Teacher fidelity | 20-step V3 with faithful long temporal behaviour | 48–100-frame clips exhibiting genuine changes like the old A1111 clips | Local RTX 3090 (CUDA) |
| 2 — Model-size / latency budget | Define a small fixed-shape latent model and benchmark its **actual computation**, decode and state update | Full end-to-end latency broken down per operation, realistic WebGPU hardware tests | CUDA + Chrome/Edge WebGPU |
| 3 — Train a **causal student** | A much smaller neural model learns to predict/evolve an image using its previous neural state and controlled stochastic innovation | It invents unseen developments over 30+ seconds, not just copying frames or degenerating into noise | Local RTX 3090 |
| 4 — Export and test | Convert validated student weights to a WebGPU-supported graph, ideally ONNX, with a matching JS runtime | Same neural computation in browser within acceptable image error versus PyTorch, then measured browser inference FPS | Chrome/Edge with WebGPU |
| 5 — Production systems | Adaptive quality and latency limits, continuous playback, cacheable model assets, show controls | Robust 15-min exhibit and live-performance run; no always-on GPU server for the browser | Visitor GPU / local stage GPU |

**Stages 2 and 3 can develop in parallel:** don't spend weeks training a student architecture that WebGPU can't execute.

## The proposed student (research architecture, not a pretrained working model)

**One neural image, plus memory of its recent history.**

```text
previous synthesized latent frame (spatial features)
        + persistent temporal memory/state
        + fresh random innovation
        + optional controls (prompt/style intensity/audio/MIDI)
                       |
           SMALL NEURAL STUDENT
                       |
       new synthesized latent + new memory
                       |
       compact image decoder / optional upscaler
                       |
       display at ~10-30 generated frames/s
                       |
             feedback to next step
```

The *learned student* generates the new content. GPU shaders may perform only image presentation and colour conversion, not masquerade as neural evolution.

### Model contract to start with

- A fixed-shape latent representation at a lower internal resolution than 720p.
- Inputs: `previous_latent`, `memory`, `innovation`, and optionally a compact conditioning vector.
- Outputs: `next_latent` and `next_memory`; a separate lightweight decoder produces RGB frames.
- No requirement to load the entire SD 1.5 text encoder and 20-step UNet inside Chrome on every output frame.
- A modest fixed context window is more plausible in browser than processing all future 100 frames at once.
- The model must be able to produce novel detail and new visual structures, not merely deform/resample the previous image.

**This is a research hypothesis, not a guarantee it will retain AnimateDiff-quality imagery.** We will compare its outputs with original videos, and replace the architecture if it becomes repetitive, blurry or simply static.

### Training

1. Keep original AnimateDiff example videos as a **held-out benchmark** where practical.
2. Generate a diverse set of good, licensed teacher videos covering scenes, seeds, temporal settings, prompt variations, transformations, etc. *A couple of 10-second videos aren't enough to teach a production video model.*
3. Preserve original contiguous sequences and metadata; do not shuffle adjacent frames across train/test.
4. Begin with 8–16-frame causal prediction, teach temporal evolution, and **train on self-generated rollouts** so that mistakes don't compound catastrophically at inference. Simple supervised frame-pairs alone risk copying/blur.
5. Validate 30 seconds, then 5 minutes; measure dynamics, diversity, identity reappearance, collapse and recovery, output quality, VRAM and FPS.
6. Only promote student weights to the website after they have passed these tests.

Work on real, temporally trained diffusion distillation (e.g. ideas from Causal Forcing / Self Forcing) rather than assuming undistilled 4-step SD 1.5 works. Research is still active and heavyweight models may not fit the 3090 or browsers.

## Browser implementation requirements

- **ONNX Runtime Web with WebGPU EP** is our practical first route; WGSL/WebGPU custom kernels remain an alternative for unsupported operations.
- Browser downloads a model bundle over HTTPS, caches it, starts WebGPU inference and maintains temporal state between frames.
- Keep tensors GPU-resident where possible; avoid copying multi-megabyte frame state CPU↔GPU every step.
- Show honest diagnostics: native generated resolution, *neural* generation FPS, end-to-end per-frame latency, GPU compatibility and fallbacks.
- Upscale to 1280×720 display if the smaller native render is visually convincing; report upscaled vs native separately.
- Automatically pause when the artwork is offscreen. Quality switches adapt internal render sizes and inference cadence.
- Use static hosting for JS and model weights (e.g. GitHub Pages + object storage/CDN); **no inference bill**, but static bandwidth/storage isn't always free.

Official starting point: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

## What to do immediately

1. **Do not reinstall PyTorch.** Python, CUDA, V3 and the RTX 3090 already worked.
2. On the Windows RTNVE checkout, update three files: `benchmarks/animatediff_long_context.py`, `scripts/03_Run_Temporal.ps1`, `03_BENCHMARK_TEMPORAL.bat`.
3. Run `03_BENCHMARK_TEMPORAL.bat`. Defaults for new test: **landscape 640×360, CFG 6, 20 steps, context 13/overlap 6, seed 611**, and prompt presets. **Start with 48 frames** to test compatibility before spending time on 100 frames.
4. If successful, run 100 frames at the same size. Share the MP4 and `temporal_results.json`.
5. Based on quality, either fix teacher parity or move to recording quality teacher sequences and choosing a small latent student architecture.

### Red lines

- Never claim 'real-time' based on playback FPS of a video rendered in seconds/minutes.
- Never claim '720p neural generation' when the actual model only generates 256×144 and a renderer stretches it.
- No face-grid GANs, static random latent walks, shallow optical-flow retexturing or replayed MP4 presented as neural generation.
- Don't train a huge model from scratch on only two short clips.
- Maintain reproducible benchmark scripts and dated results in GitHub issues.

## Hardware gates

The RTX 3090 has 24 GB. Original 512×288/16-frame V3 CUDA baseline (batch throughput only):

| Steps | End-to-end time | Batch generated FPS |
|---|---:|---:|
| 24 | 12.2s | 1.315 |
| 12 | 6.6s | 2.441 |
| 8 | 4.8s | 3.347 |
| 4 | 2.7s | 5.890 |

At 4 steps quality was badly degraded; all of those numbers are from **offline 16-frame batches** and must not be equated to live browser FPS.

### Current project decision

We choose **A1111-quality teacher + compact causal neural student + WebGPU runtime** as the main research direction. Landscape and CFG 6 are the default for current video fidelity tests.

Alternative research tracks (AnimateLCM, small one-step image-to-image diffusion, Causal Forcing) remain *benchmarks/inputs* to our architecture rather than assumed drop-in solutions.
