# Real-Time Neural Video Engine (RTNVE)

**LifeAfterLogic — research and development for continuously evolving neural visuals.**

The ultimate goal is a live AI-generated visual world for concerts, museums, installations and websites. One coherent evolving scene; genuine neural synthesis with temporally connected transformations, unexpected appearances and disappearances, never a stack of portrait tiles or a prerecorded MP4.

## Current status — research prototype, not a finished engine

- **Working:** static GitHub-hosted WebGPU neural *image* generator used for initial browser capability checks (ganlive / pretrained GAN).
- **Ready for local testing:** AnimateDiff V3 GPU benchmark on a Windows PC with an RTX 3090.
- **Not implemented:** a trained, autoregressive / temporally conditioned model that can invent a continuous new neural scene at real-time speed.
- **Not proven:** WebGPU 10 generated FPS at 1280×720, native temporal continuity, or mobile device support.

We don't label crossfades, image warps, tiled faces or video playback as neural video generation.

## Start here — Windows RTX 3090

**The original Automatic1111 installations remain untouched.** The benchmarks use their *existing* model files in a separate local Python environment.

1. Download the repository: **Code → Download ZIP** from https://github.com/tmoreno-01/RTNVE, then unzip it on your Windows computer. Alternatively, clone it with Git.
2. **Python 3.11 or 3.12 works.** The setup script detects an existing installation on PATH or in common Windows install folders (the `py` launcher is optional).
3. Double-click **01_SETUP_WINDOWS.bat**. It creates a project-local .venv and installs a CUDA-enabled PyTorch build and Diffusers. CUDA 12.6 needs a compatible NVIDIA driver; select a suitable build from https://pytorch.org/get-started/locally/ if that fails.
4. Double-click **02_BENCHMARK_V3.bat**. Choose your original CyberRealistic SD1.5 checkpoint and **v3_sd15_mm.ckpt** in the file pickers.
5. Select **1 (512×288)** for the *initial compatibility and speed check*, NOT the production-resolution goal. The script generates 16 frames at 24, 12, 8 and 4 diffusion steps and measures true generation time and peak VRAM.
6. Once the first test succeeds, repeat step 4 at **768×432** and then **1280×720** if GPU memory allows.
7. Share the generated results.csv, results.json and a few test MP4s from benchmarks/results/animatediff-v3/ in the conversation so we can inspect quality and actual performance. **Do not upload your checkpoint or motion model.**

If Windows prints an error, copy the full terminal error rather than guessing at a fix. The launcher does not edit A1111, Windows drivers, or your installed system Python.

For detailed instructions: [benchmarks/README.md](benchmarks/README.md). Benchmark source: [benchmarks/animatediff_v3.py](benchmarks/animatediff_v3.py).

## Development stages

| Stage | What we build | Completion criterion |
|---|---|---|
| 1. **Teacher parity + profiling** | A separate reproduction of SD1.5 CyberRealistic + AnimateDiff V3, and systematic CUDA measurements | Output visually comparable with the original reference and correct performance logs |
| 2. **Speed experiments** | CUDA inference optimizations, AnimateLCM/consistency tests, reducing redundant computation | Better measured generation speed **without destroying the morphing aesthetic** |
| 3. **Temporal student** | A significantly smaller model trained on full AnimateDiff frame sequences; motion state and evolving random innovation | Outputs a *new* continuous coherent clip, not interpolation between stills; sustained 30+ second validation |
| 4. **Streaming + stability** | Infinite autoregressive output, handling its own feedback, long-duration tests and controllable changes | Sustained 5–15 minute sessions without freezing, collapse or repetitious resets |
| 5. **WebGPU version** | Export/port compatible small model, GPU-resident state, performance and mobile fallback | True generated FPS measured in-browser on target GPUs |
| 6. **Performance / installation engine** | Real-time performer control and reliable high-res show outputs using local dedicated GPU | Low-latency controllable visuals suitable for a live event |

### Performance targets — aspirational, not achieved

Initial aim: **10 truly generated neural FPS**; final aspirational aim: 720p at 24–30 FPS. Browser inference should run on a visitor GPU with **no always-on cloud GPU service**. Performance rigs can use a local dedicated GPU for higher quality. Low-end hardware may require lower resolution or reduced FPS.

The architecture may diverge internally from the original AnimateDiff, provided it retains the **visually fascinating learned temporal dynamics**. We are not assuming a few-step image model automatically has these properties.

## Research files

- [TEMPORAL_RESEARCH.md](TEMPORAL_RESEARCH.md) — temporal models, architecture options and acceptance criteria.
- [training/prepare_video_frames.py](training/prepare_video_frames.py) — whole-frame dataset extraction.
- [training/build_temporal_pairs.py](training/build_temporal_pairs.py) — ordered earlier-frame → later-frame pairs.
- [training/ONE_STEP_FEEDBACK_TEST.md](training/ONE_STEP_FEEDBACK_TEST.md) — one-step image-conditioned research.
- [training/STYLEGAN_V_FIRST_TEST.md](training/STYLEGAN_V_FIRST_TEST.md) — alternative continuous-time video GAN baseline.

## Browser Lab

The root index.html is an **early neural-image test**, not RTNVE's final video engine. It is kept for browser performance tests only. The earlier tiled-face experiment was rejected and is no longer the main page. GitHub Pages may need to be reconfigured after the repo was renamed from experimentation to RTNVE. The expected default project URL, when Pages is enabled, is https://tmoreno-01.github.io/RTNVE/.

The original ganlive runtime is MIT-licensed; its model weights have separate licences. Model and dataset licences must be checked before commercial deployments.

## Keep large assets off GitHub

Do not commit checkpoint files, training samples, MP4s, or personal local results to this public repository; they are excluded by .gitignore. Training runs stay on your Windows machine. Use a separate public model host for any future deployable, properly licensed weights.
