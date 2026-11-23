# Real-Time Neural Video Engine (RTNVE)

**LifeAfterLogic — research and development for continuously evolving neural visuals.**

The ultimate goal is a live AI-generated visual world for concerts, museums, installations and websites. One coherent evolving scene; genuine neural synthesis with temporally connected transformations, unexpected appearances and disappearances, never a stack of portrait tiles or a prerecorded MP4.

## Website and phone feasibility — zero GPU inference server

**LifeAfterLogic's product requirement is an autonomous, genuinely neural evolving artwork on visitor GPUs, including supported phones, with no continuous rented GPU service.** This remains a research target; the successful 20-step AnimateDiff visuals are an artistic teacher, not yet a mobile-sized model.

**[Read cost/mobile reality and acceptance gates](MOBILE_COST_AND_MODEL_GATES.md).**

**New real PyTorch→ONNX→WebGPU portability probe:** `04_BUILD_WEBGPU_PROBE.bat` exports a tiny **untrained recurrent CNN** from your isolated Windows .venv. Open the resulting local ONNX model in the [WebGPU neural inference probe](https://tmoreno-01.github.io/RTNVE/webgpu-student-probe/) on your desktop (then optionally phone). The probe measures neural iterations/sec and GPU-resident feedback, **not art-quality video FPS**. It has no visual output; the model is deliberately random/untrained. Running it doesn't alter A1111 or require cloud inference.

**Next genuine AI milestone:** train a significantly smaller recurrent neural student on a diverse collection of your best approved temporal metamorphosis clips, then compare visual quality and end-to-end WebGPU performance. Do not train a student on only two 48-frame samples expecting usable results.

## Current experiment — non-looping autonomous neural world

**Current research defaults: landscape 640×360, 20 steps, CFG 6, context 16 / overlap 8, stride 1, closed-loop `N` (absolutely no loop), and autonomous prompt travel.** The next visual-quality test makes one semantic change gradually over **28 frames**, instead of three abrupt six-frame shifts. On a 100-frame test, it changes after the 50-frame segment with a 28-frame transition. This is still *offline AnimateDiff*, not the live browser engine.

The original extension explains `N` as its only true non-loop setting; `A` instead aggressively forces the end to resemble the start. RTNVE fixes the upstream N schedule's occasional uncovered boundary frames. Existing 16/8 R+P tests are preserved under different filenames and metadata.

**[Read the resolution and autonomy decisions](HIGH_RESOLUTION_AND_AUTONOMY.md)** — native neural generation, possible coherent upscaling to 720p/1080p, and autonomous evolution with minimal interface. We are **not** rewriting SD1.5's motion module for native 1080p before showing a quality/performance need.

**Windows action:** run `00_UPDATE_RTNVE.bat` to synchronise the Python/PowerShell files (if the updater itself is outdated, update `scripts/00_Update.ps1` first), then run `03_BENCHMARK_TEMPORAL.bat`. The launcher now defaults to N, 16/8 and autonomous travel. Upload the generated MP4 and `*_results.json` to compare against the last successful 16/8 R+P version.

## Current status — research prototype, not a finished engine

- **Working:** static GitHub-hosted WebGPU neural *image* generator used for initial browser capability checks (ganlive / pretrained GAN).
- **Ready for local testing:** AnimateDiff V3 GPU benchmark on a Windows PC with an RTX 3090.
- **Not implemented:** a trained, autoregressive / temporally conditioned model that can invent a continuous new neural scene at real-time speed.
- **Not proven:** WebGPU 10 generated FPS at 1280×720, native temporal continuity, or mobile device support.

We don't label crossfades, image warps, tiled faces or video playback as neural video generation.

## First successful long-context result and prompt-travel experiment

On 2026-11-23 the RTX 3090 produced an actual 48-frame, **640×360 landscape** AnimateDiff V3 test at **20 steps, CFG 6**, context **13 / overlap 6**. Results: **79.15 seconds end to end**, **0.6064 batch-generated FPS** and **5.083 GiB** peak CUDA allocation. The clip shows substantial architectural/vehicle/vegetation metamorphosis, far beyond the earlier single-window test, but still visibly loops towards the opening scene. These are OFFLINE numbers, not real-time WebGPU inference.

**Now available for experiment (not yet executed on RTX 3090):**

- **Context comparison:** run the existing `03_BENCHMARK_TEMPORAL.bat`, select **16-frame context / overlap 8**, same scene, same seed, frames and settings. Visually compare against 13/6; smoother movement is a hypothesis, not yet proven.
- **Prompt travel:** the launcher can optionally change the underlying CLIP **text conditioning** at randomly scheduled frames inside one global AnimateDiff denoising process, without crossfading finished pictures. The new `benchmarks/prompt_travel.py` samples surprising scene/actor/transformation combinations from a deterministic seed, then smoothly changes conditioning across each transition.
- The intended long-run prompt dwell choices are **50, 100 or 200 frames**, randomly. The 48-frame experiment uses **12–16-frame intervals** so you can see multiple transitions in a short test; the 100-frame preview uses one change after 50 frames. This is a temporary test schedule, not the desired live cadence.
- Each test produces a uniquely named MP4 and matching `*_results.json`, `*_results.csv`, and (for travel) `*_plan.json`. Local outputs remain excluded from GitHub.

**Keeping a working local ZIP updated:** The new `00_UPDATE_RTNVE.bat` fetches the benchmark scripts, prompt-travel module, and PowerShell launcher; no need to reinstall PyTorch or modify Automatic1111. When the updater itself changes, replace `scripts/00_Update.ps1` from GitHub once.

Prompt travel currently works only as an *experimental offline teacher conditioning mechanism*. It is **not proof of lifelong causal memory**. The real-time website needs a substantially smaller, temporally trained model that preserves latent state while introducing new semantic instructions, with the text embeddings cached and reused on the visitor's GPU.

## Browser engine plan

**[Read the technical bridge from A1111 teacher → small causal student → WebGPU website engine](BROWSER_ENGINE_ROADMAP.md)**.
**[WebGPU browser capability check](https://tmoreno-01.github.io/RTNVE/webgpu-check/)** — detects browser WebGPU/GPU memory limits; this is a hardware diagnostic only, not neural video or an inference speed result.
 The eventual browser engine is not the heavy 20-step A1111 pipeline converted directly to JavaScript. It will be a much smaller learned, temporally stateful model that generates new content on visitor GPUs.

**Current defaults:** landscape **640×360** (16:9), **CFG 6**, 20 steps, context 13, overlap 6, seed 611. Start the new 03 temporal test with **48 frames** to check compatibility, then test 100 frames. The earlier portrait CFG 5 screenshot is a historical reference, not the new production direction.

## New: 100-frame A1111-style temporal context test (experimental)

The 16-frame RTX 3090 benchmark worked but produces almost static shots. In contrast, the two uploaded original A1111 clips contain 100 frames at 10 FPS with substantial changes across the entire scene.

The A1111 screenshot shows: **20 steps, CFG 5, seed 611, 100 frames, 10 FPS, portrait orientation, context batch 13, stride 1, overlap 6, closed loop R+P**. These screenshot settings are a reference and are not guaranteed to be the precise settings of both old MP4s.

The new **benchmarks/animatediff_long_context.py** processes a single global latent video. During every diffusion denoising step it takes moving, overlapping 13-frame context windows, runs AnimateDiff on them, and merges predictions back into the *same latent video*. No separate short videos, shader movement, or crossfade stitching.

**How to run:** Update the local repo (the old ZIP does not update itself), then double-click **03_BENCHMARK_TEMPORAL.bat**. It calls **scripts/03_Run_Temporal.ps1**, reuses the already installed .venv, and reads the original local checkpoint and V3 motion module. No Python reinstallation required.

Choose your NYC/CCTV prompt or the apartment-window screenshot prompt; for a first full-length comparison use 100 frames at 360x640, 20 steps, CFG 5, seed 611, context 13, overlap 6. Only move to 512x896 and eventually 720x1280 if motion fidelity improves. This is a significantly heavier offline test, not real-time.

Outputs: MP4 and temporal_results.json under **benchmarks/results/animatediff-context/**, excluded from GitHub commits.

**Known differences to A1111:** Weighted prompt interpretation, full sampler/noise parity, prompt scheduling and LoRAs are not yet guaranteed identical. The screenshot LoRA names were deliberately not applied because their compatibility with V3 was unverified.

**Pass criterion:** sustained, organic, temporally connected restructuring throughout the scene. If it remains static, investigate motion checkpoint conversion and A1111 parity before distilling or training a faster model.

## Start here — Windows RTX 3090

**The original Automatic1111 installations remain untouched.** The benchmarks use their *existing* model files in a separate local Python environment.

1. Download the repository: **Code → Download ZIP** from https://github.com/tmoreno-01/RTNVE, then unzip it on your Windows computer. Alternatively, clone it with Git.
2. **Python 3.11 or 3.12 works.** The setup script detects an existing installation on PATH or in common Windows install folders (the `py` launcher is optional).
3. Double-click **01_SETUP_WINDOWS.bat**. It creates a project-local .venv and installs a CUDA-enabled PyTorch build and Diffusers. CUDA 12.6 needs a compatible NVIDIA driver; select a suitable build from https://pytorch.org/get-started/locally/ if that fails.
4. Double-click **02_BENCHMARK_V3.bat**. Choose your original CyberRealistic SD1.5 checkpoint and **v3_sd15_mm.ckpt** in the file pickers.
5. Select **1 (512×288)** for the *initial compatibility and speed check*, NOT the production-resolution goal. The script generates 16 frames at 20, 12, 8 and 4 diffusion steps and measures true generation time and peak VRAM.
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
