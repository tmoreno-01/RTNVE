# Prior art that RTNVE should build on (2026-11-23)

RTNVE is not the first endlessly generative neural artwork. It should not claim to be. The actual proposed differentiator is the **combination** of autonomous semantic invention, organically continuous neural video (not GAN latent walks or separate image generation), no obvious reset/loop, minimal manual interface, and a **visitor-GPU WebGPU runtime** with no ongoing server inference.

## Specific antecedents

### 1. Perpetual Dreaming Machine (2021)
https://artificialmind.ai/work/perpetual-dreaming-machine/
Generative installation using VQGAN+CLIP, guided diffusion and latent traversal through painting data; real-time display, Python API/WebSocket streaming. Conceptually very close to a machine that dreams endlessly, but not shown to run inference entirely in a common browser.

### 2. Refik Anadol — Unsupervised / Machine Hallucinations — MoMA (2022)
https://refikanadol.com/works/unsupervised/
A real-time AI artwork continually visiting StyleGAN2 ADA latent space. Dedicated computer for the collector; preview supplied as video because native browser inference was too heavy. Clear precedent for continuously evolving unsupervised art, but GAN latent exploration is a different mechanism from RTNVE's desired semantic video diffusion.

### 3. InfiniteStream (2026)
https://github.com/SolvAI/InfiniteStream
Extremely close technically: continuous autoregressive text-to-video with controllable changes to prompts. It reports **832×480 at 16 FPS** on a single **RTX 4090**, about 17.7 *generated* FPS in a 10-minute test, ~15.2 GB allocated, and roughly 20–23 GB reserved. Architecture: Wan2.1/Causal Forcing++, KV-ring buffer, sink-frame continuity, TAEHV lightweight decoder, prompt-change cache management. Published installation guidance requires **Linux and an NVIDIA 24 GB GPU tested on RTX 4090**. RTX 3090 (Ampere) and Windows compatibility are **not proven**; kernel/FP8 optimisation path may be different. Important: browser is a **viewer/client** receiving generated frames from a machine with GPU; this is **not browser-local neural inference**. It should inform or provide a baseline for RTNVE's local stage engine before we build a new causal model.

### 4. Training-Free Interactive World Models (2026)
https://hrdlabs.org/research/interactive-world-models/
Very close to RTNVE's smoother prompt-travel requirement. Reports interpolating old/new prompt embeddings using slerp and minimum-jerk easing across multiple autoregressive chunks, without clearing temporal KV cache. Their published demonstration uses **H100s**, not a consumer GPU browser. Their method is a valuable reference: **preserve model history, vary conditioning gradually**. Current RTNVE offline AnimateDiff does CLIP linear interpolation with smoothstep, 28-frame changes; do not claim this idea is new.

### 5. Daydream Scope (2026)
https://github.com/daydreamlive/scope
Open-source modular real-time generative pipeline with StreamDiffusionV2, LongLive and other autoregressive backends. Likely useful for evaluating models, effects and fast decoders. Current project README claims alpha **CC BY-NC-SA 4.0**: must not assume commercial reuse for LifeAfterLogic without further licence verification. Also not evidence of zero-server-GPU browser inference.

### 6. The Dreamwall
https://seethroughlab.com/projects/the-dreamwall/
Indefinite prompt-driven AI image generation + TouchDesigner live playback/transitions; this is less analogous to RTNVE because it generates discrete images and blends them in an installation, rather than one continuously neural-generated video with temporal memory.

## Distinguishing the actual milestones

- **CURRENT:** Diffusers AnimateDiff V3 teacher, one global 48-frame latent sequence using temporal windows and CLIP prompt-travel. User's N-mode 16/8, 20 steps, CFG 6, 640×360 test took **80.687s for 48 frames** = **0.5949 batch-generated FPS** on RTX 3090, ~5.424 GiB CUDA allocated. The 28-frame CLIP prompt change is visually improved but the final frames appear settled, not yet endless autonomous evolution.
- **NEAR NEXT:** Investigate a true **causal video model** (InfiniteStream/Causal Forcing++ / StreamDiffusionV2 / LongLive) for a separate native GPU engine. Reproduce and benchmark it against user target video. User RTX 3090 is not identical to published 4090/H100 systems; do not promise their performance on it.
- **BROWSER RESEARCH:** Build a much smaller causal video network with persistent memory; export to WebGPU. Existing streaming applications may have a browser UI but still run CUDA server-side. Verify this distinction for every candidate; don't claim it's browser-native unless weights actually run in the client GPU.
- **ARTISTIC CORE:** autonomous and surprise-rich prompts, natural nonlinear metamorphosis, full visual history never deliberately reset, no fixed loop, no manual requirement. A unique model training recipe/semantic controller/look and browser distribution are more defensible differentiators than claiming nobody has made an endless AI dream.

## Sources
- https://github.com/SolvAI/InfiniteStream
- https://hrdlabs.org/research/interactive-world-models/
- https://github.com/daydreamlive/scope
- https://artificialmind.ai/work/perpetual-dreaming-machine/
- https://refikanadol.com/works/unsupervised/
- https://seethroughlab.com/projects/the-dreamwall/
