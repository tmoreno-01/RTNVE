# Resolution strategy — native neural generation vs displayed resolution

Updated November 23, 2026. For RTNVE / LifeAfterLogic.

## Decision

Do **not** rewrite or retrain the original SD1.5 AnimateDiff V3 motion module solely to output 1280×720 or 1920×1080 yet.

The SD1.5 image backbone was originally trained around a 512-pixel scale, and AnimateDiff's temporal motion modules were designed around short windows. **Your own A1111 examples show this V3 setup can render at 720×1280**, so larger output is possible without rewriting the weights. But it is not guaranteed to improve quality: direct high-res generation can duplicate subjects, distort composition, increase memory requirements and generate much more slowly. We should measure before trying expensive high-resolution fine-tuning; rewriting model code does not magically change what its weights learned.

Both **720p and 1080p display output** remain goals. There are two separate tasks:

1. Produce **interesting native neural dynamics** at the lowest useful internal resolution.
2. Present those dynamics at high resolution **without introducing flicker, static overlays, or stretching**.

## Pixel-budget reality

| Resolution | Image pixels | Relative to 640×360 |
|---|---:|---:|
| 640×360 (current working teacher) | 230,400 | 1× |
| 896×504 | 451,584 | 1.96× |
| 1280×720 | 921,600 | 4× |
| 1920×1080 | 2,073,600 | 9× |

Those are pixel ratios, **not measured speed or VRAM multipliers**. UNet compute, attention memory, VAE decoding and model architecture can scale differently.

The successful NVIDIA RTX 3090 teacher at 640×360, 48 frames, 20 diffusion steps, 16-frame context/8 overlap took ~77–78 seconds and allocated ~5.4 GiB peak VRAM. It is already ~0.6 *offline batch-generated* FPS. Native 1080p teacher video will not become real time just by increasing resolution or changing the denoising scheduler.

## What we'll test, in order

**Gate A — continuity and aesthetics.** Confirm the new N-mode (no cyclic first/last frame context), 16/8 motion, and 28-frame prompt changes. A beautiful continuous scene matters much more than a bigger blurry/broken image.

**Gate B — teacher resolution scaling (offline only).** After Gate A, benchmark 48 frames at 640×360, then 896×504, and only if quality and VRAM allow, 1280×720. Keep the *same prompt, seed, loop mode, context size and CFG*. Compare:
- Motion and semantic transformations per frame;
- Duplicated objects/faces and broken perspective;
- Frame-to-frame stability and detailed textures;
- VRAM, generation latency and decoder cost.
Do not jump to 1080p yet. Avoid blind high-res fine-tuning of V3.

**Gate C — student architecture.** Train a *small causal temporal neural model* with internal latent features. Its successful native resolution might be 256×144, 384×216, 512×288 or higher depending on performance. A model that produces new, coherent evolving art at ~10 genuine generated FPS is the initial target; only claim the achieved native resolution.

**Gate D — coherent high-resolution presentation.** Test a temporally consistent, neural resolution-enhancement stage that conditions on prior output and preserves evolving edges/details, rather than individually upscaling stills and causing shimmer. It must be separately benchmarked end-to-end on WebGPU. 1280×720 display can be generated from a smaller internal neural model; users must not be told the model generates native 720p if it doesn't.

**Gate E — optional native 720p research.** If the causal student is good but high-res output clearly loses too much appearance or structure, consider larger latent features, sparse attention, tile-aware spatial encoders or fine-tuning temporal neural weights on properly prepared licensed 720p training sequences. This requires actual comparative evidence and likely substantial compute.

**Gate F — 1080p display.** Only after the browser runs a compelling 720p presentation with sustainable latency, memory and bandwidth.

## WebGPU implementation constraints

The current desktop reports NVIDIA Ampere, WebGPU and shader-f16 support, maximum storage buffer binding 128 MiB and maximum buffer 256 MiB. These are per-buffer limits, not total GPU memory. Student model weights, intermediate tensors and state must fit supported bindings and be divided/streamed where needed. Device variability is significant.

The browser target is **zero continuous server GPU inference**, not literally zero hosting/data-transfer cost.

## Autonomy requirement

The final art should not need a prompt panel, MIDI controller, or manual scrubbing to feel alive.

- **Every generated frame:** advance persistent neural visual state and inject bounded stochastic innovation.
- **Medium timescale:** naturally emergent morphing, organic deformation, appearance/disappearance of learned structures.
- **Longer timescale:** an autonomous semantic composer changes scene/subject/material/transformations on unpredictable intervals. Text-condition transitions unfold smoothly over roughly 20–30 generated frames and do NOT reset the model.
- **Rare surprises:** more radical subject shifts, still bridged through its visual memory rather than a hard cut.
- **UI:** start automatically on suitable hardware; minimal pause/fullscreen and quality diagnostics.

No fixed first/last matching, no repeating loop, no human needing to type new prompts.

**Important technical distinction:** the current offline AnimateDiff teacher does not literally feed frame N's final RGB image into the network to generate frame N+1. It jointly denoises a *whole latent video* using overlapping temporal contexts. A **causal recurrent student** will need to be separately trained for continuous output with state that survives every frame and every prompt change.
