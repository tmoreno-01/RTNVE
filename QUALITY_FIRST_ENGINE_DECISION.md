# RTNVE — quality-first architecture decision (November 2026)

## What the desktop WebGPU result proves — and doesn't

The 20,129-byte untrained recurrent ONNX probe successfully ran on the user's RTX3090 in Chrome: model load 2.603s, 60 recurrent iterations in 91.2ms, average 1.52ms/iteration, approximately 658 iterations/second.

**This is only a GPU feedback and inference plumbing success.** It has untrained random weights, NO visual decoder, NO learned temporal behaviour, and no 720p output. It cannot be extrapolated to final neural video FPS or artistic quality.

## Real goals

- Preserve the **actual uncanny 20-step AnimateDiff V3 metamorphoses** from the successful 640x360, CFG6, context 16/overlap8, nonloop N, 28-frame autonomous prompt travel examples.
- Artist wants genuine continuous neural image invention, no visible resets or repeating loops, minimal interaction.
- Visitor browser's GPU should execute inference with no continuous server GPU cost. Supported phones should eventually have an appropriately distilled tier.
- Display 1280x720, possibly 1920x1080 later; **never mislabel a lower native neural render as native 720p**.

The teacher model currently generates ~0.595 FPS on the RTX3090 at 640x360, **offline batch throughput**. It cannot be put unchanged into mobile Safari and run in real time.

## The 4 architectural possibilities

### A — Neural codec + latent world evolution (preferred first research track)

1. Train/fine-tune a small *causal* neural dynamics model that updates compressed latent image+memory given recent history, new stochastic innovation, and changing semantic conditioning.
2. Preserve highly pretrained image/video decoder weights so the smaller dynamics network does not have to learn image detail from scratch.
3. Decode genuinely newly generated latent frames, then use temporally consistent neural upscaling for 720p display. A cheap decoder might work at lower latent resolution; there is no guarantee the artist-approved aesthetic survives.
4. Train on MANY diverse teacher clips, including full-frame chronology, multiple seeds, long morphs, and gradual prompt changes. Train on self-generated rollouts to avoid freezes or drifting to static pictures.
5. Export to ONNX/WebGPU and measure the entire learned model + decoder + upscale on Chrome and supported mobile browsers.

**First gate implemented:** training/test_pretrained_video_vae.py and Windows launcher 05_TEST_NEURAL_CODEC.bat use a REAL pretrained SD1.5 VAE to encode+decode eight frames from a successful RTNVE teacher MP4. Output: side-by-side visual quality and reconstruction metrics, plus native GPU times. This is not generation, not WebGPU, and uses stabilityai/sd-vae-ft-mse which may differ from the CyberRealistic VAE. If it ruins the visuals, this architecture needs a better/fine-tuned codec before proceeding.

### B — One-to-four-step video diffusion student (alternative)

Distil the AnimateDiff-like **multi-frame generative process**, rather than training a tiny recurrent CNN from scratch. Keep an economical causal temporal state and gradual changes in prompts. This is much harder to train, but could better preserve the look than a purely deterministic image predictor.

Related credible examples:
- SnapGen-V generated 5 seconds of 512x512 video in 4.12 sec on iPhone 16 Pro Max with 0.6B parameters using 4 steps in a NATIVE mobile app (not Safari). https://snap-research.github.io/snapgen-v/
- Causal Forcing++ and Causal-rCM describe distillation of streaming causal video diffusion: research scale and nontrivial training, not off-the-shelf 3090/browser code.
- WebGPU-Video-Diffusion really runs an 8-frame SD1.5-based video pipeline in WebGPU but reports 8–10 minutes for eight frames; it does not meet the live target. https://github.com/WebGPU-Video-Diffusion/WebGPU-Video-Diffusion

### C — Precomputed teacher source plus live neural transformation

Download precomputed high-quality teacher material from static storage and have a neural network transform or recombine it on-device. Very low server GPU cost and possible 720p presentation, but restricted to precomputed source diversity; MUST be labelled transparently, not marketed as totally newly generated continuous neural video. A pragmatic fallback, not the original pure goal.

### D — Local or cloud GPU

Show/museum installations can run the high-quality model on a dedicated on-premise GPU. For a public website, 24/7 cloud inference and concurrent visitors incur ongoing cost; not compatible with the zero-server-inference constraint.

## Next engineering sequence

1. Stop interpreting synthetic tiny-network FPS as a useful video-quality number.
2. Run the pretrained VAE roundtrip quality gate on the 48-frame N/16/8 teacher. Review the contact sheet and encode/decode results.
3. Create more varied, artist-approved teacher material; two 48-frame tests are inadequate training diversity.
4. Develop a latent temporal predictor with stochastic innovations and memory that survives prompt changes, plus explicit long rollout tests.
5. Compare artistic quality to the original AnimateDiff, and profile native model resolution, upscaling and full inference pipeline. Only export a trained student once the quality passes.
6. Test the same network via WebGPU on desktop and, when feasible, a real iPhone/Android phone. If it can't sustain the desired visual result, change architecture, do not disguise replay or shader movement as new neural generation.

## Non-negotiable truth

A pretrained neural codec can reconstruct existing frames, but **does not invent new frames by itself**. Successful reconstruction is only a necessary quality gate, not the finished engine. Training and compressing a comparable generator within one RTX3090's capacity remains a major research risk, and is NOT guaranteed.

No free remote inference provider can make GPU compute cost disappear. Local GPU shows and browser visitor-side models may require separate weights and performance presets.
