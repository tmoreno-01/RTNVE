# Mobile and zero-server-cost feasibility — 2026-11-23

## Non-negotiable goals

LifeAfterLogic's RTNVE is a fully autonomous neural artwork, ideally running continuously on ordinary desktop and phone browsers. No ongoing per-visitor GPU inference server; a server-powered video stream to a browser does **not** satisfy the main website goal.

Cloud NVIDIA GPUs are not free. For example, Runpod advertises RTX 4090 instances from ~$0.34/hour in late 2026 (https://www.runpod.io/gpu-models/rtx-4090), before storage, bandwidth, demand and operating costs. Hosting a Python GPU model for every website visitor would create ongoing costs, and a 24/7 RTX 3090 at home would still consume electricity and create reliability/security problems.

## Core engineering reality

The preferred AnimateDiff teacher look (20 steps, CFG6, N no-loop, context16 overlap8, 640x360) takes ~80.7 seconds to generate 48 frames on RTX 3090. It cannot simply be loaded unchanged into Safari and expected to run in real time. Even high-end desktops need significant distillation, and phones have lower memory, battery and thermal headroom.

Phone WebGPU support makes **small local neural models plausible**. It does NOT guarantee 1280x720, 10 generated FPS, or the same look. We should prove all three constraints independently: continuity and visual quality, model size and latency, compatibility across target devices.

## Two versions of the same artwork

- Powerful desktop: larger trained neural student, higher latent resolution/detail, still on visitor GPU.
- Supported mobile: smaller distilled model, fewer channels and lower native resolution, perhaps a temporally consistent neural upscaler. It should share the same artistic evolution rules and model training direction, not become a disconnected video loop.
- Unsupported devices: show a clearly labelled preview or an honest compatibility message, never call MP4 playback or shader warping live neural video.

No promises of universal iPhone compatibility until benchmarked in actual mobile Safari.

## Concrete prototype work completed

- training/export_webgpu_student_probe.py exports a small real convolutional **recurrent neural graph** from the already installed PyTorch to ONNX. **Random weights; not trained; not an artwork.**
- 04_BUILD_WEBGPU_PROBE.bat / scripts/04_Build_Probe.ps1 install only the ONNX exporter in the separate RTNVE venv and write a tiny ONNX model locally.
- webgpu-student-probe/index.html opens that ONNX model through a file picker, runs it on the **visitor's WebGPU**, feeds output state into the next neural iteration while GPU-resident, and reports actual inference iteration latency.

This tests the PyTorch -> ONNX -> WebGPU transfer and feedback mechanism, **not the visual quality or final generated video FPS**. The script and webpage have been committed but still need actual Windows and phone browser tests; code syntax alone is insufficient.

## What comes next

1. Run untrained recurrent WebGPU probe on RTX 3090 Chrome (and later a phone), fixing any portability or memory issues. Keep actual error logs.
2. Assemble a much more varied dataset of *artist-approved teacher metamorphosis* with longer non-loop sequences and prompt transitions. Two 48-frame videos aren't enough to train a convincing engine.
3. Build/finetune a small causal temporal student. It must generate novel visually meaningful dynamics under varying semantic conditioning, not copy or blur previous images. Include rollout/self-feedback training.
4. Export that trained student and test output parity plus 30-second to 15-minute runs in PyTorch and browser. Profile native generated FPS, neural decoder, memory, thermals and display, separately.
5. Prioritise adaptive mobile/desktop model variants and native-vs-upscaled resolution honesty.

The exact RTNVE AnimateDiff look remains a **visual benchmark**, not a promise the first recurrent model will look comparable.

## No shortcuts

- No 24/7 paid GPU service billed as zero-cost.
- No prerecorded MP4 or shader latent walk marketed as true live neural evolution.
- No unmeasured 720p/1080p native-generation claims.
- No claim the untrained portability probe's speed predicts a trained model's speed.

References:
https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
https://www.runpod.io/gpu-models/rtx-4090
https://developers.cloudflare.com/pages/platform/limits/
