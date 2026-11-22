# AI LAB — living neural art research

A standalone web playground for exploring **actual in-browser neural image generation**, working toward endlessly morphing, strange AnimateDiff V3-style artwork.

The currently deployed page is just a **single-image WebGPU inference baseline**. It is not a temporally connected artwork, and we aren't presenting it as one. The earlier tiled face collage was rejected and removed from the home page. The goal is a **single evolving neural scene** with meaningful transformations.

## Run it

After this repo's static page is set up, visit `index.html` via a web server. For GitHub Pages, use **Settings → Pages → Build and deployment → Deploy from a branch → main → /(root) → Save**, then visit:

https://tmoreno-01.github.io/experimentation/

You can also run locally with `python -m http.server 8000` then visit http://localhost:8000. A local server is recommended because module loading and WebGPU require a secure context (localhost or HTTPS).

Use **Load neural model** on a modern Chrome/Edge version with WebGPU enabled. The pretrained model weights download from Hugging Face on first load. They are not stored in this repo.

## Experiments

1. **Neural morph** — actual browser-side generative image inference, with continuous latent walk, speed, variation and seed controls.
2. **Temporal learning** — reserved for a future neural network trained on sequential AnimateDiff frames.
3. **Image-conditioned feedback** — reserved for a future neural synthesizer that learns transformations rather than static views.

## Source and licenses

The WebGPU inference runtime is vendored from [ganlive](https://github.com/gustavecortal/ganlive) by Gustave Cortal (MIT license, pinned at commit `204afd323fd43ac2352b7e6e461874cd963a999c`). Retain the license in `vendor/ganlive/LICENSE` when redistributing.

The pretrained model weights are hosted by their respective authors. **Model licenses apply separately**. Their usage here is for experiments. Generated imagery is not your own trained aesthetic yet.

## Next research stage

Curate frames from the original A1111 + AnimateDiff V3 renders; train a custom lightweight image generator on a sufficiently diverse dataset; test latent continuity and temporal feedback. A model that merely interpolates recorded images, or that collapses into repetitive morphs, does not meet the eventual art goal.

## What we're building next

Read [TEMPORAL_RESEARCH.md](TEMPORAL_RESEARCH.md) before implementing new experiments. The two active research avenues are (1) time-conditioned continuous video GANs such as StyleGAN-V and (2) one-step *image-conditioned* generative models trained on consecutive video frames. Neither is ready for browser production yet; do not claim otherwise.

Local training resources:
- [training/prepare_video_frames.py](training/prepare_video_frames.py) extracts ordered sequences from the user's A1111 videos.
- [training/build_temporal_pairs.py](training/build_temporal_pairs.py) creates whole-frame earlier→later image pairs for a one-step conditional student.
- [training/STYLEGAN_V_FIRST_TEST.md](training/STYLEGAN_V_FIRST_TEST.md) outlines the RTX 3090 pretrained temporal GAN evaluation.

Only substitute the live Lab model when an actual browser-deployable learned video generator works and passes extended motion tests.
