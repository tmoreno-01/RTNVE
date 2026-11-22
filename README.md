# AI LAB — living neural art experiments

A standalone web playground for exploring **actual in-browser neural image generation**, working toward endlessly morphing, strange AnimateDiff V3-style artwork.

The first experiment is an engineering test: real WebGPU inference with publicly available pretrained GANs (not a video or shader fake). It is **not yet a custom model trained on AnimateDiff frames**, nor does it recreate A1111.

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
