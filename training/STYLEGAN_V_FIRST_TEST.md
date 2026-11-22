# First *real temporal* experiment: StyleGAN-V

This is an isolated RTX 3090 test; it will **not** alter A1111. Nothing here is deployed to the website until it shows convincing learned motion.

## 1. Run a published temporal GAN, BEFORE training anything

Official source: https://github.com/universome/stylegan-v
Research demo: https://universome.github.io/stylegan-v/

It generates frames from content **plus learned time-dependent motion**.
This is the first tested open-source architecture to examine for the
"always transforming but continuous" goal.

Its authors explicitly include `environment-ampere.yaml` for RTX 3090.

You need a Linux/CUDA environment (WSL2 Ubuntu on Windows is reasonable, but
the CUDA/custom-op setup may need troubleshooting). Install it in its own
isolated Conda env, not the A1111 Python environments:

```bash
git clone https://github.com/universome/stylegan-v
cd stylegan-v
conda env create -f environment-ampere.yaml -p ./env
conda activate ./env
```

From the README, download a published model checkpoint (provided by the
authors) and change the checkpoint location below:

```bash
python src/scripts/generate.py \
  --network_pkl /path/to/network-snapshot.pkl \
  --num_videos 1 \
  --save_as_mp4 true \
  --fps 10 \
  --video_len 300 \
  --batch_size 1 \
  --outdir preview \
  --truncation_psi 0.9
```

**What to check:** Does it actually evolve recognisable structures over
time and invent unusual intermediate forms? Does the movement look
independent from a universal morph-to-new-portrait dial? Does it retain
enough visual detail? An arbitrary 30-second test is enough to reject
an unsuitable backbone.

Published training compute was substantial (4×V100 32GB for roughly two
days on 256² datasets). Don't assume retraining from scratch on a single
3090 will be quick. Prefer a pretrained proof, then fine-tuning and
distillation, if practical.

**Browser deployment:** This checkpoint cannot be dropped directly in
our existing ganlive folder. The model has an additional time/motion
network. It would need a converter from that architecture to a WebGPU
program, or distillation into a smaller compatible recurrent network.

## 2. Prepare whole-scene motion data

The tool in this repo extracts original MP4 frames at fixed frame rates,
preserves ordering by video, and saves a manifest.

Example (PowerShell, from your `experimentation` checkout):

```powershell
python training/prepare_video_frames.py --input "C:\\path\\to\\A1111_original.mp4" --out dataset --width 256 --height 256 --motion-fps 10 --still-fps 1
```

For a *landscape* model, use 896×512 instead of the 256² pilot; never
pretend a 256² model outputs native 1600×900 detail. 256² is only a
low-cost test to verify that motion is genuinely learned.

Your existing R2 files can also serve as material if the original
A1111 files are inaccessible. More independent clips and seeds are needed
for any serious training. One ~50-second render is a reference, not a
general video-generation training corpus.

## 3. The actual art engine

Our final model should combine:
- **Content latent:** what kind of scene exists;
- **Motion state:** what is presently evolving, retained through time;
- **Stochastic innovation:** genuinely new details that can appear;
- **Learned temporal synthesis:** the network produces the *next image*
  based on the dynamical state, rather than two unrelated full frames
  being crossfaded.

The neural network can then be run entirely on the browser's WebGPU,
if it is small enough. Current static GAN browser FPS is **not** proof
that a learned temporal GAN will have similar FPS.

Alternative research: SDXS image conditioning and causal few-step video
diffusion. These may produce more interesting semantic surprises, but
shipping them as true browser-native real time is a harder target.

## 4. Commercial and licensing caution

StyleGAN-V is built on the NVIDIA StyleGAN2/INR-GAN ecosystem; review
the code, checkpoint and generated-output licences before any commercial
deployment. We only use it for research until this is settled.
