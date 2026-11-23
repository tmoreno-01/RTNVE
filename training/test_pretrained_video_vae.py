#!/usr/bin/env python3
"""RTNVE quality-first neural compression gate using a real pretrained SD1.5 VAE.

This DOES NOT generate motion or new images; it checks whether the existing
AnimateDiff visual look survives an 8x-spatial / 4-channel neural latent codec.
If not, we cannot expect a tiny temporal model plus decoder to preserve it.

Uses a published VAE (stabilityai/sd-vae-ft-mse) as a reconstruction baseline,
NOT necessarily the identical VAE in the user's CyberRealistic checkpoint.

Output:
- reconstruction_comparison.jpg (original left / VAE decode right)
- reconstruction_results.json (per-frame PSNR, timings, latent size)
- original and VAE reconstructed frames; no cloud inference
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True,help="RTNVE teacher MP4, ideally 640x360 landscape")
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/vae_quality"))
    p.add_argument("--frames",type=int,default=8,help="Number of evenly sampled frames")
    p.add_argument("--vae",default="stabilityai/sd-vae-ft-mse",help="Hugging Face VAE repository or local diffusers VAE folder")
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    return p.parse_args()

def main():
    args=parse()
    if not args.video.is_file():raise FileNotFoundError(f"Video does not exist: {args.video}")
    if not 1<=args.frames<=32:raise ValueError("frames must be between 1 and 32")
    import torch
    import numpy as np
    import imageio.v2 as imageio
    from diffusers import AutoencoderKL
    from PIL import Image, ImageDraw

    print("RTNVE neural codec reconstruction — QUALITY TEST, not video generation")
    print("Loading actual pretrained SD1.5 VAE:",args.vae,flush=True)
    print("First time may download model weights. Subsequent runs use HF cache.")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; previously working RTX3090 environment missing")
    vae=AutoencoderKL.from_pretrained(args.vae,torch_dtype=torch.float32).eval().to(args.device)
    vae.enable_slicing()
    reader=imageio.get_reader(str(args.video),format="FFMPEG")
    try:
        n=reader.count_frames()
        if n<1:raise RuntimeError("No frames decoded from input video")
        positions=np.linspace(0,n-1,min(args.frames,n),dtype=int).tolist()
        frames=[np.asarray(reader.get_data(i),dtype=np.uint8)[:,:,:3] for i in positions]
    finally:
        reader.close()
    h,w=frames[0].shape[:2]
    if w%8 or h%8:
        raise ValueError(f"Source MP4 dimensions must be divisible by 8, received {w}x{h}")
    if any(a.shape!=frames[0].shape for a in frames):
        raise RuntimeError("Video frames vary in size")
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]; results=[]
    if args.device=="cuda":
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for i,arr in zip(positions,frames):
        x=torch.from_numpy(arr.copy()).permute(2,0,1).unsqueeze(0)
        x=(x.float()/127.5-1.0).to(args.device)
        if args.device=="cuda":torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            latent=vae.encode(x).latent_dist.mode()
        if args.device=="cuda":torch.cuda.synchronize()
        enc_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        with torch.inference_mode():
            y=vae.decode(latent).sample
        if args.device=="cuda":torch.cuda.synchronize()
        dec_ms=(time.perf_counter()-start)*1000
        decoded=((y.clamp(-1,1)+1)*127.5).round().byte().cpu().permute(0,2,3,1)[0].numpy()
        original=Image.fromarray(arr,"RGB")
        reconstruction=Image.fromarray(decoded,"RGB")
        out_name=f"frame_{i:04d}"
        original.save(args.out/f"{out_name}_original.png")
        reconstruction.save(args.out/f"{out_name}_vae.png")
        # This is deterministic reconstruction from a mode latent, not sampled generation.
        mse=float(np.mean((arr.astype(np.float32)-decoded.astype(np.float32))**2))
        psnr=10*math.log10((255**2)/max(1e-12,mse))
        # Display left/right at exactly the original pixel resolution.
        sheet=Image.new("RGB",(2*w,h+30),"white")
        sheet.paste(original,(0,30));sheet.paste(reconstruction,(w,30))
        draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f"Original | frame {i}",fill="black")
        draw.text((w+12,8),f"SD1.5 VAE roundtrip | PSNR {psnr:.1f}dB",fill="black")
        rows.append(sheet)
        results.append({"frame_index":int(i),"psnr_db":round(psnr,2),
            "encode_ms":round(enc_ms,2),"decode_ms":round(dec_ms,2)})
        print(f"Frame {i}/{n-1}: PSNR {psnr:.1f} dB, encode {enc_ms:.1f} ms, decode {dec_ms:.1f} ms",flush=True)
    overview=Image.new("RGB",(2*w,(h+30)*len(rows)),"white")
    for idx,row in enumerate(rows):overview.paste(row,(0,idx*(h+30)))
    visual=args.out/"reconstruction_comparison.jpg"
    overview.save(visual,quality=93)
    meta={"project":"RTNVE","test":"pretrained VAE encode/decode fidelity",
      "created_utc":datetime.now(timezone.utc).isoformat(),
      "source_video":str(args.video),"source_frames":n,
      "width":w,"height":h,"latent_shape":[1,4,h//8,w//8],
      "vae":args.vae,"vae_matches_cyberrealistic":False,
      "device":args.device,"torch":torch.__version__,
      "cuda_allocated_peak_gib":round(torch.cuda.max_memory_allocated()/1024**3,3) if args.device=="cuda" else None,
      "mean_psnr_db":round(sum(r["psnr_db"] for r in results)/len(results),2),
      "median_decode_ms":round(float(np.median([r["decode_ms"] for r in results])),2),
      "frames":results,
      "warning":"This is ONLY codec reconstruction quality, not generative or real-time video. Image detail may be lost; visually judge comparison. It does not measure browser decoder speed."}
    (args.out/"reconstruction_results.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("QUALITY CONTACT SHEET:",visual)
    print("RESULTS:",args.out/"reconstruction_results.json")
    print("NEXT: Send contact sheet and JSON. Good reconstruction is only the first quality gate.")

if __name__=="__main__":
    main()
