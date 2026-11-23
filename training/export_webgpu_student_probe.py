#!/usr/bin/env python3
"""RTNVE WebGPU portability probe — EXPORT ONLY, NOT A TRAINED ART MODEL.

This builds a tiny actual recurrent CNN and exports an ONNX model with a
static input/output shape. The weights are deterministic but RANDOM and
untrained: the model is NOT a visual generator and its performance does
NOT predict AnimateDiff-quality browser inference.

Purpose:
  1) Confirm that a recurrent neural graph built in PyTorch runs in WebGPU.
  2) Test feeding its GPU-resident output back into itself across many frames.
  3) Measure latency and compatibility on an actual visitor device / phone.
  4) Only then consider a trained, compressed temporal student architecture.

All data and computation remain on the user's own device. No GPU server.
Requires torch (already installed in RTNVE) and ONNX (pip install onnx).
"""

from __future__ import annotations
import argparse
from pathlib import Path


def make_model(channels: int, hidden: int):
    import torch
    import torch.nn as nn

    class RecurrentProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
                nn.Tanh(),
            )

        def forward(self, state):
            # The previous frame/state influences the next output.
            # NEVER describe this untrained network as a viable art engine.
            return 0.90 * state + 0.10 * self.layers(state)

    return RecurrentProbe().eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=72)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--output", type=Path,
                   default=Path("model_probes/recurrent_probe_128x72.onnx"))
    a = p.parse_args()

    if a.width % 8 or a.height % 8 or a.width < 16 or a.height < 16:
        p.error("Width and height must be multiples of 8 and at least 16.")
    if a.width > 512 or a.height > 512:
        p.error("This is a small-device inference probe, not a 720p model.")
    if not 1 <= a.channels <= 32 or not 1 <= a.hidden <= 64:
        p.error("Invalid channel sizes (1–32 state, 1–64 hidden).")

    try:
        import torch
        import onnx
    except ImportError as e:
        raise SystemExit(
            f"Missing {e.name}. In the RTNVE .venv run: python -m pip install onnx"
        ) from e

    torch.manual_seed(a.seed)
    model = make_model(a.channels, a.hidden)
    dummy = torch.randn((1, a.channels, a.height, a.width), dtype=torch.float32)
    a.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        expected = model(dummy)
        assert expected.shape == dummy.shape
        torch.onnx.export(
            model,
            (dummy,),
            str(a.output),
            export_params=True,
            opset_version=17,
            input_names=["state"],
            output_names=["next_state"],
            dynamic_axes=None,
            do_constant_folding=True,
            dynamo=False,
        )

    graph = onnx.load(str(a.output))
    onnx.checker.check_model(graph)
    print("ONNX MODEL:", a.output.resolve())
    print("Inputs:", [v.name for v in graph.graph.input])
    print("Outputs:", [v.name for v in graph.graph.output])
    print("State shape: [1, %d, %d, %d]" % (a.channels, a.height, a.width))
    print("Model bytes:", a.output.stat().st_size)
    print("IMPORTANT: RANDOM UNTRAINED MODEL. Measures portability and speed only.")
    print("Next: open https://tmoreno-01.github.io/RTNVE/webgpu-student-probe/")
    print("Select the .onnx file and run the on-device WebGPU inference test.")


if __name__ == "__main__":
    main()
