"""Export older NVIDIA checkpoints with the upstream single-ONNX wrapper."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve()))
    from core import foundation_stereo
    from scripts import make_single_onnx as exporter

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.args.max_disp = 192
    model.args.valid_iters = 8
    model.args.mixed_precision = False
    # Match the fallback in NVIDIA's make_plugin_onnx.py for older checkpoints.
    model.args.setdefault("normalize", True)

    foundation_stereo.normalize_image = lambda image: image
    foundation_stereo.build_gwc_volume_optimized_pytorch1 = exporter._build_gwc_volume_onnx
    foundation_stereo.build_concat_volume_optimized_pytorch1 = exporter._build_concat_volume_onnx

    height, width = 384, 640
    wrapper = exporter.FastFoundationStereoSingleOnnx(model).to(args.device).eval()
    pair = tuple(torch.randn(1, 3, height, width, device=args.device) for _ in range(2))
    print(f"Exporting {width}x{height} Fast-FoundationStereo on {args.device}", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            pair,
            str(args.output),
            opset_version=17,
            input_names=["left_image", "right_image"],
            output_names=["disparity"],
            dynamo=False,
        )
    cfg = OmegaConf.to_container(model.args)
    cfg["image_size"] = [height, width]
    OmegaConf.save(OmegaConf.create(cfg), args.output.with_suffix(".yaml"))


if __name__ == "__main__":
    main()
