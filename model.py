"""Fast-FoundationStereo inference with CUDA PyTorch, ONNX Runtime or TensorRT."""

import sys

import numpy as np


class FastFoundationStereo:
    def __init__(self, path, size, provider, upstream, max_disparity=192, iterations=8):
        width, height = size
        self.input_shape = (1, 3, height, width)
        self.output_shape = (height, width)
        self.checkpoint = path.suffix == ".pth"
        if self.checkpoint:
            if upstream is None:
                raise ValueError("--upstream is required with a PyTorch checkpoint")
            sys.path.insert(0, str(upstream.resolve()))
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable to PyTorch. Run from the Thor desktop.")
            self.torch = torch
            # NVIDIA's serialized checkpoints contain Python objects.
            self.runner = torch.load(path, map_location="cpu", weights_only=False)
            self.runner.args.setdefault("normalize", True)
            self.runner.args.max_disp = max_disparity
            self.runner.args.mixed_precision = True
            self.runner.cuda().eval()
            self.iterations = iterations
            return
        if path.suffix == ".onnx":
            import onnxruntime as ort

            if provider not in ort.get_available_providers():
                raise RuntimeError(f"ONNX Runtime provider {provider} is unavailable")
            self.session = ort.InferenceSession(str(path), providers=[provider])
            inputs = self.session.get_inputs()
            shapes = {item.name: tuple(item.shape) for item in inputs}
            self.run = self.run_onnx
        elif path.suffix == ".engine":
            if upstream is None:
                raise ValueError("--upstream is required with a TensorRT engine")
            sys.path.insert(0, str(upstream.resolve()))
            import torch
            from scripts.run_demo_single_trt import SingleEngineTrtRunner

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA-enabled PyTorch is required for the NVIDIA TensorRT runner")
            self.torch = torch
            self.runner = SingleEngineTrtRunner(str(path))
            engine = self.runner.engine
            trt = self.runner.trt
            names = (engine.get_tensor_name(index) for index in range(engine.num_io_tensors))
            shapes = {name: tuple(engine.get_tensor_shape(name)) for name in names
                      if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT}
            self.run = self.run_engine
        else:
            raise ValueError("Expected an NVIDIA .pth checkpoint or a single-export .onnx or .engine file")

        if set(shapes) != {"left_image", "right_image"} or any(
            shape != self.input_shape for shape in shapes.values()
        ):
            raise ValueError(f"Expected left/right inputs of shape {self.input_shape}, got {shapes}")

    def run_onnx(self, inputs):
        return self.session.run(["disparity"], inputs)[0]

    def run_engine(self, inputs):
        with self.torch.inference_mode():
            tensors = {key: self.torch.from_numpy(value).cuda() for key, value in inputs.items()}
            return self.runner(tensors)["disparity"].float().cpu().numpy()

    def __call__(self, left, right):
        if self.checkpoint:
            with self.torch.inference_mode():
                # The native model normalizes its own RGB inputs in the 0–255 range.
                images = [self.torch.from_numpy(image).permute(2, 0, 1)[None]
                          .to(device="cuda", dtype=self.torch.float32).contiguous()
                          for image in (left, right)]
                disparity = self.runner(
                    *images, iters=self.iterations, test_mode=True,
                    optimize_build_volume="pytorch1",
                )
                return disparity.float().cpu().numpy().reshape(self.output_shape)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        inputs = {key: np.ascontiguousarray(((image - mean) / std).transpose(2, 0, 1)[None])
                  for key, image in (("left_image", left), ("right_image", right))}
        return self.run(inputs).reshape(self.output_shape)
