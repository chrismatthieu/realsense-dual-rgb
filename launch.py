"""Choose and run a D405 demo preset from the Thor desktop terminal."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("DUAL_RGB_ASSETS", ROOT)).expanduser().resolve()
CHECKPOINT = Path.home() / "Downloads" / "model_best_bp2_serialize.pth"
UPSTREAM = ASSETS / "Fast-FoundationStereo"
GPU_PYTHON = ASSETS / ".ffs-export/bin/python"
CPU_PYTHON = ASSETS / ".venv/bin/python"
MODELS = ASSETS / "models"


def presets():
    cuda = ["--model", str(CHECKPOINT), "--upstream", str(UPSTREAM)]
    return (
        ("Dual RGB preview", sys.executable, ["--camera-only"]),
        ("Dual RGB + D405 depth", sys.executable, ["--camera-only", "--sensor-depth"]),
        ("Fast-FoundationStereo · ONNX CPU", CPU_PYTHON,
         ["--model", str(MODELS / "fast_foundationstereo.onnx")]),
        ("Fast-FoundationStereo · CUDA PyTorch · 640×384, 8 iterations", GPU_PYTHON, cuda),
        ("Fast-FoundationStereo · CUDA PyTorch · 640×384, 4 iterations", GPU_PYTHON,
         [*cuda, "--iterations", "4"]),
        ("Fast-FoundationStereo · CUDA PyTorch · 448×256, 4 iterations", GPU_PYTHON,
         [*cuda, "--model-size", "448", "256", "--iterations", "4"]),
        ("Fast-FoundationStereo · TensorRT FP16 · 640×384", GPU_PYTHON,
         ["--model", str(MODELS / "fast_foundationstereo.fp16.engine"),
          "--upstream", str(UPSTREAM)]),
        ("Fast-FoundationStereo · TensorRT FP32 · 640×384", GPU_PYTHON,
         ["--model", str(MODELS / "fast_foundationstereo.fp32.engine"),
          "--upstream", str(UPSTREAM)]),
    )


def missing_files(python, args):
    roots = [Path(python)]
    if "--model" in args:
        roots.append(Path(args[args.index("--model") + 1]))
    if "--upstream" in args:
        roots.append(Path(args[args.index("--upstream") + 1]) / "core/foundation_stereo.py")
    return [path for path in roots if not path.is_file() or path.stat().st_size == 0]


def run_demo(python, args):
    env = os.environ.copy()
    library_paths = ["/usr/lib/aarch64-linux-gnu", env.get("LD_LIBRARY_PATH")]
    env["LD_LIBRARY_PATH"] = ":".join(path for path in library_paths if path)
    try:
        result = subprocess.run([python, ROOT / "live.py", *args], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return
    except OSError as error:
        print(f"Could not start demo: {error}")
        return
    if result.returncode:
        print(f"Demo exited with status {result.returncode}.")


def main():
    entries = presets()
    options = {"s": ["Include D405 depth and fusion", True, ["--sensor-depth"]],
               "m": ["Smooth model depth", False, ["--median-depth"]],
               "o": ["Save latest frame to output/latest.npz", False,
                     ["--output", "output/latest.npz"]]}
    while True:
        print("\nD405 demo launcher")
        for number, (label, python, args) in enumerate(entries, start=1):
            availability = " (unavailable)" if missing_files(python, args) else ""
            print(f"  {number}. {label}{availability}")
        for key, (label, enabled, _) in options.items():
            print(f"  {key}. {label} [{'on' if enabled else 'off'}]")
        print("  q. Quit")
        choice = input("Select a demo or toggle an option: ").strip().lower()
        if choice == "q":
            return
        if choice in options:
            options[choice][1] = not options[choice][1]
            continue
        if choice not in {str(number) for number in range(1, len(entries) + 1)}:
            print("Choose a number, a toggle, or q.")
            continue
        label, python, args = entries[int(choice) - 1]
        missing = missing_files(python, args)
        if missing:
            for path in missing:
                print(f"Missing or empty: {path}")
            continue
        extra = []
        for key, (_, enabled, flags) in options.items():
            if enabled and (key == "o" or "--camera-only" not in args):
                extra.extend(flags)
        print(f"Launching {label}. Press Q or Esc in the preview to return.", flush=True)
        run_demo(python, [*args, *extra])


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
