"""Build the Corebrum app archive and its release manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent
APP = "realsense-dual-rgb"
OWNER = "chrismatthieu"
SOURCE = f"https://github.com/{OWNER}/{APP}"
VERSION = "1.0.0"
TAG = f"corebrum-app-v{VERSION}"
PACKAGE_FILES = (
    "LICENSE", "README.md", "run", "launch.py", "live.py", "model.py",
    "stereo.py", "setup-model.sh", "export_model.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license", required=True, help="SPDX identifier of the repository LICENSE")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "marketplace")
    args = parser.parse_args()

    missing = [name for name in PACKAGE_FILES if not (ROOT / name).is_file()]
    if missing:
        parser.error(f"Missing release files: {', '.join(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"{APP}-{VERSION}-ubuntu24.04-arm64.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in PACKAGE_FILES:
            bundle.add(ROOT / name, arcname=name)
    with archive.open("rb") as data:
        sha256 = hashlib.file_digest(data, "sha256").hexdigest()

    manifest = {
        "id": f"{OWNER}/{APP}",
        "title": "RealSense Dual RGB Depth",
        "description": "Preview D405 stereo RGB and compare hardware depth with NVIDIA Fast-FoundationStereo.",
        "source": SOURCE,
        "license": args.license,
        "interface": "terminal",
        "tags": ["RealSense", "Depth", "Jetson", "TensorRT"],
        "releases": [{
            "version": VERSION,
            "entry": "run",
            "targets": [{
                "ubuntu": "24.04",
                "arch": "arm64",
                "hardware": ["NVIDIA Jetson AGX Thor", "Intel RealSense D405"],
                "requirements": [
                    "D405 firmware with USB Dual RGB support",
                    "librealsense 2.59.2 or newer with Python bindings",
                    "NumPy and OpenCV with desktop GUI support",
                    "Optional Fast-FoundationStereo model and inference runtime",
                ],
                "commands": ["bash", "python3"],
                "url": f"{SOURCE}/releases/download/{TAG}/{archive.name}",
                "sha256": sha256,
            }],
        }],
    }
    descriptor = args.output / "corebrum-app.json"
    descriptor.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(archive)
    print(descriptor)


if __name__ == "__main__":
    main()
