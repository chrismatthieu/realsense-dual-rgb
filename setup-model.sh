#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash setup-model.sh /path/to/model_best_bp2_serialize.pth\n' >&2
    printf '   or: bash setup-model.sh /path/to/fast_foundationstereo.onnx\n' >&2
    exit 2
fi

if [[ ! -f "$1" ]]; then
    printf 'Model file not found: %s\n' "$1" >&2
    printf 'Download the NVIDIA checkpoint or copy your ONNX export to the Thor, then pass its actual local path. See README.md.\n' >&2
    exit 1
fi
artifact="$(realpath -e -- "$1")"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export_python="${FFS_EXPORT_PYTHON:-python3}"
upstream="$root/Fast-FoundationStereo"
models="$root/models"
target="$models/fast_foundationstereo.onnx"
mkdir -p -- "$models"

case "$artifact" in
    *.pth)
        if ! "$export_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; import onnx'; then
            printf 'Use FFS_EXPORT_PYTHON to select the NVIDIA export environment with CUDA PyTorch and ONNX. See README.md.\n' >&2
            exit 1
        fi
        if [[ ! -f "$upstream/scripts/make_single_onnx.py" ]]; then
            git clone https://github.com/NVlabs/Fast-FoundationStereo.git "$upstream"
        fi
        "$export_python" "$root/export_model.py" "$upstream" "$artifact" "$target"
        ;;
    *.onnx)
        if [[ "$artifact" != "$target" ]]; then
            cp -- "$artifact" "$target"
        fi
        ;;
    *)
        printf 'Expected a .pth checkpoint or a single-file .onnx export.\n' >&2
        exit 2
        ;;
esac

python3 -m venv --system-site-packages "$root/.venv"
"$root/.venv/bin/python" -m pip install onnxruntime
cd -- "$root"
"$root/.venv/bin/python" - <<'PY'
from pathlib import Path
import numpy as np
from model import FastFoundationStereo

model = FastFoundationStereo(
    Path("models/fast_foundationstereo.onnx"), (640, 384), "CPUExecutionProvider", None
)
image = np.full((384, 640, 3), 128, dtype=np.uint8)
if not np.isfinite(model(image, image)).all():
    raise SystemExit("Model smoke test returned non-finite disparity")
print("Fast-FoundationStereo ONNX model loaded and passed a synthetic inference check.")
PY
