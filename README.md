# D405 Fast-FoundationStereo

Stream both RGB eyes of a USB RealSense D405 into NVIDIA Fast-FoundationStereo and
convert disparity into metric depth. NVIDIA recommends the Fast model for live
applications and the earlier FoundationStereo for higher-accuracy offline work.
See the [NVIDIA project and export instructions](https://github.com/NVlabs/Fast-FoundationStereo).

## Launch menu

Close RealSense Viewer or any running demo, then run this from a Thor desktop
terminal:

```bash
cd ~/Projects/dual-rgb
python3 launch.py
```

Choose a numbered preset for Dual RGB, D405 depth, ONNX CPU, CUDA PyTorch,
or TensorRT FP16/FP32. Use `s` to toggle sensor depth and fusion on model presets,
`m` to toggle median filtering, and `o` to save the latest frame. Press Q or Esc
in a preview to return to the menu. The FP16 preset is available for comparison.
On this Thor, the FP32 engine produced a usable preview during testing.

## Corebrum Apps packaging

The release target is Ubuntu 24.04 ARM64 on Jetson AGX Thor with a USB D405.
Place the NVIDIA checkout, model exports and Python environments under
`~/Projects/dual-rgb` using the steps below. For a different location, set
`DUAL_RGB_ASSETS` to that directory before launching the app.

With an authorized `LICENSE` at the repository root, run
`python3 package-app.py --license SPDX_ID` using the license's SPDX identifier.
The archive and its checksum-bearing `corebrum-app.json` are written to
`output/marketplace`. Attach both to the public GitHub release
`corebrum-app-v1.0.0` and submit the release URL from the
[Corebrum publish page](https://marketplace.corebrum.com/publish/app).
The release archive contains the launcher and application code. The camera
firmware, NVIDIA model files and Python environments are installed separately.

## Verify camera capture

Run from a Thor desktop terminal with RealSense Viewer and other camera clients
closed. The D405 needs firmware that exposes both RGB streams. The project defaults
to RGB8 at 848×480 and 30 fps. It uses librealsense's returned stream indices and
calibration to rectify the stereo pair for the model.

```bash
cd ~/Projects/dual-rgb
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 live.py --camera-only
```

This needs Python with `pyrealsense2`, NumPy and OpenCV with desktop GUI support.
Those imports are available in the Thor's system Python. Use a direct USB 3
connection. If profile resolution fails, check which profiles appear for both
eyes in Viewer and set `--capture-size WIDTH HEIGHT --fps FPS`.

## Export NVIDIA's model and run it

For CUDA inference directly from NVIDIA's checkpoint, use the CUDA PyTorch
environment with NVIDIA's Python dependencies and access to the system
`pyrealsense2` binding. The checkpoint runs without ONNX conversion.

```bash
LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  .ffs-export/bin/python live.py \
  --model "$HOME/Downloads/model_best_bp2_serialize.pth" \
  --upstream Fast-FoundationStereo --sensor-depth
```

For a faster preview, try `--iterations 4`. Reducing refinement can trade
accuracy for speed. The default is eight iterations.

`bash setup-model.sh /path/to/fast_foundationstereo.onnx` installs a 640×384
single-file export, creates the runtime environment and runs a synthetic
inference check. You can also pass NVIDIA's `.pth` checkpoint to export it
first. Set `FFS_EXPORT_PYTHON` to the Python executable in your CUDA export
environment. NVIDIA's upstream Dockerfile uses x86-64 dependencies, so on
Thor use an ARM64 CUDA environment or export on an x86-64 NVIDIA GPU machine
and copy the ONNX file to the Thor.

1. Clone NVIDIA's repository and download its research checkpoint
   `23-36-37/model_best_bp2_serialize.pth` using the **Weights and Trade-off**
   section of the [upstream README](https://github.com/NVlabs/Fast-FoundationStereo#weights-and-trade-off).
   The commands below use NVIDIA's research checkpoint and single-export format.
   Use a checkpoint from NVIDIA that you trust. The exporter deserializes Python objects.
2. Set up NVIDIA's [CUDA-enabled export environment](https://github.com/NVlabs/Fast-FoundationStereo#environment-setup)
   on a compatible NVIDIA GPU machine or container. Use a Jetson-compatible
   PyTorch build if exporting on the Thor.

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
cd Fast-FoundationStereo
# Place NVIDIA's downloaded checkpoint under weights/23-36-37/.
python scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path ../models --height 384 --width 640 --valid_iters 8 --max_disp 192
cd ..
```

For an initial smoke test, install ONNX Runtime in an environment that also
has the camera dependencies. A system-site-packages virtual environment can use
the Thor's installed camera bindings. CPU inference may be slow.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install onnxruntime
.venv/bin/python live.py --model models/fast_foundationstereo.onnx
```

For live GPU throughput, use the Thor's compatible
[TensorRT ARM64 installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
and build the engine on the Thor. The engine path uses NVIDIA's runner from the
upstream checkout. Run it in an environment with the upstream Python
dependencies, CUDA-enabled PyTorch, TensorRT Python bindings and the camera
dependencies above. A different inference size requires exporting at that size
and passing `--model-size WIDTH HEIGHT`. Match `--max-disparity` to the export.

```bash
trtexec --onnx=models/fast_foundationstereo.onnx \
  --saveEngine=models/fast_foundationstereo.engine --fp16
python3 live.py --model models/fast_foundationstereo.engine \
  --upstream Fast-FoundationStereo
```

On the Thor with L4T R39.2, the NVIDIA Jetson apt repository provides both
the engine builder and Python bindings:

```bash
sudo apt install libnvinfer-bin python3-libnvinfer
trtexec_path="$(dpkg -L libnvinfer-bin | awk '/\/trtexec$/ {print; exit}')"
"$trtexec_path" --onnx=models/fast_foundationstereo.onnx \
  --saveEngine=models/fast_foundationstereo.fp16.engine --fp16

LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  .ffs-export/bin/python live.py --model models/fast_foundationstereo.fp16.engine \
  --upstream Fast-FoundationStereo --sensor-depth
```

The existing ONNX export fixes the engine at 640×384 with eight refinement
iterations. To compare throughput at the same settings, run the PyTorch
checkpoint at 640×384 with eight iterations as well.

## Compare and stream depth

Add `--sensor-depth` for side-by-side estimates from the model, the D405 and
their fusion. Fast-FoundationStereo consumes only RGB. The application
deprojects the D405 depth, transforms those points to the rectified left camera,
and keeps the nearest point at each projected pixel. Fusion fills model holes
with projected sensor depth and averages pixels that agree within 1 cm or 5%,
whichever is larger. On disagreement it keeps the stereo prediction.

Add `--output output/latest.npz` to publish a continuous latest-frame file.
Readers can reopen the file and use `frame_number` or `timestamp_ms` to tell frames
apart. The file is replaced atomically and includes rectified RGB, calibration
(`k` and `baseline_m`), and `stereo_depth_m`. With `--sensor-depth`, it also
includes `sensor_depth_m` and `fused_depth_m`. Depth arrays are float32 meters
along the rectified left camera's optical axis. Zero marks unavailable or
out-of-range depth. The default display/output range is 0.07–0.75 m. Press Q
or Esc to exit. Use `--no-preview` for headless output and `--frames 1` for a
single-frame check.

Add `--median-depth` to reduce isolated speckle in the model depth before
preview and fusion. The 3×3 median filter preserves invalid pixels. With
`--output`, the unfiltered estimate is also saved as `stereo_depth_raw_m`.

For glossy parts, compare all three estimates with measured distances at the
same viewpoints. Learned stereo can fill some gaps, but reflections shift
between eyes and can create false matches. Both estimates can be wrong at the
same pixel. The fusion is a heuristic and has no calibrated confidence score.
