"""Stream D405 Dual RGB into Fast-FoundationStereo and publish metric depth."""

import argparse
import logging
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from model import FastFoundationStereo
from stereo import StereoGeometry, fuse_depth


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", type=Path, help="NVIDIA PyTorch checkpoint, ONNX, or TensorRT engine")
    mode.add_argument("--camera-only", action="store_true", help="Preview the rectified pair")
    parser.add_argument("--upstream", type=Path, help="NVIDIA checkout for PyTorch or TensorRT")
    parser.add_argument("--iterations", type=int, default=8, help="PyTorch refinement iterations")
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime provider")
    parser.add_argument("--serial")
    parser.add_argument("--capture-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        default=(848, 480))
    parser.add_argument("--model-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        default=(640, 384))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-disparity", type=int, default=192,
                        help="Match max_disp used at model export (model pixels)")
    parser.add_argument("--near", type=float, default=0.07, help="Minimum output distance in meters")
    parser.add_argument("--far", type=float, default=0.75, help="Maximum output distance in meters")
    parser.add_argument("--sensor-depth", action="store_true",
                        help="Also capture D405 Z16, register it and compute fused depth")
    parser.add_argument("--median-depth", action="store_true",
                        help="Reduce model depth speckle with a 3x3 median filter")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--output", type=Path, help="Atomically replace this latest-frame .npz")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N outputs (0 runs continuously)")
    args = parser.parse_args()
    if min(*args.capture_size, *args.model_size, args.fps, args.max_disparity, args.iterations) <= 0:
        parser.error("sizes, fps and max-disparity must be positive")
    if args.model and any(value % 32 for value in args.model_size):
        parser.error("model size must be divisible by 32 in both dimensions")
    if not 0 < args.near < args.far or args.frames < 0:
        parser.error("require 0 < near < far and frames >= 0")
    if args.output and args.output.suffix != ".npz":
        parser.error("--output must point to a .npz file")
    return args


def export_frame(path, fields):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     delete=False) as staging:
        temporary = Path(staging.name)
        try:
            np.savez(staging, **fields)
            staging.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def in_range(depth, near, far):
    return np.where((depth >= near) & (depth <= far), depth, 0).astype(np.float32)


def median_depth(depth):
    median = cv2.medianBlur(depth, 3)
    return np.where((depth > 0) & (median > 0), median, depth)


def depth_panel(name, depth, near, far):
    ramp = np.clip((far - depth) / (far - near), 0, 1)
    panel = cv2.applyColorMap((255 * ramp).astype(np.uint8), cv2.COLORMAP_TURBO)
    panel[depth <= 0] = 0
    cv2.putText(panel, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return panel


def stream(args):
    import pyrealsense2 as rs

    width, height = args.capture_size
    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    for index in (0, 1):
        cfg.enable_stream(rs.stream.color, index, width, height, rs.format.rgb8, args.fps)
    if args.sensor_depth:
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)

    # Resolve first and require both exact color indices. A stream-index 0
    # request can behave as a wildcard during librealsense profile selection.
    try:
        profile = cfg.resolve(rs.pipeline_wrapper(pipeline))
        color = {item.stream_index(): item.as_video_stream_profile() for item in
                 profile.get_streams() if item.stream_type() == rs.stream.color}
        if set(color) != {0, 1}:
            raise RuntimeError(f"Resolved color indices {sorted(color)}")
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not resolve both RGB8 streams at {width}x{height} @ {args.fps}. "
            "Close Viewer and verify Color and Color 1 at this profile. "
            "When diagnosing combined modes, try without --sensor-depth. "
            f"RealSense reported: {error}"
        ) from error

    size = tuple(args.model_size)
    model = None if args.camera_only else FastFoundationStereo(
        args.model, size, args.provider, args.upstream, args.max_disparity, args.iterations
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    geometry = StereoGeometry(rs, color[0], color[1], size)
    logging.info("Rectified %dx%d, focal length %.1f px, baseline %.4f m",
                 *size, geometry.k[0, 0], geometry.baseline_m)

    profile = pipeline.start(cfg)
    try:
        depth_to_left = None
        pointcloud = None
        if args.sensor_depth:
            depth_to_left = profile.get_stream(rs.stream.depth).get_extrinsics_to(color[0])
            pointcloud = rs.pointcloud()
        count = 0
        last_report = time.monotonic()
        last_count = 0
        last_pair_time = last_report
        reported_empty_depth = False
        while args.frames == 0 or count < args.frames:
            frames = pipeline.wait_for_frames(5000)
            colors = {frame.profile.stream_index(): frame for frame in frames
                      if frame.profile.stream_type() == rs.stream.color}
            sensor_frame = frames.get_depth_frame() if args.sensor_depth else None
            if colors.keys() != {0, 1} or (args.sensor_depth and not sensor_frame):
                if time.monotonic() - last_pair_time > 5:
                    raise RuntimeError("No complete RGB pair (and requested depth) for 5 seconds")
                continue

            left_frame, right_frame = colors[0], colors[1]
            left, right = geometry.rectify(np.asanyarray(left_frame.get_data()),
                                           np.asanyarray(right_frame.get_data()))
            last_pair_time = time.monotonic()
            output = dict(left_rgb=left, right_rgb=right, k=geometry.k,
                          baseline_m=geometry.baseline_m, timestamp_ms=left_frame.get_timestamp(),
                          frame_number=left_frame.get_frame_number())
            panels = []
            if model is not None:
                disparity = model(left, right)
                depth = geometry.disparity_to_depth(disparity, args.max_disparity)
                output["stereo_depth_m"] = in_range(depth, args.near, args.far)
                empty_depth = not np.any(output["stereo_depth_m"])
                if empty_depth and not reported_empty_depth:
                    finite = disparity[np.isfinite(disparity)]
                    valid_depth = depth[depth > 0]
                    logging.warning(
                        "Model depth is empty. Finite disparity: %d/%d, positive disparity: %d, "
                        "projected depth: %d pixels, display range: %.3f–%.3f m",
                        finite.size, disparity.size, np.count_nonzero(finite > 0),
                        valid_depth.size, args.near, args.far,
                    )
                    if valid_depth.size:
                        logging.warning("Projected depth min/median/max: %.3f / %.3f / %.3f m",
                                        valid_depth.min(), np.median(valid_depth), valid_depth.max())
                reported_empty_depth = empty_depth
                if args.median_depth:
                    output["stereo_depth_raw_m"] = output["stereo_depth_m"]
                    output["stereo_depth_m"] = median_depth(output["stereo_depth_raw_m"])
                panels.append(("Fast-FoundationStereo", output["stereo_depth_m"]))
            if pointcloud is not None:
                points = pointcloud.calculate(sensor_frame)
                vertices = np.asanyarray(points.get_vertices(2)).reshape(-1, 3)
                depth = geometry.project_sensor_depth(vertices, depth_to_left)
                output["sensor_depth_m"] = in_range(depth, args.near, args.far)
                panels.append(("D405 depth", output["sensor_depth_m"]))
                if model is not None:
                    output["fused_depth_m"] = fuse_depth(output["stereo_depth_m"],
                                                        output["sensor_depth_m"])
                    panels.append(("Fused depth", output["fused_depth_m"]))
            if args.output:
                export_frame(args.output, output)
            count += 1
            now = time.monotonic()
            if now - last_report >= 2:
                logging.info("%.1f output fps", (count - last_count) / (now - last_report))
                last_count, last_report = count, now
            if not args.no_preview:
                images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in (left, right)]
                images.extend(depth_panel(name, depth, args.near, args.far) for name, depth in panels)
                for offset in range(0, len(images), 2):
                    row = images[offset:offset + 2]
                    cv2.imshow(f"D405 stereo {offset // 2 + 1}", np.concatenate(row, axis=1))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        pipeline.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        stream(arguments())
    except KeyboardInterrupt:
        pass
    except (ImportError, OSError, RuntimeError, ValueError, cv2.error) as error:
        raise SystemExit(str(error)) from error
