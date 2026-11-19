"""Calibration, metric depth and sensor-depth fusion for a rectified stereo pair."""

import cv2
import numpy as np


def intrinsics_matrix(intrinsics):
    return np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ], dtype=np.float64)


def extrinsics_matrix(extrinsics):
    rotation = np.asarray(extrinsics.rotation).reshape(3, 3, order="F")
    return rotation, np.asarray(extrinsics.translation)


def rectification_map(rs, intrinsics, rotation, projection, size):
    # Project rays from the virtual rectified camera into the original sensor.
    height, width = size[1], size[0]
    y, x = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    rays = pixels @ np.linalg.inv(projection[:, :3]).T @ rotation
    x, y = rays[..., 0] / rays[..., 2], rays[..., 1] / rays[..., 2]
    model = intrinsics.model
    if model != rs.distortion.none:
        if model not in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady,
                         rs.distortion.inverse_brown_conrady):
            raise ValueError(f"Unsupported RealSense color distortion model {model}")
        k1, k2, p1, p2, k3 = intrinsics.coeffs
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        rx, ry = x * radial, y * radial
        # Match rs2_project_point_to_pixel for the RealSense Brown variants.
        if model != rs.distortion.brown_conrady:
            x, y = rx, ry
        x, y = (rx + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                ry + 2 * p2 * x * y + p1 * (r2 + 2 * y * y))
    return ((x * intrinsics.fx + intrinsics.ppx).astype(np.float32),
            (y * intrinsics.fy + intrinsics.ppy).astype(np.float32))


class StereoGeometry:
    def __init__(self, rs, left, right, size):
        left_intrinsics, right_intrinsics = left.get_intrinsics(), right.get_intrinsics()
        rotation, translation = extrinsics_matrix(left.get_extrinsics_to(right))
        source_size = (left.width(), left.height())
        if source_size != (right.width(), right.height()):
            raise ValueError("The two RGB streams must have matching resolutions")
        zeros = np.zeros(5)
        r_left, r_right, p_left, p_right, *_ = cv2.stereoRectify(
            intrinsics_matrix(left_intrinsics), zeros,
            intrinsics_matrix(right_intrinsics), zeros,
            source_size, rotation, translation.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0, newImageSize=size,
        )
        if not np.isfinite(p_right).all() or p_right[0, 3] >= 0 or abs(p_right[1, 3]) > 1e-6:
            raise ValueError("Expected a horizontal left-to-right RealSense stereo baseline")
        self.left_rotation = r_left
        self.k = p_left[:, :3].copy()
        self.baseline_m = -p_right[0, 3] / p_right[0, 0]
        self.left_map = rectification_map(rs, left_intrinsics, r_left, p_left, size)
        self.right_map = rectification_map(rs, right_intrinsics, r_right, p_right, size)
        self.left_valid = self.valid_map(self.left_map, source_size)
        self.right_valid = self.valid_map(self.right_map, source_size)
        self.y, self.x = np.indices((size[1], size[0]), dtype=np.float32)

    @staticmethod
    def valid_map(maps, size):
        x, y = maps
        width, height = size
        return (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)

    def rectify(self, left_rgb, right_rgb):
        return (
            cv2.remap(left_rgb, *self.left_map, cv2.INTER_LINEAR),
            cv2.remap(right_rgb, *self.right_map, cv2.INTER_LINEAR),
        )

    def disparity_to_depth(self, disparity, max_disparity):
        if disparity.shape != self.x.shape:
            raise ValueError(f"Expected disparity shape {self.x.shape}, got {disparity.shape}")
        valid = np.isfinite(disparity) & (disparity > 0) & (disparity < max_disparity)
        right_x = self.x - np.where(valid, disparity, 0)
        right_mask = cv2.remap(self.right_valid.astype(np.uint8), right_x, self.y,
                               cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        valid &= (right_x >= 0) & self.left_valid & (right_mask > 0)
        depth = np.zeros_like(disparity, dtype=np.float32)
        np.divide(self.k[0, 0] * self.baseline_m, disparity, out=depth, where=valid)
        return depth

    def project_sensor_depth(self, points, depth_to_left):
        rotation, translation = extrinsics_matrix(depth_to_left)
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
        points = (points @ rotation.T + translation) @ self.left_rotation.T
        points = points[points[:, 2] > 0]
        z = points[:, 2]
        pixels = points @ self.k.T
        x, y = np.rint(pixels[:, :2] / z[:, None]).astype(np.int32).T
        height, width = self.x.shape
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        projected = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(projected, y[inside] * width + x[inside], z[inside])
        projected[~np.isfinite(projected)] = 0
        return projected.reshape(height, width)


def fuse_depth(stereo, sensor):
    """Fill stereo holes and average agreeing depths. Retain stereo on disagreement."""
    valid_stereo = np.isfinite(stereo) & (stereo > 0)
    valid_sensor = np.isfinite(sensor) & (sensor > 0)
    # A small absolute floor plus a range-dependent tolerance.
    agree = valid_stereo & valid_sensor & (abs(stereo - sensor) <= np.maximum(0.01, 0.05 * stereo))
    fused = np.where(valid_stereo, stereo, np.where(valid_sensor, sensor, 0))
    fused[agree] = (stereo[agree] + sensor[agree]) * 0.5
    return fused
