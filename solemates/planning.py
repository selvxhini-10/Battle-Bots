from __future__ import annotations

import math

import numpy as np

from .models import GraspPlan, Pose, SockObservation


class PixelTableTransform:

    def __init__(self, image_size: tuple[int, int], bounds: tuple[float, float, float, float],
                 homography=None, calibration_image_size=None):
        self.width, self.height = image_size
        self.homography = None
        if homography is not None:
            matrix = np.asarray(homography, dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("Camera homography must be a finite 3x3 matrix")
            if np.linalg.matrix_rank(matrix) != 3:
                raise ValueError("Camera homography must be invertible")
            if calibration_image_size is not None and tuple(calibration_image_size) != tuple(image_size):
                raise ValueError("Image size differs from calibration; use the calibrated camera resolution")
            self.homography = matrix / np.max(np.abs(matrix))
        self.xmin, self.xmax, self.ymin, self.ymax = bounds

    def point(self, pixel: tuple[float, float]) -> tuple[float, float]:
        u, v = pixel
        if self.homography is not None:
            mapped = self.homography @ np.array([u, v, 1.0])
            if not np.isfinite(mapped).all() or abs(mapped[2]) < 1e-12:
                raise ValueError("Pixel cannot be mapped to a finite table position")
            return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])
        x = self.xmin + np.clip(u / self.width, 0, 1) * (self.xmax - self.xmin)
        y = self.ymax - np.clip(v / self.height, 0, 1) * (self.ymax - self.ymin)
        return float(x), float(y)


    def angle(self, pixel: tuple[float, float], angle_rad: float) -> float:
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
        if self.homography is None:
            tangent = direction * [(self.xmax - self.xmin) / self.width,
                                   -(self.ymax - self.ymin) / self.height]
        else:
            mapped = self.homography @ np.array([*pixel, 1.0])
            self.point(pixel)  # Validate the projective denominator.
            derivative = self.homography[:, :2] @ direction
            tangent = (derivative[:2] * mapped[2] - mapped[:2] * derivative[2]) / mapped[2] ** 2
        return math.atan2(float(tangent[1]), float(tangent[0]))


def make_grasp_plan(
    sock: SockObservation,
    arm: str,
    transform: PixelTableTransform,
    workspace: dict,
) -> GraspPlan:
    x, y = transform.point(sock.grasp_px)
    grasp_z = float(workspace["grasp_height_m"])
    safe_z = float(workspace["safe_height_m"])
    yaw = _normalize_yaw(transform.angle(sock.grasp_px, sock.angle_rad) + math.pi / 2.0)
    return GraspPlan(
        sock_id=sock.id,
        arm=arm,
        grasp=Pose(x, y, grasp_z, yaw),
        pregrasp=Pose(x, y, safe_z, yaw),
        lift=Pose(x, y, safe_z, yaw),
    )


def destination_pose(origin: list[float], slot: int, workspace: dict) -> Pose:
    spacing = 0.08
    return Pose(
        float(origin[0]) + spacing * (slot % 2),
        float(origin[1]) + spacing * (slot // 2),
        float(workspace["grasp_height_m"]),
        0.0,
    )


def _normalize_yaw(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value

