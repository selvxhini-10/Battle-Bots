from __future__ import annotations

import math

import numpy as np

from .models import GraspPlan, Pose, SockObservation


class PixelTableTransform:
    """Planar image-to-table mapping; defaults to configured rectangular bounds."""

    def __init__(self, image_size: tuple[int, int], bounds: tuple[float, float, float, float]):
        self.width, self.height = image_size
        self.xmin, self.xmax, self.ymin, self.ymax = bounds

    def point(self, pixel: tuple[float, float]) -> tuple[float, float]:
        u, v = pixel
        x = self.xmin + np.clip(u / self.width, 0, 1) * (self.xmax - self.xmin)
        y = self.ymax - np.clip(v / self.height, 0, 1) * (self.ymax - self.ymin)
        return float(x), float(y)


def make_grasp_plan(
    sock: SockObservation,
    arm: str,
    transform: PixelTableTransform,
    workspace: dict,
) -> GraspPlan:
    x, y = transform.point(sock.grasp_px)
    grasp_z = float(workspace["grasp_height_m"])
    safe_z = float(workspace["safe_height_m"])
    yaw = _normalize_yaw(sock.angle_rad + math.pi / 2.0)
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

