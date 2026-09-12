from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _values(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()] if text else default, dtype=float)


def _rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
        [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
        [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
    ])


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


class KinematicModel:
    """Small NumPy URDF FK/DLS-IK implementation for simulation and validation."""

    def __init__(self, urdf: Path):
        root = ET.parse(urdf).getroot()
        self.by_child: dict[str, Joint] = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            limit = element.find("limit")
            axis = element.find("axis")
            joint = Joint(
                name=element.get("name", ""),
                kind=element.get("type", "fixed"),
                parent=element.find("parent").get("link"),
                child=element.find("child").get("link"),
                xyz=_values(origin.get("xyz") if origin is not None else None),
                rpy=_values(origin.get("rpy") if origin is not None else None),
                axis=_values(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
                lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
                upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
            )
            self.by_child[joint.child] = joint

    def chain(self, base: str, target: str) -> list[Joint]:
        chain: list[Joint] = []
        current = target
        while current != base:
            if current not in self.by_child:
                raise ValueError(f"{target!r} is not below {base!r} in the URDF")
            joint = self.by_child[current]
            chain.append(joint)
            current = joint.parent
        return list(reversed(chain))

    def movable(self, base: str, target: str) -> list[Joint]:
        return [joint for joint in self.chain(base, target) if joint.kind in ("revolute", "continuous", "prismatic")]

    def forward(self, base: str, target: str, positions: dict[str, float]) -> np.ndarray:
        transform = np.eye(4)
        for joint in self.chain(base, target):
            rotation = _rpy(joint.rpy)
            translation = joint.xyz.copy()
            value = float(positions.get(joint.name, 0.0))
            if joint.kind in ("revolute", "continuous"):
                rotation = rotation @ _axis_angle(joint.axis, value)
            elif joint.kind == "prismatic":
                translation = translation + rotation @ (joint.axis * value)
            transform = transform @ _transform(rotation, translation)
        return transform

    def solve_position(
        self,
        base: str,
        target: str,
        goal: np.ndarray,
        initial: dict[str, float] | None = None,
        tolerance: float = 0.01,
        iterations: int = 350,
    ) -> tuple[dict[str, float], float, bool]:
        joints = self.movable(base, target)
        q = np.array([float((initial or {}).get(joint.name, _default(joint))) for joint in joints])

        def values(vector):
            return {joint.name: float(vector[index]) for index, joint in enumerate(joints)}

        damping = 0.025
        for _ in range(iterations):
            point = self.forward(base, target, values(q))[:3, 3]
            error = np.asarray(goal, dtype=float) - point
            norm = float(np.linalg.norm(error))
            if norm <= tolerance:
                return values(q), norm, True
            jacobian = np.zeros((3, len(joints)))
            for column, joint in enumerate(joints):
                delta = 1e-4
                perturbed = q.copy()
                perturbed[column] += delta
                changed = self.forward(base, target, values(perturbed))[:3, 3]
                jacobian[:, column] = (changed - point) / delta
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * damping * np.eye(3), error
            )
            update = np.clip(update, -0.12, 0.12)
            q = q + 0.65 * update
            for index, joint in enumerate(joints):
                if joint.lower is not None:
                    q[index] = max(q[index], joint.lower)
                if joint.upper is not None:
                    q[index] = min(q[index], joint.upper)
        error = float(np.linalg.norm(np.asarray(goal) - self.forward(base, target, values(q))[:3, 3]))
        return values(q), error, error <= tolerance


def _default(joint: Joint) -> float:
    if joint.kind == "prismatic" and joint.lower is not None:
        return max(joint.lower, -0.85)
    if joint.lower is not None and joint.upper is not None and not (joint.lower <= 0 <= joint.upper):
        return (joint.lower + joint.upper) / 2.0
    return 0.0

