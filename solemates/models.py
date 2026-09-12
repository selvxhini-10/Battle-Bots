from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RunState(str, Enum):
    SCAN = "scan"
    MATCH = "match"
    PICK_FIRST = "pick_first"
    PICK_SECOND = "pick_second"
    REUNITE = "reunite"
    PLACE_PAIR = "place_pair"
    HANDLE_SINGLE = "handle_single"
    VERIFY = "verify"
    CELEBRATE = "celebrate"
    DONE = "done"
    RECOVER = "recover"


@dataclass
class SockObservation:
    id: int
    mask: np.ndarray = field(repr=False)
    centroid_px: tuple[float, float]
    angle_rad: float
    area_px: float
    bbox_px: tuple[int, int, int, int]
    color_hist: np.ndarray = field(repr=False)
    pattern: np.ndarray = field(repr=False)
    shape: np.ndarray = field(repr=False)
    grasp_px: tuple[float, float]
    embedding: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "centroid_px": list(self.centroid_px),
            "grasp_px": list(self.grasp_px),
            "angle_rad": self.angle_rad,
            "area_px": self.area_px,
            "bbox_px": list(self.bbox_px),
            "has_embedding": self.embedding is not None,
        }


@dataclass(frozen=True)
class PairMatch:
    first_id: int
    second_id: int
    score: float
    components: dict[str, float]


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass(frozen=True)
class GraspPlan:
    sock_id: int
    arm: str
    grasp: Pose
    pregrasp: Pose
    lift: Pose


@dataclass
class Event:
    step: int
    state: RunState
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value
