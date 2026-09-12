from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .kinematics import KinematicModel
from .models import Event, Pose, RunState


class Robot(ABC):
    @abstractmethod
    def move_end_effector(self, arm: str, pose: Pose) -> None: ...

    @abstractmethod
    def set_gripper(self, arm: str, closed: bool) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class SimulatedRobot(Robot):
    """Safe task-level backend. It validates commands and records every action."""

    def __init__(
        self,
        bounds: list[float],
        emit: Callable[[RunState, str, dict], None],
        urdf: Path | None = None,
    ):
        self.bounds = bounds
        self.emit = emit
        self.held: dict[str, int | None] = {"left": None, "right": None}
        self.model = KinematicModel(urdf) if urdf else None
        self.joints: dict[str, dict[str, float]] = {"left": {}, "right": {}}

    def move_end_effector(self, arm: str, pose: Pose) -> None:
        if arm not in self.held:
            raise ValueError(f"Unknown arm: {arm}")
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin <= pose.x <= xmax and ymin <= pose.y <= ymax and pose.z >= 0):
            raise ValueError(f"Unsafe pose outside workspace: {pose}")
        data = {"arm": arm, "pose": asdict(pose)}
        if self.model:
            solution, error, solved = self.model.solve_position(
                "arm_base",
                f"{arm}_eef",
                np.array([pose.x, pose.y, pose.z]),
                initial=self.joints[arm],
                tolerance=0.015,
            )
            if not solved:
                raise RuntimeError(f"IK could not reach {pose} with {arm} arm; error={error:.3f} m")
            self.joints[arm] = solution
            data.update({"ik_error_m": error, "joints": solution})
        self.emit(RunState.VERIFY, f"Move {arm} end effector", data)

    def set_gripper(self, arm: str, closed: bool) -> None:
        self.emit(RunState.VERIFY, f"{'Close' if closed else 'Open'} {arm} gripper", {"arm": arm})

    def attach(self, arm: str, sock_id: int) -> None:
        if self.held[arm] is not None:
            raise RuntimeError(f"{arm} arm is already holding a sock")
        self.held[arm] = sock_id

    def release(self, arm: str) -> int:
        sock_id = self.held[arm]
        if sock_id is None:
            raise RuntimeError(f"{arm} arm is not holding a sock")
        self.held[arm] = None
        return sock_id

    def stop(self) -> None:
        self.emit(RunState.DONE, "Simulation stopped", {})


class BracketBotRobot(Robot):
    """Guarded placeholder: wire this only after organizers provide the SDK."""

    MESSAGE = (
        "Physical execution is intentionally disabled: implement BracketBotRobot "
        "with the organizer-provided controller, feedback, and emergency stop first."
    )

    def move_end_effector(self, arm: str, pose: Pose) -> None:
        raise RuntimeError(self.MESSAGE)

    def set_gripper(self, arm: str, closed: bool) -> None:
        raise RuntimeError(self.MESSAGE)

    def stop(self) -> None:
        raise RuntimeError(self.MESSAGE)
