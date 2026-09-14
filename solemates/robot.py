"""
solemates/robot.py

Robot backends for Solemates.

SimulatedRobot  — validates IK, tracks state, and drives the 3D Rerun
                  visualization on every command.
BracketBotRobot — guarded placeholder; wire after organizers provide the SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

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
    """
    Safe task-level backend.

    Validates workspace bounds, solves IK, tracks held socks, and
    emits every action to both the event log and the 3D Rerun visualization.
    """

    def __init__(
        self,
        bounds: list[float],
        emit: Callable[[RunState, str, dict], None],
        urdf: Optional[Path] = None,
        viz=None,          # solemates.visualization module (injected to avoid circular)
        workspace: Optional[dict] = None,
        transform=None,    # PixelTableTransform — needed for sock 3D placement
    ):
        self.bounds    = bounds
        self.emit      = emit
        self.held: dict[str, int | None] = {"left": None, "right": None}
        self.model     = KinematicModel(urdf) if urdf else None
        self.joints: dict[str, dict[str, float]] = {"left": {}, "right": {}}

        # Visualization support (optional — gracefully skipped when None)
        self._viz       = viz
        self._workspace = workspace
        self._transform = transform
        self._step      = 0        # mirrors app.step for viz timestamps

        # Current EE positions for trail + sock-carry logging
        self._ee_pos: dict[str, tuple[float, float, float]] = {
            "left":  (0.0, 0.0, 0.0),
            "right": (0.0, 0.0, 0.0),
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def move_end_effector(self, arm: str, pose: Pose) -> None:
        if arm not in self.held:
            raise ValueError(f"Unknown arm: {arm}")

        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin <= pose.x <= xmax and ymin <= pose.y <= ymax and pose.z >= 0):
            raise ValueError(f"Unsafe pose outside workspace: {pose}")

        data = {"arm": arm, "pose": asdict(pose)}
        ik_error: Optional[float] = None

        if self.model:
            solution, error, solved = self.model.solve_position(
                "arm_base",
                f"{arm}_eef",
                np.array([pose.x, pose.y, pose.z]),
                initial=self.joints[arm],
                tolerance=0.015,
            )
            if not solved:
                raise RuntimeError(
                    f"IK could not reach {pose} with {arm} arm; error={error:.3f} m"
                )
            self.joints[arm] = solution
            ik_error         = error
            data.update({"ik_error_m": error, "joints": solution})

            # --- 3D visualization: joint angles ---
            self._viz_joint_state(arm, solution)

        # Track EE position
        self._ee_pos[arm] = (pose.x, pose.y, pose.z)

        # Emit to event log
        self.emit(RunState.VERIFY, f"Move {arm} end effector", data)
        self._step += 1

        # --- 3D visualization: EE position + trail ---
        self._viz_ee_move(arm, pose, ik_error)

        # --- 3D visualization: move sock with arm if held ---
        sock_id = self.held[arm]
        if sock_id is not None:
            self._viz_sock_carried(sock_id, arm)

    def set_gripper(self, arm: str, closed: bool) -> None:
        self.emit(
            RunState.VERIFY,
            f"{'Close' if closed else 'Open'} {arm} gripper",
            {"arm": arm},
        )
        self._step += 1
        if self._viz:
            self._viz.log_gripper(arm, closed, self._step)

    def attach(self, arm: str, sock_id: int) -> None:
        if self.held[arm] is not None:
            raise RuntimeError(f"{arm} arm is already holding a sock")
        self.held[arm] = sock_id

    def release(self, arm: str) -> int:
        sock_id = self.held[arm]
        if sock_id is None:
            raise RuntimeError(f"{arm} arm is not holding a sock")
        self.held[arm] = None

        # Drop sock back to table surface at current EE xy
        if self._viz and self._workspace:
            x, y, _ = self._ee_pos[arm]
            table_z = float(self._workspace.get("table_z_m", 0.0))
            import math
            self._viz_sock_at(sock_id, x, y, table_z + 0.005, 0.0)
        return sock_id

    def stop(self) -> None:
        self.emit(RunState.DONE, "Simulation stopped", {})

    # ------------------------------------------------------------------
    # Internal visualization helpers (all no-ops when self._viz is None)
    # ------------------------------------------------------------------

    def _viz_ee_move(self, arm: str, pose: Pose, ik_error: Optional[float]) -> None:
        if self._viz is None:
            return
        self._viz.log_ee_move(arm, pose, self._step, ik_error)

    def _viz_joint_state(self, arm: str, joints: dict[str, float]) -> None:
        if self._viz is None:
            return
        self._viz.log_joint_state(arm, joints, self._step)

    def _viz_sock_carried(self, sock_id: int, arm: str) -> None:
        if self._viz is None:
            return
        self._viz.log_sock_carried(sock_id, arm, self._ee_pos[arm], self._step)

    def _viz_sock_at(
        self,
        sock_id: int,
        x: float, y: float, z: float,
        angle: float,
    ) -> None:
        if self._viz is None:
            return
        import math
        rr = getattr(self._viz, "_rr", None)
        if rr is None:
            return
        rr.set_time_sequence("step", self._step)
        from .visualization import _sock_color
        color = list(_sock_color(sock_id))
        color[3] = 200
        rr.log(
            f"world/socks/s{sock_id}",
            rr.Boxes3D(
                centers=[[x, y, z]],
                half_sizes=[[0.06, 0.035, 0.003]],
                rotation_axis_angles=[
                    rr.datatypes.RotationAxisAngle(
                        axis=[0, 0, 1],
                        angle=rr.datatypes.Angle(rad=angle),
                    )
                ],
                colors=[tuple(color)],
                labels=[f"S{sock_id}"],
            ),
        )


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