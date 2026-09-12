"""
bracketbot_robot.py — Hardware adapter for Solemates using BracketBot's bbos stack.

RUNS ON THE ROBOT (SSH in first). Requires bbos, pinocchio, and the arm daemons running.

Key facts learned from daemon.py / constants.py:
  - Joint names: lj0..lj6 (left arm), rj0..rj6 (right arm). j0 is the vertical linear stage.
  - 8 motors per arm; motor index 7 is the gripper (left_left_gripper / right_right_gripper)
  - IK solver: IKRust wrapping libhybrid_ik_lib.so, accessed via bbos Config.ik
  - IK input:  positions=[x,y,z], orientations=[qx,qy,qz,qw]  (6-DOF solve_pose)
  - IK output: 7 joint angles in URDF space (lj0..lj6)
  - Motor space ↔ URDF space: cfg.q2urdf() / cfg.urdf2q() + cfg.ik_sign
  - Commands sent via bbos Writer to "{arm_name}.ctrl" topic
  - State read via bbos Reader from "{arm_name}.state" topic  (pos, vel, torque, current, temp)
  - Gripper: motor index 7; left gripper_sign = -1 (mounted mirror-reversed)
  - Control rate: 150 Hz (dt = 1/150)

Usage:
  ssh into the robot, then:
    python bracketbot_robot.py --test-connection
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# bbos imports — only available on the robot
try:
    from bbos import Config, Reader, Type, Writer
    BBOS_AVAILABLE = True
except ImportError:
    BBOS_AVAILABLE = False
    print("[BracketBotRobot] bbos not found — running in offline/test mode only")

# Your Solemates models
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from solemates.models import Pose


# ── Gripper constants (from constants.py) ────────────────────────────────────
GRIPPER_OPEN_TURNS  = -0.10   # motor-space turns; tune on hardware
GRIPPER_CLOSE_TURNS = -0.60   # motor-space turns; tune on hardware
GRIPPER_MOVE_TIME   = 0.8     # seconds to wait for gripper to open/close

# ── Safe motion constants ─────────────────────────────────────────────────────
POSITION_SETTLE_TIME = 0.3    # seconds to wait after commanding a pose
POSITION_TOLERANCE   = 0.015  # meters; IK success threshold (matches bbos tolerances[0:3])
IK_TIMEOUT_S         = 2.0    # seconds before declaring IK failure


class BracketBotRobot:
    """
    Hardware adapter: translates Solemates Pose commands into bbos motor commands.

    Each arm runs as a separate bbos daemon (arm_left, arm_right).
    This class wraps both and provides the same interface as SimulatedRobot.
    """

    ARM_NAMES = {"left": "arm_left", "right": "arm_right"}
    EE_LINKS  = {"left": "left_eef",  "right": "right_eef"}

    def __init__(self):
        if not BBOS_AVAILABLE:
            raise RuntimeError(
                "bbos is not installed. Run this on the BracketBot (SSH in first)."
            )

        self._cfgs: dict[str, object] = {}
        self._writers: dict[str, object] = {}
        self._readers: dict[str, object] = {}

        for side, arm_name in self.ARM_NAMES.items():
            cfg = Config(arm_name)
            cfg.ik.init()
            self._cfgs[side] = cfg
            self._writers[side] = Writer(f"{arm_name}.ctrl", Type("arm_ctrl"))
            self._readers[side] = Reader(f"{arm_name}.state", keeptime=False)

        self.held: dict[str, int | None] = {"left": None, "right": None}
        print("[BracketBotRobot] Connected to arm_left and arm_right daemons.")

    # ── Public interface (matches SimulatedRobot) ─────────────────────────────

    def move_end_effector(self, arm: str, pose: Pose) -> None:
        """
        Move the end-effector to a Cartesian pose (x, y, z, yaw).
        Converts to joint angles via IK and sends to the arm daemon.
        """
        self._validate_arm(arm)
        cfg = self._cfgs[arm]

        # Convert yaw → quaternion (gripper pointing down, rotating around Z)
        quat = Rotation.from_euler("z", pose.yaw).as_quat()  # [qx, qy, qz, qw]

        # RelaxedIK solve_pose: positions=[x,y,z], orientations=[qx,qy,qz,qw]
        urdf_joints = cfg.ik.solve(
            [pose.x, pose.y, pose.z],
            quat.tolist(),
        )

        if urdf_joints is None or len(urdf_joints) == 0:
            raise RuntimeError(
                f"IK failed for {arm} arm target {pose}. "
                "Check workspace bounds and arm reachability."
            )

        # Convert URDF joint angles → motor space (applies ik_sign, gear ratios)
        urdf_full = np.zeros(cfg.dof, dtype=np.float32)
        urdf_full[:7] = np.array(urdf_joints[:7], dtype=np.float32)
        motor_pos = np.array(cfg.urdf2q(urdf_full), dtype=np.float32)

        self._send_position(arm, motor_pos)
        time.sleep(POSITION_SETTLE_TIME)

        # FK check: verify we got close
        actual_joints = self._read_joint_positions(arm)
        if actual_joints is not None:
            actual_pos, _ = cfg.ik.fk(actual_joints[:7].tolist())
            error = float(np.linalg.norm(
                np.array([pose.x, pose.y, pose.z]) - np.array(actual_pos)
            ))
            if error > POSITION_TOLERANCE * 3:
                print(
                    f"[WARN] {arm} arm position error {error*1000:.1f} mm "
                    f"(target={pose}, actual={actual_pos})"
                )

    def set_gripper(self, arm: str, closed: bool) -> None:
        """Open or close the gripper."""
        self._validate_arm(arm)
        cfg = self._cfgs[arm]
        target_turns = GRIPPER_CLOSE_TURNS if closed else GRIPPER_OPEN_TURNS

        # Gripper is motor index dof-1 (index 7). Left arm has gripper_sign=-1.
        gripper_sign = float(getattr(cfg, "gripper_sign", 1.0))
        motor_pos = self._read_motor_positions(arm)
        if motor_pos is None:
            print(f"[WARN] Could not read {arm} motor positions for gripper command")
            return

        motor_pos = motor_pos.copy()
        motor_pos[-1] = target_turns * gripper_sign
        self._send_position(arm, motor_pos)
        time.sleep(GRIPPER_MOVE_TIME)
        print(f"[BracketBotRobot] {arm} gripper {'CLOSED' if closed else 'OPENED'}")

    def stop(self) -> None:
        """Disable torque on all motors immediately."""
        for side, arm_name in self.ARM_NAMES.items():
            try:
                with self._writers[side].buf() as b:
                    b["torque_enable"] = np.zeros(
                        self._cfgs[side].dof, dtype=np.uint8
                    )
                print(f"[BracketBotRobot] {arm_name}: torque disabled (STOP)")
            except Exception as exc:
                print(f"[BracketBotRobot] {arm_name}: stop failed: {exc}")

    # ── Sim-compatible attach/release (no-ops on hardware) ───────────────────

    def attach(self, arm: str, sock_id: int) -> None:
        if self.held[arm] is not None:
            raise RuntimeError(f"{arm} arm is already holding sock {self.held[arm]}")
        self.held[arm] = sock_id

    def release(self, arm: str) -> int:
        sock_id = self.held[arm]
        if sock_id is None:
            raise RuntimeError(f"{arm} arm is not holding a sock")
        self.held[arm] = None
        return sock_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_arm(self, arm: str) -> None:
        if arm not in self.ARM_NAMES:
            raise ValueError(f"Unknown arm '{arm}'. Expected 'left' or 'right'.")

    def _send_position(self, arm: str, motor_pos: np.ndarray) -> None:
        """Write a motor-space position command to the arm daemon."""
        with self._writers[arm].buf() as b:
            b["pos"] = motor_pos
            b["tau"] = np.zeros(len(motor_pos), dtype=np.float32)

    def _read_motor_positions(self, arm: str) -> np.ndarray | None:
        """Read current motor-space positions from the arm state topic."""
        try:
            r = self._readers[arm]
            if r.ready():
                return np.array(r.data["pos"], dtype=np.float32)
        except Exception as exc:
            print(f"[WARN] Could not read {arm} state: {exc}")
        return None

    def _read_joint_positions(self, arm: str) -> np.ndarray | None:
        """Read current motor positions and convert to URDF joint space."""
        motor_pos = self._read_motor_positions(arm)
        if motor_pos is None:
            return None
        cfg = self._cfgs[arm]
        return np.array(cfg.q2urdf(motor_pos), dtype=np.float32)


# ── Standalone connection test ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BracketBot hardware adapter test")
    parser.add_argument("--test-connection", action="store_true")
    parser.add_argument("--test-ik", action="store_true",
                        help="Solve IK for a known point without moving")
    parser.add_argument("--test-move", action="store_true",
                        help="Move left arm above table center (USE WITH CAUTION)")
    args = parser.parse_args()

    if args.test_connection:
        print("Testing bbos connection...")
        robot = BracketBotRobot()
        for side in ("left", "right"):
            pos = robot._read_motor_positions(side)
            print(f"  {side} arm motor positions: {pos}")
        print("Connection OK.")

    if args.test_ik:
        print("\nTesting IK solve (no motion)...")
        if not BBOS_AVAILABLE:
            print("bbos not available — cannot test IK")
        else:
            cfg = Config("arm_left")
            cfg.ik.init()
            target = [0.0, 0.1, 0.15]
            quat = Rotation.from_euler("z", 0.0).as_quat().tolist()
            result = cfg.ik.solve(target, quat)
            print(f"  Target: {target}")
            print(f"  IK result ({len(result)} joints): {np.round(result, 3)}")
            fk_pos, fk_quat = cfg.ik.fk(result[:7])
            err = np.linalg.norm(np.array(target) - np.array(fk_pos))
            print(f"  FK check: {np.round(fk_pos, 4)} m  (error: {err*1000:.1f} mm)")

    if args.test_move:
        print("\n⚠  MOVING ARM — ensure workspace is clear!")
        input("Press Enter to continue, Ctrl+C to abort...")
        robot = BracketBotRobot()
        safe_pose = Pose(x=0.0, y=0.1, z=0.18, yaw=0.0)
        print(f"Moving left arm to safe pose above table: {safe_pose}")
        robot.move_end_effector("left", safe_pose)
        print("Done. Press Enter to open gripper...")
        input()
        robot.set_gripper("left", False)
        print("Done.")