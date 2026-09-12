"""
bracketbot_ik.py — Python wrapper for libhybrid_ik_lib.so (RelaxedIK)

This library is compiled for ARM aarch64 and runs on the BracketBot's
onboard computer, NOT on your Windows laptop. Copy this file + the .so
to the robot before use.

The .so exports:
  relaxed_ik_new(urdf_path, ee_links)  → handle
  solve_position(handle, positions)    → joint angles
  solve_pose(handle, positions, quats) → joint angles
  forward_kinematics(handle, joints)   → end effector poses
  get_ee_positions(handle, joints)     → xyz positions only
  reset(handle)
  relaxed_ik_free(handle)

Usage:
  ik = BracketBotIK("chopped_urdf_v2/urdf/chopped_urdf_v2.urdf",
                    ee_links=["left_eef", "right_eef"])
  joints = ik.solve_position(left_xyz=[0.1, 0.05, 0.15],
                             right_xyz=[-0.1, 0.05, 0.15])
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class BracketBotIK:
    """Thin ctypes wrapper around libhybrid_ik_lib.so."""

    def __init__(
        self,
        urdf_path: str | Path,
        ee_links: list[str] | None = None,
        lib_path: str | Path | None = None,
    ):
        if ee_links is None:
            ee_links = ["left_eef", "right_eef"]

        # Find the .so
        if lib_path is None:
            search = [
                Path(__file__).parent / "libhybrid_ik_lib.so",
                Path("libhybrid_ik_lib.so"),
                Path("/home/bracketbot/libhybrid_ik_lib.so"),
            ]
            for candidate in search:
                if candidate.exists():
                    lib_path = candidate
                    break
            if lib_path is None:
                raise FileNotFoundError(
                    "libhybrid_ik_lib.so not found. Place it next to bracketbot_ik.py "
                    "or pass lib_path explicitly."
                )

        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_signatures()

        urdf_bytes = str(urdf_path).encode()
        ee_bytes = ",".join(ee_links).encode()
        self._handle = self._lib.relaxed_ik_new(urdf_bytes, ee_bytes)
        if not self._handle:
            raise RuntimeError("relaxed_ik_new returned null — check URDF path and ee_links")

        self._num_ee = len(ee_links)
        print(f"BracketBotIK initialized: {len(ee_links)} end effectors, URDF={urdf_path}")

    def _setup_signatures(self):
        lib = self._lib

        lib.relaxed_ik_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.relaxed_ik_new.restype = ctypes.c_void_p

        lib.relaxed_ik_free.argtypes = [ctypes.c_void_p]
        lib.relaxed_ik_free.restype = None

        lib.reset.argtypes = [ctypes.c_void_p]
        lib.reset.restype = None

        # solve_position: takes flat array of xyz targets (3 * num_ee floats)
        lib.solve_position.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.solve_position.restype = None

        # solve_pose: xyz + quaternion targets (7 * num_ee floats)
        lib.solve_pose.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.solve_pose.restype = None

        # forward_kinematics: joint angles in → ee poses out
        lib.forward_kinematics.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.forward_kinematics.restype = None

        lib.get_ee_positions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.get_ee_positions.restype = None

    def _call_solver(self, fn, inputs: np.ndarray, out_size: int) -> np.ndarray | None:
        inputs = np.asarray(inputs, dtype=np.float64)
        out = np.zeros(out_size, dtype=np.float64)
        n_out = ctypes.c_int(out_size)
        fn(
            self._handle,
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(inputs)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(n_out),
        )
        actual = n_out.value
        if actual == 0:
            return None
        return out[:actual]

    def solve_position(
        self,
        left_xyz: list[float] | np.ndarray,
        right_xyz: list[float] | np.ndarray,
    ) -> np.ndarray | None:
        """
        Solve joint angles for both arms given end-effector xyz targets.
        Returns flat joint angle array or None on failure.

        left_xyz, right_xyz: [x, y, z] in meters, robot base frame
        """
        targets = np.array(list(left_xyz) + list(right_xyz), dtype=np.float64)
        # Output: 7 joints per arm = 14 values
        return self._call_solver(self._lib.solve_position, targets, 14)

    def solve_pose(
        self,
        left_xyz: list[float],
        left_quat: list[float],   # [x, y, z, w]
        right_xyz: list[float],
        right_quat: list[float],
    ) -> np.ndarray | None:
        """
        Solve with full pose (position + orientation) targets.
        Quaternion order: [x, y, z, w]
        """
        targets = np.array(
            list(left_xyz) + list(left_quat) + list(right_xyz) + list(right_quat),
            dtype=np.float64,
        )
        return self._call_solver(self._lib.solve_pose, targets, 14)

    def forward_kinematics(self, joint_angles: list[float] | np.ndarray) -> np.ndarray | None:
        """
        Given joint angles (14 values, left then right), return ee poses.
        Output: flat array of poses [x,y,z,qx,qy,qz,qw] * num_ee
        """
        joints = np.asarray(joint_angles, dtype=np.float64)
        return self._call_solver(self._lib.forward_kinematics, joints, 7 * self._num_ee)

    def get_ee_positions(self, joint_angles: list[float] | np.ndarray) -> np.ndarray | None:
        """
        Given joint angles, return just xyz positions for each ee.
        Output: [left_x, left_y, left_z, right_x, right_y, right_z]
        """
        joints = np.asarray(joint_angles, dtype=np.float64)
        return self._call_solver(self._lib.get_ee_positions, joints, 3 * self._num_ee)

    def reset(self):
        """Reset solver state (call between tasks if IK diverges)."""
        self._lib.reset(self._handle)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle and hasattr(self, "_lib"):
            self._lib.relaxed_ik_free(self._handle)


# ── Quick test (run on the robot: python bracketbot_ik.py) ────────────────────
if __name__ == "__main__":
    import sys

    urdf = sys.argv[1] if len(sys.argv) > 1 else "chopped_urdf_v2/chopped_urdf_v2/urdf/chopped_urdf_v2.urdf"
    print(f"Testing BracketBotIK with URDF: {urdf}")

    ik = BracketBotIK(urdf)

    # Test: reach a point above the table center
    print("\nTest solve_position: left=(0.1, 0.05, 0.15), right=(-0.1, 0.05, 0.15)")
    joints = ik.solve_position([0.1, 0.05, 0.15], [-0.1, 0.05, 0.15])
    if joints is not None:
        print(f"  Joint angles ({len(joints)}): {np.round(joints, 3)}")
        pos = ik.get_ee_positions(joints)
        if pos is not None:
            print(f"  FK check — left ee:  {np.round(pos[:3], 4)} m")
            print(f"  FK check — right ee: {np.round(pos[3:], 4)} m")
    else:
        print("  IK returned None — check URDF path and ee_link names")