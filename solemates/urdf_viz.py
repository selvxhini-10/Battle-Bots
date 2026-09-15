"""
solemates/urdf_viz.py

Logs the full articulated BracketBot URDF into Rerun so the robot arm
visually animates in the 3D world panel as joint angles change.

How it works
------------
The URDF is a tree of links connected by joints. For each step we:

1. Walk every joint in the tree (already parsed by KinematicModel).
2. Compute the 4x4 transform of the joint's child link relative to its
   parent, including the current joint-angle contribution.
3. Log rr.Transform3D at  world/robot/<child_link_name>  so Rerun
   builds the scene graph matching the URDF hierarchy.
4. On the first call, also log rr.Asset3D for any mesh files referenced
   by the URDF, parented under their link's entity path.

The result: as you scrub the Rerun timeline the arm bends and reaches
in exact correspondence with the IK solutions logged by visualization.py.

Mesh format support
-------------------
rerun 0.26 accepts OBJ, STL, GLB/GLTF via Asset3D.
The chopped_urdf_v2 package typically ships .stl or .obj meshes.
Files that cannot be found or have unsupported formats are skipped
with a warning; the transforms still animate correctly.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_meshes_logged = False   # log mesh geometry once (static), transforms every step
_urdf_root: Optional[Path] = None
_joint_tree: list[dict] = []   # parsed joint defs, ordered parent-before-child
_link_meshes: dict[str, list[Path]] = {}  # link_name → list of mesh paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_urdf(urdf_path: Path, viz_module) -> bool:
    """
    Parse the URDF and log static mesh geometry into Rerun.
    Call once after viz.init().

    Parameters
    ----------
    urdf_path : Path
        Absolute path to the .urdf file.
    viz_module : module
        The solemates.visualization module (provides _rr, _dt, _ready).
    """
    global _urdf_root, _joint_tree, _link_meshes, _meshes_logged

    if not viz_module._ready():
        return False
    if not urdf_path.is_file():
        return False

    _urdf_root = urdf_path.parent
    _joint_tree, _link_meshes = _parse_urdf(urdf_path)

    _log_static_meshes(viz_module)
    _meshes_logged = True
    return True


def log_robot_state(
    joint_positions: dict[str, float],
    step: int,
    viz_module,
) -> None:
    """
    Log Transform3D for every link at the current joint configuration.
    Call this once per move_end_effector step with the IK solution.

    Parameters
    ----------
    joint_positions : dict[str, float]
        Joint-name → angle (rad) or prismatic displacement (m).
        Unknown joints default to 0.
    step : int
        Current timeline step (matches visualization.py).
    viz_module : module
        The solemates.visualization module.
    """
    if not viz_module._ready() or not _joint_tree:
        return

    rr = viz_module._rr
    dt = viz_module._dt
    rr.set_time("step", sequence=step)

    for joint in _joint_tree:
        child = joint["child"]
        T = _joint_transform(joint, joint_positions.get(joint["name"], 0.0))
        translation = T[:3, 3]
        R = T[:3, :3]
        # Convert rotation matrix → quaternion (xyzw)
        quat_xyzw = _mat_to_quat(R)

        rr.log(
            f"world/robot/{child}",
            rr.Transform3D(
                translation=translation.tolist(),
                quaternion=dt.Quaternion(xyzw=quat_xyzw),
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def _parse_urdf(urdf_path: Path):
    """
    Returns (joint_list, link_meshes).

    joint_list  : list of joint dicts ordered so parents come before children.
    link_meshes : dict mapping link name → [mesh Path, ...]
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Collect all joints
    raw_joints = []
    for elem in root.findall("joint"):
        origin = elem.find("origin")
        axis_elem = elem.find("axis")
        limit_elem = elem.find("limit")

        xyz  = _vec3(origin.get("xyz") if origin is not None else None)
        rpy  = _vec3(origin.get("rpy") if origin is not None else None)
        axis = _vec3(axis_elem.get("xyz") if axis_elem is not None else None,
                     default=(1.0, 0.0, 0.0))

        raw_joints.append({
            "name":   elem.get("name", ""),
            "kind":   elem.get("type", "fixed"),
            "parent": elem.find("parent").get("link"),
            "child":  elem.find("child").get("link"),
            "xyz":    xyz,
            "rpy":    rpy,
            "axis":   axis,
        })

    # Topological sort (parents before children)
    ordered = _topo_sort(raw_joints)

    # Collect mesh filenames for each link
    link_meshes: dict[str, list[Path]] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name", "")
        paths = []
        for visual in link_elem.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue
            filename = mesh_elem.get("filename", "")
            # URDF filenames often use package:// or relative paths
            filename = filename.replace("package://", "")
            # Try relative to URDF dir
            candidate = (urdf_path.parent / filename).resolve()
            if not candidate.is_file():
                # Try stripping leading path components
                for part in Path(filename).parts:
                    candidate = (urdf_path.parent / part).resolve()
                    if candidate.is_file():
                        break
            if candidate.is_file():
                paths.append(candidate)
        if paths:
            link_meshes[link_name] = paths

    return ordered, link_meshes


def _topo_sort(joints: list[dict]) -> list[dict]:
    """Return joints in parent-before-child order."""
    children_of: dict[str, list[dict]] = {}
    all_parents = set()
    for j in joints:
        children_of.setdefault(j["parent"], []).append(j)
        all_parents.add(j["parent"])

    # Roots: parents that are never someone else's child
    all_children = {j["child"] for j in joints}
    roots = [p for p in all_parents if p not in all_children]
    if not roots:
        # Fallback: just return as-is
        return joints

    ordered = []
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        for child_joint in children_of.get(node, []):
            ordered.append(child_joint)
            queue.append(child_joint["child"])
    return ordered


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ], dtype=float)


def _axis_angle_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [c + x*x*one,     x*y*one - z*s,  x*z*one + y*s],
        [y*x*one + z*s,  c + y*y*one,    y*z*one - x*s],
        [z*x*one - y*s,  z*y*one + x*s,  c + z*z*one  ],
    ], dtype=float)


def _joint_transform(joint: dict, value: float) -> np.ndarray:
    """4x4 transform from parent link origin to child link origin."""
    T = np.eye(4)
    R_rpy = _rpy_to_mat(joint["rpy"])
    T[:3, :3] = R_rpy
    T[:3,  3] = joint["xyz"]

    kind = joint["kind"]
    if kind in ("revolute", "continuous"):
        R_joint = _axis_angle_mat(np.array(joint["axis"]), value)
        T[:3, :3] = R_rpy @ R_joint
    elif kind == "prismatic":
        T[:3, 3] = joint["xyz"] + R_rpy @ (np.array(joint["axis"]) * value)

    return T


def _mat_to_quat(R: np.ndarray) -> list[float]:
    """Rotation matrix → quaternion [x, y, z, w]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _vec3(text: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text:
        return np.array([float(v) for v in text.split()], dtype=float)
    return np.array(default, dtype=float)


# ---------------------------------------------------------------------------
# Static mesh logging
# ---------------------------------------------------------------------------

_SUPPORTED_SUFFIXES = {".obj", ".stl", ".glb", ".gltf"}

_LINK_COLOR = (220, 220, 230, 200)   # light grey, slightly transparent


def _log_static_meshes(viz_module) -> None:
    """Log mesh geometry once as static entities under world/robot/<link>/mesh."""
    rr = viz_module._rr

    for link_name, mesh_paths in _link_meshes.items():
        for i, mesh_path in enumerate(mesh_paths):
            suffix = mesh_path.suffix.lower()
            if suffix not in _SUPPORTED_SUFFIXES:
                continue
            try:
                entity = f"world/robot/{link_name}/mesh_{i}"
                rr.log(
                    entity,
                    rr.Asset3D(path=str(mesh_path)),
                    static=True,
                )
            except Exception as exc:
                # Non-fatal: transforms still animate, mesh just won't show
                print(f"[urdf_viz] skipped mesh {mesh_path.name}: {exc}")

    # For links with no mesh, log a small sphere placeholder so the
    # joint positions are still visible in the 3D view
    linked = set(_link_meshes.keys())
    for joint in _joint_tree:
        child = joint["child"]
        if child not in linked:
            rr.log(
                f"world/robot/{child}/marker",
                rr.Points3D(
                    [[0.0, 0.0, 0.0]],
                    radii=0.004,
                    colors=[_LINK_COLOR],
                ),
                static=True,
            )