"""
solemates/visualization.py

Full 3D Rerun visualization for Solemates.

What this logs
--------------
2D panel  (camera/*)
  raw camera frame, annotated matches frame, segmentation masks,
  grasp-point overlay, pair-score text

3D panel  (world/*)
  table surface + zone boxes (Unsorted / Couples / Singles)
  sock objects as coloured oriented boxes that move with the arm
  end-effector trajectory trails for each arm
  arm-base origin marker

Joint timeline  (joints/*)
  per-joint angle scalars so you can scrub through the IK solution

State log  (state/*)
  TextLog entry on every state-machine transition

Usage
-----
Call ``init()`` once before the run, then:
  - ``log_camera_frame()`` after perception
  - ``log_3d_scene()`` once (static geometry)
  - ``log_sock_observation()`` per detected sock
  - ``log_pairs()`` after matching
  - ``log_ee_move()`` for every move_end_effector call
  - ``log_gripper()`` on every set_gripper call
  - ``log_joint_state()`` with the IK solution dict
  - ``log_state_transition()`` on every RunState change
  - ``save()`` at the end
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import PairMatch, RunState, SockObservation

# ---------------------------------------------------------------------------
# Colour palette (RGBA tuples, uint8)
# ---------------------------------------------------------------------------
_BLUE   = (92, 170, 255, 220)
_CORAL  = (255, 110, 90, 220)
_GREEN  = (90, 210, 150, 220)
_AMBER  = (255, 190, 60, 220)
_PURPLE = (180, 120, 255, 220)
_GRAY   = (160, 160, 160, 180)
_WHITE  = (255, 255, 255, 255)
_RED    = (230, 60, 60, 255)

_ARM_COLORS = {"left": _BLUE, "right": _CORAL}
_ZONE_COLORS = {
    "unsorted": (180, 180, 120, 60),
    "couples":  (90, 200, 130, 60),
    "singles":  (200, 100, 80, 60),
}

# Map sock IDs → stable palette entry
_SOCK_PALETTE = [_BLUE, _CORAL, _GREEN, _AMBER, _PURPLE, _GRAY]


def _sock_color(sock_id: int) -> tuple:
    return _SOCK_PALETTE[(sock_id - 1) % len(_SOCK_PALETTE)]


# ---------------------------------------------------------------------------
# Module-level recorder handle (set by init())
# ---------------------------------------------------------------------------
_rr = None
_initialized = False   # True only after init() completes successfully


_dt = None   # rerun.datatypes, cached to avoid repeated lazy-import warnings


def _try_import() -> bool:
    global _rr, _dt
    try:
        import rerun as rr
        import rerun.datatypes as dt  # import explicitly — avoids rr.__getattr__ spam
        _rr = rr
        _dt = dt
        return True
    except ImportError:
        return False


def _ready() -> bool:
    """Return True only when rerun is imported AND init() has been called."""
    return _rr is not None and _initialized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init(recording_id: str = "solemates") -> bool:
    """
    Initialise the Rerun recording stream and set up the blueprint layout.
    Returns False if rerun is not installed (graceful no-op).
    """
    if not _try_import():
        return False

    rr = _rr
    from rerun.blueprint import (
        Blueprint,
        Horizontal,
        Vertical,
        Spatial2DView,
        Spatial3DView,
        TimeSeriesView,
        TextLogView,
    )

    blueprint = Blueprint(
        Horizontal(
            Vertical(
                Spatial2DView(
                    name="Camera",
                    origin="camera",
                    contents=["camera/**"],
                ),
                TimeSeriesView(
                    name="Joint angles",
                    origin="joints",
                    contents=["joints/**"],
                ),
                row_shares=[3, 1],        # Vertical splits vertically → row_shares
            ),
            Vertical(
                Spatial3DView(
                    name="3D world",
                    origin="world",
                    contents=["world/**"],
                ),
                TextLogView(
                    name="State log",
                    origin="state",
                ),
                row_shares=[3, 1],
            ),
            column_shares=[1, 1],         # Horizontal splits horizontally → column_shares
        )
    )

    rr.init(recording_id, spawn=False)
    rr.send_blueprint(blueprint)

    global _initialized
    _initialized = True

    # Declare joint time-series appearance (static, once)
    _init_joint_series()
    return True


def save(path: Path) -> bool:
    """Save the completed recording to *path* (.rrd)."""
    if not _ready():
        return False
    _rr.save(str(path))
    return True


def log_camera_frame(
    image: np.ndarray,
    annotated: np.ndarray,
    socks: list[SockObservation],
    step: int,
) -> None:
    """Log raw + annotated 2D camera frames and segmentation masks."""
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)

    rr.log("camera/raw",      rr.Image(image,      color_model="BGR"))
    rr.log("camera/annotated", rr.Image(annotated, color_model="BGR"))

    # Per-sock mask overlays and grasp points
    for sock in socks:
        color = _sock_color(sock.id)
        rr.log(
            f"camera/socks/s{sock.id}/mask",
            rr.SegmentationImage(sock.mask),
        )
        gx, gy = sock.grasp_px
        rr.log(
            f"camera/socks/s{sock.id}/grasp",
            rr.Points2D([[gx, gy]], radii=6, colors=[color],
                        labels=[f"S{sock.id}"]),
        )


def log_pairs(
    pairs: list[PairMatch],
    singles: list[int],
    step: int,
) -> None:
    """Log pair scores as a text summary on the camera panel."""
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)

    lines = []
    for p in pairs:
        lines.append(
            f"Pair S{p.first_id}+S{p.second_id}  {p.score:.0%}  "
            f"(color {p.components.get('color', 0):.2f}  "
            f"pattern {p.components.get('pattern', 0):.2f})"
        )
    for s in singles:
        lines.append(f"Single S{s}")

    rr.log(
        "camera/match_summary",
        rr.TextDocument("\n".join(lines), media_type=rr.MediaType.TEXT),
    )


def log_3d_scene(workspace: dict) -> None:
    """
    Log the static 3D world geometry:
      - table surface
      - Unsorted / Couples / Singles zone boxes
      - arm-base origin marker
    Call once before the motion loop.
    """
    if not _ready():
        return
    rr = _rr

    xmin, xmax, ymin, ymax = workspace["table_bounds_m"]
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w  = xmax - xmin
    d  = ymax - ymin
    table_z = float(workspace.get("table_z_m", 0.0))
    slab_h  = 0.01   # 1 cm slab

    # Table surface
    rr.log(
        "world/table",
        rr.Boxes3D(
            centers=[[cx, cy, table_z - slab_h / 2]],
            half_sizes=[[w / 2, d / 2, slab_h / 2]],
            colors=[(210, 180, 140, 180)],
            labels=["table"],
        ),
        static=True,
    )

    # Zone overlays (thin flat boxes just above table)
    zone_h = 0.002
    zones = _build_zones(workspace)
    for name, (zx, zy, zw, zd) in zones.items():
        color = _ZONE_COLORS.get(name, _GRAY)
        rr.log(
            f"world/zones/{name}",
            rr.Boxes3D(
                centers=[[zx, zy, table_z + zone_h / 2]],
                half_sizes=[[zw / 2, zd / 2, zone_h / 2]],
                colors=[color],
                labels=[name],
            ),
            static=True,
        )

    # Arm-base origin
    rr.log(
        "world/arm_base",
        rr.Points3D([[0.0, 0.0, 0.0]], radii=0.012, colors=[_WHITE],
                    labels=["arm_base"]),
        static=True,
    )

    # Axis arrows for orientation reference
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.08, 0, 0], [0, 0.08, 0], [0, 0, 0.08]],
            colors=[(220, 60, 60, 255), (60, 200, 60, 255), (60, 60, 220, 255)],
        ),
        static=True,
    )


def log_sock_3d(
    sock: SockObservation,
    transform: "PixelTableTransform",
    table_z: float,
    step: int,
    held_by: Optional[str] = None,
) -> None:
    """
    Log a sock as an oriented 3D box on the table surface.
    *held_by* is "left"/"right" when an arm is holding it.
    """
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)

    x, y = transform.point(sock.grasp_px)
    z = table_z + 0.005  # 5 mm above table

    # Estimate physical size from pixel area (rough)
    px_per_m = 960 / 0.70   # image width / table width
    area_m2  = sock.area_px / (px_per_m ** 2)
    side     = math.sqrt(max(area_m2, 0.001))
    hw = min(side * 0.7, 0.08)
    hd = min(side * 0.4, 0.05)

    color = list(_sock_color(sock.id))
    if held_by:
        color[3] = 255   # fully opaque when picked up

    rr.log(
        f"world/socks/s{sock.id}",
        rr.Boxes3D(
            centers=[[x, y, z]],
            half_sizes=[[hw, hd, 0.003]],
            rotation_axis_angles=[
                _dt.RotationAxisAngle(
                    axis=[0, 0, 1],
                    angle=_dt.Angle(rad=float(sock.angle_rad)),
                )
            ],
            colors=[tuple(color)],
            labels=[f"S{sock.id}" + (f" ({held_by})" if held_by else "")],
        ),
    )


def log_sock_carried(
    sock_id: int,
    arm: str,
    ee_pos: tuple[float, float, float],
    step: int,
) -> None:
    """Move a sock's 3D box to follow the end-effector while carried."""
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)
    x, y, z = ee_pos
    color = list(_sock_color(sock_id))
    color[3] = 255
    rr.log(
        f"world/socks/s{sock_id}",
        rr.Boxes3D(
            centers=[[x, y, z + 0.015]],
            half_sizes=[[0.06, 0.035, 0.003]],
            colors=[tuple(color)],
            labels=[f"S{sock_id} → {arm}"],
        ),
    )


def log_ee_move(
    arm: str,
    pose,           # solemates.models.Pose
    step: int,
    ik_error: Optional[float] = None,
) -> None:
    """
    Log end-effector position + orientation arrow, and accumulate
    a trajectory trail for this arm.
    """
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)

    color = _ARM_COLORS.get(arm, _GRAY)
    pos   = [pose.x, pose.y, pose.z]

    # EE position sphere
    label = f"{arm} EE"
    if ik_error is not None:
        label += f"  err={ik_error*1000:.1f}mm"
    rr.log(
        f"world/arms/{arm}/ee",
        rr.Points3D([pos], radii=0.008, colors=[color], labels=[label]),
    )

    # Gripper orientation arrow (points along yaw, downward)
    yaw_vec = [math.cos(pose.yaw) * 0.04, math.sin(pose.yaw) * 0.04, -0.025]
    rr.log(
        f"world/arms/{arm}/gripper_dir",
        rr.Arrows3D(origins=[pos], vectors=[yaw_vec], colors=[color]),
    )

    # Trajectory trail — append to a growing line strip
    _append_trail(arm, pos, step)


def log_gripper(arm: str, closed: bool, step: int) -> None:
    """Log gripper state as a small coloured marker."""
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)
    color = _RED if closed else _GREEN
    rr.log(
        f"world/arms/{arm}/gripper_state",
        rr.TextLog(f"{arm} gripper {'CLOSED' if closed else 'OPEN'}",
                   level="INFO" if closed else "DEBUG"),
    )
    rr.log(
        "state/gripper",
        rr.TextLog(f"[step {step}] {arm} gripper {'CLOSED ✓' if closed else 'OPEN ○'}"),
    )


def log_joint_state(
    arm: str,
    joints: dict[str, float],
    step: int,
) -> None:
    """Log each joint angle as a scalar so the timeline panel shows curves."""
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)

    for name, angle in joints.items():
        rr.log(f"joints/{arm}/{name}", rr.Scalars(angle))


def log_state_transition(
    state: RunState,
    message: str,
    step: int,
    data: Optional[dict] = None,
) -> None:
    """Log a state-machine transition to the text log panel."""
    if not _ready():
        return
    rr = _rr
    rr.set_time("step", sequence=step)

    detail = ""
    if data:
        # Pretty-print a few key fields without importing json here
        for key in ("pairs", "singles", "sock_id", "arm", "ik_error_m"):
            if key in data:
                detail += f"  {key}={data[key]}"

    rr.log(
        "state/transitions",
        rr.TextLog(
            f"[{step:03d}] {state.value.upper():16s}  {message}{detail}",
            level="INFO",
        ),
    )


# ---------------------------------------------------------------------------
# Backward-compatible helpers kept from the original visualization.py
# ---------------------------------------------------------------------------

def annotate_scene(
    image: np.ndarray,
    socks: list[SockObservation],
    pairs: list[PairMatch],
    singles: list[int],
) -> np.ndarray:
    """Draw pair lines, grasp dots, and labels onto *image*."""
    canvas = image.copy()
    by_id  = {sock.id: sock for sock in socks}
    palette = [(255, 92, 92), (92, 170, 255), (180, 120, 255), (90, 210, 150)]

    for index, pair in enumerate(pairs):
        first  = by_id[pair.first_id]
        second = by_id[pair.second_id]
        color  = palette[index % len(palette)]
        p1 = tuple(round(v) for v in first.centroid_px)
        p2 = tuple(round(v) for v in second.centroid_px)
        cv2.line(canvas, p1, p2, color, 5, cv2.LINE_AA)
        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(canvas, f"{pair.score:.0%}", midpoint,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 4)
        cv2.putText(canvas, f"{pair.score:.0%}", midpoint,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for sock in socks:
        contour, _ = cv2.findContours(
            sock.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contour, -1, (35, 35, 35), 2)
        center = tuple(round(v) for v in sock.centroid_px)
        grasp  = tuple(round(v) for v in sock.grasp_px)
        cv2.circle(canvas, grasp, 7, (0, 255, 255), -1)
        label = f"S{sock.id}" + (" SINGLE" if sock.id in singles else "")
        cv2.putText(canvas, label, (center[0] - 25, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    return canvas


def save_rerun(
    path: Path,
    image: np.ndarray,
    annotated: np.ndarray,
    socks: list[SockObservation],
) -> bool:
    """
    Minimal backward-compatible save used by app.py when --rrd is passed
    without the full enhanced pipeline.
    """
    if _rr is None and not _try_import():
        return False
    rr = _rr
    rr.init("solemates")
    rr.log("camera/raw",       rr.Image(image,      color_model="BGR"))
    rr.log("camera/annotated", rr.Image(annotated,  color_model="BGR"))
    rr.log("camera/grasps",
           rr.Points2D([sock.grasp_px for sock in socks], radii=7))
    rr.save(str(path))
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_trails: dict[str, list[list[float]]] = {}


def _append_trail(arm: str, pos: list[float], step: int) -> None:
    """Accumulate EE positions and re-log the growing polyline."""
    if not _ready():
        return
    rr = _rr
    trail = _trails.setdefault(arm, [])
    trail.append(pos)
    if len(trail) < 2:
        return
    color = _ARM_COLORS.get(arm, _GRAY)
    rr.log(
        f"world/arms/{arm}/trail",
        rr.LineStrips3D([trail], colors=[color]),
    )


def _init_joint_series() -> None:
    """Declare joint time-series appearance (static, logged once)."""
    if not _ready():
        return
    rr = _rr
    joint_names_left  = [f"lj{i}" for i in range(7)]
    joint_names_right = [f"rj{i}" for i in range(7)]
    for name in joint_names_left:
        rr.log(
            f"joints/left/{name}",
            rr.SeriesLines(colors=[_BLUE[:3]], names=[name]),
            static=True,
        )
    for name in joint_names_right:
        rr.log(
            f"joints/right/{name}",
            rr.SeriesLines(colors=[_CORAL[:3]], names=[name]),
            static=True,
        )


def _build_zones(workspace: dict) -> dict[str, tuple]:
    """
    Return {zone_name: (cx, cy, width, depth)} for the three workspace zones.
    Derives Couples and Singles from the configured origin lists; Unsorted
    covers the rest of the table.
    """
    xmin, xmax, ymin, ymax = workspace["table_bounds_m"]
    w = xmax - xmin
    d = ymax - ymin

    couples_origins = workspace.get("couples_origins_m", [[0.13, -0.22]])
    singles_origin  = workspace.get("singles_origin_m",  [-0.25, -0.22])

    # Couples zone: bounding box around all destination slots + margin
    cxs = [o[0] for o in couples_origins]
    cys = [o[1] for o in couples_origins]
    margin = 0.12
    c_cx = (min(cxs) + max(cxs)) / 2
    c_cy = (min(cys) + max(cys)) / 2
    c_w  = max(max(cxs) - min(cxs) + margin, margin)
    c_d  = margin

    s_cx = float(singles_origin[0])
    s_cy = float(singles_origin[1])

    return {
        "unsorted": (
            (xmin + xmax) / 2,
            (ymin + ymax) / 2 + d * 0.1,
            w * 0.6,
            d * 0.5,
        ),
        "couples": (c_cx, c_cy, c_w, c_d),
        "singles": (s_cx, s_cy, 0.14, 0.10),
    }