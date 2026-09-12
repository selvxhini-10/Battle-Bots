from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import PairMatch, SockObservation


def annotate_scene(
    image: np.ndarray,
    socks: list[SockObservation],
    pairs: list[PairMatch],
    singles: list[int],
) -> np.ndarray:
    canvas = image.copy()
    by_id = {sock.id: sock for sock in socks}
    palette = [(255, 92, 92), (92, 170, 255), (180, 120, 255), (90, 210, 150)]
    for index, pair in enumerate(pairs):
        first, second = by_id[pair.first_id], by_id[pair.second_id]
        color = palette[index % len(palette)]
        p1 = tuple(round(v) for v in first.centroid_px)
        p2 = tuple(round(v) for v in second.centroid_px)
        cv2.line(canvas, p1, p2, color, 5, cv2.LINE_AA)
        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(canvas, f"{pair.score:.0%}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 4)
        cv2.putText(canvas, f"{pair.score:.0%}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for sock in socks:
        contour, _ = cv2.findContours(sock.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contour, -1, (35, 35, 35), 2)
        center = tuple(round(v) for v in sock.centroid_px)
        grasp = tuple(round(v) for v in sock.grasp_px)
        cv2.circle(canvas, grasp, 7, (0, 255, 255), -1)
        label = f"S{sock.id}" + (" SINGLE" if sock.id in singles else "")
        cv2.putText(canvas, label, (center[0] - 25, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    return canvas


def save_rerun(path: Path, image: np.ndarray, annotated: np.ndarray, socks: list[SockObservation]) -> bool:
    try:
        import rerun as rr
    except ImportError:
        return False
    rr.init("solemates")
    rr.log("camera/raw", rr.Image(image, color_model="BGR"))
    rr.log("camera/annotated", rr.Image(annotated, color_model="BGR"))
    rr.log("camera/grasps", rr.Points2D([sock.grasp_px for sock in socks], radii=7))
    rr.save(str(path))
    return True

