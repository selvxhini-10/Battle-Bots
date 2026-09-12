from __future__ import annotations

import cv2
import numpy as np


def make_scene(width: int = 960, height: int = 640) -> tuple[np.ndarray, dict[int, str]]:
    """Create five separated socks: two pairs and one singleton."""
    image = np.full((height, width, 3), 242, dtype=np.uint8)
    specs = [
        ((170, 145), -18, (68, 93, 235), "stripes", "coral-stripes"),
        ((480, 140), 23, (190, 110, 65), "dots", "blue-dots"),
        ((760, 175), 14, (68, 93, 235), "stripes", "coral-stripes"),
        ((290, 395), -28, (190, 110, 65), "dots", "blue-dots"),
        ((650, 405), 31, (98, 180, 86), "plain", "green-single"),
    ]
    labels: dict[int, str] = {}
    for index, (center, angle, color, pattern, label) in enumerate(specs, 1):
        _draw_sock(image, center, angle, color, pattern)
        labels[index] = label
    return image, labels


def _draw_sock(image: np.ndarray, center, angle_deg: float, color, pattern: str) -> None:
    cx, cy = center
    local = np.array([
        [-42, -95], [42, -95], [42, 12], [88, 12], [92, 48],
        [62, 72], [-5, 65], [-5, 24], [-42, 24],
    ], dtype=np.float32)
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    polygon = (local @ rotation.T + np.array([cx, cy])).astype(np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    image[mask > 0] = color

    overlay = image.copy()
    accent = tuple(min(255, int(channel * 0.55 + 110)) for channel in color)
    if pattern == "stripes":
        for offset in range(-100, 100, 28):
            cv2.line(overlay, (cx - 110, cy + offset), (cx + 110, cy + offset), accent, 8)
    elif pattern == "dots":
        for yy in range(cy - 75, cy + 75, 30):
            for xx in range(cx - 70, cx + 80, 34):
                cv2.circle(overlay, (xx, yy), 7, accent, -1)
    image[mask > 0] = overlay[mask > 0]
    cv2.polylines(image, [polygon], True, (60, 60, 60), 3, cv2.LINE_AA)

