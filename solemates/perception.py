from __future__ import annotations

import math

import cv2
import numpy as np

from .models import SockObservation


def segment_socks(image: np.ndarray, cfg: dict) -> list[SockObservation]:
    """Segment separated socks on a known plain background."""
    expected = np.asarray(cfg["background_bgr"], dtype=np.float32)
    distance = np.linalg.norm(image.astype(np.float32) - expected, axis=2)
    foreground = (distance > float(cfg["background_distance"])).astype(np.uint8) * 255
    k = int(cfg.get("morphology_kernel", 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground)
    components: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= int(cfg["minimum_area_px"]):
            components.append((label, labels == label, stats[label], centroids[label]))
    components.sort(key=lambda item: (item[3][1], item[3][0]))

    return [
        _describe_sock(index + 1, image, mask, stat, centroid)
        for index, (_label, mask, stat, centroid) in enumerate(components)
    ]


def _describe_sock(
    sock_id: int,
    image: np.ndarray,
    mask_bool: np.ndarray,
    stat: np.ndarray,
    centroid: np.ndarray,
) -> SockObservation:
    mask = mask_bool.astype(np.uint8) * 255
    x, y, w, h, area = [int(v) for v in stat]
    points = np.column_stack(np.nonzero(mask_bool))[:, ::-1].astype(np.float32)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.atan2(float(axis[1]), float(axis[0]))

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    gy, gx = np.unravel_index(int(np.argmax(distance)), distance.shape)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256]).ravel()
    hist /= max(float(np.linalg.norm(hist)), 1e-9)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    values = laplacian[mask_bool]
    pattern = np.array([
        float(np.mean(np.abs(values))) / 64.0,
        float(np.std(values)) / 128.0,
    ], dtype=np.float32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    compactness = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-9)
    extent = area / max(float(w * h), 1.0)
    aspect = min(w, h) / max(float(max(w, h)), 1.0)
    shape = np.array([compactness, extent, aspect], dtype=np.float32)

    return SockObservation(
        id=sock_id,
        mask=mask,
        centroid_px=(float(centroid[0]), float(centroid[1])),
        angle_rad=angle,
        area_px=float(area),
        bbox_px=(x, y, w, h),
        color_hist=hist,
        pattern=pattern,
        shape=shape,
        grasp_px=(float(gx), float(gy)),
    )

