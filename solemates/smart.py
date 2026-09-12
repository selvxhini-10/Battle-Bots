from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import SockObservation
from .perception import observations_from_masks


class OptionalModelError(RuntimeError):
    """Raised with actionable setup instructions for an optional model."""


def segment_with_sam(image: np.ndarray, cfg: dict) -> list[SockObservation]:
    """Generate zero-shot masks with SAM, then reuse the classical features."""
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise OptionalModelError(
            "SAM is not installed. Run: pip install -e '.[sam]'"
        ) from exc

    checkpoint = Path(cfg.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise OptionalModelError(
            "SAM checkpoint not found. Download sam_vit_b_01ec64.pth from the "
            "official Segment Anything repository and set perception.sam.checkpoint "
            "in config.json or pass --sam-checkpoint PATH."
        )
    model_type = str(cfg.get("model_type", "vit_b"))
    if model_type not in sam_model_registry:
        raise OptionalModelError(f"Unknown SAM model type: {model_type}")
    device = _torch_device(torch, str(cfg.get("device", "auto")))
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=int(cfg.get("points_per_side", 24)),
        pred_iou_thresh=float(cfg.get("pred_iou_threshold", 0.86)),
        stability_score_thresh=float(cfg.get("stability_threshold", 0.90)),
        box_nms_thresh=float(cfg.get("box_nms_threshold", 0.7)),
        min_mask_region_area=int(cfg.get("minimum_area_px", 1800)),
    )
    proposals = generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    masks = _select_sam_masks(proposals, image.shape[:2], cfg)
    return observations_from_masks(image, masks)


def _select_sam_masks(proposals: list[dict], shape: tuple[int, int], cfg: dict) -> list[np.ndarray]:
    """Reject table/background masks and de-duplicate nested SAM proposals."""
    height, width = shape
    minimum = int(cfg.get("minimum_area_px", 1800))
    maximum = float(cfg.get("maximum_area_fraction", 0.30)) * height * width
    candidates = []
    for proposal in proposals:
        mask = np.asarray(proposal["segmentation"], dtype=bool)
        area = int(proposal.get("area", mask.sum()))
        if area < minimum or area > maximum:
            continue
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        touches_border = xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1
        if touches_border and cfg.get("reject_border_masks", True):
            continue
        quality = float(proposal.get("predicted_iou", 0.0)) + float(proposal.get("stability_score", 0.0))
        candidates.append((quality, area, mask))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[np.ndarray] = []
    threshold = float(cfg.get("mask_iou_threshold", 0.72))
    for _quality, _area, mask in candidates:
        if all(_mask_iou(mask, kept) < threshold for kept in selected):
            selected.append(mask)
    return selected


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def attach_clip_embeddings(
    image: np.ndarray,
    socks: list[SockObservation],
    cfg: dict,
) -> None:
    """Attach normalized zero-shot CLIP embeddings to masked sock crops."""
    try:
        import clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise OptionalModelError(
            "CLIP is not installed. Run: pip install -e '.[clip]'"
        ) from exc

    device = _torch_device(torch, str(cfg.get("device", "auto")))
    model, preprocess = clip.load(str(cfg.get("model", "ViT-B/32")), device=device)
    crops = []
    padding = int(cfg.get("crop_padding_px", 12))
    height, width = image.shape[:2]
    for sock in socks:
        x, y, w, h = sock.bbox_px
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
        crop = image[y0:y1, x0:x1].copy()
        mask = sock.mask[y0:y1, x0:x1] > 0
        crop[~mask] = 242
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(preprocess(Image.fromarray(rgb)))
    if not crops:
        return
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(batch).float()
        embeddings /= embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    for sock, embedding in zip(socks, embeddings.cpu().numpy()):
        sock.embedding = np.asarray(embedding, dtype=np.float32)


def _torch_device(torch, configured: str) -> str:
    if configured != "auto":
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"
