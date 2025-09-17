from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .types import ROIBox


class SamPredictorWrapper:
    """Wrapper that calls a provided SAM backend adapter.

    The backend should implement: predict(image_rgb, points, labels, box_xyxy) -> uint8 mask (0/255).
    If no backend is provided, this wrapper returns the previous mask unchanged.
    """

    def __init__(self, backend: Optional[Any] = None):
        self.backend = backend

    def predict(
        self,
        image_rgb: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        box: ROIBox,
        prev_mask: np.ndarray,
    ) -> np.ndarray:
        if self.backend is None:
            return prev_mask
        xyxy = (float(box.x0), float(box.y0), float(box.x1), float(box.y1))
        try:
            return self.backend.predict(image_rgb, points, labels, xyxy, prev_mask=prev_mask)
        except Exception:
            # Safe fallback on any runtime error
            return prev_mask


def early_stop(prev_mask: np.ndarray, new_mask: np.ndarray, iou_tol: float = 0.005, *, require_pos_points: bool = False, used_pos_count: int = 0, min_round: int = 1, current_round: int = 0) -> bool:
    """
    Decide early stopping based on IoU delta, with guards to prevent premature stop:
    - require_pos_points: only consider stopping if at least one positive point was used
    - min_round: enforce a minimum number of rounds before stopping
    - current_round: index of the current round (0-based)
    """
    if current_round < min_round:
        return False
    if require_pos_points and used_pos_count <= 0:
        return False
    p = (prev_mask > 0).astype(np.uint8)
    n = (new_mask > 0).astype(np.uint8)
    inter = (p & n).sum()
    union = (p | n).sum() + 1e-6
    iou = inter / union
    return (1.0 - iou) < iou_tol
