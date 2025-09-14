from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .types import Candidate, ROIBox
from .utils import nms_points


def estimate_neg_thresholds(
    candidates: List[Candidate], scores: Dict[int, float], B: ROIBox
) -> Tuple[float, float]:
    # Fallback fixed thresholds if stats unavailable
    arr = np.array([scores.get(c.idx, 0.0) for c in candidates], dtype=np.float32)
    if arr.size == 0:
        return 0.6, 0.45
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    tau_pos = min(max(mu + 1.0 * sd, 0.55), 0.70)  # softer than +2σ to avoid empties
    tau_neg = 0.45
    return tau_pos, tau_neg


def spatial_nms_sorted(pool: List[Candidate], scores: Dict[int, float], B: ROIBox, topk: int) -> List[Candidate]:
    pts = [c.xy for c in pool]
    sc = [float(scores.get(c.idx, 0.0)) for c in pool]
    r = 0.06 * B.diag
    keep_idx = nms_points(pts, sc, r)
    kept = [pool[i] for i in keep_idx]
    kept.sort(key=lambda c: scores.get(c.idx, 0.0), reverse=True)
    return kept[:topk]


def select_points(
    candidates: List[Candidate], scores: Dict[int, float], B: ROIBox, K_pos: int = 6, K_neg: int = 6
) -> Tuple[np.ndarray, np.ndarray, List[Candidate], List[Candidate]]:
    tau_pos, tau_neg = estimate_neg_thresholds(candidates, scores, B)
    pos_pool = [c for c in candidates if scores.get(c.idx, 0.0) >= tau_pos]
    neg_pool = [c for c in candidates if scores.get(c.idx, 0.0) <= tau_neg]
    pos = spatial_nms_sorted(pos_pool, scores, B, topk=K_pos)
    inv = {i: 1.0 - float(scores.get(i, 0.0)) for i in [c.idx for c in neg_pool]}
    neg = spatial_nms_sorted(neg_pool, inv, B, topk=K_neg)
    points = np.array([p.xy for p in pos + neg], dtype=np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    return points, labels, pos, neg

