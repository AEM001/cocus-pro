"""
【候选点采样模块】
作用：在初始掩码的ROI区域内生成三类候选控制点（内部/边界/外部）
核心功能：
  - 基于距离变换的掩码内部点采样
  - 基于形态学操作的边界带状区域采样
  - 基于纹理方差的外部背景点采样
  - 空间NMS去重，确保点分布均匀

与下游模块关系：
  - 输出给patches.py进行图像块提取
  - 输出给select_points.py进行VLM点选择
  - 最终传递给sam_refine.py作为SAM控制点

算法特点：
  - 分层采样策略（内部高权重，边界中权重，外部低权重）
  - 自适应采样密度基于ROI大小
  - 考虑纹理丰富度选择有区分性的外部点
"""

# 内部/边界/外部候选采样，带空间NMS（边界优先策略）
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .types import Candidate, ROIBox
from .utils import nms_points, uniform_grid_in_box


def _mask_bbox(mask: np.ndarray) -> ROIBox:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        H, W = mask.shape[:2]
        return ROIBox(0, 0, float(W), float(H))
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    return ROIBox(x0, y0, x1, y1)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is not None:
        m = (mask > 0).astype(np.uint8)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        dt = dt / (dt.max() + 1e-6)
        return dt
    # Fallback: normalized distance to boundary by erosion count heuristic
    m = (mask > 0).astype(np.uint8)
    dist = np.zeros_like(m, dtype=np.float32)
    cur = m.copy()
    level = 0
    while cur.any():
        dist[cur > 0] = level
        # naive erosion
        pad = np.pad(cur, 1, mode="constant")
        eroded = (
            pad[:-2, :-2]
            & pad[:-2, 1:-1]
            & pad[:-2, 2:]
            & pad[1:-1, :-2]
            & pad[1:-1, 1:-1]
            & pad[1:-1, 2:]
            & pad[2:, :-2]
            & pad[2:, 1:-1]
            & pad[2:, 2:]
        ).astype(np.uint8)
        if np.array_equal(eroded, cur):
            break
        cur = eroded
        level += 1
    dist = dist / max(1, dist.max())
    return dist


def _sample_inside(mask: np.ndarray, B: ROIBox, num: int) -> List[Tuple[float, float]]:
    # Prefer interior peaks but only a few points
    dt = _distance_transform(mask)
    H, W = mask.shape[:2]
    pts = []
    vals = []
    m = mask > 0
    ys, xs = np.where(m)
    if ys.size == 0:
        return []
    # Subsample pixels uniformly then take top by dt
    step = max(1, int(round(max(1.0, min(B.w, B.h) / 50))))
    for y in ys[::step]:
        for x in xs[::step]:
            pts.append((float(x), float(y)))
            vals.append(float(dt[int(y), int(x)]))
    if not pts:
        return []
    order = np.argsort(-np.array(vals))
    sel = [pts[int(i)] for i in order[:num]]
    return sel


def _compute_band(mask: np.ndarray, B: ROIBox, k: int) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None
    m = (mask > 0).astype(np.uint8)
    if cv2 is not None:
        ksize = max(1, int(round(0.01 * max(1.0, min(B.w, B.h)))))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1))
        dil = cv2.dilate(m, ker, iterations=k)
        ero = cv2.erode(m, ker, iterations=k)
        band = ((dil > 0) & (ero == 0)).astype(np.uint8)
    else:
        from scipy.ndimage import binary_dilation, binary_erosion  # type: ignore
        band = (binary_dilation(m, iterations=k) & (~binary_erosion(m, iterations=k))).astype(np.uint8)
    return band


def _sample_band(mask: np.ndarray, B: ROIBox, k: int, num: int) -> List[Tuple[float, float]]:
    """Sample along boundary band using contour points if available."""
    try:
        import cv2
    except Exception:
        cv2 = None
    band = _compute_band(mask, B, k)
    pts: List[Tuple[float, float]] = []
    if cv2 is not None:
        contours, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_pts: List[Tuple[float, float]] = []
        for cnt in contours:
            for p in cnt:
                x, y = float(p[0][0]), float(p[0][1])
                all_pts.append((x, y))
        if len(all_pts) <= num:
            pts = all_pts
        else:
            idx = np.linspace(0, len(all_pts) - 1, num=num, dtype=int)
            pts = [all_pts[i] for i in idx]
    else:
        # Fallback: uniform grid restricted to band region
        step = max(2, int(round(min(B.w, B.h) / 40)))
        grid = uniform_grid_in_box(B.x0, B.y0, B.x1, B.y1, step)
        pts = [(x, y) for (x, y) in grid if band[int(round(y)), int(round(x))] > 0]
        if len(pts) > num:
            idx = np.linspace(0, len(pts) - 1, num=num, dtype=int)
            pts = [pts[i] for i in idx]
    return pts


def _sample_outside(mask: np.ndarray, B: ROIBox, num: int) -> List[Tuple[float, float]]:
    H, W = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)
    # Use a thin outer ring as candidate region
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is not None:
        ker = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(m, ker, iterations=4)
        outer = (dil > 0) & (m == 0)
    else:
        outer = (np.pad(m, 1, mode="edge")[1:-1, 1:-1] > 0) & (m == 0)
    ys, xs = np.where(outer)
    if xs.size == 0:
        return []
    step = max(1, int(round(max(1.0, min(B.w, B.h)) / 60)))
    coords = list(zip(xs.tolist()[::step], ys.tolist()[::step]))
    coords = [(float(x), float(y)) for (x, y) in coords]
    if len(coords) > num:
        idx = np.linspace(0, len(coords) - 1, num=num, dtype=int)
        coords = [coords[i] for i in idx]
    return coords


def sample_candidates(mask: np.ndarray, B: ROIBox, max_total: int = 30, nms_r_ratio: float = 0.06) -> List[Candidate]:
    """Boundary-focused candidate sampling.

    Strategy:
      - Majority from boundary band (adaptive by boundary length)
      - Few inside/outside for validation
      - NMS radius scales with target bbox, not ROI
    """
    H, W = mask.shape[:2]
    Bc = B.clip_to(W, H)
    Mb = _mask_bbox(mask)
    # Estimate boundary length by band pixel count (k=1)
    band = _compute_band(mask, Bc, k=3)
    boundary_len = int((band > 0).sum())
    # Adaptive total based on boundary length
    target_total = int(max(20, min(max_total, boundary_len / 20.0)))
    # Quotas: band 70%, inside 15%, outside 15%
    band_q = max(4, int(round(target_total * 0.65)))
    inside_q = max(1, int(round(target_total * 0.10)))
    outside_q = max(1, target_total - band_q - inside_q)

    inside_pts = _sample_inside(mask, Mb, num=inside_q)
    band_pts = _sample_band(mask, Mb, k=2, num=band_q)
    outside_pts = _sample_outside(mask, Mb, num=outside_q)

    pts = inside_pts + band_pts + outside_pts
    kinds = (["inside"] * len(inside_pts)) + (["band"] * len(band_pts)) + (["outside"] * len(outside_pts))
    # bias NMS ordering: band highest, inside mid, outside low (we want boundary first)
    kind_score = {"band": 1.0, "inside": 0.7, "outside": 0.4}
    scores = [kind_score[k] for k in kinds]
    # NMS radius based on mask bbox (target size)
    r = nms_r_ratio * Mb.diag
    keep = nms_points(pts, scores, r)
    kept = [Candidate(xy=pts[i], idx=i, kind=kinds[i], score=scores[i]) for i in keep]
    kept = kept[:max_total]
    for j, c in enumerate(kept):
        c.idx = j
    return kept

