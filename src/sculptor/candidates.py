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

# 内部/边界/外部候选采样，带空间NMS
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .types import Candidate, ROIBox
from .utils import nms_points, uniform_grid_in_box


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
    dt = _distance_transform(mask)
    H, W = mask.shape[:2]
    # sample grid, sort by dt
    step = max(2, int(round(min(B.w, B.h) / 20)))
    xs = np.arange(int(B.x0), int(B.x1), step)
    ys = np.arange(int(B.y0), int(B.y1), step)
    pts = []
    vals = []
    m = mask > 0
    for y in ys:
        for x in xs:
            if 0 <= x < W and 0 <= y < H and m[int(y), int(x)]:
                pts.append((float(x), float(y)))
                vals.append(float(dt[int(y), int(x)]))
    if not pts:
        return []
    order = np.argsort(-np.array(vals))
    sel = [pts[int(i)] for i in order[:num]]
    return sel


def _sample_band(mask: np.ndarray, B: ROIBox, k: int, num: int) -> List[Tuple[float, float]]:
    try:
        import cv2
    except Exception:
        cv2 = None
    m = (mask > 0).astype(np.uint8)
    if cv2 is not None:
        ksize = max(1, int(round(0.01 * B.diag)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1))
        dil = cv2.dilate(m, ker, iterations=k)
        ero = cv2.erode(m, ker, iterations=k)
        band = ((dil > 0) & (ero == 0)).astype(np.uint8)
    else:
        from scipy.ndimage import binary_dilation, binary_erosion  # type: ignore

        band = (binary_dilation(m, iterations=k) & (~binary_erosion(m, iterations=k))).astype(np.uint8)
    # uniform in band with stride
    step = max(2, int(round(min(B.w, B.h) / 30)))
    pts = uniform_grid_in_box(B.x0, B.y0, B.x1, B.y1, step)
    pts = [(x, y) for (x, y) in pts if band[int(round(y)), int(round(x))] > 0]
    if len(pts) <= num:
        return pts
    idx = np.linspace(0, len(pts) - 1, num=num, dtype=int)
    return [pts[i] for i in idx]


def _sample_outside(mask: np.ndarray, B: ROIBox, num: int) -> List[Tuple[float, float]]:
    H, W = mask.shape[:2]
    step = max(2, int(round(min(B.w, B.h) / 20)))
    xs = np.arange(int(B.x0), int(B.x1), step)
    ys = np.arange(int(B.y0), int(B.y1), step)
    pts = []
    vals = []
    m = mask > 0
    # local variance heuristic to pick textured background
    win = max(3, int(round(0.01 * B.diag)) | 1)  # odd
    pad = win // 2
    padded = np.pad(m.astype(np.float32), pad, mode="edge")
    for y in ys:
        for x in xs:
            if 0 <= x < W and 0 <= y < H and not m[int(y), int(x)]:
                y0 = int(y) + pad
                x0 = int(x) + pad
                patch = padded[y0 - pad : y0 + pad + 1, x0 - pad : x0 + pad + 1]
                var = float(patch.var())
                pts.append((float(x), float(y)))
                vals.append(var)
    if not pts:
        return []
    order = np.argsort(-np.array(vals))
    sel = [pts[int(i)] for i in order[:num]]
    return sel


def sample_candidates(mask: np.ndarray, B: ROIBox, max_total: int = 30, nms_r_ratio: float = 0.06) -> List[Candidate]:
    """Sample candidate points from inside/band/outside of the current mask within ROI.

    Returns a list of Candidate with absolute coordinates.
    """
    H, W = mask.shape[:2]
    Bc = B.clip_to(W, H)
    # initial quotas (will be trimmed by NMS)
    quota = max_total // 3
    inside_pts = _sample_inside(mask, Bc, num=quota)
    band_pts = _sample_band(mask, Bc, k=3, num=quota)
    outside_pts = _sample_outside(mask, Bc, num=max_total - 2 * quota)

    pts = inside_pts + band_pts + outside_pts
    kinds = (["inside"] * len(inside_pts)) + (["band"] * len(band_pts)) + (["outside"] * len(outside_pts))
    # rudimentary pre-scores to bias NMS ordering: inside high, band mid, outside low
    kind_score = {"inside": 1.0, "band": 0.6, "outside": 0.4}
    scores = [kind_score[k] for k in kinds]
    r = nms_r_ratio * Bc.diag
    keep = nms_points(pts, scores, r)
    kept = [Candidate(xy=pts[i], idx=i, kind=kinds[i], score=scores[i]) for i in keep]
    # trim to max_total
    kept = kept[:max_total]
    # reindex idx to 0..N-1 for downstream dicts
    for j, c in enumerate(kept):
        c.idx = j
    return kept

