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

import math
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
        # Compute DT on the given mask only (caller should crop to ROI)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        mx = float(dt.max())
        if mx > 0:
            dt = dt / mx
        return dt.astype(np.float32)
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
    mxx = float(dist.max())
    if mxx > 0:
        dist = dist / mxx
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
        # Use CHAIN_APPROX_SIMPLE to reduce memory, and subsample evenly across contours
        contours, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_needed = max(1, num)
        # Flatten by taking uniform steps along each contour until reaching total_needed
        for cnt in contours:
            if total_needed <= 0:
                break
            cnt_len = len(cnt)
            if cnt_len == 0:
                continue
            step = max(1, cnt_len // max(1, min(total_needed, 8)))
            indices = list(range(0, cnt_len, step))
            for i in indices:
                x, y = float(cnt[i][0][0]), float(cnt[i][0][1])
                pts.append((x, y))
                total_needed -= 1
                if total_needed <= 0:
                    break
        if len(pts) > num:
            idx = np.linspace(0, len(pts) - 1, num=num, dtype=int).tolist()
            pts = [pts[i] for i in idx]
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
        dil = cv2.dilate(m, ker, iterations=2)  # reduce iterations to limit memory
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


def _roi_int_bounds(B: ROIBox, H: int, W: int):
    x0 = max(0, min(W - 1, int(round(B.x0))))
    y0 = max(0, min(H - 1, int(round(B.y0))))
    x1 = max(0, min(W, int(round(B.x1))))
    y1 = max(0, min(H, int(round(B.y1))))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1


def sample_candidates(mask: np.ndarray, B: ROIBox, max_total: int = 30, nms_r_ratio: float = 0.06) -> List[Candidate]:
    """Lightweight candidate sampling (ROI-limited, no distance transform, no contours).

    Strategy:
      - Use a coarse grid inside the ROI
      - Classify grid points by a tiny 3x3 window on the binary mask: inside / boundary / outside
      - Allocate quotas favoring boundary, but avoid expensive image-wide ops
    """
    H, W = mask.shape[:2]
    Bc = B.clip_to(W, H)
    x0, y0, x1, y1 = _roi_int_bounds(Bc, H, W)
    roi = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    Hr, Wr = roi.shape[:2]
    if Hr == 0 or Wr == 0:
        return []

    # Target counts (reduced and fixed cap)
    hard_cap = min(max_total, 18)
    target_total = max(10, hard_cap)
    band_q = max(4, int(round(target_total * 0.60)))
    inside_q = max(2, int(round(target_total * 0.20)))
    outside_q = max(2, target_total - band_q - inside_q)

    # Coarse grid step selected to roughly yield the needed number of points
    # step ~ sqrt(area / (target_total * c))
    area = max(1, Hr * Wr)
    step = int(max(6, min(64, math.sqrt(area / max(1.0, target_total * 3.0)))))

    inside_pts: List[Tuple[float, float]] = []
    band_pts: List[Tuple[float, float]] = []
    outside_pts: List[Tuple[float, float]] = []

    # Iterate coarse grid centers
    for yy in range(step // 2, Hr, step):
        if len(inside_pts) >= inside_q and len(band_pts) >= band_q and len(outside_pts) >= outside_q:
            break
        for xx in range(step // 2, Wr, step):
            if len(inside_pts) >= inside_q and len(band_pts) >= band_q and len(outside_pts) >= outside_q:
                break
            y = yy
            x = xx
            y0n = max(0, y - 1); y1n = min(Hr, y + 2)
            x0n = max(0, x - 1); x1n = min(Wr, x + 2)
            win = roi[y0n:y1n, x0n:x1n]
            vmin = int(win.min())
            vmax = int(win.max())
            center = int(roi[y, x])
            is_band = (vmin == 0 and vmax == 1)
            if is_band and len(band_pts) < band_q:
                band_pts.append((float(x + x0), float(y + y0)))
            elif center == 1 and len(inside_pts) < inside_q:
                inside_pts.append((float(x + x0), float(y + y0)))
            elif center == 0 and len(outside_pts) < outside_q:
                outside_pts.append((float(x + x0), float(y + y0)))

    # If quotas not met, relax classification priority to fill remaining
    grid_pts: List[Tuple[float, float, str]] = []
    for yy in range(step // 2, Hr, step):
        for xx in range(step // 2, Wr, step):
            y = yy; x = xx
            center = int(roi[y, x])
            kind = "inside" if center == 1 else "outside"
            grid_pts.append((float(x + x0), float(y + y0), kind))
    if len(band_pts) < band_q:
        for px, py, _ in grid_pts:
            if len(band_pts) >= band_q:
                break
            band_pts.append((px, py))
    if len(inside_pts) < inside_q:
        for px, py, kind in grid_pts:
            if len(inside_pts) >= inside_q:
                break
            if kind == "inside":
                inside_pts.append((px, py))
    if len(outside_pts) < outside_q:
        for px, py, kind in grid_pts:
            if len(outside_pts) >= outside_q:
                break
            if kind == "outside":
                outside_pts.append((px, py))

    # Compose and lightly NMS (few points, cheap)
    pts = inside_pts + band_pts + outside_pts
    kinds = (["inside"] * len(inside_pts)) + (["band"] * len(band_pts)) + (["outside"] * len(outside_pts))
    scores = [1.0 if k == "band" else (0.7 if k == "inside" else 0.4) for k in kinds]
    Mb_local = ROIBox(0, 0, float(Wr), float(Hr))
    r = nms_r_ratio * Mb_local.diag
    keep = nms_points(pts, scores, r)
    kept = [Candidate(xy=pts[i], idx=i, kind=kinds[i], score=scores[i]) for i in keep]
    kept = kept[:hard_cap]
    for j, c in enumerate(kept):
        c.idx = j
    return kept

