"""
【多尺度图像块提取模块】
作用：围绕候选点提取多尺度图像块，为VLM分析提供视觉上下文
核心功能：
  - 多尺度正方形图像块提取（基于ROI尺寸自适应）
  - 边界处理与填充策略
  - 轻度数据增强（水平翻转、亮度/对比度调整）
  - 保持图像块中心对齐候选点

与上下游模块关系：
  - 接收candidates.py生成的候选点坐标
  - 输出给vlm.py进行视觉语言模型分析
  - 为select_points.py提供标准化的图像输入

技术特点：
  - 自适应尺度计算：scale * min(ROI宽, ROI高)
  - 最小8x8像素，最大不超过图像边界
  - 轻量级增强避免过度扰动
  - 支持RGB通道标准化
"""

# 多尺度补丁提取 + 轻度数据增强
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .types import ROIBox
from .utils import to_rgb


def _crop_square(img: np.ndarray, center_xy: Tuple[float, float], size: int) -> np.ndarray:
    H, W = img.shape[:2]
    cx, cy = center_xy
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size
    # clamp
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    patch = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    patch_y0 = y0c - y0
    patch_x0 = x0c - x0
    patch[patch_y0 : patch_y0 + (y1c - y0c), patch_x0 : patch_x0 + (x1c - x0c)] = to_rgb(img[y0c:y1c, x0c:x1c])
    return patch


def _augment_light(patch: np.ndarray) -> List[np.ndarray]:
    ps = [patch]
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is None:
        return ps
    # hflip
    ps.append(cv2.flip(patch, 1))
    # brightness/contrast tweaks
    for alpha, beta in ((1.1, 0), (0.9, 0), (1.0, 10), (1.0, -10)):
        p = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)
        ps.append(p)
    return ps[:3]  # keep light set


def multi_scale_patches(I: np.ndarray, pts: Sequence[Tuple[float, float]], B: ROIBox, scales: Sequence[float]) -> Dict[int, List[np.ndarray]]:
    """Extract multi-scale square patches centered at pts.

    size = scale * min(wB, hB).
    Returns dict idx -> List[patch_images].
    """
    H, W = I.shape[:2]
    sbase = max(4, int(round(min(B.w, B.h))))
    out: Dict[int, List[np.ndarray]] = {}
    for i, (x, y) in enumerate(pts):
        lst: List[np.ndarray] = []
        for s in scales:
            size = max(8, int(round(s * sbase)))
            if size % 2 == 0:
                size += 1
            patch = _crop_square(I, (x, y), size)
            lst.append(patch)
        out[i] = lst
    return out


def multi_scale_with_aug(I: np.ndarray, pts: Sequence[Tuple[float, float]], B: ROIBox, scales: Sequence[float], aug_light: bool = True) -> Dict[int, List[np.ndarray]]:
    base = multi_scale_patches(I, pts, B, scales)
    if not aug_light:
        return base
    out: Dict[int, List[np.ndarray]] = {}
    for i, lst in base.items():
        augged: List[np.ndarray] = []
        for p in lst:
            augged.extend(_augment_light(p))
        out[i] = augged
    return out

