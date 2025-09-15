"""
【SAM精细雕刻核心模块】
作用：使用SAM模型对初始掩码进行迭代式精细化雕刻
核心功能：
  - SAM预测器封装：统一接口适配不同SAM后端
  - 迭代精炼算法：基于控制点逐步优化掩码边界
  - 早停机制：基于IoU变化的收敛判断
  - 容错处理：异常情况下返回原始掩码

与上下游模块关系：
  - 接收select_points.py选择的控制点（正/负样本点）
  - 使用candidates.py生成的ROI边界框
  - 输出最终精细掩码给下游可视化模块

算法流程：
  1. 初始化SAM预测器
  2. 设置控制点（正样本=内部点，负样本=外部点）
  3. 迭代预测直到收敛或最大迭代次数
  4. 应用早停判断IoU变化
  5. 返回最终精细掩码

技术特点：
  - 支持自定义SAM后端适配器
  - 鲁棒的异常处理机制
  - 自适应收敛阈值（默认IoU变化<0.5%）
"""

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
            return self.backend.predict(image_rgb, points, labels, xyxy)
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
