"""
【VLM点选择策略模块】
作用：基于VLM分析结果选择最优的控制点用于SAM精细雕刻
核心功能：
  - 自适应阈值估计：基于分数分布动态确定正负样本阈值
  - 空间NMS去重：确保控制点空间分布均匀
  - 分层选择策略：优先选择高置信度内部点，低置信度外部点
  - 鲁棒阈值回退：统计失效时使用固定阈值

与上下游模块关系：
  - 接收vlm.py输出的候选点分数
  - 使用candidates.py生成的Candidate对象
  - 输出控制点给sam_refine.py进行掩码精炼

算法流程：
  1. 基于分数分布估计正负阈值
  2. 根据阈值筛选正负样本池
  3. 应用空间NMS去除冗余点
  4. 按置信度排序选择top-K点
  5. 返回控制点坐标和标签（1=正样本，0=负样本）

技术特点：
  - 自适应阈值避免人工调参
  - 空间NMS半径基于ROI对角线比例（6%）
  - 支持自定义K值控制控制点数量
"""

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

