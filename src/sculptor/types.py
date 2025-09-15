"""
【数据类型定义模块】
作用：定义Sculptor系统中使用的核心数据结构和类型
核心类型：
  - ROIBox: 感兴趣区域边界框，支持坐标裁剪和几何计算
  - Candidate: 候选点数据结构，包含位置、类型、分数等元信息
  - PointArray: 点坐标数组类型别名

设计特点：
  - 使用dataclass确保数据不可变性和类型安全
  - 提供边界框的几何属性计算（宽、高、对角线）
  - 支持图像边界裁剪，防止越界访问

模块关系：
  - 被所有其他模块import作为基础类型定义
  - candidates.py生成Candidate对象列表
  - select_points.py处理Candidate对象进行点选择
  - sam_refine.py使用ROIBox定义处理区域
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ROIBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def diag(self) -> float:
        return (self.w**2 + self.h**2) ** 0.5

    def clip_to(self, W: int, H: int) -> "ROIBox":
        x0 = max(0.0, min(float(W - 1), self.x0))
        y0 = max(0.0, min(float(H - 1), self.y0))
        x1 = max(0.0, min(float(W), self.x1))
        y1 = max(0.0, min(float(H), self.y1))
        return ROIBox(x0, y0, x1, y1)


@dataclass
class Candidate:
    xy: Tuple[float, float]  # absolute image coordinates (x,y)
    idx: int                 # unique index within batch
    kind: str                # "inside" | "band" | "outside"
    score: float = 0.0       # optional pre-score for NMS ordering


PointArray = List[Tuple[float, float]]

