"""
【候选点采样模块】
作用：在初始掩码的ROI区域内生成三类候选控制点（内部/边界/外部）

与下游模块关系：
  - 输出给patches.py进行图像块提取
  - 输出给select_points.py进行VLM点选择
  - 最终传递给sam_refine.py作为SAM控制点

算法特点：
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
