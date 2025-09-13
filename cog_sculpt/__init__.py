"""Cog-Sculpt: ROI→子格→语义打分→正/负点→SAM 迭代

模块化骨架，便于替换/扩展：
- config: 全局配置
- types: 基本数据结构
- io: 路径解析与读写（支持 box_out 初始掩码）
- roi: ROI / BBox 计算（含padding）
- grid: 网格分块与递归细分
- alpha: 基于掩码构建 Alpha 软先验
- scorer: 语义打分接口与占位实现
- points: 正/负点选择
- metrics: IoU、平滑
- sam_iface: SAM/SAM2 封装接口
- pipeline: 主流程编排
- cli: 命令行入口
"""

from .core import *  # noqa: F401,F403

__all__ = [
    "Config",
    "Box",
    "Point",
    "Cell",
    "load_image",
    "load_mask_png",
    "save_mask_png",
    "ensure_dir",
    "read_json",
    "maybe_read_boxes",
    "bbox_from_mask",
    "overlay",
    "save_image",
    "roi_to_bbox",
    "box_from_xyxy_clip",
    "partition_grid",
    "subdivide_cell",
    "build_alpha_for_cell",
    "compose_rgba",
    "RegionScorer",
    "AlphaCLIPScorer",
    "SoftFocusCLIPScorer",
    "DensityScorer",
    "select_points_from_cells",
    "iou",
    "smooth_mask",
    "SamWrapper",
    "threshold_cells",
    "sculpting_pipeline",
]
