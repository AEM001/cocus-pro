"""
【Sculptor模块总览】
作用：VLM引导的迭代式掩码精细化雕刻系统
核心功能：通过视觉语言模型(VLM)分析图像块，指导SAM模型进行精确的掩码边界雕刻

模块关系链：
  1. candidates.py → 生成候选点（内部/边界/外部）
  2. patches.py → 提取多尺度图像块
  3. vlm.py → VLM模型接口和提示工程
  4. select_points.py → 基于VLM输出选择最优控制点
  5. sam_refine.py → SAM精细雕刻核心算法
  6. utils.py → 通用工具函数和可视化

数据流向：
  初始掩码 → 候选点采样 → 图像块提取 → VLM分析 → 点选择 → SAM精炼 → 精细掩码
"""

__all__ = [
    "candidates",
    "patches",
    "vlm",
    "select_points",
    "sam_refine",
    "utils",
]

