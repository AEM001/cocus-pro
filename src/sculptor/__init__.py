"""
Sculptor: VLM-driven point sculpting around an initial mask and ROI.

Modules:
- candidates: candidate point sampling inside/band/outside ROI
- patches: multi-scale patch extraction and light augmentations
- vlm: model-agnostic interface + Qwen2.5-VL stub and prompts
- select_points: selection with thresholds + spatial NMS
- sam_refine: SAM predictor hook and early-stop utilities
- utils: image/mask helpers and visualizations
"""

__all__ = [
    "candidates",
    "patches",
    "vlm",
    "select_points",
    "sam_refine",
    "utils",
]

