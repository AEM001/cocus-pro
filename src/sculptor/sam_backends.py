"""
【SAM后端适配器模块】
作用：提供统一的SAM模型接口，屏蔽不同SAM实现的差异
核心功能：
  - SegmentAnythingAdapter: 封装Meta SAM预测器，统一坐标变换和掩码选择
  - build_sam_backend: 动态加载SAM模型权重，支持多种模型规格
  - 设备自动选择：CUDA/CPU自动适配
  - 掩码标准化：统一输出0/255格式的uint8掩码

与系统其他模块关系：
  - 被sam_refine.py调用作为SAM预测器
  - 提供标准化的predict接口，接收点坐标和标签
  - 处理坐标变换和掩码选择逻辑

技术特点：
  - 适配器模式：隐藏底层SAM实现细节
  - 自动坐标变换：将图像坐标映射到SAM输入空间
  - 鲁棒的异常处理：缺失依赖时的友好错误提示
  - 支持多模型规格：vit_h/vit_l/vit_b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SegmentAnythingAdapter:
    """Adapter over Meta's segment_anything.SamPredictor.

    This class hides set_image, coordinate transforms, mask selection, and returns a uint8 mask 0/255.
    """

    predictor: object
    device: Optional[str] = None

    _last_image_shape: Optional[Tuple[int, int]] = None

    def _ensure_image(self, image_rgb: np.ndarray) -> None:
        H, W = image_rgb.shape[:2]
        if self._last_image_shape != (H, W):
            # SamPredictor handles device internally via model placement
            self.predictor.set_image(image_rgb)
            self._last_image_shape = (H, W)

    def predict(self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray, box_xyxy: Tuple[float, float, float, float], prev_mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._ensure_image(image_rgb)
        # IMPORTANT: Do NOT pre-transform coordinates here. SamPredictor.predict will handle
        # coordinate transforms internally for numpy API in most versions. Pre-transforming
        # would cause a double transform and severe misalignment.
        pt_coords = points.astype(np.float32)
        if pt_coords.ndim == 1:
            pt_coords = pt_coords[None, :]
        box_vec = np.array(box_xyxy, dtype=np.float32)

        masks, scores, logits = self.predictor.predict(
            point_coords=pt_coords if len(pt_coords) > 0 else None,
            point_labels=labels.astype(np.int32) if len(labels) > 0 else None,
            box=box_vec,
            multimask_output=True,
        )
        # Choose best proposal
        sel = 0
        if prev_mask is not None and isinstance(masks, np.ndarray) and masks.ndim == 3 and masks.shape[1:] == prev_mask.shape[:2]:
            # Compute IoU within the ROI box to prefer masks that align with previous/prior mask
            H, W = prev_mask.shape[:2]
            x0, y0, x1, y1 = int(max(0, min(W - 1, box_xyxy[0]))), int(max(0, min(H - 1, box_xyxy[1]))), int(max(0, min(W, box_xyxy[2]))), int(max(0, min(H, box_xyxy[3])))
            x1 = max(x1, x0 + 1); y1 = max(y1, y0 + 1)
            pm = (prev_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
            best_iou = -1.0
            for i in range(masks.shape[0]):
                mi = (masks[i, y0:y1, x0:x1] > 0).astype(np.uint8)
                inter = int((pm & mi).sum())
                union = int((pm | mi).sum()) or 1
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    sel = i
        elif isinstance(scores, np.ndarray) and scores.size > 0:
            sel = int(np.argmax(scores))
        m = masks[sel]
        return (m.astype(np.uint8) > 0).astype(np.uint8) * 255


def build_sam_backend(checkpoint_path: str, model_type: str = "vit_h", device: Optional[str] = None) -> SegmentAnythingAdapter:
    """Load a SAM predictor from a checkpoint using segment_anything.

    Args:
      checkpoint_path: Path to SAM weights (.pth)
      model_type: One of {'vit_h','vit_l','vit_b'} matching the checkpoint
      device: Optional device string ('cuda' or 'cpu'); if None, inferred by torch
    """
    try:
        import torch  # type: ignore
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore
    except Exception as e:
        raise ImportError(
            "segment_anything and torch are required for SAM integration. "
            "Please install them and ensure the checkpoint path is valid."
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type not in {"vit_h", "vit_l", "vit_b"}:
        raise ValueError(f"Unsupported model_type '{model_type}'. Use one of vit_h|vit_l|vit_b.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    return SegmentAnythingAdapter(predictor=predictor, device=device)

