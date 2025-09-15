from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np

from .base import VLMBase


class MockVLM(VLMBase):
    """重构后的Mock VLM，只支持anchor/quadrant指导"""

    def __init__(self, seed: int = 123):
        self.rng = random.Random(seed)

    def choose_anchors(self, image_with_anchors_rgb: np.ndarray, instance: str, global_reason: Optional[str] = None) -> Dict[str, Any]:
        _ = (instance, global_reason, image_with_anchors_rgb)
        # 固定推荐：顶部中点(2)和右侧中点(4)
        return {"anchors_to_refine": [
            {"id": 2, "reason": global_reason or "top boundary needs refinement"}, 
            {"id": 4, "reason": global_reason or "right boundary needs refinement"}
        ]}

    def quadrant_edits(self, quadrant_crop_rgb: np.ndarray, instance: str, anchor_id: int, global_reason: Optional[str] = None, anchor_reason: Optional[str] = None) -> Dict[str, Any]:
        _ = (instance, anchor_id, global_reason, anchor_reason, quadrant_crop_rgb)
        # 简单模式：区域2添加正点，区域4添加负点，why中合并上下文
        why_pos = anchor_reason or global_reason or "foreground region"
        why_neg = anchor_reason or global_reason or "background region"
        return {"anchor_id": int(anchor_id), "edits": [
            {"region_id": 2, "action": "pos", "why": str(why_pos)}, 
            {"region_id": 4, "action": "neg", "why": str(why_neg)}
        ]}

