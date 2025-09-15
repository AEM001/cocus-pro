from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np

from .base import VLMBase


class MockVLM(VLMBase):
    """重构后的Mock VLM，只支持anchor/quadrant指导"""

    def __init__(self, seed: int = 123):
        self.rng = random.Random(seed)

    def choose_anchors(self, image_with_anchors_rgb: np.ndarray, instance: str) -> Dict[str, Any]:
        _ = instance
        # 固定推荐：顶部中点(2)和右侧中点(4)
        return {"anchors_to_refine": [
            {"id": 2, "reason": "top boundary needs refinement"}, 
            {"id": 4, "reason": "right boundary needs refinement"}
        ]}

    def quadrant_edits(self, quadrant_crop_rgb: np.ndarray, instance: str, anchor_id: int) -> Dict[str, Any]:
        _ = (instance, anchor_id)
        # 简单模式：区域2添加正点，区域4添加负点
        return {"anchor_id": int(anchor_id), "edits": [
            {"region_id": 2, "action": "pos", "why": "foreground region"}, 
            {"region_id": 4, "action": "neg", "why": "background region"}
        ]}

