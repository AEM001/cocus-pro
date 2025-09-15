from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class VLMBase(ABC):
    """重构后的VLM接口，只支持anchor/quadrant指导的分割优化"""

    @abstractmethod
    def choose_anchors(self, image_with_anchors_rgb: np.ndarray, instance: str, global_reason: Optional[str] = None) -> Dict[str, Any]:
        """Return {"anchors_to_refine": [{"id": int, "reason": str}, ...]}"""
        ...

    @abstractmethod
    def quadrant_edits(self, quadrant_crop_rgb: np.ndarray, instance: str, anchor_id: int, global_reason: Optional[str] = None, anchor_reason: Optional[str] = None) -> Dict[str, Any]:
        """Return {"anchor_id": i, "edits": [{"region_id": 1-4, "action": "pos"|"neg", "why": str}]}"""
        ...


def try_parse_json(text: str) -> Dict[str, Any]:
    import json

    try:
        return json.loads(text)
    except Exception:
        # naive bracket trimming fallback
        try:
            s = text[text.find("{") : text.rfind("}") + 1]
            return json.loads(s)
        except Exception:
            return {}
