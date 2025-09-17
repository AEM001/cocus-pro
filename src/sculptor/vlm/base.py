from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class VLMBase(ABC):
    """VLM接口，仅支持API模式的anchor/quadrant指导分割优化"""

    @abstractmethod
    def choose_anchors(self, image_with_anchors_rgb: np.ndarray, instance: str, global_reason: Optional[str] = None) -> Dict[str, Any]:
        """选择需要精修的锚点
        
        Returns:
            {"anchors_to_refine": [{"id": int, "reason": str}, ...], "raw_text": str}
        """
        ...

    @abstractmethod
    def quadrant_edits(self, quadrant_crop_rgb: np.ndarray, instance: str, anchor_id: int, 
                      global_reason: Optional[str] = None, anchor_reason: Optional[str] = None) -> Dict[str, Any]:
        """对象限区域进行编辑分析
        
        Returns:
            {"anchor_id": int, "edits": [{"region_id": 1-2, "action": "pos"|"neg", "why": str}], "raw_text": str}
        """
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
