from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class VLMBase(ABC):
    """VLM接口：仅保留直接点提议(propose_points)能力，移除所有锚点/象限相关逻辑"""

    @abstractmethod
    def propose_points(self, context_image_rgb: np.ndarray, instance: str, max_total: int = 10) -> Dict[str, Any]:
        """基于上下文图让VLM输出精细正/负点（像素坐标）。
        返回：{"pos_points": [(x,y),...], "neg_points": [(x,y),...], "raw_text": str}
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
