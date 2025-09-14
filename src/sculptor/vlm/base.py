from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class VLMBase(ABC):
    """Abstract VLM interface with two tasks: peval (mask-level) and pgen (patch-level)."""

    @abstractmethod
    def peval(self, image_overlay_rgb: np.ndarray, depth_vis: Optional[np.ndarray], instance: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def pgen(self, patch_rgb: np.ndarray, instance: str, key_cues: str) -> Dict[str, Any]:
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

