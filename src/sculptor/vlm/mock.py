from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np

from .base import VLMBase


class MockVLM(VLMBase):
    """A deterministic-ish mock VLM for development without a real model."""

    def __init__(self, seed: int = 123):
        self.rng = random.Random(seed)

    def peval(self, image_overlay_rgb: np.ndarray, depth_vis: Optional[np.ndarray], instance: str) -> Dict[str, Any]:
        _ = (image_overlay_rgb, depth_vis, instance)
        # trivial template
        return {
            "missing_parts": [],
            "over_segments": [],
            "boundary_quality": "soft",
            "key_cues": ["overall shape", "texture", "edges"],
        }

    def pgen(self, patch_rgb: np.ndarray, instance: str, key_cues: str) -> Dict[str, Any]:
        _ = (instance, key_cues)
        # Use mean intensity as a weak signal to produce stable scores
        m = float(np.mean(patch_rgb)) / 255.0
        conf = max(0.0, min(1.0, 1.2 * (m - 0.4)))  # favor mid/bright patches a bit
        is_target = conf > 0.55
        return {"is_target": is_target, "conf": conf}

