from __future__ import annotations
from typing import Dict, Optional

# 简化提示词，仅保留anchor/quadrant两类，并允许注入上下文（instance/reason）

def build_anchor_prompt(instance: str, global_reason: Optional[str] = None) -> Dict[str, str]:
    sys_msg = "You are a camouflaged-object segmentation assistant. Reply JSON only."
    user_msg = (
        f"Target instance: '{instance}'.\n"
        "Image shows ROI with 8 labeled anchors around the boundary.\n"
        "Green = current mask. Camouflaged objects may have weak edges and texture similarity to background.\n"
        "Focus on boundary inconsistency: missing parts at corners/edges, leaks into background, shape breaks.\n"
        "Task: pick 1-3 anchor ids most likely to improve the mask.\n"
        "IMPORTANT: Never return an empty list. If uncertain, output exactly one most promising id.\n"
        "Return JSON strictly:\n{\"anchors_to_refine\": [ { \"id\": 1-8, \"reason\": \"short justification (no numbers)\" } ]}"
    )
    return {"system": sys_msg, "user": user_msg}


def build_quadrant_prompt(instance: str, anchor_id: int, global_reason: Optional[str] = None, anchor_reason: Optional[str] = None) -> Dict[str, str]:
    sys_msg = "You are a camouflaged-object segmentation assistant. Reply JSON only."
    user_msg = (
        f"Target instance: '{instance}'.\n"
        f"Focus anchor id: {anchor_id}.\n"
        "Square around the anchor is split into 4 labeled regions {1..4}.\n"
        "Green = current mask. Camouflaged objects: look for subtle shape/texture continuity vs background sand/rock.\n"
        "Decide where to add points for next SAM: POS (foreground) or NEG (background). Use at most 2 per type.\n"
        "Return JSON strictly:\n{\"anchor_id\": i, \"edits\": [ { \"region_id\": 1-4, \"action\": \"pos\"|\"neg\", \"why\": \"short justification (no numbers)\" } ]}"
    )
    return {"system": sys_msg, "user": user_msg}
