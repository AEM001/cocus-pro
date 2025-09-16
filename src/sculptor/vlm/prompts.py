from __future__ import annotations
from typing import Dict, Optional, List, Union

SemanticCtx = Dict[str, Union[List[str], str]]

def _fmt_list(xs: Optional[List[str]]) -> str:
    return ", ".join(xs) if xs else "none"

def build_anchor_prompt(
    instance: str,
    global_reason: Optional[str] = None,
    *,
    semantic: Optional[SemanticCtx] = None,
    K: int = 3
) -> Dict[str, str]:
    """
    不改变原有返回结构；新增 semantic 语义块与 K（最多返回多少个 anchor）
    semantic 可包含键：
      - synonyms, salient_cues, distractors, shape_prior, texture_prior, scene_context, not_target_parts
    """
    sem = semantic or {}
    sys_msg = (
        "You are a segmentation expert. Reply ONLY with valid JSON. "
        "No markdown, no explanations, no extra text. "
        "Return only the JSON response."
    )

    user_msg = f"""
I need you to analyze this segmentation image and select anchors to improve the mask for '{instance}'.

The image shows numbered anchor points (1-8) around a green mask overlay. Your task: choose which anchors need refinement.

Return EXACTLY this JSON format:
{{
  "anchors_to_refine": [
    {{ "id": 1, "intent": "fix_leak", "reason": "brief reason" }},
    {{ "id": 3, "intent": "recover_miss", "reason": "brief reason" }}
  ]
}}

Rules:
- Use anchor ids 1-8 only
- Intent: "fix_leak" (green spills to background) or "recover_miss" (missing target parts)
- Select 1-3 anchors maximum
- Keep reasons short (no quotes inside text)

Example response:
{{"anchors_to_refine":[{{"id":2,"intent":"fix_leak","reason":"background intrusion"}},{{"id":6,"intent":"recover_miss","reason":"missing edge"}}]}}
"""
    # 可选：把 global_reason 作为“全局上下文”补充
    if global_reason:
        user_msg = "Global context: " + global_reason.strip() + "\n\n" + user_msg

    return {"system": sys_msg, "user": user_msg}


def build_quadrant_prompt(
    instance: str,
    anchor_id: int,
    global_reason: Optional[str] = None,
    *,
    semantic: Optional[SemanticCtx] = None,
    anchor_hint: Optional[str] = None,
    max_pos: int = 1,
    max_neg: int = 1
) -> Dict[str, str]:
    """
    基于切线内外区域的提示词构建。
    """
    sem = semantic or {}
    sys_msg = (
        "You are a segmentation expert. Reply ONLY with valid JSON. "
        "No markdown, no explanations, no extra text."
    )

    user_msg = f"""
Analyze this anchor point image for '{instance}' segmentation.

The square region is divided into:
- Region 1 (Inner): toward object interior
- Region 2 (Outer): toward background

Select ONE region and action:
- "pos" = include this region in mask
- "neg" = exclude this region from mask

Return EXACTLY this JSON:
{{
  "anchor_id": {anchor_id},
  "edits": [
    {{ "region_id": 1, "action": "pos", "why": "brief reason" }}
  ]
}}

Example:
{{"anchor_id":5,"edits":[{{"region_id":2,"action":"neg","why":"background leak"}}]}}
"""
    # 可选：附加来自上一轮 anchor 决策的线索
    if anchor_hint:
        user_msg = f"Anchor hint: {anchor_hint.strip()}\n\n" + user_msg
    if global_reason:
        user_msg = "Global context: " + global_reason.strip() + "\n\n" + user_msg

    return {"system": sys_msg, "user": user_msg}
