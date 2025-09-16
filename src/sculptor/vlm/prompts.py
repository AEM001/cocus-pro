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
        "You are a camouflaged-object segmentation assistant. "
        "Reply with JSON only. No prose, no markdown, no trailing commas. "
        "The JSON must be valid for Python json.loads."
    )

    user_msg = f"""
Target instance (canonical, singular): '{instance}'
Synonyms/aliases: {_fmt_list(sem.get('synonyms'))}
Salient cues (positive): {_fmt_list(sem.get('salient_cues'))}
Distractors (background lookalikes): {_fmt_list(sem.get('distractors'))}
Shape prior: {sem.get('shape_prior') or "unknown"}
Texture/color prior: {_fmt_list(sem.get('texture_prior'))}
Scene context: {sem.get('scene_context') or "unknown"}
NOT-target parts to avoid: {_fmt_list(sem.get('not_target_parts'))}

Green overlay = current mask within the ROI. Anchors 1..8 are placed around boundary.
Your goal: choose up to {K} anchors to refine so the mask better matches ONLY the target instance.

Internal evaluation checklist (DO NOT OUTPUT):
1) Semantic gating: prefer anchors where local pattern matches the target cues and conflicts with distractors.
2) Leakage vs Missing:
   - If green spills onto background patterns -> intent="fix_leak".
   - If true target parts (texture/shape continuity) are missing -> intent="recover_miss".
3) Geometry sanity: respect silhouette continuity along tangent; avoid breaking plausible outline.
4) Uncertainty fallback: if unsure, output exactly one anchor with the most plausible intent.
5) Never propose anchors outside the ROI or on clearly NOT-target parts.

Return JSON STRICTLY:
{{
  "anchors_to_refine": [
    {{ "id": 1-8, "intent": "fix_leak" | "recover_miss", "reason": "short justification (no numbers)" }}
  ]
}}
Constraints:
- 1 <= len(anchors_to_refine) <= {max(1, K)}
- Use only ids from 1..8
- Keep justifications short; no lists or numbering; no quotes in the text

Few-shot examples (DO NOT COPY, DO NOT ECHO; FORMAT ONLY):
Input→ (leak suspected near rock ripples)
Output→ {{"anchors_to_refine":[{{"id":3,"intent":"fix_leak","reason":"background ripples intrude"}},{{"id":7,"intent":"recover_miss","reason":"target texture continues"}}]}}

Input→ (only one doubtful area)
Output→ {{"anchors_to_refine":[{{"id":5,"intent":"recover_miss","reason":"missing fin edge continuity"}}]}}
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
        "You are a camouflaged-object segmentation assistant. "
        "Reply with JSON only. No extra text. Valid JSON required."
    )

    user_msg = f"""
Target instance (canonical): '{instance}'
Synonyms: {_fmt_list(sem.get('synonyms'))}
Salient cues: {_fmt_list(sem.get('salient_cues'))}
Distractors: {_fmt_list(sem.get('distractors'))}
NOT-target parts: {_fmt_list(sem.get('not_target_parts'))}

Focus anchor id: {anchor_id}. A square region centered at this anchor point is divided by the contour's tangent line into two areas:
- Region 1 (Inner): The side toward the object interior
- Region 2 (Outer): The side toward the background/exterior

Internal decision rubric (DO NOT OUTPUT):
- POS if the region contains target texture/pattern that should be included in the mask
- NEG if the region contains background patterns, distractors, or should be excluded from the mask
- You MUST select exactly one region (either inner or outer, not both)
- Choose the action (pos/neg) that will most improve the segmentation quality
- Inner regions typically get POS when object extends inward; NEG when mask over-includes
- Outer regions typically get NEG when background leaks in; POS when object extends outward

Return JSON STRICTLY:
{{
  "anchor_id": {anchor_id},
  "edits": [
    {{ "region_id": 1 | 2, "action": "pos" | "neg", "why": "short justification" }}
  ]
}}
Constraints:
- len(edits) == 1 (exactly one region selection)
- region_id must be 1 (inner) or 2 (outer)
- action must be "pos" or "neg"
- Keep 'why' brief and descriptive

Few-shot examples (DO NOT COPY):
{{
  "anchor_id": 3,
  "edits": [
    {{ "region_id": 2, "action": "neg", "why": "sand ripple background leak" }}
  ]
}}

{{
  "anchor_id": 5,
  "edits": [
    {{ "region_id": 1, "action": "pos", "why": "fin texture continues inward" }}
  ]
}}
"""
    # 可选：附加来自上一轮 anchor 决策的线索
    if anchor_hint:
        user_msg = f"Anchor hint: {anchor_hint.strip()}\n\n" + user_msg
    if global_reason:
        user_msg = "Global context: " + global_reason.strip() + "\n\n" + user_msg

    return {"system": sys_msg, "user": user_msg}
