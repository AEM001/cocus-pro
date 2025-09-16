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
    K: int = 1
) -> Dict[str, str]:
    """
    默认每次只选择一个最重要的锚点，避免多个锚点同时优化导致的复杂性
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
CAMOUFLAGED TARGET: '{instance}'
This is a CAMOUFLAGED object that naturally blends with its environment. The target may have:
- Similar colors/textures to the background
- Irregular, organic shapes that mimic surroundings  
- Subtle boundaries that are hard to distinguish
- Parts that appear to "fade into" or "emerge from" the background

Camouflage analysis context:
Synonyms/aliases: {_fmt_list(sem.get('synonyms'))}
Key identifying features: {_fmt_list(sem.get('salient_cues'))}
Background mimics/distractors: {_fmt_list(sem.get('distractors'))}
Typical shape: {sem.get('shape_prior') or "unknown"}
Texture/color patterns: {_fmt_list(sem.get('texture_prior'))}
Environment: {sem.get('scene_context') or "unknown"}
Avoid including: {_fmt_list(sem.get('not_target_parts'))}

CURRENT STATE: Green overlay shows the current mask. Anchors 1..8 mark potential refinement points.
TASK: Select EXACTLY ONE anchor - the MOST CRITICAL point where the camouflaged {instance} boundary needs correction.

PRIORITY STRATEGY: Choose the single anchor that will have the MAXIMUM IMPACT on segmentation quality:
1) MOST OBVIOUS ERROR: Prefer anchors at locations with clear boundary mistakes (major leaks or missing parts)
2) HIGHEST CONFIDENCE: Select the anchor where you are most certain about the correction needed
3) BIOLOGICAL IMPORTANCE: Favor anchors at anatomically significant boundaries (head, fins, limbs)
4) LARGEST IMPROVEMENT: Choose the anchor that will correct the most pixels with one refinement

For CAMOUFLAGED targets, focus on:
- Subtle texture transitions - look for slight differences in pattern/grain between target and background
- Biological boundaries - camouflaged animals often have natural body contours despite blending
- Depth/shadow cues - slight 3D form indicators that reveal the hidden shape
- Consistency - camouflaged parts should connect logically to already-identified portions

Decision criteria (DO NOT OUTPUT):
- LEAK: Green mask extends onto clear background patterns that don't belong to the {instance}
- MISS: Obvious {instance} body parts are not included, breaking biological continuity
- Look for SUBTLE but consistent textural/tonal differences at boundaries
- Trust shape continuity over perfect color matching for camouflaged subjects

Return JSON STRICTLY:
{{
  "anchors_to_refine": [
    {{ "id": 1-8, "intent": "fix_leak" | "recover_miss", "reason": "short justification (no numbers)" }}
  ]
}}
Constraints:
- EXACTLY ONE anchor in the array (len(anchors_to_refine) == 1)
- Use only ids from 1..8
- Keep justifications short; no lists or numbering; no quotes in the text
- Choose the MOST IMPORTANT anchor, not just any suitable one

Few-shot examples (DO NOT COPY, DO NOT ECHO; FORMAT ONLY):
Input→ (major leak suspected near rock ripples at anchor 3, minor issue at anchor 7)
Output→ {{"anchors_to_refine":[{{"id":3,"intent":"fix_leak","reason":"background ripples intrude"}}]}}

Input→ (missing fin edge is most critical issue)
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
CAMOUFLAGED TARGET: '{instance}'
Key features: {_fmt_list(sem.get('salient_cues'))}
Background mimics: {_fmt_list(sem.get('distractors'))}
Avoid: {_fmt_list(sem.get('not_target_parts'))}

Focus on anchor {anchor_id}. The square region is divided by the current boundary into:
- Region 1 (Inner): Toward the {instance} body/interior
- Region 2 (Outer): Toward the background/environment

For CAMOUFLAGED {instance}, examine CLOSELY:
- Subtle texture/grain differences between target and background
- Natural body contours vs environmental patterns
- Consistent biological form vs random background elements
- Slight tonal/depth variations that reveal the hidden shape

Decision logic (DO NOT OUTPUT):
- POS: This region contains {instance} body parts that should be included (even if camouflaged)
- NEG: This region contains background/environment that mimics the {instance} but isn't actually part of it
- CRITICAL: For camouflaged subjects, trust biological/anatomical continuity over pure visual similarity
- Inner regions: POS if {instance} extends inward naturally; NEG if mask wrongly includes background
- Outer regions: NEG if background patterns leak in; POS if {instance} extends outward naturally
- When uncertain between similar textures, consider: Does this follow the expected {instance} body structure?

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
