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

CURRENT STATE: Yellow contour shows the current mask boundary. Anchors 1..8 mark potential refinement points.
TASK: Select EXACTLY ONE anchor - the MOST CRITICAL point where the camouflaged {instance} boundary needs correction.

CRITICAL ANALYSIS: Focus on the RELATIONSHIP between {instance} and background:
- OBJECT vs BACKGROUND: Does the current boundary correctly separate the {instance} from its environment?
- CAMOUFLAGE PATTERNS: The {instance} may share similar colors/textures with background but has distinct biological structure
- NATURAL BOUNDARIES: Look for anatomical edges (body outline, fins, limbs) rather than just color changes
- TEXTURE COHERENCE: {instance} body parts should have consistent organic texture, while background shows environmental patterns

PRIORITY STRATEGY - Choose the anchor with MAXIMUM BIOLOGICAL SIGNIFICANCE:
1) ANATOMICAL BOUNDARIES: Prefer anchors where {instance} body structure meets background (not just color transitions)
2) STRUCTURAL CONTINUITY: Select points that will restore natural {instance} body shape and proportions
3) CAMOUFLAGE vs REALITY: Distinguish between background mimicry and actual {instance} body parts
4) FUNCTIONAL ANATOMY: Prioritize biologically important features (head, main body, appendages)

For CAMOUFLAGED {instance}, distinguish:
- REAL {instance} PARTS: Organic textures, natural body curves, anatomical consistency
- BACKGROUND ELEMENTS: Environmental patterns, random textures, non-biological shapes
- TRANSITION ZONES: Where {instance} naturally meets its habitat

Decision criteria (DO NOT OUTPUT):
- ANATOMICAL LEAK: Mask incorrectly includes background elements that mimic {instance} but lack biological structure
- BIOLOGICAL MISS: Actual {instance} body parts are excluded, breaking anatomical continuity
- STRUCTURAL ANALYSIS: Prioritize natural {instance} body shape over superficial color similarities
- HABITAT INTEGRATION: Consider how {instance} naturally interacts with its environment
- ORGANIC vs INORGANIC: Distinguish living tissue patterns from environmental textures

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

For CAMOUFLAGED {instance}, analyze the BIOLOGICAL REALITY:
- ANATOMICAL STRUCTURE: Does this region show natural {instance} body organization?
- ORGANIC PATTERNS: Distinguish living tissue textures from environmental mimicry
- FUNCTIONAL ANATOMY: Consider {instance} body proportions and natural shape
- HABITAT RELATIONSHIP: How does the {instance} naturally interface with its environment?

Decision logic (DO NOT OUTPUT):
- POS: This region contains ACTUAL {instance} anatomy (organic structure, not just similar colors)
- NEG: This region shows background/habitat that may mimic {instance} but lacks biological structure
- BIOLOGICAL PRIMACY: Trust anatomical evidence over superficial visual similarities
- Inner regions: POS if shows {instance} internal body structure; NEG if background intrusion
- Outer regions: NEG if environmental elements; POS if natural {instance} body extension
- KEY QUESTION: Is this part of the living {instance} organism or just environmental mimicry?

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
