from __future__ import annotations

from typing import Dict


def build_peval_prompt(instance: str, word_budget: int = 40) -> Dict[str, str]:
    system = "You are a segmentation inspector. Be concise."
    user = (
        "You see an ROI crop, with a semi-transparent mask overlay.\n"
        f"Target category: \"{instance}\" (single target).\n"
        "Task: Briefly diagnose mask defects. Reply in compact JSON only:\n"
        "{\n  \"missing_parts\": [\"...\"],\n  \"over_segments\": [\"...\"],\n  \"boundary_quality\": \"sharp|soft|fragmented\",\n  \"key_cues\": [\"shape/texture/depth cues for the true object\"]\n}\n"
        f"Word budget: within {word_budget} words total."
    )
    return {"system": system, "user": user}


def build_pgen_prompt(instance: str, key_cues: str) -> Dict[str, str]:
    system = (
        "You label whether a small image patch belongs to the SINGLE target category.\n"
        "Be strict and concise. Do not reveal your reasoning."
    )
    user = (
        f"Target: \"{instance}\".\n"
        f"Visual cues (may help disambiguation): \"{key_cues}\".\n"
        "Answer JSON only:\n{\"is_target\": true|false, \"conf\": 0.00-1.00}"
    )
    return {"system": system, "user": user}

