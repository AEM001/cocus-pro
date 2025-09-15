from __future__ import annotations

from typing import Dict


# 原有的peval和pgen提示词已删除，只保留新的anchor/quadrant提示词


def build_anchor_prompt(instance: str) -> Dict[str, str]:
    """Prompt to choose which labeled anchor i around the ROI needs refinement.

    The input image contains the ROI rectangle with 8 labeled anchors {i}: 1..8 ordered as:
      1=top-left corner, 2=top-mid, 3=top-right corner, 4=mid-right,
      5=bottom-right corner, 6=bottom-mid, 7=bottom-left corner, 8=mid-left.
    The semi-transparent current mask is overlaid in green.
    """
    system = "You are a segmentation inspector. Pick labeled ROI anchor indices to refine. Reply ONLY valid JSON."
    user = (
        f"We segment a single target instance: '{instance}'.\n"
        "You see an image with an ROI rectangle and 8 labeled anchors {i}.\n"
        "Task: choose 1-3 anchors whose vicinity most needs refinement of the current mask.\n"
        "Reply JSON only with this schema:\n"
        "{\n  \"anchors_to_refine\": [ { \"id\": 1-8, \"reason\": \"...\" } ]\n}"
    )
    return {"system": system, "user": user}


def build_quadrant_prompt(instance: str, anchor_id: int) -> Dict[str, str]:
    """Prompt to choose quadrant regions and suggest pos/neg points for next SAM.

    The input is a crop around anchor i containing a square divided into 4 labeled regions {j}:
      j=1: top-left quadrant, j=2: top-right, j=3: bottom-right, j=4: bottom-left.
    Also include previously chosen anchor id i for context.
    """
    system = "You are a segmentation refiner. Suggest where to add positive/negative points. Reply ONLY valid JSON."
    user = (
        f"Single target instance: '{instance}'.\n"
        f"We focus on anchor i={anchor_id}. The square is split into 4 labeled regions {{j}} (1..4).\n"
        "Mask is shown in green overlay.\n"
        "Decide in which regions to add POSITIVE points (foreground) and NEGATIVE points (background) to improve SAM next step.\n"
        "Keep the number of points small (<=2 per type).\n"
        "Reply JSON only with this schema:\n"
        "{\n  \"anchor_id\": i,\n  \"edits\": [ { \"region_id\": 1-4, \"action\": \"pos\"|\"neg\", \"why\": \"...\" } ]\n}"
    )
    return {"system": system, "user": user}
