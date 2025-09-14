# 运行1-N轮雕刻的CLI（Peval + Pgen → P⁺/P⁻ → SAM精化）。
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sculptor.candidates import sample_candidates
from sculptor.patches import multi_scale_with_aug
from sculptor.sam_refine import SamPredictorWrapper, early_stop
from sculptor.select_points import select_points
from sculptor.types import ROIBox
from sculptor.utils import (
    draw_points,
    load_image,
    load_mask,
    overlay_mask_on_image,
    parse_sam_boxes_json,
    save_image,
    save_json,
)
from sculptor.vlm.mock import MockVLM
from sculptor.vlm.qwen import QwenVLM


def short_key_cues(items: List[str], max_words: int = 15) -> str:
    text = ", ".join(items)
    words = text.split()
    return " ".join(words[:max_words])


def main():
    ap = argparse.ArgumentParser(description="VLM-driven sculpting round runner")
    ap.add_argument("--image", required=True, help="Path to RGB image")
    ap.add_argument("--mask", required=True, help="Path to initial mask (uint8 0/255)")
    ap.add_argument("--roi_json", required=False, help="Path to ROI boxes JSON (take first)")
    ap.add_argument("--roi", nargs=4, type=float, help="x0 y0 x1 y1 ROI", default=None)
    ap.add_argument("--instance", required=True, help="Target instance/category name")
    ap.add_argument("--model", choices=["mock", "qwen"], default="mock")
    ap.add_argument("--model_dir", default=os.environ.get("QWEN_VL_MODEL_DIR", None))
    ap.add_argument("--server_url", default=os.environ.get("QWEN_VL_SERVER", "http://127.0.0.1:8000/generate"))
    # SAM integration
    ap.add_argument("--sam_checkpoint", default=os.environ.get("SAM_CHECKPOINT", None), help="Path to SAM .pth checkpoint")
    ap.add_argument("--sam_type", default=os.environ.get("SAM_TYPE", "vit_h"), help="SAM model type: vit_h|vit_l|vit_b")
    ap.add_argument("--sam_device", default=os.environ.get("SAM_DEVICE", None), help="Override device: cuda|cpu")
    ap.add_argument("--out_dir", default="outputs/sculpt")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--K_pos", type=int, default=6)
    ap.add_argument("--K_neg", type=int, default=6)
    args = ap.parse_args()

    I = load_image(args.image)
    M = load_mask(args.mask)

    if args.roi is not None:
        B = ROIBox(*args.roi)
    elif args.roi_json:
        boxes = parse_sam_boxes_json(args.roi_json)
        if not boxes:
            raise ValueError("No boxes found in ROI JSON")
        B = ROIBox(*boxes[0])
    else:
        # fallback to whole image
        H, W = I.shape[:2]
        B = ROIBox(0, 0, W, H)

    if args.model == "mock":
        vlm = MockVLM()
    else:
        # Qwen2.5-VL stub: by default server mode (no network request here)
        vlm = QwenVLM(mode="server", server_url=args.server_url, model_dir=args.model_dir)

    # SAM backend (optional)
    sam_backend = None
    if args.sam_checkpoint:
        try:
            from sculptor.sam_backends import build_sam_backend

            sam_backend = build_sam_backend(
                checkpoint_path=args.sam_checkpoint,
                model_type=args.sam_type,
                device=args.sam_device,
            )
            print(f"[SAM] Loaded checkpoint at {args.sam_checkpoint} with type {args.sam_type}")
        except Exception as e:
            print(f"[WARN] SAM backend not available: {e}. Falling back to no-op.")
    sam = SamPredictorWrapper(backend=sam_backend)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    run_dir = os.path.join(args.out_dir, base)
    os.makedirs(run_dir, exist_ok=True)

    cur_mask = M
    history = []
    for t in range(args.rounds):
        # A) candidates
        cand = sample_candidates(cur_mask, B, max_total=30, nms_r_ratio=0.06)
        pts = [c.xy for c in cand]

        # B) patches
        patches = multi_scale_with_aug(I, pts, B, scales=[0.08, 0.12, 0.18], aug_light=True)

        # C1) peval
        overlay = overlay_mask_on_image(I, cur_mask)
        Fsem = vlm.peval(overlay, depth_vis=None, instance=args.instance)
        cues = short_key_cues([str(x) for x in Fsem.get("key_cues", [])])

        # C2) pgen per point (average over extracted patches)
        scores: Dict[int, float] = {}
        for i, patch_list in patches.items():
            confs: List[float] = []
            for p in patch_list:
                resp = vlm.pgen(p, args.instance, cues)
                confs.append(float(resp.get("conf", 0.0)) if bool(resp.get("is_target", False)) else 0.0)
            if confs:
                scores[i] = float(np.mean(confs))
            else:
                scores[i] = 0.0

        # D) select P+/P-
        P, L, pos, neg = select_points(cand, scores, B, K_pos=args.K_pos, K_neg=args.K_neg)

        # log visualization
        colors = [(0, 255, 0)] * len(pos) + [(255, 0, 0)] * len(neg)
        vis_pts = draw_points(I, [c.xy for c in pos + neg], colors, r=3)
        save_image(os.path.join(run_dir, f"round{t}_points.png"), vis_pts)
        save_json(os.path.join(run_dir, f"round{t}_peval.json"), Fsem)
        save_json(os.path.join(run_dir, f"round{t}_scores.json"), {str(k): float(v) for k, v in scores.items()})

        # E) SAM refine
        new_mask = sam.predict(I, P, L, B, prev_mask=cur_mask)
        save_image(os.path.join(run_dir, f"round{t}_mask.png"), (new_mask > 0).astype(np.uint8) * 255)

        history.append({"round": t, "num_pos": len(pos), "num_neg": len(neg)})
        if early_stop(cur_mask, new_mask, iou_tol=0.005):
            cur_mask = new_mask
            break
        cur_mask = new_mask

    save_image(os.path.join(run_dir, "final_mask.png"), (cur_mask > 0).astype(np.uint8) * 255)
    save_json(os.path.join(run_dir, "meta.json"), {"history": history, "roi": [B.x0, B.y0, B.x1, B.y1]})


if __name__ == "__main__":
    main()
