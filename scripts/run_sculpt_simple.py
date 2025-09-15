"""
【简化版VLM雕刻运行脚本】
作用：提供SAM优先的简化版VLM雕刻流程，降低使用门槛
核心功能：
  - SAM优先策略：先用SAM从边界框生成初始掩码，再进行VLM雕刻
  - 自动配置：从标准目录结构自动加载所有必要文件
  - 一键运行：简化参数配置，适合快速原型验证
  - 完整集成：整合SAM生成和VLM雕刻的端到端流程

与系统模块关系：
  - 调用sculptor所有核心模块：candidates, patches, sam_refine, select_points, utils, vlm
  - 作为run_sculpt.py的简化版本，降低使用复杂度
  - 提供标准化的数据加载和配置管理

使用场景：
  - 快速原型验证：无需复杂配置即可测试雕刻效果
  - 教学演示：展示SAM+VLM协同工作的完整流程
  - 批量处理：支持标准命名规范的批量图像处理

目录结构：
  auxiliary/
    images/{name}.png          - 输入图像
    box_out/{name}/            - SAM边界框和初始掩码
    llm_out/{name}_output.json - LLM生成的实例描述
  outputs/sculpt/{name}/       - 雕刻结果输出

工作流程：
  1. 加载图像和ROI边界框
  2. SAM从边界框生成初始掩码
  3. VLM驱动的多轮雕刻优化
  4. 输出最终精细掩码和可视化结果
"""

#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

# Reduce tokenizer fork/parallelism warnings and potential deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np

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


def _roi_int_bounds(B: ROIBox, H: int, W: int):
    x0 = max(0, min(W - 1, int(round(B.x0))))
    y0 = max(0, min(H - 1, int(round(B.y0))))
    x1 = max(0, min(W, int(round(B.x1))))
    y1 = max(0, min(H, int(round(B.y1))))
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    return x0, y0, x1, y1


def _auto_build_sam_backend(base_dir: str, device: Optional[str]):
    """Try to auto-detect a SAM checkpoint under models/ and build a backend.

    Returns a SamPredictorWrapper backend or None if not found/failed.
    """
    try:
        from sculptor.sam_backends import build_sam_backend
    except Exception as e:
        print(f"[WARN] SAM backends not available: {e}")
        return None

    models_dir = os.path.join(base_dir, "models")
    if not os.path.isdir(models_dir):
        print(f"[WARN] Models directory not found: {models_dir}")
        return None

    # Search for common SAM checkpoint filenames
    candidates: List[Tuple[str, str]] = []  # (path, model_type)
    for root, _, files in os.walk(models_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith(".pth"):
                mtype = None
                if "vit_h" in low:
                    mtype = "vit_h"
                elif "vit_l" in low:
                    mtype = "vit_l"
                elif "vit_b" in low:
                    mtype = "vit_b"
                # Heuristic: filename contains 'sam'
                if "sam" in low and mtype is not None:
                    candidates.append((os.path.join(root, fn), mtype))

    if not candidates:
        print("[WARN] No SAM checkpoint found under models/. You can specify --sam_checkpoint explicitly.")
        return None

    # Prefer vit_h, then vit_l, then vit_b
    pref_order = {"vit_h": 0, "vit_l": 1, "vit_b": 2}
    candidates.sort(key=lambda it: pref_order.get(it[1], 99))
    ckpt_path, model_type = candidates[0]
    try:
        backend = build_sam_backend(checkpoint_path=ckpt_path, model_type=model_type, device=device)
        print(f"[INFO] Auto-loaded SAM checkpoint: {ckpt_path} (type={model_type})")
        return backend
    except Exception as e:
        print(f"[WARN] Failed to load SAM checkpoint {ckpt_path}: {e}")
        return None

def load_config_from_name(name: str) -> Dict:
    """Load all configuration from name, following standard directory structure."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config = {
        'name': name,
        'base_dir': base_dir,
        'image_path': os.path.join(base_dir, 'auxiliary', 'images', f'{name}.png'),
        'prior_mask_path': os.path.join(base_dir, 'auxiliary', 'box_out', name, f'{name}_prior_mask.png'),
        'roi_json_path': os.path.join(base_dir, 'auxiliary', 'box_out', name, f'{name}_sam_boxes.json'),
        'llm_output_path': os.path.join(base_dir, 'auxiliary', 'llm_out', f'{name}_output.json'),
        'output_dir': os.path.join(base_dir, 'outputs', 'sculpt', name),
    }
    
    # Load instance from LLM output
    try:
        with open(config['llm_output_path'], 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
            config['instance'] = llm_data['instance']
    except Exception as e:
        print(f"[WARN] Could not load instance from {config['llm_output_path']}: {e}")
        config['instance'] = name  # Fallback to name
    
    # Validate required files exist
    required_files = ['image_path', 'prior_mask_path', 'roi_json_path']
    for key in required_files:
        if not os.path.exists(config[key]):
            raise FileNotFoundError(f"Required file not found: {config[key]}")
    
    print(f"[INFO] Loaded config for '{name}':")
    print(f"  Instance: {config['instance']}")
    print(f"  Image: {config['image_path']}")
    print(f"  Prior mask: {config['prior_mask_path']}")
    print(f"  ROI boxes: {config['roi_json_path']}")
    print(f"  Output: {config['output_dir']}")
    
    return config

def generate_sam_mask_from_prior(image, roi_box, sam_wrapper, prior_mask, pre_points=None, pre_labels=None):
    """
    Generate initial SAM mask using ROI box and optional pre-SAM seed points, strictly within ROI.
    
    Args:
        image: RGB image array
        roi_box: ROIBox object
        sam_wrapper: SamPredictorWrapper instance (must have backend)
        prior_mask: Prior mask for guidance (only used for clipping/diagnostics)
        pre_points: Optional Nx2 float array of seed points
        pre_labels: Optional N int array of labels (1=pos, 0=neg)
    
    Returns:
        numpy.ndarray: Generated mask clipped to ROI
    """
    print("[INFO] Generating initial SAM mask (with optional VLM seeds)...")

    import numpy as np
    H, W = image.shape[:2]
    x0, y0, x1, y1 = _roi_int_bounds(roi_box, H, W)

    def _clip_to_roi(mask: np.ndarray) -> np.ndarray:
        clipped = np.zeros_like(mask, dtype=np.uint8)
        clipped[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        return clipped

    # Require SAM backend to be available
    if sam_wrapper.backend is None:
        raise RuntimeError("SAM backend is required but not available")

    # Use SAM with ROI box and optional seed points
    if pre_points is None or len(pre_points) == 0:
        pts = np.array([[]], dtype=np.float32).reshape(0, 2)
        lbs = np.array([], dtype=np.int32)
    else:
        pts = np.asarray(pre_points, dtype=np.float32)
        lbs = np.asarray(pre_labels, dtype=np.int32) if pre_labels is not None else np.ones((len(pts),), dtype=np.int32)

    initial_mask = sam_wrapper.predict(
        image,
        pts,
        lbs,
        roi_box,
        prior_mask,
    )

    initial_mask = _clip_to_roi(initial_mask)
    print(f"[INFO] Generated SAM mask (ROI-clipped) with {(initial_mask > 0).sum()} pixels")
    return initial_mask

def short_key_cues(items: List[str], max_words: int = 15) -> str:
    """Shorten key cues to fit within word limit."""
    text = ", ".join(items)
    words = text.split()
    return " ".join(words[:max_words])


# Removed VLM-based pre-seeding due to instability and misalignment in your case.


def prehint_points_simple(
    I: np.ndarray,
    B: ROIBox,
    topk: int = 4,
    negk: int = 2,
):
    """
    Very simple, deterministic seeding:
    - positives: ROI center and small offsets near the center
    - negatives: ROI corners and side midpoints
    This ignores any semantic cues or prior mask to avoid misalignment.
    """
    import numpy as np
    H, W = I.shape[:2]
    x0, y0, x1, y1 = _roi_int_bounds(B, H, W)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    dx = 0.1 * (x1 - x0)
    dy = 0.1 * (y1 - y0)

    pos_candidates = [
        (cx, cy),
        (cx + dx, cy),
        (cx - dx, cy),
        (cx, cy + dy),
        (cx, cy - dy),
    ]
    pos_pts: List[Tuple[float, float]] = []
    for p in pos_candidates:
        if len(pos_pts) >= max(1, topk):
            break
        px = float(max(0, min(W - 1, p[0])))
        py = float(max(0, min(H - 1, p[1])))
        pos_pts.append((px, py))

    neg_candidates = [
        (x0 + 2, y0 + 2), (x1 - 2, y0 + 2), (x0 + 2, y1 - 2), (x1 - 2, y1 - 2),
        (cx, y0 + 2), (cx, y1 - 2), (x0 + 2, cy), (x1 - 2, cy),
    ]
    neg_pts: List[Tuple[float, float]] = []
    for q in neg_candidates:
        if len(neg_pts) >= max(0, negk):
            break
        qx = float(max(0, min(W - 1, q[0])))
        qy = float(max(0, min(H - 1, q[1])))
        neg_pts.append((qx, qy))

    P = np.array(pos_pts + neg_pts, dtype=np.float32) if (pos_pts or neg_pts) else None
    L = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32) if (pos_pts or neg_pts) else None
    return P, L, pos_pts, neg_pts

def main():
    parser = argparse.ArgumentParser(description="Simplified VLM-driven sculpting with SAM-first workflow")
    parser.add_argument("name", help="Name of the sample (e.g., 'dog', 'f', 'q')")
    parser.add_argument("--rounds", type=int, default=2, help="Number of sculpting rounds")
    parser.add_argument("--model", choices=["mock", "qwen"], default="qwen", help="VLM model to use")
    parser.add_argument("--model_dir", default="models/Qwen2.5-VL-3B-Instruct", help="Qwen model directory (defaults to 3B)")
    parser.add_argument("--sam_checkpoint", default=None, help="SAM checkpoint path")
    parser.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    parser.add_argument("--sam_device", default=None, help="SAM device (cuda/cpu)")
    parser.add_argument("--K_pos", type=int, default=6, help="Number of positive points per round")
    parser.add_argument("--K_neg", type=int, default=6, help="Number of negative points per round")
    # Qwen generation control
    parser.add_argument("--qwen_max_new_tokens", type=int, default=96, help="Max new tokens for Qwen generation (peval/pgen)")
    # Pre-SAM hinting controls (strictly geometry-based; semantic option removed)
    parser.add_argument("--pre_sam_topk", type=int, default=3, help="Number of positive seed points near ROI center")
    parser.add_argument("--pre_sam_negk", type=int, default=2, help="Number of negative seed points at ROI corners/edges")
    parser.add_argument("--pre_sam_hints", dest="pre_sam_hints", action="store_true", help="Enable seeding before initial SAM")
    parser.add_argument("--no-pre_sam_hints", dest="pre_sam_hints", action="store_false", help="Disable seeding before initial SAM")
    parser.set_defaults(pre_sam_hints=True)
    
    args = parser.parse_args()
    
    # Load configuration from name
    try:
        config = load_config_from_name(args.name)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        return 1
    
    # Load input data
    print("\n[INFO] Loading input data...")
    I = load_image(config['image_path'])
    prior_mask = load_mask(config['prior_mask_path'])
    
    # Load ROI box
    boxes = parse_sam_boxes_json(config['roi_json_path'])
    if not boxes:
        print("[ERROR] No boxes found in ROI JSON")
        return 1
    B = ROIBox(*boxes[0])
    print(f"[INFO] Using ROI box: ({B.x0:.1f}, {B.y0:.1f}, {B.x1:.1f}, {B.y1:.1f})")
    
    # Initialize VLM
    print(f"\n[INFO] Initializing VLM: {args.model}")
    if args.model == "mock":
        vlm = MockVLM()
    else:
        vlm = QwenVLM(mode="local", model_dir=config['base_dir'] + '/' + args.model_dir, gen_max_new_tokens=args.qwen_max_new_tokens)
    
    # Initialize SAM (required)
    sam_backend = None
    if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
        from sculptor.sam_backends import build_sam_backend
        sam_backend = build_sam_backend(
            checkpoint_path=args.sam_checkpoint,
            model_type=args.sam_type,
            device=args.sam_device,
        )
        print(f"[INFO] Loaded SAM backend: {args.sam_checkpoint}")
    else:
        # Try auto-discovery under models/
        sam_backend = _auto_build_sam_backend(config['base_dir'], args.sam_device)
    
    if sam_backend is None:
        print("[ERROR] SAM backend is required but not available")
        print("Please ensure a SAM checkpoint is available or specify --sam_checkpoint")
        return 1
    
    sam = SamPredictorWrapper(backend=sam_backend)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print(f"\n[INFO] Starting SAM-first sculpting workflow...")
    print(f"[INFO] Instance: '{config['instance']}'")
    print(f"[INFO] Rounds: {args.rounds}")
    
# Optional: Pre-SAM VLM seeding to guide initial segmentation
    P0 = None
    L0 = None
    if args.pre_sam_hints:
        print("[INFO] Running pre-SAM geometry seeding (ROI center + corner negatives)...")
        P0, L0, pos_seeds, neg_seeds = prehint_points_simple(
            I,
            B,
            topk=args.pre_sam_topk,
            negk=args.pre_sam_negk,
        )
        # Visualize seeds
        if P0 is not None and len(P0) > 0:
            colors = [(0, 255, 0) if l == 1 else (255, 0, 0) for l in L0]
            vis_seeds = draw_points(I, [(float(x), float(y)) for x, y in P0], colors, r=4)
            save_image(os.path.join(config['output_dir'], "pre_sam_hints.png"), vis_seeds)
            print(f"[INFO] Pre-SAM seeds: {len(pos_seeds)} pos, {len(neg_seeds)} neg")
        else:
            print("[INFO] No reliable pre-SAM seeds found; proceeding with box-only")

    # Step 1: Generate initial SAM mask from prior (+ optional seeds)
    current_mask = generate_sam_mask_from_prior(I, B, sam, prior_mask, pre_points=P0, pre_labels=L0)
    
    # Save initial SAM mask
    save_image(os.path.join(config['output_dir'], "sam_initial_mask.png"), 
               (current_mask > 0).astype(np.uint8) * 255)
    
    # Step 2: VLM-driven sculpting rounds
    history = []
    for t in range(args.rounds):
        print(f"\n[INFO] === Sculpting Round {t} ===")
        
        # A) Sample candidate points
        cand = sample_candidates(current_mask, B, max_total=30, nms_r_ratio=0.06)
        pts = [c.xy for c in cand]
        print(f"[INFO] Sampled {len(cand)} candidate points")
        
        # B) Extract multi-scale patches (target-size & boundary-sensitive small scales)
        patches = multi_scale_with_aug(
            I,
            pts,
            B,
            scales=[0.05, 0.08, 0.12],  # smaller scales for boundary detail
            aug_light=True,
            mask=current_mask,
            context=1.15,  # include slight context to cover both sides of boundary
        )
        print(f"[INFO] Extracted patches for {len(patches)} candidates")
        
        # C1) Peval - semantic evaluation of current mask (ROI-only overlay)
        overlay_full = overlay_mask_on_image(I, current_mask)
        H, W = I.shape[:2]
        x0, y0, x1, y1 = _roi_int_bounds(B, H, W)
        overlay = overlay_full[y0:y1, x0:x1]
        print("[INFO] Running Peval (mask evaluation) on ROI crop...")
        Fsem = vlm.peval(overlay, depth_vis=None, instance=config['instance'])
        cues = short_key_cues([str(x) for x in Fsem.get("key_cues", [])])
        print(f"[INFO] Peval result: {Fsem}")
        
        # C2) Pgen - evaluate each patch
        print("[INFO] Running Pgen (patch evaluation)...")
        scores: Dict[int, float] = {}
        for i, patch_list in patches.items():
            confs: List[float] = []
            for p in patch_list:
                resp = vlm.pgen(p, config['instance'], cues)
                conf = float(resp.get("conf", 0.0)) if bool(resp.get("is_target", False)) else 0.0
                confs.append(conf)
            if confs:
                scores[i] = float(np.mean(confs))
            else:
                scores[i] = 0.0
        
        print(f"[INFO] Patch scores computed for {len(scores)} candidates")
        
        # D) Select positive and negative points
        P, L, pos, neg = select_points(cand, scores, B, K_pos=args.K_pos, K_neg=args.K_neg)
        print(f"[INFO] Selected {len(pos)} positive and {len(neg)} negative points")
        
        # Save round visualizations
        colors = [(0, 255, 0)] * len(pos) + [(255, 0, 0)] * len(neg)
        vis_pts = draw_points(I, [c.xy for c in pos + neg], colors, r=3)
        save_image(os.path.join(config['output_dir'], f"round{t}_points.png"), vis_pts)
        save_json(os.path.join(config['output_dir'], f"round{t}_peval.json"), Fsem)
        save_json(os.path.join(config['output_dir'], f"round{t}_scores.json"), 
                 {str(k): float(v) for k, v in scores.items()})
        
        # E) SAM refinement
        if len(P) > 0:  # Only refine if we have points
            print("[INFO] Refining mask with SAM...")
            new_mask = sam.predict(I, P, L, B, prev_mask=current_mask)
            
            # Check for early stopping
            if early_stop(current_mask, new_mask, iou_tol=0.005, require_pos_points=True, used_pos_count=len(pos), min_round=1, current_round=t):
                print(f"[INFO] Early stopping at round {t} (IoU change < 0.005 and conditions met)")
                current_mask = new_mask
                history.append({"round": t, "num_pos": len(pos), "num_neg": len(neg), "early_stop": True})
                break
            else:
                current_mask = new_mask
        else:
            print("[INFO] No points selected, skipping SAM refinement")
        
        # Save round mask
        save_image(os.path.join(config['output_dir'], f"round{t}_mask.png"), 
                   (current_mask > 0).astype(np.uint8) * 255)
        
        history.append({"round": t, "num_pos": len(pos), "num_neg": len(neg)})
    
    # Save final results
    save_image(os.path.join(config['output_dir'], "final_mask.png"), 
               (current_mask > 0).astype(np.uint8) * 255)
    
    # Save metadata
    meta = {
        "name": config['name'],
        "instance": config['instance'],
        "roi": [B.x0, B.y0, B.x1, B.y1], 
        "rounds_completed": len(history),
        "history": history,
        "model": args.model,
        "workflow": "sam_first_sculpting"
    }
    save_json(os.path.join(config['output_dir'], "meta.json"), meta)
    
    print(f"\n[SUCCESS] Sculpting completed!")
    print(f"[INFO] Results saved to: {config['output_dir']}")
    print(f"[INFO] Final mask coverage: {(current_mask > 0).sum()} pixels")
    
    return 0

if __name__ == "__main__":
    import numpy as np  # Import needed for the script
    sys.exit(main())