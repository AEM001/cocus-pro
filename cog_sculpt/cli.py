#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import yaml

from .core import (
    Config,
    load_image,
    load_mask_png,
    ensure_dir,
    save_image,
    overlay,
    maybe_read_boxes,
    bbox_from_mask,
    box_from_xyxy_clip,
    DensityScorer,
    AlphaCLIPScorer,
    SoftFocusCLIPScorer,
    SamWrapper,
    sculpting_pipeline,
    save_mask_png,
    Box,
    read_instance_text_from_llm_out,
)


class NoOpSam(SamWrapper):
    """占位 SAM：仅用于连通性测试。实际使用请替换为真实实现。"""

    def predict_box(self, image, bbox: Box):  # type: ignore[override]
        raise NotImplementedError("请实现 SamWrapper.predict_box 或提供 prior 掩码作为 M0")

    def predict_points(self, image, pos, neg):  # type: ignore[override]
        raise NotImplementedError("请实现 SamWrapper.predict_points 以完成互动更新")


def build_cfg_from_yaml_and_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    name = args.name or ""
    
    # 默认路径设置（基于 name）
    cfg.image_path = args.image or f"auxiliary/images/{name}.png"
    cfg.boxes_json_path = args.boxes or f"auxiliary/box_out/{name}/{name}_sam_boxes.json"
    cfg.prior_mask_path = args.prior_mask or f"auxiliary/box_out/{name}/{name}_prior_mask.png"
    cfg.out_root = args.out_root or "pipeline_output"
    
    # 其它参数覆盖（若提供）
    if args.k is not None:
        cfg.k_iters = int(args.k)
    if args.max_points is not None:
        cfg.max_points_per_iter = int(args.max_points)
    
    # 强制使用 prior 和 boxes
    cfg.use_prior_mask_as_M0 = True
    cfg.use_boxes_for_roi = True
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Cog-Sculpt pipeline runner (简化版，只需输入名称)")
    ap.add_argument("--name", required=True, help="图像名（必需）")
    ap.add_argument("--text", help="目标实例文本，若未提供则自动从 llm_out 读取")
    ap.add_argument("--k", type=int, help="迭代轮数（可选）")
    ap.add_argument("--max-points", type=int, help="每轮最大点数（可选）")
    
    # 这些参数现在有默认值，不需要用户提供
    ap.add_argument("--image", help=argparse.SUPPRESS)
    ap.add_argument("--boxes", help=argparse.SUPPRESS)
    ap.add_argument("--prior-mask", dest="prior_mask", help=argparse.SUPPRESS)
    ap.add_argument("--out-root", help=argparse.SUPPRESS)
    ap.add_argument("--use-prior", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--use-boxes", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--debug", action="store_true", default=True, help=argparse.SUPPRESS)  # 默认开启debug

    args = ap.parse_args()
    cfg = build_cfg_from_yaml_and_args(args)

    # 自动读取 instance 文本（如果未通过 --text 提供）
    instance_text = args.text
    if not instance_text and args.name:
        instance_text = read_instance_text_from_llm_out(args.name)
        if instance_text:
            print(f"Auto-loaded instance text from llm_out: '{instance_text}'")
        else:
            raise ValueError(
                f"No --text provided and failed to read instance text from auxiliary/llm_out/{args.name}_output.json. "
                "Please provide --text argument or ensure the JSON file exists with 'instance' field."
            )
    elif not instance_text:
        raise ValueError("Please provide --text argument or --name to auto-load from llm_out JSON.")

    assert cfg.image_path and os.path.isfile(cfg.image_path), f"Image not found: {cfg.image_path}"
    image = load_image(cfg.image_path)

    # ROI: 先尝试从 boxes.json 推导；否则从 prior 掩码 BBox
    H, W = image.shape[:2]
    bbox: Optional[Box] = None
    if cfg.use_boxes_for_roi and cfg.boxes_json_path:
        boxes = maybe_read_boxes(cfg.boxes_json_path)
        if boxes is not None and boxes.size > 0:
            # 使用第一只框（或可改为选择最大/中位）
            x1, y1, x2, y2 = [int(v) for v in boxes[0].tolist()]
            bbox = box_from_xyxy_clip(y1, x1, y2, x2, H, W)

    if bbox is None and cfg.prior_mask_path and os.path.isfile(cfg.prior_mask_path):
        prior = load_mask_png(cfg.prior_mask_path)
        xyxy = bbox_from_mask(prior)
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            bbox = box_from_xyxy_clip(y1, x1, y2, x2, H, W)

    if bbox is None:
        raise ValueError("无法确定 ROI：请提供 boxes.json 或 prior 掩码")

    # 初始掩码 M0：可用 prior 直接作为 M0
    M0 = None
    if cfg.use_prior_mask_as_M0 and cfg.prior_mask_path and os.path.isfile(cfg.prior_mask_path):
        M0 = load_mask_png(cfg.prior_mask_path)

    # 构造 scorer 与 sam 封装（强制要求本地权重存在）
    scorer = None
    sam = None

    # 加载 Alpha-CLIP（强制 Alpha 分支权重存在）
    from alpha_clip_rw import AlphaCLIPInference
    try:
        alpha_clip_model = AlphaCLIPInference(
            model_name="ViT-L/14@336px",
            alpha_vision_ckpt_pth="AUTO",  # 自动从 env 或 checkpoints/ 读取
        )
        scorer = AlphaCLIPScorer(alpha_clip_model)
        print("Alpha-CLIP scorer loaded successfully")
    except Exception as e:
        raise RuntimeError(
            f"Alpha-CLIP 加载失败: {e}\n"
            "请确保已下载 Alpha-CLIP 视觉分支权重到 checkpoints/clip_l14_336_grit_20m_4xe.pth，"
            "或设置环境变量 ALPHA_CLIP_ALPHA_CKPT 指向该文件。"
        )

    # 加载 SAM（强制权重存在）
    from sam_integration import create_sam_wrapper
    sam_ckpt = "models/sam_vit_h_4b8939.pth"
    if not os.path.isfile(sam_ckpt):
        raise RuntimeError(
            f"SAM 权重未找到: {sam_ckpt}\n"
            "请下载 sam_vit_h_4b8939.pth 到 models/ 目录。"
        )
    sam_wrapper = create_sam_wrapper(
        model_type="sam",
        checkpoint_path=sam_ckpt,
    )
    sam = SamWrapper(sam_wrapper)
    print("SAM wrapper loaded successfully")

    # 可选：调试钩子
    debug_hook = None
    out_dir = os.path.join(cfg.out_root, args.name or "scene")
    if args.debug:
        ensure_dir(out_dir)
        import json as _json

        def _hook(**kw):
            it = kw.get("it", 0)
            mask = kw.get("mask")
            cells = kw.get("cells", [])
            pos_points = kw.get("pos_points", [])
            neg_points = kw.get("neg_points", [])
            # 保存点位与子格分数
            points_dump = {
                "pos_points": [p.__dict__ for p in pos_points],
                "neg_points": [p.__dict__ for p in neg_points],
                "cells": [
                    {
                        "r0": c.box.r0,
                        "c0": c.box.c0,
                        "r1": c.box.r1,
                        "c1": c.box.c1,
                        "depth": c.depth,
                        "score": None if c.score is None else float(c.score),
                    }
                    for c in cells
                ],
            }
            with open(os.path.join(out_dir, f"iter_{it:02d}_debug.json"), "w", encoding="utf-8") as f:
                _json.dump(points_dump, f, ensure_ascii=False, indent=2)
            # 保存每轮掩码与叠加图
            if mask is not None:
                save_mask_png(mask, os.path.join(out_dir, f"iter_{it:02d}_mask.png"))
                save_image(overlay(image, mask), os.path.join(out_dir, f"iter_{it:02d}_overlay.png"))

        debug_hook = _hook

    M_final = sculpting_pipeline(
        image=image,
        bbox=bbox,
        instance_text=instance_text,
        sam=sam,
        scorer=scorer,
        cfg=cfg,
        initial_mask_M0=M0,
        debug_hook=debug_hook,
    )

    # 保存最终可视化
    ensure_dir(out_dir)
    save_mask_png(M_final, os.path.join(out_dir, "final_mask.png"))
    save_image(overlay(image, M_final), os.path.join(out_dir, "final_overlay.png"))


if __name__ == "__main__":
    main()
