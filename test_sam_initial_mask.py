#!/usr/bin/env python3
"""
测试SAM生成初始掩码
从ROI框生成初始掩码，模拟run_sculpt_simple.py中的generate_sam_mask_from_prior函数
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sculptor.sam_backends import build_sam_backend
from sculptor.sam_refine import SamPredictorWrapper
from sculptor.types import ROIBox
from sculptor.utils import load_image, load_mask, save_image, parse_sam_boxes_json

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

def generate_sam_mask_from_prior(image, roi_box, sam_wrapper, prior_mask):
    """
    使用SAM从ROI框生成初始掩码
    """
    print("[INFO] 使用SAM生成初始掩码...")
    
    H, W = image.shape[:2]
    x0, y0, x1, y1 = _roi_int_bounds(roi_box, H, W)
    
    def _clip_to_roi(mask: np.ndarray) -> np.ndarray:
        clipped = np.zeros_like(mask, dtype=np.uint8)
        clipped[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        return clipped
    
    # 确保SAM后端可用
    if sam_wrapper.backend is None:
        raise RuntimeError("SAM backend is required but not available")
    
    # 使用ROI框中心点作为种子点
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    
    # 创建点和标签
    pts = np.array([[cx, cy]], dtype=np.float32)
    lbs = np.array([1], dtype=np.int32)  # 正点
    
    print(f"[INFO] 使用ROI中心点 ({cx:.1f}, {cy:.1f}) 作为种子点")
    
    # 调用SAM预测
    initial_mask = sam_wrapper.predict(
        image,
        pts,
        lbs,
        roi_box,
        prior_mask,
    )
    
    # 裁剪到ROI区域
    initial_mask = _clip_to_roi(initial_mask)
    
    pixels_count = (initial_mask > 0).sum()
    print(f"[INFO] 生成的SAM掩码包含 {pixels_count} 个像素")
    
    return initial_mask

def create_result_visualization(image, roi_box, sam_mask, prior_mask):
    """创建可视化结果"""
    import cv2
    
    result = image.copy()
    
    # 绘制ROI框
    x0, y0, x1, y1 = roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1
    cv2.rectangle(result, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
    
    # 叠加SAM掩码（绿色）
    result[sam_mask > 0] = result[sam_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
    
    return result

def main():
    sample_name = "f"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    image_path = os.path.join(base_dir, 'auxiliary', 'images', f'{sample_name}.png')
    prior_mask_path = os.path.join(base_dir, 'auxiliary', 'box_out', sample_name, f'{sample_name}_prior_mask.png')
    roi_json_path = os.path.join(base_dir, 'auxiliary', 'box_out', sample_name, f'{sample_name}_sam_boxes.json')
    
    print(f"=== 测试SAM初始掩码生成 (样本: {sample_name}) ===")
    
    # 加载数据
    print("[INFO] 加载输入数据...")
    try:
        image = load_image(image_path)
        prior_mask = load_mask(prior_mask_path)
        boxes = parse_sam_boxes_json(roi_json_path)
        
        if not boxes:
            raise ValueError("未找到ROI框")
        
        roi_box = ROIBox(*boxes[0])
        print(f"[INFO] ROI框: ({roi_box.x0}, {roi_box.y0}, {roi_box.x1}, {roi_box.y1})")
        
    except Exception as e:
        print(f"[ERROR] 加载数据失败: {e}")
        return 1
    
    # 初始化SAM后端
    print("[INFO] 初始化SAM后端...")
    try:
        sam_backend = build_sam_backend(
            checkpoint_path="models/sam_vit_h_4b8939.pth",
            model_type="vit_h",
            device="cuda"
        )
        sam_wrapper = SamPredictorWrapper(backend=sam_backend)
        print("[INFO] SAM后端初始化成功")
        
    except Exception as e:
        print(f"[ERROR] SAM后端初始化失败: {e}")
        return 1
    
    # 生成SAM初始掩码
    try:
        sam_mask = generate_sam_mask_from_prior(image, roi_box, sam_wrapper, prior_mask)
        
        # 创建输出目录
        output_dir = "outputs/test_sam_initial"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存SAM掩码
        save_image(os.path.join(output_dir, f"{sample_name}_sam_initial_mask.png"), 
                   (sam_mask > 0).astype(np.uint8) * 255)
        
        # 创建可视化结果
        result_image = create_result_visualization(image, roi_box, sam_mask, prior_mask)
        save_image(os.path.join(output_dir, f"{sample_name}_sam_result_visualization.png"), result_image)
        
        print(f"[SUCCESS] SAM初始掩码生成完成!")
        print(f"[INFO] 输出目录: {output_dir}")
        print(f"[INFO] SAM掩码像素数: {(sam_mask > 0).sum()}")
        print(f"[INFO] 原始掩码像素数: {(prior_mask > 0).sum()}")
        
        # 计算与原始掩码的IoU
        sam_binary = (sam_mask > 0).astype(np.uint8)
        prior_binary = (prior_mask > 0).astype(np.uint8)
        intersection = (sam_binary & prior_binary).sum()
        union = (sam_binary | prior_binary).sum()
        iou = intersection / union if union > 0 else 0
        
        print(f"[INFO] 与原始掩码的IoU: {iou:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] SAM掩码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())