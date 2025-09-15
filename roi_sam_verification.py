#!/usr/bin/env python3
"""
简单的SAM分割验证脚本
直接使用SAM在ROI框内进行分割
"""

import cv2
import numpy as np
import torch
import os
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("请先安装segment-anything: pip install segment-anything")
    sys.exit(1)

def load_sam_model(checkpoint="models/sam_vit_h_4b8939.pth"):
    """加载SAM模型"""
    if not os.path.exists(checkpoint):
        print(f"模型文件不存在: {checkpoint}")
        return None
        
    try:
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print(f"SAM模型加载成功，使用设备: {device}")
        return predictor
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def segment_roi(predictor, image, roi_box):
    """在ROI框内进行SAM分割"""
    predictor.set_image(image)
    
    # 使用ROI框中心点
    x1, y1, x2, y2 = roi_box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # 进行分割
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[center_x, center_y]]),
        point_labels=np.array([1]),
        box=np.array(roi_box),
        multimask_output=True,
    )
    
    # 返回最佳掩码
    best_idx = np.argmax(scores)
    return masks[best_idx], scores[best_idx]

def create_result_image(image, mask, roi_box):
    """创建分割结果图像"""
    # 创建绿色掩码覆盖
    result = image.copy()
    result[mask > 0] = result[mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
    
    # 画ROI框
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return result

def process_sample(predictor, sample_name):
    """处理单个样本"""
    print(f"\n处理样本: {sample_name}")
    
    # 文件路径
    image_path = f"auxiliary/images/{sample_name}.png"
    box_path = f"auxiliary/box_out/{sample_name}/{sample_name}_sam_boxes.json"
    
    # 检查文件存在
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    if not os.path.exists(box_path):
        print(f"ROI框文件不存在: {box_path}")
        return
    
    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 读取ROI框
    with open(box_path, 'r') as f:
        roi_data = json.load(f)
    
    roi_boxes = roi_data['boxes']
    print(f"ROI框: {roi_boxes[0]}")
    
    # 进行分割
    mask, score = segment_roi(predictor, image_rgb, roi_boxes[0])
    print(f"分割得分: {score:.3f}")
    print(f"分割区域像素数: {np.sum(mask)}")
    
    # 创建结果图像（绿色掩码覆盖）
    result_image = create_result_image(image, mask, roi_boxes[0])
    
    # 保存结果
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/{sample_name}_sam_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存: {output_path}")
    
    return result_image, mask

def main():
    """主函数"""
    print("=== 简单SAM分割验证 (f样本) ===")
    
    # 加载SAM模型
    predictor = load_sam_model()
    if predictor is None:
        return
    
    # 处理f样本
    result_image, mask = process_sample(predictor, "f")
    
    print("\nf样本处理完成！")
    print(f"结果图像尺寸: {result_image.shape}")
    print(f"掩码尺寸: {mask.shape}")

if __name__ == "__main__":
    main()
