#!/usr/bin/env python3
"""
数据集样本处理脚本
功能：从COD10K数据集准备SAM Sculpt的输入文件
"""

import json
import os
import cv2
import numpy as np
from typing import Tuple

def get_bbox_from_mask(mask_path: str) -> Tuple[float, float, float, float]:
    """从掩码文件计算边界框"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩码文件: {mask_path}")
    
    # 找到非零像素的坐标
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        raise ValueError(f"掩码为空: {mask_path}")
    
    # 计算边界框
    y_coords = coords[0]
    x_coords = coords[1]
    
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    return float(x_min), float(y_min), float(x_max), float(y_max)

def prepare_sample(sample_id: int = 0, dataset_path: str = "/home/albert/code/CV/dataset"):
    """准备指定ID的样本"""
    
    # 读取样本信息
    sample_info_path = os.path.join(dataset_path, "sample_info.json")
    with open(sample_info_path, 'r') as f:
        samples = json.load(f)
    
    if sample_id >= len(samples):
        raise ValueError(f"样本ID {sample_id} 超出范围 (0-{len(samples)-1})")
    
    sample = samples[sample_id]
    
    print(f"准备样本 {sample_id}:")
    print(f"  名称: {sample['unique_id']}")
    print(f"  类别: {sample['base_class']}")
    
    # 构建文件路径
    image_path = os.path.join(dataset_path, sample['image'])
    mask_path = os.path.join(dataset_path, sample['mask'])
    
    # 验证文件存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
    
    # 从掩码计算ROI框
    x0, y0, x1, y1 = get_bbox_from_mask(mask_path)
    
    # 添加一些padding到边界框
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    padding = 20  # 像素
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    
    roi_box = {
        "boxes": [[x0, y0, x1, y1]]
    }
    
    # 创建样本名称（用于SAM Sculpt）
    image_filename = os.path.basename(sample['image']).split('.')[0]
    sample_name = f"cod10k_{sample_id:04d}"
    
    # 创建输出目录
    base_dir = "/home/albert/code/CV"
    auxiliary_dir = os.path.join(base_dir, "auxiliary")
    
    # 创建目录结构
    images_dir = os.path.join(auxiliary_dir, "images")
    box_out_dir = os.path.join(auxiliary_dir, "box_out", sample_name)
    llm_out_dir = os.path.join(auxiliary_dir, "llm_out")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(box_out_dir, exist_ok=True)
    os.makedirs(llm_out_dir, exist_ok=True)
    
    # 创建符号链接而不是复制文件（节省空间）
    target_image_path = os.path.join(images_dir, f"{sample_name}.png")
    if os.path.exists(target_image_path) or os.path.islink(target_image_path):
        os.unlink(target_image_path)
    os.symlink(image_path, target_image_path)
    
    # 保存ROI框信息
    roi_json_path = os.path.join(box_out_dir, f"{sample_name}_sam_boxes.json")
    with open(roi_json_path, 'w') as f:
        json.dump(roi_box, f, indent=2)
    
    # 创建语义信息文件
    llm_info = {
        "instance": sample['base_class'],
        "reason": f"目标是一个伪装的{sample['base_class']}，位于自然环境中。需要精确分割出其边界。",
        "original_id": sample['unique_id'],
        "source": "COD10K_TEST_DIR"
    }
    
    llm_out_path = os.path.join(llm_out_dir, f"{sample_name}_output.json")
    with open(llm_out_path, 'w', encoding='utf-8') as f:
        json.dump(llm_info, f, ensure_ascii=False, indent=2)
    
    print(f"  ROI框: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
    print(f"  图像尺寸: {w}x{h}")
    print(f"  样本名称: {sample_name}")
    print(f"  符号链接: {target_image_path}")
    print(f"  ROI文件: {roi_json_path}")
    print(f"  语义信息: {llm_out_path}")
    
    return sample_name, sample

def main():
    import argparse
    parser = argparse.ArgumentParser(description="准备COD10K样本用于SAM Sculpt")
    parser.add_argument('--id', type=int, default=0, help='样本ID (默认: 0)')
    parser.add_argument('--dataset', default='/home/albert/code/CV/dataset', help='数据集根目录')
    parser.add_argument('--list', action='store_true', help='列出前10个样本')
    
    args = parser.parse_args()
    
    if args.list:
        # 列出样本信息
        sample_info_path = os.path.join(args.dataset, "sample_info.json")
        with open(sample_info_path, 'r') as f:
            samples = json.load(f)
        
        print("前10个样本:")
        for i, sample in enumerate(samples[:10]):
            print(f"  ID {i}: {sample['unique_id']} - {sample['base_class']}")
        print(f"总共 {len(samples)} 个样本")
        return
    
    try:
        sample_name, sample = prepare_sample(args.id, args.dataset)
        print(f"\n✅ 样本准备完成！")
        print(f"现在可以运行: python clean_sam_sculpt.py --name {sample_name}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1

if __name__ == "__main__":
    main()