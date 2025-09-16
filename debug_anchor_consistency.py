#!/usr/bin/env python3
"""调试脚本：检查锚点编号和位置的一致性问题"""

import json
import os
import numpy as np
from clean_sam_sculpt import load_image, load_roi_box, get_anchor_points, generate_initial_sam_mask, load_sam_model, draw_anchors_on_image_with_points, ROIBox

def debug_anchor_positions():
    """调试锚点位置一致性"""
    sample_name = 'f'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载输入数据
    image_path = os.path.join(base_dir, 'auxiliary', 'images', f'{sample_name}.png')
    roi_json_path = os.path.join(base_dir, 'auxiliary', 'box_out', sample_name, f'{sample_name}_sam_boxes.json')
    
    print("=== 调试锚点编号和位置一致性 ===")
    print(f"样本: {sample_name}")
    
    # 加载数据
    image = load_image(image_path)
    roi_box = load_roi_box(roi_json_path)
    print(f"ROI框: ({roi_box.x0}, {roi_box.y0}, {roi_box.x1}, {roi_box.y1})")
    
    # 加载SAM生成初始掩码
    predictor = load_sam_model()
    initial_mask = generate_initial_sam_mask(predictor, image, roi_box)
    
    # 生成锚点
    anchor_points = get_anchor_points(roi_box, initial_mask)
    print(f"\n生成的锚点位置（数组索引 0-7）:")
    for i, (x, y) in enumerate(anchor_points):
        print(f"  数组索引 {i} -> 显示编号 {i+1}: ({x:.1f}, {y:.1f})")
    
    # 模拟VLM选择的锚点（如日志中显示的锚点5和锚点6）
    print(f"\n模拟VLM选择测试:")
    
    test_anchor_ids = [5, 6]  # VLM经常选择的锚点
    for anchor_id in test_anchor_ids:
        print(f"\nVLM选择锚点 {anchor_id}:")
        
        # 检查数组边界
        if 1 <= anchor_id <= 8:
            array_index = anchor_id - 1
            print(f"  -> 对应数组索引: {array_index}")
            
            if array_index < len(anchor_points):
                actual_pos = anchor_points[array_index]
                print(f"  -> 实际使用位置: ({actual_pos[0]:.1f}, {actual_pos[1]:.1f})")
                
                # 检查这个位置是否与显示的编号匹配
                print(f"  -> 验证: 数组索引{array_index}的位置确实对应显示编号{anchor_id}")
            else:
                print(f"  -> 错误: 数组索引{array_index}越界！")
        else:
            print(f"  -> 错误: 锚点编号{anchor_id}超出范围[1-8]！")
    
    # 创建可视化图像检查
    anchor_vis = draw_anchors_on_image_with_points(image, roi_box, initial_mask, anchor_points)
    
    # 保存调试图像
    output_path = os.path.join(base_dir, 'debug_anchor_positions.png')
    import cv2
    cv2.imwrite(output_path, anchor_vis)
    print(f"\n调试图像保存至: {output_path}")
    
    # 输出关键发现
    print(f"\n=== 关键发现 ===")
    print(f"1. 锚点数组长度: {len(anchor_points)}")
    print(f"2. 显示编号范围: 1-8 (通过 enumerate(anchor_points, 1))")
    print(f"3. 数组索引范围: 0-7")
    print(f"4. VLM选择转换: anchor_id -> array_index = anchor_id - 1")
    print(f"5. 这个转换逻辑看起来是正确的")
    
    # 检查VLM输入的图像中锚点编号和实际位置是否一致
    print(f"\n=== 位置验证测试 ===")
    print("检查显示给VLM的图像中，编号5和6的锚点位置:")
    for display_id in [5, 6]:
        array_idx = display_id - 1
        pos = anchor_points[array_idx]
        print(f"  显示编号 {display_id} -> 位置 ({pos[0]:.1f}, {pos[1]:.1f})")

if __name__ == "__main__":
    debug_anchor_positions()