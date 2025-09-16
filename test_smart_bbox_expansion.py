#!/usr/bin/env python3
"""
测试智能BBox扩展功能
验证当有正点提示在原BBox外时，系统能否正确扩展BBox
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from refactor_sam_sculpt import ROIBox, calculate_smart_roi_expansion

def test_smart_roi_expansion():
    print("=== 测试智能ROI扩展功能 ===")
    
    # 创建测试用的原始ROI框
    original_roi = ROIBox(100, 100, 300, 200)  # 200x100的矩形
    print(f"原始ROI: ({original_roi.x0}, {original_roi.y0}, {original_roi.x1}, {original_roi.y1})")
    
    # 图像尺寸
    image_shape = (500, 600)  # H=500, W=600
    square_size = 80.0  # 小正方形边长
    
    # 测试用例1: 没有正点，应该不扩展
    print("\n--- 测试用例1: 无正点 ---")
    pos_points_1 = []
    expanded_roi_1 = calculate_smart_roi_expansion(original_roi, pos_points_1, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_1.x0}, {expanded_roi_1.y0}, {expanded_roi_1.x1}, {expanded_roi_1.y1})")
    assert expanded_roi_1.x0 == original_roi.x0 and expanded_roi_1.y0 == original_roi.y0
    assert expanded_roi_1.x1 == original_roi.x1 and expanded_roi_1.y1 == original_roi.y1
    print("✅ 通过：无正点时不扩展")
    
    # 测试用例2: 正点在左侧外面
    print("\n--- 测试用例2: 正点在左侧外面 ---")
    pos_points_2 = [(80, 150)]  # 在ROI左边界(100)外面
    expanded_roi_2 = calculate_smart_roi_expansion(original_roi, pos_points_2, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_2.x0}, {expanded_roi_2.y0}, {expanded_roi_2.x1}, {expanded_roi_2.y1})")
    assert expanded_roi_2.x0 < original_roi.x0  # 左边界应该向左扩展
    assert expanded_roi_2.y0 == original_roi.y0  # 上下边界不变
    assert expanded_roi_2.y1 == original_roi.y1
    print("✅ 通过：正点在左侧时向左扩展")
    
    # 测试用例3: 正点在右侧外面
    print("\n--- 测试用例3: 正点在右侧外面 ---")
    pos_points_3 = [(320, 150)]  # 在ROI右边界(300)外面
    expanded_roi_3 = calculate_smart_roi_expansion(original_roi, pos_points_3, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_3.x0}, {expanded_roi_3.y0}, {expanded_roi_3.x1}, {expanded_roi_3.y1})")
    assert expanded_roi_3.x1 > original_roi.x1  # 右边界应该向右扩展
    assert expanded_roi_3.x0 == original_roi.x0  # 左边界不变
    print("✅ 通过：正点在右侧时向右扩展")
    
    # 测试用例4: 正点在上方外面
    print("\n--- 测试用例4: 正点在上方外面 ---")
    pos_points_4 = [(200, 80)]  # 在ROI上边界(100)外面
    expanded_roi_4 = calculate_smart_roi_expansion(original_roi, pos_points_4, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_4.x0}, {expanded_roi_4.y0}, {expanded_roi_4.x1}, {expanded_roi_4.y1})")
    assert expanded_roi_4.y0 < original_roi.y0  # 上边界应该向上扩展
    assert expanded_roi_4.x0 == original_roi.x0  # 左右边界不变
    assert expanded_roi_4.x1 == original_roi.x1
    print("✅ 通过：正点在上方时向上扩展")
    
    # 测试用例5: 正点在下方外面
    print("\n--- 测试用例5: 正点在下方外面 ---")
    pos_points_5 = [(200, 220)]  # 在ROI下边界(200)外面
    expanded_roi_5 = calculate_smart_roi_expansion(original_roi, pos_points_5, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_5.x0}, {expanded_roi_5.y0}, {expanded_roi_5.x1}, {expanded_roi_5.y1})")
    assert expanded_roi_5.y1 > original_roi.y1  # 下边界应该向下扩展
    assert expanded_roi_5.y0 == original_roi.y0  # 上边界不变
    print("✅ 通过：正点在下方时向下扩展")
    
    # 测试用例6: 多个正点在不同方向
    print("\n--- 测试用例6: 多个正点在不同方向 ---")
    pos_points_6 = [(80, 150), (320, 180), (200, 80)]  # 左侧、右侧、上方各一个
    expanded_roi_6 = calculate_smart_roi_expansion(original_roi, pos_points_6, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_6.x0}, {expanded_roi_6.y0}, {expanded_roi_6.x1}, {expanded_roi_6.y1})")
    assert expanded_roi_6.x0 < original_roi.x0  # 左边界向左扩展
    assert expanded_roi_6.x1 > original_roi.x1  # 右边界向右扩展
    assert expanded_roi_6.y0 < original_roi.y0  # 上边界向上扩展
    assert expanded_roi_6.y1 == original_roi.y1  # 下边界不变
    print("✅ 通过：多个方向的正点都正确扩展")
    
    # 测试用例7: 正点在ROI内部（不应扩展）
    print("\n--- 测试用例7: 正点在ROI内部 ---")
    pos_points_7 = [(200, 150), (150, 120)]  # 都在ROI内部
    expanded_roi_7 = calculate_smart_roi_expansion(original_roi, pos_points_7, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_7.x0}, {expanded_roi_7.y0}, {expanded_roi_7.x1}, {expanded_roi_7.y1})")
    assert expanded_roi_7.x0 == original_roi.x0 and expanded_roi_7.y0 == original_roi.y0
    assert expanded_roi_7.x1 == original_roi.x1 and expanded_roi_7.y1 == original_roi.y1
    print("✅ 通过：正点在ROI内部时不扩展")
    
    # 测试用例8: 边界检查（不超出图像边界）
    print("\n--- 测试用例8: 边界检查 ---")
    pos_points_8 = [(10, 150), (580, 150)]  # 接近图像左右边界
    expanded_roi_8 = calculate_smart_roi_expansion(original_roi, pos_points_8, square_size, image_shape)
    print(f"扩展结果: ({expanded_roi_8.x0}, {expanded_roi_8.y0}, {expanded_roi_8.x1}, {expanded_roi_8.y1})")
    assert expanded_roi_8.x0 >= 0  # 不超出左边界
    assert expanded_roi_8.x1 <= 600  # 不超出右边界(W=600)
    assert expanded_roi_8.y0 >= 0  # 不超出上边界
    assert expanded_roi_8.y1 <= 500  # 不超出下边界(H=500)
    print("✅ 通过：扩展结果不超出图像边界")
    
    print("\n🎉 所有测试用例通过！智能BBox扩展功能工作正常。")

def demo_expansion_visualization():
    """演示扩展效果的可视化"""
    print("\n=== 扩展效果演示 ===")
    
    original_roi = ROIBox(200, 150, 400, 250)  # 200x100的ROI
    image_shape = (600, 800)  # H=600, W=800
    square_size = 100.0
    
    # 模拟几种典型场景
    scenarios = [
        ("场景1: 目标延伸到左侧", [(180, 200), (160, 180)]),
        ("场景2: 目标延伸到右上角", [(420, 200), (450, 130)]),
        ("场景3: 目标延伸到多个方向", [(170, 200), (430, 180), (300, 120), (350, 270)]),
    ]
    
    for scenario_name, pos_points in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  正点位置: {pos_points}")
        print(f"  原始ROI: ({original_roi.x0}, {original_roi.y0}, {original_roi.x1}, {original_roi.y1})")
        
        expanded_roi = calculate_smart_roi_expansion(original_roi, pos_points, square_size, image_shape)
        print(f"  扩展ROI: ({expanded_roi.x0:.1f}, {expanded_roi.y0:.1f}, {expanded_roi.x1:.1f}, {expanded_roi.y1:.1f})")
        
        # 计算扩展量
        left_expansion = original_roi.x0 - expanded_roi.x0
        right_expansion = expanded_roi.x1 - original_roi.x1
        top_expansion = original_roi.y0 - expanded_roi.y0
        bottom_expansion = expanded_roi.y1 - original_roi.y1
        
        expansions = []
        if left_expansion > 0:
            expansions.append(f"左:{left_expansion:.1f}")
        if right_expansion > 0:
            expansions.append(f"右:{right_expansion:.1f}")
        if top_expansion > 0:
            expansions.append(f"上:{top_expansion:.1f}")
        if bottom_expansion > 0:
            expansions.append(f"下:{bottom_expansion:.1f}")
        
        if expansions:
            print(f"  扩展量: {', '.join(expansions)} 像素")
        else:
            print(f"  扩展量: 无扩展")

if __name__ == "__main__":
    test_smart_roi_expansion()
    demo_expansion_visualization()