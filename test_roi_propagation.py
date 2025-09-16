#!/usr/bin/env python3
"""
测试ROI传递功能
验证扩展后的ROI能否正确传递到下一轮
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from refactor_sam_sculpt import ROIBox, refine_mask_with_points, calculate_smart_roi_expansion

# Mock SamPredictor for testing
class MockSamPredictor:
    def __init__(self):
        pass
    
    def set_image(self, image):
        pass
    
    def predict(self, point_coords, point_labels, box, multimask_output=True):
        # 返回模拟的掩码
        H, W = 500, 600
        mask = np.zeros((H, W), dtype=bool)
        # 在box区域内创建一个简单的掩码
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        
        masks = np.array([mask])
        scores = np.array([0.9])
        logits = None
        return masks, scores, logits

def test_roi_propagation():
    print("=== 测试ROI传递功能 ===")
    
    # 创建模拟数据
    predictor = MockSamPredictor()
    image = np.zeros((500, 600, 3), dtype=np.uint8)  # H=500, W=600
    original_roi = ROIBox(100, 100, 300, 200)
    prev_mask = np.zeros((500, 600), dtype=np.uint8)
    square_size = 80.0
    
    print(f"原始ROI: ({original_roi.x0}, {original_roi.y0}, {original_roi.x1}, {original_roi.y1})")
    
    # 测试场景1: 有正点在ROI外侧
    print("\n--- 测试场景1: 正点在左侧外面 ---")
    pos_points_1 = [(80, 150)]  # 在ROI左边界外
    neg_points_1 = []
    
    refined_mask_1, updated_roi_1 = refine_mask_with_points(
        predictor, image, original_roi, pos_points_1, neg_points_1, prev_mask, square_size
    )
    
    print(f"更新后ROI: ({updated_roi_1.x0:.1f}, {updated_roi_1.y0:.1f}, {updated_roi_1.x1:.1f}, {updated_roi_1.y1:.1f})")
    
    # 验证ROI确实被扩展了
    assert updated_roi_1.x0 < original_roi.x0, "左边界应该被扩展"
    assert updated_roi_1.y0 == original_roi.y0, "上下边界不应该变化"
    assert updated_roi_1.y1 == original_roi.y1, "上下边界不应该变化"
    print("✅ ROI正确扩展到左侧")
    
    # 测试场景2: 使用上一轮扩展后的ROI
    print("\n--- 测试场景2: 使用扩展后的ROI进行下一轮 ---")
    pos_points_2 = [(330, 150)]  # 在原始ROI右边界外，但可能在扩展ROI内
    neg_points_2 = []
    
    refined_mask_2, updated_roi_2 = refine_mask_with_points(
        predictor, image, updated_roi_1, pos_points_2, neg_points_2, prev_mask, square_size
    )
    
    print(f"第二次更新后ROI: ({updated_roi_2.x0:.1f}, {updated_roi_2.y0:.1f}, {updated_roi_2.x1:.1f}, {updated_roi_2.y1:.1f})")
    
    # 验证ROI进一步扩展
    assert updated_roi_2.x1 > original_roi.x1, "右边界应该被扩展"
    print("✅ ROI正确扩展到右侧")
    
    # 测试场景3: 正点在ROI内部，不应该扩展
    print("\n--- 测试场景3: 正点在ROI内部 ---")
    pos_points_3 = [(200, 150)]  # 在更新后的ROI内部
    neg_points_3 = []
    
    refined_mask_3, updated_roi_3 = refine_mask_with_points(
        predictor, image, updated_roi_2, pos_points_3, neg_points_3, prev_mask, square_size
    )
    
    print(f"第三次更新后ROI: ({updated_roi_3.x0:.1f}, {updated_roi_3.y0:.1f}, {updated_roi_3.x1:.1f}, {updated_roi_3.y1:.1f})")
    
    # 验证ROI没有进一步变化
    assert updated_roi_3.x0 == updated_roi_2.x0, "ROI不应该进一步变化"
    assert updated_roi_3.y0 == updated_roi_2.y0, "ROI不应该进一步变化"
    assert updated_roi_3.x1 == updated_roi_2.x1, "ROI不应该进一步变化" 
    assert updated_roi_3.y1 == updated_roi_2.y1, "ROI不应该进一步变化"
    print("✅ ROI正确保持不变")
    
    # 测试场景4: 没有点时的处理
    print("\n--- 测试场景4: 没有任何点 ---")
    pos_points_4 = []
    neg_points_4 = []
    
    refined_mask_4, updated_roi_4 = refine_mask_with_points(
        predictor, image, updated_roi_3, pos_points_4, neg_points_4, prev_mask, square_size
    )
    
    print(f"无点时ROI: ({updated_roi_4.x0:.1f}, {updated_roi_4.y0:.1f}, {updated_roi_4.x1:.1f}, {updated_roi_4.y1:.1f})")
    
    # 验证ROI和掩码都没有变化
    assert updated_roi_4.x0 == updated_roi_3.x0, "ROI应该保持不变"
    assert updated_roi_4.y0 == updated_roi_3.y0, "ROI应该保持不变"
    assert updated_roi_4.x1 == updated_roi_3.x1, "ROI应该保持不变"
    assert updated_roi_4.y1 == updated_roi_3.y1, "ROI应该保持不变"
    assert np.array_equal(refined_mask_4, prev_mask), "掩码应该保持不变"
    print("✅ 无点时ROI和掩码保持不变")
    
    print("\n🎉 ROI传递功能测试完成！所有测试用例通过。")
    
    # 总结ROI变化过程
    print("\n=== ROI变化总结 ===")
    print(f"原始ROI:      ({original_roi.x0}, {original_roi.y0}, {original_roi.x1}, {original_roi.y1})")
    print(f"第1轮扩展后:  ({updated_roi_1.x0:.1f}, {updated_roi_1.y0:.1f}, {updated_roi_1.x1:.1f}, {updated_roi_1.y1:.1f})")
    print(f"第2轮扩展后:  ({updated_roi_2.x0:.1f}, {updated_roi_2.y0:.1f}, {updated_roi_2.x1:.1f}, {updated_roi_2.y1:.1f})")
    print(f"第3轮保持:    ({updated_roi_3.x0:.1f}, {updated_roi_3.y0:.1f}, {updated_roi_3.x1:.1f}, {updated_roi_3.y1:.1f})")
    print(f"第4轮保持:    ({updated_roi_4.x0:.1f}, {updated_roi_4.y0:.1f}, {updated_roi_4.x1:.1f}, {updated_roi_4.y1:.1f})")

if __name__ == "__main__":
    test_roi_propagation()