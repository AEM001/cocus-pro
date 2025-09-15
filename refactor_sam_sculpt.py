import os
import sys
import json
import numpy as np
import cv2
from typing import List, Tuple

# Runtime env tuning for stability
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("请安装segment-anything: pip install segment-anything")
    sys.exit(1)

from sculptor.vlm.qwen import QwenVLM


class ROIBox:
    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


def _auto_font_and_thickness(base_len: float) -> tuple[float, int]:
    """根据可视尺寸自适应字体大小与线宽。
    base_len: 参考长度（例如ROI或局部方框的边长）
    返回 (font_scale, thickness)
    """
    # 将100~800像素映射到0.6~2.2的字体大小
    font_scale = 0.6 + 1.6 * max(0.0, min(1.0, (base_len - 100.0) / 700.0))
    # 线宽 2~5
    thickness = int(round(2 + 3 * max(0.0, min(1.0, (base_len - 120.0) / 600.0))))
    return font_scale, max(2, thickness)


def load_sam_model(checkpoint_path: str = "models/sam_vit_b_01ec64.pth", device: str = "cuda"):
    """加载SAM模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM模型文件不存在: {checkpoint_path}")
    
    # Auto-detect model type from filename
    if "vit_h" in checkpoint_path:
        model_type = "vit_h"
    elif "vit_l" in checkpoint_path:
        model_type = "vit_l"
    elif "vit_b" in checkpoint_path:
        model_type = "vit_b"
    else:
        model_type = "vit_b"  # Default to ViT-B if unclear

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"SAM模型加载成功 ({model_type})，设备: {device}")
    return predictor


def load_image(path: str) -> np.ndarray:
    """加载图像"""
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_roi_box(json_path: str) -> ROIBox:
    """加载ROI框"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    box = data['boxes'][0]
    return ROIBox(*box)


def save_image(path: str, img: np.ndarray):
    """保存图像"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.ndim == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    cv2.imwrite(path, img_bgr)


def generate_initial_sam_mask(predictor: SamPredictor, image: np.ndarray, roi_box: ROIBox) -> np.ndarray:
    """步骤1: 使用SAM生成初始掩码 (使用ROI框中心点)"""
    predictor.set_image(image)

    # 使用ROI框中心点
    center_x = (roi_box.x0 + roi_box.x1) / 2
    center_y = (roi_box.y0 + roi_box.y1) / 2

    masks, scores, logits = predictor.predict(
        point_coords=np.array([[center_x, center_y]]),
        point_labels=np.array([1]),
        box=np.array([roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1]),
        multimask_output=True,
    )

    # 选择最佳掩码
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # 裁剪到ROI区域
    H, W = image.shape[:2]
    x0, y0, x1, y1 = int(roi_box.x0), int(roi_box.y0), int(roi_box.x1), int(roi_box.y1)
    clipped_mask = np.zeros((H, W), dtype=np.uint8)
    clipped_mask[y0:y1, x0:x1] = (best_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

    print(f"初始SAM掩码生成完成，像素数: {np.sum(clipped_mask > 0)}")
    return clipped_mask


def get_anchor_points(roi_box: ROIBox) -> List[Tuple[float, float]]:
    """获取ROI框的8个锚点坐标 (4角 + 4边中点)"""
    x0, y0, x1, y1 = roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    return [
        (x0, y0), (cx, y0), (x1, y0),  # 1,2,3: 左上，上中，右上
        (x1, cy),                       # 4: 右中
        (x1, y1), (cx, y1), (x0, y1),   # 5,6,7: 右下，下中，左下
        (x0, cy),                       # 8: 左中
    ]


def draw_anchors_on_image(image: np.ndarray, roi_box: ROIBox, mask: np.ndarray) -> np.ndarray:
    """步骤2: 简洁版锚点图（自适应字号）"""
    vis_img = image.copy()

    # 掩码覆盖（绿色）
    vis_img[mask > 0] = vis_img[mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4

    # ROI框
    roi_w = roi_box.x1 - roi_box.x0
    roi_h = roi_box.y1 - roi_box.y0
    base_len = float(min(roi_w, roi_h))
    _, box_th = _auto_font_and_thickness(base_len)
    cv2.rectangle(vis_img, (int(roi_box.x0), int(roi_box.y0)), (int(roi_box.x1), int(roi_box.y1)), (255, 0, 0), box_th)

    # 锚点与编号
    anchor_points = get_anchor_points(roi_box)
    radius = int(max(5, min(16, base_len / 25.0)))
    font_scale, text_th = _auto_font_and_thickness(base_len)
    for i, (x, y) in enumerate(anchor_points, 1):
        cv2.circle(vis_img, (int(x), int(y)), radius, (0, 255, 255), -1)
        cv2.putText(vis_img, str(i), (int(x) + radius + 4, int(y) - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), text_th)

    return vis_img


def create_quadrant_visualization(image: np.ndarray, anchor_point: Tuple[float, float],
                                roi_box: ROIBox, mask: np.ndarray, anchor_id: int,
                                ratio: float = 0.8) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """步骤3: 简洁版象限可视化（自适应字号，位置纠正）"""
    # 计算方形区域大小（相对ROI，适中比例）
    roi_width = roi_box.x1 - roi_box.x0
    roi_height = roi_box.y1 - roi_box.y0
    square_size = max(120.0, float(ratio) * float(min(roi_width, roi_height)))

    # 以锚点为中心构造正方形，边界裁剪并确保仍为正方形
    cx, cy = float(anchor_point[0]), float(anchor_point[1])
    half = square_size / 2.0
    L = max(0, int(round(cx - half)))
    T = max(0, int(round(cy - half)))
    R = min(int(image.shape[1]), int(round(cx + half)))
    Btm = min(int(image.shape[0]), int(round(cy + half)))
    side = min(R - L, Btm - T)
    if side <= 0:
        side = int(max(60, square_size))
    # 重新居中，使其为正方形
    L = max(0, int(round(cx - side / 2.0)))
    T = max(0, int(round(cy - side / 2.0)))
    R = min(int(image.shape[1]), L + side)
    Btm = min(int(image.shape[0]), T + side)

    vis = image.copy()
    vis[mask > 0] = vis[mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4

    # 线宽与字体
    font_scale, thick = _auto_font_and_thickness(side)
    line_th = max(2, thick)

    # 边框与分割线
    cv2.rectangle(vis, (L, T), (R, Btm), (0, 255, 255), line_th)
    cx_i = (L + R) // 2
    cy_i = (T + Btm) // 2
    cv2.line(vis, (cx_i, T), (cx_i, Btm), (0, 255, 255), line_th)
    cv2.line(vis, (L, cy_i), (R, cy_i), (0, 255, 255), line_th)

    # 象限标签（单色、简洁）
    centers = [
        ( (L + cx_i) // 2, (T + cy_i) // 2 ),      # 1 TL
        ( (cx_i + R) // 2, (T + cy_i) // 2 ),      # 2 TR
        ( (cx_i + R) // 2, (cy_i + Btm) // 2 ),    # 3 BR
        ( (L + cx_i) // 2, (cy_i + Btm) // 2 ),    # 4 BL
    ]
    for j, (px, py) in enumerate(centers, 1):
        cv2.putText(vis, str(j), (int(px), int(py)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thick)

    return vis, (L, T, R, Btm)


def get_quadrant_center(square_bounds: Tuple[int, int, int, int], region_id: int) -> Tuple[float, float]:
    """获取指定象限的中心点坐标"""
    x_min, y_min, x_max, y_max = square_bounds
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2

    if region_id == 1:  # 左上
        return ((x_min + mid_x) / 2, (y_min + mid_y) / 2)
    elif region_id == 2:  # 右上
        return ((mid_x + x_max) / 2, (y_min + mid_y) / 2)
    elif region_id == 3:  # 右下
        return ((mid_x + x_max) / 2, (mid_y + y_max) / 2)
    elif region_id == 4:  # 左下
        return ((x_min + mid_x) / 2, (mid_y + y_max) / 2)
    else:
        return (mid_x, mid_y)


def refine_mask_with_points(predictor: SamPredictor, image: np.ndarray, roi_box: ROIBox,
                          pos_points: List[Tuple[float, float]],
                          neg_points: List[Tuple[float, float]],
                          prev_mask: np.ndarray) -> np.ndarray:
    """步骤4: 使用正负点进行SAM分割优化"""
    predictor.set_image(image)

    # 组合所有点和标签
    all_points = pos_points + neg_points
    all_labels = [1] * len(pos_points) + [0] * len(neg_points)

    if not all_points:
        return prev_mask

    masks, scores, logits = predictor.predict(
        point_coords=np.array(all_points),
        point_labels=np.array(all_labels),
        box=np.array([roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1]),
        multimask_output=True,
    )

    # 选择最佳掩码
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # 裁剪到ROI区域
    H, W = image.shape[:2]
    x0, y0, x1, y1 = int(roi_box.x0), int(roi_box.y0), int(roi_box.x1), int(roi_box.y1)
    clipped_mask = np.zeros((H, W), dtype=np.uint8)
    clipped_mask[y0:y1, x0:x1] = (best_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

    return clipped_mask


def _resize_for_vlm(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='SAM+Qwen refinement (camouflage-friendly)')
    ap.add_argument('--name', default='f', help='sample name (e.g., f, dog, q)')
    ap.add_argument('--qwen_dir', default='/home/albert/code/CV/models/Qwen2.5-VL-3B-Instruct', help='Qwen2.5-VL model dir (3B recommended)')
    ap.add_argument('--rounds', type=int, default=4, help='refinement rounds')
    ap.add_argument('--anchors_per_round', type=int, default=2, help='limit anchors per round to save VRAM')
    ap.add_argument('--ratio', type=float, default=0.8, help='square ratio to ROI short side for quadrant crop')
    ap.add_argument('--vlm_max_side', type=int, default=720, help='resize long side before sending to VLM (<=0 to disable)')
    args = ap.parse_args()

    sample_name = args.name
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 输入文件路径
    image_path = os.path.join(base_dir, 'auxiliary', 'images', f'{sample_name}.png')
    roi_json_path = os.path.join(base_dir, 'auxiliary', 'box_out', sample_name, f'{sample_name}_sam_boxes.json')

    # 获取实例名称
    llm_out_path = os.path.join(base_dir, 'auxiliary', 'llm_out', f'{sample_name}_output.json')
    instance_name = sample_name
    if os.path.exists(llm_out_path):
        try:
            with open(llm_out_path, 'r') as f:
                instance_name = json.load(f).get('instance', sample_name)
        except:
            pass

    # 输出目录
    output_dir = os.path.join(base_dir, 'outputs', 'refactor_sculpt', sample_name)

    print(f"=== 重构后的SAM分割流程 (样本: {sample_name}, 实例: {instance_name}) ===")

    # 加载数据
    print("加载输入数据...")
    image = load_image(image_path)
    roi_box = load_roi_box(roi_json_path)
    print(f"ROI框: ({roi_box.x0}, {roi_box.y0}, {roi_box.x1}, {roi_box.y1})")

    # 加载SAM模型
    print("加载SAM模型...")
    predictor = load_sam_model()

    # 初始化VLM (仅Qwen，使用本地AWQ模型)
    print("初始化VLM (Qwen)...")
    qwen_dir = args.qwen_dir or os.environ.get('QWEN_VL_MODEL_DIR', os.path.join(base_dir, 'models', 'Qwen2.5-VL-3B-Instruct'))
    vlm = QwenVLM(mode='local', model_dir=qwen_dir, gen_max_new_tokens=128, do_sample=False)

    # 步骤1: 生成初始SAM掩码
    print("步骤1: 生成初始SAM掩码...")
    current_mask = generate_initial_sam_mask(predictor, image, roi_box)
    save_image(os.path.join(output_dir, 'step1_initial_mask.png'), current_mask)

    # 迭代优化
    max_rounds = int(args.rounds)
    for round_idx in range(max_rounds):
        print(f"\n=== 第 {round_idx + 1} 轮优化 ===")

        # 步骤2: 绘制锚点，让VLM选择需要精修的锚点
        print("步骤2: 绘制锚点并请求VLM选择...")
        anchor_vis = draw_anchors_on_image(image, roi_box, current_mask)
        save_image(os.path.join(output_dir, f'step2_round{round_idx + 1}_anchors.png'), anchor_vis)

        # 传给VLM前缩放，降低显存压力
        anchor_vis_vlm = _resize_for_vlm(anchor_vis, int(args.vlm_max_side))
        anchor_response = vlm.choose_anchors(anchor_vis_vlm, instance_name)
        selected_anchors = anchor_response.get('anchors_to_refine', [])
        # 限制每轮最多处理的锚点数量
        k = max(1, int(args.anchors_per_round))
        selected_anchors = selected_anchors[:k]

        # Fallback: if VLM returns no anchors, use a default anchor to keep pipeline going
        if not selected_anchors:
            print("[WARN] VLM未选择任何锚点，使用fallback策略选择锚点1")
            print(f"[DEBUG] VLM原始响应: '{anchor_response.get('raw_text', 'N/A')}'")
            fallback_anchor = {"id": 1, "reason": "fallback selection when VLM returned empty"}
            selected_anchors = [fallback_anchor]
            print(f"[INFO] 使用fallback锚点: {selected_anchors}")

        print(f"VLM选择的锚点: {selected_anchors}")

        # 收集所有正负点
        all_pos_points = []
        all_neg_points = []
        anchor_points = get_anchor_points(roi_box)

        for anchor_info in selected_anchors:
            anchor_id = int(anchor_info.get('id', 0))
            if not (1 <= anchor_id <= 8):
                continue

            anchor_point = anchor_points[anchor_id - 1]

            # 步骤3: 创建象限可视化vlm_max_side
            print(f"步骤3: 为锚点 {anchor_id} 创建象限...")
            quad_vis, square_bounds = create_quadrant_visualization(
                image, anchor_point, roi_box, current_mask, anchor_id, ratio=float(args.ratio)
            )
            save_image(os.path.join(output_dir, f'step3_round{round_idx + 1}_anchor{anchor_id}_quadrants.png'), quad_vis)

            # 传给VLM前缩放
            quad_vis_vlm = _resize_for_vlm(quad_vis, int(args.vlm_max_side))
            quad_response = vlm.quadrant_edits(quad_vis_vlm, instance_name, anchor_id)
            edits = quad_response.get('edits', [])

            print(f"锚点 {anchor_id} 的编辑指令: {edits}")

            # 根据VLM指令生成点
            for edit in edits:
                region_id = int(edit.get('region_id', 0))
                action = str(edit.get('action', ''))

                if region_id in [1, 2, 3, 4] and action in ['pos', 'neg']:
                    point = get_quadrant_center(square_bounds, region_id)
                    if action == 'pos':
                        all_pos_points.append(point)
                    else:
                        all_neg_points.append(point)
        
        # 卸载VLM模型以释政GPU内存供SAM使用
        print("[INFO] 卸载VLM模型释放内存...")
        vlm.unload_model()

        # 步骤4: 使用收集的点进行sam分割
        if all_pos_points or all_neg_points:
            print(f"步骤4: 使用 {len(all_pos_points)} 个正点和 {len(all_neg_points)} 个负点进行分割...")
            # 清理GPU缓存以释放内存
            import torch
            torch.cuda.empty_cache()
            current_mask = refine_mask_with_points(predictor, image, roi_box,
                                                 all_pos_points, all_neg_points, current_mask)
            save_image(os.path.join(output_dir, f'step4_round{round_idx + 1}_refined_mask.png'), current_mask)
            print(f"优化后掩码像素数: {np.sum(current_mask > 0)}")
        else:
            print("未生成任何点，跳过本轮优化")

    # 保存最终结果
    save_image(os.path.join(output_dir, 'final_result.png'), current_mask)

    # 创建最终可视化
    final_vis = image.copy()
    final_vis[current_mask > 0] = final_vis[current_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
    save_image(os.path.join(output_dir, 'final_visualization.png'), final_vis)

    print("\n=== 重构完成! ===")
    print(f"最终掩码像素数: {np.sum(current_mask > 0)}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
