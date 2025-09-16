import os
import sys
import json
import numpy as np
import cv2
from typing import List, Tuple, Optional

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


def extract_main_contour(mask: np.ndarray) -> np.ndarray:
    """从二值掩码提取主轮廓"""
    # 确保掩码是二值的
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        # 如果没有找到轮廓，返回空数组
        return np.array([])
    
    # 选择面积最大的轮廓作为主轮廓
    main_contour = max(contours, key=cv2.contourArea)
    
    # 将轮廓点reshape为 (N, 2) 的形式
    return main_contour.squeeze(axis=1) if len(main_contour.shape) == 3 else main_contour


def find_line_contour_intersections(line_start: Tuple[float, float], line_end: Tuple[float, float], contour: np.ndarray) -> List[Tuple[float, float]]:
    """找到线段与轮廓的交点"""
    if len(contour) == 0:
        return []

    intersections = []
    x1, y1 = line_start
    x2, y2 = line_end

    # 遍历轮廓的每条边，找交点
    for i in range(len(contour)):
        # 当前边的两个端点
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]

        x3, y3 = float(p1[0]), float(p1[1])
        x4, y4 = float(p2[0]), float(p2[1])

        # 计算两条线段的交点（如果存在）
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # 平行线
            continue

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # 检查交点是否在两条线段上
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            intersections.append((ix, iy))

    return intersections


def get_anchor_points(roi_box: ROIBox, mask: np.ndarray) -> List[Tuple[float, float]]:
    """基于BBox连线与轮廓交点的锚点生成方法"""
    # 提取主轮廓
    main_contour = extract_main_contour(mask)

    # 如果没有轮廓，返回基础锚点作为fallback
    if len(main_contour) == 0:
        print("[WARN] 未找到主轮廓，使用基础锚点")
        x0, y0, x1, y1 = roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return [
            (x0, y0), (cx, y0), (x1, y0),  # 1,2,3: 左上，上中，右上
            (x1, cy),                       # 4: 右中
            (x1, y1), (cx, y1), (x0, y1),   # 5,6,7: 右下，下中，左下
            (x0, cy),                       # 8: 左中
        ]

    x0, y0, x1, y1 = roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    # 定义4条关键线段
    lines = [
        # 对角线1：左上角到右下角
        ((x0, y0), (x1, y1)),
        # 对角线2：右上角到左下角
        ((x1, y0), (x0, y1)),
        # 垂直中线：上边中点到下边中点
        ((cx, y0), (cx, y1)),
        # 水平中线：左边中点到右边中点
        ((x0, cy), (x1, cy))
    ]

    # 收集所有交点
    all_intersections = []
    for line_start, line_end in lines:
        intersections = find_line_contour_intersections(line_start, line_end, main_contour)
        all_intersections.extend(intersections)

    # 如果交点不足，使用基础锚点作为补充
    if len(all_intersections) < 8:
        print(f"[INFO] 找到 {len(all_intersections)} 个交点，使用基础锚点补充")
        fallback_anchors = [
            (x0, y0), (cx, y0), (x1, y0),  # 1,2,3: 左上，上中，右上
            (x1, cy),                       # 4: 右中
            (x1, y1), (cx, y1), (x0, y1),   # 5,6,7: 右下，下中，左下
            (x0, cy),                       # 8: 左中
        ]

        # 使用交点作为前几个锚点，剩余用基础锚点填充
        result_anchors = all_intersections[:]
        for i in range(len(all_intersections), 8):
            if i < len(fallback_anchors):
                result_anchors.append(fallback_anchors[i])
    else:
        # 如果交点过多，选择8个最具代表性的点
        # 按照与BBox角点和边中点的距离进行选择
        reference_anchors = [
            (x0, y0), (cx, y0), (x1, y0),  # 1,2,3: 左上，上中，右上
            (x1, cy),                       # 4: 右中
            (x1, y1), (cx, y1), (x0, y1),   # 5,6,7: 右下，下中，左下
            (x0, cy),                       # 8: 左中
        ]
        result_anchors = []

        for ref_anchor in reference_anchors:
            # 找到距离参考锚点最近的交点
            if all_intersections:
                distances = [
                    (ix - ref_anchor[0])**2 + (iy - ref_anchor[1])**2
                    for ix, iy in all_intersections
                ]
                closest_idx = np.argmin(distances)
                closest_intersection = all_intersections.pop(closest_idx)
                result_anchors.append(closest_intersection)
            else:
                result_anchors.append(ref_anchor)

    # 打印调试信息
    for i, anchor in enumerate(result_anchors[:8]):
        print(f"锚点 {i+1}: ({anchor[0]:.1f}, {anchor[1]:.1f})")

    return result_anchors[:8]


def draw_anchors_on_image(image: np.ndarray, roi_box: ROIBox, mask: np.ndarray) -> np.ndarray:
    """步骤2: 简洁版锚点图（去除绿色覆盖，仅显示轮廓和锚点）"""
    vis_img = image.copy()

    # 提取并绘制轮廓
    main_contour = extract_main_contour(mask)
    if len(main_contour) > 0:
        # 绘制轮廓线（黄色，较粗）
        roi_w = roi_box.x1 - roi_box.x0
        roi_h = roi_box.y1 - roi_box.y0
        base_len = float(min(roi_w, roi_h))
        _, contour_th = _auto_font_and_thickness(base_len)
        contour_th = max(2, contour_th)
        cv2.drawContours(vis_img, [main_contour.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 255), contour_th)

    # ROI框（蓝色边框）
    roi_w = roi_box.x1 - roi_box.x0
    roi_h = roi_box.y1 - roi_box.y0
    base_len = float(min(roi_w, roi_h))
    _, box_th = _auto_font_and_thickness(base_len)
    cv2.rectangle(vis_img, (int(roi_box.x0), int(roi_box.y0)), (int(roi_box.x1), int(roi_box.y1)), (255, 0, 0), box_th)

    # 使用基于连线交点的锚点
    projected_anchor_points = get_anchor_points(roi_box, mask)
    radius = int(max(5, min(16, base_len / 25.0)))
    font_scale, text_th = _auto_font_and_thickness(base_len)
    
    for i, (x, y) in enumerate(projected_anchor_points, 1):
        # 绘制锚点圆点（红色）
        cv2.circle(vis_img, (int(x), int(y)), radius, (0, 0, 255), -1)
        # 绘制编号（白色文字，黑色边框）
        text_x = int(x) + radius + 4
        text_y = int(y) - radius - 2
        # 添加黑色边框使数字更清晰
        cv2.putText(vis_img, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_th + 2)
        cv2.putText(vis_img, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_th)

    return vis_img


def get_contour_tangent_at_point(point: Tuple[float, float], contour: np.ndarray, window_size: int = 5) -> Tuple[float, float]:
    """计算轮廓在指定点处的切线向量"""
    if len(contour) == 0:
        return (1.0, 0.0)  # 默认水平方向

    px, py = float(point[0]), float(point[1])

    # 找到距离锚点最近的轮廓点
    distances = np.sqrt((contour[:, 0] - px) ** 2 + (contour[:, 1] - py) ** 2)
    closest_idx = np.argmin(distances)

    # 在最近点周围取一个窗口来计算切线
    n_points = len(contour)
    start_idx = (closest_idx - window_size) % n_points
    end_idx = (closest_idx + window_size) % n_points

    # 取窗口内的点
    if start_idx <= end_idx:
        window_points = contour[start_idx:end_idx+1]
    else:
        window_points = np.concatenate([contour[start_idx:], contour[:end_idx+1]])

    if len(window_points) < 2:
        return (1.0, 0.0)

    # 计算平均切线方向
    tangent_vectors = []
    for i in range(len(window_points) - 1):
        dx = window_points[i+1][0] - window_points[i][0]
        dy = window_points[i+1][1] - window_points[i][1]
        if dx != 0 or dy != 0:
            tangent_vectors.append((dx, dy))

    if not tangent_vectors:
        return (1.0, 0.0)

    # 计算平均切线方向
    avg_dx = sum(v[0] for v in tangent_vectors) / len(tangent_vectors)
    avg_dy = sum(v[1] for v in tangent_vectors) / len(tangent_vectors)

    # 归一化
    length = np.sqrt(avg_dx * avg_dx + avg_dy * avg_dy)
    if length < 1e-6:
        return (1.0, 0.0)

    return (avg_dx / length, avg_dy / length)


def create_tangent_square_visualization(image: np.ndarray, anchor_point: Tuple[float, float],
                                      roi_box: ROIBox, mask: np.ndarray, anchor_id: int,
                                      ratio: float = 0.8) -> Tuple[np.ndarray, Tuple[int, int, int, int, Tuple[float, float], Tuple[float, float]]]:
    """基于切线的内外正方形可视化"""
    # 计算正方形半边长
    roi_width = roi_box.x1 - roi_box.x0
    roi_height = roi_box.y1 - roi_box.y0
    base_size = max(120.0, float(ratio) * float(min(roi_width, roi_height)))
    half_size = base_size / 2.0

    # 锚点坐标
    cx, cy = float(anchor_point[0]), float(anchor_point[1])

    # 提取主轮廓并计算切线方向
    main_contour = extract_main_contour(mask)
    if len(main_contour) > 0:
        tangent_dx, tangent_dy = get_contour_tangent_at_point(anchor_point, main_contour)
    else:
        tangent_dx, tangent_dy = 1.0, 0.0  # 默认水平方向

    # 计算法线方向（垂直于切线，指向内侧）
    normal_dx, normal_dy = -tangent_dy, tangent_dx

    # 计算正方形的四个角点
    # 以切线和法线方向为基础构建正方形
    corners = [
        (cx - half_size * tangent_dx - half_size * normal_dx, cy - half_size * tangent_dy - half_size * normal_dy),  # 左下
        (cx + half_size * tangent_dx - half_size * normal_dx, cy + half_size * tangent_dy - half_size * normal_dy),  # 右下
        (cx + half_size * tangent_dx + half_size * normal_dx, cy + half_size * tangent_dy + half_size * normal_dy),  # 右上
        (cx - half_size * tangent_dx + half_size * normal_dx, cy - half_size * tangent_dy + half_size * normal_dy),  # 左上
    ]

    # 计算包围盒
    min_x = min(corner[0] for corner in corners)
    max_x = max(corner[0] for corner in corners)
    min_y = min(corner[1] for corner in corners)
    max_y = max(corner[1] for corner in corners)

    L = max(0, int(round(min_x)))
    T = max(0, int(round(min_y)))
    R = min(int(image.shape[1]), int(round(max_x)))
    Btm = min(int(image.shape[0]), int(round(max_y)))

    vis = image.copy()

    # 绘制轮廓
    if len(main_contour) > 0:
        base_len = float(min(roi_width, roi_height))
        _, contour_th = _auto_font_and_thickness(base_len)
        contour_th = max(2, contour_th)
        cv2.drawContours(vis, [main_contour.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 255), contour_th // 2)

    # 线宽与字体
    font_scale, thick = _auto_font_and_thickness(base_size)
    line_th = max(2, thick)

    # 绘制正方形边框
    square_points = np.array(corners, dtype=np.int32)
    cv2.polylines(vis, [square_points], True, (0, 255, 255), line_th)

    # 绘制切线（水平分割线）
    cut_line_start = (cx - half_size * tangent_dx, cy - half_size * tangent_dy)
    cut_line_end = (cx + half_size * tangent_dx, cy + half_size * tangent_dy)
    cv2.line(vis, (int(cut_line_start[0]), int(cut_line_start[1])),
             (int(cut_line_end[0]), int(cut_line_end[1])), (0, 255, 255), line_th)

    # 计算内外区域的中心点
    inner_center = (cx + half_size * normal_dx * 0.5, cy + half_size * normal_dy * 0.5)  # 内侧区域中心
    outer_center = (cx - half_size * normal_dx * 0.5, cy - half_size * normal_dy * 0.5)  # 外侧区域中心

    # 绘制区域标签
    # "内" 标签 (1)
    cv2.putText(vis, "1", (int(inner_center[0]), int(inner_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 2)
    cv2.putText(vis, "1", (int(inner_center[0]), int(inner_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thick)

    # "外" 标签 (2)
    cv2.putText(vis, "2", (int(outer_center[0]), int(outer_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 2)
    cv2.putText(vis, "2", (int(outer_center[0]), int(outer_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thick)

    return vis, (L, T, R, Btm, inner_center, outer_center)


def get_tangent_region_center(square_bounds: Tuple[int, int, int, int, Tuple[float, float], Tuple[float, float]],
                            region_id: int) -> Tuple[float, float]:
    """获取指定切线区域的中心点坐标"""
    _, _, _, _, inner_center, outer_center = square_bounds

    if region_id == 1:  # 内侧区域
        return inner_center
    elif region_id == 2:  # 外侧区域
        return outer_center
    else:
        # 默认返回内侧中心
        return inner_center


def calculate_smart_roi_expansion(roi_box: ROIBox, pos_points: List[Tuple[float, float]], 
                                   square_size: float, image_shape: Tuple[int, int]) -> ROIBox:
    """根据正点提示智能调整ROI框，向相应边的外侧扩展小正方形的边长"""
    if not pos_points:
        return roi_box  # 没有正点，不扩展
    
    H, W = image_shape
    expanded_roi = ROIBox(roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1)
    
    for px, py in pos_points:
        # 判断正点相对于原ROI的位置，决定扩展方向
        
        # 左边扩展：如果正点在ROI左边界外侧
        if px < roi_box.x0:
            expanded_roi.x0 = max(0, min(expanded_roi.x0, px - square_size / 2))
            print(f"[扩展] 左边界扩展至 {expanded_roi.x0:.1f} (正点位置: {px:.1f})")
            
        # 右边扩展：如果正点在ROI右边界外侧  
        if px > roi_box.x1:
            expanded_roi.x1 = min(W, max(expanded_roi.x1, px + square_size / 2))
            print(f"[扩展] 右边界扩展至 {expanded_roi.x1:.1f} (正点位置: {px:.1f})")
            
        # 上边扩展：如果正点在ROI上边界外侧
        if py < roi_box.y0:
            expanded_roi.y0 = max(0, min(expanded_roi.y0, py - square_size / 2))
            print(f"[扩展] 上边界扩展至 {expanded_roi.y0:.1f} (正点位置: {py:.1f})")
            
        # 下边扩展：如果正点在ROI下边界外侧
        if py > roi_box.y1:
            expanded_roi.y1 = min(H, max(expanded_roi.y1, py + square_size / 2))
            print(f"[扩展] 下边界扩展至 {expanded_roi.y1:.1f} (正点位置: {py:.1f})")
    
    return expanded_roi


def refine_mask_with_points(predictor: SamPredictor, image: np.ndarray, roi_box: ROIBox,
                          pos_points: List[Tuple[float, float]],
                          neg_points: List[Tuple[float, float]],
                          prev_mask: np.ndarray,
                          square_size: float = 0.0,
                          allowed_update_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ROIBox]:
    """步骤4: 使用正负点进行SAM分割优化，支持智能BBox扩展与局部更新门控
    
    Args:
        allowed_update_mask: 允许更新的区域（H×W二值）。如果提供，只有该区域内使用新掩码覆盖，其他位置保持前一轮结果。
    
    返回: (refined_mask, updated_roi_box)
    """
    predictor.set_image(image)

    # 组合所有点和标签
    all_points = pos_points + neg_points
    all_labels = [1] * len(pos_points) + [0] * len(neg_points)

    if not all_points:
        return prev_mask, roi_box  # 没有点时返回原始掩码和ROI

    # 智能调整ROI框：如果有正点在原BBox外，则扩展BBox
    H, W = image.shape[:2]
    smart_roi = calculate_smart_roi_expansion(roi_box, pos_points, square_size, (H, W))
    
    # 打印调整信息
    if (smart_roi.x0 != roi_box.x0 or smart_roi.y0 != roi_box.y0 or 
        smart_roi.x1 != roi_box.x1 or smart_roi.y1 != roi_box.y1):
        print(f"[智能调整] ROI从 ({roi_box.x0:.1f},{roi_box.y0:.1f},{roi_box.x1:.1f},{roi_box.y1:.1f}) ")
        print(f"           扩展至 ({smart_roi.x0:.1f},{smart_roi.y0:.1f},{smart_roi.x1:.1f},{smart_roi.y1:.1f})")
    else:
        print(f"[智能调整] ROI无需扩展，保持原尺寸")

    masks, scores, logits = predictor.predict(
        point_coords=np.array(all_points),
        point_labels=np.array(all_labels),
        box=np.array([smart_roi.x0, smart_roi.y0, smart_roi.x1, smart_roi.y1]),
        multimask_output=True,
    )

    # 选择最佳掩码
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # 使用扩展后的ROI进行裁剪（允许超出原BBox）
    x0, y0, x1, y1 = int(smart_roi.x0), int(smart_roi.y0), int(smart_roi.x1), int(smart_roi.y1)
    clipped_mask = np.zeros((H, W), dtype=np.uint8)
    clipped_mask[y0:y1, x0:x1] = (best_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

    # 应用局部更新门控：仅在允许区域内更新，其它区域保持不变
    if allowed_update_mask is not None:
        gate = (allowed_update_mask > 0).astype(np.uint8)
        final_mask = prev_mask.copy()
        final_mask[gate.astype(bool)] = clipped_mask[gate.astype(bool)]
        return final_mask, smart_roi

    return clipped_mask, smart_roi  # 返回扩展后的ROI供下一轮使用


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
    ap = argparse.ArgumentParser(description='SAM+Qwen refinement with single-anchor optimization')
    ap.add_argument('--name', default='f', help='sample name (e.g., f, dog, q)')
    ap.add_argument('--qwen_dir', default='/home/albert/code/CV/models/Qwen2.5-VL-3B-Instruct', help='Qwen2.5-VL model dir (3B recommended)')
    ap.add_argument('--rounds', type=int, default=4, help='refinement rounds')
    ap.add_argument('--ratio', type=float, default=0.8, help='square ratio to ROI short side for tangent square crop')
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
    output_dir = os.path.join(base_dir, 'outputs', 'clean_sculpt', sample_name)

    print(f"=== 清理版SAM分割流程 (样本: {sample_name}, 实例: {instance_name}) ===")
    print("[优化策略] 使用单点优化策略: VLM每次只选择一个最重要的锚点进行精细优化")

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
    save_image(os.path.join(output_dir, 'round0_step1_initial_mask.png'), current_mask)

    # 迭代优化
    max_rounds = int(args.rounds)
    for round_idx in range(max_rounds):
        print(f"\n=== 第 {round_idx + 1} 轮优化 ===")

        # 步骤2: 绘制锚点，让VLM选择需要精修的锚点
        print("步骤2: 绘制锚点并请求VLM选择...")
        anchor_vis = draw_anchors_on_image(image, roi_box, current_mask)
        save_image(os.path.join(output_dir, f'round{round_idx + 1}_step2_anchors.png'), anchor_vis)

        # 传给VLM前缩放，降低显存压力
        anchor_vis_vlm = _resize_for_vlm(anchor_vis, int(args.vlm_max_side))
        anchor_response = vlm.choose_anchors(anchor_vis_vlm, instance_name)
        selected_anchors = anchor_response.get('anchors_to_refine', [])
        
        # VLM现在应该只返回一个锚点，但保持兼容性，只取第一个
        if len(selected_anchors) > 1:
            print(f"[INFO] VLM返回了{len(selected_anchors)}个锚点，但只使用第一个最重要的")
            selected_anchors = selected_anchors[:1]

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
        
        # 使用基于连线交点的锚点
        projected_anchor_points = get_anchor_points(roi_box, current_mask)

        for anchor_info in selected_anchors:
            anchor_id = int(anchor_info.get('id', 0))
            if not (1 <= anchor_id <= 8):
                continue

            # 使用贴边投影后的锚点位置
            anchor_point = projected_anchor_points[anchor_id - 1]

            # 步骤3: 创建切线内外区域可视化
            print(f"步骤3: 为锚点 {anchor_id} 创建切线内外区域...")
            tangent_vis, square_bounds = create_tangent_square_visualization(
                image, anchor_point, roi_box, current_mask, anchor_id, ratio=float(args.ratio)
            )
            save_image(os.path.join(output_dir, f'round{round_idx + 1}_step3_anchor{anchor_id}_tangent_square.png'), tangent_vis)

            # 传给VLM前缩放
            tangent_vis_vlm = _resize_for_vlm(tangent_vis, int(args.vlm_max_side))
            quad_response = vlm.quadrant_edits(tangent_vis_vlm, instance_name, anchor_id)
            edits = quad_response.get('edits', [])

            print(f"锚点 {anchor_id} 的编辑指令: {edits}")

            # 根据VLM指令生成点
            for edit in edits:
                region_id = int(edit.get('region_id', 0))
                action = str(edit.get('action', ''))

                if region_id in [1, 2] and action in ['pos', 'neg']:
                    point = get_tangent_region_center(square_bounds, region_id)
                    if action == 'pos':
                        all_pos_points.append(point)
                    else:
                        all_neg_points.append(point)
        
        # 卸载VLM模型以释放GPU内存供SAM使用
        print("[INFO] 卸载VLM模型释放内存...")
        vlm.unload_model()

        # 计算小正方形的尺寸（用于智能扩展BBox）
        roi_width = roi_box.x1 - roi_box.x0
        roi_height = roi_box.y1 - roi_box.y0
        square_size = max(120.0, float(args.ratio) * float(min(roi_width, roi_height)))
        
        # 步骤4: 使用收集的点进行sam分割（支持智能BBox扩展 + 局部更新门控）
        if all_pos_points or all_neg_points:
            print(f"步骤4: 使用 {len(all_pos_points)} 个正点和 {len(all_neg_points)} 个负点进行分割...")
            print(f"[参数] 小正方形尺寸: {square_size:.1f} px （用于智能扩展BBox）")
            
            # 构建严格的局部更新门控掩码（仅允许被选中锚点周围的小范围更新）
            H, W = image.shape[:2]
            allowed_update_mask = np.zeros((H, W), dtype=np.uint8)
            
            # 对每个被选中的锚点，创建更小的圆形更新区域
            anchor_radius = int(max(20, min(60, square_size * 1.1)))  # 比原来的正方形要小得多
            
            for anchor_info in selected_anchors:
                anchor_id = int(anchor_info.get('id', 0))
                if 1 <= anchor_id <= 8:
                    # 获取锚点的实际坐标
                    anchor_x, anchor_y = projected_anchor_points[anchor_id - 1]
                    
                    # 创建以锚点为中心的圆形更新区域
                    cy, cx = np.ogrid[:H, :W]
                    dist_from_anchor = np.sqrt((cx - anchor_x)**2 + (cy - anchor_y)**2)
                    circle_mask = dist_from_anchor <= anchor_radius
                    allowed_update_mask[circle_mask] = 1
                    
            updated_area = int(allowed_update_mask.sum())
            print(f"[严格门控] 本轮仅允许被选中锚点周围 {anchor_radius}px 半径内更新，共 {updated_area} 像素")

            # 清理GPU缓存以释放内存
            import torch
            torch.cuda.empty_cache()
            current_mask, updated_roi_box = refine_mask_with_points(
                predictor, image, roi_box,
                all_pos_points, all_neg_points, current_mask, square_size,
                allowed_update_mask=allowed_update_mask,
            )
            # 更新ROI框供下一轮使用
            if (updated_roi_box.x0 != roi_box.x0 or updated_roi_box.y0 != roi_box.y0 or 
                updated_roi_box.x1 != roi_box.x1 or updated_roi_box.y1 != roi_box.y1):
                print(f"[更新] 下一轮将使用新ROI: ({updated_roi_box.x0:.1f},{updated_roi_box.y0:.1f},{updated_roi_box.x1:.1f},{updated_roi_box.y1:.1f})")
                roi_box = updated_roi_box  # 更新roi_box供下一轮使用
            
            save_image(os.path.join(output_dir, f'round{round_idx + 1}_step4_refined_mask.png'), current_mask)
            print(f"优化后掩码像素数: {np.sum(current_mask > 0)}")
        else:
            print("未生成任何点，跳过本轮优化")

    # 保存最终结果
    save_image(os.path.join(output_dir, 'final_result.png'), current_mask)

    # 创建最终可视化
    final_vis = image.copy()
    final_vis[current_mask > 0] = final_vis[current_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
    save_image(os.path.join(output_dir, 'final_visualization.png'), final_vis)

    print("\n=== 清理版重构完成! ===")
    print(f"最终掩码像素数: {np.sum(current_mask > 0)}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()