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


def project_point_to_contour(point: Tuple[float, float], contour: np.ndarray) -> Tuple[float, float]:
    """将点投影到最近的轮廓点"""
    if len(contour) == 0:
        return point
    
    px, py = float(point[0]), float(point[1])
    
    # 计算到所有轮廓点的距离
    distances = np.sqrt((contour[:, 0] - px) ** 2 + (contour[:, 1] - py) ** 2)
    
    # 找到最近的点
    min_idx = np.argmin(distances)
    closest_point = contour[min_idx]
    
    return (float(closest_point[0]), float(closest_point[1]))


def _contour_orientation_sign(contour: np.ndarray) -> float:
    """返回轮廓方向符号 (CCW=+1, CW=-1)。退化情况默认+1。"""
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)
    if len(contour) < 3:
        return 1.0

    xs = contour[:, 0].astype(float)
    ys = contour[:, 1].astype(float)
    rolled_xs = np.roll(xs, -1)
    rolled_ys = np.roll(ys, -1)
    signed_area = np.sum(xs * rolled_ys - ys * rolled_xs) * 0.5
    return 1.0 if signed_area >= 0 else -1.0


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


def get_line_based_anchor_points(roi_box: ROIBox, mask: np.ndarray) -> List[Tuple[float, float]]:
    """基于BBox连线与轮廓交点的锚点生成方法"""
    # 提取主轮廓
    main_contour = extract_main_contour(mask)

    # 如果没有轮廓，返回原始锚点
    if len(main_contour) == 0:
        print("[WARN] 未找到主轮廓，使用原始锚点")
        return get_anchor_points(roi_box)

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

    # 如果交点不足，使用原始锚点作为补充
    if len(all_intersections) < 8:
        print(f"[INFO] 找到 {len(all_intersections)} 个交点，使用原始锚点补充")
        original_anchors = get_anchor_points(roi_box)

        # 使用交点作为前几个锚点，剩余用原始锚点填充
        result_anchors = all_intersections[:]
        for i in range(len(all_intersections), 8):
            if i < len(original_anchors):
                result_anchors.append(original_anchors[i])
    else:
        # 如果交点过多，选择8个最具代表性的点
        # 按照与BBox角点和边中点的距离进行选择
        original_anchors = get_anchor_points(roi_box)
        result_anchors = []

        for orig_anchor in original_anchors:
            # 找到距离原始锚点最近的交点
            if all_intersections:
                distances = [
                    (ix - orig_anchor[0])**2 + (iy - orig_anchor[1])**2
                    for ix, iy in all_intersections
                ]
                closest_idx = np.argmin(distances)
                closest_intersection = all_intersections.pop(closest_idx)
                result_anchors.append(closest_intersection)
            else:
                result_anchors.append(orig_anchor)

    # 打印调试信息
    for i, anchor in enumerate(result_anchors[:8]):
        print(f"锚点 {i+1}: ({anchor[0]:.1f}, {anchor[1]:.1f})")

    return result_anchors[:8]


def get_projected_anchor_points(roi_box: ROIBox, mask: np.ndarray) -> List[Tuple[float, float]]:
    """获取贴边投影后的8个锚点坐标（投影到轮廓上）"""
    # 首先获取原始锚点
    original_anchors = get_anchor_points(roi_box)
    
    # 提取主轮廓
    main_contour = extract_main_contour(mask)
    
    # 如果没有轮廓，返回原始锚点
    if len(main_contour) == 0:
        print("[WARN] 未找到主轮廓，使用原始锚点")
        return original_anchors
    
    # 将每个锚点投影到轮廓上
    projected_anchors = []
    for i, anchor in enumerate(original_anchors):
        projected_point = project_point_to_contour(anchor, main_contour)
        projected_anchors.append(projected_point)
        print(f"锚点 {i+1}: ({anchor[0]:.1f}, {anchor[1]:.1f}) -> ({projected_point[0]:.1f}, {projected_point[1]:.1f})")
    
    return projected_anchors


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
    projected_anchor_points = get_line_based_anchor_points(roi_box, mask)
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
        orientation = _contour_orientation_sign(main_contour)
    else:
        tangent_dx, tangent_dy = 1.0, 0.0  # 默认水平方向
        orientation = 1.0

    # 计算法线方向（垂直于切线，指向内侧）
    normal_dx, normal_dy = -tangent_dy * orientation, tangent_dx * orientation

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


def create_quadrant_visualization(image: np.ndarray, anchor_point: Tuple[float, float],
                                roi_box: ROIBox, mask: np.ndarray, anchor_id: int,
                                ratio: float = 0.8) -> Tuple[np.ndarray, Tuple[int, int, int, int, float]]:
    """步骤3: 圆形扇形分割可视化（替代原方形象限）"""
    # 计算圆形区域半径（相对ROI，适中比例）
    roi_width = roi_box.x1 - roi_box.x0
    roi_height = roi_box.y1 - roi_box.y0
    base_size = max(120.0, float(ratio) * float(min(roi_width, roi_height)))
    radius = base_size / 2.0

    # 以锚点为圆心
    cx, cy = float(anchor_point[0]), float(anchor_point[1])
    
    # 计算包围盒用于返回
    L = max(0, int(round(cx - radius)))
    T = max(0, int(round(cy - radius)))
    R = min(int(image.shape[1]), int(round(cx + radius)))
    Btm = min(int(image.shape[0]), int(round(cy + radius)))

    vis = image.copy()
    # 提取并绘制轮廓（不再使用绿色覆盖）
    main_contour = extract_main_contour(mask)
    if len(main_contour) > 0:
        base_len = float(min(roi_width, roi_height))
        _, contour_th = _auto_font_and_thickness(base_len)
        contour_th = max(2, contour_th)
        cv2.drawContours(vis, [main_contour.reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 255), contour_th // 2)

    # 线宽与字体
    font_scale, thick = _auto_font_and_thickness(radius * 2)
    line_th = max(2, thick)

    # 绘制圆形边框
    cv2.circle(vis, (int(cx), int(cy)), int(radius), (0, 255, 255), line_th)
    
    # 绘制十字分割线（将圆分成四个扇形）
    # 水平线
    cv2.line(vis, (int(cx - radius), int(cy)), (int(cx + radius), int(cy)), (0, 255, 255), line_th)
    # 垂直线
    cv2.line(vis, (int(cx), int(cy - radius)), (int(cx), int(cy + radius)), (0, 255, 255), line_th)

    # 计算四个扇形的中心点位置（在半径的2/3处）
    sector_r = radius * 2.0 / 3.0  # 标签位置在边缘的 2/3 处
    centers = [
        (cx - sector_r * 0.707, cy - sector_r * 0.707),  # 1: 左上 (315度方向)
        (cx + sector_r * 0.707, cy - sector_r * 0.707),  # 2: 右上 (45度方向)
        (cx + sector_r * 0.707, cy + sector_r * 0.707),  # 3: 右下 (135度方向)
        (cx - sector_r * 0.707, cy + sector_r * 0.707),  # 4: 左下 (225度方向)
    ]
    
    # 绘制扇形标签
    for j, (px, py) in enumerate(centers, 1):
        # 添加黑色边框使数字更清晰
        cv2.putText(vis, str(j), (int(px), int(py)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 2)
        cv2.putText(vis, str(j), (int(px), int(py)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thick)

    return vis, (L, T, R, Btm, radius)


def get_sector_center(circle_bounds: Tuple[int, int, int, int, float], region_id: int, anchor_point: Tuple[float, float]) -> Tuple[float, float]:
    """获取指定扇形的中心点坐标"""
    x_min, y_min, x_max, y_max, radius = circle_bounds
    cx, cy = float(anchor_point[0]), float(anchor_point[1])
    
    # 在每个扇形的中心方向上，距离圆心 radius/2 的位置放置点
    sector_r = radius / 2.0  # 点位置在半径的 1/2 处
    
    if region_id == 1:  # 左上扇形 (315度方向)
        angle = -45.0 * np.pi / 180.0  # -45度转换为弧度
        return (cx + sector_r * np.cos(angle), cy + sector_r * np.sin(angle))
    elif region_id == 2:  # 右上扇形 (45度方向)
        angle = 45.0 * np.pi / 180.0
        return (cx + sector_r * np.cos(angle), cy + sector_r * np.sin(angle))
    elif region_id == 3:  # 右下扇形 (135度方向)
        angle = 135.0 * np.pi / 180.0
        return (cx + sector_r * np.cos(angle), cy + sector_r * np.sin(angle))
    elif region_id == 4:  # 左下扇形 (225度方向)
        angle = 225.0 * np.pi / 180.0
        return (cx + sector_r * np.cos(angle), cy + sector_r * np.sin(angle))
    else:
        return (cx, cy)


def create_expanded_roi(roi_box: ROIBox, expansion_ratio: float = 0.2, image_shape: Tuple[int, int] = None) -> ROIBox:
    """创建扩展的ROI区域，用于精细修改阶段"""
    width = roi_box.x1 - roi_box.x0
    height = roi_box.y1 - roi_box.y0

    # 计算扩展量
    expand_x = width * expansion_ratio / 2
    expand_y = height * expansion_ratio / 2

    # 扩展边界
    expanded_x0 = roi_box.x0 - expand_x
    expanded_y0 = roi_box.y0 - expand_y
    expanded_x1 = roi_box.x1 + expand_x
    expanded_y1 = roi_box.y1 + expand_y

    # 确保不超出图像边界
    if image_shape is not None:
        H, W = image_shape
        expanded_x0 = max(0, expanded_x0)
        expanded_y0 = max(0, expanded_y0)
        expanded_x1 = min(W, expanded_x1)
        expanded_y1 = min(H, expanded_y1)

    return ROIBox(expanded_x0, expanded_y0, expanded_x1, expanded_y1)


def create_local_expansion_box(anchor_point: Tuple[float, float], region_center: Tuple[float, float],
                               image_shape: Tuple[int, int], expansion_size: int = 100) -> Tuple[int, int, int, int]:
    """为特定锚点和区域创建局部扩展框"""
    H, W = image_shape
    cx, cy = region_center

    # 创建以区域中心为中心的正方形扩展框
    half_size = expansion_size // 2
    x0 = max(0, int(cx - half_size))
    y0 = max(0, int(cy - half_size))
    x1 = min(W, int(cx + half_size))
    y1 = min(H, int(cy + half_size))

    return (x0, y0, x1, y1)


def update_roi_based_on_square_actions(roi_box: ROIBox, anchor_edits: List[tuple],
                                       ratio: float, image_shape: Tuple[int, int]) -> ROIBox:
    """根据正方形区域和VLM动作直接调整ROI边界"""
    H, W = image_shape
    new_roi = ROIBox(roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1)

    adjustments_made = []

    for anchor_point, square_bounds, edits in anchor_edits:
        if not edits:
            continue

        # 解析正方形边界信息
        L, T, R, Btm, inner_center, outer_center = square_bounds

        for edit in edits:
            region_id = int(edit.get('region_id', 0))
            action = str(edit.get('action', ''))

            if region_id not in [1, 2] or action not in ['pos', 'neg']:
                continue

            # 根据区域和动作调整ROI
            if region_id == 1:  # 内侧区域
                if action == 'pos':
                    # 内侧正点：如果锚点靠近ROI边界，可能需要扩展ROI
                    ix, iy = inner_center

                    # 检查锚点是否靠近ROI边界
                    margin = 50  # 边界阈值

                    if abs(anchor_point[0] - roi_box.x0) < margin:  # 靠近左边界
                        # 向左扩展，使用正方形的左边界
                        extend_to = L - (roi_box.x0 - L) * ratio
                        new_roi.x0 = max(0, min(new_roi.x0, extend_to))
                        adjustments_made.append(f"左边界扩展至 {new_roi.x0:.1f} (内侧正点-边界)")

                    elif abs(anchor_point[0] - roi_box.x1) < margin:  # 靠近右边界
                        # 向右扩展
                        extend_to = R + (R - roi_box.x1) * ratio
                        new_roi.x1 = min(W, max(new_roi.x1, extend_to))
                        adjustments_made.append(f"右边界扩展至 {new_roi.x1:.1f} (内侧正点-边界)")

                    if abs(anchor_point[1] - roi_box.y0) < margin:  # 靠近上边界
                        # 向上扩展
                        extend_to = T - (roi_box.y0 - T) * ratio
                        new_roi.y0 = max(0, min(new_roi.y0, extend_to))
                        adjustments_made.append(f"上边界扩展至 {new_roi.y0:.1f} (内侧正点-边界)")

                    elif abs(anchor_point[1] - roi_box.y1) < margin:  # 靠近下边界
                        # 向下扩展
                        extend_to = Btm + (Btm - roi_box.y1) * ratio
                        new_roi.y1 = min(H, max(new_roi.y1, extend_to))
                        adjustments_made.append(f"下边界扩展至 {new_roi.y1:.1f} (内侧正点-边界)")

                else:  # action == 'neg'
                    # 内侧负点：ROI应该收缩，避开这个内侧区域
                    # 根据内侧区域位置决定收缩方向
                    ix, iy = inner_center

                    # 判断内侧区域相对于ROI的位置
                    roi_cx = (roi_box.x0 + roi_box.x1) / 2
                    roi_cy = (roi_box.y0 + roi_box.y1) / 2

                    if ix < roi_cx - 50:  # 内侧区域在左侧，收缩左边界
                        new_roi.x0 = max(new_roi.x0, ix)
                        adjustments_made.append(f"左边界内缩至 {ix:.1f} (内侧负点)")
                    elif ix > roi_cx + 50:  # 内侧区域在右侧，收缩右边界
                        new_roi.x1 = min(new_roi.x1, ix)
                        adjustments_made.append(f"右边界内缩至 {ix:.1f} (内侧负点)")

                    if iy < roi_cy - 50:  # 内侧区域在上侧，收缩上边界
                        new_roi.y0 = max(new_roi.y0, iy)
                        adjustments_made.append(f"上边界内缩至 {iy:.1f} (内侧负点)")
                    elif iy > roi_cy + 50:  # 内侧区域在下侧，收缩下边界
                        new_roi.y1 = min(new_roi.y1, iy)
                        adjustments_made.append(f"下边界内缩至 {iy:.1f} (内侧负点)")

            else:  # region_id == 2, 外侧区域
                if action == 'pos':
                    # 外侧正点：ROI应该向外扩展包含这个区域
                    ox, oy = outer_center

                    # 使用ratio控制扩展激进程度
                    expansion_factor = ratio

                    # 将正方形边界作为扩展参考
                    if ox < roi_box.x0:  # 外侧区域在左侧，扩展左边界
                        extend_to = L - (roi_box.x0 - L) * expansion_factor
                        new_roi.x0 = max(0, min(new_roi.x0, extend_to))
                        adjustments_made.append(f"左边界扩展至 {new_roi.x0:.1f} (外侧正点)")

                    elif ox > roi_box.x1:  # 外侧区域在右侧，扩展右边界
                        extend_to = R + (R - roi_box.x1) * expansion_factor
                        new_roi.x1 = min(W, max(new_roi.x1, extend_to))
                        adjustments_made.append(f"右边界扩展至 {new_roi.x1:.1f} (外侧正点)")

                    if oy < roi_box.y0:  # 外侧区域在上侧，扩展上边界
                        extend_to = T - (roi_box.y0 - T) * expansion_factor
                        new_roi.y0 = max(0, min(new_roi.y0, extend_to))
                        adjustments_made.append(f"上边界扩展至 {new_roi.y0:.1f} (外侧正点)")

                    elif oy > roi_box.y1:  # 外侧区域在下侧，扩展下边界
                        extend_to = Btm + (Btm - roi_box.y1) * expansion_factor
                        new_roi.y1 = min(H, max(new_roi.y1, extend_to))
                        adjustments_made.append(f"下边界扩展至 {new_roi.y1:.1f} (外侧正点)")

                else:  # action == 'neg'
                    # 外侧负点：这种情况较少，表示外侧区域有干扰，可能需要轻微收缩
                    # 这里我们保持保守，不做大幅调整
                    continue

    # 打印调整信息
    if adjustments_made:
        for adj in adjustments_made:
            print(f"[INFO] ROI调整: {adj}")

    return new_roi


def analyze_mask_expansion(prev_mask: np.ndarray, current_mask: np.ndarray,
                          original_roi: ROIBox) -> dict:
    """分析掩码在各个方向的扩展情况"""
    # 找到新掩码和旧掩码的边界
    prev_coords = np.where(prev_mask > 0)
    current_coords = np.where(current_mask > 0)

    expansion_info = {
        'left': 0, 'right': 0, 'top': 0, 'bottom': 0,
        'has_expansion': False
    }

    if len(prev_coords[0]) == 0 or len(current_coords[0]) == 0:
        return expansion_info

    # 计算边界
    prev_bounds = {
        'top': np.min(prev_coords[0]),
        'bottom': np.max(prev_coords[0]),
        'left': np.min(prev_coords[1]),
        'right': np.max(prev_coords[1])
    }

    current_bounds = {
        'top': np.min(current_coords[0]),
        'bottom': np.max(current_coords[0]),
        'left': np.min(current_coords[1]),
        'right': np.max(current_coords[1])
    }

    # 检测各方向扩展
    expansion_info['top'] = prev_bounds['top'] - current_bounds['top']  # 正值表示向上扩展
    expansion_info['bottom'] = current_bounds['bottom'] - prev_bounds['bottom']  # 正值表示向下扩展
    expansion_info['left'] = prev_bounds['left'] - current_bounds['left']  # 正值表示向左扩展
    expansion_info['right'] = current_bounds['right'] - prev_bounds['right']  # 正值表示向右扩展

    # 判断是否有显著扩展（超过5像素）
    threshold = 5
    expansion_info['has_expansion'] = any(abs(v) > threshold for v in
                                        [expansion_info['top'], expansion_info['bottom'],
                                         expansion_info['left'], expansion_info['right']])

    return expansion_info


def update_roi_based_on_expansion(original_roi: ROIBox, expansion_info: dict,
                                 ratio: float, image_shape: Tuple[int, int]) -> ROIBox:
    """根据掩码扩展情况动态调整ROI"""
    if not expansion_info['has_expansion']:
        return original_roi

    H, W = image_shape

    # 计算调整幅度，ratio控制激进程度
    # ratio越大，调整越激进
    adjustment_factor = ratio * 0.5  # 将ratio转换为合适的调整系数

    # 基于扩展方向调整BBox边界
    new_x0 = original_roi.x0
    new_y0 = original_roi.y0
    new_x1 = original_roi.x1
    new_y1 = original_roi.y1

    # 向左扩展
    if expansion_info['left'] > 0:
        adjust_amount = expansion_info['left'] * adjustment_factor
        new_x0 = max(0, original_roi.x0 - adjust_amount)
        print(f"[INFO] ROI向左扩展: {adjust_amount:.1f}像素")

    # 向右扩展
    if expansion_info['right'] > 0:
        adjust_amount = expansion_info['right'] * adjustment_factor
        new_x1 = min(W, original_roi.x1 + adjust_amount)
        print(f"[INFO] ROI向右扩展: {adjust_amount:.1f}像素")

    # 向上扩展
    if expansion_info['top'] > 0:
        adjust_amount = expansion_info['top'] * adjustment_factor
        new_y0 = max(0, original_roi.y0 - adjust_amount)
        print(f"[INFO] ROI向上扩展: {adjust_amount:.1f}像素")

    # 向下扩展
    if expansion_info['bottom'] > 0:
        adjust_amount = expansion_info['bottom'] * adjustment_factor
        new_y1 = min(H, original_roi.y1 + adjust_amount)
        print(f"[INFO] ROI向下扩展: {adjust_amount:.1f}像素")

    # 对于收缩情况，适度内缩BBox（更保守）
    shrink_factor = adjustment_factor * 0.3  # 收缩时更保守

    if expansion_info['left'] < -10:  # 向右收缩超过10像素
        adjust_amount = abs(expansion_info['left']) * shrink_factor
        new_x0 = min(original_roi.x1 - 50, original_roi.x0 + adjust_amount)  # 确保不会过度收缩
        print(f"[INFO] ROI向右内缩: {adjust_amount:.1f}像素")

    if expansion_info['right'] < -10:  # 向左收缩
        adjust_amount = abs(expansion_info['right']) * shrink_factor
        new_x1 = max(original_roi.x0 + 50, original_roi.x1 - adjust_amount)
        print(f"[INFO] ROI向左内缩: {adjust_amount:.1f}像素")

    if expansion_info['top'] < -10:  # 向下收缩
        adjust_amount = abs(expansion_info['top']) * shrink_factor
        new_y0 = min(original_roi.y1 - 50, original_roi.y0 + adjust_amount)
        print(f"[INFO] ROI向下内缩: {adjust_amount:.1f}像素")

    if expansion_info['bottom'] < -10:  # 向上收缩
        adjust_amount = abs(expansion_info['bottom']) * shrink_factor
        new_y1 = max(original_roi.y0 + 50, original_roi.y1 - adjust_amount)
        print(f"[INFO] ROI向上内缩: {adjust_amount:.1f}像素")

    return ROIBox(new_x0, new_y0, new_x1, new_y1)


def refine_mask_with_local_expansion(predictor: SamPredictor, image: np.ndarray, roi_box: ROIBox,
                                   anchor_edits: List[dict], anchor_bounds_list: List[Tuple],
                                   prev_mask: np.ndarray) -> np.ndarray:
    """使用局部扩展进行精确的SAM分割优化"""
    predictor.set_image(image)

    # 从之前的掩码开始
    result_mask = prev_mask.copy()

    # 为每个锚点的编辑创建局部扩展并分别处理
    for (anchor_point, square_bounds, edits) in anchor_edits:
        if not edits:
            continue

        # 处理该锚点的每个编辑指令
        for edit in edits:
            region_id = int(edit.get('region_id', 0))
            action = str(edit.get('action', ''))

            if region_id not in [1, 2] or action not in ['pos', 'neg']:
                continue

            # 获取区域中心点
            region_center = get_tangent_region_center(square_bounds, region_id)

            # 创建该区域的局部扩展框
            local_box = create_local_expansion_box(anchor_point, region_center, image.shape[:2])

            print(f"[INFO] 锚点局部扩展: 区域{region_id} ({region_center[0]:.1f}, {region_center[1]:.1f}) -> "
                  f"扩展框({local_box[0]}, {local_box[1]}, {local_box[2]}, {local_box[3]})")

            # 在局部区域进行SAM分割
            point_coords = np.array([region_center])
            point_labels = np.array([1 if action == 'pos' else 0])

            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.array(local_box),
                multimask_output=True,
            )

            # 选择最佳掩码
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]

            # 将局部结果融合到主掩码
            x0, y0, x1, y1 = local_box
            # SAM返回的是全图掩码，需要提取局部区域
            local_region = (best_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

            if action == 'pos':
                # 正点：添加到掩码
                result_mask[y0:y1, x0:x1] = np.maximum(result_mask[y0:y1, x0:x1], local_region)
            else:
                # 负点：从掩码中移除
                result_mask[y0:y1, x0:x1] = np.where(local_region > 0, 0, result_mask[y0:y1, x0:x1])

    return result_mask


def refine_mask_with_points_expanded(predictor: SamPredictor, image: np.ndarray, roi_box: ROIBox,
                                   pos_points: List[Tuple[float, float]],
                                   neg_points: List[Tuple[float, float]],
                                   prev_mask: np.ndarray,
                                   expansion_ratio: float = 0.2) -> np.ndarray:
    """步骤4: 使用正负点进行SAM分割优化（扩展版本，允许在ROI外分割）"""
    predictor.set_image(image)

    # 组合所有点和标签
    all_points = pos_points + neg_points
    all_labels = [1] * len(pos_points) + [0] * len(neg_points)

    if not all_points:
        return prev_mask

    # 创建扩展的ROI区域用于SAM分割
    expanded_roi = create_expanded_roi(roi_box, expansion_ratio, image.shape[:2])

    print(f"[INFO] 扩展ROI: 原始({roi_box.x0:.1f}, {roi_box.y0:.1f}, {roi_box.x1:.1f}, {roi_box.y1:.1f}) -> "
          f"扩展({expanded_roi.x0:.1f}, {expanded_roi.y0:.1f}, {expanded_roi.x1:.1f}, {expanded_roi.y1:.1f})")

    masks, scores, logits = predictor.predict(
        point_coords=np.array(all_points),
        point_labels=np.array(all_labels),
        box=np.array([expanded_roi.x0, expanded_roi.y0, expanded_roi.x1, expanded_roi.y1]),
        multimask_output=True,
    )

    # 选择最佳掩码
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # 创建最终掩码，优先保留原始ROI内的结果，但允许适度扩展
    H, W = image.shape[:2]
    final_mask = np.zeros((H, W), dtype=np.uint8)

    # 将SAM结果应用到扩展区域
    ex0, ey0, ex1, ey1 = int(expanded_roi.x0), int(expanded_roi.y0), int(expanded_roi.x1), int(expanded_roi.y1)
    expanded_mask_region = (best_mask[ey0:ey1, ex0:ex1] > 0).astype(np.uint8) * 255
    final_mask[ey0:ey1, ex0:ex1] = expanded_mask_region

    # 获取原始ROI区域的结果
    ox0, oy0, ox1, oy1 = int(roi_box.x0), int(roi_box.y0), int(roi_box.x1), int(roi_box.y1)
    roi_mask_region = final_mask[oy0:oy1, ox0:ox1]

    # 如果扩展区域的像素数相比原始区域增加太多，进行适度约束
    original_pixels = np.sum(prev_mask > 0)
    new_pixels = np.sum(final_mask > 0)
    expansion_factor = new_pixels / max(1, original_pixels)

    # 如果扩展过度（超过3倍），回退到更保守的策略
    if expansion_factor > 3.0:
        print(f"[WARN] 分割扩展过度 ({expansion_factor:.2f}x)，使用保守策略")
        # 保守策略：主要使用原始ROI内的结果，只在边缘稍微扩展
        conservative_mask = np.zeros((H, W), dtype=np.uint8)
        conservative_mask[oy0:oy1, ox0:ox1] = roi_mask_region

        # 在原始ROI边界附近允许少量扩展
        margin = 20  # 像素
        extended_x0 = max(0, ox0 - margin)
        extended_y0 = max(0, oy0 - margin)
        extended_x1 = min(W, ox1 + margin)
        extended_y1 = min(H, oy1 + margin)

        # 只在边缘区域应用扩展结果
        edge_mask = final_mask[extended_y0:extended_y1, extended_x0:extended_x1]
        conservative_mask[extended_y0:extended_y1, extended_x0:extended_x1] = np.maximum(
            conservative_mask[extended_y0:extended_y1, extended_x0:extended_x1],
            edge_mask
        )
        final_mask = conservative_mask

    return final_mask


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

        # 收集锚点编辑信息用于局部扩展
        anchor_edits = []
        # 使用基于连线交点的锚点
        projected_anchor_points = get_line_based_anchor_points(roi_box, current_mask)

        for anchor_info in selected_anchors:
            anchor_id = int(anchor_info.get('id', 0))
            if not (1 <= anchor_id <= 8):
                continue

            # 使用贴边投影后的锚点位置
            anchor_point = projected_anchor_points[anchor_id - 1]

            # 步骤3: 创建切线内外区域可视化
            print(f"步骤3: 为锚点 {anchor_id} 创建切线内外区域...")
            quad_vis, square_bounds = create_tangent_square_visualization(
                image, anchor_point, roi_box, current_mask, anchor_id, ratio=float(args.ratio)
            )
            save_image(os.path.join(output_dir, f'step3_round{round_idx + 1}_anchor{anchor_id}_tangent_square.png'), quad_vis)

            # 传给VLM前缩放
            quad_vis_vlm = _resize_for_vlm(quad_vis, int(args.vlm_max_side))
            quad_response = vlm.quadrant_edits(quad_vis_vlm, instance_name, anchor_id)
            edits = quad_response.get('edits', [])

            print(f"锚点 {anchor_id} 的编辑指令: {edits}")

            # 收集锚点信息用于局部扩展
            anchor_edits.append((anchor_point, square_bounds, edits))
        
        # 卸载VLM模型以释政GPU内存供SAM使用
        print("[INFO] 卸载VLM模型释放内存...")
        vlm.unload_model()

        # 步骤4: 使用局部扩展进行精确SAM分割
        if anchor_edits:
            total_edits = sum(len(edits) for _, _, edits in anchor_edits)
            print(f"步骤4: 使用局部扩展方法处理 {total_edits} 个编辑区域...")
            # 清理GPU缓存以释放内存
            import torch
            torch.cuda.empty_cache()

            # 保存分割前的掩码用于对比
            prev_mask = current_mask.copy()

            current_mask = refine_mask_with_local_expansion(predictor, image, roi_box,
                                                          anchor_edits, [], current_mask)
            save_image(os.path.join(output_dir, f'step4_round{round_idx + 1}_refined_mask.png'), current_mask)
            print(f"优化后掩码像素数: {np.sum(current_mask > 0)}")

            # 步骤5: 根据VLM动作直接调整ROI
            new_roi_box = update_roi_based_on_square_actions(roi_box, anchor_edits, float(args.ratio), image.shape[:2])

            if (abs(new_roi_box.x0 - roi_box.x0) > 1 or abs(new_roi_box.y0 - roi_box.y0) > 1 or
                abs(new_roi_box.x1 - roi_box.x1) > 1 or abs(new_roi_box.y1 - roi_box.y1) > 1):
                print(f"[INFO] ROI更新: ({roi_box.x0:.1f}, {roi_box.y0:.1f}, {roi_box.x1:.1f}, {roi_box.y1:.1f}) -> "
                      f"({new_roi_box.x0:.1f}, {new_roi_box.y0:.1f}, {new_roi_box.x1:.1f}, {new_roi_box.y1:.1f})")
                roi_box = new_roi_box
            else:
                print("[INFO] ROI无需调整")
        else:
            print("未生成任何编辑指令，跳过本轮优化")

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
