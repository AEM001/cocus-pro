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

from sculptor.vlm.qwen_api import QwenAPIVLM


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


def load_sam_model(checkpoint_path: str = "models/sam_vit_h_4b8939.pth", device: str = "cuda"):
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
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}，请检查文件路径和完整性")
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
    """步骤1: 使用SAM生成初始掩码 (仅使用ROI框约束)"""
    predictor.set_image(image)

    # 仅使用ROI框作为约束，不使用点提示
    masks, scores, logits = predictor.predict(
        box=np.array([roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1]),
        multimask_output=False,  # 仅生成单个掩码
    )

    # 直接使用返回的掩码（因为multimask_output=False，只返回一个掩码）
    mask = masks[0]

    # 裁剪到ROI区域
    H, W = image.shape[:2]
    x0, y0, x1, y1 = int(roi_box.x0), int(roi_box.y0), int(roi_box.x1), int(roi_box.y1)
    clipped_mask = np.zeros((H, W), dtype=np.uint8)
    clipped_mask[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

    print(f"初始SAM掩码生成完成（仅ROI框约束），像素数: {np.sum(clipped_mask > 0)}")
    return clipped_mask



def calculate_smart_roi_expansion(roi_box: ROIBox, pos_points: List[Tuple[float, float]], 
                                   square_size: float, image_shape: Tuple[int, int]) -> ROIBox:
    """根据正点智能扩展ROI，以包含越界的正点。"""
    if not pos_points:
        return roi_box
    H, W = image_shape
    nx0, ny0, nx1, ny1 = roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1
    for px, py in pos_points:
        if px < nx0:
            nx0 = max(0, min(nx0, px - square_size / 2))
        if px > nx1:
            nx1 = min(W, max(nx1, px + square_size / 2))
        if py < ny0:
            ny0 = max(0, min(ny0, py - square_size / 2))
        if py > ny1:
            ny1 = min(H, max(ny1, py + square_size / 2))
    return ROIBox(nx0, ny0, nx1, ny1)


def refine_mask_with_points(predictor: SamPredictor, image: np.ndarray, roi_box: ROIBox,
                          pos_points: List[Tuple[float, float]],
                          neg_points: List[Tuple[float, float]],
                          prev_mask: np.ndarray,
                          square_size: float = 0.0) -> Tuple[np.ndarray, ROIBox]:
    """步骤4: 使用正负点进行SAM分割优化，支持智能BBox扩展
    
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

    return clipped_mask, smart_roi  # 返回扩展后的ROI供下一轮使用


def refine_mask_with_points_limited(predictor: SamPredictor, image: np.ndarray, roi_box: ROIBox,
                                   pos_points: List[Tuple[float, float]],
                                   neg_points: List[Tuple[float, float]],
                                   prev_mask: np.ndarray) -> Tuple[np.ndarray, ROIBox]:
    """使用正负点进行SAM分割优化，严格限制在指定ROI框内
    
    返回: (refined_mask, roi_box)
    """
    predictor.set_image(image)
    
    # 组合所有点和标签
    all_points = pos_points + neg_points
    all_labels = [1] * len(pos_points) + [0] * len(neg_points)
    
    if not all_points:
        return prev_mask, roi_box  # 没有点时返回原始掩码和ROI
    
    print(f"[SAM限制] 使用扩展ROI框: ({roi_box.x0:.1f}, {roi_box.y0:.1f}, {roi_box.x1:.1f}, {roi_box.y1:.1f})")
    
    masks, scores, logits = predictor.predict(
        point_coords=np.array(all_points),
        point_labels=np.array(all_labels),
        box=np.array([roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1]),
        multimask_output=True,
    )
    
    # 选择最佳掩码
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    
    # 严格限制在ROI框内进行裁剪
    H, W = image.shape[:2]
    x0, y0, x1, y1 = int(roi_box.x0), int(roi_box.y0), int(roi_box.x1), int(roi_box.y1)
    
    # 初始化为全零掩码
    clipped_mask = np.zeros((H, W), dtype=np.uint8)
    
    # 只在ROI框内区域应用掩码
    clipped_mask[y0:y1, x0:x1] = (best_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    
    return clipped_mask, roi_box


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


    """当VLM未返回任何点时，从当前掩码边界密集采样，生成精雕点。
    策略：
    - 使用形态学梯度得到边界带 band = dilate(mask)-erode(mask)
    - 正点：从 band 内且 mask==1 的像素均匀抽样
    - 负点：从 band 内且 mask==0 的像素均匀抽样
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        h, w = m.shape
        # 退化：均匀网格取点
        xs = np.linspace(0, w - 1, num=max_total//2 + max_total%2, dtype=int)
        ys = np.linspace(0, h - 1, num=max_total//2 + 1, dtype=int)
        pos = [(int(x), int(h//2)) for x in xs[:max_total//2]]
        neg = [(int(w//2), int(y)) for y in ys[:max_total - len(pos)]]
        return pos, neg
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(m, kernel)
    ero = cv2.erode(m, kernel)
    band = ((dil - ero) > 0).astype(np.uint8)

    pos_cands = np.column_stack(np.where((band > 0) & (m > 0)))  # (y,x)
    neg_cands = np.column_stack(np.where((band > 0) & (m == 0)))

    def sample_evenly(cands: np.ndarray, k: int) -> list[tuple[int,int]]:
        if cands.size == 0 or k <= 0:
            return []
        idxs = np.linspace(0, len(cands) - 1, num=min(k, len(cands)), dtype=int)
        pts = []
        for i in idxs:
            y, x = map(int, cands[i])
            pts.append((x, y))
        return pts

    half = max_total // 2
    pos = sample_evenly(pos_cands, half)
    neg = sample_evenly(neg_cands, max_total - len(pos))

    # 如果仍不足，随机补齐
    rng = np.random.default_rng(42)
    while len(pos) + len(neg) < max_total:
        if len(pos) < half and pos_cands.size > 0:
            i = int(rng.integers(0, len(pos_cands)))
            y, x = map(int, pos_cands[i])
            if (x, y) not in pos:
                pos.append((x, y))
        elif neg_cands.size > 0:
            i = int(rng.integers(0, len(neg_cands)))
            y, x = map(int, neg_cands[i])
            if (x, y) not in neg and (x, y) not in pos:
                neg.append((x, y))
        else:
            break
    return pos[:half], neg[: (max_total - len(pos))]


def main():
    import argparse
    ap = argparse.ArgumentParser(description='SAM+Qwen refinement with single-anchor optimization')
    ap.add_argument('--name', default='f', help='sample name (e.g., f, dog, q)')
    ap.add_argument('--rounds', type=int, default=1, help='refinement rounds')
    ap.add_argument('--ratio', type=float, default=0.6, help='square ratio to ROI short side for tangent square crop')
    ap.add_argument('--vlm_max_side', type=int, default=720, help='resize long side before sending to VLM (<=0 to disable)')
    ap.add_argument('--first_round_apply_all', action='store_true', default=True, help='In round 1, apply ALL returned anchors (batch) by generating points for each and updating mask once')
    ap.add_argument('--second_round_apply_all', action='store_true', help='In round 2, apply ALL returned anchors (batch) similar to round 1')
    ap.add_argument('--api-key', help='API key for Qwen API (default: from DASHSCOPE_API_KEY env var)')
    ap.add_argument('--api-model', default='qwen-vl-plus-latest', help='API model name')
    ap.add_argument('--use-openai-api', action='store_true', help='Use OpenAI compatible API mode')
    ap.add_argument('--high-resolution', action='store_true', help='Enable high resolution mode for API')
    ap.add_argument('--clean-output', action='store_true', help='Only keep final mask and instance info, remove intermediate files')
    ap.add_argument('--save-points-vis', action='store_true', help='Save overlay images with VLM pos/neg points on context image per round')
    ap.add_argument('--output-format', choices=['png', 'jpg'], default='png', help='Output mask format')
    args = ap.parse_args()

    sample_name = args.name
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 输入文件路径
    image_path = os.path.join(base_dir, 'dataset', 'COD10K_TEST_DIR', 'Imgs', f'{sample_name}.jpg')
    roi_json_path = os.path.join(base_dir, 'auxiliary', 'box_out', sample_name, f'{sample_name}_sam_boxes.json')

    # 加载语义信息（实例名称和推理原因）
    llm_out_path = os.path.join(base_dir, 'auxiliary', 'llm_out', f'{sample_name}_output.json')
    instance_name = sample_name
    semantic_reason = None
    if os.path.exists(llm_out_path):
        try:
            with open(llm_out_path, 'r') as f:
                llm_data = json.load(f)
                instance_name = llm_data.get('instance', sample_name)
                semantic_reason = llm_data.get('reason', None)
                print(f"[语义信息] 加载实例: {instance_name}")
                if semantic_reason:
                    print(f"[语义信息] 推理描述: {semantic_reason[:100]}...")
        except Exception as e:
            print(f"[WARN] 无法加载语义信息: {e}")
            pass

    # 输出目录
    output_dir = os.path.join(base_dir, 'outputs', 'clean_sculpt', sample_name)

    print(f"=== 清理版SAM分割流程 (样本: {sample_name}, 实例: {instance_name}) ===")

    # 加载数据
    print("加载输入数据...")
    image = load_image(image_path)
    roi_box = load_roi_box(roi_json_path)
    print(f"ROI框: ({roi_box.x0}, {roi_box.y0}, {roi_box.x1}, {roi_box.y1})")

    # 加载SAM模型
    print("加载SAM模型...")
    predictor = load_sam_model()

    # 初始化VLM (仅API模式)
    print("初始化VLM (Qwen API)...")
    try:
        vlm = QwenAPIVLM(
            api_key=args.api_key,
            model_name=args.api_model,
            use_dashscope=not args.use_openai_api,
            high_resolution=args.high_resolution,
            max_tokens=512,  # 增加到512防止截断
            temperature=0.0
        )
    except Exception as e:
        print(f"[ERROR] API初始化失败: {e}")
        raise RuntimeError(f"API初始化失败，请检查API密钥和网络连接: {e}")

    # 步骤1: 生成初始SAM掩码
    print("步骤1: 生成初始SAM掩码...")
    current_mask = generate_initial_sam_mask(predictor, image, roi_box)
    save_image(os.path.join(output_dir, 'round0_step1_initial_mask.png'), current_mask)
    
    # 扩展ROI框1.1倍用于VLM分析和SAM分割限制
    from src.sculptor.utils import expand_roi_box
    img_h, img_w = image.shape[:2]
    expanded_roi = expand_roi_box(roi_box.x0, roi_box.y0, roi_box.x1, roi_box.y1, 1.1, img_w, img_h)
    print(f"扩展ROI框: ({expanded_roi[0]:.1f}, {expanded_roi[1]:.1f}, {expanded_roi[2]:.1f}, {expanded_roi[3]:.1f})")

    # 新流程：基于VLM直接输出正/负点，单轮或多轮迭代
    max_rounds = max(1, int(args.rounds))
    for round_idx in range(max_rounds):
        print(f"\n=== VLM直接点分割 - 第 {round_idx + 1} 轮 ===")

        # 构建上下文图：原图 + 绿色掩码覆盖 + 红色扩展ROI框
        from src.sculptor.utils import overlay_mask_on_image
        context = overlay_mask_on_image(image, current_mask, color=(0, 255, 0), alpha=0.3)
        cv2.rectangle(context, (int(expanded_roi[0]), int(expanded_roi[1])), (int(expanded_roi[2]), int(expanded_roi[3])), (255, 0, 0), 3)
        
        # 保存传递给VLM的上下文图像
        save_image(os.path.join(output_dir, f'round{round_idx + 1}_context_for_vlm.png'), context)

        # 每轮请求固定数量的点：总计 10 个点（优先正点，若存在明显泄漏则包含负点）

        # 请求VLM输出点
        print("步骤2: 请求VLM输出正/负点...")
        points_resp = vlm.propose_points(context, instance_name, max_total=10)
        raw_pos_points = points_resp.get("pos_points", [])
        raw_neg_points = points_resp.get("neg_points", [])
        
        # 过滤掉扩展框外的点位
        def filter_points_in_roi(points, roi):
            filtered = []
            for x, y in points:
                if roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]:
                    filtered.append((x, y))
            return filtered
        
        pos_points = filter_points_in_roi(raw_pos_points, expanded_roi)
        neg_points = filter_points_in_roi(raw_neg_points, expanded_roi)
        
        filtered_pos = len(raw_pos_points) - len(pos_points)
        filtered_neg = len(raw_neg_points) - len(neg_points)
        if filtered_pos > 0 or filtered_neg > 0:
            print(f"[过滤] 框外点位: pos={filtered_pos}, neg={filtered_neg}")
        
        total_pts = len(pos_points) + len(neg_points)
        print(f"VLM返回点数(过滤后): pos={len(pos_points)}, neg={len(neg_points)}, total={total_pts}")
        if total_pts == 0:
            print("[INFO] 本轮未得到任何点，认为已足够或无明显可优化边界，跳过本轮。")
            break

        if not pos_points and not neg_points:
            print("[WARN] 本轮未得到任何点，提前结束迭代")
            break

        # 估计小正方形尺寸用于智能ROI扩展（用ratio与ROI短边）
        roi_w = roi_box.x1 - roi_box.x0
        roi_h = roi_box.y1 - roi_box.y0
        square_size = max(80.0, float(args.ratio) * float(min(roi_w, roi_h)))

        # 可视化点（可选）
        if args.save_points_vis:
            vis = context.copy()
            # draw pos as red, neg as blue
            for (x, y) in pos_points:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            for (x, y) in neg_points:
                cv2.circle(vis, (int(x), int(y)), 5, (255, 0, 0), -1)
            save_image(os.path.join(output_dir, f'round{round_idx + 1}_points_overlay.png'), vis)

        # 执行一次SAM精修（使用扩展ROI框限制）
        print("步骤3: 使用VLM点进行SAM分割...")
        import torch
        torch.cuda.empty_cache()
        
        # 使用扩展ROI框进行SAM分割限制
        expanded_roi_box = ROIBox(expanded_roi[0], expanded_roi[1], expanded_roi[2], expanded_roi[3])
        current_mask, _ = refine_mask_with_points_limited(
            predictor, image, expanded_roi_box,
            pos_points, neg_points, current_mask
        )
        print(f"当前掩码像素数: {int(np.sum(current_mask > 0))}")

    print("[INFO] 直接点流程完成")

    # 保存最终结果
    final_mask_name = f'final_mask.{args.output_format}'
    save_image(os.path.join(output_dir, final_mask_name), current_mask)

    # 保存instance信息到txt文件（仅包含instance）
    instance_info_path = os.path.join(output_dir, 'instance_info.txt')
    with open(instance_info_path, 'w', encoding='utf-8') as f:
        f.write(instance_name)
    
    # 如果启用清理模式，删除中间文件
    if args.clean_output:
        import glob
        # 保留的文件
        keep_files = {final_mask_name, 'instance_info.txt'}
        
        # 删除其他文件
        all_files = glob.glob(os.path.join(output_dir, '*'))
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # 保留 points overlay 如果启用
            if args.save_points_vis and 'points_overlay' in filename:
                continue
            if filename not in keep_files:
                try:
                    os.remove(file_path)
                    print(f"[CLEAN] 已删除: {filename}")
                except Exception as e:
                    print(f"[WARN] 无法删除 {filename}: {e}")
    else:
        # 创建最终可视化（仅在非清理模式下）
        final_vis = image.copy()
        final_vis[current_mask > 0] = final_vis[current_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
        save_image(os.path.join(output_dir, 'final_visualization.png'), final_vis)

    print("\n=== 清理版重构完成! ===")
    print(f"最终掩码像素数: {np.sum(current_mask > 0)}")
    print(f"输出目录: {output_dir}")
    if args.clean_output:
        print(f"[CLEAN] 输出已清理，仅保留: {final_mask_name}, instance_info.txt")


if __name__ == "__main__":
    main()