"""
实现“语义雕刻 + SAM”的核心流程，包括网格划分、
区域语义评分、正负点选择、SAM 交互细化与迭代早停。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image
import cv2


# ============================
# Config & Types
# ============================


@dataclass
class Config:
    """管线配置项

    - 网格与细分：控制初始网格大小、最小单元、最大细分深度
    - 迭代策略：控制每轮点数、正负比例、早停阈值、边界带宽
    - 提示词模板：区域语义评分所用的文本模板（支持 format(instance=...)）
    - 数据与输出：输入路径与输出目录，以及是否使用先验/box
    """
    v_segments: int = 9
    h_segments: int = 9

    bbox_margin: float = 0.2
    grid_init: Tuple[int, int] = (3, 3)

    min_cell: int = 32
    max_depth: int = 2

    k_iters: int = 3
    max_points_per_iter: int = 12
    pos_neg_ratio: float = 2.0
    iou_eps: float = 5e-3
    boundary_band: int = 10

    # 评分与选点行为
    min_sigma_for_neg: float = 0.005  # 当分数方差过小（几乎无区分度）时，禁用负点

    # 点提示的 ROI 约束（仅允许在初始 ROI 的轻度扩张区域内生成点）
    roi_point_expand_scale: float = 1.2
    gate_mask_outside_roi: bool = True  # 将最终掩码与 ROI 扩张区域做硬门控

    # 首轮/细粒度策略
    gentle_first_iter: bool = True
    first_iter_max_points: int = 4
    first_iter_disable_subdivide: bool = True
    topk_pos_first_iter: int = 3
    small_roi_ratio: float = 0.06
    small_roi_max_points: int = 6
    fine_after_first: bool = True
    fine_grid: Tuple[int, int] = (6, 6)
    fine_min_cell: int = 12
    fine_max_depth: int = 3

    # 边界精修模式（仅在边界带生成正负点，不进行全局网格打分）
    boundary_refine_only: bool = True
    skip_update_first_iter: bool = True
    boundary_samples: int = 24
    boundary_inner_band: int = 6
    boundary_alpha_radius_in: int = 8
    boundary_alpha_radius_out: int = 8

    prompt_template: str = "a photo of the {instance} camouflaged in the background."

    image_path: Optional[str] = None
    meta_path: Optional[str] = None
    boxes_json_path: Optional[str] = None
    prior_mask_path: Optional[str] = None
    out_root: str = "sculpt_out"

    use_prior_mask_as_M0: bool = True
    use_boxes_for_roi: bool = True

    # 点生成策略：semantic (Alpha-CLIP) | edge (边缘感知) | hybrid (预留)
    point_mode: str = "semantic"


@dataclass
class Box:
    """行列坐标系下的矩形框，闭开区间 [r0:r1, c0:c1]"""
    r0: int
    c0: int
    r1: int
    c1: int


@dataclass
class Point:
    """用于 SAM 交互的点提示

    - y, x: 点的像素坐标（行, 列）
    - is_positive: True 表示正点，False 表示负点
    - score: 该点来自的单元格语义分数（用于调试或加权）
    """
    y: int
    x: int
    is_positive: bool
    score: float = 1.0


@dataclass
class Cell:
    """网格单元

    - box: 单元格在整图中的位置
    - depth: 细分深度（0 为初始层）
    - score: 该单元的语义评分（None 表示尚未评分）
    """
    box: Box
    depth: int = 0
    score: Optional[float] = None


# ============================
# IO utils
# ============================


def load_image(path: str) -> np.ndarray:
    """读取图像为 RGB numpy 数组 [H, W, 3]"""
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask_png(path: str) -> np.ndarray:
    """读取灰度 PNG 掩码，并二值化为 {0,1} 的 uint8 数组"""
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def save_mask_png(mask01: np.ndarray, path: str) -> None:
    """将 {0,1} 掩码保存为 PNG（0/255）"""
    Image.fromarray((mask01.astype(np.uint8) * 255)).save(path)


def ensure_dir(path: str) -> None:
    """确保目录存在（若不存在则创建）"""
    import os

    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    """读取 JSON 文件为字典"""
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_instance_text_from_llm_out(name: str, llm_out_root: str = "auxiliary/llm_out") -> Optional[str]:
    """从 llm_out 目录中读取对应名字的 JSON 文件中的 instance 字段
    
    参数:
    - name: 图像名称（如 'q'）
    - llm_out_root: llm_out 目录路径，默认为 "auxiliary/llm_out"
    
    返回:
    - instance 文本内容，若文件不存在或字段缺失则返回 None
    """
    import os
    
    json_path = os.path.join(llm_out_root, f"{name}_output.json")
    
    if not os.path.isfile(json_path):
        return None
        
    try:
        data = read_json(json_path)
        return data.get("instance")
    except Exception as e:
        print(f"Warning: Failed to read instance text from {json_path}: {e}")
        return None


def maybe_read_boxes(path: Optional[str]) -> Optional[np.ndarray]:
    """读取候选框 JSON（若路径无效或无 boxes 字段则返回 None）

    期望 JSON 结构：{"boxes": [[x1,y1,x2,y2], ...]}
    返回：np.ndarray[int32] 或 None
    """
    import os

    if not path or not os.path.isfile(path):
        return None
    data = read_json(path)
    boxes = data.get("boxes")
    if boxes is None:
        return None
    return np.asarray(boxes, dtype=np.int32)


def bbox_from_mask(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """从二值掩码中估计紧致外接框，返回 (x_min, y_min, x_max, y_max)

    若掩码为空返回 None
    """
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay(image: np.ndarray, mask01: np.ndarray, color=(0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """将掩码渲染到图像上（半透明叠加）"""
    out = image.copy().astype(np.uint8)
    m = (mask01 > 0)
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def save_image(arr: np.ndarray, path: str) -> None:
    """保存 numpy 数组为图像文件"""
    Image.fromarray(arr).save(path)


# ============================
# ROI / Grid / Alpha
# ============================


def roi_to_bbox(ids_v, ids_h, H: int, W: int, v_segments: int, h_segments: int, margin: float) -> Box:
    """从网格索引集恢复 ROI 框，并按 margin 扩张一定比例"""
    r0 = int((min(ids_v) - 1) * (H / v_segments))
    r1 = int(max(ids_v) * (H / v_segments))
    c0 = int((min(ids_h) - 1) * (W / h_segments))
    c1 = int(max(ids_h) * (W / h_segments))
    rh, rw = r1 - r0, c1 - c0
    pr, pc = int(margin * rh), int(margin * rw)
    R0, R1 = max(0, r0 - pr), min(H, r1 + pr)
    C0, C1 = max(0, c0 - pc), min(W, c1 + pc)
    return Box(R0, C0, R1, C1)


def box_from_xyxy_clip(y1: int, x1: int, y2: int, x2: int, H: int, W: int) -> Box:
    """将任意 (y1,x1,y2,x2) 裁剪到图像范围内并确保顺序合法"""
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 < x1:
        x1, x2 = x2, x1
    return Box(y1, x1, y2, x2)


def expand_box_scale(b: Box, scale: float, H: int, W: int) -> Box:
    """按比例扩张 Box，保持中心不变，并裁剪到图像内"""
    cy = (b.r0 + b.r1) / 2.0
    cx = (b.c0 + b.c1) / 2.0
    hh = (b.r1 - b.r0) * scale / 2.0
    hw = (b.c1 - b.c0) * scale / 2.0
    r0 = int(max(0, np.floor(cy - hh)))
    r1 = int(min(H, np.ceil(cy + hh)))
    c0 = int(max(0, np.floor(cx - hw)))
    c1 = int(min(W, np.ceil(cx + hw)))
    return Box(r0, c0, r1, c1)


def partition_grid(bbox: Box, grid: Tuple[int, int]) -> List[Cell]:
    """将 bbox 均匀划分为 grid=(gh,gw) 个单元格，返回 Cell 列表（初始 depth=0）"""
    r0, c0, r1, c1 = bbox.r0, bbox.c0, bbox.r1, bbox.c1
    gh, gw = grid
    H = r1 - r0; W = c1 - c0
    cells: List[Cell] = []
    for i in range(gh):
        for j in range(gw):
            rr0 = r0 + (i * H) // gh
            rr1 = r0 + ((i + 1) * H) // gh
            cc0 = c0 + (j * W) // gw
            cc1 = c0 + ((j + 1) * W) // gw
            cells.append(Cell(Box(rr0, cc0, rr1, cc1), depth=0))
    return cells


def subdivide_cell(cell: Cell) -> List[Cell]:
    """将单元格四分细化，子格 depth = 父格 depth + 1"""
    b = cell.box
    rm = (b.r0 + b.r1) // 2
    cm = (b.c0 + b.c1) // 2
    return [
        Cell(Box(b.r0, b.c0, rm, cm), depth=cell.depth + 1),
        Cell(Box(b.r0, cm, rm, b.c1), depth=cell.depth + 1),
        Cell(Box(rm, b.c0, b.r1, cm), depth=cell.depth + 1),
        Cell(Box(rm, cm, b.r1, b.c1), depth=cell.depth + 1),
    ]


def build_alpha_for_cell(M: np.ndarray, cell: Cell) -> np.ndarray:
    """从整体掩码 M 中截取当前 Cell 区域，构造局部 alpha（float32, 与 M 同形状）"""
    alpha = np.zeros_like(M, dtype=np.float32)
    b = cell.box
    alpha[b.r0:b.r1, b.c0:b.c1] = M[b.r0:b.r1, b.c0:b.c1]
    return alpha


def compose_rgba(image: np.ndarray, alpha01: np.ndarray) -> np.ndarray:
    """将 RGB 图与 {0,1} 或 [0,255] 掩码合并为 RGBA（用于可视化/调试）"""
    a = (alpha01 * 255).astype(np.uint8) if alpha01.dtype != np.uint8 else alpha01
    return np.dstack([image, a])


# ============================
# Cell utils
# ============================


def _cell_size(cell: Cell) -> Tuple[int, int]:
    """返回单元格的 (高, 宽)"""
    b = cell.box
    return (b.r1 - b.r0, b.c1 - b.c0)


# ============================
# Scorer
# ============================


class RegionScorer:
    """区域评分抽象接口：给定图像与区域 alpha，输出该区域对给定文本的语义分数"""
    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        raise NotImplementedError


class AlphaCLIPScorer(RegionScorer):
    """使用 Alpha-CLIP 进行区域语义评分（支持多模板提示并聚合）"""
    def __init__(self, model) -> None:
        self.model = model

    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        """使用 Alpha-CLIP 进行区域语义评分

        - image: RGB 图像
        - alpha01: 当前单元对应的软掩码（0/1 或 [0,1]）
        - text: 裸 instance 文本（例如 "scorpionfish"），内部会做多模板拼接
        
        策略：返回“聚焦评分 - 无聚焦评分”的相对分数，增强区分度并抑制同质背景。
        """
        try:
            # 聚焦评分（子区域 alpha）
            s_focus = self.model.score_region_with_templates(
                image, alpha01, text,
                templates=[
                    "a photo of the {} camouflaged in the background.",
                    "a photo of the {}.",
                    "the {} blending into surroundings."
                ],
                aggregate='mean'
            )
            # 无聚焦评分（全零 alpha 作为背景基线）
            if alpha01 is None:
                s_bg = 0.0
            else:
                s_bg = self.model.score_region_with_templates(
                    image, np.zeros_like(alpha01, dtype=np.float32), text,
                    templates=[
                        "a photo of the {} camouflaged in the background.",
                        "a photo of the {}.",
                        "the {} blending into surroundings."
                    ],
                    aggregate='mean'
                )
            return float(s_focus - s_bg)
        except Exception as e:
            print(f"Warning: Alpha-CLIP scoring failed: {e}")
            return 0.0


class SoftFocusCLIPScorer(RegionScorer):
    """不使用 Alpha-CLIP 的替代评分器：对图像进行软加权后走标准 RGB 编码"""
    def __init__(self, model) -> None:
        self.model = model

    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        """使用软聚焦方法进行区域评分（非 Alpha-CLIP 的替代方案）"""
        try:
            # 软聚焦: I' = I * (0.5 + 0.5 * alpha)
            soft_focused = (image.astype(np.float32) * (0.5 + 0.5 * alpha01[..., None])).astype(np.uint8)
            
            # 使用标准RGB编码
            score = self.model.compute_similarity(
                soft_focused, 
                f"a photo of the {text} camouflaged in the background."
            )
            return score
        except Exception as e:
            print(f"Warning: Soft focus CLIP scoring failed: {e}")
            return 0.0


class DensityScorer(RegionScorer):
    """简单基线：以区域面积占比作为分数"""
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        """返回 alpha 区域像素占比（与文本无关）"""
        s = float((alpha01 > 0).sum())
        area = float(alpha01.size) + self.eps
        return s / area


class EdgeAwareScorer(RegionScorer):
    """边缘感知评分器：使用多源边缘融合衡量候选区域的“边缘吻合度”。"""
    def __init__(self, image_rgb: np.ndarray) -> None:
        self.image = image_rgb
        self.edges = self._compute_multi_source_edges(image_rgb)

    @staticmethod
    def _compute_multi_source_edges(image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        H, W = gray.shape
        # 1) Canny
        canny = cv2.Canny(gray, 50, 150).astype(np.float32)
        # 2) Sobel 强度
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobelx, sobely)
        if (sobel_mag.max() or 0) > 0:
            sobel_mag = sobel_mag / (sobel_mag.max() + 1e-6) * 255.0
        # 3) Lab 颜色边缘
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        lab_edges = np.zeros_like(gray, dtype=np.float32)
        for i in range(3):
            gx = cv2.Sobel(lab[:, :, i], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lab[:, :, i], cv2.CV_32F, 0, 1, ksize=3)
            gm = cv2.magnitude(gx, gy)
            if (gm.max() or 0) > 0:
                gm = gm / (gm.max() + 1e-6) * 255.0
            lab_edges += gm
        if (lab_edges.max() or 0) > 0:
            lab_edges = lab_edges / (lab_edges.max() + 1e-6) * 255.0
        # 融合
        combined = 0.4 * canny + 0.4 * sobel_mag + 0.2 * lab_edges
        return combined.astype(np.uint8)

    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str = "") -> float:
        # 边缘密度
        a = (alpha01 > 0).astype(np.float32)
        denom = float(a.sum()) + 1e-6
        edge_density = float((self.edges.astype(np.float32) * a).sum() / denom)
        # 边缘连续性：在候选区域边界上统计边缘连通度
        boundary = self._get_boundary_mask(a)
        continuity = self._edge_continuity(boundary)
        # 归一化维持同一量纲（粗略缩放）
        ed_norm = edge_density / 255.0
        return 0.7 * ed_norm + 0.3 * continuity

    @staticmethod
    def _get_boundary_mask(alpha01: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha01.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(alpha01.astype(np.uint8), kernel, iterations=1)
        boundary = (dilated - eroded) > 0
        return boundary.astype(np.uint8)

    def _edge_continuity(self, boundary01: np.ndarray) -> float:
        if boundary01.sum() == 0:
            return 0.0
        boundary_edges = ((self.edges > 0).astype(np.uint8) & (boundary01 > 0).astype(np.uint8)).astype(np.uint8)
        edge_pixels = int(boundary_edges.sum())
        if edge_pixels == 0:
            return 0.0
        # 连通域比例（最大连通块像素占所有边缘像素的比例）
        num_labels, labels = cv2.connectedComponents(boundary_edges)
        largest = 0
        for i in range(1, num_labels):
            size = int((labels == i).sum())
            if size > largest:
                largest = size
        edge_ratio = edge_pixels / float(boundary01.sum() + 1e-6)
        connectivity = largest / float(edge_pixels + 1e-6)
        return float(0.6 * edge_ratio + 0.4 * connectivity)


# ============================
# Points
# ============================


def _centroid_of_mask(mask01: np.ndarray) -> Optional[Tuple[int, int]]:
    """返回掩码的中位数质心 (y,x)，若掩码为空返回 None"""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return int(np.median(ys)), int(np.median(xs))


def _boundary_mask(mask01: np.ndarray) -> np.ndarray:
    """简单的边缘检测，返回边界像素的二值图"""
    k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.int32)
    pad = np.pad(mask01.astype(np.int32), 1, mode="edge")
    H, W = mask01.shape
    out = np.zeros_like(mask01)
    for y in range(H):
        for x in range(W):
            out[y, x] = abs(int((k * pad[y : y + 3, x : x + 3]).sum()))
    return (out > 0).astype(np.uint8)


def _nearest_outside(mask01: np.ndarray, y: int, x: int, band: int, limit_box: Optional[Box] = None) -> Optional[Tuple[int, int]]:
    """从 (y,x) 周围向外扩展 band，找到最近的背景像素坐标（用于负点）

    若提供 limit_box，则搜索窗口始终限制在该 box 范围内。
    """
    H, W = mask01.shape
    for r in range(1, band + 1):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        if limit_box is not None:
            y0 = max(y0, limit_box.r0); y1 = min(y1, limit_box.r1)
            x0 = max(x0, limit_box.c0); x1 = min(x1, limit_box.c1)
            if y0 >= y1 or x0 >= x1:
                continue
        sub = mask01[y0:y1, x0:x1]
        ys, xs = np.where(sub == 0)
        if len(xs) > 0:
            return int(y0 + ys[len(xs) // 2]), int(x0 + xs[len(xs) // 2])
    return None


def _nearest_inside(mask01: np.ndarray, y: int, x: int, band: int, limit_box: Optional[Box] = None) -> Optional[Tuple[int, int]]:
    """从 (y,x) 周围向内扩展 band，找到最近的前景像素坐标（用于正点）

    若提供 limit_box，则搜索窗口始终限制在该 box 范围内。
    """
    H, W = mask01.shape
    for r in range(1, band + 1):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        if limit_box is not None:
            y0 = max(y0, limit_box.r0); y1 = min(y1, limit_box.r1)
            x0 = max(x0, limit_box.c0); x1 = min(x1, limit_box.c1)
            if y0 >= y1 or x0 >= x1:
                continue
        sub = mask01[y0:y1, x0:x1]
        ys, xs = np.where(sub == 1)
        if len(xs) > 0:
            return int(y0 + ys[len(xs) // 2]), int(x0 + xs[len(xs) // 2])
    return None


def _sample_boundary_points(M: np.ndarray, allowed_box: Box, n_samples: int) -> List[Tuple[int, int]]:
    """均匀采样边界点（限制在 allowed_box 内）"""
    boundary = _boundary_mask(M)
    bys, bxs = np.where(boundary > 0)
    coords = [(int(y), int(x)) for y, x in zip(bys.tolist(), bxs.tolist())
              if (allowed_box.r0 <= y < allowed_box.r1 and allowed_box.c0 <= x < allowed_box.c1)]
    if not coords:
        return []
    if n_samples <= 0 or n_samples >= len(coords):
        return coords
    step = max(1, len(coords) // n_samples)
    return coords[::step][:n_samples]


def _disk_mask(H: int, W: int, cy: int, cx: int, r: int) -> np.ndarray:
    """生成以 (cy, cx) 为中心、半径 r 的圆盘二值掩码"""
    y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
    x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask_local = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)
    out = np.zeros((H, W), dtype=np.uint8)
    out[y0:y1, x0:x1] = mask_local.astype(np.uint8)
    return out


def make_boundary_semantic_points(
    image: np.ndarray,
    M: np.ndarray,
    allowed_mask: np.ndarray,
    allowed_box: Box,
    scorer: RegionScorer,
    instance_text: str,
    max_total: int,
    pos_neg_ratio: float,
    band_out: int,
    band_in: int,
    r_out: int,
    r_in: int,
    n_samples: int,
) -> Tuple[List[Point], List[Point]]:
    """使用 Alpha-CLIP 语义评分在边界处生成正负点：
    - 正点：在边界外侧的候选圆盘语义分数高 → 需要扩张
    - 负点：在边界内侧的候选圆盘语义分数低 → 需要收缩
    """
    H, W = M.shape
    pos_candidates: List[Tuple[float, Tuple[int, int]]] = []  # (score, (y,x) outside)
    neg_candidates: List[Tuple[float, Tuple[int, int]]] = []  # (score, (y,x) inside), score用作升序选择

    samples = _sample_boundary_points(M, allowed_box, n_samples=n_samples)
    if not samples:
        return [], []

    # 逐点评估内/外小圆盘的语义分数
    for (y, x) in samples:
        inside = _nearest_inside(M, y, x, band_in, limit_box=allowed_box)
        outside = _nearest_outside(M, y, x, band_out, limit_box=allowed_box)
        # 外侧：若存在候选点，则生成小圆盘 alpha 并评分
        if outside is not None:
            oy, ox = outside
            alpha_out = _disk_mask(H, W, oy, ox, r_out)
            alpha_out = (alpha_out & allowed_mask).astype(np.uint8)
            if alpha_out.sum() > 0:
                s_out = float(scorer.score(image, alpha_out, instance_text))
                pos_candidates.append((s_out, (oy, ox)))
        # 内侧：若存在候选点，则生成小圆盘 alpha 并评分
        if inside is not None:
            iy, ix = inside
            alpha_in = _disk_mask(H, W, iy, ix, r_in)
            alpha_in = (alpha_in & allowed_mask).astype(np.uint8)
            if alpha_in.sum() > 0:
                s_in = float(scorer.score(image, alpha_in, instance_text))
                neg_candidates.append((s_in, (iy, ix)))

    # 选择点：正点取语义分数最高的前 K 个，负点取分数最低的前 K 个
    max_pos = int(max_total * pos_neg_ratio / (1.0 + pos_neg_ratio))
    max_neg = max_total - max_pos

    pos_candidates.sort(key=lambda t: t[0], reverse=True)
    neg_candidates.sort(key=lambda t: t[0])

    pos_pts: List[Point] = []
    neg_pts: List[Point] = []

    for s, (py, px) in pos_candidates[:max_pos]:
        pos_pts.append(Point(py, px, True, score=float(s)))
    for s, (ny, nx) in neg_candidates[:max_neg]:
        neg_pts.append(Point(ny, nx, False, score=float(s)))

    return pos_pts, neg_pts



def compute_edge_directions(edge_map: np.ndarray) -> np.ndarray:
    edge_float = edge_map.astype(np.float32)
    gx = cv2.Sobel(edge_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge_float, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    dx = gx / mag
    dy = gy / mag
    dirs = np.stack([dy, dx], axis=-1)  # (y,x) 方向
    return dirs


def _explore_along_direction(mask01: np.ndarray, y: int, x: int, direction: Tuple[float, float], max_distance: int, limit_box: Box, target_val: int) -> Optional[Tuple[int, int]]:
    """沿给定方向探索最近的目标像素（target_val ∈ {0,1}）。"""
    H, W = mask01.shape
    for step in range(1, max_distance + 1):
        ny = int(round(y + step * direction[0]))
        nx = int(round(x + step * direction[1]))
        if ny < limit_box.r0 or ny >= limit_box.r1 or nx < limit_box.c0 or nx >= limit_box.c1:
            continue
        if mask01[ny, nx] == target_val:
            return (ny, nx)
    return None


def make_edge_guided_points(
    image: np.ndarray,
    M: np.ndarray,
    allowed_mask: np.ndarray,
    allowed_box: Box,
    scorer: EdgeAwareScorer,
    max_total: int,
    pos_neg_ratio: float,
    band_out: int,
    band_in: int,
    r_out: int,
    r_in: int,
    n_samples: int,
) -> Tuple[List[Point], List[Point]]:
    """边缘引导的正负点生成：
    - 正点（扩张）：沿边界法线外侧方向，寻最近背景像素并评分，取高分前K
    - 负点（收缩）：沿边界法线内侧方向，寻最近前景像素并评分，取低分前K
    """
    H, W = M.shape
    samples = _sample_boundary_points(M, allowed_box, n_samples=n_samples)
    if not samples:
        return [], []
    dirs = compute_edge_directions(scorer.edges)
    pos_candidates: List[Tuple[float, Tuple[int, int]]] = []
    neg_candidates: List[Tuple[float, Tuple[int, int]]] = []

    for (y, x) in samples:
        # 边缘方向可能为零向量，做兜底
        dy, dx = dirs[y, x]
        if abs(dy) + abs(dx) < 1e-6:
            # 默认采用垂直向外/向内两个方向
            normals = [(0.0, 1.0), (0.0, -1.0)]
        else:
            # 法线方向（旋转90度）
            normals = [(-dx, dy), (dx, -dy)]
        # 选择一组法线来尝试（取成功的第一个）
        found_out = None
        found_in = None
        for nd in normals:
            if found_out is None:
                found_out = _explore_along_direction(M, y, x, nd, band_out, allowed_box, target_val=0)
            if found_in is None:
                found_in = _explore_along_direction(M, y, x, (-nd[0], -nd[1]), band_in, allowed_box, target_val=1)
            if found_out is not None and found_in is not None:
                break
        if found_out is not None:
            oy, ox = found_out
            alpha_out = _disk_mask(H, W, oy, ox, r_out)
            alpha_out = (alpha_out & allowed_mask).astype(np.uint8)
            if alpha_out.sum() > 0:
                s_out = float(scorer.score(image, alpha_out, ""))
                pos_candidates.append((s_out, (oy, ox)))
        if found_in is not None:
            iy, ix = found_in
            alpha_in = _disk_mask(H, W, iy, ix, r_in)
            alpha_in = (alpha_in & allowed_mask).astype(np.uint8)
            if alpha_in.sum() > 0:
                s_in = float(scorer.score(image, alpha_in, ""))
                neg_candidates.append((s_in, (iy, ix)))

    max_pos = int(max_total * pos_neg_ratio / (1.0 + pos_neg_ratio))
    max_neg = max_total - max_pos
    pos_candidates.sort(key=lambda t: t[0], reverse=True)
    neg_candidates.sort(key=lambda t: t[0])
    pos_pts = [Point(y, x, True, score=float(s)) for s, (y, x) in pos_candidates[:max_pos]]
    neg_pts = [Point(y, x, False, score=float(s)) for s, (y, x) in neg_candidates[:max_neg]]
    return pos_pts, neg_pts



def select_points_from_cells(
    M: np.ndarray,
    pos_cells: List[Cell],
    neg_cells: List[Cell],
    max_total: int,
    pos_neg_ratio: float,
    boundary_band: int,
    allow_negative: bool = True,
    limit_box: Optional[Box] = None,
) -> Tuple[List[Point], List[Point]]:
    """根据正/负单元格从掩码中选择用于 SAM 的点提示

    策略：
    - 正点：在单元格内部取掩码的中位数质心
    - 负点：在单元格内的边界上，向外最近位置取一个背景点
    - 数量：受 max_total 与 pos_neg_ratio 控制
    """
    pos_pts: List[Point] = []
    neg_pts: List[Point] = []
    max_pos = int(max_total * pos_neg_ratio / (1.0 + pos_neg_ratio))
    max_neg = max_total - max_pos
    # 排序：正格按分数降序，负格按分数升序
    pos_cells_sorted = sorted(pos_cells, key=lambda c: float(c.score or 0.0), reverse=True)
    neg_cells_sorted = sorted(neg_cells, key=lambda c: float(c.score or 0.0))

    # 为每个正格挑选一个质心点（高分优先）
    for c in pos_cells_sorted:
        b = c.box
        sub = M[b.r0:b.r1, b.c0:b.c1]
        cyx = _centroid_of_mask(sub)
        if cyx:
            y = b.r0 + cyx[0]
            x = b.c0 + cyx[1]
            pos_pts.append(Point(y, x, True, score=float(c.score or 0.0)))
        if len(pos_pts) >= max_pos:
            break

    neg_pts: List[Point] = []
    if allow_negative and max_neg > 0:
        # 预先计算整个掩码的边界像素集合
        boundary = _boundary_mask(M)
        bys, bxs = np.where(boundary > 0)
        bcoords = list(zip(bys.tolist(), bxs.tolist()))
        # 为每个负格在其内部找到边界点，向外扩散寻找最近的背景点
        for c in neg_cells_sorted:
            b = c.box
            cand = [(y, x) for (y, x) in bcoords if (b.r0 <= y < b.r1 and b.c0 <= x < b.c1)]
            if not cand:
                continue
            yx = cand[len(cand) // 2]
            out = _nearest_outside(M, yx[0], yx[1], boundary_band, limit_box=limit_box)
            if out:
                neg_pts.append(Point(out[0], out[1], False, score=float(c.score or 0.0)))
            if len(neg_pts) >= max_neg:
                break
    return pos_pts, neg_pts


# ============================
# Metrics / SAM iface / Pipeline
# ============================


def iou(a01: np.ndarray, b01: np.ndarray) -> float:
    """计算两个二值掩码的 IoU（加入极小平滑项避免除零）"""
    a = a01 > 0
    b = b01 > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum()) + 1e-6
    return inter / union


def smooth_mask(M01: np.ndarray) -> np.ndarray:
    """对二值掩码做 3x3 多数投票平滑，去除毛刺/小孔"""
    M = (M01 > 0).astype(np.uint8)
    pad = np.pad(M, 1, mode="edge")
    H, W = M.shape
    out = np.zeros_like(M)
    for y in range(H):
        for x in range(W):
            win = pad[y:y + 3, x:x + 3]
            s = int(win.sum())
            out[y, x] = 1 if s >= 5 else 0
    return out


class SamWrapper:
    def __init__(self, sam_wrapper=None):
        """
        初始化SAM包装器
        
        Args:
            sam_wrapper: 外部传入的SAM实例(来自sam_integration模块)
        """
        self.sam_wrapper = sam_wrapper
        self._current_image = None

    def predict_box(self, image: np.ndarray, bbox: Box) -> np.ndarray:
        """使用边界框预测掩码

        - 确保先设置图像（避免多次重复 set_image）
        - 将 Box(r0,c0,r1,c1) 转为 SAM 接口所需的 [x1,y1,x2,y2]
        """
        if self.sam_wrapper is None:
            raise NotImplementedError("请提供真实的SAM实现")
        
        # 设置图像（如果与上次不同）
        if self._current_image is None or not np.array_equal(self._current_image, image):
            self.sam_wrapper.set_image(image)
            self._current_image = image.copy()
        
        # 转换边界框格式: Box(r0,c0,r1,c1) -> [x1,y1,x2,y2]
        box_list = [bbox.c0, bbox.r0, bbox.c1, bbox.r1]
        return self.sam_wrapper.predict_with_box(box_list)

    def predict_points(self, image: np.ndarray, pos: List[Point], neg: List[Point]) -> np.ndarray:
        """使用点提示预测掩码

        - 正点 label=1，负点 label=0
        - 输入点的坐标为 (y,x)，SAM 需要 (x,y)，内部会转换
        - 若没有点，则返回全零掩码
        """
        if self.sam_wrapper is None:
            raise NotImplementedError("请提供真实的SAM实现")
        
        # 设置图像（如果与上次不同）
        if self._current_image is None or not np.array_equal(self._current_image, image):
            self.sam_wrapper.set_image(image)
            self._current_image = image.copy()
        
        if len(pos) == 0 and len(neg) == 0:
            H, W = image.shape[:2]
            return np.zeros((H, W), dtype=np.uint8)
        
        # 准备点坐标和标签
        point_coords = []
        point_labels = []
        
        # 正点: (y,x) -> (x,y), label=1
        for pt in pos:
            point_coords.append([pt.x, pt.y])
            point_labels.append(1)
        
        # 负点: (y,x) -> (x,y), label=0  
        for pt in neg:
            point_coords.append([pt.x, pt.y])
            point_labels.append(0)
        
        if len(point_coords) > 0:
            point_coords = np.array(point_coords)
            point_labels = np.array(point_labels)
            return self.sam_wrapper.predict_with_points(point_coords, point_labels)
        else:
            H, W = image.shape[:2]
            return np.zeros((H, W), dtype=np.uint8)


def threshold_cells(cells: List[Cell]) -> Tuple[List[Cell], List[Cell], List[Cell]]:
    """基于 μ±0.5σ 自适应阈分，将单元格划为 正/负/不确定 三类"""
    scores = np.array([float(c.score) for c in cells if c.score is not None], dtype=np.float32)
    mu = float(scores.mean()) if scores.size else 0.0
    sigma = float(scores.std()) if scores.size else 1.0
    tau_pos = mu + 0.5 * sigma
    tau_neg = mu - 0.5 * sigma
    pos: List[Cell] = []
    neg: List[Cell] = []
    unc: List[Cell] = []
    for c in cells:
        s = float(c.score) if c.score is not None else mu
        if s >= tau_pos:
            pos.append(c)
        elif s <= tau_neg:
            neg.append(c)
        else:
            unc.append(c)
    return pos, neg, unc


def sculpting_pipeline(
    image: np.ndarray,
    bbox: Box,
    instance_text: str,
    sam: SamWrapper,
    scorer: RegionScorer,
    cfg: Config,
    initial_mask_M0: np.ndarray,
    debug_hook: Optional[Callable[..., None]] = None,
) -> np.ndarray:
    """主流程（仅边界语义精修）：在 prior 掩码约束下进行边界语义驱动的点提示迭代。

    流程：
    - 用 SAM 框预测与 prior 掩码求交作为初始掩码 M。
    - 每轮在边界采样，向内/外探索小圆盘，使用 Alpha-CLIP 语义评分选择正负点。
    - 用 SAM 点提示更新掩码（始终限制在 prior 区域内）。
    """
    # 初始化：SAM 框预测 ∩ prior 掩码
    M0_sam = (sam.predict_box(image, bbox) > 0).astype(np.uint8)
    M0_sam = smooth_mask(M0_sam)
    prior01 = (initial_mask_M0 > 0).astype(np.uint8)
    M = (M0_sam & prior01).astype(np.uint8)

    # 计算允许域（使用 prior 的紧致外接框）
    H, W = image.shape[:2]
    _tight = bbox_from_mask(prior01)
    allowed_box = bbox
    if _tight is not None:
        x1, y1, x2, y2 = _tight
        allowed_box = box_from_xyxy_clip(y1, x1, y2, x2, H, W)

    # 在 prior 区域内进行迭代（无需额外扩张）
    for it in range(cfg.k_iters):
        # 在边界带生成正负点（根据模式选择）
        if cfg.point_mode == "edge":
            # 边缘感知：不使用文本
            assert isinstance(scorer, EdgeAwareScorer)
            Ppos, Pneg = make_edge_guided_points(
                image,
                M,
                prior01,
                allowed_box,
                scorer,
                max_total=cfg.max_points_per_iter,
                pos_neg_ratio=cfg.pos_neg_ratio,
                band_out=cfg.boundary_band,
                band_in=cfg.boundary_inner_band,
                r_out=cfg.boundary_alpha_radius_out,
                r_in=cfg.boundary_alpha_radius_in,
                n_samples=cfg.boundary_samples,
            )
        else:
            # 语义引导（默认）
            Ppos, Pneg = make_boundary_semantic_points(
                image,
                M,
                prior01,  # 使用 prior 作为允许域
                allowed_box,
                scorer,
                instance_text,
                max_total=cfg.max_points_per_iter,
                pos_neg_ratio=cfg.pos_neg_ratio,
                band_out=cfg.boundary_band,
                band_in=cfg.boundary_inner_band,
                r_out=cfg.boundary_alpha_radius_out,
                r_in=cfg.boundary_alpha_radius_in,
                n_samples=cfg.boundary_samples,
            )

        # 调试回调
        if debug_hook is not None:
            try:
                debug_hook(
                    it=it,
                    cells=[],
                    pos_cells=[],
                    neg_cells=[],
                    pos_points=Ppos,
                    neg_points=Pneg,
                    mask=M,
                )
            except Exception:
                pass

        # 若无可用点，提前终止
        if (len(Ppos) + len(Pneg)) == 0:
            break

        # 使用 SAM 点提示更新掩码，并与 prior 求交
        M_new = (sam.predict_points(image, Ppos, Pneg) > 0).astype(np.uint8)
        M_new = (M_new & prior01).astype(np.uint8)  # 强制约束在 prior 区域内
        M_new = smooth_mask(M_new)

        # 早停：与前一轮 IoU 足够高则停止
        if iou(M_new, M) > (1.0 - cfg.iou_eps):
            break

        M = M_new

    return M
