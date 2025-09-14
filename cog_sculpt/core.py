"""
实现“语义雕刻 + SAM”的核心流程，包括网格划分、
区域语义评分、正负点选择、SAM 交互细化与迭代早停。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image


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

    prompt_template: str = "a photo of the {instance} camouflaged in the background."

    image_path: Optional[str] = None
    meta_path: Optional[str] = None
    boxes_json_path: Optional[str] = None
    prior_mask_path: Optional[str] = None
    out_root: str = "sculpt_out"

    use_prior_mask_as_M0: bool = True
    use_boxes_for_roi: bool = True


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
    initial_mask_M0: Optional[np.ndarray] = None,
    debug_hook: Optional[Callable[..., None]] = None,
) -> np.ndarray:
    """主流程：语义雕刻 + SAM 交互迭代，输出最终掩码

    参数：
    - image: 输入 RGB 图像
    - bbox: 处理的 ROI 区域
    - instance_text: 实例类别文本（将被 format 到 cfg.prompt_template 中）
    - sam: SAM 包装器（需提供 set_image / predict 接口）
    - scorer: 区域语义评分器（如 AlphaCLIPScorer）
    - cfg: 管线配置
    - initial_mask_M0: 初始掩码（若为 None 则用 bbox 初始化）
    - debug_hook: 可选调试回调，观察每轮状态
    """
    # 步骤0：无论是否提供 initial_mask_M0，都先用 SAM 的框预测得到 M0_sam
    M0_sam = (sam.predict_box(image, bbox) > 0).astype(np.uint8)
    M0_sam = smooth_mask(M0_sam)
    # 若提供了初始掩码（如 prior），则与 SAM 预测做交集，限制到 prior 指定区域
    if initial_mask_M0 is not None:
        prior01 = (initial_mask_M0 > 0).astype(np.uint8)
        M = (M0_sam & prior01).astype(np.uint8)
    else:
        M = M0_sam

    H, W = image.shape[:2]
    roi_area = (bbox.r1 - bbox.r0) * (bbox.c1 - bbox.c0)
    img_area = H * W
    roi_ratio = float(roi_area) / float(img_area + 1e-6)

    # 计算全程固定的“允许点域”（初始 ROI 的轻度扩张）
    allowed_box = expand_box_scale(bbox, cfg.roi_point_expand_scale, H, W)
    allowed_mask = np.zeros((H, W), dtype=np.uint8)
    allowed_mask[allowed_box.r0:allowed_box.r1, allowed_box.c0:allowed_box.c1] = 1

    for it in range(cfg.k_iters):
        # 迭代开始时对当前掩码进行允许域硬门控（防止外溢）
        if cfg.gate_mask_outside_roi:
            M = (M & allowed_mask).astype(np.uint8)

        # 选择本轮网格（首轮粗，后续细）
        grid_this_iter: Tuple[int, int]
        if it == 0:
            grid_this_iter = cfg.grid_init
        else:
            grid_this_iter = cfg.fine_grid if cfg.fine_after_first else cfg.grid_init

        # 根据当前掩码估计更紧致的工作 bbox
        work_bbox: Box = bbox
        tight_xyxy = bbox_from_mask(M)
        if tight_xyxy is not None:
            x1, y1, x2, y2 = tight_xyxy
            work_bbox = box_from_xyxy_clip(y1, x1, y2, x2, H, W)

        # 1) 初始网格：将 work_bbox 按 grid_this_iter 划为多个叶子单元
        leaf_cells: List[Cell] = partition_grid(work_bbox, grid_this_iter)

        # 2) 递归细分：仅对不确定单元格，且尺寸 >= min_cell，深度 < max_depth
        # 根据迭代轮次决定本轮最大细分深度（首轮可禁用细分）
        # 本轮细分阈值与深度
        max_depth_cur = cfg.max_depth if it == 0 else (cfg.fine_max_depth if cfg.fine_after_first else cfg.max_depth)
        if it == 0 and cfg.gentle_first_iter and cfg.first_iter_disable_subdivide:
            max_depth_cur = 0
        min_cell_cur = cfg.min_cell if it == 0 else (cfg.fine_min_cell if cfg.fine_after_first else cfg.min_cell)

        while True:
            # 2.1) 给当前所有叶子打分（基于当前 M 构建 alpha）
            # 传入裸的 instance 文本，由具体的 scorer 内部做模板拼接
            text_for_score = instance_text
            for c in leaf_cells:
                if c.score is None:
                    # 使用允许域门控后的掩码进行子区域 alpha 构造
                    Mg = (M & allowed_mask).astype(np.uint8)
                    alpha = build_alpha_for_cell(Mg, c)
                    c.score = float(scorer.score(image, alpha, text_for_score))

            # 2.2) 基于当前叶子阈分
            pos_now, neg_now, unc_now = threshold_cells(leaf_cells)

            # 2.3) 选择需要细分的“不确定格”
            to_subdivide: List[Cell] = []
            for c in unc_now:
                if c.depth >= max_depth_cur:
                    continue
                h, w = _cell_size(c)
                if min(h, w) < min_cell_cur:
                    continue
                to_subdivide.append(c)

            if not to_subdivide:
                break

            # 2.4) 用子格替换父格，子格待评分
            new_leaves: List[Cell] = []
            for c in leaf_cells:
                if c in to_subdivide:
                    new_leaves.extend(subdivide_cell(c))
                else:
                    new_leaves.append(c)
            leaf_cells = new_leaves
            # 2.5) 将新加入的子格 score 置空，避免复用父格分数
            for c in leaf_cells:
                if c.depth > 0 and c.score is None:
                    pass  # 明确保留 None，下一轮会评分

        # 3) 基于叶子集合的分布，决定选点候选
        scores_np = np.array([float(c.score) for c in leaf_cells if c.score is not None], dtype=np.float32)
        sigma = float(scores_np.std()) if scores_np.size else 0.0

        if it == 0 and cfg.gentle_first_iter:
            # 首轮：仅从最高分的前 K 个单元取正点，禁用负点
            K = min(cfg.topk_pos_first_iter, len(leaf_cells))
            pos_cells = sorted(leaf_cells, key=lambda c: float(c.score or 0.0), reverse=True)[:K]
            neg_cells: List[Cell] = []
            allow_neg = False
        else:
            # 常规：阈分 + 仅在区分度足够时允许负点
            pos_cells, neg_cells, _ = threshold_cells(leaf_cells)
            allow_neg = (sigma >= cfg.min_sigma_for_neg)

        # 本轮最大点数（小ROI或首轮时进一步收紧）
        iter_max_points = cfg.max_points_per_iter
        if roi_ratio <= cfg.small_roi_ratio:
            iter_max_points = min(iter_max_points, cfg.small_roi_max_points)
        if it == 0 and cfg.gentle_first_iter:
            iter_max_points = min(iter_max_points, cfg.first_iter_max_points)

        # 计算允许点生成的区域（初始 ROI 的轻度扩张）
        allowed_box = expand_box_scale(bbox, cfg.roi_point_expand_scale, H, W)

        Ppos, Pneg = select_points_from_cells(
            (M & allowed_mask).astype(np.uint8),
            pos_cells,
            neg_cells,
            max_total=iter_max_points,
            pos_neg_ratio=cfg.pos_neg_ratio,
            boundary_band=cfg.boundary_band,
            allow_negative=allow_neg,
            limit_box=allowed_box,
        )
        if debug_hook is not None:
            try:
                debug_hook(
                    it=it,
                    cells=leaf_cells,
                    pos_cells=pos_cells,
                    neg_cells=neg_cells,
                    pos_points=Ppos,
                    neg_points=Pneg,
                    mask=(M & allowed_mask).astype(np.uint8),
                )
            except Exception:
                pass
        # 若本轮没有可用点，提前终止
        if (len(Ppos) + len(Pneg)) == 0:
            break
        # 4) 使用 SAM 点提示进行掩码细化
        M_new = (sam.predict_points(image, Ppos, Pneg) > 0).astype(np.uint8)
        # 预测后先做允许域硬门控，再平滑
        if cfg.gate_mask_outside_roi:
            M_new = (M_new & allowed_mask).astype(np.uint8)
        M_new = smooth_mask(M_new)
        # 5) 早停：若两次掩码非常相似（IoU 足够高），则停止
        if iou(M_new, M) > (1.0 - cfg.iou_eps):
            break
        # 6) 否则更新掩码并进入下一轮
        M = M_new

    return M
