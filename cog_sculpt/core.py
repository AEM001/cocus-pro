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
    r0: int
    c0: int
    r1: int
    c1: int


@dataclass
class Point:
    y: int
    x: int
    is_positive: bool
    score: float = 1.0


@dataclass
class Cell:
    box: Box
    depth: int = 0
    score: Optional[float] = None


# ============================
# IO utils
# ============================


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask_png(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def save_mask_png(mask01: np.ndarray, path: str) -> None:
    Image.fromarray((mask01.astype(np.uint8) * 255)).save(path)


def ensure_dir(path: str) -> None:
    import os

    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_read_boxes(path: Optional[str]) -> Optional[np.ndarray]:
    import os

    if not path or not os.path.isfile(path):
        return None
    data = read_json(path)
    boxes = data.get("boxes")
    if boxes is None:
        return None
    return np.asarray(boxes, dtype=np.int32)


def bbox_from_mask(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay(image: np.ndarray, mask01: np.ndarray, color=(0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    out = image.copy().astype(np.uint8)
    m = (mask01 > 0)
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


# ============================
# ROI / Grid / Alpha
# ============================


def roi_to_bbox(ids_v, ids_h, H: int, W: int, v_segments: int, h_segments: int, margin: float) -> Box:
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
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 < x1:
        x1, x2 = x2, x1
    return Box(y1, x1, y2, x2)


def partition_grid(bbox: Box, grid: Tuple[int, int]) -> List[Cell]:
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
    alpha = np.zeros_like(M, dtype=np.float32)
    b = cell.box
    alpha[b.r0:b.r1, b.c0:b.c1] = M[b.r0:b.r1, b.c0:b.c1]
    return alpha


def compose_rgba(image: np.ndarray, alpha01: np.ndarray) -> np.ndarray:
    a = (alpha01 * 255).astype(np.uint8) if alpha01.dtype != np.uint8 else alpha01
    return np.dstack([image, a])


# ============================
# Cell utils
# ============================


def _cell_size(cell: Cell) -> Tuple[int, int]:
    b = cell.box
    return (b.r1 - b.r0, b.c1 - b.c0)


# ============================
# Scorer
# ============================


class RegionScorer:
    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        raise NotImplementedError


class AlphaCLIPScorer(RegionScorer):
    def __init__(self, model) -> None:
        self.model = model

    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        """使用Alpha-CLIP进行区域语义评分"""
        try:
            score = self.model.score_region_with_templates(
                image, alpha01, text,
                templates=[
                    "a photo of the {} camouflaged in the background.",
                    "a photo of the {}.",
                    "the {} blending into surroundings."
                ],
                aggregate='mean'
            )
            return score
        except Exception as e:
            print(f"Warning: Alpha-CLIP scoring failed: {e}")
            return 0.0


class SoftFocusCLIPScorer(RegionScorer):
    def __init__(self, model) -> None:
        self.model = model

    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        """使用软聚焦方法进行区域评分(非Alpha-CLIP的替代方案)"""
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
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def score(self, image: np.ndarray, alpha01: np.ndarray, text: str) -> float:
        s = float((alpha01 > 0).sum())
        area = float(alpha01.size) + self.eps
        return s / area


# ============================
# Points
# ============================


def _centroid_of_mask(mask01: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return int(np.median(ys)), int(np.median(xs))


def _boundary_mask(mask01: np.ndarray) -> np.ndarray:
    k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.int32)
    pad = np.pad(mask01.astype(np.int32), 1, mode="edge")
    H, W = mask01.shape
    out = np.zeros_like(mask01)
    for y in range(H):
        for x in range(W):
            out[y, x] = abs(int((k * pad[y : y + 3, x : x + 3]).sum()))
    return (out > 0).astype(np.uint8)


def _nearest_outside(mask01: np.ndarray, y: int, x: int, band: int) -> Optional[Tuple[int, int]]:
    H, W = mask01.shape
    for r in range(1, band + 1):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
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
) -> Tuple[List[Point], List[Point]]:
    pos_pts: List[Point] = []
    neg_pts: List[Point] = []
    max_pos = int(max_total * pos_neg_ratio / (1.0 + pos_neg_ratio))
    max_neg = max_total - max_pos
    for c in pos_cells:
        b = c.box
        sub = M[b.r0:b.r1, b.c0:b.c1]
        cyx = _centroid_of_mask(sub)
        if cyx:
            y = b.r0 + cyx[0]
            x = b.c0 + cyx[1]
            pos_pts.append(Point(y, x, True, score=float(c.score or 0.0)))
        if len(pos_pts) >= max_pos:
            break
    boundary = _boundary_mask(M)
    bys, bxs = np.where(boundary > 0)
    bcoords = list(zip(bys.tolist(), bxs.tolist()))
    for c in neg_cells:
        b = c.box
        cand = [(y, x) for (y, x) in bcoords if (b.r0 <= y < b.r1 and b.c0 <= x < b.c1)]
        if not cand:
            continue
        yx = cand[len(cand) // 2]
        out = _nearest_outside(M, yx[0], yx[1], boundary_band)
        if out:
            neg_pts.append(Point(out[0], out[1], False, score=float(c.score or 0.0)))
        if len(neg_pts) >= max_neg:
            break
    return pos_pts, neg_pts


# ============================
# Metrics / SAM iface / Pipeline
# ============================


def iou(a01: np.ndarray, b01: np.ndarray) -> float:
    a = a01 > 0
    b = b01 > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum()) + 1e-6
    return inter / union


def smooth_mask(M01: np.ndarray) -> np.ndarray:
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
        """使用边界框预测掩码"""
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
        """使用点提示预测掩码"""
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
    if initial_mask_M0 is not None:
        M = (initial_mask_M0 > 0).astype(np.uint8)
    else:
        M = (sam.predict_box(image, bbox) > 0).astype(np.uint8)
    M = smooth_mask(M)

    for it in range(cfg.k_iters):
        # 1) 初始网格
        leaf_cells: List[Cell] = partition_grid(bbox, cfg.grid_init)

        # 2) 递归细分：仅对不确定单元格，且尺寸 >= min_cell，深度 < max_depth
        while True:
            # 给当前叶子全部打分
            prompt = cfg.prompt_template.format(instance=instance_text)
            for c in leaf_cells:
                if c.score is None:
                    alpha = build_alpha_for_cell(M, c)
                    c.score = float(scorer.score(image, alpha, prompt))

            # 基于当前叶子阈分
            pos_now, neg_now, unc_now = threshold_cells(leaf_cells)

            # 选择需要细分的“不确定格”
            to_subdivide: List[Cell] = []
            for c in unc_now:
                if c.depth >= cfg.max_depth:
                    continue
                h, w = _cell_size(c)
                if min(h, w) < cfg.min_cell:
                    continue
                to_subdivide.append(c)

            if not to_subdivide:
                break

            # 用子格替换父格，子格待评分
            new_leaves: List[Cell] = []
            for c in leaf_cells:
                if c in to_subdivide:
                    new_leaves.extend(subdivide_cell(c))
                else:
                    new_leaves.append(c)
            leaf_cells = new_leaves
            # 将新加入的子格 score 置空，避免复用父格分数
            for c in leaf_cells:
                if c.depth > 0 and c.score is None:
                    pass  # 明确保留 None，下一轮会评分

        # 3) 最终基于叶子集合的阈分，产出正/负格
        pos_cells, neg_cells, _ = threshold_cells(leaf_cells)
        Ppos, Pneg = select_points_from_cells(
            M,
            pos_cells,
            neg_cells,
            max_total=cfg.max_points_per_iter,
            pos_neg_ratio=cfg.pos_neg_ratio,
            boundary_band=cfg.boundary_band,
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
                    mask=M,
                )
            except Exception:
                pass
        if (len(Ppos) + len(Pneg)) == 0:
            break
        M_new = (sam.predict_points(image, Ppos, Pneg) > 0).astype(np.uint8)
        M_new = smooth_mask(M_new)
        # 4) 早停：若两次掩码非常相似（IoU 足够高），则停止
        if iou(M_new, M) > (1.0 - cfg.iou_eps):
            break
        M = M_new

    return M
