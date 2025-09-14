替换的可行性分析
当前架构的优势
你的make_boundary_semantic_points函数设计得非常模块化，只需要：

替换评分机制：从语义评分改为边缘感知评分
保持采样框架：继续使用边界采样和内外探索
适配参数：调整探索半径和评分阈值

边缘感知优化的具体实现
1. 边缘感知评分器（替换RegionScorer）

```
class EdgeAwareScorer(RegionScorer):
    """边缘感知评分器，替代AlphaCLIPScorer"""
    
    def __init__(self, image, current_mask):
        self.image = image
        self.current_mask = current_mask
        self.edges = self._compute_multi_source_edges()
        
    def _compute_multi_source_edges(self):
        """计算多源边缘图"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        # 1. Canny边缘
        canny = cv2.Canny(gray, 50, 150)
        
        # 2. Sobel边缘强度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 3. 颜色边缘（Lab空间）
        lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        lab_edges = np.zeros_like(gray)
        for i in range(3):
            lab_edges += cv2.Sobel(lab[:,:,i], cv2.CV_64F, 1, 1, ksize=3)
        
        # 融合边缘
        combined = (
            0.4 * canny + 
            0.4 * (sobel_magnitude / sobel_magnitude.max() * 255) + 
            0.2 * (lab_edges / lab_edges.max() * 255)
        )
        
        return combined.astype(np.uint8)
    
    def score(self, image, alpha01, text=None):
        """边缘感知评分 - 完全替代语义评分"""
        # 计算alpha区域的边缘密度
        edge_density = np.sum(self.edges * alpha01) / np.sum(alpha01)
        
        # 计算边缘连续性
        boundary = self._get_boundary_pixels(alpha01)
        continuity = self._compute_edge_continuity(boundary)
        
        # 综合评分
        return 0.7 * edge_density + 0.3 * continuity
    
    def _get_boundary_pixels(self, mask):
        """获取掩码边界像素"""
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = dilated - eroded
        return boundary > 0
    
    def _compute_edge_continuity(self, boundary):
        """计算边缘连续性"""
        boundary_edges = self.edges * boundary
        if np.sum(boundary) == 0:
            return 0.0
        
        # 计算边界上边缘像素的比例
        edge_ratio = np.sum(boundary_edges > 0) / np.sum(boundary)
        
        # 计算边缘的连通性
        labeled, num_features = ndimage.label(boundary_edges > 0)
        if num_features == 0:
            return 0.0
        
        # 最大的连通区域占比
        largest_component = max(np.sum(labeled == i) for i in range(1, num_features + 1))
        connectivity = largest_component / np.sum(boundary_edges > 0)
        
        return 0.6 * edge_ratio + 0.4 * connectivity
```
2. 边缘引导的采样策略（替换make_boundary_semantic_points）
```python
def make_edge_guided_points(
    image: np.ndarray,
    M: np.ndarray,
    allowed_mask: np.ndarray,
    allowed_box: Box,
    scorer: EdgeAwareScorer,  # 替换为边缘感知评分器
    max_total: int,
    pos_neg_ratio: float,
    band_out: int,
    band_in: int,
    r_out: int,
    r_in: int,
    n_samples: int,
) -> Tuple[List[Point], List[Point]]:
    """边缘引导的正负点生成"""
    
    # 使用现有的边界采样
    samples = _sample_boundary_points(M, allowed_box, n_samples=n_samples)
    if not samples:
        return [], []
    
    # 计算边缘梯度方向
    edge_directions = compute_edge_directions(scorer.edges)
    
    pos_candidates = []
    neg_candidates = []
    
    for y, x in samples:
        # 获取当前点的边缘方向
        edge_dir = edge_directions[y, x]
        
        # 沿边缘法线方向探索
        normal_dir = [-edge_dir[1], edge_dir[0]]  # 旋转90度
        
        # 外侧探索（扩张）
        out_points = explore_along_direction(
            M, y, x, normal_dir, band_out, r_out, allowed_box
        )
        
        # 内侧探索（收缩）
        in_points = explore_along_direction(
            M, y, x, [-d for d in normal_dir], band_in, r_in, allowed_box
        )
        
        # 边缘感知评分
        for oy, ox in out_points:
            alpha_out = _disk_mask(H, W, oy, ox, r_out)
            alpha_out = (alpha_out & allowed_mask).astype(np.uint8)
            if alpha_out.sum() > 0:
                score = scorer.score(image, alpha_out, None)  # 不再需要文本
                pos_candidates.append((score, (oy, ox)))
        
        for iy, ix in in_points:
            alpha_in = _disk_mask(H, W, iy, ix, r_in)
            alpha_in = (alpha_in & allowed_mask).astype(np.uint8)
            if alpha_in.sum() > 0:
                score = scorer.score(image, alpha_in, None)
                neg_candidates.append((score, (iy, ix)))
    
    # 选择策略调整
    max_pos = int(max_total * pos_neg_ratio / (1.0 + pos_neg_ratio))
    max_neg = max_total - max_pos
    
    # 边缘强度高的区域优先
    pos_candidates.sort(key=lambda t: t[0], reverse=True)
    neg_candidates.sort(key=lambda t: t[0])  # 低分优先
    
    return (
        [Point(y, x, True, s) for s, (y, x) in pos_candidates[:max_pos]],
        [Point(y, x, False, s) for s, (y, x) in neg_candidates[:max_neg]]
    )
```
3. 方向感知探索函数
‵‵‵python
def explore_along_direction(mask, y, x, direction, max_distance, radius, limit_box):
    """沿指定方向探索有效点"""
    H, W = mask.shape
    points = []
    
    for step in range(1, max_distance + 1):
        ny = int(y + step * direction[0])
        nx = int(x + step * direction[1])
        
        # 边界检查
        if not (limit_box.r0 <= ny < limit_box.r1 and 
                limit_box.c0 <= nx < limit_box.c1):
            continue
            
        # 检查是否在允许区域内
        if mask[ny, nx] == 0:  # 背景区域
            points.append((ny, nx))
            break  # 找到最近的背景点
    
    return points
```
4. 边缘方向计算
```python
def compute_edge_directions(edge_map):
    """计算边缘方向图"""
    sobel_x = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算方向向量
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude[magnitude == 0] = 1  # 避免除零
    
    directions = np.stack([
        sobel_x / magnitude,
        sobel_y / magnitude
    ], axis=-1)
    
    return directions
```
集成到你的现有框架
1. 最小修改的集成方式
```
# 在你的sculpting_pipeline中，只需要修改两行
def sculpting_pipeline_edge_aware(
    image: np.ndarray,
    bbox: Box,
    instance_text: str,  # 可以保留但不使用
    sam: SamWrapper,
    scorer: RegionScorer,  # 实际传入EdgeAwareScorer
    cfg: Config,
    initial_mask_M0: np.ndarray,
) -> np.ndarray:
    
    # ... 保持原有流程不变 ...
    
    # 替换评分器
    edge_scorer = EdgeAwareScorer(image, M)
    
    # 替换点生成
    Ppos, Pneg = make_edge_guided_points(
        image, M, prior01, bbox, edge_scorer,
        max_total=cfg.max_points_per_iter,
        pos_neg_ratio=cfg.pos_neg_ratio,
        band_out=cfg.boundary_band,
        band_in=cfg.boundary_inner_band,
        r_out=cfg.boundary_alpha_radius_out,
        r_in=cfg.boundary_alpha_radius_in,
        n_samples=cfg.boundary_samples,
    )
    
    # ... 其余流程完全不变 ...
```
