"""
SAM/SAM2 集成模块，用于语义引导的交互式分割
"""

import torch
import numpy as np
from typing import List, Optional, Union, Tuple
import cv2
from abc import ABC, abstractmethod

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: segment-anything not installed. SAM functionality will be limited.")
    SAM_AVAILABLE = False

try:
    # SAM2 support (when available)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class BaseSAMWrapper(ABC):
    """SAM包装器基类"""
    
    @abstractmethod
    def set_image(self, image: np.ndarray) -> None:
        """设置要分割的图像"""
        pass
    
    @abstractmethod
    def predict_with_box(self, box: List[int]) -> np.ndarray:
        """使用边界框预测掩码"""
        pass
    
    @abstractmethod 
    def predict_with_points(
        self, 
        point_coords: np.ndarray, 
        point_labels: np.ndarray
    ) -> np.ndarray:
        """使用点提示预测掩码"""
        pass


class SAMWrapper(BaseSAMWrapper):
    """
    SAM (Segment Anything Model) 封装类
    
    支持边界框和点提示的交互式分割
    """
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: str = "models/sam_vit_h_4b8939.pth",
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        初始化SAM模型
        
        Args:
            model_type: 模型类型 ("vit_h", "vit_l", "vit_b")
            checkpoint_path: 模型检查点路径
            device: 计算设备
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything not installed. Please install it first.")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        
        # 加载SAM模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        self.predictor = SamPredictor(sam)
        self.current_image = None
        
        print(f"SAM loaded: {model_type} on {self.device}")
    
    def set_image(self, image: np.ndarray) -> None:
        """
        设置要分割的图像
        
        Args:
            image: RGB图像，shape为(H, W, 3)
        """
        self.predictor.set_image(image)
        self.current_image = image
    
    def predict_with_box(self, box: List[int]) -> np.ndarray:
        """
        使用边界框预测掩码
        
        Args:
            box: 边界框 [x1, y1, x2, y2]
            
        Returns:
            二值掩码，shape为(H, W)
        """
        if self.current_image is None:
            raise RuntimeError("Please call set_image() first")
        
        box_np = np.array(box)
        masks, scores, _ = self.predictor.predict(
            box=box_np,
            multimask_output=True
        )
        
        # 选择得分最高的掩码
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8)
    
    def predict_with_points(
        self, 
        point_coords: np.ndarray, 
        point_labels: np.ndarray
    ) -> np.ndarray:
        """
        使用点提示预测掩码
        
        Args:
            point_coords: 点坐标，shape为(N, 2) [x, y]
            point_labels: 点标签，shape为(N,)，1为正点，0为负点
            
        Returns:
            二值掩码，shape为(H, W)
        """
        if self.current_image is None:
            raise RuntimeError("Please call set_image() first")
        
        if len(point_coords) == 0:
            # 如果没有点，返回空掩码
            H, W = self.current_image.shape[:2]
            return np.zeros((H, W), dtype=np.uint8)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # 选择得分最高的掩码
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8)


class SAM2Wrapper(BaseSAMWrapper):
    """SAM2封装类（当可用时）"""
    
    def __init__(
        self,
        model_cfg: str = "sam2_hiera_l.yaml",
        checkpoint_path: str = "models/sam2_hiera_large.pt",
        device: Optional[Union[str, torch.device]] = None
    ):
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 not available. Please install sam2 package.")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        
        # 构建SAM2模型
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.current_image = None
        
        print(f"SAM2 loaded on {self.device}")
    
    def set_image(self, image: np.ndarray) -> None:
        """设置要分割的图像"""
        self.predictor.set_image(image)
        self.current_image = image
    
    def predict_with_box(self, box: List[int]) -> np.ndarray:
        """使用边界框预测掩码"""
        if self.current_image is None:
            raise RuntimeError("Please call set_image() first")
        
        box_np = np.array(box)
        masks, scores, _ = self.predictor.predict(
            box=box_np,
            multimask_output=True
        )
        
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8)
    
    def predict_with_points(
        self, 
        point_coords: np.ndarray, 
        point_labels: np.ndarray
    ) -> np.ndarray:
        """使用点提示预测掩码"""
        if self.current_image is None:
            raise RuntimeError("Please call set_image() first")
        
        if len(point_coords) == 0:
            H, W = self.current_image.shape[:2]
            return np.zeros((H, W), dtype=np.uint8)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8)


class SemanticGuidedSAM:
    """
    语义引导的SAM分割器
    
    结合Alpha-CLIP的语义评分和SAM的精确分割能力
    """
    
    def __init__(
        self,
        sam_wrapper: BaseSAMWrapper,
        alpha_clip_model = None
    ):
        """
        初始化语义引导SAM
        
        Args:
            sam_wrapper: SAM包装器实例
            alpha_clip_model: Alpha-CLIP推理模型
        """
        self.sam = sam_wrapper
        self.alpha_clip = alpha_clip_model
    
    def segment_with_semantic_points(
        self,
        image: np.ndarray,
        bbox: List[int],
        instance_text: str,
        grid_size: Tuple[int, int] = (3, 3),
        max_points: int = 12,
        pos_neg_ratio: float = 2.0,
        iterations: int = 3
    ) -> np.ndarray:
        """
        使用语义引导的点提示进行分割
        
        Args:
            image: RGB图像
            bbox: 初始边界框 [x1, y1, x2, y2]  
            instance_text: 目标实例的文本描述
            grid_size: 网格划分大小
            max_points: 每轮最大点数
            pos_neg_ratio: 正负点比例
            iterations: 迭代次数
            
        Returns:
            最终的分割掩码
        """
        # 设置图像
        self.sam.set_image(image)
        
        # 初始掩码（使用边界框）
        current_mask = self.sam.predict_with_box(bbox)
        
        if self.alpha_clip is None:
            return current_mask
        
        # 迭代优化
        for i in range(iterations):
            # 网格化并评分
            pos_points, neg_points = self._generate_semantic_points(
                image, current_mask, instance_text, 
                grid_size, max_points, pos_neg_ratio
            )
            
            if len(pos_points) == 0 and len(neg_points) == 0:
                break
            
            # 合并点坐标和标签
            all_points = []
            all_labels = []
            
            for pt in pos_points:
                all_points.append([pt[1], pt[0]])  # SAM expects [x, y]
                all_labels.append(1)
            
            for pt in neg_points:
                all_points.append([pt[1], pt[0]])
                all_labels.append(0)
            
            if len(all_points) > 0:
                point_coords = np.array(all_points)
                point_labels = np.array(all_labels)
                
                # 使用点提示更新掩码
                new_mask = self.sam.predict_with_points(point_coords, point_labels)
                
                # 检查收敛
                iou = self._compute_iou(current_mask, new_mask)
                if iou > 0.99:  # 收敛阈值
                    break
                    
                current_mask = new_mask
        
        return current_mask
    
    def _generate_semantic_points(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        instance_text: str,
        grid_size: Tuple[int, int],
        max_points: int,
        pos_neg_ratio: float
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """生成语义引导的正负点"""
        if self.alpha_clip is None:
            return [], []
        
        H, W = mask.shape
        gh, gw = grid_size
        
        # 网格化评分
        scores = []
        cells = []
        
        for i in range(gh):
            for j in range(gw):
                y1 = (i * H) // gh
                y2 = ((i + 1) * H) // gh  
                x1 = (j * W) // gw
                x2 = ((j + 1) * W) // gw
                
                # 创建子区域alpha掩码
                alpha = np.zeros_like(mask, dtype=np.float32)
                alpha[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                
                # 语义评分
                score = self.alpha_clip.score_region_with_templates(
                    image, alpha, instance_text
                )
                
                scores.append(score)
                cells.append((y1, y2, x1, x2))
        
        scores = np.array(scores)
        
        # 自适应阈值
        mu = scores.mean()
        sigma = scores.std() + 1e-6
        
        pos_thresh = mu + 0.5 * sigma
        neg_thresh = mu - 0.5 * sigma
        
        pos_cells = [cells[i] for i, s in enumerate(scores) if s >= pos_thresh]
        neg_cells = [cells[i] for i, s in enumerate(scores) if s <= neg_thresh]
        
        # 生成点
        max_pos = int(max_points * pos_neg_ratio / (1 + pos_neg_ratio))
        max_neg = max_points - max_pos
        
        pos_points = self._extract_positive_points(mask, pos_cells[:max_pos])
        neg_points = self._extract_negative_points(mask, neg_cells[:max_neg])
        
        return pos_points, neg_points
    
    def _extract_positive_points(
        self, 
        mask: np.ndarray, 
        cells: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int]]:
        """从正例网格中提取质心点"""
        points = []
        for y1, y2, x1, x2 in cells:
            sub_mask = mask[y1:y2, x1:x2]
            ys, xs = np.where(sub_mask > 0)
            if len(ys) > 0:
                cy = int(np.median(ys)) + y1
                cx = int(np.median(xs)) + x1
                points.append((cy, cx))
        return points
    
    def _extract_negative_points(
        self, 
        mask: np.ndarray, 
        cells: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int]]:
        """从负例网格中提取边界外点"""
        points = []
        
        # 计算边界
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        
        for y1, y2, x1, x2 in cells:
            # 在网格内寻找边界点
            sub_boundary = boundary[y1:y2, x1:x2]
            ys, xs = np.where(sub_boundary > 0)
            
            if len(ys) > 0:
                # 选择中位数位置的边界点
                idx = len(ys) // 2
                by, bx = ys[idx] + y1, xs[idx] + x1
                
                # 沿法向量向外寻找负点
                neg_point = self._find_outside_point(mask, by, bx, distance=10)
                if neg_point:
                    points.append(neg_point)
                    
        return points
    
    def _find_outside_point(
        self, 
        mask: np.ndarray, 
        y: int, 
        x: int, 
        distance: int = 10
    ) -> Optional[Tuple[int, int]]:
        """沿边界法向量向外寻找负点"""
        H, W = mask.shape
        
        # 简单方法：在边界点周围搜索第一个mask=0的点
        for r in range(1, distance + 1):
            for dy in [-r, 0, r]:
                for dx in [-r, 0, r]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W and 
                        mask[ny, nx] == 0):
                        return (ny, nx)
        return None
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个掩码的IoU"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-6)


def create_sam_wrapper(
    model_type: str = "sam", 
    **kwargs
) -> BaseSAMWrapper:
    """
    工厂函数：创建合适的SAM包装器
    
    Args:
        model_type: "sam" 或 "sam2"
        **kwargs: 传递给具体包装器的参数
        
    Returns:
        SAM包装器实例
    """
    if model_type == "sam":
        return SAMWrapper(**kwargs)
    elif model_type == "sam2":
        return SAM2Wrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")