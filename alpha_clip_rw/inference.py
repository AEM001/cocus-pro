"""
推理封装层，实现语义分割和区域评分
实现权重加载和预处理
实现文字、图像、Alpha通道的输入编码
相似度计算与多模版评分，然后进行聚合，得到更为robust的评分结果
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, List, Optional, Tuple
import warnings

from .alpha_clip import load, tokenize, available_models
from .alpha_clip import _transform, mask_transform


class AlphaCLIPInference:
    """
    Alpha-CLIP推理封装类，支持RGBA输入和语义评分
    
    主要用于语义分割中的区域评分，支持：
    1. 标准3通道RGB输入 
    2. 4通道RGBA输入（Alpha通道作为软注意力掩码）
    3. 批量文本-图像相似度计算
    4. 多模板文本增强
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        alpha_vision_ckpt_pth: Optional[str] = "AUTO",
        device: Optional[Union[str, torch.device]] = None,
        lora_adapt: bool = False,
        rank: int = 16
    ):
        """
        初始化Alpha-CLIP模型
        
        Args:
            model_name: CLIP模型名称
            alpha_vision_ckpt_pth: Alpha视觉编码器检查点路径
            device: 计算设备
            lora_adapt: 是否使用LoRA适配
            rank: LoRA秩数
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        
        # 解析 Alpha 分支权重：强制要求本地存在，便于脱网运行
        ckpt_path = None
        if isinstance(alpha_vision_ckpt_pth, str) and alpha_vision_ckpt_pth.upper() == "AUTO":
            import os
            ckpt_path = os.environ.get("ALPHA_CLIP_ALPHA_CKPT")
            if ckpt_path is None:
                default_path = os.path.join("checkpoints", "clip_l14_336_grit_20m_4xe.pth")
                if os.path.isfile(default_path):
                    ckpt_path = default_path
            # 若仍未找到，抛错并提示用户准备文件
            if ckpt_path is None or not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    "未找到 Alpha-CLIP 视觉分支权重。请下载并放置到 'checkpoints/clip_l14_336_grit_20m_4xe.pth'，"
                    "或通过环境变量 ALPHA_CLIP_ALPHA_CKPT 指定完整路径。"
                )
        else:
            ckpt_path = alpha_vision_ckpt_pth

        # 若本地提供了 CLIP 主干权重，则优先使用本地文件，避免在线下载
        import os as _os
        if model_name == "ViT-L/14@336px":
            local_clip = _os.environ.get("CLIP_MODEL_PATH") or _os.path.join("checkpoints", "ViT-L-14-336px.pt")
            if _os.path.isfile(local_clip):
                model_name = local_clip

        # 加载模型（注意：alpha_vision_ckpt_pth 强制为本地文件）
        self.model, self.preprocess = load(
            model_name,
            alpha_vision_ckpt_pth=ckpt_path,
            device=self.device,
            lora_adapt=lora_adapt,
            rank=rank,
        )
        
        # 获取输入分辨率
        self.input_resolution = self.model.visual.input_resolution
        
        # Alpha通道变换
        self.mask_preprocess = mask_transform(self.input_resolution)
        
        # 缓存文本特征
        self._text_cache = {}
        
        print(f"Alpha-CLIP loaded: {model_name} on {self.device}")
        print(f"Input resolution: {self.input_resolution}")
    
    def encode_text(self, texts: Union[str, List[str]], use_cache: bool = True) -> torch.Tensor:
        """
        编码文本为特征向量
        
        Args:
            texts: 单个文本或文本列表
            use_cache: 是否使用缓存
            
        Returns:
            归一化的文本特征张量 [N, D]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 检查缓存
        if use_cache:
            cache_key = tuple(texts)
            if cache_key in self._text_cache:
                return self._text_cache[cache_key]
        
        # 分词和编码
        text_tokens = tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        # 缓存结果
        if use_cache:
            self._text_cache[cache_key] = text_features
            
        return text_features
    
    def encode_image_rgb(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        编码RGB图像
        
        Args:
            image: RGB图像 (H, W, 3) 或 PIL Image
            
        Returns:
            归一化的图像特征 [1, D]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
        return image_features
    
    def encode_image_rgba(
        self, 
        image: Union[np.ndarray, Image.Image], 
        alpha: Union[np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        编码RGBA图像（Alpha通道作为注意力掩码）
        
        Args:
            image: RGB图像 (H, W, 3) 或 PIL Image
            alpha: Alpha掩码 (H, W) 或 PIL Image，值范围[0,1]
            
        Returns:
            归一化的图像特征 [1, D]
        """
        # 预处理RGB图像
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
        
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        
        # 预处理Alpha掩码
        if isinstance(alpha, np.ndarray):
            if alpha.max() <= 1.0:
                alpha = (alpha * 255).astype(np.uint8)
            alpha_pil = Image.fromarray(alpha, mode='L')
        else:
            alpha_pil = alpha
        
        alpha_tensor = self.mask_preprocess(alpha_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 使用Alpha-CLIP的4通道编码
            image_features = self.model.visual(image_tensor, alpha_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
        return image_features
    
    def compute_similarity(
        self, 
        image: Union[np.ndarray, Image.Image],
        text: Union[str, List[str]],
        alpha: Optional[Union[np.ndarray, Image.Image]] = None,
        temperature: float = 0.01
    ) -> Union[float, torch.Tensor]:
        """
        计算图像-文本相似度
        
        Args:
            image: RGB图像
            text: 文本或文本列表  
            alpha: 可选的Alpha掩码
            temperature: 温度参数
            
        Returns:
            相似度分数，单个文本返回float，多个文本返回tensor
        """
        # 编码图像
        if alpha is not None:
            image_features = self.encode_image_rgba(image, alpha)
        else:
            image_features = self.encode_image_rgb(image)
        
        # 编码文本
        text_features = self.encode_text(text)
        
        # 计算相似度
        similarity = (image_features @ text_features.T) / temperature
        similarity = torch.softmax(similarity, dim=-1)
        
        if isinstance(text, str):
            return float(similarity.squeeze())
        else:
            return similarity.squeeze()
    
    def score_region_with_templates(
        self,
        image: Union[np.ndarray, Image.Image],
        alpha: Union[np.ndarray, Image.Image], 
        instance_text: str,
        templates: Optional[List[str]] = None,
        aggregate: str = 'mean'
    ) -> float:
        """
        使用多个模板对区域进行语义评分
        
        Args:
            image: RGB图像
            alpha: Alpha掩码，表示要评分的区域
            instance_text: 实例类别文本
            templates: 文本模板列表，{}处会被instance_text替换
            aggregate: 聚合方式：'mean', 'max', 'weighted_mean'
            
        Returns:
            聚合后的语义相似度分数
        """
        if templates is None:
            templates = [
                "a photo of the {} camouflaged in the background.",
                "a photo of the {}.",
                "the {} blending into surroundings.",
            ]
        
        # 生成所有提示
        prompts = [template.format(instance_text) for template in templates]
        
        # 批量计算相似度
        similarities = self.compute_similarity(image, prompts, alpha)
        
        # 聚合结果
        if aggregate == 'mean':
            return float(similarities.mean())
        elif aggregate == 'max':
            return float(similarities.max())
        elif aggregate == 'weighted_mean':
            # 简单加权：第一个模板权重更高
            weights = torch.tensor([2.0, 1.0, 1.0], device=similarities.device)
            weights = weights[:len(similarities)] / weights[:len(similarities)].sum()
            return float((similarities * weights).sum())
        else:
            return float(similarities.mean())
    
    def clear_cache(self):
        """清除文本特征缓存"""
        self._text_cache.clear()
        
    @property
    def available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return available_models()
