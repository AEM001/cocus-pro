"""
阿里云百炼 Qwen VL API调用模块
支持通过DashScope SDK调用Qwen VL模型进行视觉语言理解
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional, List
import json
import base64

import numpy as np
from PIL import Image

from .base import VLMBase, try_parse_json
from .prompts import build_point_prompt


class QwenAPIVLM(VLMBase):
    """
    阿里云百炼 Qwen VL API 调用后端
    
    支持两种调用方式：
    1. DashScope SDK (推荐，支持文件路径直传)
    2. OpenAI兼容模式 (需要Base64编码)
    """

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "qwen-vl-plus-latest",
        use_dashscope: bool = True,
        high_resolution: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.0
    ):
        """
        初始化API客户端
        
        Args:
            api_key: API密钥（推荐通过环境变量/文件加载，不要硬编码）
            model_name: 模型名称，支持 qwen-vl-plus-latest, qwen-vl-max-latest 等
            use_dashscope: 是否使用DashScope SDK (推荐)，否则使用OpenAI兼容模式
            high_resolution: 是否开启高分辨率模式 (仅DashScope SDK支持)
            max_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.api_key = "sk-a20473de00b94f668cbcd4d2725947b2"
        
        self.model_name = model_name
        self.use_dashscope = use_dashscope
        self.high_resolution = high_resolution
        self.max_tokens = max(max_tokens, 512)  # 至少512 tokens
        self.temperature = temperature
        
        self._client = None
        self._temp_files: List[str] = []  # 用于清理临时文件
        
        # 初始化客户端
        self._init_client()

    def _init_client(self):
        """初始化API客户端"""
        if self.use_dashscope:
            try:
                import dashscope
                dashscope.api_key = self.api_key
                self._client = dashscope
                print(f"[INFO] 已初始化 DashScope SDK 客户端，模型: {self.model_name}")
            except ImportError:
                raise ImportError("DashScope SDK not installed. Run: pip install dashscope")
        else:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                print(f"[INFO] 已初始化 OpenAI 兼容客户端，模型: {self.model_name}")
            except ImportError:
                raise ImportError("OpenAI SDK not installed. Run: pip install openai")


    def propose_points(self, context_image_rgb: np.ndarray, instance: str, max_total: int = 10) -> Dict[str, Any]:
        """直接基于上下文图（叠加当前mask与ROI）让VLM输出正/负点。
        返回：{"pos_points": [(x,y),...], "neg_points": [(x,y),...], "raw_text": str}
        坐标以像素为单位，参照传入的 context_image_rgb 尺寸。
        为避免缩放引入的坐标歧义，提示词要求输出归一化坐标 xy_norm ∈ [0,1]，本函数再映射回像素。
        """
        H, W = int(context_image_rgb.shape[0]), int(context_image_rgb.shape[1])
        prompts = build_point_prompt(instance, W, H, max_total=max_total)
        text = self._inference([context_image_rgb], prompts["system"], prompts["user"])
        print(f"[DEBUG] Raw VLM points response: '{text}'")
        # Strip markdown fences if any
        data = try_parse_json(text)
        if not data:
            # 尝试修复截断的JSON
            try:
                fixed = self._fix_truncated_json(text)
                data = json.loads(fixed) if fixed and fixed != "{}" else {}
            except Exception:
                data = {}
        # Support two formats: separate lists, or combined "points" with type
        pos_items = data.get("pos_points", []) if isinstance(data, dict) else []
        neg_items = data.get("neg_points", []) if isinstance(data, dict) else []
        if not pos_items and not neg_items and isinstance(data, dict) and "points" in data:
            pts_combined = data.get("points", [])
            for it in pts_combined:
                if isinstance(it, dict) and it.get("type") in ("pos", "neg"):
                    if it["type"] == "pos":
                        pos_items.append(it)
                    else:
                        neg_items.append(it)
        pos_pts = self._extract_points(pos_items, W, H)
        neg_pts = self._extract_points(neg_items, W, H)
        # Respect VLM output: do not force count; return as is
        return {"pos_points": pos_pts, "neg_points": neg_pts, "raw_text": text}

    def _extract_points(self, items: List[Any], W: int, H: int) -> List[tuple[int,int]]:
        pts: List[tuple[int,int]] = []
        if not isinstance(items, (list, tuple)):
            return pts
        for it in items:
            try:
                x = y = None
                if isinstance(it, dict):
                    # Prefer pixel coordinates first
                    if "point_2d" in it and isinstance(it["point_2d"], (list, tuple)) and len(it["point_2d"]) == 2:
                        xp, yp = float(it["point_2d"][0]), float(it["point_2d"][1])
                        x = int(round(xp))
                        y = int(round(yp))
                    elif ("x" in it and "y" in it):
                        x = int(round(float(it["x"])))
                        y = int(round(float(it["y"])))
                    elif "xy_norm" in it and isinstance(it["xy_norm"], (list, tuple)) and len(it["xy_norm"]) == 2:
                        xn, yn = float(it["xy_norm"][0]), float(it["xy_norm"][1])
                        x = int(round(max(0.0, min(1.0, xn)) * (W - 1)))
                        y = int(round(max(0.0, min(1.0, yn)) * (H - 1)))
                    elif ("x_norm" in it and "y_norm" in it):
                        xn, yn = float(it["x_norm"]), float(it["y_norm"])
                        x = int(round(max(0.0, min(1.0, xn)) * (W - 1)))
                        y = int(round(max(0.0, min(1.0, yn)) * (H - 1)))
                if x is None or y is None:
                    continue
                x = max(0, min(W - 1, x))
                y = max(0, min(H - 1, y))
                if (x, y) not in pts:
                    pts.append((x, y))
            except Exception:
                continue
        return pts

        """If fewer than target points are returned, densify by jittering around existing points.
        Preference order: densify positives first, then negatives. Ensure uniqueness and image bounds.
        """
        pos_set = list(pos)
        neg_set = list(neg)
        offsets = [(dx,dy) for dx in (-3,-2,-1,1,2,3) for dy in (-3,-2,-1,1,2,3)]
        i = 0
        while len(pos_set) + len(neg_set) < target and (pos_set or neg_set):
            src_list = pos_set if pos_set else neg_set
            base = src_list[i % len(src_list)]
            jittered = False
            for dx, dy in offsets:
                nx = max(0, min(W - 1, base[0] + dx))
                ny = max(0, min(H - 1, base[1] + dy))
                if (nx, ny) not in pos_set and (nx, ny) not in neg_set:
                    if src_list is pos_set:
                        pos_set.append((nx, ny))
                    else:
                        neg_set.append((nx, ny))
                    jittered = True
                    break
            i += 1
            if not jittered and i > (len(pos_set)+len(neg_set))*len(offsets):
                break
        # Trim if overshoot
        total = len(pos_set) + len(neg_set)
        if total > target:
            cut = total - target
            if cut <= len(neg_set):
                neg_set = neg_set[:-cut]
            else:
                cut2 = cut - len(neg_set)
                neg_set = []
                pos_set = pos_set[:-cut2] if cut2 < len(pos_set) else pos_set[:1]
        return pos_set, neg_set

    def _inference(self, images: List[np.ndarray], system: str, user: str) -> str:
        """执行API推理"""
        try:
            if self.use_dashscope:
                return self._infer_dashscope(images, system, user)
            else:
                return self._infer_openai_compatible(images, system, user)
        except Exception as e:
            print(f"[ERROR] API推理失败: {e}")
            return "{}"

    def _infer_dashscope(self, images: List[np.ndarray], system: str, user: str) -> str:
        """使用DashScope SDK进行推理"""
        # 保存图像到临时文件
        image_paths = []
        for i, img in enumerate(images):
            temp_path = self._save_temp_image(img, f"temp_img_{i}")
            image_paths.append(f"file://{temp_path}")
        
        # 构建消息
        content = []
        for img_path in image_paths:
            content.append({"image": img_path})
        content.append({"text": user})
        
        messages = [
            {"role": "system", "content": [{"text": system}]},
            {"role": "user", "content": content}
        ]
        
        # API调用参数 - 增加tokens数量防止截断
        call_params = {
            "api_key": self.api_key,
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max(self.max_tokens, 512),  # 至少512 tokens
            "temperature": self.temperature,
        }
        
        # 如果启用高分辨率模式
        if self.high_resolution:
            call_params["vl_high_resolution_images"] = True
        
        try:
            response = self._client.MultiModalConversation.call(**call_params)
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                if isinstance(content, list) and content:
                    return content[0].get("text", "")
                return str(content)
            else:
                print(f"[ERROR] API调用失败，状态码: {response.status_code}")
                print(f"[ERROR] 错误信息: {response.message}")
                return "{}"
                
        except Exception as e:
            print(f"[ERROR] DashScope API调用异常: {e}")
            return "{}"

    def _infer_openai_compatible(self, images: List[np.ndarray], system: str, user: str) -> str:
        """使用OpenAI兼容模式进行推理"""
        # 构建消息内容
        content = []
        for img in images:
            img_b64 = self._numpy_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
        content.append({"type": "text", "text": user})
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content}
        ]
        
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max(self.max_tokens, 512),  # 至少512 tokens
                temperature=self.temperature,
            )
            
            return completion.choices[0].message.content or ""
            
        except Exception as e:
            print(f"[ERROR] OpenAI兼容API调用异常: {e}")
            return "{}"

    def _save_temp_image(self, img_array: np.ndarray, prefix: str = "temp") -> str:
        """保存numpy数组为临时图像文件"""
        # 确保图像是uint8格式
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        
        # 创建PIL图像
        pil_img = Image.fromarray(img_array)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', prefix=prefix, delete=False) as tmp_file:
            pil_img.save(tmp_file.name)
            self._temp_files.append(tmp_file.name)  # 记录以便后续清理
            return tmp_file.name

    def _numpy_to_base64(self, img_array: np.ndarray) -> str:
        """将numpy数组转换为base64字符串"""
        # 确保图像是uint8格式
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        
        # 转换为PIL图像
        pil_img = Image.fromarray(img_array)
        
        # 编码为base64
        import io
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_b64





    def _fix_truncated_json(self, text: str) -> str:
        """尝试修复截断的JSON"""
        fixed_text = text.strip()
        
        # 提取JSON代码块
        if "```json" in fixed_text:
            json_start = fixed_text.find("{")
            if json_start >= 0:
                fixed_text = fixed_text[json_start:]
                # 移除可能的markdown结尾
                if "```" in fixed_text:
                    fixed_text = fixed_text[:fixed_text.find("```")]
        
        # 如果没有开始的花括号，直接返回空JSON
        if "{" not in fixed_text:
            return "{}"
        
        # 修复截断的JSON
        if not fixed_text.endswith("}"):
            open_braces = fixed_text.count("{")
            close_braces = fixed_text.count("}")
            open_brackets = fixed_text.count("[")
            close_brackets = fixed_text.count("]")
            
            # 修复未闭合的字符串 - 更智能的处理
            quote_count = fixed_text.count('"')
            if quote_count % 2 == 1:
                # 查找最后一个未闭合的引号位置
                last_quote_pos = fixed_text.rfind('"')
                # 检查是否是属性名后面缺少值
                after_last_quote = fixed_text[last_quote_pos+1:].strip()
                if after_last_quote.startswith(':') and not after_last_quote[1:].strip():
                    # 属性名后缺少值，添加空字符串
                    fixed_text += '""'
                else:
                    # 普通的未闭合字符串
                    fixed_text += '"'
            
            # 闭合数组和对象
            while close_brackets < open_brackets:
                fixed_text += "]"
                close_brackets += 1
            while close_braces < open_braces:
                fixed_text += "}"
                close_braces += 1
        
        return fixed_text

    def _validate_anchor_response(self, result: dict) -> dict:
        """验证锚点响应格式"""
        if not isinstance(result.get("anchors_to_refine"), list):
            print("[ERROR] anchors_to_refine is not a list, using empty list")
            result["anchors_to_refine"] = []
        
        valid = []
        for i, anchor in enumerate(result["anchors_to_refine"]):
            if isinstance(anchor, dict) and 'id' in anchor:
                try:
                    anchor['id'] = int(anchor['id'])
                    if 'score' in anchor:
                        try:
                            anchor['score'] = float(anchor['score'])
                        except Exception:
                            anchor.pop('score', None)
                    valid.append(anchor)
                except Exception:
                    print(f"[ERROR] Invalid anchor id at index {i}")
            else:
                print(f"[ERROR] Invalid anchor format at index {i}, removing")
        
        result["anchors_to_refine"] = valid
        return result

    def _validate_edit_response(self, result: dict) -> dict:
        """验证编辑响应格式"""
        if not isinstance(result.get("edits"), list):
            print(f"[ERROR] edits is not a list for anchor {result.get('anchor_id')}, using empty list")
            result["edits"] = []
        
        valid = []
        for i, edit in enumerate(result["edits"]):
            if isinstance(edit, dict) and 'region_id' in edit and 'action' in edit:
                try:
                    edit['region_id'] = int(edit['region_id'])
                    if edit['region_id'] in [1, 2, 3, 4] and edit['action'] in ['pos', 'neg']:
                        valid.append(edit)
                    else:
                        print(f"[ERROR] Invalid edit values at index {i} for anchor {result.get('anchor_id')}")
                except Exception:
                    print(f"[ERROR] Invalid edit format at index {i} for anchor {result.get('anchor_id')}")
            else:
                print(f"[ERROR] Invalid edit format at index {i} for anchor {result.get('anchor_id')}")
        
        result["edits"] = valid
        return result

    def cleanup(self):
        """清理临时文件"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"[WARN] Failed to remove temp file {temp_file}: {e}")
        self._temp_files.clear()

    def __del__(self):
        """析构函数，确保清理临时文件"""
        self.cleanup()