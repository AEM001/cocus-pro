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
from .prompts import build_anchor_prompt, build_quadrant_prompt


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
            api_key: API密钥，如果为None则从环境变量DASHSCOPE_API_KEY读取
            model_name: 模型名称，支持 qwen-vl-plus-latest, qwen-vl-max-latest 等
            use_dashscope: 是否使用DashScope SDK (推荐)，否则使用OpenAI兼容模式
            high_resolution: 是否开启高分辨率模式 (仅DashScope SDK支持)
            max_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter")
            
        self.model_name = model_name
        self.use_dashscope = use_dashscope
        self.high_resolution = high_resolution
        self.max_tokens = max(max_tokens, 512)  # 确保至少256 tokens
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

    def choose_anchors(self, image_with_anchors_rgb: np.ndarray, instance: str, global_reason: Optional[str] = None) -> Dict[str, Any]:
        """选择锚点的API调用"""
        # 构建语义上下文
        semantic_context = None
        if global_reason:
            semantic_context = {
                'salient_cues': [f'{instance} anatomical features', 'natural body contours', 'organic textures'],
                'distractors': ['background camouflage patterns', 'environmental mimicry', 'habitat elements'],
                'shape_prior': f'natural {instance} body structure and proportions',
                'scene_context': f'{instance} in natural camouflaged habitat'
            }
        
        prompts = build_anchor_prompt(instance, global_reason, semantic=semantic_context)
        text = self._inference([image_with_anchors_rgb], prompts["system"], prompts["user"])
        
        # DEBUG: Print raw VLM response
        print(f"[DEBUG] Raw VLM anchor response: '{text}'")
        print(f"[DEBUG] Response length: {len(text)} chars")
        
        data = try_parse_json(text)
        anchors = data.get("anchors_to_refine", [])

        # 标准化锚点数据
        norm = self._normalize_anchors(anchors)
        
        # 如果解析失败，尝试修复
        if not norm:
            norm = self._salvage_anchors_from_text(text)
        
        print(f"[DEBUG] Final anchors count: {len(norm)}")
        
        # 返回标准格式
        result = {"anchors_to_refine": norm, "raw_text": text}
        return self._validate_anchor_response(result)

    def quadrant_edits(self, quadrant_crop_rgb: np.ndarray, instance: str, anchor_id: int, 
                      global_reason: Optional[str] = None, anchor_reason: Optional[str] = None) -> Dict[str, Any]:
        """象限编辑的API调用"""
        # 构建语义上下文
        semantic_context = None
        if global_reason:
            semantic_context = {
                'salient_cues': [f'{instance} anatomical structures', 'biological boundaries', 'organic patterns'],
                'distractors': ['environmental textures', 'background camouflage', 'habitat patterns'],
                'shape_prior': f'natural {instance} anatomy and form',
                'scene_context': f'{instance} blending with natural habitat'
            }
        
        prompts = build_quadrant_prompt(instance, anchor_id, global_reason, 
                                       semantic=semantic_context, anchor_hint=anchor_reason)
        text = self._inference([quadrant_crop_rgb], prompts["system"], prompts["user"])
        
        # DEBUG: Print raw VLM response
        print(f"[DEBUG] Raw VLM quadrant response for anchor {anchor_id}: '{text}'")
        print(f"[DEBUG] Response length: {len(text)} chars")
        
        data = try_parse_json(text)
        edits = data.get("edits", [])
        
        # 标准化编辑指令
        norm = self._normalize_edits(edits)
        
        # 如果解析失败，尝试修复
        if not norm:
            norm = self._salvage_edits_from_text(text, anchor_id)
        
        print(f"[DEBUG] Final edits count for anchor {anchor_id}: {len(norm)}")
        
        # 返回标准格式
        result = {"anchor_id": int(anchor_id), "edits": norm, "raw_text": text}
        return self._validate_edit_response(result)

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

    def _normalize_anchors(self, anchors) -> List[dict]:
        """标准化锚点响应"""
        norm = []
        for it in anchors if isinstance(anchors, (list, tuple)) else []:
            try:
                aid = int(it.get("id", 0))
                rsn = str(it.get("reason", ""))
                sc = it.get("score", None)
                try:
                    sc = float(sc) if sc is not None else None
                except Exception:
                    sc = None
                
                anchor_dict = {"id": aid, "reason": rsn}
                if sc is not None:
                    anchor_dict["score"] = sc
                norm.append(anchor_dict)
            except Exception:
                continue
        
        # 如果有分数，按分数降序排列
        if norm and any("score" in a for a in norm):
            norm = sorted(norm, key=lambda x: -(x.get("score", -1.0)))
        
        return norm

    def _normalize_edits(self, edits) -> List[dict]:
        """标准化编辑指令"""
        norm = []
        for it in edits if isinstance(edits, (list, tuple)) else []:
            try:
                rid = int(it.get("region_id", 0))
                act = str(it.get("action", ""))
                why = str(it.get("why", ""))
                if rid in (1, 2, 3, 4) and act in ("pos", "neg"):
                    norm.append({"region_id": rid, "action": act, "why": why})
            except Exception:
                continue
        return norm

    def _salvage_anchors_from_text(self, text: str) -> List[dict]:
        """从文本中抢救锚点信息"""
        import re
        
        norm = []
        # 尝试修复截断的JSON
        fixed_text = self._fix_truncated_json(text)
        if fixed_text != text:
            try:
                data = json.loads(fixed_text)
                anchors = data.get("anchors_to_refine", [])
                norm = self._normalize_anchors(anchors)
                if norm:
                    print(f"[DEBUG] Fixed and parsed {len(norm)} anchors from truncated JSON")
                    return norm
            except Exception as e:
                print(f"[DEBUG] JSON fix attempt failed: {e}")
        
        # 正则表达式抢救
        m = re.search(r"anchors\s*_?to\s*_?refine\s*:\s*(\[.*?\])", text, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                sub = m.group(1)
                parsed = json.loads(sub)
                if isinstance(parsed, list):
                    norm = self._normalize_anchors(parsed)
                    print(f"[DEBUG] Salvaged {len(norm)} anchors from regex pattern")
            except Exception as e:
                print(f"[DEBUG] Regex salvage failed: {e}")
        
        return norm

    def _salvage_edits_from_text(self, text: str, anchor_id: int) -> List[dict]:
        """从文本中抢救编辑信息"""
        norm = []
        # 尝试修复截断的JSON
        fixed_text = self._fix_truncated_json(text)
        if fixed_text != text:
            try:
                data = json.loads(fixed_text)
                edits = data.get("edits", [])
                norm = self._normalize_edits(edits)
                if norm:
                    print(f"[DEBUG] Fixed and parsed {len(norm)} edits from truncated JSON for anchor {anchor_id}")
            except Exception as e:
                print(f"[DEBUG] JSON fix attempt failed for anchor {anchor_id}: {e}")
        
        return norm

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