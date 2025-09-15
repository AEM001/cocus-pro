"""
【Qwen2.5-VL模型集成模块】
作用：集成Qwen2.5-VL-7B-Instruct视觉语言模型，实现图像分析和点评估
核心功能：
  - 双模式支持：本地推理（AWQ量化）和HTTP服务器模式
  - 图像编码：将numpy数组转换为PNG格式进行传输
  - 提示工程：构建PEval（整体评估）和PGen（点生成）的提示词
  - 结果解析：将VLM输出解析为结构化JSON数据

与上下游模块关系：
  - 接收patches.py提取的图像块
  - 输出结构化分析结果给select_points.py
  - 通过base.py定义的VLMBase接口实现标准化调用

技术特点：
  - 零依赖启动：默认返回空JSON，避免网络调用
  - 本地推理支持：完整的Qwen2.5-VL-AWQ加载和推理流程
  - 错误恢复：模型加载失败时优雅降级
  - 标准化接口：统一的peval/pgen方法签名
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import base64
import io
import os

import numpy as np

from .base import VLMBase, try_parse_json
from .prompts import build_anchor_prompt, build_quadrant_prompt


def _to_png_bytes(img_rgb: np.ndarray) -> bytes:
    from PIL import Image

    pil = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


class QwenVLM(VLMBase):
    """
    Qwen2.5-VL integration scaffold (supports 3B/7B checkpoints).

    Modes:
    - server: POST to a local HTTP endpoint you host, with fields {images: [b64], system, user}
    - local:  load a local model from `model_dir`.

    Parameters
    - mode: "server" | "local"
    - server_url: e.g., http://127.0.0.1:8000/generate  (no network calls here by default)
    - model_dir: path to local Qwen2.5-VL weights (3B or 7B)
    - gen_max_new_tokens: generation cap to reduce memory/time
    """

    def __init__(self, mode: str = "server", server_url: Optional[str] = None, model_dir: Optional[str] = None, gen_max_new_tokens: int = 96, do_sample: bool = False):
        self.mode = mode
        self.server_url = server_url or "http://127.0.0.1:8000/generate"
        self.model_dir = model_dir
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        self.do_sample = bool(do_sample)
        self._local_model = None  # Will be set to True after successful loading
        self._model = None       # The actual model instance
        self._processor = None   # The processor instance

    # -------- 重构后的API，只支持anchor/quadrant指导 --------

    def choose_anchors(self, image_with_anchors_rgb: np.ndarray, instance: str, global_reason: Optional[str] = None) -> Dict[str, Any]:
        prompts = build_anchor_prompt(instance, global_reason)
        text = self._inference([image_with_anchors_rgb], prompts["system"], prompts["user"])  # type: ignore
        
        # DEBUG: Print raw VLM response
        print(f"[DEBUG] Raw VLM anchor response: '{text}'")
        print(f"[DEBUG] Response length: {len(text)} chars")
        
        data = try_parse_json(text)
        anchors = data.get("anchors_to_refine", [])

        # Normalize if available
        norm: list[dict] = []
        for it in anchors if isinstance(anchors, (list, tuple)) else []:
            try:
                norm.append({"id": int(it.get("id", 0)), "reason": str(it.get("reason", ""))})
            except Exception:
                pass

        # Heuristic salvage if empty
        if not norm:
            import re, json as _json
            print(f"[DEBUG] No anchors parsed from JSON, attempting heuristic salvage...")
            
            # Try to fix common truncation issues and parse JSON
            fixed_text = text.strip()
            # If JSON starts with markdown code block, extract it
            if "```json" in fixed_text:
                json_start = fixed_text.find("{")
                if json_start >= 0:
                    fixed_text = fixed_text[json_start:]
            
            # Try to fix truncated JSON by adding missing closures
            if not fixed_text.endswith("}"):
                # Count braces to see what we're missing
                open_braces = fixed_text.count("{")
                close_braces = fixed_text.count("}")
                open_brackets = fixed_text.count("[")
                close_brackets = fixed_text.count("]")
                
                # Add missing quotes if string is cut off
                if fixed_text.count('"') % 2 == 1:
                    fixed_text += '"'
                
                # Close arrays and objects as needed
                while close_brackets < open_brackets:
                    fixed_text += "]"
                    close_brackets += 1
                while close_braces < open_braces:
                    fixed_text += "}"
                    close_braces += 1
                    
                print(f"[DEBUG] Attempting to fix truncated JSON: '{fixed_text}'")
                
                try:
                    data = _json.loads(fixed_text)
                    anchors = data.get("anchors_to_refine", [])
                    for it in anchors if isinstance(anchors, (list, tuple)) else []:
                        try:
                            anchor_id = it.get("id", 0)
                            # Handle string IDs
                            if isinstance(anchor_id, str):
                                anchor_id = int(anchor_id)
                            norm.append({"id": int(anchor_id), "reason": str(it.get("reason", ""))})
                        except Exception:
                            pass
                    print(f"[DEBUG] Fixed and parsed {len(norm)} anchors from truncated JSON")
                except Exception as e:
                    print(f"[DEBUG] JSON fix attempt failed: {e}")
            
            # Fallback: try to directly capture the list after anchors_to_refine
            if not norm:
                m = re.search(r"anchors\s*_?to\s*_?refine\s*:\s*(\[.*?\])", text, re.IGNORECASE | re.DOTALL)
                if m:
                    try:
                        sub = m.group(1)
                        parsed = _json.loads(sub)
                        if isinstance(parsed, list):
                            for it in parsed:
                                try:
                                    anchor_id = it.get("id", 0)
                                    if isinstance(anchor_id, str):
                                        anchor_id = int(anchor_id)
                                    norm.append({"id": int(anchor_id), "reason": str(it.get("reason", ""))})
                                except Exception:
                                    pass
                        print(f"[DEBUG] Salvaged {len(norm)} anchors from regex pattern")
                    except Exception as e:
                        print(f"[DEBUG] Regex salvage failed: {e}")
                else:
                    print(f"[DEBUG] No anchor pattern found in response")
        
        print(f"[DEBUG] Final anchors count: {len(norm)}")
        return {"anchors_to_refine": norm, "raw_text": text}

    def quadrant_edits(self, quadrant_crop_rgb: np.ndarray, instance: str, anchor_id: int, global_reason: Optional[str] = None, anchor_reason: Optional[str] = None) -> Dict[str, Any]:
        prompts = build_quadrant_prompt(instance, anchor_id, global_reason, anchor_reason)
        text = self._inference([quadrant_crop_rgb], prompts["system"], prompts["user"])  # type: ignore
        
        # DEBUG: Print raw VLM response
        print(f"[DEBUG] Raw VLM quadrant response for anchor {anchor_id}: '{text}'")
        print(f"[DEBUG] Response length: {len(text)} chars")
        
        data = try_parse_json(text)
        edits = data.get("edits", [])
        norm = []
        for it in edits if isinstance(edits, (list, tuple)) else []:
            try:
                rid = int(it.get("region_id", 0))
                act = str(it.get("action", ""))
                why = str(it.get("why", ""))
                if rid in (1, 2, 3, 4) and act in ("pos", "neg"):
                    norm.append({"region_id": rid, "action": act, "why": why})
            except Exception:
                pass
        
        # If no edits parsed, try to fix truncated JSON similar to anchor processing
        if not norm:
            import re, json as _json
            print(f"[DEBUG] No edits parsed from JSON, attempting repair for anchor {anchor_id}...")
            
            # Try to fix common truncation issues and parse JSON
            fixed_text = text.strip()
            # If JSON starts with markdown code block, extract it
            if "```json" in fixed_text:
                json_start = fixed_text.find("{")
                if json_start >= 0:
                    fixed_text = fixed_text[json_start:]
            
            # Try to fix truncated JSON by adding missing closures
            if not fixed_text.endswith("}"):
                # Count braces to see what we're missing
                open_braces = fixed_text.count("{")
                close_braces = fixed_text.count("}")
                open_brackets = fixed_text.count("[")
                close_brackets = fixed_text.count("]")
                
                # Add missing quotes if string is cut off
                if fixed_text.count('"') % 2 == 1:
                    fixed_text += '"'
                
                # Close arrays and objects as needed
                while close_brackets < open_brackets:
                    fixed_text += "]"
                    close_brackets += 1
                while close_braces < open_braces:
                    fixed_text += "}"
                    close_braces += 1
                    
                print(f"[DEBUG] Attempting to fix truncated JSON: '{fixed_text}'")
                
                try:
                    data = _json.loads(fixed_text)
                    edits = data.get("edits", [])
                    for it in edits if isinstance(edits, (list, tuple)) else []:
                        try:
                            rid = int(it.get("region_id", 0))
                            act = str(it.get("action", ""))
                            why = str(it.get("why", ""))
                            if rid in (1, 2, 3, 4) and act in ("pos", "neg"):
                                norm.append({"region_id": rid, "action": act, "why": why})
                        except Exception:
                            pass
                    print(f"[DEBUG] Fixed and parsed {len(norm)} edits from truncated JSON")
                except Exception as e:
                    print(f"[DEBUG] JSON fix attempt failed: {e}")
        
        print(f"[DEBUG] Final edits count for anchor {anchor_id}: {len(norm)}")
        return {"anchor_id": int(anchor_id), "edits": norm, "raw_text": text}

    # -------- internal helpers --------
    def _inference(self, images: list[np.ndarray], system: str, user: str) -> str:
        if self.mode == "server":
            return self._infer_via_server(images, system, user)
        elif self.mode == "local":
            return self._infer_via_local(images, system, user)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _infer_via_server(self, images: list[np.ndarray], system: str, user: str) -> str:
        # Placeholder: do not perform HTTP by default (no network). Return empty JSON.
        # To enable, implement your local server to accept this payload and return a JSON string.
        # Example payload shape:
        # {
        #   "system": system,
        #   "user": user,
        #   "images": ["data:image/png;base64,<...>", ...]
        # }
        # Then parse response["text"].
        _ = (images, system, user)
        return "{}"

    def _infer_via_local(self, images: list[np.ndarray], system: str, user: str) -> str:
        # Local python inference with Qwen2.5-VL-AWQ
        if self._local_model is None:
            self._load_local_model()
        
        if self._local_model is None:
            print(f"[WARN] Model not loaded from {self.model_dir}, returning empty JSON")
            return "{}"
            
        try:
            import torch
            from PIL import Image
            try:
                from qwen_vl_utils import process_vision_info
            except Exception:
                process_vision_info = None
            # Convert numpy images to PIL Images
            pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
            
            # Prepare messages in Qwen2.5-VL format
            content = []
            for pil_img in pil_images:
                content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": user})
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content}
            ]
            
            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs per Qwen docs
            image_inputs = None
            video_inputs = None
            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = pil_images if pil_images else None
                video_inputs = None
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            inputs = inputs.to(self._model.device)
            
            # Generate response (greedy by default to avoid CUDA multinomial assertions)
            gen_kwargs = {
                "max_new_tokens": int(self.gen_max_new_tokens),
                "pad_token_id": self._processor.tokenizer.eos_token_id,
                "eos_token_id": self._processor.tokenizer.eos_token_id,
                "do_sample": bool(self.do_sample),
            }
            if self.do_sample:
                gen_kwargs.update({"temperature": 0.4, "top_p": 0.9})
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Clear GPU cache to free memory for other operations
            torch.cuda.empty_cache()
            
            return output_text.strip()
            
        except Exception as e:
            print(f"[WARN] Local inference failed: {e}")
            return "{}"
    
    def _load_local_model(self):
        """Load the local Qwen2.5-VL-AWQ model"""
        if not self.model_dir or not os.path.exists(self.model_dir):
            print(f"[WARN] Model directory not found: {self.model_dir}")
            return
            
        try:
            from transformers import AutoProcessor
            from transformers import Qwen2_5_VLForConditionalGeneration
            import torch
            
            print(f"[INFO] Loading Qwen2.5-VL-AWQ model from {self.model_dir}")
            
            # Check if model files exist
            config_path = os.path.join(self.model_dir, "config.json")
            if not os.path.exists(config_path):
                print(f"[WARN] config.json not found in {self.model_dir}")
                return
            
            # Load processor and model
            self._processor = AutoProcessor.from_pretrained(
                self.model_dir,
            )
            
            # Load model (3B/7B). Prefer FP16 on CUDA, CPU fallback otherwise.
            dtype = torch.float16 if torch.cuda.is_available() else None
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype=dtype,
                device_map="auto",
            )
            self._model.eval()
            
            print(f"[INFO] Model loaded successfully on device: {self._model.device}")
            self._local_model = True  # Mark as successfully loaded
            
        except ImportError as e:
            print(f"[WARN] Required packages not installed: {e}")
            print("[INFO] Please ensure transformers>=4.37.0 and torch are installed")
        except Exception as e:
            print(f"[WARN] Failed to load model: {e}")
            self._local_model = None
    
    def unload_model(self):
        """Unload the VLM model to free GPU memory"""
        if self._model is not None:
            print(f"[INFO] Unloading VLM model to free GPU memory")
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._local_model = None
        # Clear GPU cache
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

