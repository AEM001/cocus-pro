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
from .prompts import build_peval_prompt, build_pgen_prompt


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

    def __init__(self, mode: str = "server", server_url: Optional[str] = None, model_dir: Optional[str] = None, gen_max_new_tokens: int = 96):
        self.mode = mode
        self.server_url = server_url or "http://127.0.0.1:8000/generate"
        self.model_dir = model_dir
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        self._local_model = None  # Will be set to True after successful loading
        self._model = None       # The actual model instance
        self._processor = None   # The processor instance

    # -------- public API --------
    def peval(self, image_overlay_rgb: np.ndarray, depth_vis: Optional[np.ndarray], instance: str) -> Dict[str, Any]:
        prompts = build_peval_prompt(instance)
        text = self._inference([image_overlay_rgb], prompts["system"], prompts["user"])  # type: ignore
        data = try_parse_json(text)
        # sane defaults
        data.setdefault("missing_parts", [])
        data.setdefault("over_segments", [])
        data.setdefault("boundary_quality", "")
        data.setdefault("key_cues", [])
        return data

    def pgen(self, patch_rgb: np.ndarray, instance: str, key_cues: str) -> Dict[str, Any]:
        prompts = build_pgen_prompt(instance, key_cues)
        text = self._inference([patch_rgb], prompts["system"], prompts["user"])  # type: ignore
        data = try_parse_json(text)
        # normalize
        is_target = bool(data.get("is_target", False))
        conf = float(data.get("conf", 0.0))
        conf = max(0.0, min(1.0, conf))
        return {"is_target": is_target, "conf": conf}

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
            # Convert numpy images to PIL Images
            pil_images = []
            for img in images:
                from PIL import Image
                pil_img = Image.fromarray(img.astype(np.uint8))
                pil_images.append(pil_img)
            
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
            
            # Process inputs
            inputs = self._processor(
                text=[text], 
                images=pil_images if pil_images else None, 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = inputs.to(self._model.device)
            
            # Generate response
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=int(self.gen_max_new_tokens),
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self._processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
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
            from transformers import AutoModelForVision2Seq, AutoProcessor
            from transformers import AutoConfig
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
                trust_remote_code=True
            )
            
            # Load model (works for 3B/7B). Prefer FP16 on CUDA, CPU fallback otherwise.
            import torch
            dtype = torch.float16 if torch.cuda.is_available() else None
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_dir,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
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

