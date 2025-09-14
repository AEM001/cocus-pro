from __future__ import annotations

from typing import Any, Dict, Optional

import base64
import io

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
    Qwen2.5-VL-7B-Instruct integration scaffold.

    Modes:
    - server: POST to a local HTTP endpoint you host, with fields {images: [b64], system, user}
    - local:  load a local model from `model_dir` (placeholder left for user to implement)

    Parameters
    - mode: "server" | "local"
    - server_url: e.g., http://127.0.0.1:8000/generate  (no network calls here by default)
    - model_dir: path to local Qwen2.5-VL-7B-Instruct weights (not loaded in this stub)
    """

    def __init__(self, mode: str = "server", server_url: Optional[str] = None, model_dir: Optional[str] = None):
        self.mode = mode
        self.server_url = server_url or "http://127.0.0.1:8000/generate"
        self.model_dir = model_dir
        self._local_model = None  # user to implement if desired

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
        # Placeholder for local python inference. Users can implement loading with vLLM/Transformers.
        # You can set up your runner here using `self.model_dir`.
        _ = (images, system, user, self.model_dir, self._local_model)
        return "{}"

