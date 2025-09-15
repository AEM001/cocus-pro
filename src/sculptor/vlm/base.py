from __future__ import annotations

"""
【VLM基础接口模块】
作用：定义视觉语言模型的抽象基类和通用工具函数
核心功能：
  - VLMBase协议：定义标准化的peval/pgen接口规范
  - JSON解析工具：将VLM输出字符串解析为结构化数据
  - 错误处理：提供安全的JSON解析和异常恢复机制

与系统其他模块关系：
  - 被qwen.py和mock.py继承实现具体VLM模型
  - 为select_points.py提供标准化的VLM调用接口
  - 确保不同VLM实现之间的接口一致性

设计特点：
  - 协议类模式：使用Python Protocol定义接口规范
  - 类型安全：完整的类型注解确保编译时检查
  - 错误恢复：try_parse_json提供优雅的解析失败处理
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class VLMBase(ABC):
    """Abstract VLM interface with two tasks: peval (mask-level) and pgen (patch-level)."""

    @abstractmethod
    def peval(self, image_overlay_rgb: np.ndarray, depth_vis: Optional[np.ndarray], instance: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def pgen(self, patch_rgb: np.ndarray, instance: str, key_cues: str) -> Dict[str, Any]:
        ...


def try_parse_json(text: str) -> Dict[str, Any]:
    import json

    try:
        return json.loads(text)
    except Exception:
        # naive bracket trimming fallback
        try:
            s = text[text.find("{") : text.rfind("}") + 1]
            return json.loads(s)
        except Exception:
            return {}

