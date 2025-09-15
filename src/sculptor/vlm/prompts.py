"""
【VLM提示词工程模块】
作用：构建视觉语言模型的提示词模板，指导VLM进行图像分析和点评估
核心功能：
  - build_peval_prompt: 构建整体掩码评估的提示词
  - build_pgen_prompt: 构建单个图像块点评估的提示词
  - 结构化输出：强制JSON格式响应
  - 领域知识注入：包含分割任务的先验知识

与系统其他模块关系：
  - 被qwen.py和mock.py调用生成VLM输入提示
  - 为peval/pgen接口提供标准化的提示模板
  - 影响select_points.py接收到的VLM输出格式

提示设计原则：
  - 角色设定：专业图像分割专家
  - 任务明确：区分整体评估vs局部点评估
  - 输出约束：强制JSON格式确保可解析
  - 上下文丰富：提供关键线索指导分析
"""

from __future__ import annotations

from typing import Dict


def build_peval_prompt(instance: str, word_budget: int = 40) -> Dict[str, str]:
    system = "You are a segmentation inspector. Be concise."
    user = (
        "You see an ROI crop, with a semi-transparent mask overlay.\n"
        f"Target category: \"{instance}\" (single target).\n"
        "Task: Briefly diagnose mask defects. Reply in compact JSON only:\n"
        "{\n  \"missing_parts\": [\"...\"],\n  \"over_segments\": [\"...\"],\n  \"boundary_quality\": \"sharp|soft|fragmented\",\n  \"key_cues\": [\"shape/texture/depth cues for the true object\"]\n}\n"
        f"Word budget: within {word_budget} words total."
    )
    return {"system": system, "user": user}


def build_pgen_prompt(instance: str, key_cues: str) -> Dict[str, str]:
    system = (
        "You label whether a small image patch belongs to the SINGLE target category.\n"
        "Be strict and concise. Do not reveal your reasoning."
    )
    user = (
        f"Target: \"{instance}\".\n"
        f"Visual cues (may help disambiguation): \"{key_cues}\".\n"
        "Answer JSON only:\n{\"is_target\": true|false, \"conf\": 0.00-1.00}"
    )
    return {"system": system, "user": user}

