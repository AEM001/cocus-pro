#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测API调用脚本
使用阿里云百炼Qwen VL API进行网格化目标检测与定位

工作流程：
1. 读取两张网格标注图像（垂直分割 + 水平分割）
2. 调用VLM API进行目标检测和定位
3. 输出包含网格ID的JSON结果

替换原有的本地模型调用流程
"""

import os
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

# 尝试导入必要的API SDK
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TargetDetectionAPI:
    """目标检测API调用类"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "qwen-vl-plus-latest",
                 use_dashscope: bool = True,
                 high_resolution: bool = False):
        """
        初始化API客户端
        
        Args:
            api_key: API密钥，默认从环境变量获取
            model_name: 模型名称
            use_dashscope: 是否使用DashScope SDK
            high_resolution: 是否启用高分辨率模式
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter")
            
        self.model_name = model_name
        self.use_dashscope = use_dashscope
        self.high_resolution = high_resolution
        
        # 检查依赖
        if self.use_dashscope and not DASHSCOPE_AVAILABLE:
            print("[WARN] DashScope SDK not available, falling back to OpenAI compatible mode")
            self.use_dashscope = False
            
        if not self.use_dashscope and not OPENAI_AVAILABLE:
            raise ImportError("Neither DashScope nor OpenAI SDK is available. Please install one of them.")
        
        # 初始化客户端
        self._init_client()

    def _init_client(self):
        """初始化API客户端"""
        if self.use_dashscope:
            dashscope.api_key = self.api_key
            self._client = dashscope
            print(f"[INFO] 已初始化 DashScope SDK 客户端，模型: {self.model_name}")
        else:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            print(f"[INFO] 已初始化 OpenAI 兼容客户端，模型: {self.model_name}")

    def detect_target_location(self, 
                             vertical_grid_path: str, 
                             horizontal_grid_path: str, 
                             target_description: str,
                             grid_size: int = 9) -> Dict[str, Any]:
        """
        调用API进行目标检测
        
        Args:
            vertical_grid_path: 垂直网格标注图路径
            horizontal_grid_path: 水平网格标注图路径
            target_description: 目标描述
            grid_size: 网格大小
            
        Returns:
            包含检测结果的字典
        """
        
        # 构建提示词
        system_prompt = self._build_system_prompt(grid_size)
        user_prompt = f"questions:\n<<{target_description}>>"
        
        try:
            if self.use_dashscope:
                return self._detect_with_dashscope(vertical_grid_path, horizontal_grid_path, 
                                                 system_prompt, user_prompt)
            else:
                return self._detect_with_openai(vertical_grid_path, horizontal_grid_path,
                                              system_prompt, user_prompt)
        except Exception as e:
            print(f"[ERROR] API调用失败: {e}")
            return self._get_fallback_result(target_description)

    def _build_system_prompt(self, grid_size: int) -> str:
        """构建系统提示词"""
        return f"""You are now acting as an image annotator. I will give you several abstract descriptions which refer to the same object which may appear in the given image and you need to locate them if they exists.

Now I will provide you with two images featuring exactly the same scene but annotated with vertical and horizontal visual segmentation prompt.

One image is divided into <<{grid_size}>> segments using up-to-down lines, and the other one is divided with left-to-right lines into <<{grid_size}>> segments. Each segment is labeled with a number from 1 to <<{grid_size}>>.

Now please carefully follow the below instruction step by step:
1. Infer what is (are) the objects that the questions are jointly referring to, carefully review the given image and find the object(s) that is the most probable candidate(s). (Exactly one category should be decided.)
2. Denote from which labeled segment to which contain any parts of the object(s) that the questions described, both vertically and horizontally. Include ALL segment numbers that contain ANY part of the target, even if tiny or on the border.

Tell me these segment ids in json as follows:

{{  
"instance": "<words(adj+noun) short summarative description of the answer object>",  
"ids_line_vertical": [1,2,3,...],  
"ids_line_horizontal": [1,2,3,...],  
"reason": "your rationale about what is(are) being referred"  
}}

Attention: you MUST give all segment numbers that contain any parts of the object(s) that the questions described. The answer object may not be the salient object in the frame, and may even not appear in the frame.

If the answer is not in the image, please return empty ids like:

{{  
"instance": "<words(adj+noun) short summarative description of the answer object>",  
"ids_line_vertical": [],  
"ids_line_horizontal": [],  
"reason": "your rationale about what is being referred and why none of the object in the scene matches"  
}}

Output RULES:
- Output ONLY the JSON object specified above. No extra text, no explanations, no markdown fences.
- Keys must be exactly: instance, ids_line_vertical, ids_line_horizontal, reason.
- Exactly ONE category in "instance".

Now, the questions are presented as follows, do NOT output anything other than the required json."""

    def _detect_with_dashscope(self, vertical_path: str, horizontal_path: str, 
                             system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """使用DashScope SDK进行检测"""
        
        # 构建消息
        content = [
            {"image": f"file://{os.path.abspath(vertical_path)}"},
            {"image": f"file://{os.path.abspath(horizontal_path)}"},
            {"text": user_prompt}
        ]
        
        messages = [
            {"role": "system", "content": [{"text": system_prompt}]},
            {"role": "user", "content": content}
        ]
        
        # API调用参数 - 增加tokens数量防止截断
        call_params = {
            "api_key": self.api_key,
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,  # 增加到512防止截断
            "temperature": 0.0,
        }
        
        if self.high_resolution:
            call_params["vl_high_resolution_images"] = True
        
        response = self._client.MultiModalConversation.call(**call_params)
        
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            if isinstance(content, list) and content:
                response_text = content[0].get("text", "")
            else:
                response_text = str(content)
            
            print(f"[DEBUG] API响应: {response_text}")
            return self._parse_response(response_text)
        else:
            raise Exception(f"API调用失败，状态码: {response.status_code}, 错误: {response.message}")

    def _detect_with_openai(self, vertical_path: str, horizontal_path: str,
                           system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """使用OpenAI兼容模式进行检测"""
        
        import base64
        
        # 编码图像
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        vertical_b64 = encode_image(vertical_path)
        horizontal_b64 = encode_image(horizontal_path)
        
        # 构建消息
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{vertical_b64}"}
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{horizontal_b64}"}
            },
            {"type": "text", "text": user_prompt}
        ]
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content}
        ]
        
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,  # 增加到512防止截断
            temperature=0.0,
        )
        
        response_text = completion.choices[0].message.content or ""
        print(f"[DEBUG] API响应: {response_text}")
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析API响应"""
        try:
            # 清理响应文本
            clean_text = response_text.strip()
            
            # 如果响应包含markdown代码块，提取JSON部分
            if "```json" in clean_text:
                start = clean_text.find("{")
                end = clean_text.rfind("}") + 1
                if start != -1 and end != 0:
                    clean_text = clean_text[start:end]
            
            # 解析JSON
            result = json.loads(clean_text)
            
            # 验证必需字段
            required_fields = ["instance", "ids_line_vertical", "ids_line_horizontal", "reason"]
            for field in required_fields:
                if field not in result:
                    result[field] = [] if "ids_" in field else ""
            
            # 确保ID列表是整数类型
            if result["ids_line_vertical"]:
                result["ids_line_vertical"] = [int(x) for x in result["ids_line_vertical"] if isinstance(x, (int, str)) and str(x).isdigit()]
            if result["ids_line_horizontal"]:
                result["ids_line_horizontal"] = [int(x) for x in result["ids_line_horizontal"] if isinstance(x, (int, str)) and str(x).isdigit()]
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON解析失败: {e}")
            print(f"[DEBUG] 原始响应: {response_text}")
            return self._get_fallback_result("unknown object", f"JSON解析失败: {str(e)}")
        except Exception as e:
            print(f"[ERROR] 响应解析失败: {e}")
            return self._get_fallback_result("unknown object", f"响应解析失败: {str(e)}")

    def _get_fallback_result(self, target_description: str, error_reason: str = "API调用失败") -> Dict[str, Any]:
        """获取备用结果"""
        return {
            "instance": f"unknown {target_description}",
            "ids_line_vertical": [],
            "ids_line_horizontal": [],
            "reason": error_reason
        }


def main():
    parser = argparse.ArgumentParser(description="使用API进行目标检测和网格定位")
    parser.add_argument("--name", required=True, help="样本名称（如 f, dog, q）")
    parser.add_argument("--target", help="目标描述，如果不提供则使用默认查询")
    parser.add_argument("--api-key", help="API密钥，默认从环境变量获取")
    parser.add_argument("--model", default="qwen-vl-plus-latest", help="模型名称")
    parser.add_argument("--use-openai", action="store_true", help="使用OpenAI兼容模式")
    parser.add_argument("--high-res", action="store_true", help="启用高分辨率模式")
    parser.add_argument("--grid-size", type=int, default=9, help="网格大小")
    
    args = parser.parse_args()
    
    # 路径设置
    base_dir = Path(__file__).parent.parent
    name = args.name
    
    # 输入路径
    vertical_path = base_dir / "out" / name / f"{name}_vertical_{args.grid_size}.png"
    horizontal_path = base_dir / "out" / name / f"{name}_horizontal_{args.grid_size}.png"
    
    # 输出路径
    output_dir = base_dir / "llm_out"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{name}_output.json"
    
    # 检查输入文件
    if not vertical_path.exists() or not horizontal_path.exists():
        print(f"[ERROR] 网格图像文件不存在:")
        print(f"  垂直: {vertical_path}")
        print(f"  水平: {horizontal_path}")
        print("请先运行 make_region_prompts.py 生成网格图像")
        return 1
    
    # 设置目标描述
    if args.target:
        target_description = args.target
    else:
        # 默认目标描述映射
        default_targets = {
            "f": "find the camouflaged scorpionfish",
            "dog": "find the camouflaged dog", 
            "q": "find the camouflaged animal",
        }
        target_description = default_targets.get(name, f"find the target in {name}")
    
    print(f"[INFO] 处理样本: {name}")
    print(f"[INFO] 目标描述: {target_description}")
    print(f"[INFO] 垂直网格图: {vertical_path}")
    print(f"[INFO] 水平网格图: {horizontal_path}")
    
    try:
        # 初始化API客户端
        api_client = TargetDetectionAPI(
            api_key=args.api_key,
            model_name=args.model,
            use_dashscope=not args.use_openai,
            high_resolution=args.high_res
        )
        
        # 执行目标检测
        print("[INFO] 调用API进行目标检测...")
        result = api_client.detect_target_location(
            str(vertical_path),
            str(horizontal_path), 
            target_description,
            args.grid_size
        )
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] 检测结果已保存: {output_path}")
        print(f"[RESULT] 检测到目标: {result['instance']}")
        print(f"[RESULT] 垂直区域: {result['ids_line_vertical']}")
        print(f"[RESULT] 水平区域: {result['ids_line_horizontal']}")
        print(f"[RESULT] 原因: {result['reason']}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())