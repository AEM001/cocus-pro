#!/usr/bin/env python3
"""
完整的API调用流程脚本
演示从网格生成到SAM精修的全流程API调用

使用示例：
  # 设置环境变量
  export DASHSCOPE_API_KEY="your_api_key_here"
  
  # 运行完整流程
  python run_api_pipeline.py --name f --target "find the camouflaged scorpionfish"
  
  # 只运行目标检测
  python run_api_pipeline.py --name f --target "find the camouflaged scorpionfish" --only-detection
  
  # 使用OpenAI兼容模式
  python run_api_pipeline.py --name f --use-openai-api
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description="", check=True):
    """运行命令并打印输出"""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True, cwd="/home/albert/code/CV")
        
        if result.stdout:
            print(f"[STDOUT] {result.stdout}")
        if result.stderr:
            print(f"[STDERR] {result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 命令执行失败: {e}")
        if e.stdout:
            print(f"[STDOUT] {e.stdout}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        return False

def check_api_key():
    """检查API密钥"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] 未设置API密钥")
        print("请设置环境变量: export DASHSCOPE_API_KEY='your_api_key_here'")
        return False
    
    print(f"[INFO] API密钥已设置: {api_key[:8]}...")
    return True

def check_dependencies():
    """检查必要的依赖"""
    print("[INFO] 检查依赖...")
    
    try:
        import dashscope
        print("[✓] DashScope SDK 可用")
        dashscope_available = True
    except ImportError:
        print("[✗] DashScope SDK 不可用")
        dashscope_available = False
    
    try:
        from openai import OpenAI
        print("[✓] OpenAI SDK 可用")
        openai_available = True
    except ImportError:
        print("[✗] OpenAI SDK 不可用")
        openai_available = False
    
    if not dashscope_available and not openai_available:
        print("[ERROR] 至少需要安装 dashscope 或 openai 其中一个SDK")
        print("安装命令:")
        print("  pip install dashscope")
        print("  pip install openai")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="完整的API调用流程")
    parser.add_argument("--name", required=True, help="样本名称 (如 f, dog, q)")
    parser.add_argument("--target", help="目标描述 (如 'find the camouflaged scorpionfish')")
    parser.add_argument("--api-key", help="API密钥")
    parser.add_argument("--model", default="qwen-vl-plus-latest", help="模型名称")
    parser.add_argument("--use-openai-api", action="store_true", help="使用OpenAI兼容模式")
    parser.add_argument("--high-resolution", action="store_true", help="启用高分辨率模式")
    parser.add_argument("--grid-size", type=int, default=9, help="网格大小")
    parser.add_argument("--rounds", type=int, default=4, help="精修轮数")
    parser.add_argument("--ratio", type=float, default=0.8, help="象限比例")
    parser.add_argument("--vlm-max-side", type=int, default=720, help="VLM输入图像最大边长")
    parser.add_argument("--only-detection", action="store_true", help="只执行目标检测，不运行SAM精修")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 启动完整的API调用流程")
    print("=" * 60)
    
    # 检查环境
    if not check_api_key() and not args.api_key:
        return 1
    
    if not check_dependencies():
        return 1
    
    # 设置路径
    base_dir = Path("/home/albert/code/CV")
    name = args.name
    
    # 设置默认目标描述
    if not args.target:
        default_targets = {
            "f": "find the camouflaged scorpionfish",
            "dog": "find the camouflaged dog",
            "q": "find the camouflaged animal"
        }
        args.target = default_targets.get(name, f"find the target in {name}")
    
    print(f"[INFO] 样本: {name}")
    print(f"[INFO] 目标: {args.target}")
    print(f"[INFO] 模型: {args.model}")
    print(f"[INFO] API模式: {'OpenAI兼容' if args.use_openai_api else 'DashScope SDK'}")
    
    # 步骤1：生成网格标注图
    print("\n" + "="*50)
    print("📊 步骤1: 生成网格标注图")
    print("="*50)
    
    grid_cmd = f"""cd auxiliary/scripts && python make_region_prompts.py \\
        --name {name} \\
        --image ../images/{name}.png \\
        --rows {args.grid_size} \\
        --cols {args.grid_size}"""
    
    if not run_command(grid_cmd, "生成网格标注图"):
        return 1
    
    # 验证网格图是否生成成功
    vertical_path = base_dir / "auxiliary" / "out" / name / f"{name}_vertical_{args.grid_size}.png"
    horizontal_path = base_dir / "auxiliary" / "out" / name / f"{name}_horizontal_{args.grid_size}.png"
    
    if not vertical_path.exists() or not horizontal_path.exists():
        print(f"[ERROR] 网格图生成失败")
        print(f"  期望路径: {vertical_path}")
        print(f"  期望路径: {horizontal_path}")
        return 1
    
    print(f"[✓] 网格图生成成功:")
    print(f"  垂直: {vertical_path}")
    print(f"  水平: {horizontal_path}")
    
    # 步骤2：调用API进行目标检测
    print("\n" + "="*50)
    print("🎯 步骤2: API目标检测")
    print("="*50)
    
    detection_cmd = f"""cd auxiliary/scripts && python detect_target_api.py \\
        --name {name} \\
        --target "{args.target}" \\
        --model {args.model} \\
        --grid-size {args.grid_size}"""
    
    if args.api_key:
        detection_cmd += f" --api-key {args.api_key}"
    
    if args.use_openai_api:
        detection_cmd += " --use-openai"
    
    if args.high_resolution:
        detection_cmd += " --high-res"
    
    if not run_command(detection_cmd, "执行API目标检测"):
        return 1
    
    # 验证检测结果
    llm_output_path = base_dir / "auxiliary" / "llm_out" / f"{name}_output.json"
    if not llm_output_path.exists():
        print(f"[ERROR] 目标检测结果文件未生成: {llm_output_path}")
        return 1
    
    print(f"[✓] 目标检测完成: {llm_output_path}")
    
    # 步骤3：生成SAM输入
    print("\n" + "="*50)
    print("📦 步骤3: 生成SAM输入")
    print("="*50)
    
    build_cmd = f"""python auxiliary/scripts/build_prior_and_boxes.py \\
        --name {name} \\
        --meta auxiliary/scripts/out/{name}/{name}_meta.json \\
        --pred auxiliary/llm_out/{name}_output.json"""
    
    if not run_command(build_cmd, "生成SAM输入文件"):
        return 1
    
    # 验证SAM输入文件
    sam_box_path = base_dir / "auxiliary" / "box_out" / name / f"{name}_sam_boxes.json"
    sam_mask_path = base_dir / "auxiliary" / "box_out" / name / f"{name}_prior_mask.png"
    
    if not sam_box_path.exists() or not sam_mask_path.exists():
        print(f"[ERROR] SAM输入文件生成失败")
        return 1
    
    print(f"[✓] SAM输入文件生成成功:")
    print(f"  边界框: {sam_box_path}")
    print(f"  初始掩码: {sam_mask_path}")
    
    # 如果只要求目标检测，到此为止
    if args.only_detection:
        print("\n" + "="*50)
        print("✅ 目标检测完成!")
        print("="*50)
        print(f"结果文件:")
        print(f"  检测结果: {llm_output_path}")
        print(f"  SAM边界框: {sam_box_path}")
        print(f"  初始掩码: {sam_mask_path}")
        return 0
    
    # 步骤4：运行SAM精修（使用API）
    print("\n" + "="*50)
    print("🎨 步骤4: SAM精修 (使用API)")
    print("="*50)
    
    sam_cmd = f"""python clean_sam_sculpt.py \\
        --name {name} \\
        --rounds {args.rounds} \\
        --ratio {args.ratio} \\
        --vlm_max_side {args.vlm_max_side} \\
        --use-api"""
    
    if args.api_key:
        sam_cmd += f" --api-key {args.api_key}"
    
    sam_cmd += f" --api-model {args.model}"
    
    if args.use_openai_api:
        sam_cmd += " --use-openai-api"
    
    if args.high_resolution:
        sam_cmd += " --high-resolution"
    
    if not run_command(sam_cmd, "执行SAM精修"):
        return 1
    
    # 验证最终结果
    output_dir = base_dir / "outputs" / "clean_sculpt" / name
    final_result = output_dir / "final_result.png"
    final_vis = output_dir / "final_visualization.png"
    
    if not final_result.exists() or not final_vis.exists():
        print(f"[ERROR] 最终结果文件未生成")
        return 1
    
    print("\n" + "="*50)
    print("🎉 完整流程执行成功!")
    print("="*50)
    print(f"最终结果:")
    print(f"  掩码结果: {final_result}")
    print(f"  可视化结果: {final_vis}")
    print(f"  完整输出目录: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())