#!/usr/bin/env python3
"""
批量图像处理脚本
自动处理auxiliary/images/目录下的所有图像文件

支持的功能：
1. 自动发现图像文件
2. 批量API目标检测
3. 批量SAM精修
4. 并行处理支持
5. 错误处理和重试
6. 进度显示
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import glob

# 支持的图像格式
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

# 默认目标检测查询映射
DEFAULT_TARGETS = {
    'f': 'find the camouflaged scorpionfish',
    'dog': 'find the camouflaged dog',
    'cat': 'find the camouflaged cat',
    'q': 'find the camouflaged animal',
    'person': 'find the camouflaged person',
    'person2': 'find the camouflaged person'
}

def get_image_files(image_dir: str) -> List[Tuple[str, str]]:
    """
    获取所有支持的图像文件
    返回: [(文件名, 完整路径)] 列表
    """
    image_files = []
    for ext in SUPPORTED_FORMATS:
        pattern = os.path.join(image_dir, f"*{ext}")
        pattern_upper = os.path.join(image_dir, f"*{ext.upper()}")
        
        for file_path in glob.glob(pattern) + glob.glob(pattern_upper):
            name = os.path.splitext(os.path.basename(file_path))[0]
            image_files.append((name, file_path))
    
    return sorted(image_files)

def run_command_with_retry(cmd: str, description: str, max_retries: int = 2) -> Tuple[bool, str]:
    """
    运行命令并重试
    """
    for attempt in range(max_retries + 1):
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
            if attempt > 0:
                print(f"  重试 {attempt}/{max_retries}")
            
            result = subprocess.run(
                cmd, shell=True, check=True,
                capture_output=True, text=True,
                cwd="/home/albert/code/CV"
            )
            
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            error_msg = f"命令失败: {e}\n输出: {e.stdout}\n错误: {e.stderr}"
            
            if attempt < max_retries:
                print(f"  [WARN] {error_msg}")
                print(f"  等待2秒后重试...")
                time.sleep(2)
            else:
                print(f"  [ERROR] {error_msg}")
                return False, error_msg
    
    return False, "超过最大重试次数"

def process_single_image(name: str, image_path: str, args) -> Dict:
    """
    处理单个图像
    """
    start_time = time.time()
    result = {
        'name': name,
        'image_path': image_path,
        'success': False,
        'error': None,
        'steps_completed': [],
        'processing_time': 0,
        'output_files': []
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"🖼️  处理图像: {name}")
        print(f"   路径: {image_path}")
        print(f"{'='*60}")
        
        # 设置目标查询
        target_query = args.target or DEFAULT_TARGETS.get(name, f'find the camouflaged object in {name}')
        
        # 构建基本命令参数 - 直接使用python路径
        python_path = "/home/albert/anaconda3/envs/camo-vlm/bin/python"
        base_env = f"export DASHSCOPE_API_KEY='{args.api_key}'"
        
        # 步骤1：生成网格标注图
        if not args.skip_grid:
            grid_cmd = f"""{base_env} && cd auxiliary/scripts && {python_path} make_region_prompts.py \\
                --name {name} \\
                --image ../images/{os.path.basename(image_path)} \\
                --rows {args.grid_size} \\
                --cols {args.grid_size}"""
            
            success, output = run_command_with_retry(grid_cmd, f"生成网格标注图", args.retry_count)
            if not success:
                result['error'] = f"网格生成失败: {output}"
                return result
            result['steps_completed'].append('grid_generation')
        
        # 步骤2：API目标检测
        if not args.skip_detection:
            detection_cmd = f"""{base_env} && cd auxiliary/scripts && {python_path} detect_target_api.py \\
                --name {name} \\
                --target "{target_query}" \\
                --model {args.api_model} \\
                --grid-size {args.grid_size}"""
            
            if args.use_openai_api:
                detection_cmd += " --use-openai"
            if args.high_resolution:
                detection_cmd += " --high-res"
            
            success, output = run_command_with_retry(detection_cmd, f"API目标检测", args.retry_count)
            if not success:
                result['error'] = f"目标检测失败: {output}"
                return result
            result['steps_completed'].append('target_detection')
        
        # 步骤3：生成SAM输入
        if not args.skip_build:
            build_cmd = f"""{base_env} && {python_path} auxiliary/scripts/build_prior_and_boxes.py \\
                --name {name} \\
                --meta auxiliary/scripts/out/{name}/{name}_meta.json \\
                --pred auxiliary/llm_out/{name}_output.json"""
            
            success, output = run_command_with_retry(build_cmd, f"生成SAM输入", args.retry_count)
            if not success:
                result['error'] = f"SAM输入生成失败: {output}"
                return result
            result['steps_completed'].append('sam_input_generation')
        
        # 步骤4：SAM精修
        if not args.skip_sculpt:
            sculpt_cmd = f"""{base_env} && {python_path} clean_sam_sculpt.py \\
                --name {name} \\
                --rounds {args.rounds} \\
                --ratio {args.ratio} \\
                --vlm_max_side {args.vlm_max_side} \\
                --output-format {args.output_format}"""
            
            # API选项
            if args.use_api:
                sculpt_cmd += f" --use-api --api-model {args.api_model}"
                if args.use_openai_api:
                    sculpt_cmd += " --use-openai-api"
                if args.high_resolution:
                    sculpt_cmd += " --high-resolution"
            
            # 清理输出选项
            if args.clean_output:
                sculpt_cmd += " --clean-output"
            
            success, output = run_command_with_retry(sculpt_cmd, f"SAM精修", args.retry_count)
            if not success:
                result['error'] = f"SAM精修失败: {output}"
                return result
            result['steps_completed'].append('sam_sculpting')
        
        # 检查输出文件
        output_dir = f"outputs/clean_sculpt/{name}"
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            result['output_files'] = output_files
        
        result['success'] = True
        result['processing_time'] = time.time() - start_time
        
        print(f"✅ {name} 处理完成! 用时: {result['processing_time']:.1f}秒")
        
    except Exception as e:
        result['error'] = f"处理异常: {str(e)}"
        result['processing_time'] = time.time() - start_time
        print(f"❌ {name} 处理失败: {result['error']}")
    
    return result

def save_batch_report(results: List[Dict], output_dir: str):
    """保存批处理报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    total_count = len(results)
    success_count = sum(1 for r in results if r['success'])
    failed_count = total_count - success_count
    total_time = sum(r['processing_time'] for r in results)
    avg_time = total_time / total_count if total_count > 0 else 0
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_images': total_count,
            'successful': success_count,
            'failed': failed_count,
            'success_rate': success_count / total_count * 100 if total_count > 0 else 0,
            'total_processing_time': total_time,
            'average_processing_time': avg_time
        },
        'results': results
    }
    
    # 保存JSON报告
    report_path = os.path.join(output_dir, f'batch_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存简化的文本报告
    txt_report_path = os.path.join(output_dir, f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("批量处理报告\n")
        f.write("="*60 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总图像数: {total_count}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"成功率: {success_count / total_count * 100:.1f}%\n")
        f.write(f"总处理时间: {total_time:.1f}秒\n")
        f.write(f"平均处理时间: {avg_time:.1f}秒/图像\n\n")
        
        # 成功的图像
        if success_count > 0:
            f.write("成功处理的图像:\n")
            f.write("-" * 40 + "\n")
            for result in results:
                if result['success']:
                    f.write(f"✅ {result['name']} ({result['processing_time']:.1f}s)\n")
                    f.write(f"   输出文件: {', '.join(result['output_files'])}\n")
        
        # 失败的图像
        if failed_count > 0:
            f.write(f"\n失败的图像:\n")
            f.write("-" * 40 + "\n")
            for result in results:
                if not result['success']:
                    f.write(f"❌ {result['name']}: {result['error']}\n")
    
    print(f"\n📊 批处理报告已保存:")
    print(f"   JSON报告: {report_path}")
    print(f"   文本摘要: {txt_report_path}")

def main():
    parser = argparse.ArgumentParser(description="批量处理图像文件")
    
    # 基本参数
    parser.add_argument('--image-dir', default='auxiliary/images', help='图像文件目录')
    parser.add_argument('--include', nargs='+', help='只处理指定的图像名称 (不含扩展名)')
    parser.add_argument('--exclude', nargs='+', help='排除指定的图像名称 (不含扩展名)')
    parser.add_argument('--target', help='统一的目标检测查询，覆盖默认映射')
    
    # API参数
    parser.add_argument('--api-key', help='API密钥 (默认从环境变量获取)')
    parser.add_argument('--api-model', default='qwen-vl-plus-latest', help='API模型名称')
    parser.add_argument('--use-api', action='store_true', help='使用API模式')
    parser.add_argument('--use-openai-api', action='store_true', help='使用OpenAI兼容模式')
    parser.add_argument('--high-resolution', action='store_true', help='启用高分辨率模式')
    
    # 处理参数
    parser.add_argument('--grid-size', type=int, default=9, help='网格大小')
    parser.add_argument('--rounds', type=int, default=1, help='SAM精修轮数')
    parser.add_argument('--ratio', type=float, default=0.6, help='象限比例')
    parser.add_argument('--vlm-max-side', type=int, default=720, help='VLM输入图像最大边长')
    parser.add_argument('--output-format', choices=['png', 'jpg'], default='png', help='输出掩码格式')
    parser.add_argument('--clean-output', action='store_true', help='只保留最终掩码和instance信息')
    
    # 批处理参数
    parser.add_argument('--parallel', type=int, default=1, help='并行处理数 (1=串行)')
    parser.add_argument('--retry-count', type=int, default=2, help='失败重试次数')
    parser.add_argument('--report-dir', default='batch_reports', help='报告保存目录')
    
    # 跳过步骤选项
    parser.add_argument('--skip-grid', action='store_true', help='跳过网格生成')
    parser.add_argument('--skip-detection', action='store_true', help='跳过目标检测')
    parser.add_argument('--skip-build', action='store_true', help='跳过SAM输入生成')
    parser.add_argument('--skip-sculpt', action='store_true', help='跳过SAM精修')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 批量图像处理系统")
    print("="*60)
    
    # 检查API密钥
    if not args.api_key:
        args.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not args.api_key:
            print("[ERROR] 需要设置API密钥")
            print("使用 --api-key 参数或设置环境变量 DASHSCOPE_API_KEY")
            return 1
    
    print(f"API密钥: {args.api_key[:8]}...")
    print(f"API模型: {args.api_model}")
    
    # 获取图像文件
    image_files = get_image_files(args.image_dir)
    if not image_files:
        print(f"[ERROR] 在 {args.image_dir} 中未找到支持的图像文件")
        print(f"支持的格式: {', '.join(SUPPORTED_FORMATS)}")
        return 1
    
    # 过滤图像文件
    if args.include:
        image_files = [(name, path) for name, path in image_files if name in args.include]
        print(f"包含过滤: {args.include}")
    
    if args.exclude:
        image_files = [(name, path) for name, path in image_files if name not in args.exclude]
        print(f"排除过滤: {args.exclude}")
    
    if not image_files:
        print("[ERROR] 过滤后没有图像文件需要处理")
        return 1
    
    print(f"待处理图像: {len(image_files)}")
    for name, path in image_files:
        print(f"  - {name}")
    
    print(f"处理模式: {'API' if args.use_api else '本地'}")
    print(f"并行数: {args.parallel}")
    print(f"清理输出: {'是' if args.clean_output else '否'}")
    
    # 开始批处理
    start_time = time.time()
    results = []
    
    if args.parallel == 1:
        # 串行处理
        for i, (name, image_path) in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理 {name}...")
            result = process_single_image(name, image_path, args)
            results.append(result)
    else:
        # 并行处理
        print(f"\n开始并行处理 (并发数: {args.parallel})...")
        
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # 提交任务
            future_to_name = {}
            for name, image_path in image_files:
                future = executor.submit(process_single_image, name, image_path, args)
                future_to_name[future] = name
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_name):
                completed += 1
                name = future_to_name[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{completed}/{len(image_files)}] {name} 完成")
                except Exception as e:
                    error_result = {
                        'name': name,
                        'success': False,
                        'error': f"并行处理异常: {str(e)}",
                        'processing_time': 0,
                        'steps_completed': [],
                        'output_files': []
                    }
                    results.append(error_result)
                    print(f"[{completed}/{len(image_files)}] {name} 异常: {e}")
    
    # 统计结果
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    
    print("\n" + "="*60)
    print("📊 批处理完成!")
    print("="*60)
    print(f"总处理时间: {total_time:.1f}秒")
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {failed_count}/{len(results)}")
    print(f"成功率: {success_count/len(results)*100:.1f}%")
    
    # 保存报告
    save_batch_report(results, args.report_dir)
    
    # 显示失败的图像
    if failed_count > 0:
        print(f"\n❌ 失败的图像:")
        for result in results:
            if not result['success']:
                print(f"  - {result['name']}: {result['error']}")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    exit(main())