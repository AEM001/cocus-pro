#!/usr/bin/env python3
"""
测试仅API模式的SAM分割系统
验证新的路径结构和API调用功能
"""

import os
import sys
import argparse
from pathlib import Path

def test_environment():
    """测试环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        print(f"✅ API密钥已设置: {api_key[:8]}...")
    else:
        print("❌ 未设置DASHSCOPE_API_KEY环境变量")
        return False
    
    # 检查Python依赖
    try:
        import torch
        import cv2
        import segment_anything
        import dashscope
        from PIL import Image
        print("✅ 核心依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        return False
    
    return True

def test_paths():
    """测试路径结构"""
    print("\n📁 检查路径结构...")
    
    base_dir = Path(__file__).parent
    
    # 检查输入路径
    img_dir = base_dir / "dataset" / "COD10K_TEST_DIR" / "Imgs"
    if img_dir.exists():
        img_files = list(img_dir.glob("*.jpg"))
        print(f"✅ 输入图像目录存在: {img_dir}")
        print(f"   找到 {len(img_files)} 个jpg文件")
        if img_files:
            print(f"   示例文件: {img_files[0].name}")
    else:
        print(f"❌ 输入图像目录不存在: {img_dir}")
        return False
    
    # 检查输出路径
    aux_dir = base_dir / "auxiliary"
    if aux_dir.exists():
        print(f"✅ 辅助目录存在: {aux_dir}")
        
        # 检查子目录
        subdirs = ["scripts", "out", "llm_out", "box_out"]
        for subdir in subdirs:
            sub_path = aux_dir / subdir
            if sub_path.exists():
                print(f"   ✅ {subdir}/ 存在")
            else:
                print(f"   ⚠️  {subdir}/ 不存在，将自动创建")
    else:
        print(f"❌ 辅助目录不存在: {aux_dir}")
        return False
    
    return True

def test_api_import():
    """测试API导入"""
    print("\n🔌 测试API后端导入...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from sculptor.vlm.qwen_api import QwenAPIVLM
        from sculptor.vlm.base import VLMBase
        print("✅ API后端导入成功")
        
        # 检查是否删除了本地VLM
        try:
            from sculptor.vlm.qwen import QwenVLM
            print("⚠️  本地VLM仍然存在，应该已被删除")
            return False
        except ImportError:
            print("✅ 本地VLM已正确移除")
            
        return True
    except ImportError as e:
        print(f"❌ API后端导入失败: {e}")
        return False

def test_sample_image():
    """测试样本图像"""
    print("\n🖼️  测试样本图像...")
    
    img_dir = Path(__file__).parent / "dataset" / "COD10K_TEST_DIR" / "Imgs"
    sample_files = list(img_dir.glob("COD10K_*.jpg"))
    
    if not sample_files:
        print("❌ 未找到COD10K样本文件")
        return False
    
    sample_file = sample_files[0]
    sample_name = sample_file.stem
    
    try:
        from PIL import Image
        img = Image.open(sample_file)
        print(f"✅ 成功加载样本图像: {sample_name}")
        print(f"   图像尺寸: {img.size}")
        print(f"   图像格式: {img.format}")
        return sample_name
    except Exception as e:
        print(f"❌ 无法加载样本图像: {e}")
        return False

def run_quick_test(sample_name):
    """运行快速测试"""
    print(f"\n🚀 运行快速测试 (样本: {sample_name})...")
    
    # 测试脚本路径
    clean_script = Path(__file__).parent / "clean_sam_sculpt.py"
    if not clean_script.exists():
        print(f"❌ 主脚本不存在: {clean_script}")
        return False
    
    # 构建测试命令
    test_cmd = [
        sys.executable,
        str(clean_script),
        "--name", sample_name,
        "--clean-output",
        "--rounds", "1",
        "--vlm-max-side", "512"  # 较小的尺寸以便快速测试
    ]
    
    print(f"测试命令: {' '.join(test_cmd)}")
    print("注意: 这将进行实际的API调用和SAM处理")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="测试仅API模式的SAM分割系统")
    parser.add_argument("--run-full-test", action="store_true", 
                       help="运行完整测试包括实际API调用")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 SAM Sculpt API模式测试")
    print("=" * 60)
    
    # 基础环境测试
    if not test_environment():
        print("\n❌ 环境测试失败")
        return 1
    
    if not test_paths():
        print("\n❌ 路径测试失败")
        return 1
    
    if not test_api_import():
        print("\n❌ API导入测试失败")
        return 1
    
    sample_name = test_sample_image()
    if not sample_name:
        print("\n❌ 样本图像测试失败")
        return 1
    
    # 可选的完整测试
    if args.run_full_test:
        if not run_quick_test(sample_name):
            print("\n❌ 快速测试设置失败")
            return 1
        print("\n⚠️  完整测试需要手动运行上述命令")
    
    print("\n" + "=" * 60)
    print("✅ 所有基础测试通过!")
    print("🎯 系统已准备好进行API模式的SAM分割")
    print("=" * 60)
    
    print(f"\n📋 快速开始命令:")
    print(f"python clean_sam_sculpt.py --name {sample_name} --clean-output")
    print(f"python batch_process.py --include {sample_name} --parallel 1")
    
    return 0

if __name__ == "__main__":
    exit(main())