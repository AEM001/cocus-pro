#!/usr/bin/env python3
"""
CV分割管道主入口脚本
整合 VLM定位、Alpha-CLIP评分、SAM分割的完整流程
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(
        description="CV分割管道 - 整合VLM定位、Alpha-CLIP评分、SAM分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行完整流程
  cv-segment --name scorpionfish --text "scorpionfish"
  
  # 仅运行区域提示生成
  cv-segment --stage region_prompts --name scorpionfish
  
  # 运行sculpting优化
  cv-segment --stage sculpt --name scorpionfish --text "scorpionfish"
        """
    )
    
    # 基本参数
    parser.add_argument("--config", type=str, default=None,
                       help="可选：配置文件路径（若不提供，则使用内置默认路径推断）")
    parser.add_argument("--name", type=str, required=True,
                       help="图像名称 (不含扩展名)")
    parser.add_argument("--text", type=str,
                       help="目标文本描述 (用于语义分割)")
    
    # 流程控制
    parser.add_argument("--stage", type=str, 
                       choices=["all", "region_prompts", "build_prior", "sculpt"],
                       default="all",
                       help="运行阶段: all(全流程), region_prompts(区域提示), build_prior(初始掩码), sculpt(精细雕刻)")
    
    # 输出控制
    parser.add_argument("--output-dir", type=str, default="pipeline_output",
                       help="输出目录")
    parser.add_argument("--debug", action="store_true",
                       help="保存调试信息")
    parser.add_argument("--visualize", action="store_true", 
                       help="生成可视化结果")
    
    # 模型参数
    parser.add_argument("--alpha-clip-ckpt", type=str,
                       help="Alpha-CLIP检查点路径")
    parser.add_argument("--sam-ckpt", type=str, 
                       help="SAM检查点路径") 
    parser.add_argument("--sam-type", type=str, default="vit_h",
                       choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM模型类型")
    
    # 高级参数
    parser.add_argument("--grid-size", type=str, default="3x3",
                       help="网格大小 (如 3x3)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Sculpting迭代次数")
    parser.add_argument("--max-points", type=int, default=12,
                       help="每轮最大点数")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 开始处理图像: {args.name}")
    print(f"📁 输出目录: {output_dir}")
    
    try:
        if args.stage in ["all", "region_prompts"]:
            print("\n📊 步骤 1: 生成区域提示图")
            run_region_prompts(args)
        
        if args.stage in ["all", "build_prior"]:
            print("\n🎯 步骤 2: 构建初始边界框和掩码")  
            run_build_prior(args)
        
        if args.stage in ["all", "sculpt"]:
            if not args.text:
                print("⚠️  Warning: --text is required for sculpting stage")
                return
                
            print(f"\n🎨 步骤 3: 语义引导精细雕刻 (目标: {args.text})")
            run_sculpting(args, output_dir)
        
        print(f"\n✅ 处理完成! 结果保存在: {output_dir}")
        
        # 可视化结果
        if args.visualize:
            print("\n🖼️  生成可视化结果...")
            generate_visualization(args, output_dir)
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """（保留作为兼容）可选加载配置；若未提供或不存在则返回空字典"""
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        print(f"⚠️  配置文件不存在: {config_path}，将使用内置默认路径")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def run_region_prompts(args):
    """运行区域提示生成"""
    try:
        from auxiliary.scripts.make_region_prompts import main as make_prompts
        
        # 准备参数
        # 无需外部配置：按约定从 auxiliary/images/{name}.png 读取
        img_path = f"auxiliary/images/{args.name}.png"
        sys.argv = [
            "make_region_prompts.py",
            "--name", args.name,
            "--image", img_path,
        ]
        
        make_prompts()
        print("   ✓ 区域提示图生成完成")
        
    except ImportError:
        print("   ❌ 无法导入make_region_prompts模块")
    except Exception as e:
        print(f"   ❌ 区域提示生成失败: {e}")


def run_build_prior(args):
    """运行初始边界框和掩码构建"""
    try:
        from auxiliary.scripts.build_prior_and_boxes import main as build_prior
        
        # 准备参数
        # 无需外部配置：按约定从 auxiliary/images/{name}.png 读取
        img_path = f"auxiliary/images/{args.name}.png"
        sys.argv = [
            "build_prior_and_boxes.py",
            "--name", args.name,
            "--image", img_path,
        ]
        
        build_prior()
        print("   ✓ 初始边界框和掩码构建完成")
        
    except ImportError:
        print("   ❌ 无法导入build_prior_and_boxes模块")
    except Exception as e:
        print(f"   ❌ 边界框构建失败: {e}")


def run_sculpting(args, output_dir: Path):
    """运行语义引导精细雕刻"""
    try:
        from cog_sculpt.cli import main as sculpt_main
        from cog_sculpt.cli import build_cfg_from_yaml_and_args
        
        # 构造sculpting参数
        # 直接在此处按 name 推断默认路径，避免外部配置依赖
        img_path = f"auxiliary/images/{args.name}.png"
        boxes_path = f"auxiliary/box_out/{args.name}/{args.name}_sam_boxes.json"
        prior_path = f"auxiliary/box_out/{args.name}/{args.name}_prior_mask.png"
        sculpt_args = argparse.Namespace(
            config=None,
            name=args.name,
            text=args.text,
            image=img_path,
            meta=None,
            boxes=boxes_path,
            prior_mask=prior_path,
            out_root=str(output_dir),
            grid=args.grid_size,
            k=args.iterations,
            margin=None,
            max_points=args.max_points,
            use_prior=True,
            use_boxes=True,
            debug=args.debug
        )
        
        # 构建配置
        cfg = build_cfg_from_yaml_and_args(sculpt_args)
        
        # 检查必要文件
        if not cfg.image_path or not os.path.exists(cfg.image_path):
            print(f"   ❌ 图像文件不存在: {cfg.image_path}")
            return
        print(f"   📸 图像: {cfg.image_path}")
        if cfg.boxes_json_path and os.path.exists(cfg.boxes_json_path):
            print(f"   📦 boxes: {cfg.boxes_json_path}")
        if cfg.prior_mask_path and os.path.exists(cfg.prior_mask_path):
            print(f"   🧩 prior: {cfg.prior_mask_path}")
        
        print(f"   🎯 目标: {args.text}")
        print(f"   📐 网格: {args.grid_size}")
        print(f"   🔄 迭代: {args.iterations}轮")
        
        # 运行sculpting
        sculpt_main()
        print("   ✓ 语义引导精细雕刻完成")
        
    except ImportError as e:
        print(f"   ❌ 无法导入sculpting模块: {e}")
    except Exception as e:
        print(f"   ❌ 精细雕刻失败: {e}")


def generate_visualization(args, output_dir: Path):
    """生成可视化结果"""
    try:
        # 寻找结果文件（兼容两种保存位置）
        flat_mask = output_dir / "final_mask.png"
        flat_overlay = output_dir / "final_overlay.png"
        nested_dir = output_dir / (args.name or "scene")
        nested_mask = nested_dir / "final_mask.png"
        nested_overlay = nested_dir / "final_overlay.png"
        final_mask_path = flat_mask if flat_mask.exists() else nested_mask
        final_overlay_path = flat_overlay if flat_overlay.exists() else nested_overlay
        
        if final_mask_path.exists():
            print(f"   📄 最终掩码: {final_mask_path}")
        if final_overlay_path.exists():
            print(f"   🖼️  叠加可视化: {final_overlay_path}")
        
        # 如果有调试信息，列出调试文件
        debug_dir = output_dir if (output_dir / "iter_00_debug.json").exists() else nested_dir
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("iter_*_debug.json"))
            if debug_files:
                print(f"   🔍 调试文件: {len(debug_files)} 个迭代记录")
        
    except Exception as e:
        print(f"   ⚠️  可视化生成出错: {e}")


if __name__ == "__main__":
    main()