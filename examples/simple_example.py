#!/usr/bin/env python3
"""
CV语义分割管道简单使用示例

本示例展示如何使用Alpha-CLIP和SAM进行语义引导的图像分割。
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("🚀 CV语义分割管道示例")
    print("=" * 50)
    
    # 检查测试图像是否存在
    test_image_path = "auxiliary/images/q.png"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        print("请确保auxiliary/data/images/目录下有测试图像")
        return
    
    try:
        # 1. 加载图像
        print("\n📸 加载测试图像...")
        image = np.array(Image.open(test_image_path))
        print(f"   原始图像尺寸: {image.shape}")

        # 为SAM调整图像尺寸（长边不超过1024）
        if len(image.shape) == 3 and image.shape[2] == 4:
            # 如果是RGBA，转换为RGB
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image

        # 调整尺寸以适配SAM
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image_rgb)
        w, h = pil_img.size
        max_side = max(w, h)
        if max_side > 1024:
            scale = 1024 / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            image_for_sam = np.array(pil_img)
        else:
            image_for_sam = image_rgb

        print(f"   SAM处理后尺寸: {image_for_sam.shape}")

        # 保持原始图像用于Alpha-CLIP
        image = image_rgb
        
        # 2. 初始化Alpha-CLIP模型
        print("\n🎯 初始化Alpha-CLIP模型（强制 Alpha 分支权重）...")
        from alpha_clip_rw import AlphaCLIPInference
        try:
            alpha_clip = AlphaCLIPInference(
                model_name="ViT-L/14@336px",
                alpha_vision_ckpt_pth="AUTO",  # 从 env 或 checkpoints/ 自动查找
                device="cpu"
            )
            print("   ✓ Alpha-CLIP模型加载成功")
        except Exception as e:
            print(f"   ❌ Alpha-CLIP加载失败: {e}")
            print("   请将 Alpha-CLIP 视觉分支权重放在 checkpoints/clip_l14_336_grit_20m_4xe.pth，"
                  "或设置环境变量 ALPHA_CLIP_ALPHA_CKPT 后重试。")
            return
        
        # 3. 初始化SAM模型
        print("\n🔧 初始化SAM模型（需要本地权重）...")
        from sam_integration import create_sam_wrapper
        sam_ckpt = "models/sam_vit_h_4b8939.pth"
        if not os.path.isfile(sam_ckpt):
            print(f"   ❌ 未找到 SAM 权重: {sam_ckpt}")
            print("   请下载 sam_vit_h_4b8939.pth 到 models/ 目录后重试。")
            return
        sam_wrapper = create_sam_wrapper(
            model_type="sam",
            checkpoint_path=sam_ckpt
        )
        print("   ✓ SAM模型加载成功")
        
        # 4. 演示Alpha-CLIP语义评分
        if alpha_clip:
            print("\n🧠 演示Alpha-CLIP语义评分...")
            
            # 创建一个简单的测试掩码
            H, W = image.shape[:2]
            test_alpha = np.zeros((H, W), dtype=np.float32)
            test_alpha[H//4:3*H//4, W//4:3*W//4] = 1.0  # 中心区域
            
            # 计算语义相似度
            score = alpha_clip.score_region_with_templates(
                image=image,
                alpha=test_alpha,
                instance_text="blurred person standing in field",
                templates=[
                    "a photo of the {} camouflaged in the background.",
                    "a photo of the {}.",
                    "the {} in the scene."
                ]
            )
            
            print(f"   语义相似度分数: {score:.4f}")
            
            # 测试不同区域的评分
            print("   测试不同区域的语义评分:")
            regions = [
                ("左上角", test_alpha * 0, [0, H//2, 0, W//2]),
                ("中心区域", test_alpha, [H//4, 3*H//4, W//4, 3*W//4]),
                ("右下角", test_alpha * 0, [H//2, H, W//2, W])
            ]
            
            for name, alpha_region, bbox in regions:
                alpha_mask = np.zeros_like(test_alpha)
                alpha_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1.0
                
                score = alpha_clip.score_region_with_templates(
                    image=image, 
                    alpha=alpha_mask, 
                    instance_text="blurred person standing in field"
                )
                print(f"     {name}: {score:.4f}")
        
        # 5. 演示语义引导SAM（如果模型可用）
        if alpha_clip and sam_wrapper:
            print("\n🎨 演示语义引导SAM分割...")
            
            from sam_integration import SemanticGuidedSAM
            
            # 创建语义引导SAM
            semantic_sam = SemanticGuidedSAM(sam_wrapper, alpha_clip)
            
            # 定义测试边界框 - 基于调整后的图像
            H_sam, W_sam = image_for_sam.shape[:2]
            bbox = [W_sam//4, H_sam//4, 3*W_sam//4, 3*H_sam//4]  # [x1, y1, x2, y2]
            
            # 运行语义分割 - 使用调整后的图像
            result_mask = semantic_sam.segment_with_semantic_points(
                image=image_for_sam,
                bbox=bbox,
                instance_text="",
                grid_size=(2, 2),  # 简化网格
                max_points=8,
                iterations=2
            )
            
            print(f"   生成掩码尺寸: {result_mask.shape}")
            print(f"   掩码覆盖率: {result_mask.sum() / result_mask.size * 100:.1f}%")
            
            # 保存结果
            output_dir = Path("examples/output")
            output_dir.mkdir(exist_ok=True)
            
            # 保存掩码
            Image.fromarray((result_mask * 255).astype(np.uint8)).save(
                output_dir / "semantic_mask.png"
            )
            
            # 保存可视化叠加 - 使用调整后的图像
            overlay = image_for_sam.copy()
            overlay[result_mask > 0] = (
                overlay[result_mask > 0] * 0.7 +
                np.array([0, 255, 0], dtype=np.uint8) * 0.3
            ).astype(np.uint8)
            Image.fromarray(overlay).save(output_dir / "semantic_overlay.png")
            
            print(f"   结果已保存到: {output_dir}")
        
        # 6. 演示Cog-Sculpt核心API
        print("\n⚙️ 演示Cog-Sculpt核心API...")
        
        from cog_sculpt.core import (
            Config, Box, DensityScorer, 
            partition_grid, threshold_cells
        )
        
        # 创建配置
        config = Config(
            grid_init=(2, 2),
            k_iters=2,
            max_points_per_iter=6
        )
        
        # 创建测试边界框
        bbox = Box(r0=H//4, c0=W//4, r1=3*H//4, c1=3*W//4)
        
        # 网格化
        cells = partition_grid(bbox, config.grid_init)
        print(f"   生成网格单元数: {len(cells)}")
        
        # 模拟评分
        scorer = DensityScorer()
        test_mask = np.ones((H, W), dtype=np.uint8)
        test_mask[H//3:2*H//3, W//3:2*W//3] = 0  # 创建洞
        
        for cell in cells:
            b = cell.box
            alpha = np.zeros_like(test_mask, dtype=np.float32)
            alpha[b.r0:b.r1, b.c0:b.c1] = test_mask[b.r0:b.r1, b.c0:b.c1]
            cell.score = scorer.score(image, alpha, "test")
        
        # 阈值分类
        pos_cells, neg_cells, unc_cells = threshold_cells(cells)
        print(f"   正例网格: {len(pos_cells)}, 负例网格: {len(neg_cells)}, 不确定: {len(unc_cells)}")
        
        print("\n✅ 示例演示完成!")
        
        # 生成总结
        print("\n📋 功能总结:")
        print("   ✓ 图像加载和预处理")
        if alpha_clip:
            print("   ✓ Alpha-CLIP语义评分")
        if alpha_clip and sam_wrapper:
            print("   ✓ 语义引导SAM分割")
        print("   ✓ Cog-Sculpt核心算法")
        print("\n🔗 更多用法请参考docs/USAGE.md")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
