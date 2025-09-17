#!/usr/bin/env python3
"""
样本查找工具 - 映射图像文件和辅助数据
"""
import os
import glob

def find_matching_samples():
    """找到有完整数据的样本"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 扫描所有图像文件
    img_dir = os.path.join(base_dir, 'dataset', 'COD10K_TEST_DIR', 'Imgs')
    box_dir = os.path.join(base_dir, 'auxiliary', 'box_out')
    llm_dir = os.path.join(base_dir, 'auxiliary', 'llm_out')
    
    print("=== 可用样本检查 ===")
    
    # 查找box_out中的样本
    box_samples = []
    if os.path.exists(box_dir):
        for item in os.listdir(box_dir):
            if os.path.isdir(os.path.join(box_dir, item)):
                box_file = os.path.join(box_dir, item, f"{item}_sam_boxes.json")
                if os.path.exists(box_file):
                    box_samples.append(item)
    
    print(f"找到 {len(box_samples)} 个具有BOX数据的样本:")
    for i, sample in enumerate(box_samples[:10]):  # 显示前10个
        llm_file = os.path.join(llm_dir, f"{sample}_output.json")
        has_llm = "✓" if os.path.exists(llm_file) else "✗"
        print(f"  {i+1:2d}. {sample} [LLM: {has_llm}]")
    
    if len(box_samples) > 10:
        print(f"  ... 还有 {len(box_samples) - 10} 个样本")
    
    return box_samples

def check_sample_data(sample_name):
    """检查特定样本的数据完整性"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n=== 样本数据检查: {sample_name} ===")
    
    # 检查图像文件
    img_patterns = [
        f"dataset/COD10K_TEST_DIR/Imgs/{sample_name}.jpg",
        f"dataset/COD10K_TEST_DIR/Imgs/{sample_name}*.jpg"
    ]
    
    img_found = None
    for pattern in img_patterns:
        matches = glob.glob(os.path.join(base_dir, pattern))
        if matches:
            img_found = matches[0]
            break
    
    print(f"图像文件: {'✓' if img_found else '✗'} {img_found or 'NOT FOUND'}")
    
    # 检查BOX数据
    box_file = os.path.join(base_dir, 'auxiliary', 'box_out', sample_name, f"{sample_name}_sam_boxes.json")
    print(f"BOX数据:  {'✓' if os.path.exists(box_file) else '✗'} {box_file}")
    
    # 检查LLM数据
    llm_file = os.path.join(base_dir, 'auxiliary', 'llm_out', f"{sample_name}_output.json")
    print(f"LLM数据:  {'✓' if os.path.exists(llm_file) else '✗'} {llm_file}")
    
    return img_found, os.path.exists(box_file), os.path.exists(llm_file)

if __name__ == "__main__":
    samples = find_matching_samples()
    
    if samples:
        print(f"\n推荐使用样本: {samples[0]}")
        check_sample_data(samples[0])
        print(f"\n运行命令示例:")
        print(f"python clean_sam_sculpt.py --name {samples[0]} --rounds 1 --save-points-vis")