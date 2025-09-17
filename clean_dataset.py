#!/usr/bin/env python3
"""
数据集清理脚本
功能：只保留路径以 "COD10K_TEST_DIR/Imgs" 开头的样本，并同步清理class_info.json
"""

import json
import os
from collections import defaultdict

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_sample_info(sample_info_path):
    """清理sample_info.json，只保留COD10K_TEST_DIR/Imgs路径的样本"""
    print(f"加载 {sample_info_path}...")
    samples = load_json(sample_info_path)
    
    print(f"原始样本数量: {len(samples)}")
    
    # 过滤样本
    filtered_samples = []
    kept_classes = set()
    
    for sample in samples:
        image_path = sample.get('image', '')
        if image_path.startswith('COD10K_TEST_DIR/Imgs/'):
            filtered_samples.append(sample)
            kept_classes.add(sample.get('base_class', ''))
            print(f"保留: {sample['unique_id']} - {sample['base_class']} - {image_path}")
    
    print(f"\n过滤后样本数量: {len(filtered_samples)}")
    print(f"保留的类别: {sorted(kept_classes)}")
    
    # 重新分配ID
    for i, sample in enumerate(filtered_samples):
        sample['id'] = i
    
    return filtered_samples, kept_classes

def clean_class_info(class_info_path, kept_classes):
    """清理class_info.json，只保留出现在kept_classes中的类别"""
    print(f"\n加载 {class_info_path}...")
    classes = load_json(class_info_path)
    
    print(f"原始类别数量: {len(classes)}")
    
    # 过滤类别
    filtered_classes = []
    
    for class_info in classes:
        class_name = class_info.get('name', '')
        if class_name in kept_classes:
            filtered_classes.append(class_info)
            print(f"保留类别: {class_name} (ID: {class_info.get('id')}, 样本数: {class_info.get('num_samples')})")
    
    print(f"\n过滤后类别数量: {len(filtered_classes)}")
    
    # 重新分配ID
    for i, class_info in enumerate(filtered_classes):
        class_info['id'] = i
    
    return filtered_classes

def verify_images_exist(samples, base_path="/home/albert/code/CV/dataset"):
    """验证图像文件是否存在"""
    print(f"\n验证图像文件存在性...")
    missing_files = []
    existing_files = []
    
    for sample in samples:
        image_path = sample.get('image', '')
        full_path = os.path.join(base_path, image_path)
        
        if os.path.exists(full_path):
            existing_files.append(image_path)
        else:
            missing_files.append(image_path)
            print(f"⚠️  文件不存在: {full_path}")
    
    print(f"存在的文件: {len(existing_files)}")
    print(f"缺失的文件: {len(missing_files)}")
    
    return existing_files, missing_files

def main():
    base_dir = "/home/albert/code/CV/dataset"
    sample_info_path = os.path.join(base_dir, "sample_info.json")
    class_info_path = os.path.join(base_dir, "class_info.json")
    
    # 备份原始文件
    print("备份原始文件...")
    os.system(f"cp {sample_info_path} {sample_info_path}.backup")
    os.system(f"cp {class_info_path} {class_info_path}.backup")
    print("✓ 备份完成")
    
    # 清理sample_info.json
    filtered_samples, kept_classes = clean_sample_info(sample_info_path)
    
    # 验证图像文件存在性
    existing_files, missing_files = verify_images_exist(filtered_samples)
    
    # 如果有缺失文件，进一步过滤
    if missing_files:
        print(f"\n发现 {len(missing_files)} 个缺失文件，将从样本中移除...")
        final_samples = []
        for sample in filtered_samples:
            if sample.get('image', '') in existing_files:
                final_samples.append(sample)
        
        # 重新分配ID
        for i, sample in enumerate(final_samples):
            sample['id'] = i
            
        filtered_samples = final_samples
        print(f"最终样本数量: {len(filtered_samples)}")
    
    # 清理class_info.json
    filtered_classes = clean_class_info(class_info_path, kept_classes)
    
    # 保存清理后的文件
    print(f"\n保存清理后的文件...")
    save_json(filtered_samples, sample_info_path)
    save_json(filtered_classes, class_info_path)
    
    # 统计信息
    print(f"\n" + "="*60)
    print(f"清理完成！")
    print(f"="*60)
    print(f"样本信息 ({sample_info_path}):")
    print(f"  - 最终样本数量: {len(filtered_samples)}")
    print(f"  - 保留的类别: {len(kept_classes)} 个")
    
    print(f"\n类别信息 ({class_info_path}):")
    print(f"  - 最终类别数量: {len(filtered_classes)}")
    
    # 显示样本分布
    class_counts = defaultdict(int)
    for sample in filtered_samples:
        class_counts[sample['base_class']] += 1
    
    print(f"\n类别分布:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  - {class_name}: {count} 个样本")
    
    print(f"\n备份文件:")
    print(f"  - {sample_info_path}.backup")
    print(f"  - {class_info_path}.backup")
    
    print(f"\n验证:")
    print(f"  - 所有保留样本的图像路径都以 'COD10K_TEST_DIR/Imgs/' 开头")
    print(f"  - 所有图像文件都存在于磁盘上")
    print(f"  - ID已重新分配 (0 到 {len(filtered_samples)-1})")

if __name__ == "__main__":
    main()