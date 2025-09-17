#!/usr/bin/env python3
import os
import json
import glob

def check_specific_sample(sample_name):
    """检查特定样本的数据完整性"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"=== 检查样本: {sample_name} ===")
    
    # 检查图像文件
    img_path = os.path.join(base_dir, 'dataset', 'COD10K_TEST_DIR', 'Imgs', f'{sample_name}.jpg')
    img_exists = os.path.exists(img_path)
    print(f"图像文件: {'✓' if img_exists else '✗'} {img_path}")
    
    # 检查BOX数据
    box_file = os.path.join(base_dir, 'auxiliary', 'box_out', sample_name, f"{sample_name}_sam_boxes.json")
    box_exists = os.path.exists(box_file)
    print(f"BOX数据:  {'✓' if box_exists else '✗'} {box_file}")
    
    # 检查LLM数据
    llm_file = os.path.join(base_dir, 'auxiliary', 'llm_out', f"{sample_name}_output.json")
    llm_exists = os.path.exists(llm_file)
    print(f"LLM数据:  {'✓' if llm_exists else '✗'} {llm_file}")
    
    if box_exists:
        try:
            with open(box_file, 'r') as f:
                box_data = json.load(f)
            print(f"BOX内容: {box_data}")
        except Exception as e:
            print(f"BOX读取错误: {e}")
    
    if llm_exists:
        try:
            with open(llm_file, 'r') as f:
                llm_data = json.load(f)
            print(f"LLM实例: {llm_data.get('instance', 'N/A')}")
        except Exception as e:
            print(f"LLM读取错误: {e}")
    
    return img_exists, box_exists, llm_exists

if __name__ == "__main__":
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else "COD10K-CAM-1-Aquatic-13-Pipefish-532"
    check_specific_sample(sample)