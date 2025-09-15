#!/usr/bin/env python3
"""
Debug version of sculpt_simple.py with memory monitoring and step-by-step execution
"""
import os
import sys
import json
import psutil
import gc
import torch

# Add src to path
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def print_memory_usage(step):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_mem_cached = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"[{step}] Memory: CPU={mem_mb:.1f}MB, GPU_alloc={gpu_mem_alloc:.1f}MB, GPU_cached={gpu_mem_cached:.1f}MB")
    else:
        print(f"[{step}] Memory: CPU={mem_mb:.1f}MB")

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "f"
    print(f"Debug run for sample: {name}")
    
    print_memory_usage("START")
    
    # Step 1: Load basic modules
    print("\n=== Step 1: Loading modules ===")
    from sculptor.utils import load_image, load_mask, parse_sam_boxes_json
    from sculptor.types import ROIBox
    print_memory_usage("MODULES_LOADED")
    
    # Step 2: Load input data
    print("\n=== Step 2: Loading input data ===")
    base_dir = THIS_DIR
    image_path = os.path.join(base_dir, 'auxiliary', 'images', f'{name}.png')
    prior_mask_path = os.path.join(base_dir, 'auxiliary', 'box_out', name, f'{name}_prior_mask.png')
    roi_json_path = os.path.join(base_dir, 'auxiliary', 'box_out', name, f'{name}_sam_boxes.json')
    
    print(f"Loading image: {image_path}")
    I = load_image(image_path)
    print(f"Image shape: {I.shape}")
    
    print(f"Loading prior mask: {prior_mask_path}")
    prior_mask = load_mask(prior_mask_path)
    print(f"Prior mask shape: {prior_mask.shape}")
    
    print(f"Loading ROI: {roi_json_path}")
    boxes = parse_sam_boxes_json(roi_json_path)
    B = ROIBox(*boxes[0])
    print(f"ROI: {B.x0:.1f}, {B.y0:.1f}, {B.x1:.1f}, {B.y1:.1f}")
    
    print_memory_usage("DATA_LOADED")
    
    # Step 3: Try SAM loading
    print("\n=== Step 3: Loading SAM ===")
    try:
        from sculptor.sam_backends import build_sam_backend
        models_dir = os.path.join(base_dir, "models")
        sam_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and 'sam' in f.lower()]
        if sam_files:
            sam_path = os.path.join(models_dir, sam_files[0])
            print(f"Found SAM checkpoint: {sam_path}")
            print(f"File size: {os.path.getsize(sam_path) / 1024 / 1024:.1f} MB")
            
            print("Building SAM backend...")
            sam_backend = build_sam_backend(
                checkpoint_path=sam_path,
                model_type="vit_h",
                device="cuda"
            )
            print("SAM backend loaded successfully")
            print_memory_usage("SAM_LOADED")
        else:
            print("No SAM checkpoint found")
            return
    except Exception as e:
        print(f"SAM loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Try VLM loading
    print("\n=== Step 4: Loading VLM ===")
    try:
        from sculptor.vlm.qwen import QwenVLM
        model_dir = os.path.join(base_dir, "models/Qwen2.5-VL-7B-Instruct-AWQ")
        print(f"Loading Qwen from: {model_dir}")
        
        vlm = QwenVLM(mode="local", model_dir=model_dir)
        print("VLM instance created")
        print_memory_usage("VLM_CREATED")
        
        # Try a simple peval to trigger model loading
        import numpy as np
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        print("Testing VLM with small image...")
        result = vlm.peval(test_image, None, "test")
        print(f"VLM test result: {result}")
        print_memory_usage("VLM_LOADED")
        
    except Exception as e:
        print(f"VLM loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== All components loaded successfully ===")
    print_memory_usage("ALL_LOADED")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print_memory_usage("CLEANUP")

if __name__ == "__main__":
    main()