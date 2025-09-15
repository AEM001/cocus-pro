#!/usr/bin/env python3
"""
Test script for Qwen VLM integration
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sculptor.vlm.qwen import QwenVLM
from sculptor.utils import load_image
import numpy as np

def test_qwen_vlm():
    # Test parameters
    model_dir = "/home/albert/code/CV/models/Qwen2.5-VL-7B-Instruct-AWQ"
    image_path = "/home/albert/code/CV/auxiliary/images/dog.png"
    
    print(f"Testing Qwen VLM with model_dir: {model_dir}")
    print(f"Using test image: {image_path}")
    
    # Create VLM instance
    vlm = QwenVLM(mode="local", model_dir=model_dir)
    
    # Load test image
    try:
        image = load_image(image_path)
        print(f"✓ Image loaded: shape {image.shape}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False
    
    # Test peval
    print("\n--- Testing Peval ---")
    try:
        result = vlm.peval(image, None, "dog")
        print(f"Peval result: {result}")
    except Exception as e:
        print(f"✗ Peval failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test pgen with a small patch
    print("\n--- Testing Pgen ---") 
    try:
        # Create a small patch from the center of the image
        h, w = image.shape[:2]
        patch = image[h//2-50:h//2+50, w//2-50:w//2+50]
        result = vlm.pgen(patch, "dog", "furry animal")
        print(f"Pgen result: {result}")
    except Exception as e:
        print(f"✗ Pgen failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_qwen_vlm()
    sys.exit(0 if success else 1)