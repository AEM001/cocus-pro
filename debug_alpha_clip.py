#!/usr/bin/env python3
"""
Debug script to isolate the Alpha-CLIP loading issue
"""

import sys
sys.path.insert(0, '.')

print("Testing Alpha-CLIP loading...")

try:
    from alpha_clip_rw.alpha_clip import load
    print("✓ Alpha-CLIP module imported successfully")

    # Test loading with explicit paths
    model_name = "checkpoints/ViT-L-14-336px.pt"
    alpha_ckpt = "checkpoints/clip_l14_336_grit_20m_4xe.pth"

    print(f"Loading model: {model_name}")
    print(f"Alpha checkpoint: {alpha_ckpt}")

    model, preprocess = load(
        model_name,
        alpha_vision_ckpt_pth=alpha_ckpt,
        device="cpu"
    )

    print("✓ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Input resolution: {model.visual.input_resolution}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()