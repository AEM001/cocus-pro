#!/usr/bin/env python3
"""
Download Qwen2.5-VL-3B-Instruct from ModelScope into models/ and verify files.
"""
import os
from modelscope.hub.snapshot_download import snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
TARGET = os.path.join(ROOT, 'models', 'Qwen2.5-VL-3B-Instruct')

def main():
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    print(f"[INFO] Downloading to: {TARGET}")
    model_dir = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir=TARGET)
    print(f"[INFO] Downloaded: {model_dir}")
    # Quick check
    needed = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json']
    missing = [f for f in needed if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"[WARN] Missing files: {missing}")
    else:
        print("[OK] Basic files present.")

if __name__ == '__main__':
    main()
