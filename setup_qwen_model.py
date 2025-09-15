#!/usr/bin/env python3
"""
Setup script for Qwen2.5-VL-7B-Instruct-AWQ model
Provides multiple download methods and verification
"""
import os
import subprocess
import sys
from pathlib import Path

MODEL_DIR = "/home/albert/code/CV/models/Qwen2.5-VL-7B-Instruct-AWQ"
REPO_URL = "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

def check_git_lfs():
    """Check if git lfs is available"""
    try:
        result = subprocess.run(["git", "lfs", "version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def download_with_git():
    """Download model using git clone"""
    try:
        print("Downloading with git clone...")
        os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
        
        cmd = ["git", "clone", REPO_URL, MODEL_DIR]
        result = subprocess.run(cmd, cwd=os.path.dirname(MODEL_DIR))
        
        if result.returncode == 0:
            print(f"Successfully downloaded to {MODEL_DIR}")
            return True
        else:
            print(f"Git clone failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Git clone failed: {e}")
        return False

def verify_model_files():
    """Verify that essential model files exist"""
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        return False, "Model directory doesn't exist"
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        return False, f"Missing files: {missing_files}"
    
    # Check for model weights (AWQ format)
    weight_files = list(model_path.glob("*.safetensors")) or list(model_path.glob("pytorch_model*.bin"))
    if not weight_files:
        return False, "No model weight files found"
    
    return True, f"Model verified. Found {len(weight_files)} weight file(s)"

def print_manual_instructions():
    """Print manual download instructions"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print(f"If automatic download fails, please manually download from:")
    print(f"{REPO_URL}")
    print(f"\nPlace all files in: {MODEL_DIR}")
    print("\nAlternatively, use one of these methods:")
    print("\n1. Using wget/curl (for individual files):")
    print(f"   mkdir -p {MODEL_DIR}")
    print(f"   cd {MODEL_DIR}")
    print("   # Download each file individually from the HF repo")
    
    print("\n2. Using git with LFS:")
    print(f"   git lfs install")
    print(f"   git clone {REPO_URL} {MODEL_DIR}")
    
    print("\n3. Using browser download:")
    print(f"   Visit: {REPO_URL}")
    print("   Click 'Download' on each file and save to the model directory")
    
    print("\nRequired files include:")
    print("   - config.json")
    print("   - tokenizer_config.json") 
    print("   - preprocessor_config.json")
    print("   - *.safetensors (model weights)")
    print("   - tokenizer.json")
    print("="*60)

def main():
    print(f"Setting up Qwen2.5-VL-7B-Instruct-AWQ model...")
    print(f"Target directory: {MODEL_DIR}")
    
    # Check if already downloaded
    is_valid, message = verify_model_files()
    if is_valid:
        print(f"✓ Model already exists and verified: {message}")
        return True
    else:
        print(f"✗ Model verification failed: {message}")
    
    # Try downloading with git
    if check_git_lfs():
        print("Git LFS detected, attempting download...")
        if download_with_git():
            is_valid, message = verify_model_files()
            if is_valid:
                print(f"✓ Download successful: {message}")
                return True
            else:
                print(f"✗ Download completed but verification failed: {message}")
    else:
        print("Git LFS not available, skipping git download")
    
    # If all automatic methods fail, provide manual instructions
    print_manual_instructions()
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ Setup complete! You can now use the Qwen model with:")
        print("python scripts/run_sculpt.py --model qwen --model_dir /home/albert/code/CV/models/Qwen2.5-VL-7B-Instruct-AWQ ...")
    else:
        print("\n✗ Automatic setup failed. Please follow the manual instructions above.")
        sys.exit(1)