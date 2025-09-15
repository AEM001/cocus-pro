#!/usr/bin/env python3
"""
Download Qwen2.5-VL-7B-Instruct-AWQ model with retry logic
"""
import os
import time
from huggingface_hub import snapshot_download, logging

# Enable debug logging
logging.set_verbosity_info()

def download_with_retries(repo_id, local_dir, max_retries=3):
    """Download model with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Downloading {repo_id}")
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True,
                local_dir_use_symlinks=False,
                repo_type="model"
            )
            print(f"Successfully downloaded to: {path}")
            return path
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("All attempts failed!")
                raise

if __name__ == "__main__":
    model_dir = "/home/albert/code/CV/models/Qwen2.5-VL-7B-Instruct-AWQ"
    os.makedirs(model_dir, exist_ok=True)
    
    download_with_retries(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        local_dir=model_dir
    )