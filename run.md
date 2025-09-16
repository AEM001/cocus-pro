cd /home/albert/code/CV && timeout 360 python clean_sam_sculpt.py --name f --qwen_dir /home/albert/code/CV/models/Qwen2.5-VL-7B-Instruct-AWQ --ratio 0.5 --vlm_max_side 720 --rounds 6 2>&1

cd /home/albert/code/CV && timeout 360 python clean_sam_sculpt.py --name f --qwen_dir /home/albert/code/CV/models/Qwen2.5-VL-7B-Instruct-AWQ --ratio 0.5 --vlm_max_side 720 --rounds 1 --first_round_apply_all 2>&1