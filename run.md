python refactor_sam_sculpt.py --name f --qwen_dir /home/albert/code/CV/models/Qwen2.5-VL-3B-Instruct --ratio 0.5 --vlm_max_side 720 --rounds 5

# 默认模式：绝对标号追踪（推荐）
python refactor_sam_sculpt.py --name f

# 可选模式：位置追踪（复杂情况下使用）
python refactor_sam_sculpt.py --name f --use_position_tracking --position_threshold 25

# 每个锚点只能使用1次（严格模式）
python refactor_sam_sculpt.py --name f --max_anchor_uses 1

# 每个锚点可以使用3次（宽松模式）  
python refactor_sam_sculpt.py --name f --max_anchor_uses 3

python refactor_sam_sculpt.py --name f --ratio 0.6 --vlm_max_side 1024 --rounds 5 



python clean_sam_sculpt.py --name f --ratio 0.5 --rounds 3

# 支持的参数（简化版）
--name          # 样本名称
--qwen_dir      # Qwen模型路径  
--rounds        # 优化轮数
--ratio         # 切线正方形与ROI的比例
--vlm_max_side  # VLM输入图像最大边长