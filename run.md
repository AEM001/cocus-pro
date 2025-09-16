python refactor_sam_sculpt.py --name f --qwen_dir /home/albert/code/CV/models/Qwen2.5-VL-3B-Instruct --ratio 0.5 --anchors_per_round 1 --vlm_max_side 720 --rounds 5

# 默认模式：绝对标号追踪（推荐）
python refactor_sam_sculpt.py --name f

# 可选模式：位置追踪（复杂情况下使用）
python refactor_sam_sculpt.py --name f --use_position_tracking --position_threshold 25

# 每个锚点只能使用1次（严格模式）
python refactor_sam_sculpt.py --name f --max_anchor_uses 1

# 每个锚点可以使用3次（宽松模式）  
python refactor_sam_sculpt.py --name f --max_anchor_uses 3

python refactor_sam_sculpt.py --name f --ratio 0.6 --anchors_per_round 1 --vlm_max_side 720 --rounds 5 --max_anchor_uses 2