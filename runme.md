# 激活环境
conda activate camo-vlm
export TOKENIZERS_PARALLELISM=false

# 使用 Qwen2.5-VL 运行雕刻流水线
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model qwen \
  --model_dir models/Qwen2.5-VL-7B-Instruct-AWQ \
  --rounds 2