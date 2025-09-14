conda run -n camo-vlm python -m cog_sculpt.cli \
  --name f \
  --image auxiliary/images/f.png \
  --boxes auxiliary/box_out/f/f_sam_boxes.json \
  --prior-mask auxiliary/box_out/f/f_prior_mask.png \
  --out-root pipeline_output \
  --use-prior --use-boxes --debug

•  通过修改 Config 默认值（cog_sculpt/core.py）来微调边界策略力度：
•  boundary_samples：每轮用于评分的边界采样点数（默认 24）
•  boundary_band（外带宽）/ boundary_inner_band（内带宽）
•  boundary_alpha_radius_out / boundary_alpha_radius_in：外/内圆盘半径
•  pos_neg_ratio：正负点比例
•  k_iters、max_points_per_iter、first_iter_max_points、small_roi_max_points 等