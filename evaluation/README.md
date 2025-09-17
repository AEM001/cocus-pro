# Evaluation for OVCOS on COD10K subset

This folder contains scripts to:
- Select the first N samples from dataset/sample_info.json
- Prepare ROI inputs (via prepare_sample.py) and run the 3-round SAM+VLM refinement
- Collect predicted masks and GT masks
- Compute COS metrics (S/E/Fwβ/MAE) and optionally class-aware metrics if label files are provided

Usage quickstart:

1) Run 100-sample evaluation (3 rounds, API model qwen-vl-plus-latest):
   DASHSCOPE_API_KEY=... python evaluation/run_eval.py --limit 100 --rounds 3 --api-model qwen-vl-plus-latest

2) Results will be under evaluation/runs/<timestamp>/
   - pred_masks/: collected predictions
   - gt_masks/: GT masks copied from dataset
   - per_image_metrics.csv: per-image COS metrics
   - summary.json: aggregated COS and (if available) class-aware metrics
