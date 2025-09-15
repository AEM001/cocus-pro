# runme.md

This document summarizes how to run the SAM-first VLM sculpting workflow and all configurable parameters.

Quick start

- Activate environment and set tokenizer behavior

```bash
conda activate camo-vlm
export TOKENIZERS_PARALLELISM=false
```

- Run with local Qwen2.5-VL-AWQ and auto-SAM

```bash
python scripts/run_sculpt_simple.py <name> [--rounds N]
```

Examples

```bash
# Using Qwen (default), auto-detect SAM checkpoint under models/
python scripts/run_sculpt_simple.py f --rounds 2

# Using Mock VLM (no GPU needed)
python scripts/run_sculpt_simple.py f --rounds 2 --model mock

# Explicitly specify SAM checkpoint and device
python scripts/run_sculpt_simple.py f \
  --sam_checkpoint models/sam_vit_h_4b8939.pth \
  --sam_type vit_h \
  --sam_device cuda \
  --rounds 2

# Use a custom Qwen model directory
python scripts/run_sculpt_simple.py f --model qwen --model_dir models/Qwen2.5-VL-7B-Instruct-AWQ
```

Standard data layout

- Input image: auxiliary/images/{name}.png
- ROI + prior: auxiliary/box_out/{name}/{name}_sam_boxes.json and {name}_prior_mask.png
- Instance spec: auxiliary/llm_out/{name}_output.json (field: instance)
- Outputs: outputs/sculpt/{name}/

What the script does

1) Loads the image, ROI, prior mask, and instance (auto)
2) Generates an initial SAM mask from the ROI box (strictly clipped to ROI)
3) Performs N rounds of VLM-driven sculpting
4) Saves per-round points, peval JSON, scores JSON, masks, and final outputs

All configurable parameters

scripts/run_sculpt_simple.py

- name (positional)
  - Target sample name, matching files under auxiliary/.

- --rounds int (default: 2)
  - Number of sculpting rounds.

- --model [mock|qwen] (default: qwen)
  - Choose the VLM backend.

- --model_dir str (default: models/Qwen2.5-VL-7B-Instruct-AWQ)
  - Path to the local Qwen2.5-VL model directory.

- --sam_checkpoint str (optional)
  - Path to SAM checkpoint (.pth). If omitted, the script auto-scans under models/ for a checkpoint named like sam_*vit_[hlb]*.pth

- --sam_type [vit_h|vit_l|vit_b] (default: vit_h)
  - SAM model type matching the checkpoint.

- --sam_device [cuda|cpu] (optional)
  - Device for SAM. If omitted, inferred by torch.

- --K_pos int (default: 6)
  - Number of positive points per round.

- --K_neg int (default: 6)
  - Number of negative points per round.

Environment variables

- TOKENIZERS_PARALLELISM=false
  - Prevents tokenizers from spawning parallel workers which can cause warnings.

Notes

- SAM initial mask is strictly clipped to the ROI box to ensure initialization is contained.
- Candidate sampling prioritizes boundary points with density adapted to boundary length.
- Patch scales are small and based on target size (mask bbox) with slight context to capture both sides of the boundary.