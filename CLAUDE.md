# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision research project implementing semantic-guided iterative segmentation using Alpha-CLIP and SAM (Segment Anything Model). The system performs ROI-based segmentation through iterative "sculpting" - using semantic scoring to guide positive/negative point selection for SAM refinement over ~3 iterations.

## Key Architecture Components

### Core Modules
- **`alpha_clip_rw/`**: Alpha-CLIP inference wrapper supporting 4-channel RGBA input for semantic scoring
  - `alpha_clip.py`: CLIP/Alpha-CLIP model loading and image transforms
  - `inference.py`: Main inference class with `score_region_with_templates()` for multi-template semantic scoring
  - `model.py`: Model definitions with Alpha branch for vision intervention

- **`cog_sculpt/`**: Main algorithmic pipeline implementing the semantic sculpting process
  - `core.py`: Core pipeline functions including grid partitioning, scoring, point selection, and SAM integration. Entry point: `sculpting_pipeline()`
  - `cli.py`: Command-line interface for running the segmentation pipeline

- **`sam_integration.py`**: SAM wrapper classes providing unified interface for SAM/SAM2
  - `SAMWrapper`/`SAM2Wrapper`: Unified API for `set_image`, `predict_with_box`, `predict_with_points`
  - `SemanticGuidedSAM`: Integrated class combining semantic scoring with SAM interaction

### Supporting Infrastructure
- **`auxiliary/scripts/`**: Data preprocessing utilities
  - `make_region_prompts.py`: Creates 9×9 region grid prompts and metadata
  - `build_prior_and_boxes.py`: Generates prior masks and SAM boxes from LLM outputs

## Common Development Commands

### Installation
```bash
# Install the package in development mode
pip install -e .

# Note: requirements.txt referenced in setup.py but not present in repo
# Install dependencies manually as needed:
# pip install numpy pillow pyyaml opencv-python torch torchvision segment-anything
```

### Running the Pipeline
```bash
# Main semantic sculpting pipeline
python -m cog_sculpt.cli \
  --text "scorpionfish" \
  --image auxiliary/images/q.png \
  --boxes auxiliary/box_out/q/q_sam_boxes.json \
  --prior-mask auxiliary/box_out/q/q_prior_mask.png \
  --name q \
  --out-root pipeline_output \
  --use-prior --use-boxes --debug

# Preprocessing: Generate prior masks and boxes
python auxiliary/scripts/build_prior_and_boxes.py \
  --image auxiliary/images/q.png \
  --meta auxiliary/out/q/q_meta.json \
  --pred auxiliary/llm_out/q_output.json \
  --out auxiliary/box_out
```

### Testing
```bash
# No formal test suite detected
# Test imports and basic functionality:
python examples/simple_example.py
python debug_alpha_clip.py
```

## Algorithm Flow (guide.md Implementation)

The pipeline implements a 6-step iterative refinement process:

1. **Input**: Image, ROI (from bbox), initial mask M0, text description T
2. **Grid Partitioning**: Divide ROI into 3×3 cells (recursive subdivision for uncertain regions)
3. **Alpha Soft Prior Scoring**: Use Alpha-CLIP with mask as alpha channel for semantic confidence scoring
4. **Threshold & Point Selection**: Adaptive thresholds (μ±0.5σ) select positive/negative points from high/low confidence cells
5. **SAM Interaction**: Use selected points to refine mask via SAM
6. **Iteration**: Repeat for k≈3 rounds with early stopping

## Configuration

Key parameters in `cog_sculpt/core.py:Config`:
- `k_iters: int = 3` - Number of sculpting iterations
- `grid_init: Tuple[int, int] = (3, 3)` - Initial grid size
- `pos_neg_ratio: float = 2.0` - Ratio of positive to negative points
- `iou_eps: float = 5e-3` - Early stopping threshold
- `prompt_template: str` - Text template for semantic scoring

## Known Issues & Improvements

- Early stopping condition at `cog_sculpt/core.py:487` uses `if iou(M_new, M) < cfg.iou_eps: break` which may be counterintuitive (suggest using high IoU for convergence)
- `requirements.txt` referenced in `setup.py` but not present in repository
- Recursive cell subdivision implemented but not enabled in main loop

## Dependencies

- Core: `numpy`, `Pillow`, `PyYAML`, `opencv-python`
- Alpha-CLIP: `torch`, `torchvision`, `tqdm`, `packaging`, `loralib`
- SAM: `segment-anything` (or `sam2`) with model weights