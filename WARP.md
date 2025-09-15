# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This is a VLM-driven segmentation refinement system that implements an iterative "sculpting" approach. The system takes an initial mask and ROI, then uses Vision-Language Models (VLM) to evaluate the mask quality and guide point selection for SAM (Segment Anything Model) refinement over multiple rounds.

## Common Development Commands

### Basic Development
```bash
# Run with mock VLM (for development/testing)
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model mock \
  --rounds 2 \
  --out_dir outputs/sculpt/dog

# Run with local Qwen2.5-VL-AWQ model
export TOKENIZERS_PARALLELISM=false
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model qwen \
  --model_dir models/Qwen2.5-VL-7B-Instruct-AWQ \
  --rounds 2

# Run with server mode Qwen2.5-VL (requires server setup)
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model qwen \
  --server_url http://127.0.0.1:8000/generate \
  --rounds 2

# Run with SAM integration
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model mock \
  --sam_checkpoint models/sam_vit_h_4b8939.pth \
  --sam_type vit_h \
  --sam_device cuda \
  --rounds 2
```

### Testing and Validation
```bash
# Test single components (examine auxiliary/ outputs)
ls auxiliary/box_out/  # View available test images
ls auxiliary/images/   # RGB test images

# Check outputs
ls outputs/sculpt_test/  # View recent test outputs

# Test Qwen2.5-VL integration directly
export TOKENIZERS_PARALLELISM=false
python test_qwen.py

# Setup Qwen2.5-VL model (download and verify)
python setup_qwen_model.py

# Validate mask progression
python -c "
from src.sculptor.utils import load_mask, load_image
mask = load_mask('outputs/sculpt/dog/final_mask.png')
print(f'Final mask coverage: {(mask > 0).sum()} pixels')
"
```

## Architecture Overview

### Core Pipeline Flow
1. **Candidate Sampling** (`candidates.py`): Samples points inside mask, on boundary band, and outside mask using distance transform and spatial heuristics
2. **Patch Extraction** (`patches.py`): Extracts multi-scale patches around candidate points for VLM evaluation
3. **Semantic Evaluation** (`vlm/`): 
   - **Peval**: VLM analyzes full mask overlay to diagnose defects
   - **Pgen**: VLM scores individual patches for target membership
4. **Point Selection** (`select_points.py`): Converts VLM scores to positive/negative point sets using adaptive thresholds and spatial NMS
5. **SAM Refinement** (`sam_refine.py`): Uses selected points to refine mask via SAM, with early stopping

### Key Design Patterns

**Modular Pipeline**: Each step is isolated and file-cached, enabling easy debugging and component swapping.

**VLM Abstraction**: All VLMs implement `VLMBase` interface with `peval()` and `pgen()` methods. This allows switching between MockVLM (deterministic testing) and real models.

**SAM Backend Injection**: SAM integration is optional and injected via `SamPredictorWrapper`, defaulting to no-op if not available.

**Coordinate System**: All algorithms work in absolute image coordinates. ROI boxes handle clipping and coordinate transforms.

### Module Relationships

- `scripts/run_sculpt.py` orchestrates the full pipeline
- `src/sculptor/types.py` defines core data structures (`ROIBox`, `Candidate`)
- `src/sculptor/utils.py` provides shared utilities (image I/O, NMS, overlays)
- VLM modules (`vlm/`) are completely interchangeable via the `VLMBase` interface
- SAM backends (`sam_backends.py`) adapt different SAM implementations

### Configuration and Extensibility

**Sampling Parameters**: Located in respective modules and easily adjustable:
- Candidate sampling: `max_total=30`, `nms_r_ratio=0.06` 
- Multi-scale patches: `scales=[0.08, 0.12, 0.18]` (relative to ROI size)
- Point selection: `K_pos=6`, `K_neg=6` (positive/negative points per round)

**VLM Integration Points**:
- For new VLM: Extend `VLMBase` and implement `peval`/`pgen` methods
- Prompts are centralized in `vlm/prompts.py` with forced JSON output format
- Mock VLM provides stable outputs for development without real models

**SAM Integration Points**:
- Implement `predict(image_rgb, points, labels, box_xyxy) -> mask` interface
- `SegmentAnythingAdapter` shows integration pattern for Meta's SAM
- System gracefully degrades to no-op if SAM unavailable

### Environment Variables

```bash
# VLM configuration
export QWEN_VL_SERVER="http://127.0.0.1:8000/generate"  # Server mode
export QWEN_VL_MODEL_DIR="models/Qwen2.5-VL-7B-Instruct-AWQ"  # Local mode

# SAM configuration  
export SAM_CHECKPOINT="models/sam_vit_h_4b8939.pth"
export SAM_TYPE="vit_h"
export SAM_DEVICE="cuda"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false
```

### Important Implementation Details

**Distance Transform Fallback**: Uses OpenCV when available, falls back to erosion-based approximation for robust interior point sampling.

**JSON Parsing Robustness**: VLM responses use `try_parse_json()` which handles imperfect model outputs by extracting the outermost JSON object.

**Spatial NMS Strategy**: Prevents point clustering using radius proportional to ROI diagonal (`0.06 × diag(ROI)`).

**Adaptive Thresholds**: Point classification thresholds adapt to score distributions to handle varying VLM confidence ranges.

**Early Stopping**: Rounds terminate when IoU change < 0.005 to prevent unnecessary computation.

## File Structure Context

- `auxiliary/`: Test data (images, masks, ROI boxes, expected outputs)
- `src/sculptor/`: Core algorithm modules
- `scripts/`: CLI entry points
- `models/`: Place SAM checkpoints and VLM models here (gitignored)
- `outputs/`: Generated results and visualizations

The system is designed to be research-friendly with comprehensive logging (per-round visualizations, JSON metadata, score tracking) while maintaining production-ready modularity and error handling.