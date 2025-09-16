# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a VLM-driven SAM segmentation sculpting system that combines Segment Anything Model (SAM) with Vision-Language Models for iterative mask refinement. The system uses anchor points and quadrant-based analysis to progressively improve image segmentation results.

## Environment Setup

```bash
# Activate the conda environment
conda activate camo-vlm

# Set environment variables for stability
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Running the System

### Main Script
```bash
# Basic run with default settings
python refactor_sam_sculpt.py

# Full command with all parameters
python refactor_sam_sculpt.py --name f --qwen_dir /home/albert/code/CV/models/Qwen2.5-VL-3B-Instruct --ratio 1.0 --anchors_per_round 1 --vlm_max_side 720 --rounds 5
```

### Quick Testing Commands
```bash
# Test Qwen VLM integration
python test_qwen.py

# Verify model setup
python setup_qwen_model.py
```

## Code Architecture

### Core Components

1. **Main Pipeline** (`refactor_sam_sculpt.py`): Orchestrates the entire segmentation sculpting process
2. **VLM Integration** (`src/sculptor/vlm/`): Vision-Language Model backends
   - `base.py`: Abstract VLM interface
   - `qwen.py`: Qwen2.5-VL implementation with local inference
   - `prompts.py`: Prompt engineering for anchor/quadrant analysis
3. **SAM Integration** (`src/sculptor/sam_*.py`): Segment Anything Model integration
   - `sam_backends.py`: SAM predictor adapter
   - `sam_refine.py`: Iterative mask refinement engine
4. **Utilities** (`src/sculptor/utils.py`, `src/sculptor/types.py`): Support functions and type definitions

### Data Flow

1. **Image Input**: Load image and initial ROI boxes from `auxiliary/` directory
2. **Anchor Selection**: VLM analyzes image and selects anchor points for refinement
3. **Quadrant Analysis**: For each anchor, VLM analyzes quadrant crops and provides edit recommendations
4. **SAM Refinement**: Apply positive/negative points to SAM for mask refinement
5. **Visualization**: Generate outputs in `outputs/refactor_sculpt/[sample_name]/`

### VLM Integration Architecture

The system supports multiple VLM backends through the `VLMBase` interface:
- **Local Inference**: Qwen2.5-VL with AWQ quantization (implemented)
- **Server Mode**: HTTP endpoint integration (stub)
- **Mock Mode**: Testing fallback

## Configuration

### Sample Selection
Edit line 259 in `refactor_sam_sculpt.py`:
```python
sample_name = "f"  # Change to "dog" or "q"
```

### Quadrant Box Size
Edit line 155 in `refactor_sam_sculpt.py` (in `create_quadrant_visualization` function):
```python
ratio: float = 0.35  # Increase to 0.5 or 0.6 for larger quadrant boxes
```

## Model Dependencies

### Required Models
- SAM checkpoint: `models/sam_vit_b_01ec64.pth` (or other SAM variants)
- Qwen2.5-VL model: `models/Qwen2.5-VL-7B-Instruct-AWQ/` or `models/Qwen2.5-VL-3B-Instruct/`

### Hardware Requirements
- GPU with 7-8GB memory for Qwen2.5-VL-7B-Instruct-AWQ
- CUDA support recommended for performance
- CPU fallback available but significantly slower

## Output Structure

Results are saved in `outputs/refactor_sculpt/[sample_name]/`:
- Step-by-step visualization images
- Final refined masks
- Intermediate processing results

## Key Technical Notes

- The system uses iterative refinement with early stopping based on IoU convergence
- VLM prompts are engineered for anchor selection and quadrant-based mask editing
- Error handling includes graceful fallbacks when models fail to load
- Memory management includes environment tuning for CUDA allocation