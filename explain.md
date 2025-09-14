# Codebase Overview and Module Relationships

This document summarizes the repository’s structure, each file’s purpose, and how modules connect to implement the VLM‑driven segmentation sculpting workflow described in `followme.md`.

## High‑Level Flow

Per sculpting round (2–3 typical):
1) Candidate points are sampled inside the current mask, in a boundary band, and outside within the ROI.
2) Multi‑scale patches around each candidate point are extracted with light augmentations.
3) VLM runs Peval once on the ROI overlay to diagnose mask defects and provide key cues, and Pgen on each patch to score “is target” with confidence.
4) Points are selected into P⁺/P⁻ via spatial NMS and adaptive thresholds.
5) SAM is called (hook) to refine the mask using P⁺/P⁻ and the ROI box. Early‑stop checks convergence.

Entry point: `scripts/run_sculpt.py`.

---

## Directory Map (Key Files)

- `followme.md` — Problem statement and algorithm blueprint for sculpting.
- `README.md` — Quick usage and integration notes.
- `models/README.md` — How to wire a local Qwen2.5‑VL.
- `scripts/run_sculpt.py` — CLI runner orchestrating a full sculpting round.
- `src/sculptor/` — Core sculpting modules:
  - `__init__.py` — Package marker.
  - `types.py` — Lightweight types: `ROIBox`, `Candidate`.
  - `utils.py` — IO, drawing, overlay, JSON helpers, NMS and grid utilities.
  - `candidates.py` — Candidate sampling inside/band/outside with distance transform and NMS.
  - `patches.py` — Multi‑scale patch extraction around points, light augmentations.
  - `select_points.py` — Adaptive thresholds, spatial NMS, compose P⁺/P⁻ and SAM inputs.
  - `sam_refine.py` — SAM predictor wrapper (injectable) + early‑stop utility.
  - `vlm/` — VLM abstractions and stubs:
    - `base.py` — `VLMBase` interface; robust JSON parsing helper.
    - `prompts.py` — Peval/Pgen prompts (forced JSON), aligned with `followme.md`.
    - `qwen.py` — Qwen2.5‑VL stub (server/local modes; no network by default).
    - `mock.py` — Deterministic mock VLM for development without a real model.
- `auxiliary/` — Sample images, masks, boxes, and docs for quick runs (not code).

---

## Module‑by‑Module Details

- `src/sculptor/types.py:1`
  - `ROIBox` stores an ROI rectangle and provides width/height/diagonal/clip utilities.
  - `Candidate` describes a sampled point (`xy`, `idx`, `kind`, `score`).

- `src/sculptor/utils.py:1`
  - Image helpers: `ensure_uint8`, `to_rgb`, `load_image`, `save_image`.
  - Mask helpers: `load_mask`, `overlay_mask_on_image` (for Peval overlay).
  - Geometry: `nms_points` (greedy point NMS), `uniform_grid_in_box`.
  - IO: `save_json`, `parse_sam_boxes_json`.

- `src/sculptor/candidates.py:1`
  - `_distance_transform` (OpenCV if available; degraded fallback otherwise).
  - `_sample_inside` uses distance peaks within the mask for stable interior positives.
  - `_sample_band` builds a dilate–erode ring (“band”) for potential fix points (pos/neg determined by VLM).
  - `_sample_outside` samples B\M with a simple variance heuristic for harder negatives.
  - `sample_candidates` merges routes, applies spatial NMS (radius = 0.06×diag(B)), returns up to ~30 candidates.

- `src/sculptor/patches.py:1`
  - `_crop_square` centers a square crop on a point, clamps image edges.
  - `_augment_light` adds horizontal flip and small brightness/contrast tweaks.
  - `multi_scale_patches` extracts sizes proportional to min(wB,hB) for robustness.
  - `multi_scale_with_aug` wraps multi‑scale with optional light augmentations.

- `src/sculptor/vlm/prompts.py:1`
  - `build_peval_prompt(instance, word_budget)` — compact JSON diagnosis (missing/over/boundary/key_cues).
  - `build_pgen_prompt(instance, key_cues)` — patch‑level yes/no with confidence, JSON only.

- `src/sculptor/vlm/base.py:1`
  - `VLMBase` defines `peval(image_overlay_rgb, depth_vis, instance)` and `pgen(patch_rgb, instance, key_cues)`.
  - `try_parse_json` handles imperfect VLM outputs by trimming to the outer JSON object.

- `src/sculptor/vlm/qwen.py:1`
  - `QwenVLM(mode, server_url, model_dir)` implements the `VLMBase` interface.
  - `peval`/`pgen` build prompts then call `_inference`.
  - `_infer_via_server` and `_infer_via_local` are stubs (safe no‑op returning `{}`) — you implement either a local HTTP server or in‑process loading.
  - Environment hooks: `QWEN_VL_SERVER`, `QWEN_VL_MODEL_DIR`.

- `src/sculptor/vlm/mock.py:1`
  - `MockVLM` returns stable, deterministic outputs to validate the pipeline without a real model.

- `src/sculptor/select_points.py:1`
  - `estimate_neg_thresholds` computes adaptive thresholds (μ+σ, clamped) with safe fallbacks.
  - `spatial_nms_sorted` applies NMS then sorts by score.
  - `select_points` partitions into P⁺/P⁻, returns `points` (Nx2) and `labels` (N,) for SAM.

- `src/sculptor/sam_refine.py:1`
  - `SamPredictorWrapper.predict` calls an injected SAM backend; current default is a safe no‑op returning the previous mask.
  - `early_stop` checks IoU deltas to decide convergence.
  - 配合 `sam_backends.py` 可加载真实 SAM 预测器。

- `src/sculptor/sam_backends.py:1`
  - `SegmentAnythingAdapter`：对 `segment_anything.SamPredictor` 的适配层，封装了 `set_image`、坐标变换与掩码选择。
  - `build_sam_backend(checkpoint_path, model_type, device)`：从检查点加载 SAM 并返回适配器。

- `scripts/run_sculpt.py:1`
  - CLI orchestrator. Steps: candidates → patches → Peval → Pgen → selection → SAM refine → early‑stop.
  - Options: `--image`, `--mask`, `--roi_json` or `--roi`, `--instance`, `--model {mock,qwen}`, `--server_url`, `--model_dir`, `--rounds`, `--K_pos`, `--K_neg`, `--out_dir`；SAM相关：`--sam_checkpoint`, `--sam_type {vit_h|vit_l|vit_b}`, `--sam_device {cuda|cpu}`。
  - Logging: saves per‑round points visualization, Peval JSON, point scores, masks, and final summary.

---

## How Modules Connect (Data Flow)

- Input: RGB image `I`, initial mask `M0`, ROI `B`.
- `candidates.sample_candidates(Mt, B)` → `List[Candidate]` with absolute coordinates.
- `patches.multi_scale_with_aug(I, [c.xy], B, scales)` → dict of patches per candidate.
- `vlm.QwenVLM/MockVLM.peval(overlay, None, instance)` → semantic cues and compact diagnosis.
- `vlm.*.pgen(patch, instance, key_cues)` per patch → per‑candidate fused score in [0,1].
- `select_points.select_points(candidates, scores, B)` → `points` (N×2) & `labels` (N,) for SAM.
- `sam_refine.SamPredictorWrapper.predict(I, points, labels, B, prev_mask)` → refined `Mt+1`.
- `sam_refine.early_stop(Mt, Mt+1)` → decide whether to stop.

---

## Extension Points and Configuration

- VLM backend:
  - Use `MockVLM` for testing.
  - Switch to `QwenVLM` and implement either:
    - a local HTTP server (set `--server_url` / `QWEN_VL_SERVER`), or
    - in‑process loader in `QwenVLM._infer_via_local` (set `--model_dir` / `QWEN_VL_MODEL_DIR`).

- SAM backend:
  - Inject your predictor into `SamPredictorWrapper(backend=...)` and implement its `predict` call as needed.

- Heuristics:
  - Sampling sizes, scales (`{0.08, 0.12, 0.18}`), NMS radius (`0.06×diag(B)`), and thresholds are centralized in their respective modules and easily tunable.

---

## Outputs

- Round artifacts (per `--out_dir/<image_basename>/`):
  - `roundX_points.png` — green=positive, red=negative point visualization.
  - `roundX_peval.json` — VLM mask‑level diagnosis.
  - `roundX_scores.json` — candidate fused scores.
  - `roundX_mask.png` — refined mask (or unchanged if SAM not injected).
  - `final_mask.png`, `meta.json` — final outputs and small log.

---

## Quick Usage

Mock VLM:

```
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model mock \
  --rounds 2 \
  --out_dir outputs/sculpt/dog
```

Qwen stub (server mode; implement your local endpoint first):

```
python scripts/run_sculpt.py \
  --image auxiliary/images/dog.png \
  --mask auxiliary/box_out/dog/dog_prior_mask.png \
  --roi_json auxiliary/box_out/dog/dog_sam_boxes.json \
  --instance "dog" \
  --model qwen \
  --server_url http://127.0.0.1:8000/generate \
  --rounds 2
```
