Great insight. I’ve implemented a boundary-refinement mode that keeps iteration 0 as your strong baseline (initial SAM mask), and from iteration 1 onward only refines along the current mask’s boundary with local positive/negative pairs constrained strictly to the ROI-expanded region. This avoids global “whole-ROI re-sculpting” that was destabilizing iter01.

What changed
•  Strict ROI gating everywhere
•  All points and masks are hard-clamped to an allowed region: initial ROI expanded by a small scale (default 1.2x).
•  Any SAM-updated mask is also clamped back to this region each iteration.
•  Bootstrap with SAM, then boundary-only refinement
•  Always compute M0 via SAM box. If prior is provided, intersect: M = M0_sam & prior.
•  Iteration 0 simply records this baseline; no update (skip_update_first_iter=True).
•  Iterations >= 1 use boundary_refine_only mode: sample along current mask boundary and place:
◦  Positive points slightly inside the boundary (nearest inside within a small band).
◦  Negative points slightly outside the boundary (nearest outside within a configurable band).
•  Dynamic fine grid is still available (but bypassed when boundary_refine_only=True). If you later want mixed modes, we can toggle it.

Relevant new config fields in cog_sculpt/core.py Config
•  boundary_refine_only: True
•  skip_update_first_iter: True
•  boundary_inner_band: 6 (pixels inward for positive point search)
•  boundary_band: 10 (already existed; used for outward negative search)
•  roi_point_expand_scale: 1.2
•  gate_mask_outside_roi: True

What is saved per-iteration
•  iter_00_mask.png and iter_00_overlay.png: initial M0 (baseline)
•  iter_01_mask.png / overlay, iter_02_mask.png / overlay, etc.
•  iter_XX_debug.json contains the points used each iteration, so you can verify they’re on the boundary in the allowed region.

How to run
Use your same command; it now produces per-iteration masks with the boundary refinement behavior:
python -m cog_sculpt.cli \
  --name q \
  --image auxiliary/images/q.png \
  --boxes auxiliary/box_out/q/q_sam_boxes.json \
  --prior-mask auxiliary/box_out/q/q_prior_mask.png \
  --out-root pipeline_output \
  --use-prior --use-boxes --debug

Tuning suggestions
•  If refinement is still too aggressive:
•  Reduce boundary_band (e.g., 6–8) so negative points are placed closer to the boundary.
•  Reduce max points per iter: --max-points 6.
•  If refinement is too conservative:
•  Increase boundary_inner_band to allow positive points a bit deeper inside (e.g., 8–10).
•  Slightly increase max points per iter.

Would you like me to expose boundary_inner_band and roi_point_expand_scale on the CLI (e.g., --boundary-inner 6, --roi-expand 1.15) so you can tune quickly without changing code?