from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SegmentAnythingAdapter:
    """Adapter over Meta's segment_anything.SamPredictor.

    This class hides set_image, coordinate transforms, mask selection, and returns a uint8 mask 0/255.
    """

    predictor: object
    device: Optional[str] = None

    _last_image_shape: Optional[Tuple[int, int]] = None

    def _ensure_image(self, image_rgb: np.ndarray) -> None:
        H, W = image_rgb.shape[:2]
        if self._last_image_shape != (H, W):
            # SamPredictor handles device internally via model placement
            self.predictor.set_image(image_rgb)
            self._last_image_shape = (H, W)

    def predict(self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray, box_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
        self._ensure_image(image_rgb)
        # Transform coordinates to SAM's input space
        pt_coords = points.astype(np.float32)
        if pt_coords.ndim == 1:
            pt_coords = pt_coords[None, :]
        pt_coords = self.predictor.transform.apply_coords(pt_coords, image_rgb.shape[:2])
        box = np.array([box_xyxy], dtype=np.float32)
        if hasattr(self.predictor.transform, "apply_boxes"):
            box_t = self.predictor.transform.apply_boxes(box, image_rgb.shape[:2])[0]
        else:
            # Some versions may not expose apply_boxes; pass raw and rely on predictor
            box_t = box[0]

        masks, scores, logits = self.predictor.predict(
            point_coords=pt_coords if len(pt_coords) > 0 else None,
            point_labels=labels.astype(np.int32) if len(labels) > 0 else None,
            box=box_t,
            multimask_output=True,
            return_logits=False,
        )
        # Select best by highest score
        if isinstance(scores, np.ndarray) and scores.size > 0:
            k = int(np.argmax(scores))
            m = masks[k]
        else:
            # Fallback to first
            m = masks[0]
        return (m.astype(np.uint8) > 0).astype(np.uint8) * 255


def build_sam_backend(checkpoint_path: str, model_type: str = "vit_h", device: Optional[str] = None) -> SegmentAnythingAdapter:
    """Load a SAM predictor from a checkpoint using segment_anything.

    Args:
      checkpoint_path: Path to SAM weights (.pth)
      model_type: One of {'vit_h','vit_l','vit_b'} matching the checkpoint
      device: Optional device string ('cuda' or 'cpu'); if None, inferred by torch
    """
    try:
        import torch  # type: ignore
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore
    except Exception as e:
        raise ImportError(
            "segment_anything and torch are required for SAM integration. "
            "Please install them and ensure the checkpoint path is valid."
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type not in {"vit_h", "vit_l", "vit_b"}:
        raise ValueError(f"Unsupported model_type '{model_type}'. Use one of vit_h|vit_l|vit_b.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    return SegmentAnythingAdapter(predictor=predictor, device=device)

