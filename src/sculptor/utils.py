from __future__ import annotations

import json
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 3:
        return img
    if img.shape[2] == 4:
        return img[..., :3]
    raise ValueError("Unsupported channel count")


def overlay_mask_on_image(img_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.3) -> np.ndarray:
    img = to_rgb(ensure_uint8(img_rgb))
    m = (mask > 0).astype(np.uint8)
    overlay = img.copy()
    overlay[m.astype(bool)] = (
        (1 - alpha) * overlay[m.astype(bool)] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    from PIL import Image

    m = Image.open(path)
    if m.mode != "L":
        m = m.convert("L")
    return (np.array(m) > 127).astype(np.uint8) * 255


def save_image(path: str, img: np.ndarray) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(ensure_uint8(to_rgb(img))).save(path)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_sam_boxes_json(path: str) -> List[Tuple[float, float, float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    # Expect list of boxes or dict with key 'boxes'
    if isinstance(data, dict) and "boxes" in data:
        items = data["boxes"]
    elif isinstance(data, list):
        items = data
    else:
        items = []
    for it in items:
        if isinstance(it, dict) and all(k in it for k in ("x0", "y0", "x1", "y1")):
            boxes.append((float(it["x0"]), float(it["y0"]), float(it["x1"]), float(it["y1"])) )
        elif isinstance(it, (list, tuple)) and len(it) == 4:
            boxes.append(tuple(map(float, it)))
    return boxes


def draw_points(img: np.ndarray, pts: Sequence[Tuple[float, float]], colors: Sequence[Tuple[int, int, int]], r: int = 3) -> np.ndarray:
    try:
        import cv2
    except Exception:
        cv2 = None

    vis = to_rgb(ensure_uint8(img)).copy()
    for (x, y), c in zip(pts, colors):
        xi, yi = int(round(x)), int(round(y))
        if cv2 is not None:
            cv2.circle(vis, (xi, yi), r, c, -1)
        else:
            y0, y1 = max(yi - r, 0), min(yi + r + 1, vis.shape[0])
            x0, x1 = max(xi - r, 0), min(xi + r + 1, vis.shape[1])
            vis[y0:y1, x0:x1] = c
    return vis


def nms_points(points: List[Tuple[float, float]], scores: List[float], r: float) -> List[int]:
    """Simple greedy NMS over 2D points. Returns kept indices."""
    if not points:
        return []
    pts = np.array(points, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    order = np.argsort(-s)
    keep = []
    taken = np.zeros(len(points), dtype=bool)
    r2 = r * r
    for i in order:
        if taken[i]:
            continue
        keep.append(int(i))
        di = pts - pts[i]
        dist2 = (di[:, 0] ** 2 + di[:, 1] ** 2)
        suppress = dist2 <= r2
        taken[suppress] = True
        taken[i] = True
    return keep


def uniform_grid_in_box(x0: float, y0: float, x1: float, y1: float, step: float) -> List[Tuple[float, float]]:
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    return pts


