from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ROIBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def diag(self) -> float:
        return (self.w**2 + self.h**2) ** 0.5

    def clip_to(self, W: int, H: int) -> "ROIBox":
        x0 = max(0.0, min(float(W - 1), self.x0))
        y0 = max(0.0, min(float(H - 1), self.y0))
        x1 = max(0.0, min(float(W), self.x1))
        y1 = max(0.0, min(float(H), self.y1))
        return ROIBox(x0, y0, x1, y1)


@dataclass
class Candidate:
    xy: Tuple[float, float]  # absolute image coordinates (x,y)
    idx: int                 # unique index within batch
    kind: str                # "inside" | "band" | "outside"
    score: float = 0.0       # optional pre-score for NMS ordering


PointArray = List[Tuple[float, float]]

