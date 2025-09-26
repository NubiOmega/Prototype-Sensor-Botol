"""ROI data structures and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from .db import RoiPreset, session_scope

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(slots=True)
class ROIShape:
    id: Optional[int]
    line: str
    name: str
    kind: Literal["rect", "poly"]
    points: List[Point] = field(default_factory=list)

    @classmethod
    def from_preset(cls, preset: RoiPreset) -> "ROIShape":
        raw = preset.points_json or {}
        points = _parse_points(raw.get("points") if isinstance(raw, dict) else raw)
        return cls(
            id=preset.id,
            line=preset.line,
            name=preset.name,
            kind=preset.kind,
            points=points,
        )

    def as_payload(self) -> dict:
        return {
            "line": self.line,
            "name": self.name,
            "kind": self.kind,
            "points_json": {"points": self.points},
        }


class ROIManager:
    def list_presets(self, line: Optional[str] = None) -> List[ROIShape]:
        with session_scope() as session:
            query = session.query(RoiPreset)
            if line:
                query = query.filter(RoiPreset.line == line)
            presets = query.order_by(RoiPreset.created_at.desc()).all()
            return [ROIShape.from_preset(preset) for preset in presets]

    def save(self, shape: ROIShape) -> ROIShape:
        with session_scope() as session:
            return self._save(session, shape)

    def _save(self, session: Session, shape: ROIShape) -> ROIShape:
        points_payload = {"points": shape.points}
        if shape.id:
            preset = session.get(RoiPreset, shape.id)
            if not preset:
                raise ValueError(f"ROI preset {shape.id} not found")
            preset.line = shape.line
            preset.name = shape.name
            preset.kind = shape.kind
            preset.points_json = points_payload
            session.flush()
            return ROIShape.from_preset(preset)
        preset = RoiPreset(
            line=shape.line,
            name=shape.name,
            kind=shape.kind,
            points_json=points_payload,
        )
        session.add(preset)
        session.flush()
        return ROIShape.from_preset(preset)

    def delete(self, preset_id: int) -> None:
        with session_scope() as session:
            preset = session.get(RoiPreset, preset_id)
            if preset:
                session.delete(preset)
                session.flush()


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def roi_contains(shape: ROIShape, bbox: BBox) -> bool:
    if not shape.points:
        return True
    if shape.kind == "rect":
        return _rect_contains(shape.points, bbox)
    return _poly_contains(shape.points, bbox)


def roi_any_contains(rois: Iterable[ROIShape], bbox: BBox) -> bool:
    shapes = list(rois)
    if not shapes:
        return True
    return any(roi_contains(shape, bbox) for shape in shapes)


def _rect_contains(points: Sequence[Point], bbox: BBox) -> bool:
    if len(points) < 2:
        return True
    (rx1, ry1), (rx2, ry2) = points[0], points[1]
    x1 = min(rx1, rx2)
    x2 = max(rx1, rx2)
    y1 = min(ry1, ry2)
    y2 = max(ry1, ry2)
    cx, cy = bbox_center(bbox)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _poly_contains(points: Sequence[Point], bbox: BBox) -> bool:
    cx, cy = bbox_center(bbox)
    inside = False
    n = len(points)
    if n < 3:
        return True
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        intersects = ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _parse_points(raw: object) -> List[Point]:
    if not raw:
        return []
    points: List[Point] = []
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                try:
                    x = float(entry[0])
                    y = float(entry[1])
                except (TypeError, ValueError):
                    continue
                points.append((x, y))
    return points


__all__ = [
    "BBox",
    "ROIManager",
    "ROIShape",
    "bbox_center",
    "roi_any_contains",
    "roi_contains",
]
