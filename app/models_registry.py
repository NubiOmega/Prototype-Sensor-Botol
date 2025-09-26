"""Model registry helpers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from .db import ModelRecord, get_active_model, list_models, register_model, session_scope, set_active_model


@dataclass(slots=True)
class ModelInfo:
    id: int
    name: str
    path: Path
    size: int
    hash: str
    type: str
    notes: str
    metrics: Dict[str, float]
    is_active: bool

    @classmethod
    def from_record(cls, record: ModelRecord) -> "ModelInfo":
        return cls(
            id=record.id,
            name=record.name,
            path=Path(record.path),
            size=record.size,
            hash=record.hash,
            type=record.type,
            notes=record.notes or "",
            metrics=record.metrics_json or {},
            is_active=bool(record.is_active),
        )


def compute_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_model_type(name: str) -> str:
    lowered = name.lower()
    for tag in ("nano", "n"), ("small", "s"), ("medium", "m"), ("large", "l"), ("xlarge", "x"):
        if tag[0] in lowered or lowered.endswith(f"{tag[1]}.pt"):
            return tag[1]
    return ""


def register_model_path(
    model_path: Path,
    *,
    name: Optional[str] = None,
    notes: str = "",
    metrics: Optional[Dict[str, float]] = None,
    set_active: bool = False,
    session: Optional[Session] = None,
) -> ModelInfo:
    path = model_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    display_name = name or path.stem
    model_type = infer_model_type(path.name)
    file_hash = compute_hash(path)
    size = path.stat().st_size
    metrics = metrics or {}

    def _save(session: Session) -> ModelInfo:
        existing = session.query(ModelRecord).filter(ModelRecord.path == str(path)).one_or_none()
        if existing:
            existing.name = display_name
            existing.size = size
            existing.hash = file_hash
            existing.type = model_type
            existing.notes = notes
            existing.metrics_json = metrics
            session.flush()
            if set_active:
                set_active_model(session, existing.id)
            return ModelInfo.from_record(existing)
        record = register_model(
            session,
            name=display_name,
            path=str(path),
            size=size,
            file_hash=file_hash,
            model_type=model_type,
            notes=notes,
            metrics=metrics,
            set_active=set_active,
        )
        session.flush()
        return ModelInfo.from_record(record)

    if session is not None:
        return _save(session)
    with session_scope() as scoped:
        return _save(scoped)


def list_model_info() -> List[ModelInfo]:
    with session_scope() as session:
        return [ModelInfo.from_record(record) for record in list_models(session)]


def active_model_info() -> Optional[ModelInfo]:
    with session_scope() as session:
        record = get_active_model(session)
        return ModelInfo.from_record(record) if record else None


def set_active(model_id: int) -> Optional[ModelInfo]:
    with session_scope() as session:
        record = set_active_model(session, model_id)
        if not record:
            return None
        session.flush()
        return ModelInfo.from_record(record)


def prune_missing_models() -> List[int]:
    removed: List[int] = []
    with session_scope() as session:
        for record in list_models(session):
            if not Path(record.path).exists():
                removed.append(record.id)
                session.delete(record)
        if removed:
            session.flush()
    return removed


__all__ = [
    "ModelInfo",
    "active_model_info",
    "compute_hash",
    "infer_model_type",
    "list_model_info",
    "prune_missing_models",
    "register_model_path",
    "set_active",
]
