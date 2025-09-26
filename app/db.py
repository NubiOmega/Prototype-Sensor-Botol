"""SQLite persistence layer for the QC suite."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "qc.db"
_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=text("CURRENT_TIMESTAMP"))


class ModelRecord(TimestampMixin, Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    size: Mapped[int] = mapped_column(Integer, default=0)
    hash: Mapped[str] = mapped_column(String(64), default="")
    type: Mapped[str] = mapped_column(String(16), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    metrics_json: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    inspections: Mapped[List["Inspection"]] = relationship(back_populates="model")


class Batch(TimestampMixin, Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    line: Mapped[str] = mapped_column(String(64), nullable=False)
    shift: Mapped[str] = mapped_column(String(32), nullable=False)
    operator: Mapped[str] = mapped_column(String(128), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    inspections: Mapped[List["Inspection"]] = relationship(back_populates="batch", cascade="all, delete-orphan")


class Inspection(TimestampMixin, Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[Optional[int]] = mapped_column(ForeignKey("batches.id"))
    frame_path: Mapped[str] = mapped_column(String(512), default="")
    pass_fail: Mapped[str] = mapped_column(String(8), default="PASS")
    rule_version: Mapped[str] = mapped_column(String(64), default="")
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    model_id: Mapped[Optional[int]] = mapped_column(ForeignKey("models.id"))
    notes: Mapped[str] = mapped_column(Text, default="")

    batch: Mapped[Optional[Batch]] = relationship(back_populates="inspections")
    model: Mapped[Optional[ModelRecord]] = relationship(back_populates="inspections")
    detections: Mapped[List["Detection"]] = relationship(back_populates="inspection", cascade="all, delete-orphan")
    review_truth: Mapped[List["ReviewTruth"]] = relationship(back_populates="inspection", cascade="all, delete-orphan")


class Detection(TimestampMixin, Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey("inspections.id"), nullable=False)
    cls: Mapped[str] = mapped_column(String(64), nullable=False)
    conf: Mapped[float] = mapped_column(Float, default=0.0)
    x1: Mapped[float] = mapped_column(Float, default=0.0)
    y1: Mapped[float] = mapped_column(Float, default=0.0)
    x2: Mapped[float] = mapped_column(Float, default=0.0)
    y2: Mapped[float] = mapped_column(Float, default=0.0)

    inspection: Mapped[Inspection] = relationship(back_populates="detections")


class RoiPreset(TimestampMixin, Base):
    __tablename__ = "roi_presets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    points_json: Mapped[Dict[str, Sequence[float]]] = mapped_column(JSON, default=dict)


class QCRule(TimestampMixin, Base):
    __tablename__ = "qc_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    json: Mapped[Dict[str, object]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=False)


class ReviewTruth(TimestampMixin, Base):
    __tablename__ = "review_truth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey("inspections.id"), nullable=False)
    cls: Mapped[str] = mapped_column(String(64), nullable=False)
    x1: Mapped[float] = mapped_column(Float, default=0.0)
    y1: Mapped[float] = mapped_column(Float, default=0.0)
    x2: Mapped[float] = mapped_column(Float, default=0.0)
    y2: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="added")

    inspection: Mapped[Inspection] = relationship(back_populates="review_truth")


class Operator(TimestampMixin, Base):
    __tablename__ = "operators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)


class Device(TimestampMixin, Base):
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    url_or_index: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), default="camera")


@dataclass(slots=True)
class DetectionPayload:
    cls: str
    conf: float
    bbox: Tuple[float, float, float, float]


def init_db(db_path: Optional[Path | str] = None) -> Path:
    """Initialise the SQLite database and keep a global engine/session factory."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{path.as_posix()}"
    engine = create_engine(
        uri,
        future=True,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    global _ENGINE, _SESSION_FACTORY
    _ENGINE = engine
    _SESSION_FACTORY = sessionmaker(engine, expire_on_commit=False, class_=Session)
    with session_scope() as session:
        _ensure_single_active_rule(session)
    return path


def get_engine() -> Engine:
    if _ENGINE is None:
        init_db()
    assert _ENGINE is not None
    return _ENGINE


def get_session() -> Session:
    if _SESSION_FACTORY is None:
        init_db()
    assert _SESSION_FACTORY is not None
    return _SESSION_FACTORY()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def register_model(
    session: Session,
    *,
    name: str,
    path: str,
    size: int,
    file_hash: str,
    model_type: str = "",
    notes: str = "",
    metrics: Optional[Dict[str, float]] = None,
    set_active: bool = False,
) -> ModelRecord:
    metrics = metrics or {}
    record = ModelRecord(
        name=name,
        path=path,
        size=size,
        hash=file_hash,
        type=model_type,
        notes=notes,
        metrics_json=metrics,
    )
    if set_active:
        _clear_active_model(session)
        record.is_active = True
    session.add(record)
    session.flush()
    return record


def list_models(session: Session) -> List[ModelRecord]:
    stmt = select(ModelRecord).order_by(ModelRecord.created_at.desc())
    return list(session.scalars(stmt))


def _clear_active_model(session: Session) -> None:
    for model in session.scalars(select(ModelRecord).where(ModelRecord.is_active.is_(True))):
        model.is_active = False


def set_active_model(session: Session, model_id: int) -> Optional[ModelRecord]:
    _clear_active_model(session)
    model = session.get(ModelRecord, model_id)
    if model:
        model.is_active = True
    return model


def get_active_model(session: Session) -> Optional[ModelRecord]:
    stmt = select(ModelRecord).where(ModelRecord.is_active.is_(True)).limit(1)
    return session.scalars(stmt).first()


def upsert_batch(
    session: Session,
    *,
    lot_id: str,
    line: str,
    shift: str,
    operator: str,
    notes: str = "",
    reopen: bool = False,
) -> Batch:
    stmt = (
        select(Batch)
        .where(Batch.lot_id == lot_id)
        .where(Batch.line == line)
        .where(Batch.shift == shift)
        .where(Batch.operator == operator)
        .order_by(Batch.created_at.desc())
    )
    batch = session.scalars(stmt).first()
    if batch and batch.end_time and reopen:
        batch.end_time = None
        batch.notes = notes
        session.flush()
        return batch
    if batch and not batch.end_time:
        return batch
    batch = Batch(
        lot_id=lot_id,
        line=line,
        shift=shift,
        operator=operator,
        notes=notes,
    )
    session.add(batch)
    session.flush()
    return batch


def close_batch(session: Session, batch_id: int) -> None:
    batch = session.get(Batch, batch_id)
    if not batch:
        return
    batch.end_time = datetime.utcnow()
    session.flush()


def record_inspection(
    session: Session,
    *,
    batch_id: Optional[int],
    frame_path: str,
    pass_fail: str,
    rule_version: str,
    inference_ms: float,
    fps: float,
    model_id: Optional[int],
    detections: Sequence[DetectionPayload],
    notes: str = "",
) -> Inspection:
    inspection = Inspection(
        batch_id=batch_id,
        frame_path=frame_path,
        pass_fail=pass_fail,
        rule_version=rule_version,
        inference_ms=inference_ms,
        fps=fps,
        model_id=model_id,
        notes=notes,
    )
    session.add(inspection)
    session.flush()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        session.add(
            Detection(
                inspection_id=inspection.id,
                cls=det.cls,
                conf=det.conf,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    session.flush()
    return inspection


def recent_detections(session: Session, batch_id: Optional[int], limit: int = 50) -> List[Detection]:
    stmt = (
        select(Detection)
        .join(Inspection)
        .where(Inspection.batch_id == batch_id)
        .order_by(Detection.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def batch_counters(session: Session, batch_id: int) -> Dict[str, int]:
    stmt = (
        select(Detection.cls, func.count(Detection.id))
        .join(Inspection)
        .where(Inspection.batch_id == batch_id)
        .group_by(Detection.cls)
    )
    return {row[0]: row[1] for row in session.execute(stmt)}


def _ensure_single_active_rule(session: Session) -> None:
    stmt = select(QCRule).limit(1)
    existing = session.scalars(stmt).first()
    if existing:
        active_stmt = select(QCRule).where(QCRule.active.is_(True)).limit(1)
        if not session.scalars(active_stmt).first():
            existing.active = True
        return
    default_rule = QCRule(
        name="Default",
        json={
            "reject_if": [
                "crack_hairline",
                "crack_major",
                "chip_mouth",
                "chip_body",
                "no_cap",
                "label_torn",
                "other_defect",
            ],
            "min_conf": 0.35,
        },
        active=True,
    )
    session.add(default_rule)
    session.flush()


__all__ = [
    "Batch",
    "Detection",
    "DetectionPayload",
    "Device",
    "Inspection",
    "ModelRecord",
    "Operator",
    "QCRule",
    "ReviewTruth",
    "RoiPreset",
    "batch_counters",
    "close_batch",
    "get_active_model",
    "get_engine",
    "get_session",
    "init_db",
    "list_models",
    "recent_detections",
    "record_inspection",
    "register_model",
    "session_scope",
    "set_active_model",
    "upsert_batch",
]
