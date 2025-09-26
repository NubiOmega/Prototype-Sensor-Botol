"""Analytics and chart generation for QC reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import Select, select

from .db import Batch, Detection, Inspection, ModelRecord, ReviewTruth, get_engine

plt.switch_backend("Agg")


@dataclass(slots=True)
class AnalyticsFilters:
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    lot_id: Optional[str] = None
    line: Optional[str] = None
    shift: Optional[str] = None
    model_id: Optional[int] = None


@dataclass(slots=True)
class AnalyticsResult:
    filters: AnalyticsFilters
    inspections: pd.DataFrame
    detections: pd.DataFrame
    review_truth: pd.DataFrame
    kpis: Dict[str, object]
    chart_paths: Dict[str, Path] = field(default_factory=dict)
    confusion: Optional[pd.DataFrame] = None


def _apply_common_filters(stmt: Select, filters: AnalyticsFilters) -> Select:
    if filters.start:
        stmt = stmt.where(Inspection.created_at >= filters.start)
    if filters.end:
        stmt = stmt.where(Inspection.created_at <= filters.end)
    if filters.lot_id:
        stmt = stmt.where(Batch.lot_id == filters.lot_id)
    if filters.line:
        stmt = stmt.where(Batch.line == filters.line)
    if filters.shift:
        stmt = stmt.where(Batch.shift == filters.shift)
    if filters.model_id:
        stmt = stmt.where(Inspection.model_id == filters.model_id)
    return stmt


def _inspections_stmt(filters: AnalyticsFilters) -> Select:
    stmt = (
        select(
            Inspection.id.label("inspection_id"),
            Inspection.created_at,
            Inspection.pass_fail.label("status"),
            Inspection.rule_version,
            Inspection.inference_ms,
            Inspection.fps,
            Inspection.frame_path,
            Inspection.batch_id,
            Batch.lot_id,
            Batch.line,
            Batch.shift,
            Batch.operator,
            ModelRecord.id.label("model_id"),
            ModelRecord.name.label("model_name"),
        )
        .select_from(Inspection)
        .join(Batch, Inspection.batch_id == Batch.id, isouter=True)
        .join(ModelRecord, Inspection.model_id == ModelRecord.id, isouter=True)
        .order_by(Inspection.created_at.asc())
    )
    return _apply_common_filters(stmt, filters)


def _detections_stmt(filters: AnalyticsFilters) -> Select:
    stmt = (
        select(
            Detection.id.label("detection_id"),
            Detection.inspection_id,
            Detection.cls,
            Detection.conf,
            Detection.x1,
            Detection.y1,
            Detection.x2,
            Detection.y2,
            Detection.created_at.label("created_at"),
            Inspection.created_at.label("inspection_created_at"),
            Batch.lot_id,
            Batch.line,
            Batch.shift,
        )
        .select_from(Detection)
        .join(Inspection, Detection.inspection_id == Inspection.id)
        .join(Batch, Inspection.batch_id == Batch.id, isouter=True)
        .order_by(Detection.created_at.asc())
    )
    return _apply_common_filters(stmt, filters)


def _review_truth_stmt(filters: AnalyticsFilters) -> Select:
    stmt = (
        select(
            ReviewTruth.id.label("truth_id"),
            ReviewTruth.inspection_id,
            ReviewTruth.cls,
            ReviewTruth.x1,
            ReviewTruth.y1,
            ReviewTruth.x2,
            ReviewTruth.y2,
            ReviewTruth.status,
            ReviewTruth.created_at.label("created_at"),
            Inspection.created_at.label("inspection_created_at"),
        )
        .select_from(ReviewTruth)
        .join(Inspection, ReviewTruth.inspection_id == Inspection.id)
        .join(Batch, Inspection.batch_id == Batch.id, isouter=True)
        .order_by(ReviewTruth.created_at.asc())
    )
    return _apply_common_filters(stmt, filters)


def _fetch_dataframe(stmt: Select) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(stmt, engine)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"] = df["created_at"].dt.date
    return df


def compute_kpis(inspections_df: pd.DataFrame, detections_df: pd.DataFrame) -> Dict[str, object]:
    total = int(len(inspections_df)) if not inspections_df.empty else 0
    pass_count = int((inspections_df["status"] == "PASS").sum()) if total else 0
    fail_count = int((inspections_df["status"] == "FAIL").sum()) if total else 0
    pass_rate = (pass_count / total * 100.0) if total else 0.0
    defect_counts: Dict[str, int] = {}
    if not detections_df.empty:
        defect_counts = (
            detections_df.groupby("cls")["detection_id"].count().sort_values(ascending=False).to_dict()
        )
    top_defects = list(defect_counts.items())[:3]
    return {
        "total_inspections": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "defect_counts": defect_counts,
        "top_defects": top_defects,
    }


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chart_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"{name}.png"


def _plot_bar(data: Dict[str, int], title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    if data:
        labels = list(data.keys())
        values = list(data.values())
        plt.bar(labels, values, color="#1f77b4")
        plt.xticks(rotation=45, ha="right")
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_trend(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    if not df.empty:
        grouped = df.groupby("date")["status"].apply(lambda s: (s == "FAIL").mean() * 100.0)
        grouped.plot(marker="o", color="#d62728")
        plt.ylabel("Fail rate (%)")
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xticks([])
        plt.yticks([])
    plt.title("Defect rate over time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pie(pass_count: int, fail_count: int, path: Path) -> None:
    plt.figure(figsize=(4, 4))
    total = pass_count + fail_count
    if total:
        plt.pie(
            [pass_count, fail_count],
            labels=["Pass", "Fail"],
            colors=["#2ca02c", "#d62728"],
            autopct="%1.1f%%",
        )
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xticks([])
        plt.yticks([])
    plt.title("Pass vs Fail")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_charts(
    inspections_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    output_dir: Path,
    kpis: Dict[str, object],
) -> Dict[str, Path]:
    charts_dir = _ensure_dir(output_dir)
    defect_path = _chart_path(charts_dir, "defects_by_class")
    trend_path = _chart_path(charts_dir, "defect_trend")
    pie_path = _chart_path(charts_dir, "pass_fail")

    _plot_bar(kpis.get("defect_counts", {}) or {}, "Defects by class", defect_path)
    _plot_trend(inspections_df, trend_path)
    _plot_pie(kpis.get("pass_count", 0), kpis.get("fail_count", 0), pie_path)

    return {
        "defects_by_class": defect_path,
        "defect_trend": trend_path,
        "pass_fail": pie_path,
    }


def compute_confusion(detections_df: pd.DataFrame, truth_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if detections_df.empty or truth_df.empty:
        return None
    preds = detections_df[["inspection_id", "cls"]].rename(columns={"cls": "pred"})
    truth = truth_df[["inspection_id", "cls", "status"]].rename(columns={"cls": "truth"})
    truth = truth[truth["status"] != "deleted"]
    merged = preds.merge(truth, on="inspection_id", how="outer")
    merged["pred"].fillna("__none__", inplace=True)
    merged["truth"].fillna("__none__", inplace=True)
    confusion = pd.crosstab(merged["truth"], merged["pred"], dropna=False)
    return confusion


def run_analytics(filters: AnalyticsFilters, output_dir: Path) -> AnalyticsResult:
    inspections_df = _fetch_dataframe(_inspections_stmt(filters))
    detections_df = _fetch_dataframe(_detections_stmt(filters))
    review_truth_df = _fetch_dataframe(_review_truth_stmt(filters))
    kpis = compute_kpis(inspections_df, detections_df)
    chart_paths = build_charts(inspections_df, detections_df, output_dir, kpis)
    confusion = compute_confusion(detections_df, review_truth_df)
    return AnalyticsResult(
        filters=filters,
        inspections=inspections_df,
        detections=detections_df,
        review_truth=review_truth_df,
        kpis=kpis,
        chart_paths=chart_paths,
        confusion=confusion,
    )


__all__ = [
    "AnalyticsFilters",
    "AnalyticsResult",
    "build_charts",
    "compute_confusion",
    "compute_kpis",
    "run_analytics",
]
