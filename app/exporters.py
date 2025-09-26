"""Export helpers for analytics and datasets."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .analytics import AnalyticsResult
from .reports import ReportMetadata, build_pdf_report


SUMMARY_LABELS = {
    "total_inspections": "Total Inspections",
    "pass_count": "Pass Count",
    "fail_count": "Fail Count",
    "pass_rate": "Pass Rate (%)",
    "defect_counts": "Defect Counts",
    "top_defects": "Top Defects",
}


@dataclass(slots=True)
class ExportPaths:
    detail_csv: Optional[Path] = None
    summary_csv: Optional[Path] = None
    confusion_csv: Optional[Path] = None
    pdf_report: Optional[Path] = None


def export_csv(analytics: AnalyticsResult, output_dir: Path) -> ExportPaths:
    """Persist inspection detail, summary metrics, and confusion matrix to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "inspections.csv"
    summary_path = output_dir / "summary.csv"
    confusion_path = output_dir / "confusion.csv"

    analytics.inspections.to_csv(detail_path, index=False)

    summary_rows = [["Metric", "Value"]]
    for key, label in _ordered_summary_items(analytics.kpis):
        value = analytics.kpis.get(key)
        summary_rows.append([label, _format_summary_value(key, value)])
    _write_rows(summary_path, summary_rows)

    if analytics.confusion is not None and not analytics.confusion.empty:
        analytics.confusion.to_csv(confusion_path)
    else:
        confusion_path = None

    return ExportPaths(
        detail_csv=detail_path,
        summary_csv=summary_path,
        confusion_csv=confusion_path,
    )


def export_pdf(
    analytics: AnalyticsResult,
    metadata: ReportMetadata,
    output_path: Path,
    notes: str = "",
    chart_order: Optional[Sequence[str]] = None,
    defect_images: Optional[Sequence[Path]] = None,
) -> Path:
    """Build a PDF QC report embedding charts and metrics."""
    resolved_defect_images = list(defect_images) if defect_images else None
    return build_pdf_report(
        output_path=output_path,
        analytics=analytics,
        metadata=metadata,
        notes=notes,
        chart_order=chart_order,
        defect_images=resolved_defect_images,
    )


def _ordered_summary_items(kpis: Mapping[str, object]) -> Iterable[tuple[str, str]]:
    yielded = set()
    for key, label in SUMMARY_LABELS.items():
        if key in kpis:
            yielded.add(key)
            yield key, label
    for key in kpis:
        if key not in yielded:
            yield key, key.replace("_", " ").title()


def _format_summary_value(key: str, value: object) -> str:
    if value is None:
        return "-"
    if key == "pass_rate":
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return str(value)
    if key == "top_defects":
        if isinstance(value, Sequence):
            formatted = [
                f"{item[0]} ({item[1]})"
                for item in value
                if isinstance(item, Sequence) and len(item) >= 2
            ]
            return "; ".join(formatted) if formatted else "-"
        return str(value)
    if key == "defect_counts" and isinstance(value, Mapping):
        formatted = [f"{cls}: {count}" for cls, count in value.items()]
        return "; ".join(formatted) if formatted else "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_rows(path: Path, rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(row)


__all__ = [
    "ExportPaths",
    "export_csv",
    "export_pdf",
]
