import csv
from pathlib import Path

import pandas as pd
from PIL import Image

from app.analytics import AnalyticsFilters, AnalyticsResult
from app.exporters import export_csv, export_pdf
from app.reports import ReportMetadata


def _build_sample_result(tmp_path: Path) -> AnalyticsResult:
    filters = AnalyticsFilters()
    inspections = pd.DataFrame(
        [
            {"inspection_id": 1, "created_at": "2024-09-01T10:00:00", "status": "PASS", "rule_version": "v1", "inference_ms": 25.0, "fps": 28.0, "frame_path": "frame1.jpg", "batch_id": 10, "lot_id": "LOT1", "line": "L1", "shift": "A", "operator": "OP1", "model_id": 5, "model_name": "Model-A"},
            {"inspection_id": 2, "created_at": "2024-09-01T10:05:00", "status": "FAIL", "rule_version": "v1", "inference_ms": 27.0, "fps": 27.0, "frame_path": "frame2.jpg", "batch_id": 10, "lot_id": "LOT1", "line": "L1", "shift": "A", "operator": "OP1", "model_id": 5, "model_name": "Model-A"},
            {"inspection_id": 3, "created_at": "2024-09-01T10:10:00", "status": "PASS", "rule_version": "v1", "inference_ms": 24.0, "fps": 29.0, "frame_path": "frame3.jpg", "batch_id": 10, "lot_id": "LOT1", "line": "L1", "shift": "A", "operator": "OP1", "model_id": 5, "model_name": "Model-A"},
            {"inspection_id": 4, "created_at": "2024-09-01T10:15:00", "status": "PASS", "rule_version": "v1", "inference_ms": 23.0, "fps": 29.0, "frame_path": "frame4.jpg", "batch_id": 10, "lot_id": "LOT1", "line": "L1", "shift": "A", "operator": "OP1", "model_id": 5, "model_name": "Model-A"},
        ]
    )
    detections = pd.DataFrame(
        [
            {"detection_id": 1, "inspection_id": 2, "cls": "chip", "conf": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5, "created_at": "2024-09-01T10:05:01", "inspection_created_at": "2024-09-01T10:05:00", "lot_id": "LOT1", "line": "L1", "shift": "A"},
            {"detection_id": 2, "inspection_id": 2, "cls": "chip", "conf": 0.8, "x1": 0.2, "y1": 0.2, "x2": 0.6, "y2": 0.6, "created_at": "2024-09-01T10:05:02", "inspection_created_at": "2024-09-01T10:05:00", "lot_id": "LOT1", "line": "L1", "shift": "A"},
            {"detection_id": 3, "inspection_id": 3, "cls": "crack", "conf": 0.7, "x1": 0.15, "y1": 0.15, "x2": 0.55, "y2": 0.55, "created_at": "2024-09-01T10:10:05", "inspection_created_at": "2024-09-01T10:10:00", "lot_id": "LOT1", "line": "L1", "shift": "A"},
        ]
    )
    truth = pd.DataFrame(
        [
            {"truth_id": 1, "inspection_id": 2, "cls": "chip", "x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5, "status": "added", "created_at": "2024-09-02T08:00:00", "inspection_created_at": "2024-09-01T10:05:00"},
            {"truth_id": 2, "inspection_id": 3, "cls": "crack", "x1": 0.15, "y1": 0.15, "x2": 0.55, "y2": 0.55, "status": "added", "created_at": "2024-09-02T08:10:00", "inspection_created_at": "2024-09-01T10:10:00"},
        ]
    )

    kpis = {
        "total_inspections": 4,
        "pass_count": 3,
        "fail_count": 1,
        "pass_rate": 75.0,
        "defect_counts": {"chip": 2, "crack": 1},
        "top_defects": [("chip", 2), ("crack", 1)],
    }

    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = {}
    for name in ("defects_by_class", "defect_trend", "pass_fail"):
        chart_file = charts_dir / f"{name}.png"
        Image.new("RGB", (10, 10), color="white").save(chart_file)
        chart_paths[name] = chart_file

    confusion = pd.DataFrame(
        [[5, 1], [2, 7]],
        index=["ok", "chip"],
        columns=["ok", "chip"],
    )

    return AnalyticsResult(
        filters=filters,
        inspections=inspections,
        detections=detections,
        review_truth=truth,
        kpis=kpis,
        chart_paths=chart_paths,
        confusion=confusion,
    )


def test_export_csv_creates_summary(tmp_path):
    analytics = _build_sample_result(tmp_path)
    output_dir = tmp_path / "exports"
    paths = export_csv(analytics, output_dir)

    assert paths.detail_csv and paths.detail_csv.exists()
    assert paths.summary_csv and paths.summary_csv.exists()
    assert paths.confusion_csv and paths.confusion_csv.exists()

    with paths.summary_csv.open(newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    assert any("Pass Rate (%)" in row and "75.00" in row for row in rows)
    assert any("Top Defects" in row and "chip (2)" in row[1] for row in rows if len(row) >= 2)


def test_export_pdf_writes_file(tmp_path):
    analytics = _build_sample_result(tmp_path)
    pdf_path = tmp_path / "report.pdf"
    metadata = ReportMetadata(
        title="QC Summary",
        company="GlassCo",
        line="Line 1",
        lot_id="LOT1",
        shift="A",
        operator="OP1",
        model_name="Model-A",
        rule_version="rule-v1",
    )

    export_pdf(analytics, metadata, pdf_path, notes="Automated test")

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
