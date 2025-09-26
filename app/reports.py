"""PDF report generation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .analytics import AnalyticsResult


@dataclass(slots=True)
class ReportMetadata:
    title: str
    company: str = ""
    line: str = ""
    lot_id: str = ""
    shift: str = ""
    operator: str = ""
    model_name: str = ""
    rule_version: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, str] = field(default_factory=dict)


def _build_table(data: List[List[str]]) -> Table:
    table = Table(data, hAlign="LEFT")
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
    table.setStyle(style)
    return table


def build_pdf_report(
    output_path: Path,
    analytics: AnalyticsResult,
    metadata: ReportMetadata,
    notes: str = "",
    chart_order: Optional[Iterable[str]] = None,
    defect_images: Optional[List[Path]] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, title=metadata.title)
    styles = getSampleStyleSheet()
    header_style = styles["Heading1"]
    sub_style = styles["Heading3"]
    normal = styles["BodyText"]
    small = ParagraphStyle("Small", parent=normal, fontSize=9)

    story: List = []
    story.append(Paragraph(metadata.title, header_style))
    subtitle_parts = []
    if metadata.company:
        subtitle_parts.append(metadata.company)
    subtitle_parts.append(metadata.generated_at.strftime("%Y-%m-%d %H:%M"))
    story.append(Paragraph(" | ".join(subtitle_parts), sub_style))
    story.append(Spacer(1, 0.4 * cm))

    info_rows = [
        ["Lot", metadata.lot_id or "-", "Line", metadata.line or "-"],
        ["Shift", metadata.shift or "-", "Operator", metadata.operator or "-"],
        ["Model", metadata.model_name or "-", "QC Rule", metadata.rule_version or "-"],
    ]
    for key, value in metadata.extra.items():
        info_rows.append([key, value, "", ""])
    info_table = _build_table([["Key", "Value", "Key", "Value"]] + info_rows)
    story.append(info_table)
    story.append(Spacer(1, 0.6 * cm))

    kpis = analytics.kpis
    story.append(Paragraph("Key Metrics", sub_style))
    kpi_table = _build_table([
        ["Metric", "Value"],
        ["Total Inspections", str(kpis.get("total_inspections", 0))],
        ["Pass Count", str(kpis.get("pass_count", 0))],
        ["Fail Count", str(kpis.get("fail_count", 0))],
        ["Pass Rate", f"{kpis.get('pass_rate', 0):.2f}%"],
    ])
    story.append(kpi_table)
    story.append(Spacer(1, 0.6 * cm))

    chart_keys = list(chart_order) if chart_order else list(analytics.chart_paths.keys())
    for key in chart_keys:
        chart_path = analytics.chart_paths.get(key)
        if not chart_path or not Path(chart_path).exists():
            continue
        story.append(Paragraph(key.replace("_", " ").title(), sub_style))
        story.append(Image(str(chart_path), width=14 * cm, height=8 * cm))
        story.append(Spacer(1, 0.4 * cm))

    if notes:
        story.append(Paragraph("Notes", sub_style))
        story.append(Paragraph(notes, normal))
        story.append(Spacer(1, 0.6 * cm))

    if defect_images:
        story.append(PageBreak())
        story.append(Paragraph("Sample Defect Images", sub_style))
        for image_path in defect_images:
            if not Path(image_path).exists():
                continue
            story.append(Image(str(image_path), width=10 * cm, height=6 * cm))
            story.append(Spacer(1, 0.3 * cm))

    if analytics.confusion is not None and not analytics.confusion.empty:
        story.append(PageBreak())
        story.append(Paragraph("Confusion Matrix", sub_style))
        headers = ["Truth\\Pred"] + list(analytics.confusion.columns)
        rows = [headers]
        for truth_label, row in analytics.confusion.iterrows():
            rows.append([truth_label] + [str(int(value)) for value in row.values])
        story.append(_build_table(rows))
        story.append(Spacer(1, 0.4 * cm))

    doc.build(story)
    return output_path


__all__ = [
    "ReportMetadata",
    "build_pdf_report",
]
