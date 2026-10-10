from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.analytics import get_promotion_summary, get_salary_summary


PURPLE = colors.HexColor("#6C4CF1")
NAVY = colors.HexColor("#20213E")
CSV_LIMITATIONS = (
    "Descriptive demonstration only; differences do not prove discrimination. "
    "Human review and appropriate context are required."
)


def _pdf_document() -> tuple[BytesIO, SimpleDocTemplate, list[Any], Any]:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="FairVision AI report",
        author="FairVision AI",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="FairVisionTitle",
            parent=styles["Title"],
            textColor=NAVY,
            alignment=TA_CENTER,
            fontSize=22,
            leading=27,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FairVisionSection",
            parent=styles["Heading2"],
            textColor=PURPLE,
            fontSize=13,
            leading=17,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    story: list[Any] = []
    return buffer, document, story, styles


def _add_table(story: list[Any], rows: list[list[str]]) -> None:
    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E4E2ED")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F4FC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)


def _metadata(
    story: list[Any],
    source: str,
    filters: dict[str, Any],
    report_date: datetime,
) -> None:
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    story.extend(
        [
            Paragraph(
                f"Generated: {report_date.strftime('%Y-%m-%d %H:%M')} · Data source: {escape(str(source))}",
                styles["BodyText"],
            ),
            Paragraph(
                "Applied filters: "
                + escape(
                    f"Departments: {', '.join(map(str, filters.get('departments', []))) or 'none'}; "
                    f"experience: {filters.get('experience', ('all', 'all'))[0]}–{filters.get('experience', ('all', 'all'))[1]} years"
                ),
                styles["BodyText"],
            ),
            Spacer(1, 10),
        ]
    )


def serialize_report_csv(
    df: pd.DataFrame,
    source: str,
    filters: dict[str, Any],
    report_date: datetime | None = None,
) -> bytes:
    """Export a clean CSV table with provenance, filter, and limitation columns."""
    report_date = report_date or datetime.now()
    exported = df.copy()
    applied_filters = (
        f"Departments: {', '.join(map(str, filters.get('departments', []))) or 'none'}; "
        f"experience: {filters.get('experience', ('all', 'all'))[0]}–"
        f"{filters.get('experience', ('all', 'all'))[1]} years"
    )
    metadata = {
        "report_generated_at": report_date.strftime("%Y-%m-%d %H:%M"),
        "data_source": str(source),
        "applied_filters": applied_filters,
        "limitations": CSV_LIMITATIONS,
    }
    for column, value in metadata.items():
        while column in exported.columns:
            column = f"fairvision_{column}"
        exported[column] = value
    return exported.to_csv(index=False).encode("utf-8")


def create_summary_report(
    df: pd.DataFrame,
    source: str,
    filters: dict[str, Any],
    report_date: datetime | None = None,
) -> bytes:
    report_date = report_date or datetime.now()
    buffer, document, story, styles = _pdf_document()
    story.append(Paragraph("FairVision AI", styles["FairVisionTitle"]))
    story.append(Paragraph("Gender Equality Analytics · Summary Report", styles["Heading2"]))
    _metadata(story, source, filters, report_date)

    story.append(Paragraph("Workforce snapshot", styles["FairVisionSection"]))
    total = len(df)
    female = int(df["gender"].eq("Female").sum())
    male = int(df["gender"].eq("Male").sum())
    known = df["promotion_status"].isin(["Yes", "No"])
    promoted = int(df.loc[known, "promotion_status"].eq("Yes").sum())
    promotion_rate = promoted / int(known.sum()) * 100 if known.any() else None
    _add_table(
        story,
        [
            ["Measure", "Value"],
            ["Employees", str(total)],
            ["Female employees", f"{female} ({female / total:.1%})" if total else "0"],
            ["Male employees", f"{male} ({male / total:.1%})" if total else "0"],
            ["Other / unknown gender", str(total - female - male)],
            ["Recorded promotion rate", f"{promotion_rate:.1f}%" if promotion_rate is not None else "Not available"],
        ],
    )

    salary = get_salary_summary(df)
    story.append(Paragraph("Salary summary", styles["FairVisionSection"]))
    salary_rows = [["Gender", "With salary", "Average", "Median"]]
    for row in salary.itertuples(index=False):
        salary_rows.append(
            [
                escape(str(row.gender)),
                str(row.employees_with_salary),
                f"${row.average_salary:,.2f}",
                f"${row.median_salary:,.2f}",
            ]
        )
    _add_table(story, salary_rows)

    promotion = get_promotion_summary(df)
    story.append(Paragraph("Promotion outcomes", styles["FairVisionSection"]))
    promotion_rows = [["Gender", "Recorded outcomes", "Promoted", "Rate"]]
    for row in promotion.itertuples(index=False):
        promotion_rows.append(
            [
                escape(str(row.gender)),
                str(row.employees_with_recorded_outcome),
                str(int(row.promoted)),
                f"{row.promotion_rate:.1f}%",
            ]
        )
    _add_table(story, promotion_rows)
    story.append(Paragraph("Limitations", styles["FairVisionSection"]))
    story.append(
        Paragraph(
            "This report is descriptive and reflects only the records provided and selected. "
            "The bundled dataset is synthetic demonstration data, not real-world evidence. "
            "Uploaded data provenance is not independently verified. Differences do not prove discrimination "
            "and may reflect missing context, data quality, small samples, job level, tenure, working hours, "
            "or other factors. Human review is required.",
            styles["BodyText"],
        )
    )
    document.build(story)
    return buffer.getvalue()


def create_ml_report(
    result: dict[str, Any],
    source: str,
    filters: dict[str, Any] | None = None,
    report_date: datetime | None = None,
) -> bytes:
    report_date = report_date or datetime.now()
    buffer, document, story, styles = _pdf_document()
    story.append(Paragraph("FairVision AI", styles["FairVisionTitle"]))
    story.append(Paragraph("Experimental ML Evaluation", styles["Heading2"]))
    _metadata(
        story,
        source,
        filters or {"departments": [], "experience": ("all", "all")},
        report_date,
    )
    story.append(Paragraph("Holdout evaluation", styles["FairVisionSection"]))
    _add_table(
        story,
        [
            ["Metric", "Observed value"],
            ["Accuracy", f"{result['accuracy']:.1%}"],
            ["Precision", f"{result['precision']:.1%}"],
            ["Recall", f"{result['recall']:.1%}"],
            ["F1 score", f"{result['f1']:.1%}"],
            ["Train / test records", f"{result['train_size']} / {result['test_size']}"],
            ["Inputs", escape(", ".join(result["feature_columns"]))],
            ["Gender excluded", "Yes"],
        ],
    )
    story.append(Paragraph("Feature importance (encoded model inputs)", styles["FairVisionSection"]))
    importance_rows = [["Feature", "Importance"]]
    for row in result["feature_importance"].head(15).itertuples(index=False):
        importance_rows.append([escape(str(row.feature)), f"{row.importance:.5f}"])
    _add_table(story, importance_rows)
    story.append(Paragraph("Confusion matrix (actual rows × predicted columns)", styles["FairVisionSection"]))
    matrix = result["confusion_matrix"]
    labels = result["labels"]
    _add_table(
        story,
        [
            ["Actual / predicted", *labels],
            [labels[0], *[str(value) for value in matrix[0]]],
            [labels[1], *[str(value) for value in matrix[1]]],
        ],
    )
    story.append(Paragraph("Model limitations", styles["FairVisionSection"]))
    story.append(
        Paragraph(
            "This random-forest model predicts recorded outcomes from non-gender inputs, using a stratified "
            "held-out test set and preprocessing fitted on training data. Scores are specific to this split and "
            "may be unstable, especially for small or synthetic data. Feature importance is not causal evidence. "
            "The model is experimental, is not a hiring or promotion recommendation, and must not be used to make "
            "decisions about individuals. Human review is required.",
            styles["BodyText"],
        )
    )
    document.build(story)
    return buffer.getvalue()
