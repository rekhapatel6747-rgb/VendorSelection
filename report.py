"""
report.py
---------
Generates a downloadable PDF procurement report containing the executive
summary, AI recommendation narrative, vendor comparison table, and charts.

Uses reportlab for PDF layout and plotly's `to_image` (via kaleido) to
rasterize charts. If kaleido is unavailable, the report degrades gracefully
by omitting chart images and noting it in the PDF.
"""

from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils import SCORING_CRITERIA, get_logger

logger = get_logger(__name__)

BRAND_BLUE = colors.HexColor("#1a237e")
LIGHT_BLUE = colors.HexColor("#e8eaf6")


def _fig_to_image(fig, width_cm: float = 16) -> Image | None:
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        buf = io.BytesIO(png_bytes)
        img = Image(buf, width=width_cm * cm, height=width_cm * cm * 0.55)
        return img
    except Exception:  # noqa: BLE001
        logger.warning("Chart-to-image conversion failed (kaleido missing?). Skipping chart.", exc_info=True)
        return None


def build_pdf_report(
    scored_df: pd.DataFrame,
    narrative: Dict[str, str],
    weights: Dict[str, float],
    figures: Dict[str, object],
    company_name: str = "Procurement Team",
) -> bytes:
    """
    Build the PDF report and return it as bytes, ready for st.download_button.

    figures: dict of {"score_bar": fig, "radar": fig, "price": fig,
                       "delivery": fig, "heatmap": fig}
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        title="AI Vendor Selection Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=BRAND_BLUE, fontSize=22, spaceAfter=6
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=BRAND_BLUE, spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)

    story: List = []

    # --- Cover / header ---
    story.append(Paragraph("AI Vendor Selection Assistant", title_style))
    story.append(Paragraph("Procurement Decision Report", styles["Heading3"]))
    story.append(Paragraph(f"Prepared for: {company_name}", body))
    story.append(Spacer(1, 12))

    # --- Executive summary ---
    story.append(Paragraph("Executive Summary", h2))
    story.append(Paragraph(narrative.get("executive_summary", ""), body))

    # --- Recommendation ---
    top = scored_df.iloc[0]
    story.append(Paragraph("Final Recommendation", h2))
    story.append(
        Paragraph(
            f"<b>{top['Vendor']}</b> is recommended with a composite weighted score of "
            f"<b>{top['Weighted Score']:.2f}/100</b> and confidence level "
            f"<b>{narrative.get('confidence', 'N/A')}</b>.",
            body,
        )
    )
    story.append(Paragraph(narrative.get("selection_reasoning", ""), body))

    # --- Weight configuration ---
    story.append(Paragraph("Weight Configuration Used", h2))
    weight_data = [["Criterion", "Weight (%)"]] + [[k, f"{v}"] for k, v in weights.items()]
    wt_table = Table(weight_data, colWidths=[8 * cm, 4 * cm])
    wt_table.setStyle(_table_style())
    story.append(wt_table)

    # --- Vendor comparison table ---
    story.append(PageBreak())
    story.append(Paragraph("Vendor Comparison Table", h2))
    cols = ["Rank", "Vendor", "Weighted Score"] + SCORING_CRITERIA
    table_df = scored_df.copy()
    table_df.columns = [c.replace(" Norm", "") for c in table_df.columns]
    display_cols = ["Rank", "Vendor", "Weighted Score"] + [f"{c} Norm" for c in SCORING_CRITERIA]
    disp = scored_df[display_cols].copy()
    disp.columns = cols
    data = [cols] + disp.round(1).astype(str).values.tolist()
    comp_table = Table(data, repeatRows=1)
    comp_table.setStyle(_table_style(small=True))
    story.append(comp_table)

    # --- Charts ---
    story.append(PageBreak())
    story.append(Paragraph("Visual Analysis", h2))
    for key, caption in [
        ("score_bar", "Vendor Weighted Score Ranking"),
        ("radar", "Multi-Criteria Comparison"),
        ("price", "Price Comparison"),
        ("delivery", "Delivery Comparison"),
        ("heatmap", "Criteria Heatmap"),
    ]:
        fig = figures.get(key)
        if fig is not None:
            img = _fig_to_image(fig)
            if img is not None:
                story.append(Paragraph(caption, styles["Heading4"]))
                story.append(img)
                story.append(Spacer(1, 10))

    # --- Strengths / weaknesses ---
    story.append(PageBreak())
    story.append(Paragraph("Vendor Strengths & Weaknesses", h2))
    for line in narrative.get("strengths_weaknesses", "").split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("- ", "&bull; "), body))

    # --- Risks ---
    story.append(Paragraph("Identified Risks", h2))
    for line in narrative.get("risks", "").split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("- ", "&bull; "), body))

    # --- Trade-offs ---
    story.append(Paragraph("Cost vs. Quality Trade-offs", h2))
    story.append(Paragraph(narrative.get("trade_offs", ""), body))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _table_style(small: bool = False) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7 if small else 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )
