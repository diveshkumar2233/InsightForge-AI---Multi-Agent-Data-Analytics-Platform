"""
utils/pdf_utils.py
───────────────────
PDF report generation using ReportLab Platypus.
Produces professional multi-page reports with cover, TOC, charts, and tables.
"""

from __future__ import annotations

import io
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Fonts ────────────────────────────────────────────────────────────────────
_FONT_DIR = Path("/usr/share/fonts/truetype/custom")
_REGISTERED: set = set()

def _reg(name: str, filename: str) -> None:
    path = _FONT_DIR / filename
    if path.exists() and name not in _REGISTERED:
        pdfmetrics.registerFont(TTFont(name, str(path)))
        _REGISTERED.add(name)

_reg("Poppins",       "Poppins-Regular.ttf")
_reg("Poppins-Bold",  "Poppins-Bold.ttf")
_reg("Inter",         "Inter-Regular.ttf")
_reg("Inter-Bold",    "Inter-Bold.ttf")

_BODY_FONT    = "Inter"         if "Inter"       in _REGISTERED else "Helvetica"
_HEADING_FONT = "Poppins-Bold"  if "Poppins-Bold" in _REGISTERED else "Helvetica-Bold"

# ── Colour Palette ────────────────────────────────────────────────────────────
PRIMARY   = colors.HexColor("#1B4FD8")
SECONDARY = colors.HexColor("#0EA5E9")
ACCENT    = colors.HexColor("#7C3AED")
DARK      = colors.HexColor("#0F172A")
LIGHT_BG  = colors.HexColor("#F8FAFC")
BORDER    = colors.HexColor("#E2E8F0")
WHITE     = colors.white

# ── Styles ────────────────────────────────────────────────────────────────────

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}

    s["title"] = ParagraphStyle(
        "IFTitle", fontName=_HEADING_FONT, fontSize=28,
        textColor=WHITE, alignment=TA_CENTER, leading=36, spaceAfter=8,
    )
    s["subtitle"] = ParagraphStyle(
        "IFSubtitle", fontName=_BODY_FONT, fontSize=13,
        textColor=colors.HexColor("#CBD5E1"), alignment=TA_CENTER, leading=18,
    )
    s["h1"] = ParagraphStyle(
        "IFH1", fontName=_HEADING_FONT, fontSize=18,
        textColor=PRIMARY, spaceBefore=14, spaceAfter=6, leading=24,
    )
    s["h2"] = ParagraphStyle(
        "IFH2", fontName=_HEADING_FONT, fontSize=13,
        textColor=DARK, spaceBefore=10, spaceAfter=4, leading=18,
    )
    s["body"] = ParagraphStyle(
        "IFBody", fontName=_BODY_FONT, fontSize=10,
        textColor=DARK, leading=15, spaceAfter=6,
    )
    s["caption"] = ParagraphStyle(
        "IFCaption", fontName=_BODY_FONT, fontSize=8,
        textColor=colors.HexColor("#64748B"), leading=12, alignment=TA_CENTER,
    )
    s["kpi_val"] = ParagraphStyle(
        "IFKpiVal", fontName=_HEADING_FONT, fontSize=22,
        textColor=PRIMARY, alignment=TA_CENTER, leading=26,
    )
    s["kpi_label"] = ParagraphStyle(
        "IFKpiLabel", fontName=_BODY_FONT, fontSize=9,
        textColor=colors.HexColor("#64748B"), alignment=TA_CENTER, leading=12,
    )
    return s


STYLES = _build_styles()

# ── Page Template Callbacks ───────────────────────────────────────────────────

def _header_footer(canvas, doc, title: str = "InsightForge AI Report") -> None:
    w, h = A4
    canvas.saveState()
    # Header bar
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, h - 18*mm, w, 18*mm, fill=1, stroke=0)
    canvas.setFont(_HEADING_FONT if _HEADING_FONT != "Helvetica-Bold" else "Helvetica-Bold", 9)
    canvas.setFillColor(WHITE)
    canvas.drawString(15*mm, h - 11*mm, title)
    canvas.drawRightString(w - 15*mm, h - 11*mm, datetime.now().strftime("%B %d, %Y"))
    # Footer
    canvas.setFillColor(LIGHT_BG)
    canvas.rect(0, 0, w, 10*mm, fill=1, stroke=0)
    canvas.setFont(_BODY_FONT if _BODY_FONT != "Helvetica" else "Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(15*mm, 3.5*mm, "InsightForge AI | Confidential")
    canvas.drawRightString(w - 15*mm, 3.5*mm, f"Page {doc.page}")
    canvas.restoreState()


def _cover_page(canvas, doc) -> None:
    w, h = A4
    canvas.saveState()
    # Gradient-like background via stacked rects
    canvas.setFillColor(colors.HexColor("#0F172A"))
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, h * 0.55, w, h * 0.45, fill=1, stroke=0)
    # Decorative circle
    canvas.setFillColor(SECONDARY)
    canvas.setStrokeColor(WHITE)
    canvas.setLineWidth(0)
    canvas.circle(w - 30*mm, h - 30*mm, 40*mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#0EA5E920"))
    canvas.circle(30*mm, 30*mm, 55*mm, fill=1, stroke=0)
    canvas.restoreState()


# ── Table Helper ─────────────────────────────────────────────────────────────

def _df_to_table(df: pd.DataFrame, max_rows: int = 20) -> Table:
    df_display = df.head(max_rows)
    header = [Paragraph(f"<b>{c}</b>", STYLES["caption"]) for c in df_display.columns]
    rows   = [[Paragraph(str(v), STYLES["caption"]) for v in row]
              for row in df_display.values]
    data   = [header] + rows

    col_count = len(df_display.columns)
    available = A4[0] - 30*mm
    col_width = available / col_count

    tbl = Table(data, colWidths=[col_width] * col_count, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID",        (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",(0, 0), (-1, -1), 5),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ]))
    return tbl


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    output_path: str | Path,
    report_title: str,
    sections: list[dict[str, Any]],
    kpis: list[dict] | None = None,
    dataframe: pd.DataFrame | None = None,
) -> Path:
    """
    Build a PDF report.

    sections: list of {"heading": str, "content": str, "chart_path": str (optional)}
    kpis:     list of {"label": str, "value": str}
    """
    output_path = Path(output_path)
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=25*mm,  bottomMargin=18*mm,
        title=report_title,
        author="InsightForge AI",
    )

    elements: list[Any] = []

    # ── Cover ──────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 60*mm))
    elements.append(Paragraph(report_title, STYLES["title"]))
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph(
        f"Generated by InsightForge AI &nbsp;·&nbsp; {datetime.now().strftime('%B %d, %Y')}",
        STYLES["subtitle"],
    ))
    elements.append(PageBreak())

    # ── KPIs ───────────────────────────────────────────────────────────────
    if kpis:
        elements.append(Paragraph("Key Performance Indicators", STYLES["h1"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6))

        gap = 5*mm
        cols = min(len(kpis), 4)
        avail = A4[0] - 30*mm
        kpi_w = (avail - gap * (cols - 1)) / cols

        kpi_data = [[
            Table(
                [[Paragraph(k["value"], STYLES["kpi_val"])],
                 [Paragraph(k["label"], STYLES["kpi_label"])]],
                colWidths=[kpi_w],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                    ("LINEBELOW",  (0, 0), (-1, 0), 2, PRIMARY),
                    ("BOX",        (0, 0), (-1, -1), 0.5, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]),
            )
            for k in kpis[:cols]
        ]]

        widths = []
        for i in range(cols):
            widths.append(kpi_w)
            if i < cols - 1:
                widths.append(gap)

        # insert gap columns
        row_with_gaps = []
        for i, cell in enumerate(kpi_data[0]):
            row_with_gaps.append(cell)
            if i < len(kpi_data[0]) - 1:
                row_with_gaps.append(Paragraph("", STYLES["body"]))

        kpi_tbl = Table([row_with_gaps], colWidths=widths)
        kpi_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(kpi_tbl)
        elements.append(Spacer(1, 8*mm))

    # ── Data Preview ───────────────────────────────────────────────────────
    if dataframe is not None and not dataframe.empty:
        elements.append(Paragraph("Data Preview", STYLES["h1"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6))
        elements.append(_df_to_table(dataframe))
        elements.append(Spacer(1, 6*mm))

    # ── Content Sections ───────────────────────────────────────────────────
    for sec in sections:
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        chart   = sec.get("chart_path")

        if heading:
            elements.append(Paragraph(heading, STYLES["h1"]))
            elements.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6))

        if content:
            for para in content.split("\n\n"):
                para = para.strip()
                if para:
                    elements.append(Paragraph(para, STYLES["body"]))

        if chart and Path(chart).exists():
            from reportlab.platypus import Image as RLImage
            from PIL import Image as PILImage
            img = PILImage.open(chart)
            iw, ih = img.size
            max_w = A4[0] - 30*mm
            scale = min(max_w / iw, 80*mm / ih)
            elements.append(Spacer(1, 4*mm))
            elements.append(RLImage(chart, width=iw*scale, height=ih*scale))
            elements.append(Spacer(1, 4*mm))

        elements.append(Spacer(1, 4*mm))

    # ── Build ──────────────────────────────────────────────────────────────
    def _first_page(c, d):
        _cover_page(c, d)

    def _later_pages(c, d):
        _header_footer(c, d, title=report_title)

    doc.build(elements, onFirstPage=_first_page, onLaterPages=_later_pages)
    logger.success(f"PDF report written: {output_path}")
    return output_path
