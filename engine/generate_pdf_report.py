
from pathlib import Path
import re

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Preformatted,
)


ROOT = Path(__file__).resolve().parent.parent

MARKDOWN_FILE = ROOT / "reports" / "comprehensive_research_report.md"
PDF_FILE = ROOT / "AI_Quant_Research_Lab_Final_Report_v2.pdf"


def create_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=29,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10 * mm,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceBefore=8 * mm,
        spaceAfter=4 * mm,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="SubHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="BodyTextCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=3 * mm,
    ))

    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontSize=6.5,
        leading=8,
        spaceAfter=0,
    ))

    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=8,
        textColor=colors.white,
        spaceAfter=0,
    ))

    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7,
        leading=9,
        backColor=colors.HexColor("#F2F2F2"),
        borderPadding=5,
    ))

    return styles


def inline_markup(element):
    """Convert supported HTML inline tags to ReportLab markup."""

    result = "".join(
        str(child) if isinstance(child, NavigableString)
        else inline_markup(child)
        for child in element.children
    )

    replacements = {
        "<strong>": "<b>",
        "</strong>": "</b>",
        "<em>": "<i>",
        "</em>": "</i>",
        "<code>": '<font name="Courier">',
        "</code>": "</font>",
        "<br/>": "<br/>",
        "<br>": "<br/>",
    }

    for old, new in replacements.items():
        result = result.replace(old, new)

    # Remove unsupported link formatting while retaining link text.
    result = re.sub(r'<a [^>]*>', "", result)
    result = result.replace("</a>", "")

    return result


def build_table(element, available_width, styles):
    rows = []

    for row_index, tr in enumerate(element.find_all("tr")):
        cells = []

        for cell in tr.find_all(["th", "td"], recursive=False):
            content = inline_markup(cell)

            style = (
                styles["TableHeader"]
                if row_index == 0
                else styles["TableCell"]
            )

            cells.append(Paragraph(content or " ", style))

        if cells:
            rows.append(cells)

    if not rows:
        return None

    column_count = max(len(row) for row in rows)

    # Normalize rows in case a Markdown table has uneven rows.
    for row in rows:
        while len(row) < column_count:
            row.append(Paragraph(" ", styles["TableCell"]))

    column_width = available_width / column_count

    table = Table(
        rows,
        colWidths=[column_width] * column_count,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#263746")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35,
         colors.HexColor("#B8B8B8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F3F6F8"),
        ]),
    ]))

    return table


def process_blocks(parent, story, styles, available_width):
    for element in parent.children:

        if isinstance(element, NavigableString):
            continue

        if not isinstance(element, Tag):
            continue

        tag = element.name

        if tag == "h1":
            story.append(Paragraph(
                inline_markup(element),
                styles["ReportTitle"],
            ))

        elif tag == "h2":
            story.append(Paragraph(
                inline_markup(element),
                styles["SectionHeading"],
            ))

        elif tag in ("h3", "h4"):
            story.append(Paragraph(
                inline_markup(element),
                styles["SubHeading"],
            ))

        elif tag == "p":
            story.append(Paragraph(
                inline_markup(element),
                styles["BodyTextCustom"],
            ))

        elif tag == "table":
            table = build_table(element, available_width, styles)

            if table:
                story.append(Spacer(1, 2 * mm))
                story.append(table)
                story.append(Spacer(1, 3 * mm))

        elif tag in ("ul", "ol"):
            for index, item in enumerate(
                element.find_all("li", recursive=False), start=1
            ):
                prefix = "• " if tag == "ul" else f"{index}. "

                story.append(Paragraph(
                    prefix + inline_markup(item),
                    styles["BodyTextCustom"],
                    bulletText=None,
                ))

        elif tag == "pre":
            code = element.get_text()
            story.append(Preformatted(
                code,
                styles["CodeBlock"],
            ))
            story.append(Spacer(1, 3 * mm))

        elif tag == "blockquote":
            story.append(Paragraph(
                inline_markup(element),
                styles["BodyTextCustom"],
            ))

        elif tag == "hr":
            story.append(Spacer(1, 2 * mm))
            story.append(HRFlowable(
                width="100%",
                thickness=0.6,
                color=colors.HexColor("#BBBBBB"),
            ))
            story.append(Spacer(1, 2 * mm))

        else:
            process_blocks(
                element,
                story,
                styles,
                available_width,
            )


def add_page_number(canvas, document):
    canvas.saveState()

    page_width, _ = landscape(A4)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))

    canvas.drawString(
        18 * mm,
        10 * mm,
        "AI Quant Research Lab | Comprehensive Research Report",
    )

    canvas.drawRightString(
        page_width - 18 * mm,
        10 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def main():
    if not MARKDOWN_FILE.exists():
        raise FileNotFoundError(
            f"Markdown report not found: {MARKDOWN_FILE}"
        )

    markdown_text = MARKDOWN_FILE.read_text(encoding="utf-8")

    html = markdown.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
        ],
    )

    soup = BeautifulSoup(html, "html.parser")

    page_size = landscape(A4)

    document = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=page_size,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Quant Research Lab - Comprehensive Research Report",
        author="AI Quant Research Lab",
    )

    styles = create_styles()
    story = []

    available_width = (
        page_size[0] - document.leftMargin - document.rightMargin
    )

    process_blocks(
        soup,
        story,
        styles,
        available_width,
    )

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    print("=" * 55)
    print("PDF GENERATED SUCCESSFULLY")
    print("=" * 55)
    print(f"Output: {PDF_FILE}")
    print(f"Size: {PDF_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
