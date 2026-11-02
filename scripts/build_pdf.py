"""Build a PDF from a small markdown subset: # title, ## headings, paragraphs, * bullets, | tables |, ![caption](image.png).

    .venv/bin/python scripts/build_pdf.py docs/midway-report.md ~/Downloads/CS576-Midway-Singhal.pdf
"""
import os
import re
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SRC, OUT = sys.argv[1], sys.argv[2]
styles = getSampleStyleSheet()
title_style = ParagraphStyle("t", parent=styles["Title"], fontName="Times-Bold", fontSize=15, leading=19, alignment=TA_CENTER, spaceAfter=4)
sub_style = ParagraphStyle("s", parent=styles["Normal"], fontName="Times-Roman", fontSize=10.5, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#333333"), spaceAfter=10)
h_style = ParagraphStyle("h", parent=styles["Heading2"], fontName="Times-Bold", fontSize=12, leading=14, spaceBefore=9, spaceAfter=3)
body_style = ParagraphStyle("b", parent=styles["Normal"], fontName="Times-Roman", fontSize=11, leading=14, spaceAfter=6)
bullet_style = ParagraphStyle("bl", parent=body_style, spaceAfter=2)
cell_style = ParagraphStyle("c", parent=body_style, fontSize=8.5, leading=10.5, spaceAfter=0)
caption_style = ParagraphStyle("cap", parent=body_style, fontSize=9.5, leading=12, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)


def is_rule(line):
    return set(line.replace("|", "").strip()) <= set("-: ")


def inline(text):
    text = escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


text = open(SRC, encoding="utf-8").read()
blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text) if b.strip()]
story, first_para = [], True
for block in blocks:
    lines = block.split("\n")
    if lines[0].startswith("# "):
        story.append(Paragraph(inline(lines[0][2:].strip()), title_style))
        continue
    if lines[0].startswith("## "):
        story.append(Paragraph(inline(lines[0][3:].strip()), h_style))
        rest = "\n".join(lines[1:]).strip()
        if rest:
            story.append(Paragraph(inline(" ".join(l.strip() for l in rest.split("\n"))), body_style))
        continue
    if len(lines) >= 2 and lines[0].startswith("|") and is_rule(lines[1]):
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not is_rule(l)]
        ncol = max(len(r) for r in rows)
        data = [[Paragraph(inline(c), cell_style) for c in r] + [""] * (ncol - len(r)) for r in rows]
        avail = letter[0] - 2 * inch
        t = Table(data, colWidths=[avail / ncol] * ncol, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
        story.append(t)
        story.append(Spacer(1, 6))
        continue
    m = re.fullmatch(r"!\[(.*?)\]\((.+?)\)", block.strip())
    if m:
        caption, path = m.group(1), m.group(2)
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(SRC)), path)
        from PIL import Image as PILImage
        w, h = PILImage.open(path).size
        max_w, max_h = letter[0] - 2 * inch, 5.2 * inch
        scale = min(max_w / w, max_h / h)
        story.append(Image(path, width=w * scale, height=h * scale))
        if caption:
            story.append(Paragraph(inline(caption), caption_style))
        continue
    if all(l.lstrip().startswith("* ") for l in lines):
        items = [ListItem(Paragraph(inline(l.lstrip()[2:].strip()), bullet_style), leftIndent=14) for l in lines]
        story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9))
        story.append(Spacer(1, 4))
        continue
    style = sub_style if first_para else body_style
    first_para = False
    story.append(Paragraph(inline(" ".join(l.strip() for l in lines)), style))

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=1 * inch, rightMargin=1 * inch, topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                        title="CS576 Term Project", author="Om Singhal")
doc.build(story)
print("wrote", OUT)
