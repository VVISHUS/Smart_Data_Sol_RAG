"""
Render docs/METHODOLOGY.md to the PDF required by the brief.

    python -m utils.build_pdf

WHY A SCRIPT RATHER THAN EXPORTING FROM AN EDITOR
-------------------------------------------------
The methodology document quotes measured results (element counts, column counts,
evaluation accuracy). Those numbers change whenever the pipeline changes. A
hand-exported PDF silently goes stale the moment it is regenerated from an older
copy, and a write-up whose numbers disagree with the code is worse than no
write-up. Building it with a command makes the PDF a derived artefact that can be
regenerated in one step after any change.

WHY REPORTLAB AND NOT PANDOC/WEASYPRINT
---------------------------------------
Pandoc and WeasyPrint produce nicer typography but require system-level installs
(LaTeX, GTK) that a reviewer on a clean Windows machine will not have. ReportLab
is pure Python and installs from pip, so `pip install -r requirements.txt` is
genuinely sufficient to reproduce every artefact in this repo.

SCOPE: this renders the subset of Markdown the methodology document actually
uses -- headings, paragraphs, bullet and numbered lists, tables, fenced code
blocks, bold/italic/inline code, and horizontal rules. It is not a general
Markdown engine, and deliberately so.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# WHY a muted slate rather than pure black for headings: the document is dense
# and mostly prose. A slightly lighter heading colour separates structure from
# body text without the visual noise of a second font family.
HEADING_COLOR = colors.HexColor("#1f2933")
RULE_COLOR = colors.HexColor("#cbd2d9")
CODE_BG = colors.HexColor("#f5f7fa")
TABLE_HEADER_BG = colors.HexColor("#e4e7eb")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=19, leading=23,
                             spaceBefore=4, spaceAfter=10, textColor=HEADING_COLOR),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=14, leading=18,
                             spaceBefore=16, spaceAfter=7, textColor=HEADING_COLOR),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11.5, leading=15,
                             spaceBefore=12, spaceAfter=5, textColor=HEADING_COLOR),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5,
                               leading=14, alignment=TA_LEFT, spaceAfter=7),
        # WHY a dedicated cell style: table text must wrap inside its column.
        # Passing bare strings to ReportLab's Table makes long cells overflow the
        # page rather than wrapping, which silently truncates content.
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8,
                               leading=11, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=base["BodyText"], fontSize=8,
                                   leading=11, spaceAfter=0, fontName="Helvetica-Bold"),
    }


# WHY THIS TABLE EXISTS
# ReportLab's built-in Courier and Helvetica are WinAnsi fonts. They have no
# glyph for box-drawing characters or arrows, and ReportLab substitutes a
# fallback rather than failing -- so the architecture diagram silently rendered
# as a block of "I" and "M" characters in the first build. Verified by extracting
# text back out of the generated PDF.
#
# Substituting ASCII is preferred over embedding a Unicode TTF because font
# availability differs across machines, and this PDF must build identically on a
# reviewer's. An ASCII diagram is also legible if the PDF is ever copied as text.
_ASCII_MAP = {
    # box drawing
    "─": "-", "│": "|",
    "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
    # arrows and markers
    "▼": "v", "▲": "^", "◀": "<", "▶": ">",
    "→": "->", "←": "<-", "•": "*",
    # punctuation
    "—": "--", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "×": "x", "…": "...",
    " ": " ", "≥": ">=", "≤": "<=",
}


# WHY A SECOND, WIDTH-PRESERVING MAP: in prose "->" reads better than a bare
# ">", but inside the architecture diagram every substitution must be exactly
# one character wide. "→" becoming "->" shifts everything after it by one column
# and the box borders no longer line up -- which looks like a rendering bug even
# though the content is correct. Monospace blocks therefore get 1:1 replacements.
_MONO_OVERRIDES = {"→": ">", "←": "<", "—": "-", "…": "."}


def _ascii_safe(text: str, monospace: bool = False) -> str:
    """
    Replace glyphs the built-in PDF fonts cannot render.

    `monospace=True` uses single-character substitutions so that column
    alignment in a preformatted block survives the conversion.
    """
    mapping = {**_ASCII_MAP, **_MONO_OVERRIDES} if monospace else _ASCII_MAP
    for bad, good in mapping.items():
        text = text.replace(bad, good)
    return text


def _inline(text: str) -> str:
    """
    Convert inline Markdown to ReportLab's mini-HTML.

    WHY escape FIRST and in this order: ReportLab parses its own markup, so a
    literal '&' or '<' in the source (common in "R&D" and in the ASCII diagram)
    would raise a parse error or vanish. Escaping before inserting our own tags
    means our tags survive and the content is safe.
    """
    # Normalise unrenderable glyphs BEFORE escaping, for the reason recorded
    # at _ASCII_MAP. Order matters: _ascii_safe can introduce < and > (from
    # arrows), which the escaping below must then handle.
    text = _ascii_safe(text)

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    # WHY a background on inline code: it distinguishes identifiers from prose
    # without a font change, which keeps line heights uniform.
    text = re.sub(r"`(.+?)`",
                  r'<font face="Courier" size="8.5" backColor="#f0f2f5">\1</font>',
                  text)
    # Markdown links -> just the label; the PDF is read on paper as often as on
    # screen, and a bare URL mid-sentence hurts more than the link helps.
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", text)
    return text


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build(md_path: Path, pdf_path: Path) -> None:
    st = _styles()
    lines = md_path.read_text(encoding="utf-8").splitlines()
    flow: list = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- fenced code / diagrams ----------------------------------------
        if stripped.startswith("```"):
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            # WHY Preformatted and not Paragraph: it preserves leading spaces,
            # which is the entire point for the architecture diagram.
            flow.append(Preformatted(
                # WHY normalise here too: this path bypasses _inline(), and it
                # is the one that renders the architecture diagram. Missing it
                # was the bug that turned every box-drawing character into "I".
                _ascii_safe("\n".join(block), monospace=True),
                ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8.2,
                               backColor=CODE_BG, borderPadding=6, leftIndent=2),
            ))
            flow.append(Spacer(1, 8))
            continue

        # --- tables ----------------------------------------------------------
        if stripped.startswith("|") and i + 1 < len(lines) and set(
            lines[i + 1].strip().replace("|", "").replace(" ", "")
        ) <= {"-", ":"} and lines[i + 1].strip().startswith("|"):
            header = _split_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1

            data = [[Paragraph(_inline(c), st["cellhead"]) for c in header]]
            for r in rows:
                r = (r + [""] * len(header))[: len(header)]
                data.append([Paragraph(_inline(c), st["cell"]) for c in r])

            # WHY compute column widths rather than letting ReportLab guess: its
            # default distributes width by content length, which makes a column
            # of short numbers as wide as a column of prose.
            avail = A4[0] - 40 * mm
            ncols = len(header)
            widths = [avail / ncols] * ncols
            if ncols > 1:
                # First column usually holds the label/description: give it more.
                widths = [avail * 0.34] + [avail * 0.66 / (ncols - 1)] * (ncols - 1)

            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE_COLOR),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 10))
            continue

        # --- headings ---------------------------------------------------------
        if stripped.startswith("### "):
            flow.append(Paragraph(_inline(stripped[4:]), st["h3"]))
        elif stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[3:]), st["h2"]))
        elif stripped.startswith("# "):
            flow.append(Paragraph(_inline(stripped[2:]), st["h1"]))

        # --- horizontal rule ---------------------------------------------------
        elif stripped in ("---", "***", "___"):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE_COLOR))
            flow.append(Spacer(1, 8))

        # --- lists --------------------------------------------------------------
        elif re.match(r"^(\d+\.|[-*])\s+", stripped):
            items = []
            ordered = bool(re.match(r"^\d+\.", stripped))
            while i < len(lines) and re.match(r"^(\d+\.|[-*])\s+", lines[i].strip()):
                text = re.sub(r"^(\d+\.|[-*])\s+", "", lines[i].strip())
                # Absorb continuation lines so a wrapped bullet stays one item.
                # WHY `startswith((" ", "\t"))` and not a fixed indent width:
                # the original check required three spaces, but the source wraps
                # list items at two. Continuation lines therefore failed the test
                # and were emitted as separate top-level paragraphs, so a wrapped
                # bullet broke into a bullet plus an orphaned block of text.
                # Any leading whitespace marks a continuation; the block-start
                # check below is what actually ends the item.
                j = i + 1
                while (j < len(lines) and lines[j].strip()
                       and not re.match(r"^(\d+\.\s|[-*]\s|#{1,3}\s|\||```)", lines[j].strip())
                       and lines[j].startswith((" ", "\t"))):
                    text += " " + lines[j].strip()
                    j += 1
                items.append(ListItem(Paragraph(_inline(text), st["body"]),
                                      leftIndent=12))
                i = j
            flow.append(ListFlowable(
                items, bulletType="1" if ordered else "bullet",
                bulletFontSize=8, leftIndent=14,
            ))
            flow.append(Spacer(1, 4))
            continue

        # --- HTML comment marker (the results placeholder) ----------------------
        elif stripped.startswith("<!--"):
            pass

        # --- paragraph ------------------------------------------------------------
        elif stripped:
            # WHY WE JOIN CONSECUTIVE LINES INTO ONE PARAGRAPH
            # The source Markdown is hard-wrapped at ~80 columns. Emitting one
            # Paragraph per physical line produced two visible defects in the
            # rendered PDF:
            #
            #   1. Every wrapped line inherited `spaceAfter`, so body text came
            #      out double-spaced and the document looked broken.
            #   2. Inline emphasis spanning a line break never closed. `*"how did
            #      iPhone revenue compare to last year?"*` has its opening marker
            #      on one line and its closing marker on the next, so the regex
            #      in _inline() matched neither and both asterisks were printed
            #      literally.
            #
            # A Markdown paragraph is a run of non-blank lines, so gathering them
            # before rendering fixes both at once -- and lets ReportLab do the
            # line breaking, which is what it is good at.
            buf: list[str] = []
            while i < len(lines):
                nxt = lines[i].strip()
                # Stop at a blank line or anything that starts a new block.
                if not nxt or re.match(r"^(#{1,3}\s|[-*]\s|\d+\.\s|\||```|---|<!--)", nxt):
                    break
                buf.append(nxt)
                i += 1
            flow.append(Paragraph(_inline(" ".join(buf)), st["body"]))
            continue

        i += 1

    def _footer(canvas, doc):
        """Page numbers. A multi-page report without them is hard to discuss."""
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#7b8794"))
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
        canvas.drawString(20 * mm, 12 * mm,
                          _ascii_safe("RAG over Apple Form 10-Q - Methodology"))
        canvas.restoreState()

    SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title=_ascii_safe("RAG over Apple Form 10-Q - Methodology"),
    ).build(flow, onFirstPage=_footer, onLaterPages=_footer)


if __name__ == "__main__":
    src = PROJECT_ROOT / "docs" / "METHODOLOGY.md"
    out = PROJECT_ROOT / "docs" / "METHODOLOGY.pdf"
    build(src, out)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
