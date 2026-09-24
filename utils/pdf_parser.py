"""
Extract text, tables and figures from a PDF as a flat list of `Element`s.

WHY TWO PDF LIBRARIES
---------------------
Using both pdfplumber and PyMuPDF looks like indecision. It is not; they fail at
different things and we use each only where it is strongest.

  pdfplumber  -- superior TABLE detection. It reasons about ruling lines and
                 character positions, and crucially it exposes each detected
                 table's bounding box. We need those boxes (see DEDUPLICATION
                 below), and PyMuPDF does not give them.

  PyMuPDF     -- the only one of the two that cleanly extracts EMBEDDED RASTER
                 IMAGES together with their on-page placement. pdfplumber can
                 list image objects but getting usable pixel data out of it is
                 painful. PyMuPDF is also far faster at text, which matters when
                 iterating on the pipeline.

THE DEDUPLICATION PROBLEM (the subtle bug this module exists to avoid)
----------------------------------------------------------------------
A naive pipeline calls `extract_text()` and `extract_tables()` on the same page
and indexes both. But `extract_text()` returns the table's contents too --
flattened into an unreadable run of numbers with the column structure destroyed:

    "iPhone $ 40,665 $ 39,570 $ 162,863 $ 153,105 Mac 7,382 8,235 28,669 26,012"

That string is then embedded and can be retrieved. When it is, the answering
model sees digits with no column headers and cannot tell which number belongs to
which period -- so it guesses. This is the single most likely source of wrong
numeric answers in a financial RAG system, and it is invisible unless you look
for it.

Our fix: detect tables first, collect their bounding boxes, then extract text
from the page with everything inside those boxes REMOVED. Each table is indexed
exactly once, in a structure-preserving Markdown form. Nothing is lost and
nothing is duplicated.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from utils.config import PROJECT_ROOT, AppConfig
from utils.elements import Element, ElementKind
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """
    Render a pdfplumber table as a Markdown pipe table.

    WHY MARKDOWN and not CSV, JSON, or the raw nested list:

    1. LLMs read it natively. Markdown tables are ubiquitous in training data,
       so the model reliably associates a cell with its column header. JSON
       costs 3-4x the tokens for the same grid; CSV loses the visual alignment
       that makes header association easy.
    2. It survives chunking as a visible unit -- a human debugging a retrieved
       context sees a table rather than a wall of commas.
    3. It round-trips into the methodology PDF without reformatting.

    WHY `or ""` on every cell: pdfplumber yields None for empty cells (very
    common in financial tables, where a blank means "same as the column to the
    left" or simply spacing). `None` would crash the join; "" renders as an
    empty cell, which is the truthful representation.

    WHY we collapse whitespace inside cells: a wrapped header such as
    "Three Months<newline>Ended" contains a literal newline that would break the
    pipe-table row structure and make the whole table unparseable to the model.
    """
    if not rows:
        return ""

    def clean(cell: str | None) -> str:
        # str.split() with no argument splits on ANY whitespace run, including
        # newlines and tabs, so this both trims and collapses in one step.
        return " ".join(str(cell or "").split())

    header = [clean(c) for c in rows[0]]
    width = len(header)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]

    for row in rows[1:]:
        cells = [clean(c) for c in row]
        # WHY pad/truncate to the header width: pdfplumber occasionally returns
        # a ragged row when a merged cell spans columns. A row with the wrong
        # cell count silently corrupts every column to its right in the model's
        # reading of the table, so we normalise rather than trust it.
        cells = (cells + [""] * width)[:width]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _normalise_table_grid(rows: list[list[str | None]]) -> list[list[str | None]]:
    """
    Strip layout artefacts and restore a rectangular, correctly aligned grid.

    THE PROBLEM (observed in real output, not anticipated)
    Financial PDFs typeset the currency symbol in its own cell so the symbols
    align down the column. pdfplumber faithfully reports that as a real column,
    so the p.4 income statement extracts as ELEVEN columns:

        | Products | $ | 63,355 | $ | 63,948 |  | $ | 245,241 |  | $ | 232,309 |

    when it has five meaningful ones. The cost is not merely cosmetic:
      * roughly half the table's tokens are "$" and empty padding, crowding the
        answering model's context window;
      * an LLM counting columns to locate "the third period" lands on a
        separator and misreads the value;
      * worst of all, the grid is RAGGED -- see below.

    WHY WHOLE-COLUMN DELETION DOES NOT WORK
    ---------------------------------------
    The obvious fix is to delete any column that is entirely artefacts. It fails
    here, because the currency symbol makes rows disagree about what each column
    holds. From the same table:

        Products : ["Products", "$", "63,355", "$", "63,948", ...]
        Services : ["Services", "19,604", "",   "17,486", "",   ...]

    Column 1 holds "$" in one row and a real value in the next. No column is
    uniformly droppable, so whole-column analysis preserves the mess.

    THE FIX: squeeze each row INDEPENDENTLY, then normalise every row to the
    modal width. Squeezing collapses both rows above to the same five logical
    fields and, as a side effect, realigns them into a true grid.
    """
    if not rows:
        return rows

    # WHY this specific set: these are typesetting artefacts, never data. We
    # deliberately do NOT include "%" -- a lone "%" cell can carry meaning in a
    # percentage-change column, and dropping it would lose information.
    artefacts = {"", "$", "(", ")", "|", "—", "–", "-"}

    def is_artefact(cell: str | None) -> bool:
        return " ".join(str(cell or "").split()) in artefacts

    def squeeze(row: list[str | None]) -> list[str]:
        """
        Drop artefact cells, and re-attach a stranded "%" to the value it
        belongs to.

        WHY "%" IS MERGED RATHER THAN DROPPED OR KEPT AS ITS OWN COLUMN:
        the percentage-change columns on p.18 typeset the number and the sign in
        separate cells, so a row arrives as [..., "3", "%", ...]. Keeping "%"
        as a column inflates the width past the header's column count (7 header
        labels vs 9 data cells), which is exactly the misalignment this function
        exists to remove. Dropping it instead would turn "3%" into a bare "3" --
        losing the unit and letting the model read a percentage as a dollar
        amount. Merging preserves both the alignment and the meaning.
        """
        out: list[str] = []
        for cell in row:
            text = " ".join(str(cell or "").split())
            if text == "%" and out:
                out[-1] = f"{out[-1]}%"
                continue
            if is_artefact(text):
                continue
            out.append(text)
        return out

    # Row-wise, for the reason set out in the docstring: the grid is ragged, so
    # alignment can only be recovered per row. Result for the two rows above:
    #   ["Products", "63,355", "63,948", "245,241", "232,309"]
    #   ["Services", "19,604", "17,486", "58,941",  "50,148"]
    squeezed = [squeeze(list(r)) for r in rows]

    # WHY the modal width and not the max: section-label rows ("Cost of sales:")
    # squeeze to a single cell, and a stray merged cell can squeeze to more than
    # the true width. The MODE is the width that most data rows agree on, which
    # is the robust estimate of the real column count.

    widths = Counter(len(r) for r in squeezed if len(r) >= 2)
    if not widths:
        return rows
    target = widths.most_common(1)[0][0]

    out: list[list[str | None]] = []
    for row in squeezed:
        if len(row) == target:
            out.append(list(row))
        elif len(row) < target:
            # WHY pad on the RIGHT: a short row is almost always a label or
            # section heading whose text belongs in the first (label) column.
            # Padding right keeps that label under the label column.
            out.append(list(row) + [""] * (target - len(row)))
        else:
            # WHY keep the label and the LAST (target-1) values when a row is
            # too long: the overflow comes from a label that pdfplumber split
            # across cells (e.g. a footnote marker). The numeric values are
            # right-aligned in the source, so the trailing cells are the real
            # data and the leading extras belong to the label.
            label = " ".join(row[: len(row) - (target - 1)])
            out.append([label] + list(row[len(row) - (target - 1) :]))

    return out


def _header_context_above(
    page, table_bbox: tuple[float, float, float, float], floor_y: float
) -> str:
    """
    Recover the column headers that sit ABOVE a detected table's bounding box.

    THE PROBLEM THIS SOLVES (the most important fix in this module)
    --------------------------------------------------------------
    pdfplumber's bbox for the p.4 income statement starts at the first DATA row.
    The period headers --

        Three Months Ended        Nine Months Ended
        June 25, 2022  June 26, 2021   June 25, 2022  June 26, 2021

    -- are rendered above the ruling line and fall outside the bbox, so
    `table.extract()` never returns them. The table therefore reaches the index
    as a grid of unlabelled numbers.

    That is fatal for this assignment. "What were iPhone net sales in Q3 2022?"
    has FOUR candidate values in that table and no way to choose between them.
    The model picks one, sounds confident, and is wrong 75% of the time.

    THE FIX: crop the page region between the previous table (or the page top)
    and this table's top edge, and attach that text to the table.

    WHY `floor_y`: on pages with several stacked tables, cropping to the page top
    would sweep up the PREVIOUS table's contents as this one's header. Passing
    the previous table's bottom edge bounds the search correctly.

    WHY we keep only the last few lines: the region can include a whole
    paragraph of narrative. The header is always immediately adjacent to the
    table; text further up is context we already index separately as narrative.
    """
    top = max(floor_y, table_bbox[1] - 90)  # ~90pt ≈ 5-6 lines of 10pt type
    bottom = table_bbox[1]

    # WHY the degenerate check: if a table starts at the very top of a page,
    # top >= bottom and page.crop() raises on an invalid rectangle.
    if bottom - top < 2:
        return ""

    try:
        region = page.crop((page.bbox[0], top, page.bbox[2], bottom))
        text = region.extract_text() or ""
    except Exception:
        # WHY swallow: a failed header lookup degrades the table slightly. An
        # exception here would abort ingestion of the entire document.
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-6:])


def _is_meaningful_table(rows: list[list[str | None]], cfg: AppConfig) -> bool:
    """
    Reject pdfplumber's false positives.

    WHY this filter is necessary: on the 10-Q cover page, the checkbox rows
    ("Large accelerated filer [X]   Accelerated filer [ ]") are laid out with
    ruling lines and get detected as tables. So do some page headers. Indexing
    them wastes a summarisation API call each and, worse, injects meaningless
    grids into retrieval results for numeric queries.

    The tests below encode what "tabular data" actually means here: enough rows
    and columns, AND more than one row carrying real content -- a single
    populated row is a heading, not a table.
    """
    if len(rows) < cfg.ingestion.min_table_rows:
        return False
    if not rows or len(rows[0]) < cfg.ingestion.min_table_cols:
        return False

    populated = sum(
        1 for row in rows if sum(1 for c in row if c and str(c).strip()) >= 2
    )
    return populated >= cfg.ingestion.min_table_rows


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------
def _is_meaningful_figure(width: int, height: int, cfg: AppConfig) -> tuple[bool, str]:
    """
    Decide whether an embedded image carries answerable information.

    Returns (keep, reason). The reason is returned even on success so the
    ingestion log can report *why* each image was kept or dropped. That audit
    trail is what lets the methodology PDF state, with evidence, what the figure
    path actually found in this document.

    WHY filter at all: a 10-Q's embedded images are almost entirely corporate
    logos, horizontal rules and signature glyphs. Each one we keep costs a
    vision-model call and adds a noise element to the index. Each one we wrongly
    drop is a silently missing answer -- so the thresholds are deliberately
    permissive (see config.yaml) and every decision is logged.
    """
    # WHY the degenerate check comes first: a zero dimension would raise
    # ZeroDivisionError in the aspect-ratio maths below and abort the whole
    # ingestion run over one malformed embedded object.
    if width <= 0 or height <= 0:
        return False, "degenerate dimensions"
    if width < cfg.ingestion.min_figure_width_px:
        return False, f"width {width}px below {cfg.ingestion.min_figure_width_px}px"
    if height < cfg.ingestion.min_figure_height_px:
        return False, f"height {height}px below {cfg.ingestion.min_figure_height_px}px"

    aspect = max(width / height, height / width)
    if aspect > cfg.ingestion.max_figure_aspect_ratio:
        return False, f"aspect ratio {aspect:.1f} suggests a rule or border"

    return True, f"{width}x{height}px"


def _extract_figures(
    doc: fitz.Document, cfg: AppConfig, figures_dir: Path
) -> list[Element]:
    """
    Pull embedded raster images out of the PDF and persist them to disk.

    WHY WRITE THE IMAGE FILES OUT rather than holding bytes in memory:
      * the vision model is called in a later, separate stage -- decoupling
        parsing from enrichment means a failed API call does not force a
        re-parse;
      * a human reviewer (and we, while tuning the filters) can open
        data/processed/figures/ and see exactly what the filter decided;
      * the methodology PDF can include them as evidence.

    WHY `get_images(full=True)`: it returns the xref table including each
    image's true pixel dimensions, which the filter needs. Without `full` we
    would have to decode every image just to measure it.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    elements: list[Element] = []
    kept = dropped = 0

    # WHY dedupe across the WHOLE document, not per page: a logo placed in a
    # running header appears on every page but is ONE embedded object reused by
    # reference. Scoping this set per page would extract, store and caption the
    # same logo 28 times -- 28 wasted vision calls and 28 duplicate elements.
    seen_xrefs: set[int] = set()

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_no = page_index + 1

        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            width, height = img[2], img[3]
            keep, reason = _is_meaningful_figure(width, height, cfg)
            if not keep:
                dropped += 1
                log.debug("p.%d image xref=%d dropped: %s", page_no, xref, reason)
                continue

            try:
                pix = fitz.Pixmap(doc, xref)
                # WHY convert CMYK to RGB: PNG cannot encode CMYK, and vision
                # APIs reject it. `pix.n - pix.alpha >= 4` is PyMuPDF's idiom
                # for "more than three colour channels", i.e. CMYK.
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                out_path = figures_dir / f"p{page_no:02d}_x{xref}.png"
                pix.save(out_path)
                pix = None  # release the C-level buffer promptly
            except Exception as exc:
                # WHY catch broadly and continue: one corrupt embedded image
                # must not abort ingestion of a 28-page document. Logged, so the
                # failure is visible rather than silent.
                log.warning(
                    "p.%d image xref=%d failed to extract: %s", page_no, xref, exc
                )
                dropped += 1
                continue

            kept += 1
            elements.append(
                Element(
                    kind=ElementKind.FIGURE,
                    page=page_no,
                    # WHY content is empty here: the real content is the vision
                    # model's description, filled in by the enrichment stage.
                    # Parsing stays pure and offline -- no network calls in this
                    # module, which keeps it fast and unit-testable.
                    content="",
                    metadata={
                        # WHY store a PROJECT-RELATIVE path: this value is
                        # persisted into the docstore, which is a reviewable
                        # artefact. An absolute path bakes in this machine's
                        # directory layout, so the store would break the moment
                        # the repo is cloned elsewhere -- and it leaks the
                        # author's local filesystem into a deliverable.
                        "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "width": width,
                        "height": height,
                        "xref": xref,
                        "label": f"embedded image {xref}",
                        "filter_reason": reason,
                    },
                )
            )

    log.info("Figures: %d kept, %d rejected by filters", kept, dropped)
    return elements


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def parse_pdf(cfg: AppConfig) -> list[Element]:
    """
    Parse the configured PDF into text, table and figure elements.

    Order of operations matters and is deliberate:
      1. detect tables and record their bounding boxes;
      2. extract page text with those boxes masked out   <- the dedup step;
      3. extract figures.

    Doing (2) before (1) would make the masking impossible.
    """
    pdf_path = cfg.paths.absolute("raw_pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source PDF not found at {pdf_path}. "
            "Place the 10-Q there (see README) or update config.yaml -> paths.raw_pdf."
        )

    elements: list[Element] = []
    table_count = 0

    log.info("Parsing %s", pdf_path.name)

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_no = page_index + 1

            # --- 1. tables -------------------------------------------------
            # WHY `find_tables()` rather than `extract_tables()`: find_tables
            # returns Table OBJECTS carrying `.bbox`, which step 2 requires.
            # extract_tables() returns only cell values and throws the geometry
            # away.
            # WHY sort by vertical position: find_tables() returns tables in
            # detection order, not reading order. The header-recovery step below
            # needs the PREVIOUS table in page order to bound its search, so the
            # list must be top-to-bottom for that bound to be correct.
            found = sorted(page.find_tables(), key=lambda t: t.bbox[1])
            table_bboxes: list[tuple[float, float, float, float]] = []

            # WHY track this: it becomes `floor_y` for the next table's header
            # search, stopping it from swallowing the table above. Starts at the
            # page's top edge.
            prev_table_bottom = page.bbox[1]

            for t_index, table in enumerate(found):
                rows = table.extract()

                # WHY we record the bbox even for REJECTED tables: the region is
                # still visually a grid. Letting its text flow back into the
                # narrative would reintroduce exactly the flattened-numbers
                # problem this module exists to prevent.
                table_bboxes.append(table.bbox)

                if not _is_meaningful_table(rows, cfg):
                    log.debug("p.%d table %d rejected as non-tabular", page_no, t_index)
                    prev_table_bottom = max(prev_table_bottom, table.bbox[3])
                    continue

                # Order matters: collapse the noise columns BEFORE rendering, so
                # the Markdown header row and data rows share a column count.
                rows = _normalise_table_grid(rows)
                header_context = _header_context_above(
                    page, table.bbox, prev_table_bottom
                )
                prev_table_bottom = max(prev_table_bottom, table.bbox[3])

                # WHY the header text is prepended as a plain block rather than
                # spliced in as a Markdown header row: the recovered text is a
                # visual two-line header ("Three Months Ended / June 25, 2022
                # ...") that does not map one-to-one onto columns. Forcing it
                # into the grid would invent an alignment we cannot verify.
                # Presented above the table it gives the model exactly what a
                # human reader uses -- the caption and period labels -- without
                # asserting a false structure.
                body = _table_to_markdown(rows)
                content = f"{header_context}\n\n{body}" if header_context else body

                table_count += 1
                elements.append(
                    Element(
                        kind=ElementKind.TABLE,
                        page=page_no,
                        content=content,
                        metadata={
                            "n_rows": len(rows),
                            "n_cols": len(rows[0]) if rows else 0,
                            "label": f"table {t_index + 1}",
                            "has_header_context": bool(header_context),
                        },
                    )
                )

            # --- 2. narrative text, with table regions removed -------------
            def _outside_tables(obj: dict) -> bool:
                """
                Keep a PDF object only if it lies outside every detected table.

                WHY midpoint containment rather than rectangle overlap: a
                character's box may clip a table border by a fraction of a point
                without belonging to the table. Testing the midpoint is the
                standard robust approximation and avoids dropping the first word
                of a paragraph that sits flush against a table edge.

                WHY a closure over `table_bboxes`: pdfplumber's `filter()` takes
                a single-argument predicate, so the boxes must be captured
                rather than passed.
                """
                cx = (obj["x0"] + obj["x1"]) / 2
                cy = (obj["top"] + obj["bottom"]) / 2
                for x0, top, x1, bottom in table_bboxes:
                    if x0 <= cx <= x1 and top <= cy <= bottom:
                        return False
                return True

            filtered = page.filter(_outside_tables) if table_bboxes else page
            text = filtered.extract_text() or ""

            # WHY the length threshold: pages that are entirely tables (most of
            # this filing) legitimately leave nothing behind but a page number
            # and a footer. Indexing "Apple Inc. | Q3 2022 Form 10-Q | 7" as a
            # retrievable element is pure noise that competes with real content.
            if len(text.strip()) > 80:
                elements.append(
                    Element(
                        kind=ElementKind.TEXT,
                        page=page_no,
                        content=text.strip(),
                        metadata={"label": "narrative"},
                    )
                )

    # --- 3. figures ----------------------------------------------------------
    # WHY a separate `fitz` pass instead of folding into the loop above: the two
    # libraries maintain independent file handles and page caches. Interleaving
    # them roughly doubles peak memory for no benefit, and keeping them separate
    # makes each stage independently testable.
    doc = fitz.open(pdf_path)
    try:
        elements.extend(_extract_figures(doc, cfg, cfg.paths.absolute("figures_dir")))
    finally:
        # WHY try/finally: PyMuPDF holds an OS file handle. On Windows an
        # unclosed handle prevents the PDF being moved or overwritten until the
        # process exits -- an annoying, hard-to-attribute bug during iteration.
        doc.close()

    n_text = sum(1 for e in elements if e.kind is ElementKind.TEXT)
    n_fig = sum(1 for e in elements if e.kind is ElementKind.FIGURE)
    log.info(
        "Parsed %d elements: %d text, %d tables, %d figures",
        len(elements), n_text, table_count, n_fig,
    )
    return elements
