"""
Generate the retrieval-facing `summary` for tables and figures.

WHY THIS STAGE EXISTS AT ALL
----------------------------
This is the step that makes the whole single-pipeline design work, and it is the
part most naive implementations skip.

Embedding a raw financial table is close to useless. Its text is essentially
"iPhone 40,665 39,570 162,863 153,105" -- digits with almost no semantic signal.
A user asking "how did iPhone revenue compare to last year?" shares no
vocabulary with that string, so cosine similarity ranks it low and the correct
table is never retrieved. The system then answers from a narrative chunk that
merely *mentions* iPhone, and produces a confident wrong number.

By generating a natural-language description -- "Net sales by product category
for the three and nine months ended June 25 2022 versus June 26 2021, covering
iPhone, Mac, iPad, Wearables and Services" -- we create text that matches the
question semantically. We embed that, and return the raw table for answering.

The same argument applies to figures, which have NO text at all: without a
vision-model caption they are simply unretrievable.

WHY SUMMARISE TABLES BUT NOT NARRATIVE TEXT
-------------------------------------------
Narrative prose is already natural language; it embeds well as-is. Summarising
it would discard the specific phrasing a user might quote ("substantial doubt",
"material weakness"), trading recall for nothing. Asymmetric treatment is not an
inconsistency -- it follows from which modality has a semantic-signal problem.
"""

from __future__ import annotations

from pathlib import Path

from utils.config import PROJECT_ROOT, AppConfig
from utils.elements import Element, ElementKind
from utils.provider import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# WHY the prompt forbids restating the numbers: we are building a SEARCH KEY,
# not a replacement for the table. Numbers in the summary would (a) waste tokens,
# (b) risk the summariser transcribing a figure wrongly, and (c) create a second
# copy of the data that could be retrieved and answered from -- defeating the
# entire point of returning the verbatim table.
#
# WHY it asks for row/column LABELS and the time periods: those are exactly the
# terms a user's question will contain ("iPhone", "nine months", "segment",
# "gross margin"). Naming them maximises lexical and semantic overlap.
TABLE_SUMMARY_PROMPT = """You are indexing tables from an SEC quarterly report (Form 10-Q).

Write a single dense paragraph (40-70 words) describing WHAT THIS TABLE CONTAINS,
so that it can be found by semantic search.

Include:
- the financial concept being reported (e.g. net sales, gross margin, share repurchases)
- the row labels / line items present
- the column labels, especially the time periods and units being compared

Do NOT restate any numeric values. Do NOT add commentary or analysis.

Table (Markdown):
{table}

Description:"""


# WHY this prompt is explicitly conditional ("if it is a chart... if it is a
# logo..."): the extractor cannot know what an image is before the model sees
# it. A single prompt that handles both cases avoids a pre-classification step,
# and the "state that plainly" instruction is what produces the honest
# "this is a decorative logo" answers we need in order to REPORT that this
# document contains no analytical figures.
FIGURE_CAPTION_PROMPT = """You are indexing images extracted from an SEC quarterly report (Form 10-Q).

Describe this image factually in 30-60 words.

If it is a chart or graph: state the chart type, what is plotted on each axis,
the series shown, and the overall trend. Report any axis labels, legend text or
data labels you can read.

If it is a logo, signature, decorative rule, or other non-informational graphic:
say so plainly in one sentence and do not invent detail.

Description:"""


def _summarize_table(element: Element, provider: LLMProvider) -> str:
    """
    Produce the search key for one table.

    WHY we truncate the table before sending it: a handful of tables in a 10-Q
    (the marketable-securities breakdown on p.11) run to dozens of rows. The
    summariser does not need every row to describe the table's SUBJECT -- the
    header plus the first rows establish it. Truncating bounds token cost and
    latency, and the full table is still what gets returned at answer time, so
    nothing is lost where it matters.
    """
    table_md = element.content
    if len(table_md) > 4000:
        table_md = table_md[:4000] + "\n| ... (truncated for summarisation only) |"

    return provider.complete(TABLE_SUMMARY_PROMPT.format(table=table_md)).strip()


def _caption_figure(element: Element, provider: LLMProvider) -> str:
    """
    Produce both the content AND the search key for one figure.

    WHY the caption serves as both: unlike a table, a figure has no extractable
    text. The vision model's description IS the only textual representation that
    exists, so it must play both roles -- the thing we embed and the thing the
    answering model reads.

    WHY this returns "" on failure rather than raising: one unreadable image
    should degrade to a missing element, not abort ingestion of the whole
    document. The caller drops empty captions and logs the loss.
    """
    # WHY resolve against PROJECT_ROOT: the parser stores a project-relative
    # path so the docstore stays portable. Resolving here means this works
    # regardless of the current working directory -- which Streamlit changes.
    image_path = PROJECT_ROOT / element.metadata["image_path"]
    if not image_path.exists():
        log.warning("Figure file missing, skipping: %s", image_path)
        return ""

    try:
        return provider.describe_image(image_path, FIGURE_CAPTION_PROMPT).strip()
    except Exception as exc:
        log.warning("Vision call failed for %s: %s", image_path.name, exc)
        return ""


def enrich(
    elements: list[Element], provider: LLMProvider, cfg: AppConfig
) -> list[Element]:
    """
    Fill in `summary` for tables and `content`+`summary` for figures.

    Mutates and returns the same list.

    WHY MUTATE IN PLACE rather than building a new list: `Element.id` is derived
    from `kind|page|content`. Tables keep their content unchanged, so their IDs
    are stable across enrichment -- which is what makes re-running ingestion
    idempotent. (Figures DO change ID here, because their content goes from ""
    to the caption. That is correct: before captioning a figure element is not
    yet a meaningful indexable unit.)

    WHY a sequential loop and not concurrency: free-tier rate limits are the
    binding constraint, not latency. Parallel requests would trigger 429s and,
    after backoff, finish no sooner. ~30 sequential calls is roughly a minute --
    acceptable, and the code stays simple enough to reason about under deadline.
    """
    tables = [e for e in elements if e.kind is ElementKind.TABLE]
    figures = [e for e in elements if e.kind is ElementKind.FIGURE]

    log.info("Enriching %d tables and %d figures", len(tables), len(figures))

    for i, element in enumerate(tables, start=1):
        try:
            element.summary = _summarize_table(element, provider)
            # WHY INFO and not DEBUG: enrichment is the slowest stage (one API
            # call per table) and produces no other output. At DEBUG the run
            # looked frozen for two minutes, which is indistinguishable from a
            # hang -- and during this build an actual hang WAS misread as slow
            # progress. A per-item counter makes the difference visible.
            log.info("summarised table %d/%d (p.%d)", i, len(tables), element.page)
        except Exception as exc:
            # WHY we fall back to a deterministic, non-LLM summary instead of
            # failing: a table with a mediocre search key is far better than a
            # table that is absent from the index. The first row of a financial
            # table is its header, which is a genuinely usable (if weaker) key.
            header = element.content.split("\n", 1)[0]
            element.summary = f"Financial table on page {element.page}. Columns: {header}"
            log.warning("Summarisation failed for table on p.%d (%s); using header fallback",
                        element.page, exc)

    captioned = 0
    for i, element in enumerate(figures, start=1):
        caption = _caption_figure(element, provider)
        if caption:
            element.content = caption
            element.summary = caption
            captioned += 1
            log.info("captioned figure %d/%d (p.%d)", i, len(figures), element.page)

    # WHY we DROP figures that produced no caption: an element whose content and
    # summary are both empty embeds to a meaningless vector and can still be
    # returned by a similarity search, injecting an empty context block into the
    # prompt. Better to have no element than a blank one.
    before = len(elements)
    elements = [
        e for e in elements if not (e.kind is ElementKind.FIGURE and not e.content)
    ]
    if dropped := before - len(elements):
        log.info("Dropped %d figure(s) that could not be captioned", dropped)

    log.info("Enrichment complete: %d tables summarised, %d figures captioned",
             len(tables), captioned)
    return elements
