"""
Retrieval with guaranteed table coverage.

THE PROBLEM THIS SOLVES
-----------------------
Plain top-k similarity search is not sufficient for this document, for a reason
specific to financial filings: the narrative sections *discuss* the same topics
the tables *quantify*, using richer prose. So for a question like

    "What were iPhone net sales in the third quarter of 2022?"

the MD&A paragraph -- "iPhone net sales increased during the third quarter of
2022 compared to the same period in 2021 due primarily to higher net sales from
the Company's new iPhone models" -- is an excellent semantic match. It is far
closer in wording to the question than any table summary. So the top-k fills
with narrative, the table never makes the context window, and the model answers
"iPhone net sales increased" -- fluent, relevant, sourced, and useless, because
the user asked for a number that was never retrieved.

THE FIX
-------
Run a second, modality-restricted search and reserve slots for tables. This
guarantees that if a table relevant to the query exists, it reaches the
answering model -- while still letting narrative chunks win the remaining slots
on pure similarity.

WHY NOT AN LLM ROUTER INSTEAD
-----------------------------
The textbook alternative is to classify the question ("is this numeric?") and
route to one modality. We rejected that:
  * it adds a full LLM round-trip of latency to every question;
  * it introduces a new failure mode -- a misclassification produces a
    confidently wrong answer with no recovery path;
  * many real questions are genuinely BOTH ("how did iPhone revenue change and
    why?") and a router must pick one.
Reserving slots is cheaper, has no classification step to get wrong, and
degrades gracefully: at worst we spend two extra context slots on a table that
was not needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.config import AppConfig
from utils.vector_store import MultiVectorStore
from utils.elements import Element, ElementKind
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    """
    Retrieved context plus the diagnostics needed to explain it.

    WHY carry `distances` and `strategy` rather than returning a bare list: the
    evaluation harness and the CLI's --verbose mode both need to show WHY a
    given answer was produced. A retrieval system you cannot inspect is one you
    cannot debug or defend in a write-up.
    """

    elements: list[Element]
    distances: list[float]
    strategy: str

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.elements:
            counts[e.kind.value] = counts.get(e.kind.value, 0) + 1
        return counts


class Retriever:
    def __init__(self, store: MultiVectorStore, cfg: AppConfig):
        self.store = store
        self.cfg = cfg

    def retrieve(self, question: str) -> RetrievalResult:
        """
        Fetch context for a question, topping up with tables if under-represented.
        """
        k = self.cfg.retrieval.top_k
        min_tables = self.cfg.retrieval.min_tables_in_context

        # --- pass 1: unrestricted similarity ---------------------------------
        primary = self.store.search(question, k=k)

        # WHY we track seen IDs explicitly: the same element can legitimately be
        # returned by both passes. Without deduplication it would occupy two
        # context slots and be double-weighted in the model's attention.
        seen = {e.id for e, _ in primary}
        tables_present = sum(1 for e, _ in primary if e.kind is ElementKind.TABLE)

        strategy = "similarity"
        topped_up: list[tuple[Element, float]] = []

        # --- pass 2: reserved table slots ------------------------------------
        if tables_present < min_tables:
            shortfall = min_tables - tables_present
            # WHY request more than the shortfall: some of the top table hits may
            # already be in `primary` and get deduplicated away. Over-fetching by
            # the shortfall again makes it very likely we still net the slots we
            # need, at no extra API cost (the query embedding is already computed
            # inside search(), and Chroma lookups are local and cheap).
            candidates = self.store.search(
                question, k=shortfall + min_tables, kind=ElementKind.TABLE
            )
            for element, distance in candidates:
                if element.id in seen:
                    continue
                topped_up.append((element, distance))
                seen.add(element.id)
                if len(topped_up) >= shortfall:
                    break

            if topped_up:
                strategy = f"similarity + {len(topped_up)} reserved table slot(s)"
                log.info(
                    "Only %d table(s) in top-%d; added %d via reserved slots",
                    tables_present, k, len(topped_up),
                )

        # --- pass 3: reserved figure slot for questions ABOUT figures ---------
        # WHY THIS IS NEEDED, AND WHY PLAIN SIMILARITY CANNOT COVER IT
        # A figure is indexed by its vision caption -- here, "A solid black
        # silhouette of the iconic Apple logo set against a plain white
        # background." A user asking "what charts or figures are in this
        # document?" shares almost no vocabulary with that sentence, because the
        # question is ABOUT the document's composition while the caption
        # describes the image's content. Similarity search therefore never
        # returns it, and the system answers "the excerpts do not contain this
        # information" -- which is wrong, since it does have a figure indexed.
        #
        # WHY A KEYWORD TEST AND NOT AN LLM CLASSIFIER: we rejected a routing
        # model because a misroute silently produces a wrong answer, with no
        # way to recover. This is not that. It never *replaces* the similarity results,
        # only appends to them, so a false positive costs one context slot and a
        # false negative leaves behaviour exactly as it was. A transparent word
        # list is auditable; a classifier's judgement is not.
        figure_terms = (
            "figure", "figures", "chart", "charts", "graph", "graphs",
            "image", "images", "diagram", "diagrams", "picture", "pictures",
            "illustration", "visual", "visuals", "plot", "plots",
        )
        asks_about_figures = any(
            term in question.casefold().split() or term in question.casefold()
            for term in figure_terms
        )
        if asks_about_figures:
            for element, distance in self.store.search(
                question, k=2, kind=ElementKind.FIGURE
            ):
                if element.id not in seen:
                    topped_up.append((element, distance))
                    seen.add(element.id)
                    strategy += " + figure slot"
                    log.info("Question mentions figures; reserved a figure slot")
                    break

        combined = primary + topped_up

        # WHY sort by distance at the end: the two passes produce independently
        # ranked lists. Presenting them concatenated would put a weak table above
        # a strong narrative chunk purely because of which pass found it. LLMs
        # weight earlier context more heavily, so ordering by true similarity
        # puts the best evidence where the model reads it most attentively.
        combined.sort(key=lambda pair: pair[1])

        elements = [e for e, _ in combined]
        distances = [d for _, d in combined]

        log.info(
            "Retrieved %d elements (%s) via %s",
            len(elements),
            ", ".join(f"{k_}={v}" for k_, v in
                      RetrievalResult(elements, distances, strategy).kind_counts().items()),
            strategy,
        )

        return RetrievalResult(elements=elements, distances=distances, strategy=strategy)
