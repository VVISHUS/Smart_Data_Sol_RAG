"""
The single data structure that flows through the entire pipeline.

WHY ONE TYPE FOR THREE MODALITIES
---------------------------------
The brief demands answers over text, tables and figures. The obvious design is
three parallel pipelines -- three parsers, three indexes, three retrievers, and
a router that guesses which to query. We deliberately rejected that (see
see the retriever). It triples the code, and the router becomes a new and
unreliable failure point: "how many iPhones were sold" is a table question,
"how did iPhone sales change" is arguably both, and a misroute yields a
confidently wrong answer with no recovery path.

Instead every modality is normalised into one `Element` with two text faces:

    .content  -- the VERBATIM payload. Prose, or a Markdown-rendered table, or
                 a vision model's description of a figure. This is what the
                 answering LLM reads. For tables it preserves every digit
                 exactly, which is the whole reason the system can be trusted
                 with financial figures.

    .summary  -- a short natural-language restatement. This is what gets
                 EMBEDDED and searched.

WHY THE SPLIT (the core insight of this design)
-----------------------------------------------
Embedding a raw financial table is close to useless. Its text is "iPhone 40,665
39,570 162,863 153,105" -- almost no semantic signal, and a user asking "how did
iPhone revenue compare to last year?" shares no vocabulary with it, so cosine
similarity fails to retrieve it.

But embedding a *summary* -- "Quarterly and nine-month net sales broken out by
product line: iPhone, Mac, iPad, Wearables, Services, with year-over-year
comparison" -- matches that question strongly.

So: search the summary, return the raw. Retrieval gets semantic richness;
generation gets exact numbers. Neither is compromised. This is the
multi-vector / "summary-index" retriever pattern, and it is what lets a single
uniform pipeline serve all three required modalities.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ElementKind(str, Enum):
    """
    WHY a str-Enum rather than bare strings: the kind is used for filtering
    (`retrieval.min_tables_in_context`), for prompt shaping, and is written to
    the vector store's metadata. A typo'd literal "tabel" would silently create
    a category that never matches any filter. Subclassing `str` keeps it
    directly JSON-serialisable, so persistence needs no custom encoder.
    """

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


@dataclass
class Element:
    """One retrievable unit of the source document."""

    kind: ElementKind
    page: int  # 1-indexed, matching what a human sees in a PDF reader
    content: str
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """
        A content-addressed, deterministic identifier.

        WHY content-addressed (hash of kind+page+content) rather than a running
        counter or uuid4():

        1. IDEMPOTENCY. Re-running ingestion on an unchanged PDF produces the
           same IDs, so re-indexing overwrites rather than duplicates. With
           uuid4 every run would silently double the size of the vector store
           and skew retrieval toward whichever content was indexed most often.
        2. CHANGE DETECTION. If the ID changed, the content changed. That makes
           incremental re-indexing possible later without a diffing layer.
        3. DEBUGGABILITY. A given table always has the same ID across runs, so
           an eval failure logged yesterday is still traceable today.

        WHY 16 hex chars: 64 bits. For a few hundred elements the collision
        probability is vanishingly small, and short IDs stay readable in logs.
        """
        digest = hashlib.sha256(
            f"{self.kind.value}|{self.page}|{self.content}".encode("utf-8")
        ).hexdigest()
        return digest[:16]

    @property
    def embedding_text(self) -> str:
        """
        The text actually handed to the embedding model.

        WHY prefer summary and fall back to content: figures and tables are only
        findable via their summary (see the module docstring). Plain text
        elements need no summary -- their content is already natural language,
        and summarising it would *lose* the specific wording a user might quote.
        The fallback also keeps the system functional if summarisation was
        skipped or its API call failed, degrading quality instead of crashing.
        """
        return self.summary if self.summary else self.content

    def citation(self) -> str:
        """
        Human-readable provenance, e.g. "table on p.10" or "figure on p.1".

        WHY every answer must carry this: an unsourced number from an LLM over a
        financial filing is unusable -- the reader cannot tell recall from
        hallucination. Forcing a citation into the answer makes the system
        auditable, and makes its errors visible rather than plausible.
        """
        label = self.metadata.get("label")
        base = f"{self.kind.value} on p.{self.page}"
        return f"{base} ({label})" if label else base

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage and for the JSON export. Enum -> str."""
        d = asdict(self)
        d["kind"] = self.kind.value
        d["id"] = self.id  # WHY store it: it is a property, so asdict() omits
        # it, but the docstore is keyed by it and a reader of the export
        # should not have to recompute a hash to understand the file.
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Element":
        """Rehydrate from the docstore, tolerating the extra derived `id` key."""
        return cls(
            kind=ElementKind(d["kind"]),
            page=d["page"],
            content=d["content"],
            summary=d.get("summary"),
            metadata=d.get("metadata", {}),
        )
