"""
Wiring: assemble the stages into the two operations the system supports.

WHY A SEPARATE ORCHESTRATION MODULE
-----------------------------------
Every stage so far (parse, enrich, index, retrieve, generate) depends only on
its inputs and knows nothing about the others. That is what makes each one
testable in isolation -- but something has to connect them, and if that wiring
lives in the CLI then the Streamlit app and the eval harness must each duplicate
it, and the three will drift apart.

Putting the wiring here means all three entrypoints are thin and provably
consistent: they all call the same two functions below.
"""

from __future__ import annotations

from utils.config import AppConfig
from utils.summarizer import enrich
from utils.qa_chain import QAChain
from utils.vector_store import MultiVectorStore
from utils.pdf_parser import parse_pdf
from utils.provider import get_provider
from utils.logger import get_logger
from utils.retriever import Retriever

log = get_logger(__name__)


def build_index(cfg: AppConfig, rebuild: bool = False) -> MultiVectorStore:
    """
    Run the full ingestion pipeline: parse -> enrich -> embed -> persist.

    WHY THE STAGE ORDER IS FIXED AND NOT CONFIGURABLE:
      parse   must precede enrich  -- enrich needs the extracted tables/images;
      enrich  must precede index   -- indexing embeds `embedding_text`, which for
                                      tables and figures IS the summary. Indexing
                                      first would embed empty strings for every
                                      figure and raw digit-soup for every table,
                                      silently producing a useless index that
                                      still "works".

    WHY `rebuild` is an explicit opt-in: re-embedding costs API calls and time.
    The common case (asking another question) should never trigger it. But any
    change to extraction or filtering parameters REQUIRES it, because stale
    elements from the old configuration would otherwise persist in the index
    forever. Making it a visible flag keeps that trade-off in the operator's
    hands.
    """
    provider = get_provider(cfg)
    store = MultiVectorStore(cfg, provider)

    if rebuild:
        store.reset()
    elif store.size > 0:
        # WHY short-circuit: ingestion is expensive and idempotent. If an index
        # already exists, silently rebuilding it on every invocation would make
        # the CLI feel broken and burn quota for no gain.
        log.info("Index already contains %d elements; skipping ingestion "
                 "(pass --rebuild to force)", store.size)
        return store

    elements = parse_pdf(cfg)
    elements = enrich(elements, provider, cfg)
    store.add(elements)

    log.info("Index built: %d elements", store.size)
    return store


def build_qa_chain(cfg: AppConfig) -> QAChain:
    """
    Assemble the query-time stack against an already-built index.

    WHY this does NOT call build_index: querying must be fast and must not
    silently trigger an expensive ingestion run as a side effect. If the index is
    missing, `MultiVectorStore.search` raises with an instruction to run
    `ingest` -- an explicit error beats a surprising five-minute delay.
    """
    provider = get_provider(cfg)
    store = MultiVectorStore(cfg, provider)
    retriever = Retriever(store, cfg)
    return QAChain(retriever, provider, cfg)
