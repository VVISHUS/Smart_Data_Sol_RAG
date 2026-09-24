"""
The multi-vector index: embedded summaries in Chroma, verbatim content in SQLite.

THE CENTRAL MECHANISM OF THIS PROJECT
-------------------------------------
Two stores, joined by a single id:

  Chroma (vector store)   holds ONE embedding per element, computed from
                          `element.embedding_text` - the SUMMARY for tables and
                          figures, the raw prose for text. This is what is
                          SEARCHED. A query returns ids.

  SQLite (docstore)       holds the FULL element, keyed by that same id. This is
                          what is FETCHED and handed to the answering model.

The flow is: question -> vector -> Chroma -> ids -> SQLite -> verbatim tables.
The id is the join key between the two stores, exactly like a foreign key.

A search therefore matches against a semantically rich description and returns a
verbatim financial table. Retrieval quality and answer fidelity are optimised
independently instead of being traded off against each other.

WHY SQLITE AND NOT A JSON FILE
------------------------------
The first version of this module kept the docstore in a JSON file. It worked,
and at 59 elements nothing was visibly wrong, but every scaling property was
against it:

  * it loaded the ENTIRE file into RAM at startup, whether one element was
    needed or all of them;
  * saving a single element rewrote the WHOLE file - re-ingesting one changed
    page rewrote all 108 KB, and at 100k elements that is hundreds of megabytes
    rewritten to change one row;
  * two processes writing at once would silently lose data;
  * a crash mid-write left a truncated, unparseable file with no recovery.

SQLite fixes all four, is in the standard library (no new dependency), and costs
about the same number of lines. The only thing JSON was better at was being
readable in an editor, and `export_json()` below keeps that.

WHY METADATA IS REAL COLUMNS AND NOT A JSON BLOB
------------------------------------------------
An intermediate version stored metadata as a JSON text column, on the argument
that it is heterogeneous by kind: figures carry image_path/width/height, tables
carry n_rows/n_cols. That argument does not survive contact with the data. The
parser emits exactly nine metadata keys, all scalar (str, int, bool), and the
set is stable.

So they are nine nullable columns instead. What that buys:

  * the database enforces types, rather than trusting whatever was serialised;
  * they are queryable and indexable - `WHERE n_rows > 10` or
    `WHERE image_path IS NOT NULL` work directly, with no json_extract();
  * no parse step on every read;
  * a malformed value fails at write time, not silently at read time.

The cost is that adding a tenth metadata key needs an ALTER TABLE. For a fixed,
known set that is the right trade - the schema documenting itself is a feature,
not overhead.

`Element.metadata` stays a plain dict, so no caller changed. The normalisation
happens only at the storage boundary: dict spread into columns on write, NULLs
dropped back into a dict on read.

WHY WE PASS OUR OWN EMBEDDINGS TO CHROMA
----------------------------------------
Chroma's default embedding function downloads and runs a local
sentence-transformers model. That would mean the index is built with one
embedding model and queried through our provider with another - vectors from
different models are not comparable, and retrieval would be effectively random.
Supplying embeddings explicitly makes the model choice single and visible.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import chromadb

from utils.config import AppConfig
from utils.elements import Element, ElementKind
from utils.provider import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)

_COLLECTION_NAME = "filing_elements"

# WHY `id` is the PRIMARY KEY and not an autoincrement integer: the id is the
# content hash Chroma also stores, so it is the join key between the two stores.
# Making it the primary key gives us the index we need for free, and makes
# INSERT OR REPLACE the natural upsert.
#
# WHY the two secondary indexes: `kind` is filtered on for modality-restricted
# searches and for the inspect command; `page` is used for reporting. Both are
# cheap at this size and prevent full scans as the table grows.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS elements (
    id                 TEXT PRIMARY KEY,
    kind               TEXT    NOT NULL,
    page               INTEGER NOT NULL,
    content            TEXT    NOT NULL,
    summary            TEXT,

    -- shared metadata
    label              TEXT,

    -- table-only metadata
    n_rows             INTEGER,
    n_cols             INTEGER,
    has_header_context INTEGER,

    -- figure-only metadata
    image_path         TEXT,
    width              INTEGER,
    height             INTEGER,
    xref               INTEGER,
    filter_reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_elements_kind ON elements(kind);
CREATE INDEX IF NOT EXISTS idx_elements_page ON elements(page);
"""

# WHY a declared list rather than reading the table schema at runtime: it keeps
# the write path, the read path and the CREATE TABLE above in one obvious
# correspondence. If a key is added to the parser without being added here, the
# round-trip test in the smoke check fails loudly instead of dropping it.
#
# WHY `has_header_context` is an INTEGER: SQLite has no BOOLEAN type. Python
# bools round-trip through it as 0/1, and the read path below restores the bool.
_METADATA_COLUMNS = (
    "label",
    "n_rows",
    "n_cols",
    "has_header_context",
    "image_path",
    "width",
    "height",
    "xref",
    "filter_reason",
)

_BOOL_COLUMNS = frozenset({"has_header_context"})


class MultiVectorStore:
    """Owns both stores and keeps them consistent."""

    def __init__(self, cfg: AppConfig, provider: LLMProvider):
        self.cfg = cfg
        self.provider = provider

        self._persist_dir = cfg.paths.absolute("vector_store_dir")
        self._db_path = cfg.paths.absolute("docstore_path")

        # WHY PersistentClient and not the in-memory Client: ingestion and
        # querying are separate processes (`cli.py ingest` then `cli.py ask`).
        # An in-memory index would vanish between them, forcing re-ingestion -
        # and re-paying every embedding call - on every question.
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))

        self._db = self._connect()

    # -----------------------------------------------------------------------
    @property
    def _collection(self):
        """
        Fetch the collection on every access rather than holding a handle.

        WHY THIS IS A PROPERTY AND NOT AN ATTRIBUTE SET IN __init__
        A Chroma collection handle is bound to a specific collection UUID. When
        another process rebuilds the index, `delete_collection` retires that
        UUID, and every later call on the old handle raises
        `NotFoundError: Collection [uuid] does not exist`.

        That is not hypothetical. The Streamlit app caches this store for the
        life of the server (`@st.cache_resource`), so a single
        `ingest --rebuild` from a terminal left the running UI permanently
        broken, reporting an empty index while the index was in fact fine. The
        user has no way to connect those two events.

        Re-fetching is a local SQLite lookup, so the cost is negligible next to
        an embedding call, and it makes the store self-healing across external
        rebuilds.

        WHY cosine and not Chroma's default L2: our embeddings are not
        normalised to unit length, and with L2 a longer document's larger
        vector magnitude biases the distance independently of its meaning.
        Cosine compares direction only, which is what "semantic similarity"
        means for text embeddings.
        """
        return self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # -----------------------------------------------------------------------
    # SQLite plumbing
    # -----------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """
        Open the docstore, creating the schema on first use.

        WHY `check_same_thread=False`: Streamlit runs script reruns on worker
        threads, and sqlite3 otherwise refuses a connection created on another
        thread. We only issue short reads and one batched write per ingestion,
        so there is no concurrent-use hazard here. A served multi-user version
        would use a connection pool instead.

        WHY WAL mode: it lets a reader (an `ask`) proceed while a writer (an
        `ingest`) holds the database, instead of failing with "database is
        locked". One line that removes a whole class of operational annoyance.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        # WHY Row: lets us read columns by name, so adding a column later does
        # not silently shift positional unpacking in _row_to_element.
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    @staticmethod
    def _row_to_element(row: sqlite3.Row) -> Element:
        """
        Rebuild an Element from a row, reassembling the metadata dict.

        WHY NULLs are dropped rather than kept as None: a table element has no
        `image_path`, and carrying `image_path: None` through the rest of the
        system would make every consumer test for it. Omitting the key restores
        exactly the dict the parser produced, so the round-trip is lossless and
        `metadata.get("label")` behaves identically before and after storage.
        """
        metadata = {}
        for column in _METADATA_COLUMNS:
            value = row[column]
            if value is None:
                continue
            metadata[column] = bool(value) if column in _BOOL_COLUMNS else value

        return Element(
            kind=ElementKind(row["kind"]),
            page=row["page"],
            content=row["content"],
            summary=row["summary"],
            metadata=metadata,
        )

    def reset(self) -> None:
        """
        Delete both stores so the next build starts clean.

        WHY this is offered explicitly rather than relying on upsert: changing an
        extraction or filtering parameter changes which elements exist, and stale
        elements from a previous configuration would linger in the index
        forever, polluting results in ways that are extremely hard to diagnose.
        An explicit `--rebuild` is the honest way to re-run an experiment.
        """
        try:
            self._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            # WHY tolerate failure: on a first run the collection does not
            # exist. That is not an error condition for a reset.
            pass
        # No need to recreate it here: `_collection` is a property, so the next
        # access creates it.

        # WHY DELETE and not DROP TABLE: keeping the table means the schema is
        # created in exactly one place (_connect), rather than in two that could
        # drift apart.
        self._db.execute("DELETE FROM elements")
        self._db.commit()

        figures = self.cfg.paths.absolute("figures_dir")
        if figures.exists():
            shutil.rmtree(figures)
        log.info("Index reset")

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------
    def add(self, elements: list[Element], batch_size: int = 50) -> None:
        """
        Embed and index a list of elements.

        WHY BATCHED: embedding APIs charge and rate-limit per request, not per
        token. Sending 60 texts in two requests instead of 60 keeps us far below
        the free-tier request ceiling and cuts wall-clock time by roughly the
        batch factor.

        WHY `upsert` / `INSERT OR REPLACE`: combined with content-addressed IDs
        (see Element.id), re-running ingestion over an unchanged document
        overwrites identical records instead of creating duplicates. Without
        this, every re-run would inflate the index and skew similarity toward
        whatever was indexed most often.
        """
        if not elements:
            log.warning("Nothing to index")
            return

        for start in range(0, len(elements), batch_size):
            batch = elements[start : start + batch_size]
            texts = [e.embedding_text for e in batch]

            vectors = self.provider.embed(texts)

            self._collection.upsert(
                ids=[e.id for e in batch],
                embeddings=vectors,
                documents=texts,  # WHY store the embedded text: it makes the
                                  # Chroma record self-explanatory when
                                  # debugging why something did or did not match.
                metadatas=[
                    {
                        # WHY duplicate `page`/`kind` into Chroma metadata when
                        # SQLite already has them: metadata filtering happens
                        # INSIDE the vector search. Fetching from SQLite first to
                        # filter would defeat the index.
                        "kind": e.kind.value,
                        "page": e.page,
                        "label": str(e.metadata.get("label", "")),
                    }
                    for e in batch
                ],
            )

            # WHY executemany in one transaction: 50 separate commits would each
            # fsync. One commit per batch is a single disk flush, which is the
            # difference between milliseconds and seconds.
            columns = ("id", "kind", "page", "content", "summary") + _METADATA_COLUMNS
            placeholders = ",".join("?" * len(columns))
            self._db.executemany(
                f"INSERT OR REPLACE INTO elements ({','.join(columns)}) "
                f"VALUES ({placeholders})",
                [
                    (e.id, e.kind.value, e.page, e.content, e.summary)
                    # WHY `.get(col)` rather than indexing: a table element has no
                    # image_path and a figure has no n_rows. Missing keys become
                    # NULL, which is exactly what a nullable column is for.
                    + tuple(e.metadata.get(col) for col in _METADATA_COLUMNS)
                    for e in batch
                ],
            )
            self._db.commit()

            log.info("Indexed %d/%d elements", min(start + batch_size, len(elements)),
                     len(elements))

    # -----------------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------------
    def search(
        self, query: str, k: int, kind: ElementKind | None = None
    ) -> list[tuple[Element, float]]:
        """
        Return the k nearest elements as (element, distance) pairs.

        `kind` optionally restricts the search to one modality - used by the
        retriever to guarantee table coverage for numeric questions.

        WHY `embed_query` and not `embed`: task-aware embedding models encode
        queries and documents differently on purpose (see provider.py). Using
        the document path for a query silently costs recall.

        WHY we return the DISTANCE alongside the element: the retriever needs it
        to merge two result sets sensibly, and surfacing it in the CLI makes it
        possible to see *how* confident a match was rather than guessing.
        """
        if self._collection.count() == 0:
            # WHY the message distinguishes the two cases: "the index is empty"
            # was actively misleading when the docstore held 59 rows and only
            # the vector side had been cleared. Reporting both counts turns a
            # dead end into a diagnosis.
            raise RuntimeError(
                f"No vectors in the index (docstore holds {self.docstore_size} "
                "rows). Run `python -m app.cli ingest` to build it, or "
                "`--rebuild` if the two have drifted apart."
            )

        vector = self.provider.embed_query(query)

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            where={"kind": kind.value} if kind else None,
        )

        # WHY the [0] everywhere: Chroma's query API is batched - it accepts N
        # query vectors and returns N result lists. We always send one, so we
        # always unwrap the first.
        ids = result["ids"][0]
        distances = result["distances"][0]
        if not ids:
            return []

        # WHY ONE QUERY WITH `IN` RATHER THAN K SEPARATE LOOKUPS: this is the
        # payoff of moving off JSON. We fetch exactly the k rows the search
        # returned, in a single round-trip, and never touch the rest of the
        # table. The JSON version had to hold every element in memory to do the
        # same thing.
        placeholders = ",".join("?" * len(ids))
        rows = {
            row["id"]: row
            for row in self._db.execute(
                f"SELECT * FROM elements WHERE id IN ({placeholders})", ids
            )
        }

        out: list[tuple[Element, float]] = []
        # WHY iterate `ids` rather than the SQL result: `IN` returns rows in
        # arbitrary order, but ranking is the entire point of a search. Walking
        # the id list preserves Chroma's ordering.
        for element_id, distance in zip(ids, distances):
            row = rows.get(element_id)
            if row is None:
                # WHY this can happen and why we warn rather than crash: the two
                # stores can drift if a run is interrupted between the Chroma
                # upsert and the SQLite commit. Warning names the inconsistency
                # so the fix (`--rebuild`) is obvious.
                log.warning("Vector %s has no docstore row; index may be stale",
                            element_id)
                continue
            out.append((self._row_to_element(row), distance))

        return out

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    def stats(self) -> dict[str, tuple[int, list[int]]]:
        """
        Per-modality counts and the pages each occurs on.

        WHY THIS IS A STORE METHOD rather than the CLI reaching into internals:
        the CLI previously iterated a private in-memory dict, which only worked
        because everything was loaded. Expressing it as SQL keeps the CLI honest
        and does the aggregation where the data lives.
        """
        rows = self._db.execute(
            "SELECT kind, COUNT(*) AS n, GROUP_CONCAT(DISTINCT page) AS pages "
            "FROM elements GROUP BY kind"
        ).fetchall()
        return {
            row["kind"]: (
                row["n"],
                sorted(int(p) for p in (row["pages"] or "").split(",") if p),
            )
            for row in rows
        }

    def export_json(self, path: Path) -> int:
        """
        Dump the whole docstore to a readable JSON file, for human review only.

        NOTHING IN THE PIPELINE READS THIS FILE. It is a one-way export, written
        only when `ingest --export-json` is passed, and deleting it has no effect
        on ingestion, retrieval or answering. This is the only JSON left in the
        project, and it is deliberately outside the data path.

        WHY KEEP IT AT ALL: readability was the single genuine advantage the old
        JSON docstore had - a reviewer could open the file and see exactly what
        the system indexed. Moving to SQLite should not cost that, so the
        capability survives as an explicit action rather than as the storage
        format. (`sqlite3 docstore.sqlite3 "SELECT ..."` works too, but not
        everyone reviewing this will reach for it.)
        """
        payload = {
            row["id"]: self._row_to_element(row).to_dict()
            for row in self._db.execute("SELECT * FROM elements ORDER BY page, kind")
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Exported %d elements to %s", len(payload), path)
        return len(payload)

    @property
    def size(self) -> int:
        return self._collection.count()

    @property
    def docstore_size(self) -> int:
        """
        WHY EXPOSED SEPARATELY FROM `size`: if these two ever disagree, the two
        stores have drifted apart. Having both makes that visible rather than
        latent.
        """
        return self._db.execute("SELECT COUNT(*) FROM elements").fetchone()[0]

    def close(self) -> None:
        self._db.close()
