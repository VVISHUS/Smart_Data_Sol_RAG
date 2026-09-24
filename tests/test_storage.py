"""
Offline tests for the parts that have no business calling an API.

    python -m tests.test_storage          (no pytest needed)
    pytest tests/                         (also works)

WHY THESE TESTS AND NOT OTHERS
------------------------------
The expensive, slow, non-deterministic parts of this system (embedding,
summarisation, answering) are covered by the evaluation harness, which measures
end-to-end quality against real ground truth. Duplicating that with mocks would
test the mocks.

What the eval CANNOT catch is a silent data-layer bug: a metadata key dropped on
the way into SQLite, a ragged table normalised wrongly, or an element id that
changes between runs and quietly doubles the index. Those produce no error and
no obviously wrong answer - they just make the system a bit worse. They are also
fast and deterministic to test, which is exactly the trade that makes a test
worth writing.

WHY PLAIN ASSERTS AND A __main__ RUNNER: adding pytest to requirements.txt to
run six assertions is a dependency a reviewer has to install for no gain. These
run with the standard library, and pytest will still collect them if it is
present.
"""

from __future__ import annotations

import sqlite3

from utils.vector_store import (
    _BOOL_COLUMNS,
    _METADATA_COLUMNS,
    _SCHEMA,
    MultiVectorStore,
)
from utils.elements import Element, ElementKind
from utils.pdf_parser import _normalise_table_grid, _table_to_markdown


# ---------------------------------------------------------------------------
# storage round-trip
# ---------------------------------------------------------------------------
def _in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _write(conn: sqlite3.Connection, elements: list[Element]) -> None:
    columns = ("id", "kind", "page", "content", "summary") + _METADATA_COLUMNS
    placeholders = ",".join("?" * len(columns))
    conn.executemany(
        f"INSERT OR REPLACE INTO elements ({','.join(columns)}) "
        f"VALUES ({placeholders})",
        [
            (e.id, e.kind.value, e.page, e.content, e.summary)
            + tuple(e.metadata.get(c) for c in _METADATA_COLUMNS)
            for e in elements
        ],
    )
    conn.commit()


SAMPLES = [
    Element(ElementKind.TABLE, 4, "| Products | 63,355 |", "net sales by product",
            {"label": "table 1", "n_rows": 28, "n_cols": 5,
             "has_header_context": True}),
    Element(ElementKind.FIGURE, 1, "an Apple logo", "an Apple logo",
            {"label": "embedded image 5", "image_path": "data/processed/figures/p01_x5.png",
             "width": 46, "height": 56, "xref": 5, "filter_reason": "46x56px"}),
    Element(ElementKind.TEXT, 9, "Japan net sales decreased...", None,
            {"label": "narrative"}),
]


def test_metadata_round_trip_is_lossless():
    """
    Every metadata key must survive the trip through typed columns.

    WHY THIS IS THE MOST IMPORTANT TEST HERE: metadata was a JSON blob until
    recently. Normalising it into nine nullable columns is strictly better, but
    it introduces a way to lose data that JSON did not have - forget to list a
    key in _METADATA_COLUMNS and it silently vanishes on write, with no error
    anywhere. This asserts the exact dict comes back.
    """
    conn = _in_memory_db()
    _write(conn, SAMPLES)
    for original in SAMPLES:
        row = conn.execute("SELECT * FROM elements WHERE id=?", (original.id,)).fetchone()
        restored = MultiVectorStore._row_to_element(row)
        assert restored.kind == original.kind
        assert restored.page == original.page
        assert restored.content == original.content
        assert restored.summary == original.summary
        assert restored.metadata == original.metadata, (
            f"metadata changed for {original.kind.value}: "
            f"{original.metadata} -> {restored.metadata}"
        )
    conn.close()


def test_booleans_survive_as_booleans():
    """
    SQLite has no BOOLEAN type, so bools are stored as 0/1.

    Without the explicit cast in _row_to_element, `has_header_context` would come
    back as the integer 1. That compares truthy, so nothing would appear broken -
    it would just quietly change type between a fresh parse and a reload.
    """
    conn = _in_memory_db()
    _write(conn, SAMPLES)
    row = conn.execute("SELECT * FROM elements WHERE kind='table'").fetchone()
    restored = MultiVectorStore._row_to_element(row)
    assert isinstance(restored.metadata["has_header_context"], bool)
    conn.close()


def test_absent_metadata_keys_stay_absent():
    """
    A text element has no `image_path`. It must not come back as None.

    WHY: callers use `metadata.get("image_path")` and treat presence as meaning
    "this element has an image". A NULL column resurfacing as an explicit
    `image_path: None` key would break that contract.
    """
    conn = _in_memory_db()
    _write(conn, SAMPLES)
    row = conn.execute("SELECT * FROM elements WHERE kind='text'").fetchone()
    restored = MultiVectorStore._row_to_element(row)
    assert "image_path" not in restored.metadata
    assert restored.metadata == {"label": "narrative"}
    conn.close()


def test_upsert_does_not_duplicate():
    """
    Re-ingesting unchanged content must overwrite, not accumulate.

    This is what content-addressed ids buy us. If ids were uuid4, this test
    would fail with 6 rows and the index would double on every run.
    """
    conn = _in_memory_db()
    _write(conn, SAMPLES)
    _write(conn, SAMPLES)
    assert conn.execute("SELECT COUNT(*) FROM elements").fetchone()[0] == len(SAMPLES)
    conn.close()


def test_element_id_is_stable_and_content_sensitive():
    """Same content -> same id. Changed content -> different id."""
    a = Element(ElementKind.TABLE, 4, "| x |", "summary A")
    b = Element(ElementKind.TABLE, 4, "| x |", "summary B")
    c = Element(ElementKind.TABLE, 4, "| y |", "summary A")
    # WHY the summary is deliberately NOT part of the id: re-running enrichment
    # produces slightly different wording each time. If the summary fed the hash,
    # every re-run would create new ids for unchanged tables and the index would
    # grow without bound.
    assert a.id == b.id
    assert a.id != c.id


# ---------------------------------------------------------------------------
# table repair
# ---------------------------------------------------------------------------
def test_ragged_currency_rows_are_realigned():
    """
    The defect this function exists for, reduced to its smallest form.

    pdfplumber puts "$" in its own cell only on rows that print one, so column 1
    holds "$" in the first row and a real value in the second. Both rows must
    come out as the same five aligned fields.
    """
    rows = [
        ["Products", "$", "63,355", "$", "63,948"],
        ["Services", "19,604", "", "17,486", ""],
    ]
    out = _normalise_table_grid(rows)
    assert out[0] == ["Products", "63,355", "63,948"]
    assert out[1] == ["Services", "19,604", "17,486"]
    assert len(out[0]) == len(out[1]), "rows must share a column count"


def test_stranded_percent_merges_into_its_value():
    """`3` followed by a lone `%` must become `3%`, not two cells and not `3`."""
    rows = [["iPhone", "40,665", "39,570", "3", "%"],
            ["Mac", "7,382", "8,235", "(10)", "%"]]
    out = _normalise_table_grid(rows)
    assert out[0][-1] == "3%"
    assert out[1][-1] == "(10)%", "negative percentages must keep their brackets"


def test_markdown_table_is_rectangular():
    """Every rendered row must have the same number of pipes as the header."""
    md = _table_to_markdown([["a", "b", "c"], ["1", "2"], ["1", "2", "3", "4"]])
    widths = {line.count("|") for line in md.splitlines()}
    assert len(widths) == 1, f"ragged markdown produced: {md}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
