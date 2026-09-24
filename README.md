# RAG over a Financial Filing

Ask questions about Apple's Form 10-Q (quarter ended June 25, 2022) and get
answers grounded in the document, with page citations.

Handles all three content types the document has: **text**, **tables** and
**figures**.

**Result: 19/20 on a fixed evaluation set.**

---

## Run it

```bash
python -m venv venv
venv\Scripts\activate                 # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt

cp .env.example .env                  # add GOOGLE_API_KEY or GEMINI_API_KEY

python -m app.cli ingest              # one-off, a few minutes
python -m app.cli ask "What were iPhone net sales in Q3 2022?"
```

The filing is already in `data/raw/`, so there is nothing to download.

Other commands:

```bash
python -m app.cli inspect                    # what the index contains
python -m app.cli ask "..." --show-context   # see the exact text used
python -m tests.test_storage                 # offline tests, no API key needed
python -m eval.run_eval                      # the 20-question evaluation
streamlit run streamlit_app.py               # browser UI
```

---

## How it works

```
   PDF
    |
    |  pdfplumber reads tables, PyMuPDF reads images,
    |  page text is extracted with table areas removed
    v
 elements  (28 text, 30 tables, 1 figure)
    |
    |  each table gets an LLM-written description
    |  each image gets a vision-model caption
    v
 two stores, joined by one id
    |
    +-- Chroma : the embedding of the DESCRIPTION  -> what we search
    +-- SQLite : the VERBATIM table text           -> what we return
    |
    v
 question -> vector -> Chroma gives ids -> SQLite gives the tables
          -> LLM answers using only those, citing pages
```

### The one idea worth understanding

Embedding a raw financial table does not work. Its text is
`iPhone 40,665 39,570 162,863 153,105` - digits with no meaning attached. Someone
asking *"how did iPhone revenue compare to last year?"* uses none of those words,
so the search never finds it.

So we embed a **description** of each table instead:

> *"Net sales by product category for the three and nine months ended June 25
> 2022 versus June 26 2021, covering iPhone, Mac, iPad, Wearables and Services."*

That matches the question. But we hand the model the **original table**, so the
numbers are exact.

Search on the description, answer from the original. The same trick makes
figures work, since a vision caption plays the same role.

---

## The problem this document poses

Before writing any code, the PDF was measured:

| | |
|---|---|
| Pages | 28 |
| Tables | 30 |
| Images in the whole document | **1** (a 46x56px logo on page 1) |

**There are no charts or graphs.** The figure pipeline is built and runs, and it
reports their absence rather than inventing them.

**The difficulty is the tables.** Every income statement has four value columns:

```
                      Three Months Ended        Nine Months Ended
                 June 25 2022  June 26 2021  June 25 2022  June 26 2021
iPhone                 40,665        39,570       162,863       153,105
```

Ask "what were iPhone net sales in Q3 2022?" and a naive system answers
**$162,863M**. That number is real, it is in the table, and it is wrong by 4x.
Most of the work here exists to stop that.

---

## Four problems found by reading the output

**1. Tables were being indexed twice.** `extract_text()` also returns the table,
flattened into `iPhone $ 40,665 $ 39,570 $ 162,863` with the columns destroyed.
That string is searchable, and when found the model sees digits with no headers
and guesses. Fixed by removing table areas from the page text.

**2. Column headers were missing entirely.** pdfplumber's table box starts at the
first *data* row, so `Three Months Ended / Nine Months Ended` was silently
dropped. Without it there is no way to pick the right column. Fixed by reading
the area just above each table. **All 30 tables now keep their headers.**

**3. Tables came out ragged.** Currency symbols get their own cell, so `$` lands
in column 1 of one row and a real number in column 1 of the next. Fixed by
cleaning each row separately, which realigns the grid. Page 4 went from 11
columns to 5.

**4. Narrative crowded out the tables.** The sentence "iPhone net sales increased
during the third quarter..." matches the question better than any table does, so
tables never reached the model. Fixed by reserving slots in the context for
tables.

---

## Evaluation

20 questions with answers taken straight from the filing, across five types:
text, single-table, multi-table, figure, and questions the document **cannot**
answer.

| Type | Score |
|---|---|
| table | 10/10 |
| table_multi | 3/3 |
| text | 3/4 |
| figure | 1/1 |
| refusal | 2/2 |
| **total** | **19/20** |

Two details that make the test harder than it looks:

- Some questions **forbid** a specific wrong answer. Q1 must contain `82,959`
  and must **not** contain `304,182`, the nine-month figure sitting in the next
  column. Producing both means you cannot tell the periods apart.
- There is a **refusal** category. "How many iPhone units did Apple sell?" is not
  in any 10-Q. A system that never says "I don't know" is not accurate, just
  agreeable.

**The one failure** is q16, "Why did Japan's net sales decrease?". The answer
gives the main cause (lower iPhone and iPad sales, cited to page 19) but leaves
out the weak yen, which the filing also mentions. Correct but incomplete. Kept as
a failure rather than loosened, because one honest failure is worth more than a
perfect score you adjusted the test to get.

Full detail: [`eval/results.md`](eval/results.md).

---

## Files

```
utils/                 everything the pipeline is made of, in one place
  config.py            settings, validated at startup
  logger.py            one configured logger
  pipeline.py          joins the stages together
  elements.py          the single type everything flows through
  pdf_parser.py        extraction, header recovery, table repair
  summarizer.py        table descriptions and image captions
  vector_store.py      Chroma for vectors, SQLite for documents
  retriever.py         search, with slots reserved for tables
  qa_chain.py          the prompt and the answer
  provider.py          Gemini/OpenAI behind one interface
  build_pdf.py         renders the methodology doc to PDF

app/     cli.py, streamlit_app.py      how you run it
eval/    questions.yaml, run_eval.py, results.md
tests/   test_storage.py               offline, no API key
docs/    METHODOLOGY.md/.pdf           the write-up
data/    raw/ the filing, processed/ the built index
```

The pipeline modules sit flat in `utils/` on purpose. An earlier layout split
them across six single-file packages (`ingestion/`, `enrichment/`, `indexing/`,
`retrieval/`, `generation/`, `llm/`), which is more directories than modules and
makes you click through a folder to reach one file. Flat is easier to read.

---

## Notes

**Settings** live in [`config.yaml`](config.yaml), with the reason for each value
written next to it. After changing anything that affects extraction, rebuild:

```bash
python -m app.cli ingest --rebuild
```

**Models.** `gemini-3.5-flash-lite` for descriptions, captions and answers;
`gemini-embedding-001` for embeddings. Answering was originally on the stronger
`gemini-3.6-flash` and moved only because its free-tier quota ran out. That is
one line in `config.yaml`.

**Not done, on purpose:** loading the numbers into SQL columns so they can be
queried rather than read, a re-ranker, and containerisation. Reasons in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
