"""
Root-level entry point for the CLI, so it can be run from the repository root.

    python cli.py ingest
    python cli.py ask "What were iPhone net sales in Q3 2022?"

Equivalent to `python -m app.cli ...`.

WHY A SHIM AND NOT A COPY: this file began as a byte-for-byte duplicate of
`app/cli.py`. Two copies of the same 200 lines will drift, and the sibling root
copy of the Streamlit app had already drifted by the time this was written. The
implementation stays in one place.
"""

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
