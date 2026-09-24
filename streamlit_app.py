"""
Entry point for Streamlit Cloud, which looks for the app at the repository root.

    streamlit run streamlit_app.py      (deployment)
    streamlit run app/streamlit_app.py  (equivalent, local)

The implementation lives in `app/streamlit_app.py`. Importing that module runs
it, because a Streamlit script is executed top to bottom on import, so this file
needs to do nothing else.

WHY A SHIM AND NOT A COPY
-------------------------
This file began as a full copy of `app/streamlit_app.py`, and was already out of
date within the hour: a bug fix landed in the `app/` version and the root copy
silently kept the broken behaviour. Since the root copy is the one that actually
gets deployed, the stale version is the one users would have seen.

Two files that must stay identical will not stay identical. One implementation,
imported from wherever it is needed.
"""

import app.streamlit_app  # noqa: F401  - the import is the app
