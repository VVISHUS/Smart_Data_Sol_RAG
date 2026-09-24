"""
Streamlit demo UI.

    streamlit run streamlit_app.py

WHY THIS FILE IS AT THE REPOSITORY ROOT
---------------------------------------
Streamlit Cloud looks for the app at the root, and Streamlit re-executes the
whole script on every interaction.

It briefly lived in app/ with a one-line shim here that imported it. That was
wrong, and broke the app in a way worth recording: Python caches imports, so the
first page load ran the module body and rendered the UI, but every rerun after
that found the module already in sys.modules, skipped the body, and rendered
nothing. Pressing Enter on a question blanked the page.

A Streamlit script has to be the file Streamlit executes, not a module something
imports. So the implementation lives here, in one place.

WHY THIS IS THE SECONDARY INTERFACE
-----------------------------------
The CLI is the real interface (scriptable, pipeable, what the eval harness
mirrors). This exists because a reviewer should be able to see the system work
in thirty seconds without reading argparse help, and because showing the
retrieved context beside the answer is far more convincing in a browser than in
a terminal.

WHY THE CONTEXT PANEL IS ALWAYS VISIBLE, NOT HIDDEN BEHIND A TOGGLE
-------------------------------------------------------------------
The claim this project makes is "grounded, cited answers". A UI that shows only
the answer asks the user to take that on faith and looks identical to a chatbot
answering from memory. Showing the exact tables that produced the number is the
demonstration.
"""

from __future__ import annotations

import os

import streamlit as st

from utils.config import PROJECT_ROOT, load_config
from utils.elements import ElementKind
from utils.pipeline import build_qa_chain

st.set_page_config(page_title="10-Q RAG", page_icon="📄", layout="wide")


# WHY BRIDGE st.secrets INTO THE ENVIRONMENT
# On Streamlit Cloud the API key is supplied through the app's Secrets settings,
# while every other entry point in this project (the CLI, the eval harness)
# reads credentials from environment variables. Copying them across once at
# startup keeps a single credential path instead of putting a Streamlit-specific
# branch inside config.py, which the CLI would then carry for no reason.
for _var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
    try:
        if _var in st.secrets and not os.getenv(_var):
            os.environ[_var] = str(st.secrets[_var])
    except Exception:
        # WHY swallow: st.secrets raises if no secrets file exists at all, which
        # is the normal case when running locally against a .env. Not an error.
        break


@st.cache_resource
def _chain():
    """
    Build the QA stack once per server process.

    WHY `cache_resource` and not `cache_data`: Streamlit re-runs this entire
    script top-to-bottom on every widget interaction. Without caching, each
    keystroke would re-open Chroma, reload the docstore and re-instantiate the
    model clients -- several seconds of latency per interaction.

    `cache_resource` is the correct decorator for unserialisable, long-lived
    objects like DB connections and model clients; `cache_data` would try to
    pickle them and fail.

    CAVEAT THIS CACHE INTRODUCES: it lives for the life of the server, so a
    change made from outside (an `ingest --rebuild` in a terminal) is invisible
    here until the cache is cleared. The store is now written to survive that on
    its own, and the sidebar has a Reload button for whatever it cannot.
    """
    return build_qa_chain(load_config())


st.title("Apple 10-Q - Retrieval-Augmented QA")
st.caption(
    "Form 10-Q for the quarterly period ended June 25, 2022. "
    "Answers are generated only from retrieved excerpts of that filing."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
Every answer is grounded in retrieved excerpts, shown below it.

**Try:**
- What were iPhone net sales in Q3 2022?
- Compare Americas and Europe net sales.
- What did Apple say about COVID-19 supply disruption?
- How many iPhone units were sold? *(should refuse)*
        """
    )
    # WHY surface this in the UI: the figure finding is a headline result of the
    # project, and a reviewer opening the demo should meet it immediately rather
    # than having to dig it out of the write-up.
    st.info(
        "**On figures:** this filing contains exactly one embedded image "
        "(a 46×56px logo on p.1) and no charts or graphs. The figure pipeline "
        "runs and reports their absence rather than inventing them."
    )

    # WHY THIS BUTTON EXISTS: if the index is rebuilt from a terminal while this
    # server is running, the cached store can be out of step with what is on
    # disk. Rather than telling the user to restart the server, give them the
    # one action that fixes it.
    if st.button("Reload index"):
        st.cache_resource.clear()
        st.rerun()

    # WHY SHOW THE RAW COUNTS: the two stores are joined by id, and when they
    # disagree the symptom is silent. Chroma returns ids, none of them resolve
    # to a docstore row, the context comes back empty, and the model correctly
    # answers "the provided document excerpts do not contain this information"
    # for every single question. That looks like a model problem and is not one.
    #
    # Two numbers make the difference obvious: 59/59 is healthy, 59/0 means the
    # docstore did not load, 0/59 means the vectors did not.
    st.divider()
    try:
        _store = _chain().retriever.store
        _v, _d = _store.size, _store.docstore_size
        (st.caption if _v == _d else st.warning)(
            f"index: {_v} vectors / {_d} docstore rows"
            + ("" if _v == _d else "  <- these should match")
        )
    except Exception as _exc:
        st.warning(f"index unavailable: {type(_exc).__name__}: {_exc}")

question = st.text_input(
    "Ask a question about the filing",
    placeholder="What were Apple's total net sales for the quarter?",
)

if question:
    try:
        with st.spinner("Retrieving and generating…"):
            answer = _chain().ask(question)
    except Exception as exc:
        # WHY catch broadly here and not just RuntimeError: the first version
        # caught only our own error, so a stale Chroma handle (which raises the
        # vendor's NotFoundError) reached the browser as a raw traceback. Any
        # failure at this point has the same two remedies, so state them.
        st.error(f"{type(exc).__name__}: {exc}")
        st.caption(
            "If the index was rebuilt while this app was running, press "
            "**Reload index** in the sidebar. If it has never been built, run "
            "`python -m app.cli ingest`."
        )
        st.stop()

    st.markdown("### Answer")
    st.markdown(answer.text)

    st.markdown("### Retrieved context")
    st.caption(f"strategy: {answer.retrieval.strategy}")

    for element, distance in zip(answer.retrieval.elements, answer.retrieval.distances):
        with st.expander(f"{element.citation()}  ·  distance {distance:.3f}"):
            if element.kind is ElementKind.TABLE:
                # WHY render as Markdown: the content already IS a Markdown pipe
                # table, so Streamlit displays it as a real grid -- which is the
                # clearest possible evidence that the extraction preserved
                # structure.
                st.markdown(element.content)
            elif element.kind is ElementKind.FIGURE:
                image_path = element.metadata.get("image_path")
                if image_path:
                    # resolve project-relative path (see pdf_parser)
                    st.image(str(PROJECT_ROOT / image_path), width=240)
                st.caption("Vision-model description:")
                st.write(element.content)
            else:
                st.write(element.content)
