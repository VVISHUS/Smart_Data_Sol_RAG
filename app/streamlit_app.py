"""
Streamlit demo UI.

    streamlit run app/streamlit_app.py

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

import streamlit as st

from utils.config import PROJECT_ROOT, load_config
from utils.elements import ElementKind
from utils.pipeline import build_qa_chain

st.set_page_config(page_title="10-Q RAG", page_icon="📄", layout="wide")


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

question = st.text_input(
    "Ask a question about the filing",
    placeholder="What were Apple's total net sales for the quarter?",
)

if question:
    try:
        with st.spinner("Retrieving and generating…"):
            answer = _chain().ask(question)
    except RuntimeError as exc:
        # WHY handle this specific case in the UI: an empty index is the single
        # most likely first-run problem, and the fix is one command. A raw
        # traceback in the browser would not communicate that.
        st.error(f"{exc}")
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
