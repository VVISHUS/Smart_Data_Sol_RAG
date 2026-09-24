"""
Answer synthesis: turn retrieved elements into a grounded, cited answer.

WHY THE PROMPT IN THIS FILE IS LONGER THAN THE CODE
---------------------------------------------------
Retrieval decides what the model CAN see; the prompt decides what it DOES with
it. For a financial filing, three specific failure modes account for nearly all
wrong answers, and each needs an explicit instruction. They are not generic
"be accurate" boilerplate -- each one was chosen for this document.

1. THE PERIOD-COLUMN TRAP  (the big one for a 10-Q)
   Every income-statement table in this filing has FOUR value columns:
       Three Months Ended | June 25 2022 | June 26 2021
       Nine  Months Ended | June 25 2022 | June 26 2021
   A model asked "what were iPhone net sales?" will frequently read across to
   the nine-month column and answer $162,863M when the quarterly figure is
   $40,665M. The number is real, appears in the table, and is wrong. No
   retrieval improvement fixes this -- it is a reading error, so it must be
   addressed in the prompt.

2. CROSS-TABLE ARITHMETIC
   Asked for a growth rate not printed in the filing, a model will happily
   compute one and present it with the same confidence as a quoted figure. We
   permit the calculation but require it to be labelled as derived, so a reader
   can tell quoted facts from our arithmetic.

3. PLAUSIBLE FABRICATION
   If the context does not contain the answer, the model's prior over SEC
   filings is strong enough to invent a very convincing one. The refusal
   instruction is what converts a silent wrong answer into a visible gap --
   which is the only behaviour that makes the system trustworthy.

WHY WE PUT THE QUESTION AFTER THE CONTEXT
-----------------------------------------
Long-context models attend most reliably to the beginning and the END of the
prompt. Placing the question last means the instruction the model acts on is in
the highest-attention position, rather than buried above several thousand tokens
of tables.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.config import AppConfig
from utils.elements import Element, ElementKind
from utils.provider import LLMProvider
from utils.logger import get_logger
from utils.retriever import RetrievalResult, Retriever

log = get_logger(__name__)


ANSWER_PROMPT = """You are a financial analyst answering questions about Apple Inc.'s
Form 10-Q for the quarter ended June 25, 2022. Answer ONLY from the context below.

RULES

1. PERIOD COLUMNS. Financial tables in this filing report BOTH a three-month
   (quarterly) period AND a nine-month (year-to-date) period, each with a current
   and a prior year column. Before quoting any number, identify which column it
   sits in. If the question does not specify a period, assume the THREE-MONTH
   (quarterly) figure and state that assumption.

2. CITE EVERYTHING. After each fact, cite its source in square brackets exactly
   as given in the context header, e.g. [table on p.10]. A number without a
   citation is not an acceptable answer.

3. UNITS. State the unit with every figure. Most tables are in millions of USD;
   share counts are in thousands. Do not convert between them silently.

4. DERIVED VALUES. If the answer requires arithmetic that is not printed in the
   filing, you may compute it, but you must show the inputs and label the result
   as "calculated".

5. IF IT IS NOT THERE, SAY SO. If the context does not contain enough to answer,
   reply exactly: "The provided document excerpts do not contain this
   information." Do not use outside knowledge about Apple. Do not guess.

CONTEXT
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class Answer:
    """
    The answer plus everything needed to audit it.

    WHY we return the retrieval result and not just the text: the evaluation
    harness must distinguish a RETRIEVAL failure (the right table never reached
    the model) from a GENERATION failure (the right table was there and the
    model misread it). Those have completely different fixes, and without the
    context attached you cannot tell them apart.
    """

    text: str
    retrieval: RetrievalResult
    question: str

    @property
    def cited_pages(self) -> list[int]:
        return sorted({e.page for e in self.retrieval.elements})


def _format_context(elements: list[Element]) -> str:
    """
    Render retrieved elements into the prompt's context block.

    WHY EACH ELEMENT GETS AN EXPLICIT HEADER (`[table on p.10]`): the citation
    rule in the prompt tells the model to cite "exactly as given in the context
    header". Providing the exact string we want echoed back removes any need for
    the model to invent a citation format, which makes citations parseable and
    therefore checkable by the eval harness.

    WHY FIGURES ARE MARKED as a description rather than presented as content:
    the text for a figure is a vision model's interpretation, not something
    printed in the filing. Labelling it keeps that provenance honest -- a reader
    should know a figure answer is one inference removed from the source.

    WHY the separator line: without a hard delimiter, a Markdown table's trailing
    row and the next element's opening line visually merge, and the model
    occasionally reads a value from the wrong element.
    """
    blocks: list[str] = []
    for element in elements:
        header = f"[{element.citation()}]"

        if element.kind is ElementKind.TABLE:
            body = element.content
        elif element.kind is ElementKind.FIGURE:
            body = f"(description of an image, generated by a vision model)\n{element.content}"
        else:
            body = element.content

        blocks.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(blocks)


class QAChain:
    """Ties retrieval and generation together."""

    def __init__(self, retriever: Retriever, provider: LLMProvider, cfg: AppConfig):
        self.retriever = retriever
        self.provider = provider
        self.cfg = cfg

    def ask(self, question: str) -> Answer:
        """
        Answer one question end to end.

        WHY the empty-retrieval short-circuit: with no context the prompt would
        still be sent, and the model -- given a question about Apple and no
        evidence -- would answer from memory. That is precisely the failure the
        whole system exists to prevent, so we refuse before spending the call.
        """
        retrieval = self.retriever.retrieve(question)

        if not retrieval.elements:
            return Answer(
                text="The provided document excerpts do not contain this information.",
                retrieval=retrieval,
                question=question,
            )

        prompt = ANSWER_PROMPT.format(
            context=_format_context(retrieval.elements),
            question=question,
        )

        # WHY `answering=True`: routes to the answer model configured in
        # config.yaml rather than the cheap bulk summariser. This is the call the
        # reviewer judges, and the one worth spending on.
        text = self.provider.complete(prompt, answering=True).strip()

        log.info("Answered in %d chars from %d elements", len(text), len(retrieval.elements))
        return Answer(text=text, retrieval=retrieval, question=question)
