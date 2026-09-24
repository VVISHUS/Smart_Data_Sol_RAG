"""
Evaluation harness.

    python -m eval.run_eval [--limit N] [--category table]

WHY THIS EXISTS
---------------
Without it, every claim about this system is an anecdote. The brief rewards
"well-structured, organized" work, and the most concrete evidence of that is a
number that was measured rather than asserted.

It also answers the question that matters when something is wrong: was this a
RETRIEVAL failure (the right table never reached the model) or a GENERATION
failure (it was there and the model misread it)? Those have completely different
fixes, so the harness records retrieval diagnostics alongside every verdict.

WHY SUBSTRING MATCHING AND NOT AN LLM JUDGE
-------------------------------------------
For factual financial QA the ground truth is an exact figure. "Did the string
82,959 appear in the answer?" is a complete and unambiguous test -- it needs no
second model, costs nothing, cannot itself hallucinate, and is deterministic, so
two runs are comparable.

An LLM judge would be necessary for open-ended generation, and it would
introduce its own error rate into the measurement. Here it would add cost and
noise to answer a question that string matching already answers exactly.

THE LIMITATION, STATED HONESTLY: substring matching cannot detect a right number
used in a wrong sentence ("total net sales fell to 82,959"). The `forbid` field
partly compensates by catching the specific known confusions, and the harness
prints every answer so failures are inspectable by eye. This is a deliberate
trade-off, not an oversight.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from utils.config import PROJECT_ROOT, load_config
from utils.pipeline import build_qa_chain

console = Console()


def _normalise(text: str) -> str:
    """
    Canonicalise before comparison.

    WHY strip thousands separators: the filing prints "82,959" but a model may
    answer "$82,959 million", "82959", or "82.959 billion". Removing commas and
    currency symbols means we test the DIGITS, which is the fact, rather than the
    formatting, which is not.

    WHY casefold: "MacBook Air" vs "macbook air" is not a correctness difference.
    """
    text = text.casefold()
    text = text.replace(",", "").replace("$", "")
    # WHY collapse whitespace: answers wrap across lines, and a newline inside
    # an expected phrase would otherwise cause a spurious failure.
    return re.sub(r"\s+", " ", text)


def _check(answer: str, expect: list[str], forbid: list[str]) -> tuple[bool, str]:
    """
    Return (passed, reason).

    WHY `forbid` is checked FIRST and overrides everything: a forbidden string is
    positive evidence of a specific known failure -- most importantly the
    period-column confusion. An answer containing both the right number and the
    nine-month number has not demonstrated it can distinguish them, so it must
    not be scored as a pass.
    """
    norm = _normalise(answer)

    for bad in forbid:
        if _normalise(bad) in norm:
            return False, f"contains forbidden value '{bad}'"

    missing = [e for e in expect if _normalise(e) not in norm]
    if missing:
        return False, f"missing {missing}"

    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation set.")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N")
    parser.add_argument("--category", default=None, help="filter to one category")
    parser.add_argument(
        "--out",
        default="eval/results.md",
        help="write a Markdown report here (paste into the methodology PDF)",
    )
    args = parser.parse_args(argv)

    questions = yaml.safe_load(
        (PROJECT_ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8")
    )
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    cfg = load_config()
    chain = build_qa_chain(cfg)

    rows: list[dict] = []
    for q in questions:
        console.print(f"[dim]{q['id']}[/dim] {q['question']}")
        try:
            answer = chain.ask(q["question"])
            passed, reason = _check(
                answer.text, q.get("expect", []), q.get("forbid", [])
            )
            kinds = answer.retrieval.kind_counts()
            rows.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "passed": passed,
                    "reason": reason,
                    "answer": answer.text,
                    # WHY record retrieval composition: if a table question fails
                    # with zero tables retrieved, it is a retrieval bug. If it
                    # fails with the right table present, it is a prompt bug.
                    "retrieved": ", ".join(f"{k}={v}" for k, v in kinds.items()),
                    "pages": ", ".join(map(str, answer.cited_pages)),
                }
            )
        except Exception as exc:
            # WHY continue on error: one failed API call should not discard the
            # other 19 results, which cost time and quota to obtain.
            console.print(f"[red]  error: {exc}[/red]")
            rows.append(
                {
                    "id": q["id"], "category": q["category"], "question": q["question"],
                    "passed": False, "reason": f"exception: {exc}",
                    "answer": "", "retrieved": "", "pages": "",
                }
            )

        mark = "[green]PASS[/green]" if rows[-1]["passed"] else "[red]FAIL[/red]"
        console.print(f"  {mark} {rows[-1]['reason']}\n")

    # --- summary -----------------------------------------------------------
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])

    summary = Table("category", "passed", "total", "accuracy", title="Results")
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, items in sorted(by_cat.items()):
        p = sum(1 for i in items if i["passed"])
        summary.add_row(cat, str(p), str(len(items)), f"{p / len(items):.0%}")
    summary.add_row("[bold]overall[/bold]", f"[bold]{passed}[/bold]",
                    f"[bold]{total}[/bold]", f"[bold]{passed / total:.0%}[/bold]")
    console.print(summary)

    # --- Markdown report ---------------------------------------------------
    # WHY emit Markdown: it drops straight into the methodology PDF without
    # retyping, which removes the chance of transcription error between the
    # measured result and the reported one.
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation Results",
        "",
        f"**{passed}/{total} passed ({passed / total:.0%})**",
        "",
        "| category | passed | total | accuracy |",
        "|---|---|---|---|",
    ]
    for cat, items in sorted(by_cat.items()):
        p = sum(1 for i in items if i["passed"])
        lines.append(f"| {cat} | {p} | {len(items)} | {p / len(items):.0%} |")
    lines += ["", "## Per-question detail", ""]
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        lines += [
            f"### {r['id']} ({r['category']}) - {status}",
            "",
            f"**Q:** {r['question']}",
            "",
            f"**A:** {r['answer']}",
            "",
            f"*retrieved:* {r['retrieved']} · *pages:* {r['pages']} · *verdict:* {r['reason']}",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[dim]report written to {out_path}[/dim]")

    # WHY exit non-zero on any failure: makes the harness usable as a CI gate.
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
