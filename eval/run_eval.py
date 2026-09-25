"""Eval harness: run the 20-question golden set, score with a judge LLM, report.

Metrics:
  faithfulness        — judge LLM: does the answer make claims supported by the citations?
  context_recall      — does the retrieved context contain the information needed?
  context_precision   — are the retrieved chunks relevant to the question?
  refusal_correctness — boolean: agent refused iff expected_refusal=True
  keyword_match       — boolean: at least half of expected_keywords appear in answer

Run:
  python eval/run_eval.py
  python eval/run_eval.py --sample 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Make backend src importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

import anthropic  # noqa: E402

from lp_diligence import config  # noqa: E402
from lp_diligence.checklist import _build_client, answer_item  # noqa: E402
from lp_diligence.embeddings import Embedder  # noqa: E402
from lp_diligence.retrieval import Citation, format_context  # noqa: E402
from lp_diligence.vectorstore import VectorStore  # noqa: E402


GOLDEN_PATH = Path(__file__).parent / "golden_set" / "golden_set.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"


JUDGE_PROMPT = """You are evaluating an LP diligence agent's answer.

Question: {question}

Retrieved context:
{context}

Agent answer: {answer}

Score the answer on these dimensions (each 0.0 to 1.0):

1. faithfulness — Are the agent's claims supported by the retrieved context? 1.0 means every claim is grounded. 0.0 means hallucinated.
2. context_recall — Does the retrieved context contain the information needed to answer the question? 1.0 means yes, fully. 0.0 means the relevant info is missing.
3. context_precision — Of the retrieved chunks, how many are actually relevant to the question? 1.0 means all chunks are relevant. 0.0 means none are.

Return JSON: {{"faithfulness": <float>, "context_recall": <float>, "context_precision": <float>, "rationale": "<one sentence>"}}
Output ONLY the JSON object."""


@dataclass
class EvalRow:
    id: str
    doc_id: str
    checklist_item: str
    question: str
    expected_refusal: bool
    expected_keywords: list[str]
    # Filled in by the run
    answer: str = ""
    confidence: str = ""
    refusal_correct: bool = False
    keyword_match: bool = False
    faithfulness: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    judge_rationale: str = ""
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    citations: list[dict] = field(default_factory=list)


def _load_golden(path: Path, sample: Optional[int]) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append(
            EvalRow(
                id=d["id"],
                doc_id=d["doc_id"],
                checklist_item=d["checklist_item"],
                question=d["question"],
                expected_refusal=bool(d.get("expected_refusal", False)),
                expected_keywords=list(d.get("expected_keywords", [])),
            )
        )
    if sample:
        rows = rows[:sample]
    return rows


def _judge_context(citations: list[dict]) -> str:
    """The exact context block the agent saw, rebuilt from the answer's citations.

    The judge must see everything the agent saw. An earlier version passed only the
    first 6 excerpts, cut to 6,000 characters, so detailed answers that drew on later
    excerpts were scored "unsupported" for facts the judge never received.
    """
    return format_context([Citation(**c) for c in citations])


def _parse_judge(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # The judge sometimes wraps the object in prose; take the outermost {...} block.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"faithfulness": 0.0, "context_recall": 0.0, "context_precision": 0.0,
            "rationale": "judge JSON parse failed", "parse_failed": True}


def _judge(client: anthropic.Anthropic, question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    kwargs = {}
    # Newer models think by default; a judge that returns one JSON object doesn't need to.
    if config.JUDGE_MODEL.startswith(("claude-sonnet-5", "claude-opus-5")):
        kwargs["thinking"] = {"type": "disabled"}
    resp = client.messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return _parse_judge("".join(b.text for b in resp.content if b.type == "text"))


def _keyword_match(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits >= max(1, len(keywords) // 2)


def run_eval(sample: Optional[int] = None) -> dict:
    rows = _load_golden(GOLDEN_PATH, sample)
    client = _build_client()
    embedder = Embedder()
    store = VectorStore(config.VECTOR_DB_PATH, dim=embedder.dim)

    for row in rows:
        t0 = time.time()
        ans = answer_item(
            row.checklist_item,
            row.question,
            doc_id=row.doc_id,
            client=client,
            embedder=embedder,
            store=store,
        )
        row.answer = ans.answer
        row.confidence = ans.confidence
        row.citations = ans.citations
        row.latency_ms = ans.latency_ms
        row.tokens_in = ans.tokens_in
        row.tokens_out = ans.tokens_out

        is_refusal = ans.confidence == "refused"
        row.refusal_correct = is_refusal == row.expected_refusal
        row.keyword_match = _keyword_match(ans.answer, row.expected_keywords)

        # Judge — but skip when expected refusal and agent refused (no answer to score)
        if not (row.expected_refusal and is_refusal):
            scores = _judge(client, row.question, _judge_context(ans.citations), ans.answer)
            row.faithfulness = float(scores.get("faithfulness", 0.0))
            row.context_recall = float(scores.get("context_recall", 0.0))
            row.context_precision = float(scores.get("context_precision", 0.0))
            row.judge_rationale = str(scores.get("rationale", ""))
        else:
            row.faithfulness = 1.0
            row.context_recall = 1.0
            row.context_precision = 1.0
            row.judge_rationale = "expected refusal, agent refused"

        print(f"  [{row.id}] conf={row.confidence} faith={row.faithfulness:.2f} latency={row.latency_ms}ms")

    store.close()

    # Aggregate
    n = len(rows)
    summary = {
        "n_questions": n,
        "rerank_enabled": config.RERANK_ENABLED,
        "rerank_model": config.RERANK_MODEL if config.RERANK_ENABLED else None,
        "rerank_candidates": config.RERANK_CANDIDATES if config.RERANK_ENABLED else None,
        "refusal_correctness": sum(1 for r in rows if r.refusal_correct) / max(1, n),
        "keyword_match": sum(1 for r in rows if r.keyword_match) / max(1, n),
        "faithfulness_mean": sum(r.faithfulness for r in rows) / max(1, n),
        "context_recall_mean": sum(r.context_recall for r in rows) / max(1, n),
        "context_precision_mean": sum(r.context_precision for r in rows) / max(1, n),
        "avg_latency_ms": int(sum(r.latency_ms for r in rows) / max(1, n)),
        "total_tokens_in": sum(r.tokens_in for r in rows),
        "total_tokens_out": sum(r.tokens_out for r in rows),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"eval_{timestamp}.json"
    md_path = REPORTS_DIR / f"eval_{timestamp}.md"

    payload = {"summary": summary, "rows": [asdict(r) for r in rows]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(summary, rows), encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(summary, indent=2))
    return summary


def _render_markdown(summary: dict, rows: list[EvalRow]) -> str:
    lines = [
        "# LP Diligence Agent — Eval Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{config.MODEL}` (judge: `{config.JUDGE_MODEL}`)",
        (
            f"Reranking: **on** (`{summary['rerank_model']}`, "
            f"{summary['rerank_candidates']} candidates -> top {config.RETRIEVAL_K})"
            if summary.get("rerank_enabled")
            else "Reranking: **off** (vector similarity only)"
        ),
        "",
        "## Summary",
        "",
        f"- **Questions**: {summary['n_questions']}",
        f"- **Refusal correctness**: {summary['refusal_correctness']:.2f}",
        f"- **Keyword match**: {summary['keyword_match']:.2f}",
        f"- **Faithfulness (mean)**: {summary['faithfulness_mean']:.2f}",
        f"- **Context recall (mean)**: {summary['context_recall_mean']:.2f}",
        f"- **Context precision (mean)**: {summary['context_precision_mean']:.2f}",
        f"- **Avg latency**: {summary['avg_latency_ms']} ms",
        f"- **Total tokens**: {summary['total_tokens_in']} in / {summary['total_tokens_out']} out",
        "",
        "## Per-question results",
        "",
        "| ID | doc | conf | refusal ok | kw ok | faith | recall | precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.id} | {r.doc_id} | {r.confidence} | "
            f"{'✓' if r.refusal_correct else '✗'} | "
            f"{'✓' if r.keyword_match else '✗'} | "
            f"{r.faithfulness:.2f} | {r.context_recall:.2f} | {r.context_precision:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Run only the first N questions (smoke test).")
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=None,
        help="Force cross-encoder reranking ON for this run.",
    )
    rerank_group.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Force cross-encoder reranking OFF for this run (the A/B baseline).",
    )
    args = parser.parse_args()

    # Retrieval reads config at call time, so setting it here covers every
    # question in the run without threading a flag through the call chain.
    if args.rerank is not None:
        config.RERANK_ENABLED = args.rerank

    state = "ON" if config.RERANK_ENABLED else "OFF"
    detail = ""
    if config.RERANK_ENABLED:
        detail = (
            f" (model={config.RERANK_MODEL.split('/')[-1]}, "
            f"candidates={config.RERANK_CANDIDATES}, max_len={config.RERANK_MAX_LENGTH})"
        )
    print(f"Reranking: {state}{detail}\n")

    run_eval(sample=args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
