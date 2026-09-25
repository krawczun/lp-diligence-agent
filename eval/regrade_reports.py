"""Re-grade stored eval runs with the fixed judge, without regenerating any answers.

Each stored row keeps the citations the agent actually saw, so the judge can be given
exactly that context. Only faithfulness, context recall and context precision change;
refusal correctness and keyword match are mechanical and are carried over as-is.

Writes <run>.regraded.json next to each input. Originals are never modified.

Run:
  python eval/regrade_reports.py --dry-run
  python eval/regrade_reports.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lp_diligence import config  # noqa: E402
from lp_diligence.checklist import _build_client  # noqa: E402
import run_eval  # noqa: E402

EVAL = Path(__file__).resolve().parent
# The five distinct runs behind every number the README and portfolio quote.
RUNS = {
    "published/baseline.json": "May 2026 baseline (vector search only)",
    "reports/eval_20260728_103953.json": "July: vector search only",
    "reports/eval_20260728_104323.json": "July: vector + cross-encoder reranking",
    "reports/eval_20260728_104630.json": "July: hybrid (vector + BM25), the shipped default",
    "reports/eval_20260728_105034.json": "July: hybrid + reranking",
}
METRICS = ("faithfulness", "context_recall", "context_precision")
# Sonnet 5 list price, USD per million tokens; used only for the spend guard.
PRICE_IN, PRICE_OUT = 2.00, 10.00


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ceiling", type=float, default=3.00)
    args = ap.parse_args()

    todo = sum(1 for rel in RUNS for r in json.loads((EVAL / rel).read_text(encoding="utf-8"))["rows"]
               if not (r["expected_refusal"] and r["confidence"] == "refused"))
    print(f"Judge: {config.JUDGE_MODEL}. {len(RUNS)} runs, {todo} answers to grade, "
          f"about ${todo * 0.012:.2f} (hard stop ${args.ceiling:.2f})")
    if args.dry_run:
        return 0

    client = _build_client()
    spent = 0.0

    class Metered:
        """Forward to the real client and keep a running spend total."""
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            nonlocal spent
            resp = client.messages.create(**kw)
            spent += (resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT) / 1e6
            if spent > args.ceiling:
                raise SystemExit(f"Spend ceiling ${args.ceiling:.2f} exceeded; stopping.")
            return resp

    judge_client = Metered()
    for rel, label in RUNS.items():
        path = EVAL / rel
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        failures = 0
        for r in rows:
            for m in METRICS:
                r[f"{m}_first_judge"] = r[m]
            if r["expected_refusal"] and r["confidence"] == "refused":
                continue  # same rule as run_eval: nothing to grade, scored 1.0
            scores = run_eval._judge(judge_client, r["question"], run_eval._judge_context(r["citations"]), r["answer"])
            failures += bool(scores.get("parse_failed"))
            for m in METRICS:
                r[m] = float(scores.get(m, 0.0))
            r["judge_rationale"] = str(scores.get("rationale", ""))
        n = len(rows)
        s = data["summary"]
        s["first_judge"] = {f"{m}_mean": s[f"{m}_mean"] for m in METRICS}
        for m in METRICS:
            s[f"{m}_mean"] = sum(r[m] for r in rows) / n
        s["judge_model"] = config.JUDGE_MODEL
        s["judge_context"] = "full context the agent saw"
        s["regraded"] = "2026-09-25"
        out = path.with_suffix(".regraded.json")
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        change = "  ".join(f"{m.split('_')[-1]} {s['first_judge'][m + '_mean']:.2f}->{s[m + '_mean']:.2f}" for m in METRICS)
        print(f"{label:<52} {change}  parse failures: {failures}  (${spent:.2f})", flush=True)
    print(f"Spent ${spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
