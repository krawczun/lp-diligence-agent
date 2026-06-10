# LP Diligence Agent

An agentic AI prototype that performs LP-side diligence on private-equity fund quarterly reports. Given a quarterly report, the agent runs a 9-item diligence checklist against the document and returns structured, citation-backed answers with explicit confidence tags. Designed for sophisticated LPs and fund-of-fund managers who read hundreds of these per quarter.

**Live demo:** [lp-diligence.krawczun.com](https://lp-diligence.krawczun.com) · **Project writeup:** [krawczun.com/projects/lp-diligence-agent](https://krawczun.com/projects/lp-diligence-agent)

## What it does

- Ingests LP-format quarterly reports (PDF) and SEC 10-Q filings (HTML)
- Runs a 9-item diligence checklist against each document via a multi-step agent
- Returns structured JSON with citations, confidence tags, and a refusal mode when data is missing
- Exposes itself two ways:
  - **MCP server** (`lp_diligence.mcp_server`): usable directly from Claude Desktop
  - **FastAPI server** (`lp_diligence.api`): backs a Next.js demo UI
- Ships with a 20-question eval golden set and an LLM-judge eval harness

## The diligence checklist

1. NAV change drivers
2. Capital called this period (and cumulative vs. committed)
3. Distributions and DPI trajectory
4. IRR / TVPI / DPI vs. prior periods
5. Top holdings movement (gains, losses, exits)
6. Unfunded commitment remaining
7. Management fees and expense anomalies
8. Key-person / GP-level events
9. Valuation policy and Level 1/2/3 hierarchy

## Corpus

Five publicly available documents:
- **PSERS Hamilton Lane Quarterly Reports**: Q2/Q3/Q4 2017, FOIA-released via the Pennsylvania Joint State Government Commission Act 5 archive. These are the closest public analog to true GP-to-LP quarterly communications and provide three consecutive quarters of the same portfolio for change-over-time analysis.
- **Blackstone Private Equity Strategies Fund 10-Q**: Q1 and Q3 2025, registered SEC filings. Cover the fund-level fee detail and Level 1/2/3 valuation hierarchy that the PSERS reports redact.

See [data/corpus/MANIFEST.md](data/corpus/MANIFEST.md) for source URLs and provenance.

## Architecture

```
backend/src/lp_diligence/
  config.py        # env-driven constants
  documents.py     # PDF + HTML loaders + section detection
  chunking.py      # deterministic sentence-aware chunker
  embeddings.py    # local sentence-transformers (default) or OpenAI
  vectorstore.py   # sqlite-vec wrapper for cosine similarity
  ingest.py        # load → chunk → embed → store pipeline
  retrieval.py     # query → top-k chunks + citation formatting
  checklist.py     # the 9 items + the agent that fills them in
  api.py           # FastAPI server backing the Next.js demo
  mcp_server.py    # MCP server for Claude Desktop
  cli.py           # `lp-diligence` command-line interface

eval/
  golden_set/      # 20-question hand-curated set
  run_eval.py      # judge LLM scores faithfulness / recall / precision
  reports/         # eval output (generated)

docs/
  solution-design.md  # 1-page solution memo
  architecture.md     # diagrams + decisions

frontend/          # Next.js demo UI (see frontend/README.md)
```

## Guardrails

This is a prototype, not a production system. The interesting design choices are the guardrails:

- **Citation required for every numeric claim.** The system prompt instructs the LLM to return a structured "Data not available" response when the retrieved excerpts don't support a citation-backed answer. The agent does not fall back to general knowledge.
- **Confidence tag per answer.** Each item gets `high`, `medium`, or `refused`. A refused item flags missing data without faking continuity.
- **Eval harness with refusal correctness.** The golden set includes questions targeting redacted or out-of-scope content; the agent must refuse those to score well. Honest negative coverage matters more than headline accuracy.
- **Audit trail.** Every API and MCP call returns the full list of retrieved chunks, the chunk IDs cited, the model used, and the input/output token counts.
- **Data handling.** All document processing is local. The only network egress is to the Anthropic API for synthesis and to the embedding provider (local by default).

## Quick start

```powershell
# 1. Python 3.11+
python --version

# 2. Create venv and install
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[embeddings]"

# 3. Configure
cd ..
copy .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Ingest the corpus
lp-diligence ingest

# 5. Run the checklist
lp-diligence checklist PSERS_HL_2Q17
```

For Claude Desktop MCP integration, see `docs/mcp-setup.md`. For running the full Next.js + FastAPI demo locally and sharing it via Cloudflare Tunnel, see [`docs/demo.md`](docs/demo.md).

## Eval results

The committed baseline is at [eval/published/baseline.md](eval/published/baseline.md). Numbers published verbatim regardless of whether they're flattering. Headline metrics from the baseline:

- Faithfulness: 0.78 mean across the 20-question golden set
- Refusal correctness: 0.80 (16/20 refusal decisions matched expectations)
- Context recall: 0.62, context precision: 0.60
- Average latency: 4.2s per checklist item

Local eval runs land in `eval/reports/` (gitignored). Next iteration would add chunk re-ranking and per-section query rewriting to lift retrieval precision.

## License

MIT
