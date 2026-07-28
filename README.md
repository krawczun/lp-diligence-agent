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
  vectorstore.py   # sqlite-vec cosine similarity + FTS5 BM25 keyword search
  hybrid.py        # reciprocal rank fusion of the dense and sparse rankings
  reranking.py     # optional cross-encoder reranking stage
  ingest.py        # load → chunk → embed → store pipeline
  retrieval.py     # query → retrieve → fuse → (rerank) → cited chunks
  checklist.py     # the 9 items + the agent that fills them in
  api.py           # FastAPI server backing the Next.js demo
  mcp_server.py    # MCP server for Claude Desktop
  cli.py           # `lp-diligence` command-line interface

eval/
  golden_set/         # 20-question hand-curated set
  run_eval.py         # judge LLM scores faithfulness / recall / precision
  compare_rerank.py   # retrieval-only A/B (runs offline, no API key)
  published/          # committed baselines
  reports/            # eval output (generated)

docs/
  solution-design.md  # 1-page solution memo
  architecture.md     # diagrams + decisions

frontend/          # Next.js demo UI (see frontend/README.md)
```

## Retrieval: hybrid search, and why reranking is off

Retrieval runs in up to three stages: dense vector search, optional fusion with BM25 keyword search, and an optional cross-encoder reranking pass. Each stage is independently switchable so any combination can be measured.

**Hybrid search is on by default. Cross-encoder reranking is off.** That second decision was measured, not assumed.

### Why hybrid

Dense embedding search understands meaning but smears exact tokens. A query naming `2Q17` ranks chunks from 3Q17 and 4Q17 nearly as highly, because the period code dissolves into the surrounding semantics. BM25 has the mirror failure: it nails the literal token and has no idea "returns" and "performance" are related. Fusing them covers both.

The two score scales are not comparable (cosine distance versus `bm25()`, different ranges, different signs), so any weighted sum of raw scores would be arbitrary. [Reciprocal rank fusion](backend/src/lp_diligence/hybrid.py) discards magnitudes and uses only rank position, which is comparable across retrievers and rewards cross-retriever agreement.

### The measurement

All four configurations, run end to end over the 20-question golden set with an LLM judge:

| Config | Faithfulness | Context recall | Context precision | Refusal correctness |
|---|---|---|---|---|
| Vector only | 0.84 | 0.61 | 0.59 | 0.75 |
| Rerank only | 0.89 | 0.69 | 0.68 | 0.70 |
| **Hybrid only** | **0.91** | **0.73** | **0.71** | **0.80** |
| Hybrid + rerank | 0.89 | 0.61 | 0.65 | 0.75 |

Retrieval-level cost, measured separately (baseline vector search is ~25 ms/query):

| Config | Top-8 chunks replaced | Added latency |
|---|---|---|
| Hybrid only | 46% | **+6 ms** |
| Rerank only | 51% | +2038 ms |
| Hybrid + rerank | 67% | +3799 ms |

### Three findings

**Hybrid won on every quality metric and cost 6 ms.** Reranking spent 2 seconds per query to finish second.

**Stacking both was worse than hybrid alone.** Context recall fell from 0.73 to 0.61. The likely mechanism: RRF ranks by cross-retriever agreement, and re-scoring the fused set on pairwise relevance discards exactly that signal. Two rankers optimizing different objectives in sequence can undo each other.

**Reranking alone lowered refusal correctness below baseline** (0.70 versus 0.75), the only metric where it lost outright. On a corpus where refusing unsupported questions is a safety property, surfacing more plausible-looking context can make the model less willing to refuse. That is a quality regression, not just a latency cost.

Reranking stays in the codebase, documented and one flag away. It is the right tool when recall is good and ordering is bad, and a GPU changes the latency arithmetic entirely. It is simply not the right default *here*.

### Caveats

One corpus, one embedding model (`all-MiniLM-L6-v2`, 384 dims), 20 questions, single run per configuration, CPU-only inference. The direction is consistent across metrics and the effect sizes are larger than the gaps between adjacent configurations, but this is not a claim about hybrid search in general. It is a claim about this corpus.

Reproduce with:

```powershell
python eval\compare_rerank.py --mode hybrid    # retrieval-only, no API key needed
python eval\run_eval.py --no-rerank            # end-to-end, needs ANTHROPIC_API_KEY
$env:LP_DILIGENCE_HYBRID=1; python eval\run_eval.py --no-rerank
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

Numbers are published verbatim regardless of whether they're flattering. Committed baselines:

- [`eval/published/baseline_hybrid.json`](eval/published/baseline_hybrid.json): the current default configuration
- [`eval/published/baseline_vector_only.json`](eval/published/baseline_vector_only.json): dense retrieval alone, for comparison
- [`eval/published/baseline.md`](eval/published/baseline.md): the original pre-hybrid run

Headline metrics on the current default (hybrid search, 20-question golden set):

- Faithfulness: 0.91 mean
- Refusal correctness: 0.80
- Context recall: 0.73, context precision: 0.71
- Average latency: 5.0s per checklist item

The full four-way comparison and the reasoning behind the default configuration are in [Retrieval](#retrieval-hybrid-search-and-why-reranking-is-off) above.

Local eval runs land in `eval/reports/` (gitignored). Next iteration would add per-section query rewriting, and would test whether reranking earns its place on a GPU where the latency cost largely disappears.

## License

MIT
