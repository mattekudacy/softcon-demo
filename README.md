# Suki Inbox

A small reference app for the talk **Crawl, Walk, Run: Applying the FinOps Maturity Model to an LLM Product**.

Suki Inbox triages the Messenger inbox of a small Philippine online shop. Customers write in English, Filipino, or Taglish. For each ticket the app runs three features:

| Feature | Output | Difficulty |
|---|---|---|
| `classify` | intent label (8 classes) | easy |
| `extract` | `{order_id, item, qty}` | easy |
| `reply` | draft reply that follows the shop policy | hard |

It is a reference implementation, not a production system. Traffic is 60 hand-written synthetic tickets. The numbers are real measurements from recorded runs.

## Run it offline (no API key, no cost)

Python 3.9+, standard library only.

```sh
python3 -m suki.report            # rebuild results/ from the traces
open results/dashboard.html
python3 -m suki.traffic run --runs 3   # replay the Run config from recordings/
```

Replay mode is the default. Every model response is stored in `recordings/responses.jsonl`, keyed by a hash of the exact request. Replay and record share one code path, so replayed traces match the recorded ones exactly.

## Record your own runs

```sh
cp .env.example .env              # then paste a key from https://ollama.com/settings/keys
SUKI_MODE=record SUKI_BUDGET_USD=2 python3 -m suki.traffic crawl --runs 3
```

`SUKI_BUDGET_USD` is a hard stop on spend recorded so far. Calls go straight to `https://ollama.com/api/chat` with the key. The local Ollama daemon is not used. Prices live in `pricing.json`, copied from https://ollama.com/pricing on the date shown there. Thinking tokens bill as output.

## The three phases

Tags `crawl`, `walk` and `run` mark each phase in git.

**Crawl** (`crawl` config). One model (`gpt-oss:120b`) for everything. The whole shop policy goes into the system prompt of every call, and every call gets the full conversation history. The wrapper in `suki/llm.py` logs feature, model, tokens, latency and cost for every call.

**Walk** (`suki/report.py`). No new model calls. It adds attribution over the crawl traces: spend by feature, by customer and by thread length, plus an alert when a ticket costs more than 2x the baseline median.

**Run** (`run-*` configs). Four levers, each measured alone, then combined in `run`:

| Config | Lever |
|---|---|
| `run-route` | `gpt-oss:20b` for classify and extract, and for replies to low-risk intents |
| `run-think` | low reasoning effort for classify and extract |
| `run-trim` | only the policy sections the intent needs, and only the last 3 turns |
| `run-cache` | app-level cache: identical requests in the same run are free |

## How it maps to the FinOps papers

| Paper | Used for |
|---|---|
| [FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/) | Crawl/Walk/Run phases, the cost per inference and model choice quality alignment KPIs, the optimization levers (model selection, trimming, caching) |
| [How to Build a Generative AI Cost and Usage Tracker](https://www.finops.org/wg/how-to-build-a-generative-ai-cost-and-usage-tracker/) | Trace record: actual token counts from the response, a `request_id`, a price table with an effective date, one wrapper for every call |
| [Cost Estimation of AI Workloads](https://www.finops.org/wg/cost-estimation-of-ai-workloads/) | `suki/estimate.py`: estimate tokens per request × volume × price before running, then compare with measured spend (the paper's target is over 90% accuracy) |

Ours, not the papers': cost per feature and per user (an answer to the Overview's allocation challenge), the reasoning-effort lever, and run-to-run spread.

Not measured: batching. The Overview lists it, but Ollama Cloud has no batch API.

The Tracker advises against keeping raw prompts and outputs long-term. These traces keep model outputs because the data is synthetic and the reply scorer needs them. Do not do this with real customer data.

## Quality

`suki/evaluate.py` scores against hand labels in `data/make_tickets.py`. No LLM judge, so scoring is free and deterministic.

- classify: exact intent match
- extract: share of order_id, item and qty that match
- reply: share of required policy facts the reply mentions (for example "7 days" for a return)

The reply check is a floor, not a full quality measure. It catches a reply that drops or contradicts a required fact. It does not judge tone.

## Layout

```
data/       policy.md, tickets.jsonl (built by make_tickets.py)
suki/       llm.py (wrapper, record/replay), app.py (features, configs), traffic.py, evaluate.py, report.py
recordings/ every model response, keyed by request hash
traces/     one JSONL record per call, per config and run, plus scores
results/    results.md, summary.json, dashboard.html
```

## Known limits

- 60 tickets is small. Run-to-run spread is reported next to every mean.
- The cheap and expensive models differ by about 2x in price. Routing savings depend on that gap.
- Ollama Cloud does not report cached input tokens or offer a batch API, so provider prompt caching and batch discounts are not measured here. The cache lever is an app-level response cache.
- The duplicate rate in the tickets (about 15%) is an assumption, and it sets the cache hit rate.
