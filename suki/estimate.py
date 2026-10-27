"""Cost estimate before running, then checked against measured spend.

Follows FinOps Foundation, "Cost Estimation of AI Workloads": estimate tokens per request,
multiply by volume and price, then track actuals against the estimate. The paper's target is
over 90% expenditure accuracy.

Two estimates, both made without calling a model:
  naive: input tokens from prompt length (about 4 characters per token), output tokens from the
         length of the visible answer a developer would expect. Set before any run, never tuned.
  pilot: measured cost of the first PILOT tickets, scaled to the full set.
"""
from . import app
from .llm import PRICING

CHARS_PER_TOKEN = 4
# What a developer writes down from the visible answer: a label, a small JSON object, a short reply.
VISIBLE_OUTPUT_TOKENS = {"classify": 5, "extract": 40, "reply": 150}
PILOT = 10
N_TICKETS = 60


def prompt_chars(feature, ticket):
    thread = ticket["thread"]
    prompt = {"classify": app.classify_prompt, "extract": app.extract_prompt, "reply": app.reply_prompt}[feature](thread)
    return len(app.SYSTEM_FULL) + len(prompt)


def naive(tickets, model=app.BIG):
    p = PRICING[model]
    total = 0.0
    for t in tickets:
        for f, out in VISIBLE_OUTPUT_TOKENS.items():
            total += (prompt_chars(f, t) / CHARS_PER_TOKEN * p["input"] + out * p["output"]) / 1e6
    return total


def accuracy(estimate, actual):
    """Expenditure accuracy: 1 minus the relative error."""
    return 1 - abs(estimate - actual) / actual


def compare(tickets, runs):
    """runs: loaded crawl runs. Compares both estimates with the measured mean total cost of a run."""
    actual = sum(sum(c["cost_usd"] for c in r["calls"]) for r in runs) / len(runs)
    first = {t["id"] for t in tickets[:PILOT]}
    pilot_cost = sum(c["cost_usd"] for c in runs[0]["calls"] if c["ticket_id"] in first)
    pilot = pilot_cost * len(tickets) / PILOT
    n = naive(tickets)
    in_tokens = sum(prompt_chars(f, t) for t in tickets for f in VISIBLE_OUTPUT_TOKENS) / CHARS_PER_TOKEN
    out_est = sum(VISIBLE_OUTPUT_TOKENS.values()) * len(tickets)
    out_actual = sum(c["output_tokens"] for c in runs[0]["calls"])
    in_actual = sum(c["input_tokens"] for c in runs[0]["calls"])
    return {
        "actual": actual,
        "naive": {"estimate": n, "accuracy": accuracy(n, actual)},
        "pilot": {"estimate": pilot, "accuracy": accuracy(pilot, actual), "tickets": PILOT},
        "tokens": {"input_estimated": in_tokens, "input_actual": in_actual,
                   "output_estimated": out_est, "output_actual": out_actual},
    }
