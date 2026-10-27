"""Sends every ticket through the app and writes traces plus scored outputs.

  SUKI_MODE=record python3 -m suki.traffic crawl --runs 2   # calls the API, saves recordings
  python3 -m suki.traffic crawl --runs 2                    # replays offline, same numbers
"""
import argparse
import json

from . import app, evaluate, llm
from .llm import ROOT


def load_tickets():
    return [json.loads(x) for x in (ROOT / "data" / "tickets.jsonl").read_text().splitlines()]


def run(config, n):
    tickets = load_tickets()
    for r in range(1, n + 1):
        tracer = llm.Tracer(config, r)
        outputs = {}
        for i, t in enumerate(tickets, 1):
            outputs[t["id"]] = app.handle(t, app.CONFIGS[config], tracer)
            print(f"\r{config} run{r}: {i}/{len(tickets)}", end="", flush=True)
        tracer.close()
        scores = evaluate.score(tickets, outputs)
        path = ROOT / "traces" / config / f"run{r}.scores.json"
        path.write_text(json.dumps({"summary": scores["summary"], "tickets": scores["tickets"],
                                    "outputs": outputs}, ensure_ascii=False, indent=1))
        print(f"  quality={scores['summary']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=list(app.CONFIGS))
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()
    run(a.config, a.runs)
