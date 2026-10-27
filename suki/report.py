"""Walk: attribution and alerts over the traces. Also builds results.md and dashboard.html.

  python3 -m suki.report      # reads traces/, writes results/ ; fully offline
"""
import html
import json
import statistics as st
from collections import defaultdict

from . import estimate
from .llm import ROOT, cost_usd
from .traffic import load_tickets

CONFIG_ORDER = ["crawl", "run-route", "run-think", "run-trim", "run-cache", "run"]
LABELS = {"crawl": "Crawl baseline", "run-route": "Routing only", "run-think": "Low reasoning only",
          "run-trim": "Context trimming only", "run-cache": "Response cache only", "run": "Run: all levers"}
FEATURES = ["classify", "extract", "reply"]
ANOMALY_X = 2.0  # alert when a ticket costs more than 2x the baseline median ticket


def load(config):
    runs = []
    for p in sorted((ROOT / "traces" / config).glob("run*.jsonl")):
        scores = p.with_name(p.stem + ".scores.json")
        if not scores.exists():
            continue  # incomplete run
        calls = [json.loads(x) for x in p.read_text().splitlines()]
        runs.append({"calls": calls, "scores": json.loads(scores.read_text())})
    return runs


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]


def run_metrics(run, n_tickets):
    calls = run["calls"]
    per_ticket_ms = defaultdict(int)
    for c in calls:
        per_ticket_ms[c["ticket_id"]] += c["latency_ms"]
    return {
        "cost_per_ticket": sum(c["cost_usd"] for c in calls) / n_tickets,
        "cost_per_call": sum(c["cost_usd"] for c in calls) / len(calls),
        "tokens_per_ticket": sum(c["input_tokens"] + c["output_tokens"] for c in calls) / n_tickets,
        "input_per_ticket": sum(c["input_tokens"] for c in calls) / n_tickets,
        "output_per_ticket": sum(c["output_tokens"] for c in calls) / n_tickets,
        "p95_ms": p95(list(per_ticket_ms.values())),
        "cache_hit_rate": sum(c["cache_hit"] for c in calls) / len(calls),
        **run["scores"]["summary"],
    }


def mean_sd(vals):
    return {"mean": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else 0.0}


def summarize(configs, n_tickets):
    out = {}
    for name, runs in configs.items():
        ms = [run_metrics(r, n_tickets) for r in runs]
        out[name] = {"runs": len(runs), **{k: mean_sd([m[k] for m in ms]) for k in ms[0]}}
    base = out["crawl"]["cost_per_ticket"]["mean"]
    for v in out.values():
        v["savings"] = 1 - v["cost_per_ticket"]["mean"] / base
    return out


def walk(runs, tickets):
    """Attribution over the crawl traces: where the money goes, and which tickets trip the alert."""
    calls = [c for r in runs for c in r["calls"]]
    total = sum(c["cost_usd"] for c in calls)
    by_feature = {}
    for f in FEATURES:
        fc = [c for c in calls if c["feature"] == f]
        cost = sum(c["cost_usd"] for c in fc)
        by_feature[f] = {
            "cost_share": cost / total,
            "avg_input": st.mean(c["input_tokens"] for c in fc),
            "avg_output": st.mean(c["output_tokens"] for c in fc),
            "avg_cost": cost / len(fc),
        }
    in_cost = sum(cost_usd(c["model"], c["input_tokens"], 0) for c in calls)

    turns = {t["id"]: len(t["thread"]) for t in tickets}
    user_of = {t["id"]: t["user_id"] for t in tickets}
    tickets_per_user = defaultdict(int)
    for t in tickets:
        tickets_per_user[t["user_id"]] += 1
    by_user = defaultdict(float)
    for c in calls:
        by_user[c["user_id"]] += c["cost_usd"]
    by_user = sorted(({"user": u, "cost_share": v / total, "ticket_share": tickets_per_user[u] / len(tickets)}
                      for u, v in by_user.items()), key=lambda x: -x["cost_share"])

    reply_by_turns = defaultdict(list)
    for c in calls:
        if c["feature"] == "reply":
            reply_by_turns[turns[c["ticket_id"]]].append(c["input_tokens"])
    reply_by_turns = {k: st.mean(v) for k, v in sorted(reply_by_turns.items())}

    # Alert check, as Walk would run it: baseline median from run 1, applied to every ticket of every run.
    def ticket_costs(run):
        tc = defaultdict(float)
        for c in run["calls"]:
            tc[c["ticket_id"]] += c["cost_usd"]
        return tc
    median = st.median(ticket_costs(runs[0]).values())
    flagged = defaultdict(int)
    for r in runs:
        for tid, v in ticket_costs(r).items():
            if v > ANOMALY_X * median:
                flagged[tid] += 1
    anomalies = [{"ticket": t, "user": user_of[t], "turns": turns[t], "runs_flagged": n}
                 for t, n in sorted(flagged.items())]
    return {"by_feature": by_feature, "input_cost_share": in_cost / total, "by_user": by_user,
            "reply_input_by_turns": reply_by_turns, "median_ticket_cost": median, "anomalies": anomalies}


def alignment(configs):
    """Quality vs cost for each feature on each model, from the crawl (big) and run-route (small) configs."""
    rows = []
    for name in ("crawl", "run-route"):
        for f in ("classify", "extract"):
            if name not in configs:
                continue
            calls = [c for r in configs[name] for c in r["calls"] if c["feature"] == f]
            q = st.mean(r["scores"]["summary"][f] for r in configs[name])
            rows.append({"config": name, "feature": f, "model": calls[0]["model"], "quality": q,
                         "cost_per_call": st.mean(c["cost_usd"] for c in calls)})
    return rows


def usd(x, digits=5):
    return f"${x:.{digits}f}"


def write_markdown(summary, w, est, align, path):
    L = ["# Suki Inbox results", "", "Generated by `python3 -m suki.report` from the trace files. "
         "Mean of N runs, ± one standard deviation. Costs use Ollama Cloud list prices in `pricing.json`.", "",
         "A ticket is 3 calls (classify, extract, reply). FinOps \"cost per inference\" is the per-call column.", "",
         "| Config | Runs | Cost/ticket | Cost/call | Savings | Tokens/ticket | Quality | Classify | Extract | Reply facts | p95 latency/ticket | Cache hits |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in CONFIG_ORDER:
        if name not in summary:
            continue
        s = summary[name]
        L.append(f"| {LABELS[name]} | {s['runs']} | {usd(s['cost_per_ticket']['mean'])} ± {s['cost_per_ticket']['sd']:.5f} "
                 f"| {usd(s['cost_per_call']['mean'], 6)} | {s['savings']:+.0%} | {s['tokens_per_ticket']['mean']:,.0f} | {s['quality']['mean']:.3f} ± {s['quality']['sd']:.3f} "
                 f"| {s['classify']['mean']:.3f} | {s['extract']['mean']:.3f} | {s['reply']['mean']:.3f} "
                 f"| {s['p95_ms']['mean'] / 1000:.1f}s | {s['cache_hit_rate']['mean']:.0%} |")
    L += ["", "## Walk: attribution over the crawl baseline", "",
          f"Input tokens are {w['input_cost_share']:.0%} of crawl spend.", "",
          "| Feature | Share of spend | Avg input tokens | Avg output tokens | Avg cost/call |", "|---|---|---|---|---|"]
    for f, v in w["by_feature"].items():
        L.append(f"| {f} | {v['cost_share']:.0%} | {v['avg_input']:,.0f} | {v['avg_output']:,.0f} | {usd(v['avg_cost'], 6)} |")
    L += ["", "| User | Share of spend | Share of tickets |", "|---|---|---|"]
    L += [f"| {u['user']} | {u['cost_share']:.0%} | {u['ticket_share']:.0%} |" for u in w["by_user"][:5]]
    L += ["", "Reply input tokens by thread length (turns): " +
          ", ".join(f"{k} turns: {v:,.0f}" for k, v in w["reply_input_by_turns"].items()), "",
          f"Alert rule: ticket cost > {ANOMALY_X:g}x baseline median ({usd(w['median_ticket_cost'], 6)}). "
          f"Flagged: {', '.join(a['ticket'] + ' (' + a['user'] + ')' for a in w['anomalies']) or 'none'}.", "",
          "## Estimate before running (FinOps Foundation, Cost Estimation of AI Workloads)", "",
          f"Measured crawl cost for {estimate.N_TICKETS} tickets: {usd(est['actual'], 4)}. Target accuracy in the paper: over 90%.", "",
          "| Estimate | Value | Accuracy |", "|---|---|---|",
          f"| Naive (prompt length / 4, visible answer length) | {usd(est['naive']['estimate'], 4)} | {est['naive']['accuracy']:.0%} |",
          f"| Pilot ({est['pilot']['tickets']} tickets, scaled) | {usd(est['pilot']['estimate'], 4)} | {est['pilot']['accuracy']:.0%} |", "",
          f"Input tokens: estimated {est['tokens']['input_estimated']:,.0f}, measured {est['tokens']['input_actual']:,.0f}. "
          f"Output tokens: estimated {est['tokens']['output_estimated']:,.0f}, measured {est['tokens']['output_actual']:,.0f}. "
          "Reasoning tokens bill as output and are not in the visible answer.", "",
          "## Model choice quality alignment (FinOps KPI)", "",
          "| Feature | Model | Quality | Cost/call |", "|---|---|---|---|"]
    L += [f"| {a['feature']} | {a['model']} | {a['quality']:.3f} | {usd(a['cost_per_call'], 6)} |" for a in align] + [""]
    path.write_text("\n".join(L))


def write_dashboard(summary, w, est, align, path):
    e = html.escape
    names = [n for n in CONFIG_ORDER if n in summary]

    def saved(n):
        return "" if n == "crawl" else format(summary[n]["savings"], "+.0%")
    base, run = summary["crawl"], summary.get("run", summary["crawl"])

    def tile(label, a, b, fmt, lower_is_better=True):
        delta = (b - a) / a if a else 0
        good = (delta < 0) == lower_is_better
        sign = "▼" if delta < 0 else "▲"
        return (f'<div class="tile"><div class="tl">{e(label)}</div><div class="tv">{fmt(b)}</div>'
                f'<div class="ts">crawl {fmt(a)} <span class="{"good" if good else "bad"}">{sign} {abs(delta):.0%}</span></div></div>')

    tiles = "".join([
        tile("Cost per ticket", base["cost_per_ticket"]["mean"], run["cost_per_ticket"]["mean"], lambda x: f"${x * 1000:.3f}<small>/1k</small>"),
        tile("Tokens per ticket", base["tokens_per_ticket"]["mean"], run["tokens_per_ticket"]["mean"], lambda x: f"{x:,.0f}"),
        tile("Quality", base["quality"]["mean"], run["quality"]["mean"], lambda x: f"{x:.3f}", lower_is_better=False),
        tile("p95 latency / ticket", base["p95_ms"]["mean"], run["p95_ms"]["mean"], lambda x: f"{x / 1000:.1f}s"),
    ])

    maxc = max(summary[n]["cost_per_ticket"]["mean"] for n in names)
    lever_rows = "".join(
        f'<div class="row" data-tip="{e(LABELS[n])}: ${summary[n]["cost_per_ticket"]["mean"] * 1000:.3f} per 1k tickets, '
        f'quality {summary[n]["quality"]["mean"]:.3f}, {summary[n]["runs"]} runs">'
        f'<div class="lbl">{e(LABELS[n])}</div><div class="track"><div class="bar{" hi" if n == "run" else ""}" '
        f'style="width:{summary[n]["cost_per_ticket"]["mean"] / maxc * 100:.1f}%"></div></div>'
        f'<div class="val">{saved(n)}</div>'
        f'<div class="q">{summary[n]["quality"]["mean"]:.3f}</div></div>' for n in names)

    feat = "".join(
        f'<div class="seg s{i}" style="width:{w["by_feature"][f]["cost_share"] * 100:.2f}%" '
        f'data-tip="{f}: {w["by_feature"][f]["cost_share"]:.0%} of spend, avg {w["by_feature"][f]["avg_input"]:,.0f} in / '
        f'{w["by_feature"][f]["avg_output"]:,.0f} out tokens">{f} {w["by_feature"][f]["cost_share"]:.0%}</div>'
        for i, f in enumerate(FEATURES))
    calls_bar = "".join(f'<div class="seg s{i}" style="width:33.33%" data-tip="{f}: 1 of 3 calls per ticket">{f} 33%</div>'
                        for i, f in enumerate(FEATURES))

    maxu = w["by_user"][0]["cost_share"]
    users = "".join(
        f'<div class="row" data-tip="{u["user"]}: {u["cost_share"]:.0%} of spend from {u["ticket_share"]:.0%} of tickets">'
        f'<div class="lbl mono">{u["user"]}</div><div class="track"><div class="bar{" hi" if i == 0 else ""}" '
        f'style="width:{u["cost_share"] / maxu * 100:.1f}%"></div></div><div class="val">{u["cost_share"]:.0%}</div>'
        f'<div class="q">{u["ticket_share"]:.0%}</div></div>' for i, u in enumerate(w["by_user"][:6]))

    maxt = max(w["reply_input_by_turns"].values())
    turns = "".join(
        f'<div class="row" data-tip="{k}-turn threads: {v:,.0f} input tokens per reply call">'
        f'<div class="lbl">{k} turn{"s" if k > 1 else ""}</div><div class="track"><div class="bar" '
        f'style="width:{v / maxt * 100:.1f}%"></div></div><div class="val">{v:,.0f}</div><div class="q"></div></div>'
        for k, v in w["reply_input_by_turns"].items())

    table = "".join(
        f'<tr><td>{e(LABELS[n])}</td><td>{summary[n]["runs"]}</td>'
        f'<td>${summary[n]["cost_per_ticket"]["mean"] * 1000:.3f} ± {summary[n]["cost_per_ticket"]["sd"] * 1000:.3f}</td>'
        f'<td>{saved(n)}</td>'
        f'<td>{summary[n]["tokens_per_ticket"]["mean"]:,.0f}</td>'
        f'<td>{summary[n]["quality"]["mean"]:.3f} ± {summary[n]["quality"]["sd"]:.3f}</td>'
        f'<td>{summary[n]["classify"]["mean"]:.2f} / {summary[n]["extract"]["mean"]:.2f} / {summary[n]["reply"]["mean"]:.2f}</td>'
        f'<td>{summary[n]["p95_ms"]["mean"] / 1000:.1f}s</td><td>{summary[n]["cache_hit_rate"]["mean"]:.0%}</td></tr>'
        for n in names)

    est_rows = "".join(
        f'<div class="row" data-tip="{e(lbl)}: ${v["estimate"]:.4f} estimated vs ${est["actual"]:.4f} measured">'
        f'<div class="lbl">{e(lbl)}</div><div class="track"><div class="bar{" hi" if hi else ""}" '
        f'style="width:{min(v["estimate"] / est["actual"], 1.2) / 1.2 * 100:.1f}%"></div></div>'
        f'<div class="val">{v["accuracy"]:.0%}</div><div class="q"></div></div>'
        for lbl, v, hi in (("Naive estimate", est["naive"], False), ("10-ticket pilot", est["pilot"], True)))
    align_rows = "".join(
        f'<tr><td>{a["feature"]}</td><td class="mono">{e(a["model"])}</td><td>{a["quality"]:.3f}</td>'
        f'<td>${a["cost_per_call"] * 1000:.3f} / 1k calls</td></tr>' for a in align)
    anomalies = ", ".join(f'{a["ticket"]} ({a["user"]}, {a["turns"]} turns)' for a in w["anomalies"]) or "none"

    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Suki Inbox Costs</title>
<style>
:root{{--bg:#0f1420;--panel:#171e2e;--line:#2a3448;--text:#e8edf5;--muted:#93a0b8;--crawl:#5fb3f9;--walk:#7dd87d;--run:#f5a524;
--s0:#3987e5;--s1:#d95926;--s2:#199e70;--good:#7dd87d;--bad:#ff7b72}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Inter,system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}h1{{font-size:26px;margin:0 0 4px}}
h2{{font-size:17px;margin:36px 0 4px}}.sub{{color:var(--muted);margin:0 0 14px;font-size:14px}}
.mono{{font-family:"JetBrains Mono",ui-monospace,monospace}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:20px}}
.tile{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
.tl{{color:var(--muted);font-size:13px}}.tv{{font-size:28px;font-weight:700;margin:4px 0}}.tv small{{font-size:14px;color:var(--muted);font-weight:400}}
.ts{{color:var(--muted);font-size:13px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}
.row{{display:grid;grid-template-columns:170px 1fr 56px 56px;gap:10px;align-items:center;padding:4px 0}}
.lbl{{font-size:14px}}.val,.q{{text-align:right;font-variant-numeric:tabular-nums;font-size:14px}}.q{{color:var(--muted)}}
.track{{height:18px}}.bar{{height:100%;background:var(--muted);border-radius:0 4px 4px 0;min-width:2px}}.bar.hi{{background:var(--run)}}
.head{{color:var(--muted);font-size:12px}}
.stack{{display:flex;gap:2px;height:30px;margin:6px 0 14px}}.seg{{border-radius:4px;color:#fff;font-size:13px;padding:5px 8px;white-space:nowrap;overflow:hidden}}
.s0{{background:var(--s0)}}.s1{{background:var(--s1)}}.s2{{background:var(--s2)}}
table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:8px;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-weight:500}}.scroll{{overflow-x:auto}}
#tip{{position:fixed;pointer-events:none;background:#0a0e17;border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:13px;display:none;max-width:320px;z-index:9}}
[data-tip]:hover{{filter:brightness(1.15)}}
@media(max-width:600px){{.row{{grid-template-columns:110px 1fr 44px 44px}}}}
</style></head><body><main>
<h1>Suki Inbox: cost per ticket, Crawl to Run</h1>
<p class="sub">Reference app on {sum(s["runs"] for s in summary.values())} recorded runs of 60 synthetic Taglish support tickets.
Ollama Cloud list prices. Generated offline from trace files.</p>
<div class="tiles">{tiles}</div>

<h2>Run levers: cost per ticket</h2><p class="sub">Each lever alone, then combined. Right columns: savings vs crawl, quality.</p>
<div class="card"><div class="row head"><div></div><div></div><div class="val">saved</div><div class="q">quality</div></div>{lever_rows}</div>

<h2>Walk: where crawl spend goes</h2>
<p class="sub">Every feature is one call per ticket, but they do not cost the same. Input tokens are {w["input_cost_share"]:.0%} of spend.</p>
<div class="card"><div class="head">Share of calls</div><div class="stack">{calls_bar}</div>
<div class="head">Share of spend</div><div class="stack">{feat}</div></div>

<h2>Walk: spend by customer</h2><p class="sub">Right columns: share of spend, share of tickets.</p>
<div class="card">{users}</div>

<h2>Walk: reply input tokens by thread length</h2><p class="sub">Crawl resends the full policy and the full history on every call.</p>
<div class="card">{turns}</div>
<p class="sub">Alert rule: ticket cost over {ANOMALY_X:g}x the baseline median. Flagged: {e(anomalies)}.</p>

<h2>Estimate before running</h2>
<p class="sub">FinOps Cost Estimation method: tokens per request × volume × price. Bar is the estimate as a share of measured cost; right column is accuracy (paper target: over 90%).
Input tokens: {est["tokens"]["input_estimated"]:,.0f} estimated, {est["tokens"]["input_actual"]:,.0f} measured. Output: {est["tokens"]["output_estimated"]:,.0f} estimated, {est["tokens"]["output_actual"]:,.0f} measured, because reasoning tokens bill as output.</p>
<div class="card">{est_rows}</div>

<h2>Model choice quality alignment</h2><p class="sub">Does the cheaper model hold quality on the easy features? (FinOps KPI)</p>
<div class="card scroll"><table><tr><th>Feature</th><th>Model</th><th>Quality</th><th>Cost</th></tr>{align_rows}</table></div>

<h2>All configs</h2><div class="card scroll"><table>
<tr><th>Config</th><th>Runs</th><th>$ / 1k tickets</th><th>Saved</th><th>Tokens/ticket</th><th>Quality</th><th>Classify / extract / reply</th><th>p95/ticket</th><th>Cache hits</th></tr>
{table}</table></div>
<p class="sub">Quality: classify exact match, extract field match, reply share of required policy facts mentioned. Hand-labeled, no LLM judge.</p>
</main><div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.addEventListener('mousemove',ev=>{{const t=ev.target.closest('[data-tip]');
if(!t){{tip.style.display='none';return}}tip.textContent=t.dataset.tip;tip.style.display='block';
tip.style.left=Math.min(ev.clientX+14,innerWidth-330)+'px';tip.style.top=(ev.clientY+14)+'px'}});
</script></body></html>""")


def main():
    tickets = load_tickets()
    configs = {n: r for n in CONFIG_ORDER if (r := load(n))}
    summary = summarize(configs, len(tickets))
    w = walk(configs["crawl"], tickets)
    est = estimate.compare(tickets, configs["crawl"])
    align = alignment(configs)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"configs": summary, "walk": w, "estimate": est, "alignment": align}, indent=1))
    write_markdown(summary, w, est, align, out / "results.md")
    write_dashboard(summary, w, est, align, out / "dashboard.html")
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main()
