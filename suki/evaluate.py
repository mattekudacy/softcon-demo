"""Scores outputs against hand labels. No LLM judge, so scores are free and reproducible.

classify: exact intent match.
extract:  share of order_id, item, qty that match.
reply:    share of required policy facts mentioned (any listed alternative counts).
quality:  mean of the three.
"""


# Models emit narrow no-break spaces and non-breaking hyphens ("2\u202fPM", "cut\u2011off").
_ASCII = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2011": "-", "\u2010": "-", "\u2013": "-"})


def _norm(x):
    return str(x).strip().lower() if x not in (None, "", "null") else None


def score_extract(t, f):
    oid = _norm(f.get("order_id"))
    ok_oid = oid == _norm(t["order_id"])
    item = _norm(f.get("item"))
    ok_item = (item is None) if not t["item"] else (item is not None and any(a in item for a in t["item"]))
    try:
        qty = int(f.get("qty")) if f.get("qty") not in (None, "", "null") else None
    except (TypeError, ValueError):
        qty = None
    ok_qty = qty == t["qty"]
    return (ok_oid + ok_item + ok_qty) / 3


def score_reply(t, reply):
    if not t["facts"]:
        return None
    r = reply.translate(_ASCII).lower()
    return sum(any(a.translate(_ASCII).lower() in r for a in group) for group in t["facts"]) / len(t["facts"])


def score(tickets, outputs):
    rows = {}
    for t in tickets:
        o = outputs[t["id"]]
        rows[t["id"]] = {
            "classify": float(o["intent"] == t["intent"]),
            "extract": score_extract(t, o["fields"] if isinstance(o["fields"], dict) else {}),
            "reply": score_reply(t, o["reply"]),
        }
    summary = {}
    for k in ("classify", "extract", "reply"):
        vals = [r[k] for r in rows.values() if r[k] is not None]
        summary[k] = round(sum(vals) / len(vals), 3)
    summary["quality"] = round(sum(summary.values()) / 3, 3)
    return {"summary": summary, "tickets": rows}
