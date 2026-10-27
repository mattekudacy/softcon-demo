"""Suki Inbox: triage for a small online shop's Taglish customer inbox.

Three features per ticket: classify (intent), extract (order fields), reply (draft answer).
"""
import json
import re

from . import llm
from .llm import ROOT

BIG, SMALL = "gpt-oss:120b", "gpt-oss:20b"
POLICY = (ROOT / "data" / "policy.md").read_text()
INTENTS = ["order_status", "return_refund", "change_order", "cancel_order",
           "payment", "shipping_fee", "product_question", "other"]

CRAWL = {"models": {"classify": BIG, "extract": BIG, "reply": BIG}}
LEVERS = {
    # Small model for classify and extract, and for replies to low-risk intents.
    "route": {"models": {"classify": SMALL, "extract": SMALL, "reply": BIG}, "route_reply": True},
    # Low reasoning effort where the answer is a label or a JSON object.
    "think": {"think": {"classify": "low", "extract": "low"}},
    # Only the policy sections the intent needs, and only the last 3 turns.
    "trim": {"trim": True},
    # App-level cache: identical requests in the same run are served for free.
    "cache": {"cache": True},
}
# Crawl: the prototype. Best model everywhere, whole policy in every prompt, full history.
# Walk makes no new calls; it adds attribution and alerts on top of the crawl traces (suki/report.py).
# Run: each lever alone, then all together.
CONFIGS = {"crawl": CRAWL}
CONFIGS.update({f"run-{k}": {**CRAWL, **v} for k, v in LEVERS.items()})
CONFIGS["run"] = {**CRAWL, **{k: v for lever in LEVERS.values() for k, v in lever.items()}}

EASY_INTENTS = {"order_status", "shipping_fee", "payment", "product_question"}
SECTIONS_FOR = {
    "order_status": ["Shipping"],
    "shipping_fee": ["Shipping", "Shipping fees"],
    "payment": ["Payment"],
    "change_order": ["Order changes and cancellation"],
    "cancel_order": ["Order changes and cancellation", "Returns and refunds"],
    "return_refund": ["Returns and refunds", "Warranty"],
    "product_question": ["Products", "Warranty"],
    "other": ["Shipping", "Store hours and contact"],
}
SECTIONS = {b.split("\n", 1)[0].strip(): "## " + b.strip() for b in POLICY.split("\n## ")[1:]}

SYSTEM_FULL = f"You are Suki, the customer support assistant for Bahay Basics.\n\n{POLICY}"
SYSTEM_SHORT = "You are a support triage assistant for Bahay Basics, a small online home goods shop."


def system_for_reply(intent):
    picked = "\n\n".join(SECTIONS[s] for s in SECTIONS_FOR[intent] + ["Tone"])
    return f"You are Suki, the customer support assistant for Bahay Basics.\n\n{picked}"


def render(thread):
    return "\n".join(f"{m['role'].upper()}: {m['text']}" for m in thread)


def classify_prompt(thread):
    return (f"Classify the intent of the customer's latest message.\n"
            f"Labels: {', '.join(INTENTS)}.\n"
            f"order_status = where is my order or delivery time for an existing order. "
            f"shipping_fee = shipping cost or delivery time before buying. "
            f"return_refund = returns, refunds, damaged or wrong items, warranty claims. "
            f"other = anything else.\n"
            f"Answer with the label only.\n\nConversation:\n{render(thread)}")


def extract_prompt(thread):
    return ("Extract fields from the conversation as JSON with keys order_id, item, qty.\n"
            "order_id looks like SK-12345. item is the product the request is about. "
            "qty is the quantity the customer wants. Use null when not mentioned.\n"
            f"Return JSON only.\n\nConversation:\n{render(thread)}")


def reply_prompt(thread, known=""):
    return ("Write the shop's next reply to the customer. Follow the policy exactly.\n"
            f"Return only the reply text.\n{known}\nConversation:\n{render(thread)}")


def parse_intent(text):
    t = text.strip().lower()
    for i in INTENTS:
        if i in t:
            return i
    return "other"


def parse_fields(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def handle(ticket, cfg, tracer):
    """Runs all three features on one ticket. Returns parsed outputs."""
    tid, uid, thread = ticket["id"], ticket["user_id"], ticket["thread"]
    trim, think = cfg.get("trim", False), cfg.get("think", {})
    cache = tracer.cache if cfg.get("cache") else None

    def ask(feature, system, prompt, model=None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return llm.call(tracer, feature, model or cfg["models"][feature], msgs, ticket_id=tid, user_id=uid,
                        think=think.get(feature), cache=cache)

    recent = thread[-3:] if trim else thread
    intent = parse_intent(ask("classify", SYSTEM_SHORT if trim else SYSTEM_FULL, classify_prompt(recent)))
    fields = parse_fields(ask("extract", SYSTEM_SHORT if trim else SYSTEM_FULL, extract_prompt(thread)))

    reply_model = SMALL if cfg.get("route_reply") and intent in EASY_INTENTS else None
    if trim:
        oid = fields.get("order_id") if isinstance(fields, dict) else None
        known = f"Known order ID from earlier in the chat: {oid}\n" if oid and len(thread) > 3 else ""
        reply = ask("reply", system_for_reply(intent), reply_prompt(recent, known), reply_model)
    else:
        reply = ask("reply", SYSTEM_FULL, reply_prompt(thread), reply_model)
    return {"intent": intent, "fields": fields, "reply": reply}
