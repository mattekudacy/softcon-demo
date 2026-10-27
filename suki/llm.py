"""LLM wrapper: every call gets a price and a trace record.

Record and replay share one code path. In record mode a request missing from
recordings/ is sent to Ollama Cloud and saved. In replay mode (the default) it
is served from recordings/ and never touches the network, so the numbers on
stage are the numbers from the recorded run.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICING = json.loads((ROOT / "pricing.json").read_text())["models"]
RECORDINGS = ROOT / "recordings" / "responses.jsonl"

MODE = os.environ.get("SUKI_MODE", "replay")
BUDGET_USD = float(os.environ.get("SUKI_BUDGET_USD", "2.00"))
HOST = "https://ollama.com"


def _api_key():
    """OLLAMA_API_KEY from the environment, else from a .env file in the repo root or its parent folder."""
    key = os.environ.get("OLLAMA_API_KEY")
    for env in (ROOT / ".env", ROOT.parent / ".env"):
        if not key and env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OLLAMA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
    return key


def cost_usd(model, input_tokens, output_tokens):
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1e6


def _key(model, messages, think, run):
    blob = json.dumps({"model": model, "messages": messages, "think": think, "run": run}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_recordings():
    store = {}
    if RECORDINGS.exists():
        for line in RECORDINGS.read_text().splitlines():
            r = json.loads(line)
            store[r["key"]] = r
    return store


_store = _load_recordings()
_spent = sum(cost_usd(r["model"], r["input_tokens"], r["output_tokens"]) for r in _store.values())


def _post(model, messages, think):
    key = _api_key()
    if not key:
        raise RuntimeError("record mode calls https://ollama.com directly and needs OLLAMA_API_KEY "
                           "(environment or .env). The local Ollama daemon is not used.")
    body = {"model": model, "messages": messages, "stream": False}
    if think is not None:
        body["think"] = think
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    req = urllib.request.Request(f"{HOST}/api/chat", json.dumps(body).encode(), headers)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())
        except OSError as e:  # URLError, HTTPError and socket.timeout are all OSError
            code = getattr(e, "code", None)
            if code in (400, 401, 402, 404) or attempt == 5:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))


def _fetch(model, messages, think, run):
    """Returns the recorded response for this request, recording it first if needed."""
    global _spent
    key = _key(model, messages, think, run)
    if key in _store:
        return _store[key]
    if MODE != "record":
        raise RuntimeError(f"no recording for request {key}; run with SUKI_MODE=record")
    if _spent >= BUDGET_USD:
        raise RuntimeError(f"budget cap reached: ${_spent:.4f} >= ${BUDGET_USD:.2f}")
    t0 = time.time()
    d = _post(model, messages, think)
    rec = {
        "key": key,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "model": model,
        "think": think,
        "input_tokens": d.get("prompt_eval_count", 0),
        "output_tokens": d.get("eval_count", 0),
        "latency_ms": round((time.time() - t0) * 1000),
        "output": d["message"]["content"],
    }
    RECORDINGS.parent.mkdir(exist_ok=True)
    with RECORDINGS.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _store[key] = rec
    _spent += cost_usd(model, rec["input_tokens"], rec["output_tokens"])
    return rec


class Tracer:
    """Writes one JSONL record per call to traces/<config>/run<N>.jsonl."""

    def __init__(self, config, run):
        self.config, self.run = config, run
        self.cache = {}  # app-level response cache, fresh each run
        self.n = 0
        path = ROOT / "traces" / config / f"run{run}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w")

    def write(self, **rec):
        self.n += 1
        # One id per call, as in the FinOps Tracker schema. Deterministic so replayed traces match recorded ones.
        request_id = f"{self.config}-r{self.run}-{self.n:04d}"
        self.f.write(json.dumps({"request_id": request_id, "config": self.config, "run": self.run, **rec},
                                ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _cache_key(model, messages, think):
    text = " ".join(json.dumps([model, messages, think]).lower().split())
    return hashlib.sha256(text.encode()).hexdigest()


def call(tracer, feature, model, messages, *, ticket_id, user_id, think=None, cache=None):
    """The only way the app talks to a model. Returns the text output.

    cache: optional dict for an app-level response cache. A hit costs nothing and is traced as such.
    """
    if cache is not None:
        ck = _cache_key(model, messages, think)
        if ck in cache:
            tracer.write(
                ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), ticket_id=ticket_id, user_id=user_id,
                feature=feature, model=model, think=think, input_tokens=0, output_tokens=0, latency_ms=0,
                cost_usd=0.0, cache_hit=True, output=cache[ck],
            )
            return cache[ck]
    r = _fetch(model, messages, think, tracer.run)
    if cache is not None:
        cache[ck] = r["output"]
    tracer.write(
        ts=r["ts"], ticket_id=ticket_id, user_id=user_id, feature=feature, model=model, think=think,
        input_tokens=r["input_tokens"], output_tokens=r["output_tokens"], latency_ms=r["latency_ms"],
        cost_usd=cost_usd(model, r["input_tokens"], r["output_tokens"]), cache_hit=False, output=r["output"],
    )
    return r["output"]
