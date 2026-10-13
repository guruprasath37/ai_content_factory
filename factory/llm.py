"""LLM access for the content factory: local Ollama by default, or any OpenAI-compatible API.

Providers are tried in order and the next one is used if a call fails (rate limit, outage, bad key):
  LLM_PROVIDERS  comma list, default "groq,ollama"  (groq|openai|ollama; ones without credentials are skipped)
  GROQ_API_KEY / GROQ_MODELS (comma list, tried in order; default openai/gpt-oss-120b,openai/gpt-oss-20b - Groq limits each model
  separately, so the second is real extra capacity) / GROQ_REASONING (low|medium|high, default low)
Env:
  OLLAMA_URL     default http://localhost:11434
  OLLAMA_MODEL   default: best installed model (prefers qwen/llama/mistral/phi over gemma)
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
"""
from __future__ import annotations
import json, os, re
from typing import Any, Callable

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
PREFERRED = ("qwen2.5", "qwen3", "llama3", "mistral", "phi4", "phi3", "gemma2", "gemma3", "gemma")


GROQ_URL = "https://api.groq.com/openai/v1"
LAST: dict = {"provider": None, "model": None, "fell_back_from": None}   # what answered the most recent call


def chain() -> list[str]:
    """Configured providers that have credentials, in order. Ollama is always kept as the last resort."""
    legacy = os.getenv("LLM_PROVIDER", "").lower()
    names = [x.strip().lower() for x in os.getenv("LLM_PROVIDERS", legacy or "groq,openai,ollama").split(",") if x.strip()]
    ok = {"groq": bool(os.getenv("GROQ_API_KEY", "").strip()), "openai": bool(os.getenv("OPENAI_API_KEY", "").strip()), "ollama": True}
    out = [n for n in names if ok.get(n)]
    if "ollama" not in out: out.append("ollama")
    return out


def provider() -> str:
    return chain()[0]


def ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3); r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_model() -> str | None:
    if os.getenv("OLLAMA_MODEL"): return os.getenv("OLLAMA_MODEL")
    names = ollama_models()
    for pref in PREFERRED:
        for n in names:
            if n.startswith(pref): return n
    return names[0] if names else None


def groq_models() -> list[str]:
    return [m.strip() for m in os.getenv("GROQ_MODELS", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b,openai/gpt-oss-20b")).split(",") if m.strip()]


def _attempts() -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for p in chain():
        out += [(p, m) for m in groq_models()] if p == "groq" else [(p, _model_for(p))]
    return out


def _model_for(p: str) -> str | None:
    return {"groq": groq_models()[0], "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "ollama": ollama_model()}[p]


def status() -> dict[str, Any]:
    c = chain(); first = c[0]
    return {"provider": first, "model": _model_for(first), "ready": bool(_model_for(first)) or len(c) > 1, "chain": c,
            "models": [f"{p}:{m}" for p, m in _attempts()], "installed": ollama_models(), "last_used": LAST["provider"], "last_model": LAST["model"]}


def _openai_style(base: str, key: str, model: str, messages: list[dict], temperature: float, json_mode: bool, extra: dict | None = None) -> str:
    body = {"model": model, "messages": messages, "temperature": temperature, **(extra or {})}
    if json_mode: body["response_format"] = {"type": "json_object"}
    for attempt in range(2):
        r = requests.post(f"{base}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=int(os.getenv("LLM_TIMEOUT", "120")))
        if r.status_code == 400 and attempt == 0 and ("reasoning" in r.text or "response_format" in r.text):
            body.pop("reasoning_effort", None); body.pop("response_format", None); continue   # model doesn't support an optional param
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"] or ""
    raise RuntimeError("request rejected")


def _call(p: str, model: str | None, messages: list[dict], temperature: float, json_mode: bool) -> str:
    if p == "groq":
        extra = {"reasoning_effort": os.getenv("GROQ_REASONING", "low")} if "gpt-oss" in model else {}
        return _openai_style(GROQ_URL, os.getenv("GROQ_API_KEY", ""), model, messages, temperature, json_mode, extra)
    if p == "openai":
        return _openai_style((os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"), os.getenv("OPENAI_API_KEY", ""), _model_for("openai"), messages, temperature, json_mode)
    if not model: raise RuntimeError("Ollama has no models. Run: ollama pull qwen2.5:7b")
    body = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature, "num_ctx": 4096}, "keep_alive": "30m"}
    if json_mode: body["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=int(os.getenv("OLLAMA_TIMEOUT", "600")))
    r.raise_for_status()
    return r.json()["message"]["content"]


def _raw_chat(messages: list[dict], temperature: float, json_mode: bool) -> str:
    """Try each provider/model in turn; a failure (429, outage, bad key, timeout) moves on to the next one."""
    errors, first = [], None
    for p, model in _attempts():
        try:
            text = _call(p, model, messages, temperature, json_mode)
            LAST.update(provider=p, model=model, fell_back_from=first)
            return text
        except Exception as e:
            first = first or f"{p}:{model}"; errors.append(f"{p}:{model}: {str(e)[:140]}")
    raise RuntimeError("all LLM providers failed - " + " | ".join(errors))


def parse_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)   # tolerate chatter around the JSON
        if not m: raise
        return json.loads(m.group(0))


def chat_json(system: str, user: str, validate: Callable[[Any], Any] | None = None, temperature: float = 0.8, tries: int = 3) -> Any:
    """Ask for JSON; if parsing or `validate` (which may fix up and return the data, or raise ValueError) fails,
    tell the model what was wrong and retry. Small local models need this."""
    messages = [{"role": "system", "content": system + "\nReturn ONLY valid JSON. No Markdown, no commentary."}, {"role": "user", "content": user}]
    last: Exception | None = None
    for attempt in range(tries):
        text = ""
        try:
            text = _raw_chat(messages, max(0.2, temperature - 0.2 * attempt), True)
            data = parse_json(text)
            return validate(data) if validate else data
        except (ValueError, KeyError, TypeError) as e:     # JSONDecodeError is a ValueError
            last = e
            messages += [{"role": "assistant", "content": text or "{}"},
                         {"role": "user", "content": f"That was rejected: {e}. Fix it and return the complete corrected JSON only."}]
    raise ValueError(f"LLM did not return usable JSON after {tries} tries: {last}")
