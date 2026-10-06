"""vLLM backend injection for the upstream AuditBench v2 harness.

The upstream `inference_helpers.call_model` routes `--llm gpt` through the OpenAI
*Responses* API with no base_url, no temperature and no reasoning_effort. We keep
the upstream checkout pristine and instead monkeypatch that single network call so
that, for our study, every AuditBench query goes to our local vLLM endpoint via
chat.completions under the PLAN's fairness controls:

  - base_url / model from the .env (aligned or ablated endpoint),
  - reasoning_effort fixed (high), temperature fixed, max_tokens fixed,
  - a reproducible per-prompt seed sequence so the N attempts differ but replay
    identically across the aligned and ablated runs (only the weights differ),
  - the analysis/reasoning channel captured to a trace JSONL for the §6.2
    "knew but withheld" vs "never identified" labelling.

Importing the upstream `inference_helpers` pulls in the Gemini client, whose module
body runs `vertexai.init(project="YOUR_PROJECT_ID")` at import time. We only use the
GPT path, so we install lightweight stubs for the Gemini/Together clients in
sys.modules before that import. Nothing else in the upstream is altered.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# 1. Stub the unused provider clients so `inference_helpers` imports cleanly.
# --------------------------------------------------------------------------- #
def install_provider_stubs() -> None:
    """Register dummy llm_base.interact_{gemini,llama} modules.

    Must run before `import inference_helpers`. Safe to call more than once.
    """
    for mod_name, func_name in (
        ("llm_base.interact_gemini", "call_gemini"),
        ("llm_base.interact_llama", "call_llama"),
    ):
        if mod_name in sys.modules:
            continue
        stub = types.ModuleType(mod_name)

        def _unavailable(*_a, _mod=mod_name, **_k):
            raise RuntimeError(
                f"{_mod} is stubbed in the cyber-offdef-gap harness; only the "
                f"vLLM/gpt path is supported."
            )

        setattr(stub, func_name, _unavailable)
        sys.modules[mod_name] = stub
        # bind as an attribute on the package too, so `from llm_base import X` works
        pkg = sys.modules.get("llm_base")
        if pkg is not None:
            setattr(pkg, mod_name.split(".")[-1], stub)


# --------------------------------------------------------------------------- #
# 2. The patched network call.
# --------------------------------------------------------------------------- #
_CFG: Dict[str, Any] = {}
_LOCK = threading.Lock()
_SEED_COUNTER: Dict[str, int] = {}
_TRACE_PATH: Path | None = None


def configure(
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    reasoning_effort: str,
    max_tokens: int,
    base_seed: int,
    trace_path: Path,
    timeout: float = 1800.0,
) -> None:
    """Set the parity knobs and open the trace sink. Call once per run."""
    global _TRACE_PATH
    _CFG.update(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
        temperature=float(temperature),
        reasoning_effort=reasoning_effort,
        max_tokens=int(max_tokens),
        base_seed=int(base_seed),
        timeout=float(timeout),
    )
    _TRACE_PATH = Path(trace_path)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SEED_COUNTER.clear()


def _next_seed(prompt_sha: str) -> int:
    """Reproducible distinct seed per attempt of a given prompt.

    base_seed + k for the k-th call on an identical prompt. Because the prompt
    bytes are identical across the aligned and ablated runs, the seed sequence is
    identical too, so the only intended difference between the pair is the weights.
    """
    with _LOCK:
        k = _SEED_COUNTER.get(prompt_sha, 0)
        _SEED_COUNTER[prompt_sha] = k + 1
    return _CFG["base_seed"] + k


def patched_call_model(llm: str, model: str, message: str, temp: float) -> str:
    """Drop-in replacement for inference_helpers.call_model (gpt path -> vLLM).

    Returns the assistant content string exactly as the upstream expects (the
    upstream appends it to the per-scenario output .txt that the metrics script
    parses). The `model`/`temp`/`llm` the upstream passes are ignored in favour of
    the configured parity values; the upstream model arg is used only for output
    file naming, which the orchestrator sets separately.
    """
    from openai import OpenAI  # imported lazily so stubs/venv are in place

    if not _CFG:
        raise RuntimeError("vllm_backend.configure() was not called before inference")

    prompt_sha = hashlib.sha256(message.encode("utf-8")).hexdigest()
    seed = _next_seed(prompt_sha)
    client = OpenAI(base_url=_CFG["base_url"], api_key=_CFG["api_key"], timeout=_CFG["timeout"])

    messages: List[Dict[str, str]] = [{"role": "user", "content": message}]
    t0 = time.time()
    resp = client.chat.completions.create(
        model=_CFG["model"],
        messages=messages,
        temperature=_CFG["temperature"],
        max_tokens=_CFG["max_tokens"],
        seed=seed,
        extra_body={"reasoning_effort": _CFG["reasoning_effort"]},
    )
    latency = time.time() - t0

    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    usage = getattr(resp, "usage", None)
    usage_d = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}

    _append_trace(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_url": _CFG["base_url"],
            "served_model": _CFG["model"],
            "reasoning_effort": _CFG["reasoning_effort"],
            "temperature": _CFG["temperature"],
            "max_tokens": _CFG["max_tokens"],
            "seed": seed,
            "prompt_sha256": prompt_sha,
            "prompt_chars": len(message),
            "finish_reason": resp.choices[0].finish_reason,
            "latency_s": round(latency, 2),
            "usage": usage_d,
            "reasoning_content": reasoning,
            "content": content,
        }
    )
    return content


def _append_trace(record: Dict[str, Any]) -> None:
    if _TRACE_PATH is None:
        return
    with _LOCK:
        with _TRACE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
