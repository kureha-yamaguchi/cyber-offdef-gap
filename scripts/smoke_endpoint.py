#!/usr/bin/env python3
"""M0 endpoint smoke + reasoning-effort/harmony parity check.
Usage: BASE_URL=https://<pod>-8000.proxy.runpod.net/v1 MODEL=openai/gpt-oss-120b python3 scripts/smoke_endpoint.py
Hits the vLLM OpenAI endpoint, confirms a well-formed chat completion, exercises
reasoning_effort low/high, and reports whether a reasoning/analysis channel is returned.
"""
import os, json, sys, urllib.request

BASE = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("MODEL", "openai/gpt-oss-120b")
KEY = os.environ.get("OPENAI_API_KEY", "sk-noauth")

def chat(effort):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful security analyst."},
            {"role": "user", "content": "In one sentence: what is lateral movement in an intrusion?"},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
        "reasoning_effort": effort,
    }
    req = urllib.request.Request(BASE + "/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def main():
    # models list
    req = urllib.request.Request(BASE + "/models", headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        models = [m["id"] for m in json.load(r).get("data", [])]
    print("served models:", models)
    for effort in ("low", "high"):
        resp = chat(effort)
        msg = resp["choices"][0]["message"]
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        content = msg.get("content") or ""
        usage = resp.get("usage", {})
        print(f"\n--- reasoning_effort={effort} ---")
        print("reasoning_tokens:", usage.get("completion_tokens_details", {}).get("reasoning_tokens"))
        print("has_reasoning_channel:", bool(reasoning), "| reasoning_chars:", len(reasoning))
        print("answer:", content[:200].replace("\n", " "))
    print("\nSMOKE_OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("SMOKE_FAIL:", repr(e)); sys.exit(1)
