#!/usr/bin/env python3
"""M0 AuditBench smoke: assemble the REAL upstream v2 classification prompt on real
scenario logs and run it through the vLLM endpoint. Confirms the defensive harness
path works end to end (no Docker needed). Not a metric run — that's M1."""
import os, sys, json, glob, re
AB = "/Users/kyamaguchi/Desktop/auditlogsbench/inference"
sys.path.insert(0, AB)
from prompt_v2 import Prompt
from prompt_template.input_format import InputFormat
from prompt_template.ostype import OSType
from prompt_template.classification_template_v2 import ClassificationPrompt
from openai import OpenAI

BASE = os.environ["OPENAI_BASE_URL"]; MODEL = os.environ.get("MODEL", "openai/gpt-oss-120b")
client = OpenAI(base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY", "sk-noauth"))
INP = f"{AB}/labgen_expt_dataset/task_classification/edge/input"
MAXCHARS = 55000  # cap input to stay under 32k ctx for the smoke

def pick(pattern):
    fs = sorted(glob.glob(f"{INP}/*{pattern}*.log"))
    return fs[0] if fs else None

def run(fpath):
    name = os.path.basename(fpath)
    os_type = OSType.windows if "windows" in name else OSType.linux
    with open(fpath) as f:
        lines = f.read().splitlines()
    logs = "\n".join(lines)[:MAXCHARS]
    prompt = Prompt(task="attack", os_type=os_type, goal=ClassificationPrompt.GOAL,
        output_description=ClassificationPrompt.OUTPUT_DESCRIPTION,
        json_output_format=ClassificationPrompt.OUTPUT_FORMAT,
        log_input_format=InputFormat.edge).populate(logs)
    resp = client.chat.completions.create(model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, extra_body={"reasoning_effort": "high"})
    out = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", out, re.S)
    verdict, parsed_ok = None, False
    if m:
        try:
            verdict = json.loads(m.group(0)).get("verdict"); parsed_ok = True
        except Exception:
            mv = re.search(r"(LOW|MEDIUM|HIGH)_SUSPICIOUS", out); verdict = mv.group(0) if mv else None
    print(f"  file={name}\n  os={os_type.value} logs_chars_sent={min(len(chr(10).join(lines)),MAXCHARS)} prompt_chars={len(prompt)}")
    print(f"  json_parsed={parsed_ok} verdict={verdict} answer_chars={len(out)}")
    return verdict

print("AuditBench v2 classification smoke via", BASE)
for label, pat in [("ATTACK", "attack-linux"), ("BENIGN", "benign")]:
    f = pick(pat)
    if not f: print(f"[{label}] no scenario found for '{pat}'"); continue
    print(f"\n[{label}] {pat}"); run(f)
print("\nAUDITBENCH_SMOKE_OK")
