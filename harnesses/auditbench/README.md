# AuditBench harness (defensive axis)

Thin wrapper that runs the upstream **auditlogsbench** v2 benchmark against our vLLM
endpoints under the PLAN's fairness controls, and scores it with the upstream metrics.

- Upstream checkout: `~/auditlogsbench` @ `369ad441` (override with `AUDITBENCH_DIR`).
- The upstream tree is kept **pristine**. We only monkeypatch one function,
  `inference_helpers.call_model`, to send each query to our vLLM endpoint via
  chat.completions instead of the OpenAI Responses API (see `vllm_backend.py`).

## What it controls (parity)
- `base_url` / `model` from `.env` (`ALIGNED_*` vs `ABLATED_*`) — the **only**
  intended difference between the two runs is the weights.
- `reasoning_effort=high`, fixed `temperature` (default 0.7), fixed `max_tokens`.
- Reproducible per-prompt seed sequence (`base_seed + attempt_index`), so the N
  attempts differ but replay identically across aligned/ablated.
- The analysis/reasoning channel of every call is captured to `traces.jsonl` for the
  §6.2 "knew but withheld" vs "never identified" labelling.

## Setup (one-time, no GPU)
```bash
bash scripts/setup_auditbench_venv.sh
source .venv/bin/activate
```
The Gemini/Together clients are stubbed at runtime, so their SDKs are not installed.

## Usage
```bash
# offline sanity check — prints planned commands, no endpoint calls, no spend
python harnesses/auditbench/run_auditbench.py --side aligned \
  --tasks classification --reps edge --types attack benign --limit 2 --dry-run

# validation slice against the live aligned endpoint (needs the pod up)
python harnesses/auditbench/run_auditbench.py --side aligned \
  --tasks classification --reps edge --types attack benign --limit 2 --num-runs 3

# full aligned run (all four tasks, edge+raw, labgen)
python harnesses/auditbench/run_auditbench.py --side aligned \
  --tasks classification lm persistence exfiltration --reps edge raw --types attack benign
# optc dataset is edge-only:
python harnesses/auditbench/run_auditbench.py --side aligned --dataset optc \
  --tasks classification lm persistence exfiltration --reps edge --types attack benign
```
The ablated half is the same command with `--side ablated` (needs `ABLATED_BASE_URL`).

## Outputs (per run, under `results/auditbench/<UTC-stamp>_<side>/`)
- `manifest.json` — full parity config, upstream SHA, per-scenario pass/fail, and the
  parsed F1/TPR/FPR per task/rep.
- `traces.jsonl` — one line per model call: seed, usage, reasoning_content, content.
- `outputs/<task>_<rep>/` — the raw upstream `.txt` answer files (scoring source).
- Metric raw stdout is stored verbatim in `manifest.json` (`metrics[].raw`); the
  parsed numbers are best-effort and the raw text is the source of truth.

## Metrics
Scored with the upstream `metrics_comp/task_computemetrics.py` (`--all_scenarios`,
once per `--scenario_type`). F1 = 2·TP / (2·TP + FP + FN), as in the paper (§4.3).
Per-attempt raw outputs + traces are retained so pass@1 / pass@3 (PLAN §5) can be
computed separately from the upstream averaged metric.
```
