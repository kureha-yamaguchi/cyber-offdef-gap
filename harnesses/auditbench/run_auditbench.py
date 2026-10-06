#!/usr/bin/env python3
"""Run AuditBench (defensive axis) against one of our vLLM endpoints and score it.

Keeps the upstream auditlogsbench checkout pristine: it enumerates scenarios from the
upstream `benchmark_scenarios` registry, drives the upstream `task-query-v2.py` per
scenario via runpy (with our single network-call monkeypatch applied), then scores
with the upstream `metrics_comp/task_computemetrics.py`. Raw outputs, reasoning
traces, metric stdout and a parity manifest are snapshotted into results/auditbench/.

Build the aligned half first (step a); the ablated side only differs by --side.

Example (validation slice — one task, 2 scenarios/type):
  python harnesses/auditbench/run_auditbench.py --side aligned \
    --dataset labgen --tasks classification --reps edge \
    --types attack benign --limit 2 --num-runs 3

Full aligned run:
  python harnesses/auditbench/run_auditbench.py --side aligned \
    --dataset labgen --tasks classification lm persistence exfiltration \
    --reps edge raw --types attack benign --num-runs 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

# repo + upstream locations ---------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("AUDITBENCH_DIR", str(Path.home() / "auditlogsbench")))
INFER = UPSTREAM / "inference"
UPSTREAM_SHA = "369ad441f2876245d0db19990c77ccbcb74a0a3a"  # pinned; verified at runtime

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for vllm_backend
import vllm_backend  # noqa: E402


def load_env() -> dict:
    env = {}
    envfile = REPO / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    # live environment overrides the file
    for k in list(env):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def verify_upstream_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"
    if sha != UPSTREAM_SHA:
        print(f"[warn] auditlogsbench HEAD {sha[:10]} != pinned {UPSTREAM_SHA[:10]}")
    return sha


def enumerate_scenarios(bm, dataset, task, rep, typ, limit):
    var = f"{dataset}_{task}_{rep}_{typ}"
    lst = getattr(bm, var, None)
    if not lst:
        return []
    return lst[:limit] if limit else list(lst)


def run_inference(args, env, run_dir):
    """Drive upstream task-query-v2.py per scenario with the vLLM patch applied."""
    base_url = env[f"{args.side.upper()}_BASE_URL"]
    model = env[f"{args.side.upper()}_MODEL"]
    api_key = env.get("OPENAI_API_KEY", "sk-noauth")
    if not base_url:
        sys.exit(f"[fatal] {args.side.upper()}_BASE_URL is empty in .env — endpoint not set")

    trace_path = run_dir / "traces.jsonl"

    # stub unused providers, import upstream helpers, patch the single network call
    vllm_backend.install_provider_stubs()
    os.chdir(INFER)
    sys.path.insert(0, str(INFER))
    import inference_helpers  # noqa: E402  (after chdir + stubs)

    inference_helpers.call_model = vllm_backend.patched_call_model
    import benchmark_scenarios as bm  # noqa: E402

    vllm_backend.configure(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=args.temp,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        base_seed=args.base_seed,
        trace_path=trace_path,
    )

    model_tag = f"gptoss120b-{args.side}"  # names output files; API model is `model`
    planned, done, failed = [], [], []
    for task in args.tasks:
        for rep in args.reps:
            num_lines = args.num_lines or (1000 if rep == "raw" else 400)
            for typ in args.types:
                scenarios = enumerate_scenarios(bm, args.dataset, task, rep, typ, args.limit)
                for fname in scenarios:
                    log_fpath = f"./{args.dataset}_expt_dataset/task_{task}/{rep}/input/{fname}"
                    prompt_key = f"{task}-investigation_v2_{task}_{args.dataset}_{rep}"
                    argv = [
                        "task-query-v2.py",
                        "--dataset", args.dataset,
                        "--log_fpath", log_fpath,
                        "--num_lines", str(num_lines),
                        "--prompt_key", prompt_key,
                        "--num_runs", str(args.num_runs),
                        "--temp", str(args.temp),
                        "--llm", "gpt",
                        "--model", model_tag,
                        "--task", task,
                    ]
                    planned.append(log_fpath)
                    if args.dry_run:
                        print("[dry-run]", " ".join(argv))
                        continue
                    print(f"[run] {task}/{rep}/{typ}: {fname}")
                    sys.argv = argv
                    try:
                        runpy.run_path(str(INFER / "task-query-v2.py"), run_name="__main__")
                        done.append(log_fpath)
                    except SystemExit:
                        done.append(log_fpath)
                    except Exception as e:  # keep going; one bad scenario shouldn't kill the run
                        print(f"[error] {fname}: {e!r}")
                        failed.append({"scenario": log_fpath, "error": repr(e)})
    return {"model_tag": model_tag, "planned": planned, "done": done, "failed": failed,
            "base_url": base_url, "served_model": model, "num_lines_note": "400 edge / 1000 raw unless overridden"}


def score(args, run_dir, model_tag):
    """Run the upstream metrics script per task/rep/type; capture raw stdout + parse."""
    metrics = []
    for task in args.tasks:
        for rep in args.reps:
            per = {"task": task, "rep": rep, "raw": {}, "parsed": {}}
            counts = {}
            for typ in args.types:
                cmd = [
                    sys.executable, "./metrics_comp/task_computemetrics.py",
                    "--dataset", args.dataset, "--rep", rep, "--llm", "gpt",
                    "--task", task, "--version", "v2",
                    "--all_scenarios", "1", "--scenario_type", typ,
                    "--model", model_tag, "--quiet", "1",
                ]
                try:
                    out = subprocess.check_output(cmd, cwd=str(INFER), text=True,
                                                  stderr=subprocess.STDOUT, timeout=600)
                except subprocess.CalledProcessError as e:
                    out = f"[metrics-error rc={e.returncode}]\n{e.output}"
                except Exception as e:
                    out = f"[metrics-exception] {e!r}"
                per["raw"][typ] = out.strip()
                counts[typ] = [int(n) for n in __import__("re").findall(r"-?\d+", out)]
            per["parsed"] = compute_f1(task, counts)
            metrics.append(per)
    return metrics


def compute_f1(task, counts):
    """Best-effort F1/TPR/FPR from the --quiet integer pairs. Raw stdout is truth."""
    try:
        if task == "classification":
            detected, total_atk = counts.get("attack", [0, 0])[:2]
            mal, total_chunks = counts.get("benign", [0, 0])[:2]
            tp, fn, fp = detected, max(total_atk - detected, 0), mal
            tpr = detected / total_atk if total_atk else None
            fpr = mal / total_chunks if total_chunks else None
        else:
            tp, tpfn = counts.get("attack", [0, 0])[:2]
            fp, fptn = counts.get("benign", [0, 0])[:2]
            fn = max(tpfn - tp, 0)
            tpr = tp / tpfn if tpfn else None
            fpr = fp / fptn if fptn else None
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else None
        return {"tp": tp, "fp": fp, "fn": fn, "tpr": tpr, "fpr": fpr, "f1": f1}
    except Exception as e:
        return {"parse_error": repr(e)}


def snapshot_outputs(args, run_dir, model_tag):
    """Copy the upstream output .txt files this run produced into the results dir."""
    copied = 0
    for task in args.tasks:
        for rep in args.reps:
            src = INFER / f"{args.dataset}_expt_dataset" / f"task_{task}" / rep / "output" / "gptout"
            if not src.is_dir():
                continue
            dst = run_dir / "outputs" / f"{task}_{rep}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob(f"*{model_tag}*"):
                shutil.copy2(f, dst / f.name)
                copied += 1
    return copied


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=["aligned", "ablated"], required=True)
    p.add_argument("--dataset", choices=["labgen", "optc"], default="labgen")
    p.add_argument("--tasks", nargs="+", default=["classification", "lm", "persistence", "exfiltration"])
    p.add_argument("--reps", nargs="+", default=["edge"])
    p.add_argument("--types", nargs="+", default=["attack", "benign"])
    p.add_argument("--num-runs", type=int, default=3, help="attempts per scenario (pass@k)")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--max-tokens", type=int, default=24000)
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument("--num-lines", type=int, default=0, help="0 = paper default (400 edge / 1000 raw)")
    p.add_argument("--limit", type=int, default=0, help="cap scenarios per (task,rep,type); 0 = all")
    p.add_argument("--dry-run", action="store_true", help="print planned commands, no inference")
    p.add_argument("--no-score", action="store_true")
    args = p.parse_args()

    env = load_env()
    if not INFER.is_dir():
        sys.exit(f"[fatal] upstream inference dir not found: {INFER} (set AUDITBENCH_DIR)")
    sha = verify_upstream_sha()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO / "results" / "auditbench" / f"{stamp}_{args.side}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": f"{stamp}_{args.side}", "side": args.side, "dataset": args.dataset,
        "tasks": args.tasks, "reps": args.reps, "types": args.types,
        "num_runs": args.num_runs, "temp": args.temp,
        "reasoning_effort": args.reasoning_effort, "max_tokens": args.max_tokens,
        "base_seed": args.base_seed, "limit": args.limit,
        "upstream_sha": sha, "upstream_sha_pinned": UPSTREAM_SHA,
        "dry_run": args.dry_run,
    }

    inf = run_inference(args, env, run_dir)
    manifest.update(inf)

    if not args.dry_run and not args.no_score:
        manifest["copied_output_files"] = snapshot_outputs(args, run_dir, inf["model_tag"])
        manifest["metrics"] = score(args, run_dir, inf["model_tag"])

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n=== run complete ===")
    print("results:", run_dir)
    if manifest.get("metrics"):
        for m in manifest["metrics"]:
            pr = m["parsed"]
            print(f"  {m['task']}/{m['rep']}: F1={pr.get('f1')} TPR={pr.get('tpr')} FPR={pr.get('fpr')}")
    if inf["failed"]:
        print(f"  [!] {len(inf['failed'])} scenario(s) errored — see manifest.json")


if __name__ == "__main__":
    main()
