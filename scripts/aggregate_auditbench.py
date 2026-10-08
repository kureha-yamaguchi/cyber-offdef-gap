#!/usr/bin/env python3
"""Aggregate the AuditBench aligned-vs-ablated comparison.

The per-partition harness runs already captured raw model outputs + reasoning traces
into results/auditbench/<stamp>_<side>/ and left the scored .txt files in the pristine
upstream scratch tree (tagged `gptoss120b-aligned` / `gptoss120b-ablated`). This script:

  1. Normalises the model's verdict SPELLING in the scratch-tree output files
     (gpt-oss sometimes emits e.g. HIGH_SUSPIOUS for HIGH_SUSPICIOUS, which crashes the
     upstream metric). Only the spelling is repaired; the severity level the model chose
     (HIGH/MEDIUM/LOW) is preserved. Applied IDENTICALLY to both sides, so it cannot bias
     the aligned->ablated delta. Originals are backed up (.rawbak) and the untouched raw
     generations remain in results/.../outputs + traces.jsonl.
  2. Re-runs the upstream metric per (dataset, task, rep, scenario_type) for BOTH sides,
     reusing the harness's own score()/compute_f1 (the paper's F1 = 2TP/(2TP+FP+FN)).
  3. Writes results/auditbench/comparison.{json,md} with per-cell aligned/ablated
     F1/TPR/FPR and the aligned->ablated deltas (the defensive-axis signal in PLAN §3/§6).

Usage:  .venv/bin/python scripts/aggregate_auditbench.py
Run it only after BOTH sides' inference has finished.
"""
from __future__ import annotations
import argparse, json, os, re, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "harnesses" / "auditbench"
sys.path.insert(0, str(HARNESS))
import run_auditbench as rb  # reuse score() + compute_f1 + UPSTREAM/INFER

INFER = rb.INFER
PLAN = [("labgen", ["edge", "raw"]), ("optc", ["edge"])]
TASKS = ["classification", "lm", "persistence", "exfiltration"]
TYPES = ["attack", "benign"]
SIDES = {"aligned": "gptoss120b-aligned", "ablated": "gptoss120b-ablated"}

VERDICT_RE = re.compile(r'(?i)\b(HIGH|MEDIUM|LOW)[_-]SUSP[A-Z]*\b')


def normalise_verdicts(model_tag: str) -> dict:
    """Repair verdict spelling in-place in the scratch-tree outputs for one model tag.
    Returns {files_touched, replacements}. Backs up each changed file once to .rawbak."""
    touched = repl = 0
    for f in INFER.glob(f"*_expt_dataset/task_*/*/output/gptout/*{model_tag}*"):
        if f.suffix == ".rawbak":
            continue
        txt = f.read_text(errors="replace")
        new, n = VERDICT_RE.subn(lambda m: f"{m.group(1).upper()}_SUSPICIOUS", txt)
        if n and new != txt:
            bak = f.with_suffix(f.suffix + ".rawbak")
            if not bak.exists():
                bak.write_text(txt)
            f.write_text(new)
            touched += 1
            repl += n
    return {"files_touched": touched, "replacements": repl}


def score_side(side: str) -> dict:
    """Run the upstream metric for every cell of one side; return nested metrics."""
    model_tag = SIDES[side]
    out = {}
    for dataset, reps in PLAN:
        args = argparse.Namespace(
            dataset=dataset, tasks=TASKS, reps=reps, types=TYPES,
            side=side,  # unused by score() but keep for clarity
        )
        metrics = rb.score(args, REPO / "results" / "auditbench", model_tag)
        # metrics: list of {task, rep, raw{type:...}, parsed{f1,tpr,fpr,tp,fp,fn}}
        for m in metrics:
            out[f"{dataset}/{m['task']}/{m['rep']}"] = {
                "parsed": m["parsed"],
                "raw": m["raw"],
            }
    return out


def has_outputs(model_tag: str) -> int:
    return sum(1 for _ in INFER.glob(f"*_expt_dataset/task_*/*/output/gptout/*{model_tag}*"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", nargs="+", default=["aligned", "ablated"])
    a = ap.parse_args()

    report = {"normalisation": {}, "sides": {}, "deltas": {}}
    for side in a.sides:
        tag = SIDES[side]
        n = has_outputs(tag)
        print(f"[{side}] {n} scratch output files tagged {tag}")
        if n == 0:
            print(f"[{side}] no outputs found — skip (did this side's run finish?)")
            continue
        report["normalisation"][side] = normalise_verdicts(tag)
        print(f"[{side}] verdict normalisation: {report['normalisation'][side]}")
        report["sides"][side] = score_side(side)

    # deltas = ablated - aligned per cell (negative dF1 on defence = the signature)
    al, ab = report["sides"].get("aligned", {}), report["sides"].get("ablated", {})
    for cell in sorted(set(al) | set(ab)):
        pa = al.get(cell, {}).get("parsed", {}) or {}
        pb = ab.get(cell, {}).get("parsed", {}) or {}
        d = {}
        for k in ("f1", "tpr", "fpr"):
            va, vb = pa.get(k), pb.get(k)
            d[k] = {"aligned": va, "ablated": vb,
                    "delta": (round(vb - va, 4) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None)}
        report["deltas"][cell] = d

    outdir = REPO / "results" / "auditbench"
    (outdir / "comparison.json").write_text(json.dumps(report, indent=2))

    # markdown table
    lines = ["# AuditBench: aligned vs refusal-ablated (defensive axis)", "",
             "F1 = 2·TP/(2·TP+FP+FN). delta = ablated − aligned (negative F1 delta = defence drop).",
             "Verdict spelling normalised identically on both sides; see comparison.json.", "",
             "| cell (dataset/task/rep) | F1 aligned | F1 ablated | ΔF1 | TPR a→b | FPR a→b |",
             "|---|---|---|---|---|---|"]
    def fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "—"
    for cell, d in sorted(report["deltas"].items()):
        f1 = d["f1"]; tpr = d["tpr"]; fpr = d["fpr"]
        lines.append(f"| {cell} | {fmt(f1['aligned'])} | {fmt(f1['ablated'])} | "
                     f"{fmt(f1['delta'])} | {fmt(tpr['aligned'])}→{fmt(tpr['ablated'])} | "
                     f"{fmt(fpr['aligned'])}→{fmt(fpr['ablated'])} |")
    (outdir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\nwrote", outdir / "comparison.json", "and comparison.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
