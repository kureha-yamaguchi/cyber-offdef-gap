#!/usr/bin/env python3
"""Aligned-vs-ablated AuditBench comparison chart (matplotlib PNG).

Reads results/auditbench/verdict_rates_both.json; writes results/auditbench/aligned_vs_ablated.png.
Two panels:
  (left)  attack-vs-benign HIGH_SUSPICIOUS rate, 4 bars/task (aligned atk/ben, ablated atk/ben)
  (right) discrimination = attack% - benign% separation per task, aligned vs ablated
Reliable (verdict extracted directly from model JSON). Chunk-level proxy, not per-scenario F1.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
d = json.load(open(REPO / "results/auditbench/verdict_rates_both.json"))
TASKS = [("classification", "Classification"), ("lm", "Lateral\nmovement"),
         ("persistence", "Persistence"), ("exfiltration", "Exfiltration")]
labels = [t[1] for t in TASKS]

AL, AB = "#4f46e5", "#b5540b"          # aligned indigo, ablated rust
AL_L, AB_L = "#a5a0f5", "#e0a070"      # lighter = benign
FG, MUT, GRID = "#1a1d23", "#59616e", "#d8dce3"
x = np.arange(len(labels))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=150,
                               gridspec_kw={"width_ratios": [1.35, 1]})
fig.patch.set_facecolor("white")

# ---- left: 4 bars per task ----
w = 0.2
al_a = [d[k]["aligned_attack"] for k, _ in TASKS]
al_b = [d[k]["aligned_benign"] for k, _ in TASKS]
ab_a = [d[k]["ablated_attack"] for k, _ in TASKS]
ab_b = [d[k]["ablated_benign"] for k, _ in TASKS]
axL.set_facecolor("white")
axL.bar(x - 1.5*w, al_a, w, label="Aligned · attack", color=AL, zorder=3)
axL.bar(x - 0.5*w, al_b, w, label="Aligned · benign", color=AL_L, zorder=3)
axL.bar(x + 0.5*w, ab_a, w, label="Ablated · attack", color=AB, zorder=3)
axL.bar(x + 1.5*w, ab_b, w, label="Ablated · benign", color=AB_L, zorder=3)
axL.set_ylim(0, 100); axL.set_yticks(range(0, 101, 25))
axL.set_ylabel("% chunks flagged HIGH_SUSPICIOUS", color=FG, fontsize=10)
axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9.5, color=FG)
axL.tick_params(axis="y", colors=MUT); axL.tick_params(axis="x", colors=FG, length=0)
axL.set_axisbelow(True); axL.yaxis.grid(True, color=GRID, lw=1)
for s in ("top", "right", "left"): axL.spines[s].set_visible(False)
axL.spines["bottom"].set_color(MUT)
axL.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right", labelcolor=FG)
axL.set_title("Flagging rate: attack vs benign", fontsize=12, color=FG, weight="bold", loc="left", pad=8)

# ---- right: discrimination (attack - benign) ----
sep_al = [round(d[k]["aligned_attack"] - d[k]["aligned_benign"], 1) for k, _ in TASKS]
sep_ab = [round(d[k]["ablated_attack"] - d[k]["ablated_benign"], 1) for k, _ in TASKS]
w2 = 0.36
axR.set_facecolor("white")
r1 = axR.bar(x - w2/2, sep_al, w2, label="Aligned", color=AL, zorder=3)
r2 = axR.bar(x + w2/2, sep_ab, w2, label="Ablated", color=AB, zorder=3)
for bars, vals in ((r1, sep_al), (r2, sep_ab)):
    for rect, v in zip(bars, vals):
        off = 0.8 if v >= 0 else -2.4
        axR.text(rect.get_x()+rect.get_width()/2, v+off, f"{v:+.0f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9, color=FG)
axR.axhline(0, color=MUT, lw=1.2, zorder=4)
axR.set_ylim(-10, 36); axR.set_ylabel("Attack − benign separation (pp)", color=FG, fontsize=10)
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9.5, color=FG)
axR.tick_params(axis="y", colors=MUT); axR.tick_params(axis="x", colors=FG, length=0)
axR.set_axisbelow(True); axR.yaxis.grid(True, color=GRID, lw=1)
for s in ("top", "right", "left"): axR.spines[s].set_visible(False)
axR.spines["bottom"].set_color(MUT)
axR.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=FG)
axR.set_title("Discrimination (higher = better defence)", fontsize=12, color=FG, weight="bold", loc="left", pad=8)

fig.suptitle("AuditBench defence: aligned vs refusal-ablated gpt-oss-120b",
             fontsize=14, color=FG, weight="bold", x=0.06, ha="left", y=0.99)
fig.text(0.06, 0.935, "Refusal-ablation collapses attack/benign discrimination on 3 of 4 tasks "
         "while inflating overall suspicion.", fontsize=9, color=MUT, ha="left")
fig.text(0.06, 0.01, "Reliable verdict extraction from model JSON (upstream line-parser crashes on gpt-oss). "
         "Chunk-level proxy, not per-scenario F1. 3 attempts/scenario.", fontsize=7.5, color=MUT, ha="left")

fig.tight_layout(rect=(0, 0.035, 1, 0.915))
out = REPO / "results/auditbench/aligned_vs_ablated.png"
fig.savefig(out, facecolor="white"); print("wrote", out)
