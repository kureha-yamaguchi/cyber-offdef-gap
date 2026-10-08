#!/usr/bin/env python3
"""Render the ablated AuditBench verdict-rate preview as a matplotlib PNG.

Reads results/auditbench/ablated_verdict_rates.json; writes results/auditbench/ablated_detection.png.
Grouped bars: % of log chunks flagged HIGH_SUSPICIOUS, attack vs benign, per task.
Reliable (verdict extracted directly from the model JSON), ablated side only.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
data = json.load(open(REPO / "results/auditbench/ablated_verdict_rates.json"))
TASKS = [("classification", "Classification"), ("lm", "Lateral\nmovement"),
         ("persistence", "Persistence"), ("exfiltration", "Exfiltration")]

labels = [t[1] for t in TASKS]
attack = [data[f"{k}/attack"]["high_pct"] for k, _ in TASKS]
benign = [data[f"{k}/benign"]["high_pct"] for k, _ in TASKS]
n_att = [data[f"{k}/attack"]["N"] for k, _ in TASKS]
n_ben = [data[f"{k}/benign"]["N"] for k, _ in TASKS]

ATT, BEN, FG, MUT, GRID = "#b5540b", "#0e7a70", "#1a1d23", "#59616e", "#d8dce3"
x = np.arange(len(labels)); w = 0.38

fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
b1 = ax.bar(x - w/2, attack, w, label="Attack scenarios", color=ATT, zorder=3)
b2 = ax.bar(x + w/2, benign, w, label="Benign scenarios", color=BEN, zorder=3)

for bars, vals in ((b1, attack), (b2, benign)):
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 1.5, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9, color=FG)

ax.set_ylim(0, 100)
ax.set_ylabel("% of log chunks flagged HIGH_SUSPICIOUS", color=FG, fontsize=10)
ax.set_yticks(range(0, 101, 25))
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={a}/{b})" for l, a, b in zip(labels, n_att, n_ben)], fontsize=9.5, color=FG)
ax.tick_params(axis="y", colors=MUT); ax.tick_params(axis="x", colors=FG, length=0)
ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, lw=1)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUT)

ax.set_title("Refusal-ablated gpt-oss-120b: attack-flagging rate by task",
             fontsize=13, color=FG, weight="bold", pad=12, loc="left")
fig.text(0.125, 0.905, "AuditBench defensive axis · ablated side only (aligned baseline pending) · "
         "equal bars = no attack/benign discrimination",
         fontsize=8.5, color=MUT, ha="left")
ax.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=FG)
fig.text(0.125, 0.01, "Verdict extracted directly from model JSON (upstream line-parser crashes on gpt-oss "
         "output). Chunk-level proxy, not per-scenario F1.", fontsize=7.5, color=MUT, ha="left")

fig.tight_layout(rect=(0, 0.03, 1, 0.93))
out = REPO / "results/auditbench/ablated_detection.png"
fig.savefig(out, facecolor="white"); print("wrote", out)
