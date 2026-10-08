#!/usr/bin/env python3
"""Generate the AuditBench 'results so far' chart (ablated side, reliable verdict extraction).

Reads results/auditbench/ablated_verdict_rates.json and emits notes/results-chart.html —
a theme-aware grouped bar chart of the % of log chunks the refusal-ablated model flagged
HIGH_SUSPICIOUS, attack vs benign, per task. Bars drawn as SVG with exact geometry.

This is a reliable preview computed by extracting the verdict field directly from the model's
(valid, pretty-printed) JSON, NOT the upstream positional line-parser, which crashes on gpt-oss's
output structure. It is the ablated model's chunk-level suspicion rate, not the paper's per-scenario F1.
"""
import json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
data = json.load(open(REPO / "results/auditbench/ablated_verdict_rates.json"))
TASKS = [("classification", "Classification"), ("lm", "Lateral movement"),
         ("persistence", "Persistence"), ("exfiltration", "Exfiltration")]

# SVG plot geometry
W, H = 560, 300
L, R, T, B = 52, 16, 20, 56           # margins
x0, x1, y0, y1 = L, W - R, T, H - B    # plot box; y0=top(100%), y1=baseline(0%)
plot_h = y1 - y0
ngroups = len(TASKS)
gw = (x1 - x0) / ngroups
bw = gw * 0.26                          # bar width
gap = gw * 0.06

def y_of(pct): return y1 - (pct / 100) * plot_h

grid = ""
for p in (0, 25, 50, 75, 100):
    yy = y_of(p)
    grid += f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}"/>'
    grid += f'<text class="ylab" x="{x0-8}" y="{yy+3.5:.1f}" text-anchor="end">{p}</text>'

bars = ""; xlabels = ""
for i, (key, label) in enumerate(TASKS):
    cx = x0 + gw * (i + 0.5)
    a = data.get(f"{key}/attack", {}); b = data.get(f"{key}/benign", {})
    ah, bh = a.get("high_pct", 0), b.get("high_pct", 0)
    # attack bar (left), benign bar (right)
    ax = cx - bw - gap/2; bx = cx + gap/2
    bars += f'<rect class="b-att" x="{ax:.1f}" y="{y_of(ah):.1f}" width="{bw:.1f}" height="{y1-y_of(ah):.1f}" rx="2"/>'
    bars += f'<text class="val" x="{ax+bw/2:.1f}" y="{y_of(ah)-5:.1f}" text-anchor="middle">{ah:.0f}</text>'
    bars += f'<rect class="b-ben" x="{bx:.1f}" y="{y_of(bh):.1f}" width="{bw:.1f}" height="{y1-y_of(bh):.1f}" rx="2"/>'
    bars += f'<text class="val" x="{bx+bw/2:.1f}" y="{y_of(bh)-5:.1f}" text-anchor="middle">{bh:.0f}</text>'
    xlabels += f'<text class="xlab" x="{cx:.1f}" y="{y1+18:.1f}" text-anchor="middle">{label}</text>'
    na, nb = a.get("N", 0), b.get("N", 0)
    xlabels += f'<text class="xsub" x="{cx:.1f}" y="{y1+32:.1f}" text-anchor="middle">n={na}/{nb}</text>'

svg = f'''<svg viewBox="0 0 {W} {H}" role="img" aria-label="HIGH_SUSPICIOUS flag rate by task, attack vs benign">
<text class="axtitle" x="{L-44}" y="{T-6}" transform="rotate(0)"></text>
{grid}
<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>
{bars}{xlabels}
</svg>'''

# tidy table rows
rows = ""
for key, label in TASKS:
    a = data.get(f"{key}/attack", {}); b = data.get(f"{key}/benign", {})
    gap_pp = a.get("high_pct", 0) - b.get("high_pct", 0)
    sign = "+" if gap_pp >= 0 else "−"
    rows += (f'<tr><td class="tk">{label}</td>'
             f'<td class="num att">{a.get("high_pct",0):.1f}%</td>'
             f'<td class="num ben">{b.get("high_pct",0):.1f}%</td>'
             f'<td class="num gap">{sign}{abs(gap_pp):.1f} pp</td></tr>')

html = f'''<title>Ablated Detection Preview</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{
  --bg:#f3f4f7;--surface:#fff;--surface-2:#eceef2;--fg:#1a1d23;--muted:#59616e;--line:#d8dce3;
  --att:#b5540b;--att-soft:#f6e6d6;--ben:#0e7a70;--ben-soft:#d7eeeb;--accent:#4f46e5;
  --display:"Fraunces",Georgia,serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#0f1217;--surface:#161b22;--surface-2:#1d232c;--fg:#e6e9ef;--muted:#98a1af;--line:#29313c;
  --att:#e0923f;--att-soft:#33230f;--ben:#52d3c6;--ben-soft:#0d2b29;--accent:#8e88f6;color-scheme:dark;
}}}}
:root[data-theme="dark"]{{
  --bg:#0f1217;--surface:#161b22;--surface-2:#1d232c;--fg:#e6e9ef;--muted:#98a1af;--line:#29313c;
  --att:#e0923f;--att-soft:#33230f;--ben:#52d3c6;--ben-soft:#0d2b29;--accent:#8e88f6;color-scheme:dark;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font-family:var(--body);line-height:1.6;margin:0;font-size:15px}}
.wrap{{max-width:760px;margin:0 auto;padding-inline:20px;padding-block:34px 44px}}
h1{{font-family:var(--display);font-weight:600;font-size:clamp(24px,5vw,34px);line-height:1.15;margin:0;text-wrap:balance;letter-spacing:-.01em}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}}
.lede{{color:var(--fg);margin-top:14px;max-width:62ch}}
.status{{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;padding:4px 10px;border-radius:999px;
  border:1px solid color-mix(in srgb,var(--accent) 40%,var(--line));color:var(--accent);background:color-mix(in srgb,var(--accent) 8%,transparent);margin-top:18px}}
.card{{border:1px solid var(--line);border-radius:14px;background:var(--surface);padding:20px;margin-top:22px}}
.chart-h{{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:6px}}
.chart-h .t{{font-family:var(--display);font-weight:600;font-size:16px}}
.chart-h .u{{font-family:var(--mono);font-size:11px;color:var(--muted)}}
.legend{{display:flex;gap:16px;margin:4px 0 12px}}
.legend span{{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;font-family:var(--mono)}}
.sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}
.sw.att{{background:var(--att)}} .sw.ben{{background:var(--ben)}}
svg{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--line);stroke-width:1}}
.axis{{stroke:var(--muted);stroke-width:1.2}}
.ylab,.xsub{{fill:var(--muted);font-family:var(--mono);font-size:10px}}
.xlab{{fill:var(--fg);font-family:var(--body);font-size:11.5px;font-weight:500}}
.val{{fill:var(--fg);font-family:var(--mono);font-size:11px;font-weight:500}}
.b-att{{fill:var(--att)}} .b-ben{{fill:var(--ben)}}
table{{width:100%;border-collapse:collapse;margin-top:6px;font-size:13.5px}}
th,td{{text-align:right;padding:7px 10px;border-top:1px solid var(--line)}}
th{{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}}
th:first-child,td.tk{{text-align:left}}
.num{{font-variant-numeric:tabular-nums}} td.att{{color:var(--att)}} td.ben{{color:var(--ben)}}
td.gap{{font-family:var(--mono);color:var(--muted)}}
.read{{margin-top:22px;display:grid;gap:12px}}
.read .r{{border-left:3px solid var(--line);padding:3px 0 3px 14px;font-size:14px}}
.read .r b{{font-weight:600}}
.read .r.key{{border-left-color:var(--accent)}}
.note{{margin-top:20px;font-size:12.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}}
.note code{{font-family:var(--mono);font-size:12px}}
</style>
<div class="wrap">
  <p class="eyebrow">AuditBench · defensive axis · preview</p>
  <h1>Does the refusal-ablated model actually flag attacks?</h1>
  <p class="lede">The share of audit-log chunks the ablated <code style="font-family:var(--mono)">gpt-oss-120b-abliterated</code>
    model rated <b>HIGH_SUSPICIOUS</b>, split by whether the scenario is an attack or benign. A model that
    discriminates should raise the orange bar well above the teal one.</p>
  <span class="status">● Ablated side complete · aligned baseline still running</span>

  <div class="card">
    <div class="chart-h"><span class="t">HIGH_SUSPICIOUS flag rate by task</span><span class="u">% of chunks · y-axis 0–100</span></div>
    <div class="legend"><span><i class="sw att"></i>Attack scenarios</span><span><i class="sw ben"></i>Benign scenarios</span></div>
    {svg}
    <table>
      <tr><th>Task</th><th>Attack</th><th>Benign</th><th>Separation</th></tr>
      {rows}
    </table>
  </div>

  <div class="read">
    <div class="r key"><b>Weak separation on three of four tasks.</b> For lateral movement, persistence and
      exfiltration the model flags benign chunks about as often as attack chunks — the orange and teal bars
      nearly match, so at chunk level it is not distinguishing attacks from normal activity.</div>
    <div class="r"><b>Classification is the exception.</b> It flags 90% of attack chunks vs 82% of benign —
      the only task with a clear gap, though the benign rate is still high (many false alarms).</div>
    <div class="r"><b>This is behaviour, not the verdict.</b> The real comparison is the aligned→ablated delta
      per task, which needs the aligned run now in progress. Absolute rates alone are not the signal.</div>
  </div>

  <p class="note">Computed by extracting the <code>"verdict"</code> field directly from the model's JSON
    outputs (reliable), because the upstream metric's positional line-parser crashes on gpt-oss's multi-line
    pretty-printed JSON. Chunk-level suspicion rate is a proxy for the paper's per-scenario F1, not a
    substitute. n = attack chunks / benign chunks scored per task, across 3 attempts.</p>
</div>'''

out = REPO / "notes/results-chart.html"
out.write_text(html)
print("wrote", out, f"({len(html)} bytes)")
