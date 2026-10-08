#!/usr/bin/env python3
"""Generate the AuditBench results spec page (notes/results-spec.html) from results/auditbench/rescored.json.

Visual-first: ΔF1 with CIs, per-task confusion tiles (TP/FN/FP/TN with TPR/FNR/FPR/TNR), rate
micro-bars, the precision-collapse mechanism, empty-response rates, pass@k dumbbells, threshold
robustness. All numbers come from the re-scorer; SVG geometry is computed here (no hand arithmetic).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = json.load(open(REPO / "results/auditbench/rescored.json"))
T = R["thresholds"]["HIGH_only"]; T2 = R["thresholds"]["HIGH+MEDIUM"]; FE = R["format_errors"]
TASKS = [("classification", "Classification"), ("lm", "Lateral movement"),
         ("persistence", "Persistence"), ("exfiltration", "Exfiltration")]
SIDES = ["aligned", "ablated"]

def rates(m):
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, tpr=tpr, fnr=1 - tpr, fpr=fpr, tnr=1 - fpr, prec=prec, f1=m["f1"] or 0.0)

D = {t: {s: rates(T["per_task"][t][s]) for s in SIDES} for t, _ in TASKS}
PK = {t: {s: T["passk"][t][s] for s in SIDES} for t, _ in TASKS}
EMPTY = {t: {s: FE[f"{s}/{t}"]["empty_responses"] / FE[f"{s}/{t}"]["blocks"] for s in SIDES} for t, _ in TASKS}

def f3(x): return f"{x:.3f}"
def pct(x): return f"{100*x:.0f}%"
def num(x): return f"{x:,.0f}" if x >= 100 else f"{x:.2f}".rstrip("0").rstrip(".")

# ------------------------------------------------------------------ SVG helpers (classes carry theme fills)
def svg_ci(deltas, W=640, H=190):
    """Dot + 95% CI per task, horizontal, zero line. deltas: list of (label, d, lo, hi)."""
    L, Rm, Tm, Bm = 150, 20, 18, 34
    xmin, xmax = -0.42, 0.08
    def X(v): return L + (v - xmin) / (xmax - xmin) * (W - L - Rm)
    rowh = (H - Tm - Bm) / len(deltas)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Delta F1 per task with 95% CI">']
    for v in (-0.4, -0.3, -0.2, -0.1, 0.0):
        x = X(v); cls = "zero" if v == 0 else "grid"
        s.append(f'<line class="{cls}" x1="{x:.1f}" y1="{Tm}" x2="{x:.1f}" y2="{H-Bm}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{H-Bm+16}" text-anchor="middle">{v:+.1f}</text>')
    for i, (lab, d, lo, hi) in enumerate(deltas):
        y = Tm + rowh * (i + 0.5)
        sig = hi < 0
        s.append(f'<text class="rowlab" x="{L-12}" y="{y+4:.1f}" text-anchor="end">{lab}</text>')
        s.append(f'<line class="{"ci sig" if sig else "ci"}" x1="{X(lo):.1f}" y1="{y:.1f}" x2="{X(hi):.1f}" y2="{y:.1f}"/>')
        s.append(f'<circle class="{"dot sig" if sig else "dot"}" cx="{X(d):.1f}" cy="{y:.1f}" r="6"/>')
        s.append(f'<text class="val" x="{X(d):.1f}" y="{y-11:.1f}" text-anchor="middle">{d:+.3f}</text>')
    s.append(f'<text class="axlab" x="{(L+W-Rm)/2:.0f}" y="{H-4}" text-anchor="middle">ΔF1 = ablated − aligned (← defence drop)</text>')
    s.append("</svg>")
    return "\n".join(s)

def svg_pairbars(rows, W=640, H=None, xmax=1.0, fmt=pct, unit=""):
    """Horizontal paired bars (aligned over ablated) per row. rows: list of (label, a, b)."""
    L, Rm, Tm = 150, 56, 10
    rowh = 44; H = H or Tm + rowh * len(rows) + 8
    def X(v): return L + (v / xmax) * (W - L - Rm)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for i, (lab, a, b) in enumerate(rows):
        y = Tm + rowh * i
        s.append(f'<text class="rowlab" x="{L-12}" y="{y+24:.1f}" text-anchor="end">{lab}</text>')
        for j, (v, cls) in enumerate(((a, "al"), (b, "ab"))):
            yy = y + 6 + j * 16
            s.append(f'<rect class="bar {cls}" x="{L}" y="{yy}" width="{max(X(v)-L, 1):.1f}" height="12" rx="2"/>')
            s.append(f'<text class="val" x="{X(v)+6:.1f}" y="{yy+10}">{fmt(v)}{unit}</text>')
    s.append("</svg>")
    return "\n".join(s)

def svg_dumbbell(rows, W=640, H=None, xmax=1.0, fmt=pct):
    """aligned→ablated dumbbell per row. rows: list of (label, a, b)."""
    L, Rm, Tm = 170, 40, 14
    rowh = 36; H = H or Tm + rowh * len(rows) + 26
    def X(v): return L + (v / xmax) * (W - L - Rm)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for v in (0, .25, .5, .75, 1.0):
        s.append(f'<line class="grid" x1="{X(v):.1f}" y1="{Tm}" x2="{X(v):.1f}" y2="{H-22}"/>')
        s.append(f'<text class="tick" x="{X(v):.1f}" y="{H-6}" text-anchor="middle">{fmt(v)}</text>')
    for i, (lab, a, b) in enumerate(rows):
        y = Tm + rowh * (i + 0.5)
        s.append(f'<text class="rowlab" x="{L-12}" y="{y+4:.1f}" text-anchor="end">{lab}</text>')
        s.append(f'<line class="link" x1="{X(a):.1f}" y1="{y:.1f}" x2="{X(b):.1f}" y2="{y:.1f}"/>')
        s.append(f'<circle class="dot al" cx="{X(a):.1f}" cy="{y:.1f}" r="6"/>')
        s.append(f'<circle class="dot ab" cx="{X(b):.1f}" cy="{y:.1f}" r="6"/>')
        lo, hi = (a, b) if a <= b else (b, a)
        s.append(f'<text class="val" x="{X(lo)-9:.1f}" y="{y+4:.1f}" text-anchor="end">{fmt(lo)}</text>')
        s.append(f'<text class="val" x="{X(hi)+9:.1f}" y="{y+4:.1f}">{fmt(hi)}</text>')
    s.append("</svg>")
    return "\n".join(s)

# ------------------------------------------------------------------ content blocks
ci_rows = [(lab, T["delta"][t]["delta_f1"], *T["delta"][t]["ci95"]) for t, lab in TASKS]
ci_svg = svg_ci(ci_rows)

def tile(t, s):
    d = D[t][s]; o = D[t]["aligned" if s == "ablated" else "ablated"]
    fpx = (d["fp"] / o["fp"]) if (s == "ablated" and o["fp"]) else None
    badge = f'<span class="x">{fpx:.0f}× aligned</span>' if fpx and fpx >= 2 else ""
    return f'''
    <div class="tile {s}">
      <div class="th">{s}<span class="f1">F1 {f3(d["f1"])}</span></div>
      <div class="cm">
        <div class="corner"></div><div class="col">Flagged</div><div class="col">Not flagged</div>
        <div class="rowh">Attack<small>actual</small></div>
        <div class="c ok"><b>TP</b><span>{num(d["tp"])}</span><em>TPR {pct(d["tpr"])}</em></div>
        <div class="c err"><b>FN</b><span>{num(d["fn"])}</span><em>FNR {pct(d["fnr"])}</em></div>
        <div class="rowh">Benign<small>actual</small></div>
        <div class="c err"><b>FP</b><span>{num(d["fp"])}</span>{badge}<em>FPR {d["fpr"]*100:.2f}%</em></div>
        <div class="c ok"><b>TN</b><span>{num(d["tn"])}</span><em>TNR {d["tnr"]*100:.2f}%</em></div>
      </div>
      <div class="prec">Precision <b>{pct(d["prec"])}</b></div>
    </div>'''

tiles = "".join(f'<div class="taskrow"><h3>{lab}</h3><div class="tiles">{tile(t,"aligned")}{tile(t,"ablated")}</div></div>'
                for t, lab in TASKS)

def microbar(a, b, mx):
    wa, wb = 100 * a / mx, 100 * b / mx
    return (f'<div class="mb"><i class="al" style="width:{wa:.1f}%"></i><i class="ab" style="width:{wb:.1f}%"></i></div>')

rate_rows = ""
for t, lab in TASKS:
    a, b = D[t]["aligned"], D[t]["ablated"]
    fpx = b["fpr"] / a["fpr"] if a["fpr"] else 0
    rate_rows += f'''<tr><td class="tk">{lab}</td>
      <td>{microbar(a["tpr"], b["tpr"], 1)}<span class="pair"><b class="al">{pct(a["tpr"])}</b>→<b class="ab">{pct(b["tpr"])}</b></span></td>
      <td>{microbar(a["fnr"], b["fnr"], 1)}<span class="pair"><b class="al">{pct(a["fnr"])}</b>→<b class="ab">{pct(b["fnr"])}</b></span></td>
      <td><span class="pair"><b class="al">{a["fpr"]*100:.2f}%</b>→<b class="ab">{b["fpr"]*100:.2f}%</b></span><span class="mult">×{fpx:.0f}</span></td>
      <td><span class="pair"><b class="al">{a["tnr"]*100:.2f}%</b>→<b class="ab">{b["tnr"]*100:.2f}%</b></span></td>
      <td>{microbar(a["prec"], b["prec"], 0.2)}<span class="pair"><b class="al">{pct(a["prec"])}</b>→<b class="ab">{pct(b["prec"])}</b></span></td></tr>'''

fp_rows = [(lab, D[t]["aligned"]["fp"], D[t]["ablated"]["fp"]) for t, lab in TASKS]
fp_svg = svg_pairbars(fp_rows, xmax=1250, fmt=lambda v: f"{v:,.0f}")
tpr_rows = [(lab, D[t]["aligned"]["tpr"], D[t]["ablated"]["tpr"]) for t, lab in TASKS]
tpr_svg = svg_dumbbell(tpr_rows)
empty_rows = [(lab, EMPTY[t]["aligned"], EMPTY[t]["ablated"]) for t, lab in TASKS]
empty_svg = svg_pairbars(empty_rows, xmax=0.6)

def pk(t, s, typ): return PK[t][s].get(f"{typ}/lenient", {}).get("pass@1", 0.0)
pk_att = [(lab, pk(t, "aligned", "attack"), pk(t, "ablated", "attack")) for t, lab in TASKS]
pk_ben = [(lab, pk(t, "aligned", "benign"), pk(t, "ablated", "benign")) for t, lab in TASKS]
pk_att_svg = svg_dumbbell(pk_att); pk_ben_svg = svg_dumbbell(pk_ben)

thr_rows = ""
for t, lab in TASKS:
    d1, d2 = T["delta"][t], T2["delta"][t]
    thr_rows += (f'<tr><td class="tk">{lab}</td><td class="num">{d1["delta_f1"]:+.3f} <small>[{d1["ci95"][0]:+.2f}, {d1["ci95"][1]:+.2f}]</small></td>'
                 f'<td class="num">{d2["delta_f1"]:+.3f} <small>[{d2["ci95"][0]:+.2f}, {d2["ci95"][1]:+.2f}]</small></td></tr>')

n_scn = T["delta"]["classification"]["n_scenarios"]

html = f'''<title>AuditBench Results</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: single reading column, chart cards full-width within it. Indigo = aligned, rust = ablated,
   red only for error cells / significant drops. */
:root{{
  --bg:#f3f4f7;--surface:#fff;--surface-2:#eceef2;--fg:#1a1d23;--muted:#59616e;--line:#d8dce3;
  --al:#4f46e5;--al-soft:#e7e6fb;--ab:#b5540b;--ab-soft:#f6e6d6;
  --err:#b2271f;--err-soft:#f8e3e1;--ok-soft:#e4f1ea;--accent:#4f46e5;
  --display:"Fraunces",Georgia,serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#0f1217;--surface:#161b22;--surface-2:#1d232c;--fg:#e6e9ef;--muted:#98a1af;--line:#29313c;
  --al:#8e88f6;--al-soft:#232144;--ab:#e0923f;--ab-soft:#33230f;--err:#f4746a;--err-soft:#3a1c1a;--ok-soft:#15302a;--accent:#8e88f6;color-scheme:dark;
}}}}
:root[data-theme="dark"]{{
  --bg:#0f1217;--surface:#161b22;--surface-2:#1d232c;--fg:#e6e9ef;--muted:#98a1af;--line:#29313c;
  --al:#8e88f6;--al-soft:#232144;--ab:#e0923f;--ab-soft:#33230f;--err:#f4746a;--err-soft:#3a1c1a;--ok-soft:#15302a;--accent:#8e88f6;color-scheme:dark;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font-family:var(--body);line-height:1.6;margin:0;font-size:15px}}
.wrap{{max-width:860px;margin:0 auto;padding-inline:20px;padding-block:40px 56px}}
h1,h2,h3{{font-family:var(--display);text-wrap:balance;line-height:1.15;margin:0;font-weight:600}}
h1{{font-size:clamp(28px,5.5vw,44px);letter-spacing:-.01em}}
h2{{font-size:clamp(20px,3.2vw,26px)}} h3{{font-size:16px}}
p{{margin:0}} code,.mono{{font-family:var(--mono)}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}}
.lede{{font-size:clamp(16px,2.3vw,18px);margin-top:16px;max-width:62ch}}
.sub{{color:var(--muted);margin-top:10px;max-width:66ch}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:18px}}
.legend span{{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);font-size:12px}}
.sw{{width:12px;height:12px;border-radius:3px;display:inline-block}} .sw.al{{background:var(--al)}} .sw.ab{{background:var(--ab)}}
section{{padding-block:30px;border-top:1px solid var(--line)}}
.sec-head{{display:flex;flex-direction:column;gap:7px;margin-bottom:18px}}
.sec-head p{{color:var(--muted);max-width:64ch}}
.card{{border:1px solid var(--line);border-radius:14px;background:var(--surface);padding:18px 20px}}
.card+.card{{margin-top:14px}}
.card .ct{{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}}
svg{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--line);stroke-width:1}} .zero{{stroke:var(--muted);stroke-width:1.4}}
.tick,.axlab{{fill:var(--muted);font-family:var(--mono);font-size:10.5px}}
.rowlab{{fill:var(--fg);font-family:var(--body);font-size:12.5px;font-weight:500}}
.val{{fill:var(--fg);font-family:var(--mono);font-size:11px;font-weight:500}}
.ci{{stroke:var(--muted);stroke-width:2.5;stroke-linecap:round}} .ci.sig{{stroke:var(--err)}}
.dot{{fill:var(--muted)}} .dot.sig{{fill:var(--err)}} .dot.al{{fill:var(--al)}} .dot.ab{{fill:var(--ab)}}
.bar.al{{fill:var(--al)}} .bar.ab{{fill:var(--ab)}} .link{{stroke:var(--line);stroke-width:3}}

/* confusion tiles */
.taskrow{{margin-top:18px}} .taskrow h3{{margin-bottom:10px}}
.tiles{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.tile{{border:1px solid var(--line);border-radius:12px;background:var(--surface);padding:14px;border-top:3px solid var(--line);min-width:0}}
.tile.aligned{{border-top-color:var(--al)}} .tile.ablated{{border-top-color:var(--ab)}}
.tile .th{{display:flex;justify-content:space-between;align-items:baseline;font-family:var(--mono);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px}}
.tile.aligned .th{{color:var(--al)}} .tile.ablated .th{{color:var(--ab)}}
.tile .f1{{font-family:var(--display);font-size:15px;letter-spacing:0;text-transform:none;color:var(--fg);font-weight:600}}
.cm{{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;align-items:stretch}}
.cm .col,.cm .rowh{{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
.cm .col{{text-align:center;padding-bottom:2px}} .cm .rowh{{display:flex;flex-direction:column;justify-content:center;padding-right:6px;line-height:1.25}}
.cm .rowh small{{text-transform:none;letter-spacing:0;font-size:9.5px;color:var(--muted)}}
.c{{border-radius:8px;padding:9px 10px;display:flex;flex-direction:column;gap:1px;min-height:66px;position:relative}}
.c.ok{{background:var(--ok-soft)}} .c.err{{background:var(--err-soft)}}
.c b{{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;color:var(--muted)}}
.c span{{font-family:var(--display);font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1}}
.c em{{font-style:normal;font-family:var(--mono);font-size:11px;color:var(--fg)}}
.c .x{{position:absolute;top:8px;right:8px;font-family:var(--mono);font-size:10px;color:var(--err);border:1px solid color-mix(in srgb,var(--err) 45%,var(--line));border-radius:5px;padding:1px 6px;background:var(--surface)}}
.prec{{margin-top:10px;font-size:13px;color:var(--muted)}} .prec b{{color:var(--fg)}}

/* rate table */
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:9px 8px;border-top:1px solid var(--line);vertical-align:middle;text-align:left}}
th{{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:500;border-top:0}}
td.tk{{font-weight:600;white-space:nowrap}} td.num{{font-variant-numeric:tabular-nums;font-family:var(--mono)}}
td.num small{{color:var(--muted);font-size:11px}}
.mb{{display:flex;flex-direction:column;gap:2px;width:100%;max-width:140px;margin-bottom:4px}}
.mb i{{display:block;height:6px;border-radius:2px;min-width:2px}} .mb i.al{{background:var(--al)}} .mb i.ab{{background:var(--ab)}}
.pair{{font-family:var(--mono);font-size:12px;white-space:nowrap}} .pair b{{font-weight:600}} .pair .al{{color:var(--al)}} .pair .ab{{color:var(--ab)}}
.mult{{display:inline-block;margin-left:6px;font-family:var(--mono);font-size:11px;color:var(--err)}}
.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.read{{display:grid;gap:12px;margin-top:4px}}
.r{{border-left:3px solid var(--line);padding:3px 0 3px 14px;font-size:14.5px}} .r b{{font-weight:600}} .r.key{{border-left-color:var(--accent)}}
.note{{font-size:12.5px;color:var(--muted)}}
.notes{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
footer{{border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:12.5px}}
@media (max-width:640px){{.tiles,.twocol,.notes{{grid-template-columns:1fr}} .c span{{font-size:17px}} table{{font-size:12px}} th,td{{padding:7px 5px}}}}
.tablewrap{{overflow-x:auto}}
</style>

<div class="wrap">
  <p class="eyebrow">Results · AuditBench defensive axis · aligned vs refusal-ablated gpt-oss-120b</p>
  <h1>Defence drops — and it's precision that collapses, not recall</h1>
  <p class="lede">Across all four attack-investigation tasks the refusal-ablated model's F1 falls. It still <b>finds the
    attacks</b> (true-positive rate flat or higher), but it <b>stops being able to call anything benign</b>: false positives
    rise 3–44×, and it never once returns "nothing suspicious here."</p>
  <p class="sub">Scored with the upstream AuditBench metric (unchanged scoring logic, robust parser), HIGH_SUSPICIOUS
    threshold, {n_scn} scenarios per task pooled across Lab and OpTC datasets, 3 attempts each. Δ = ablated − aligned.</p>
  <div class="legend"><span><i class="sw al"></i>Aligned (control)</span><span><i class="sw ab"></i>Ablated (refusal-removed)</span></div>

  <!-- 1. HEADLINE -->
  <section>
    <div class="sec-head"><p class="eyebrow">Headline</p><h2>Did defensive F1 drop?</h2>
      <p>Point estimate and 95% paired-bootstrap interval (scenarios resampled, 2000 draws). Red = interval excludes zero.</p></div>
    <div class="card"><div class="ct">ΔF1 per task</div>{ci_svg}</div>
  </section>

  <!-- 2. CONFUSION TILES -->
  <section>
    <div class="sec-head"><p class="eyebrow">The full picture</p><h2>Confusion matrices, side by side</h2>
      <p>Rows are what actually happened, columns are what the model said. Green cells are correct, red are errors.
        Counts are mean-over-3-attempts, so they can be fractional. <b>TN is counted per benign log line</b> (tens of
        thousands), which is why FPR and TNR look almost unchanged even as FP counts explode — read FP and precision
        for the benign side.</p></div>
    {tiles}
  </section>

  <!-- 3. RATES -->
  <section>
    <div class="sec-head"><p class="eyebrow">Rates at a glance</p><h2>TPR · FNR · FPR · TNR · Precision</h2>
      <p>Each cell reads aligned → ablated. Bars are drawn to the same scale within a column (precision column scaled to 20%).</p></div>
    <div class="card tablewrap"><table>
      <tr><th>Task</th><th>TPR (recall)</th><th>FNR (missed)</th><th>FPR <small>(rel. change)</small></th><th>TNR</th><th>Precision</th></tr>
      {rate_rows}
    </table></div>
  </section>

  <!-- 4. MECHANISM -->
  <section>
    <div class="sec-head"><p class="eyebrow">Mechanism</p><h2>Recall held. Precision collapsed.</h2>
      <p>The two halves of the story: the ablated model detects attacks about as well as the aligned one, but produces
        vastly more false alarms.</p></div>
    <div class="twocol">
      <div class="card"><div class="ct">True-positive rate (attack recall)</div>{tpr_svg}</div>
      <div class="card"><div class="ct">False-positive count (mean over attempts)</div>{fp_svg}</div>
    </div>
  </section>

  <!-- 5. EMPTY RESPONSES -->
  <section>
    <div class="sec-head"><p class="eyebrow">Behavioural signature</p><h2>"Nothing suspicious here" disappears</h2>
      <p>Share of chunk-attempts where the model returned an empty response — an explicit judgement that the window is
        benign. The aligned model does this on a third to a half of windows. The ablated model never does.</p></div>
    <div class="card"><div class="ct">Empty-response rate per task</div>{empty_svg}</div>
  </section>

  <!-- 6. PASS@K -->
  <section>
    <div class="sec-head"><p class="eyebrow">pass@1 · lenient definition</p><h2>Attack scenarios vs benign scenarios</h2>
      <p>Attack success = at least one ground-truth item detected in the attempt. Benign success = zero false positives
        in the attempt. The benign side is where the ablated model fails almost completely.</p></div>
    <div class="twocol">
      <div class="card"><div class="ct">Attack scenarios — pass@1</div>{pk_att_svg}</div>
      <div class="card"><div class="ct">Benign scenarios — pass@1</div>{pk_ben_svg}</div>
    </div>
  </section>

  <!-- 7. ROBUSTNESS -->
  <section>
    <div class="sec-head"><p class="eyebrow">Robustness</p><h2>Same direction under a looser threshold</h2>
      <p>Counting MEDIUM_SUSPICIOUS as a detection too shrinks the gap but every delta stays negative.</p></div>
    <div class="card tablewrap"><table>
      <tr><th>Task</th><th>ΔF1 · HIGH only (headline)</th><th>ΔF1 · HIGH + MEDIUM</th></tr>
      {thr_rows}
    </table></div>
  </section>

  <!-- 8. INTERPRETATION -->
  <section>
    <div class="sec-head"><p class="eyebrow">What this means for the research question</p><h2>Reading the signature</h2></div>
    <div class="read">
      <div class="r key"><b>This is not sandbagging by omission.</b> Deliberate defensive under-performance would show up as
        missed attacks — a lower TPR. Instead TPR is flat or higher. The ablated model is not hiding detections.</div>
      <div class="r"><b>The lost capability is restraint.</b> Removing the refusal direction appears to have removed the model's
        ability to decline to flag — to judge a window benign. Everything becomes suspicious, so precision and F1 collapse.</div>
      <div class="r"><b>Defence-only so far.</b> Whether this is targeted or a general degradation still needs the offence
        benchmarks as the capability anchor (the plan's 2×2), plus the general-capability control.</div>
    </div>
  </section>

  <!-- 9. METHOD -->
  <section>
    <div class="sec-head"><p class="eyebrow">Method &amp; caveats</p><h2>How the numbers were produced</h2></div>
    <div class="notes">
      <p class="note"><b>Scoring.</b> Upstream <code>task_computemetrics.py</code> with its TP/FP/FN logic, entity fuzzy-matching
        and aggregation untouched; only the brittle positional line-parser was replaced by a JSON parser (gpt-oss emits
        multi-line JSON arrays the original could not read).</p>
      <p class="note"><b>Unparseable findings</b> count as no detection and are reported: 3–10% of findings per side, comparable
        across models. Verdict spelling (<code>HIGH_SUSPIOUS</code>) repaired symmetrically; date-less timestamps given the
        chunk's date.</p>
      <p class="note"><b>Pooling.</b> Per-task F1 is micro-averaged over Lab + OpTC and edge + raw cells (summed counts), so
        OpTC's large FP mass dominates. Several Lab entity cells score 0 for both models.</p>
      <p class="note"><b>Parity.</b> Identical vLLM 0.31, TP=2, 131k context, temp 0.7, seed 1234, reasoning high, prompts and
        scaffold. Only the weights differ.</p>
    </div>
  </section>

  <footer>Source: <span class="mono">results/auditbench/rescored.json</span> · <span class="mono">scripts/rescore_auditbench.py</span>.
    Aligned <span class="mono">openai/gpt-oss-120b</span> vs ablated <span class="mono">wangzhang/gpt-oss-120b-abliterated</span>.</footer>
</div>'''

out = REPO / "notes/results-spec.html"
out.write_text(html)
print("wrote", out, f"({len(html)} bytes)")
