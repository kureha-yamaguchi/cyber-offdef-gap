#!/usr/bin/env python3
"""Robust AuditBench re-scorer: swap ONLY the output parser, keep upstream scoring intact.

Why: the upstream metric (`metrics_comp/task_computemetrics.py`) reads model output with a
positional line-parser that assumes one JSON field per line in a fixed order. gpt-oss emits valid
but pretty-printed, multi-line JSON (arrays of findings, occasionally `"..." +` string
concatenation and misspelled verdict labels), which that parser mis-reads -> "format following
error" -> None verdict -> crash. The paper's TP/FP/FN semantics are fine; only parsing is brittle.

What this does (decisions agreed 2026-10-08):
  * Fidelity: monkeypatch the four upstream `parse_*_llmoutput` methods with a robust JSON parser
    that yields the IDENTICAL `llmresponse_dict` tuple layouts, then run the UNCHANGED upstream
    `TaskMetrics.compute_correctness` / `ErrorInvestigation` matching + aggregation.
  * Unparseable findings (no verdict, or entity/timestamp missing) count as "no detection":
    they are dropped (no TP, no FP), and the format-error rate is reported per model/task.
  * Verdict spelling repaired symmetrically (HIGH_SUSPIOUS -> HIGH_SUSPICIOUS); severity kept.
  * Both thresholds reported: more_defensive=0 (HIGH only, paper default = headline) and =1 (HIGH+MED).
  * pass@1 / pass@3 (PLAN §5) in two definitions: lenient (attack: >=1 GT item detected;
    benign: zero FPs) and strict (attack: all GT recovered i.e. FN==0; benign: zero FPs).
  * Paired scenario-level bootstrap (2000 draws) for the aligned->ablated delta-F1 per task.

Outputs: results/auditbench/rescored.json, results/auditbench/rescored.md
Usage:   .venv/bin/python scripts/rescore_auditbench.py   (run from repo; needs ~/auditlogsbench)
"""
from __future__ import annotations
import json, os, re, sys, random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("AUDITBENCH_DIR", str(Path.home() / "auditlogsbench")))
INFER = UPSTREAM / "inference"
os.chdir(INFER)
sys.path.insert(0, str(INFER))

import numpy as np                                   # noqa: E402
from metrics_comp import task_computemetrics as tc    # noqa: E402
from metrics_comp.factory import ModuleFactory        # noqa: E402
import parser.parser_lmv2 as plm                      # noqa: E402
import parser.parser_persistencev2 as pps             # noqa: E402
import parser.parser_exfiltrationv2 as pex            # noqa: E402
import parser.parser_classificationv2 as pcl          # noqa: E402

SIDES = {"aligned": "gptoss120b-aligned", "ablated": "gptoss120b-ablated"}
PLAN = [("labgen", ["edge", "raw"]), ("optc", ["edge"])]
TASKS = ["classification", "lm", "persistence", "exfiltration"]
TYPES = ["attack", "benign"]
HEADER_RX = re.compile(r"\*\*\*\*Starting line num: (\d+)\. Num lines: (\d+)\. (.*?)\*\*\*\*")
CONCAT_RX = re.compile(r'"\s*\+\s*\n\s*"')            # joins  "...abc" +\n "def..."  -> "...abcdef..."
FENCE_RX = re.compile(r"```(?:json)?", re.I)
VERDICT_RX = re.compile(r"^(HIGH|MEDIUM|LOW)[_\- ]?SUSP", re.I)

# format-error accounting, attributed to the side currently being scored
FMT: dict = defaultdict(lambda: {"blocks": 0, "blocks_unparsed": 0, "empty_responses": 0, "objects": 0,
                                 "objects_dropped": 0, "dropped_no_verdict": 0, "dropped_no_entity": 0,
                                 "dropped_bad_ts": 0, "ts_repaired": 0, "ts_date_from_chunk": 0})
CUR = {"side": None, "task": None, "count": True}   # count=False during re-parses so stats tally once

import datetime as _dt
ISO_TS_RX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
CHUNK_DATE_RX = re.compile(r"\d{4}-\d{2}-\d{2}")


def tolerant_ts(obj, ts_raw, chunk_date):
    """Normalise a model timestamp to the GT-friendly format the upstream matcher expects.

    1) Unicode cleanup (narrow no-break spaces, typographic dashes), then upstream's own converter.
    2) If that fails: take the start of a range ("2:04:44-46 PM" -> "2:04:44 PM"), pull a time token,
       and a date from the string if present ("...PM 11/27/2024" or ISO) else from the chunk window.
    Returns (gt_timestamp | None, how) with how in {direct, repaired, date_from_chunk, fail}.
    """
    if ts_raw in (None, ""):
        return None, "fail"
    s = str(ts_raw).replace(" ", " ").replace(" ", " ")
    s = re.sub(r"[‐-―−]", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    out = obj.convert_ts_to_groundtruth_friendly(s)
    if out and (obj.dataset != "optc" or ISO_TS_RX.match(out)):
        return out, "direct"
    s2 = re.sub(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*\d{1,2}(?::\d{2}){0,2}", r"\1", s)   # range -> start
    tm = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s2)
    if not tm:
        return None, "fail"
    h, mi, se = int(tm.group(1)), int(tm.group(2)), int(tm.group(3) or 0)
    ampm = re.search(r"\b(AM|PM)\b", s2, re.I)
    if ampm:
        p = ampm.group(1).upper()
        if p == "PM" and h < 12: h += 12
        if p == "AM" and h == 12: h = 0
    date = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s2)
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", s2)
    how = "repaired"
    if date:
        y, mo, d = int(date.group(3)), int(date.group(1)), int(date.group(2))
    elif iso:
        y, mo, d = map(int, iso.groups())
    elif chunk_date:
        y, mo, d = map(int, chunk_date.split("-")); how = "date_from_chunk"
    else:
        return None, "fail"
    try:
        dt = _dt.datetime(y, mo, d, h, mi, se)
    except ValueError:
        return None, "fail"
    fmt = "%Y-%m-%d %H:%M:%S" if (obj.dataset == "optc" or obj.task == "lm") else "%Y-%m-%d %H:%M"
    return dt.strftime(fmt), how


# ----------------------------------------------------------------------------- robust parsing
def canon_verdict(v):
    if v is None:
        return None
    m = VERDICT_RX.match(str(v).strip())
    return f"{m.group(1).upper()}_SUSPICIOUS" if m else None


def getkey(d: dict, *names):
    """Case/space/underscore-insensitive key lookup."""
    norm = {re.sub(r"[\s_]+", "", k).lower(): k for k in d}
    for n in names:
        k = norm.get(re.sub(r"[\s_]+", "", n).lower())
        if k is not None:
            return d[k]
    return None


def extract_objects(block: str) -> list[dict]:
    """Return the list of JSON finding-objects in one chunk-round block (tolerant)."""
    txt = FENCE_RX.sub("", CONCAT_RX.sub("", block))
    dec = json.JSONDecoder()
    # 1) primary: decode the first JSON value found
    for i, ch in enumerate(txt):
        if ch in "[{":
            try:
                val, _ = dec.raw_decode(txt[i:])
                if isinstance(val, dict):
                    return [val]
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
            except json.JSONDecodeError:
                break
            break
    # 2) fallback: balanced-brace scan, decode each top-level {...} that parses
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(txt):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    o = json.loads(txt[start:i + 1])
                    if isinstance(o, dict) and ("verdict" in {k.lower() for k in o}):
                        out.append(o)
                except json.JSONDecodeError:
                    pass
                start = None
    if out:
        return out
    # 3) last resort: regex a single object's string fields
    o = {}
    for m in re.finditer(r'"([^"\n]{1,40})"\s*:\s*"((?:[^"\\]|\\.)*)"', txt):
        o.setdefault(m.group(1), m.group(2))
    return [o] if o and getkey(o, "verdict") is not None else []


def robust_parse(self, out_fpath):
    """Drop-in for the upstream parse_*_llmoutput methods (same return contract)."""
    task, side = self.task, CUR["side"]
    st = FMT[(side, task)] if CUR["count"] else defaultdict(int)   # throwaway when not counting
    text = Path(out_fpath).read_text(errors="replace")
    llm = defaultdict(dict)
    cnt = defaultdict(int)
    heads = list(HEADER_RX.finditer(text))
    for j, m in enumerate(heads):
        key = (int(m.group(1)), int(m.group(2)), m.group(3))
        cnt[key] += 1
        r = cnt[key]
        llm[key][r] = []
        cd = CHUNK_DATE_RX.search(m.group(3))
        chunk_date = cd.group(0) if cd else None
        block = text[m.end(): heads[j + 1].start() if j + 1 < len(heads) else len(text)]
        block = block.split("-----------------------")[0]
        objs = extract_objects(block)
        st["blocks"] += 1
        if not objs:
            st["blocks_unparsed"] += 1
            continue
        for o in objs:
            if not o:                                   # `{}` = model reports nothing suspicious here
                st["empty_responses"] += 1              # legitimate no-detection, NOT a format error
                continue
            st["objects"] += 1
            verdict = canon_verdict(getkey(o, "verdict"))
            if verdict is None:
                st["objects_dropped"] += 1; st["dropped_no_verdict"] += 1
                continue
            kll = getkey(o, "key log lines", "key log line")
            ef, ea, de = getkey(o, "evidence for"), getkey(o, "evidence against"), getkey(o, "deliberation")
            pid = getkey(o, "pid")
            if task == "classification":
                llm[key][r].append((r, kll, ef, ea, de, verdict))
                continue
            ts, how = tolerant_ts(self, getkey(o, "timestamp"), chunk_date)
            if how == "repaired": st["ts_repaired"] += 1
            elif how == "date_from_chunk": st["ts_repaired"] += 1; st["ts_date_from_chunk"] += 1
            ent_raw = getkey(o, {"lm": "external host", "persistence": "MITRE technique",
                                 "exfiltration": "exfiltrated data"}[task])
            # a list-valued entity = several assertions -> one finding per element (stringified)
            ents = ent_raw if isinstance(ent_raw, list) else [ent_raw]
            ents = [str(e).strip() for e in ents if e not in (None, "") and str(e).strip()]
            if not ents:
                st["objects_dropped"] += 1; st["dropped_no_entity"] += 1   # no entity named -> no detection
                continue
            if ts in (None, ""):
                st["objects_dropped"] += 1; st["dropped_bad_ts"] += 1      # unparseable ts -> no detection
                continue
            pid_s = str(pid) if pid is not None else None
            for ent in ents:
                if task == "lm":
                    tup = (r, ts, ent, pid_s, getkey(o, "username"), kll, ef, ea, de, verdict)
                elif task == "persistence":
                    tup = (r, ts, ent, pid_s, kll, ef, ea, de, verdict)
                else:  # exfiltration
                    tup = (r, ts, ent, pid_s, getkey(o, "exfiltration method"), kll, ef, ea, de, verdict)
                llm[key][r].append(tup)
    num_rounds = max(cnt.values()) if cnt else 0
    return llm, num_rounds


# patch the four upstream parsers (class objects shared with ModuleFactory)
plm.ParserLateralMovement.parse_lateralmovement_llmoutput = robust_parse
pps.ParserPersistence.parse_persistence_llmoutput = robust_parse
pex.ParserExfiltration.parse_exfiltration_llmoutput = robust_parse
pcl.ParserClassification.parse_classification_llmoutput = robust_parse


# ----------------------------------------------------------------------------- scoring
def outpaths(dataset, rep, task, typ, tag):
    return tc.computation_setup("v2", dataset, rep, "gpt", None, 1, tag, typ, task, 1)


def score_cell(dataset, rep, task, side, more_defensive):
    """Per-scenario contributions (upstream compute_correctness) + per-round pass@k."""
    tag = SIDES[side]
    CUR.update(side=side, task=task, count=(more_defensive == 0))   # tally format stats once (first pass)
    modules = ModuleFactory("v2")
    per_scn = {}            # scn_id -> {tp,fp,fn,tn,type}
    passk = {}              # scn_id -> {"lenient":[...rounds], "strict":[...]}
    for typ in TYPES:
        lst = outpaths(dataset, rep, task, typ, tag)
        if not lst:
            continue
        tm = tc.TaskMetrics(task, "v2", lst, dataset)
        tp, fp, fn, tn, _acc = tm.compute_correctness(more_defensive)
        # per-round detail for pass@k (reuse upstream matching)
        for tup in lst:
            fname = os.path.basename(tup[0])
            scn = f"{dataset}/{rep}/{typ}/{fname.split('_edge_')[-1].split('_raw_')[-1].split('.log')[0]}"
            # per-scenario aggregate: keys in compute_correctness dicts are input_fname
            in_key = "_".join(tup[1].split("_")[3:6]) if dataset == "labgen" else "_".join(tup[1].split("_")[3:7])
            per_scn[scn] = {"type": typ, "tp": float(tp.get(in_key, 0) or 0), "fp": float(fp.get(in_key, 0) or 0),
                            "fn": float(fn.get(in_key, 0) or 0), "tn": float(tn.get(in_key, 0) or 0)}
            # rounds
            obj = {"lm": modules.ParserLateralMovement, "persistence": modules.ParserPersistence,
                   "exfiltration": modules.ParserExfiltration, "classification": modules.ParserClassification}[task](dataset, task)
            _c = CUR["count"]; CUR["count"] = False            # pass@k re-parse: don't double count
            llm, _ = robust_parse(obj, tup[0])
            CUR["count"] = _c
            ei = modules.ErrorInvestigation(dataset, task)
            gtkey = tc.get_groundtruth_key(tup, dataset)
            len_r, str_r = [], []
            if task == "classification":
                ben, mal, _ = ei.compute_chunkdecision(llm, more_defensive)
                for r in sorted(mal):
                    ok = (mal[r] > 0) if typ == "attack" else (mal[r] == 0)
                    len_r.append(ok); str_r.append(ok)
            else:
                gt = obj.read_task_groudtruth(gtkey)
                fpd, tpd, fnd, _vd = ei.compute_confusionmatrix(llm, gt, gtkey, more_defensive)
                rounds = sorted(set(fpd) | set(tpd) | set(fnd))
                for r in rounds:
                    tpc = sum(len(v) for v in tpd.get(r, {}).values())
                    fpc = sum(len(v) for v in fpd.get(r, {}).values())
                    if typ == "attack":
                        len_r.append(tpc > 0); str_r.append(tpc > 0 and (fnd.get(r, 0) or 0) == 0)
                    else:
                        len_r.append(fpc == 0); str_r.append(fpc == 0)
            passk[scn] = {"type": typ, "lenient": len_r, "strict": str_r}
    return per_scn, passk


def f1_from(tp, fp, fn):
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else None


def summarise(per_scn):
    tp = sum(v["tp"] for v in per_scn.values()); fp = sum(v["fp"] for v in per_scn.values())
    fn = sum(v["fn"] for v in per_scn.values()); tn = sum(v["tn"] for v in per_scn.values())
    return {"tp": round(tp, 2), "fp": round(fp, 2), "fn": round(fn, 2), "tn": round(tn, 2),
            "tpr": round(tp / (tp + fn), 4) if tp + fn else None,
            "fpr": round(fp / (fp + tn), 4) if fp + tn else None,
            "f1": round(f1_from(tp, fp, fn), 4) if f1_from(tp, fp, fn) is not None else None,
            "n_scenarios": len(per_scn)}


def passk_summary(passk):
    out = {}
    for typ in TYPES:
        for mode in ("lenient", "strict"):
            rows = [v[mode] for v in passk.values() if v["type"] == typ and v[mode]]
            if not rows:
                continue
            p1 = float(np.mean([np.mean(r) for r in rows])); p3 = float(np.mean([any(r) for r in rows]))
            out[f"{typ}/{mode}"] = {"pass@1": round(p1, 4), "pass@3": round(p3, 4), "n": len(rows)}
    return out


def paired_bootstrap(al, ab, iters=2000, seed=1234):
    keys = sorted(set(al) & set(ab))
    if not keys:
        return None
    rng = random.Random(seed)
    def f1(side, idx):
        tp = sum(side[keys[i]]["tp"] for i in idx); fp = sum(side[keys[i]]["fp"] for i in idx)
        fn = sum(side[keys[i]]["fn"] for i in idx); return f1_from(tp, fp, fn)
    n = len(keys); deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        a, b = f1(al, idx), f1(ab, idx)
        if a is not None and b is not None:
            deltas.append(b - a)
    if not deltas:
        return None
    deltas.sort()
    return {"delta_f1": round(float(np.mean(deltas)), 4),
            "ci95": [round(deltas[int(.025 * len(deltas))], 4), round(deltas[int(.975 * len(deltas)) - 1], 4)],
            "n_scenarios": n, "iters": iters}


# ----------------------------------------------------------------------------- main
def main():
    report = {"thresholds": {}, "format_errors": {}}
    for md in (0, 1):
        thr = "HIGH_only" if md == 0 else "HIGH+MEDIUM"
        T = {"cells": {}, "per_task": {}, "passk": {}, "delta": {}}
        pooled = {t: {s: {} for s in SIDES} for t in TASKS}      # task -> side -> per_scn
        pooled_pk = {t: {s: {} for s in SIDES} for t in TASKS}
        for dataset, reps in PLAN:
            for rep in reps:
                for task in TASKS:
                    for side in SIDES:
                        per_scn, passk = score_cell(dataset, rep, task, side, md)
                        T["cells"][f"{dataset}/{task}/{rep}/{side}"] = summarise(per_scn)
                        pooled[task][side].update(per_scn); pooled_pk[task][side].update(passk)
                        print(f"[{thr}] {dataset}/{task}/{rep}/{side}: F1={T['cells'][f'{dataset}/{task}/{rep}/{side}']['f1']}")
        for task in TASKS:
            T["per_task"][task] = {s: summarise(pooled[task][s]) for s in SIDES}
            T["passk"][task] = {s: passk_summary(pooled_pk[task][s]) for s in SIDES}
            T["delta"][task] = paired_bootstrap(pooled[task]["aligned"], pooled[task]["ablated"])
        report["thresholds"][thr] = T
    # format-error rates (parsing is threshold-independent; FMT accumulates over both passes -> rates unaffected)
    for (side, task), st in FMT.items():
        report["format_errors"][f"{side}/{task}"] = {
            **st,
            "block_unparsed_rate": round(st["blocks_unparsed"] / st["blocks"], 4) if st["blocks"] else None,
            "object_drop_rate": round(st["objects_dropped"] / st["objects"], 4) if st["objects"] else None}

    out = REPO / "results" / "auditbench"
    (out / "rescored.json").write_text(json.dumps(report, indent=2))

    # markdown
    L = ["# AuditBench re-scored (robust parser, upstream scoring)", "",
         "F1 = 2·TP/(2·TP+FP+FN) from upstream per-scenario contributions. Δ = ablated − aligned, "
         "paired scenario bootstrap (2000 draws, 95% CI). Unparseable findings = no detection.", ""]
    for thr, T in report["thresholds"].items():
        L += [f"## Threshold: {thr}" + (" (headline / paper default)" if thr == "HIGH_only" else ""), "",
              "| task | side | TP | FP | FN | TPR | FPR | **F1** | ΔF1 [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
        for task in TASKS:
            d = T["delta"][task]
            dtxt = f"{d['delta_f1']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else "—"
            for i, side in enumerate(SIDES):
                m = T["per_task"][task][side]
                f = lambda x: f"{x:.3f}" if isinstance(x, float) else "—"
                L.append(f"| {task if i == 0 else ''} | {side} | {m['tp']} | {m['fp']} | {m['fn']} | {f(m['tpr'])} | {f(m['fpr'])} | **{f(m['f1'])}** | {dtxt if i == 0 else ''} |")
        L += ["", f"### pass@k ({thr})", "", "| task | type/def | aligned p@1 | aligned p@3 | ablated p@1 | ablated p@3 |", "|---|---|---|---|---|---|"]
        for task in TASKS:
            keys = sorted(set(T["passk"][task]["aligned"]) | set(T["passk"][task]["ablated"]))
            for k in keys:
                a = T["passk"][task]["aligned"].get(k, {}); b = T["passk"][task]["ablated"].get(k, {})
                g = lambda m, kk: f"{m[kk]:.3f}" if kk in m else "—"
                L.append(f"| {task} | {k} | {g(a,'pass@1')} | {g(a,'pass@3')} | {g(b,'pass@1')} | {g(b,'pass@3')} |")
        L.append("")
    L += ["## Format handling (parser robustness, per side/task; counted once)", "",
          "`empty {}` = model explicitly reported nothing suspicious (legitimate no-detection, scored as benign). "
          "`dropped` = unparseable finding, counted as *no detection*. `ts repaired` = timestamp normalised "
          "(in parentheses: date inferred from the chunk window).", "",
          "| side/task | chunk-rounds | unparsed | empty {} | findings | dropped (rate) | no verdict | no entity | bad ts | ts repaired (date←chunk) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(report["format_errors"].items()):
        L.append(f"| {k} | {v['blocks']} | {v['blocks_unparsed']} | {v['empty_responses']} | {v['objects']} | "
                 f"{v['objects_dropped']} ({v['object_drop_rate']}) | {v['dropped_no_verdict']} | {v['dropped_no_entity']} "
                 f"| {v['dropped_bad_ts']} | {v['ts_repaired']} ({v['ts_date_from_chunk']}) |")
    (out / "rescored.md").write_text("\n".join(L) + "\n")
    print("\nwrote", out / "rescored.json", "and rescored.md")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
