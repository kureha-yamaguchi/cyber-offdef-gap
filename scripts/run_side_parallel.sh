#!/usr/bin/env bash
# Run the FULL AuditBench benchmark for ONE side (aligned|ablated) with 8 harness
# processes in parallel, partitioned by (dataset x task) so they write to disjoint
# upstream output paths. This saturates the shared vLLM endpoint's concurrency
# (~8x) instead of the single-stream ~30h/side the serial harness would take.
#
# Every process uses IDENTICAL sampling/parity flags — only --side (weights) and the
# (dataset,task) partition differ. Each writes its own results/auditbench/<stamp>_<side>
# dir; aggregate across them afterwards with scripts/aggregate_auditbench.py.
#
# Usage: bash scripts/run_side_parallel.sh <aligned|ablated>
set -euo pipefail
SIDE="${1:?usage: run_side_parallel.sh <aligned|ablated>}"
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TASKS=(classification lm persistence exfiltration)
MAXTOK=32000
NUMRUNS=3
LOGDIR="results/auditbench/_logs_${SIDE}"
mkdir -p "$LOGDIR"

pids=()
launch () { # $1=dataset  $2="reps..."
  local ds="$1"; shift
  local reps="$*"
  for t in "${TASKS[@]}"; do
    echo "[launch] $SIDE $ds $t reps=[$reps]"
    $PY harnesses/auditbench/run_auditbench.py --side "$SIDE" \
      --dataset "$ds" --tasks "$t" --reps $reps --types attack benign \
      --num-runs "$NUMRUNS" --max-tokens "$MAXTOK" \
      > "$LOGDIR/${ds}_${t}.log" 2>&1 &
    pids+=($!)
    sleep 2   # stagger so per-second result-dir stamps stay unique
  done
}

launch labgen edge raw
launch optc edge

echo "[info] launched ${#pids[@]} parallel runs for side=$SIDE; waiting..."
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[done] side=$SIDE all runs finished (rc=$rc)"
echo "[dirs]"; ls -1d results/auditbench/*_"${SIDE}" 2>/dev/null | tail -20
exit $rc
