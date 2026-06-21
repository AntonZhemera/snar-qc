#!/usr/bin/env bash
# Parallel POC batch launcher.
#
# Runs scripts/run_poc.py over the Lu_74 slice with several substrates in flight at
# once. Each worker handles a disjoint set of lu_ids (via --only) in its own process,
# writing resumable per-substrate sidecars, so the batch survives a crash and re-running
# skips finished substrates. The transition-state Hessian dominates the ~40 min/substrate
# cost, so a handful of workers (each given a few threads) finishes ~10 substrates in a
# couple of hours on a 16-core box.
#
# Usage:
#   conda run -n snar-qc bash scripts/run_poc_batch.sh \
#       data/external/lu74_poc_slice.csv data/processed/poc_run
#
# Env overrides: N_WORKERS (default 5), THREADS (default 3), MEM_GB (default 6).
set -euo pipefail

SLICE="${1:-data/external/lu74_poc_slice.csv}"
OUTDIR="${2:-data/processed/poc_run}"
N_WORKERS="${N_WORKERS:-5}"
THREADS="${THREADS:-3}"
MEM_GB="${MEM_GB:-6}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# Read the lu_id column (skip header) and round-robin into N_WORKERS buckets.
mapfile -t IDS < <(tail -n +2 "$SLICE" | cut -d, -f1)
echo "Batch: ${#IDS[@]} substrates, ${N_WORKERS} workers x ${THREADS} threads, outdir=${OUTDIR}"

declare -a BUCKETS
for i in "${!IDS[@]}"; do
    w=$(( i % N_WORKERS ))
    BUCKETS[$w]="${BUCKETS[$w]:-}${BUCKETS[$w]:+,}${IDS[$i]}"
done

mkdir -p "$OUTDIR"
pids=()
for w in "${!BUCKETS[@]}"; do
    only="${BUCKETS[$w]}"
    [ -z "$only" ] && continue
    log="${OUTDIR}/worker_${w}.log"
    echo "  worker ${w}: lu_ids=${only} -> ${log}"
    OMP_NUM_THREADS="$THREADS" python -u scripts/run_poc.py \
        --substrates "$SLICE" --only "$only" --outdir "$OUTDIR" \
        --n-procs "$THREADS" --mem "$MEM_GB" --retry \
        >"$log" 2>&1 &
    pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
echo "All workers finished (rc=${rc}). Sidecars under ${OUTDIR}."
exit "$rc"
