#!/usr/bin/env bash
# Single-command launcher for the solvated cross-leaving-group ΔG‡ re-validation.
#
# One command runs the whole campaign on the extended Lu slice:
#   1. the resumable parallel batch (run_poc_batch.sh) with PCM solvation on, then
#   2. the validation join + per-leaving-group stats + scatter (validate_poc.py).
# Each substrate writes a resumable per-substrate sidecar, so a crash/restart just
# re-runs the unfinished ones. The transition-state Hessian dominates the multi-hour
# per-substrate cost; PCM adds per-SCF overhead, so budget generously.
#
# Run inside the snar-qc conda env, e.g. from the repo root:
#   conda run -n snar-qc bash scripts/run_solvated_validation.sh
# or, after `conda activate snar-qc`:
#   bash scripts/run_solvated_validation.sh
#
# Env overrides (all optional):
#   SLICE       extended slice CSV   (default data/external/lu74_solv_slice.csv)
#   OUTDIR      run output dir       (default data/processed/solv_run)
#   SOLVENT     PCMSolver solvent    (default DMSO; matches Lu's reaction solvent)
#   COORDINATE  concerted|addition  (default concerted)
#   N_WORKERS / THREADS / MEM_GB     parallelism knobs (see run_poc_batch.sh)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

SLICE="${SLICE:-data/external/lu74_solv_slice.csv}"
OUTDIR="${OUTDIR:-data/processed/solv_run}"
export SOLVENT="${SOLVENT:-DMSO}"
export COORDINATE="${COORDINATE:-concerted}"
export N_WORKERS="${N_WORKERS:-4}"
export THREADS="${THREADS:-4}"
export MEM_GB="${MEM_GB:-5}"

echo "=== solvated ΔG‡ re-validation ==="
echo "slice=${SLICE} outdir=${OUTDIR} solvent=${SOLVENT} coordinate=${COORDINATE}"
echo "workers=${N_WORKERS} threads=${THREADS} mem_gb=${MEM_GB}"

# 1. compute (resumable; PCM solvation on via SOLVENT).
bash scripts/run_poc_batch.sh "$SLICE" "$OUTDIR"

# 2. validate (correlation, per-leaving-group breakdown, scatter).
python scripts/validate_poc.py --slice "$SLICE" --run "$OUTDIR" --outdir notes/assets

echo "Done. Stats/scatter under notes/assets; sidecars under ${OUTDIR}."
