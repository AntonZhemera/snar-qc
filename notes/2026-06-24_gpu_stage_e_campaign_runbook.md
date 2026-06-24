# Runbook — Stage E POC revalidation on the GPU backend (run manually)

Closes the gpu4pyscf backend plan (`plans/2026-06-23_gpu4pyscf_backend/`, Stage E /
`prompt_05_Revalidation.md`). Re-run the 10-substrate Lu POC slice through the GPU backend
and confirm the within-leaving-group correlations hold vs the Psi4 baseline. The pipeline
is **unblocked and proven end to end** (`notes/2026-06-24_gpu_stage_e_unblock.md`); this is
the production run.

## Why it needs no babysitting

Unlike the 4-worker CPU DMSO campaign (which OOM-killed workers and was babysat), this is:

- **Sequential, one GPU job at a time** — no multi-worker memory contention.
- **VRAM-stable** — `free_device_memory()` keeps free VRAM flat across the scan (the bug
  that tripped a 4 GB card mid-scan is fixed).
- **Resumable** — `scripts/run_poc.py` writes a per-substrate `result.json` sidecar and
  skips terminal ones; `--retry` re-runs only the incomplete.
- **Failure is data** — `compute_barrier` catches a flaky TS and records `status`
  (`ts_not_saddle` / `no_peak` / `error`); one bad substrate never sinks the batch.

So: launch it and walk away. Expected ~**31 min/substrate** (TS opt+freq ~24 min dominates)
⇒ ~**5 h** for 10 substrates on the RTX 3050 Ti.

## Prerequisites

- Env **`gpuqc`** with a CUDA device free. It now carries **xtb** (the relaxed-scan CLI) —
  no `PATH` hack needed. Recreate from `environment-gpu.yml` if absent.
- Run from the repo root (`scripts/run_poc.py` resolves `src/` itself).

## Launch

```bash
conda activate gpuqc            # GPU-on-device env (has xtb + gpu4pyscf)
export SNAR_QC_BACKEND=gpu4pyscf
export SNAR_QC_REQUIRE_GPU=1    # error out instead of silently falling back to Psi4
                               # (Psi4 isn't installed here -- a fallback would fail anyway)

python scripts/run_poc.py \
  --substrates data/external/lu74_poc_slice.csv \
  --outdir     data/processed/gpu_stage_e \
  --n-procs 1 --mem 2          # ignored by the GPU backend; harmless

# Resume after an interruption (re-runs only substrates without a terminal sidecar):
python scripts/run_poc.py --substrates data/external/lu74_poc_slice.csv \
  --outdir data/processed/gpu_stage_e --n-procs 1 --mem 2 --retry
```

Gas phase (no `--solvent`) and the default concerted coordinate — matching the
2026-06-21 POC baseline. Run it under `nohup`/`tmux` if the session may disconnect.

## Watch (optional, not required)

- Progress: `data/processed/gpu_stage_e/<lu_id>/result.json` appears per substrate; the
  final `summary.json` rolls up `N/10 reached a confirmed saddle`.
- Health: `nvidia-smi` (~1 GB used, one job); a clean run never approaches the 4 GB ceiling.

## Validate (after the batch)

```bash
python scripts/validate_poc.py \
  --slice data/external/lu74_poc_slice.csv \
  --run   data/processed/gpu_stage_e \
  --outdir notes/assets/gpu_stage_e      # separate dir: do NOT overwrite the Psi4 baseline
```

**Pass criterion** (vs `notes/2026-06-21_poc_deltag_validation.md`): within-leaving-group
Spearman/Pearson hold — **Cl ρ≈0.96 / R²≈0.95, F ρ=1.0** — within noise. Drift beyond noise
⇒ investigate the thermochem convention (analytic vs FD frequencies, Stage C).

## On success

Record a dated findings note with the GPU correlation table, then move
`plans/2026-06-23_gpu4pyscf_backend/` → `plans/archive/`. PCM/SMD stays a separate
decision, gated to the `2026-06-22_solvation_revalidation` plan.
