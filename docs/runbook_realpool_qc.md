# Runbook — real-pool ΔG‡ QC run (shared-queue orchestrator)

First-principles ΔG‡ (DMSO, gas-Hessian + PCM single point) for a compact, diverse set
of **non-Lu** aryl-halide substrates, via a shared work queue. Companion to the Lu_74
solvated run; uses the gas-Hessian + PCM single-point method documented in
`notes/2026-06-23_solvent_freq_pcm_singlepoint.md`.

## Prerequisites

```
conda activate snar-qc          # Psi4 / xTB / autodE + RDKit
pip install -e . --no-deps      # once, so workers can import snar_qc
```

## 1. Build the substrate slice (one-off, fast)

`scripts/build_realpool_slice.py` selects substrates from an arylator pool TSV, excluding
a reference set (matched on InChIKey connectivity block). Paths are arguments — nothing
about the pool is baked into the repo.

```
python scripts/build_realpool_slice.py \
    --pool <pool.tsv> --reference <reference.csv> \
    --out-dir data/external/realpool_qc --target 150 --seed 42
```

Selection: leaving group ∈ {F, Cl, Br} (sampled Cl≈60 % / F≈20 % / Br≈20 %); heavy-atom
count ≤ the pool median ("prefer small"); long **flexible** chains dropped
(`--max-rotatable`, `--max-chain`); both monocyclic- and fused-aromatic ring systems
represented; distinct Bemis–Murcko scaffolds preferred within each stratum. Writes
`realpool_qc_slice.csv` + a provenance `README.md` and prints the funnel + composition.

## 2. Smoke test (optional, before the multi-hour run)

Run a couple of substrates end to end to confirm compute → sidecar → zip → archive →
heartbeat all wire up:

```
python scripts/run_qc_queue.py \
    --substrates data/external/realpool_qc/realpool_qc_slice.csv \
    --outdir data/processed/realpool_smoke --solvent DMSO \
    --limit 2 --heartbeat-min 2 --archive-dir <ARCHIVE_DIR>
```

## 3. Full run

```
python scripts/run_qc_queue.py \
    --substrates data/external/realpool_qc/realpool_qc_slice.csv \
    --outdir data/processed/realpool_dmso \
    --solvent DMSO --coordinate concerted \
    --archive-dir <ARCHIVE_DIR>
```

- **Shared queue.** Every substrate is one task on a `ProcessPoolExecutor`; idle workers
  pull the next task (no pre-set shards). Concurrency is **auto-tuned** from cores + RAM
  (override with `--workers N --threads M`); `--mem` sets GB/worker for Psi4.
- **Per-task archive.** On each completion the task dir is zipped and the `.zip` copied to
  `--archive-dir` (created if missing). Omit `--archive-dir` to skip archiving.
- **Resumable.** Re-run the same command to continue; finished sidecars are skipped.
  `--retry` re-runs non-completed substrates; `--force` re-runs everything.

## Heartbeat (every `--heartbeat-min`, default 30)

```
---- heartbeat @ 1h 02m (4x3 threads) ----
progress : 12/150 done (8 completed, 2 no-saddle, 2 failed) | 4 running | 134 pending
errors   : ts_opt_freq/MemoryError x2
running  : EN300_17347[ts_opt_freq,41m]  EN300_19151[dft_sps,12m] ...
resources: CPU 78% | RAM 21.3/63.7 GB (33%)
pace     : mean 38m/task | ETA ~3h 50m
--------------------------------------------------------
```

Phase per running task is inferred from on-disk artifacts (xtb scan → DFT scan points →
TS opt+freq). Error reasons are `<stage>/<type>` from each failed sidecar.

## Outputs

- `data/processed/realpool_dmso/<tag>/result.json` — per-substrate sidecar (gitignored).
- `data/processed/realpool_dmso/summary.json` — roll-up of all sidecars.
- `<ARCHIVE_DIR>/<tag>.zip` — per-task archive copied as each finishes.

`<tag>` is the catalogue code (e.g. `EN300_17347`). On Windows an `--archive-dir` under
`$env:OneDrive` syncs results for remote viewing.
