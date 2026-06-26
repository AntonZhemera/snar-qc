# Runbook — GPU DMSO revalidation campaign (IEF-PCM + SMD), run manually

> **Superseded as the standing procedure** by the living SOP
> [`docs/sop_snar_deltag.md`](../docs/sop_snar_deltag.md) (2026-06-26). This file is kept as
> the historical record of the one-off DMSO revalidation campaign; outcome:
> `notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`.

Brings the **gpu4pyscf backend** up to the DMSO frontier that the CPU/Psi4 path reached in
`notes/2026-06-24_dmso_recalc_lu74_arylators.md` (assets: `notes/assets/cpu_dmso/`). GPU
implicit solvation is now implemented (IEF-PCM **and** SMD; the SP-on-gas recipe that
mirrors `cpu_dmso`); this is the production run that validates it.

Two goals:

1. **Engine parity** — a GPU **IEF-PCM** run on the same 18-substrate slice `cpu_dmso`
   used, compared per-leaving-group against the Psi4 IEFPCM baseline. This is the solvated
   analogue of the gas-phase `cpu_gas` ↔ `gpu_stage_e` check (which held to <0.2 kcal/mol).
2. **New capability** — a GPU **SMD** run, a model the Psi4 1.10.2 path cannot provide.
   No CPU baseline exists; characterise the SMD−IEFPCM delta.

## Recipe (mirrors `cpu_dmso`, only the engine changes)

Gas-phase geometries and Hessians throughout; the solvent enters as an implicit-solvent
**single-point correction** on each gas geometry (`E(solv) − E(gas)`), plus solvated DFT
scan single points. xTB-GFN2 relaxed scan → DFT scan SPs (solvated) → **gas** TS opt+freq
(geomeTRIC, analytic Hessian) → **gas** ArX/amine references → ΔG‡(qh). The continuum model
is selected by `--solvent-model`.

## Why it needs no babysitting

Same as the gas Stage E run (`notes/2026-06-24_gpu_stage_e_campaign_runbook.md`):
sequential one-GPU-job-at-a-time, VRAM-stable (`free_device_memory()` between scan points),
resumable (`--retry` re-runs only incomplete substrates), failure-is-data (per-substrate
`status`). The solvated SP adds one extra SCF per species and shifts the scan SPs to the
continuum — modest cost on top of the gas pipeline.

## Reuse the gas backbone — run it once, sweep solvents on top

The gas backbone (xTB scan, gas TS/ArX/amine opt+freq) is **solvent-independent**, so a
different solvent/model only needs the **three** implicit-solvent single points on the
cached gas geometries. A gas run now persists that cache (`gas_thermo.json` + `*_opt.xyz`
per substrate), and `scripts/sweep_solvent.py` re-evaluates any solvent/model from it.

Per-substrate cost: a full gas+solvent run is ~**33–36 min** (gas TS opt+freq ~24 min
dominates); a **sweep** is just 3 SCFs, ~**1–2 min**. ETA for the 18-substrate slice:

| approach | ETA |
|---|---|
| two full runs (iefpcm, then smd) | ~**20 h** |
| **gas once + 2 sweeps** (recommended) | ~**11 h** (one gas pass + ~0.9 h of sweeps) |
| each *further* model (e.g. water, C-PCM) | ~**0.5 h** |

## Prerequisites

- Env **`gpuqc`** with a free CUDA device. It carries **xtb** and **gpu4pyscf with
  solvation** (IEF-PCM via `mf.PCM()`, SMD via `mf.SMD()` against gpu4pyscf's `solvent_db`).
- Run from the repo root.

## Launch (recommended: gas once, then sweep)

```bash
conda activate gpuqc
export SNAR_QC_BACKEND=gpu4pyscf
export SNAR_QC_REQUIRE_GPU=1     # error out instead of silently falling back to Psi4

# --- 1. Gas backbone once (writes the reusable gas_thermo.json + *_opt.xyz cache) ---
python scripts/run_poc.py \
  --substrates data/external/lu74_solv_slice.csv \
  --outdir     data/processed/gpu_dmso_gas \
  --n-procs 1 --mem 2            # ignored by the GPU backend; harmless

# --- 2. Sweep DMSO under both models off that one gas run (~1-2 min/substrate each) ---
python scripts/sweep_solvent.py --gas-run data/processed/gpu_dmso_gas \
  --solvent DMSO --solvent-model iefpcm --outdir data/processed/gpu_dmso_iefpcm
python scripts/sweep_solvent.py --gas-run data/processed/gpu_dmso_gas \
  --solvent DMSO --solvent-model smd    --outdir data/processed/gpu_dmso_smd

# Resume after an interruption: add --retry to any command, same --outdir.
```

`gpu_dmso_gas` is itself a valid gas run (it is the `lu74_solv_slice` analogue of
`gpu_stage_e`), so validate it too if you want the gas baseline on the 18-substrate cohort.
The existing `gpu_stage_e` run predates geometry persistence and **cannot** be swept —
sweeping needs the cache, so the gas pass above is required.

**Full-run fallback** (no separate cache step; recomputes the gas backbone per model):

```bash
python scripts/run_poc.py --substrates data/external/lu74_solv_slice.csv \
  --outdir data/processed/gpu_dmso_iefpcm --solvent DMSO --solvent-model iefpcm \
  --n-procs 1 --mem 2
```

**Faster first pass:** swap `lu74_solv_slice.csv` → `lu74_poc_slice.csv` (10 substrates, no
Br) to overlap the gas-phase POC slice; the 18-substrate run is the one that compares
cleanly to `cpu_dmso`.

**Optional H3 probe (arylators):** the CPU DMSO run's `a2`
(`O=[N+]([O-])c1ccc(Cl)s1`) died three times on a PCMSolver "S matrix not positive-definite"
cavity. gpu4pyscf PCM uses no PCMSolver, so this should not recur — test it:

```bash
python scripts/run_poc.py --substrates data/external/qc_test_5ring_arylators.csv \
  --outdir data/processed/gpu_dmso_arylators_iefpcm \
  --solvent DMSO --solvent-model iefpcm --n-procs 1 --mem 2
```

## Watch (optional)

- Progress: `data/processed/<outdir>/<lu_id>/result.json` per substrate; `summary.json`
  rolls up `N/… reached a confirmed saddle`. Each sidecar now records `solvent` **and**
  `solvent_model` for provenance.
- Health: `nvidia-smi` (~1 GB, one job); a clean run never nears the 4 GB ceiling.

## Validate (after each batch) — separate asset dirs

```bash
python scripts/validate_poc.py --slice data/external/lu74_solv_slice.csv \
  --run data/processed/gpu_dmso_iefpcm --outdir notes/assets/gpu_dmso_iefpcm
python scripts/validate_poc.py --slice data/external/lu74_solv_slice.csv \
  --run data/processed/gpu_dmso_smd    --outdir notes/assets/gpu_dmso_smd
```

Do **not** overwrite `notes/assets/cpu_dmso/`. Drop a `README.md` in each new asset dir
(what / when / method / stack / comparability), as the other folders carry.

## Pass criteria

- **IEF-PCM engine parity** vs `notes/assets/cpu_dmso/`: within-leaving-group correlations
  hold — CPU was **Br r≈0.998, F r≈0.891**, Cl moderate, with an LG-dependent offset
  (Br +5.3 / Cl +1.5 / F +13.3 kcal/mol). GPU IEF-PCM should land in that band. Drift
  beyond noise ⇒ check the dielectric (`_SOLVENT_EPS["DMSO"]=46.826`) and PCM variant.
- **H1 sidestep (`lu_27`, `N#Cc1ccnc(Cl)c1`):** CPU optking failed this nitrile TS (150-iter
  linear-bend non-convergence). The GPU gas Stage E **already completed lu_27** via
  geomeTRIC (ΔG‡ 29.53), so the DMSO run (gas TS opt = geomeTRIC) should complete it too —
  confirm it does.
- **H3 sidestep (`a2`, optional arylator run):** completes instead of the PCMSolver cavity
  death.
- **SMD:** no CPU counterpart; report the SMD−IEFPCM offset per leaving group and whether
  SMD narrows the F over-penalisation.

## On success

Record a dated findings note (`notes/YYYY-MM-DD_gpu_dmso_*`) with both correlation tables
and the H1/H3 outcomes; add the two new cells to the engine × solvent provenance matrix in
`notes/assets/gpu_stage_e/README.md` (Psi4/DMSO ✓, gpu4pyscf/DMSO ✓×2). Then consider
archiving the solvation-revalidation plan.
