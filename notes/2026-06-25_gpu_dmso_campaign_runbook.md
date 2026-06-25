# Runbook — GPU DMSO revalidation campaign (IEF-PCM + SMD), run manually

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

Expected ~**33–36 min/substrate** (gas TS opt+freq ~24 min still dominates) ⇒ ~**10–11 h**
for the 18-substrate slice, **per model**.

## Prerequisites

- Env **`gpuqc`** with a free CUDA device. It carries **xtb** and now **gpu4pyscf with
  solvation** (IEF-PCM via `mf.PCM()`, SMD via `mf.SMD()` against gpu4pyscf's `solvent_db`).
- Run from the repo root.

## Launch

```bash
conda activate gpuqc
export SNAR_QC_BACKEND=gpu4pyscf
export SNAR_QC_REQUIRE_GPU=1     # error out instead of silently falling back to Psi4

# --- 1. IEF-PCM, 18-substrate slice (direct cpu_dmso engine-parity check) ---
python scripts/run_poc.py \
  --substrates data/external/lu74_solv_slice.csv \
  --outdir     data/processed/gpu_dmso_iefpcm \
  --solvent DMSO --solvent-model iefpcm \
  --n-procs 1 --mem 2            # ignored by the GPU backend; harmless

# --- 2. SMD, same slice (new capability; no CPU baseline) ---
python scripts/run_poc.py \
  --substrates data/external/lu74_solv_slice.csv \
  --outdir     data/processed/gpu_dmso_smd \
  --solvent DMSO --solvent-model smd \
  --n-procs 1 --mem 2

# Resume after an interruption (re-runs only substrates without a terminal sidecar):
#   add --retry to either command, same --outdir.
```

**Faster first pass:** swap `lu74_solv_slice.csv` → `lu74_poc_slice.csv` (10 substrates, no
Br) to get an IEF-PCM run in ~5 h that overlaps the gas-phase POC slice; the 18-substrate
run is the one that compares cleanly to `cpu_dmso`.

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
