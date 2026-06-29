# `cpu_geom_smd/` — the 5 hard-failure substrates (hybrid CPU-geometry / GPU-SMD)

**What:** the validation join + stats for the **5 large arylators** that exceeded the 4 GB GPU
at the analytic Hessian during the Lu_74 campaign and could not be optimised on `gpu4pyscf`.
They are recovered here on a **hybrid recipe** so they enter the united model on the **same SMD
solvent model** as the other 69.

**Recipe (per substrate).**
1. **Gas backbone on CPU/Psi4** (`run_poc.py … --solvent DMSO --solvent-model iefpcm`,
   env `snar-qc`) — CPU has no VRAM limit, so the TS-opt + analytic-Hessian that OOM'd on the
   GPU complete here. Run dir: `data/processed/cpu_hard_iefpcm/` (gitignored).
2. **GPU SMD single point on the CPU geometry** (`sweep_solvent.py … --solvent-model smd`,
   env `gpuqc`) — a single SCF (no Hessian) fits in 4 GB easily. Run dir:
   `data/processed/cpu_geom_smd/` (gitignored). This swaps the solvent model from the CPU run's
   IEF-PCM to **SMD**, matching the primary.

**The 5 (DMSO SMD ΔG‡, quasi-harmonic; all clean first-order saddles `n_imag_ts=1`):**

| lu_id | LG | SMILES | ΔG‡(qh) kcal/mol |
|---|---|---|---|
| 20 | Cl | `FC(F)(F)c1cc(Cl)c2ccccc2n1`     | 20.35 |
| 21 | Cl | `FC(F)(F)c1ccc2nccc(Cl)c2c1`     | 22.62 |
| 36 | Cl | `Clc1cc(N2CCOCC2)ncn1`           | 24.12 |
| 53 | Cl | `COc1cc2nccc(Cl)c2cc1OC`         | 26.52 |
| 10 | Br | `FC(F)(F)c1cc(Br)cc(C(F)(F)F)c1` | 28.58 |

**Provenance caveat (`source_model = cpu_geom_smd`).** These 5 share the SMD solvent model with
the primary but keep a **CPU-optimised geometry** (Psi4 optking) rather than the GPU geometry
(geomeTRIC) of the other 69. A matched spot-check — lu_23, a 25-atom aryl chloride run on **both**
engines with SMD held fixed — put the pure geometry-engine shift at **CPU-geom-SMD 22.31 vs
GPU-geom-SMD 20.97 = ~1.3 kcal/mol**, i.e. on the order of the model's own ~0.94 kcal/mol MAE.
The refit absorbed the 5 with no MAE penalty (it improved marginally, 0.94→0.937), so the shift
is within method noise. Spot-check dirs: `data/processed/cpu_spotcheck/` (CPU gas) +
`data/processed/cpu_spotcheck_smd/` (GPU-SMD SP), gitignored.

**When:** 2026-06-29. Write-up: `notes/2026-06-28_lu74_full_deltag_analysis.md` (hard-failure
recovery section). Feeds `../united_model/` as the fallback that lifts the cohort to **74/74**.

**Files:** `poc_validation_join.csv`, `poc_validation_stats.json`,
`poc_validation_scatter.png`, `poc_validation_per_lg.png` — part of the Lu_74 published
test/calibration set.
