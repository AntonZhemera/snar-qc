# `gpu_stage_e/` — GPU/gpu4pyscf gas-phase ΔG‡ revalidation (Stage E)

**What:** the 10-substrate Lu POC slice re-run through the **gpu4pyscf backend**, gas phase,
to confirm the within-leaving-group correlations hold after porting the engine off Psi4
(plan `plans/2026-06-23_gpu4pyscf_backend/`, Stage E).

- `poc_validation_join.csv` — experimental vs computed ΔG‡, per substrate.
- `poc_validation_stats.json` — Pearson / Spearman / MAE / offset-corrected MAE.
- `poc_validation_scatter.png` — computed vs experimental scatter.

**When:** QC run + validation 2026-06-25 (`data/processed/gpu_stage_e`).

**Method (model chemistry): identical to `../cpu_gas/`** — B3LYP-D3BJ / def2-SVP, gas
phase, methylamine, concerted coordinate, qRRHO (100 cm⁻¹, 298.15 K), separated-reactants
reference. **Implementation differs** (same level of theory, different stack):

| | CPU gas (`../cpu_gas/`) | GPU gas (here) |
|---|---|---|
| QC engine | Psi4 1.10.2 | gpu4pyscf / pyscf (density fitting) |
| TS optimizer | optking `OPT_TYPE=TS` | geomeTRIC |
| TS Hessian | finite-difference | **analytic** |
| Mayer bond orders | `Psi4BondOrders` (wavefunction) | `PyscfBondOrders` (mean-field) |
| scan | xtb 6.7.1 (identical binary) | xtb 6.7.1 (identical binary) |

**Stack:** gpu4pyscf (pip CUDA wheels) + pyscf + geomeTRIC, xtb 6.7.1, conda env `gpuqc`,
RTX 3050 Ti. ~31 min/substrate.

**Source:** `data/processed/gpu_stage_e`, slice `data/external/lu74_poc_slice.csv` (same
10 substrates as `../cpu_gas/`).

| set | n | Spearman ρ | Pearson r | mean offset | offset-corr MAE |
|:--|:--:|--:|--:|--:|--:|
| all | 10 | 0.66 | 0.52 | +15.9 | 6.24 |
| Cl  | 7  | **0.96** | **0.98** | +11.5 | **0.86** |
| F   | 3  | **1.00** | 0.90 | +26.3 | 0.95 |

**Result — backend port validated.** GPU reproduces the CPU gas baseline to **< 0.2
kcal/mol per substrate** and identical correlation stats, despite the four implementation
changes above. The two LG clusters are the documented gas-phase fluoride over-penalisation,
present identically on both engines — not a GPU artefact.

## Provenance — engine × solvent matrix

These assets span two axes. Pool results **only within a cell**, or across cells where
equivalence has been demonstrated.

| | gas phase | DMSO (PCM-SP) |
|---|---|---|
| **Psi4 (CPU)** | `../cpu_gas/` | `../cpu_dmso/` |
| **gpu4pyscf (GPU)** | `gpu_stage_e/` (here) | *not possible yet — PCM is Psi4-only* |

Only **`cpu_gas` ↔ `gpu_stage_e`** equivalence is established (< 0.2 kcal/mol, above).
`cpu_dmso` differs by real solvation physics **and** a larger cohort (18, incl. Br) — do
not diff gas results against it.
