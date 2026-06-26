# `gpu_gas/` — GPU/gpu4pyscf gas-phase ΔG‡ (18-substrate cohort)

**What:** gas-phase ΔG‡ for the **18-substrate** `lu74_solv_slice` (Br+Cl+F) via the
**gpu4pyscf** backend. This is the reusable **gas backbone** the DMSO campaign swept solvents
off of (`data/processed/gpu_dmso_gas`); it is also a valid gas validation run in its own
right — the 18-substrate analogue of the 10-substrate `../gpu_stage_e/`.

- `poc_validation_join.csv` / `poc_validation_stats.json` / `poc_validation_scatter.png`.

**When:** QC run + validation 2026-06-26 (`data/processed/gpu_dmso_gas`). Write-up:
`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`.

**Method:** B3LYP-D3BJ / def2-SVP, **gas phase**, methylamine, concerted coordinate,
qRRHO (100 cm⁻¹, 298.15 K), separated-reactants reference. xTB-GFN2 scan → gas DFT scan SPs →
gas TS opt+freq (geomeTRIC, analytic Hessian) → gas ArX/amine refs → ΔG‡(qh). Persists the
`gas_thermo.json` + `*_opt.xyz` cache that `sweep_solvent.py` reuses.

**Stack:** gpu4pyscf (CUDA wheels) + pyscf + geomeTRIC, xtb 6.7.1, env `gpuqc`, RTX 3050 Ti.

**Source:** `data/processed/gpu_dmso_gas`, slice `data/external/lu74_solv_slice.csv`.

**Headline:** 18/18 completed. Pooled ρ=0.610, r=0.430, MAE 14.55. Per-LG offsets Br +10.1 /
Cl +11.5 / F +24.2 — the documented gas-phase fluoride over-penalisation, ~2× the Br/Cl
cluster. Solvation roughly halves these (see `../gpu_dmso_iefpcm/`, `../gpu_dmso_smd/`).

**Comparability:** engine-comparable to `../cpu_gas/` / `../gpu_stage_e/` for the overlapping
substrates (gas parity <0.2 kcal/mol). It is the gas reference for the solvated GPU folders
(same engine & cohort).
