# `gpu_dmso_smd/` — GPU/gpu4pyscf DMSO SMD ΔG‡

**What:** DMSO ΔG‡ for the 18-substrate `lu74_solv_slice` (Br+Cl+F) via the **gpu4pyscf**
backend with **SMD** solvation. A model the Psi4 1.10.2 path cannot provide, so there is
**no CPU counterpart**; characterises the SMD−IEF-PCM delta. New-capability arm of the
2026-06-25 DMSO campaign.

- `poc_validation_join.csv` / `poc_validation_stats.json` / `poc_validation_scatter.png`.

**When:** QC sweep + validation 2026-06-26. Write-up:
`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`.

**Method:** B3LYP-D3BJ / def2-SVP, methylamine, concerted coordinate, **DMSO via SMD single
points on gas geometries** (gas Hessian). Same pipeline as `../gpu_dmso_iefpcm/`, only
`--solvent-model smd`. Produced by `sweep_solvent.py` off `data/processed/gpu_dmso_gas`.

**Stack:** gpu4pyscf (CUDA wheels) + pyscf + geomeTRIC, xtb 6.7.1, env `gpuqc`, RTX 3050 Ti.

**Source:** `data/processed/gpu_dmso_smd`, slice `data/external/lu74_solv_slice.csv`.

**Headline — best single model.** Pooled ρ=**0.900**, r=0.857, MAE **5.76** (best of the
three). Per-LG offsets Br +4.1 / Cl +4.7 / F +9.1 — SMD narrows the F over-penalisation most
(F−Cl gap +4.4 vs IEF-PCM's +6.0). **17/18 completed:** `lu_48`
(`Clc1cc(N2CCCC2)ncn1`) lost to an SMD-SCF failure — the one robustness gap vs IEF-PCM (18/18).

**Comparability:** model-comparable to `gpu_dmso_iefpcm/` and `gpu_gas/` (same engine &
cohort). No CPU/Psi4 SMD baseline exists. **Not** comparable to gas folders for absolute ΔG‡.
