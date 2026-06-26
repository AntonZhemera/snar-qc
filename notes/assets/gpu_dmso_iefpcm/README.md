# `gpu_dmso_iefpcm/` — GPU/gpu4pyscf DMSO IEF-PCM ΔG‡

**What:** DMSO ΔG‡ for the 18-substrate `lu74_solv_slice` (Br+Cl+F) via the **gpu4pyscf**
backend with **IEF-PCM** implicit solvation (SP-on-gas recipe). GPU analogue of
`../cpu_dmso/`; the engine-parity arm of the 2026-06-25 DMSO campaign.

- `poc_validation_join.csv` — experimental vs computed ΔG‡, per substrate.
- `poc_validation_stats.json` — Pearson / Spearman / MAE / offset-corrected MAE + per-LG.
- `poc_validation_scatter.png` — computed vs experimental scatter.

**When:** QC sweep + validation 2026-06-26. Write-up:
`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`.

**Method:** B3LYP-D3BJ / def2-SVP, methylamine, concerted coordinate, **DMSO via IEF-PCM
single points on gas geometries** (gas Hessian; ε=46.826). Pipeline: xTB-GFN2 scan → DFT
scan SPs (PCM) → gas TS opt+freq (geomeTRIC, analytic Hessian) → gas ArX/amine refs →
ΔG‡(qh). Produced by `sweep_solvent.py` off `data/processed/gpu_dmso_gas`.

**Stack:** gpu4pyscf (CUDA wheels) + pyscf + geomeTRIC, xtb 6.7.1, env `gpuqc`, RTX 3050 Ti.

**Source:** `data/processed/gpu_dmso_iefpcm`, slice `data/external/lu74_solv_slice.csv`.

**Headline:** **18/18 completed** (incl. `lu_27` and `lu_65`, both of which the CPU path
failed/broke). Pooled ρ=0.769, r=0.774, MAE 7.16. Per-LG offsets Br +4.8 / Cl +5.8 / F +11.8;
within-LG ρ Br 1.00 / Cl 0.96 / F 0.90.

**Comparability:** engine-comparable to `../cpu_dmso/` (same level of theory & cohort,
Psi4→gpu4pyscf); Br/Cl agree within ~0.5–0.8 kcal/mol, F diverges 1–3 (PCM cavity model).
Solvent-comparable to `gpu_dmso_smd/` and `gpu_gas/` (same engine & cohort, model swapped).
**Not** comparable to gas folders for absolute ΔG‡ (different solvation physics).
