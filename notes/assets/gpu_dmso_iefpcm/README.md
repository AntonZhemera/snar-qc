# `gpu_dmso_iefpcm/` — GPU/gpu4pyscf DMSO IEF-PCM ΔG‡ (full Lu_74 cohort)

**What:** DMSO ΔG‡ for the **full 74-substrate Lu_74 cohort** (Cl 51 / Br 14 / F 9) via the
**gpu4pyscf** backend with **IEF-PCM** implicit solvation (SP-on-gas recipe). GPU analogue of
`../cpu_dmso/`; the per-substrate **fallback** model where SMD is unavailable (e.g. a CPU host)
in the standard workflow.

- `poc_validation_join.csv` — experimental vs computed ΔG‡, per substrate.
- `poc_validation_stats.json` — Pearson / Spearman / MAE / offset-corrected MAE + per-LG.
- `poc_validation_scatter.png` — computed vs experimental scatter.
- `poc_validation_per_lg.png` — per-leaving-group calibration panels (F / Cl / Br).

**When:** full-cohort sweep 2026-06-26; full-cohort revalidation 2026-06-28. Write-up:
`notes/2026-06-28_lu74_full_deltag_analysis.md` (slice-era comparison:
`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`).

**Method:** B3LYP-D3BJ / def2-SVP, methylamine, concerted coordinate, **DMSO via IEF-PCM
single points on gas geometries** (gas Hessian; ε=46.826). Pipeline: xTB-GFN2 scan → DFT
scan SPs (PCM) → gas TS opt+freq (geomeTRIC, analytic Hessian) → gas ArX/amine refs →
ΔG‡(qh). Produced by `sweep_solvent.py` off `data/processed/gpu_dmso_gas`.

**Stack:** gpu4pyscf (CUDA wheels) + pyscf + geomeTRIC, xtb 6.7.1, env `gpuqc`, RTX 3050 Ti (4 GB).

**Source:** `data/processed/gpu_dmso_iefpcm`; input `data/external/lu74_full.csv` (committed / public — the full 74-row
literature Lu_74 set). A smaller `data/external/lu74_solv_slice.csv` slice is also committed.

**Headline:** **69/74 completed.** Pooled ρ=0.76, r=0.76, MAE 7.29 (offset-corrected 2.13).
Per-LG mean offset Br +5.98 / Cl +6.79 / **F +11.76**; within-LG R² Br 0.65 / Cl 0.69 / F 0.82.
Solvation roughly halves the gas offsets and lifts ranking from ρ 0.52 → 0.76; SMD
(`../gpu_dmso_smd/`) is better still on every metric.

**Failures (5):** lu 10, 20, 21, 36, 53 — the IEF-PCM single point reuses the gas backbone,
so it fails exactly where the **gas** run did: `cudaErrorMemoryAllocation` (4 GB GPU OOM) at
the gas opt+freq for the larger arylators. No additional IEF-PCM-specific failures.

**Comparability:** engine-comparable to `../cpu_dmso/` (same level of theory & cohort,
Psi4→gpu4pyscf); Br/Cl agree within ~0.5–0.8 kcal/mol, F diverges 1–3 (PCM cavity model).
Model-comparable to `gpu_dmso_smd/` and `gpu_gas/` (same engine & cohort, model swapped).
**Not** comparable to gas folders for absolute ΔG‡ (different solvation physics).
