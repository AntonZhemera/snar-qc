# `gpu_gas/` — GPU/gpu4pyscf gas-phase ΔG‡ (full Lu_74 cohort)

**What:** gas-phase ΔG‡ for the **full 74-substrate Lu_74 arylator cohort** (Cl 51 / Br 14 /
F 9) via the **gpu4pyscf** backend. This is the reusable **gas backbone** the DMSO campaign
sweeps solvents off of (`data/processed/gpu_dmso_gas`); it is also a gas validation run in its
own right — the full-cohort successor to the 18-substrate `lu74_solv_slice` era.

- `poc_validation_join.csv` / `poc_validation_stats.json` / `poc_validation_scatter.png`.
- `poc_validation_per_lg.png` — per-leaving-group calibration panels (F / Cl / Br).

**When:** full-cohort QC run 2026-06-25/26; full-cohort revalidation 2026-06-28. Write-up:
`notes/2026-06-28_lu74_full_deltag_analysis.md` (slice-era comparison:
`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`).

**Method:** B3LYP-D3BJ / def2-SVP, **gas phase**, methylamine, concerted coordinate,
qRRHO (100 cm⁻¹, 298.15 K), separated-reactants reference. xTB-GFN2 scan → gas DFT scan SPs →
gas TS opt+freq (geomeTRIC, analytic Hessian) → gas ArX/amine refs → ΔG‡(qh). Persists the
`gas_thermo.json` + `*_opt.xyz` cache that `sweep_solvent.py` reuses.

**Stack:** gpu4pyscf (CUDA wheels) + pyscf + geomeTRIC, xtb 6.7.1, env `gpuqc`, RTX 3050 Ti (4 GB).

**Source:** `data/processed/gpu_dmso_gas`; input `data/external/lu74_full.csv` (internal,
gitignored). The committed, publicly reproducible subset is `data/external/lu74_solv_slice.csv`.

**Headline:** **69/74 completed.** Pooled ρ=0.52, r=0.46, MAE 13.92 (offset-corrected 4.18).
Per-LG mean offset Br +11.2 / Cl +12.7 / **F +24.5** — the documented gas-phase fluoride
over-penalisation, ~2× the Br/Cl cluster. Solvation roughly halves these (see
`../gpu_dmso_iefpcm/`, `../gpu_dmso_smd/`). Gas is the **worst-ranking** model and the per-LG
slopes are heterogeneous (F 1.75 / Cl 1.27 / Br 0.98) — it needs solvation to be usable.

**Failures (5):** lu 10, 20, 21, 36, 53 — all `cudaErrorMemoryAllocation` (4 GB GPU **out of
memory**) at the gas TS/ArX opt+freq (the analytic Hessian is the memory peak), **not** a
chemistry failure. The larger arylators (bis-CF₃ arene, CF₃-quinolines, a dimethoxy-quinoline)
exceed the 4 GB card; recover on a larger GPU or via the CPU/Psi4 fallback.

**Comparability:** engine-comparable to `../cpu_gas/` / `../gpu_stage_e/` for the overlapping
substrates (gas parity <0.2 kcal/mol). It is the gas reference for the solvated GPU folders
(same engine & cohort).
