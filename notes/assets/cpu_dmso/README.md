# `cpu_dmso/` — CPU/Psi4 DMSO (PCM-SP) ΔG‡ recalculation

**What:** fresh **DMSO** ΔG‡ recalculation of the lu74 solvation slice (18) and the 5-ring
arylators (7). This is the CPU science's current frontier and the live validation
artifact set. (Previously these three files sat loose at the `notes/assets/` root, where
they shadowed the gas-phase baseline; moved into this folder 2026-06-25.)

- `poc_validation_join.csv` — experimental vs computed ΔG‡, per substrate.
- `poc_validation_stats.json` — Pearson / Spearman / MAE / offset-corrected MAE.
- `poc_validation_scatter.png` — computed vs experimental scatter.

**When:** QC run + validation 2026-06-24. Full write-up:
`notes/2026-06-24_dmso_recalc_lu74_arylators.md`.

**Method:** B3LYP-D3BJ / def2-SVP, methylamine, concerted coordinate, **DMSO via IEFPCM
single points on gas-phase geometries** (gas Hessian; no PCM Hessian). Pipeline: xTB-GFN2
scan → DFT scan SPs (**PCM**) → gas TS opt+freq → gas ArX/amine references → ΔG‡(qh) with
soft-mode folding.

**Stack:** Psi4 1.10.2 + PCMSolver (IEFPCM / Bondi radii) + optking, xtb 6.7.1, conda env
`snar-qc` (CPU).

**Source:** `data/processed/lu74_solv_dmso` (+ `data/processed/qc_5ring_arylators_dmso`),
slice `data/external/lu74_solv_slice.csv` (18 substrates: Br + Cl + F).

**Yield / headline:** 17/18 lu74 completed (lu_65 spurious TS, lu_27 failed). Pooled
correlation is weak; per-leaving-group is the meaningful view.

| set | Pearson r | Spearman ρ | offset-corr MAE |
|:--|--:|--:|--:|
| all completed (n=17, incl. lu_65) | 0.151 | 0.375 | 4.33 |
| excluding lu_65 (n=16) | **0.703** | **0.650** | 3.38 |

Per-LG (ex-lu_65): **Br r=0.998** (offset +5.3), Cl moderate (offset +1.5),
**F r=0.891** (offset +13.3). Three method-level failure modes characterised in the note
(H1 nitrile-TS non-convergence, H2 spurious-TS gate gap, H3 PCM cavity death) — none
retry-fixable.

**Comparability:** **not** comparable to the gas-phase folders (`../cpu_gas/`,
`../gpu_stage_e/`) — different solvent **and** a different, larger cohort. See
`../gpu_stage_e/README.md` for the engine × solvent provenance matrix.
