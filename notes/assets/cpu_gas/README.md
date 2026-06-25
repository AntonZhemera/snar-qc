# `cpu_gas/` — CPU/Psi4 gas-phase ΔG‡ POC baseline

**What:** the canonical **gas-phase** validation of the first-principles ΔG‡ engine on the
10-substrate Lu POC slice. Previously this baseline lived only as prose in
`notes/2026-06-21_poc_deltag_validation.md`; regenerated here as on-disk artifacts so the
gas-vs-gas backend comparison has a solid file to diff against.

- `poc_validation_join.csv` — experimental vs computed ΔG‡, per substrate.
- `poc_validation_stats.json` — Pearson / Spearman / MAE / offset-corrected MAE.
- `poc_validation_scatter.png` — computed vs experimental scatter.

**When:** QC run 2026-06-21 (`data/processed/poc_run`); assets regenerated 2026-06-25.

**Method (model chemistry):** B3LYP-D3BJ / def2-SVP, **gas phase**, methylamine model
nucleophile, concerted antisymmetric coordinate d(C–Nu) − d(C–LG). Pipeline: xTB-GFN2
relaxed scan → Psi4 DFT scan single points + **Psi4 Mayer** peak validation → optking
`OPT_TYPE=TS` saddle (**finite-difference** Hessian) + freq → qRRHO (100 cm⁻¹, 298.15 K).
Separated-reactants reference; bimolecular standard-state correction not applied (a
ranking-invariant constant).

**Stack:** Psi4 1.10.2 + optking, xtb 6.7.1, conda env `snar-qc` (CPU).

**Source:** `data/processed/poc_run`, slice `data/external/lu74_poc_slice.csv`
(10 substrates: 7 Cl + 3 F, exp range 15.3–22.85 kcal/mol).

**Headline:** within-leaving-group ranking is excellent; the pooled correlation is weak
because the gas-phase offset is leaving-group-dependent (F over-penalised ~15 kcal/mol).

| set | n | Spearman ρ | Pearson r | mean offset | offset-corr MAE |
|:--|:--:|--:|--:|--:|--:|
| all | 10 | 0.66 | 0.52 | +15.9 | 6.24 |
| Cl  | 7  | **0.96** | **0.98** | +11.5 | **0.88** |
| F   | 3  | **1.00** | 0.90 | +26.3 | 0.96 |

**Comparability:** directly comparable to `../gpu_stage_e/` (same model chemistry, both
gas phase). **Not** comparable to `../cpu_dmso/` (different solvent and cohort).
See `../gpu_stage_e/README.md` for the engine × solvent provenance matrix.
