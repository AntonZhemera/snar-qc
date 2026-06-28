# Full Lu_74 ΔG‡ analysis: per-leaving-group + united calibration, and the 4 GB-GPU OOM recovery

**Date:** 2026-06-28
**Scope:** post-QC analysis of the full 74-substrate Lu_74 arylator cohort (Cl 51 / Br 14 /
F 9) computed on the gpu4pyscf backend across three model chemistries — gas, DMSO IEF-PCM,
DMSO SMD. No new optimisations: this is modelling, plotting, README renewal, and the recovery
of failed single points. Builds on the 18-substrate slice picture in
`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`; standard workflow in
`docs/sop_snar_deltag.md`.

## Summary

1. **The full cohort reproduces the slice picture and the model ordering.** Monotone
   gas → IEF-PCM → SMD on every aggregate metric; **SMD is the best single model**
   (69/74, ρ=0.82, MAE 5.95). Fluoride is a distinct, over-penalised cluster in every model.
2. **SMD's apparent robustness gap was a 4 GB-GPU artefact, not a method limitation** (H1).
   The 9 "SMD-only" failures all recovered once each single point ran in a fresh process;
   SMD now matches IEF-PCM's coverage (69/74) *and* beats it on every metric.
3. **The united F/Cl/Br calibration is a shared slope with per-LG intercepts** (H2). A partial
   F-test shows per-LG *slopes* are not justified — fluoride's distinctness is an **offset**,
   not a slope.

## Aggregate metrics (full cohort, after OOM recovery)

| Model | Completed | Spearman ρ | Pearson r | MAE | Offset-corr. MAE |
|---|---|---|---|---|---|
| GPU gas | 69/74 | 0.52 | 0.46 | 13.92 | 4.18 |
| GPU DMSO IEF-PCM | 69/74 | 0.76 | 0.76 | 7.29 | 2.13 |
| GPU DMSO SMD | **69/74** | **0.82** | **0.81** | **5.95** | **1.88** |

Per-leaving-group mean offset (computed − experimental, kcal/mol):

| Model | Br | Cl | F | F−Cl gap |
|---|---|---|---|---|
| gas | 11.17 | 12.65 | 24.46 | +11.8 |
| IEF-PCM | 5.98 | 6.79 | 11.76 | +5.0 |
| SMD | 5.07 | 5.57 | 9.01 | +3.4 |

Solvation roughly halves the gas offsets and narrows the fluoride gap; SMD narrows it most.
The fluoride offset never merges into the Br≈Cl cluster, so **per-leaving-group offset
calibration is mandatory** — a single global offset is wrong by ~3–12 kcal/mol depending on LG.

## H1 — the SMD failures were cumulative GPU OOM, not SCF non-convergence

The first full-cohort SMD sweep completed only 60/74. All 14 failures across the three models
raised the **same** error: `CUDARuntimeError: cudaErrorMemoryAllocation: out of memory` on the
4 GB laptop GPU — never an SCF/optimiser non-convergence.

- **9 "SMD-only" failures** (lu 48, 52, 58, 60, 61, 62, 63, 64, 68): gas + IEF-PCM had
  completed and the gas cache existed; only the heavier SMD single point OOM'd, ~7 s in, at
  SCF init. Re-running **each SMD single point in its own process** recovered **all 9** (peak
  ~1.1 GB, well under the 4 GB ceiling). The cause is **cumulative GPU memory**: the batch
  runs every substrate in one Python process and CuPy's memory pool is not released between
  them, so later/larger substrates OOM even when individually small (tiny chloropicolines
  lu 60/61/63 "failed" only because they came late in the batch). **Fix:** one substrate per
  process, or free the CuPy pool between substrates (see Next steps).
- **5 "hard" failures** (lu 10, 20, 21, 36, 53): OOM upstream in the **gas backbone**, at the
  TS/ArX `opt+freq` (the analytic Hessian is the memory peak) during the batch run, so no gas
  cache and no solvent single point exist. These are the genuinely large arylators (a bis-CF₃
  bromoarene, two CF₃-quinolines, a chloro-triazine-morpholine, a dimethoxy-chloroquinoline).
  A **solo GPU probe** (one process, clean card) was run on the smallest of them, lu_36: it no
  longer OOM'd but its **TS optimisation did not converge** within a 20-min cap (67+ geomeTRIC
  cycles), so it remains uncompleted. So the hard failures are **memory- *and* convergence-
  limited**, not cheaply GPU-recoverable; the cohort stays **69/74**. Recovery is deferred to a
  larger GPU (memory) with a longer/looser TS-opt budget, or the **CPU/Psi4** path (no 4 GB
  limit; IEF-PCM only) — see Next steps.

**Consequence:** SMD's earlier "least robust (60/74)" reading is retired. On adequate memory
SMD is **both** the best-ranking model **and** as complete as IEF-PCM (69/74). The current
`docs/sop_snar_deltag.md` §7 "SMD SCF non-convergence on electron-rich arylators" wording is
contradicted by the full-cohort evidence and should be corrected to "4 GB-GPU OOM; re-run one
substrate per process or use a larger card."

## H2 — the united model is a shared slope with per-LG intercepts

Per leaving group, computed-on-experimental regression `comp = slope·exp + intercept`
(`scripts/validate_poc.py`, the `by_leaving_group` block). On SMD:

| LG | n | slope | R² | within-LG ρ | mean offset |
|---|---|---|---|---|---|
| F | 9 | 1.38 | 0.83 | 0.97 | 9.01 |
| Cl | 47 | 1.56 | 0.71 | 0.84 | 5.57 |
| Br | 13 | 1.50 | 0.69 | 0.85 | 5.07 |

The solvated per-LG slopes cluster tightly (1.38–1.56); only the intercepts/offsets differ.
The united fit (`scripts/fit_united_model.py`) bears this out:

- **United (shared slope, per-LG intercept)** `comp = β·exp + γ(LG)`: β≈1.53, R²=0.75. The
  shared slope >1 means the computed barriers **over-disperse** ~1.5× relative to experiment.
- **Per-LG slope (full)**: R² gains nothing (0.746 → 0.746) and adjusted R² *drops*.
- **Partial F-test:** F≈0.07, p≈0.93 — per-LG slopes are **not** justified.

So a **shared slope with three per-LG intercepts** is the calibration: it removes the fluoride
over-penalisation as an offset without the overfit risk of per-LG slopes (fluoride n=9). After
calibration, pooled ranking is **ρ≈0.85** and the calibrated-prediction **MAE ≈0.94 kcal/mol**
(vs 1.88 for a slope-1 offset-only correction — the shared slope removes the over-dispersion).
This is the headline deliverable: `notes/assets/united_model/` and the
`comp = 1.53·exp + γ(LG)` model with per-LG offsets F +9.0 / Cl +5.6 / Br +5.1.

## Provenance

After the OOM recovery, SMD covers the same 69 substrates as IEF-PCM, so the headline united
model is **pure SMD with zero backfilled rows**; the "SMD primary, IEF-PCM fallback" precedence
is retained in `fit_united_model.py` for hosts/substrates where SMD is unavailable, and every
row of `united_model_join.csv` carries a `source_model` tag. The recovered SMD single points
are same-engine, same-recipe re-runs of the cached gas geometries — no cross-engine mixing.

## Next steps

1. **Free the CuPy memory pool between substrates in the batch runners** (`run_poc.py`,
   `sweep_solvent.py`) — `cupy.get_default_memory_pool().free_all_blocks()` (and the pinned
   pool) per substrate. This is the highest-leverage fix: it would have prevented the entire
   60/74 → 69/74 artefact and lets large cohorts run unattended on a 4 GB card.
2. **Recover the 5 hard failures.** The solo-GPU probe is exhausted (lu_36 didn't converge in
   20 min), so this needs either a **larger GPU** (memory headroom) with a longer/looser
   TS-opt budget, or the **CPU/Psi4** path (no 4 GB limit; ~20-40 min/substrate), then a DMSO
   sweep and a united-model re-fit on the enlarged cohort. CPU gives gas + IEF-PCM (no SMD on
   Psi4), so any recovered rows enter as IEF-PCM-fallback provenance. The conclusions are
   already stable at 69/74; this is completeness, not a correctness gap.
3. **Anionic nucleophile (methoxide).** The per-LG offsets *are* the systematic error of the
   neutral model amine standing in for the real anionic alkoxide; this is the deferred stretch
   from `plans/2026-06-22_solvation_revalidation.md` (Step 4) and the hypothesis-driven path to
   shrink — possibly merge — the fluoride offset. Highest scientific value.
4. **Hand the calibration downstream.** The SMD per-LG offsets + shared slope are the
   calibrated ΔG‡ ranker a reactivity consumer applies across F/Cl/Br.
5. **Guard against overfit.** With F n=9, confirm the per-LG calibration generalises
   (leave-one-out within F, or a held-out check) before treating the offsets as fixed.
