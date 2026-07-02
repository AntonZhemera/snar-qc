# `united_model/` — united three-leaving-group ΔG‡ calibration (full Lu_74 cohort)

**What:** the single **calibrated ΔG‡ model across F / Cl / Br**, folding each leaving group's
individual offset into one regression. Built by `scripts/fit_united_model.py` from the
per-model validation joins; the headline join is **SMD primary, IEF-PCM fallback** per missing
substrate (the standard workflow). The companion to the per-model `../gpu_dmso_*/` folders.

- `united_model_stats.json` — shared-slope coefficients, calibrated-prediction metrics, the
  per-LG-slope comparison, and the partial F-test that justifies the slope choice.
- `united_model_scatter.png` — left: raw computed-vs-experimental with the three parallel
  shared-slope lines; right: calibrated prediction (per-LG offset removed) vs experimental.
- `united_model_join.csv` — the blended per-substrate frame with a `source_model` provenance
  tag. The Lu_74 set is this repo's **published test/calibration set** (Lu et al.'s barriers are
  literature; the computed values are the repo's contribution).

**When:** 2026-06-28; extended to the full **74/74** cohort 2026-06-29 (hard-failure recovery).
Write-up: `notes/2026-06-28_lu74_full_deltag_analysis.md`.

**Model.** Two nested calibrations are fit (computed-on-experimental, `comp = slope·exp + γ`):

- **United (shared slope, per-LG intercept):** `comp = β·exp + γ(LG)`. One slope across all
  leaving groups; the per-LG intercept `γ(LG)` is the individual offset correction. **This is
  the deliverable** — the calibrated ranker a consumer applies across F/Cl/Br.
- **Per-LG slope (full):** `comp = β(LG)·exp + γ(LG)` — the three independent fits stacked,
  used only as the comparison arm.

**Slope choice (the "F slope vs F offset" question).** A partial F-test compares the two: on
the full 74-substrate cohort it is **not significant** (F≈0.05, p≈0.95), and the per-LG-slope
model's *adjusted* R² is lower (0.731 vs 0.738). So fluoride's distinctness is an **offset**, not
a slope — a **shared slope with per-LG intercepts** is the right, non-overfit calibration. (The
solvated per-LG slopes cluster tightly: SMD F 1.38 / Cl 1.53 / Br 1.55.)

**Headline (SMD, 74/74).** Shared slope β≈1.53 (computed over-disperses ~1.5× vs experiment);
per-LG mean offsets F +9.0 / Cl +5.5 / Br +5.3. After per-LG calibration the pooled ranking is
**ρ≈0.86**, and the calibrated-prediction MAE is **≈0.94 kcal/mol** (R²≈0.75). Adding the 5
recovered hard substrates left the calibration essentially unchanged (69-row MAE was also 0.94),
confirming completeness without a correctness shift.

**Provenance.** Every row carries `source_model`: **smd 69 + cpu_geom_smd 5**. The 5 are the
large arylators that exceeded the 4 GB GPU at the analytic Hessian; recovered on **CPU/Psi4**
(gas backbone) and finished with a **GPU SMD single point on the CPU geometry** (`cpu_geom_smd`),
so all 74 share the **same SMD solvent model**. The residual cross-engine difference is the
*geometry* engine only (CPU optking vs GPU geomeTRIC); a matched spot-check (lu_23) put that at
**~1.3 kcal/mol** — comparable to the model's own MAE, and absorbed without degrading the fit.
The IEF-PCM-fallback path is retained in `fit_united_model.py` for hosts where SMD is
unavailable.

**Reproduce:**
```bash
python scripts/fit_united_model.py \
  --primary  notes/assets/gpu_dmso_smd/poc_validation_join.csv     --primary-model smd \
  --fallback notes/assets/cpu_geom_smd/poc_validation_join.csv     --fallback-model cpu_geom_smd \
  --outdir   notes/assets/united_model
```
