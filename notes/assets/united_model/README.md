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
  tag (**internal / per-substrate values; not for the public tree** — see the root note).

**When:** 2026-06-28. Write-up: `notes/2026-06-28_lu74_full_deltag_analysis.md`.

**Model.** Two nested calibrations are fit (computed-on-experimental, `comp = slope·exp + γ`):

- **United (shared slope, per-LG intercept):** `comp = β·exp + γ(LG)`. One slope across all
  leaving groups; the per-LG intercept `γ(LG)` is the individual offset correction. **This is
  the deliverable** — the calibrated ranker a consumer applies across F/Cl/Br.
- **Per-LG slope (full):** `comp = β(LG)·exp + γ(LG)` — the three independent fits stacked,
  used only as the comparison arm.

**Slope choice (the "F slope vs F offset" question).** A partial F-test compares the two: on
the SMD 69-substrate cohort it is **not significant** (F≈0.07, p≈0.93), and the per-LG-slope
model's *adjusted* R² is lower. So fluoride's distinctness is an **offset**, not a slope — a
**shared slope with per-LG intercepts** is the right, non-overfit calibration. (The solvated
per-LG slopes cluster tightly: SMD F 1.38 / Cl 1.56 / Br 1.50.)

**Headline (SMD, 69/74).** Shared slope β≈1.53 (computed over-disperses ~1.5× vs experiment);
per-LG mean offsets F +9.0 / Cl +5.6 / Br +5.1. After per-LG calibration the pooled ranking is
**ρ≈0.85**, and the calibrated-prediction MAE is **≈0.94 kcal/mol** (vs 1.88 for a slope-1
offset-only correction — the shared slope removes the over-dispersion).

**Provenance.** Every row carries `source_model`. After the 2026-06-28 OOM recovery SMD covers
the same 69 substrates as IEF-PCM, so the headline model is **pure SMD (0 backfilled rows)**;
the IEF-PCM-fallback path is retained for hosts/substrates where SMD is unavailable.

**Reproduce:**
```bash
python scripts/fit_united_model.py \
  --primary  notes/assets/gpu_dmso_smd/poc_validation_join.csv     --primary-model smd \
  --fallback notes/assets/gpu_dmso_iefpcm/poc_validation_join.csv  --fallback-model iefpcm \
  --outdir   notes/assets/united_model
```
