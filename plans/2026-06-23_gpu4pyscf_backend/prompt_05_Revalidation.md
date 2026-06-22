# Stage E — POC revalidation on the GPU backend

**Goal:** confirm the gpu4pyscf backend reproduces the POC validation result, so it can be
used in the descriptor-labelling campaign in place of (or alongside) Psi4.

Depends on Stage D.

**Approach**
- Re-run the 10-substrate Lu POC slice (`data/external/lu74_poc_slice.csv`) through the
  GPU backend (`SNAR_QC_BACKEND=gpu4pyscf`).
- Recompute the correlation against experiment and compare to
  `notes/2026-06-21_poc_deltag_validation.md`: within-leaving-group Spearman/Pearson must
  hold (Cl ρ≈0.96, R²≈0.95; F ρ=1.0). Any drift beyond noise → investigate (likely the
  thermochem convention, Stage C).

**PCM/SMD — decide here, separately.** gpu4pyscf offers SMD, which the Psi4 path lacked
(it used IEFPCM). Do **not** fold an SMD switch into this backend swap; route it through the
`2026-06-22_solvation_revalidation` plan so the model change is validated on its own.

**Tests**
- Re-run the **full suite** one final time with the complete backend in place.
- Record the GPU-backend correlation table in a dated findings note.

**Done when:** GPU-backend POC correlation matches the published POC within noise, full suite
green, PCM/SMD decision recorded, uncommitted.
