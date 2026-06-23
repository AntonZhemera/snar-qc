# Plan — implicit solvation on the Psi4 ΔG‡ engine + cross-leaving-group re-validation

**Date:** 2026-06-22
**Status:** active
**Mode:** dev-only on Linux (write + test code, build the initial datafiles, no
production QC here). The full solvated campaign is launched manually on the Windows
workstation with one command.

## Motivation

The gas-phase ΔG‡ POC ranks **within** a leaving-group family superbly (Cl n=7:
ρ=0.96, R²=0.95) but the offset is leaving-group dependent (fluoride ~+15 kcal/mol above
the Cl trend), so the pooled correlation collapses to ρ=0.66. The cause is mechanistic:
the concerted gas-phase coordinate forces full C–LG cleavage in the TS, ruinous without
solvent to stabilise the developing halide. Implicit solvation is the hypothesis-driven
fix. See `notes/2026-06-21_poc_deltag_validation.md`.

## Decisions taken at kickoff

- **Solvent = DMSO.** Confirmed from Lu, Paci & Leitch 2022 (DOI 10.1039/d2sc04041g) ESI:
  *"DMSO was used as the reaction solvent."* (Acetonitrile in the ESI is only the LC
  mobile phase.)
- **PCM, not SMD.** Psi4 1.10.2 has **no native SMD**; its implicit-solvation route is
  PCMSolver (IEFPCM/CPCM). Use **IEFPCM with Bondi radii** in DMSO. SMD is the
  predict-snar/Gaussian default but is unavailable here; document the divergence.
- **Coordinate stays explicit.** Add an `addition`-only coordinate alongside the validated
  `concerted` one as a selectable option; record which was used per substrate. Do **not**
  silently auto-switch or run both (cost) — the chemist picks per family from the runs.

## Steps

### Step 1 — wire PCM into `Psi4Calculator` (TDD) ✅ this session
- Add `solvent` / `pcm_solver` / `pcm_radii` options (gas phase when `solvent is None`).
- All calc types already route through `run_calc`, so PCM is wired once there
  (single point / opt / freq / opt_freq / ts / ts_freq all inherit it).
- Tests: fast monkeypatched test that PCM options are set iff `solvent` given; a slow
  real solvated single-point energy pinned (NH3-in-DMSO B3LYP-D3BJ/def2-SVP).

### Step 2 — coordinate option in the barrier driver ✅ this session
- `compute_barrier(..., coordinate="concerted"|"addition")`; addition scans only C···Nu.
- Record `coordinate` on `BarrierResult`; expose `--coordinate` on `run_poc.py`.

### Step 3 — extended solvated-validation slice + single launcher ✅ this session
- Build `data/external/lu74_solv_slice.csv` (~18 substrates: existing 10 + 6 Br + 2 F),
  spanning the published range across all three leaving groups.
- One launcher `scripts/run_solvated_validation.{sh,ps1}`: prep → run (resumable batch)
  → validate, so the workstation step is one command. PowerShell invocation documented.

### Step 4 — anionic nucleophile (stretch, deferred)
- Optionally test methoxide vs the neutral amine once the solvated runs are in.

## Run-time watch-list + refinements (added 2026-06-23, post method-fix)

The campaign now runs the **gas-Hessian + PCM single-point** method (gas opt+freq; solvent
applied as one PCM single point per gas geometry) with the soft-mode-tolerant TS gate — see
`notes/2026-06-23_solvent_freq_pcm_singlepoint.md`. Two items to carry through the run:

- **Triage fast `stage=scan` failures as their own class.** Substrate `lu_67`
  (`Fc1cccc(Cl)n1`, F leaving) failed in ~10 s at the `scan` stage with
  `FileNotFoundError: scan.xyz` — the xTB relaxed scan emitted no `scan.xyz`. That is a
  scan/geometry-setup failure, **not** a ΔG‡ miss and not the TS-step cost issue; expect it
  on some ortho-substituted pyridines. Record such cases as scan-stage failures in the
  findings note and triage the xTB scan setup for them separately, rather than counting them
  as solvation misses.

- **Refinement — fix the geometry for a pure single-point solvent correction.** The
  relaxed-scan DFT single points use PCM, so the located scan peak shifts between phases
  (5-ring: node 8 → 7) and the gas vs DMSO TS optimisations seed from different guesses,
  converging to different saddles (reaction modes −147 vs −264 cm⁻¹). The end-to-end
  gas-vs-DMSO comparison stays internally consistent, but the reported solvent effect then
  mixes the PCM electrostatic stabilisation with a geometry/guess difference. For a clean
  single-point correction, seed the TS from the **gas** scan peak in both phases (or optimise
  one gas TS and apply both the gas and PCM single points to it), so ΔΔG‡(solv) reflects only
  `E(PCM) − E(gas)` at a fixed structure. Apply on the next re-validation pass.

## Out of scope this session
- Running the multi-hour solvated QC campaign (happens on Windows).
- Auto coordinate-selection / running both coordinates per substrate.

## Deliverables
- `src/snar_qc/qc/psi4_calculator.py` — PCM option (TDD).
- `src/snar_qc/poc/barrier.py`, `scripts/run_poc.py` — coordinate option.
- `data/external/lu74_solv_slice.csv` — extended slice (initial datafile).
- `scripts/run_solvated_validation.{sh,ps1}` — the single launcher.
- Updated `docs/` runbook + tests green; smoke path confirmed without a real QC run.
