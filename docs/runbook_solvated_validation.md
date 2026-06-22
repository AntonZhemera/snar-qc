# Runbook — solvated cross-leaving-group ΔG‡ re-validation

Stable reference for running the PCM-solvated ΔG‡ campaign that re-validates the engine
across leaving groups (F / Cl / Br). Pairs with the gas-phase POC findings note
(`notes/2026-06-21_poc_deltag_validation.md`) and the plan
(`plans/2026-06-22_solvation_revalidation.md`).

## Method summary

- **Level:** B3LYP-D3BJ / def2-SVP (unchanged from the POC).
- **Solvation:** PCMSolver **IEFPCM**, **Bondi** radii, solvent **DMSO** (Lu's reaction
  solvent, per the *Chem. Sci.* 2022 ESI). Psi4 1.10.2 has **no native SMD**, so IEFPCM
  is the continuum model. The solvent is applied to *every* SCF in the chain (scan DFT
  single points, TS opt+freq, both reference opt+freqs) because all calculation types
  route through `Psi4Calculator.run_calc`.
- **Reaction coordinate:** `concerted` (the gas-phase-validated antisymmetric
  d(C–Nu) − d(C–LG) scan) by default; `addition` (forming C···Nu only) is selectable for
  the stepwise coordinate that may acquire a saddle once solvent stabilises the
  Meisenheimer charge. The coordinate used is recorded per substrate; paths are never
  auto-switched.
- **Reference / nucleophile:** unchanged from the POC — separated-reactants reference,
  neutral methylamine model nucleophile (ranking metric is the headline).

## Data

`data/external/lu74_solv_slice.csv` — 18 Lu substrates (7 Cl, 6 Br, 5 F) spanning the
published 15.3–22.9 kcal/mol range, including matched ring pairs across leaving groups
(e.g. nitro-pyridine 17 Cl ↔ 1 Br; cyano-pyridine 22 Cl ↔ 3 Br) so the
leaving-group offset can be read off directly. Built read-only from the Lu_74 source.

## One-command run (Windows workstation)

The full campaign is one launcher. PCM solvation (DMSO) is the default.

```powershell
# from the repo root, snar-qc conda env active
conda activate snar-qc
pwsh scripts/run_solvated_validation.ps1
```

```bash
# Linux / WSL / Git Bash equivalent
conda run -n snar-qc bash scripts/run_solvated_validation.sh
```

The launcher runs the resumable parallel batch (`run_poc_batch.sh`, PCM on) then the
validation (`validate_poc.py`). Outputs: per-substrate sidecars under
`data/processed/solv_run/`, and stats/scatter under `notes/assets/`.

Overrides (PowerShell params / shell env): `Solvent`/`SOLVENT` (default DMSO),
`Coordinate`/`COORDINATE` (`concerted`|`addition`), `NWorkers`/`Threads`/`MemGb`.

## Cost & resume

The finite-difference TS Hessian dominates (≈2–8 h/substrate gas phase); PCM adds
per-SCF overhead and PCM frequencies are finite-difference, so budget generously for
~18 substrates. The run is resumable: each substrate writes a terminal-status sidecar
and a re-run (the launcher passes `--retry`) skips the finished ones.

## Verifying without a full run

Fast tests (no QC): `pytest -m "not slow"`. The real solvated single point is pinned in
`tests/test_psi4_calculator.py::test_nh3_dmso_pcm_single_point_energy` (`-m slow`).
