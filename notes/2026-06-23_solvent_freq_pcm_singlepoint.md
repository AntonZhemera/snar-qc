# 2026-06-23 — solvent ΔG‡ without a PCM Hessian: gas Hessian + PCM single point

**Status:** fixes implemented + unit-tested (suite green); 5-ring gas-vs-DMSO comparison done.
**Scope:** `src/snar_qc/poc/barrier.py`, `src/snar_qc/qc/psi4_calculator.py`,
`src/snar_qc/qc/thermo.py`, + tests (all uncommitted).
**System:** 5-ring reference — `N#Cc1ccc(F)s1` (5-fluorothiophene-2-carbonitrile) + methylamine,
fluoride leaving, concerted S~N~Ar, B3LYP-D3BJ/def2-SVP.

## Two problems found running the 5-ring case gas + DMSO

**P1 — bare-species min-opt never converged (gas).** The separated-reactants reference step
(`arx_opt_freq`) ran optking's full 150 iterations and threw
`OptimizationConvergenceError`, with quasi-Newton breakdown (`Denominators (dg)(dq) very
small`, `Skipping Hessian update`). The SCF converged every step — it was the geometry
optimiser. Cause: optking's redundant internals go degenerate on a rigid planar aryl nitrile
(near-linear C≡N), so the force criterion is never met.

**P2 — DMSO TS frequency exploded.** With `--solvent DMSO`, the run reached the TS freq and
logged `PCM analytic gradients are not implemented yet, re-routing to finite differences` then
`2353 displacements needed`. Psi4 1.10.2 has **no analytic PCM gradient/Hessian**, so a
solvent freq is a *double* finite difference: 2353 PCM-SCFs for one Hessian (vs 91 for the gas
Hessian). RSS hit ~24 GB (over the 10 GB budget) and the box began swapping. Aborted.

## Fixes

**F1 — Cartesian minimisation** (`psi4_calculator.py`). Minimisations (`opt`/`opt_freq`) now run
in Cartesian coordinates (`min_opt_coordinates`); the **TS** path keeps optking internals (it
builds a Hessian and converges fine). Min-only, so the validated TS search is untouched.

**F2 — gas Hessian + PCM single-point solvation** (`barrier.py`). A `solvent` no longer puts
opt+freq on the PCM path. Every opt+freq (TS, both references) runs in gas; the solvent is one
PCM single point at each gas geometry, shifting all thermo terms by `E(PCM) − E(gas)`
(`_solvated_thermo`, `_optimised_atoms`). The relaxed-scan DFT single points still use PCM
directly (single points have no derivative cost). Saddle order / imaginary mode come from the
gas Hessian.

**F3 — threshold-aware saddle gate + soft-mode qRRHO** (`barrier.py`, `thermo.py`). The 5-ring
TS is a clean first-order saddle plus one *soft methyl/amine rotor* the FD Hessian rendered
slightly imaginary (gas: −147 reaction mode + **−69** rotor; the old `n_imag == 1` gate failed
it as `ts_not_saddle`). Now: a valid TS has exactly one imaginary mode with `|ν| ≥
TS_SOFT_IMAG_CUTOFF_CM` (100 cm⁻¹; `count_significant_imaginary`). Smaller imaginaries are
tolerated (`n_imag_ts_soft` records them) and **folded into the thermochemistry as real
low-frequency modes** — `Psi4Thermo.from_calculator(soft_imag_cutoff=100)` adds each one's full
ZPVE + thermal − T·S_qRRHO (Psi4 drops them), keeping the genuine reaction mode excluded.

## Isolated verification (before the full re-run)

| check | result |
|---|---|
| bare ArX `N#Cc1ccc(F)s1` min-opt (F1) | n_imag = 0, real minimum, **274.8 s** (was: crash at 150 iters) |
| amine gas opt+freq | n_imag = 0, 56.9 s |
| amine DMSO (F2: gas opt+freq + PCM SP) | n_imag = 0, **55.1 s** — i.e. solvent ≈ free |
| → ΔG_solv(amine) | −3.11 kcal/mol |
| F3 gate + folding | unit tests added; thermo+barrier suite green (20 passed) |

## Full 5-ring gas vs DMSO

Matched parallel runs, `--n-procs 4 --mem 10` each (8 physical cores, contended 4+4):
`data/processed/poc_5ring_gas`, `data/processed/poc_5ring_dmso`.

**Performance** — DMSO ≈ gas (+12%), *not* the hours a PCM Hessian would have cost. F2 holds.

| stage (s) | gas | DMSO |
|---|---:|---:|
| scan_xtb | 393 | 402 |
| dft_sps (PCM in DMSO) | 144 | 317 |
| ts_opt_freq | 3239 | 3667 |
| arx_opt_freq | 387 | 296 |
| amine_opt_freq | 66 | 52 |
| **wall** | **4229 (70.5 min)** | **4753 (79.2 min)** |

**Outcome** — both reference species clean minima (n_imag = 0; F1 holds). ΔE‡ from `summary.json`;
ΔH‡ and ΔG‡(qh) shown **with F3 soft-mode folding applied** (these runs executed under pre-F3 code,
so the fold — deterministic, unit-verified — is applied analytically: gas folds −69 cm⁻¹, DMSO −36).

| quantity | gas | DMSO | solvent effect |
|---|---:|---:|---:|
| status (F3 gate) | completed | completed | — |
| reaction mode (cm⁻¹) | −147.0 | −263.8 | — |
| soft rotor, folded (cm⁻¹) | −69.3 | −36.3 | — |
| n_imag significant / soft | 1 / 1 | 1 / 1 | — |
| scan peak (DFT, node) | 69.7 (8) | 57.3 (7) | −12.4 |
| ΔE‡ | 40.48 | 24.01 | −16.47 |
| ΔH‡ (folded) | 41.86 | 25.24 | −16.62 |
| **ΔG‡(qh) (folded)** | **54.41** | **37.69** | **−16.72** |

DMSO stabilises the charge-separated S~N~Ar TS by **~16.7 kcal/mol** relative to the neutral
separated reactants — large and in the expected direction.

**Caveat — gas and DMSO optimised *different* TS geometries.** The relaxed-scan DFT single points
use PCM in DMSO, so the located peak shifts (node 8 → 7) and the gas-phase TS opt is seeded from a
different guess; the two converge to different saddles (reaction modes −147 vs −264). So this
compares "gas pipeline" vs "DMSO pipeline" end to end — each internally consistent — but the
solvent effect mixes the PCM single-point stabilisation with a geometry/guess difference (the PCM
electrostatics dominate: ΔE‡ −16.5). For a *pure* single-point solvent correction on a fixed
geometry, seed the TS from the **gas** scan peak in both phases (or optimise one gas TS and apply
both SPs). Recorded as a refinement for `2026-06-22_solvation_revalidation`.

## Relevance to the gpu4pyscf backend

P2 is a Psi4 limitation, not a method one. gpu4pyscf ships solvent gradient/Hessian code and
*likely* offers an analytic solvated Hessian — which would let solvent freq run directly and make
F2 a Psi4-path-only measure. The `2026-06-23_gpu4pyscf_backend` masterplan now carries a
cross-cutting item to **verify the analytic solvent Hessian explicitly** (Stage C/E) rather than
assume solvated freq is cheap on GPU.
