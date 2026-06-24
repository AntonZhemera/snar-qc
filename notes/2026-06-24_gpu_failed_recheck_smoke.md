# 2026-06-24 — GPU gas-phase mini-smoke on the 5 prior DMSO failures

**What:** a quick gpu4pyscf sanity run on the five substrates the committed Psi4/DMSO
pipeline could not produce a valid ΔG‡ for (cohort `data/external/gpu_failed_recheck.csv`).
Goal: do these "unordinary" scaffolds work on the GPU backend at all, and where does the
prior failure actually live?

**Method:** bare substrate from SMILES (RDKit embed, seed `0xC0FFEE`, MMFF) →
`GPU4PySCFCalculator.opt_freq` (B3LYP-D3BJ/def2-SVP, gas phase). Exercises SCF + GPU-gradient
geomeTRIC minimisation + analytic Hessian on each scaffold. **Gas phase only** — gpu4pyscf
PCM solvation is not yet wired (gated to the solvation-revalidation plan).

## Result — all 5 clean (n_imag = 0, SCF + opt + analytic Hessian)

| id | substrate | prior failure | n_atoms | E (Ha) | G (Ha) | n_imag | freq range (cm⁻¹) | wall |
|---|---|---|---|---|---|---|---|---|
| lu_27 | `N#Cc1ccnc(Cl)c1` | TS 150-iter non-conv | 12 | −799.766053 | −799.720818 | 0 | 133.6 – 3232.4 | 75 s |
| lu_65 | `COc1ccc(Cl)nc1` | spurious TS | 15 | −822.038099 | −821.959740 | 0 | 74.0 – 3223.0 | 102 s |
| a2 | `O=[N+]([O-])c1ccc(Cl)s1` | PCM cavity fail | 11 | −1216.611867 | −1216.585337 | 0 | 80.5 – 3246.4 | 77 s |
| a4 | `N#Cc1ccc(F)s1` | TS 150-iter non-conv | 10 | −744.125555 | −744.099183 | 0 | 113.0 – 3241.4 | 47 s |
| a5 | `N#Cc1nnc(Cl)s1` | TS 150-iter non-conv | 8 | −1136.458596 | −1136.458473 | 0 | 95.3 – 2373.6 | 38 s |

~5.6 min total. `a4` (= the 5-ring reference) reproduces its Stage C minimum
(−744.125555 Ha) exactly. `a5` is H-free (cyano-thiadiazole), hence its top mode is the
C≡N stretch (2373.6 cm⁻¹), not an aromatic C–H stretch.

## Interpretation

The five substrates are **not** pathological for gpu4pyscf: each converges an SCF, relaxes
under geomeTRIC, and gives a clean first-order-free minimum from the analytic Hessian. So
the prior Psi4 failures ([`2026-06-24_dmso_recalc_lu74_arylators.md`](2026-06-24_dmso_recalc_lu74_arylators.md))
live in the **TS and PCM steps**, not in handling these scaffolds:

- **Modes 1 (lu_27/a4/a5 TS non-convergence) and 2 (lu_65 spurious TS)** — the GPU TS path
  (geomeTRIC `transition=True` seeded by the analytic Hessian, Stage D) is the candidate
  fix; whether it converges *these specific* saddles where optking did not is the next test.
- **Mode 3 (a2 PCM cavity "S matrix not positive-definite")** — solvent-specific; rides on
  the not-yet-wired analytic GPU solvation.

## Scope / next (deferred to dedicated sessions)

This is a **substrate-level gas-phase smoke** — it does **not** yet test whether the GPU
backend resolves the actual TS/PCM failures (that needs the reaction complex + a TS search
per substrate, and the PCM path). Those, plus the Psi4 ΔG‡ cross-check and Stage E POC
revalidation, are deferred to dedicated sessions.
