# 2026-06-24 — Fresh DMSO ΔG‡ recalculation: lu74 slice + 5-ring arylators

**Status:** complete. Two batches recomputed from scratch in DMSO with the committed
gas-Hessian + PCM-single-point pipeline (`b8a394f`/`32ae266`/`a072fa4`).
**Method:** B3LYP-D3BJ/def2-SVP, methylamine nucleophile, concerted coordinate, DMSO via
PCM single points on gas-phase geometries (no PCM Hessian). xTB relaxed scan → DFT scan SPs
(PCM) → gas TS opt+freq → gas ArX/amine references → ΔG‡(qh) with soft-mode folding.
**Inputs:** `data/external/lu74_solv_slice.csv` (18), `data/external/qc_test_5ring_arylators.csv` (7).
**Outputs:** `data/processed/lu74_solv_dmso/`, `data/processed/qc_5ring_arylators_dmso/`
(per-substrate `result.json` sidecars + audit `.out`). Validation in `notes/assets/`.

## Yield

| Batch | valid ΔG‡ | spurious | failed | total |
|---|---:|---:|---:|---:|
| lu74 slice | 16 | 1 (lu_65) | 1 (lu_27) | 18 |
| 5-ring arylators | 4 | 0 | 3 (a2, a4, a5) | 7 |

Mean wall ≈ 4.0 h/substrate (min 1.3 h, max 11.6 h for the pyrrolidinyl pyrimidine lu_48),
running 4 workers × 4 threads on an 8-physical-core / 31 GB box; the run is machine-bound
(two finite-difference Hessians per substrate dominate). The PCM single-point path held — DMSO
cost ≈ gas, no PCM-Hessian blow-up.

## lu74 ΔG‡(qh) — chemically coherent activation ladder

NO₂-pyridine (lu_1/17 ≈ 18.1–18.5) < CN-pyridine (lu_3/22 ≈ 21.3–21.9) < CF₃-pyridine /
NO₂-benzene (lu_31/8 ≈ 24.0–24.2) < bare 2-Br-pyridine (lu_11 ≈ 26.4) < EDG Me/alkoxy
(lu_12/57/14 ≈ 27.9–29.7) < F-leaving fluoropyridines (lu_66/67/69/72 ≈ 29.6–34.3) <
unactivated F-leaving chlorobenzene (lu_74 ≈ 37.6). Ordering tracks activation strength and
leaving-group ability. Full per-substrate values in `notes/assets/poc_validation_join.csv`.

## Validation vs the empirical descriptor column (`dg_kcal`)

| set | Pearson r | Spearman | offset-corr MAE |
|---|---:|---:|---:|
| all completed (n=17, incl. lu_65) | 0.151 | 0.375 | 4.33 |
| **excluding lu_65 (n=16)** | **0.703** | **0.650** | **3.38** |

Per leaving group (ex-lu_65): **Br r=0.998**, **F r=0.891**, Cl moderate. The method
rank-orders excellently *within* a leaving group but carries a large **LG-dependent offset**
(Br +5.3, Cl +1.5, F +13.3 kcal/mol) — F barriers run systematically ~13 kcal/mol high,
consistent with the concerted coordinate over-penalising C–F cleavage. Pooled correlation is
therefore weak even on clean data; per-LG is the meaningful view.

## Failure modes (3 distinct; all need a method fix, none fixable by retry)

**H1 — TS optimiser 150-iteration non-convergence on specific heteroaromatic-nitrile
saddles.** `lu_27` (N#Cc1ccnc(Cl)c1), `a4` (N#Cc1ccc(F)s1), `a5` (N#Cc1nnc(Cl)s1) all died
with `OptimizationConvergenceError: Could not converge geometry optimization in 150
iterations` (optking `AlgError: Linear bends detected` on the near-linear C≡N). **Not** all
nitriles — `a3` (N#Cc1csc(Br)n1), `a6` (N#Cc1coc(Br)n1), and lu74 `lu_22`/`lu_3` converged
fine. The F1 Cartesian-coordinate fix covers minimisations only; the TS path keeps optking
internals (needed for the saddle Hessian), so the linear-bend degeneracy still bites on these
geometries. Fix candidate: hybrid/Cartesian TS coordinates or a better TS seed for
nitrile-bearing scaffolds. **Notable:** `a4` is the designated 5-ring reference molecule —
it converged in the 2026-06-23 pre-commit run (ΔG‡ 37.69) but fails under the committed
pipeline at 4-way contention, i.e. TS convergence here is seed/coordinate-sensitive.

**H2 — `lu_65` spurious TS the `n_imag==1` gate cannot catch.** `COc1ccc(Cl)nc1` "completed"
with ΔG‡(qh)=4.05 and **ΔE‡ = −9.02 kcal/mol** (TS electronically *below* separated
reactants). References are clean minima; the DFT scan peak is +25.1 kcal/mol. The optimiser
drifted off the reaction coordinate to a conformational saddle in a stabilised pre-complex —
a valid first-order saddle (−407 cm⁻¹) but **not the bond-exchange TS**. The single-imaginary
gate passes it because it never checks the imaginary mode's displacement vector. **Exclude
from analysis.** Fix candidate: validate the reaction mode (displacement along C–Nu/C–LG) or
a short IRC before accepting a TS. The bromo analog `lu_14` is sane (29.73).

**H3 — `a2` deterministic C-level process termination on the ArX PCM single point.**
`O=[N+]([O-])c1ccc(Cl)s1` died **three times** (original worker, two recovery runs) at the
identical point: TS + ts_pcm complete, ArX opt+freq completes, then the process exits **rc=0
with no `result.json`, no Python exception, no amine step**. `arx_pcm.out` truncates mid-init
at JKFIT basis loading — death is *inside* the ArX PCM single-point setup. rc=0 with no
exception ⇒ a C-level exit (not OOM, which is SIGKILL/137), accompanied by persistent
PCMSolver `S matrix is not positive-definite` cavity warnings. The TS PCM SP on a slightly
different geometry survived; the ArX cavity is fatal. The bromo analog `a1` completed (21.01).
Fix candidate: set the PCMSolver cavity `Area` (finer/coarser finite elements, as the warning
suggests) or an alternative continuum model. **Not recoverable by retry.**

## Operational note — memory

At 4 workers × 4 threads × 4 GB on 31 GB RAM, a worker (`w1`) was OOM-killed during the
simultaneous-Hessian peak (swap climbed to ~17 GB; abrupt log stop, no traceback), orphaning
two arylators (a2, a6). Gas Hessians cap per-worker RSS at ~6–7 GB, so no single process
runs away, but 4 concurrent peaks can exhaust RAM+swap. Recommendation for future DMSO
campaigns on this box: **3 workers, or ≤3 GB budget with more headroom**, or stagger starts.
a6 was recovered (18.89); a2 is the deterministic H3 failure.

## Recovered ΔG‡(qh) — 5-ring arylators (4 of 7)

| id | SMILES | LG | ΔG‡(qh) | ΔE‡ |
|---|---|---|---:|---:|
| a1 | O=[N+]([O-])c1ccc(Br)s1 | Br | 21.01 | 7.31 |
| a3 | N#Cc1csc(Br)n1 | Br | 21.36 | 7.56 |
| a6 | N#Cc1coc(Br)n1 | Br | 18.89 | 5.13 |
| a7 | CC(=O)c1cnc(Cl)s1 | Cl | 18.50 | 4.61 |

## Follow-ups

- TS reaction-mode/IRC validation gate (kills H2-type false positives).
- Nitrile TS coordinate handling (addresses H1; would recover lu_27, a4, a5).
- PCM cavity `Area` control (addresses H3; would recover a2).
- These map onto the `2026-06-23_gpu4pyscf_backend` plan: an analytic solvated Hessian +
  different optimiser/PCM stack on GPU may sidestep H1 and H3 outright.
