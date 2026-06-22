# 2026-06-21 — First-principles ΔG‡ POC vs the Lu reactivity set (go/no-go)

**Status:** complete — engine validated end to end on 11/11 substrates (every one a
confirmed saddle); 10-substrate validation and the go/no-go verdict are below.

End-to-end proof of concept for computing S~N~Ar activation free energies (ΔG‡) directly
from transition-state quantum chemistry, validated against the Lu, Paci & Leitch
reactivity set (*Chem. Sci.* 2022, 13, 12681; DOI 10.1039/d2sc04041g). This closes the
ΔG‡ engine adoption: it drives the Psi4-backed engine (B3LYP-D3BJ/def2-SVP) from a
SMILES to a quasi-harmonic ΔG‡ and confronts it with published barriers.

## Question

Does a first-principles ΔG‡, computed with a **neutral model amine** in the **gas
phase** (no PCM/SMD yet), reproduce the **ordering** of published S~N~Ar barriers well
enough to justify scaling the engine to a full descriptor-labelling campaign?

## What was computed

For each substrate (a (hetero)aryl halide), the engine computes the concerted-S~N~Ar
activation free energy for the reaction with **methylamine** as a model nucleophile:

1. **Reaction complex.** Built from SMILES with RDKit: the amine is docked above the
   ipso carbon (the carbon bearing the leaving halide) along the aromatic-ring normal.
2. **Concerted relaxed scan.** An xTB (GFN2) relaxed scan drives the **antisymmetric**
   reaction coordinate d(C–Nu) − d(C–LG): the forming C···N bond contracts while the
   breaking C–LG bond extends, in one concerted scan. Psi4 B3LYP-D3BJ/def2-SVP single
   points along the scan give the DFT energy profile; Psi4 **Mayer** bond orders
   validate the peak (both the forming and breaking bonds change order across it).
3. **Transition state.** The highest validated scan maximum seeds an optking
   `OPT_TYPE=TS` saddle optimisation (with a Hessian); a frequency calculation confirms
   the saddle by **exactly one imaginary mode** and supplies the harmonic
   thermochemistry. The Grimme quasi-RRHO correction (cutoff 100 cm⁻¹, 298.15 K) gives
   the quasi-harmonic Gibbs free energy.
4. **Reference & barrier.** ΔG‡ = G(TS) − [G(aryl halide) + G(methylamine)] (see
   *Reference choice* below).

## Key methodological findings (validated on the 4-nitrofluorobenzene smoke test)

- **The reaction coordinate must be antisymmetric.** Scanning only the forming C···N
  bond drives the neutral amine into a high-energy zwitterionic adduct **monotonically
  uphill, with no gas-phase saddle**. Driving d(C–Nu) − d(C–LG) concertedly (Nu in *and*
  LG out together) traverses a genuine concerted-S~N~Ar barrier with a clean maximum.
  This is the single most important wiring decision for the gas-phase POC.
- **A real saddle is found.** On the smoke test (methylamine + 1-fluoro-4-nitrobenzene)
  the located TS has exactly one imaginary frequency (−237 cm⁻¹), corresponding to the
  coupled C–N formation / C–F cleavage motion — the engine produces a true
  transition state end to end.

## Reference choice (decided here)

The clean choice for a *unimolecular* step is a reaction-complex reference 
**(aryl halide + amine as one supermolecule)**, 
which cancels the gas-phase standard-state term. In practice the gas-phase 
pre-association complex is a **floppy, orientation-dependent
van-der-Waals minimum on a flat surface that does not converge** — the optimisation
failed even after the (expensive) TS search. The **separated-reactants** reference
[G(ArX) + G(amine)] is used instead because:

- it is **numerically robust** — a rigid aromatic and a tiny amine, both easy minima;
- for a **ranking** validation it is equivalent up to a constant: the amine term and the
  bimolecular 1 atm → 1 M standard-state correction are identical for every substrate,
  so they shift all barriers together without changing the order.

The standard-state correction is therefore **not applied** (a constant; the POC weights
ranking over magnitude). Consequently the reported ΔG‡ carries the full, roughly
constant entropy-of-association penalty and a large positive absolute offset vs the
solution-phase experiment — expected and irrelevant to the ranking question.

## Caveats (why absolute magnitude is not the metric)

- **Nucleophile mismatch.** Lu's barriers were measured with an anionic benzyl
  **alkoxide**; gas-phase/SMD QC handles anions poorly, so the POC substitutes a neutral
  **methylamine**. The two nucleophiles differ in absolute reactivity but the
  substrate-driven *ordering* (ring/substituent activation) is expected to track.
- **Gas phase, no solvation.** No PCM/SMD on the Psi4 path yet — a large, systematic
  absolute offset is expected; ranking (Spearman/Pearson) is the headline metric.
- **Single conformer / no CREST.** One docked geometry per substrate; no conformer
  search on the TS or references.

## Results — 10 Lu substrates (F & Cl leaving groups)

All 11 runs (10 validation substrates + the smoke test) located a genuine transition
state — **exactly one imaginary mode each, 100 % convergence** — with the concerted-scan
peak as the saddle seed and `geom_maxiter` raised to 150. Wall time 2.3–8.4 h/substrate
(the finite-difference TS Hessian dominates).

| lu_id | LG | exp ΔG‡ | comp ΔG‡(qh) | comp ΔE‡ | offset (ΔG‡) |
|---:|:--:|---:|---:|---:|---:|
| 17 | Cl | 15.3 | 25.3 | 11.3 | +10.0 |
| 22 | Cl | 16.9 | 28.3 | 14.4 | +11.4 |
| 66 | **F** | 17.4 | 42.6 | 28.8 | **+25.2** |
| 27 | Cl | 17.5 | 29.5 | 16.0 | +12.0 |
| 31 | Cl | 17.9 | 28.6 | 15.1 | +10.7 |
| 72 | **F** | 19.6 | 47.4 | 33.0 | **+27.8** |
| 48 | Cl | 19.7 | 30.3 | 16.7 | +10.6 |
| 57 | Cl | 21.3 | 33.2 | 19.9 | +11.9 |
| 74 | **F** | 21.8 | 47.7 | 33.9 | **+25.9** |
| 65 | Cl | 22.9 | 36.3 | 23.3 | +13.4 |

All energies kcal/mol. Correlation of computed ΔG‡(qh) against experiment:

| Set | n | Spearman ρ | Pearson r | mean offset | offset-corrected MAE |
|:--|:--:|--:|--:|--:|--:|
| **all** | 10 | 0.66 (p=0.04) | 0.52 | +15.9 | 6.24 |
| **Cl only** | 7 | **0.96** | **0.98** (R²=0.95) | +11.5 | **0.88** |
| **F only** | 3 | **1.00** | 0.90 | +26.3 | 0.96 |

Two clear signals:

1. **Within a leaving-group family the computed ordering is excellent.** Cl (n=7)
   reproduces the experimental rank almost perfectly (ρ=0.96, R²=0.95 — above the
   aspirational predict-SNAr R²=0.93) with an offset-corrected MAE of **0.88 kcal/mol**;
   the three F substrates are correctly ordered (ρ=1.0). The direct ΔG‡ does track
   substrate activation.
2. **The offset is leaving-group-dependent**, so the *pooled* correlation collapses to
   ρ=0.66. F substrates sit ~15 kcal/mol above the Cl trend (offset +26 vs +11) because
   the concerted gas-phase coordinate forces full C–F cleavage in the TS, which is very
   costly without solvent/H-bond stabilisation of the developing fluoride — whereas in
   solution F is a competent S~N~Ar leaving group (the addition step, not C–F cleavage,
   is rate-limiting). This is the expected gas-phase / no-PCM artefact, concentrated on
   fluoride.

_Validation slice: `data/external/lu74_poc_slice.csv` (10 substrates spanning the
published 15.3–22.9 kcal/mol range, 7 Cl + 3 F leaving groups). Computed values and the
join: `notes/assets/poc_validation_join.csv`; statistics:
`notes/assets/poc_validation_stats.json`; scatter:
`notes/assets/poc_validation_scatter.png`._

## Go / no-go

**Qualified GO — scale the direct ΔG‡ engine as a within-leaving-group ranker; add
solvation before pooling across leaving groups or trusting absolute magnitudes.**

What the POC establishes:

- The end-to-end machinery is sound and reliable: SMILES → docked complex → concerted
  relaxed scan → DFT profile + Mayer-bond-order peak validation → optking TS saddle →
  qRRHO ΔG‡, **11/11 confirmed saddles, no hand-holding**.
- The computed barrier **carries real reactivity information**: within a leaving-group
  family it reproduces the published ranking at R² ≈ 0.95 and offset-corrected MAE < 1
  kcal/mol — already good enough to **label substrates the empirical six-ring model
  cannot reach** (e.g. five-membered heterocycles), provided labels are used *within*
  a leaving-group family or after an offset calibration.

What it does *not* yet support, and the conditions to lift them:

- **Cross-leaving-group pooling and absolute magnitudes need solvation.** The
  leaving-group-dependent offset (fluoride over-penalised by ~15 kcal/mol) is a direct
  consequence of the gas-phase, no-PCM path. **Add PCM/SMD** (already the next deferred
  item) before mixing F/Cl/Br or comparing to experiment in absolute terms.
- **Consider the addition (Meisenheimer) coordinate once solvation is in.** With solvent
  stabilising the developing charge, the stepwise addition TS becomes well-defined and
  is the experimentally rate-limiting step for many activated arenes; the gas-phase POC
  had to use the concerted coordinate precisely because the zwitterionic adduct is not a
  gas-phase minimum.
- **Nucleophile.** A neutral model amine was a deliberate, validated stand-in for Lu's
  alkoxide; revisit once solvation allows the anionic nucleophile to be modelled.

Net: the approach is validated and worth scaling; the immediate next investment is
PCM/SMD on the Psi4 path, after which a cross-leaving-group re-validation is warranted.

## Reproduce

```bash
# one substrate (smoke test)
python scripts/run_poc.py --smiles "O=[N+]([O-])c1ccc(F)cc1" --leaving-group F \
    --outdir data/processed/poc_smoke
# the 10-substrate batch (parallel, resumable)
N_WORKERS=4 THREADS=4 MEM_GB=5 bash scripts/run_poc_batch.sh \
    data/external/lu74_poc_slice.csv data/processed/poc_run
# validation (correlation, magnitude, scatter)
python scripts/validate_poc.py --slice data/external/lu74_poc_slice.csv \
    --run data/processed/poc_run --outdir notes/assets
```
