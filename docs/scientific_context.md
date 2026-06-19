# Scientific context — snar-qc

## Purpose

`snar-qc` computes the **activation free energy (ΔG‡)** of nucleophilic aromatic
substitution (S~N~Ar) **directly**, from explicit ground and transition states, rather
than reading it off an empirical descriptor regression. It also computes the ground-state
electronic descriptors such regressions use. The computed quantities are the *product*;
downstream reactivity analysis is the *consumer*.

## Why this exists (the gap)

Empirical S~N~Ar reactivity models — for example Lu, Paci & Leitch (2022), which regresses
ΔG‡ on ground-state descriptors (−LUMO / EA, ESP features) — are fit on a set of
six-membered and fused 6–6 electrophiles. **Five-membered heterocycles are absent**, and
the empirical model is undefined there. Computing ΔG‡ from first principles supplies the
labels the empirical line cannot obtain from that data alone.

## Strategy

1. **Direct (expensive, general).** Build the S~N~Ar σ-adduct / transition-state guess
   from an atom-mapped reaction template (SMIRKS) for a model amine nucleophile; locate
   the saddle point; compute ΔG‡.
2. **Validate.** Check computed ΔG‡ against published S~N~Ar kinetics on a subsample of a
   well-characterised reactivity set.
3. **Cheap surrogate.** Train a regression on cheap ground-state descriptors using the
   *computed* ΔG‡ as ground truth — now defined across ring sizes, including the
   five-membered heterocycles the published six-ring model cannot reach.

## Anchors and references

- **Validation set:** Lu, Paci & Leitch, *Chem. Sci.* 2022, 13, 12681–12695
  (DOI 10.1039/d2sc04041g). Nucleophile: benzyl alcohol (alkoxide).
- **Prior art for the TS search:** autodE (Young, Silcock, Sterling & Duarte, *Angew.
  Chem. Int. Ed.* 2021, DOI 10.1002/anie.202011941); TS-tools (Stuyver, *J. Comput. Chem.*
  2024, DOI 10.1002/jcc.27374); the `predict-SNAr` workflow (Jorner, Brinck, Norrby &
  Buttar, *Chem. Sci.* 2021, DOI 10.1039/d0sc04896h).

## Status

Greenfield. Repository scaffolded; no implementation yet. The first execution is the ΔG‡
proof of concept: a model amine nucleophile against a spread of validation substrates,
B3LYP/def2-SVP with geometry optimisation, checking correlation and magnitude against the
published ordering.
