# snar-qc

Quantum-chemistry tooling for **nucleophilic aromatic substitution (S~N~Ar)** reactivity:
first-principles **activation free energies (ΔG‡)** from explicit transition states, plus
ground-state electronic descriptors.

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-early-lightgrey.svg)

> In the course of my doctoral research I repeatedly needed to run series of
> quantum-chemical calculations. This repository collects the tooling I built and used for
> that purpose. If it proves useful to the wider community, so much the better. The
> substantive credit belongs to the authors of the underlying tools this work stands on —
> my thanks and respect to them. AI coding agents were used extensively throughout
> development (with thanks to Anthropic).

## What it does

S~N~Ar reactivity is commonly predicted from empirical regressions on ground-state
descriptors. Those models are fit on six-membered (and fused 6–6) electrophiles, so
**five-membered heterocycles fall outside their domain**. `snar-qc` computes ΔG‡ directly
from explicit ground and transition states, supplying activation-energy labels where the
empirical models are undefined — so a reactivity model can be extended across ring sizes.

The computed barriers (and ground-state descriptors) are the **product**: they serve as
labels and inputs for downstream reactivity analysis.

Full framing and references: [`docs/scientific_context.md`](docs/scientific_context.md).

## Approach

1. Build the S~N~Ar σ-adduct / transition-state guess from an atom-mapped reaction
   template (SMIRKS) for a model amine nucleophile.
2. Locate the saddle point; compute ΔG‡ (B3LYP/def2-SVP to start).
3. Validate against published S~N~Ar kinetics.
4. Train a cheap ground-state-descriptor surrogate on the computed ΔG‡ — now defined for
   five-membered heterocycles too.

Reuse before reinventing the TS search: established tools such as autodE, TS-tools, and
S~N~Ar-specific transition-state workflows.

## Status

**Early / greenfield.** First work is the ΔG‡ proof of concept above; the package's
ground-state descriptor tooling is consolidated here over time.

## Layout

```
snar-qc/
├── src/
│   ├── snar_qc/       # core package
│   └── predict_snar/  # vendored MIT scaffolding (see src/predict_snar/VENDORED.md)
├── scripts/         # pipeline / batch scripts
├── tests/           # pytest suite
├── data/            # raw / processed / external (gitignored payloads)
├── assets/          # reaction-template (SMIRKS) catalogues
├── notes/           # dated interpretive notes
├── plans/           # work plans (+ archive/)
└── docs/            # stable reference (scientific context, runbooks)
```

## Environment

Python 3.10 via conda / Mamba (conda-forge). The QC stack (Psi4 / xTB / autodE + RDKit)
lives in the conda env:

```bash
mamba env create -f environment.yml   # creates 'snar-qc'
mamba activate snar-qc
pip install -e . --no-deps
```

A lightweight `pip install -e .[dev]` venv is enough for the package, analysis, and tests.

## Contributing

Conventional Commits; Black + Ruff; a documentation → tests → implementation (DDD/TDD)
flow. Please discuss non-trivial changes in an issue first.

## License

[Apache-2.0](LICENSE).

## Citation

If you use this work, please cite it — see [`CITATION.cff`](CITATION.cff).

## Author

Anton Zhemera — Taras Shevchenko National University of Kyiv.
