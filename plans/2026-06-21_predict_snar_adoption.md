# Plan — Adopt predict-SNAr and swap its QC engine to Psi4 (ΔG‡ PoC)

**Date:** 2026-06-21
**Status:** active (Stage 1 in progress / landing)
**Scope owner:** snar-qc

## Goal

A first-principles S~N~Ar **ΔG‡ proof of concept** that reuses Kjell Jorner's
**predict-SNAr** code (Jorner et al., *Chem. Sci.* 2021, DOI 10.1039/D0SC04896H)
but replaces its quantum-chemistry engine — **Gaussian 16** — with **Psi4** (open
source, already in our conda env). The validation target is reproducing published
S~N~Ar barriers (Lu_74 reference set) on ~10 substrates.

## Pragmatic-reuse strategy (what we reuse vs. rebuild)

predict-SNAr is a large, working pipeline. We reuse as much of it **verbatim** as
possible and add only a thin Psi4 layer in `snar_qc`, so the vendored MIT code stays
unedited.

| Concern | Decision |
|---|---|
| predict-snar source | **Vendor verbatim** as top-level `src/predict_snar/` (absolute `from predict_snar import …` imports must keep working). Our code never edits it. |
| DFT engine | New **`Psi4Calculator(predict_snar.calculators.Calculator)`** — drop-in for `G16Calculator`, B3LYP-D3BJ/def2-SVP via the Psi4 Python API. |
| Relaxed scan along the reaction coordinate | **Reuse the xTB-native `TSScan`** (`TSScan.run_scan` → `XTBCalculator.opt`). xTB is engine-agnostic and already in the env; no Gaussian needed for the scan. |
| DFT single points along the scan | Inject `Psi4Calculator` where `TSScan` builds `G16Calculator` (`TSScan.__init__` line ~65 and `TSScan.run_sps` line ~283), via a `snar_qc` subclass `Psi4TSScan(TSScan)` (keeps vendored code unedited). |
| Bond orders (reactive-atom / peak validation) | **Replace Gaussian NBO `bndidx`** (parsed from `sps/{i}.log`) with **Psi4 Mayer/Wiberg bond orders** read from the Psi4 wavefunction. |
| ΔG‡ extraction | GoodVibes-style free-energy corrections from `opt`+`freq` on the located TS / σ-adduct (Psi4 `optimize`/`frequencies`). |
| Validation | ΔG‡ vs **Lu_74** on ~10 substrates. |

### Explicitly deferred / out of scope (whole effort)

- **`GICTSScan` (Gaussian GIC relaxed scan) on Psi4** — *not* reimplemented. Psi4 has
  no Gaussian-GIC equivalent; we deliberately route the relaxed scan through the
  **xTB-native `TSScan`** instead. (`GICTSScan` stays in the vendored tree, unused.)
- **`predict_snar/descriptors.py`** — *not* ported. The ground-state descriptor
  pipeline is a separate future consolidation (per snar-qc CLAUDE.md), not part of the
  ΔG‡ PoC.
- Solvation (PCM) on the Psi4 path, async/Popen+parse adaptation to `jobs.py`, and the
  full `__main__`/`jobs` SMILES→ΔG‡ driver — later stages only.

## Stages

### Stage 1 — Adoption + `Psi4Calculator` skeleton  *(this stage)*
- DDD plan (this file).
- Vendor `predict_snar` verbatim → `src/predict_snar/` (+ `src/data/`, MIT `LICENSE`,
  `VENDORED.md`). Apache `NOTICE` at repo root. Packaging so both `snar_qc` and
  `predict_snar` import from `src/`.
- `Psi4Calculator(Calculator)` — single point is the must-have; `opt`/`freq` thin
  wrappers. TDD test: real NH3 B3LYP-D3BJ/def2-SVP single point, finite + on a pinned
  reference.
- **Out of scope here:** wiring `jobs.py`, running any S~N~Ar substrate end-to-end,
  bond orders, descriptors, GIC.

### Stage 2 — Psi4 DFT single points inside the xTB-native scan
- `Psi4TSScan(TSScan)` in `snar_qc`: override `__init__` (build `Psi4Calculator`
  instead of `G16Calculator`), `run_sps`, and `read_sp_output` so DFT single points run
  synchronously through Psi4 and energies come back directly (no `sps/{i}.log` cclib
  parse). Reproduce the relaxed-scan → DFT-SP **energy profile** on one substrate.

### Stage 3 — Bond orders + thermochemistry
- Psi4 **Mayer/Wiberg bond orders** from the wavefunction → reactive-atom identification
  and `TSScan.validate_peaks` analogue (replacing NBO `bndidx`).
- ΔG‡ extraction: `opt`+`freq` on the located TS / σ-adduct; GoodVibes-style free-energy
  correction to a barrier.

### Stage 4 — Validation vs Lu_74
- Run ~10 substrates end-to-end; compare computed ΔG‡ to Lu_74 (and to predict-snar /
  experiment). Findings note in `notes/`.

## `Psi4Calculator` contract

Subclasses `predict_snar.calculators.Calculator`; the base's
`single_point` / `opt` / `opt_freq` / `freq` set the `opt`/`freq`/`ts` flags on
`self.options` and dispatch to `run_calc` — so **`run_calc` is the one method a
`Calculator` subclass must supply** (as `G16Calculator`/`XTBCalculator` do).

**Implemented in Stage 1**

| Member | Role |
|---|---|
| `__init__(atoms, file, options)` | Seed B3LYP-D3BJ/def2-SVP option defaults; read total charge from `atoms.info["charge"]`; name the Psi4 output file. |
| `run_calc(n_procs=1, mem=2.0) -> float` | Build the Psi4 molecule from the ASE geometry, set memory/threads/SCF options, dispatch on the `opt`/`freq` flags to `psi4.energy` / `psi4.optimize` / `psi4.frequencies`, return the energy **in Hartree** (also `self.energy`; wavefunction on `self.wavefunction`). |
| `single_point` / `opt` / `opt_freq` / `freq` | Inherited; work via `run_calc`. |
| `_build_molecule`, `_multiplicity`, `_method_string` | Helpers: ASE→Psi4 geometry (`no_com`/`no_reorient`/`symmetry c1`); singlet/doublet by electron-count parity (mirrors `G16Calculator`); `b3lyp-d3bj/def2-svp` method string. |

**Known divergence (flag for Stage 2):** `G16Calculator.run_calc` is asynchronous
(returns a `subprocess.Popen`; energies parsed later from the log). The Psi4 Python API
is synchronous, so `run_calc` runs in-process and returns the energy directly. The
`Psi4TSScan` override absorbs this (no Popen/parse loop); we do **not** retrofit Psi4
into predict-snar's async wait/parse machinery.

**Deferred contract surface (later stages):** PCM solvation option; Mayer/Wiberg
bond-order accessor off the wavefunction; free-energy/thermochemistry accessor;
`ts`/`ts_freq` TS optimisation; optional parser-compatible output for `jobs.py`.

## Environment / packaging notes

- Vendoring forced four predict-snar runtime deps into the env (needed just to import
  `predict_snar.calculators`): **ase, cclib, joblib, mendeleev** (added to
  `environment.yml`).
- Psi4 `-d3bj` needs a dispersion backend: **dftd3-python** (s-dftd3) added to
  `environment.yml`.
- `src/data/` is a plain data dir (no `__init__.py`); the vendored package resolves it
  via `Path(__file__).parent / "../data"`. Works for the editable dev install; a built
  wheel would need explicit data inclusion (Stage 2+ packaging concern, if ever wheeled).
- Repo stays **Apache-2.0**; vendored code is MIT (NOTICE + `src/predict_snar/LICENSE`).
