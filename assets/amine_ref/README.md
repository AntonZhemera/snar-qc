# `assets/amine_ref/` — fixed model-amine reference cache

Cached gas-phase references for the **fixed model amine** (methylamine) used by the
separated-reactants ΔG‡ barrier. The amine is invariant across substrates and runs, so its
opt+freq is computed/copied **once** and reused, skipping a redundant ~12 s opt+freq on every
later substrate (`snar_qc.poc.amine_cache`; consumed by `barrier.cached_amine_reference`).

**One pair per `(amine, level-of-theory, backend)`:**

- `<amine>__<level>__<backend>.json` — gas thermochemistry scalars (electronic energy,
  harmonic/quasi-harmonic Gibbs, enthalpy, ZPVE, all Hartree), `n_imag`, and provenance.
- `<amine>__<level>__<backend>.xyz` — the gas-optimised amine geometry (extended XYZ,
  carries the charge), needed for the follow-up implicit-solvent single point.

e.g. `CN__b3lyp-d3bj_def2-svp__gpu4pyscf.{json,xyz}`.

**Backend-keyed on purpose.** Psi4 and gpu4pyscf absolute energies differ (density fitting,
screening), so a Psi4-seeded amine must never combine with a gpu4pyscf TS/ArX — the key keeps
them apart. The runtime validates the stored `(amine, backend, level)` against the request and
treats any mismatch as a miss (never a wrong-reference hit).

**Seeding (no recompute).** Every finished gas run already holds this reference in each
substrate's `gas_thermo.json` (`species.amine`) + `amine_opt.xyz`, so copy it:

```bash
python scripts/extract_amine_ref.py --run data/processed/<run>_gas --backend gpu4pyscf
```

The seeder reads **all** substrates in the run and verifies the amine energy is constant
(<1e-6 Hartree) before writing one canonical asset — a built-in check that the reference
really is fixed. To override the location use `$SNAR_QC_AMINE_CACHE`; to force a fresh
recompute, delete the pair and the next run regenerates + re-stores it.
