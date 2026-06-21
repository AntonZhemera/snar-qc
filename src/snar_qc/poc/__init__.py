"""snar_qc.poc -- proof-of-concept end-to-end ΔG‡ runner for the predict-SNAr adoption.

This subpackage wires the Stage 1-3 engine pieces (``Psi4TSScan``,
``Psi4Calculator.ts_freq``, ``Psi4Thermo`` / ``activation_free_energy``) into a single,
resumable substrate runner and confronts it with the **Lu_74** S~N~Ar reactivity set.

The validation strategy is deliberately a *ranking* one. Lu_74's published barriers were
measured with an anionic benzyl-alkoxide nucleophile; gas-phase/SMD QC handles anions
poorly, so the POC computes ΔG‡ for a neutral **model amine** (methylamine) instead and
asks whether the computed ordering tracks the experimental one (Spearman / Pearson),
accepting a systematic absolute offset (no PCM/SMD on the Psi4 path yet).

Public names are imported explicitly from their modules (e.g.
``from snar_qc.poc.complex import build_reaction_complex``) so importing this subpackage
does not pull in RDKit / Psi4.

- :mod:`snar_qc.poc.complex` -- build an amine + aryl-halide reaction complex (ASE
  ``Atoms`` + 1-indexed central/nu/lg atom indices) from SMILES, via RDKit.
- :mod:`snar_qc.poc.barrier` -- drive the engine from a reaction complex to a
  quasi-harmonic ΔG‡ (relaxed scan -> DFT SPs -> peak -> TS opt+freq -> thermo).
"""
