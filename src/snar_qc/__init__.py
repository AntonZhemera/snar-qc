"""snar_qc -- quantum chemistry for nucleophilic aromatic substitution (SNAr) reactivity.

Builds SNAr sigma-adduct / transition-state geometries (from atom-mapped reaction SMARTS),
computes the activation free energy (delta-G-double-dagger) and ground-state electronic
descriptors, and validates the computed barriers against published SNAr kinetics. The
computed quantities are emitted as tables for downstream reactivity analysis.

See ``CLAUDE.md`` for the operating contract and ``docs/scientific_context.md`` for the
scientific framing. Greenfield as of 2026-06-15 -- no implementation yet.
"""

__version__ = "0.1.0"
