"""Tests for snar_qc.poc.complex.build_reaction_complex.

These are fast (RDKit embedding only, no QC): they check that the builder identifies
the right reactive atoms, docks a neutral amine at the requested distance on the ring
normal, and keeps the engine's 1-indexed atom contract self-consistent.
"""

import numpy as np
import pytest

from snar_qc.poc.complex import build_molecule, build_reaction_complex


def _distance(atoms, i_one_indexed, j_one_indexed):
    """Distance (Angstrom) between two 1-indexed atoms of an ASE Atoms object."""
    return float(atoms.get_distance(i_one_indexed - 1, j_one_indexed - 1))


def test_para_fluoronitrobenzene_reactive_atoms():
    """1-fluoro-4-nitrobenzene + methylamine: F leaves, ipso C is its neighbour, N is Nu."""
    rc = build_reaction_complex("O=[N+]([O-])c1ccc(F)cc1", leaving_group="F")

    symbols = rc.atoms.get_chemical_symbols()
    # Leaving group is the fluorine; central atom is an aromatic carbon; Nu is nitrogen.
    assert symbols[rc.lg_atom - 1] == "F"
    assert symbols[rc.central_atom - 1] == "C"
    assert symbols[rc.nu_atom - 1] == "N"
    assert rc.leaving_group == "F"

    # The leaving F is bonded to the central carbon (within a C-F bond length).
    assert _distance(rc.atoms, rc.central_atom, rc.lg_atom) < 1.6
    # Neutral closed-shell complex.
    assert rc.atoms.info["charge"] == 0


def test_amine_docked_at_requested_distance():
    """The nucleophilic N sits at ~approach Angstrom from the ipso carbon."""
    rc = build_reaction_complex(
        "O=[N+]([O-])c1ccc(F)cc1", leaving_group="F", approach=3.0
    )
    assert _distance(rc.atoms, rc.central_atom, rc.nu_atom) == pytest.approx(
        3.0, abs=0.3
    )


def test_indices_in_range_and_distinct():
    """central / nu / lg are valid distinct 1-indexed atoms; Nu is in the amine block."""
    rc = build_reaction_complex("Clc1ccncc1", leaving_group="Cl")
    n_atoms = len(rc.atoms)
    for idx in (rc.central_atom, rc.nu_atom, rc.lg_atom):
        assert 1 <= idx <= n_atoms
    assert len({rc.central_atom, rc.nu_atom, rc.lg_atom}) == 3
    # The nucleophile nitrogen belongs to the appended amine, after the aryl block.
    assert rc.nu_atom > rc.lg_atom
    assert rc.nu_atom > rc.central_atom


def test_picks_specified_leaving_element_over_other_halogen():
    """With both Cl and F present, leaving_group selects which halide leaves."""
    # 1-chloro-4-fluorobenzene: choose the chlorine explicitly.
    rc_cl = build_reaction_complex("Fc1ccc(Cl)cc1", leaving_group="Cl")
    assert rc_cl.atoms.get_chemical_symbols()[rc_cl.lg_atom - 1] == "Cl"
    # And the fluorine when asked for F.
    rc_f = build_reaction_complex("Fc1ccc(Cl)cc1", leaving_group="F")
    assert rc_f.atoms.get_chemical_symbols()[rc_f.lg_atom - 1] == "F"


def test_charge_is_neutral_and_geometry_finite():
    """The combined geometry is finite and the complex is charge-neutral."""
    rc = build_reaction_complex("N#Cc1ccc(F)cc1", leaving_group="F")
    assert rc.atoms.info["charge"] == 0
    assert np.all(np.isfinite(rc.atoms.get_positions()))


def test_build_molecule_reference_species():
    """build_molecule embeds a bare species with explicit Hs and the formal charge."""
    # Methylamine: C, N, and 5 H once hydrogens are added -> 7 atoms, neutral.
    amine = build_molecule("CN")
    assert amine.info["charge"] == 0
    assert sorted(amine.get_chemical_symbols()) == ["C", "H", "H", "H", "H", "H", "N"]
    assert np.all(np.isfinite(amine.get_positions()))
    # A bare aryl halide keeps its halogen and stays neutral.
    arx = build_molecule("O=[N+]([O-])c1ccc(F)cc1")
    assert arx.info["charge"] == 0
    assert "F" in arx.get_chemical_symbols()
