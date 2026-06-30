"""Tests for snar_qc.poc.complex.build_reaction_complex.

These are fast (RDKit embedding only, no QC): they check that the builder identifies
the right reactive atoms, docks a neutral amine at the requested distance on the ring
normal, and keeps the engine's 1-indexed atom contract self-consistent.
"""

import numpy as np
import pytest

from rdkit import Chem

from snar_qc.poc.complex import (
    _activation_score,
    build_molecule,
    build_reaction_complex,
)


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


def _count_neighbours(atoms, centre_one_indexed, symbol, cutoff):
    """Number of atoms of ``symbol`` within ``cutoff`` Angstrom of a 1-indexed atom."""
    centre = atoms.get_positions()[centre_one_indexed - 1]
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    count = 0
    for i, sym in enumerate(symbols):
        if i == centre_one_indexed - 1 or sym != symbol:
            continue
        if float(np.linalg.norm(positions[i] - centre)) <= cutoff:
            count += 1
    return count


# 2-chloro-4-nitropyridine: the ipso-S~N~Ar probe substrate. C2 bears Cl (Path A), C4
# bears NO2 (Path B); the two pathways must resolve to *different* ipso carbons.
_NITROPYRIDINE = "Clc1nccc([N+](=O)[O-])c1"


def test_nitro_leaving_group_selects_nitrogen_and_ipso_carbon():
    """leaving_group='NO2' picks the nitro N as LG and its aromatic carbon as ipso."""
    rc = build_reaction_complex(_NITROPYRIDINE, leaving_group="NO2")

    symbols = rc.atoms.get_chemical_symbols()
    # The leaving atom is the nitro nitrogen; the ipso is an aromatic carbon; Nu is N.
    assert symbols[rc.lg_atom - 1] == "N"
    assert symbols[rc.central_atom - 1] == "C"
    assert symbols[rc.nu_atom - 1] == "N"
    assert rc.leaving_group == "NO2"

    # The leaving nitrogen is bonded to the ipso carbon (within a C-N bond length) ...
    assert _distance(rc.atoms, rc.central_atom, rc.lg_atom) < 1.6
    # ... and carries the two nitro oxygens (so it really is a nitro nitrogen).
    assert _count_neighbours(rc.atoms, rc.lg_atom, "O", cutoff=1.4) == 2
    assert rc.atoms.info["charge"] == 0


def test_nitro_token_is_case_insensitive():
    """'nitro' selects the same nitro-displacement site as 'NO2'."""
    rc_no2 = build_reaction_complex(_NITROPYRIDINE, leaving_group="NO2")
    rc_nitro = build_reaction_complex(_NITROPYRIDINE, leaving_group="nitro")
    assert rc_nitro.central_atom == rc_no2.central_atom
    assert rc_nitro.lg_atom == rc_no2.lg_atom
    assert rc_nitro.leaving_group == "NO2"


def test_nitro_and_halide_paths_pick_different_ipso_carbons():
    """On 2-Cl-4-NO2-pyridine, Path A (Cl) and Path B (NO2) target distinct carbons."""
    rc_cl = build_reaction_complex(_NITROPYRIDINE, leaving_group="Cl")
    rc_no2 = build_reaction_complex(_NITROPYRIDINE, leaving_group="NO2")

    # Path A leaves chlorine; Path B leaves the nitro nitrogen.
    assert rc_cl.atoms.get_chemical_symbols()[rc_cl.lg_atom - 1] == "Cl"
    assert rc_no2.atoms.get_chemical_symbols()[rc_no2.lg_atom - 1] == "N"
    # The two pathways attack different ring carbons (C2 vs C4) -- the crux of the probe.
    assert rc_cl.central_atom != rc_no2.central_atom


# 1,2,4-trinitrobenzene: the Senger 2012 nitro-leaving-group anchor. All three nitro
# carbons score equally under a position-blind activator count, but only C1 is ortho
# *and* para to the other two nitros -- the kinetically favoured ipso for nitrite loss.
_TNB = "O=[N+]([O-])c1ccc([N+](=O)[O-])c([N+](=O)[O-])c1"


def _doubly_activated_nitro_carbon(smiles):
    """Heavy-atom index of the ring carbon with the most ortho/para nitro activators."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ring = max(mol.GetRingInfo().AtomRings(), key=len)

    def bears_nitro(idx):
        atom = mol.GetAtomWithIdx(idx)
        return any(
            nbr.GetIdx() not in ring
            and nbr.GetSymbol() == "N"
            and any(b.GetSymbol() == "O" for b in nbr.GetNeighbors())
            for nbr in atom.GetNeighbors()
        )

    nitro_carbons = [i for i in ring if bears_nitro(i)]
    return max(nitro_carbons, key=lambda i: _activation_score(mol, i))


def test_activation_score_is_ortho_para_aware_on_tnb():
    """On TNB the C1 nitro (ortho+para to the other two) outscores the C2/C4 nitros."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_TNB))
    ring = max(mol.GetRingInfo().AtomRings(), key=len)

    def bears_nitro(idx):
        atom = mol.GetAtomWithIdx(idx)
        return any(
            nbr.GetIdx() not in ring
            and nbr.GetSymbol() == "N"
            and any(b.GetSymbol() == "O" for b in nbr.GetNeighbors())
            for nbr in atom.GetNeighbors()
        )

    scores = sorted(_activation_score(mol, i) for i in ring if bears_nitro(i))
    # Two singly-activated carbons (C2, C4) and one doubly-activated (C1).
    assert scores == [1, 1, 2]


def test_nitro_leaving_group_picks_doubly_activated_carbon_on_tnb():
    """build_reaction_complex(TNB, NO2) targets the doubly-activated C1, not a tie-break."""
    rc = build_reaction_complex(_TNB, leaving_group="NO2")
    # The ipso carbon the builder chose must be the doubly-activated one.
    assert rc.central_atom - 1 == _doubly_activated_nitro_carbon(_TNB)
    # The leaving atom is its nitro nitrogen, bonded to that ipso carbon.
    assert rc.atoms.get_chemical_symbols()[rc.lg_atom - 1] == "N"
    assert _distance(rc.atoms, rc.central_atom, rc.lg_atom) < 1.6
    assert _count_neighbours(rc.atoms, rc.lg_atom, "O", cutoff=1.4) == 2


def test_nitro_leaving_group_requires_a_nitro_group():
    """Requesting NO2 on a substrate with no aromatic nitro raises a clear error."""
    with pytest.raises(ValueError, match="nitro"):
        build_reaction_complex("Clc1ccncc1", leaving_group="NO2")


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
