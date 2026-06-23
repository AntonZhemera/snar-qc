"""Tests for the real-pool slice builder's buildability gate.

The gate must agree with ``build_reaction_complex``: a halogen on a 'fragile'
aromatic ring that RDKit aromatises but MMFF demotes (coumarin pyranone,
azolo-fused C=N) is NOT a usable S~N~Ar site and must be rejected up front,
rather than slipping into the slice and failing at build time.
"""

import sys
from pathlib import Path

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_realpool_slice as b  # noqa: E402


def test_gate_rejects_fragile_aromatic_halides():
    # 4-chlorocoumarin (Cl on the non-aromatic pyranone) and an azolo-fused C-Cl.
    assert b.buildable_lg_site(Chem.MolFromSmiles("O=c1cc(Cl)c2ccccc2o1"), "Cl") is False
    assert b.buildable_lg_site(Chem.MolFromSmiles("Clc1nccn2nnnc12"), "Cl") is False


def test_gate_accepts_genuine_aryl_halides():
    assert b.buildable_lg_site(Chem.MolFromSmiles("O=[N+]([O-])c1ccc(Cl)nc1"), "Cl") is True
    assert b.buildable_lg_site(Chem.MolFromSmiles("N#Cc1ccc(Cl)nc1"), "Cl") is True
    assert b.buildable_lg_site(Chem.MolFromSmiles("Fc1ccc(F)cc1"), "F") is True
