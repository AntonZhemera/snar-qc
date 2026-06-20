"""Tests for snar_qc.qc.bond_orders.Psi4BondOrders.

The bond-order values are validated against unambiguous references at the production
DFT level (B3LYP-D3BJ/def2-SVP): H2O O-H is a single bond (Mayer ~ 1.0) and N2 is a
triple bond (Mayer well above 2.5). Those need a real Psi4 SCF, so they are marked
slow; the molecules are tiny (sub-second to ~2 s). A fast guard test (no SCF) covers
the None-wavefunction error path.
"""

import math

import pytest
from ase import Atoms

from snar_qc.qc.bond_orders import Psi4BondOrders
from snar_qc.qc.psi4_calculator import Psi4Calculator


def _h2o() -> Atoms:
    """Near-equilibrium H2O (Angstrom), neutral closed shell. Order: O, H, H."""
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.0000, 0.0000, 0.1173),
            (0.0000, 0.7572, -0.4692),
            (0.0000, -0.7572, -0.4692),
        ],
    )
    atoms.info["charge"] = 0
    return atoms


def _n2(distance: float = 1.10) -> Atoms:
    """N2 (Angstrom) at a given bond length, neutral closed shell."""
    atoms = Atoms(symbols=["N", "N"], positions=[(0.0, 0.0, 0.0), (0.0, 0.0, distance)])
    atoms.info["charge"] = 0
    return atoms


def test_rejects_none_wavefunction():
    """Building from a missing wavefunction is a clear ValueError (no SCF needed)."""
    with pytest.raises(ValueError):
        Psi4BondOrders(None)


@pytest.mark.slow
def test_h2o_single_bond_and_indexing(tmp_path, monkeypatch):
    """H2O O-H is a Mayer single bond (~1.0) with NBOParser-compatible indexing."""
    # Temp dir so Psi4 scratch (psi4.out, timer.dat) never touches the repo.
    monkeypatch.chdir(tmp_path)

    calc = Psi4Calculator(atoms=_h2o())
    calc.single_point()
    bo = Psi4BondOrders(calc.wavefunction)
    assert bo.kind == "mayer"

    oh_1 = bo.get_bo(1, 2)  # O-H
    oh_2 = bo.get_bo(1, 3)  # O-H (equivalent by symmetry)
    hh = bo.get_bo(2, 3)  # H...H, non-bonded

    assert math.isfinite(oh_1)
    assert 0.9 < oh_1 < 1.1  # single bond ~ 1.0 (pinned 1.0104)
    # Same 1-indexed, symmetric convention as NBOParser.get_bo.
    assert bo.get_bo(2, 1) == oh_1
    assert abs(oh_1 - oh_2) < 1e-6
    # The non-bonded H...H pair is near zero, far below a real bond.
    assert hh < 0.1
    assert oh_1 > hh


@pytest.mark.slow
def test_n2_triple_bond(tmp_path, monkeypatch):
    """N2 is a Mayer triple bond -- clearly above 2.5 (pinned 2.79 at B3LYP)."""
    monkeypatch.chdir(tmp_path)

    calc = Psi4Calculator(atoms=_n2(1.10))
    calc.single_point()
    bo = Psi4BondOrders(calc.wavefunction)

    nn = bo.get_bo(1, 2)
    assert math.isfinite(nn)
    assert nn > 2.5
    assert bo.get_bo(2, 1) == nn  # symmetric
