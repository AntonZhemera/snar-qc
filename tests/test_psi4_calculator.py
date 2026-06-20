"""Tests for snar_qc.qc.psi4_calculator.Psi4Calculator.

The energy test is a real Psi4 B3LYP-D3BJ/def2-SVP single point on NH3, asserted
against a pinned reference. The multiplicity test is a fast, pure-arithmetic check
of the singlet/doublet parity rule (no QC calculation).
"""

import math

import pytest
from ase import Atoms

from snar_qc.qc.psi4_calculator import Psi4Calculator

# B3LYP-D3BJ / def2-SVP single-point energy of NH3 at the geometry in ``_nh3``,
# pinned from Psi4 1.10.2 with the s-dftd3 dispersion backend. The tolerance is
# wide enough to absorb threading / BLAS numerical noise yet far tighter than any
# method/basis/dispersion regression (1e-4 Ha ~ 0.06 kcal/mol).
NH3_REFERENCE_HARTREE = -56.51059216
ENERGY_TOL_HARTREE = 1e-4


def _nh3() -> Atoms:
    """A near-equilibrium NH3 geometry (Angstrom), neutral closed shell."""
    atoms = Atoms(
        symbols=["N", "H", "H", "H"],
        positions=[
            (0.0000, 0.0000, 0.1173),
            (0.0000, 0.9377, -0.2737),
            (0.8121, -0.4689, -0.2737),
            (-0.8121, -0.4689, -0.2737),
        ],
    )
    atoms.info["charge"] = 0
    return atoms


def test_multiplicity_parity():
    """Even electron count -> singlet; odd -> doublet (mirrors G16Calculator)."""
    calc = Psi4Calculator(atoms=_nh3())
    assert calc._multiplicity(0) == 1  # NH3: 10 electrons -> singlet
    assert calc._multiplicity(1) == 2  # odd electron count -> doublet
    assert calc._multiplicity(-1) == 2  # odd electron count -> doublet


@pytest.mark.slow
def test_nh3_single_point_energy(tmp_path, monkeypatch):
    """Psi4 B3LYP-D3BJ/def2-SVP single point on NH3 is finite and on reference."""
    # Run in a temp dir so Psi4 scratch (psi4.out, timer.dat) never touches the repo.
    monkeypatch.chdir(tmp_path)

    calc = Psi4Calculator(atoms=_nh3())
    energy = calc.single_point()

    assert isinstance(energy, float)
    assert math.isfinite(energy)
    assert calc.energy == energy
    assert calc.wavefunction is not None
    assert abs(energy - NH3_REFERENCE_HARTREE) < ENERGY_TOL_HARTREE
