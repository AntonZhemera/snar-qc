"""Tests for snar_qc.qc.gpu4pyscf_calculator.GPU4PySCFCalculator (Stage A).

The whole module is skipped where the optional ``[gpu]`` stack is absent (CI, the
Windows workstation, CPU-only dev hosts) -- ``gpu4pyscf`` is imported at module scope by
the calculator, so importing it here requires that stack. The CPU-fallback / factory
behaviour is covered GPU-free in tests/test_backend.py.

The slow energy test is a real gpu4pyscf B3LYP-D3BJ/def2-SVP single point on NH3, asserted
for **parity** against the same pinned Psi4 reference the Psi4 calculator test uses
(it additionally needs a usable CUDA device). The contract tests (multiplicity, option
merge, the Stage-A NotImplementedError guards) need only the importable stack, not a
device.
"""

import math

import pytest
from ase import Atoms

pytest.importorskip("gpu4pyscf", reason="GPU backend stack ([gpu] extra) not installed")

from snar_qc.qc.backend import GPUUnavailableError, probe_gpu  # noqa: E402
from snar_qc.qc.gpu4pyscf_calculator import GPU4PySCFCalculator  # noqa: E402

# B3LYP-D3BJ/def2-SVP NH3 single-point energy, pinned from Psi4 (see
# tests/test_psi4_calculator.py). The GPU backend must reproduce it: the 5-ring
# reference complex matched Psi4 to 3e-7 Ha (notes/2026-06-23_gpu_hessian_benchmark.md),
# so 1e-4 Ha here is a generous parity bound, not a loose one.
NH3_REFERENCE_HARTREE = -56.51059216
ENERGY_TOL_HARTREE = 1e-4


def _has_gpu_device() -> bool:
    try:
        probe_gpu()
        return True
    except GPUUnavailableError:
        return False


requires_gpu_device = pytest.mark.skipif(
    not _has_gpu_device(), reason="no usable CUDA device / sufficient VRAM"
)


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


# -- contract tests (no device needed) -------------------------------------------


def test_multiplicity_parity():
    """Even electron count -> singlet; odd -> doublet (mirrors Psi4Calculator)."""
    calc = GPU4PySCFCalculator(atoms=_nh3())
    assert calc._multiplicity(0) == 1  # NH3: 10 electrons -> singlet
    assert calc._multiplicity(1) == 2
    assert calc._multiplicity(-1) == 2


def test_options_merge_onto_defaults():
    """Overrides apply; untouched defaults (B3LYP-D3BJ/def2-SVP) are preserved."""
    calc = GPU4PySCFCalculator(atoms=_nh3(), options={"functional": "pbe"})
    assert calc.options["functional"] == "pbe"
    assert calc.options["basis_set"] == "def2-svp"
    assert calc.options["dispersion"] == "d3bj"


def test_charge_read_from_atoms():
    atoms = _nh3()
    atoms.info["charge"] = 0
    calc = GPU4PySCFCalculator(atoms=atoms)
    assert calc.options["charge"] == 0


def test_solvent_not_supported_in_stage_a():
    """Stage A is gas phase: a configured solvent raises, it does not run silently."""
    calc = GPU4PySCFCalculator(atoms=_nh3(), options={"solvent": "DMSO"})
    with pytest.raises(NotImplementedError, match="solvation"):
        calc.single_point()


def test_opt_not_supported_in_stage_a():
    calc = GPU4PySCFCalculator(atoms=_nh3())
    with pytest.raises(NotImplementedError, match="single_point only"):
        calc.opt()


def test_freq_not_supported_in_stage_a():
    calc = GPU4PySCFCalculator(atoms=_nh3())
    with pytest.raises(NotImplementedError, match="single_point only"):
        calc.freq()


def test_ts_flag_not_supported_in_stage_a():
    """The ``ts`` guard fires even if the flag is set directly (no ts() in Stage A)."""
    calc = GPU4PySCFCalculator(atoms=_nh3())
    calc.options["ts"] = True
    with pytest.raises(NotImplementedError, match="single_point only"):
        calc.run_calc()


# -- energy parity (needs a real device) -----------------------------------------


@pytest.mark.slow
@requires_gpu_device
def test_nh3_single_point_energy_parity():
    """gpu4pyscf B3LYP-D3BJ/def2-SVP single point on NH3 matches the Psi4 reference."""
    calc = GPU4PySCFCalculator(atoms=_nh3())
    energy = calc.single_point()

    assert isinstance(energy, float)
    assert math.isfinite(energy)
    assert calc.energy == energy
    assert calc.mean_field is not None
    assert abs(energy - NH3_REFERENCE_HARTREE) < ENERGY_TOL_HARTREE
    # A single point captures no thermochemistry.
    assert calc.free_energy is None
    assert calc.frequencies is None
