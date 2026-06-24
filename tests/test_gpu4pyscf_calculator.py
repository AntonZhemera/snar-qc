"""Tests for snar_qc.qc.gpu4pyscf_calculator.GPU4PySCFCalculator (Stages A-B).

The whole module is skipped where the optional ``[gpu]`` stack is absent (CI, the
Windows workstation, CPU-only dev hosts) -- ``gpu4pyscf`` is imported at module scope by
the calculator, so importing it here requires that stack. The CPU-fallback / factory
behaviour is covered GPU-free in tests/test_backend.py.

The slow tests are real gpu4pyscf B3LYP-D3BJ/def2-SVP runs (and need a usable CUDA
device): a single point on NH3 and a geometry minimisation of the rigid aryl-nitrile
that broke optking, each asserted for **parity** against a pinned reference. The contract
tests (multiplicity, option merge, coordinate selection, the NotImplementedError guards)
need only the importable stack, not a device.
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

# The 5-ring reference aryl halide N#Cc1ccc(F)s1 -- the rigid planar aryl nitrile whose
# near-linear C#N drove optking's internals degenerate (forcing Cartesian min-opt on the
# Psi4 path). geomeTRIC's TRIC relaxes it cleanly on GPU gradients; the minimum it reaches
# is pinned here. Start geometry is an MMFF-relaxed embedding (deterministic, so the test
# needs no RDKit); the B3LYP-D3BJ/def2-SVP minimum is basin-stable regardless of start.
ARYL_HALIDE_MIN_HARTREE = -744.12555
# Psi4 (B3LYP-D3BJ/def2-SVP) opt_freq reference on the same aryl halide, for the Stage C
# analytic-Hessian thermochemistry parity. Cs/sigma=1, so no rotational-symmetry-number
# ambiguity between the backends. Enthalpy/ZPVE match the GPU analytic Hessian to <3e-5 Ha;
# Gibbs differs ~1.8e-4 Ha (0.11 kcal/mol) via the low-mode entropy -- the expected
# few-cm^-1 analytic-vs-finite-difference frequency difference.
PSI4_ARYL_GIBBS_HARTREE = -744.0993651
PSI4_ARYL_ENTHALPY_HARTREE = -744.0601765
PSI4_ARYL_ZPVE_HARTREE = 0.0578450
GIBBS_TOL_HARTREE = 5e-4  # covers the analytic-vs-FD low-mode entropy difference
ARYL_HALIDE_START = [
    ("N", (3.48380, -0.42810, -0.32213)),
    ("C", (2.32788, -0.38165, -0.23039)),
    ("C", (0.90669, -0.30208, -0.11405)),
    ("C", (0.19164, 0.85764, 0.12191)),
    ("C", (-1.21442, 0.61781, 0.18654)),
    ("C", (-1.49965, -0.70930, -0.00293)),
    ("F", (-2.75066, -1.19599, 0.01127)),
    ("S", (-0.12471, -1.66346, -0.25448)),
    ("H", (0.65045, 1.83302, 0.24298)),
    ("H", (-1.97101, 1.37211, 0.36129)),
]


def _aryl_halide() -> Atoms:
    """The 5-ring reference aryl halide (neutral, closed shell), MMFF start geometry."""
    atoms = Atoms(
        symbols=[s for s, _ in ARYL_HALIDE_START],
        positions=[p for _, p in ARYL_HALIDE_START],
    )
    atoms.info["charge"] = 0
    return atoms


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


def test_solvent_not_supported_yet():
    """A configured solvent raises (gas phase only); it does not run silently."""
    calc = GPU4PySCFCalculator(atoms=_nh3(), options={"solvent": "DMSO"})
    with pytest.raises(NotImplementedError, match="solvation"):
        calc.single_point()


def test_ts_flag_not_supported_yet():
    """The ``ts`` guard fires even if the flag is set directly (no ts() until Stage D)."""
    calc = GPU4PySCFCalculator(atoms=_nh3())
    calc.options["ts"] = True
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        calc.run_calc()


def test_coordsys_default_is_tric():
    """The min-opt default is geomeTRIC's TRIC (robust on the aryl-nitrile case)."""
    assert GPU4PySCFCalculator(atoms=_nh3())._geometric_coordsys() == "tric"


def test_coordsys_cartesian_alias():
    """``"cartesian"`` (the Psi4 option value) maps to geomeTRIC's ``"cart"``."""
    calc = GPU4PySCFCalculator(
        atoms=_nh3(), options={"min_opt_coordinates": "Cartesian"}
    )
    assert calc._geometric_coordsys() == "cart"


def test_coordsys_passthrough():
    """A geomeTRIC coordsys name is passed through (lower-cased)."""
    calc = GPU4PySCFCalculator(atoms=_nh3(), options={"min_opt_coordinates": "DLC"})
    assert calc._geometric_coordsys() == "dlc"


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


@pytest.mark.slow
@requires_gpu_device
def test_aryl_halide_min_opt_converges():
    """geomeTRIC min-opt relaxes the hard aryl-nitrile to the pinned B3LYP minimum.

    This is the Stage-B risk the masterplan flags: the rigid planar aryl nitrile that
    broke optking's internals. geomeTRIC (GPU gradients, TRIC) must converge it.
    """
    atoms = _aryl_halide()
    start = atoms.get_positions().copy()

    calc = GPU4PySCFCalculator(atoms=atoms)
    energy = calc.opt()

    assert math.isfinite(energy)
    assert calc.energy == energy
    assert abs(energy - ARYL_HALIDE_MIN_HARTREE) < ENERGY_TOL_HARTREE
    # The relaxed geometry is written back onto the Atoms (it moved off the MMFF start).
    assert calc.atoms is atoms
    assert not (atoms.get_positions() == start).all()


@pytest.mark.slow
@requires_gpu_device
def test_aryl_halide_freq_thermo_parity():
    """Analytic-Hessian frequencies + thermochemistry match the Psi4 FD-freq reference.

    Stage C: opt_freq builds the analytic Hessian and harmonic thermochemistry. Enthalpy
    and ZPVE track Psi4 to <1e-4 Ha; Gibbs to <5e-4 Ha (the low-mode entropy carries the
    few-cm^-1 analytic-vs-FD difference). The frequency list is signed cm^-1 with this
    minimum showing no imaginary mode, and feeds snar_qc.qc.thermo unchanged.
    """
    calc = GPU4PySCFCalculator(atoms=_aryl_halide())
    calc.opt_freq()

    # Frequency list: 3N-6 = 24 modes, all real (a minimum), signed-cm^-1 convention.
    assert len(calc.frequencies) == 24
    assert all(f > 0.0 for f in calc.frequencies)

    # Harmonic thermochemistry parity vs Psi4.
    assert abs(calc.zpve - PSI4_ARYL_ZPVE_HARTREE) < ENERGY_TOL_HARTREE
    assert abs(calc.enthalpy - PSI4_ARYL_ENTHALPY_HARTREE) < ENERGY_TOL_HARTREE
    assert abs(calc.free_energy - PSI4_ARYL_GIBBS_HARTREE) < GIBBS_TOL_HARTREE

    # The GPU calculator duck-types into snar_qc.qc.thermo (the Psi4-named helper).
    from snar_qc.qc.thermo import Psi4Thermo

    thermo = Psi4Thermo.from_calculator(calc)
    assert thermo.imaginary_frequencies == []
    assert math.isfinite(thermo.gibbs_qh)
