"""Tests for snar_qc.qc.gpu4pyscf_calculator.GPU4PySCFCalculator (Stages A-D).

The whole module is skipped where the optional ``[gpu]`` stack is absent (CI, the
Windows workstation, CPU-only dev hosts) -- ``gpu4pyscf`` is imported at module scope by
the calculator, so importing it here requires that stack. The CPU-fallback / factory
behaviour is covered GPU-free in tests/test_backend.py.

The slow tests are real gpu4pyscf B3LYP-D3BJ/def2-SVP runs (and need a usable CUDA
device): an NH3 single point, an aryl-nitrile minimisation, its analytic-Hessian
frequencies/thermochemistry, and an SNAr transition-state search -- each asserted against
a pinned reference (vs Psi4 where available; vs the GPU's own converged result for the TS,
whose Psi4 barrier parity is deferred). The contract tests (multiplicity, option merge,
coordinate selection, the dispatch-flag and solvation guards) need only the importable
stack, not a device.
"""

import math
import types

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


# Smoke-substrate SNAr transition state: lu_0 = para-fluoronitrobenzene + methylamine
# (21 atoms). The TS guess is the relaxed-scan peak (data/processed/poc_smoke/lu_0/scan.xyz);
# geomeTRIC transition=True seeded by the analytic Hessian relaxes it to a first-order
# saddle. The energy / imaginary frequency are pinned from the GPU backend's own converged
# result as a regression guard -- the Psi4 barrier (delta-G-dagger) parity cross-check is
# DEFERRED (Stage D was scoped "code only, defer the Psi4 reference").
TS_ENERGY_HARTREE = -631.3593
TS_IMAG_FREQ_CM = -294.1
TS_GUESS = [
    ("O", (3.2164, -0.52209, 0.88584)),
    ("N", (2.53687, 0.42334, 0.53757)),
    ("O", (2.98019, 1.51219, 0.22974)),
    ("C", (1.11587, 0.24705, 0.50234)),
    ("C", (0.57153, -0.98675, 0.85759)),
    ("C", (-0.79903, -1.15746, 0.8525)),
    ("C", (-1.5472, -0.07031, 0.47644)),
    ("F", (-3.10711, -0.66545, -0.78939)),
    ("C", (-1.06748, 1.15825, 0.09723)),
    ("C", (0.30509, 1.31119, 0.1086)),
    ("H", (1.23191, -1.79623, 1.12923)),
    ("H", (-1.24735, -2.10496, 1.0966)),
    ("H", (-1.71808, 1.95222, -0.22633)),
    ("H", (0.76358, 2.24262, -0.18698)),
    ("C", (-3.01921, 0.62454, 2.88657)),
    ("N", (-3.25129, 0.09257, 1.5646)),
    ("H", (-2.56236, 1.61008, 2.80595)),
    ("H", (-3.94752, 0.71447, 3.45799)),
    ("H", (-2.33817, -0.03091, 3.42793)),
    ("H", (-3.59288, -0.86127, 1.5303)),
    ("H", (-3.80627, 0.67548, 0.94765)),
]


def _ts_guess() -> Atoms:
    """The SNAr scan-peak TS guess for lu_0 (neutral, closed shell)."""
    atoms = Atoms(symbols=[s for s, _ in TS_GUESS], positions=[p for _, p in TS_GUESS])
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


def _h2o() -> Atoms:
    """Near-equilibrium H2O (Angstrom), neutral closed shell. Order: O, H, H.

    Same geometry as tests/test_bond_orders.py so the GPU Mayer orders are compared to
    the very references the Psi4 adapter is pinned against.
    """
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


class _FakeSolventMF:
    """Mimics the gpu4pyscf mean-field solvation surface for ``_apply_solvent`` unit tests.

    ``mf.PCM()`` / ``mf.SMD()`` each return a fresh handle carrying a ``with_solvent``
    namespace whose attributes the calculator sets -- exactly the gpu4pyscf contract,
    without needing a CUDA device or an SCF.
    """

    def __init__(self, kind: str | None = None) -> None:
        self.kind = kind
        self.with_solvent = types.SimpleNamespace()

    def PCM(self) -> "_FakeSolventMF":
        return _FakeSolventMF(kind="pcm")

    def SMD(self) -> "_FakeSolventMF":
        return _FakeSolventMF(kind="smd")


def test_solvent_model_default_is_iefpcm():
    """The default continuum model matches the cpu_dmso Psi4 IEFPCM baseline."""
    assert GPU4PySCFCalculator(atoms=_nh3()).options["solvent_model"] == "iefpcm"


def test_apply_solvent_iefpcm_sets_method_and_eps():
    """The default (IEF-PCM) path sets the method and the solvent dielectric."""
    calc = GPU4PySCFCalculator(atoms=_nh3(), options={"solvent": "DMSO"})
    mf = calc._apply_solvent(_FakeSolventMF())
    assert mf.kind == "pcm"
    assert mf.with_solvent.method == "IEF-PCM"
    assert mf.with_solvent.eps == pytest.approx(46.826)


def test_apply_solvent_smd_sets_solvent_name():
    """SMD (a model the Psi4 path lacks) sets the gpu4pyscf SMD solvent key."""
    calc = GPU4PySCFCalculator(
        atoms=_nh3(), options={"solvent": "DMSO", "solvent_model": "smd"}
    )
    mf = calc._apply_solvent(_FakeSolventMF())
    assert mf.kind == "smd"
    assert mf.with_solvent.solvent == "dimethylsulfoxide"


def test_apply_solvent_unknown_model_raises():
    """An unrecognised solvent_model fails loudly rather than silently going gas."""
    calc = GPU4PySCFCalculator(
        atoms=_nh3(), options={"solvent": "DMSO", "solvent_model": "bogus"}
    )
    with pytest.raises(ValueError, match="solvent_model"):
        calc._apply_solvent(_FakeSolventMF())


def test_apply_solvent_unknown_solvent_raises():
    """A PCM solvent with no tabulated dielectric raises rather than defaulting to water."""
    calc = GPU4PySCFCalculator(atoms=_nh3(), options={"solvent": "unobtanium"})
    with pytest.raises(ValueError, match="dielectric"):
        calc._apply_solvent(_FakeSolventMF())


def test_ts_sets_saddle_flags(monkeypatch):
    """ts() requests an opt + saddle search (opt=ts=True, freq off), then runs."""
    calc = GPU4PySCFCalculator(atoms=_nh3())
    monkeypatch.setattr(calc, "run_calc", lambda *a, **k: 0.0)
    calc.ts()
    assert calc.options["opt"] is True
    assert calc.options["ts"] is True
    assert calc.options["freq"] is False


def test_ts_freq_sets_saddle_and_freq_flags(monkeypatch):
    """ts_freq() requests opt + saddle + a validating frequency run."""
    calc = GPU4PySCFCalculator(atoms=_nh3())
    monkeypatch.setattr(calc, "run_calc", lambda *a, **k: 0.0)
    calc.ts_freq()
    assert calc.options["opt"] is True
    assert calc.options["ts"] is True
    assert calc.options["freq"] is True


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
def test_dmso_solvation_shifts_single_point_energy():
    """IEF-PCM and SMD each stabilise the solute below gas, and the two models differ.

    The implicit continuum lowers the electronic energy of a polar solute (H2O) versus
    gas phase; IEF-PCM and SMD are different physics, so they must not coincide. This is
    the end-to-end on-device check that ``_apply_solvent`` actually engages the SCF (the
    unit tests above only verify the option wiring).
    """
    gas = GPU4PySCFCalculator(atoms=_h2o()).single_point()
    pcm = GPU4PySCFCalculator(atoms=_h2o(), options={"solvent": "DMSO"}).single_point()
    smd = GPU4PySCFCalculator(
        atoms=_h2o(), options={"solvent": "DMSO", "solvent_model": "smd"}
    ).single_point()

    assert math.isfinite(pcm) and math.isfinite(smd)
    assert pcm < gas  # reaction field stabilises the polar solute
    assert abs(smd - pcm) > 1e-4  # distinct continuum models -> distinct energies


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


@pytest.mark.slow
@requires_gpu_device
def test_snar_ts_search_one_imaginary_mode():
    """geomeTRIC TS search (analytic-Hessian seed) finds the SNAr saddle: one imag mode.

    Stage D, and the masterplan's flagged risk -- geomeTRIC must converge the SNAr saddle
    that optking's FD-Hessian path located on Psi4. EXPENSIVE (~24 min on the RTX 3050 Ti,
    measured 2026-06-24: the saddle search's per-step analytic Hessian seed + the saddle
    steps + the validating Hessian on 21 atoms). Pinned to the GPU backend's own converged
    result as a regression guard; the Psi4 barrier-parity cross-check is deferred (Stage D
    scoped code-only).
    """
    calc = GPU4PySCFCalculator(atoms=_ts_guess())
    calc.ts_freq()

    # A clean first-order saddle: exactly one imaginary mode, the SNAr reaction coordinate.
    imag = [f for f in calc.frequencies if f < 0.0]
    assert len(imag) == 1
    assert abs(imag[0] - TS_IMAG_FREQ_CM) < 20.0
    assert abs(calc.energy - TS_ENERGY_HARTREE) < 1e-3
    assert math.isfinite(calc.free_energy)  # thermochemistry captured for the barrier

    # The genuine reaction mode (|nu| well above the 100 cm^-1 soft cutoff) is kept out of
    # the qRRHO sum, not folded back as a soft rotor.
    from snar_qc.qc.thermo import Psi4Thermo

    thermo = Psi4Thermo.from_calculator(calc)
    assert len(thermo.imaginary_frequencies) == 1
    assert math.isfinite(thermo.gibbs_qh)


@pytest.mark.slow
@requires_gpu_device
def test_pyscf_bond_orders_mayer_matches_textbook():
    """PyscfBondOrders reproduces Mayer orders from a GPU mean-field (vs the Psi4 pins).

    This is what lets the relaxed-scan peak validation run GPU-native: the GPU calculator
    exposes a ``mean_field`` (no Psi4 ``Wavefunction``), and PyscfBondOrders turns it into
    the same 1-indexed Mayer ``get_bo`` surface ``validate_peaks`` consumes. Same
    references as the Psi4 adapter (tests/test_bond_orders.py): H2O O-H ~ 1.0 single bond
    (non-bonded H...H ~ 0), N2 ~ 3.0 triple bond -- so the 0.05/0.5 validate_peaks
    thresholds read the same bond-order scale on either backend.
    """
    from snar_qc.qc.bond_orders import PyscfBondOrders, bond_orders_from_calculator

    water = GPU4PySCFCalculator(atoms=_h2o())
    water.single_point()
    bo = PyscfBondOrders(water.mean_field)
    assert bo.kind == "mayer"

    oh_1 = bo.get_bo(1, 2)  # O-H
    oh_2 = bo.get_bo(1, 3)  # O-H (equivalent by symmetry)
    hh = bo.get_bo(2, 3)  # H...H, non-bonded
    assert 0.9 < oh_1 < 1.1  # single bond ~ 1.0 (Psi4 pin 1.0104)
    assert bo.get_bo(2, 1) == oh_1  # 1-indexed, symmetric -- NBOParser convention
    assert abs(oh_1 - oh_2) < 1e-6
    assert hh < 0.1 and oh_1 > hh

    # The backend factory returns this adapter for a GPU calculator.
    assert isinstance(bond_orders_from_calculator(water), PyscfBondOrders)

    nitrogen = GPU4PySCFCalculator(atoms=_n2(1.10))
    nitrogen.single_point()
    nn = PyscfBondOrders(nitrogen.mean_field).get_bo(1, 2)
    assert nn > 2.5  # triple bond
