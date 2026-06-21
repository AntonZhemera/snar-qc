"""Tests for snar_qc.qc.thermo (Grimme qRRHO free energies and ΔG‡).

The quasi-harmonic correction and the barrier arithmetic are pure functions, tested
fast (no SCF). One slow test runs a real Psi4 opt+freq on H2O and pins the harmonic
Gibbs free energy, mirroring the pinned-reference style of test_psi4_calculator.py.
"""

import math

import pytest
from ase import Atoms

from predict_snar.data import HARTREE_TO_KCAL
from snar_qc.qc.psi4_calculator import Psi4Calculator
from snar_qc.qc.thermo import (
    Psi4Thermo,
    activation_free_energy,
    grimme_qh_gibbs,
)

# Harmonic Gibbs free energy of H2O at B3LYP-D3BJ/def2-SVP (opt+freq), pinned from
# Psi4 1.10.2. Tolerance absorbs optimiser/threading noise while staying far tighter
# than any method/basis regression.
H2O_GIBBS_REFERENCE_HARTREE = -76.35539418
GIBBS_TOL_HARTREE = 2e-3


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


# -- grimme_qh_gibbs (pure, no SCF) --------------------------------------------


def test_high_frequencies_leave_gibbs_unchanged():
    """With no low modes the qRRHO correction is negligible (weights ~ 1)."""
    g = -100.0
    out = grimme_qh_gibbs(g, [2000.0, 3000.0])
    assert abs(out - g) < 1e-7  # measured ~7e-10 Ha


def test_low_frequency_raises_gibbs_by_sub_kcal():
    """A floppy 30 cm^-1 mode caps the harmonic entropy, so G_qh > G_harmonic."""
    g = -100.0
    out = grimme_qh_gibbs(g, [30.0, 1500.0, 3000.0])
    diff_kcal = (out - g) * HARTREE_TO_KCAL
    assert out > g  # quasi-harmonic raises G (less spurious low-mode entropy)
    assert 0.0 < diff_kcal < 1.0  # measured ~0.53 kcal/mol for one 30 cm^-1 mode


def test_imaginary_zero_and_negative_modes_are_dropped():
    """Imaginary (negative), zero and non-positive modes are ignored, not summed."""
    g = -100.0
    with_junk = grimme_qh_gibbs(g, [-500.0, 0.0, -0.1, 2000.0])
    real_only = grimme_qh_gibbs(g, [2000.0])
    assert with_junk == real_only


def test_temperature_and_cutoff_are_tunable():
    """Both knobs change the result for a low mode (raising cutoff folds more modes)."""
    g = -100.0
    base = grimme_qh_gibbs(g, [80.0], temperature=298.15, cutoff=100.0)
    hotter = grimme_qh_gibbs(g, [80.0], temperature=400.0, cutoff=100.0)
    wider_cutoff = grimme_qh_gibbs(g, [80.0], temperature=298.15, cutoff=200.0)
    assert hotter != base
    assert wider_cutoff != base


# -- Psi4Thermo (pure construction) --------------------------------------------


def test_psi4thermo_computes_gibbs_qh_and_flags_imaginary():
    """Construction computes gibbs_qh and exposes the imaginary (negative) modes."""
    thermo = Psi4Thermo(
        electronic_energy=-100.0,
        gibbs=-99.0,
        gibbs_qh=grimme_qh_gibbs(-99.0, [-1200.0, 50.0, 1800.0]),
        enthalpy=-98.5,
        zpve=0.05,
        frequencies=[-1200.0, 50.0, 1800.0],
    )
    assert thermo.imaginary_frequencies == [-1200.0]
    assert math.isfinite(thermo.gibbs_qh)


def test_from_calculator_without_freq_run_raises():
    """A calculator that never ran a frequency calc cannot yield thermochemistry."""
    calc = Psi4Calculator(atoms=_h2o())  # no run_calc -> frequencies/free_energy None
    with pytest.raises(ValueError):
        Psi4Thermo.from_calculator(calc)


# -- activation_free_energy (pure) ---------------------------------------------


def test_activation_free_energy_from_floats():
    """Raw Hartree floats: ΔG‡ = (G_ts - sum G_ref) * HARTREE_TO_KCAL."""
    dg = activation_free_energy(-100.0, -100.5)
    assert math.isclose(dg, 0.5 * HARTREE_TO_KCAL, rel_tol=0, abs_tol=1e-9)


def test_activation_free_energy_selects_attribute_and_sums_references():
    """``which`` picks the attribute; multiple references are summed."""
    ts = Psi4Thermo(
        -100.0,
        gibbs=-100.0,
        gibbs_qh=-100.1,
        enthalpy=-100.2,
        zpve=0.0,
        frequencies=[1800.0],
    )
    ref_a = Psi4Thermo(
        -60.0,
        gibbs=-60.0,
        gibbs_qh=-60.05,
        enthalpy=-60.1,
        zpve=0.0,
        frequencies=[1800.0],
    )
    ref_b = Psi4Thermo(
        -40.0,
        gibbs=-40.0,
        gibbs_qh=-40.05,
        enthalpy=-40.1,
        zpve=0.0,
        frequencies=[1800.0],
    )

    dg_qh = activation_free_energy(ts, ref_a, ref_b, which="gibbs_qh")
    expected_qh = (-100.1 - (-60.05 + -40.05)) * HARTREE_TO_KCAL
    assert math.isclose(dg_qh, expected_qh, rel_tol=0, abs_tol=1e-9)

    dg_harm = activation_free_energy(ts, ref_a, ref_b, which="gibbs")
    expected_harm = (-100.0 - (-60.0 + -40.0)) * HARTREE_TO_KCAL
    assert math.isclose(dg_harm, expected_harm, rel_tol=0, abs_tol=1e-9)


def test_activation_free_energy_requires_a_reference():
    """A barrier needs something to measure the TS against."""
    with pytest.raises(ValueError):
        activation_free_energy(-100.0)


# -- real Psi4 thermochemistry (slow) ------------------------------------------


@pytest.mark.slow
def test_h2o_opt_freq_thermochemistry(tmp_path, monkeypatch):
    """Real H2O opt+freq -> Psi4Thermo: finite, 3 real modes, pinned harmonic G."""
    # Temp dir so Psi4 scratch (psi4.out, timer.dat) never touches the repo.
    monkeypatch.chdir(tmp_path)

    calc = Psi4Calculator(atoms=_h2o())
    calc.opt_freq()
    thermo = Psi4Thermo.from_calculator(calc)

    assert math.isfinite(thermo.gibbs)
    assert math.isfinite(thermo.gibbs_qh)
    assert thermo.zpve > 0.0

    # H2O is a minimum: 3N-6 = 3 real vibrations, none imaginary.
    assert len([nu for nu in thermo.frequencies if nu > 0.0]) == 3
    assert thermo.imaginary_frequencies == []

    # No low modes, so the quasi-harmonic correction is negligible.
    assert abs(thermo.gibbs_qh - thermo.gibbs) < 5e-4

    # Ordering: electronic < Gibbs < enthalpy (G = H - TS, S > 0; +ZPE over E_elec).
    assert thermo.electronic_energy < thermo.gibbs < thermo.enthalpy

    # Pinned harmonic Gibbs free energy.
    assert abs(thermo.gibbs - H2O_GIBBS_REFERENCE_HARTREE) < GIBBS_TOL_HARTREE
