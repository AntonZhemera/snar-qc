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

# Same NH3 single point but in DMSO via IEFPCM / Bondi radii (the solvation path),
# pinned from Psi4 1.10.2 + PCMSolver. PCM stabilises NH3 by ~4 kcal/mol vs gas phase,
# which is the headline assertion: the solvated energy is real, finite, and below the
# gas-phase reference.
NH3_DMSO_REFERENCE_HARTREE = -56.51698628


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


def _record_psi4_options(monkeypatch):
    """Stub psi4.set_options / psi4.optimize so run_calc drives no real SCF.

    Returns a dict merging every (lower-cased) option run_calc sets, so a test can
    assert on the optimiser type without running a calculation.
    """
    import psi4

    recorded: dict[str, str] = {}

    def _set_options(options):
        for key, value in options.items():
            recorded[key.lower()] = str(value).lower()

    monkeypatch.setattr(psi4, "set_options", _set_options)
    monkeypatch.setattr(psi4, "optimize", lambda *args, **kwargs: (-1.0, object()))
    return recorded


def _record_psi4_solvation(monkeypatch):
    """Stub psi4 so run_calc drives no SCF but records the PCM wiring.

    Returns ``(recorded, pcm_inputs)``: ``recorded`` merges every option run_calc set
    (so a test can assert ``pcm`` / ``pcm_scf_type``); ``pcm_inputs`` collects every
    PCMSolver block passed to ``psi4.pcm_helper`` (empty when no solvent is configured).
    """
    import psi4

    recorded: dict[str, str] = {}
    pcm_inputs: list[str] = []

    def _set_options(options):
        for key, value in options.items():
            recorded[key.lower()] = str(value).lower()

    monkeypatch.setattr(psi4, "set_options", _set_options)
    monkeypatch.setattr(psi4, "pcm_helper", lambda inp: pcm_inputs.append(inp))
    monkeypatch.setattr(psi4, "energy", lambda *a, **k: (-1.0, object()))
    return recorded, pcm_inputs


def test_solvent_configures_pcm(tmp_path, monkeypatch):
    """A configured solvent loads a PCMSolver block and switches on the PCM SCF."""
    monkeypatch.chdir(tmp_path)
    recorded, pcm_inputs = _record_psi4_solvation(monkeypatch)

    calc = Psi4Calculator(atoms=_nh3(), options={"solvent": "DMSO"})
    calc.single_point()

    assert recorded.get("pcm") == "true"
    assert recorded.get("pcm_scf_type") == "total"
    assert len(pcm_inputs) == 1
    block = pcm_inputs[0]
    assert "Solvent = DMSO" in block
    assert "SolverType = IEFPCM" in block  # default solver
    assert "RadiiSet = Bondi" in block  # default radii


def test_no_solvent_stays_gas_phase(tmp_path, monkeypatch):
    """With no solvent the PCM path is never touched (gas phase, as before)."""
    monkeypatch.chdir(tmp_path)
    recorded, pcm_inputs = _record_psi4_solvation(monkeypatch)

    calc = Psi4Calculator(atoms=_nh3())  # solvent defaults to None
    calc.single_point()

    assert pcm_inputs == []
    assert "pcm" not in recorded


def test_pcm_solver_and_radii_overrides(tmp_path, monkeypatch):
    """Non-default ``pcm_solver`` / ``pcm_radii`` flow into the PCMSolver block."""
    monkeypatch.chdir(tmp_path)
    _, pcm_inputs = _record_psi4_solvation(monkeypatch)

    calc = Psi4Calculator(
        atoms=_nh3(),
        options={"solvent": "Water", "pcm_solver": "CPCM", "pcm_radii": "UFF"},
    )
    calc.single_point()

    block = pcm_inputs[0]
    assert "SolverType = CPCM" in block
    assert "Solvent = Water" in block
    assert "RadiiSet = UFF" in block


def test_ts_request_drives_optking_ts(tmp_path, monkeypatch):
    """``ts()`` sets opt+ts and drives optking's OPT_TYPE=TS with a Hessian (no SCF)."""
    monkeypatch.chdir(tmp_path)
    recorded = _record_psi4_options(monkeypatch)

    calc = Psi4Calculator(atoms=_nh3())
    calc.ts()

    assert calc.options["opt"] is True
    assert calc.options["ts"] is True
    assert calc.options["freq"] is False
    assert recorded.get("opt_type") == "ts"
    assert "full_hess_every" in recorded


def test_min_opt_does_not_request_ts(tmp_path, monkeypatch):
    """A normal ``opt()`` leaves the optimiser at its default minimum search."""
    monkeypatch.chdir(tmp_path)
    recorded = _record_psi4_options(monkeypatch)

    calc = Psi4Calculator(atoms=_nh3())
    calc.opt()

    assert calc.options["ts"] is False
    assert "opt_type" not in recorded


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
    # A single point captures no thermochemistry.
    assert calc.free_energy is None
    assert calc.frequencies is None


@pytest.mark.slow
def test_nh3_dmso_pcm_single_point_energy(tmp_path, monkeypatch):
    """Psi4 B3LYP-D3BJ/def2-SVP single point on NH3 in DMSO (IEFPCM) is on reference.

    Exercises the full PCM path end to end (real PCMSolver SCF) and checks both the
    pinned solvated energy and that PCM lowers the energy relative to the gas-phase
    reference (continuum stabilisation of a polar solute).
    """
    monkeypatch.chdir(tmp_path)

    calc = Psi4Calculator(atoms=_nh3(), options={"solvent": "DMSO"})
    energy = calc.single_point()

    assert math.isfinite(energy)
    assert abs(energy - NH3_DMSO_REFERENCE_HARTREE) < ENERGY_TOL_HARTREE
    assert energy < NH3_REFERENCE_HARTREE  # solvation stabilises the molecule
