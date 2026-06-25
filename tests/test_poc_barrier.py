"""Tests for the deterministic helpers of snar_qc.poc.barrier.

The heavy ``compute_barrier`` chain (xTB scan + Psi4 DFT + TS opt/freq) is exercised by
the Stage 4 smoke run, not here. These fast tests pin the pure logic: imaginary-mode
counting, rate-determining-peak selection, and the result's JSON round-trip.
"""

import json
import os
from collections import namedtuple

import pytest

import snar_qc.poc.barrier as barrier
from snar_qc.poc.barrier import BarrierResult, count_imaginary, select_peak_index

# Minimal stand-in for predict_snar.calculators.Peak (only the fields we read).
_Peak = namedtuple("Peak", ["maximum", "energy"])


class _FakeAtoms:
    """Minimal ASE-Atoms stand-in: a charge and a constant bond distance."""

    info = {"charge": 0}

    def get_distance(self, i, j):  # noqa: D401 - stub
        return 1.5


class _FakeRC:
    """Minimal ReactionComplex stand-in for the scan-construction branch tests."""

    atoms = _FakeAtoms()
    aryl_halide_smiles = "Fc1ccccc1"
    amine_smiles = "CN"
    leaving_group = "F"
    central_atom = 1
    nu_atom = 8
    lg_atom = 2


class _RecordingScan:
    """Psi4TSScan stand-in that records the scans built, then halts the chain.

    ``read_scan_output`` raises so ``compute_barrier`` stops right after the scan is
    constructed (the failure is caught and recorded as status 'error'); the test then
    inspects how many scans/constraints were added and the solvent passed through.
    """

    instances: list["_RecordingScan"] = []

    def __init__(
        self,
        atoms,
        xtb_options,
        dft_options,
        general_options,
        solvent=None,
        solvent_model=None,
    ):
        self.solvent = solvent
        self.solvent_model = solvent_model
        self.scans: list[tuple] = []
        self.constraints: list[tuple] = []
        _RecordingScan.instances.append(self)

    def constrain_bond(self, a, b, c):
        self.constraints.append((a, b))

    def add_scan(self, *args):
        self.scans.append(args)

    def run_scan(self, n_procs=2):
        class _Proc:
            def wait(self_inner):
                return None

        return _Proc()

    def read_scan_output(self):
        raise RuntimeError("stop-after-scan")


class _FakeScan:
    """A scan stub exposing only the ``peaks`` attribute select_peak_index reads."""

    def __init__(self, peaks):
        self.peaks = peaks


def test_count_imaginary_handles_none_and_signs():
    """None -> 0; a minimum -> 0; one negative mode -> 1; two -> 2."""
    assert count_imaginary(None) == 0
    assert count_imaginary([]) == 0
    assert count_imaginary([12.0, 340.0, 1500.0]) == 0
    assert count_imaginary([-450.0, 200.0, 900.0]) == 1
    assert count_imaginary([-450.0, -30.0, 900.0]) == 2


def test_count_significant_imaginary_ignores_soft_modes():
    """Only imaginary modes at/above the cutoff count; soft sub-cutoff ones don't."""
    from snar_qc.poc.barrier import (
        TS_SOFT_IMAG_CUTOFF_CM,
        count_significant_imaginary,
    )

    assert TS_SOFT_IMAG_CUTOFF_CM == 100.0
    assert count_significant_imaginary(None) == 0
    # A clean minimum / clean saddle.
    assert count_significant_imaginary([12.0, 340.0, 1500.0]) == 0
    assert count_significant_imaginary([-450.0, 200.0, 900.0]) == 1
    # The real 5-ring case: one reaction mode (-143) + one soft rotor (-70) -> 1.
    assert count_significant_imaginary([-143.3, -70.1, 86.9, 200.0]) == 1
    # Two genuine imaginaries -> a real higher-order saddle.
    assert count_significant_imaginary([-450.0, -260.0, 900.0]) == 2
    # Cutoff is tunable.
    assert count_significant_imaginary([-70.1, 200.0], cutoff=50.0) == 1


def test_select_peak_index_picks_highest_energy_peak():
    """The rate-determining peak is the highest-energy surviving maximum."""
    scan = _FakeScan([_Peak(3, 5.0), _Peak(7, 12.0), _Peak(5, 9.0)])
    assert select_peak_index(scan) == 7


def test_select_peak_index_none_when_no_peaks():
    """No surviving peak (empty or absent) -> None (status becomes 'no_peak')."""
    assert select_peak_index(_FakeScan([])) is None
    assert select_peak_index(_FakeScan(None)) is None


def test_unknown_coordinate_is_an_error(monkeypatch):
    """An unrecognised coordinate fails fast with a clear error, no QC attempted."""
    monkeypatch.setattr(barrier, "Psi4TSScan", _RecordingScan)
    _RecordingScan.instances.clear()
    result = barrier.compute_barrier(_FakeRC(), coordinate="diagonal")
    assert result.status == "error"
    assert "coordinate" in (result.error or "")
    assert _RecordingScan.instances == []  # never even built a scan


def test_concerted_coordinate_builds_two_scans(monkeypatch):
    """The concerted coordinate drives both the forming and breaking bonds."""
    monkeypatch.setattr(barrier, "Psi4TSScan", _RecordingScan)
    _RecordingScan.instances.clear()
    result = barrier.compute_barrier(
        _FakeRC(), solvent="DMSO", solvent_model="smd", coordinate="concerted"
    )
    scan = _RecordingScan.instances[-1]
    assert len(scan.scans) == 2  # C...Nu and C-LG
    assert len(scan.constraints) == 2
    assert scan.solvent == "DMSO"  # solvent threaded into the scan
    assert scan.solvent_model == "smd"  # model threaded into the scan
    assert result.coordinate == "concerted"
    assert result.solvent == "DMSO"
    assert result.solvent_model == "smd"  # recorded on the result for provenance


def test_addition_coordinate_builds_one_scan(monkeypatch):
    """The addition coordinate drives only the forming C...Nu bond (C-LG left intact)."""
    monkeypatch.setattr(barrier, "Psi4TSScan", _RecordingScan)
    _RecordingScan.instances.clear()
    result = barrier.compute_barrier(_FakeRC(), solvent="DMSO", coordinate="addition")
    scan = _RecordingScan.instances[-1]
    assert len(scan.scans) == 1  # only C...Nu
    assert len(scan.constraints) == 1
    assert result.coordinate == "addition"


def test_optimised_atoms_reads_gpu_atoms_when_no_wavefunction():
    """GPU backend (no wavefunction): the optimised geometry is read off ``calc.atoms``.

    The gpu4pyscf calculator writes the relaxed coordinates back onto ``calc.atoms`` and
    exposes ``mean_field`` (not ``wavefunction``), so ``_optimised_atoms`` must dispatch
    on the absent wavefunction and return a copy of those atoms carrying the charge.
    """
    from ase import Atoms

    relaxed = Atoms("HH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])

    class _FakeGPUCalc:
        wavefunction = None
        atoms = relaxed
        options = {"charge": 0}

    opt = barrier._optimised_atoms(_FakeGPUCalc())
    assert list(opt.get_chemical_symbols()) == ["H", "H"]
    assert opt.info["charge"] == 0
    assert opt is not relaxed  # a copy, not the live atoms


def test_barrier_result_records_coordinate_and_solvent_defaults():
    """Defaults: gas-phase (solvent None) concerted, surviving the JSON round trip."""
    result = BarrierResult(
        aryl_halide_smiles="Fc1ccccc1",
        amine_smiles="CN",
        leaving_group="F",
        central_atom=1,
        nu_atom=8,
        lg_atom=2,
    )
    restored = json.loads(json.dumps(result.to_dict()))
    assert restored["coordinate"] == "concerted"
    assert restored["solvent"] is None
    assert restored["solvent_model"] is None


def test_barrier_result_json_round_trip():
    """BarrierResult.to_dict is JSON-serialisable and preserves the key fields."""
    result = BarrierResult(
        aryl_halide_smiles="O=[N+]([O-])c1ccc(F)cc1",
        amine_smiles="CN",
        leaving_group="F",
        central_atom=7,
        nu_atom=16,
        lg_atom=8,
        status="completed",
        lu_id=42,
        delta_g_qh_kcal=21.3,
        n_imag_ts=1,
    )
    blob = json.dumps(result.to_dict())
    restored = json.loads(blob)
    assert restored["status"] == "completed"
    assert restored["lu_id"] == 42
    assert restored["delta_g_qh_kcal"] == 21.3
    assert restored["n_imag_ts"] == 1
    assert restored["scan_dft_energies_kcal"] == []


# -- gas cache + solvent sweep (geometry-persistence reuse path) ------------------


def test_geometry_persist_round_trip(tmp_path):
    """_persist_geometry -> _read_geometry preserves symbols, positions, and charge."""
    from ase import Atoms

    atoms = Atoms("HF", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.92)])
    atoms.info["charge"] = -1
    path = tmp_path / "sp.xyz"
    barrier._persist_geometry(atoms, str(path))
    back = barrier._read_geometry(str(path))

    assert list(back.get_chemical_symbols()) == ["H", "F"]
    assert back.info["charge"] == -1
    assert back.get_positions()[1][2] == pytest.approx(0.92)


def _thermo(e, gqh):
    """A Psi4Thermo with electronic + gibbs_qh set (others tracked alongside)."""
    from snar_qc.qc.thermo import Psi4Thermo

    return Psi4Thermo(
        electronic_energy=e, gibbs=gqh, gibbs_qh=gqh, enthalpy=gqh, zpve=0.0, frequencies=[]
    )


def _write_gas_cache_fixture(tmp_path):
    """Build a minimal gas cache (gas_thermo.json + 3 geometries) in tmp_path."""
    from ase import Atoms

    # Distinct atom counts so the fake SP can return a per-species energy.
    geoms = {
        "ts": Atoms("H3", positions=[(0, 0, 0), (0, 0, 0.7), (0, 0.7, 0)]),
        "arx": Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.7)]),
        "amine": Atoms("H", positions=[(0, 0, 0)]),
    }
    for a in geoms.values():
        a.info["charge"] = 0
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for key, a in geoms.items():
            barrier._persist_geometry(a, barrier._GEOMETRY_FILES[key])
        result = BarrierResult(
            aryl_halide_smiles="Fc1ccccc1",
            amine_smiles="CN",
            leaving_group="F",
            central_atom=1,
            nu_atom=8,
            lg_atom=2,
            lu_id=7,
            n_imag_ts=1,
            n_imag_ts_soft=0,
        )
        gas_thermos = {
            "ts": _thermo(-100.5, -100.4),
            "arx": _thermo(-10.3, -10.25),
            "amine": _thermo(-1.2, -1.15),
        }
        barrier._write_gas_cache(result, gas_thermos)
    finally:
        os.chdir(cwd)


def test_write_gas_cache_structure(tmp_path):
    """_write_gas_cache emits a self-contained cache with per-species thermo + geometry."""
    _write_gas_cache_fixture(tmp_path)
    cache = json.loads((tmp_path / barrier._GAS_CACHE_FILE).read_text())

    assert cache["lu_id"] == 7
    assert cache["n_imag_ts"] == 1
    assert set(cache["species"]) == {"ts", "arx", "amine"}
    assert cache["species"]["ts"]["geometry"] == "ts_opt.xyz"
    assert cache["species"]["ts"]["electronic_energy"] == -100.5
    assert cache["species"]["amine"]["gibbs_qh"] == -1.15


def test_solvent_sweep_reuses_gas_cache(tmp_path, monkeypatch):
    """solvent_sweep recombines ΔG‡ from the cache + 3 solvent SPs, no gas recompute."""
    from predict_snar.data import HARTREE_TO_KCAL

    _write_gas_cache_fixture(tmp_path)

    # Fake SP: return a per-species solvated electronic energy keyed by atom count.
    solv_e = {3: -100.0, 2: -10.0, 1: -1.0}
    calls = {"n": 0}

    class _FakeSPCalc:
        def __init__(self, atoms):
            self.atoms = atoms

        def single_point(self, n_procs=1, mem=1.0):
            calls["n"] += 1
            return solv_e[len(self.atoms)]

    monkeypatch.setattr(
        barrier, "make_calculator", lambda atoms, file=None, options=None: _FakeSPCalc(atoms)
    )

    outdir = tmp_path / "out"
    outdir.mkdir()
    cwd = os.getcwd()
    os.chdir(outdir)
    try:
        result = barrier.solvent_sweep(tmp_path, "DMSO", "iefpcm")
    finally:
        os.chdir(cwd)

    assert calls["n"] == 3  # exactly one SP per species -- the gas backbone is reused
    assert result.status == "completed"
    assert result.solvent == "DMSO"
    assert result.solvent_model == "iefpcm"
    assert result.lu_id == 7

    # shift_sp = E_solv - E_gas; G_qh_solv = G_qh_gas + shift.
    gqh = {  # G_qh_gas + (E_solv - E_gas)
        "ts": -100.4 + (-100.0 - -100.5),
        "arx": -10.25 + (-10.0 - -10.3),
        "amine": -1.15 + (-1.0 - -1.2),
    }
    expected = (gqh["ts"] - gqh["arx"] - gqh["amine"]) * HARTREE_TO_KCAL
    assert result.delta_g_qh_kcal == pytest.approx(expected)


def test_solvent_sweep_without_cache_raises(tmp_path):
    """A gas run predating geometry persistence (no cache) fails with a clear message."""
    with pytest.raises(FileNotFoundError, match="gas cache"):
        barrier.solvent_sweep(tmp_path, "DMSO", "iefpcm")
