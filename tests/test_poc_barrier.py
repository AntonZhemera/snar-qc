"""Tests for the deterministic helpers of snar_qc.poc.barrier.

The heavy ``compute_barrier`` chain (xTB scan + Psi4 DFT + TS opt/freq) is exercised by
the Stage 4 smoke run, not here. These fast tests pin the pure logic: imaginary-mode
counting, rate-determining-peak selection, and the result's JSON round-trip.
"""

import json
from collections import namedtuple

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

    def __init__(self, atoms, xtb_options, dft_options, general_options, solvent=None):
        self.solvent = solvent
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
    result = barrier.compute_barrier(_FakeRC(), solvent="DMSO", coordinate="concerted")
    scan = _RecordingScan.instances[-1]
    assert len(scan.scans) == 2  # C...Nu and C-LG
    assert len(scan.constraints) == 2
    assert scan.solvent == "DMSO"  # solvent threaded into the scan
    assert result.coordinate == "concerted"
    assert result.solvent == "DMSO"


def test_addition_coordinate_builds_one_scan(monkeypatch):
    """The addition coordinate drives only the forming C...Nu bond (C-LG left intact)."""
    monkeypatch.setattr(barrier, "Psi4TSScan", _RecordingScan)
    _RecordingScan.instances.clear()
    result = barrier.compute_barrier(_FakeRC(), solvent="DMSO", coordinate="addition")
    scan = _RecordingScan.instances[-1]
    assert len(scan.scans) == 1  # only C...Nu
    assert len(scan.constraints) == 1
    assert result.coordinate == "addition"


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
