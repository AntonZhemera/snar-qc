"""Tests for the deterministic helpers of snar_qc.poc.barrier.

The heavy ``compute_barrier`` chain (xTB scan + Psi4 DFT + TS opt/freq) is exercised by
the Stage 4 smoke run, not here. These fast tests pin the pure logic: imaginary-mode
counting, rate-determining-peak selection, and the result's JSON round-trip.
"""

import json
from collections import namedtuple

from snar_qc.poc.barrier import BarrierResult, count_imaginary, select_peak_index

# Minimal stand-in for predict_snar.calculators.Peak (only the fields we read).
_Peak = namedtuple("Peak", ["maximum", "energy"])


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
