"""Tests for snar_qc.ts.psi4_tsscan.Psi4TSScan.

``test_init_swaps_dft_calculator`` is a fast, no-SCF check that the DFT calculator is
swapped to Psi4 while the xTB scan and reactive-atom wiring are inherited unchanged.
``test_run_sps_read_sp_output`` feeds a tiny fixture (N2 at three bond lengths) straight
through ``run_sps`` / ``read_sp_output`` -- bypassing the xTB scan -- to exercise the
synchronous Psi4 single points and the Psi4 bond-order handoff. It runs three real Psi4
SCFs (~a few seconds) and is marked slow.
"""

import math

import pytest
from ase import Atoms

import predict_snar.config as predict_snar_config
from predict_snar.calculators import XTBCalculator
from snar_qc.qc.psi4_calculator import Psi4Calculator
from snar_qc.ts.psi4_tsscan import Psi4TSScan


def _n2(distance: float = 1.10) -> Atoms:
    """N2 (Angstrom) at a given bond length, neutral closed shell."""
    atoms = Atoms(symbols=["N", "N"], positions=[(0.0, 0.0, 0.0), (0.0, 0.0, distance)])
    atoms.info["charge"] = 0
    return atoms


# General options the base TSScan.__init__ reads (central/nu/lg atom indices,
# 1-indexed). For N2 the "central" and the bonded partner are atoms 1 and 2.
_GENERAL_OPTIONS = {"central_atom": 1, "nu_atom": 2, "lg_atom": 2}


def test_init_swaps_dft_calculator(monkeypatch):
    """__init__ replaces the G16 DFT calculator with Psi4 and inherits the rest."""
    # Base TSScan.__init__ reads config.general_info["azide_nucleophile"]; provide it.
    monkeypatch.setattr(
        predict_snar_config, "general_info", {"azide_nucleophile": False}
    )

    scan = Psi4TSScan(
        _n2(1.10), xtb_options={}, dft_options={}, general_options=_GENERAL_OPTIONS
    )

    # DFT calculator swapped to Psi4; self.g16 rebound to the same Psi4 backend.
    assert isinstance(scan.dft, Psi4Calculator)
    assert scan.g16 is scan.dft
    # xTB scan reused unchanged; reactive atoms wired identically to the base.
    assert isinstance(scan.xtb, XTBCalculator)
    assert (scan.central_atom, scan.nu_atom, scan.lg_atom) == (1, 2, 2)


@pytest.mark.slow
def test_run_sps_read_sp_output(tmp_path, monkeypatch):
    """run_sps + read_sp_output produce finite normalized energies and Psi4 BOs."""
    # Temp dir so Psi4 outputs (sps/*.out, timer.dat) never touch the repo.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        predict_snar_config, "general_info", {"azide_nucleophile": False}
    )

    scan = Psi4TSScan(
        _n2(1.10), xtb_options={}, dft_options={}, general_options=_GENERAL_OPTIONS
    )

    # Bypass the (inherited) xTB scan: hand in three pre-made geometries directly,
    # as read_scan_output would have populated them.
    scan.geometries = [_n2(1.05), _n2(1.10), _n2(1.20)]

    scan.run_sps(n_procs=2, mem=2)
    scan.read_sp_output()

    # Energies: one per geometry, finite, first referenced to 0.0 (kcal/mol).
    assert len(scan.dft_energies) == 3
    assert scan.dft_energies[0] == 0.0
    assert all(math.isfinite(energy) for energy in scan.dft_energies)
    # Compressing N2 from 1.10 to 1.05 raises the energy: E(1.10) < E(1.05) = 0.0.
    assert scan.dft_energies[1] < scan.dft_energies[0]

    # Bond orders: one Psi4BondOrders per scan point, sane N2 triple-bond values,
    # and the NBOParser-compatible 1-indexed / symmetric get_bo.
    assert len(scan.nbo_data) == 3
    for parser in scan.nbo_data:
        bond_order = parser.get_bo(scan.central_atom, scan.nu_atom)
        assert math.isfinite(bond_order)
        assert 1.0 < bond_order < 4.0
        assert parser.get_bo(scan.nu_atom, scan.central_atom) == bond_order
