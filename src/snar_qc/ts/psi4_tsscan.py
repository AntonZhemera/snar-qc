"""Psi4-backed relaxed-scan transition-state search for snar_qc.

:class:`Psi4TSScan` subclasses predict-snar's :class:`predict_snar.calculators.TSScan`
(the chosen xTB-native relaxed-scan route) and overrides **only** what the synchronous
Psi4 contract forces. The xTB relaxed scan itself is reused verbatim (``run_scan`` /
``read_scan_output`` are inherited), so are ``find_peaks`` / ``validate_peaks`` /
``make_plot`` and the constraint helpers. What changes:

* the DFT single points along the scan run through :class:`snar_qc.qc.psi4_calculator.
  Psi4Calculator` (B3LYP-D3BJ/def2-SVP) instead of Gaussian 16, and
* the peak-validation bond orders come from :class:`snar_qc.qc.bond_orders.
  Psi4BondOrders` (Mayer) instead of Gaussian NBO Wiberg indices.

Sync vs. async
--------------
``G16Calculator.run_calc`` is *asynchronous*: ``TSScan.run_sps`` launches one Gaussian
job per scan geometry via ``subprocess.Popen`` (joblib-parallel) and ``read_sp_output``
later parses the ``sps/*.log`` files. ``Psi4Calculator.run_calc`` is *synchronous*: it
returns the energy (Hartree) in-process and stores the wavefunction. So :meth:`run_sps`
runs each single point in-process and **stashes** the energies and wavefunctions on the
instance, and :meth:`read_sp_output` consumes those stashes (no log files, no wait-loop)
to populate :attr:`dft_energies` (kcal/mol, referenced to the first scan point) and
:attr:`nbo_data`. The scan-level handoff is unchanged: the caller still does
``scan.run_scan(...).wait()`` (xTB Popen) then ``read_scan_output()`` before ``run_sps``.

DFT level (Stage 2 scope)
-------------------------
The Psi4 single points use the ``Psi4Calculator`` default level -- B3LYP-D3BJ/def2-SVP,
the same level predict-snar uses for its DFT single points. The Gaussian-flavoured
``dft_options`` from the config (e.g. ``dispersion="gd3bj"``, a Gaussian solvent name,
``solvation_model`` for PCM) are **not** forwarded: their names do not map onto Psi4's
method string and PCM solvation is out of Stage 2 scope. They are kept on
:attr:`dft_options` for traceability; translating them (and wiring PCM) is deferred.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import TYPE_CHECKING, Any

import ase.io

from predict_snar.calculators import TSScan
from predict_snar.data import HARTREE_TO_KCAL

from snar_qc.qc.bond_orders import Psi4BondOrders
from snar_qc.qc.psi4_calculator import Psi4Calculator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ase import Atoms


class Psi4TSScan(TSScan):
    """Relaxed-scan TS search with an xTB scan and Psi4 (sync) DFT single points.

    Args:
        atoms: ASE ``Atoms`` for the reaction complex (carries ``info["charge"]``).
        xtb_options: Options for the xTB scan (passed through to the base class).
        dft_options: Gaussian-flavoured DFT options from the config. Retained on
            :attr:`dft_options` for traceability but not forwarded to Psi4 in Stage 2
            (see the module docstring).
        general_options: Carries ``central_atom`` / ``nu_atom`` / ``lg_atom`` (handled
            identically to the base class).

    Attributes:
        dft: The Psi4 DFT calculator template (B3LYP-D3BJ/def2-SVP).
        dft_options: The raw Gaussian-flavoured DFT options (unused in Stage 2).
        dft_energies: DFT energies along the scan (kcal/mol, first point = 0.0).
        nbo_data: ``Psi4BondOrders`` per scan point (Mayer), in scan order.
    """

    def __init__(
        self,
        atoms: "Atoms",
        xtb_options: dict[str, Any],
        dft_options: dict[str, Any],
        general_options: dict[str, Any],
    ) -> None:
        # Reuse the base set-up verbatim: the xTB relaxed scan (constraints, force
        # constants, azide angle handling), the central/nu/lg atom indices and the
        # result containers. This also assigns ``self.g16 = G16Calculator(...)``.
        super().__init__(atoms, xtb_options, dft_options, general_options)

        # Swap the Gaussian DFT calculator for the synchronous Psi4 one. ``self.g16``
        # is rebound as an alias so any inherited reference resolves to the Psi4
        # backend; the only method that used it (``run_sps``) is overridden below.
        # The G16-specific SP flags the base set (int_acc / scf_acc / nbo / chk) are
        # intentionally dropped -- Psi4 always returns the wavefunction and bond
        # orders come from oeprop, so there is no NBO/accuracy flag to honour.
        self.dft = Psi4Calculator(atoms)
        self.g16 = self.dft
        self.dft_options = dft_options

        # In-memory stashes filled by run_sps and drained by read_sp_output, taking
        # the place of the Gaussian sps/*.log files in the synchronous Psi4 flow.
        self._sp_energies: list[float] = []
        self._sp_wavefunctions: list[Any] = []

    # xtb energies live in each scan frame's comment line: ``energy: <Eh> xtb: ...``.
    _SCAN_ENERGY_RE = re.compile(r"energy:\s*(-?\d+\.\d+)")

    def read_scan_output(self) -> None:
        """Read the xTB relaxed-scan geometries and energies (xtb-6.7.1 compatible).

        Overrides ``TSScan.read_scan_output``, which builds a ``predict_snar.parsers.
        XTBParser`` over ``xtb.out`` to recover the scan energies. That parser also
        scrapes the Wiberg-bond-order block, keyed on an output header
        (``"total WBO ... WBO to atom ..."``) that **xTB 6.7.1 no longer prints** (the
        block is now headed ``"largest (>0.10) Wiberg bond orders for each atom"``), so
        the parser leaves its ``bo_matrix`` unbound and raises ``UnboundLocalError``
        before returning any energy. The bond orders it would compute are unused on the
        Psi4 path anyway -- peak validation uses Psi4 Mayer orders (:meth:`read_sp_
        output`).

        This override sidesteps the parser entirely: it reads the geometries from the
        scan trajectory and the per-frame energy straight from each frame's comment line
        (xTB writes ``energy: <Eh>`` there), referencing energies to the first point in
        kcal/mol exactly as the base method does. The HOMO-LUMO gaps the base method
        also collected are not needed by the POC runner (which never calls
        :meth:`check_electronic_temperature`), so ``hl_gaps`` is left empty.
        """
        if os.path.isfile("xtbscan.log"):
            shutil.move("xtbscan.log", "scan.xyz")

        self.geometries = list(ase.io.iread("scan.xyz"))
        for geometry in self.geometries:
            geometry.info["charge"] = self.atoms.info["charge"]

        energies = self._read_scan_energies("scan.xyz")
        self.xtb_energies = [
            (energy - energies[0]) * HARTREE_TO_KCAL for energy in energies
        ]
        self.hl_gaps = []

    @classmethod
    def _read_scan_energies(cls, path: str) -> list[float]:
        """Extract per-frame energies (Hartree) from an xTB scan xyz trajectory.

        Each frame in an xTB scan trajectory has a comment line of the form
        ``energy: -38.160514552507 xtb: 6.7.1 (...)``; this pulls the float out of the
        comment line of every frame (the line after each atom-count line).

        Args:
            path: Path to the scan trajectory (``scan.xyz``).

        Returns:
            The per-frame energies in Hartree, in trajectory order.

        Raises:
            ValueError: If a frame's comment line carries no parseable energy.
        """
        energies: list[float] = []
        with open(path) as handle:
            lines = handle.readlines()
        index = 0
        while index < len(lines):
            n_atoms = int(lines[index].strip())
            comment = lines[index + 1]
            match = cls._SCAN_ENERGY_RE.search(comment)
            if match is None:
                raise ValueError(
                    f"No energy in scan-frame comment line: {comment.strip()!r}"
                )
            energies.append(float(match.group(1)))
            index += n_atoms + 2
        return energies

    def run_sps(self, n_procs: int, mem: float) -> None:
        """Run a synchronous Psi4 single point for each scan geometry.

        Each point is computed in-process (no ``Popen``, no ``single_point_job``
        wait-loop). The energy (Hartree) and the Psi4 wavefunction are stashed on the
        instance for :meth:`read_sp_output`. Unlike the Gaussian path, which spreads
        ``n_procs`` across simultaneous jobs with joblib, the points run sequentially
        and each Psi4 single point is given the full ``n_procs`` (as SCF threads) and
        ``mem``.

        Args:
            n_procs: Number of threads handed to each Psi4 single point.
            mem: Memory budget (GB) per single point.
        """
        # Mirror the base class's per-geometry output directory so Psi4's output
        # files stay tidy; exist_ok keeps a re-run from blowing up on the dir.
        os.makedirs("sps", exist_ok=True)

        energies: list[float] = []
        wavefunctions: list[Any] = []
        for counter, geometry in enumerate(self.geometries, start=1):
            calc = Psi4Calculator(atoms=geometry, file=f"sps/{counter}.in")
            energy = calc.single_point(n_procs=n_procs, mem=mem)
            energies.append(energy)
            wavefunctions.append(calc.wavefunction)

        self._sp_energies = energies
        self._sp_wavefunctions = wavefunctions

    def read_sp_output(self) -> None:
        """Populate ``dft_energies`` and ``nbo_data`` from the stashed Psi4 results.

        Energies are converted Hartree -> kcal/mol and referenced to the first scan
        point (first point = 0.0), matching ``TSScan.read_sp_output``'s normalization.
        ``nbo_data`` is filled with ``Psi4BondOrders`` (Mayer) per scan point, so the
        inherited ``find_peaks`` / ``validate_peaks`` work unchanged.
        """
        # Hartree -> kcal/mol, then reference to the first scan point.
        dft_energies = [energy * HARTREE_TO_KCAL for energy in self._sp_energies]
        normalized_energies = [energy - dft_energies[0] for energy in dft_energies]
        self.dft_energies = normalized_energies

        # Bond orders straight off each stashed wavefunction (Mayer via oeprop).
        self.nbo_data = [
            Psi4BondOrders(wavefunction) for wavefunction in self._sp_wavefunctions
        ]
