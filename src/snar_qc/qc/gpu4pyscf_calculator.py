"""gpu4pyscf calculator for snar_qc -- the GPU backend (opt-in).

``GPU4PySCFCalculator`` is the GPU sibling of
:class:`snar_qc.qc.psi4_calculator.Psi4Calculator`: it subclasses the vendored
:class:`predict_snar.calculators.Calculator`, exposes the same public surface
(``single_point`` / ``opt`` / ``opt_freq`` / ``freq`` dispatch via ``run_calc``), and
runs the same method -- **B3LYP-D3BJ / def2-SVP** with density fitting -- returning the
energy in Hartree. Energy parity with Psi4 at this level was shown to 3x10^-7 Ha on the
5-ring reference complex (``notes/2026-06-23_gpu_hessian_benchmark.md``).

**Stage A scope: gas-phase single points only.** ``opt`` / ``freq`` / ``ts`` and the PCM
solvation path raise ``NotImplementedError`` here; they arrive in Stages B-E. The Psi4
backend remains the default and the fallback for everything (CPU hosts, large
substrates, anything exceeding the 4 GB VRAM ceiling).

**Import discipline (CPU-fallback contract).** This module imports ``pyscf`` /
``gpu4pyscf`` / ``cupy`` at *module* scope -- which is exactly why it must only ever be
imported lazily, by :func:`snar_qc.qc.backend.make_calculator`, after its device probe
succeeds. It is **not** imported at the ``snar_qc`` package top level, so ``import
snar_qc`` and the entire Psi4 path stay GPU-free on hosts without the ``[gpu]`` extra.
Do not add an eager import of this module anywhere.

Like ``Psi4Calculator``, this is **synchronous**: ``run_calc`` runs the calculation
in-process and returns the energy directly (the predict-snar ``G16Calculator`` contract
is async). Drive it from the snar_qc orchestrators, not the vendored ``jobs.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pyscf import gto
from gpu4pyscf.dft import rks as _gpu_rks
from gpu4pyscf.dft import uks as _gpu_uks

from predict_snar.calculators import Calculator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ase import Atoms


# Default DFT level: B3LYP-D3BJ / def2-SVP with density fitting -- mirrors
# ``Psi4Calculator``'s option surface so the two backends are interchangeable behind
# the factory. ``solvent`` is carried for surface parity but rejected in Stage A.
_DEFAULT_OPTIONS: dict[str, Any] = {
    "functional": "b3lyp",
    "basis_set": "def2-svp",
    "dispersion": "d3bj",
    "charge": None,
    "opt": None,
    "freq": None,
    "ts": None,
    "scf_type": "df",  # density fitting (parity with Psi4 scf_type="df")
    "reference": None,  # resolved per-calculation from multiplicity (rks / uks)
    "solvent": None,
}


class GPU4PySCFCalculator(Calculator):
    """Run B3LYP-D3BJ/def2-SVP single points through gpu4pyscf.

    Args:
        atoms: ASE ``Atoms`` carrying the geometry (Angstrom) and the total molecular
            charge in ``atoms.info["charge"]``.
        file: Optional base name (kept for interface parity with ``Psi4Calculator``;
            gpu4pyscf runs in-process and writes no output file).
        options: Optional overrides merged onto the defaults (e.g. a different
            ``functional`` / ``basis_set`` / ``dispersion``).

    Attributes:
        options: Calculation options (see ``_DEFAULT_OPTIONS``).
        energy: Energy (Hartree) of the most recent ``run_calc`` call, or ``None``.
        mean_field: The converged gpu4pyscf mean-field object from the most recent
            ``run_calc`` call (the GPU analogue of ``Psi4Calculator.wavefunction``).
        free_energy / enthalpy / zpve / frequencies: Thermochemistry placeholders for
            interface parity; populated only by the frequency path (Stage C), ``None``
            here.
    """

    def __init__(
        self,
        atoms: Optional["Atoms"] = None,
        file: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(atoms, file)

        self.options = dict(_DEFAULT_OPTIONS)
        if options:
            self.set_options(options)

        if self.options.get("charge") is None and atoms is not None:
            self.options["charge"] = atoms.info.get("charge", 0)

        self.energy: Optional[float] = None
        self.mean_field: Any = None

        # Thermochemistry -- populated by the frequency path in a later stage.
        self.free_energy: Optional[float] = None
        self.enthalpy: Optional[float] = None
        self.zpve: Optional[float] = None
        self.frequencies: Optional[list[float]] = None

    # -- geometry / spin helpers ------------------------------------------------

    def _multiplicity(self, charge: int) -> int:
        """Singlet for an even electron count, doublet for odd (as ``Psi4Calculator``)."""
        n_electrons = int(sum(self.atoms.numbers)) + int(charge)
        return 1 if n_electrons % 2 == 0 else 2

    def _build_mol(self) -> "gto.Mole":
        """Construct a pyscf ``Mole`` from the ASE Atoms geometry, charge, and spin."""
        if self.atoms is None:
            raise ValueError(
                "GPU4PySCFCalculator requires an ASE Atoms geometry (got None)."
            )

        charge = int(self.options.get("charge") or 0)
        mult = self._multiplicity(charge)

        atom_spec = [
            (
                atom.symbol,
                (
                    float(atom.position[0]),
                    float(atom.position[1]),
                    float(atom.position[2]),
                ),
            )
            for atom in self.atoms
        ]
        return gto.M(
            atom=atom_spec,
            unit="Angstrom",
            basis=self.options["basis_set"],
            charge=charge,
            spin=mult - 1,  # pyscf spin = number of unpaired electrons = 2S = mult - 1
        )

    def _build_mean_field(self, mol: "gto.Mole") -> Any:
        """Build the (RKS/UKS) mean-field object: functional, density fitting, D3BJ."""
        functional = self.options["functional"]
        reference = self.options.get("reference") or ("uks" if mol.spin else "rks")
        if reference == "uks":
            mf = _gpu_uks.UKS(mol, xc=functional)
        else:
            mf = _gpu_rks.RKS(mol, xc=functional)

        if (self.options.get("scf_type") or "df").lower() == "df":
            mf = mf.density_fit()

        dispersion = self.options.get("dispersion")
        if dispersion:
            mf.disp = dispersion  # gpu4pyscf D3BJ via the dftd3 integration

        return mf

    # -- driver -----------------------------------------------------------------

    def run_calc(self, n_procs: int = 1, mem: float = 2.0) -> float:
        """Run the gpu4pyscf calculation selected by the ``opt`` / ``freq`` flags.

        The inherited ``single_point`` sets ``opt=freq=ts=False`` before calling this.
        Stage A implements the single-point path only.

        Args:
            n_procs: Accepted for interface parity with ``Psi4Calculator``; gpu4pyscf
                manages its own GPU threading, so this is not used to set thread counts.
            mem: Accepted for interface parity; VRAM is governed by the device, not this
                budget. (The factory's probe enforces the VRAM floor.)

        Returns:
            The total energy in Hartree. Also stored on ``self.energy``; the mean-field
            object is stored on ``self.mean_field``.
        """
        if self.options.get("solvent"):
            raise NotImplementedError(
                "GPU4PySCFCalculator solvation (PCM/SMD) is not in Stage A "
                "(gas-phase single points only); it is gated through the solvation "
                "revalidation plan. Use the Psi4 backend for solvent runs."
            )
        if any(self.options.get(flag) for flag in ("opt", "freq", "ts")):
            raise NotImplementedError(
                "GPU4PySCFCalculator supports single_point only in Stage A; "
                "opt / freq / ts arrive in Stages B-D. Use the Psi4 backend meanwhile."
            )

        mol = self._build_mol()
        mf = self._build_mean_field(mol)
        energy = float(mf.kernel())

        self.energy = energy
        self.mean_field = mf
        return self.energy
