"""gpu4pyscf calculator for snar_qc -- the GPU backend (opt-in).

``GPU4PySCFCalculator`` is the GPU sibling of
:class:`snar_qc.qc.psi4_calculator.Psi4Calculator`: it subclasses the vendored
:class:`predict_snar.calculators.Calculator`, exposes the same public surface
(``single_point`` / ``opt`` / ``opt_freq`` / ``freq`` dispatch via ``run_calc``), and
runs the same method -- **B3LYP-D3BJ / def2-SVP** with density fitting -- returning the
energy in Hartree. Energy parity with Psi4 at this level was shown to 3x10^-7 Ha on the
5-ring reference complex (``notes/2026-06-23_gpu_hessian_benchmark.md``).

**Scope: gas-phase single point (A), minimisation (B), frequencies (C), TS search (D).**
``single_point`` runs an SCF; ``opt`` minimises with geomeTRIC on GPU gradients; ``freq``
/ ``opt_freq`` build the **analytic** Hessian and harmonic thermochemistry; ``ts`` /
``ts_freq`` locate a saddle (geomeTRIC ``transition=True`` seeded by the analytic Hessian).
Only the PCM **solvation** path still raises ``NotImplementedError`` (gated to the
solvation-revalidation plan). The Psi4 backend remains the default and the fallback for
everything (CPU hosts, large substrates, anything exceeding the 4 GB VRAM ceiling).

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

import numpy as np
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
    # Coordinate system for a minimisation (the opt path). geomeTRIC's native TRIC,
    # not the Psi4 path's "cartesian": optking's *redundant internals* went degenerate
    # on rigid planar aryl nitriles (near-linear C#N), forcing Psi4 onto Cartesians,
    # but geomeTRIC's translation-rotation internal coordinates converge that exact
    # hard case cleanly and in fewer steps (verified on N#Cc1ccc(F)s1: TRIC 10 vs
    # Cartesian 20, same minimum). Accepts geomeTRIC coordsys names; "cartesian" is
    # aliased to "cart" for cross-backend option parity.
    "min_opt_coordinates": "tric",
    "geom_maxiter": 150,  # geomeTRIC step cap (mirrors the Psi4 optking cap)
}


class GPU4PySCFCalculator(Calculator):
    """Run B3LYP-D3BJ/def2-SVP single points, opts, frequencies, and TS searches on gpu4pyscf.

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
        free_energy / enthalpy / zpve / frequencies: Harmonic Gibbs / enthalpy / ZPVE
            (Hartree) and the signed cm^-1 frequency list, populated by a frequency run
            (``freq`` / ``opt_freq``); ``None`` for single points / opts.
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

    # -- minimisation / transition-state search ---------------------------------

    # Cross-backend aliases onto geomeTRIC coordinate-system names.
    _COORDSYS_ALIASES = {"cartesian": "cart", "internal": "tric"}

    def _geometric_coordsys(self) -> str:
        """Resolve the configured ``min_opt_coordinates`` to a geomeTRIC coordsys name."""
        raw = (self.options.get("min_opt_coordinates") or "tric").strip().lower()
        return self._COORDSYS_ALIASES.get(raw, raw)

    def _run_min_opt(self, mf: Any) -> "gto.Mole":
        """Minimise the geometry with geomeTRIC on gpu4pyscf gradients.

        gpu4pyscf has no ``geometric_solver``; pyscf's drives the optimisation and calls
        the gpu4pyscf gradient (``mf.nuc_grad_method()`` -> a GPU ``Gradients``) each
        step, so the gradient evaluations run on the GPU. Returns the optimised ``Mole``.
        """
        from pyscf.geomopt.geometric_solver import optimize  # noqa: PLC0415 -- lazy

        maxsteps = int(self.options.get("geom_maxiter") or 150)
        return optimize(mf, maxsteps=maxsteps, coordsys=self._geometric_coordsys())

    def _run_ts_opt(self, mf: Any) -> "gto.Mole":
        """Locate a transition state with geomeTRIC, seeded by the analytic Hessian.

        ``transition=True`` runs a saddle search; ``hessian="first"`` builds gpu4pyscf's
        **analytic** Hessian at the first step to identify the reaction mode (not a
        finite-difference seed) and BFGS-updates it through the search -- the
        hours->minutes payoff over the Psi4 optking + FD-Hessian TS path. geomeTRIC's TRIC
        internals locate the SNAr saddle cleanly (verified on the smoke substrate: one
        imaginary mode). Returns the optimised ``Mole``; the subsequent ``freq`` validates
        the single imaginary mode.
        """
        from pyscf.geomopt.geometric_solver import optimize  # noqa: PLC0415 -- lazy

        maxsteps = int(self.options.get("geom_maxiter") or 150)
        return optimize(
            mf,
            transition=True,
            hessian="first",
            coordsys=self._geometric_coordsys(),
            maxsteps=maxsteps,
        )

    def _write_optimised_geometry(self, mol: "gto.Mole") -> None:
        """Write the optimised coordinates back onto ``self.atoms`` (Angstrom).

        The GPU analogue of reading the relaxed geometry off a Psi4 wavefunction. pyscf
        preserves atom order, so the row order matches ``self.atoms``.
        """
        if self.atoms is not None:
            self.atoms.set_positions(mol.atom_coords(unit="Angstrom"))

    # -- transition-state requests ----------------------------------------------

    def ts(self, *args: Any, **kwargs: Any) -> float:
        """Optimise a transition state (geomeTRIC saddle search, analytic-Hessian seed).

        Mirrors :meth:`Psi4Calculator.ts` -- the inherited ``single_point`` / ``opt`` /
        ``opt_freq`` / ``freq`` all force ``ts`` off, so a TS request needs its own entry
        point. The Hessian for the saddle search is gpu4pyscf's analytic one (see
        :meth:`_run_ts_opt`), not a finite difference.
        """
        self.options["opt"] = True
        self.options["ts"] = True
        self.options["freq"] = False
        return self.run_calc(*args, **kwargs)

    def ts_freq(self, *args: Any, **kwargs: Any) -> float:
        """Optimise a transition state and run a frequency analysis on it.

        Mirrors :meth:`Psi4Calculator.ts_freq`. The frequency run both validates the
        saddle (one imaginary mode) and captures the thermochemistry :mod:`snar_qc.qc.thermo`
        needs for ΔG‡.
        """
        self.options["opt"] = True
        self.options["ts"] = True
        self.options["freq"] = True
        return self.run_calc(*args, **kwargs)

    # -- driver -----------------------------------------------------------------

    def run_calc(self, n_procs: int = 1, mem: float = 2.0) -> float:
        """Run the gpu4pyscf calculation selected by the ``opt`` / ``freq`` flags.

        The inherited ``single_point`` / ``opt`` / ``opt_freq`` / ``freq`` (and this
        class's ``ts`` / ``ts_freq``) set those flags before calling this. Implemented:
        ``single_point`` (A), ``opt`` minimisation (B), ``freq`` / ``opt_freq`` --
        analytic-Hessian harmonic thermochemistry (C), and ``ts`` / ``ts_freq`` -- a
        geomeTRIC saddle search seeded by the analytic Hessian (D).

        Args:
            n_procs: Accepted for interface parity with ``Psi4Calculator``; gpu4pyscf
                manages its own GPU threading, so this is not used to set thread counts.
            mem: Accepted for interface parity; VRAM is governed by the device, not this
                budget. (The factory's probe enforces the VRAM floor.)

        Returns:
            The total energy in Hartree. Also stored on ``self.energy``; the converged
            mean-field object on ``self.mean_field``. For an ``opt`` the energy and
            mean-field are those of the relaxed geometry, which is written back onto
            ``self.atoms``.
        """
        # Fresh result each call: a reused calculator must not report stale thermo
        # (e.g. a single_point after a freq run leaves these None, as Psi4 does).
        self.free_energy = self.enthalpy = self.zpve = self.frequencies = None

        if self.options.get("solvent"):
            raise NotImplementedError(
                "GPU4PySCFCalculator solvation (PCM/SMD) is not implemented yet "
                "(gas phase only); it is gated through the solvation revalidation plan. "
                "Use the Psi4 backend for solvent runs."
            )

        mol = self._build_mol()
        mf = self._build_mean_field(mol)

        if self.options.get("opt"):
            # Relax to the stationary point (a minimum, or a saddle for a TS request),
            # then re-build the mean-field on the optimised geometry so self.energy /
            # self.mean_field describe the converged stationary point cleanly.
            if self.options.get("ts"):
                mol = self._run_ts_opt(mf)
            else:
                mol = self._run_min_opt(mf)
            self._write_optimised_geometry(mol)
            mf = self._build_mean_field(mol)

        energy = float(mf.kernel())
        self.energy = energy
        self.mean_field = mf

        if self.options.get("freq"):
            self._capture_thermo(mf, mol)

        return self.energy

    # -- frequency / thermochemistry --------------------------------------------

    def _capture_thermo(self, mf: Any, mol: "gto.Mole") -> None:
        """Populate signed frequencies + harmonic thermochemistry from the analytic Hessian.

        Mirrors ``Psi4Calculator._capture_thermo`` so ``snar_qc.qc.thermo`` consumes
        either backend unchanged:

        - ``self.frequencies`` -- signed real vibrational wavenumbers (cm^-1) with
          imaginary modes carried as **negative** numbers (cclib/Gaussian convention), so
          a transition state shows exactly one negative entry.
        - ``free_energy`` / ``enthalpy`` / ``zpve`` -- harmonic Gibbs / enthalpy / ZPVE
          (Hartree, 298.15 K / 1 atm) over the **real positive** modes only. Imaginary
          modes are excluded from the partition function exactly as Psi4's thermo does;
          soft imaginary modes are folded back from the frequency list downstream by
          ``Psi4Thermo.from_calculator``.

        The Hessian is gpu4pyscf's **analytic** one-shot (the structural win over Psi4's
        finite-difference Hessian), and includes the D3BJ second-derivative because the
        dispersion is set on ``mf`` (``mf.disp``).
        """
        from pyscf.hessian import thermo as pyscf_thermo  # noqa: PLC0415 -- lazy

        hess = mf.Hessian().kernel()
        harmonic = pyscf_thermo.harmonic_analysis(mol, hess)
        freq_wavenumber = np.atleast_1d(harmonic["freq_wavenumber"])
        freq_au = np.atleast_1d(harmonic["freq_au"])

        # Signed cm^-1, plus a mask of the real positive modes for the partition function.
        signed: list[float] = []
        real_positive: list[bool] = []
        for w in freq_wavenumber:
            val = complex(w)
            if (
                abs(val.imag) > 1e-6
            ):  # imaginary mode: pyscf carries it in the imag part
                signed.append(-abs(val.imag))
                real_positive.append(False)
            else:
                signed.append(float(val.real))
                real_positive.append(val.real > 0.0)
        self.frequencies = signed

        mask = np.array(real_positive, dtype=bool)
        th = pyscf_thermo.thermo(mf, np.real(freq_au)[mask], 298.15, 101325)
        self.free_energy = float(th["G_tot"][0])
        self.enthalpy = float(th["H_tot"][0])
        self.zpve = float(th["ZPE"][0])

    # -- device memory ----------------------------------------------------------

    def free_device_memory(self) -> None:
        """Release cached device memory so a long sequence of GPU jobs holds VRAM flat.

        cupy / gpu4pyscf cache freed device blocks in a memory pool rather than returning
        them to the driver, so the *driver-visible* free VRAM (what
        :func:`snar_qc.qc.backend.probe_gpu` checks) drifts down across a run of single
        points -- the relaxed-scan DFT points are ~14 in a row -- and eventually trips
        ``GPUUnavailableError`` mid-scan on the 4 GB card. Dropping the retained
        ``mean_field`` (its device arrays are otherwise "in use", so the pool keeps them)
        and freeing the pools' unused blocks after a point is consumed keeps free VRAM
        flat. ``free_all_blocks`` only releases blocks not currently in use, so a caller
        that has already read what it needs off ``mean_field`` (energy, bond orders,
        thermochemistry) loses nothing. A no-op-safe call: the orchestrators invoke it
        duck-typed (``getattr``), so the Psi4 path -- which has no such method -- skips it.
        """
        self.mean_field = None
        import cupy  # noqa: PLC0415 -- lazy: optional [gpu] dependency

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
