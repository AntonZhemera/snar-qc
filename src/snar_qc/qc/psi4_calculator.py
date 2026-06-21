"""Psi4 calculator for snar_qc.

``Psi4Calculator`` is a Psi4 (Python-API) backend for the vendored predict-snar
calculator framework. It subclasses :class:`predict_snar.calculators.Calculator`,
so the inherited ``single_point`` / ``opt`` / ``opt_freq`` / ``freq`` methods work
unchanged -- each sets the ``opt`` / ``freq`` / ``ts`` flags on ``self.options`` and
then dispatches to :meth:`Psi4Calculator.run_calc`, exactly as predict-snar's
``G16Calculator`` does for Gaussian 16.

The method is **B3LYP-D3BJ / def2-SVP** (functional ``b3lyp``, dispersion ``d3bj``,
basis ``def2-svp``), the same DFT level predict-snar uses for its DFT single points.
Geometry and charge are read from the ASE ``Atoms`` object (``atoms`` and
``atoms.info["charge"]``); the spin multiplicity is assigned singlet / doublet by
electron-count parity, mirroring ``G16Calculator``'s rule.

Scope (Stage 1): single-point energies are the must-have. ``opt`` and ``freq`` are
thin wrappers over ``psi4.optimize`` / ``psi4.frequencies``. Solvation (PCM) and
parser-compatible output files are intentionally **not** wired here.

Scope (Stage 3): a frequency run (``freq`` / ``opt_freq``) now captures harmonic
thermochemistry off the Psi4 globals -- Gibbs free energy, enthalpy, ZPVE -- plus the
signed real vibrational frequencies (imaginary modes carried as negative numbers, the
cclib/Gaussian convention), so a transition state shows exactly one negative entry.
These feed :mod:`snar_qc.qc.thermo` (Grimme quasi-RRHO free energy and ΔG‡). A TS
optimisation path is wired too: when the inherited ``ts`` option is set on an opt, the
optimiser is driven through optking's ``OPT_TYPE TS`` (with ``FULL_HESS_EVERY`` so a
Hessian is built). Real TS convergence on substrates is exercised in Stage 4; here the
option is only wired and tested, not driven to convergence.

Divergence from the predict-snar engine contract worth noting for downstream wiring:
``G16Calculator.run_calc`` is *asynchronous* -- it launches Gaussian via
``subprocess.Popen`` and returns the process; the pipeline later waits on it and
parses energies from the log file. The Psi4 Python API is *synchronous*, so
:meth:`run_calc` runs the calculation in-process and returns the **energy in Hartree**
directly (also stored on ``self.energy``, with the Psi4 wavefunction on
``self.wavefunction``). Adapting this to predict-snar's async wait/parse loop in
``jobs.py`` is deferred to a later stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import psi4

from predict_snar.calculators import Calculator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ase import Atoms


# Default DFT level: B3LYP-D3BJ / def2-SVP (matches predict-snar's DFT single points).
_DEFAULT_OPTIONS: dict[str, Any] = {
    "functional": "b3lyp",
    "basis_set": "def2-svp",
    "dispersion": "d3bj",
    "charge": None,
    "opt": None,
    "freq": None,
    "ts": None,
    "scf_type": "df",
    "reference": None,  # resolved per-calculation from multiplicity (rks / uks)
    "e_convergence": 1e-8,
    "d_convergence": 1e-8,
    "geom_maxiter": 150,  # optking iteration cap (raised from Psi4's default 50)
}


class Psi4Calculator(Calculator):
    """Run B3LYP-D3BJ/def2-SVP calculations through the Psi4 Python API.

    Args:
        atoms: ASE ``Atoms`` object carrying the geometry (Angstrom) and, in
            ``atoms.info["charge"]``, the total molecular charge.
        file: Optional base name; used to name the Psi4 output file
            (``<prefix>.out``). Defaults to ``psi4.out``.
        options: Optional overrides merged onto the default options (e.g. a
            different ``functional`` / ``basis_set`` / ``dispersion``).

    Attributes:
        options: Calculation options (see ``_DEFAULT_OPTIONS``).
        energy: Energy (Hartree) of the most recent ``run_calc`` call, or ``None``.
        wavefunction: Psi4 ``Wavefunction`` from the most recent ``run_calc`` call.
        free_energy: Harmonic Gibbs free energy G (Hartree, 298.15 K / 1 atm ideal
            gas) captured from a frequency run; ``None`` for single points / opts.
        enthalpy: Total enthalpy H (Hartree) from a frequency run, else ``None``.
        zpve: Zero-point vibrational energy (Hartree) from a frequency run, else
            ``None``.
        frequencies: Signed real vibrational frequencies (cm^-1) from a frequency run.
            Imaginary modes are carried as **negative** numbers (cclib/Gaussian
            convention), so a transition state shows exactly one negative entry.
            ``None`` for single points / opts.
    """

    def __init__(
        self,
        atoms: Optional["Atoms"] = None,
        file: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(atoms, file)

        # Seed defaults, then apply any user overrides via the base-class helper.
        self.options = dict(_DEFAULT_OPTIONS)
        if options:
            self.set_options(options)

        # Charge: prefer an explicit option, else read it from the ASE Atoms info.
        if self.options.get("charge") is None and atoms is not None:
            self.options["charge"] = atoms.info.get("charge", 0)

        # Output file name, mirroring G16Calculator's prefix handling.
        if file:
            prefix = file.rsplit(".", 1)[0]
            self.output = f"{prefix}.out"
        else:
            self.file = "psi4.in"
            self.output = "psi4.out"

        self.energy: Optional[float] = None
        self.wavefunction: Any = None

        # Thermochemistry, populated only by a frequency run (freq / opt_freq).
        self.free_energy: Optional[float] = None
        self.enthalpy: Optional[float] = None
        self.zpve: Optional[float] = None
        self.frequencies: Optional[list[float]] = None

    # -- geometry / spin helpers ------------------------------------------------

    def _multiplicity(self, charge: int) -> int:
        """Singlet for an even electron count, doublet for odd.

        Mirrors ``G16Calculator`` (``n_electrons = sum(Z) + charge``); only the
        parity matters, so adding vs. subtracting the charge is equivalent here.
        """
        n_electrons = int(sum(self.atoms.numbers)) + int(charge)
        return 1 if n_electrons % 2 == 0 else 2

    def _build_molecule(self) -> "psi4.core.Molecule":
        """Construct a Psi4 molecule from the ASE Atoms geometry.

        ``no_com`` / ``no_reorient`` / ``symmetry c1`` keep Psi4's atom order and
        coordinate frame aligned with the input, which matters when the resulting
        wavefunction is reused (e.g. for bond orders) downstream.
        """
        if self.atoms is None:
            raise ValueError(
                "Psi4Calculator requires an ASE Atoms geometry (got None)."
            )

        charge = int(self.options.get("charge") or 0)
        mult = self._multiplicity(charge)

        lines = [f"{charge} {mult}"]
        for atom in self.atoms:
            x, y, z = atom.position
            lines.append(f"{atom.symbol} {x:.10f} {y:.10f} {z:.10f}")
        lines += ["units angstrom", "symmetry c1", "no_reorient", "no_com"]

        return psi4.geometry("\n".join(lines))

    def _method_string(self) -> str:
        """Assemble the ``method[-dispersion]/basis`` string for the Psi4 driver."""
        functional = self.options["functional"]
        dispersion = self.options.get("dispersion")
        method = f"{functional}-{dispersion}" if dispersion else functional
        return f"{method}/{self.options['basis_set']}"

    # -- transition-state requests ----------------------------------------------

    def ts(self, *args: Any, **kwargs: Any) -> float:
        """Optimise a transition state (opt driven through optking ``OPT_TYPE TS``).

        Mirrors :meth:`predict_snar.calculators.G16Calculator.ts` -- the base
        ``single_point`` / ``opt`` / ``opt_freq`` / ``freq`` all force ``ts`` off, so a
        TS request needs its own entry point. The Hessian for the saddle search is
        handled by ``run_calc``'s TS branch (Psi4 ``FULL_HESS_EVERY``), not a Gaussian
        ``calc_fc`` keyword.
        """
        self.options["opt"] = True
        self.options["ts"] = True
        self.options["freq"] = False
        return self.run_calc(*args, **kwargs)

    def ts_freq(self, *args: Any, **kwargs: Any) -> float:
        """Optimise a transition state and run a subsequent frequency calculation.

        Mirrors :meth:`predict_snar.calculators.G16Calculator.ts_freq`. The frequency
        run both validates the saddle point (one imaginary mode) and captures the
        thermochemistry :mod:`snar_qc.qc.thermo` needs for ΔG‡.
        """
        self.options["opt"] = True
        self.options["ts"] = True
        self.options["freq"] = True
        return self.run_calc(*args, **kwargs)

    # -- driver -----------------------------------------------------------------

    def run_calc(self, n_procs: int = 1, mem: float = 2.0) -> float:
        """Run the Psi4 calculation selected by the ``opt`` / ``freq`` flags.

        The inherited ``single_point`` / ``opt`` / ``opt_freq`` / ``freq`` set those
        flags before calling this method.

        Args:
            n_procs: Number of threads for Psi4.
            mem: Memory budget in GB.

        Returns:
            The total energy in Hartree. Also stored on ``self.energy``; the Psi4
            wavefunction is stored on ``self.wavefunction``.
        """
        do_opt = bool(self.options.get("opt"))
        do_freq = bool(self.options.get("freq"))

        # Fresh Psi4 state for a deterministic, side-effect-free run.
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_memory(f"{mem} GB")
        psi4.set_num_threads(int(n_procs))
        psi4.core.set_output_file(self.output, False)

        molecule = self._build_molecule()
        reference = self.options.get("reference") or (
            "uks" if molecule.multiplicity() != 1 else "rks"
        )
        psi4.set_options(
            {
                "reference": reference,
                "scf_type": self.options["scf_type"],
                "e_convergence": self.options["e_convergence"],
                "d_convergence": self.options["d_convergence"],
            }
        )

        # Raise optking's iteration cap on any optimisation (Psi4 defaults to 50,
        # which is short for floppy or strained geometries).
        if do_opt and self.options.get("geom_maxiter"):
            psi4.set_options({"geom_maxiter": int(self.options["geom_maxiter"])})

        # Transition-state optimisation: optking's OPT_TYPE=TS, building a Hessian.
        # Only on the opt path; the normal MIN opt is left untouched.
        if do_opt and self.options.get("ts"):
            psi4.set_options({"opt_type": "TS", "full_hess_every": 0})

        method = self._method_string()
        if do_opt and do_freq:
            psi4.optimize(method, molecule=molecule)
            energy, wfn = _frequencies(method, molecule)
        elif do_opt:
            energy, wfn = psi4.optimize(method, molecule=molecule, return_wfn=True)
        elif do_freq:
            energy, wfn = _frequencies(method, molecule)
        else:
            energy, wfn = psi4.energy(method, molecule=molecule, return_wfn=True)

        # Capture thermochemistry from the freq run's Psi4 globals *now*: a later
        # psi4.core.clean() (next run_calc) would overwrite these scalars.
        if do_freq:
            self._capture_thermo(wfn)

        self.energy = float(energy)
        self.wavefunction = wfn
        return self.energy

    def _capture_thermo(self, wfn: Any) -> None:
        """Store harmonic thermochemistry and frequencies from a freq run.

        Reads the Gibbs free energy, enthalpy, and ZPVE psi4 globals set by the
        frequency calculation, and the signed real vibrational frequencies off
        ``wfn.frequency_analysis``. Imaginary modes are carried as negative numbers
        (cclib/Gaussian convention) so a transition state shows one negative entry.
        """
        self.free_energy = float(psi4.variable("GIBBS FREE ENERGY"))
        self.enthalpy = float(psi4.variable("ENTHALPY"))
        self.zpve = float(psi4.variable("ZPVE"))

        analysis = wfn.frequency_analysis
        omega = analysis["omega"].data
        trv = analysis["TRV"].data
        frequencies: list[float] = []
        for mode, kind in zip(omega, trv):
            if kind != "V":  # skip translations / rotations
                continue
            value = complex(mode)
            if abs(value.imag) > 1e-9:  # imaginary mode -> negative real number
                frequencies.append(-abs(value.imag))
            else:
                frequencies.append(float(value.real))
        self.frequencies = frequencies


def _frequencies(method: str, molecule: "psi4.core.Molecule") -> tuple[float, Any]:
    """Run a Psi4 frequency calculation, tolerant of the API alias name."""
    func = getattr(psi4, "frequencies", None) or psi4.frequency
    return func(method, molecule=molecule, return_wfn=True)
