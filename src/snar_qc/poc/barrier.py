"""Drive a reaction complex to a quasi-harmonic ΔG‡ with the Psi4 engine.

This is the Stage 4 glue that wires the Stage 1-3 pieces into one substrate run:

1. **Relaxed scan** of the forming C(ipso)...N(amine) bond with :class:`snar_qc.ts.
   psi4_tsscan.Psi4TSScan` -- an xTB relaxed scan, Psi4 (B3LYP-D3BJ/def2-SVP) DFT
   single points along it, and Psi4 Mayer bond orders for peak validation.
2. **Peak location** via the inherited ``find_peaks`` / ``validate_peaks`` -- the
   highest-energy validated maximum is taken as the S~N~Ar addition-TS guess.
3. **TS optimisation + frequencies** on that geometry with ``Psi4Calculator.ts_freq``
   (optking ``OPT_TYPE=TS`` + Hessian); the saddle is confirmed by exactly one
   imaginary frequency.
4. **Reference** -- the bare aryl halide and the bare amine are each optimised +
   frequency-analysed (``opt_freq``) to give the **separated-reactants** reference
   G(ArX) + G(amine).
5. **ΔG‡** -- ``activation_free_energy`` of the TS against that reference, on the Grimme
   quasi-harmonic Gibbs free energy.

Reference choice (decided in Stage 4a). The first attempt used a *reaction-complex*
reference -- a single supermolecule of aryl halide + amine, so the step would be
unimolecular and the standard-state term would cancel. In the gas phase that
pre-association complex is a floppy, orientation-dependent van-der-Waals minimum on a
flat surface that would not converge (``OptimizationConvergenceError``) even after the
expensive TS search. The **separated-reactants** reference is used instead: it is
numerically robust (a rigid aromatic and a tiny amine, both easy minima) and, for a
*ranking* validation, equivalent up to a constant -- the amine term and the bimolecular
1 atm -> 1 M standard-state correction are identical across every substrate, so they
shift all barriers together without changing the order. The standard-state correction
is therefore **not applied** (it is a constant; the POC weights ranking over absolute
magnitude); the resulting ΔG‡ includes the full (large, roughly constant) entropy of
association.

Failure is data, not an exception: :func:`compute_barrier` catches engine failures and
returns a :class:`BarrierResult` whose ``status`` records where the chain stopped
(``no_peak`` / ``ts_not_saddle`` / ``error`` ...), so a flaky TS search on one substrate
never sinks a batch.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import predict_snar.config as predict_snar_config

from snar_qc.qc.psi4_calculator import Psi4Calculator
from snar_qc.qc.thermo import Psi4Thermo, activation_free_energy
from snar_qc.ts.psi4_tsscan import Psi4TSScan

if TYPE_CHECKING:  # pragma: no cover - typing only
    from snar_qc.poc.complex import ReactionComplex

# Scan defaults. The S~N~Ar reaction coordinate is the *antisymmetric* combination
# d(C-Nu) - d(C-LG): a concerted relaxed scan that forms the C...Nu bond while breaking
# the C-LG bond. (Scanning only the forming bond traps the neutral amine in a
# zwitterionic adduct with no gas-phase saddle; driving both bond changes together
# traverses the concerted-S~N~Ar barrier.) The scan pulls C...Nu in to a bonded length
# and pushes C-LG out to a near-dissociated length over this many steps.
DEFAULT_SCAN_STOP = 1.45  # Angstrom, ~ a formed C-N single bond (forming bond end)
DEFAULT_SCAN_STOP_LG = 2.6  # Angstrom, ~ a near-broken C-LG bond (breaking bond end)
DEFAULT_SCAN_STEPS = 14


def count_imaginary(frequencies: Optional[list[float]]) -> int:
    """Number of imaginary (negative) vibrational modes.

    The engine carries imaginary modes as negative numbers (cclib/Gaussian convention),
    so a transition state has exactly one negative entry and a minimum has none.

    Args:
        frequencies: Signed vibrational frequencies (cm^-1), or ``None``.

    Returns:
        The count of negative frequencies (0 if ``frequencies`` is ``None``).
    """
    if not frequencies:
        return 0
    return sum(1 for nu in frequencies if nu < 0.0)


def select_peak_index(scan: Any) -> Optional[int]:
    """Index (into the scan geometries) of the rate-determining validated peak.

    After ``find_peaks`` / ``validate_peaks`` the surviving peaks are the reactive
    maxima; the highest-energy one is the rate-determining S~N~Ar addition-TS guess.

    Args:
        scan: A ``Psi4TSScan`` whose ``peaks`` have been found and validated.

    Returns:
        The ``maximum`` index of the highest-energy surviving peak, or ``None`` if no
        peak survived validation.
    """
    peaks = getattr(scan, "peaks", None)
    if not peaks:
        return None
    best = max(peaks, key=lambda peak: peak.energy)
    return int(best.maximum)


@dataclass
class BarrierResult:
    """Outcome of one substrate's ΔG‡ run (serialisable to a resumable sidecar).

    ``status`` is the single source of truth for whether the run reached a confirmed
    saddle point:

    - ``completed`` -- TS located with exactly one imaginary frequency; ΔG‡ available.
    - ``ts_not_saddle`` -- TS opt+freq finished but the imaginary-mode count != 1.
    - ``no_peak`` -- the relaxed scan produced no validated maximum (no TS guess).
    - ``error`` -- an engine call raised; see ``error``.

    All energies are kcal/mol; frequencies cm^-1; Hartree raw values kept for audit.
    """

    aryl_halide_smiles: str
    amine_smiles: str
    leaving_group: str
    central_atom: int
    nu_atom: int
    lg_atom: int
    status: str = "pending"
    stage: str = "init"
    lu_id: Optional[int] = None
    delta_g_qh_kcal: Optional[float] = None
    delta_g_kcal: Optional[float] = None
    delta_h_kcal: Optional[float] = None
    delta_e_kcal: Optional[float] = None
    n_imag_ts: Optional[int] = None
    n_imag_arx: Optional[int] = None
    n_imag_amine: Optional[int] = None
    ts_imag_freq_cm: Optional[float] = None
    ts_energy_hartree: Optional[float] = None
    arx_energy_hartree: Optional[float] = None
    amine_energy_hartree: Optional[float] = None
    ts_gibbs_qh_hartree: Optional[float] = None
    arx_gibbs_qh_hartree: Optional[float] = None
    amine_gibbs_qh_hartree: Optional[float] = None
    reference: str = "separated_reactants"
    peak_index: Optional[int] = None
    n_scan_points: Optional[int] = None
    scan_dft_energies_kcal: list[float] = field(default_factory=list)
    scan_xtb_energies_kcal: list[float] = field(default_factory=list)
    error: Optional[str] = None
    timing_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for JSON serialisation."""
        return asdict(self)


def _configure_engine() -> None:
    """Seed the predict-snar config the engine reads at run time.

    Two globals are needed for the xTB relaxed scan:

    - ``config.general_info["azide_nucleophile"]`` -- consulted by ``TSScan.__init__``;
      the POC amine nucleophiles are never azides, so the azide constraint path is off.
    - ``config.xtb`` -- the xTB binary the ``XTBCalculator`` shells out to. predict-snar
      sets this from an install directory; here we resolve it from ``PATH`` (the conda
      env's ``xtb``), falling back to the bare name.
    """
    info = getattr(predict_snar_config, "general_info", None)
    if not isinstance(info, dict):
        info = {}
    info.setdefault("azide_nucleophile", False)
    predict_snar_config.general_info = info

    if not getattr(predict_snar_config, "xtb", None):
        predict_snar_config.xtb = shutil.which("xtb") or "xtb"


def _species_thermo(
    atoms: Any, file: str, n_procs: int, mem: float
) -> tuple[Psi4Thermo, int, float, float]:
    """Optimise + frequency-analyse a reference species and bundle its thermochemistry.

    Args:
        atoms: ASE ``Atoms`` for the species (carries ``info["charge"]``).
        file: Psi4 output-file base name.
        n_procs: Threads for the Psi4 calculation.
        mem: Memory budget (GB).

    Returns:
        ``(thermo, n_imaginary, electronic_energy_hartree, gibbs_qh_hartree)``.
    """
    calc = Psi4Calculator(atoms, file=file)
    calc.opt_freq(n_procs=n_procs, mem=mem)
    thermo = Psi4Thermo.from_calculator(calc)
    return thermo, count_imaginary(calc.frequencies), calc.energy, thermo.gibbs_qh


def compute_barrier(
    rc: "ReactionComplex",
    *,
    scan_stop: float = DEFAULT_SCAN_STOP,
    scan_stop_lg: float = DEFAULT_SCAN_STOP_LG,
    scan_steps: int = DEFAULT_SCAN_STEPS,
    n_procs: int = 8,
    mem: float = 12.0,
    make_plot: bool = True,
    lu_id: Optional[int] = None,
) -> BarrierResult:
    """Compute ΔG‡ for one reaction complex, returning a status-carrying result.

    Runs the full chain (relaxed scan -> DFT SPs -> peak -> TS opt+freq -> reference
    opt+freq -> ΔG‡). Must be called with the current working directory set to a
    per-substrate scratch directory (the engine writes ``scan.xyz`` / ``sps/`` / Psi4
    output there).

    Args:
        rc: The reaction complex to run (carries geometry, charge, reactive indices).
        scan_stop: Final C...N (forming-bond) distance (Angstrom) for the scan.
        scan_stop_lg: Final C-LG (breaking-bond) distance (Angstrom) for the scan.
        scan_steps: Number of relaxed-scan steps (both bonds move concertedly).
        n_procs: Threads handed to each Psi4 calculation.
        mem: Memory budget (GB) per Psi4 calculation.
        make_plot: Whether to write the engine's scan plot (``GSM.png``).
        lu_id: Optional Lu_74 identifier, copied onto the result.

    Returns:
        A :class:`BarrierResult`; ``status == "completed"`` only for a confirmed saddle.
    """
    result = BarrierResult(
        aryl_halide_smiles=rc.aryl_halide_smiles,
        amine_smiles=rc.amine_smiles,
        leaving_group=rc.leaving_group,
        central_atom=rc.central_atom,
        nu_atom=rc.nu_atom,
        lg_atom=rc.lg_atom,
        lu_id=lu_id,
    )

    try:
        _configure_engine()
        general_options = {
            "central_atom": rc.central_atom,
            "nu_atom": rc.nu_atom,
            "lg_atom": rc.lg_atom,
        }

        # --- 1. concerted relaxed scan: form C...Nu while breaking C-LG --------
        result.stage = "scan"
        t0 = time.time()
        scan = Psi4TSScan(rc.atoms, {}, {}, general_options)
        start_nu = float(rc.atoms.get_distance(rc.central_atom - 1, rc.nu_atom - 1))
        start_lg = float(rc.atoms.get_distance(rc.central_atom - 1, rc.lg_atom - 1))
        # Two constraints + two scans; xTB's default concerted mode advances both bonds
        # together, tracing the antisymmetric d(C-Nu) - d(C-LG) coordinate.
        scan.constrain_bond(rc.central_atom, rc.nu_atom, "auto")
        scan.constrain_bond(rc.central_atom, rc.lg_atom, "auto")
        scan.add_scan(
            rc.central_atom, rc.nu_atom, "auto", start_nu, scan_stop, scan_steps
        )
        scan.add_scan(
            rc.central_atom, rc.lg_atom, "auto", start_lg, scan_stop_lg, scan_steps
        )
        scan.run_scan(n_procs=2).wait()
        scan.read_scan_output()
        result.timing_s["scan_xtb"] = time.time() - t0

        # --- 2. DFT single points + peak location ------------------------------
        result.stage = "dft_sps"
        t0 = time.time()
        scan.run_sps(n_procs=n_procs, mem=mem)
        scan.read_sp_output()
        scan.find_peaks()
        scan.validate_peaks(intermediate=True, threshold=0.5)
        if make_plot:
            try:
                scan.make_plot()
            except Exception:  # plotting is cosmetic; never fail a run on it
                pass
        result.timing_s["dft_sps"] = time.time() - t0
        result.n_scan_points = len(scan.geometries)
        result.scan_dft_energies_kcal = [float(e) for e in scan.dft_energies]
        result.scan_xtb_energies_kcal = [float(e) for e in scan.xtb_energies]

        peak_index = select_peak_index(scan)
        result.peak_index = peak_index
        if peak_index is None:
            result.status = "no_peak"
            return result

        # --- 3. TS optimisation + frequencies on the peak geometry -------------
        result.stage = "ts_opt_freq"
        t0 = time.time()
        ts_guess = scan.geometries[peak_index].copy()
        ts_guess.info["charge"] = rc.atoms.info["charge"]
        ts_calc = Psi4Calculator(ts_guess, file="ts.in")
        ts_calc.ts_freq(n_procs=n_procs, mem=mem)
        result.timing_s["ts_opt_freq"] = time.time() - t0

        n_imag = count_imaginary(ts_calc.frequencies)
        result.n_imag_ts = n_imag
        imag = [nu for nu in (ts_calc.frequencies or []) if nu < 0.0]
        result.ts_imag_freq_cm = float(min(imag)) if imag else None
        result.ts_energy_hartree = ts_calc.energy
        ts_thermo = Psi4Thermo.from_calculator(ts_calc)
        result.ts_gibbs_qh_hartree = ts_thermo.gibbs_qh

        # --- 4. separated-reactants reference: bare ArX + bare amine -----------
        from snar_qc.poc.complex import build_molecule

        result.stage = "arx_opt_freq"
        t0 = time.time()
        arx_atoms = build_molecule(rc.aryl_halide_smiles)
        arx_thermo, n_imag_arx, arx_e, arx_gqh = _species_thermo(
            arx_atoms, "arx.in", n_procs, mem
        )
        result.timing_s["arx_opt_freq"] = time.time() - t0
        result.n_imag_arx = n_imag_arx
        result.arx_energy_hartree = arx_e
        result.arx_gibbs_qh_hartree = arx_gqh

        result.stage = "amine_opt_freq"
        t0 = time.time()
        amine_atoms = build_molecule(rc.amine_smiles)
        amine_thermo, n_imag_amine, amine_e, amine_gqh = _species_thermo(
            amine_atoms, "amine.in", n_procs, mem
        )
        result.timing_s["amine_opt_freq"] = time.time() - t0
        result.n_imag_amine = n_imag_amine
        result.amine_energy_hartree = amine_e
        result.amine_gibbs_qh_hartree = amine_gqh

        # --- 5. ΔG‡ = G(TS) - [G(ArX) + G(amine)] -----------------------------
        result.stage = "barrier"
        result.delta_g_qh_kcal = activation_free_energy(
            ts_thermo, arx_thermo, amine_thermo, which="gibbs_qh"
        )
        result.delta_g_kcal = activation_free_energy(
            ts_thermo, arx_thermo, amine_thermo, which="gibbs"
        )
        result.delta_h_kcal = activation_free_energy(
            ts_thermo, arx_thermo, amine_thermo, which="enthalpy"
        )
        result.delta_e_kcal = activation_free_energy(
            ts_thermo, arx_thermo, amine_thermo, which="electronic_energy"
        )

        result.status = "completed" if n_imag == 1 else "ts_not_saddle"
        return result

    except Exception as exc:  # noqa: BLE001 -- failure is recorded, not raised
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        return result
