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

Solvation and coordinate (added for the cross-leaving-group re-validation). A ``solvent``
argument adds implicit solvation as a **continuum single-point correction on gas-phase
geometries and Hessians**: every opt+freq (TS and both references) runs in gas phase,
then a single solvent SCF at each gas geometry shifts the energetics by E(solv) - E(gas)
while keeping the gas thermal corrections. This is deliberate -- a *solvent-phase* freq
on Psi4 falls back to a double finite difference (thousands of PCM-SCF displacements:
hours of wall time and tens of GB of RAM), whereas the SP correction captures the
dominant electrostatic solvation at one SCF. The SP runs on the active backend; a
``solvent_model`` selects the continuum model (default IEF-PCM, matching the ``cpu_dmso``
Psi4 baseline; the GPU backend also offers ``"smd"``, which Psi4 1.10.2 cannot). The
relaxed scan's DFT single points use the same continuum directly (single points have no
such cost). ``None`` keeps the whole chain gas phase. A
``coordinate`` argument selects the relaxed-scan coordinate: ``"concerted"`` (the
gas-phase-validated antisymmetric d(C-Nu) - d(C-LG) scan) or ``"addition"`` (scan only
the forming C...Nu bond). The choice is recorded on the result and never auto-switched.

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

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import predict_snar.config as predict_snar_config

from snar_qc.qc.backend import GPU4PYSCF, PSI4, make_calculator
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

# Saddle-acceptance cutoff (cm^-1). A valid TS has exactly one imaginary mode with a
# magnitude at/above this; smaller extra imaginaries are tolerated as soft modes (e.g. a
# near-free methyl rotor the finite-difference Hessian rendered slightly imaginary) and
# folded into the thermochemistry as real modes (see Psi4Thermo.from_calculator's
# ``soft_imag_cutoff``, which this matches). Keeps a clean saddle bar one soft rotor from
# being failed as ``ts_not_saddle``.
TS_SOFT_IMAG_CUTOFF_CM = 100.0


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


def count_significant_imaginary(
    frequencies: Optional[list[float]], cutoff: float = TS_SOFT_IMAG_CUTOFF_CM
) -> int:
    """Number of imaginary modes large enough to be a real saddle direction.

    Counts imaginary (negative) modes whose magnitude is at/above ``cutoff`` (cm^-1). A
    valid transition state has exactly one. Smaller imaginaries are soft modes (e.g. a
    near-free rotor rendered slightly imaginary by the finite-difference Hessian); they
    are tolerated by the saddle gate and folded into the thermochemistry as real modes
    (see :data:`TS_SOFT_IMAG_CUTOFF_CM` and ``Psi4Thermo.from_calculator``'s
    ``soft_imag_cutoff``).

    Args:
        frequencies: Signed vibrational frequencies (cm^-1), or ``None``.
        cutoff: Magnitude threshold in cm^-1 (default :data:`TS_SOFT_IMAG_CUTOFF_CM`).

    Returns:
        The count of imaginary modes with ``abs(nu) >= cutoff`` (0 if ``None``).
    """
    if not frequencies:
        return 0
    return sum(1 for nu in frequencies if nu < 0.0 and abs(nu) >= cutoff)


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

    - ``completed`` -- TS located with exactly one imaginary mode at/above
      ``TS_SOFT_IMAG_CUTOFF_CM`` (soft sub-cutoff extras tolerated; ``n_imag_ts_soft``
      records how many); ΔG‡ available.
    - ``ts_not_saddle`` -- TS opt+freq finished but the count of *significant* imaginary
      modes (magnitude >= cutoff) is not 1.
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
    n_imag_ts_soft: Optional[int] = None
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
    coordinate: str = "concerted"
    solvent: Optional[str] = None
    solvent_model: Optional[str] = None
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


_BOHR_TO_ANGSTROM = 0.52917721067

# Gas-cache artefacts written by ``compute_barrier`` so a different solvent/model can be
# evaluated later without re-running the gas backbone (see :func:`solvent_sweep`). The
# SP-on-gas recipe makes the gas geometries + thermochemistry solvent-independent, so a
# new solvent only needs the three implicit-solvent single points on these cached inputs.
_GAS_CACHE_FILE = "gas_thermo.json"
# species key -> persisted gas-optimised geometry file (extended XYZ, carries the charge).
_GEOMETRY_FILES = {"ts": "ts_opt.xyz", "arx": "arx_opt.xyz", "amine": "amine_opt.xyz"}


def _persist_geometry(atoms: Any, path: str) -> None:
    """Write an optimised geometry to extended XYZ, preserving ``info["charge"]``.

    Extended XYZ round-trips the ASE ``info`` dict in the comment line, so the charge
    survives for the follow-up solvent single point in :func:`solvent_sweep`.
    """
    from ase.io import write  # noqa: PLC0415 -- lazy (ASE only needed when running)

    write(path, atoms, format="extxyz")


def _read_geometry(path: str) -> Any:
    """Read an extended-XYZ geometry back, coercing the charge to an int."""
    from ase.io import read  # noqa: PLC0415 -- lazy

    atoms = read(path, format="extxyz")
    atoms.info["charge"] = int(round(float(atoms.info.get("charge", 0))))
    return atoms


def _optimised_atoms(calc: Any) -> Any:
    """ASE ``Atoms`` for the optimised geometry, read back per backend.

    Psi4 optimises its own internal molecule, leaving the input ``Atoms`` untouched, so
    the relaxed geometry is read back off ``calc.wavefunction.molecule()`` (Bohr ->
    Angstrom). The gpu4pyscf backend instead writes the optimised coordinates straight
    back onto ``calc.atoms`` (it exposes ``mean_field``, not ``wavefunction``), so the
    relaxed geometry is just a copy of those atoms -- used to seed the follow-up
    implicit-solvent single point on either backend.
    """
    from ase import Atoms

    if getattr(calc, "wavefunction", None) is None:
        opt = calc.atoms.copy()
        opt.info.setdefault("charge", int(calc.options.get("charge") or 0))
        return opt

    mol = calc.wavefunction.molecule()
    symbols = [mol.symbol(i).capitalize() for i in range(mol.natom())]
    positions = [
        [
            mol.x(i) * _BOHR_TO_ANGSTROM,
            mol.y(i) * _BOHR_TO_ANGSTROM,
            mol.z(i) * _BOHR_TO_ANGSTROM,
        ]
        for i in range(mol.natom())
    ]
    opt = Atoms(symbols=symbols, positions=positions)
    opt.info["charge"] = int(round(mol.molecular_charge()))
    return opt


def _solvated_thermo(
    gas_thermo: Psi4Thermo,
    atoms: Any,
    prefix: str,
    n_procs: int,
    mem: float,
    solvent: Optional[str],
    solvent_model: Optional[str] = None,
) -> Psi4Thermo:
    """Apply an implicit solvent as a single-point correction on gas-phase thermochem.

    Gas phase (``solvent`` None): returns ``gas_thermo`` unchanged. With a ``solvent``:
    runs one implicit-solvent single point at the gas-optimised ``atoms`` and shifts every
    energy term by ``E(solv) - E(gas)``, keeping the gas-phase Hessian's thermal
    corrections, ZPVE and frequencies. This is the standard gas-geometry +
    implicit-solvent single-point protocol. It deliberately sidesteps a *solvent*
    opt+freq: it captures the dominant electrostatic solvation at one SCF, and on the
    Psi4 path a PCM frequency would fall back to a double finite difference (thousands of
    PCM-SCF displacements). The single point runs on the **active backend** (the GPU
    backend offers IEF-PCM and SMD; ``solvent_model`` selects, defaulting to the
    calculator's IEF-PCM), so a GPU chain stays on the GPU instead of needing Psi4.

    Taking the already-optimised ``atoms`` (not a calculator) lets :func:`solvent_sweep`
    reuse this exact shift logic against a cached gas geometry + thermochemistry.
    """
    if not solvent:
        return gas_thermo
    sp_options: dict[str, Any] = {"solvent": solvent}
    if solvent_model:
        sp_options["solvent_model"] = solvent_model
    sp = make_calculator(atoms, file=f"{prefix}_pcm.in", options=sp_options)
    e_pcm = sp.single_point(n_procs=n_procs, mem=mem)
    shift = e_pcm - gas_thermo.electronic_energy
    return Psi4Thermo(
        electronic_energy=e_pcm,
        gibbs=gas_thermo.gibbs + shift,
        gibbs_qh=gas_thermo.gibbs_qh + shift,
        enthalpy=gas_thermo.enthalpy + shift,
        zpve=gas_thermo.zpve,
        frequencies=gas_thermo.frequencies,
    )


def _species_gas_thermo(
    atoms: Any, file: str, n_procs: int, mem: float
) -> tuple[Psi4Thermo, Any, int]:
    """Gas-phase opt+freq for a reference species.

    Always gas phase -- the solvent is applied afterwards as a single-point correction
    by :func:`_solvated_thermo`, so this stays solvent-independent and its geometry +
    thermochemistry are the reusable gas cache.

    Returns:
        ``(gas_thermo, optimised_atoms, n_imaginary)``.
    """
    calc = make_calculator(atoms, file=file, options=None)
    calc.opt_freq(n_procs=n_procs, mem=mem)
    n_imag = count_imaginary(calc.frequencies)
    return Psi4Thermo.from_calculator(calc), _optimised_atoms(calc), n_imag


def cached_amine_reference(
    amine_smiles: str, n_procs: int, mem: float
) -> tuple[Psi4Thermo, Any, int]:
    """Gas opt+freq for the fixed model amine, reusing a cached reference when present.

    The model amine is invariant across substrates and runs, so its gas reference is cached
    on disk keyed by ``(amine, level-of-theory, backend)`` (see
    :mod:`snar_qc.poc.amine_cache`). A cache hit returns the stored ``(thermo, atoms,
    n_imag)`` and skips the opt+freq entirely (~12 s/substrate of pure redundancy); a miss
    computes it via :func:`_species_gas_thermo` and stores it for the next substrate/run.

    Backend and level are read off a freshly constructed (not-yet-run) calculator, so the
    key reflects the engine that *actually* runs: a GPU request that fell back to Psi4 is
    keyed -- and stored -- as Psi4, never crossing backends (whose absolute energies differ).
    """
    from snar_qc.poc import amine_cache
    from snar_qc.poc.complex import build_molecule

    atoms = build_molecule(amine_smiles)
    # Construct (no QC run) to read the resolved level and the concrete backend class.
    probe = make_calculator(atoms, file="amine.in", options=None)
    backend = GPU4PYSCF if type(probe).__name__.startswith("GPU") else PSI4
    level = amine_cache.level_tag(probe.options)

    hit = amine_cache.load(amine_smiles, backend, level)
    if hit is not None:
        thermo, cached_atoms, n_imag = hit
        return thermo, cached_atoms, int(n_imag or 0)

    gas_thermo, opt_atoms, n_imag = _species_gas_thermo(atoms, "amine.in", n_procs, mem)
    amine_cache.store(amine_smiles, backend, level, gas_thermo, opt_atoms, n_imag)
    return gas_thermo, opt_atoms, n_imag


def _thermo_to_dict(thermo: Psi4Thermo) -> dict[str, float]:
    """The energy terms needed to re-shift a species into a different solvent."""
    return {
        "electronic_energy": thermo.electronic_energy,
        "gibbs": thermo.gibbs,
        "gibbs_qh": thermo.gibbs_qh,
        "enthalpy": thermo.enthalpy,
        "zpve": thermo.zpve,
    }


def _write_gas_cache(
    result: "BarrierResult", gas_thermos: dict[str, Psi4Thermo]
) -> None:
    """Write ``gas_thermo.json`` (gas thermochemistry + geometry refs) to the CWD.

    Paired with the ``*_opt.xyz`` files :func:`compute_barrier` persists, this is the
    self-contained gas cache :func:`solvent_sweep` reuses to evaluate any other
    solvent/model without recomputing the gas backbone.
    """
    cache = {
        "lu_id": result.lu_id,
        "aryl_halide_smiles": result.aryl_halide_smiles,
        "amine_smiles": result.amine_smiles,
        "leaving_group": result.leaving_group,
        "central_atom": result.central_atom,
        "nu_atom": result.nu_atom,
        "lg_atom": result.lg_atom,
        "coordinate": result.coordinate,
        "reference": result.reference,
        "peak_index": result.peak_index,
        "n_imag_ts": result.n_imag_ts,
        "n_imag_ts_soft": result.n_imag_ts_soft,
        "ts_imag_freq_cm": result.ts_imag_freq_cm,
        "n_imag_arx": result.n_imag_arx,
        "n_imag_amine": result.n_imag_amine,
        "species": {
            key: {"geometry": _GEOMETRY_FILES[key], **_thermo_to_dict(gas_thermos[key])}
            for key in ("ts", "arx", "amine")
        },
    }
    Path(_GAS_CACHE_FILE).write_text(json.dumps(cache, indent=2))


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
    solvent: Optional[str] = None,
    solvent_model: Optional[str] = None,
    coordinate: str = "concerted",
) -> BarrierResult:
    """Compute ΔG‡ for one reaction complex, returning a status-carrying result.

    Runs the full chain (relaxed scan -> DFT SPs -> peak -> TS opt+freq -> reference
    opt+freq -> ΔG‡). Must be called with the current working directory set to a
    per-substrate scratch directory (the engine writes ``scan.xyz`` / ``sps/`` / Psi4
    output there).

    Args:
        rc: The reaction complex to run (carries geometry, charge, reactive indices).
        scan_stop: Final C...N (forming-bond) distance (Angstrom) for the scan.
        scan_stop_lg: Final C-LG (breaking-bond) distance (Angstrom) for the scan, used
            only by the ``concerted`` coordinate.
        scan_steps: Number of relaxed-scan steps.
        n_procs: Threads handed to each Psi4 calculation.
        mem: Memory budget (GB) per Psi4 calculation.
        make_plot: Whether to write the engine's scan plot (``GSM.png``).
        lu_id: Optional Lu_74 identifier, copied onto the result.
        solvent: Optional continuum solvent name (e.g. ``"DMSO"``). When set, the scan
            DFT single points and the implicit-solvent SP corrections on the gas TS and
            both references run with the continuum-solvation path; ``None`` keeps the
            whole chain gas phase.
        solvent_model: Optional continuum model used when ``solvent`` is set. ``None``
            (default) uses the active calculator's default (IEF-PCM, matching the
            ``cpu_dmso`` Psi4 baseline). On the GPU backend ``"smd"`` selects SMD (a model
            the Psi4 path cannot provide). Recorded on the result.
        coordinate: Reaction coordinate for the relaxed scan. ``"concerted"`` (default)
            drives the antisymmetric d(C-Nu) - d(C-LG) coordinate (Nu in *and* LG out),
            the gas-phase-validated path. ``"addition"`` scans only the forming C...Nu
            bond, leaving C-LG intact -- the stepwise-addition coordinate, which has no
            gas-phase saddle but can acquire one once solvent stabilises the developing
            Meisenheimer charge. The choice is recorded on the result; this never
            silently switches paths.

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
        solvent=solvent,
        solvent_model=solvent_model,
        coordinate=coordinate,
    )
    if coordinate not in ("concerted", "addition"):
        result.status = "error"
        result.error = (
            f"Unknown coordinate {coordinate!r}; expected 'concerted' or 'addition'."
        )
        return result

    try:
        _configure_engine()
        general_options = {
            "central_atom": rc.central_atom,
            "nu_atom": rc.nu_atom,
            "lg_atom": rc.lg_atom,
        }

        # --- 1. relaxed scan along the chosen reaction coordinate --------------
        result.stage = "scan"
        t0 = time.time()
        scan = Psi4TSScan(
            rc.atoms, {}, {}, general_options, solvent=solvent, solvent_model=solvent_model
        )
        start_nu = float(rc.atoms.get_distance(rc.central_atom - 1, rc.nu_atom - 1))
        # The forming C...Nu bond is always scanned inwards.
        scan.constrain_bond(rc.central_atom, rc.nu_atom, "auto")
        scan.add_scan(
            rc.central_atom, rc.nu_atom, "auto", start_nu, scan_stop, scan_steps
        )
        if coordinate == "concerted":
            # Second constraint + scan; xTB advances both bonds together, tracing the
            # antisymmetric d(C-Nu) - d(C-LG) coordinate (Nu in while LG out).
            start_lg = float(rc.atoms.get_distance(rc.central_atom - 1, rc.lg_atom - 1))
            scan.constrain_bond(rc.central_atom, rc.lg_atom, "auto")
            scan.add_scan(
                rc.central_atom, rc.lg_atom, "auto", start_lg, scan_stop_lg, scan_steps
            )
        # else "addition": only the forming bond is driven; C-LG stays intact and
        # relaxes. No gas-phase saddle, but a solvated one is possible (see docstring).
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
        # TS opt+freq always in gas phase; the solvent enters as an implicit-solvent
        # single-point correction at the gas saddle (matches the cpu_dmso recipe; see
        # _solvated_thermo).
        ts_calc = make_calculator(ts_guess, file="ts.in", options=None)
        ts_calc.ts_freq(n_procs=n_procs, mem=mem)
        result.timing_s["ts_opt_freq"] = time.time() - t0

        # Saddle order and imaginary mode come from the gas-phase Hessian. Split the
        # imaginary modes into *significant* (|nu| >= cutoff: a genuine reaction mode)
        # and *soft* (below it: a near-free rotor tolerated and folded into the thermo).
        n_imag = count_imaginary(ts_calc.frequencies)
        result.n_imag_ts = n_imag
        imag = [nu for nu in (ts_calc.frequencies or []) if nu < 0.0]
        result.ts_imag_freq_cm = float(min(imag)) if imag else None
        n_imag_significant = count_significant_imaginary(ts_calc.frequencies)
        result.n_imag_ts_soft = n_imag - n_imag_significant

        ts_gas_thermo = Psi4Thermo.from_calculator(ts_calc)
        ts_atoms = _optimised_atoms(ts_calc)
        _persist_geometry(ts_atoms, _GEOMETRY_FILES["ts"])
        ts_thermo = _solvated_thermo(
            ts_gas_thermo, ts_atoms, "ts", n_procs, mem, solvent, solvent_model
        )
        result.ts_energy_hartree = ts_thermo.electronic_energy
        result.ts_gibbs_qh_hartree = ts_thermo.gibbs_qh

        # --- 4. separated-reactants reference: bare ArX + bare amine -----------
        from snar_qc.poc.complex import build_molecule

        result.stage = "arx_opt_freq"
        t0 = time.time()
        arx_gas_thermo, arx_atoms, n_imag_arx = _species_gas_thermo(
            build_molecule(rc.aryl_halide_smiles), "arx.in", n_procs, mem
        )
        _persist_geometry(arx_atoms, _GEOMETRY_FILES["arx"])
        arx_thermo = _solvated_thermo(
            arx_gas_thermo, arx_atoms, "arx", n_procs, mem, solvent, solvent_model
        )
        result.timing_s["arx_opt_freq"] = time.time() - t0
        result.n_imag_arx = n_imag_arx
        result.arx_energy_hartree = arx_thermo.electronic_energy
        result.arx_gibbs_qh_hartree = arx_thermo.gibbs_qh

        result.stage = "amine_opt_freq"
        t0 = time.time()
        # Fixed model amine: reuse the cached gas reference when present, else compute it
        # once and cache it (skips a redundant ~12 s opt+freq on every later substrate/run).
        amine_gas_thermo, amine_atoms, n_imag_amine = cached_amine_reference(
            rc.amine_smiles, n_procs, mem
        )
        _persist_geometry(amine_atoms, _GEOMETRY_FILES["amine"])
        amine_thermo = _solvated_thermo(
            amine_gas_thermo, amine_atoms, "amine", n_procs, mem, solvent, solvent_model
        )
        result.timing_s["amine_opt_freq"] = time.time() - t0
        result.n_imag_amine = n_imag_amine
        result.amine_energy_hartree = amine_thermo.electronic_energy
        result.amine_gibbs_qh_hartree = amine_thermo.gibbs_qh

        # Persist the gas cache (gas thermochemistry + the three optimised geometries) so a
        # different solvent/model can be evaluated later via solvent_sweep without re-running
        # the gas backbone. Solvent-independent by construction (the SP-on-gas recipe).
        _write_gas_cache(
            result,
            {"ts": ts_gas_thermo, "arx": arx_gas_thermo, "amine": amine_gas_thermo},
        )

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

        result.status = "completed" if n_imag_significant == 1 else "ts_not_saddle"
        return result

    except Exception as exc:  # noqa: BLE001 -- failure is recorded, not raised
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        return result


def solvent_sweep(
    gas_dir: "str | Path",
    solvent: str,
    solvent_model: Optional[str] = None,
    n_procs: int = 8,
    mem: float = 12.0,
) -> BarrierResult:
    """Recompute ΔG‡ in a continuum solvent by reusing a cached gas run.

    The SP-on-gas recipe makes the gas backbone (scan, gas TS/ArX/amine opt+freq)
    solvent-independent, so a different solvent or model costs only the **three**
    implicit-solvent single points on the cached gas geometries -- minutes, not the
    ~30+ that a full gas+solvent run takes. Reads the ``gas_thermo.json`` cache and the
    ``*_opt.xyz`` geometries that :func:`compute_barrier` wrote into ``gas_dir``, runs one
    solvent SP per species (shifting its thermochemistry by ``E(solv) - E(gas)`` exactly
    as the full pipeline does), and recombines into a :class:`BarrierResult` with
    ``solvent`` / ``solvent_model`` set.

    Must be called with the CWD set to the (per-substrate) output directory: the SP
    outputs are written to the CWD, while the cached inputs are read from ``gas_dir``.
    """
    gas_dir = Path(gas_dir)
    cache_path = gas_dir / _GAS_CACHE_FILE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No gas cache at {cache_path}; this gas run predates geometry persistence. "
            f"Re-run the gas backbone to produce {_GAS_CACHE_FILE} + *_opt.xyz."
        )
    cache = json.loads(cache_path.read_text())

    result = BarrierResult(
        aryl_halide_smiles=cache["aryl_halide_smiles"],
        amine_smiles=cache["amine_smiles"],
        leaving_group=cache["leaving_group"],
        central_atom=cache["central_atom"],
        nu_atom=cache["nu_atom"],
        lg_atom=cache["lg_atom"],
        lu_id=cache.get("lu_id"),
        solvent=solvent,
        solvent_model=solvent_model,
        coordinate=cache.get("coordinate", "concerted"),
    )
    result.reference = cache.get("reference", "separated_reactants")
    result.peak_index = cache.get("peak_index")
    result.n_imag_ts = cache.get("n_imag_ts")
    result.n_imag_ts_soft = cache.get("n_imag_ts_soft")
    result.ts_imag_freq_cm = cache.get("ts_imag_freq_cm")
    result.n_imag_arx = cache.get("n_imag_arx")
    result.n_imag_amine = cache.get("n_imag_amine")

    try:
        thermos: dict[str, Psi4Thermo] = {}
        for key in ("ts", "arx", "amine"):
            spec = cache["species"][key]
            geom = gas_dir / spec["geometry"]
            if not geom.exists():
                raise FileNotFoundError(f"missing cached geometry {geom}")
            t0 = time.time()
            gas_thermo = Psi4Thermo(
                electronic_energy=spec["electronic_energy"],
                gibbs=spec["gibbs"],
                gibbs_qh=spec["gibbs_qh"],
                enthalpy=spec["enthalpy"],
                zpve=spec["zpve"],
                frequencies=[],
            )
            thermos[key] = _solvated_thermo(
                gas_thermo, _read_geometry(str(geom)), key, n_procs, mem,
                solvent, solvent_model,
            )
            result.timing_s[f"{key}_solvent_sp"] = time.time() - t0

        ts_t, arx_t, amine_t = thermos["ts"], thermos["arx"], thermos["amine"]
        result.ts_energy_hartree = ts_t.electronic_energy
        result.ts_gibbs_qh_hartree = ts_t.gibbs_qh
        result.arx_energy_hartree = arx_t.electronic_energy
        result.arx_gibbs_qh_hartree = arx_t.gibbs_qh
        result.amine_energy_hartree = amine_t.electronic_energy
        result.amine_gibbs_qh_hartree = amine_t.gibbs_qh
        result.delta_g_qh_kcal = activation_free_energy(
            ts_t, arx_t, amine_t, which="gibbs_qh"
        )
        result.delta_g_kcal = activation_free_energy(
            ts_t, arx_t, amine_t, which="gibbs"
        )
        result.delta_h_kcal = activation_free_energy(
            ts_t, arx_t, amine_t, which="enthalpy"
        )
        result.delta_e_kcal = activation_free_energy(
            ts_t, arx_t, amine_t, which="electronic_energy"
        )
        # The saddle was validated in the gas run; carry that verdict (significant
        # imaginary modes = n_imag_ts - soft), since the geometry is unchanged.
        n_sig = (result.n_imag_ts or 0) - (result.n_imag_ts_soft or 0)
        result.status = "completed" if n_sig == 1 else "ts_not_saddle"
        result.stage = "solvent_sweep"
    except Exception as exc:  # noqa: BLE001 -- failure is recorded, not raised
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"

    return result
