"""Quasi-harmonic free energies and activation barriers for snar_qc.

This module turns the harmonic thermochemistry captured by
:class:`snar_qc.qc.psi4_calculator.Psi4Calculator` (Gibbs free energy, enthalpy, ZPVE,
and the vibrational frequency list, all off a Psi4 frequency run) into a **Grimme
quasi-RRHO** free energy and, from a set of such states, an activation free energy
ΔG‡. It mirrors the GoodVibes post-processing the predict-snar pipeline relied on.

Grimme quasi-rigid-rotor-harmonic-oscillator (qRRHO)
----------------------------------------------------
The rigid-rotor-harmonic-oscillator (RRHO) entropy of a low-frequency vibration
diverges as the frequency goes to zero, which makes harmonic Gibbs free energies of
floppy modes unreliable. Grimme's quasi-harmonic correction (Chem. Eur. J. 2012, 18,
9955) damps each low mode's harmonic-oscillator entropy ``S_HO`` towards a free-rotor
entropy ``S_FR`` with an interpolating weight that switches around a frequency cutoff.
Following the predict-snar GoodVibes settings, the cutoff is **100 cm^-1**, the
temperature **298.15 K**, and only the vibrational-entropy term is corrected -- the
electronic energy, ZPVE, thermal enthalpy, and rotational/translational entropy are
left at their harmonic values. Imaginary (negative), zero, and non-positive modes are
dropped from the sum (Psi4's own thermo already excludes imaginary modes).

Units
-----
Energies in and out of :func:`grimme_qh_gibbs` are **Hartree**; frequencies are
**cm^-1**. :func:`activation_free_energy` returns ΔG‡ in **kcal/mol** via
``predict_snar.data.HARTREE_TO_KCAL`` (never a hardcoded 627.5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

from predict_snar.data import HARTREE_TO_KCAL

if TYPE_CHECKING:  # pragma: no cover - typing only
    from snar_qc.qc.psi4_calculator import Psi4Calculator

# SI physical constants (CODATA 2018, as used by GoodVibes).
_R = 8.314462618  # gas constant, J mol^-1 K^-1
_H = 6.62607015e-34  # Planck constant, J s
_KB = 1.380649e-23  # Boltzmann constant, J K^-1
_N_A = 6.02214076e23  # Avogadro constant, mol^-1
_C_CM = 2.99792458e10  # speed of light, cm s^-1 (so nu[cm^-1] * c -> s^-1)

# Average moment of inertia used to cap the free-rotor entropy (Grimme), kg m^2.
_BAV = 1.0e-44

# Hartree -> J/mol. HARTREE_TO_KCAL is Hartree -> kcal/mol; *4184 -> J/mol.
_HARTREE_TO_J_PER_MOL = HARTREE_TO_KCAL * 4184.0


def grimme_qh_gibbs(
    harmonic_gibbs: float,
    frequencies: list[float],
    temperature: float = 298.15,
    cutoff: float = 100.0,
) -> float:
    """Grimme quasi-harmonic Gibbs free energy (Hartree).

    Corrects only the vibrational-entropy term of a harmonic Gibbs free energy:
    ``G_qh = harmonic_gibbs - T * (S_qh - S_HO)``, summed over real positive
    vibrational modes. Imaginary (negative), zero, and non-positive frequencies are
    dropped.

    For each mode ``nu`` (cm^-1) at temperature ``T`` (K):

    - ``x = h*c*nu/(kB*T)``
    - harmonic-oscillator entropy ``S_HO = R*[x/(e^x - 1) - ln(1 - e^-x)]``
    - free-rotor entropy from the (capped) moment of inertia ``mu'``:
      ``S_FR = R*[1/2 + ln((8*pi^3*mu'*kB*T/h^2)^0.5)]``
    - Grimme damping weight ``w = 1/(1 + (cutoff/nu)^4)``
    - ``S_qRRHO = w*S_HO + (1 - w)*S_FR``

    Args:
        harmonic_gibbs: Harmonic Gibbs free energy G (Hartree).
        frequencies: Vibrational frequencies (cm^-1); negative/imaginary, zero, and
            non-positive entries are ignored.
        temperature: Temperature in Kelvin (default 298.15).
        cutoff: Grimme frequency cutoff in cm^-1 (default 100).

    Returns:
        The quasi-harmonic Gibbs free energy in Hartree.
    """
    delta_s = 0.0  # J mol^-1 K^-1, sum of (S_qRRHO - S_HO) over real positive modes
    for nu in frequencies:
        if nu <= 0.0:  # drop imaginary (negative), zero, non-positive modes
            continue

        omega = nu * _C_CM  # vibrational frequency in s^-1
        x = _H * omega / (_KB * temperature)

        # Harmonic-oscillator vibrational entropy.
        s_ho = _R * (x / (math.exp(x) - 1.0) - math.log(1.0 - math.exp(-x)))

        # Free-rotor entropy from the moment of inertia, capped via Bav.
        mu = _H / (8.0 * math.pi**2 * omega)
        mu_eff = mu * _BAV / (mu + _BAV)
        s_fr = _R * (
            0.5
            + math.log((8.0 * math.pi**3 * mu_eff * _KB * temperature / _H**2) ** 0.5)
        )

        # Grimme interpolation between harmonic and free-rotor entropy.
        weight = 1.0 / (1.0 + (cutoff / nu) ** 4)
        s_qrrho = weight * s_ho + (1.0 - weight) * s_fr

        delta_s += s_qrrho - s_ho

    # -T*delta_S in J/mol, converted to Hartree.
    correction_hartree = -temperature * delta_s / _HARTREE_TO_J_PER_MOL
    return harmonic_gibbs + correction_hartree


class Psi4Thermo:
    """Thermochemical state of a single species from a Psi4 frequency run.

    Bundles the harmonic thermochemistry with the Grimme quasi-harmonic Gibbs free
    energy so a barrier can be assembled from a transition state and its references.

    Args:
        electronic_energy: Total electronic energy (Hartree).
        gibbs: Harmonic Gibbs free energy G (Hartree).
        gibbs_qh: Grimme quasi-harmonic Gibbs free energy (Hartree).
        enthalpy: Total enthalpy H (Hartree).
        zpve: Zero-point vibrational energy (Hartree).
        frequencies: Signed real vibrational frequencies (cm^-1); imaginary modes are
            negative (cclib/Gaussian convention).

    Attributes:
        Mirror the constructor arguments.
    """

    def __init__(
        self,
        electronic_energy: float,
        gibbs: float,
        gibbs_qh: float,
        enthalpy: float,
        zpve: float,
        frequencies: list[float],
    ) -> None:
        self.electronic_energy = electronic_energy
        self.gibbs = gibbs
        self.gibbs_qh = gibbs_qh
        self.enthalpy = enthalpy
        self.zpve = zpve
        self.frequencies = list(frequencies)

    @property
    def imaginary_frequencies(self) -> list[float]:
        """The negative (imaginary) entries of :attr:`frequencies`."""
        return [nu for nu in self.frequencies if nu < 0.0]

    @classmethod
    def from_calculator(
        cls,
        calc: "Psi4Calculator",
        temperature: float = 298.15,
        cutoff: float = 100.0,
    ) -> "Psi4Thermo":
        """Build from a ``Psi4Calculator`` after a ``freq`` / ``opt_freq`` run.

        Reads the captured harmonic thermochemistry off the calculator and computes
        the Grimme quasi-harmonic Gibbs free energy.

        Args:
            calc: A ``Psi4Calculator`` whose ``frequencies`` / ``free_energy`` /
                ``enthalpy`` / ``zpve`` were populated by a frequency run.
            temperature: Temperature in Kelvin for the quasi-harmonic correction.
            cutoff: Grimme frequency cutoff in cm^-1.

        Returns:
            A populated :class:`Psi4Thermo`.

        Raises:
            ValueError: If the calculator has no captured frequencies (no freq run).
        """
        if calc.frequencies is None or calc.free_energy is None:
            raise ValueError(
                "Psi4Thermo.from_calculator requires a completed frequency run "
                "(calc.frequencies / calc.free_energy are None). Call freq() or "
                "opt_freq() first."
            )
        gibbs_qh = grimme_qh_gibbs(
            calc.free_energy,
            calc.frequencies,
            temperature=temperature,
            cutoff=cutoff,
        )
        return cls(
            electronic_energy=calc.energy,
            gibbs=calc.free_energy,
            gibbs_qh=gibbs_qh,
            enthalpy=calc.enthalpy,
            zpve=calc.zpve,
            frequencies=calc.frequencies,
        )


def activation_free_energy(
    ts: Union["Psi4Thermo", float],
    *references: Union["Psi4Thermo", float],
    which: str = "gibbs_qh",
) -> float:
    """Activation free energy ΔG‡ (kcal/mol).

    ``ΔG‡ = G(ts) - sum(G(references))``, converted to kcal/mol via
    ``HARTREE_TO_KCAL``. Each argument is either a :class:`Psi4Thermo` (the attribute
    named by ``which`` is read, in Hartree) or a float already in Hartree.

    For a unimolecular reaction-complex -> TS reference the gas-phase standard-state
    term cancels, so no concentration correction is applied here (POC assumption). A
    bimolecular reference with a change in mole number (Δn != 0) needs a standard-state
    concentration correction, which is **out of scope** for this proof of concept.

    Args:
        ts: The transition state (``Psi4Thermo`` or Hartree float).
        *references: One or more reference states (``Psi4Thermo`` or Hartree floats).
        which: Which energy attribute to read from ``Psi4Thermo`` inputs; one of
            ``"gibbs_qh"`` (default), ``"gibbs"``, ``"enthalpy"``, ``"electronic_energy"``.

    Returns:
        ΔG‡ in kcal/mol.

    Raises:
        ValueError: if no reference state is given (a barrier needs something to
            measure the transition state against).
    """
    if not references:
        raise ValueError(
            "activation_free_energy needs at least one reference state to measure "
            "the transition state against."
        )

    def _value(state: Union["Psi4Thermo", float]) -> float:
        if isinstance(state, Psi4Thermo):
            return float(getattr(state, which))
        return float(state)

    barrier_hartree = _value(ts) - sum(_value(ref) for ref in references)
    return barrier_hartree * HARTREE_TO_KCAL
