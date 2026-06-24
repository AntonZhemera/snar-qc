"""Bond-order adapters for snar_qc (Psi4 and gpu4pyscf), and a backend factory.

``Psi4BondOrders`` wraps a Psi4 ``Wavefunction`` and exposes bond orders through the
**same 1-indexed ``get_bo(atom_1, atom_2)`` contract** as predict-snar's
:class:`predict_snar.parsers.NBOParser` (which reads Gaussian's *Wiberg bond index
matrix in the NAO basis*). This lets the relaxed-scan transition-state search reuse
its bond-order change criterion (:meth:`predict_snar.calculators.TSScan.validate_peaks`)
unchanged, with Psi4 standing in for Gaussian + NBO.

``PyscfBondOrders`` is the GPU sibling: it computes **Mayer** bond orders straight from a
(gpu4)pyscf mean-field object (``GPU4PySCFCalculator.mean_field``), exposing the identical
``get_bo`` contract so the same peak-validation works on the GPU backend, where no Psi4
``Wavefunction`` exists. :func:`bond_orders_from_calculator` is the single dispatch point:
given either calculator it returns the matching adapter, so callers (the TS scan) stay
backend-agnostic.

**Import discipline (mirror of the CPU-fallback contract).** ``import psi4`` is *lazy*
(inside :class:`Psi4BondOrders`), so importing this module costs no Psi4 -- the gas-phase
GPU path runs on the ``gpuqc`` env, which deliberately carries no Psi4. ``PyscfBondOrders``
needs only NumPy and the mean-field's own pyscf objects (``make_rdm1`` / ``get_ovlp`` /
``mol``), so it imports neither psi4 nor pyscf at module scope either.

Bond-order definition
---------------------
**Mayer bond orders** are used by default. They are the natural population-analysis
analogue of the NBO Wiberg-in-NAO indices the original pipeline relied on, and in
Psi4 1.10.2 they reproduce textbook bond multiplicities cleanly (N2 Mayer = 3.00 at
HF/STO-3G; H2O O-H Mayer = 1.01 at B3LYP-D3BJ/def2-SVP). Psi4's Wiberg-Löwdin indices
are offered as a fallback (``kind="wiberg"``) but run hotter on the same systems
(N2 = 3.51, H2O O-H = 1.16 at B3LYP-D3BJ/def2-SVP), so they map less faithfully onto
the 0.05 / 0.5 bond-order thresholds baked into ``validate_peaks``.

Accessor (Psi4 1.10.2, determined empirically)
----------------------------------------------
``psi4.oeprop(wfn, "MAYER_INDICES")`` computes the matrix and stores it on the
wavefunction; it is then read back with ``wfn.array_variable("MAYER INDICES")`` -- note
the stored key uses a **space**, not the underscore of the oeprop argument. The matrix
is symmetric, diagonal-zeroed, and ordered exactly as the input geometry (the
``Psi4Calculator`` builds molecules with ``no_com`` / ``no_reorient`` / ``symmetry c1``,
so atom order is preserved). Computing the property on a stored wavefunction is
self-contained: it remains correct even after intervening ``psi4.core.clean()`` calls
from subsequent single points, which is what lets ``Psi4TSScan.read_sp_output`` build
these objects from wavefunctions stashed earlier in ``run_sps``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psi4.core import Wavefunction


# Per-kind (oeprop property argument, stored array-variable key). The oeprop name
# uses underscores; the variable Psi4 writes back uses spaces.
_BOND_ORDER_KINDS: dict[str, tuple[str, str]] = {
    "mayer": ("MAYER_INDICES", "MAYER INDICES"),
    "wiberg": ("WIBERG_LOWDIN_INDICES", "WIBERG LOWDIN INDICES"),
}


class Psi4BondOrders:
    """Bond orders from a Psi4 wavefunction with ``NBOParser``-compatible indexing.

    Args:
        wavefunction: A converged Psi4 ``Wavefunction`` (e.g. ``Psi4Calculator.
            wavefunction`` after a single point).
        kind: Bond-order flavour, ``"mayer"`` (default) or ``"wiberg"``.

    Attributes:
        bo_matrix: Symmetric ``(n_atoms, n_atoms)`` bond-order matrix (``ndarray``),
            atom order matching the input geometry; mirrors ``NBOParser.bo_matrix``.
        kind: The bond-order flavour used.
        wavefunction: The source wavefunction.
    """

    def __init__(self, wavefunction: "Wavefunction", kind: str = "mayer") -> None:
        if wavefunction is None:
            raise ValueError("Psi4BondOrders requires a Psi4 Wavefunction (got None).")
        kind = kind.lower()
        if kind not in _BOND_ORDER_KINDS:
            raise ValueError(
                f"Unknown bond-order kind {kind!r}; "
                f"choose one of {sorted(_BOND_ORDER_KINDS)}."
            )

        oeprop_name, variable_key = _BOND_ORDER_KINDS[kind]

        # Lazy import: keep ``import snar_qc.qc.bond_orders`` Psi4-free so the GPU path
        # (gpuqc env, no Psi4) can import the module and use ``PyscfBondOrders`` /
        # ``bond_orders_from_calculator`` without ever pulling Psi4 in.
        import psi4  # noqa: PLC0415

        # Compute the property onto the wavefunction, then read the matrix back.
        psi4.oeprop(wavefunction, oeprop_name)
        matrix: Any = wavefunction.array_variable(variable_key)
        self.bo_matrix = np.asarray(matrix)
        self.kind = kind
        self.wavefunction = wavefunction

    def get_bo(self, atom_1: int, atom_2: int) -> float:
        """Return the bond order between two atoms.

        Matches :meth:`predict_snar.parsers.NBOParser.get_bo` exactly: atom indices
        are **1-indexed** and the value read is ``bo_matrix[atom_1 - 1][atom_2 - 1]``.

        Args:
            atom_1: Index of atom 1 (1-indexed).
            atom_2: Index of atom 2 (1-indexed).

        Returns:
            The bond order (float).
        """
        bond_order = self.bo_matrix[atom_1 - 1][atom_2 - 1]

        return float(bond_order)

    def __repr__(self) -> str:
        n = self.bo_matrix.shape[0] if self.bo_matrix.ndim == 2 else 0
        return f"{self.__class__.__name__}(kind={self.kind!r}, n_atoms={n})"


class PyscfBondOrders:
    """Mayer bond orders from a (gpu4)pyscf mean-field, NBOParser-compatible indexing.

    The GPU analogue of :class:`Psi4BondOrders`: it exposes the same 1-indexed
    ``get_bo(atom_1, atom_2)`` / ``bo_matrix`` surface so the relaxed-scan peak
    validation (``TSScan.validate_peaks``) works unchanged on the GPU backend, where the
    calculator stores a ``mean_field`` (``GPU4PySCFCalculator.mean_field``) rather than a
    Psi4 ``Wavefunction``.

    Mayer definition
    ----------------
    ``B_AB = 2 * sum_{mu in A, nu in B} [ (Pa S)_mu,nu (Pa S)_nu,mu
                                          + (Pb S)_mu,nu (Pb S)_nu,mu ]`` with ``Pa`` /
    ``Pb`` the alpha/beta AO density matrices and ``S`` the AO overlap. For a closed shell
    ``Pa = Pb = P/2`` this reduces to the textbook ``sum (PS)(PS)^T`` over the atom blocks.
    It reproduces Psi4's ``MAYER INDICES`` to the bond-validation tolerance (N2 = 3.00;
    H2O O-H = 1.01 at B3LYP-D3BJ/def2-SVP, matching the :class:`Psi4BondOrders` pins), so
    the 0.05 / 0.5 thresholds in ``validate_peaks`` behave identically across backends. The
    AO overlap is the exact ``get_ovlp()`` even under density fitting (DF approximates J/K,
    not S), so the orders are unaffected by the DF used for the energy.

    Args:
        mean_field: A converged (gpu4)pyscf mean-field (RKS/UKS, optionally
            density-fitted), e.g. ``GPU4PySCFCalculator.mean_field`` after a single point.
        kind: Bond-order flavour. Only ``"mayer"`` is implemented (the flavour
            ``validate_peaks`` uses); any other value raises.

    Attributes:
        bo_matrix: Symmetric, diagonal-zeroed ``(n_atoms, n_atoms)`` Mayer matrix
            (``ndarray``), atom order matching the molecule; mirrors ``Psi4BondOrders``.
        kind: The bond-order flavour used (``"mayer"``). The source mean-field is
            intentionally not retained (it would pin GPU memory -- see ``__init__``).
    """

    def __init__(self, mean_field: Any, kind: str = "mayer") -> None:
        if mean_field is None:
            raise ValueError(
                "PyscfBondOrders requires a converged (gpu4)pyscf mean-field (got None)."
            )
        kind = kind.lower()
        if kind != "mayer":
            raise ValueError(
                f"PyscfBondOrders implements 'mayer' only (got {kind!r}); the Psi4 "
                "backend offers 'wiberg' as well."
            )

        # The mean-field is deliberately NOT retained: holding it pins the GPU
        # density-fit tensors (~0.2 GB each), and the relaxed scan keeps ~14 of these
        # bond-order objects alive in ``nbo_data`` -- enough to exhaust a 4 GB card
        # mid-scan. Only the host-side Mayer matrix is kept; ``get_bo`` needs nothing else.
        self.bo_matrix = self._mayer_matrix(mean_field)
        self.kind = kind

    @staticmethod
    def _to_numpy(array: Any) -> np.ndarray:
        """CuPy / gpu4pyscf array -> NumPy (a no-op for a NumPy array)."""
        getter = getattr(array, "get", None)
        if callable(getter):  # cupy.ndarray.get() copies device -> host
            return getter()
        return np.asarray(array)

    @classmethod
    def _mayer_matrix(cls, mean_field: Any) -> np.ndarray:
        """Symmetric, diagonal-zeroed Mayer bond-order matrix from a mean-field."""
        mol = mean_field.mol
        dm = cls._to_numpy(mean_field.make_rdm1())
        overlap = cls._to_numpy(mean_field.get_ovlp())
        if dm.ndim == 3:  # UKS/UHF: [dm_alpha, dm_beta]
            dm_alpha, dm_beta = dm[0], dm[1]
        else:  # RKS/RHF total density -> split evenly between the spins
            dm_alpha = dm_beta = dm * 0.5

        ps_alpha = dm_alpha @ overlap
        ps_beta = dm_beta @ overlap
        # Elementwise (PS) * (PS)^T accumulates (PS)_mn (PS)_nm per AO pair.
        per_ao = 2.0 * (ps_alpha * ps_alpha.T + ps_beta * ps_beta.T)

        aoslices = mol.aoslice_by_atom()
        n_atoms = mol.natm
        matrix = np.zeros((n_atoms, n_atoms))
        for atom_a in range(n_atoms):
            a0, a1 = aoslices[atom_a][2], aoslices[atom_a][3]
            for atom_b in range(n_atoms):
                b0, b1 = aoslices[atom_b][2], aoslices[atom_b][3]
                matrix[atom_a, atom_b] = per_ao[a0:a1, b0:b1].sum()
        # Symmetrise: B_AB and B_BA are equal in exact arithmetic but the two block sums
        # round differently (~1e-15), so average them for a bit-exact symmetric matrix --
        # matching Psi4's stored-symmetric MAYER INDICES and the get_bo(i,j)==get_bo(j,i)
        # contract. (a+b == b+a in IEEE 754, so this is exactly symmetric.)
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)  # match Psi4's diagonal-zeroed MAYER INDICES
        return matrix

    def get_bo(self, atom_1: int, atom_2: int) -> float:
        """Bond order between two atoms (1-indexed, symmetric -- see Psi4BondOrders)."""
        return float(self.bo_matrix[atom_1 - 1][atom_2 - 1])

    def __repr__(self) -> str:
        n = self.bo_matrix.shape[0] if self.bo_matrix.ndim == 2 else 0
        return f"{self.__class__.__name__}(kind={self.kind!r}, n_atoms={n})"


def bond_orders_from_calculator(calc: Any, kind: str = "mayer") -> Any:
    """Return the bond-order adapter native to a calculator's backend.

    The single dispatch point that lets the relaxed-scan TS search stay backend-agnostic:

    - a :class:`snar_qc.qc.psi4_calculator.Psi4Calculator` carries a Psi4 ``wavefunction``
      -> :class:`Psi4BondOrders`;
    - a :class:`snar_qc.qc.gpu4pyscf_calculator.GPU4PySCFCalculator` carries a
      ``mean_field`` -> :class:`PyscfBondOrders`.

    Dispatch is on which QC handle the calculator exposes (``wavefunction`` vs
    ``mean_field``), so neither backend's heavy library is imported on the other's path.

    Args:
        calc: A calculator that has run a single point (so its QC handle is populated).
        kind: Bond-order flavour passed through to the adapter (default ``"mayer"``).

    Returns:
        A bond-order adapter exposing the 1-indexed ``get_bo`` / ``bo_matrix`` contract.

    Raises:
        TypeError: If the calculator exposes neither a populated ``wavefunction`` nor a
            ``mean_field`` (e.g. no single point has been run).
    """
    wavefunction = getattr(calc, "wavefunction", None)
    if wavefunction is not None:
        return Psi4BondOrders(wavefunction, kind=kind)
    mean_field = getattr(calc, "mean_field", None)
    if mean_field is not None:
        return PyscfBondOrders(mean_field, kind=kind)
    raise TypeError(
        f"{type(calc).__name__} exposes no bond-order source (wavefunction / "
        "mean_field); run a single point first."
    )
