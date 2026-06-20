"""Psi4 bond-order adapter for snar_qc.

``Psi4BondOrders`` wraps a Psi4 ``Wavefunction`` and exposes bond orders through the
**same 1-indexed ``get_bo(atom_1, atom_2)`` contract** as predict-snar's
:class:`predict_snar.parsers.NBOParser` (which reads Gaussian's *Wiberg bond index
matrix in the NAO basis*). This lets the relaxed-scan transition-state search reuse
its bond-order change criterion (:meth:`predict_snar.calculators.TSScan.validate_peaks`)
unchanged, with Psi4 standing in for Gaussian + NBO.

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
import psi4

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
