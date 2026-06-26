"""A reusable on-disk cache for the **fixed model-amine reference** (methylamine).

The separated-reactants reference G(ArX) + G(amine) re-optimises and re-freqs the model
amine for *every* substrate (``barrier.py`` stage ``amine_opt_freq``), even though the
amine is invariant across the whole cohort and across runs -- ~12 s of pure redundancy per
substrate. The aryl halide is substrate-specific and cannot be cached; the amine can.

This module is the cache: a small ``assets/amine_ref/<key>.{json,xyz}`` pair per
``(amine, level-of-theory, backend)``. The JSON carries the gas thermochemistry scalars
(the same five terms ``_thermo_to_dict`` persists) plus ``n_imag`` and provenance; the
sidecar XYZ carries the gas-optimised amine geometry (needed for the follow-up implicit
solvent single point). On a hit, ``barrier.cached_amine_reference`` returns the cached
``(Psi4Thermo, atoms, n_imag)`` and skips the opt+freq entirely.

**The cache is keyed by backend and level of theory on purpose.** Absolute Hartree
energies differ between Psi4 and gpu4pyscf (density fitting, integral screening), so a
Psi4-seeded amine reference must never be combined with a gpu4pyscf TS/ArX -- the key keeps
them apart.

Seeding does **not** require recomputation: the amine reference already exists inside every
finished gas run's ``gas_thermo.json`` (``species.amine``) + ``amine_opt.xyz``, so
``scripts/extract_amine_ref.py`` copies it straight into the cache. This mirrors the
``extract_gas_cache.py`` "backfill, no recompute" pattern. The runtime path only computes
(and stores) on a genuine miss.

Pure I/O: this module imports only ASE (lazily) and :class:`Psi4Thermo`; it never pulls in
Psi4 / gpu4pyscf, so it is importable in any env and from the per-substrate working dir.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from snar_qc.qc.thermo import Psi4Thermo

_CACHE_ENV = "SNAR_QC_AMINE_CACHE"
# Default cache root: <repo-root>/assets/amine_ref. amine_cache.py lives at
# src/snar_qc/poc/, so parents[3] is the repo root. Absolute, so it resolves the same way
# regardless of the per-substrate working directory compute_barrier runs in.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "assets" / "amine_ref"

# The five gas thermochemistry scalars that define the reference (Hartree); frequencies are
# not needed downstream (the qh Gibbs is already folded in), matching solvent_sweep's reuse.
_THERMO_KEYS = ("electronic_energy", "gibbs", "gibbs_qh", "enthalpy", "zpve")


def cache_root() -> Path:
    """The cache directory: ``$SNAR_QC_AMINE_CACHE`` if set, else ``assets/amine_ref``."""
    env = os.environ.get(_CACHE_ENV)
    return Path(env).expanduser() if env else _DEFAULT_ROOT


def canonical_amine(amine_smiles: str) -> str:
    """Canonical SMILES for the amine, so ``CN`` and ``NC`` map to one key.

    Falls back to the input string if RDKit is unavailable or the SMILES does not parse
    (the cache key stays self-consistent either way -- store and load use this same fn).
    """
    try:
        from rdkit import Chem  # noqa: PLC0415 -- lazy; RDKit is a runtime dep

        mol = Chem.MolFromSmiles(amine_smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001 -- a missing/broken RDKit must not break caching
        pass
    return amine_smiles


def level_tag(options: dict[str, Any]) -> str:
    """Level-of-theory tag from a calculator's options, e.g. ``b3lyp-d3bj_def2-svp``."""
    functional = str(options.get("functional", "")).lower()
    dispersion = str(options.get("dispersion") or "").lower()
    basis = str(options.get("basis_set", "")).lower()
    method = f"{functional}-{dispersion}" if dispersion else functional
    return f"{method}_{basis}"


def cache_key(amine_smiles: str, backend: str, level: str) -> str:
    """Filesystem-safe stem for the ``(amine, backend, level)`` triple."""
    slug = (
        re.sub(r"[^A-Za-z0-9]+", "_", canonical_amine(amine_smiles)).strip("_")
        or "amine"
    )
    return f"{slug}__{level}__{backend}"


def _paths(amine_smiles: str, backend: str, level: str) -> tuple[Path, Path]:
    stem = cache_key(amine_smiles, backend, level)
    root = cache_root()
    return root / f"{stem}.json", root / f"{stem}.xyz"


def _read_geometry(path: Path) -> Any:
    """Read an extended-XYZ geometry, coercing the charge to int (mirrors barrier)."""
    from ase.io import read  # noqa: PLC0415 -- lazy

    atoms = read(str(path), format="extxyz")
    atoms.info["charge"] = int(round(float(atoms.info.get("charge", 0))))
    return atoms


def _write_geometry(atoms: Any, path: Path) -> None:
    """Write an optimised geometry to extended XYZ, preserving ``info["charge"]``."""
    from ase.io import write  # noqa: PLC0415 -- lazy

    write(str(path), atoms, format="extxyz")


def load(
    amine_smiles: str, backend: str, level: str
) -> Optional[tuple[Psi4Thermo, Any, Optional[int]]]:
    """Return the cached ``(Psi4Thermo, atoms, n_imag)`` for the triple, or ``None``.

    A miss (no file, or stored provenance that does not match the requested triple) returns
    ``None`` so the caller computes instead -- never a wrong-reference hit.
    """
    json_path, xyz_path = _paths(amine_smiles, backend, level)
    if not (json_path.exists() and xyz_path.exists()):
        return None
    rec = json.loads(json_path.read_text())
    # Defensive: the key already encodes the triple, but guard against a hash-collision or a
    # hand-edited file by confirming the stored provenance matches what was asked for.
    if (
        rec.get("amine_smiles") != canonical_amine(amine_smiles)
        or rec.get("backend") != backend
        or rec.get("level") != level
    ):
        return None
    thermo = Psi4Thermo(
        electronic_energy=rec["electronic_energy"],
        gibbs=rec["gibbs"],
        gibbs_qh=rec["gibbs_qh"],
        enthalpy=rec["enthalpy"],
        zpve=rec["zpve"],
        frequencies=[],
    )
    return thermo, _read_geometry(xyz_path), rec.get("n_imag")


def store(
    amine_smiles: str,
    backend: str,
    level: str,
    thermo: Psi4Thermo,
    atoms: Any,
    n_imag: Optional[int],
    *,
    provenance: Optional[dict[str, Any]] = None,
) -> Path:
    """Write the cache pair for the triple; return the JSON path.

    Args:
        amine_smiles: Model amine SMILES (canonicalised for the key).
        backend: ``"psi4"`` or ``"gpu4pyscf"`` -- absolute energies are backend-specific.
        level: Level-of-theory tag (see :func:`level_tag`).
        thermo: Gas-phase amine thermochemistry.
        atoms: Gas-optimised amine geometry (ASE ``Atoms`` with ``info["charge"]``).
        n_imag: Number of imaginary modes (0 for a clean minimum).
        provenance: Optional extra provenance (e.g. the source run for a no-recompute seed).
    """
    json_path, xyz_path = _paths(amine_smiles, backend, level)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geometry(atoms, xyz_path)
    rec: dict[str, Any] = {
        "amine_smiles": canonical_amine(amine_smiles),
        "backend": backend,
        "level": level,
        "n_imag": n_imag,
        "geometry": xyz_path.name,
        **{k: getattr(thermo, k) for k in _THERMO_KEYS},
    }
    if provenance:
        rec["_provenance"] = provenance
    json_path.write_text(json.dumps(rec, indent=2))
    return json_path
