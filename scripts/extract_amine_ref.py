#!/usr/bin/env python
"""Seed the fixed model-amine reference cache from a finished gas run -- **no recompute**.

The separated-reactants reference re-optimises + re-freqs the model amine (methylamine) for
every substrate, even though it is invariant. :mod:`snar_qc.poc.amine_cache` caches that
reference so the runtime path can skip the redundant opt+freq -- but the cache does not have
to be *computed*: every finished gas run already carries the amine reference inside each
substrate's ``gas_thermo.json`` (``species.amine``) + ``amine_opt.xyz``. This tool copies it
straight into the cache, mirroring the ``extract_gas_cache.py`` "backfill, no recompute"
pattern.

Because the amine is fixed, every completed substrate in a run holds the *same* reference;
this tool reads them all and **verifies they agree** (electronic energy identical to
<1e-6 Hartree) before writing one canonical asset -- a built-in faithfulness check that the
amine really is a constant reference and that no substrate's amine job drifted.

The backend is not recorded in ``gas_thermo.json`` and absolute energies are
backend-specific, so ``--backend`` is **required** (a ``gpu_*`` run is ``gpu4pyscf``; a
Psi4/CPU run is ``psi4``). The level defaults to the POC's fixed B3LYP-D3BJ/def2-SVP.

Pure I/O (ASE + json); does not import Psi4 / gpu4pyscf, so it runs in any env. Examples::

    # Seed the gpu4pyscf amine reference from the GPU DMSO gas backbone:
    python scripts/extract_amine_ref.py --run data/processed/gpu_dmso_gas --backend gpu4pyscf

    # Seed the Psi4 amine reference from a CPU gas run:
    python scripts/extract_amine_ref.py --run data/processed/cpu_gas_run --backend psi4

    # Inspect without writing:
    python scripts/extract_amine_ref.py --run data/processed/gpu_dmso_gas \\
        --backend gpu4pyscf --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Make ``snar_qc`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snar_qc.poc import amine_cache  # noqa: E402  (pure I/O; no Psi4 import chain)
from snar_qc.qc.thermo import Psi4Thermo  # noqa: E402

_ENERGY_TOL_HARTREE = 1e-6  # the fixed amine must be bit-stable across substrates


class SeedError(RuntimeError):
    """The amine reference could not be extracted faithfully."""


def _collect_amine_refs(run_dir: Path) -> list[tuple[str, dict, int, Path]]:
    """Per-substrate ``(tag, amine_thermo_dict, n_imag, amine_xyz_path)`` from a run dir."""
    refs = []
    for cache_path in sorted(run_dir.glob("*/gas_thermo.json")):
        task_dir = cache_path.parent
        cache = json.loads(cache_path.read_text())
        amine = cache.get("species", {}).get("amine")
        if not amine:
            continue
        xyz = task_dir / amine.get("geometry", "amine_opt.xyz")
        if not xyz.exists():
            continue
        refs.append((task_dir.name, cache, cache.get("n_imag_amine"), xyz))
    return refs


def extract(
    run_dir: Path, backend: str, level: str
) -> tuple[str, Psi4Thermo, object, int, dict]:
    """Read + cross-check the amine reference across a run; return what :func:`store` needs.

    Returns ``(amine_smiles, thermo, atoms, n_imag, provenance)``. Raises :class:`SeedError`
    if no eligible substrate is found or the amine reference is not consistent across the run.
    """
    refs = _collect_amine_refs(run_dir)
    if not refs:
        raise SeedError(
            f"no substrate under {run_dir} has both gas_thermo.json (species.amine) and "
            f"amine_opt.xyz -- run the gas backbone first, or point at a different run"
        )

    amine_smiles = refs[0][1]["amine_smiles"]
    ref_energy = refs[0][1]["species"]["amine"]["electronic_energy"]
    for tag, cache, _n_imag, _xyz in refs:
        if cache["amine_smiles"] != amine_smiles:
            raise SeedError(
                f"{tag}: amine SMILES {cache['amine_smiles']!r} != {amine_smiles!r} "
                f"(run mixes amines; seed per amine)"
            )
        drift = abs(cache["species"]["amine"]["electronic_energy"] - ref_energy)
        if drift > _ENERGY_TOL_HARTREE:
            raise SeedError(
                f"{tag}: amine electronic energy drifts {drift:.2e} Eh from the first "
                f"substrate -- the amine reference is not constant across this run"
            )

    tag0, cache0, n_imag0, xyz0 = refs[0]
    amine = cache0["species"]["amine"]
    thermo = Psi4Thermo(
        electronic_energy=amine["electronic_energy"],
        gibbs=amine["gibbs"],
        gibbs_qh=amine["gibbs_qh"],
        enthalpy=amine["enthalpy"],
        zpve=amine["zpve"],
        frequencies=[],
    )
    from ase.io import read  # noqa: PLC0415 -- lazy

    atoms = read(str(xyz0), format="extxyz")
    atoms.info["charge"] = int(round(float(atoms.info.get("charge", 0))))
    provenance = {
        "seeded_by": "scripts/extract_amine_ref.py",
        "source_run": str(run_dir),
        "source_substrate": tag0,
        "n_substrates_checked": len(refs),
        "note": "copied from a finished gas run (no recomputation)",
    }
    return amine_smiles, thermo, atoms, int(n_imag0 or 0), provenance


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        required=True,
        help="finished gas run dir (per-substrate gas_thermo.json)",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("psi4", "gpu4pyscf"),
        help="engine that produced the run (absolute energies are backend-specific)",
    )
    parser.add_argument(
        "--level",
        default="b3lyp-d3bj_def2-svp",
        help="level-of-theory tag (default the POC's B3LYP-D3BJ/def2-SVP)",
    )
    parser.add_argument(
        "--cache-dir",
        help="cache root (default $SNAR_QC_AMINE_CACHE or assets/amine_ref)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing cached reference"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report only; write nothing"
    )
    args = parser.parse_args(argv)

    if args.cache_dir:
        import os

        os.environ[amine_cache._CACHE_ENV] = args.cache_dir

    run_dir = Path(args.run).resolve()
    try:
        amine_smiles, thermo, atoms, n_imag, provenance = extract(
            run_dir, args.backend, args.level
        )
    except SeedError as exc:
        print(f"[fail] {exc}")
        return 2

    dest_json, _dest_xyz = amine_cache._paths(amine_smiles, args.backend, args.level)
    print(
        f"Amine reference: {amine_smiles!r}  backend={args.backend}  level={args.level}\n"
        f"  source: {provenance['source_run']} "
        f"({provenance['n_substrates_checked']} substrate(s) checked, consistent)\n"
        f"  E_e={thermo.electronic_energy:.8f}  G_qh={thermo.gibbs_qh:.8f} Eh  "
        f"n_imag={n_imag}\n"
        f"  dest: {dest_json}"
    )
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0
    if dest_json.exists() and not args.force:
        print(f"[skip] {dest_json.name} exists (use --force)")
        return 0

    written = amine_cache.store(
        amine_smiles,
        args.backend,
        args.level,
        thermo,
        atoms,
        n_imag,
        provenance=provenance,
    )
    print(f"[ ok ] wrote {written} (+ {written.with_suffix('.xyz').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
