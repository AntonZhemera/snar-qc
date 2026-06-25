#!/usr/bin/env python
"""Backfill the reusable gas cache (``gas_thermo.json`` + ``*_opt.xyz``) for runs that
predate geometry persistence -- **without recomputing any QC**.

Why this exists
---------------
The ``run_poc`` / ``run_qc_queue`` pipeline gained a *gas cache* so a finished gas
backbone can be re-evaluated in another solvent/model for the price of three single
points (see ``snar_qc.poc.barrier._write_gas_cache`` and ``scripts/sweep_solvent.py``).
The cache is two artefacts per substrate: the gas-optimised geometries
(``ts_opt.xyz`` / ``arx_opt.xyz`` / ``amine_opt.xyz``, extended XYZ carrying the charge)
plus ``gas_thermo.json`` (the per-species gas thermochemistry + geometry refs).

The real-pool CPU/Psi4 campaign (``docs/runbook_realpool_qc.md``) was already running when
that persistence landed, so its per-task archives carry the **expensive** result -- the
gas TS/ArX/amine opt+freq -- but not in the cache's tidy form. Nothing is lost, though:
the optimised geometry is in each Psi4 ``ts.out`` / ``arx.out`` / ``amine.out`` (every
optimisation step prints a ``Geometry (in Angstrom)`` block; the last one is the relaxed
geometry), and the gas thermochemistry is in the same file's thermochemistry block. This
tool reconstructs the cache from those archived ``*.out`` files plus ``result.json``, so
runs that finished (or are still finishing) become sweepable like a native run -- with
**no** recomputation.

How it reconstructs each species (gas phase)
--------------------------------------------
- **geometry / charge** -- the last ``Geometry (in Angstrom), charge = q`` block of the
  ``*.out`` (the relaxed geometry the frequency job used), written back through the exact
  extended-XYZ writer the sweep reads (charge in ``info``).
- **electronic_energy / enthalpy / gibbs / zpve** -- parsed straight from the Psi4
  ``==> Thermochemistry Energy Analysis <==`` block (``Total E_e`` / ``Total H`` /
  ``Total G`` / ``Correction ZPVE to E_e``), all gas phase (the opt+freq is gas; the PCM
  single point is a separate ``*_pcm.out``).
- **gibbs_qh** -- recovered from ``result.json`` by undoing the implicit-solvent
  single-point shift: ``G_qh(gas) = E_e(gas) + (G_qh(solv) - E(solv))``. This reuses the
  run's *own* quasi-harmonic Gibbs (Grimme qRRHO + any soft-imaginary folding done at
  runtime), so it is exact rather than re-derived from scraped frequencies. For a gas run
  the shift is zero and the identity still holds.

Each substrate is checked before its cache is written (atom-count balance
``n(ts) == n(arx) + n(amine)``; thermochemical ordering; and a round-trip that re-applies
the solvent shift and reproduces ``result.json``'s ``delta_g_qh_kcal`` to <0.01 kcal/mol).
A substrate that fails any check is reported and skipped; it never sinks the batch.

Caveats
-------
- Only ``status == "completed"`` substrates (all three species + a barrier) are eligible.
- ``electronic_energy`` etc. carry Psi4's printed precision (8 dp, ~1e-5 kcal/mol -- far
  below method error and the solvent-shift size).
- If a species had a *soft* imaginary mode folded into its thermochemistry at runtime
  (``n_imag_*_soft > 0``), the parsed gas ``gibbs`` / ``enthalpy`` / ``zpve`` are Psi4's
  raw (unfolded) values, so they differ slightly from the live run; ``gibbs_qh`` (the
  headline number) is still exact via the shift identity above. The tool **warns** when
  this applies. (It does not apply to the real-pool DMSO cohort, where all soft counts
  are 0.)

This depends only on ASE + ``predict_snar.data`` -- it does **not** import Psi4, so it
runs in either QC env (``snar-qc`` or ``gpuqc``).

Run inside a QC conda env. Examples::

    # In place over a finished/active run directory (per-substrate subdirs):
    python scripts/extract_gas_cache.py --run data/processed/realpool_dmso

    # From the per-task archive zips (e.g. the runbook --archive-dir / a datadump mirror):
    python scripts/extract_gas_cache.py \\
        --archive ../datadump/snar-qc-runs/realpool_dmso \\
        --outdir  data/processed/realpool_dmso_gas

Then sweep a solvent off the reconstructed cache (use the **same engine** the geometries
were optimised with -- Psi4, i.e. the ``snar-qc`` env -- to stay method-consistent)::

    python scripts/sweep_solvent.py --gas-run data/processed/realpool_dmso_gas \\
        --solvent DMSO --solvent-model iefpcm --outdir data/processed/realpool_dmso_sweep
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional

# Make ``snar_qc`` / ``predict_snar`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_snar.data import HARTREE_TO_KCAL  # noqa: E402  (lightweight; no Psi4)

# --- Cache contract -------------------------------------------------------------------
# These MIRROR snar_qc.poc.barrier._GAS_CACHE_FILE / _GEOMETRY_FILES so the cache this
# tool writes is byte-compatible with what scripts/sweep_solvent.py consumes. They are
# duplicated (not imported) only to keep this tool free of the Psi4 import chain that
# barrier pulls in; keep them in sync with barrier if that contract ever changes.
GAS_CACHE_FILE = "gas_thermo.json"
GEOMETRY_FILES = {"ts": "ts_opt.xyz", "arx": "arx_opt.xyz", "amine": "amine_opt.xyz"}
OUT_FILES = {"ts": "ts.out", "arx": "arx.out", "amine": "amine.out"}
SPECIES = ("ts", "arx", "amine")

# --- Psi4 .out parsing ----------------------------------------------------------------
_GEO_HEADER = re.compile(r"Geometry \(in Angstrom\), charge = (-?\d+), multiplicity")
_COORD = re.compile(
    r"^\s*([A-Za-z]{1,3})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
_THERMO_PATTERNS = {
    "electronic_energy": r"Total E_e, Electronic energy at well bottom\s+(-?\d+\.\d+)",
    "enthalpy": r"Total H, Enthalpy at\s+[\d.]+ \[K\]\s+(-?\d+\.\d+)",
    "gibbs": r"Total G, Gibbs energy at\s+[\d.]+ \[K\]\s+(-?\d+\.\d+)",
    "zpve": (
        r"Correction ZPVE to E_e\s+[\d.]+ \[kcal/mol\]\s+[\d.]+ \[kJ/mol\]\s+"
        r"(-?\d+\.\d+) \[Eh\]"
    ),
}


class ExtractError(RuntimeError):
    """A substrate could not be reconstructed faithfully (reported, then skipped)."""


def _parse_final_geometry(text: str) -> tuple[list[str], list[list[float]], int]:
    """Element symbols, Cartesian positions (Angstrom) and charge of the *last*
    ``Geometry (in Angstrom)`` block -- the relaxed geometry the frequency job used.

    The clean 3-column ``Geometry (in Angstrom)`` block (symbol + x/y/z) is matched; the
    interleaved ``==> Geometry <==`` tables (which carry a 4th mass column) are skipped
    by the exact "three floats then end of line" shape of :data:`_COORD`.
    """
    lines = text.splitlines()
    header_idx: Optional[int] = None
    charge = 0
    for i, line in enumerate(lines):
        match = _GEO_HEADER.search(line)
        if match:
            header_idx = i
            charge = int(match.group(1))
    if header_idx is None:
        raise ExtractError("no 'Geometry (in Angstrom)' block found")

    symbols: list[str] = []
    positions: list[list[float]] = []
    started = False
    for line in lines[header_idx + 1 :]:
        match = _COORD.match(line)
        if match:
            started = True
            symbols.append(match.group(1).capitalize())  # CL -> Cl, BR -> Br, C -> C
            positions.append(
                [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            )
        elif started:
            break
    if not symbols:
        raise ExtractError("geometry block had no coordinate rows")
    return symbols, positions, charge


def _parse_gas_thermo(text: str) -> dict[str, float]:
    """The gas thermochemistry terms (Hartree) from the Psi4 thermochemistry block.

    The last match of each pattern is taken (one opt+freq writes one block).
    """
    out: dict[str, float] = {}
    for key, pattern in _THERMO_PATTERNS.items():
        hits = re.findall(pattern, text)
        if not hits:
            raise ExtractError(f"thermochemistry term '{key}' not found in .out")
        out[key] = float(hits[-1])
    return out


# --- extended-XYZ writer (mirrors barrier._persist_geometry) --------------------------
def _persist_geometry(
    symbols: list[str], positions: list[list[float]], charge: int, path: Path
) -> None:
    """Write an optimised geometry to extended XYZ, preserving ``info["charge"]``.

    Identical in effect to ``snar_qc.poc.barrier._persist_geometry`` so the file
    round-trips through the sweep's ``_read_geometry`` (extended XYZ keeps the charge in
    the comment line). ASE is imported lazily so the heavy import is paid only on write.
    """
    from ase import Atoms  # noqa: PLC0415 -- lazy
    from ase.io import write  # noqa: PLC0415 -- lazy

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info["charge"] = int(charge)
    write(str(path), atoms, format="extxyz")


# --- core reconstruction --------------------------------------------------------------
def build_gas_cache(
    tag: str,
    read_text: Callable[[str], str],
    dest_dir: Path,
) -> dict:
    """Reconstruct ``gas_thermo.json`` + ``*_opt.xyz`` for one substrate into ``dest_dir``.

    Args:
        tag: Substrate tag (catalogue code / ``lu_id`` subdir name).
        read_text: Reads a member of the substrate's task dir by basename (e.g.
            ``"result.json"`` / ``"ts.out"``), returning its text. Abstracts over an
            on-disk run directory vs. a per-task archive zip.
        dest_dir: Where the cache + geometries are written.

    Returns:
        The written cache dict.

    Raises:
        ExtractError: if the run is not eligible or any faithfulness check fails.
    """
    result = json.loads(read_text("result.json"))
    if result.get("status") != "completed":
        raise ExtractError(f"status={result.get('status')!r} (not a completed barrier)")

    species_thermo: dict[str, dict[str, float]] = {}
    natoms: dict[str, int] = {}
    geometries: dict[str, tuple[list[str], list[list[float]], int]] = {}

    for sp in SPECIES:
        out_text = read_text(OUT_FILES[sp])
        symbols, positions, charge = _parse_final_geometry(out_text)
        thermo = _parse_gas_thermo(out_text)
        natoms[sp] = len(symbols)
        geometries[sp] = (symbols, positions, charge)

        # gibbs_qh: undo the implicit-solvent single-point shift on the run's own qRRHO
        # Gibbs (G_qh(gas) = E_e(gas) + (G_qh(solv) - E(solv))). Exact -- reuses the
        # runtime qRRHO/soft-imag folding rather than re-deriving from frequencies.
        gibbs_qh = thermo["electronic_energy"] + (
            result[f"{sp}_gibbs_qh_hartree"] - result[f"{sp}_energy_hartree"]
        )
        species_thermo[sp] = {
            "electronic_energy": thermo["electronic_energy"],
            "gibbs": thermo["gibbs"],
            "gibbs_qh": gibbs_qh,
            "enthalpy": thermo["enthalpy"],
            "zpve": thermo["zpve"],
        }

        # Per-species sanity: H > E_e, G < H, ZPVE > 0, and qRRHO close to harmonic G.
        if not (thermo["enthalpy"] > thermo["electronic_energy"] > -1e6):
            raise ExtractError(f"{sp}: enthalpy !> electronic_energy")
        if not thermo["gibbs"] < thermo["enthalpy"]:
            raise ExtractError(f"{sp}: gibbs !< enthalpy")
        if not thermo["zpve"] > 0:
            raise ExtractError(f"{sp}: zpve !> 0")
        if abs(gibbs_qh - thermo["gibbs"]) > 0.05:
            raise ExtractError(
                f"{sp}: |gibbs_qh - gibbs| too large ({gibbs_qh - thermo['gibbs']:.4f} Eh)"
            )

    # Atom-count balance: the TS is the aryl halide plus the amine.
    if natoms["ts"] != natoms["arx"] + natoms["amine"]:
        raise ExtractError(
            f"atom-count imbalance: ts={natoms['ts']} != arx={natoms['arx']} + amine={natoms['amine']}"
        )

    # Round-trip: re-apply the solvent electronic shift and reproduce result.json's barrier.
    solv_gibbs_qh = {
        sp: species_thermo[sp]["gibbs_qh"]
        + (result[f"{sp}_energy_hartree"] - species_thermo[sp]["electronic_energy"])
        for sp in SPECIES
    }
    recomputed = (
        solv_gibbs_qh["ts"] - solv_gibbs_qh["arx"] - solv_gibbs_qh["amine"]
    ) * HARTREE_TO_KCAL
    delta = abs(recomputed - result["delta_g_qh_kcal"])
    if delta > 1e-2:
        raise ExtractError(
            f"round-trip ΔG‡(qh) mismatch: {recomputed:.4f} vs result {result['delta_g_qh_kcal']:.4f} "
            f"(Δ={delta:.4f} kcal/mol) -- parse likely picked the wrong file/field"
        )

    # Assemble the cache (schema mirrors barrier._write_gas_cache) and write artefacts.
    dest_dir.mkdir(parents=True, exist_ok=True)
    for sp in SPECIES:
        symbols, positions, charge = geometries[sp]
        _persist_geometry(symbols, positions, charge, dest_dir / GEOMETRY_FILES[sp])

    cache = {
        "lu_id": result.get("lu_id"),
        "aryl_halide_smiles": result["aryl_halide_smiles"],
        "amine_smiles": result["amine_smiles"],
        "leaving_group": result["leaving_group"],
        "central_atom": result["central_atom"],
        "nu_atom": result["nu_atom"],
        "lg_atom": result["lg_atom"],
        "coordinate": result.get("coordinate", "concerted"),
        "reference": result.get("reference", "separated_reactants"),
        "peak_index": result.get("peak_index"),
        "n_imag_ts": result.get("n_imag_ts"),
        "n_imag_ts_soft": result.get("n_imag_ts_soft"),
        "ts_imag_freq_cm": result.get("ts_imag_freq_cm"),
        "n_imag_arx": result.get("n_imag_arx"),
        "n_imag_amine": result.get("n_imag_amine"),
        "species": {
            sp: {"geometry": GEOMETRY_FILES[sp], **species_thermo[sp]} for sp in SPECIES
        },
        # Not read by the sweep; marks this cache as a no-recompute backfill for provenance.
        "_provenance": {
            "reconstructed_by": "scripts/extract_gas_cache.py",
            "source": "psi4 *.out + result.json (no recomputation)",
            "roundtrip_dG_qh_kcal": round(recomputed, 6),
        },
    }
    (dest_dir / GAS_CACHE_FILE).write_text(json.dumps(cache, indent=2))
    return cache


# --- input adapters (on-disk run dir / archive zips) ----------------------------------
def _dir_reader(task_dir: Path) -> Callable[[str], str]:
    def read(name: str) -> str:
        path = task_dir / name
        if not path.exists():
            raise ExtractError(f"missing {name} in {task_dir}")
        return path.read_text()

    return read


def _zip_reader(zip_path: Path, tag: str) -> Callable[[str], str]:
    zf = zipfile.ZipFile(zip_path)
    members = set(zf.namelist())

    def read(name: str) -> str:
        member = f"{tag}/{name}"
        if member not in members:
            raise ExtractError(f"missing {name} in {zip_path.name}")
        return zf.read(member).decode()

    return read


def _iter_run_tasks(
    run_dir: Path, only: Optional[set[str]]
) -> list[tuple[str, Callable[[str], str], Path]]:
    tasks = []
    for sidecar in sorted(run_dir.glob("*/result.json")):
        task_dir = sidecar.parent
        tag = task_dir.name
        if only and tag not in only and tag.removeprefix("lu_") not in only:
            continue
        tasks.append((tag, _dir_reader(task_dir), task_dir))
    return tasks


def _iter_archive_tasks(
    archive_dir: Path, outdir: Path, only: Optional[set[str]]
) -> list[tuple[str, Callable[[str], str], Path]]:
    tasks = []
    for zip_path in sorted(archive_dir.glob("*.zip")):
        tag = zip_path.stem
        if only and tag not in only and tag.removeprefix("lu_") not in only:
            continue
        tasks.append((tag, _zip_reader(zip_path, tag), outdir / tag))
    return tasks


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run", help="run dir with per-substrate subdirs (result.json + *.out)"
    )
    src.add_argument("--archive", help="dir of per-task <tag>.zip archives")
    parser.add_argument(
        "--outdir",
        help="where to write the cache (default: in place for --run; required for --archive)",
    )
    parser.add_argument("--only", help="comma-separated tag / lu_id filter")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing gas cache"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report only; write nothing"
    )
    args = parser.parse_args(argv)

    only = {s.strip() for s in args.only.split(",")} if args.only else None

    if args.run:
        run_dir = Path(args.run).resolve()
        outdir = Path(args.outdir).resolve() if args.outdir else None
        tasks = _iter_run_tasks(run_dir, only)
        # In place: write into each task dir unless an --outdir was given.
        if outdir is not None:
            tasks = [(tag, reader, outdir / tag) for tag, reader, _ in tasks]
    else:
        if not args.outdir:
            parser.error("--archive requires --outdir")
        archive_dir = Path(args.archive).resolve()
        outdir = Path(args.outdir).resolve()
        tasks = _iter_archive_tasks(archive_dir, outdir, only)

    if not tasks:
        print("No eligible tasks found.")
        return 1

    print(
        f"Backfilling gas cache for {len(tasks)} task(s)"
        + (" [dry-run]" if args.dry_run else "")
    )
    n_done = n_skip = n_fail = n_warn = 0
    for tag, reader, dest in tasks:
        if (dest / GAS_CACHE_FILE).exists() and not args.force and not args.dry_run:
            print(f"[skip] {tag}: {GAS_CACHE_FILE} exists (use --force)")
            n_skip += 1
            continue
        try:
            if args.dry_run:
                # Reconstruct into a throwaway dir to exercise every check without writing.
                import tempfile

                with tempfile.TemporaryDirectory() as tmp:
                    cache = build_gas_cache(tag, reader, Path(tmp))
            else:
                cache = build_gas_cache(tag, reader, dest)
        except ExtractError as exc:
            print(f"[fail] {tag}: {exc}")
            n_fail += 1
            continue
        except KeyError as exc:
            print(f"[fail] {tag}: missing result.json field {exc}")
            n_fail += 1
            continue

        soft = cache.get("n_imag_ts_soft") or 0
        warn = ""
        if soft:
            warn = f"  [warn] {soft} soft imag mode(s): gas G/H/ZPVE unfolded (gibbs_qh still exact)"
            n_warn += 1
        verb = "would write" if args.dry_run else "wrote"
        print(
            f"[ ok ] {tag}: {verb} cache (ΔG‡(qh) round-trip {cache['_provenance']['roundtrip_dG_qh_kcal']}){warn}"
        )
        n_done += 1

    print(
        f"\nDone: {n_done} reconstructed, {n_skip} skipped, {n_fail} failed"
        + (f", {n_warn} with soft-imag warnings" if n_warn else "")
    )
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
