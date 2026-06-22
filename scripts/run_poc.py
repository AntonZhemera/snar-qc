#!/usr/bin/env python
"""Resumable POC runner: SMILES (aryl halide) + model amine -> ΔG‡.

Drives :func:`snar_qc.poc.barrier.compute_barrier` over a set of substrates, one
per-substrate scratch directory, writing a JSON **sidecar** (``result.json``) per
substrate. Re-running skips substrates that already have a sidecar, so a crashed or slow
transition-state search never loses the rest of the batch (``--retry`` re-runs the ones
that did not complete; ``--force`` re-runs everything).

Two input modes:

* ``--smiles SMILES`` -- a single ad-hoc substrate (the Stage 4a smoke test, e.g.
  ``--smiles "O=[N+]([O-])c1ccc(F)cc1" --leaving-group F``).
* ``--substrates CSV`` -- a CSV with at least ``smiles_canonical`` and ``leaving_group``
  columns (and optionally ``lu_id``), e.g. the Lu_74 slice (the Stage 4b batch).

Run inside the ``snar-qc`` conda env. Example:

    python scripts/run_poc.py --substrates data/external/lu74_poc_slice.csv \\
        --outdir data/processed/poc_run --n-procs 8 --mem 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make ``snar_qc`` / ``predict_snar`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snar_qc.poc.barrier import compute_barrier  # noqa: E402
from snar_qc.poc.complex import (  # noqa: E402
    DEFAULT_AMINE_SMILES,
    build_reaction_complex,
)

# A sidecar with one of these statuses counts as "done" -- not re-run unless --force.
_TERMINAL_STATUSES = {"completed", "no_peak", "ts_not_saddle", "error"}


def _slug(text: str) -> str:
    """A filesystem-safe short tag derived from a SMILES string."""
    keep = [c if c.isalnum() else "_" for c in text]
    return "".join(keep)[:48].strip("_")


def _load_substrates(args: argparse.Namespace) -> list[dict]:
    """Build the substrate work-list from --smiles or --substrates."""
    if args.smiles:
        return [
            {
                "lu_id": args.lu_id,
                "smiles_canonical": args.smiles,
                "leaving_group": args.leaving_group,
            }
        ]
    rows: list[dict] = []
    with open(args.substrates, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        rows = [r for r in rows if str(r.get("lu_id", "")).strip() in wanted]
    return rows


def _tag(row: dict) -> str:
    """Per-substrate directory tag: lu_<id> when available, else a SMILES slug."""
    lu_id = row.get("lu_id")
    if lu_id not in (None, ""):
        return f"lu_{lu_id}"
    return _slug(row["smiles_canonical"])


def _should_skip(sidecar: Path, retry: bool, force: bool) -> Optional[str]:
    """Return the existing status if the substrate should be skipped, else None."""
    if force or not sidecar.exists():
        return None
    try:
        status = json.loads(sidecar.read_text()).get("status")
    except (json.JSONDecodeError, OSError):
        return None
    if status == "completed":
        return status
    if status in _TERMINAL_STATUSES and not retry:
        return status
    return None


def _run_one(row: dict, args: argparse.Namespace) -> dict:
    """Build the complex and compute ΔG‡ for a single substrate, writing its sidecar."""
    tag = _tag(row)
    workdir = Path(args.outdir).resolve() / tag
    workdir.mkdir(parents=True, exist_ok=True)
    sidecar = workdir / "result.json"

    skip_status = _should_skip(sidecar, args.retry, args.force)
    if skip_status is not None:
        print(f"[skip] {tag}: already {skip_status}")
        return json.loads(sidecar.read_text())

    lu_id = row.get("lu_id")
    lu_id = int(lu_id) if lu_id not in (None, "") else None
    leaving_group = (row.get("leaving_group") or "").strip() or None

    print(f"[run ] {tag}: {row['smiles_canonical']} (LG={leaving_group})")
    rc = build_reaction_complex(
        row["smiles_canonical"],
        amine_smiles=args.amine,
        leaving_group=leaving_group,
        approach=args.approach,
    )
    # Keep the input complex geometry for audit.
    from ase.io import write as ase_write

    ase_write(str(workdir / "complex.xyz"), rc.atoms)

    cwd = os.getcwd()
    os.chdir(workdir)
    started = time.time()
    try:
        result = compute_barrier(
            rc,
            scan_steps=args.scan_steps,
            scan_stop=args.scan_stop,
            scan_stop_lg=args.scan_stop_lg,
            n_procs=args.n_procs,
            mem=args.mem,
            lu_id=lu_id,
            solvent=args.solvent,
            coordinate=args.coordinate,
        )
    finally:
        os.chdir(cwd)

    payload = result.to_dict()
    payload["wall_s"] = round(time.time() - started, 1)
    sidecar.write_text(json.dumps(payload, indent=2))
    print(
        f"[done] {tag}: status={payload['status']} "
        f"ΔG‡(qh)={payload.get('delta_g_qh_kcal')} "
        f"n_imag_ts={payload.get('n_imag_ts')} ({payload['wall_s']}s)"
    )
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="single aryl-halide SMILES (ad-hoc smoke test)")
    src.add_argument("--substrates", help="CSV with smiles_canonical / leaving_group")
    parser.add_argument("--leaving-group", help="leaving halide element for --smiles")
    parser.add_argument("--lu-id", type=int, help="optional id for --smiles")
    parser.add_argument("--only", help="comma-separated lu_id filter for --substrates")
    parser.add_argument(
        "--amine", default=DEFAULT_AMINE_SMILES, help="model amine SMILES (default CN)"
    )
    parser.add_argument("--outdir", default="data/processed/poc_run")
    parser.add_argument("--n-procs", type=int, default=8)
    parser.add_argument("--mem", type=float, default=12.0)
    parser.add_argument("--scan-steps", type=int, default=14)
    parser.add_argument("--scan-stop", type=float, default=1.45)
    parser.add_argument("--scan-stop-lg", type=float, default=2.6)
    parser.add_argument("--approach", type=float, default=3.0)
    parser.add_argument(
        "--solvent",
        default=None,
        help="PCMSolver solvent name for implicit solvation (e.g. DMSO); "
        "omit for gas phase",
    )
    parser.add_argument(
        "--coordinate",
        choices=("concerted", "addition"),
        default="concerted",
        help="relaxed-scan coordinate: concerted d(C-Nu)-d(C-LG) (default) or "
        "addition-only C...Nu",
    )
    parser.add_argument(
        "--retry", action="store_true", help="re-run substrates that did not complete"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-run all substrates from scratch"
    )
    args = parser.parse_args(argv)

    rows = _load_substrates(args)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    print(
        f"POC runner: {len(rows)} substrate(s), amine={args.amine}, "
        f"solvent={args.solvent or 'gas'}, coordinate={args.coordinate}, "
        f"outdir={args.outdir}"
    )

    summary = []
    for row in rows:
        try:
            summary.append(_run_one(row, args))
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- one bad substrate must not stop batch
            tag = _tag(row)
            print(f"[FAIL] {tag}: {type(exc).__name__}: {exc}")
            summary.append({"tag": tag, "status": "error", "error": str(exc)})

    # Batch roll-up next to the per-substrate sidecars.
    roll = Path(args.outdir) / "summary.json"
    roll.write_text(json.dumps(summary, indent=2))
    done = sum(1 for s in summary if s.get("status") == "completed")
    print(f"\nBatch complete: {done}/{len(summary)} reached a confirmed saddle.")
    print(f"Roll-up: {roll}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
