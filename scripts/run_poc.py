#!/usr/bin/env python
"""Resumable POC runner: SMILES (aryl halide) + model amine -> ΔG‡.

Drives :func:`snar_qc.poc.worker.run_substrate` over a set of substrates,
sequentially, one per-substrate scratch directory with a JSON **sidecar**
(``result.json``). Re-running skips substrates that already have a terminal
sidecar (``--retry`` re-runs the ones that did not complete; ``--force`` re-runs
everything). The per-substrate work itself lives in ``snar_qc.poc.worker`` so the
sequential runner and the shared-queue orchestrator (``scripts/run_qc_queue.py``)
share one code path.

Two input modes:

* ``--smiles SMILES`` -- a single ad-hoc substrate, e.g.
  ``--smiles "O=[N+]([O-])c1ccc(F)cc1" --leaving-group F``.
* ``--substrates CSV`` -- a CSV with at least ``smiles_canonical`` and
  ``leaving_group`` columns (and optionally ``lu_id``), e.g. the Lu_74 slice.

Run inside the ``snar-qc`` conda env. Example:

    python scripts/run_poc.py --substrates data/external/lu74_poc_slice.csv \\
        --outdir data/processed/poc_run --n-procs 8 --mem 12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

# Make ``snar_qc`` / ``predict_snar`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snar_qc.poc.complex import DEFAULT_AMINE_SMILES  # noqa: E402
from snar_qc.poc.worker import WorkerConfig, run_substrate, task_tag  # noqa: E402


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
    with open(args.substrates, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        rows = [r for r in rows if str(r.get("lu_id", "")).strip() in wanted]
    return rows


def _config(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        outdir=args.outdir,
        amine=args.amine,
        approach=args.approach,
        scan_steps=args.scan_steps,
        scan_stop=args.scan_stop,
        scan_stop_lg=args.scan_stop_lg,
        n_procs=args.n_procs,
        mem=args.mem,
        solvent=args.solvent,
        solvent_model=args.solvent_model,
        coordinate=args.coordinate,
        retry=args.retry,
        force=args.force,
        resume=args.resume,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="single aryl-halide SMILES (ad-hoc smoke test)")
    src.add_argument("--substrates", help="CSV with smiles_canonical / leaving_group")
    parser.add_argument(
        "--leaving-group",
        help="leaving group for --smiles: a halide element (F/Cl/Br/I) or NO2 for ipso "
        "nitro displacement",
    )
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
        help="continuum solvent name for implicit solvation (e.g. DMSO); "
        "omit for gas phase",
    )
    parser.add_argument(
        "--solvent-model",
        default=None,
        help="continuum model for --solvent: iefpcm (default, matches the cpu_dmso "
        "Psi4 baseline) or smd (GPU backend only). Omit to use the backend default.",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="within a substrate, skip stages already checkpointed in its progress.json "
        "(scan, TS opt+freq, ArX opt+freq) after an interrupted run; --force overrides",
    )
    args = parser.parse_args(argv)

    rows = _load_substrates(args)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    print(
        f"POC runner: {len(rows)} substrate(s), amine={args.amine}, "
        f"solvent={args.solvent or 'gas'}"
        f"{('/' + args.solvent_model) if (args.solvent and args.solvent_model) else ''}, "
        f"coordinate={args.coordinate}, outdir={args.outdir}"
    )

    cfg = _config(args)
    summary = []
    for row in rows:
        try:
            summary.append(run_substrate(row, cfg, log=print))
        except Exception as exc:  # noqa: BLE001 -- one bad substrate must not stop batch
            tag = task_tag(row)
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
