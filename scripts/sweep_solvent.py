#!/usr/bin/env python
"""Solvent sweep: re-evaluate ΔG‡ in a continuum solvent by reusing a cached gas run.

The SP-on-gas recipe makes the gas backbone (xTB scan, gas TS/ArX/amine opt+freq)
**solvent-independent**, so a different solvent or continuum model only needs the three
implicit-solvent single points on the cached gas geometries -- minutes per substrate
instead of the ~30+ a full gas+solvent run takes. This runner reuses a completed gas run
(one that wrote ``gas_thermo.json`` + ``*_opt.xyz`` per substrate, i.e. produced by the
current ``run_poc.py``) and evaluates one (solvent, model) on top of it.

Resumable, like ``run_poc.py``: each substrate gets a ``result.json`` sidecar in --outdir;
a terminal sidecar is skipped unless ``--retry`` / ``--force``.

Run inside the active QC env (gpuqc for GPU IEF-PCM/SMD). Example -- evaluate DMSO under
both models from one gas run::

    python scripts/sweep_solvent.py --gas-run data/processed/gpu_stage_e \\
        --solvent DMSO --solvent-model iefpcm --outdir data/processed/gpu_dmso_iefpcm
    python scripts/sweep_solvent.py --gas-run data/processed/gpu_stage_e \\
        --solvent DMSO --solvent-model smd    --outdir data/processed/gpu_dmso_smd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make ``snar_qc`` / ``predict_snar`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snar_qc.poc.barrier import _GAS_CACHE_FILE, solvent_sweep  # noqa: E402
from snar_qc.poc.worker import should_skip  # noqa: E402


def _gas_substrate_dirs(gas_run: Path, only: Optional[set[str]]) -> list[Path]:
    """Per-substrate subdirs of a gas run that carry a reusable gas cache."""
    dirs = sorted(p.parent for p in gas_run.glob(f"*/{_GAS_CACHE_FILE}"))
    if only:
        dirs = [d for d in dirs if d.name in only or d.name.removeprefix("lu_") in only]
    return dirs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gas-run", required=True, help="completed gas run dir (per-substrate subdirs)"
    )
    parser.add_argument("--outdir", required=True, help="where to write the solvent run")
    parser.add_argument("--solvent", required=True, help="continuum solvent name, e.g. DMSO")
    parser.add_argument(
        "--solvent-model",
        default=None,
        help="continuum model: iefpcm (default) or smd (GPU backend only)",
    )
    parser.add_argument("--n-procs", type=int, default=8)
    parser.add_argument("--mem", type=float, default=12.0)
    parser.add_argument("--only", help="comma-separated lu_id / tag filter")
    parser.add_argument(
        "--retry", action="store_true", help="re-run substrates that did not complete"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-run all substrates from scratch"
    )
    args = parser.parse_args(argv)

    gas_run = Path(args.gas_run).resolve()
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    sub_dirs = _gas_substrate_dirs(gas_run, only)
    if not sub_dirs:
        print(f"No gas caches ({_GAS_CACHE_FILE}) under {gas_run}.")
        return 1

    outroot = Path(args.outdir).resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    model = args.solvent_model or "iefpcm (default)"
    print(
        f"Solvent sweep: {len(sub_dirs)} substrate(s) from {gas_run}, "
        f"solvent={args.solvent}/{model}, outdir={outroot}"
    )

    summary = []
    for gas_dir in sub_dirs:
        tag = gas_dir.name
        workdir = outroot / tag
        workdir.mkdir(parents=True, exist_ok=True)
        sidecar = workdir / "result.json"

        skip = should_skip(sidecar, args.retry, args.force)
        if skip is not None:
            print(f"[skip] {tag}: already {skip}")
            payload = json.loads(sidecar.read_text())
            payload["skipped"] = skip
            summary.append(payload)
            continue

        print(f"[run ] {tag}: {args.solvent}/{model}")
        cwd = os.getcwd()
        os.chdir(workdir)
        started = time.time()
        try:
            result = solvent_sweep(
                gas_dir, args.solvent, args.solvent_model, args.n_procs, args.mem
            )
            payload = result.to_dict()
        except Exception as exc:  # noqa: BLE001 -- one bad substrate must not stop batch
            payload = {"tag": tag, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            os.chdir(cwd)

        payload["tag"] = tag
        payload["wall_s"] = round(time.time() - started, 1)
        payload["gas_dir"] = str(gas_dir)
        sidecar.write_text(json.dumps(payload, indent=2))
        print(
            f"[done] {tag}: status={payload.get('status')} "
            f"ΔG‡(qh)={payload.get('delta_g_qh_kcal')} ({payload['wall_s']}s)"
        )
        summary.append(payload)

    roll = outroot / "summary.json"
    roll.write_text(json.dumps(summary, indent=2))
    done = sum(1 for s in summary if s.get("status") == "completed")
    print(f"\nSweep complete: {done}/{len(summary)} completed. Roll-up: {roll}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
