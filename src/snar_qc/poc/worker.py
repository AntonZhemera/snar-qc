"""Per-substrate ΔG‡ work unit, shared by the sequential and queued runners.

Extracted from ``scripts/run_poc.py`` so that the sequential runner and the
shared-queue orchestrator (``scripts/run_qc_queue.py``) drive the *same*
resumable computation: build the reaction complex, run ``compute_barrier``, and
write the per-substrate ``result.json`` sidecar. Keeping one code path means the
resume / skip semantics and the sidecar schema can't drift between the two.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from snar_qc.poc.barrier import compute_barrier
from snar_qc.poc.complex import DEFAULT_AMINE_SMILES, build_reaction_complex
from snar_qc.qc.backend import free_gpu_memory

# A sidecar with one of these statuses counts as "done" -- not re-run unless --force.
TERMINAL_STATUSES = {"completed", "no_peak", "ts_not_saddle", "error"}


@dataclass
class WorkerConfig:
    """Everything ``run_substrate`` needs that is constant across a batch."""

    outdir: str
    amine: str = DEFAULT_AMINE_SMILES
    approach: float = 3.0
    scan_steps: int = 14
    scan_stop: float = 1.45
    scan_stop_lg: float = 2.6
    n_procs: int = 4
    mem: float = 6.0
    solvent: Optional[str] = None
    solvent_model: Optional[str] = None
    coordinate: str = "concerted"
    retry: bool = False
    force: bool = False
    resume: bool = False


def slug(text: str) -> str:
    """A filesystem-safe short tag derived from a string (e.g. a SMILES)."""
    return "".join(c if c.isalnum() else "_" for c in text)[:48].strip("_")


def task_tag(row: dict) -> str:
    """Per-substrate directory tag.

    Prefers an explicit id (``substrate_id`` / catalogue code), then ``lu_id``
    (rendered ``lu_<id>`` for the Lu slices), else a SMILES slug.
    """
    for key in ("substrate_id", "lu_id", "arylator_catcode"):
        val = row.get(key)
        if val not in (None, ""):
            text = str(val).strip()
            return f"lu_{text}" if key == "lu_id" else slug(text)
    return slug(row["smiles_canonical"])


def should_skip(sidecar: Path, retry: bool, force: bool) -> Optional[str]:
    """Return the existing status if the substrate should be skipped, else None."""
    if force or not sidecar.exists():
        return None
    try:
        status = json.loads(sidecar.read_text()).get("status")
    except (json.JSONDecodeError, OSError):
        return None
    if status == "completed":
        return status
    if status in TERMINAL_STATUSES and not retry:
        return status
    return None


def run_substrate(
    row: dict, cfg: WorkerConfig, log: Optional[Callable[[str], None]] = None
) -> dict:
    """Build the complex and compute ΔG‡ for one substrate; write its sidecar.

    Returns the result payload (with ``tag`` and any id columns stamped on). A
    skipped substrate returns its cached payload with ``skipped`` set. ``log`` is
    an optional progress callback (the sequential runner passes ``print``; the
    queue keeps workers quiet and reports via the heartbeat instead).
    """
    tag = task_tag(row)
    workdir = Path(cfg.outdir).resolve() / tag
    workdir.mkdir(parents=True, exist_ok=True)
    sidecar = workdir / "result.json"

    skip_status = should_skip(sidecar, cfg.retry, cfg.force)
    if skip_status is not None:
        if log:
            log(f"[skip] {tag}: already {skip_status}")
        payload = json.loads(sidecar.read_text())
        payload["skipped"] = skip_status
        payload["tag"] = tag
        return payload

    lu_id = row.get("lu_id")
    lu_id = int(lu_id) if lu_id not in (None, "") else None
    leaving_group = (row.get("leaving_group") or "").strip() or None

    if log:
        log(f"[run ] {tag}: {row['smiles_canonical']} (LG={leaving_group})")
    rc = build_reaction_complex(
        row["smiles_canonical"],
        amine_smiles=cfg.amine,
        leaving_group=leaving_group,
        approach=cfg.approach,
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
            scan_steps=cfg.scan_steps,
            scan_stop=cfg.scan_stop,
            scan_stop_lg=cfg.scan_stop_lg,
            n_procs=cfg.n_procs,
            mem=cfg.mem,
            lu_id=lu_id,
            solvent=cfg.solvent,
            solvent_model=cfg.solvent_model,
            coordinate=cfg.coordinate,
            # --force always recomputes from scratch; otherwise honour --resume.
            resume=cfg.resume and not cfg.force,
        )
    finally:
        os.chdir(cwd)
        # Return CuPy's pooled VRAM to the driver so a long batch in one process does
        # not accumulate memory and OOM later/larger substrates (no-op off the GPU path).
        free_gpu_memory()

    payload = result.to_dict()
    payload["wall_s"] = round(time.time() - started, 1)
    payload["tag"] = tag
    # Stamp traceability ids onto the sidecar (lu_id stays whatever the barrier set).
    for key in ("substrate_id", "arylator_id"):
        if row.get(key) not in (None, ""):
            payload[key] = row[key]
    sidecar.write_text(json.dumps(payload, indent=2))
    if log:
        log(
            f"[done] {tag}: status={payload['status']} "
            f"ΔG‡(qh)={payload.get('delta_g_qh_kcal')} "
            f"n_imag_ts={payload.get('n_imag_ts')} ({payload['wall_s']}s)"
        )
    return payload
