#!/usr/bin/env python
"""Shared-queue ΔG‡ orchestrator: idle workers pull the next substrate.

Replaces the pre-set ``--only`` sharding (``scripts/run_poc_batch.sh``) with a
single work queue. Every substrate is one task submitted to a
``concurrent.futures.ProcessPoolExecutor``; the pool feeds whichever worker is
free, and ``as_completed`` lets us pack each result the moment it finishes -- no
manual sharding, no idle workers waiting on a slow shard. Per-substrate sidecars
keep the run **resumable**: re-running the same command skips finished substrates
(``--retry`` re-runs the non-completed ones, ``--force`` re-runs everything).

On each finished task: zip its work dir and copy the ``.zip`` into ``--archive-dir``
(a generic path, created if missing) so results can be watched from elsewhere. On
resume, a finished task whose archive is missing is re-zipped (backfill).

Every ``--heartbeat-min`` minutes it reports -- to the console **and**, when an
archive dir is set, to ``<archive-dir>/run.log`` -- total progress, error
breakdown by reason, what is running and in which phase, system resources, ETA.

Run in the snar-qc conda env. Example:

    python scripts/run_qc_queue.py \\
        --substrates data/external/realpool_qc/realpool_qc_slice.csv \\
        --outdir data/processed/realpool_dmso --solvent DMSO \\
        --archive-dir <ARCHIVE_DIR>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import psutil

_SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _SRC)
from snar_qc.poc.worker import (  # noqa: E402
    TERMINAL_STATUSES,
    WorkerConfig,
    run_substrate,
    task_tag,
)

# Make the Δ in "ΔG‡" survive a Windows cp1252 console (see CLAUDE.md portability).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                  #
# --------------------------------------------------------------------------- #
def plan_concurrency(
    logical: Optional[int],
    physical: Optional[int],
    avail_ram_gb: float,
    mem_per_worker_gb: float = 6.0,
    threads_pref: int = 3,
    want_workers: Optional[int] = None,
    want_threads: Optional[int] = None,
) -> tuple[int, int, float]:
    """Pick ``(workers, threads, mem_gb)``; explicit ``want_*`` override the guess.

    Fills physical cores at ``threads`` per worker, but never asks for more
    concurrent workers than ~85 % of RAM can hold at ``mem_per_worker_gb`` each.
    """
    threads = max(1, want_threads or threads_pref)
    if want_workers:
        workers = max(1, want_workers)
    else:
        cores = physical or logical or 2
        by_cpu = max(1, cores // threads)
        by_ram = max(1, int((avail_ram_gb * 0.85) // mem_per_worker_gb))
        workers = max(1, min(by_cpu, by_ram))
    return workers, threads, mem_per_worker_gb


def infer_phase(names: set[str]) -> str:
    """Infer a task's pipeline phase from the files present in its work dir."""
    if "result.json" in names:
        return "done"
    if "ts.out" in names:
        return "ts_opt_freq"
    if "sps" in names:
        return "dft_sps"
    if "scan.xyz" in names:
        return "scan_dft"
    if names & {"complex.xyz", "xtb.out", "xtb.xyz"}:
        return "scan_xtb"
    return "starting"


def fmt_dur(seconds: Optional[float]) -> str:
    """Human duration: '3h 50m', '12m', '45s', or '?' for unknown."""
    if seconds is None or seconds < 0 or math.isinf(seconds):
        return "?"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 90:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def eta_seconds(mean_wall_s: Optional[float], remaining: int, workers: int) -> Optional[float]:
    """ETA = mean task wall-time x remaining tasks / workers (None if no data)."""
    if remaining <= 0:
        return 0.0
    if not mean_wall_s:
        return None
    return mean_wall_s * remaining / max(1, workers)


def format_heartbeat(snap: dict) -> str:
    """Render a heartbeat block from a precomputed snapshot dict (no I/O)."""
    running = "  ".join(
        f"{tag}[{phase},{fmt_dur(age)}]" for tag, phase, age in snap["running_tasks"]
    ) or "(none)"
    errors = (
        "  ".join(f"{reason} x{n}" for reason, n in snap["error_reasons"])
        if snap["error_reasons"]
        else "(none)"
    )
    return "\n".join(
        [
            f"---- heartbeat @ {fmt_dur(snap['elapsed_s'])} "
            f"({snap['workers']}x{snap['threads']} threads) ----",
            f"progress : {snap['terminal']}/{snap['total']} done "
            f"({snap['completed']} completed, {snap['terminal_other']} no-saddle, "
            f"{snap['failed']} failed) | {snap['running']} running | "
            f"{snap['pending']} pending",
            f"errors   : {errors}",
            f"running  : {running}",
            f"resources: CPU {snap['cpu_pct']:.0f}% | "
            f"RAM {snap['ram_used_gb']:.1f}/{snap['ram_total_gb']:.1f} GB "
            f"({snap['ram_pct']:.0f}%)",
            f"pace     : mean {fmt_dur(snap['mean_wall_s'])}/task | "
            f"ETA ~{fmt_dur(snap['eta_s'])}",
            "-" * 56,
        ]
    )


def error_reason(payload: dict) -> str:
    """Compact failure label: '<stage>/<first error line / exc type>'."""
    stage = payload.get("stage") or "?"
    err = (payload.get("error") or "").strip()
    head = err.splitlines()[0] if err else "unknown"
    if ":" in head:  # keep the exception/type token, drop the long message tail
        head = head.split(":", 1)[0]
    return f"{stage}/{head}"[:60]


# --------------------------------------------------------------------------- #
# Output sink + live state                                                    #
# --------------------------------------------------------------------------- #
class Emitter:
    """Write run output to the console and, if given, append it to a log file.

    Thread-safe: the main loop and the heartbeat thread both emit. The log file
    lives next to the per-task archives so a synced ``--archive-dir`` shows live
    progress without shell access to the running host.
    """

    def __init__(self, log_path: Optional[Path]):
        self.log_path = log_path
        self._lock = threading.Lock()
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, msg: str) -> None:
        with self._lock:
            print(msg, flush=True)
            if self.log_path is not None:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as handle:
                        handle.write(msg + "\n")
                except OSError as exc:
                    print(f"[warn] log write failed: {exc}", flush=True)


class Tracker:
    """Thread-safe roll-up the heartbeat reads while the main loop records."""

    def __init__(self, total: int, workers: int, threads: int, outdir: Path, tags: list[str]):
        self.total = total
        self.workers = workers
        self.threads = threads
        self.outdir = outdir
        self.tags = tags
        self.start = time.time()
        self._lock = threading.Lock()
        self.completed = self.terminal_other = self.failed = self.skipped = 0
        self.wall_times: list[float] = []
        self.error_reasons: Counter = Counter()

    def record(self, payload: dict) -> None:
        with self._lock:
            status = payload.get("status")
            if payload.get("skipped"):
                self.skipped += 1
            if status == "completed":
                self.completed += 1
            elif status == "error":
                self.failed += 1
                self.error_reasons[error_reason(payload)] += 1
            elif status in TERMINAL_STATUSES:
                self.terminal_other += 1
            wall = payload.get("wall_s")
            if isinstance(wall, (int, float)) and not payload.get("skipped"):
                self.wall_times.append(float(wall))

    def snapshot(self) -> dict:
        """Build a heartbeat snapshot: live counts + a disk scan for phases."""
        with self._lock:
            completed, other, failed = self.completed, self.terminal_other, self.failed
            errors = self.error_reasons.most_common(6)
            mean_wall = (sum(self.wall_times) / len(self.wall_times)) if self.wall_times else None
        terminal = completed + other + failed
        running_tasks, now = [], time.time()
        for tag in self.tags:
            workdir = self.outdir / tag
            if not workdir.is_dir():
                continue
            names = set(os.listdir(workdir))
            if "result.json" in names:
                continue
            try:
                age = now - workdir.stat().st_ctime
            except OSError:
                age = None
            running_tasks.append((tag, infer_phase(names), age))
        running_tasks.sort(key=lambda t: (t[2] is None, -(t[2] or 0)))
        vm = psutil.virtual_memory()
        return {
            "elapsed_s": now - self.start,
            "total": self.total,
            "terminal": terminal,
            "completed": completed,
            "terminal_other": other,
            "failed": failed,
            "running": len(running_tasks),
            "pending": max(0, self.total - terminal - len(running_tasks)),
            "running_tasks": running_tasks[: self.workers + 2],
            "error_reasons": errors,
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_used_gb": (vm.total - vm.available) / 1e9,
            "ram_total_gb": vm.total / 1e9,
            "ram_pct": vm.percent,
            "workers": self.workers,
            "threads": self.threads,
            "mean_wall_s": mean_wall,
            "eta_s": eta_seconds(mean_wall, self.total - terminal, self.workers),
        }


def archive_task(workdir: Path, archive_dir: Path) -> Path:
    """Zip ``workdir`` and move the archive to ``archive_dir/<tag>.zip``."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = workdir.parent / f".{workdir.name}.partial"
    tmp_zip = shutil.make_archive(
        str(tmp_base), "zip", root_dir=str(workdir.parent), base_dir=workdir.name
    )
    dest = archive_dir / f"{workdir.name}.zip"
    shutil.move(tmp_zip, dest)
    return dest


def _init_worker(src_path: str) -> None:
    """Ensure ``snar_qc`` is importable in spawned workers (Windows spawn)."""
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _load_rows(path: str, only: Optional[str], limit: Optional[int]) -> list[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if only:
        wanted = {x.strip() for x in only.split(",")}
        rows = [
            r
            for r in rows
            if str(r.get("substrate_id", "")).strip() in wanted
            or str(r.get("lu_id", "")).strip() in wanted
        ]
    if limit:
        rows = rows[:limit]
    return rows


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrates", required=True, help="CSV: smiles_canonical, leaving_group")
    ap.add_argument("--outdir", default="data/processed/qc_queue_run")
    ap.add_argument("--archive-dir", default=None, help="copy each finished task's .zip here")
    ap.add_argument(
        "--log-file", default=None, help="mirror output here (default <archive-dir>/run.log)"
    )
    ap.add_argument("--solvent", default=None, help="PCMSolver solvent (e.g. DMSO); omit for gas")
    ap.add_argument("--coordinate", choices=("concerted", "addition"), default="concerted")
    ap.add_argument("--amine", default=None, help="model amine SMILES (default CN)")
    ap.add_argument("--workers", default="auto", help="int or 'auto'")
    ap.add_argument("--threads", default="auto", help="Psi4 threads/worker: int or 'auto'")
    ap.add_argument("--mem", type=float, default=6.0, help="GB per worker for Psi4")
    ap.add_argument("--heartbeat-min", type=float, default=30.0)
    ap.add_argument("--only", help="comma-separated substrate_id/lu_id filter (smoke test)")
    ap.add_argument("--limit", type=int, help="only the first N substrates (smoke test)")
    ap.add_argument("--retry", action="store_true", help="re-run non-completed substrates")
    ap.add_argument("--force", action="store_true", help="re-run everything from scratch")
    args = ap.parse_args(argv)

    rows = _load_rows(args.substrates, args.only, args.limit)
    if not rows:
        print("No substrates to run.")
        return 1
    tags = [task_tag(r) for r in rows]

    want_workers = None if args.workers == "auto" else int(args.workers)
    want_threads = None if args.threads == "auto" else int(args.threads)
    workers, threads, mem_gb = plan_concurrency(
        psutil.cpu_count(),
        psutil.cpu_count(logical=False),
        psutil.virtual_memory().available / 1e9,
        mem_per_worker_gb=args.mem,
        want_workers=want_workers,
        want_threads=want_threads,
    )

    cfg_kwargs = dict(
        outdir=args.outdir,
        n_procs=threads,
        mem=mem_gb,
        solvent=args.solvent,
        coordinate=args.coordinate,
        retry=args.retry,
        force=args.force,
    )
    if args.amine:
        cfg_kwargs["amine"] = args.amine
    cfg = WorkerConfig(**cfg_kwargs)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(args.archive_dir).expanduser().resolve() if args.archive_dir else None
    log_path = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else (archive_dir / "run.log" if archive_dir else None)
    )
    emitter = Emitter(log_path)

    emitter.emit(
        f"QC queue: {len(rows)} substrates | {workers} workers x {threads} threads "
        f"({mem_gb:g} GB each) | solvent={args.solvent or 'gas'} | "
        f"coordinate={args.coordinate}\n"
        f"  outdir : {outdir}\n"
        f"  archive: {archive_dir or '(none)'}\n"
        f"  log    : {log_path or '(console only)'}\n"
        f"  heartbeat every {args.heartbeat_min:g} min"
    )

    tracker = Tracker(len(rows), workers, threads, outdir, tags)
    stop = threading.Event()

    def heartbeat_loop() -> None:
        interval = max(1.0, args.heartbeat_min * 60.0)
        while not stop.wait(interval):
            emitter.emit(format_heartbeat(tracker.snapshot()))

    beat = threading.Thread(target=heartbeat_loop, name="heartbeat", daemon=True)
    beat.start()

    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(_SRC,)
        ) as pool:
            futures = {pool.submit(run_substrate, row, cfg): tag for row, tag in zip(rows, tags)}
            for future in as_completed(futures):
                tag = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad task must not stop the run
                    payload = {"tag": tag, "status": "error", "stage": "orchestrator",
                               "error": f"{type(exc).__name__}: {exc}"}
                tracker.record(payload)
                state = "skip" if payload.get("skipped") else payload.get("status")
                line = f"[{state}] {tag}"
                if state == "completed":
                    line += f"  ΔG‡(qh)={payload.get('delta_g_qh_kcal')}"
                emitter.emit(line)
                # Archive fresh results; on resume, backfill a finished task whose zip is gone.
                if archive_dir is not None:
                    dest = archive_dir / f"{tag}.zip"
                    if not payload.get("skipped") or not dest.exists():
                        try:
                            archive_task(outdir / tag, archive_dir)
                        except Exception as exc:  # noqa: BLE001
                            emitter.emit(f"[warn] archive failed for {tag}: {exc}")
    finally:
        stop.set()

    # Final heartbeat + roll-up.
    emitter.emit(format_heartbeat(tracker.snapshot()))
    summary = []
    for tag in tags:
        sidecar = outdir / tag / "result.json"
        if sidecar.exists():
            try:
                summary.append(json.loads(sidecar.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    emitter.emit(
        f"\nDone: {tracker.completed}/{len(rows)} completed, {tracker.failed} failed. "
        f"Roll-up: {outdir / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
