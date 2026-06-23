"""Tests for the shared-queue orchestrator's pure helpers (no QC, no pool).

Auto-tune maths, phase inference from on-disk artifacts, duration/ETA formatting,
error-reason compaction, and heartbeat rendering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_qc_queue as q  # noqa: E402


def test_plan_concurrency_auto_fills_cores():
    assert q.plan_concurrency(20, 14, 52.0, mem_per_worker_gb=6, threads_pref=3)[:2] == (4, 3)


def test_plan_concurrency_capped_by_ram():
    workers, threads, _ = q.plan_concurrency(20, 14, 10.0, mem_per_worker_gb=6, threads_pref=3)
    assert workers == 1 and threads == 3  # 10*0.85/6 -> 1


def test_plan_concurrency_explicit_overrides():
    assert q.plan_concurrency(20, 14, 52.0, want_workers=6, want_threads=2)[:2] == (6, 2)


def test_infer_phase_from_artifacts():
    assert q.infer_phase({"result.json", "ts.out"}) == "done"
    assert q.infer_phase({"ts.out", "sps", "scan.xyz"}) == "ts_opt_freq"
    assert q.infer_phase({"sps", "scan.xyz"}) == "dft_sps"
    assert q.infer_phase({"scan.xyz"}) == "scan_dft"
    assert q.infer_phase({"complex.xyz"}) == "scan_xtb"
    assert q.infer_phase(set()) == "starting"


def test_fmt_dur():
    assert q.fmt_dur(None) == "?"
    assert q.fmt_dur(45) == "45s"
    assert q.fmt_dur(125) == "2m"
    assert q.fmt_dur(3 * 3600 + 50 * 60) == "3h 50m"


def test_eta_seconds():
    assert q.eta_seconds(None, 10, 4) is None  # no pace data yet
    assert q.eta_seconds(100, 0, 4) == 0.0  # nothing left
    assert q.eta_seconds(120, 8, 4) == 240.0  # 120 * 8 / 4


def test_error_reason_compacts_stage_and_type():
    assert (
        q.error_reason({"stage": "ts_opt_freq", "error": "MemoryError: bad allocation\n..."})
        == "ts_opt_freq/MemoryError"
    )
    assert q.error_reason({"stage": "scan", "error": ""}) == "scan/unknown"


def test_format_heartbeat_renders_fields():
    snap = {
        "elapsed_s": 3720, "total": 150, "terminal": 12, "completed": 8,
        "terminal_other": 2, "failed": 2, "running": 4, "pending": 134,
        "running_tasks": [("EN300_1", "ts_opt_freq", 2460), ("EN300_2", "dft_sps", 600)],
        "error_reasons": [("ts_opt_freq/MemoryError", 2)],
        "cpu_pct": 78.0, "ram_used_gb": 21.3, "ram_total_gb": 63.7, "ram_pct": 33.0,
        "workers": 4, "threads": 3, "mean_wall_s": 2310, "eta_s": 79350,
    }
    text = q.format_heartbeat(snap)
    assert "progress : 12/150" in text
    assert "ts_opt_freq/MemoryError x2" in text
    assert "EN300_1[ts_opt_freq,41m]" in text
    assert "CPU 78%" in text
    assert "ETA ~" in text
