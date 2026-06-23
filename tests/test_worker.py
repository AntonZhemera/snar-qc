"""Tests for the shared per-substrate work unit (no QC).

The heavy ``compute_barrier`` chain is exercised by the manual run, not here.
These pin the resumable logic: tag derivation, skip semantics, and that the
compute path writes a sidecar and stamps traceability ids (with the QC calls
monkeypatched out).
"""

import json

import pytest

from snar_qc.poc import worker
from snar_qc.poc.worker import WorkerConfig, run_substrate, should_skip, slug, task_tag


def test_slug_keeps_alnum_replaces_rest():
    assert slug("EN300-17347") == "EN300_17347"
    assert slug("Fc1ccccc1") == "Fc1ccccc1"


def test_task_tag_prefers_substrate_id():
    assert task_tag({"substrate_id": "EN300-17347", "smiles_canonical": "x"}) == "EN300_17347"


def test_task_tag_lu_id_then_smiles():
    assert task_tag({"lu_id": 17, "smiles_canonical": "x"}) == "lu_17"
    assert task_tag({"smiles_canonical": "Fc1ccccc1"}) == "Fc1ccccc1"


def test_should_skip_semantics(tmp_path):
    sidecar = tmp_path / "result.json"
    assert should_skip(sidecar, retry=False, force=False) is None  # missing -> run

    sidecar.write_text(json.dumps({"status": "completed"}))
    assert should_skip(sidecar, retry=False, force=False) == "completed"
    assert should_skip(sidecar, retry=True, force=False) == "completed"  # always skip success
    assert should_skip(sidecar, retry=False, force=True) is None  # force re-runs

    sidecar.write_text(json.dumps({"status": "error"}))
    assert should_skip(sidecar, retry=False, force=False) == "error"  # terminal -> skip
    assert should_skip(sidecar, retry=True, force=False) is None  # --retry re-runs errors

    sidecar.write_text("not json")
    assert should_skip(sidecar, retry=False, force=False) is None  # unreadable -> run


def test_run_substrate_skip_returns_cached(tmp_path):
    workdir = tmp_path / "EN300_1"
    workdir.mkdir()
    (workdir / "result.json").write_text(json.dumps({"status": "completed", "delta_g_qh_kcal": 12.3}))
    cfg = WorkerConfig(outdir=str(tmp_path))
    out = run_substrate(
        {"substrate_id": "EN300-1", "smiles_canonical": "x", "leaving_group": "F"}, cfg
    )
    assert out["skipped"] == "completed"
    assert out["tag"] == "EN300_1"
    assert out["delta_g_qh_kcal"] == 12.3


def test_run_substrate_computes_writes_and_stamps(tmp_path, monkeypatch):
    ase_io = pytest.importorskip("ase.io")

    class _FakeResult:
        def to_dict(self):
            return {"status": "completed", "delta_g_qh_kcal": 1.0, "n_imag_ts": 1}

    class _FakeRC:
        atoms = object()

    monkeypatch.setattr(worker, "build_reaction_complex", lambda *a, **k: _FakeRC())
    monkeypatch.setattr(worker, "compute_barrier", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(ase_io, "write", lambda *a, **k: None)

    cfg = WorkerConfig(outdir=str(tmp_path), solvent="DMSO")
    out = run_substrate(
        {
            "substrate_id": "EN300-9",
            "arylator_id": 42,
            "smiles_canonical": "Fc1ccccc1",
            "leaving_group": "F",
        },
        cfg,
    )
    assert out["status"] == "completed"
    assert out["tag"] == "EN300_9"
    assert out["substrate_id"] == "EN300-9" and out["arylator_id"] == 42
    assert out["wall_s"] >= 0
    sidecar = tmp_path / "EN300_9" / "result.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["status"] == "completed"
