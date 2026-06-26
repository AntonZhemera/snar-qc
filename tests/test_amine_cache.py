"""Tests for the fixed model-amine reference cache and its no-recompute seeder.

Pure I/O -- no Psi4 / gpu4pyscf / GPU needed. The cache root is redirected to a tmp dir via
the ``SNAR_QC_AMINE_CACHE`` env var so nothing touches the repo's ``assets/``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from ase import Atoms

from snar_qc.poc import amine_cache
from snar_qc.qc.thermo import Psi4Thermo


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(amine_cache._CACHE_ENV, str(tmp_path))
    return tmp_path


def _amine_atoms() -> Atoms:
    atoms = Atoms("CN", positions=[[0.0, 0.0, 0.0], [1.47, 0.0, 0.0]])
    atoms.info["charge"] = 0
    return atoms


def _thermo() -> Psi4Thermo:
    return Psi4Thermo(
        electronic_energy=-95.78927687,
        gibbs=-95.74856679,
        gibbs_qh=-95.74856855,
        enthalpy=-95.72131775,
        zpve=0.06361173,
        frequencies=[],
    )


def test_level_tag_includes_dispersion():
    opts = {"functional": "B3LYP", "dispersion": "D3BJ", "basis_set": "def2-SVP"}
    assert amine_cache.level_tag(opts) == "b3lyp-d3bj_def2-svp"


def test_level_tag_without_dispersion():
    opts = {"functional": "pbe", "dispersion": None, "basis_set": "sto-3g"}
    assert amine_cache.level_tag(opts) == "pbe_sto-3g"


def test_cache_key_is_filesystem_safe_and_stable():
    key = amine_cache.cache_key("CN", "gpu4pyscf", "b3lyp-d3bj_def2-svp")
    assert key == "CN__b3lyp-d3bj_def2-svp__gpu4pyscf"
    assert "/" not in key and "#" not in key and "=" not in key


def test_cache_key_canonicalises_equivalent_smiles():
    # CN and NC are the same molecule; RDKit canonicalisation must collapse them to one key.
    pytest.importorskip("rdkit")
    assert amine_cache.cache_key("CN", "psi4", "lv") == amine_cache.cache_key(
        "NC", "psi4", "lv"
    )


def test_store_then_load_round_trips(cache_dir):
    thermo, atoms = _thermo(), _amine_atoms()
    amine_cache.store("CN", "gpu4pyscf", "b3lyp-d3bj_def2-svp", thermo, atoms, 0)

    hit = amine_cache.load("CN", "gpu4pyscf", "b3lyp-d3bj_def2-svp")
    assert hit is not None
    got_thermo, got_atoms, n_imag = hit
    assert n_imag == 0
    for key in amine_cache._THERMO_KEYS:
        assert getattr(got_thermo, key) == pytest.approx(getattr(thermo, key))
    assert got_atoms.get_chemical_symbols() == ["C", "N"]
    assert got_atoms.positions == pytest.approx(atoms.positions)
    assert got_atoms.info["charge"] == 0


def test_load_miss_returns_none(cache_dir):
    assert amine_cache.load("CN", "psi4", "b3lyp-d3bj_def2-svp") is None


def test_load_does_not_cross_backends(cache_dir):
    amine_cache.store("CN", "gpu4pyscf", "lv", _thermo(), _amine_atoms(), 0)
    # Same amine + level, different backend -> miss (absolute energies are backend-specific).
    assert amine_cache.load("CN", "psi4", "lv") is None


def test_load_rejects_tampered_provenance(cache_dir):
    amine_cache.store("CN", "gpu4pyscf", "lv", _thermo(), _amine_atoms(), 0)
    json_path, _ = amine_cache._paths("CN", "gpu4pyscf", "lv")
    rec = json.loads(json_path.read_text())
    rec["backend"] = "psi4"  # now disagrees with the filename's key
    json_path.write_text(json.dumps(rec))
    assert amine_cache.load("CN", "gpu4pyscf", "lv") is None


def test_store_records_provenance(cache_dir):
    amine_cache.store(
        "CN",
        "psi4",
        "lv",
        _thermo(),
        _amine_atoms(),
        0,
        provenance={"seeded_by": "test"},
    )
    json_path, _ = amine_cache._paths("CN", "psi4", "lv")
    rec = json.loads(json_path.read_text())
    assert rec["_provenance"] == {"seeded_by": "test"}


# --- runtime wiring (barrier.cached_amine_reference) ---------------------------------
def test_cached_amine_reference_hits_cache_without_recompute(cache_dir, monkeypatch):
    """A cache hit returns the stored reference and never calls the opt+freq."""
    # barrier pulls the engine chain; skip (don't error) where it is absent (e.g. a
    # minimal/no-Psi4 host). The cache I/O tests above need none of this.
    barrier = pytest.importorskip("snar_qc.poc.barrier")

    amine_cache.store(
        "CN", "gpu4pyscf", "b3lyp-d3bj_def2-svp", _thermo(), _amine_atoms(), 0
    )

    class GPU4PySCFCalculator:  # name drives backend detection (startswith "GPU")
        options = {"functional": "b3lyp", "dispersion": "d3bj", "basis_set": "def2-svp"}

    monkeypatch.setattr(
        barrier, "make_calculator", lambda *a, **k: GPU4PySCFCalculator()
    )

    def _no_compute(*a, **k):
        raise AssertionError("opt+freq must not run on a cache hit")

    monkeypatch.setattr(barrier, "_species_gas_thermo", _no_compute)

    thermo, atoms, n_imag = barrier.cached_amine_reference("CN", 1, 1.0)
    assert thermo.electronic_energy == pytest.approx(-95.78927687)
    assert atoms.get_chemical_symbols() == ["C", "N"]
    assert n_imag == 0


# --- seeder (scripts/extract_amine_ref.py) -------------------------------------------
def _load_seeder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_amine_ref.py"
    spec = importlib.util.spec_from_file_location("extract_amine_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_run(run_dir: Path, energies: dict[str, float]) -> None:
    """Fake a finished gas run: one subdir per tag with gas_thermo.json + amine_opt.xyz."""
    for tag, e in energies.items():
        d = run_dir / tag
        d.mkdir(parents=True)
        (d / "gas_thermo.json").write_text(
            json.dumps(
                {
                    "amine_smiles": "CN",
                    "n_imag_amine": 0,
                    "species": {
                        "amine": {
                            "geometry": "amine_opt.xyz",
                            "electronic_energy": e,
                            "gibbs": e + 0.04,
                            "gibbs_qh": e + 0.04,
                            "enthalpy": e + 0.07,
                            "zpve": 0.0636,
                        }
                    },
                }
            )
        )
        from ase.io import write

        write(str(d / "amine_opt.xyz"), _amine_atoms(), format="extxyz")


def test_seeder_extracts_consistent_run(tmp_path):
    seeder = _load_seeder()
    run = tmp_path / "run"
    _write_run(run, {"lu_1": -95.78927687, "lu_2": -95.78927687})
    smiles, thermo, atoms, n_imag, prov = seeder.extract(run, "gpu4pyscf", "lv")
    assert smiles == "CN"
    assert thermo.electronic_energy == pytest.approx(-95.78927687)
    assert n_imag == 0
    assert prov["n_substrates_checked"] == 2


def test_seeder_rejects_drifting_amine(tmp_path):
    seeder = _load_seeder()
    run = tmp_path / "run"
    _write_run(run, {"lu_1": -95.78927687, "lu_2": -95.70000000})  # 0.09 Eh drift
    with pytest.raises(seeder.SeedError, match="not constant"):
        seeder.extract(run, "gpu4pyscf", "lv")


def test_seeder_errors_on_empty_run(tmp_path):
    seeder = _load_seeder()
    (tmp_path / "empty").mkdir()
    with pytest.raises(seeder.SeedError, match="no substrate"):
        seeder.extract(tmp_path / "empty", "psi4", "lv")
