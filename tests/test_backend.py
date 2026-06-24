"""Tests for the QC backend factory (snar_qc.qc.backend).

These are the **CPU-fallback contract** tests from the gpu4pyscf-backend masterplan:
they run on any host with no GPU (CI, the Windows workstation, CPU-only dev boxes).
gpu4pyscf / cupy are never imported here -- the GPU device probe and the GPU calculator
module are monkeypatched, so the whole file is green without a CUDA stack. The real GPU
energy parity lives in tests/test_gpu4pyscf_calculator.py (skipped when no device).
"""

import logging
import sys
import types

import pytest

from snar_qc.qc import backend as backend_mod
from snar_qc.qc.backend import GPUUnavailableError, make_calculator, probe_gpu


def _fake_cupy(n_devices: int, free_bytes: int = 8 * 1024**3):
    """A stand-in ``cupy`` module exposing just the runtime calls ``probe_gpu`` uses."""
    cupy = types.ModuleType("cupy")
    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: n_devices,
        memGetInfo=lambda: (free_bytes, 16 * 1024**3),
    )
    cupy.cuda = types.SimpleNamespace(runtime=runtime)
    return cupy


def _fake_gpu_module(monkeypatch):
    """Inject a fake snar_qc.qc.gpu4pyscf_calculator so the factory needs no CUDA stack.

    Returns the fake ``GPU4PySCFCalculator`` class; instances record their constructor
    args so a test can assert the factory threaded atoms / file / options through.
    """

    class FakeGPUCalculator:
        def __init__(self, atoms=None, file=None, options=None):
            self.atoms = atoms
            self.file = file
            self.options = options

    module = types.ModuleType("snar_qc.qc.gpu4pyscf_calculator")
    module.GPU4PySCFCalculator = FakeGPUCalculator
    monkeypatch.setitem(sys.modules, "snar_qc.qc.gpu4pyscf_calculator", module)
    return FakeGPUCalculator


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    """No ambient backend selection leaks into a test from the real environment."""
    monkeypatch.delenv("SNAR_QC_BACKEND", raising=False)
    monkeypatch.delenv("SNAR_QC_REQUIRE_GPU", raising=False)


# -- default / explicit selection -------------------------------------------------


def test_default_backend_is_psi4_and_never_probes_gpu(monkeypatch):
    """No selection -> Psi4, silently, with no GPU probe or import attempted."""
    monkeypatch.setattr(
        backend_mod,
        "probe_gpu",
        lambda *a, **k: pytest.fail("probed GPU on default path"),
    )
    calc = make_calculator()
    from snar_qc.qc.psi4_calculator import Psi4Calculator

    assert isinstance(calc, Psi4Calculator)


def test_explicit_psi4_backend(monkeypatch):
    """``backend="psi4"`` is the same Psi4 path as the default."""
    monkeypatch.setattr(
        backend_mod, "probe_gpu", lambda *a, **k: pytest.fail("probed GPU on Psi4 path")
    )
    from snar_qc.qc.psi4_calculator import Psi4Calculator

    assert isinstance(make_calculator(backend="psi4"), Psi4Calculator)


def test_options_threaded_to_psi4():
    """Option overrides flow through the factory onto the constructed calculator."""
    calc = make_calculator(options={"functional": "pbe", "basis_set": "def2-tzvp"})
    assert calc.options["functional"] == "pbe"
    assert calc.options["basis_set"] == "def2-tzvp"


def test_unknown_backend_arg_raises():
    with pytest.raises(ValueError, match="Unknown QC backend"):
        make_calculator(backend="quantum-espresso")


def test_unknown_backend_env_raises(monkeypatch):
    monkeypatch.setenv("SNAR_QC_BACKEND", "nope")
    with pytest.raises(ValueError, match="Unknown QC backend"):
        make_calculator()


# -- GPU selection: device present ------------------------------------------------


def test_gpu_selected_with_device_returns_gpu_class(monkeypatch):
    """GPU requested + usable device -> the GPU calculator, with args threaded."""
    monkeypatch.setattr(backend_mod, "probe_gpu", lambda *a, **k: None)
    fake_cls = _fake_gpu_module(monkeypatch)

    calc = make_calculator(
        atoms=None, file="x.in", options={"functional": "b3lyp"}, backend="gpu4pyscf"
    )

    assert isinstance(calc, fake_cls)
    assert calc.file == "x.in"
    assert calc.options == {"functional": "b3lyp"}


def test_env_var_selects_gpu(monkeypatch):
    """``SNAR_QC_BACKEND=gpu4pyscf`` selects GPU without an explicit ``backend`` arg."""
    monkeypatch.setenv("SNAR_QC_BACKEND", "gpu4pyscf")
    monkeypatch.setattr(backend_mod, "probe_gpu", lambda *a, **k: None)
    fake_cls = _fake_gpu_module(monkeypatch)

    assert isinstance(make_calculator(), fake_cls)


# -- GPU selection: unavailable -> fallback / strict ------------------------------


def _raise_unavailable(*args, **kwargs):
    raise GPUUnavailableError("no usable device (test)")


def test_gpu_unavailable_falls_back_to_psi4_with_warning(monkeypatch, caplog):
    """GPU requested but unavailable -> Psi4 fallback + a WARNING (loud override)."""
    monkeypatch.setattr(backend_mod, "probe_gpu", _raise_unavailable)
    from snar_qc.qc.psi4_calculator import Psi4Calculator

    with caplog.at_level(logging.WARNING, logger="snar_qc"):
        calc = make_calculator(backend="gpu4pyscf")

    assert isinstance(calc, Psi4Calculator)
    assert any("falling back to Psi4" in r.getMessage() for r in caplog.records)


def test_require_gpu_arg_raises_instead_of_fallback(monkeypatch):
    monkeypatch.setattr(backend_mod, "probe_gpu", _raise_unavailable)
    with pytest.raises(GPUUnavailableError):
        make_calculator(backend="gpu4pyscf", require_gpu=True)


def test_require_gpu_env_raises_instead_of_fallback(monkeypatch):
    monkeypatch.setenv("SNAR_QC_REQUIRE_GPU", "1")
    monkeypatch.setattr(backend_mod, "probe_gpu", _raise_unavailable)
    with pytest.raises(GPUUnavailableError):
        make_calculator(backend="gpu4pyscf")


# -- the device probe (fake cupy: no CUDA stack needed) ---------------------------


def test_probe_passes_with_device_and_vram(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(1, free_bytes=8 * 1024**3))
    probe_gpu()  # must not raise


def test_probe_raises_when_no_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(0))
    with pytest.raises(GPUUnavailableError, match="no CUDA device"):
        probe_gpu()


def test_probe_raises_on_insufficient_vram(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(1, free_bytes=100 * 1024**2))
    with pytest.raises(GPUUnavailableError, match="insufficient free VRAM"):
        probe_gpu()


def test_probe_raises_when_cupy_absent(monkeypatch):
    # A None entry in sys.modules makes ``import cupy`` raise ImportError.
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(GPUUnavailableError, match="not importable"):
        probe_gpu()


def test_probe_raises_when_driver_query_fails(monkeypatch):
    cupy = types.ModuleType("cupy")

    def _boom():
        raise RuntimeError("CUDA driver version is insufficient")

    cupy.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(getDeviceCount=_boom)
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    with pytest.raises(GPUUnavailableError, match="driver query failed"):
        probe_gpu()
