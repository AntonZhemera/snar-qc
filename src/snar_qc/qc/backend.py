"""Backend factory: choose the QC calculator (Psi4 default, gpu4pyscf opt-in).

This module is the **single chokepoint** for the CPU-fallback contract (see the
gpu4pyscf-backend masterplan): backend selection, the lazy GPU import, the device
probe, and the fall-back decision all live here. Calculators never import GPU
libraries directly -- one place to reason about, one place to test.

Key guarantees for non-GPU hosts (the Windows workstation, CI, CPU-only dev boxes):

- **Default is Psi4/CPU.** A fresh checkout behaves exactly as before; the GPU path is
  opt-in via ``backend="gpu4pyscf"`` or the ``SNAR_QC_BACKEND=gpu4pyscf`` env var.
- **gpu4pyscf / cupy are optional ``[gpu]`` extras, never imported at package top
  level.** They are imported lazily here only when the GPU backend is selected, so
  ``import snar_qc`` and the whole Psi4 path succeed with them absent.
- **Capability probe before use.** A missing driver/library or an undersized device
  becomes a typed, catchable :class:`GPUUnavailableError`, not a bare ``ImportError``
  or a CUDA segfault.
- **Silent default, loud override.** No GPU requested -> Psi4, silently. GPU requested
  but unavailable -> fall back to Psi4 and log a WARNING, unless ``require_gpu`` is set
  (then the error propagates, for benchmarking).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from predict_snar.calculators import Calculator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ase import Atoms

logger = logging.getLogger("snar_qc")

PSI4 = "psi4"
GPU4PYSCF = "gpu4pyscf"
_VALID_BACKENDS = frozenset({PSI4, GPU4PYSCF})

_BACKEND_ENV = "SNAR_QC_BACKEND"
_REQUIRE_GPU_ENV = "SNAR_QC_REQUIRE_GPU"

# Minimum free VRAM to attempt a GPU job. The 5-ring def2-SVP reference complex used
# ~0.66 GB (notes/2026-06-23_gpu_hessian_benchmark.md); 1 GB keeps the 4 GB card off
# the fallback edge for the POC-sized complexes and leaves room for the DF tensors.
_MIN_FREE_VRAM_BYTES = 1 * 1024**3


class GPUUnavailableError(RuntimeError):
    """No usable CUDA device / gpu4pyscf stack: missing driver, library, or VRAM.

    Raised by :func:`probe_gpu`. :func:`make_calculator` catches it to fall back to
    Psi4 (logging a WARNING) unless ``require_gpu`` is set.
    """


def _resolve_backend(backend: Optional[str]) -> str:
    """Resolve the backend name: explicit arg > ``SNAR_QC_BACKEND`` env > Psi4 default."""
    name = (backend or os.environ.get(_BACKEND_ENV) or PSI4).strip().lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown QC backend {name!r}; expected one of {sorted(_VALID_BACKENDS)} "
            f"(set via the 'backend' argument or ${_BACKEND_ENV})."
        )
    return name


def _resolve_require_gpu(require_gpu: Optional[bool]) -> bool:
    """Resolve strict mode: explicit arg > ``SNAR_QC_REQUIRE_GPU`` env > ``False``."""
    if require_gpu is not None:
        return bool(require_gpu)
    return os.environ.get(_REQUIRE_GPU_ENV, "").strip().lower() in {"1", "true", "yes"}


def probe_gpu(min_free_vram_bytes: int = _MIN_FREE_VRAM_BYTES) -> None:
    """Raise :class:`GPUUnavailableError` unless a usable CUDA device is present.

    Imports ``cupy`` lazily and queries the runtime, so that on a CPU host a missing
    driver / library / device surfaces as a typed, catchable error rather than an
    ``ImportError`` or a hard CUDA failure. Checks, in order: ``cupy`` importable, the
    CUDA driver answers ``getDeviceCount()``, at least one device present, and enough
    free VRAM for a def2-SVP job.
    """
    try:
        import cupy  # noqa: PLC0415 -- lazy: optional [gpu] dependency
    except Exception as exc:  # ImportError, or a broken CUDA runtime at import time
        raise GPUUnavailableError(f"cupy / CUDA runtime not importable: {exc}") from exc

    try:
        n_devices = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise GPUUnavailableError(f"CUDA driver query failed: {exc}") from exc
    if n_devices < 1:
        raise GPUUnavailableError("no CUDA device present (getDeviceCount() == 0)")

    try:
        free_bytes, _total = cupy.cuda.runtime.memGetInfo()
    except Exception as exc:
        raise GPUUnavailableError(f"could not query free VRAM: {exc}") from exc
    if free_bytes < min_free_vram_bytes:
        raise GPUUnavailableError(
            f"insufficient free VRAM: {free_bytes / 1024**3:.2f} GB free < "
            f"{min_free_vram_bytes / 1024**3:.2f} GB required"
        )


def make_calculator(
    atoms: Optional["Atoms"] = None,
    file: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    backend: Optional[str] = None,
    require_gpu: Optional[bool] = None,
) -> Calculator:
    """Construct a QC calculator for the selected backend (Psi4 default, GPU opt-in).

    Args:
        atoms: ASE ``Atoms`` geometry (with ``info["charge"]``), passed to the calculator.
        file: Optional output-file base name, passed through unchanged.
        options: Optional calculator option overrides, passed through unchanged.
        backend: ``"psi4"`` (default) or ``"gpu4pyscf"``. If ``None``, read from
            ``$SNAR_QC_BACKEND``, then default to Psi4.
        require_gpu: Strict mode. When the GPU backend is requested but no usable device
            is available, raise instead of falling back to Psi4. If ``None``, read from
            ``$SNAR_QC_REQUIRE_GPU``, then default to ``False`` (fall back).

    Returns:
        A :class:`predict_snar.calculators.Calculator` subclass instance. Psi4 is
        imported only on the Psi4 path; gpu4pyscf / cupy only on the GPU path.
    """
    name = _resolve_backend(backend)
    if name == PSI4:
        return _make_psi4(atoms, file, options)

    # GPU requested -- probe for a usable device before importing the heavy GPU stack.
    try:
        probe_gpu()
    except GPUUnavailableError as exc:
        if _resolve_require_gpu(require_gpu):
            raise
        logger.warning(
            "GPU backend requested but unavailable (%s); falling back to Psi4.", exc
        )
        return _make_psi4(atoms, file, options)

    # Device is usable: import the GPU calculator lazily (it imports gpu4pyscf/cupy at
    # module scope, so this import must stay inside the function, never at top level).
    from snar_qc.qc.gpu4pyscf_calculator import GPU4PySCFCalculator  # noqa: PLC0415

    return GPU4PySCFCalculator(atoms, file=file, options=options)


def _make_psi4(
    atoms: Optional["Atoms"], file: Optional[str], options: Optional[dict[str, Any]]
) -> Calculator:
    """Construct a Psi4Calculator (imported lazily to keep the default path light)."""
    from snar_qc.qc.psi4_calculator import Psi4Calculator  # noqa: PLC0415

    return Psi4Calculator(atoms, file=file, options=options)
