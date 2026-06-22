# Stage A — GPU4PySCFCalculator.single_point

**Goal:** a gpu4pyscf backend that produces B3LYP-D3BJ/def2-SVP single-point energies at
parity with `Psi4Calculator`, behind the same `Calculator` interface.

Stage A also lands the **CPU-fallback contract** (masterplan) — the factory, lazy import, and
probe — because it is the first stage that constructs a backend.

**Files**
- New `src/snar_qc/qc/gpu4pyscf_calculator.py`: `GPU4PySCFCalculator(Calculator)`.
- Backend selection helper (factory by `SNAR_QC_BACKEND` env / config), wired where
  `Psi4Calculator` is constructed. **Default = Psi4.**
- `pyproject.toml`: add a `gpu` optional-dependency extra (`gpu4pyscf-cuda12x`, `cupy`,
  `cutensor-cu12`); keep them out of the core requirements.

**Approach**
- Mirror `Psi4Calculator`'s option surface (`functional / basis_set / dispersion /
  charge / scf_type`); build the molecule from the ASE `Atoms` (geometry + `info["charge"]`),
  spin by electron-count parity (RKS singlet / UKS doublet), as `Psi4Calculator._multiplicity`.
- `mol = gto.M(...)`; `mf = gpu4pyscf.dft.rks.RKS(mol, xc=functional).density_fit();
  mf.disp = "d3bj"`; `self.energy = float(mf.kernel())`.
- **Lazy import + probe:** the factory imports `gpu4pyscf_calculator` only when GPU is
  selected; the GPU module imports gpu4pyscf/cupy at module scope (never at package top level).
  Probe a usable device (driver present, `getDeviceCount() > 0`, free VRAM); wrap import/probe
  failures in a typed, catchable error.
- **Fallback:** GPU selected but unavailable/insufficient → fall back to `Psi4Calculator` and
  **log a WARNING**; a `require_gpu` flag raises instead. No GPU selected → Psi4, silently.

**Tests** (the fallback tests run on CI / any CPU host — no GPU required)
- Fast: factory returns the GPU class iff selected and a device is present; options merge.
- **CPU-only:** with gpu4pyscf monkeypatched to `ImportError`, `import snar_qc` and the Psi4
  path work; selecting GPU falls back to Psi4 and logs a WARNING; `require_gpu` raises.
- **Probe:** monkeypatch `getDeviceCount()` → 0 ⇒ fallback + warning; default (unset) ⇒ Psi4,
  no GPU import attempted.
- Slow (GPU, skip-marked when no device): pin the 5-ring reference complex energy to the Psi4
  reference (−839.808359 Ha, tol 1e-5 Ha) from `notes/2026-06-23_gpu_hessian_benchmark.md`.
- Re-run the full suite; Psi4 path unchanged.

**Done when:** energy parity shown (on a GPU host), CPU-fallback tests green on a CPU-only
host, full suite green, changes left uncommitted.
