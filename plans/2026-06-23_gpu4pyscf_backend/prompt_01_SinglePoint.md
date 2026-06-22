# Stage A — GPU4PySCFCalculator.single_point

**Goal:** a gpu4pyscf backend that produces B3LYP-D3BJ/def2-SVP single-point energies at
parity with `Psi4Calculator`, behind the same `Calculator` interface.

**Files**
- New `src/snar_qc/qc/gpu4pyscf_calculator.py`: `GPU4PySCFCalculator(Calculator)`.
- Backend selection helper (factory by `SNAR_QC_BACKEND` env / config), wired where
  `Psi4Calculator` is constructed.

**Approach**
- Mirror `Psi4Calculator`'s option surface (`functional / basis_set / dispersion /
  charge / scf_type`); build the molecule from the ASE `Atoms` (geometry + `info["charge"]`),
  spin by electron-count parity (RKS singlet / UKS doublet), as `Psi4Calculator._multiplicity`.
- `mol = gto.M(...)`; `mf = gpu4pyscf.dft.rks.RKS(mol, xc=functional).density_fit();
  mf.disp = "d3bj"`; `self.energy = float(mf.kernel())`.
- **Fallback:** if no CUDA device or VRAM insufficient, raise a typed error the factory
  catches to fall back to Psi4 (log the reason).

**Tests**
- Fast: monkeypatched — backend factory returns the GPU class iff selected; options merge.
- Slow (GPU): pin the 5-ring reference complex energy to the Psi4 reference
  (−839.808359 Ha, tol 1e-5 Ha) from `notes/2026-06-23_gpu_hessian_benchmark.md`.
- Re-run the full suite; Psi4 path unchanged.

**Done when:** energy parity shown, new + full suite green, changes left uncommitted.
