# Stage C — analytic Hessian + thermochemistry

**Goal:** `GPU4PySCFCalculator.freq` producing harmonic frequencies and qRRHO-ready
thermochemistry from the **analytic** Hessian, at parity with the Psi4 FD-freq path.

**Files**
- Extend `src/snar_qc/qc/gpu4pyscf_calculator.py` with `freq` (and the `opt_freq` combo).

**Approach**
- `hess = mf.Hessian().kernel()` (confirm the D3BJ contribution is included; add it if not).
- Frequencies + thermochem: `pyscf.hessian.thermo.harmonic_analysis(mol, hess)` →
  `pyscf.hessian.thermo.thermo(...)` for Gibbs / enthalpy / ZPVE.
- Populate `self.frequencies` in the **signed cm⁻¹ convention** the Psi4 path uses (imaginary
  modes as negative reals — see `Psi4Calculator._capture_thermo`), so `snar_qc.qc.thermo`
  consumes them unchanged. A TS must show exactly one negative entry.

**Tests**
- Slow (GPU): pin frequencies + ΔG‡ on a known case; compare to the Psi4 freq result.
  Allow a few cm⁻¹ tolerance (analytic vs FD) but **flag if the ΔG‡ offset shifts** the POC
  ranking — see masterplan "thermochemistry convention".
- Re-run the full suite.

**Clean re-benchmark (deliverable):** on an **idle** host, time the analytic Hessian (GPU)
vs a **dedicated 16-thread** Psi4 FD Hessian on the same substrate. Replace the indicative
~117× in `notes/2026-06-23_gpu_hessian_benchmark.md` with the clean per-job figure.

**Done when:** frequency/ΔG‡ parity shown, clean Hessian benchmark recorded, suite green,
uncommitted.
