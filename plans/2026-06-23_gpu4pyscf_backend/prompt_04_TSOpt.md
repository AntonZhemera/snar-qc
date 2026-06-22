# Stage D — TS optimisation (the payoff)

**Goal:** `GPU4PySCFCalculator.ts` / `ts_freq` — a saddle search driven by GPU gradients and
**seeded with the analytic Hessian**, replacing optking's FD-Hessian TS path.

Depends on Stage C (analytic Hessian).

**Files**
- Extend `src/snar_qc/qc/gpu4pyscf_calculator.py` with the `ts` / `ts_freq` paths.

**Approach**
- geomeTRIC TS search: `optimize(mf, transition=True, ...)`, providing the analytic Hessian
  (Stage C) as the initial/recomputed Hessian rather than finite-difference.
- Validate the saddle by the subsequent `freq` (exactly one imaginary mode), as the Psi4
  `ts_freq` path does.
- Confirm the converged TS matches the Psi4-located saddle on a known substrate (geometry,
  imaginary frequency, ΔG‡ within tolerance).

**Tests**
- Slow (GPU): TS on the smoke-test substrate (1-fluoro-4-nitrobenzene + methylamine) — one
  imaginary mode, ΔG‡ within tol of the Psi4 result.
- Re-run the full suite.

**Clean benchmark (deliverable):** on an idle host, full TS pipeline GPU vs Psi4 per-substrate,
**and** throughput — 4 concurrent CPU workers (the production `N_WORKERS=4` config) vs one
4 GB GPU job at a time. Record both in the benchmark note.

**Done when:** TS parity + one imaginary mode, latency *and* throughput benchmark recorded,
suite green, uncommitted.
