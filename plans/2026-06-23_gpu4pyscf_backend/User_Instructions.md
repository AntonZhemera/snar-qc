# User Instructions — gpu4pyscf backend

How this multi-stage plan is run (mirrors the repo's master/sub-session model).

- The **master session** reads `masterplan.md`, dispatches one `prompt_NN_*.md` per
  sub-session, reviews the diff, runs tests, and commits.
- A **sub-session** executes **exactly one stage**, adds its tests, and **leaves changes
  uncommitted** for the master to review. It does not touch the vendored `predict_snar`
  package (the GPU backend is a new `snar_qc.qc` class).
- **Stages are ordered A→E** but A/B/C are independently shippable; D depends on C, E on D.

Environment for GPU stages (Linux dev host with the RTX 3050 Ti):

```bash
conda activate gpuqc      # gpu4pyscf-cuda12x 1.7.3 / pyscf 2.13.1 / cutensor active
```

The `gpuqc` env already has the NVIDIA component wheels and an `activate.d` hook that puts
the cuTENSOR/cuBLAS lib dirs on `LD_LIBRARY_PATH`. Psi4 stays in the existing `snar-qc` env.

Acceptance for every stage:

1. New stage tests pass.
2. **The full existing test suite is re-run and still green** — required after each stage and
   again once the whole backend lands. The GPU backend must not regress the Psi4 path or the
   shared `barrier` / `thermo` drivers.
3. Numerical parity vs Psi4 is shown where the stage produces an energy/frequency/ΔG‡.

Benchmarking note: the timings in `notes/2026-06-23_gpu_hessian_benchmark.md` are **indicative
only** (CPU reference came from a contended 4-worker batch). Stages C and D must produce a
**clean re-benchmark on an idle host** — dedicated 16-thread Psi4 vs an otherwise-idle GPU —
and report per-job latency *and* batch throughput (4 concurrent CPU workers vs one 4 GB GPU job).
