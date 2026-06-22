# Masterplan — gpu4pyscf backend for the snar-qc TS engine

**Date:** 2026-06-23
**Status:** draft (not started)
**Mode:** dev on Linux (write + test code; small pinned-energy tests only). Production QC
campaigns are launched on the Windows workstation. Code is multiplatform by default.

## Motivation

Per-substrate cost in the ΔG‡ engine is dominated by the **finite-difference TS Hessian**
(`notes/2026-06-21_poc_deltag_validation.md`; ~169 of 199 wall-minutes on lu_17). A
gpu4pyscf backend computes the Hessian **analytically** in one shot. Benchmark on the
5-ring reference complex (`notes/2026-06-23_gpu_hessian_benchmark.md`): analytic Hessian ≈87 s vs
FD ≈169 min, with B3LYP-D3BJ/def2-SVP energies matching Psi4 to 3×10⁻⁷ Ha. The win is
structural (analytic vs FD), not just clock speed.

> **Timing caveat carried from the note:** the CPU reference was a contended 4-worker batch
> (`THREADS=4`/worker), so the reported speedups are upper bounds. A clean dedicated-hardware
> re-benchmark is an explicit deliverable of Stage C/D, not an assumption.

## Seam

The engine already exposes a clean calculator interface. `Psi4Calculator`
(`src/snar_qc/qc/psi4_calculator.py`) subclasses the vendored
`predict_snar.calculators.Calculator` and routes `single_point / opt / opt_freq / freq /
ts / ts_freq` through `run_calc`, returning energy in Hartree and populating
`energy / free_energy / enthalpy / zpve / frequencies`. The GPU backend is a **sibling class
`GPU4PySCFCalculator(Calculator)` with the same public surface** — it does **not** edit the
vendored `predict_snar` package, so no VENDORED.md obligation is triggered. Backend is chosen
by config/env (`SNAR_QC_BACKEND=psi4|gpu4pyscf`); **Psi4 stays the default and fallback**
(CPU hosts, large substrates, anything exceeding 4 GB VRAM).

## Operation mapping

| Engine op | Psi4 today | gpu4pyscf |
|---|---|---|
| `single_point` | `psi4.energy` | `gpu4pyscf.dft.rks.RKS(mol,xc).density_fit(); mf.disp='d3bj'` (parity proven) |
| `opt` (min) | optking, Cartesian | geomeTRIC via `pyscf.geomopt.geometric_solver.optimize` on GPU gradients |
| `freq` + thermo | `psi4.frequencies` (FD) + Gibbs/ZPVE globals | `mf.Hessian()` (analytic) → `pyscf.hessian.thermo.harmonic_analysis` + `.thermo` |
| `ts` | optking `OPT_TYPE=TS`, FD Hessian | geomeTRIC `transition=True` **seeded by the analytic Hessian** |
| Mayer peak check (`qc/bond_orders.py`) | Psi4 Mayer | needs a pyscf Mayer equivalent, **or** keep this step on Psi4/xTB |
| solvation (`solvent`) | PCMSolver IEFPCM | `gpu4pyscf.solvent.pcm` — also offers SMD (gate separately) |

## Cross-cutting decisions

- **Psi4 default; GPU opt-in.** Backend selection must degrade gracefully and fall back to
  Psi4 when VRAM is insufficient or no CUDA device is present. See **CPU-fallback contract**.
- **4 GB VRAM ceiling.** Fine at def2-SVP for ≤~20-atom complexes; def2-TZVP / larger →
  Psi4/CPU. Make the fallback automatic and logged, not a hard crash.
- **PCM→SMD is a method change, not a free swap.** gpu4pyscf gains SMD (which Psi4 1.10.2
  lacked), but switching the solvation model alters results — gate it behind the
  `2026-06-22_solvation_revalidation` plan's revalidation, *not* this backend swap.
- **Thermochemistry convention.** Psi4 FD frequencies vs pyscf *analytic* frequencies differ
  by a few cm⁻¹; `snar_qc.qc.thermo` (Grimme qRRHO) consumes them. Re-confirm the single
  imaginary mode and the ΔG‡ offset — do not assume identical thermochem.

## CPU-fallback contract (non-GPU hosts)

snar-qc runs across hosts with **no NVIDIA GPU / CUDA** (the Windows workstation, CI, other
dev machines). The backend must never break those — this is a hard, tested requirement, not a
nice-to-have:

1. **Default is Psi4/CPU.** GPU is opt-in via `SNAR_QC_BACKEND=gpu4pyscf` (or config). A fresh
   checkout on any host behaves exactly as today; the GPU code path is not touched unless
   explicitly selected.
2. **gpu4pyscf / cupy are optional dependencies, never imported at package top level.** The GPU
   calculator module is imported **lazily**, only when the GPU backend is selected. `import
   snar_qc` and the entire Psi4 path must succeed with gpu4pyscf/cupy absent. Declare them as an
   extra (`pip install snar-qc[gpu]`), not a core requirement.
3. **Capability probe before use.** When GPU is selected, probe for a usable device (CUDA driver
   present, `cupy.cuda.runtime.getDeviceCount() > 0`, enough free VRAM for the job) — wrapped so a
   missing driver/library raises a typed, catchable error rather than a bare `ImportError`/segfault.
4. **Fallback policy — silent default, loud override.** No GPU requested → run Psi4 silently (the
   norm on CPU hosts, no noise). GPU requested but unavailable/insufficient → fall back to Psi4 and
   **log a WARNING** (a job you expected on the GPU must not silently degrade unnoticed). Provide a
   strict mode (e.g. a `require_gpu` flag) that errors instead of falling back, for benchmarking.
5. **Single chokepoint.** The backend factory owns probe + lazy import + fallback decision;
   calculators never import GPU libraries directly. One place to reason about, one place to test.

## Stages (each independently shippable; sub-session executes exactly one, leaves uncommitted)

- **A — single point** (`prompt_01_SinglePoint.md`): `GPU4PySCFCalculator.single_point`,
  energy parity vs Psi4 on POC geometries. Lowest risk.
- **B — gradient + min-opt** (`prompt_02_GradientMinOpt.md`): GPU gradients, geomeTRIC
  minimisation.
- **C — analytic Hessian + thermo** (`prompt_03_HessianThermo.md`): frequencies + qRRHO ΔG‡
  parity vs Psi4; **first clean dedicated-hardware Hessian re-benchmark.**
- **D — TS optimisation** (`prompt_04_TSOpt.md`): geomeTRIC `transition=True` seeded by the
  analytic Hessian — the hours→minutes payoff; clean per-job + throughput benchmark.
- **E — revalidation** (`prompt_05_Revalidation.md`): re-run the 10-substrate POC slice on the
  GPU backend; confirm the note's Spearman/Pearson hold. Decide PCM/SMD separately.

## Tests

- Each stage adds tests (fast monkeypatched contract tests + a slow pinned-energy/-frequency
  test), in the repo's existing TDD style.
- **The full test suite must be re-run after each stage and again after the complete backend
  is in place** — not only the new tests. A backend that passes its own stage tests can still
  regress the Psi4 path or the shared `barrier` / `thermo` drivers.
- **CPU-fallback tests (CI-runnable, no GPU needed):** with gpu4pyscf absent (monkeypatched
  `ImportError`) `import snar_qc` and the Psi4 path still work; a probe returning zero devices
  falls back to Psi4 and logs a WARNING; the default (no backend selected) picks Psi4 and never
  imports a GPU library; `require_gpu` raises instead of falling back. GPU-only tests carry a
  skip marker when no CUDA device is present, so the suite stays green on CPU hosts and CI.

## Risks

- geomeTRIC TS convergence on rigid planar aryl-nitriles (the same ill-conditioning that
  forced Cartesian min-opt on the Psi4 path) — validate on the known-hard substrates early.
- Analytic-Hessian memory blow-up at larger basis on 4 GB — keep the Psi4 fallback wired.
- D3BJ contribution to the analytic Hessian — confirm gpu4pyscf includes it, else add it.
