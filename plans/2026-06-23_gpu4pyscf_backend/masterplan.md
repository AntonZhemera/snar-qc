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
- **Verify the analytic *solvent* Hessian explicitly — it is the whole point in solution.**
  The operation mapping lists `freq → mf.Hessian()` (gas) and `solvation → solvent.pcm` as
  separate rows; no stage composes them. Psi4 1.10.2 has **no analytic PCM gradient/Hessian**,
  so its solvent freq degrades to a *double* finite difference (a single 5-ring DMSO TS
  needed **2353 PCM-SCF displacements**, ~24 GB RSS, aborted on 2026-06-22). The Psi4 path
  therefore now does gas opt+freq + a PCM single-point correction (`barrier.py::_solvated_thermo`).
  gpu4pyscf ships solvent gradient/Hessian code, so it *likely* offers an analytic solvated
  Hessian — confirm this in **Stage C or E** on a PCM-wrapped `mf`. If it holds, the GPU
  backend removes the workaround for solution-phase freq (a fresh argument for the backend);
  if not, carry the gas-Hessian + PCM-SP protocol onto the GPU path too.
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

## Vendored-code boundary (predict_snar)

The GPU backend must **not** touch the vendored third-party package `src/predict_snar/`
(MIT, © Kjell Jorner; see `src/predict_snar/VENDORED.md`). The boundary that keeps this an
additive change with **no** vendoring obligation:

1. **Subclass, don't edit.** `GPU4PySCFCalculator` lives in `src/snar_qc/qc/` and subclasses
   `predict_snar.calculators.Calculator`, using its public interface only — exactly as
   `Psi4Calculator` does. No file under `src/predict_snar/` is modified.
2. **Selection stays snar_qc-side.** The backend factory (CPU-fallback contract) chooses the
   calculator. Do **not** add the GPU class to `predict_snar.jobs.py`'s by-name imports
   (`G16Calculator, XTBCalculator, CRESTCalculator`) or its dispatch.
3. **Orchestrate from snar_qc, not vendored `jobs.py`.** The vendored `Calculator` contract is
   async (`single_point/opt/freq` return a process to `.wait()` on) and `jobs.py` drives it via
   `.wait()` + `calculation_monitor()` + cclib file-parsing. The GPU/Psi4 calculators are
   **synchronous** (return energy in Hartree). Drive them from the snar_qc orchestrators
   (`snar_qc.poc.barrier`, `snar_qc.ts.psi4_tsscan`), which already accommodate the sync model.
   The trap a TS-path stage (D) can fall into: patching vendored `jobs.py` to accept a sync
   calculator. Don't — keep the GPU TS flow in snar_qc.
4. **No coupling to vendored data.** The GPU path uses pyscf basis sets
   (`gto.M(basis="def2-svp")`), not the vendored def2 pickles `predict_snar.data.get_basis` /
   `get_ecp` resolve from `../data`. Keep it that way.

**If a stage genuinely cannot avoid editing `predict_snar`**, it becomes a vendored
modification and the three-step obligation applies (`CLAUDE.md`): (a) mark the site inline
`# snar-qc: <reason>`, (b) add a *Local modifications* row to `VENDORED.md`, (c) correct the
stale header claims — it currently asserts every module but the two Windows-patched files is
"byte-for-byte upstream" and that "all snar-qc functionality lives in `snar_qc`", with `jobs`
listed among the verbatim modules. Flag this to the master session before proceeding; prefer a
snar_qc-side alternative.

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
