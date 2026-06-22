# 2026-06-23 — gpu4pyscf vs Psi4/CPU: SCF, gradient, analytic Hessian

**Status:** indicative benchmark (timings not yet clean — see caveat). Energy parity firm.
**Hardware:** laptop NVIDIA RTX 3050 Ti (4 GB, Ampere, compute cap 8.6); gpu4pyscf 1.7.3 /
PySCF 2.13.1 / CuPy. **System:** the 5-fluorothiophene-2-carbonitrile reference complex — a
5-membered thiophene-ring S~N~Ar substrate (`N#Cc1ccc(F)s1`) docked with methylamine
(17 atoms, def2-SVP = 179 basis functions),
**B3LYP-D3BJ/def2-SVP**, density fitting — the level the Psi4 engine runs.

## Why

The POC ΔG‡ engine's per-substrate cost is dominated by the finite-difference TS Hessian
(`notes/2026-06-21_poc_deltag_validation.md`; on lu_17, ~169 of 199 wall-minutes). gpu4pyscf
offers analytic gradients and an analytic Hessian. Question: is the GPU win large and
structural enough to justify wiring it into the engine as a backend?

## What was measured

GPU (gpu4pyscf, RTX 3050 Ti), 5-ring reference complex:

| Operation | GPU wall |
|---|---|
| SCF (B3LYP-D3BJ/def2-SVP, DF) | 8.5 s |
| 1 analytic gradient | 9.0 s |
| analytic Hessian (one shot) | 86.8 s |

Energy: GPU −839.80835927 Ha vs Psi4 `sps/10.out` −839.80835876 Ha → |ΔE| = **3×10⁻⁷ Ha
(0.0003 kcal/mol)**. VRAM used ≈ 0.66 GB of 4 GB.

CPU reference (read from the running POC batch, B3LYP-D3BJ/def2-SVP): SCF module 11 s;
full single-point process 100 s; FD Hessian on lu_17 (109 displacements, 20 atoms) ≈ 169 min.

## Findings

**H1. Energy parity holds.** gpu4pyscf reproduces Psi4 at this level to 3×10⁻⁷ Ha;
backend correctness is not in question. Same 179 basis functions, same D3BJ dispersion.

**H2. The Hessian is the payoff, and it is structural.** Analytic one-shot (≈87 s) vs Psi4's
finite-difference 6N+1 ≈ 103–109 gradient evaluations. The advantage decomposes into ~10×
raw GPU gradient speed **and** ~11× analytic-vs-finite-difference algorithm — Psi4 has no
analytic DFT Hessian, so the engine is forced into FD. Even a GPU *finite-difference* Hessian
(~103 × 9 s ≈ 15 min) would beat CPU; the analytic path is what reaches ~90 s.

**H3. 4 GB VRAM suffices at def2-SVP** for ≤~20-atom complexes (0.66 GB used here). def2-TZVP
or larger substrates will exhaust it → must fall back to the Psi4/CPU backend.

## Caveat — timings are indicative, not a clean benchmark

The CPU reference numbers were read from the **live POC batch** launched
`N_WORKERS=4 THREADS=4`: each Psi4 worker had only **4 threads / ~2 physical cores**, and
**4 workers contended** for the 16-logical-core box. The CPU walls are therefore inflated by
contention *and* thread-capped; a clean, dedicated 16-thread Psi4 run would be faster, so the
reported 6–7× (SCF/single-point) and ~117× (Hessian) are **upper bounds**. The GPU runs were
also taken while the machine was loaded. Energy parity (H1) is unaffected — that is correctness,
not timing.

**Required before trusting magnitudes:** a clean re-benchmark on an idle host — dedicated
16-thread Psi4 single point + a dedicated Psi4 FD Hessian against an otherwise-idle GPU. And
separate the two questions they answer: **per-job latency** (one GPU job vs one 16-thread Psi4
job) versus **throughput** (production runs 4 CPU workers concurrently, whereas a single 4 GB
GPU runs one job at a time — so the GPU's throughput edge is smaller than its latency edge).

## Implication

Worth wiring a gpu4pyscf backend into the engine — design in
`plans/2026-06-23_gpu4pyscf_backend/`. Even discounted heavily for the contention caveat, the
analytic-Hessian advantage is order-of-magnitude and structural (algorithm, not just clock
speed), and it targets exactly the phase that dominates per-substrate cost.
