# 2026-06-24 — GPU barrier pipeline runs end-to-end (Stage E unblocked)

**What:** made the full ΔG‡ barrier pipeline (`snar_qc.poc.barrier.compute_barrier`) run
end-to-end on the **gpu4pyscf backend**, gas phase, and proved it on the lu_0 smoke
substrate. This is the prerequisite the Stage E POC revalidation
(`plans/2026-06-23_gpu4pyscf_backend/prompt_05_Revalidation.md`) was blocked on.

## The two blockers (both resolved)

The kickoff flagged one gap (the Mayer peak check reads a Psi4 wavefunction); on-device
work surfaced a second (the pipeline was not even *importable* on the GPU host).

### 1. The barrier pipeline was not importable on `gpuqc` (no Psi4)

`gpuqc` deliberately carries no Psi4 (conda-only; would risk the pinned cupy/cuTENSOR
wheels). But `barrier.py` eagerly imported `Psi4Calculator` and `bond_orders.py` did
`import psi4` at module scope, so `import snar_qc.poc.barrier` died with
`ModuleNotFoundError: psi4` on the only env with the GPU stack. Fix — the **mirror of the
CPU-fallback contract** (make Psi4 lazy on the GPU-needed path):

- `bond_orders.py`: `import psi4` moved into `Psi4BondOrders.__init__`.
- `barrier.py`: `Psi4Calculator` import moved into `_solvated_thermo` (only reached when a
  solvent is requested, which the gas-phase GPU path never is); its annotations moved to
  `TYPE_CHECKING`.

`import snar_qc.poc.barrier` now succeeds on `gpuqc` with `psi4` absent from `sys.modules`,
and the Psi4 env is unaffected.

### 2. The Mayer peak check was Psi4-only — added a pyscf equivalent

The relaxed-scan peak validation built `Psi4BondOrders(calc.wavefunction)`; the GPU
calculator exposes `mean_field`, not `wavefunction`. Because `gpuqc` has no Psi4, "keep the
step on Psi4" was not an option — the pyscf Mayer adapter was forced. Added to
`bond_orders.py`:

- **`PyscfBondOrders`** — Mayer orders straight from a (gpu4)pyscf mean-field
  (`B_AB = 2·Σ (PaS)(PaS)ᵀ + (PbS)(PbS)ᵀ` over atom blocks; NumPy-only, cupy arrays
  pulled to host via `.get()`). Validated vs the references the Psi4 adapter is pinned to:
  **N₂ = 3.0000, H₂O O–H = 1.0104** (Psi4 pin 1.0104), H–H ≈ 0.006. So the 0.05/0.5
  `validate_peaks` thresholds read the same scale on either backend. The matrix is
  symmetrised (`0.5·(M+Mᵀ)`) for a bit-exact `get_bo(i,j)==get_bo(j,i)`.
- **`bond_orders_from_calculator(calc)`** — dispatch factory (`wavefunction` → Psi4,
  `mean_field` → pyscf). `Psi4TSScan.run_sps` builds bond orders through it per scan point,
  so the TS scan is backend-agnostic.

## VRAM bug the smoke caught (4 GB card)

The first end-to-end run errored mid-scan: `insufficient free VRAM: 0.94 GB < 1.00 GB`.
Two compounding causes, both fixed:

1. **`PyscfBondOrders` retained `self.mean_field`** → pinned the GPU density-fit tensors
   (~0.2 GB each). The scan keeps ~14 bond-order objects alive in `nbo_data` → ~2.8 GB
   pinned → card exhausted. Fix: `PyscfBondOrders` keeps only the host-side `bo_matrix`.
2. **cupy pools freed device blocks** rather than returning them to the driver, so the
   probe-visible free VRAM drifts down across the ~14 sequential single points. Fix:
   `GPU4PySCFCalculator.free_device_memory()` (drops `mean_field` + frees the cupy pools),
   called duck-typed from `run_sps` after each point (the Psi4 path has no such method).

Confirmed in isolation: 16 sequential 21-atom single points with **all 16 bond-order
objects retained** now hold free VRAM **flat at 3.46 GB** (was collapsing to 0.94 GB).

## End-to-end validation — lu_0 (para-fluoronitrobenzene + methylamine, gas phase)

`SNAR_QC_BACKEND=gpu4pyscf SNAR_QC_REQUIRE_GPU=1`, concerted coordinate. Result:
**`status=completed`, 1/1 confirmed saddle.**

| field | value |
|---|---|
| ΔG‡(qh) | **42.56 kcal/mol** (in the POC F-substrate band 42.6–47.7) |
| ΔG / ΔH / ΔE‡ | 42.22 / 29.85 / 28.57 kcal/mol |
| n_imag_ts | 1 — TS imag freq **−295.7 cm⁻¹** (Stage D pin −294.1 ✓) |
| references | n_imag_arx = 0, n_imag_amine = 0 |
| peak | index 8 of 14 scan points |

### Measured per-stage wall (RTX 3050 Ti, one job) — feeds the deferred T2 TS benchmark

| stage | wall |
|---|---|
| scan (xtb) | 157 s (2.6 min) |
| DFT SPs + Mayer peak validation (GPU) | 121 s (2.0 min) |
| **TS opt+freq (GPU)** | **1461 s (24.4 min)** |
| ArX reference opt+freq | 100 s (1.7 min) |
| amine reference opt+freq | 15 s (0.2 min) |
| **total** | **1854 s (30.9 min)** |

The TS opt+freq dominates (~79%). This **confirms** the ~25 min figure for the slow TS
test (the test docstring's "~20 min" was corrected to ~24 min); the per-step analytic
Hessian in the geomeTRIC saddle search, not the validating Hessian alone, is the cost.

## Operational gaps on `gpuqc` (for the Stage E campaign)

`gpuqc` has neither Psi4 nor **xtb** (the relaxed-scan binary). xtb is a subprocess (no
ABI risk), so this smoke pointed `PATH` at the `snar-qc` env's xtb 6.7.1 — verified it runs
from a `gpuqc` shell. The 10-substrate Stage E campaign needs xtb reachable from `gpuqc`
(a light `conda install xtb`, or the PATH approach).

## Tests

- New: `PyscfBondOrders` None/kind guards + `bond_orders_from_calculator` dispatch (fast,
  `tests/test_bond_orders.py`); on-device Mayer parity vs N₂/H₂O (slow,
  `tests/test_gpu4pyscf_calculator.py::test_pyscf_bond_orders_mayer_matches_textbook`).
- Green: snar-qc fast **69 passed**, slow Psi4 tsscan/bond_orders **9 passed**, gpuqc fast
  contract **9 passed** + the new Mayer test. ruff clean.

## What remains to close the plan

The Stage E **10-substrate POC revalidation campaign** (`data/external/lu74_poc_slice.csv`)
on the GPU backend — ~31 min/substrate ⇒ ~5 h GPU — then confirm the within-leaving-group
correlations hold (Cl ρ≈0.96/R²≈0.95, F ρ=1.0) vs `2026-06-21_poc_deltag_validation.md`,
and archive the plan. Smoke artefacts: `data/processed/gpu_smoke_lu0/lu_0/`.
