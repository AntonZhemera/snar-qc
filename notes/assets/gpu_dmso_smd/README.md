# `gpu_dmso_smd/` — GPU/gpu4pyscf DMSO SMD ΔG‡ (full Lu_74 cohort)

**What:** DMSO ΔG‡ for the **full 74-substrate Lu_74 cohort** (Cl 51 / Br 14 / F 9) via the
**gpu4pyscf** backend with **SMD** solvation. A model the Psi4 1.10.2 path cannot provide, so
there is **no CPU counterpart**. The **default / primary** model in the standard workflow —
best ranking and lowest MAE of the three.

- `poc_validation_join.csv` / `poc_validation_stats.json` / `poc_validation_scatter.png`.
- `poc_validation_per_lg.png` — per-leaving-group calibration panels (F / Cl / Br).

**When:** full-cohort sweep 2026-06-26; **OOM recovery + full-cohort revalidation 2026-06-28**.
Write-up: `notes/2026-06-28_lu74_full_deltag_analysis.md` (slice-era comparison:
`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`).

**Method:** B3LYP-D3BJ / def2-SVP, methylamine, concerted coordinate, **DMSO via SMD single
points on gas geometries** (gas Hessian). Same pipeline as `../gpu_dmso_iefpcm/`, only
`--solvent-model smd`. Produced by `sweep_solvent.py` off `data/processed/gpu_dmso_gas`.

**Stack:** gpu4pyscf (CUDA wheels) + pyscf + geomeTRIC, xtb 6.7.1, env `gpuqc`, RTX 3050 Ti (4 GB).

**Source:** `data/processed/gpu_dmso_smd`; input `data/external/lu74_full.csv` (internal,
gitignored). The committed, publicly reproducible subset is `data/external/lu74_solv_slice.csv`.

**Headline — best single model.** **69/74 completed.** Pooled ρ=**0.82**, r=0.81, MAE **5.95**
(offset-corrected 1.88) — best of the three on every metric. Per-LG mean offset Br +5.07 /
Cl +5.57 / **F +9.01** — SMD narrows the F over-penalisation most (F−Cl gap +3.4 vs IEF-PCM's
+5.0). Within-LG R² Br 0.69 / Cl 0.71 / F 0.83.

**SMD failures were a 4 GB-GPU artefact, not SCF non-convergence.** The first full-cohort
sweep completed only 60/74 — the 9 "SMD-only" failures (lu 48, 52, 58, 60, 61, 62, 63, 64, 68)
all raised `cudaErrorMemoryAllocation` at SCF init, ~7 s in. The cause was **cumulative GPU
memory** across the batch (CuPy's pool is not released between substrates in one process), not
a property of SMD: re-running each SMD single point in a **fresh process** recovered all 9
(peak ~1.1 GB, well under the 4 GB ceiling). This supersedes the earlier "SMD-SCF failure"
reading (e.g. `lu_48` / `Clc1cc(N2CCCC2)ncn1`).

**Remaining failures (5):** lu 10, 20, 21, 36, 53 — fail upstream in the **gas** backbone
(gas opt+freq OOM), so no SMD single point is attempted. Recover the gas backbone (larger GPU
or CPU/Psi4) first.

**Comparability:** model-comparable to `gpu_dmso_iefpcm/` and `gpu_gas/` (same engine &
cohort). No CPU/Psi4 SMD baseline exists. **Not** comparable to gas folders for absolute ΔG‡.
