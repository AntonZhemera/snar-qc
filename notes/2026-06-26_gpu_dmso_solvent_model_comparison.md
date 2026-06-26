# GPU DMSO campaign — solvent-model & engine comparison (gas / IEF-PCM / SMD)

**When:** 2026-06-26. **Runs validated:** `data/processed/gpu_dmso_gas`,
`gpu_dmso_iefpcm`, `gpu_dmso_smd` (18-substrate `lu74_solv_slice`, Br+Cl+F).
**Seed:** runbook `notes/2026-06-25_gpu_dmso_campaign_runbook.md`.
**Assets:** `notes/assets/{gpu_gas,gpu_dmso_iefpcm,gpu_dmso_smd}/` (per-run joins/stats),
comparison figures in `notes/assets/comparison/`.

This closes the runbook's three goals: IEF-PCM **engine parity** vs the Psi4 baseline, the
new **SMD** capability, and the gas→solvated **model comparison**. Headline: solvation
makes the POC track Lu_74 far better, **SMD** is the best single model, the GPU engine is at
parity with Psi4 in gas and **more robust** on the solvated slice, and the **F leaving-group
stays a distinct, over-penalised cluster** in every model — exactly as flagged.

> Scope note (public/internal red line): this is **method characterisation** of the QC
> tooling against the literature Lu_74 barriers — which engine/solvent model the tool should
> use, and how well it ranks. It is not the project's own reactivity analysis. The full-cohort
> ΔG‡ production run (`data/external/lu74_full.csv`, gitignored) is the research-adjacent part
> and stays internal.

## 1. Solvent-model comparison (GPU, 18-substrate slice)

Pooled metrics, computed ΔG‡(qh) vs Lu_74. Ranking (Spearman/Pearson) is the headline; the
amine/charge mismatch (gas-phase methylamine vs solution alkoxide) guarantees a positive
offset, so magnitude is secondary.

| model | n | Spearman ρ | Pearson r | MAE | offset-corr MAE | mean offset |
|:--|:--:|--:|--:|--:|--:|--:|
| **gas**    | 18 | 0.610 | 0.430 | 14.55 | 5.39 | +14.55 |
| **IEF-PCM**| 18 | 0.769 | 0.774 | 7.16  | 2.80 | +7.16  |
| **SMD**    | 17 | **0.900** | **0.857** | **5.76** | **2.33** | +5.76 |

Every metric improves monotonically gas → IEF-PCM → SMD (`fig2_metrics_by_model.png`,
`fig3_model_scatter.png`). SMD roughly **doubles** the rank correlation over gas (0.61→0.90)
and **halves** the offset-corrected MAE (5.4→2.3). IEF-PCM captures most of the gain; SMD's
non-electrostatic terms add the rest.

**One caveat — robustness:** IEF-PCM completed **18/18**; SMD completed **17/18**, losing
`lu_48` (`Clc1cc(N2CCCC2)ncn1`, the pyrrolidinyl-triazine arylator) to an SMD-SCF failure on
the gas geometry. So SMD wins on accuracy, IEF-PCM on coverage. Both are nearly free off one
gas backbone (3 SCFs, ~1–2 min/substrate), so the practical answer is **produce both**.

## 2. Partition by leaving group — F is its own cluster

Mean offset (computed − exp), per LG, per model (`fig1_offset_by_lg_model.png`):

| LG (n) | gas | IEF-PCM | SMD |
|:--|--:|--:|--:|
| **Br** (6) | +10.1 | +4.8 | +4.1 |
| **Cl** (7) | +11.5 | +5.8 | +4.7 |
| **F**  (5) | +24.2 | +11.8 | +9.1 |

The hypothesis from the memo holds precisely: **Br ≈ Cl** (offsets within ~1 kcal/mol in
every model) while **F sits ~2× higher** — a separate cluster, not a continuum. Solvation
shrinks the absolute offsets but does **not** merge the clusters: the F − Cl offset gap is
+12.8 (gas) → +6.0 (IEF-PCM) → **+4.4 (SMD)**. SMD narrows the F over-penalisation the most,
but it does not remove it.

Within-LG ranking is excellent once solvated — IEF-PCM: Br ρ=1.00, Cl ρ=0.96, F ρ=0.90;
SMD: Br ρ=1.00, Cl ρ=0.94, F ρ=1.00. **Consequence for downstream use:** the offset is
irreducibly LG-dependent, so correct it **per leaving group**, never with one global shift.
The pooled offset-corrected MAE understates per-LG accuracy (per-LG oc-MAE is ~1–1.9 kcal/mol
for both solvated models).

## 3. CPU vs GPU engine comparison

`fig4_cpu_gpu_parity.png` (left = gas, right = DMSO IEF-PCM).

- **Gas — parity confirmed.** `cpu_gas` ↔ `gpu_stage_e` (same 10 substrates):
  **max |Δ| = 0.20 kcal/mol**, identical correlation stats, despite four implementation
  differences (optking→geomeTRIC, FD→analytic Hessian, Psi4→pyscf bond orders). The two LG
  clusters appear identically on both engines — they are physics, not a GPU artefact.
- **DMSO IEF-PCM — GPU is at parity on Br/Cl and more robust overall.** Br/Cl agree within
  ~0.5–0.8 kcal/mol; **F diverges 1–3 kcal/mol** (PCMSolver Bondi cavity vs gpu4pyscf's PCM
  for the small fluorine — a real continuum-model difference, not noise). More importantly,
  the GPU run **fixed two CPU failures**: `lu_27` (nitrile TS, CPU optking 150-iter
  non-convergence) completed via geomeTRIC, and `lu_65` (CPU returned a **spurious** TS,
  ΔG‡ 4.05 kcal/mol — below reactants) came back sensible at 32.07. `lu_65` is the lone
  off-parity point on the right panel; it is a CPU defect, not a GPU one.

So GPU is not merely "as good as" Psi4 here — on the solvated slice it is the **more robust**
engine (no PCMSolver cavity deaths, geomeTRIC clears the nitrile TS), and ~10× cheaper via
the gas-once-then-sweep pattern.

## 4. Recommended standard workflow

| axis | choice | why | fallback |
|:--|:--|:--|:--|
| **Engine** | **GPU / gpu4pyscf** | gas parity (<0.2), more robust solvated, ~10× cheaper (gas+sweep) | **CPU / Psi4** where no CUDA device — the only portable path |
| **Solvent model** | **SMD** (primary) | best ranking (ρ=0.90), lowest MAE, narrowest LG spread, best on F | **IEF-PCM** for substrates where SMD SCF fails (e.g. `lu_48`) |
| **Calibration** | **per-LG offset** + report ranking | offset is irreducibly LG-dependent (F cluster) | — |

GPU is laptop-specific here (one RTX 3050 Ti); the CPU/Psi4 path remains the documented
portable fallback for hosts without a GPU. Both solvent models cost ~1–2 min/substrate once
the gas backbone exists, so the production recipe is **gas backbone once → sweep SMD and
IEF-PCM**, prefer SMD, fall back to IEF-PCM per failed substrate.

## 5. Side question — is the fixed amine reference reused?

**No.** `compute_barrier` recomputes the methylamine reference (gas opt+freq **and** the
solvent SP) from scratch for **every substrate** (`barrier.py`, stage `amine_opt_freq`).
Methylamine is invariant across the whole cohort, so this is pure redundancy. Cost is small —
median **~12 s/substrate** vs **~21 min** for the TS opt+freq and ~2 min for the ArX — so it
is not a bottleneck, but it is ~12 s × N of wasted work (≈15 min over the full Lu_74) and a
tiny source of per-run amine-geometry noise.

The per-substrate `gas_thermo.json` cache **does** let `sweep_solvent.py` reuse the gas
backbone across solvents, but there is **no shared fixed-component asset** across substrates
or runs. **Suggested low-priority polish:** cache the methylamine gas thermo once (a
`assets/amine_ref/CN_b3lyp_def2svp.json` keyed by amine SMILES + level of theory) and reuse
it for the separated-reactants reference; the ArX is substrate-specific and cannot be cached,
but the amine can. Tracked as a follow-up, not blocking.

## 6. Next — full Lu_74 production

Standard workflow decided (GPU + SMD/IEF-PCM), the remaining cohort can be computed. Input
prepared at **`data/external/lu74_full.csv`** (74 rows, gitignored — research-adjacent, not a
general tool): Cl 51 / Br 14 / F 9, exp ΔG‡ 14.67–22.85 kcal/mol. The 18-substrate slice is
already done; **56 remain** (Cl 44 / Br 8 / F 4). At ~35 min/substrate for the gas backbone
this is a ~30 h GPU campaign + cheap sweeps — a separate dispatched run, see the SOP
(`docs/sop_snar_deltag.md`). Validate per-LG against the slice's offsets above.

## Provenance matrix update

Two new cells confirmed in `notes/assets/gpu_stage_e/README.md`: gpu4pyscf × DMSO now ✓ for
**both** IEF-PCM and SMD (the Psi4 path cannot provide SMD at all). See that file.
