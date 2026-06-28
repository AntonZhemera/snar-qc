# SOP — Compute S~N~Ar ΔG‡ (aryl halide + methylamine) for any input table

**Status:** living reference (refresh in place; not dated). **Supersedes**, as the standing
procedure, the dated campaign runbooks under `notes/` (`2026-06-24_gpu_stage_e_campaign_runbook.md`,
`2026-06-25_gpu_dmso_campaign_runbook.md`) — those remain as historical run records.

This is the standard operating procedure for computing a first-principles S~N~Ar activation
free energy ΔG‡ for the **aryl halide + methylamine** model reaction, for *almost any* input
list of substrates. The model amine is fixed (methylamine, `CN`); you supply the aryl
halides.

> **What the number means.** ΔG‡(qh) is the quasi-harmonic Gibbs activation free energy of
> the concerted S~N~Ar TS relative to separated reactants (gas-phase methylamine + aryl
> halide), with an implicit-solvent single-point correction. It uses a **neutral model amine
> in place of the real nucleophile**, so against an experimental (anionic, solution) dataset
> it carries a **systematic, leaving-group-dependent offset** — the headline output is the
> **ranking**, not the absolute value. See
> `notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`.

## 1. Input contract

The runner accepts a single ad-hoc SMILES (`--smiles`) or a CSV work-list (`--substrates`).
For a CSV, the only hard requirement is a SMILES column.

| Field | Required? | Meaning / behaviour if omitted |
|:--|:--|:--|
| `smiles_canonical` | **Mandatory** | The aryl halide SMILES. The one field every row must have. |
| `lu_id` / `substrate_id` / `arylator_catcode` | **Recommended** | Traceability id and per-substrate output-dir tag. If none is present, the tag **falls back to a slug of the SMILES** (`task_tag`, `worker.py`). Provide one so runs/sidecars are human-readable and joinable. |
| `leaving_group` | Optional | Element (`F`/`Cl`/`Br`/`I`) of the halide to displace. If omitted, it is **auto-detected**: the halogen on the most-activated aromatic carbon (most ring heteroatoms / nearby nitro), ties broken on lowest atom index (`_find_leaving_halide`, `complex.py`). **Supply it explicitly when a substrate carries more than one halogen** and you want to pin which one leaves (e.g. `Fc1ccc(Cl)nc1` → `Cl`). |
| anything else (`inchikey`, `delta_g_kJmol`, `EA`, descriptors, …) | Optional | Ignored by the runner; carried only for your own downstream joins. |

So a minimal valid input is a one-column CSV of `smiles_canonical`; a good input adds an id
column; add `leaving_group` only where the halide choice is ambiguous.

A single substrate without a file:
```bash
python scripts/run_poc.py --smiles "O=[N+]([O-])c1ccc(F)cc1" --leaving-group F \
  --outdir data/processed/adhoc_smoke
```

## 2. Method (model chemistry)

B3LYP-D3BJ / def2-SVP; concerted S~N~Ar coordinate d(C–Nu) − d(C–LG); qRRHO thermo
(100 cm⁻¹ cutoff, 298.15 K); separated-reactants reference. Pipeline per substrate:

1. xTB-GFN2 relaxed scan along the concerted coordinate → guess TS region.
2. DFT scan single points (solvated, if a solvent is set).
3. **Gas** TS opt + freq (geomeTRIC, analytic Hessian on GPU) → one significant imaginary mode.
4. **Gas** ArX and amine references, opt + freq.
5. Solvent enters as an implicit **single-point correction** `E(solv) − E(gas)` on each gas
   geometry (the "SP-on-gas" recipe). ΔG‡(qh) = G(TS) − [G(ArX) + G(amine)].

The gas backbone (steps 1, 3, 4) is **solvent-independent** and is cached
(`gas_thermo.json` + `*_opt.xyz` per substrate), so any solvent/model is a cheap re-run of
only step 5 via `scripts/sweep_solvent.py`.

## 3. Engine & solvent-model choice (the standard workflow)

Established by the 2026-06-26 comparison (`notes/2026-06-26_gpu_dmso_solvent_model_comparison.md`):

| axis | **Default** | Fallback |
|:--|:--|:--|
| Engine | **GPU / gpu4pyscf** (env `gpuqc`) — gas parity with Psi4 (<0.2 kcal/mol), more robust solvated, ~10× cheaper via gas+sweep | **CPU / Psi4** (env `snar-qc`) on hosts with no CUDA device — the portable path |
| Solvent model | **SMD** — best ranking (ρ≈0.82 full Lu_74, lowest MAE, narrowest leaving-group spread) | **IEF-PCM** where SMD is unavailable (CPU/Psi4 host) — SMD is GPU-only. SMD does **not** SCF-fail on these substrates; its only gap is the heavier VRAM footprint on a small card (mitigated by the per-substrate pool free). |
| Calibration | report **ranking** + a **per-leaving-group** offset (shared slope, per-LG intercept; see `scripts/fit_united_model.py`) | — never a single global offset (F is a distinct, over-penalised cluster) |

Both solvent models cost ~1–2 min/substrate off one gas backbone, so the recommended
production recipe computes **both** and prefers SMD.

## 4. Procedure (recommended: gas once, then sweep)

```bash
conda activate gpuqc
export SNAR_QC_BACKEND=gpu4pyscf
export SNAR_QC_REQUIRE_GPU=1            # fail loudly instead of silently falling back to Psi4

# 1. Gas backbone once — writes the reusable gas_thermo.json + *_opt.xyz cache.
python scripts/run_poc.py \
  --substrates data/external/<your_input>.csv \
  --outdir     data/processed/<run>_gas \
  --n-procs 1 --mem 2                   # ignored by GPU backend; harmless

# 2. Sweep both solvent models off that one gas run (~1-2 min/substrate each).
python scripts/sweep_solvent.py --gas-run data/processed/<run>_gas \
  --solvent DMSO --solvent-model smd    --outdir data/processed/<run>_smd
python scripts/sweep_solvent.py --gas-run data/processed/<run>_gas \
  --solvent DMSO --solvent-model iefpcm --outdir data/processed/<run>_iefpcm

# Resume after an interruption: add --retry to any command, same --outdir.
```

Swap `--solvent DMSO` for any solvent in gpu4pyscf's `solvent_db`; each further model off the
same gas cache is ~0.5 h for an 18-substrate slice.

**CPU fallback** (no GPU; IEF-PCM only — Psi4 1.10.2 has no SMD):
```bash
conda activate snar-qc
python scripts/run_poc.py --substrates data/external/<your_input>.csv \
  --outdir data/processed/<run>_iefpcm --solvent DMSO --solvent-model iefpcm \
  --n-procs 8 --mem 12
```

## 5. Outputs, monitoring, resumability

- Per substrate: `data/processed/<run>/<tag>/result.json` (ΔG‡(qh), n_imag, solvent +
  solvent_model provenance, per-stage timing). `summary.json` rolls up the batch.
- Health: `nvidia-smi` (~1 GB per substrate, one job at a time). A *single* POC-sized job
  stays well under the 4 GB ceiling, but a long **batch in one process** accumulates VRAM
  because CuPy does not return its pool between substrates — the runner now frees the pool
  per substrate (`backend.free_gpu_memory`), so batches stay flat. Genuinely large arylators
  (CF₃-quinolines, bis-CF₃ arenes) can still exceed 4 GB at the analytic Hessian even solo.
- Resumable & failure-is-data: `--retry` re-runs only incomplete substrates; a failed
  substrate is recorded with `status: error`, never aborts the batch.

## 6. Validate against a reference (optional)

If your input carries an experimental `delta_g_kJmol` column, score ranking + magnitude:
```bash
python scripts/validate_poc.py --slice data/external/<your_input>.csv \
  --run data/processed/<run>_smd --outdir notes/assets/<run>_smd
```
Outputs Pearson/Spearman/MAE + a **per-leaving-group** breakdown and a scatter. Drop a
`README.md` in each new asset dir (what / when / method / stack / comparability).

## 7. Known failure modes (characterised)

- **GPU cumulative OOM** (`cudaErrorMemoryAllocation`): the dominant failure on the 4 GB card,
  and **retry-fixable** — it is *not* SCF non-convergence. A batch in one process accumulates
  CuPy's pool, so later/larger substrates OOM even when individually small (e.g. SMD on
  `Clc1cc(N2CCCC2)ncn1` OOM'd in the full-Lu_74 batch, then completed fine on retry). The
  runner now frees the pool per substrate; to recover sidecars left by an older batch, re-run
  with `--retry` (each substrate gets a fresh allocation). See
  `notes/2026-06-28_lu74_full_deltag_analysis.md`.
- **Genuinely memory- *and* convergence-limited substrates** (large arylators: CF₃-quinolines,
  bis-CF₃ arenes, fused dimethoxy-quinolines): the gas analytic Hessian can exceed 4 GB, and
  the TS optimisation can need many cycles. Use a larger GPU (more VRAM, looser TS budget) or
  the CPU/Psi4 path (no VRAM limit; IEF-PCM only, no SMD).
- **Nitrile-bearing TS** (e.g. `N#Cc1ccnc(Cl)c1`): CPU optking can fail to converge (linear-bend
  oscillation); the GPU geomeTRIC path clears it. Prefer GPU for nitrile substrates.
- **PCM cavity death** (Psi4/PCMSolver "S matrix not positive-definite"): a CPU-only failure;
  gpu4pyscf's PCM uses no PCMSolver and does not hit it.
- **Spurious TS** (a saddle below reactants, ΔG‡ implausibly low): gate on `n_imag_ts == 1`
  *and* a sane ΔE; cross-check engines if a single substrate is a parity outlier.

## 8. Fixed-amine reference cache

The model amine is invariant, so its gas reference is **cached** in `assets/amine_ref/`
keyed by `(amine, level-of-theory, backend)` (`snar_qc.poc.amine_cache`). `compute_barrier`
reuses it via `cached_amine_reference` and skips the redundant ~12 s opt+freq on every
substrate after the first; a genuine miss (new amine or new backend) computes it once and
stores it. The key is backend-specific on purpose — Psi4 and gpu4pyscf absolute energies
differ, so the two never mix.

You do **not** need to compute the seed: every finished gas run already contains the amine
reference, so copy it (no recompute), exactly like `extract_gas_cache.py`:

```bash
# Seed the gpu4pyscf reference from a finished GPU gas run (checks all substrates agree):
python scripts/extract_amine_ref.py --run data/processed/<run>_gas --backend gpu4pyscf
# Seed the Psi4 reference from a CPU gas run:
python scripts/extract_amine_ref.py --run data/processed/<cpu_run> --backend psi4
```

Override the location with `$SNAR_QC_AMINE_CACHE`; delete the asset to force a fresh
recompute on the next run. Because the amine enters every barrier identically, a cached
reference shifts all ΔG‡ by the same constant — ranking (the headline) is invariant even if
library versions drift slightly.
