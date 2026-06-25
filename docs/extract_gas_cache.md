# Tool — `extract_gas_cache.py`

Backfill the reusable **gas cache** (`gas_thermo.json` + `ts_opt.xyz` / `arx_opt.xyz` /
`amine_opt.xyz`) for runs produced *before* the pipeline persisted it — **without
recomputing any quantum chemistry**.

- Script: [`../scripts/extract_gas_cache.py`](../scripts/extract_gas_cache.py)
- Consumes: a finished/active run directory **or** the per-task archive zips.
- Produces: the same cache `scripts/sweep_solvent.py` reads, so a backfilled run becomes
  solvent-sweepable like a native one.

## Why this tool exists

The ΔG‡ pipeline gained a *gas cache* so a finished gas backbone (xTB scan + gas
TS/ArX/amine opt+freq) can be re-evaluated in another solvent or continuum model for the
price of **three single points** instead of a full ~30+ min rebuild — see
[`2026-06-25_gpu_dmso_campaign_runbook.md`](../notes/2026-06-25_gpu_dmso_campaign_runbook.md)
and `snar_qc.poc.barrier._write_gas_cache` / `solvent_sweep`. The cache is two artefacts
per substrate:

- the gas-optimised geometries `*_opt.xyz` (extended XYZ, charge in the comment line), and
- `gas_thermo.json` (per-species gas thermochemistry + geometry refs).

The **real-pool CPU/Psi4 campaign** ([`runbook_realpool_qc.md`](runbook_realpool_qc.md))
was already running when that persistence landed. Its per-task archives therefore carry
the *expensive* result — the gas TS/ArX/amine opt+freq — but not in the cache's tidy form.

**Nothing was lost.** The optimised geometry is inside every Psi4 `ts.out` / `arx.out` /
`amine.out` (each optimisation step prints a `Geometry (in Angstrom)` block; the last is
the relaxed geometry the frequency job used), and the gas thermochemistry is in the same
file's `==> Thermochemistry Energy Analysis <==` block. This tool reconstructs the cache
from those archived `*.out` files plus `result.json`. So **stopping and recomputing the
campaign is unnecessary** — finished *and* still-finishing CPU runs become sweepable with
no extra compute.

## How it reconstructs each species (all gas phase)

| Quantity | Source |
|---|---|
| geometry, charge | last `Geometry (in Angstrom), charge = q` block of the `*.out` |
| `electronic_energy`, `enthalpy`, `gibbs`, `zpve` | Psi4 thermochemistry block (`Total E_e` / `Total H` / `Total G` / `Correction ZPVE to E_e`) |
| `gibbs_qh` | from `result.json`, undoing the solvent single-point shift: `G_qh(gas) = E_e(gas) + (G_qh(solv) − E(solv))` |

The `gibbs_qh` step deliberately reuses the run's **own** quasi-harmonic Gibbs (Grimme
qRRHO + any soft-imaginary folding applied at runtime) rather than re-deriving it from
scraped frequencies, so it is *exact*. For a gas run the shift is zero and the identity
still holds.

### Faithfulness checks (per substrate, before anything is written)

1. **Atom balance** — `n(ts) == n(arx) + n(amine)` (the TS is the aryl halide plus the amine).
2. **Thermochemical ordering** — `H > E_e`, `G < H`, `ZPVE > 0`, `|G_qh − G|` small.
3. **Round-trip** — re-apply the solvent shift to the reconstructed gas `gibbs_qh` and
   reproduce `result.json`'s `delta_g_qh_kcal` to **< 0.01 kcal/mol**. This proves the
   parser picked the right file and fields. (Validated: all 39 completed real-pool DMSO
   substrates round-trip to 5 dp.)

A substrate that fails any check is reported and skipped; it never sinks the batch.

## Usage

Run inside a QC conda env. The tool needs only **ASE** + `predict_snar.data` — it does
**not** import Psi4, so it runs in either `snar-qc` or `gpuqc`.

```bash
# In place over a finished/active run directory (per-substrate subdirs):
python scripts/extract_gas_cache.py --run data/processed/realpool_dmso

# From the per-task archive zips (the runbook --archive-dir / a datadump mirror):
python scripts/extract_gas_cache.py \
    --archive ../datadump/snar-qc-runs/realpool_dmso \
    --outdir  data/processed/realpool_dmso_gas

# Preview without writing; or re-do an existing cache:
python scripts/extract_gas_cache.py --run data/processed/realpool_dmso --dry-run
python scripts/extract_gas_cache.py --run data/processed/realpool_dmso --force
```

Flags: `--only tag1,tag2` filters by tag / `lu_id`; `--force` overwrites an existing
`gas_thermo.json`; `--dry-run` runs every check but writes nothing. It is idempotent —
re-running skips substrates that already have a cache.

### Then sweep a solvent off the reconstructed cache

```bash
python scripts/sweep_solvent.py --gas-run data/processed/realpool_dmso_gas \
    --solvent DMSO --solvent-model iefpcm --outdir data/processed/realpool_dmso_sweep
```

> **Engine caveat.** `sweep_solvent.py` runs its solvent single point on the **active
> backend**. The real-pool geometries were optimised with **Psi4**, so sweep them in the
> `snar-qc` (Psi4) env to stay method-consistent with how they were made. Sweeping
> Psi4 geometries in `gpuqc` is a *cross-engine* single point (gpu4pyscf SCF on a Psi4
> geometry) — a method change that needs its own comparability check before the deltas
> are trusted. This is the only reason the backfill is not "free for any solvent" the way
> a native gas run is.

## Limitations

- Only `status == "completed"` substrates (all three species + a barrier) are eligible.
- Energies carry Psi4's printed precision (8 dp, ~1e-5 kcal/mol — far below method error
  and the solvent-shift magnitude).
- If a species had a **soft imaginary mode folded** into its thermochemistry at runtime
  (`n_imag_*_soft > 0`), the parsed gas `gibbs` / `enthalpy` / `zpve` are Psi4's raw
  (unfolded) values and so differ slightly from the live run; `gibbs_qh` (the headline
  number) is still exact via the shift identity. The tool **warns** when this applies. It
  does *not* apply to the real-pool DMSO cohort, where every soft count is 0.

## Provenance

A backfilled `gas_thermo.json` carries a `_provenance` block (`reconstructed_by`,
`source: "psi4 *.out + result.json (no recomputation)"`, and the round-trip ΔG‡). The
sweep ignores it; it exists so a backfilled cache is never mistaken for a natively-emitted
one.

## See also

- [`runbook_realpool_qc.md`](runbook_realpool_qc.md) — the CPU campaign this backfills.
- [`2026-06-25_gpu_dmso_campaign_runbook.md`](../notes/2026-06-25_gpu_dmso_campaign_runbook.md)
  — the gas-once-then-sweep workflow the cache was built for.
- `snar_qc.poc.barrier` (`_write_gas_cache`, `_GEOMETRY_FILES`, `solvent_sweep`) — the
  cache contract this tool mirrors. The schema constants are duplicated in the script (to
  keep it off the Psi4 import chain); keep them in sync if that contract changes.
