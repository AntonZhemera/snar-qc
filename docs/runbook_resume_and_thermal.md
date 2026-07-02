# Runbook — resume an interrupted ΔG‡ run, and run under a thermal cap

**Status:** living reference (refresh in place; not dated). **Applies to:** the ΔG‡ pipeline
driven by `scripts/run_poc.py` (+ `scripts/sweep_solvent.py` for the solvent stage).

A long QC batch can die for many reasons — an OOM, a killed session, a power loss, or a
hardware thermal cutoff on a machine with no working thermal daemon. This runbook covers the
two tools that make that survivable: **`run_poc.py --resume`** (skip stages already finished)
and **`scripts/thermal_run.py`** (duty-cycle the job under a temperature cap). See
`docs/sop_snar_deltag.md` for the pipeline itself.

---

## 1. What resumes automatically

The runner is resumable at three granularities:

| Granularity | Resumable | Mechanism |
|---|---|---|
| **Per substrate** | yes | `run_poc.py` writes a `result.json` sidecar per molecule; on rerun, `worker.should_skip` skips any with a terminal status (`completed` / `no_peak` / `ts_not_saddle` / `error`). `--retry` re-runs the non-`completed` ones; `--force` redoes all. |
| **Per solvent / model** | yes | a finished gas backbone writes `gas_thermo.json` + `*_opt.xyz`; `sweep_solvent.py` recomputes ΔG‡ in another solvent from 3 single points (no gas recompute). |
| **Within a substrate (`--resume`)** | yes | `compute_barrier` checkpoints each expensive stage to the workdir's `progress.json`; with `--resume` it reloads a finished stage instead of recomputing. |

### `--resume` stage checkpoints

As each stage completes, the per-substrate workdir accumulates a `progress.json` keyed by
`(smiles, coordinate)`:

| Stage | Persisted | Reloaded on `--resume` instead of |
|---|---|---|
| scan + DFT single points | `ts_guess.xyz` + peak index + scan energies | the xTB relaxed scan + the DFT single points |
| TS opt+freq | `ts_opt.xyz` + gas thermochemistry + imaginary-mode counts | the TS optimisation + Hessian (the expensive stage) |
| ArX opt+freq | `arx_opt.xyz` + gas thermochemistry + n_imag | the reference opt+freq |
| model amine | (already cached across runs by `amine_cache`) | — |

`--resume` is safe to pass on **every** run: a fresh workdir has no checkpoint and runs from
scratch, then checkpoints as it goes. A stale/mismatched-key or unreadable `progress.json`
degrades to a clean recompute (never raises). `--force` overrides `--resume`. The complete
`gas_thermo.json` is still emitted for `sweep_solvent.py`, and the cheap solvent single-point
corrections always re-run.

The only thing **not** checkpointed is progress *inside* a single TS optimisation: a death
mid-`ts_opt_freq` reruns that optimisation from the scan peak (the scan + single points before
it are kept).

---

## 2. Resume procedure

**Triage** the dead run dir `data/processed/<run>/`:

```bash
# Per-substrate status (these are skipped on rerun unless --retry/--force):
for d in data/processed/<run>/*/; do
  python - "$d/result.json" <<'PY' 2>/dev/null || echo "$d: NO-SIDECAR (will rerun)"
import json, sys; print(sys.argv[1], json.load(open(sys.argv[1]))["status"])
PY
done
# Which substrates carry a finished gas backbone (solvent-resumable)?
ls data/processed/<run>/*/gas_thermo.json 2>/dev/null
# Which carry partial stage checkpoints (--resume will use these)?
ls data/processed/<run>/*/progress.json 2>/dev/null
```

**Resume the batch** — rerun the *same* command with `--resume`:

```bash
python scripts/run_poc.py --substrates <csv> --amine <SMILES> --resume \
  --n-procs <N> --mem <GB> --outdir data/processed/<run>
```

Completed substrates are skipped; the interrupted one picks up from its last checkpointed
stage; unstarted ones run fresh. Add `--retry` to also re-run terminal failures (they then
resume from their last good stage too).

**Add / change the solvent** on a finished gas run (minutes, not a full recompute):

```bash
python scripts/sweep_solvent.py --gas-run data/processed/<gas_run> \
  --solvent <name> --solvent-model <iefpcm|smd> --outdir data/processed/<solv_run>
```

---

## 3. Running under a thermal cap (`scripts/thermal_run.py`)

On a machine with weak cooling or no thermal daemon, a sustained all-core QC load can drive
the CPU into a firmware thermal cutoff (a hard power-off) — and a parallel run of several QC
processes makes it far worse. Two rules:

- **Run one heavy job at a time.** A single `run_poc.py` already processes substrates serially
  (one at a time, `--n-procs` threads each); do not launch several in parallel.
- **Run it under `thermal_run.py`.** It launches the job in its own process group and
  **pauses** it (SIGSTOP) when the CPU (`k10temp`) or NVIDIA GPU (`nvidia-smi`) reaches a high
  cap, **resuming** (SIGCONT) after it cools below a low cap, with hysteresis. This
  duty-cycles a sustained load so the package temperature cannot run away.

> **POSIX-only at this time.** `thermal_run.py` uses POSIX process-group signals
> (SIGSTOP/SIGCONT) and reads CPU temperature from `/sys/class/hwmon` (`k10temp`). It is
> **Linux-only and has not been tested on Windows** — run it only on a POSIX host. On other
> platforms, run the pipeline without it (or port the throttle to a platform-native
> mechanism first). The temperature sources are also vendor-specific (AMD `k10temp`, NVIDIA
> `nvidia-smi`); with no readable sensor it logs a warning and runs unthrottled.

**Launch mode** — start and guard a command:

```bash
python scripts/thermal_run.py --cpu-high 85 --cpu-low 74 --poll 1 --heartbeat 60 \
  --log run.thermal.log -- \
  python scripts/run_poc.py --substrates <csv> --amine <SMILES> --resume \
    --n-procs <N> --mem <GB> --outdir data/processed/<run>
```

**Attach mode** — guard (or re-cap) an already-running job without restarting it, e.g. to
change the caps live:

```bash
# find the run's pid, then:
python scripts/thermal_run.py --attach-pid <PID> --cpu-high 85 --cpu-low 74 \
  --poll 1 --heartbeat 60 --log run.thermal.log
```

Pick caps with headroom for overshoot: a 12-thread SCF ramping from idle can spike several °C
past the cap before the next poll, so set the high cap a few °C below the temperature you must
not exceed (lower `--poll` shrinks the overshoot). Pair `thermal_run.py` with `--resume`: if
the cap is ever insufficient and the box still dies, the rerun resumes from the last
checkpointed stage.

---

## See also
- `docs/sop_snar_deltag.md` — the standard ΔG‡ procedure.
- `docs/runbook_solvated_validation.md`, `docs/runbook_realpool_qc.md` — campaign runbooks.
