# Vendored: predict-SNAr

This directory is a **vendored copy** of the `predict_snar` Python package from the
*predict-SNAr* code release. It was taken verbatim, then **minimally patched for
Windows portability** — two files, documented under [Local modifications](#local-modifications)
below. Everything else is byte-for-byte upstream, and all snar-qc functionality lives
in `snar_qc`. The MIT licence permits modification (the notice in `LICENSE` is preserved).

## Provenance

| Field | Value |
|---|---|
| Package | `predict_snar` (predict-SNAr) |
| Authors | Kjell Jorner et al. |
| Source | Paper_00130 ESI archive `d0sc04896h2.zip`, `ESI data/Code/predict-snar/` |
| Reference | Jorner, K. et al. *Chem. Sci.* **2021**, 12, 1163–1175. DOI: 10.1039/D0SC04896H |
| Licence | MIT — © 2020 Kjell Jorner (see `LICENSE` in this directory) |
| Vendored | 2026-06-21, verbatim; then 2 files patched 2026-06-22 (see Local modifications) |

## What was vendored

- The full `predict_snar/` package (all modules: `calculators`, `jobs`, `smiles`,
  `helpers`, `parsers`, `descriptors`, `data`, `config`, `results`, `legacy_code`,
  `__main__`, and the `script_*` entry points).
- The package data directory at `../data/` (sibling of this package, i.e.
  `src/data/`) — the vendored modules resolve it via `Path(__file__).parent / "../data"`,
  so the relative layout is preserved. Contains the def2 basis-set / ECP pickles,
  the solvent pickles, and the acid/base agent pickles.
- The MIT `LICENSE` (placed inside this package directory).

## What was omitted, and why

- **`data/solvent/opsin-2.4.0-jar-with-dependencies.jar` (~7 MB)** — omitted. It is
  referenced only inside `SolventPicker` (`data.py`), and only at *call* time of the
  trivial-name → InChIKey solvent lookup (`inchikey_from_name`), not at import time and
  not by anything on the `predict_snar.calculators` import path that Stage 1 uses. Its
  absence therefore cannot raise an import error. If a later stage needs trivial-name
  solvent parsing, copy the jar back from the ESI archive into `src/data/solvent/`.

## Local modifications

The package was vendored verbatim on 2026-06-21. On 2026-06-22 two files were
patched for Windows portability — the upstream code uses POSIX-only process APIs
that fail on Windows before any calculation runs. The changes are
behaviour-preserving on POSIX. Each edited site carries an inline
`# snar-qc: Windows portability` marker so it stands out against upstream.

| File | Change | Why |
|---|---|---|
| `calculators.py` | `subprocess.Popen(..., preexec_fn=os.setsid)` → `start_new_session=True` (3 sites: `G16Calculator`, `XTBCalculator`, `CRESTCalculator`) | `os.setsid` is POSIX-only; `preexec_fn` raised `AttributeError` on Windows. `start_new_session=True` calls `setsid()` on POSIX (identical) and is a no-op on Windows. |
| `helpers.py` | `calculation_monitor` dissociation-abort: guard `os.killpg(os.getpgid(...))` with a `hasattr(os, "killpg")` check, falling back to `process.terminate()` | `os.killpg`/`os.getpgid` are POSIX-only; Windows has no process groups here. |

To re-vendor cleanly from upstream, re-apply these two patches (or re-run them
from the snar-qc git history: commit `ab17da7`).

## Why a top-level package (not nested under `snar_qc`)

The vendored modules use absolute imports (`from predict_snar import config`,
`from predict_snar.data import ...`). Keeping the package importable as top-level
`predict_snar` means those imports work unchanged — a prerequisite for vendoring
the package essentially as-is.
