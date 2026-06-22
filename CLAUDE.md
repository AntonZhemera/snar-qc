# CLAUDE.md — Project context for AI coding agents

## Project overview

**snar_qc** is quantum-chemistry tooling for **nucleophilic aromatic substitution
(S~N~Ar)** reactivity. It computes reactivity-relevant quantities from first principles:

- **Activation free energies (ΔG‡)** from explicit ground and transition states (rather
  than from an empirical descriptor regression), and
- **Ground-state electronic descriptors** (e.g. −LUMO / EA and electrostatic-potential
  features) at the reactive site.

It is a **producer**: it emits computed barriers and descriptor tables that downstream
reactivity-analysis tools consume through a data contract (tables written to a shared data
directory). The package does not itself fit or apply the reactivity model.

> **Status:** greenfield / early. The first work is the ΔG‡ proof of concept (see
> [`docs/scientific_context.md`](docs/scientific_context.md)). Ground-state descriptor
> tooling is consolidated here over time, so the project's S~N~Ar quantum chemistry lives
> behind one boundary.

## Scientific framing

See [`docs/scientific_context.md`](docs/scientific_context.md) for the motivation (why
first-principles ΔG‡ labels are needed where the empirical model is undefined), the
strategy, and the literature anchors.

## Repository layout

```
snar-qc/
├── src/
│   ├── snar_qc/            # core package (TS building, QC wrappers, ΔG‡, descriptors, surrogate)
│   └── predict_snar/       # vendored MIT scaffolding (see src/predict_snar/VENDORED.md)
├── scripts/                # pipeline / batch scripts
├── tests/                  # pytest suite (TDD)
├── data/
│   ├── raw/                # raw inputs (gitignored payloads)
│   ├── processed/          # pipeline outputs (gitignored payloads)
│   └── external/           # external reference sets (e.g. published validation data)
├── assets/                 # reaction-template (SMIRKS) catalogues, nucleophile definitions
├── notes/                  # dated interpretive notes, findings
├── plans/                  # work plans (active); plans/archive/ for completed
├── docs/                   # stable reference (scientific_context, runbooks)
├── environment.yml         # conda env for the QC stack (Psi4 / xTB / autodE + RDKit)
├── pyproject.toml          # package metadata
├── VERSION                 # semver
└── LICENSE                 # Apache-2.0
```

## Development standards

### Environment

- **Python 3.10** via conda / Mamba (conda-forge). The QC stack — **Psi4**, **xTB**,
  **autodE** — is not reliably pip-installable, so it lives in the conda env
  (`environment.yml`, env name `snar-qc`). A lightweight pip venv (`pip install -e .[dev]`)
  is enough for package / analysis / tests; switch to the conda env to run calculations.
- Editable install once the conda env is active: `pip install -e . --no-deps`.

### DDD / TDD

- **Documentation-driven:** write the docstring / spec contract before the code; update
  `docs/` before changing behaviour.
- **Test-driven:** documentation → tests → implementation. Published ΔG‡ values make
  natural fixtures.
- **Routine work** (bug fixes, small refactors, doc touch-ups): execute directly.
  **Non-routine work** (new pipeline stages, method changes, new external data, schema
  changes): write a short plan under `plans/` first.

### Code style

- Formatter **Black**; linter **Ruff**.
- Type hints on public APIs.
- RDKit `Mol` objects can be `None` — always guard.

### Commit conventions

Strict [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Use for |
|:--|:--|
| `feat` | New functionality |
| `fix` | Bug fixes |
| `refactor` | Restructuring without behaviour change |
| `docs` | Documentation |
| `test` | Tests |
| `chore` | Build, environment, CI, scaffold |

Scopes: `qc`, `ts`, `template`, `descriptor`, `surrogate`, `model`, `data`, `psi4`,
`xtb`, `autode`.

Rules: imperative mood, lowercase, no trailing period, ≤ ~72 chars.

## Vendored third-party code

`src/predict_snar/` is a **vendored MIT package** (© 2020 Kjell Jorner, predict-SNAr) —
upstream code, not ours. Default: **do not edit it**; build on top in `src/snar_qc/`.

If you genuinely must touch it (e.g. a portability fix that can't live in `snar_qc`),
treat all three of these as a single non-optional unit — never one without the others:

1. **Mark the site inline** — add a `# snar-qc: <reason>` comment at each edited line, so
   local patches stand out against upstream and a future re-vendor diff is obvious.
2. **Record it** in [`src/predict_snar/VENDORED.md`](src/predict_snar/VENDORED.md) under
   *Local modifications* (file, exact change, why, behaviour impact).
3. **Never leave a false provenance claim standing** — any "unmodified / verbatim /
   byte-for-byte" wording in `NOTICE` or `VENDORED.md` must be corrected in the same change.

MIT permits the modification; the obligations are to preserve `src/predict_snar/LICENSE`
and to not misrepresent what was changed. Precedent: `snar-qc` `ab17da7` + `1ee7349`.

## Plans / notes / docs

- **Plans** (`plans/`): a single dated file `YYYY-MM-DD_Short_Description.md`, or a
  subfolder for multi-stage work. Completed plans move to `plans/archive/`.
- **Notes** (`notes/`): dated `YYYY-MM-DD_short_topic.md` — findings and hypotheses
  (labelled `H1.`, `H2.`…). Scientific content only.
- **Docs** (`docs/`): stable reference, refreshed in place. Reserve CLAUDE.md edits for
  workflow / convention changes.

## Data hygiene

`data/` payloads are gitignored by default. Commit only small, shareable, curated tables —
explicitly, once a consumer artefact stabilises. Never commit large or non-shareable data.
