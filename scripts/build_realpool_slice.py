#!/usr/bin/env python
"""Build a compact, diverse QC substrate slice from an arylator pool.

Selects aryl-halide substrates from a pool TSV for a first-principles ΔG‡ run,
excluding any that appear in a reference set (matched on InChIKey connectivity
block). Selection funnel (all thresholds tunable):

1. valid SMILES and leaving group in {F, Cl, Br};
2. not in the reference set (InChIKey first-block / connectivity match);
3. small molecules only -- heavy-atom count <= the median of the eligible pool
   (the lowest two quartiles, "prefer small");
4. long *flexible* chains excluded -- many rotatable bonds or a long acyclic
   (non-ring) run make the conformer + geometry-optimisation problem harder.
   Rigid motifs (alkynes, nitriles, conjugation) are kept: they optimise cleanly;
5. both monocyclic-aromatic and fused-aromatic ring systems represented;
6. stratified diverse sample to a target size: leaving-group quota (default
   Cl 60 % / F 20 % / Br 20 %), both ring classes per LG, and within each
   (LG x ring-class) bucket distinct Bemis-Murcko scaffolds first.

Input columns required: ``smiles_canonical``, ``leaving_group``. Carried through
when present: an id column (default ``arylator_catcode``), ``arylator_id``,
``inchikey``, ``mean_yield_pct``. The reference file is matched on ``inchikey``.

Run in the snar-qc conda env (RDKit + pandas). Example:

    python scripts/build_realpool_slice.py \\
        --pool <pool.tsv> --reference <reference.csv> \\
        --out-dir data/external/realpool_qc --target 150 --seed 42
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # quiet RDKit parse warnings; we guard None ourselves

HALOGENS = {"F", "Cl", "Br"}


# --------------------------------------------------------------------------- #
# Per-molecule descriptors                                                    #
# --------------------------------------------------------------------------- #
def parse(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def ikey_block(inchikey: Optional[str]) -> str:
    """The connectivity (first) block of an InChIKey, '' if missing."""
    return (inchikey or "").split("-")[0].strip()


def longest_acyclic_chain(mol: Chem.Mol) -> int:
    """Longest run of connected non-ring heavy atoms (forest diameter, in atoms).

    Non-ring atoms induce a forest, so the longest simple path in each component is
    its diameter (two-BFS). Captures a floppy tail length regardless of branching.
    """
    adj: dict[int, list[int]] = {
        a.GetIdx(): [] for a in mol.GetAtoms() if not a.IsInRing()
    }
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

    def bfs(start: int) -> tuple[int, int, set[int]]:
        dist = {start: 1}
        far, far_d = start, 1
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if dist[v] > far_d:
                        far, far_d = v, dist[v]
                    queue.append(v)
        return far, far_d, set(dist)

    best, seen = 0, set()
    for node in adj:
        if node in seen:
            continue
        f1, _, comp = bfs(node)
        seen |= comp
        _, diameter, _ = bfs(f1)
        best = max(best, diameter)
    return best


def ring_class(mol: Chem.Mol) -> str:
    """'mono' (one aromatic ring), 'fused' (ortho-fused aromatic system),
    'other' (isolated multi-ring / biaryl), or 'none'."""
    arom_rings = [
        set(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
    ]
    if not arom_rings:
        return "none"
    for i in range(len(arom_rings)):
        for j in range(i + 1, len(arom_rings)):
            if len(arom_rings[i] & arom_rings[j]) >= 2:  # share a bond -> fused
                return "fused"
    return "mono" if len(arom_rings) == 1 else "other"


def buildable_lg_site(mol: Chem.Mol, element: str) -> bool:
    """True iff a halogen of ``element`` sits on an aromatic carbon *as
    ``build_reaction_complex`` will perceive it*.

    The builder embeds + MMFF-optimises, and MMFF atom typing demotes 'fragile'
    aromatic rings that RDKit's default model had aromatised (e.g. the coumarin
    pyranone in ``O=c1cc(Cl)c2ccccc2o1`` or the azolo-fused C-Cl in
    ``Clc1nccn2nnnc12``) -- so a naive default-RDKit check passes but the build
    then raises "No leaving halide ... on an aromatic carbon". ``AddHs`` +
    ``MMFFGetMoleculeProperties`` reproduces that exact demotion as an in-place
    side effect, with no 3D embed, giving a fast and faithful buildability gate.
    """
    mol = Chem.AddHs(mol)
    try:
        AllChem.MMFFGetMoleculeProperties(mol)  # side effect: MMFF aromaticity perception
    except Exception:  # noqa: BLE001 - unparametrised atom; fall through to the check
        pass
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == element:
            nbrs = atom.GetNeighbors()
            if len(nbrs) == 1 and nbrs[0].GetSymbol() == "C" and nbrs[0].GetIsAromatic():
                return True
    return False


def murcko(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:  # noqa: BLE001 - degenerate scaffolds shouldn't kill the build
        return ""
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


# --------------------------------------------------------------------------- #
# Diverse sampling                                                            #
# --------------------------------------------------------------------------- #
def diverse_pick(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Pick k rows maximising Murcko-scaffold spread (round-robin over scaffolds)."""
    if k >= len(rows):
        return list(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["murcko_scaffold"]].append(row)
    keys = list(groups)
    rng.shuffle(keys)
    queues = {}
    for key in keys:
        bucket = groups[key]
        rng.shuffle(bucket)
        queues[key] = deque(bucket)
    picked: list[dict] = []
    while len(picked) < k and any(queues[key] for key in keys):
        for key in keys:
            if queues[key]:
                picked.append(queues[key].popleft())
                if len(picked) >= k:
                    break
    return picked


def allocate_lg(
    rows: list[dict], quota: int, rng: random.Random, fused_frac: float = 0.35
) -> list[dict]:
    """Fill an LG quota: a fused floor first, then mono, then leftovers."""
    by_class: dict[str, list[dict]] = {"mono": [], "fused": [], "other": []}
    for row in rows:
        by_class.get(row["ring_class"], by_class["other"]).append(row)

    want_fused = min(len(by_class["fused"]), round(quota * fused_frac))
    picked = diverse_pick(by_class["fused"], want_fused, rng)
    chosen = {id(r) for r in picked}

    remaining = quota - len(picked)
    mono = diverse_pick(by_class["mono"], min(remaining, len(by_class["mono"])), rng)
    picked += mono
    chosen |= {id(r) for r in mono}

    remaining = quota - len(picked)
    if remaining > 0:
        leftovers = [r for r in rows if id(r) not in chosen]
        picked += diverse_pick(leftovers, min(remaining, len(leftovers)), rng)
    return picked


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="arylator pool TSV")
    ap.add_argument("--reference", required=True, help="reference set to exclude (CSV)")
    ap.add_argument("--out-dir", default="data/external/realpool_qc")
    ap.add_argument("--id-col", default="arylator_catcode")
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--cl-frac", type=float, default=0.60)
    ap.add_argument("--f-frac", type=float, default=0.20)
    ap.add_argument("--br-frac", type=float, default=0.20)
    ap.add_argument("--max-rotatable", type=int, default=4)
    ap.add_argument("--max-chain", type=int, default=4, help="longest acyclic run (atoms)")
    ap.add_argument("--fused-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    pool = pd.read_csv(args.pool, sep="\t" if str(args.pool).endswith(".tsv") else ",")
    ref = pd.read_csv(args.reference)
    ref_blocks = {ikey_block(k) for k in ref.get("inchikey", pd.Series([], dtype=str))}

    funnel = Counter()
    rows: list[dict] = []
    for rec in pool.to_dict("records"):
        funnel["pool"] += 1
        lg = str(rec.get("leaving_group", "")).strip()
        mol = parse(rec.get("smiles_canonical"))
        if mol is None or lg not in HALOGENS:
            funnel["drop_invalid_or_lg"] += 1
            continue
        if ikey_block(rec.get("inchikey")) in ref_blocks:
            funnel["drop_in_reference"] += 1
            continue
        rows.append(
            {
                "substrate_id": str(rec.get(args.id_col, "")).strip(),
                "arylator_id": rec.get("arylator_id", ""),
                "smiles_canonical": rec["smiles_canonical"],
                "leaving_group": lg,
                "lg_buildable": buildable_lg_site(mol, lg),
                "ring_class": ring_class(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "n_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True),
                "longest_acyclic_chain": longest_acyclic_chain(mol),
                "murcko_scaffold": murcko(mol),
                "inchikey": rec.get("inchikey", ""),
                "mean_yield_pct": rec.get("mean_yield_pct", ""),
            }
        )
    funnel["eligible_base"] = len(rows)

    # (3) small molecules: HAC <= median of the eligible base pool.
    median_hac = statistics.median(r["heavy_atoms"] for r in rows)
    rows = [r for r in rows if r["heavy_atoms"] <= median_hac]
    funnel["after_small_filter"] = len(rows)

    # (4) drop long flexible chains; (5) require an aromatic ring class.
    clean: list[dict] = []
    for r in rows:
        if r["n_rotatable"] > args.max_rotatable:
            funnel["drop_flexible_rotatable"] += 1
        elif r["longest_acyclic_chain"] > args.max_chain:
            funnel["drop_flexible_chain"] += 1
        elif r["ring_class"] not in ("mono", "fused", "other"):
            funnel["drop_no_aromatic"] += 1
        elif not r["lg_buildable"]:
            funnel["drop_lg_unbuildable"] += 1
        else:
            clean.append(r)
    funnel["clean_pool"] = len(clean)

    # (6) stratified diverse sample.
    fracs = {"Cl": args.cl_frac, "F": args.f_frac, "Br": args.br_frac}
    selected: list[dict] = []
    for lg, frac in fracs.items():
        quota = round(args.target * frac)
        lg_rows = [r for r in clean if r["leaving_group"] == lg]
        selected += allocate_lg(lg_rows, quota, rng, args.fused_frac)

    # ---- write ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_path = out_dir / "realpool_qc_slice.csv"
    cols = [
        "substrate_id", "arylator_id", "smiles_canonical", "leaving_group",
        "ring_class", "heavy_atoms", "n_rotatable", "longest_acyclic_chain",
        "murcko_scaffold", "inchikey", "mean_yield_pct",
    ]
    pd.DataFrame(selected, columns=cols).to_csv(slice_path, index=False)

    # ---- report ----
    lg_counts = Counter(r["leaving_group"] for r in selected)
    rc_counts = Counter(r["ring_class"] for r in selected)
    cross = Counter((r["leaving_group"], r["ring_class"]) for r in selected)
    hac = sorted(r["heavy_atoms"] for r in selected)
    n_scaffolds = len({r["murcko_scaffold"] for r in selected})
    lines = [
        "=== selection funnel ===",
        *(f"  {k:<26} {v}" for k, v in funnel.items()),
        f"  median heavy-atom count   {median_hac}",
        "",
        f"=== selected: {len(selected)} substrates -> {slice_path} ===",
        f"  leaving group : {dict(lg_counts)}  "
        f"(target Cl/F/Br {args.cl_frac:.0%}/{args.f_frac:.0%}/{args.br_frac:.0%})",
        f"  ring class    : {dict(rc_counts)}",
        f"  LG x class    : "
        + ", ".join(f"{lg}-{rc}:{n}" for (lg, rc), n in sorted(cross.items())),
        f"  heavy atoms   : min {hac[0]}, median {statistics.median(hac)}, max {hac[-1]}",
        f"  unique Murcko scaffolds: {n_scaffolds}",
    ]
    report = "\n".join(lines)
    print(report)
    (out_dir / "README.md").write_text(
        f"# real-pool QC slice (generated)\n\n"
        f"Generated by `scripts/build_realpool_slice.py` "
        f"(seed={args.seed}, target={args.target}).\n\n```\n{report}\n```\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
