"""Build an amine + aryl-halide S~N~Ar reaction complex from SMILES.

The POC needs a reaction complex (an ASE ``Atoms`` object carrying ``info["charge"]``
plus the 1-indexed central / nucleophile / leaving-group atom indices the
``Psi4TSScan`` engine consumes) for each Lu_74 aryl halide reacting with a neutral
**model amine** (methylamine by default).

predict-snar's own ``SmilesToXYZ`` builds such complexes, but it pulls in the full
config / agent-detection / conformer-search machinery (and is tuned for the Gaussian
path). For a handful of well-defined neutral-amine cases it is cleaner to build the
complex directly with RDKit, which is what this module does:

1. Embed the aryl halide in 3D (ETKDG + MMFF) and locate the leaving halide and its
   ipso (central) aromatic carbon.
2. Embed the amine; identify its nucleophilic nitrogen.
3. Dock the amine above the ipso carbon along the aromatic-ring normal at a chosen
   N...C(ipso) distance, lone pair pointing at the ring, leaving the relaxed scan to
   refine the approach.

The complex is **neutral closed shell** (neutral aryl halide + neutral amine); the
S~N~Ar addition step keeps the charge at zero until a later proton transfer that the
POC does not model. The returned indices are 1-indexed to match the engine's
``central_atom`` / ``nu_atom`` / ``lg_atom`` contract and ``Psi4BondOrders.get_bo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit.Chem import Mol

# Halogen elements that can act as the S~N~Ar leaving group, by symbol.
_HALOGENS = ("F", "Cl", "Br", "I")

# Leaving-group selector tokens for *ipso* nitro displacement: when ``leaving_group`` is
# one of these (case-insensitive), the builder locates an aromatic-carbon-bound nitro
# group instead of a halide and treats the nitro **nitrogen** as the leaving atom (the
# group departs as nitrite, NO2-). Recorded on the complex as the symbol ``"NO2"``.
_NITRO_TOKENS = frozenset({"NO2", "NITRO"})
_NITRO_SYMBOL = "NO2"

# Default neutral model nucleophile: methylamine (CH3-NH2). Its single nitrogen is the
# nucleophilic atom. Used for the whole POC so 4a and 4b share one nucleophile.
DEFAULT_AMINE_SMILES = "CN"

# Default docking distance N...C(ipso), Angstrom -- a pre-reaction complex separation
# from which the relaxed scan drives the forming bond inwards.
DEFAULT_APPROACH = 3.0


@dataclass
class ReactionComplex:
    """An amine + aryl-halide reaction complex ready for the ``Psi4TSScan`` engine.

    Attributes:
        atoms: ASE ``Atoms`` for the whole complex (aryl-halide atoms first, then the
            amine atoms), carrying the total charge in ``atoms.info["charge"]``.
        central_atom: 1-indexed ipso aromatic carbon (the S~N~Ar reaction centre).
        nu_atom: 1-indexed nucleophilic nitrogen of the amine.
        lg_atom: 1-indexed leaving atom -- the halide for halide departure, or the nitro
            **nitrogen** for ipso nitro displacement (nitrite leaves; its C-N bond is the
            one the scan elongates).
        aryl_halide_smiles: The source aryl-halide SMILES.
        amine_smiles: The source amine SMILES.
        leaving_group: Symbol of the leaving group: a halide ("F"/"Cl"/"Br"/"I") or
            ``"NO2"`` for ipso nitro displacement.
    """

    atoms: Atoms
    central_atom: int
    nu_atom: int
    lg_atom: int
    aryl_halide_smiles: str
    amine_smiles: str
    leaving_group: str


def _embed(mol: "Mol", seed: int) -> "Mol":
    """Add hydrogens, embed a 3D conformer (ETKDG), and MMFF-optimise.

    Args:
        mol: An RDKit molecule (parsed from SMILES).
        seed: Random seed for the deterministic ETKDG embedding.

    Returns:
        The molecule with explicit Hs and one optimised 3D conformer.

    Raises:
        ValueError: If a 3D conformer cannot be embedded.
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Fall back to random-coordinate embedding for awkward systems.
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed) != 0:
            raise ValueError("Could not embed a 3D conformer for the molecule.")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _find_leaving_halide(mol: "Mol", leaving_group: Optional[str]) -> tuple[int, int]:
    """Locate the leaving halide and its ipso aromatic carbon.

    The leaving halide is a halogen atom singly bonded to an aromatic carbon. If
    ``leaving_group`` is given, only halogens of that element are considered. When more
    than one candidate matches, the one whose ipso carbon carries the most
    electron-withdrawing environment (the largest number of ring heteroatoms / nearby
    nitro groups) is chosen as the most activated S~N~Ar position; ties break on the
    lowest atom index for determinism.

    Args:
        mol: An embedded aryl-halide molecule.
        leaving_group: Element symbol to restrict the search to, or ``None`` for any
            halogen.

    Returns:
        ``(halide_index, ipso_index)`` as 0-indexed atom indices.

    Raises:
        ValueError: If no halogen on an aromatic carbon is found.
    """
    wanted = leaving_group.capitalize() if leaving_group else None
    candidates: list[tuple[int, int, int]] = []  # (activation, halide_idx, ipso_idx)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in _HALOGENS:
            continue
        if wanted and symbol != wanted:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            continue
        ipso = neighbors[0]
        if not ipso.GetIsAromatic() or ipso.GetSymbol() != "C":
            continue
        candidates.append(
            (_activation_score(mol, ipso.GetIdx()), atom.GetIdx(), ipso.GetIdx())
        )

    if not candidates:
        raise ValueError(
            "No leaving halide (halogen on an aromatic carbon) found"
            + (f" matching element {wanted!r}" if wanted else "")
            + f" in {Chem.MolToSmiles(mol)!r}."
        )

    # Most activated ipso first; break ties on lowest halide index.
    candidates.sort(key=lambda c: (-c[0], c[1]))
    _, halide_idx, ipso_idx = candidates[0]
    return halide_idx, ipso_idx


def _find_leaving_nitro(mol: "Mol") -> tuple[int, int]:
    """Locate an *ipso* nitro group's nitrogen and its aromatic carbon.

    In ipso S~N~Ar a ring nitro group departs as nitrite (NO2-), so the leaving atom
    handed to the scan engine is the nitro **nitrogen** -- the single atom bonded to the
    ipso aromatic carbon, whose C-N bond the relaxed scan elongates as nitrite leaves
    (mirroring how the C-halide bond is elongated for halide departure). A nitro nitrogen
    is identified structurally: a nitrogen bonded to exactly two oxygens and one aromatic
    carbon (the ipso). When more than one ipso nitro group is present, the one on the most
    activated ring carbon is chosen (the same ``_activation_score`` proxy used for the
    halide search); ties break on the lowest nitrogen index for determinism.

    Args:
        mol: An embedded aryl-nitro molecule.

    Returns:
        ``(nitro_nitrogen_index, ipso_index)`` as 0-indexed atom indices.

    Raises:
        ValueError: If no aromatic-carbon-bound nitro group is found.
    """
    candidates: list[tuple[int, int, int]] = []  # (activation, nitro_n_idx, ipso_idx)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue
        neighbors = atom.GetNeighbors()
        oxygens = [nbr for nbr in neighbors if nbr.GetSymbol() == "O"]
        aromatic_carbons = [
            nbr for nbr in neighbors if nbr.GetSymbol() == "C" and nbr.GetIsAromatic()
        ]
        # A nitro group: N with two O's and a single aromatic-carbon (ipso) attachment.
        if len(oxygens) != 2 or len(aromatic_carbons) != 1:
            continue
        ipso = aromatic_carbons[0]
        candidates.append(
            (_activation_score(mol, ipso.GetIdx()), atom.GetIdx(), ipso.GetIdx())
        )

    if not candidates:
        raise ValueError(
            "No ipso nitro group (a nitro on an aromatic carbon) found in "
            f"{Chem.MolToSmiles(mol)!r}."
        )

    # Most activated ipso first; break ties on lowest nitrogen index.
    candidates.sort(key=lambda c: (-c[0], c[1]))
    _, nitro_idx, ipso_idx = candidates[0]
    return nitro_idx, ipso_idx


def _activation_score(mol: "Mol", ipso_idx: int) -> int:
    """Crude S~N~Ar activation score for an ipso carbon.

    Counts the activators that are **ortho or para to the ipso carbon** -- aromatic ring
    nitrogens and ring-borne nitro / N-oxide substituents -- since only ortho/para
    activators stabilise the anionic Meisenheimer intermediate (the developing negative
    charge resonates onto the ortho/para positions); a meta activator does not, and the
    ipso carbon's own substituent (the leaving group) is excluded. Position is read from
    the ring's topological distance: ortho = 1, meta = 2, para = 3 bonds around the ring.

    Used only to disambiguate which leaving group departs when a substrate carries several
    equivalent-element candidates -- e.g. on 1,2,4-trinitrobenzene it selects the C1 nitro
    (ortho *and* para to the other two nitros) over the C2/C4 nitros (one ortho/para
    activator each), matching the kinetically favoured ipso position.

    Args:
        mol: The aryl-halide molecule.
        ipso_idx: 0-indexed ipso carbon.

    Returns:
        An integer activation score (higher = more activated).
    """
    ring_info = mol.GetRingInfo()
    rings = [r for r in ring_info.AtomRings() if ipso_idx in r]
    if not rings:
        return 0
    ring = max(rings, key=len)
    # ``AtomRings`` orders atoms by ring connectivity, so the topological distance around
    # the cycle is the gap between positions (wrapping at the ring size).
    size = len(ring)
    position = {idx: i for i, idx in enumerate(ring)}
    ipso_pos = position[ipso_idx]
    score = 0
    for idx in ring:
        if idx == ipso_idx:
            continue
        gap = abs(position[idx] - ipso_pos)
        ring_dist = min(gap, size - gap)
        if ring_dist not in (1, 3):  # only ortho (1) and para (3) activate
            continue
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N" and atom.GetIsAromatic():
            score += 2  # ortho/para ring aza nitrogen
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring:
                continue
            if nbr.GetSymbol() == "N" and any(
                b.GetSymbol() == "O" for b in nbr.GetNeighbors()
            ):
                score += 1  # ortho/para nitro / N-oxide substituent
    return score


def _nucleophile_nitrogen(mol: "Mol") -> int:
    """Return the 0-indexed nucleophilic nitrogen of an amine.

    Picks the nitrogen with the most attached hydrogens (the least hindered N-H site),
    breaking ties on the lowest index.

    Args:
        mol: An embedded amine molecule (with explicit Hs).

    Returns:
        The 0-indexed nitrogen atom index.

    Raises:
        ValueError: If the molecule has no nitrogen.
    """
    nitrogens = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    if not nitrogens:
        raise ValueError(f"No nitrogen in amine {Chem.MolToSmiles(mol)!r}.")
    nitrogens.sort(key=lambda a: (-a.GetTotalNumHs(includeNeighbors=True), a.GetIdx()))
    return nitrogens[0].GetIdx()


def _ring_normal(positions: np.ndarray, ring: tuple[int, ...]) -> np.ndarray:
    """Unit normal of the aromatic ring plane (best-fit via SVD).

    Args:
        positions: ``(n_atoms, 3)`` coordinate array.
        ring: Atom indices of the ring.

    Returns:
        A unit normal vector to the ring's mean plane.
    """
    ring_coords = positions[list(ring)]
    centroid = ring_coords.mean(axis=0)
    _, _, vh = np.linalg.svd(ring_coords - centroid)
    normal = vh[2]
    return normal / np.linalg.norm(normal)


def _to_ase(mol: "Mol") -> tuple[list[str], np.ndarray]:
    """Extract element symbols and the conformer coordinates from an RDKit molecule."""
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.array(conf.GetPositions())
    return symbols, positions


def build_molecule(smiles: str, seed: int = 0x5217) -> Atoms:
    """Build a single optimised-ready ASE ``Atoms`` from SMILES (for a reference state).

    Used for the separated-reactants reference: the bare aryl halide and the bare amine
    are each built here, then optimised + frequency-analysed downstream to supply
    G(ArX) and G(amine). The total formal charge is read from the SMILES and stored in
    ``info["charge"]`` for the Psi4 calculator.

    Args:
        smiles: SMILES of the molecule.
        seed: Random seed for the deterministic 3D embedding.

    Returns:
        An ASE ``Atoms`` object with explicit Hs, a 3D conformer, and ``info["charge"]``.

    Raises:
        ValueError: If the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES {smiles!r}.")
    charge = Chem.GetFormalCharge(mol)
    mol = _embed(mol, seed=seed)
    symbols, positions = _to_ase(mol)
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info["charge"] = int(charge)
    return atoms


def build_reaction_complex(
    aryl_halide_smiles: str,
    amine_smiles: str = DEFAULT_AMINE_SMILES,
    leaving_group: Optional[str] = None,
    approach: float = DEFAULT_APPROACH,
    seed: int = 0x5217,
) -> ReactionComplex:
    """Build an amine + aryl-halide S~N~Ar reaction complex from SMILES.

    The aryl halide and amine are each embedded in 3D; the amine is then docked above
    the ipso carbon along the aromatic-ring normal with its nucleophilic nitrogen at
    ``approach`` Angstrom from the ipso carbon and its lone pair pointing at the ring.
    The relaxed scan downstream refines the approach.

    Args:
        aryl_halide_smiles: SMILES of the aryl halide (the S~N~Ar electrophile).
        amine_smiles: SMILES of the neutral amine nucleophile (default methylamine).
        leaving_group: Symbol of the leaving group. A halide element ("F"/"Cl"/"Br"/"I")
            selects halide departure (``None`` picks the only / most activated aromatic
            halide). ``"NO2"`` (or ``"nitro"``, case-insensitive) selects ipso nitro
            displacement: an aromatic-carbon-bound nitro group is located and its nitrogen
            becomes the leaving atom (the group departs as nitrite).
        approach: Initial N...C(ipso) distance in Angstrom.
        seed: Random seed for the deterministic 3D embeddings.

    Returns:
        A :class:`ReactionComplex` with the combined ASE ``Atoms`` (charge 0) and the
        1-indexed central / nu / lg atom indices.

    Raises:
        ValueError: If either SMILES cannot be parsed, no leaving group of the requested
            kind is found (halide or ipso nitro), or the amine has no nitrogen.
    """
    aryl = Chem.MolFromSmiles(aryl_halide_smiles)
    if aryl is None:
        raise ValueError(f"Could not parse aryl-halide SMILES {aryl_halide_smiles!r}.")
    amine = Chem.MolFromSmiles(amine_smiles)
    if amine is None:
        raise ValueError(f"Could not parse amine SMILES {amine_smiles!r}.")

    aryl = _embed(aryl, seed=seed)
    amine = _embed(amine, seed=seed)

    # Halide departure (default) vs ipso nitro displacement (leaving_group "NO2"/"nitro").
    if leaving_group and leaving_group.strip().upper() in _NITRO_TOKENS:
        lg_idx, ipso_idx = _find_leaving_nitro(aryl)
        lg_symbol = _NITRO_SYMBOL
    else:
        lg_idx, ipso_idx = _find_leaving_halide(aryl, leaving_group)
        lg_symbol = aryl.GetAtomWithIdx(lg_idx).GetSymbol()
    nu_idx = _nucleophile_nitrogen(amine)

    aryl_symbols, aryl_pos = _to_ase(aryl)
    amine_symbols, amine_pos = _to_ase(amine)

    # Ring normal at the ipso carbon: the amine approaches out of the ring plane.
    ring_info = aryl.GetRingInfo()
    ring = max((r for r in ring_info.AtomRings() if ipso_idx in r), key=len)
    normal = _ring_normal(aryl_pos, ring)
    ipso_pos = aryl_pos[ipso_idx]

    # Orient the amine so its nitrogen's lone pair points back at the ring: place the
    # centroid of the nitrogen's substituents on the far side of N (along +normal),
    # which leaves the lone-pair lobe pointing toward the ipso carbon (-normal).
    nu_pos = amine_pos[nu_idx]
    nbrs = [a.GetIdx() for a in amine.GetAtomWithIdx(nu_idx).GetNeighbors()]
    subst_centroid = amine_pos[nbrs].mean(axis=0) if nbrs else nu_pos + normal
    lp_dir = nu_pos - subst_centroid  # points away from substituents ~ lone-pair lobe
    norm = np.linalg.norm(lp_dir)
    lp_dir = lp_dir / norm if norm > 1e-6 else normal

    # Rotation aligning the lone-pair direction with -normal (toward the ring).
    rot = _rotation_between(lp_dir, -normal)
    amine_pos = (amine_pos - nu_pos) @ rot.T  # rotate about the nitrogen
    # Translate so the nitrogen sits at approach Angstrom above the ipso carbon.
    target_n = ipso_pos + approach * normal
    amine_pos = amine_pos + target_n

    symbols = aryl_symbols + amine_symbols
    positions = np.vstack([aryl_pos, amine_pos])
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info["charge"] = 0

    n_aryl = len(aryl_symbols)
    return ReactionComplex(
        atoms=atoms,
        central_atom=ipso_idx + 1,
        nu_atom=n_aryl + nu_idx + 1,
        lg_atom=lg_idx + 1,
        aryl_halide_smiles=aryl_halide_smiles,
        amine_smiles=amine_smiles,
        leaving_group=lg_symbol,
    )


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix turning unit vector ``a`` onto unit vector ``b`` (Rodrigues).

    Args:
        a: A unit vector.
        b: A unit vector.

    Returns:
        A ``(3, 3)`` rotation matrix ``R`` with ``R @ a ~ b``.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        # Parallel or antiparallel: identity, or 180 deg about any orthogonal axis.
        if c > 0:
            return np.eye(3)
        # Pick an axis orthogonal to a for the 180 deg flip.
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
