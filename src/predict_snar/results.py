"""A module to hold the results from the calculations."""
rdkit_mols = {
    "substrate": None,
    "nucleophile": None,
    "product": None,
    "leaving_group": None,
    "reaction_complex": None,
    "agent": None,
}

mm_atoms = {
    "substrate": None,
    "nucleophile": None,
    "product": None,
    "leaving_group": None,
    "agent": None,
}

xtb_atoms = {
    "substrate": None,
    "nucleophile": None,
    "product": None,
    "leaving_group": None,
    "reaction_complex": None,
    "product_complex": None,
    "intermediate": None,
    "agent": None,
}

dft_atoms = {
    "substrate": None,
    "nucleophile": None,
    "product": None,
    "leaving_group": None,
    "intermediate": None,
    "ts": [],
    "agent": None,
}

descriptors = {
    "substrate": None,
    "nucleophile": None,
    "product": None,
    "leaving_group": None,
    "ts": [],
    "agent": None,
}

clustering_energies ={
    "substrate": None,
    "nucleophile": None,
    "ts": [],
}

smiles = {}
inchi = {}
inchi_key = {}
run_time = None
end_time = None