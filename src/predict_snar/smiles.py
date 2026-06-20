import itertools
import logging
import os
from pathlib import Path
import pickle


from ase import Atoms
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS, GetMolFrags
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Descriptors import NumRotatableBonds
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator

from predict_snar.calculators import XTBCalculator
from predict_snar.helpers import cd
from predict_snar.parsers import XTBParser

logger = logging.getLogger("predict_snar")

class SmilesToXYZ:
    """Converts a reaction smiles to a set of optimized structures and provides
    methods for writing the coordinates, smiles and config files.

    Args:
        reaction_smiles (str): Smiles of the reaction
        n_confs (int): Number of configuarations to try. If not set, a
            reasonable number is set based on the number of rotatable bonds.
        detect_agent_atom (bool): Attempt to detect agent atom automatically
            based on localized molecular orbitals if it is not given explicitly
            in the reaction SMILES.

    Attributes:
        reaction_smiles_orig (str): Original smiles of the reaction
        reaction_smiles (str): Reaction smiles for parsed reaction.
        substrate (object): Molecule object for substrate
        nu (object): Molecule object for nucleophile
        product (object): Molecule object for the product
        lg (object): Molecule object for the leaving group
        agent (object): Molecule object for agent.
        reaction_complex (object): Molecule object for reaction complex
        reaction_complex_pre (object): Molecule object for reaction complex
            prior to geometry optimization.
        reaction_complex_agent (object): Molecule object for reaction complex
            with agent present.
        reaction_complex_agent_pre (object): Molecule object for reaction complex
            with agent present before geometry optimization.            
        central_atom (object): Atom object for central atom
        central_atom_prod (object): Atom object for central atom in the product.
        nu_atom (object): Atom object for nucleophilic atom
        lg_atom (object): Atom object for leaving atom
        lg_atom_orig (object): Atom object for leaving atom in the leaving group.
        added_atom (object): Atom object for nucleophilic atom in the product.
        agent_atom (object): Atom object for active atom in agent.
        charges (dict): Molecular charges for species
        core_atoms (list): Atom indices for core
        nu_atoms (list): Atom indices for nucleophile
        lg_atoms (list): Atom indices for leaving_group
        agent_atoms (list): Atom indices in the agent
        ring_atoms (list): Atom indices for aromatic ring atoms.
        ortho_carbons (list): Atom indices for carbons ortho to leaving group.
        smiles (dict): Dictionary of the smiles
        atom_nos (dict): Dictionary of the atom numbers
        proton_transfer (bool): Whether proton has been transferred in the
            reaction.
        ortho_nitro (bool): Whether there are nitro groups ortho to the leaving
            group.
        has_agent (bool): Whether an agent is present.
        sp3 (bool): Whether reaction involes sp3 carbon for SN2 reactivity.
        intramolecular (bool): Whether reaction is intramolecular.
        reaction_smiles_parser (object): Reaction smiles parser used to parse
            the reaction smiles.
        agent_detector (object): Agent detector used to parse the agents.
        clustering (dict): Which species to cluster with explicit solvation.
        ff_type (str): Type of force field used during the optimization 'mmff'
            or 'uff'
        inchi (dict): Inchi for the molecules
        inchi_keys (dict): Inchi keys for the molecules.
        nu_sp3_neighgbors (list): Atom indices for sp3 neighbors of nucleophilic
            atom in reaction complex.
        nu_sp3_neighbors_orig (list): Atom indices for sp3 neighbors of
            nucleophilic atom in nucleophile 
    """
    def __init__(self, reaction_smiles, n_confs=None, detect_agent_atom=False):
        # Set up attributes.
        self.agent_atom = None

        self.smiles = {}
        self.atom_nos = {}
        self.charge = {}

        self.agent_atoms = []
        self.ring_atoms = []

        self.reaction_smiles_orig = reaction_smiles

        self.charge = {}
        self.sp3 = None
        self.azide = None
        self.azide_angle = None
        self.azide_dihedral = None
        self.has_agent = False
        self.ortho_nitro = False
        self.proton_transfer = False

        rsmp = ReactionSmilesProcessor(reaction_smiles)
        if rsmp.reaction_type == "intramolecular":
            self.intramolecular = True
        else:
            self.intramolecular = False
        self.substrate = rsmp.substrate
        self.product = rsmp.products[0]
        self.nu_atom = rsmp.nu_atom[0]
        self.nu_atoms = rsmp.nu_atoms
        if not self.intramolecular:
            self.nu = rsmp.nu
        else:
            self.nu = None
        self.lg = rsmp.lgs[0]
        self.lg_atom_orig = rsmp.lg_atom_orig[0]
        self.central_atom = rsmp.central_atom[0]
        self.central_atom_prod = rsmp.central_atom_prod[0]
        self.added_atom = rsmp.added_atom[0]
        self.lg_atom = rsmp.lg_atom[0]
        self.lg_atoms = rsmp.lg_atoms[0]
        self.core_atoms = rsmp.core_atoms[0]

        self.reaction_complex = None
        self.reaction_complex_pre = None
        self.reaction_complex_agent = None
        self.reaction_complex_agent_pre = None

        # Run agents through AgentDetector to see if they are relevant for the
        # reaction.
        ad = AgentDetector()
        for mol in rsmp.agents:
            ad.process_mol(mol)
        
        # Count number of hydrogens on nu atom
        if not self.intramolecular:
            num_nu_h = self.nu.GetAtomWithIdx(self.nu_atom).GetTotalNumHs()
        else:
            num_nu_h = self.substrate.GetAtomWithIdx(self.nu_atom).GetTotalNumHs()

        # TODO Parse and handle other types of acids and bases.
        # Take out strongest weakest base
        if len(ad.mols["weak_bases"]) > 0 and num_nu_h > 0:
            self.has_agent = True
            self.agent = ad.mols["weak_bases"][np.argmax(ad.pkas["weak_bases"])]
            self.agent_atom = ad.active_atoms["weak_bases"][np.argmax(ad.pkas["weak_bases"])]
        else:
            self.has_agent = False
            self.agent = None
            self.agent_atom = None

        # Add hydrogens and optimize the 3D structures
        self.substrate = self.get_conformer(self.substrate, n_confs=n_confs)
        self.product = self.get_conformer(self.product, n_confs=n_confs)
        if not self.intramolecular:
            self.nu = self.get_conformer(self.nu, n_confs=n_confs)
        self.lg = self.get_conformer(self.lg, n_confs=n_confs)
        
        # Optimize agent and take out important atoms. Active atom can be
        # tagged with atom mapping 1.
        if self.has_agent:
            self.agent = self.get_conformer(self.agent)
            self.agent_atoms = [atom.GetIdx() for atom in self.agent.GetAtoms()]
            # If active atom is not given, estimate with xtb
            if detect_agent_atom:
                logger.info("Determining basic site in agent from localized molecular orbitals")
                # Create ASE atoms object
                symbols = [atom.GetSymbol() for atom in self.agent.GetAtoms()]
                positions = self.agent.GetConformer().GetPositions()
                charge = Chem.GetFormalCharge(self.agent)
                atoms = Atoms(symbols=symbols, positions=positions)
                atoms.info["charge"] = charge

                # Run xtb calculation with localization
                os.mkdir("agent")
                with cd("agent"):
                    xtb = XTBCalculator(atoms, "agent.xyz")
                    xtb.options["lmo"] = True
                    xtb.single_point().wait()
                    data = XTBParser(xtb.output)
                    basic_atom = max(data.lmo_dict, key=data.lmo_dict.get)
                    self.agent_atom = basic_atom - 1
                logger.info(f"Most basic site is atom {self.agent_atom}")

        # Add H atoms to indices lists
        for atom in self.substrate.GetAtoms():
            if atom.GetSymbol() == "H":
                neighbor = atom.GetNeighbors()[0]
                if neighbor.GetIdx() in self.core_atoms:
                    self.core_atoms.append(atom.GetIdx())
                elif neighbor.GetIdx() in self.lg_atoms:
                    self.lg_atoms.append(atom.GetIdx())

        # Take out atom indices of reactive ring
        rings = self.substrate.GetRingInfo().AtomRings()
        ring_atoms = set()
        ortho_carbons = []
        for ring in rings:
            if self.central_atom in ring:
                ring_atoms.update(ring)
                for neighbor in self.substrate.GetAtomWithIdx(self.central_atom).GetNeighbors():
                   if neighbor.GetIdx() in ring:
                      ortho_carbons.append(neighbor.GetIdx())
        ring_atoms.remove(self.central_atom)
        self.ring_atoms = ring_atoms
        self.ortho_carbons = ortho_carbons

        # Find ortho nitro groups
        substituent = Chem.MolFromSmarts("[N+](=O)[O-]")
        sub_matches = self.substrate.GetSubstructMatches(substituent)

        no2_substituted = []
        for i in self.ring_atoms:
            neighbors = self.substrate.GetAtomWithIdx(i).GetNeighbors()
            for neighbor in neighbors:
                idx = neighbor.GetIdx()
                for match in sub_matches:
                    if idx in match:
                        no2_substituted.append(atom.GetIdx())
        if len(no2_substituted) > 0:
            self.ortho_nitro = True
        
        # Create reaction complex after computing offset. Use different
        # distances for inter and intramolecular cases.
        if self.intramolecular:
            self.reaction_complex = Chem.Mol(self.substrate)
            freeze_dist = 3
            self.nu_atom_orig = None
        else:
            # Calculate vector and distance for adding nucleophile.
            center_pos = self.substrate.GetConformer().GetPositions()[self.central_atom]
            lg_pos = self.substrate.GetConformer().GetPositions()[self.lg_atom]
            
            # If sp3, set angle of attack from the back
            hybridization = self.substrate.GetAtomWithIdx(self.central_atom).GetHybridization().name
            if hybridization == "SP3":
                freeze_dist = 6
                vec = center_pos - lg_pos
                vec = vec / np.linalg.norm(vec)
                fix_point = center_pos + vec * freeze_dist
                self.sp3 = True
            if hybridization == "SP2":
                freeze_dist = 3
                coordinates = self.substrate.GetConformer().GetPositions()
                center_coordinates = np.array(self.substrate.GetConformer().GetAtomPosition(self.central_atom))
                neighbor_coordinates = []
                for atom in self.substrate.GetAtomWithIdx(self.central_atom).GetNeighbors():
                    if atom.GetIdx() in self.ring_atoms:
                        coordinates = np.array(self.substrate.GetConformer().GetAtomPosition(atom.GetIdx()))
                        neighbor_coordinates.append(coordinates)
                v1 = neighbor_coordinates[0] - center_coordinates
                v2 = neighbor_coordinates[1] - center_coordinates
                vec = np.cross(v1, v2)
                vec /= np.linalg.norm(vec)
                fix_point = center_coordinates + vec * freeze_dist
            # If sp2, set angle of attack 90 from above plane formed by SP2 atoms.
            offset = vec * 15
            offset = rdkit.Geometry.rdGeometry.Point3D(*offset)
            
            # Combine the two molecules
            self.reaction_complex = AllChem.CombineMols(self.substrate, self.nu, offset=offset)
            Chem.SanitizeMol(self.reaction_complex)
    
            # Update the atom numbers
            self.nu_atom_orig = self.nu_atom
            self.nu_atom = self.nu_atom + self.substrate.GetNumAtoms()
            self.nu_atoms = [idx + self.substrate.GetNumAtoms() for idx in self.nu_atoms]

        # Save reaction complex for reference
        self.reaction_complex_pre = Chem.Mol(self.reaction_complex)

        # Take out Nu-H atoms and LG-H atoms
        nu_h_atoms = [neighbor.GetIdx() for neighbor in self.reaction_complex.GetAtomWithIdx(self.nu_atom).GetNeighbors() if neighbor.GetAtomicNum() == 1]
        lg_h_atoms = [neighbor.GetIdx() for neighbor in self.lg.GetAtomWithIdx(self.lg_atom_orig).GetNeighbors() if neighbor.GetAtomicNum() == 1]
        
        # Set up the agent if it exists
        if self.has_agent:
            # Get vector from nucleophile to one of its H atoms (the first).
            nu_pos = self.reaction_complex.GetConformer().GetPositions()[self.nu_atom]
            h_pos = self.reaction_complex.GetConformer().GetPositions()[nu_h_atoms[0]]
            vec = h_pos - nu_pos
            vec = vec / np.linalg.norm(vec)
            offset = vec * 15
            offset = rdkit.Geometry.rdGeometry.Point3D(*offset)
            
            # Combine the two molecules
            self.reaction_complex_agent = AllChem.CombineMols(self.reaction_complex, self.agent, offset=offset)
            Chem.SanitizeMol(self.reaction_complex_agent)
            
            # Update the atom numbers
            self.agent_atoms = [idx + self.reaction_complex.GetNumAtoms() for idx in self.agent_atoms]
            self.agent_atom_orig = self.agent_atom
            self.agent_atom = self.agent_atom + self.reaction_complex.GetNumAtoms()
            
            self.reaction_complex_agent_pre = Chem.Mol(self.reaction_complex_agent)
        
        # Perform constrained optimization of reaction complex.
        ff = AllChem.MMFFGetMoleculeForceField(self.reaction_complex, AllChem.MMFFGetMoleculeProperties(self.reaction_complex),
                                               ignoreInterfragInteractions=False)
        if not ff:
            ff = AllChem.UFFGetMoleculeForceField(self.reaction_complex, ignoreInterfragInteractions=False)
            self.ff_type = "uff"
        else:
            self.ff_type = "mmff"
        
        if not self.intramolecular:
            # Add fix point and constrain Nu atom to it.
            fix_index = ff.AddExtraPoint(*fix_point) - 1 #point indexing starts from 1 apparently
            if self.ff_type == "mmff":
                ff.MMFFAddDistanceConstraint(self.nu_atom, fix_index, False, 0.0, 0.0, 1e6)
            elif self.ff_type == "uff":
                ff.UFFAddDistanceConstraint(self.nu_atom, fix_index, False, 0.0, 0.0, 1e6)
        
            #Fix positions of the substrate
            for atom in self.substrate.GetAtoms():
                ff.AddFixedPoint(atom.GetIdx())
            # Fix atoms of nucleophile so that they are at least d(central_atom-nu_atom) away
            for idx in self.nu_atoms:
                if idx != self.nu_atom:
                    if self.ff_type == "mmff":
                        ff.MMFFAddDistanceConstraint(self.central_atom, idx, False, freeze_dist, 100, 1e6)
                    elif self.ff_type == "uff":
                        ff.UFFAddDistanceConstraint(self.central_atom, idx, False, freeze_dist, 100, 1e6)                        
        else:
            if self.ff_type == "mmff":
                ff.MMFFAddDistanceConstraint(self.central_atom, self.nu_atom, False, freeze_dist, freeze_dist, 1e6)
            elif self.ff_type == "uff":
                ff.UFFAddDistanceConstraint(self.central_atom, self.nu_atom, False, freeze_dist, freeze_dist, 1e6)
        ff.Initialize()
        ff.Minimize(maxIts=500)
        Chem.rdMolTransforms.CanonicalizeConformer(self.reaction_complex.GetConformer())

        # Perform constrained optimization of reaction complex with agent.
        if self.has_agent:
            # Set up force field.
            if self.ff_type == "mmff":
                ff = AllChem.MMFFGetMoleculeForceField(self.reaction_complex_agent, AllChem.MMFFGetMoleculeProperties(self.reaction_complex_agent),
                                                   ignoreInterfragInteractions=False)
                ff.MMFFAddDistanceConstraint(self.agent_atom, nu_h_atoms[0], False, 2.0, 2.0, 1e6)
                ff.MMFFAddDistanceConstraint(self.central_atom, self.agent_atom, False, freeze_dist, 100, 1e6)                                                 
            elif self.ff_type == "uff":
                ff = AllChem.UFFGetMoleculeForceField(self.reaction_complex_agent, ignoreInterfragInteractions=False)
                ff.UFFAddDistanceConstraint(self.agent_atom, nu_h_atoms[0], False, 2.0, 2.0, 1e6)
                ff.UFFAddDistanceConstraint(self.central_atom, self.agent_atom, False, freeze_dist, 100, 1e6)                
            if not self.intramolecular:
                # Add fix point and constrain Nu atom to it.
                fix_index = ff.AddExtraPoint(*fix_point) - 1 #point indexing starts from 1 apparently
                if self.ff_type == "mmff":
                    ff.MMFFAddDistanceConstraint(self.nu_atom, fix_index, False, 0.0, 0.0, 1e6)
                elif self.ff_type == "uff":
                    ff.UFFAddDistanceConstraint(self.nu_atom, fix_index, False, 0.0, 0.0, 1e6)
                #Fix positions of the substrate
                for atom in self.substrate.GetAtoms():
                    ff.AddFixedPoint(atom.GetIdx())
                # Fix atoms of nucleophile so that they are at least d(central_atom-nu_atom) away
                for idx in self.nu_atoms:
                    if idx != self.nu_atom:
                        if self.ff_type == "mmff":
                            ff.MMFFAddDistanceConstraint(self.central_atom, idx, False, freeze_dist, 100, 1e6)
                        elif self.ff_type == "uff":
                            ff.UFFAddDistanceConstraint(self.central_atom, idx, False, freeze_dist, 100, 1e6)
            else:
                if self.ff_type == "mmff":
                    ff.MMFFAddDistanceConstraint(self.central_atom, self.nu_atom, False, freeze_dist, freeze_dist, 1e6)
                elif self.ff_type == "uff":
                    ff.UFFAddDistanceConstraint(self.central_atom, self.nu_atom, False, freeze_dist, freeze_dist, 1e6)
            
            # Do FF optimization for 500 cycles.
            ff.Initialize()
            ff.Minimize(maxIts=500)
            Chem.rdMolTransforms.CanonicalizeConformer(self.reaction_complex_agent.GetConformer())

        # Populate dictionaries.
        self.smiles["substrate"] = Chem.MolToSmiles(Chem.RemoveHs(self.substrate))
        self.smiles["product"] = Chem.MolToSmiles(Chem.RemoveHs(self.product))
        if not self.intramolecular:
            self.smiles["nu"] = Chem.MolToSmiles(Chem.RemoveHs(self.nu))
        else:
            self.smiles["nu"] = None
        self.smiles["lg"] = Chem.MolToSmiles(Chem.RemoveHs(self.lg))
        self.reaction_smiles = self.smiles["substrate"]
        if not self.intramolecular:
            self.reaction_smiles += "." + self.smiles["nu"] + ">"
        if self.has_agent:
            self.smiles["agent"] = Chem.MolToSmiles(Chem.RemoveHs(self.agent))
            self.reaction_smiles += self.smiles["agent"]
        self.reaction_smiles += ">" + self.smiles["product"] + "." + self.smiles["lg"]
        self.smiles["reaction"] = self.reaction_smiles 
        self.smiles["reaction_orig"] = self.reaction_smiles_orig

        self.inchi = {}
        self.inchi["substrate"] = Chem.MolToInchi(Chem.RemoveHs(self.substrate))
        self.inchi["product"] = Chem.MolToInchi(Chem.RemoveHs(self.product))
        if not self.intramolecular:
            self.inchi["nu"] = Chem.MolToInchi(Chem.RemoveHs(self.nu))
        else: 
            self.inchi["nu"] = None
        self.inchi["lg"] = Chem.MolToInchi(Chem.RemoveHs(self.lg))
        if self.has_agent:
            self.inchi["agent"] = Chem.MolToInchi(Chem.RemoveHs(self.agent))

        self.inchi_key = {}
        self.inchi_key["substrate"] = Chem.MolToInchiKey(Chem.RemoveHs(self.substrate))
        self.inchi_key["product"] = Chem.MolToInchiKey(Chem.RemoveHs(self.product))
        if not self.intramolecular:
            self.inchi_key["nu"] = Chem.MolToInchiKey(Chem.RemoveHs(self.nu))
        else:
            self.inchi_key["nu"] = None
        self.inchi_key["lg"] = Chem.MolToInchiKey(Chem.RemoveHs(self.lg))
        if self.has_agent:
            self.inchi_key["agent"] = Chem.MolToInchiKey(Chem.RemoveHs(self.agent))

        self.atom_nos["core"] = self.core_atoms
        self.atom_nos["nu_atom"] = self.nu_atom
        self.atom_nos["nu_h_atoms"] = nu_h_atoms
        self.atom_nos["lg_h_atoms"] = lg_h_atoms
        if not self.intramolecular:
            self.atom_nos["nu"] = self.nu_atoms
            self.charge["nu"] = Chem.GetFormalCharge(self.nu)
        else:
            self.atom_nos["nu"] = None
            self.charge["nu"] = None

        self.atom_nos["lg"] = self.lg_atoms
        self.atom_nos["central_atom"] = self.central_atom
        self.atom_nos["lg_atom"] = self.lg_atom
        self.atom_nos["central_atom_prod"] = self.central_atom_prod
        self.atom_nos["added_atom"] = self.added_atom
        self.atom_nos["nu_atom_orig"] = self.nu_atom_orig
        self.atom_nos["lg_atom_orig"] = self.lg_atom_orig
        self.atom_nos["ring_atoms"] = self.ring_atoms
        self.atom_nos["ortho_carbons"] = self.ortho_carbons
        self.charge["substrate"] = Chem.GetFormalCharge(self.substrate)
        self.charge["product"] = Chem.GetFormalCharge(self.product)
        self.charge["lg"] = Chem.GetFormalCharge(self.lg)
        self.charge["nu_atom"] = self.nu.GetAtomWithIdx(self.nu_atom_orig).GetFormalCharge()

        if self.has_agent:
            self.charge["agent"] = Chem.GetFormalCharge(self.agent)
            self.charge["agent_atom"] = self.agent.GetAtomWithIdx(self.agent_atom).GetFormalCharge()
            self.atom_nos["agent"] = self.agent_atoms
            self.atom_nos["agent_atom"] = self.agent_atom
            self.atom_nos["agent_atom_orig"] = self.agent_atom_orig

        # Detect azide
        if self.smiles["nu"] == '[N-]=[N+]=[N-]' and not self.sp3:
            self.azide = True
            neighbor = self.reaction_complex.GetAtomWithIdx(self.nu_atom).GetNeighbors()[0]
            nn_neighbor = neighbor.GetNeighbors()[0]
            self.azide_angle = [self.nu_atom, neighbor.GetIdx(), nn_neighbor.GetIdx()]
            self.azide_dihedral = [self.lg_atom, self.central_atom, self.nu_atom, neighbor.GetIdx()]
        
        # Set up clustering dictionary 
        self.clustering = {}

        # Detect if nucleophile should be clustered
        if not self.intramolecular:
            atom = self.nu.GetAtomWithIdx(self.nu_atom_orig)
            charge = atom.GetFormalCharge()
            atomic_number = atom.GetAtomicNum()
            if charge < 0 and atomic_number < 18:
                self.clustering["nucleophile"] = True
            else:
                self.clustering["nucleophile"] = False
        else:
            self.clustering["nucleophile"] = None

        # Detect if leaving group should be clustered
        atom = self.lg.GetAtomWithIdx(self.lg_atom_orig)
        charge = atom.GetFormalCharge()
        atomic_number = atom.GetAtomicNum()
        if charge < 0 and atomic_number < 18:
            self.clustering["leaving_group"] = True
        else:
            self.clustering["leaving_group"] = False

        # Detect if agent should be clustered
        if self.has_agent:
            atom = self.agent.GetAtomWithIdx(self.agent_atom_orig)
            charge = atom.GetFormalCharge()
            atomic_number = atom.GetAtomicNum()
            if charge < 0 and atomic_number < 18:
                self.clustering["agent"] = True
            else:
                self.clustering["agent"] = False
        
        # Determine if TS should be clustered
        if self.has_agent:
            if any([self.clustering["nucleophile"], self.clustering["leaving_group"], self.clustering["agent"]]):
                self.clustering["ts"] = True
            else:
                self.clustering["ts"] = False
        else:
            if any([self.clustering["nucleophile"], self.clustering["leaving_group"]]):
                self.clustering["ts"] = True
            else:
                self.clustering["ts"] = False
        
        # Detect sp3 neighbors of nu atom in nucleophile
        neighbor_atoms = self.nu.GetAtomWithIdx(self.nu_atom_orig).GetNeighbors()
        sp3_neighbor_indices = [atom.GetIdx() for atom in neighbor_atoms if atom.GetHybridization() == Chem.HybridizationType.SP3]
        self.nu_sp3_neighbors_orig = sp3_neighbor_indices

        # Detect sp3 neighbors of nu atom in reaction complex
        neighbor_atoms = self.reaction_complex.GetAtomWithIdx(self.nu_atom).GetNeighbors()
        sp3_neighbor_indices = [atom.GetIdx() for atom in neighbor_atoms if atom.GetHybridization() == Chem.HybridizationType.SP3]
        self.nu_sp3_neighbors = sp3_neighbor_indices

        # Detect proton transfer
        nu_lost = sum([neighbor.GetAtomicNum() == 1 for neighbor in self.nu.GetAtomWithIdx(self.nu_atom_orig).GetNeighbors()]) - \
                  sum([neighbor.GetAtomicNum() == 1 for neighbor in self.product.GetAtomWithIdx(self.added_atom).GetNeighbors()])
        lg_gained = sum([neighbor.GetAtomicNum() == 1 for neighbor in self.lg.GetAtomWithIdx(self.lg_atom_orig).GetNeighbors()]) - \
                    sum([neighbor.GetAtomicNum() == 1 for neighbor in self.substrate.GetAtomWithIdx(self.lg_atom).GetNeighbors()])
        if nu_lost == 1 and lg_gained and nu_lost == 1:
            self.proton_transfer = True
        
        # Store parsers
        self.agent_detector = ad
        self.reaction_smiles_parser = rsmp

    @staticmethod
    def get_xyz(mol):
        """Get the xyz file text from RDKit mol object.

        Args:
            mol (object): RDKit mol object.

        Returns:
            string (str): Text corresponding to XMOL .xyz file.
        """
        # Get the coordinate and symbols.
        coordinates = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        # Create the xys text
        string = f"{mol.GetNumAtoms()}\n"
        string += "\n"
        for atom, coordinate in zip(symbols, coordinates):
            string += f"{atom:10s}{coordinate[0]:15.5f}{coordinate[1]:15.5f}{coordinate[2]:15.5f}\n"

        return string

    @staticmethod
    def get_conformer(mol, n_confs=None):
        """Generates conformers for an RDKit mol object.

        Args:
            mol (object): RDKit mol object
            n_confs (int): Number of conformers to generate. If 'None', a 
                reasonable number will be set depending on the number of
                rotatable bonds.

        Returns:
            mol (object): RDKit mol object.
        """
        # Add hydrogens
        mol = Chem.AddHs(mol)

        # If n_confs is not set, set number of conformers based on number of
        # rotatable bonds 
        if not n_confs:
            n_rot_bonds = NumRotatableBonds(mol)
    
            if n_rot_bonds <= 7:
                n_confs = 50
            elif n_rot_bonds >= 8 and n_rot_bonds <= 12:
                n_confs = 200
            else:
                n_confs = 300
        # Get all the conformers and rank with MMFF. Set the lowest energy
        # conformer with MMFF as the active conformer.
        if mol.GetNumAtoms() > 2:
            AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=1)
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
            energy_list = [x[1] for x in res]
            min_conformer_index = energy_list.index(min(energy_list))
            min_conformer = mol.GetConformer(min_conformer_index)
            mol = Chem.Mol(mol)
            mol.RemoveAllConformers()
            mol.AddConformer(min_conformer)
            mol.GetConformer().SetId(0)
        else:
            AllChem.EmbedMolecule(mol, randomSeed=1)
            res = AllChem.MMFFOptimizeMolecule(mol)            

        return mol

    def write_xyz(self):
        """Write structures to xyz files"""

        # Create directory.
        if not os.path.isdir("xyz_from_smiles"):
            os.mkdir("xyz_from_smiles")
        
        # Write the structures to file.
        with open("xyz_from_smiles/substrate.xyz", "w") as file:
            file.write(self.get_xyz(self.substrate))
        if not self.intramolecular:
            with open("xyz_from_smiles/nucleophile.xyz", "w") as file:
                file.write(self.get_xyz(self.nu))
        with open("xyz_from_smiles/product.xyz", "w") as file:
            file.write(self.get_xyz(self.product))
        with open("xyz_from_smiles/leaving_group.xyz", "w") as file:
            file.write(self.get_xyz(self.lg))
        with open("xyz_from_smiles/reaction_complex.xyz", "w") as file:
            file.write(self.get_xyz(self.reaction_complex))
        if self.has_agent:
            with open("xyz_from_smiles/reaction_complex_agent.xyz", "w") as file:
                file.write(self.get_xyz(self.reaction_complex_agent))
            with open("xyz_from_smiles/agent.xyz", "w") as file:
                file.write(self.get_xyz(self.agent))                

    def write_info(self):
        """Write system information file input for the main program. Write smiles.txt"""
        # Write config file for the TS finding script
        with open("system_info", "w") as file:
            file.write("[GENERAL]\n")
            if self.has_agent:
                file.write("agent = True\n")
            if self.sp3:
                file.write("find_intermediate = False\n")
            if self.ortho_nitro:
                file.write("ortho_nitro = True\n")
            file.write(f"proton_transfer = {self.proton_transfer}\n")
            if self.azide:
                file.write("azide_nucleophile = True\n")
                file.write(f"azide_angle = {' '.join([str(i + 1) for i in self.azide_angle])}\n")
                file.write(f"azide_dihedral = {' '.join([str(i + 1) for i in self.azide_dihedral])}\n")
            else:
                file.write("azide_nucleophile = False\n")
                file.write("azide_angle = \n")
                file.write("azide_dihedral = \n")
            file.write("[REACTIVE ATOMS]\n")
            file.write(f"central_atom = {self.atom_nos['central_atom'] + 1}\n")
            file.write(f"nucleophilic_atom = {self.atom_nos['nu_atom'] + 1}\n")
            file.write(f"leaving_atom = {self.atom_nos['lg_atom'] + 1}\n")
            file.write(f"central_atom_prod = {self.atom_nos['central_atom_prod'] + 1}\n")
            file.write(f"added_atom = {self.atom_nos['added_atom'] + 1}\n")
            if not self.intramolecular:
                file.write(f"nu_atom_orig = {self.atom_nos['nu_atom_orig'] + 1}\n")
                file.write(f"lg_atom_orig = {self.atom_nos['lg_atom_orig'] + 1}\n")
                file.write(f"fragment_nu = {' '.join([str(i + 1) for i in self.atom_nos['nu']])}\n")
            file.write(f"ring_atoms = {' '.join([str(i + 1) for i in self.atom_nos['ring_atoms']])}\n")              
            file.write(f"ortho_carbons = {' '.join([str(i + 1) for i in self.atom_nos['ortho_carbons']])}\n")
            file.write(f"fragment_lg = {' '.join([str(i + 1) for i in self.atom_nos['lg']])}\n")
            file.write(f"nu_h_atoms = {' '.join([str(i + 1) for i in self.atom_nos['nu_h_atoms']])}\n")
            file.write(f"lg_h_atoms = {' '.join([str(i + 1) for i in self.atom_nos['lg_h_atoms']])}\n")
            file.write(f"nu_sp3_neighbors = {' '.join([str(i + 1) for i in self.nu_sp3_neighbors])}\n")
            file.write(f"nu_sp3_neighbors_orig = {' '.join([str(i + 1) for i in self.nu_sp3_neighbors_orig])}\n")            
            file.write(f"fragment_substrate = {' '.join([str(i + 1) for i in self.atom_nos['core']])}\n")
            if self.has_agent:
                file.write(f"agent_atom = {self.atom_nos['agent_atom'] + 1}\n")
                file.write(f"agent_atom_orig = {self.atom_nos['agent_atom_orig'] + 1}\n")    
                file.write(f"fragment_agent = {' '.join([str(i + 1) for i in self.atom_nos['agent']])}\n")
            file.write("[CHARGES]\n")
            file.write(f"substrate = {self.charge['substrate']}\n")
            if not self.intramolecular:
                file.write(f"nucleophile = {self.charge['nu']}\n")
            file.write(f"product = {self.charge['product']}\n")
            file.write(f"leaving_group = {self.charge['lg']}\n")
            if self.has_agent:
                file.write(f"agent = {self.charge['agent']}\n")
            file.write(f"nucleophilic_atom = {self.charge['nu_atom']}\n")
            if self.has_agent:
                file.write(f"agent_atom = {self.charge['agent_atom']}\n")
            file.write("[CLUSTERING]\n")
            if not self.intramolecular:
                file.write(f"nucleophile = {self.clustering['nucleophile']}\n")
            file.write(f"leaving_group = {self.clustering['leaving_group']}\n")
            file.write(f"ts = {self.clustering['ts']}\n")
            if self.has_agent:
                file.write(f"agent = {self.clustering['agent']}\n")
        
        # Write the SMILES to file.
        with open("smiles.txt", "w") as file:
            file.write(f"Reaction: {self.smiles['reaction']}\n")
            file.write(f"Substrate: {self.smiles['substrate']}\n")
            file.write(f"Product: {self.smiles['product']}\n")
            if not self.intramolecular:
                file.write(f"Nucleophile: {self.smiles['nu']}\n")
            file.write(f"Leaving_group: {self.smiles['lg']}\n")
            if self.has_agent:
                file.write(f"Agent: {self.smiles['agent']}\n")
            file.write(f"Original input: {self.reaction_smiles_orig}\n")

    def __repr__(self):
        return f"{self.__class__.__name__}(substrate='{self.smiles['substrate']}')"


class ReactionSmilesProcessor:
    """Processes reaction SMILES for SNAR to pick out reactants, products
    and reactive atoms.
    
    Args:
        reaction_smiles (str): Reaction SMILES to be processed

    Attributes:
        substrate (object): Mol object for substrate
        products (list): Mol objects for product(s)
        nu (object): Mol object for nucleophile
        lgs (list): Mol objects for leaving group(s)
        agents (list): Mol objects for agents
        byproducts (list): Mol objects for byproducts

        central_atom (int): Index of central atom
        central_atom_prod (int): Index of central atom in product
        added_atom (int): Index of nucleophilic atom in product
        nu_atom (int): Index of nucleophilic atom in nucleophile
        lg_atom (int): Index of leaving atom in substrate
        lg_atom_orig (int): Index of leaving atom in leaving group.
        
        lg_atoms (list): Indices of atoms in leaving group
        nu_atoms (list): Indices of atoms in nucleophile
        core_atoms (list): Indices of atoms in the reactive core.

        reaction (object): RDKit reaction object.
        reaction_smiles (str): Reaction SMILES after processing
        reaction_smiles_orig (str): Reaction SMILES before processing.
        reaction_type (str): Type of reaction 'intramolecular' or 'conventional'        
        unreacted_starting_material (bool): Whether unreacted starting material
            is present on the right hand side.
    """
    def __init__(self, reaction_smiles):
        # Set up attributes
        self.substrate = None
        self.products = []
        self.nu = None
        self.nu_tautomers = None
        self.lgs = []
        self.agents = []
        self.byproducts = []

        self.central_atom = []
        self.central_atom_prod = []
        self.added_atom = []
        self.nu_atom = []
        self.lg_atom = []
        self.lg_atom_orig = []

        self.nu_atoms = []
        self.lg_atoms = []
        self.core_atoms = []
        self._core_patterns = []

        self.reaction_smiles_orig = reaction_smiles
        self.reaction_smiles = None

        self.reaction_type = None
        self.unreacted_starting_material = False

        # Initialize reaction
        self.reaction = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)

        # Remove salts and duplicates
        self._cleanup_reaction()
       
        # Set up reactant and products lists
        self._reactant_mols = list(self.reaction.GetReactants())
        self._product_mols = list(self.reaction.GetProducts())
       
        # Determine reactant identities
        self._identify_roles()

        # Create LG
        self._create_lgs()   

        # Identify agents
        self._identify_agents()

        # Identify byproducts
        self._identify_byproducts()

        # Test that atom counts are balanced
        for product, lg in zip(self.products, self.lgs):
            atoms_left = self.substrate.GetNumAtoms()
            if self.nu:
                atoms_left += self.nu.GetNumAtoms()
            atoms_right = product.GetNumAtoms()
            atoms_right += lg.GetNumAtoms()
            if atoms_left != atoms_right:
                raise Exception(f"Unequal number of atoms on each side of reaction. L: {atoms_left}, R: {atoms_right}")
            
        # Identify reactive atoms
        self._identify_reactive()
        
        # Create reaction object
        self._create_reaction()
        
        # Set reaction smiles
        self.reaction_smiles = AllChem.ReactionToSmiles(self.reaction)

    def _create_reaction(self):
        self.reaction = AllChem.ChemicalReaction()
        self.reaction.AddReactantTemplate(self.substrate)
        if self.nu:
            self.reaction.AddReactantTemplate(self.nu)
        if self.agents:
            for agent in self.agents:
                self.reaction.AddAgentTemplate(agent)
        for product, lg in zip(self.products, self.lgs):
            self.reaction.AddProductTemplate(product)
            self.reaction.AddProductTemplate(lg)        

    def _identify_roles(self):
        # Set up dictionary of MCS pairs
        mcs_record = {}

        # Set up lists of core patterns for substrates and products
        core_patterns_substrate = []
        core_patterns_product = []
        
        # Set up lists for candidates
        nu_candidates = []
        lg_candidates = []
        substrate_candidates = []
        product_candidates = []        

        for mol_l, mol_r in itertools.product(self._reactant_mols, self._product_mols):
            # Special treatment for single atoms as FindMCS cannot handle that
            if mol_l.GetNumAtoms() == 1 or mol_r.GetNumAtoms() == 1:
                # Set up default values
                matches_l = []
                matches_r = []
                mcs_list = []

                # Neutralize as HasSubstructMatch is very specific about this
                neutral_l = Chem.Mol(mol_l)
                neutral_r = Chem.Mol(mol_r)
                for atom in [*neutral_l.GetAtoms(), *neutral_r.GetAtoms()]:
                    atom.SetFormalCharge(0)
                # If atom is contained the other molecule, then this is the MCS
                if neutral_l.GetNumAtoms() == 1:
                    if neutral_r.HasSubstructMatch(neutral_l):
                        mcs_list = [neutral_l]
                        matches_l = neutral_l.GetSubstructMatches(neutral_l)
                        matches_r = neutral_r.GetSubstructMatches(neutral_l)
                if neutral_r.GetNumAtoms() == 1:
                    if neutral_l.HasSubstructMatch(neutral_r):
                        mcs_list = [neutral_r]
                        matches_l = neutral_l.GetSubstructMatches(neutral_r)
                        matches_r = neutral_r.GetSubstructMatches(neutral_r)
            else:
                # Find the MCS between left and right molecule
                mcs_list = []
                mcs_result = Chem.rdFMCS.FindMCS([mol_l, mol_r], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact, ringMatchesRingOnly=True)
                if mcs_result.numAtoms > 0:
                    mcs = mcs_result.queryMol
                else:
                    mcs = Chem.Mol()

                # If MCS found, remove mcs from first left and then right molecule
                # to see if there are MCS with same number of atoms.
                if mcs.GetNumAtoms() > 0:
                    mcs_list.append(mcs)
                    mcs_mol_l = Chem.Mol(mol_l)                    
                    while True:
                        mcs_mol_l = Chem.DeleteSubstructs(mcs_mol_l, mcs)
                        mcs_result = Chem.rdFMCS.FindMCS([mcs_mol_l, mol_r], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact, ringMatchesRingOnly=True)
                        if mcs_result.numAtoms > 0:
                            mcs = mcs_result.queryMol
                        else:
                            mcs = Chem.Mol()
                        
                        if mcs.GetNumAtoms() == mcs_list[-1].GetNumAtoms():
                            mcs_list.append(mcs)
                        else:
                            break                   
                    mcs_mol_r = Chem.Mol(mol_r)
                    while True:
                        mcs_mol_r = Chem.DeleteSubstructs(mcs_mol_r, mcs)
                        mcs_result = Chem.rdFMCS.FindMCS([mol_l, mcs_mol_r], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact, ringMatchesRingOnly=True)
                        if mcs_result.numAtoms > 0:
                            mcs = mcs_result.queryMol
                        else:
                            mcs = Chem.Mol()                        

                        if mcs.GetNumAtoms() == mcs_list[-1].GetNumAtoms():
                            mcs_list.append(mcs)
                        else:
                            break
            
            # Save list of MCS hits for further use
            if len(mcs_list) > 0:
                mcs_record[frozenset([mol_l, mol_r])] = mcs_list
            
            # Loop over MCS in list to find candidates
            for mcs in mcs_list:
                # Find substructurematches with MCS unless one mol has
                # only one atom
                if not(mol_l.GetNumAtoms() == 1 or mol_r.GetNumAtoms() == 1):
                    matches_l = mol_l.GetSubstructMatches(mcs)
                    matches_r = mol_r.GetSubstructMatches(mcs)
                
                # Nucleophile or leaving group must be fully contained in the MCS
                if not (mcs.GetNumAtoms() == mol_l.GetNumAtoms() or mcs.GetNumAtoms() == mol_r.GetNumAtoms()):
                    continue
                # Loop over matches to left mol to find candidates for lg
                # and substrate
                for match in matches_l:
                    matched_atoms = list(match)
                    Chem.SanitizeMol(mol_l)

                    # Find aromatic ring atoms not in match
                    rings = self._get_aromatic_rings(mol_l)

                    non_matched_ring_atoms = []
                    fragments_rings = False
                    for ring in rings:
                        overlapping_atoms = set(matched_atoms).intersection(ring)
                        if len(overlapping_atoms) == 0:
                            non_matched_ring_atoms.extend(ring)
                        # If match is partly part of aromatic ring the match
                        # is wrong.
                        elif len(overlapping_atoms) < len(ring):
                            fragments_rings = True
                    if fragments_rings:
                        continue
                    
                    # Find instances where there is a bond between matched atoms
                    # and non-matched ring atom which is an aromatic C atom
                    for i, j in itertools.product(matched_atoms, non_matched_ring_atoms):
                        bond = mol_l.GetBondBetweenAtoms(i, j)
                        if bond:
                            if bond.GetBondType() == Chem.BondType.SINGLE:
                                if mol_l.GetAtomWithIdx(j).GetSymbol() == "C" and mol_l.GetAtomWithIdx(j).GetIsAromatic() == True:
                                    # Construct core pattern by removing atoms not
                                    # in the corrected MCS
                                    remove_indices = matched_atoms
                                    remove_indices.sort(reverse=True)
                                    rw_mol = Chem.RWMol(mol_l)
                                    
                                    atom = rw_mol.GetAtomWithIdx(j)
                                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                                    for k in remove_indices:
                                        rw_mol.RemoveAtom(k)
                                    
                                    # Append candidates
                                    core_pattern_substrate = rw_mol.GetMol()
                                    n_frags = len(AllChem.GetMolFrags(core_pattern_substrate))
                                    if n_frags > 1:
                                        continue                                    
                                    Chem.SanitizeMol(core_pattern_substrate)
                                    core_patterns_substrate.append(core_pattern_substrate)
                                    lg_candidates.append(mol_r)
                                    substrate_candidates.append(mol_l)                              

                # Loop over matches to right mol to find candidates for nu
                # and product                
                for match in matches_r:
                    matched_atoms = list(match)
                    Chem.SanitizeMol(mol_r)

                    # Find aromatic ring atoms not in match
                    rings = self._get_aromatic_rings(mol_r)

                    non_matched_ring_atoms = []
                    fragments_rings = False
                    for ring in rings:
                        overlapping_atoms = set(matched_atoms).intersection(ring)
                        if len(overlapping_atoms) == 0:
                            non_matched_ring_atoms.extend(ring)
                        # If match is partly part of aromatic ring the match
                        # is wrong.
                        elif len(overlapping_atoms) < len(ring):
                            fragments_rings = True
                    if fragments_rings:
                        continue

                    # Find instances where there is a bond between matched atoms
                    # and non-matched ring atom which is a C atom                            
                    for i, j in itertools.product(matched_atoms, non_matched_ring_atoms):
                        bond = mol_r.GetBondBetweenAtoms(i, j)
                        if bond:
                            if bond.GetBondType() == Chem.BondType.SINGLE:
                                if mol_r.GetAtomWithIdx(j).GetSymbol() == "C" and mol_r.GetAtomWithIdx(j).GetIsAromatic() == True:
                                    # Construct core pattern by removing atoms not
                                    # in the corrected MCS
                                    remove_indices = matched_atoms
                                    remove_indices.sort(reverse=True)
                                    rw_mol = Chem.RWMol(mol_r)
                                    atom = rw_mol.GetAtomWithIdx(j)
                                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                                    for k in remove_indices:
                                        rw_mol.RemoveAtom(k)
                                    
                                    # Append candidates
                                    core_pattern_product = rw_mol.GetMol()
                                    n_frags = len(AllChem.GetMolFrags(core_pattern_product))
                                    if n_frags > 1:
                                        continue
                                    Chem.SanitizeMol(core_pattern_product)
                                    core_patterns_product.append(core_pattern_product)
                                    nu_candidates.append(mol_l)
                                    product_candidates.append(mol_r)        

        # Find roles in case that both lg and nu candidates exist
        if len(core_patterns_substrate) > 0 and len(core_patterns_product) > 0:
            # Loop over candidate pairs
            for (nu_candidate, product_candidate, core_pattern_product), (lg_candidate, substrate_candidate, core_pattern_substrate) in \
                itertools.product(zip(nu_candidates, product_candidates, core_patterns_product), zip(lg_candidates, substrate_candidates, core_patterns_substrate)):
                mcs_num_atoms = Chem.rdFMCS.FindMCS([core_pattern_product, core_pattern_substrate]).numAtoms
                # Use criteria that:
                # (1) the core_pattern must match each other
                # (2) atom sum of substrate + nu matches product + lg
                if mcs_num_atoms == core_pattern_product.GetNumAtoms() and mcs_num_atoms == core_pattern_substrate.GetNumAtoms() \
                        and substrate_candidate.GetNumAtoms() == core_pattern_substrate.GetNumAtoms() + lg_candidate.GetNumAtoms() \
                        and product_candidate.GetNumAtoms() == core_pattern_product.GetNumAtoms() + nu_candidate.GetNumAtoms():
                    self.substrate = substrate_candidate
                    self.nu = nu_candidate
                    if not product_candidate in self.products:
                        self.products.append(product_candidate)
                    if not lg_candidate in self.lgs:
                        self.lgs.append(lg_candidate)
                    if not core_pattern_product in self._core_patterns:
                        self._core_patterns.append(core_pattern_product)
                        core_atoms = list(self.substrate.GetSubstructMatch(core_pattern_product))
                    if not core_atoms in self.core_atoms:
                        self.core_atoms.append(core_atoms)
        # Find roles in case that only one of lg and nu candidates exist
        if not all([self.substrate, self.nu, self.products]):
            # Loop over mol pairs and their associated MCS lists
            for mols, mcs_list in mcs_record.items():
                for mcs in mcs_list:
                    # Loop over candidates from nu matches
                    for nu_candidate, product_candidate, core_pattern_product in zip(nu_candidates, product_candidates, core_patterns_product):
                        core_pattern_product = Chem.MolFromSmarts(Chem.MolToSmarts(core_pattern_product))
                        mcs = Chem.MolFromSmarts(Chem.MolToSmarts(mcs))
                        check_num_atoms = Chem.rdFMCS.FindMCS([core_pattern_product, mcs]).numAtoms
                        substrate_mols = [mol for mol in mols if mol not in [product_candidate, nu_candidate]]
                        # Use criteria that:
                        # (1) MCS fully matches core_pattern of the product
                        # (2) Atom sum of core pattern and nu adds up to product
                        #     candidate
                        if check_num_atoms == core_pattern_product.GetNumAtoms() \
                                and product_candidate.GetNumAtoms() == core_pattern_product.GetNumAtoms() + nu_candidate.GetNumAtoms() \
                                and len(substrate_mols) == 1:
                            # Avoid duplication of products
                            if not product_candidate in self.products:
                                # Check that core pattern only has one neighbor to other atoms
                                substrate_candidate = substrate_mols[0]
                                mcs_result = Chem.rdFMCS.FindMCS([core_pattern_product, substrate_candidate], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact)                               
                                if mcs_result.numAtoms == 0:
                                    continue
                                else:
                                    query_mol = mcs_result.queryMol
                                indices = substrate_candidate.GetSubstructMatch(query_mol)                               
                                neighbors = 0
                                for i in indices:
                                    atom = substrate_candidate.GetAtomWithIdx(i)
                                    for neighbor in atom.GetNeighbors():
                                        if neighbor.GetIdx() not in indices:
                                            neighbors += 1
                                if neighbors > 1:
                                    continue

                                core_atoms_mcs = Chem.rdFMCS.FindMCS([substrate_candidate, core_pattern_product], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact)
                                core_atoms_pattern = Chem.MolFromSmarts(core_atoms_mcs.smartsString)
                                core_atoms = list(substrate_candidate.GetSubstructMatch(core_atoms_pattern))
                                if len(core_atoms) == core_pattern_product.GetNumAtoms():
                                    self.core_atoms.append(core_atoms)
                                    self.substrate = substrate_candidate
                                    self.products.append(product_candidate)
                                    self.nu = nu_candidate
                                    self._core_patterns.append(core_pattern_product)

                    # Loop over candidates from lg matches                                
                    for lg_candidate, substrate_candidate, core_pattern_substrate in zip(lg_candidates, substrate_candidates, core_patterns_substrate):
                        core_pattern_substrate = Chem.MolFromSmarts(Chem.MolToSmarts(core_pattern_substrate))
                        mcs = Chem.MolFromSmarts(Chem.MolToSmarts(mcs))                        
                        check_num_atoms = Chem.rdFMCS.FindMCS([core_pattern_substrate, mcs], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact).numAtoms
                        # Use criteria that:
                        # (1) MCS fully matches core_pattern of the substrate
                        # (2) Atom sum of core pattern and lg adds up to
                        #     substrate candidate
                        product_mols = [mol for mol in mols if mol is not substrate_candidate]
                        if check_num_atoms == core_pattern_substrate.GetNumAtoms() \
                                and substrate_candidate.GetNumAtoms() == core_pattern_substrate.GetNumAtoms() + lg_candidate.GetNumAtoms() \
                                and len(product_mols) == 1:
                            # Avoid duplication of substrates
                            if not substrate_candidate is not self.substrate:
                                substrate_candidate = substrate_candidate
                                core_atoms_mcs = Chem.rdFMCS.FindMCS([substrate_candidate, core_pattern_product], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact)
                                core_atoms_pattern = Chem.MolFromSmarts(core_atoms_mcs.smartsString)
                                core_atoms = list(substrate_candidate.GetSubstructMatch(core_atoms_pattern))
                                if len(core_atoms) == core_pattern_product.GetNumAtoms():
                                    self.substrate = substrate_candidate
                                    self.products.append(product_mols[0])
                                    self.lgs.append(lg_candidate)
                                    self._core_patterns.append(core_pattern_substrate)
                                    self.core_atoms.append(core_atoms)
        
        # If previous attempts failed, assume intramolecular reaction
        if not all([self.substrate, self.nu, self.products]):
            # Go through the MCS matches
            for mols in mcs_record.keys():
                mols = list(mols)
                mcs_results = Chem.rdFMCS.FindMCS([mols[0], mols[1]], bondCompare=rdkit.Chem.rdFMCS.BondCompare.CompareAny)
                mcs_num_atoms = mcs_results.numAtoms
                mol_num_atoms = [mol.GetNumAtoms() for mol in mols]
                # Use criteria that:
                # (1) Product will have one more atom in the MCS as LG has
                #     left
                if mcs_num_atoms in mol_num_atoms and (mcs_num_atoms + 1) in mol_num_atoms:
                    self.reaction_type = "intramolecular"
                    product = [mol for mol in mols if mol.GetNumAtoms() == min(mol_num_atoms)][0]
                    # Avoid duplication
                    if not product in self.products:
                        self.products.append(product)
                        self.substrate = [mol for mol in mols if mol is not product][0]
                        mcs = mcs_results.queryMol
                        self._core_patterns.append(mcs)
                        core_atoms = list(self.substrate.GetSubstructMatch(mcs))
                        self.core_atoms.append(core_atoms)                 
        
        # Set reaction type if not intramolecular
        if not self.reaction_type:
            self.reaction_type = "conventional"

    def _identify_agents(self):
        # Take out potential agents from reactants
        agents = []
        remove = []
        for mol in self._reactant_mols:
            if mol is not self.substrate and mol is not self.nu:
                agents.append(mol)
                remove.append(mol)
        for mol in remove:
            self._reactant_mols.remove(mol)

        # Add explicit agents
        agents += self.reaction.GetAgents()
        for mol in self.reaction.GetAgents():
            Chem.SanitizeMol(mol)

        # Remove duplicate agents
        identicals = {mol: set() for mol in agents}
        for mol_1, mol_2 in itertools.combinations(agents, 2):
            if mol_1.GetNumAtoms() == mol_2.GetNumAtoms():
                if len(mol_1.GetSubstructMatch(mol_2)) == mol_1.GetNumAtoms():
                    identicals[mol_1].add(mol_2)
                    identicals[mol_2].add(mol_1)
        
        keep = set()
        remove = set()
        for mol, identical_mols in identicals.items():
            if mol not in remove:
                keep.add(mol)
            for identical_mol in identical_mols:
                if identical_mol not in keep:
                    remove.add(identical_mol)

        for mol in remove:
            agents.remove(mol)

        self.agents = agents
    
    def _identify_byproducts(self):
        # Separate byproducts from genuine products
        byproducts = []
        for mol in self._product_mols:
            if mol not in self.products and mol not in self.lgs:
                byproducts.append(mol)
        for mol in byproducts:
            self._product_mols.remove(mol)
        
        # Identify unreacted starting material
        for mol in byproducts:
            mcs = Chem.rdFMCS.FindMCS([mol, self.substrate])
            if mcs.numAtoms == self.substrate.GetNumAtoms():
                byproducts.remove(mol)
                self.unreacted_starting_material = True
        
        self.byproducts = byproducts

    def _identify_reactive(self):
        for core_pattern, product, lg_atoms, core_atoms in zip(self._core_patterns, self.products, self.lg_atoms, self.core_atoms):
            if self.reaction_type == "intramolecular":
                # Get atoms not matched by the core pattern
                matches_product = product.GetSubstructMatches(core_pattern, uniquify=False)
                matches_substrate = self.substrate.GetSubstructMatches(core_pattern, uniquify=False)
                all_atoms = set([atom.GetIdx() for atom in self.substrate.GetAtoms()])
                non_matched_atoms = all_atoms - set(matches_substrate[0])
    
                # Get LG atom. It is the only one unmatched by the MCS
                lg_atom = list(non_matched_atoms)[0]
    
                # Get central atom
                central_atom = self.substrate.GetAtomWithIdx(lg_atom).GetNeighbors()[0].GetIdx()
                
                # Get nu atom. Here we go through all matches due to possible
                # symmetry
                for match_substrate, match_product in itertools.product(matches_substrate, matches_product):
                    if len(self.nu_atom) < 1:
                        central_atom_prod = match_product[match_substrate.index(central_atom)]
                        neighbors_prod = [atom.GetIdx() for atom in product.GetAtomWithIdx(central_atom_prod).GetNeighbors()]
                        
                        # Product must have three neighbors, unless H acts as nucleophile
                        if len(neighbors_prod) < 3:
                            continue
    
                        # Find neighbors of substrate and product central atoms
                        # the added atom cannot be a neighbor in the substrate
                        neighbors_substrate = [atom.GetIdx() for atom in self.substrate.GetAtomWithIdx(central_atom).GetNeighbors() if atom.GetIdx() in match_substrate]
                        mapped_neighbors_prod = [match_product.index(i) for i in neighbors_prod]
                        mapped_neighbors_substrate = [match_substrate.index(i) for i in neighbors_substrate]
                        added_atom_index = list(set(mapped_neighbors_prod) - set(mapped_neighbors_substrate))[0]
                        added_atom = match_product[added_atom_index]
    
                        # Check that added atom has product central atom as neighbor
                        neighbors_added_atom = [atom.GetIdx() for atom in product.GetAtomWithIdx(added_atom).GetNeighbors()]
                        if central_atom_prod not in neighbors_added_atom:
                            continue
                        
                        # Nu atom correspond to added atom in substrate
                        self.nu_atom.append(match_substrate[added_atom_index])
                        self.central_atom_prod.append(central_atom_prod)
                        self.added_atom.append(added_atom)
                self.central_atom.append(central_atom)
                self.lg_atom.append(lg_atom)
            else:
                # Determine the center atom
                for i in lg_atoms:
                    atom = self.substrate.GetAtomWithIdx(i)
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIdx() in core_atoms:
                            self.central_atom.append(neighbor.GetIdx())
                            self.lg_atom.append(i)
        
                # Determine the atoms added to the core in the product
                product_core_indices_all = list(product.GetSubstructMatches(Chem.MolFromSmarts(rdFMCS.FindMCS([core_pattern, self.substrate]).smartsString)))
                for product_core_indices in product_core_indices_all:
                    added_atoms = [atom.GetIdx() for atom in product.GetAtoms() if atom.GetIdx() not in product_core_indices]
            
                    # Determine the atom which is connected to the core in the product
                    for i, j in itertools.product(product_core_indices, added_atoms):
                        if product.GetBondBetweenAtoms(i, j):
                            self.central_atom_prod.append(i)
                            added_atom = j
                    
                    # Determine which atom in the nucleophile is the active one
                    if self.nu.GetNumAtoms() == 1:
                        self.nu_atom.append(0)
                    else:
                        matches = product.GetSubstructMatches(Chem.MolFromSmarts(Chem.rdFMCS.FindMCS([product, self.nu]).smartsString))
                        nu_matches = self.nu.GetSubstructMatches(Chem.MolFromSmarts(Chem.rdFMCS.FindMCS([product, self.nu]).smartsString), uniquify=False)
    
                        real_matches = []
                        for match in matches:
                            if set(match).issubset(added_atoms):
                                real_matches.append(match)
                        if len(real_matches) == 0:
                            continue
                        nu_atom_candidates = []
                        nu_atom_charges = []
                        nu_atom_hs = []
                        added_heavy_neighbors = product.GetAtomWithIdx(added_atom).GetTotalDegree() - product.GetAtomWithIdx(added_atom).GetTotalNumHs(includeNeighbors=True)
                        for match in nu_matches:
                            for real_match in real_matches:
                                nu_atom = match[real_match.index(added_atom)]
                                nu_charge = self.nu.GetAtomWithIdx(nu_atom).GetFormalCharge()
                                nu_hs = self.nu.GetAtomWithIdx(nu_atom).GetTotalNumHs()
                                nu_heavy_neighbors = self.nu.GetAtomWithIdx(nu_atom).GetTotalDegree() - self.nu.GetAtomWithIdx(nu_atom).GetTotalNumHs(includeNeighbors=True)
                                if nu_heavy_neighbors == added_heavy_neighbors - 1:
                                    nu_atom_candidates.append(nu_atom)
                                    nu_atom_charges.append(nu_charge)
                                    nu_atom_hs.append(nu_hs)
                        # Pick atom with most negative charge if tie
                        if len(np.unique(nu_atom_charges)) > 1:
                            nu_atom = nu_atom_candidates[np.argmin(nu_atom_charges)]
                        # Otherwise pick the first candidate
                        else:
                            nu_atom = nu_atom_candidates[0]
                        # Take out the reactive tautomer 
                        atom = self.nu.GetAtomWithIdx(nu_atom)
                        nu_symbol = atom.GetSymbol()
                        nu_charge = atom.GetFormalCharge()
                        nu_rings = self._get_aromatic_rings(self.nu, merge=False)
                        n_n_h = 0
                        n_n = 0
                        reactive_rings = [ring for ring in nu_rings if nu_atom in ring]
                        for ring in reactive_rings:
                            for i in ring:
                                atom = self.nu.GetAtomWithIdx(i)
                                symbol = atom.GetSymbol()
                                if symbol == "N":
                                    n_n += 1 
                                    n_h = atom.GetTotalNumHs()
                                    n_n_h += 1
                        
                        if nu_symbol == "N" and nu_charge == 0 and len(reactive_rings) > 0 and n_n > 1 and n_n_h > 0:
                            n_hs = sum([self.nu.GetAtomWithIdx(i).GetTotalNumHs() for i in reactive_rings[0]])
                            n_atoms = self.nu.GetNumAtoms()

                            # Enumerate tautomers 
                            enumerator = TautomerEnumerator()
                            tautomers = enumerator.Enumerate(self.nu)

                            # Add additional tautomers in case of symmetry
                            test_mol = Chem.Mol(self.nu)
                            for atom in test_mol.GetAtoms():
                                atom.SetNumExplicitHs(0)
                            symms = np.array(Chem.CanonicalRankAtoms(test_mol, breakTies=False))
                            symm_nu_atom = symms[nu_atom]
                            same_symm_indices = [int(i) for i in np.where(symms == symm_nu_atom)[0] if i != nu_atom]
                            for i in same_symm_indices:
                                new_tautomer = Chem.Mol(self.nu)
                                atom_1 = new_tautomer.GetAtomWithIdx(i)
                                atom_2 = new_tautomer.GetAtomWithIdx(nu_atom)
                                n_h_1 = atom_1.GetNumExplicitHs()
                                n_h_2 = atom_2.GetNumExplicitHs()
                                if n_h_1 > n_h_2:
                                    atom_1.SetNumExplicitHs(n_h_1 - 1)
                                    atom_2.SetNumExplicitHs(n_h_2 + 1)
                                else:
                                    atom_1.SetNumExplicitHs(n_h_1 + 1)
                                    atom_2.SetNumExplicitHs(n_h_2 - 1)
                                tautomers.append(new_tautomer)

                            ring_tautomers = []
                            for tautomer in tautomers:
                                mcs_result = Chem.rdFMCS.FindMCS([self.nu, tautomer])
                                if mcs_result.numAtoms == n_atoms:
                                    ring_tautomers.append(tautomer)
                            n_h_atoms = []
                            new_nu_atoms = []
                            new_tautomers = []
                            for tautomer in ring_tautomers:
                                all_indices = tautomer.GetSubstructMatches(self.nu)
                                for indices in all_indices:
                                    all_new_nu_atoms = []
                                    all_n_h_atoms = []
                                    if len(indices) == n_atoms:
                                        new_nu_atom = indices.index(nu_atom)
                                        n_h = tautomer.GetAtomWithIdx(new_nu_atom).GetTotalNumHs()
                                        tautomer_rings = self._get_aromatic_rings(tautomer, merge=False)
                                        tautomer_reactive_rings = [ring for ring in tautomer_rings if new_nu_atom in ring]
                                        tautomer_n_hs =  sum([self.nu.GetAtomWithIdx(i).GetTotalNumHs() for i in tautomer_reactive_rings[0]])
                                        if n_hs == tautomer_n_hs and tautomer.GetAtomWithIdx(new_nu_atom).GetIsAromatic():
                                            all_new_nu_atoms.append(new_nu_atom)
                                            all_n_h_atoms.append(n_h)
                                    if len(all_new_nu_atoms) > 0:
                                        new_nu_atoms.append(all_new_nu_atoms[np.argmin(all_n_h_atoms)])
                                        n_h_atoms.append(all_n_h_atoms[np.argmin(all_n_h_atoms)])
                                        new_tautomers.append(tautomer)
                            self.nu = new_tautomers[np.argmin(n_h_atoms)]
                            self.nu_tautomers = new_tautomers
                            nu_atom = new_nu_atoms[np.argmin(n_h_atoms)]
                        self.nu_atom.append(nu_atom)

                    self.added_atom.append(added_atom)
                    self.nu_atoms = [atom.GetIdx() for atom in self.nu.GetAtoms()]

    def _create_lgs(self):
        for core_atoms, product in zip(self.core_atoms, self.products):
            # Find atom in LG connected to core
            remove_indices = core_atoms
            remaining_indices = [atom.GetIdx() for atom in self.substrate.GetAtoms() if atom.GetIdx() not in remove_indices]
            connected_index = None
            for i in remaining_indices:
                neighbors = list(self.substrate.GetAtomWithIdx(i).GetNeighbors())
                for neighbor in neighbors:
                    if neighbor.GetIdx() in remove_indices:
                        connected_index = i
    
            self.lg_atoms.append(remaining_indices)
            
            # Create LG mol object                    
            remove_indices.sort(reverse=True)
            rw_mol = Chem.RWMol(self.substrate)
            for i in remove_indices:
                if i < connected_index:
                    connected_index -= 1
                rw_mol.RemoveAtom(i)
            lg = rw_mol.GetMol()
            Chem.SanitizeMol(lg)
            if len(Chem.GetMolFrags(lg)) > 1:
                raise Exception(f"Several leaving groups: {Chem.MolToSmiles(lg)}")
           
            # Take out connected atom
            connected_atom = lg.GetAtomWithIdx(connected_index)
            if connected_atom.GetFormalCharge() == 1:
                connected_atom.SetFormalCharge(0)
                connected_atom.UpdatePropertyCache()
            self.lg_atom_orig.append(connected_atom.GetIdx())
            
            # Set the right charge and H atoms
            left_hs = sum([atom.GetTotalNumHs(includeNeighbors=True) for atom in self.substrate.GetAtoms()])
            if self.nu:
                left_hs += sum([atom.GetTotalNumHs(includeNeighbors=True) for atom in self.nu.GetAtoms()])
            right_hs = sum([atom.GetTotalNumHs(includeNeighbors=True) for atom in product.GetAtoms()])
            right_hs += sum([atom.GetTotalNumHs(includeNeighbors=True) for atom in lg.GetAtoms()])
            if right_hs > left_hs:
                connected_atom.SetFormalCharge(-1)
                connected_atom.UpdatePropertyCache()
            elif right_hs < left_hs:
                # Check charge on atom and adjust to protonate
                if connected_atom.GetFormalCharge() == -1:
                    connected_atom.SetFormalCharge(0)
                    connected_atom.SetNumExplicitHs(connected_atom.GetNumExplicitHs() + 1)
                    connected_atom.UpdatePropertyCache()
                else:
                    # Check charges on neighbors and adjust to protonate
                    neighbors = connected_atom.GetNeighbors()
                    for neighbor in neighbors:
                        if neighbor.GetFormalCharge() == -1:
                            neighbor.SetFormalCharge(0)
                            neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
                            neighbor.UpdatePropertyCache()
                            break 
    
            Chem.SanitizeMol(lg)
            self.lgs.append(lg)
    
    def _cleanup_reaction(self):
        # Sanitize
        for mol in self.reaction.GetReactants():
            Chem.SanitizeMol(mol)
        for mol in self.reaction.GetProducts():
            Chem.SanitizeMol(mol)
        for mol in self.reaction.GetAgents():
            Chem.SanitizeMol(mol)

        # Remove metals
        reactants = []
        for mol in self.reaction.GetReactants():
            cleaned_mol = self._remove_metals(mol)
            if cleaned_mol:
                reactants.append(cleaned_mol)

        products = []
        for mol in self.reaction.GetProducts():
            cleaned_mol = self._remove_metals(mol)
            if cleaned_mol:
                products.append(cleaned_mol)

        agents = []
        for mol in self.reaction.GetAgents():
            cleaned_mol = self._remove_metals(mol)
            if cleaned_mol:
                agents.append(cleaned_mol)                

        # Remove identical species in each series
        for mols in [reactants, products, agents]:
            identicals = {mol: set() for mol in mols}
            for mol_1, mol_2 in itertools.combinations(mols, 2):
                if mol_1.GetNumAtoms() == mol_2.GetNumAtoms():
                    if len(mol_1.GetSubstructMatch(mol_2)) == mol_1.GetNumAtoms():
                        identicals[mol_1].add(mol_2)
                        identicals[mol_2].add(mol_1)
            
            keep = set()
            remove = set()
            for mol, identical_mols in identicals.items():
                if mol not in remove:
                    keep.add(mol)
                for identical_mol in identical_mols:
                    if identical_mol not in keep:
                        remove.add(identical_mol)

            for mol in remove:
                mols.remove(mol)
        
        # Remove agents which are in the reactants
        remove = []
        for mol in agents:
            for reactant_mol in reactants:
                if mol.GetNumAtoms() == reactant_mol.GetNumAtoms():
                    if len(mol.GetSubstructMatch(reactant_mol)) == mol.GetNumAtoms():
                        remove.append(mol)

        for mol in remove:
            agents.remove(mol)        

        # Construct cleaned up reaction                        
        cleaned_reaction = AllChem.ChemicalReaction()
        for mol in reactants:
            if mol:
                cleaned_reaction.AddReactantTemplate(mol)

        for mol in products:
            if mol:
                cleaned_reaction.AddProductTemplate(mol)             
        
        for mol in agents:
            if mol:
                cleaned_reaction.AddAgentTemplate(mol)
        
        self.reaction = cleaned_reaction

        # Process reaction smiles and clean up components
        for mol in self.reaction.GetReactants():
            Chem.SanitizeMol(mol)
        for mol in self.reaction.GetProducts():
            Chem.SanitizeMol(mol)
        for mol in self.reaction.GetAgents():
            Chem.SanitizeMol(mol)
       
    @staticmethod
    def _get_aromatic_rings(mol, merge=True):
        # Identify aromatic rings in the molecule.
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        rings = mol.GetRingInfo().AtomRings()
        aromatic_rings = []
        for ring in rings:
            aromaticities = [mol.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring]
            if all(aromaticities):
                aromatic_rings.append(list(ring))
        
        # Merge fused aromatic rings into one
        if merge:
            merged_rings = []
            while len(aromatic_rings) > 0:
                merged = False
                test_ring = aromatic_rings.pop(0)
                for ring in aromatic_rings:
                    if len(set(test_ring).intersection(ring)) > 1:
                        aromatic_rings.remove(ring)
                        new_ring = test_ring + ring
                        aromatic_rings.insert(0, new_ring)
                        merged = True
                if merged:
                    continue
                else:
                    merged_rings.append(test_ring)
    
            # Remove duplicated atoms
            merged_rings = [list(set(ring)) for ring in merged_rings]
            aromatic_rings = merged_rings
                
        return aromatic_rings
    
    @staticmethod
    def _remove_metals(mol):
        # Set up lists of metals and coordinating atoms
        metals = [3, 11, 12, 19, 20, 55]
        coordinating = [7, 8, 16]
        
        # Initialize salt remover
        salt_remover = SaltRemover(defnData="[Li,Na,K,Cs,Ca,Mg]")
        
        mol = Chem.Mol(mol)
        bonds = mol.GetBonds()
        
        # Fragment any metal bonds
        fragment_indices = []
        for bond in bonds:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if (begin_atom.GetAtomicNum() in metals and end_atom.GetAtomicNum() in coordinating):
                begin_atom.SetFormalCharge(1)
                end_atom.SetFormalCharge(-1)
                fragment_indices.append(bond.GetIdx())
            elif (end_atom.GetAtomicNum() in metals and begin_atom.GetAtomicNum() in coordinating):
                begin_atom.SetFormalCharge(-1)
                end_atom.SetFormalCharge(1)
                fragment_indices.append(bond.GetIdx())            
        if len(fragment_indices) > 0:
            mol = Chem.FragmentOnBonds(mol, fragment_indices, addDummies=False)
        
        # Remove metal salts
        mol = salt_remover.StripMol(mol)
        
        if mol.GetNumAtoms() > 0:
            return mol
        else:
            return None

    @staticmethod
    def delete_atoms(mol, indices):
        # Sort indices so that atoms are removed in descending order
        remove_indices = list(indices)
        remove_indices.sort(reverse=True)

        # Remove atoms 
        rw_mol = Chem.RWMol(mol)
        for k in remove_indices:
            rw_mol.RemoveAtom(k)
        new_mol = core_pattern_product = rw_mol.GetMol()

        return new_mol


class AgentDetector:
    """Detect acid/base agents from SMILES or RDKit Mol file.

    Attributes:
        active_atoms (dict): Index of active atom for each molecule
        mols (dict): RDKit mol objects for matches in each category
        names (dict): Type of agent for each molecule
        pkas (dict): pKa values of each molecule
        smiles (dict): SMILES for matches in each category
    """
    # List of different acids and bases.
    _bronstedt_categories = ["strong_bases", "weak_bases", "strong_acids", "weak_acids"]
    _lewis_categories = ["lewis_acids"]
    
    def __init__(self):
        # Set up attributes
        self.smiles = {}
        self.mols = {}
        self.active_atoms = {}
        self.pkas = {}
        self.names = {}
        for d in [self.smiles, self.mols, self.active_atoms, self.names, self.pkas]:
            for category in self._bronstedt_categories + self._lewis_categories:
                d[category] = []
        
        # Load in data and set up SMARTS pattern.
        base_path = Path(__file__).parent
        agent_path = base_path / "../data/agents/"

        patterns = {}
        for i, category in enumerate(self._bronstedt_categories + self._lewis_categories):
            with open(agent_path / f"{category}.pickle", "rb") as file:
                patterns[category] = pickle.load(file)
            for value in patterns[category].values():
                value["pattern"] = Chem.MolFromSmarts(value["smarts"])
        self._patterns = patterns
    
    def process_smiles(self, smiles):
        """Process a SMILES to detect agents.

        Args:
            smiles (str): SMILES of potential agent.

        Returns:
            matched (bool): Whether the SMILES was matched as an agent. 
        """
        # Create mol object and match.
        mol = Chem.MolFromSmiles(smiles)
        matched = self.process_mol(mol)

        return matched
    
    def process_mol(self, mol):
        """Process a RDKit mol object to detect agents.

        Args:
            mol (str): RDKit mol object of potential agent.

        Returns:
            matched (bool): Whether the mol object was matched as an agent. 
        """        
        matched = False
        # Loop over categories and match SMARTS patterns.
        for matching in self._bronstedt_categories + self._lewis_categories:
            for value in self._patterns[matching].values():
                pattern = value["pattern"]
                match = mol.GetSubstructMatch(pattern)
                if match:
                    matched = True
                    # Add to dictionaries.
                    self.mols[matching].append((mol))
                    self.smiles[matching].append((Chem.MolToSmiles(mol)))
                    self.active_atoms[matching].append(match[0])
                    self.names[matching].append(value["name"])
                    if "pka" in value:
                        self.pkas[matching].append(value["pka"])
                    else:
                        self.pkas[matching].append(None)
        return matched
          