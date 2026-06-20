from functools import reduce
import logging
from operator import add
import os
import re
import shutil
import subprocess
from textwrap import dedent

#TODO change to scipy units.
from ase.units import eV, Hartree, mol, kcal
import cclib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from predict_snar import config, results
from predict_snar.calculators import G16Calculator
from predict_snar.data import get_ecp
from predict_snar.helpers import cd, calculation_monitor
from predict_snar.parsers import ChargeMolParser, EPNParser, HirshfeldParser, BrinckParser
from steriplus import SASA, Dispersion

logger = logging.getLogger("predict_snar")

class RFPCalculator:
    # TODO refactor code to eliminate repetitions.
    """Calculator for reaction fingerprint from reaction SMILES.

    Args:
        reaction_smiles (str): Reaction smiles of the reaction of interest

    Attributes:
        reaction (object): Reaction as RDkit reaction object
        reactants (object): Reactants as RDKit mol objects
        products (object): Products as RDKit mol objects
    """
    def __init__(self, reaction_smiles):
        # Set up reaction and sanitize
        reaction = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        for mol in reaction.GetReactants():
            Chem.SanitizeMol(mol)
        for mol in reaction.GetProducts():
            Chem.SanitizeMol(mol)

        # Set up attributes
        self.reactants = reaction.GetReactants()
        self.products = reaction.GetProducts()
        self.reaction = reaction

    def get_morgan(self, radius=2, size=2048, vector_type="bit", all_positive=False, use_features=False, extend_negative=False):
        """Gets the Morgan fingerprint. 

        Args:
            radius (int): Radius of Morgan fingerprint
            size (int): Size of final fingerprint vector
            vector_type (str): Type of vector, either "count" or "bit"
            all_positive (bool): Whether to take all values as positive.
            use_features (bool): Whether to use chemical features
            extend_negative (bool): Whether to treate negative values by a
                fineprint of twice the length with only positive values.

        Returns:
            reaction_fp (ndarray): Reaction fingerprint
        """
        # Calculate reaction fingerprint.
        reactant_fps = [AllChem.GetMorganFingerprint(reactant, radius=radius, useFeatures=use_features) for reactant in self.reactants]
        product_fps = [AllChem.GetMorganFingerprint(product, radius=radius, useFeatures=use_features) for product in self.products]
        reaction_fp = reduce(add, product_fps) - reduce(add, reactant_fps) 

        # Fold the fingerprint to specified size
        reaction_fp = self._fold_sparse(reaction_fp, size=size)
        
        # Convert to bit vector
        if vector_type == "bit":
            reaction_fp = self._count_to_bit(reaction_fp)
        
        # Convert to only positive numbers
        if all_positive:
            reaction_fp = np.abs(reaction_fp)
        
        # Extend fingerprint to treat negative values
        if extend_negative:
            reaction_fp = self._extend_negative(reaction_fp)            

        return reaction_fp

    def get_atom_pair(self, max_length=30, size=2048, vector_type="bit", all_positive=False, extend_negative=False):
        """Gets the Atom Pair fingerprint. 

        Args:
            max_length (int): Maximum length between atom pairs.
            size (int): Size of final fingerprint vector
            vector_type (str): Type of vector, either "count" or "bit"
            all_positive (bool): Whether to take all values as positive.
            extend_negative (bool): Whether to treate negative values by a
                fineprint of twice the length with only positive values.

        Returns:
            reaction_fp (ndarray): Reaction fingerprint
        """
        # Calculate reaction fingerprint.
        reactant_fps = [AllChem.GetAtomPairFingerprint(reactant, maxLength=max_length) for reactant in self.reactants]
        product_fps = [AllChem.GetAtomPairFingerprint(product, maxLength=max_length) for product in self.products]
        reaction_fp = reduce(add, product_fps) - reduce(add, reactant_fps)

        # Fold the fingerprint to specified size
        reaction_fp = self._fold_sparse(reaction_fp, size=size)

        # Convert to bit vector
        if vector_type == "bit":
            reaction_fp = self._count_to_bit(reaction_fp)

        # Convert to only positive numbers
        if all_positive:
            reaction_fp = np.abs(reaction_fp)

        # Extend fingerprint to treat negative values
        if extend_negative:
            reaction_fp = self._extend_negative(reaction_fp)

        return reaction_fp

    def get_hashed_atom_pair(self, max_length=30, size=2048, vector_type="bit", all_positive=False, extend_negative=False):
        """Gets the Atom Pair fingerprint. 

        Args:
            max_length (int): Maximum length between atom pairs.
            size (int): Size of final fingerprint vector
            vector_type (str): Type of vector, either "count" or "bit"
            all_positive (bool): Whether to take all values as positive.
            extend_negative (bool): Whether to treate negative values by a
                fineprint of twice the length with only positive values.

        Returns:
            reaction_fp (ndarray): Reaction fingerprint
        """
        # Calculate reaction fingerprint.
        reactant_fps = [AllChem.GetHashedAtomPairFingerprint(reactant, maxLength=max_length, nBits=size) for reactant in self.reactants]
        product_fps = [AllChem.GetHashedAtomPairFingerprint(product, maxLength=max_length, nBits=size) for product in self.products]
        reaction_fp = reduce(add, product_fps) - reduce(add, reactant_fps)

        # Fold the fingerprint to specified size
        reaction_fp = self._sparse_to_dense(reaction_fp)

        # Convert to bit vector
        if vector_type == "bit":
            reaction_fp = self._count_to_bit(reaction_fp)

        # Convert to only positive numbers
        if all_positive:
            reaction_fp = np.abs(reaction_fp)

        # Extend fingerprint to treat negative values
        if extend_negative:
            reaction_fp = self._extend_negative(reaction_fp)            

        return reaction_fp

    @staticmethod
    def _fold_sparse(sparse_vector, size):
        dense_vector = np.zeros(size, dtype=int)
        for index, value in sparse_vector.GetNonzeroElements().items():
            fold_number = index // size
            folded_index = index % (size * fold_number)
            dense_vector[folded_index] += value
            
        return dense_vector
    
    @staticmethod
    def _extend_negative(vector):
        # Make new vector twice the size of the old
        extended_vector = np.zeros(len(vector) * 2)

        # Get the indices of the old and new array.
        positive_indices = np.where(vector > 0)
        old_negative_indices = np.where(vector < 0)
        new_negative_indices = (np.multiply(*old_negative_indices, 2),)

        # Set values of new array.
        extended_vector[positive_indices] = vector[positive_indices]
        extended_vector[new_negative_indices] = np.abs(vector[old_negative_indices])
        
        return extended_vector
    
    @staticmethod
    def _sparse_to_dense(sparse_vector):
        dense_vector = np.zeros(sparse_vector.GetLength(), dtype=int)
        for index, value in sparse_vector.GetNonzeroElements().items():
            dense_vector[index] = value
        
        return dense_vector
    
    @staticmethod
    def _count_to_bit(count_vector):
        bit_vector = np.clip(count_vector, -1, 1)

        return bit_vector

    def __repr__(self):
        return f"{self.__class__.__name__}('{AllChem.ReactionToSmiles(self.reaction)}')"


class ChargemolCalculator:
    """Calculator of charges and bond orders with the Chargemol program.

    Args:
        atomic_densities (str): Directory with atomic densities
        file (str): Name wfx file

    Attributes:
        atomic_densities (str): Directory with atomic densities
        file (str): Name of wfx file
    """
    def __init__(self, file, atomic_densities):
        self.file = file
        self.atomic_densities = atomic_densities

    def run(self, n_procs):
        """Run Chargemol
        Args:
            n_procs (int): Number of processors.

        Returns:
            process (object): Running process.
        """
        prefix = os.path.splitext(self.file)[0]
        
        # Set up environment with the right number of processors
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(n_procs)

        # Start calculation.
        process = subprocess.Popen(f"{str(config.chargemol)} &> {prefix}.out &".split(), shell=True, env=env)

        # Return running calculation.
        return process

    def write_control_file(self):
        """Write the control file for ChargeMol."""
        string = dedent(
            f"""\
            <atomic densities directory complete path>
            {self.atomic_densities}
            </atomic densities directory complete path>

            <input filename>
            {self.file}
            </input filename>

            <charge type>
            DDEC6
            </charge type>

            <compute BOs>
            .true.
            </compute BOs>
            """)
        with open("job_control.txt", "w") as file:
            file.write(string)


class BrinckCalculator:
    """Calculator of surface properties with Brinck's HS95 program.

    Args:
        isosurface_value (float): Isosurface value
        file (str): Name 

    Attributes:
        atomic_densities (str)   :   Directory with atomic densities
        file (str)               :   Name wfx file
    """
    def __init__(self, file, isosurface_value):
        # Set up file name
        self.file = file
        prefix = os.path.splitext(file)[0]
        self.outfile = prefix + "_hs.out"
        
        # Set value of isosurface.
        if not (isosurface_value == 0.004 or isosurface_value == 0.001):
            raise Exception("The allowed values are 0.001 (suitable for Is_min) or 0.004 (suitable for Es_max)")
        self.isosurface_value = isosurface_value

    def run(self):
        """Run the HS95 program.
        
        Returns:
            process (object): Running process.
        """
        # Choose the appropriate script file.
        if self.isosurface_value == 0.004:
            cmd = config.hs95_4 
        elif self.isosurface_value == 0.001:
            cmd = config.hs95_1
        # Run the program
        process = subprocess.Popen(cmd + " " + self.file, shell=True)
        return process


class DescriptorCalculator:
    # TODO change ESmin and ISmin to av
    """Calculate descriptors of different kinds
    Args:
        atoms (object): ASE Atoms object
        dft_options (dict): General options for dft calculations
        descriptor_dft_options (dict): Special options for descriptor calculations.
        atomic_densities (str): Path to the atomic densities directory.
        file (str): Name of input file.
    
    Attributes:
        atoms (object): ASE Atoms object
        dft_options (dict): Options for dft calculations
        descriptor_dft_options (dict): Options for descriptor calculations.
        atomic_densities (str): Path to the atomic densities directory.
        file (str): Name of input file.
        prefix (str): Prefix of the file
        epn_parser (object): Parser for the EPN descriptor
        hirshfeld_parser (object): Parser for the Hirshfeld charges
        chargemol_parser (object): Parser for the ChargeMol charges
        brinck_parsers (dict): Parsers for the Brinck surface descriptors
    """
    def __init__(self, atoms, dft_options, descriptor_dft_options, atomic_densities=None, file="descriptors.gjf"):
        self.atoms = atoms
        self.file = file
        self.prefix = os.path.splitext(self.file)[0]
        self.epn_parser = None
        self.hirshfeld_parser = None
        self.chargemol_parser = None
        self.brinck_parsers = {}
        self.dft_options = dft_options
        self.descriptor_dft_options = descriptor_dft_options
        self.atomic_densities = atomic_densities

    def run_dft(self, n_procs, mem):
        """Run DFT calculation as basis for descriptors.
        
        Args:
            n_procs (int): Number of processors
            mem (float): Memory (GB)
        """
        # Set up calculator options
        g16 = G16Calculator(self.atoms, self.file, self.dft_options)
        g16.set_options(self.descriptor_dft_options)
        g16.options["wfx"] = True
        g16.options["brinck"] = True
        g16.options["hirshfeld"] = True
        g16.options["epn"] = True
        g16.options["chk"] = False
        g16.options["xqc"] = True

        # Run calculation
        g16.single_point(n_procs=n_procs, mem=mem)
        calculation_monitor(g16)
        
        # Test if ECPs are used and modify output file in that case
        symbol_list = [atom.symbol for atom in self.atoms]
        ecp_dict = get_ecp(self.dft_options['ecp'])
        ecp_list = [symbol.capitalize() in ecp_dict.keys() for symbol in symbol_list]
        if any(ecp_list):
            shutil.move(g16.output, "old_" + g16.output)
            modify_log_ecp("old_" + g16.output, g16.output)

    def get_epn(self, atom):
        """Get the EPN value of an atom.

        Args:
            atom (int): Index of atom (1-indexed)

        Returns:
            epn (float): EPN value
        """
        if not self.epn_parser:
            self.epn_parser = EPNParser(self.prefix + ".log")
        epn = self.epn_parser.get_potential(atom)

        return epn

    def get_hirshfeld(self, atom):
        """Get the Hirshfeld charge of an atom.
        
        Args:
            atom (int): Index of atom (1-indexed)

        Returns:
            hirshfeld_charge (float): Hirshfeld charge
        """
        if not self.hirshfeld_parser:
            self.hirshfeld_parser = HirshfeldParser(self.prefix + ".log")
        hirshfeld_charge = self.hirshfeld_parser.get_charge(atom)

        return hirshfeld_charge

    def get_ddec6_charge(self, atom):
        """Get the DDEC6 charge of an atom.
        
        Args:
            atom (int): Index of atom (1-indexed)

        Returns:
            ddec6_charge (float): DDEC6 charge
        """
        # Set up and run ChargeMol if it's not done.
        if not self.chargemol_parser:
            self.setup_charge_mol()
        ddec6_charge = self.chargemol_parser.get_charge(atom)

        return ddec6_charge

    def setup_charge_mol(self):
        """Set up and run ChargeMol"""
        # Set up ChargeMol calculation
        c_mol = ChargemolCalculator(self.prefix + ".wfx", self.atomic_densities)
        c_mol.write_control_file()

        # Run calculation
        c_mol.run(config.n_procs).wait()

        # Parse results
        self.chargemol_parser = ChargeMolParser(self.prefix + ".output")

    def setup_brinck(self, isosurface_value):
        """Set up and run HS95 program.

        Args:
            isosurface_value (float): Isosurface value.
        """
        # Set up and run HS95 calculation.
        brinck_calc = BrinckCalculator(self.prefix + ".log", isosurface_value)
        brinck_calc.run().wait()

        # Parse results and clean up files.
        self.brinck_parsers[isosurface_value] = BrinckParser(brinck_calc.outfile)
        shutil.move(brinck_calc.outfile, f"hs95_{isosurface_value}")

    def get_ddec6_bo(self, atom_1, atom_2):
        """Get DDEC6 bond order between two atoms.
        
        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)

        Returns:
            ddec6_bo (float): Bond order
        """
        # Set up and run ChargeMol if not already done
        if not self.chargemol_parser:
            self.setup_charge_mol()

        # Get and return bond order.
        ddec6_bo = self.chargemol_parser.get_bo(atom_1, atom_2)

        return ddec6_bo

    def get_es_min(self, atom):
        """Get Es_min value of an atom
        
        Args:
            atom (int): Index of atom (1-indexed)
        
        Returns:
            es_min (float): Es_min value
        """
        # Set up and run HS95 calculation if not already done.
        if not self.brinck_parsers.get(0.004):
            self.setup_brinck(0.004)

        # Parse results
        es_min = self.brinck_parsers[0.004].es_min_all_dict.get(atom)

        return es_min

    def get_is_min(self, atom):
        """Get Is_min value of an atom
        
        Args:
            atom (int): Index of atom (1-indexed)
        
        Returns:
            is_min (float): Is_min value
        """       
        # Set up and run HS95 calculation if not already done.         
        if not self.brinck_parsers.get(0.001):
            self.setup_brinck(0.001)

        # Parse results            
        is_min = self.brinck_parsers[0.001].is_min_all_dict.get(atom)

        return is_min

    def get_v_av(self, atom):
        """Get V_av value of an atom
        
        Args:
            atom (int): Index of atom (1-indexed)
        
        Returns:
            v_av (float): V_av value
        """
        # Set up and run HS95 calculation if not already done.         
        if not self.brinck_parsers.get(0.001):
            self.setup_brinck(0.001)

        # Parse results            
        v_av = self.brinck_parsers[0.001].v_av_dict.get(atom)

        return v_av

    def get_vs_min(self, atom):
        """Get Vs_min value of an atom
        
        Args:
            atom (int): Index of atom (1-indexed)
        
        Returns:
            vs_min (float): Vs_min value
        """    
        # Set up and run HS95 calculation if not already done.         
        if not self.brinck_parsers.get(0.001):
            self.setup_brinck(0.001)

        # Parse results            
        vs_min = self.brinck_parsers[0.001].vs_min_all_dict.get(atom)

        return vs_min

    def get_vs_max(self, atom):
        """Get Vs_max value of an atom
        
        Args:
            atom (int): Index of atom (1-indexed)
        
        Returns:
            vs_max (float): Vs_max value
        """          
        # Set up and run HS95 calculation if not already done.         
        if not self.brinck_parsers.get(0.001):
            self.setup_brinck(0.001)

        # Parse results            
        vs_max = self.brinck_parsers[0.001].vs_max_all_dict.get(atom)

        return vs_max

    def get_electronic_energy(self):
        """Get the electronic energy of the DFT calculation.

        Returns:
            energy (float): Electronic energy (a.u.)
        """
        outfile = self.prefix + ".log"
        data = cclib.io.ccread(outfile)
        energy = data.scfenergies[-1] * eV / Hartree
        return energy


def get_SASA(atoms, atom_indices=None, radii=None):
    """Returns the solvent accesible surface area (SASA) of the whole molecule
    and a subset of atoms. If no atoms are specified, all atom areas are
    returned.

    Args:
        atoms (object): ASE Atoms object
        atom_indices (list): Atom indices (1-indexed)
        radii (list): Atomic vdW radii (Å)
    
    Returns:
        total_area (float): Total SASA (Å^2)
        atom_areas (list): Atom areas (Å^2)
        atom_ratios (list): Ratio of accessible to occluded area.
    """
    # Set up list of atom indices. Defaults to all atoms.
    if not atom_indices:
        atom_indices = range(1, atoms.get_number_of_atoms() + 1)

    # Get elements and coordinates
    elements = atoms.get_chemical_symbols()
    coordinates = atoms.get_positions()
    
    # Calculate SASA
    sasa = SASA(elements, coordinates, radii=radii, radii_type="crc")

    total_area = sasa.area
    atom_areas = sasa.atom_areas

    # Take out atom areas. 
    atom_areas = [atom_areas[i] for i in atom_indices]

    # Take out atom ratios.
    atom_ratios = []
    for i in atom_indices:
        atom = sasa._atoms[i - 1]
        ratio = len(atom.accessible_points) / (len(atom.accessible_points) + len(atom.occluded_points))
        atom_ratios.append(ratio)

    return total_area, atom_areas, atom_ratios


def get_dispersion(atoms, atom_indices=None):
    """Returns dispersion descriptors.

    Args:
        atoms (object): ASE Atoms object
        atom_indices (list): Atom indices (1-indexed)
    
    Returns:
        atom_p_ints (list): Atom P_int ((kcal/mol)^(1/2))
        atom_p_ints_areas (list): Atom P_int multiplied by atom area 
                                  ((kcal/mol)^(1/2) * Å^2)
        p_int (float): Total P_int ((kcal/mol)^(1/2))
        p_int_area (float): Total P_int multiplied by total area
                            ((kcal/mol)^(1/2) * Å^2)
    """
    if not atom_indices:
        atom_indices = range(1, atoms.get_number_of_atoms() + 1)

    # Get elements and coordinates
    elements = atoms.get_chemical_symbols()
    coordinates = atoms.get_positions()
    
    # Calculate Dispersion
    dispersion = Dispersion(elements, coordinates)

    area = dispersion.area
    atom_areas = dispersion.atom_areas
    p_int = dispersion.p_int
    all_atom_p_ints = dispersion.atom_p_ints

    # Calculate p_int * area
    p_int_area = p_int * area

    # Calculate atom_p_int * atom_area
    atom_p_ints = [all_atom_p_ints[i] for i in atom_indices]
    atom_p_ints_areas = [all_atom_p_ints[i] * atom_areas[i] for 
                         i in atom_indices]

    return p_int, atom_p_ints, p_int_area, atom_p_ints_areas


def calculate_descriptors():
    # Set up reactive atoms
    central_atom = config.reactive_atoms["central_atom"]
    nu_atom = config.reactive_atoms["nu_atom"]
    lg_atom = config.reactive_atoms["lg_atom"]
    lg_atom_orig = config.reactive_atoms["lg_atom_orig"]
    central_atom_prod = config.reactive_atoms["central_atom_prod"]
    if not config.intramolecular:
        nu_atom_orig = config.reactive_atoms["nu_atom_orig"]
    agent_atom_orig = config.reactive_atoms["agent_atom_orig"]
    added_atom = config.reactive_atoms["added_atom"]

    # Take out structures from DFT optimizations
    substrate = results.dft_atoms["substrate"]
    product = results.dft_atoms["product"]
    leaving_group = results.dft_atoms["leaving_group"]
    if not config.intramolecular:
        nucleophile = results.dft_atoms["nucleophile"]
    else:
        nucleophile = None
    ts_list = results.dft_atoms["ts"]
    if config.agent:
        agent = results.dft_atoms["agent"]
    
    # Calculate descriptors for substrate.
    if substrate:
        os.mkdir("substrate")
        with cd("substrate"):
            logger.info("Calculating descriptors for substrate...")

            # Calculate descriptors
            calculator = DescriptorCalculator(substrate, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)

            # Set up results dictionary.
            substrate_dict = {"epn": {},
                             "hirshfeld": {},
                             "hirshfeld_plus": {},
                             "hirshfeld_minus": {},
                             "ddec6_charge": {},
                             "ddec6_bo": {},
                             "v_av": {},
                             "vs_max": {},
                             "es_min_b3lyp": {},
                             "es_min_blyp": {},
                             "sasa": {},
                             "sasa_ratio": {},
                             "ip": None,
                             "ea": None,
                             "atom_p_int": {},
                             "atom_p_int_area": {},
                             }
            
            # Get descriptors.
            for atom in [central_atom, lg_atom]:
                substrate_dict["epn"][atom] = calculator.get_epn(atom)
                substrate_dict["hirshfeld"][atom] = calculator.get_hirshfeld(atom)
                substrate_dict["ddec6_charge"][atom] = calculator.get_ddec6_charge(atom)
                substrate_dict["vs_max"][atom] = calculator.get_vs_max(atom)
                substrate_dict["v_av"][atom] = calculator.get_v_av(atom)

            _, sasa, ratio = get_SASA(substrate, [central_atom])
            substrate_dict["sasa"][central_atom] = sasa[0]
            substrate_dict["sasa_ratio"][central_atom] = ratio[0]
            
            p_int, atom_p_ints, p_int_area, atom_p_ints_areas = get_dispersion(substrate, [central_atom])
            substrate_dict["atom_p_int"][central_atom] = atom_p_ints[0]
            substrate_dict["atom_p_int_area"][central_atom] = atom_p_ints_areas[0]
            substrate_dict["p_int"] = p_int
            substrate_dict["p_int_area"] = p_int_area
            
            substrate_dict["es_min_b3lyp"][central_atom] = calculator.get_es_min(central_atom)
            substrate_dict["ddec6_bo"][(central_atom, lg_atom)] = calculator.get_ddec6_bo(central_atom, lg_atom)

            energy_neutral = calculator.get_electronic_energy()

            # In case of intramolecular reaction, nucleophile is part of 
            # substrate and its descriptors are calculated here
            if config.intramolecular:
                # Set up results dictionary.
                nu_dict = {"epn": {},
                           "hirshfeld": {},
                           "hirshfeld_plus": {},
                           "hirshfeld_minus": {},
                           "ddec6_charge": {},
                           "vs_min": {},
                           "v_av": {},
                           "is_min": {},
                           "sasa": {},
                           "sasa_ratio": {},
                           "hirshfeld_plus": {},
                           "hirshfeld_minus": {},
                           "atom_p_int": {},
                           "atom_p_int_area": {},
                           "ip": None,
                           "ea": None,
                           }            

                # Get descriptor results
                nu_dict["epn"][nu_atom] = calculator.get_epn(nu_atom)
                nu_dict["hirshfeld"][nu_atom] = calculator.get_hirshfeld(nu_atom)
                nu_dict["ddec6_charge"][nu_atom] = calculator.get_ddec6_charge(nu_atom)
                nu_dict["vs_min"][atom] = calculator.get_vs_min(atom)
                nu_dict["v_av"][nu_atom] = calculator.get_v_av(nu_atom)
                nu_dict["is_min"][nu_atom] = calculator.get_is_min(nu_atom)
    
                _, sasa, ratio = get_SASA(substrate, [nu_atom])
                nu_dict["sasa"][nu_atom] = sasa[0]
                nu_dict["sasa_ratio"][nu_atom] = ratio[0]
    
                p_int, atom_p_ints, p_int_area, atom_p_ints_areas = get_dispersion(substrate, [nu_atom])
                nu_dict["atom_p_int"][nu_atom] = atom_p_ints[0]
                nu_dict["atom_p_int_area"][nu_atom] = atom_p_ints_areas[0]
                nu_dict["p_int"] = p_int
                nu_dict["p_int_area"] = p_int_area

            # Calculate Es_min also with BLYP to make sure to get an Es_min value
            descriptor_dft_options = config.descriptor_dft_options.copy()
            descriptor_dft_options["functional"] = "blyp"
            calculator = DescriptorCalculator(substrate, config.dft_options, descriptor_dft_options, config.directories["atomic_densities"], file="blyp.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            substrate_dict["es_min_blyp"][central_atom] = calculator.get_es_min(central_atom)

            # Calculate cation and anion to get the Fukui indices.
            substrate_plus = substrate.copy()
            substrate_plus.info["charge"] += 1
            calculator = DescriptorCalculator(substrate_plus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_plus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            substrate_dict["hirshfeld_plus"][central_atom] = calculator.get_hirshfeld(central_atom)
            energy_plus = calculator.get_electronic_energy()

            if config.intramolecular:
                nu_dict["hirshfeld_plus"][nu_atom] = calculator.get_hirshfeld(nu_atom)

            substrate_minus = substrate.copy()
            substrate_minus.info["charge"] -= 1
            calculator = DescriptorCalculator(substrate_minus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_minus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            substrate_dict["hirshfeld_minus"][central_atom] = calculator.get_hirshfeld(central_atom)
            energy_minus = calculator.get_electronic_energy()

            # Calculate ionizaation potential and electron affinity.
            ea = (energy_minus - energy_neutral) / eV * Hartree
            ip = (energy_plus - energy_neutral) / eV * Hartree
            substrate_dict["ea"] = ea
            substrate_dict["ip"] = ip

            if config.intramolecular:
                nu_dict["hirshfeld_minus"][nu_atom] = calculator.get_hirshfeld(nu_atom)
                nu_dict["ea"] = ea
                nu_dict["ip"] = ip

            logger.info("...substrate descriptors completed.")
    else:
        substrate_dict = {}

    if product:
        os.mkdir("product")
        with cd("product"):
            logger.info("Calculating descriptors for product...")
            # Set up and run descriptor calculation.
            calculator = DescriptorCalculator(product, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)

            # Set up results dictionary.
            product_dict = {"epn": {},
                            "hirshfeld": {},
                            "ddec6_charge": {},
                            "ddec6_bo": {},
                            "v_av": {},
                            "vs_max": {},
                            }
            # Get descriptor results.
            for atom in [central_atom_prod, added_atom]:
                product_dict["epn"][atom] = calculator.get_epn(atom)
                product_dict["hirshfeld"][atom] = calculator.get_hirshfeld(atom)
                product_dict["ddec6_charge"][atom] = calculator.get_ddec6_charge(atom)
                product_dict["v_av"][atom] = calculator.get_v_av(atom)
                product_dict["vs_max"][atom] = calculator.get_vs_max(atom)

            product_dict["ddec6_bo"][(central_atom_prod, added_atom)] = calculator.get_ddec6_bo(central_atom_prod, added_atom)
            logger.info("...product descriptors completed.")
    else:
        product_dict = {}

    if leaving_group:
        os.mkdir("leaving_group")
        with cd("leaving_group"):
            logger.info("Calculating descriptors for leaving_group...")

            # Set up and run descriptor calculations.
            calculator = DescriptorCalculator(leaving_group, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)

            # Set up results dictionary.
            lg_dict = {"epn": {},
                       "hirshfeld": {},
                       "hirshfeld_plus": {},
                       "hirshfeld_minus": {},
                       "ddec6_charge": {},
                       "v_av": {},
                       "vs_max": {},
                       "vs_min": {},
                       "ip": None,
                       "ea": None,
                        }                           
            
            lg_dict["epn"][lg_atom_orig] = calculator.get_epn(lg_atom_orig)
            lg_dict["hirshfeld"][lg_atom_orig] = calculator.get_hirshfeld(lg_atom_orig)
            lg_dict["ddec6_charge"][lg_atom_orig] = calculator.get_ddec6_charge(lg_atom_orig)

            energy_neutral = calculator.get_electronic_energy()

            # Calculate cation and anion to get the Fukui indices.
            lg_plus = leaving_group.copy()
            lg_plus.info["charge"] += 1
            calculator = DescriptorCalculator(lg_plus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_plus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            lg_dict["hirshfeld_plus"][lg_atom_orig] = calculator.get_hirshfeld(lg_atom_orig)
            energy_plus = calculator.get_electronic_energy()

            lg_minus = leaving_group.copy()
            lg_minus.info["charge"] -= 1
            calculator = DescriptorCalculator(lg_minus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_minus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            lg_dict["hirshfeld_minus"][lg_atom_orig] = calculator.get_hirshfeld(lg_atom_orig)
            energy_minus = calculator.get_electronic_energy()

            # Calculate ionization potential and electron affinity.
            ea = (energy_minus - energy_neutral) / eV * Hartree
            ip = (energy_plus - energy_neutral) / eV * Hartree
            lg_dict["ea"] = ea
            lg_dict["ip"] = ip

            logger.info("...leaving group descriptors completed.")
    else:
        lg_dict = {}

    if not config.intramolecular and nucleophile:
        os.mkdir("nucleophile")
        with cd("nucleophile"):
            logger.info("Calculating descriptors for nucleophile...")

            # Set up and run calculation of descriptors
            calculator = DescriptorCalculator(nucleophile, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)

            # Set up results dictionary.
            nu_dict = {"epn": {},
                       "hirshfeld": {},
                       "hirshfeld_plus": {},
                       "hirshfeld_minus": {},
                       "ddec6_charge": {},
                       "vs_min": {},
                       "v_av": {},
                       "is_min": {},
                       "sasa": {},
                       "sasa_ratio": {},
                       "hirshfeld_plus": {},
                       "atom_p_int": {},
                       "atom_p_int_area": {},
                       "ip": None,
                       "ea": None,
                       }
            
            # Get descriptors.
            nu_dict["epn"][nu_atom_orig] = calculator.get_epn(nu_atom_orig)
            nu_dict["hirshfeld"][nu_atom_orig] = calculator.get_hirshfeld(nu_atom_orig)
            nu_dict["ddec6_charge"][nu_atom_orig] = calculator.get_ddec6_charge(nu_atom_orig)
            nu_dict["vs_min"][nu_atom_orig] = calculator.get_vs_min(nu_atom_orig)
            nu_dict["v_av"][nu_atom_orig] = calculator.get_v_av(nu_atom_orig)
            nu_dict["is_min"][nu_atom_orig] = calculator.get_is_min(nu_atom_orig)

            _, sasa, ratio = get_SASA(nucleophile, [nu_atom_orig])
            nu_dict["sasa"][nu_atom_orig] = sasa[0]
            nu_dict["sasa_ratio"][nu_atom_orig] = ratio[0]

            p_int, atom_p_ints, p_int_area, atom_p_ints_areas = get_dispersion(nucleophile, [nu_atom_orig])
            nu_dict["atom_p_int"][nu_atom_orig] = atom_p_ints[0]
            nu_dict["atom_p_int_area"][nu_atom_orig] = atom_p_ints_areas[0]
            nu_dict["p_int"] = p_int
            nu_dict["p_int_area"] = p_int_area

            energy_neutral = calculator.get_electronic_energy()

            # Calculate cation and anion to get the Fukui indices.
            nu_plus = nucleophile.copy()
            nu_plus.info["charge"] += 1
            calculator = DescriptorCalculator(nu_plus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_plus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            nu_dict["hirshfeld_plus"][nu_atom_orig] = calculator.get_hirshfeld(nu_atom_orig)
            energy_plus = calculator.get_electronic_energy()

            nu_minus = nucleophile.copy()
            nu_minus.info["charge"] -= 1
            calculator = DescriptorCalculator(nu_minus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_minus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            nu_dict["hirshfeld_minus"][nu_atom_orig] = calculator.get_hirshfeld(nu_atom_orig)
            energy_minus = calculator.get_electronic_energy()

            # Calculate ionization potential and electron affinity.
            ea = (energy_minus - energy_neutral) / eV * Hartree
            ip = (energy_plus - energy_neutral) / eV * Hartree
            nu_dict["ea"] = ea
            nu_dict["ip"] = ip

            logger.info("...nucleophile descriptors completed.")
    else:
        nu_dict = {}

    if config.agent:
        os.mkdir("agent")
        with cd("agent"):
            logger.info("Calculating descriptors for agent...")

            # Set up and run descriptors calculations
            calculator = DescriptorCalculator(agent, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)

            # Set up results dictionary.
            agent_dict = {"epn": {},
                       "hirshfeld": {},
                       "hirshfeld_plus": {},
                       "hirshfeld_minus": {},
                       "ddec6_charge": {},
                       "vs_min": {},
                       "v_av": {},
                       "is_min": {},
                       "sasa": {},
                       "sasa_ratio": {},
                       "hirshfeld_plus": {},
                       "hirshfeld_minus": {},
                       "atom_p_int": {},
                       "atom_p_int_area": {},
                       "ip": None,
                       "ea": None,
                       }
            
            # Get descriptors.
            agent_dict["epn"][agent_atom_orig] = calculator.get_epn(agent_atom_orig)
            agent_dict["hirshfeld"][agent_atom_orig] = calculator.get_hirshfeld(agent_atom_orig)
            agent_dict["ddec6_charge"][agent_atom_orig] = calculator.get_ddec6_charge(agent_atom_orig)
            agent_dict["vs_min"][agent_atom_orig] = calculator.get_vs_min(agent_atom_orig)
            agent_dict["v_av"][agent_atom_orig] = calculator.get_v_av(agent_atom_orig)
            agent_dict["is_min"][agent_atom_orig] = calculator.get_is_min(agent_atom_orig)

            _, sasa, ratio = get_SASA(agent, [agent_atom_orig])
            agent_dict["sasa"][agent_atom_orig] = sasa[0]
            agent_dict["sasa_ratio"][agent_atom_orig] = ratio[0]

            p_int, atom_p_ints, p_int_area, atom_p_ints_areas = get_dispersion(agent, [agent_atom_orig])
            agent_dict["atom_p_int"][agent_atom_orig] = atom_p_ints[0]
            agent_dict["atom_p_int_area"][agent_atom_orig] = atom_p_ints_areas[0]
            agent_dict["p_int"] = p_int
            agent_dict["p_int_area"] = p_int_area

            energy_neutral = calculator.get_electronic_energy()

            # Calculate cation and anion to get the Fukui indices.
            agent_plus = agent.copy()
            agent_plus.info["charge"] += 1
            calculator = DescriptorCalculator(agent_plus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_plus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            agent_dict["hirshfeld_plus"][agent_atom_orig] = calculator.get_hirshfeld(agent_atom_orig)
            energy_plus = calculator.get_electronic_energy()

            agent_minus = agent.copy()
            agent_minus.info["charge"] -= 1
            calculator = DescriptorCalculator(agent_minus, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp_minus.gjf")
            calculator.run_dft(n_procs=config.n_procs, mem=config.mem)
            agent_dict["hirshfeld_minus"][agent_atom_orig] = calculator.get_hirshfeld(agent_atom_orig)
            energy_minus = calculator.get_electronic_energy()

            # Calculate ionization potential and electron affinity.
            ea = (energy_minus - energy_neutral) / eV * Hartree
            ip = (energy_plus - energy_neutral) / eV * Hartree
            agent_dict["ea"] = ea
            agent_dict["ip"] = ip

            logger.info("...agent descriptors completed.")
    else:
        agent_dict = {}

    if any(ts_list):
        ts_dict_list = []
        for counter, ts in enumerate(ts_list, start=1):
            os.mkdir(f"ts_{counter}")
            with cd(f"ts_{counter}"):
                logger.info(f"Calculating descriptors for TS {counter}")

                # Set up and run descriptor calculations.
                calculator = DescriptorCalculator(ts, config.dft_options, config.descriptor_dft_options, config.directories["atomic_densities"], file="b3lyp.gjf")
                calculator.run_dft(n_procs=config.n_procs, mem=config.mem)

                # Set up results dictionary.
                ts_dict = {"epn": {},
                           "hirshfeld": {},
                           "ddec6_charge": {},
                           "ddec6_bo": {},
                           "v_av": {},
                           }
                
                # Get descriptors.
                for atom in [central_atom, nu_atom, lg_atom]:
                    ts_dict["epn"][atom] = calculator.get_epn(atom)
                    ts_dict["hirshfeld"][atom] = calculator.get_hirshfeld(atom)
                    ts_dict["ddec6_charge"][atom] = calculator.get_ddec6_charge(atom)
                    ts_dict["v_av"][atom] = calculator.get_v_av(atom)

                ts_dict["ddec6_bo"][(central_atom, nu_atom)] = calculator.get_ddec6_bo(central_atom, nu_atom)
                ts_dict["ddec6_bo"][(central_atom, lg_atom)] = calculator.get_ddec6_bo(central_atom, lg_atom)

                ts_dict_list.append(ts_dict)
                logger.info(f"...TS {counter} descriptors completed.")
    else:
        ts_dict_list = []

    # Put descriptors in results module
    results.descriptors["substrate"] = substrate_dict
    results.descriptors["product"] = product_dict
    results.descriptors["nucleophile"] = nu_dict
    results.descriptors["leaving_group"] = lg_dict
    results.descriptors["ts"] = ts_dict_list
    if config.agent:
        results.descriptors["agent"] = agent_dict


def modify_log_ecp(infile, outfile):
    """Modify Gaussian 16 log file with ECP for the HS95 program.
    
    Args:
        infile (str): Path to log file.
        outfile (str): Path to modified log file.
    """
    lines = open(infile).readlines()

    # Make dictionary of valence electrons
    read = False
    pattern = r"\s{2,4}\d{1,3}\s{8,10}\d{1,3}\s{10,12}\d{1,3}"
    valence_dict = {}
    for line in lines:
        if read:
            if counter > 3:
                match = re.match(pattern, line)
                if match:
                    split_match = match.group(0).strip().split()
                    atom_number = int(split_match[1])
                    electrons = int(split_match[2])
                    valence_dict[atom_number] = electrons
                if "======================================================================================================" in line:
                    read = False
            counter += 1
    
        if "Pseudopotential Parameters" in line:
            read = True
            counter = 0
    
    # Modify the valence electrons for atoms with ECPs.
    new_lines = []
    modify = False
    for line in lines:
        if modify:
            if counter > 3:
                if "---------------------------------------------------------------------" in line:
                    modify = False
                    new_lines.append(line)
                else:
                    split_line = line.strip().split()
                    atom_number = int(split_line[1])
                    if atom_number in valence_dict.keys():
                        electrons = valence_dict[atom_number]
                        mod_line = f"{int(split_line[0]):7d}{electrons:11d}{int(split_line[2]):12d}{float(split_line[3]):16.6f}{float(split_line[4]):12.6f}{float(split_line[5]):12.6f}\n"
                        new_lines.append(mod_line)
                    else:
                        new_lines.append(line)
            else:
                new_lines.append(line)
            counter += 1
        else:
            new_lines.append(line)
        if "Standard orientation:" in line:
            modify = True
            counter = 0
    
    # Write lines to new log file
    with open(outfile, "w") as file:
        file.writelines(new_lines)