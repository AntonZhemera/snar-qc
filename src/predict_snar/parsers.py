import configparser
import logging
import re

import ase
import numpy as np

from predict_snar.data import SolventPicker, xtb_avoid_solvents

logger = logging.getLogger("predict_snar")

class GICParser:
    """Parse the results of a GIC scan.
    
    Args:
        file (str): Path to file

    Attributes:
        file (str): Path to file
        geometries (list): Geometries as list of lists of coordinates (Å)
        energies (list): Electronic energies (a.u.)
    """
    def __init__(self, file):
        # Set up attributes
        self.file = file
        self.geometries = []
        self.energies = []

        # Read file and parse it
        lines = open(self.file, 'r').readlines()

        read_geometries = False
        geom_list = []
        energy_list = []
        coordinate_list = []
        nosymm = False
        for line in lines:
            if "Symmetry turned off by external request." in line or "nosymm" in line.lower():
                nosymm = True
            if "Energy= " in line:
                energy = float(line.strip().split()[1])
            if read_geometries:
                counter += 1
                if counter > 4:
                    if "---------------------------------------------------------------------" in line:
                        read_geometries = False
                    else:
                        line_strip = line.strip().split()
                        coordinate = [float(line_strip[3]), float(line_strip[4]), float(line_strip[5])]
                        coordinate_list.append(coordinate)
            if "Standard orientation:" in line or ("Input orientation:" in line and nosymm):
                read_geometries = True
                coordinate_list = []
                counter = 0
            if "-- Stationary point found." in line:
                geom_list.append(coordinate_list)
                energy_list.append(energy)

        # Set up attributes.
        self.geometries = geom_list
        self.energies = energy_list

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class ConfigParser:
    """Parser for the program's configuration file

    Args:
        file (str): Filename of configuration file

    Attributes:
        crest_options (dict): Options for the CREST program
        descriptor_options (dict): Options for descriptor calculations
        dft_options (dict): Options for dft calculations
        directories (dict): Directories for programs and databases
        file (str): Filename of configuration file
        general_options (dict): General options for optimizations
        options (obj): ConfigParser object
        xtb_options (dict): Options for xtb calculations
    """
    def __init__(self, file):
        # Read config file
        self.file = file
        options = configparser.ConfigParser()
        options.read(file)
        self.options = options
        
        # Set up dictionaries to hold different types of options.
        self.general_options = {}
        self.xtb_options = {}
        self.dft_options = {}
        self.crest_options = {}
        self.descriptor_options = {}
        self.directories = {}

        # Initialize solvent picker.
        solvent_picker = SolventPicker()

        # Parse the general options
        general_options = {}
        general_options["ts_max_step"] = options["GENERAL"].getint("ts_max_step")
        general_options["find_intermediate"] = options["GENERAL"].getboolean("find_intermediate")
        general_options["opt_reactants"] = options["GENERAL"].getboolean("opt_reactants")
        general_options["opt_products"] = options["GENERAL"].getboolean("opt_products")
        try:
            general_options["cluster_nucleophile"] = options["GENERAL"].getboolean("cluster_nucleophile")
        except ValueError:
            general_options["cluster_nucleophile"] = None
        general_options["cluster_leaving_group"] = options["GENERAL"].getboolean("cluster_leaving_group")
        general_options["cluster_agent"] = options["GENERAL"].getboolean("cluster_agent")
        general_options["cluster_ts"] = options["GENERAL"].getboolean("cluster_ts")        
        try:
            general_options["temperature"] = options["GENERAL"].getfloat("temperature")
        except ValueError:
            general_options["temperature"] = 298.15

        self.general_options = general_options

        # Parse the dft options
        dft_options = {}
        dft_options["functional"] = options["DFT"].get("functional")
        dft_options["basis_set"] = options["DFT"].get("basis_set")
        dft_options["ecp"] = options["DFT"].get("ecp")
        gaussian_solvent_name = options["DFT"].get("solvent")
        standard_state = solvent_picker.get_standard_state(gaussian_solvent_name)
        gaussian_solvent = solvent_picker.get_gaussian_solvent(gaussian_solvent_name)
        solvent_smiles = solvent_picker.smiles_from_name(gaussian_solvent_name)
        general_options["solvent_smiles"] = solvent_smiles
        dft_options["solvent"] = gaussian_solvent
        dft_options["dispersion"] = options["DFT"].get("dispersion_model")
        dft_options["solvation_model"] = options["DFT"].get("solvation_model")
        general_options["standard_state"] = standard_state
        dft_options["nosymm"] = options["DFT"].get("nosymm")

        dft_sp_options = {}
        if options["DFT"].get("sp_functional"):
            dft_sp_options["functional"] = options["DFT"].get("sp_functional")
        else:
            dft_sp_options["functional"] = options["DFT"].get("functional")
        if options["DFT"].get("sp_basis_set"):
            dft_sp_options["basis_set"] = options["DFT"].get("sp_basis_set")
        else:
             dft_sp_options["basis_set"] = options["DFT"].get("basis_set")
        if options["DFT"].get("sp_ecp"):
            dft_sp_options["ecp"] = options["DFT"].get("sp_ecp")
        else:
            dft_sp_options["ecp"] = options["DFT"].get("ecp")
        if options["DFT"].get("sp_functional"):
            dft_sp_options["dispersion"] = options["DFT"].get("sp_dispersion_model")
        else:
            dft_sp_options["dispersion"] = options["DFT"].get("dispersion_model") 
        if options["DFT"].get("sp_solvation_model"):
            dft_sp_options["solvation_model"] = options["DFT"].get("sp_solvation_model")
        else:
           dft_sp_options["solvation_model"] = options["DFT"].get("solvation_model")

        self.dft_options = dft_options
        self.dft_sp_options = dft_sp_options

        # Parse the xtb options
        xtb_options = {}
        try:
            xtb_options["el_temp"] = options["XTB"].getfloat("electronic_temperature")
        except ValueError:
            xtb_options["el_temp"] = None
        xtb_solvent_name = options["XTB"].get("solvent")
        if xtb_solvent_name:
            xtb_solvent = solvent_picker.get_xtb_solvent(xtb_solvent_name)
            xtb_solvent_cluster = xtb_solvent
        else:
            xtb_solvent = solvent_picker.get_xtb_solvent(gaussian_solvent_name)
            xtb_solvent_cluster = xtb_solvent
        if xtb_solvent.lower() in xtb_avoid_solvents.keys():
            logger.info(f"Avoiding solvent {xtb_solvent} with xtb.")
            xtb_solvent_cluster = xtb_solvent
            xtb_solvent = xtb_avoid_solvents[xtb_solvent.lower()]
            logger.info(f"Using solvent {xtb_solvent} instead.")
        xtb_options["solvent"] = xtb_solvent
        xtb_options["solvent_cluster"] = xtb_solvent_cluster
        xtb_options["gfn_version"] = options["XTB"].get("gfn_version")

        self.xtb_options = xtb_options

        # Parse the CREST options
        crest_options = {}

        crest_options["energy_window"] = options["CREST"].getfloat("energy_window")
        crest_options["speed"] = options["CREST"].get("speed").strip().lower()

        self.crest_options = crest_options

        # Parse the descriptor options
        descriptor_options = {}
        descriptor_options["calculate_descriptors"] = options["DESCRIPTORS"].getboolean("calculate_descriptors")

        self.descriptor_options = descriptor_options

        # Parse the scratch directories
        directories = {}
        directories["xtb"] = options["DIRECTORIES"].get("xtb")
        directories["crest"] = options["DIRECTORIES"].get("crest")
        directories["chargemol"] = options["DIRECTORIES"].get("chargemol")
        directories["hs95"] = options["DIRECTORIES"].get("hs95")
        directories["interface_script"] = options["DIRECTORIES"].get("interface_script")
        directories["crest_scratch"] = options["DIRECTORIES"].get("crest_scratch")
        directories["gaussian_scratch"] = options["DIRECTORIES"].get("gaussian_scratch")
        directories["database"] = options["DIRECTORIES"].get("database")
        directories["atomic_densities"] = options["DIRECTORIES"].get("atomic_densities")

        self.directories = directories
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class SystemParser:
    """Parser for the program's system information file

    Args:
        file (str): Filename of configuration file

    Attributes:
        file (str): Filename of configuration file
        charges (dict): Atomic charges
        clustering (dict): Clustering based on charges
        general_info (dict): General system info
        options (obj): ConfigParser object
        reactive_atoms (dict): Reactive atoms
    """
    def __init__(self, file):
        # Read configuration file
        self.file = file
        options = configparser.ConfigParser()
        options.read(file)
        self.options = options

        # Read general information
        general_info = {}      
        general_info["agent"] = options["GENERAL"].getboolean("agent")
        general_info["ortho_nitro"] = options["GENERAL"].getboolean("ortho_nitro")
        general_info["azide_nucleophile"] = options["GENERAL"].getboolean("azide_nucleophile")
        general_info["proton_transfer"] = options["GENERAL"].getboolean("proton_transfer")
        general_info["azide_angle"] = [int(x) for x in options["GENERAL"].get("azide_angle").split()]
        general_info["azide_dihedral"] = [int(x) for x in options["GENERAL"].get("azide_dihedral").split()]

        self.general_info = general_info         

        # Read reactive atoms
        reactive_atoms = {}
        reactive_atoms["central_atom"] = options["REACTIVE ATOMS"].getint("central_atom")
        reactive_atoms["nu_atom"] = options["REACTIVE ATOMS"].getint("nucleophilic_atom")
        reactive_atoms["lg_atom"] = options["REACTIVE ATOMS"].getint("leaving_atom")
        reactive_atoms["central_atom_prod"] = options["REACTIVE ATOMS"].getint("central_atom_prod")
        reactive_atoms["added_atom"] = options["REACTIVE ATOMS"].getint("added_atom")
        try:
            reactive_atoms["nu_atom_orig"] = options["REACTIVE ATOMS"].getint("nu_atom_orig")
        except ValueError:
            reactive_atoms["nu_atom_orig"] = None
        try:
            reactive_atoms["lg_atom_orig"] = options["REACTIVE ATOMS"].getint("lg_atom_orig")
        except ValueError:
            reactive_atoms["lg_atom_orig"] = None            
        try:
            reactive_atoms["agent_atom"] = options["REACTIVE ATOMS"].getint("agent_atom")
        except ValueError:
            reactive_atoms["agent_atom"] = None
        try:
            reactive_atoms["agent_atom_orig"] = options["REACTIVE ATOMS"].getint("agent_atom_orig")
        except ValueError:
            reactive_atoms["agent_atom_orig"] = None
        try:
            reactive_atoms["ring_atoms"] = [int(x) for x in options["REACTIVE ATOMS"].get("ring_atoms").split()]
        except ValueError:
            reactive_atoms["ring_atoms"] = None
        try:
            reactive_atoms["ortho_carbons"] = [int(x) for x in options["REACTIVE ATOMS"].get("ortho_carbons").split()]
        except ValueError:
            reactive_atoms["ortho_carbons"] = None            
        try:
            reactive_atoms["fragment_nu"] = [int(x) for x in options["REACTIVE ATOMS"].get("fragment_nu").split()]
        except AttributeError:
            reactive_atoms["fragment_nu"] = None
        reactive_atoms["nu_h_atoms"] = [int(x) for x in options["REACTIVE ATOMS"].get("nu_h_atoms").split()]
        reactive_atoms["lg_h_atoms"] = [int(x) for x in options["REACTIVE ATOMS"].get("lg_h_atoms").split()]
        reactive_atoms["nu_sp3_neighbors"] = [int(x) for x in options["REACTIVE ATOMS"].get("nu_sp3_neighbors").split()]
        reactive_atoms["nu_sp3_neighbors_orig"] = [int(x) for x in options["REACTIVE ATOMS"].get("nu_sp3_neighbors_orig").split()]
        reactive_atoms["fragment_lg"] = [int(x) for x in options["REACTIVE ATOMS"].get("fragment_lg").split()]
        reactive_atoms["fragment_substrate"] = [int(x) for x in options["REACTIVE ATOMS"].get("fragment_substrate").split()]
        try:
            reactive_atoms["fragment_agent"] = [int(x) for x in options["REACTIVE ATOMS"].get("fragment_agent").split()]
        except AttributeError:
            reactive_atoms["fragment_agent"] = None
        
        self.reactive_atoms = reactive_atoms

        # Parse the charges
        charges = {}
        charges["substrate"] = options["CHARGES"].getint("substrate")
        try:
            charges["nucleophile"] = options["CHARGES"].getint("nucleophile")
            charges["nu_atom"] = options["CHARGES"].getint("nucleophilic_atom")
        except ValueError:
            charges["nucleophile"] = 0
        charges["product"] = options["CHARGES"].getint("product")
        charges["leaving_group"] = options["CHARGES"].getint("leaving_group")
        try:
            charges["agent"] = options["CHARGES"].getint("agent")
            charges["agent_atom"] = options["CHARGES"].getint("agent_atom")
        except ValueError:
            charges["agent"] = None

        self.charges = charges

        # Parser the clustering
        clustering = {}
        try:
            clustering["nucleophile"] = options["CLUSTERING"].getboolean("nucleophile")
        except ValueError:
            clustering["nucleophile"] = None
        clustering["leaving_group"] = options["CLUSTERING"].getboolean("leaving_group")
        clustering["ts"] = options["CLUSTERING"].getboolean("ts")
        try:
            clustering["agent"] = options["CLUSTERING"].getboolean("agent")
        except ValueError:
            pass

        self.clustering = clustering            


class NBOParser:
    """Parser for Gaussian output file with NBO Wiberg bond indices.
    
    Args:
        file (str): Path to file
    
    Attributes:
        file (str): Path to file
        bo_matrix (ndarray): Bond order matrix.
    """
    def __init__(self, file):
        self.file = file
        self.bo_matrix = np.zeros(())

        # Parse the file
        lines = open(self.file, 'r').readlines()

        read = False
        bo_matrix = np.zeros(())
        for line in lines:
            if "NAtoms" in line:
                n_atoms = int(line.strip().split()[1])
                bo_matrix = np.zeros((n_atoms, n_atoms))
                n_lines = n_atoms + 3

            if read == True:
                if line_counter == 2:
                    cols = line.strip().split()[1:]
                    cols = [ int(x) for x in cols ]
                    if not line.strip():
                        read = False
                elif (line_counter <= n_lines) & (line_counter > 3):
                    values = line.strip().split()[2:]
                    for counter, value in enumerate(values):
                        bo_matrix[line_counter - 4][cols[counter] - 1] = float(value)
                if line_counter == n_lines:
                    line_counter = 1
                else:
                    line_counter += 1
            if "Wiberg bond index matrix in the NAO basis" in line:
                read = True
                line_counter = 1

        # Makes sure that the output is be complete (not partial)
        if not read:
            self.bo_matrix = bo_matrix

    def get_bo(self, atom_1, atom_2):
        """Returns the bond order between two atoms.
        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
        
        Returns:
            bond_order (float): Bond order
        """
        bond_order = self.bo_matrix[atom_1 - 1][atom_2 - 1]
        
        return bond_order

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class EPNParser:
    """Parser for Gaussian output file with electrostatic potential at nuclei.

    Args:
        file (str): Name of Gaussian output file

    Attributes:
        file (str): Name of Gaussian output file.
        potential_list (list): Atomic potentials.
    """
    def __init__(self, file):
        # Set up attributes
        self.file = file
        self.potential_list = []

        # Parse file
        lines = open(self.file, 'r').readlines()

        read = False
        potential_list = []
        for line in lines:
            if read:
                counter += 1
                if counter > 5:
                    if "-------------------------" in line:
                        read = False
                    else:
                        potential = float(line.strip().split()[2])
                        potential_list.append(potential)
            if "Electrostatic Properties (Atomic Units)" in line:
                read = True
                counter = 0
                potential_list = []

        # Set attributes
        self.potential_list = potential_list

    def get_potential(self, atom_index):
        """Get EPN for an atom.
        Args:
            atom_index (int): Atom index (1-indexed)

        Returns:
            potential (float): EPN of atom
        """
        potential = self.potential_list[atom_index - 1]

        return potential

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class HirshfeldParser:
    """Parser for Gaussian output file with Hirshfeld charges.

    Args:
        file (str): Name of Gaussian output file

    Attributes:
        file (str): Name of Gaussian output file.
        charges (list): Hirshfeld charges
    """
    def __init__(self, file):
        # Set up attributes
        self.file = file
        self.charges = []

        #Parse file
        lines = open(self.file, 'r').readlines()

        read = False
        charge_list = []
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if "Tot" in line:
                        read = False
                    else:
                        charge = float(line.strip().split()[2])
                        charge_list.append(charge)
            if "Hirshfeld charges, spin densities" in line:
                read = True
                counter = 0
                charge_list = []

        # Set up attributes
        self.charge_list = charge_list

    def get_charge(self, atom_index):
        """Returns the atomic charge of an atom.

        Args:
            atom_index (int): Atom index (1-indexed)

        Returns:
            charge (float): Atomic charge
        """
        charge = self.charge_list[atom_index - 1]

        return charge

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class XTBParser:
    """Parser for xtb output file giving total energy, homo and lumo energies.

    Args:
        file (str): Output file from xtb calculation

    Attributes:
        grad (list): Gradients (a.u.)
        homo (list): HOMO energies (eV)
        lmos (dict): LMO energies (eV) by atom index (starting at 1)
        lumo (list): LUMO energies (eV)
        energy (list): Electronic energies (a.u.)
        free_energy (list): Free energy (a.u.)
        restrained_energy (list): Restrained energy for constrained optimization (a.u.)
    """
    def __init__(self, file):
        # Parse file and set up attributes.
        self.file = file
        lines = open(self.file, 'r', encoding="utf-8").readlines()

        self.homo = []
        self.lumo = []
        self.energy = []
        self.free_energy = None
        self.restrained_energy = []
        self.grad = []
        self.lmos = None
        self.charges = []

        read_lmos = False
        read_wbo = False
        read_charges = False
        lmo_dict = {}

        for line in lines:
            if ":: total energy" in line:
                total_energy = float(line.strip().split()[3])
            if ":: add. restraining" in line:
                restraining_energy = float(line.strip().split()[3])
                if restraining_energy != 0:
                    self.restrained_energy.append(total_energy)
                unrestrained_energy = total_energy - restraining_energy
                self.energy.append(unrestrained_energy)
            if ":: total free energy" in line:
                self.free_energy = float(line.strip().split()[4])
            if ":: gradient norm" in line:
                self.grad.append(float(line.strip().split()[3]))
            if "number of atoms" in line:
                n_atoms = int(line.strip().split()[4])
            if ":: HOMO orbital eigv." in line:
                 self.homo.append(float(line.strip().split()[4]))
            if ":: LUMO orbital eigv." in line:
                 self.lumo.append(float(line.strip().split()[4]))
            if "LMO Fii/eV" in line:
                read_lmos = True
            if read_charges:
                if not line.strip():
                    read_charges = False
                else:
                    strip_line = line.strip().split()
                    q = float(strip_line[4])
                    self.charges.append(q)
            if "#   Z        covCN         q      C6AA      α(0)" in line:
                read_charges = True
                self.charges = []
            if read_wbo:
                if not line.strip():
                    read_wbo = False
                else:
                    strip_line = line.strip().split()
                    this_atom = strip_line[:3]
                    other_atoms = strip_line[3:]
                    atom_1 = int(this_atom[0]) - 1
                    wbo_atom_1 = float(this_atom[2])
                    bo_matrix[atom_1, atom_1] = wbo_atom_1
                    for i in range(int(len(other_atoms) / 3)):
                        other_atom = other_atoms[i * 3:i * 3 + 3]
                        atom_2 = int(other_atom[1]) - 1
                        wbo = float(other_atom[2])
                        bo_matrix[atom_1, atom_2] = wbo
                        bo_matrix[atom_2, atom_1] = wbo                            
            if "total WBO             WBO to atom ..." in line:
                read_wbo = True
                bo_matrix = np.zeros((n_atoms, n_atoms))             
            elif "starting deloc pi regularization" in line or "files" in line:
                read_lmos = False
            if read_lmos:
                strip_line = line.strip().split()
                if n_atoms > 1:
                    if strip_line[1] == "LP":
                        energy = float(strip_line[2])
                        atom = int(re.sub(r"\D", "", strip_line[7]).replace(":", ""))
                        old_energy = lmo_dict.get(atom)
                        if old_energy:
                            if old_energy < energy:
                                lmo_dict[atom] = energy
                        else:
                            lmo_dict[atom] = energy
                    # Parse polarized pi orbitals
                    if strip_line[1] == "pi":
                        weight_info = strip_line[7:]
                        atoms = weight_info[::3]
                        weights = weight_info[2::3]
                        atoms = [int(re.sub(r"\D", "", i)) for i in atoms]
                        weights = [float(i) for i in weights]
                        for atom, weight in zip(atoms, weights):
                            if weight > 0.7 or (len(atoms) > 2 and weight > 0.6):
                                logger.info(f"Polarized pi orbital with weight {weight} on atom {atom} detected. "
                                            "Taking as potential nucleophilic center. ")
                                energy = float(strip_line[2])
                                old_energy = lmo_dict.get(atom)
                                if old_energy:
                                    if old_energy < energy:
                                        lmo_dict[atom] = energy
                                else:
                                    lmo_dict[atom] = energy
                else:
                    if strip_line[1] == "sigma":
                        energy = float(strip_line[2])
                        atom = int(re.sub(r"\D", "", strip_line[7]).replace(":", ""))
                        old_energy = lmo_dict.get(atom)
                        if old_energy:
                            if old_energy < energy:
                                lmo_dict[atom] = energy
                        else:
                            lmo_dict[atom] = energy

        # Set up attributes.
        self.lmo_dict = lmo_dict
        self.bo_matrix = bo_matrix
        
        if len(self.homo) == 0:
            self.homo = None
        elif len(self.homo) <= 2:
            self.homo = self.homo[-1]
        else:
            self.homo = self.homo[2:]

        if len(self.lumo) == 0:
            self.lumo = None       
        elif len(self.lumo) <= 2:
            self.lumo = self.lumo[-1]
        else:
            self.lumo = self.lumo[2:]
        
        if len(self.energy) == 0:
            self.energy = None
        elif len(self.energy) <= 2:
            self.energy = self.energy[-1]
        else:
            self.energy = self.energy[2:]

        if len(self.restrained_energy) == 0:
            self.restrained_energy = None
        elif len(self.restrained_energy) <= 2:
            self.restrained_energy = self.restrained_energy[-1]
        else:
            self.restrained_energy = self.restrained_energy[2:]
        
        if len(self.grad) == 0:
            self.grad = None
        elif len(self.grad) <= 2:
            self.grad = self.grad[-1]
        else:
            self.grad = self.grad[2:]

    def get_bo(self, atom_1, atom_2):
        """Returns the bond order between two atoms.
        
        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)

        Returns:
            bond_order (float): Bond order
        """
        bond_order = self.bo_matrix[atom_1 - 1][atom_2 - 1]

        return bond_order            

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class ChargeMolParser:
    """Parser for the Chargemol output. Extracts charges and bond orders.

    Args:
        file (str): Name of output file from xtb calculation

    Attributes:
        file (str): Filename
        charge_list (list): List of charges
        bo_matrix (ndarray): Bond order matrix
    """
    def __init__(self, file):
        # Set up attributes
        self.file = file
        self.charge_list = []
        self.bo_matrix = np.empty(())

        lines = open(self.file, 'r', encoding="utf-8").readlines()

        # Parse charges
        charge_list = []
        read = False
        for line in lines:
            if read:
                if not line.strip().split():
                    read = False
                else:
                    charge = float(line.strip().split()[5])
                    charge_list.append(charge)

            if "center number, atomic number, x, y, z, net_charge" in line:
                charge_list = []
                read = True

        self.charge_list = charge_list

        # Parse bond order matrix.
        bo_matrix = np.empty(())
        read = False
        for line in lines:
            if "ncenters=" in line:
                n_atoms = int(line.strip().split()[1])
            if read:
                if "The legend for the bond pair matrix follows:" in line:
                    read = False
                else:
                    split_line = line.strip().split()
                    atom_1 = int(split_line[0]) - 1
                    atom_2 = int(split_line[1]) - 1
                    bo = float(split_line[19])
                    bo_matrix[atom_1][atom_2] = bo
            if "The final bond pair matrix is" in line:
                bo_matrix = np.zeros((n_atoms, n_atoms))
                read = True
        self.bo_matrix = bo_matrix + bo_matrix.T

    def get_bo(self, atom_index_1, atom_index_2):
        """Returns the bond order between two atoms.
        
        Args:
            atom_index_1 (int): Atom index of atom 1 (1-indexed)
            atom_index_2 (int): Atom index of atom 2 (1-indexed)

        Returns:
            bond_order (float): Bond order.
        """
        bond_order = self.bo_matrix[atom_index_1 - 1][atom_index_2 - 1]

        return bond_order

    def get_charge(self, atom_index):
        """Returns the charge for an atom.

        Args:
            atom_index (int): Atom index (1-indexed)

        Returns:
            charge (int): Atomic charge
        """
        charge = self.charge_list[atom_index - 1]

        return charge

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class BrinckParser:
    """Parser for the HS95 output. Extracts Es_min, is_min, vs_min, vs_max,
    and vs_av.

    Args:
        file (str):         Output file from xtb calculation

    Attributes:
        file (str): Filename
        es_min_dict (dict): Es_min as values and atom indices as keys. Taken
            from the extreme points.
        es_min_all_dict (dict): Es_min as values and atom indices as keys. Taken
            as the minimum over the surface of the atom.
        is_min_dict (dict): Is_min as values and atom indices as keys. Taken
            from the extreme points.
        is_min_all_dict (dict): Is_min as values and atom indices as keys. Taken
            from the minimum over surface of the atom.
        vs_min_dict (dict): Vs_min as values and atom indices as keys. Taken
            from the extreme points.
        vs_min_all_dict (dict): Vs_min as values and atom indices as keys. Taken
            from the minimum over the surface of the atom.
        vs_max_dict (dict): Vs_max as values and atom indices as keys. Taken
            from the extreme points.
        vs_max_all_dict (dict): Vs_max as values and atom indices as keys. Taken
            from the maximum over the surface of the atom.
        v_av_dict (dict): V_av as values and atom indices as keys. Taken from
            the average over the surface of the atom.
    """
    def __init__(self, file):
        # Set up attributes.
        self.file = file
        self.es_min_dict = {}
        self.es_min_all_dict = {}
        self.is_min_dict = {}
        self.is_min_all_dict = {}
        self.vs_min_dict = {}
        self.vs_min_all_dict = {}
        self.vs_max_dict = {}
        self.vs_max_all_dict = {}
        self.v_av_dict = {}

        # Parse file.
        lines = open(self.file, 'r', encoding="utf-8").readlines()

        es_min_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[3])
                        es_min = float(line.strip().split()[4])
                        if es_min_dict.get(atom_index):
                            old_es_min = es_min_dict[atom_index]
                            es_min = min(es_min, old_es_min)
                        es_min_dict[atom_index] = es_min
            if "Info from  Electron Attachment Energy min (Esmin)  search" in line:
                counter = 0
                read = True
        self.es_min_dict = es_min_dict

        es_min_all_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[0])
                        es_min = float(line.strip().split()[7])
                        es_min_all_dict[atom_index] = es_min
            if "****Statistical data atom by atom 2****" in line:
                counter = 0
                read = True
        self.es_min_all_dict = es_min_all_dict

        is_min_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[3])
                        is_min = float(line.strip().split()[4])
                        if is_min_dict.get(atom_index):
                            old_is_min = is_min_dict[atom_index]
                            is_min = min(is_min, old_is_min)
                        is_min_dict[atom_index] = is_min
            if "Information from Ismin search" in line:
                counter = 0
                read = True
        self.is_min_dict = is_min_dict

        is_min_all_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[0])
                        is_min = float(line.strip().split()[16])
                        is_min_all_dict[atom_index] = is_min
            if "****Statistical data atom by atom 1****" in line:
                counter = 0
                read = True
        self.is_min_all_dict = is_min_all_dict

        vs_min_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[3])
                        vs_min = float(line.strip().split()[4])
                        if vs_min_dict.get(atom_index):
                            old_vs_min = vs_min_dict[atom_index]
                            vs_min = min(vs_min, old_vs_min)
                        vs_min_dict[atom_index] = vs_min
            if "Information from Vsmin search" in line:
                counter = 0
                read = True
        self.vs_min_dict = vs_min_dict

        vs_min_all_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[0])
                        vs_min = float(line.strip().split()[13])
                        vs_min_all_dict[atom_index] = vs_min
            if "****Statistical data atom by atom 1****" in line:
                counter = 0
                read = True
        self.vs_min_all_dict = vs_min_all_dict

        vs_max_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[3])
                        vs_max = float(line.strip().split()[4])
                        if vs_max_dict.get(atom_index):
                            old_vs_max = vs_max_dict[atom_index]
                            vs_max = min(vs_max, old_vs_max)
                        vs_max_dict[atom_index] = vs_max
            if "Information from Vsmax search" in line:
                counter = 0
                read = True
        self.vs_max_dict = vs_max_dict

        vs_max_all_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[0])
                        vs_max = float(line.strip().split()[12])
                        vs_max_all_dict[atom_index] = vs_max
            if "****Statistical data atom by atom 1****" in line:
                counter = 0
                read = True
        self.vs_max_all_dict = vs_max_all_dict

        v_av_dict = {}
        read = False
        for line in lines:
            if read:
                counter += 1
                if counter > 1:
                    if not line.strip().split():
                        read = False
                    else:
                        atom_index = int(line.strip().split()[0])
                        v_av = float(line.strip().split()[5])
                        v_av_dict[atom_index] = v_av
            if "****Statistical data atom by atom 1****" in line:
                counter = 0
                read = True
        self.v_av_dict = v_av_dict

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r})"


class CRESTParser:
    """Parser of CREST output file to get energies, weights and degeneracies
    of conformers.

    Args:
        file (str): Name of file

    Attributes:
        file (str): Name of file
        energies (list): Energeis (kcal/mol)
        weights (list): Weights (fractions summing to 1)
        degeneracies (list): Degeneracies of each conformer.
    """
    def __init__(self, file):
        # Read file and parse.
        self. file = file
        lines = open(file).readlines()

        read = False
        energies = []
        weights = []
        degeneracies = []

        for line in lines:
            if read:
                split_line = line.strip().split()
                if len(split_line) == 8:
                    energy = float(split_line[1])
                    weight = float(split_line[4])
                    degeneracy = int(split_line[6])
                    
                    energies.append(energy)
                    weights.append(weight)
                    degeneracies.append(degeneracy)
                if "T /K " in line:
                    read = False
            if "Erel/kcal     Etot      weight/tot conformer  set degen    origin" in line:
                read = True
        
        # Set up attributes.
        self.energies = energies
        self.weights = weights
        self.degeneracies = degeneracies


class GaussianParser:
    """Parse Gaussian output file for energies of double hybrid calculations.

    Args:
        file (str): Name of log file
    
    Attributes:
        dh_energy (float): Full double hybrid energy (a.u.)
        scf_energy (float): SCF part of double hybrid energy (a.u.)
    """
    def __init__(self, file):
        # Read file
        lines = open(file).readlines()
        
        # Parse lines for the SCF and double hybrid energy
        pattern = re.compile(r"^ E2\(\w+\)")
        for line in lines:
            match = pattern.match(line)
            if match:
                split_line = line.strip().split()
                dh_energy = float(split_line[5].replace("D", "E"))
            if "SCF Done" in line:
                split_line = line.strip().split()
                scf_energy = float(split_line[4])
        
        # Store attributes
        self.dh_energy = dh_energy
        self.scf_energy = scf_energy