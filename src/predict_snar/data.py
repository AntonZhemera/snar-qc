from pathlib import Path
import pickle
import re
import scipy.constants
from scipy.spatial.distance import cdist
from subprocess import Popen, PIPE, DEVNULL
import warnings

import numpy as np

# Solvents to avoid in xtb and their replacements
xtb_avoid_solvents = {"dmso": "methanol"}

# Units and constants
AVOGADRO_CONSTANT = scipy.constants.physical_constants["Avogadro constant"][0]
MOL = AVOGADRO_CONSTANT
HARTREE = scipy.constants.physical_constants["atomic unit of energy"][0]
KCAL = scipy.constants.calorie * 1000
EV = scipy.constants.electron_volt
BOHR = scipy.constants.physical_constants["atomic unit of length"][0]
ANGSTROM = scipy.constants.angstrom
PLANCK = scipy.constants.Planck
ATM = scipy.constants.atm
BOLTZMANN = scipy.constants.Boltzmann
AMU = scipy.constants.physical_constants["atomic mass constant"][0]
GAS_CONSTANT = scipy.constants.R

# Conversion factors
HARTREE_TO_KCAL = HARTREE / KCAL * MOL
KCAL_TO_HARTREE = KCAL / HARTREE / MOL
EV_TO_KCAL = EV / KCAL * MOL
EV_TO_HARTREE = EV / HARTREE
BOHR_TO_ANGSTROM = BOHR / ANGSTROM
ANGSTROM_TO_BOHR = ANGSTROM / BOHR
STANDARD_STATE_UNIT_CORRECTION = 1 / HARTREE / AVOGADRO_CONSTANT

# Atomic masses
atomic_masses = {1: 1.0078250322, 2: 3.01602932, 3: 6.015122887, 4: 9.0121831,
5: 10.012937, 6: 12.0, 7: 14.003074004, 8: 15.99491462, 9: 18.998403163,
10: 19.99244018, 11: 22.98976928, 12: 23.9850417, 13: 26.9815385,
14: 27.976926535, 15: 30.973761998, 16: 31.972071174, 17: 34.9688527,
18: 35.9675451, 19: 38.96370649, 20: 39.9625909, 21: 44.955908, 22: 45.952628,
23: 49.947156, 24: 49.946042, 25: 54.938044, 26: 53.939609, 27: 58.933194,
28: 57.935342, 29: 62.929598, 30: 63.929142, 31: 68.925574, 32: 69.924249,
33: 74.921595, 34: 73.9224759, 35: 78.918338, 36: 77.920365, 37: 84.91178974,
38: 83.913419, 39: 88.90584, 40: 89.9047, 41: 92.90637, 42: 91.906808,
43: 97.90721, 44: 95.90759, 45: 102.9055, 46: 101.9056, 47: 106.90509,
48: 105.90646, 49: 112.904062, 50: 111.904824, 51: 120.90381, 52: 119.90406,
53: 126.90447, 54: 123.90589, 55: 132.90545196, 56: 129.90632, 57: 137.90712,
58: 135.907129, 59: 140.90766, 60: 141.90773, 61: 144.91276, 62: 143.91201,
63: 150.91986, 64: 151.9198, 65: 158.92535, 66: 155.92428, 67: 164.93033,
68: 161.92879, 69: 168.93422, 70: 167.93389, 71: 174.94078, 72: 173.94005,
73: 179.94746, 74: 179.94671, 75: 184.952955, 76: 183.952489, 77: 190.96059,
78: 189.95993, 79: 196.966569, 80: 195.96583, 81: 202.972345, 82: 203.973044,
83: 208.9804, 84: 207.98125, 85: 209.98715, 86: 209.98969, 87: 211.99623,
88: 226.02541, 89: 225.02323, 90: 230.03313, 91: 231.03588, 92: 233.03964,
93: 236.0466, 94: 238.04956, 95: 241.05683, 96: 243.06139, 97: 247.07031,
98: 249.07485, 99: 252.083, 100: 253.08519, 101: 258.09843, 102: 255.0932,
103: 261.107, 104: 265.117, 105: 268.126, 106: 269.129, 107: 270.133,
108: 269.1338, 109: 276.152, 110: 280.161, 111: 281.166, 112: 283.173,
113: 285.18, 114: 287.187, 115: 288.193, 116: 291.201, 117: 293.208,
118: 294.214}
"""dict: Atomic numbers as keys and masses as values."""

class SolventPicker:
    """Pick solvents for Gaussian and xtb based on a IUPAC or trivial name.
    
    xtb solvents will be picked with as close dielectric constant as possible to
    the correponding Gaussian solvent. If solvent is not in the library of
    supported Gaussian solvents, the closest solvent in terms of epsilon and
    hydrogen-bonding parameters will be chosen.

    Attributes:
        gaussian_solvent (str): Name of Gaussian solvent
        solvent_trivial_names (dict): Mapping between trivial names and names 
                                      accepted by inchikey constructor
        xtb_solvent (str): Name of xtb solvent
    """
    def __init__(self):
        # Load solvent dictionaries
        base_path = Path(__file__).parent

        xtb_file = base_path / "../data/solvent/xtb_solvents.pickle"
        with open(xtb_file, "rb") as file:
            self._xtb_solvents = pickle.load(file)
        
        gaussian_file = base_path / "../data/solvent/gaussian_solvents.pickle"
        with open(gaussian_file, "rb") as file:
            self._gaussian_solvents = pickle.load(file)
        
        ss_file = base_path / "../data/solvent/standard_states.pickle"
        with open(ss_file, "rb") as file:
            self._standard_states = pickle.load(file)

        epsilon_file = base_path / "../data/solvent/epsilon.pickle"
        with open(epsilon_file, "rb") as file:
            self._epsilon = pickle.load(file)

        epsilon_h_file = base_path / "../data/solvent/epsilon_h.pickle"
        with open(epsilon_h_file, "rb") as file:
            self._epsilon_h = pickle.load(file)
        
        # Set up dictionary to map common trivial names to systematic names
        self.solvent_trivial_names = {"dmf": "dimethylformamide",
                                      "dmso": "dimethyl sulfoxide",
                                      "h2o": "water",
                                      "hmpt": "hexamethylphosphoramide",
                                      "mek": "butanone",
                                      "meno2": "nitromethane",
                                      "nmp": "n-methyl-2-pyrrolidone",
                                      "pc": "propylene carbonate",
                                      "thf": "tetrahydrofuran",
                                      "tms": "tetramethylsilane",
                                      }
        
        # Path to OPSIN used for trivial name parsing.
        self._jar_path = base_path / "../data/solvent/opsin-2.4.0-jar-with-dependencies.jar"
    
    def get_gaussian_solvent(self, name):
        """Get the name of the Gaussian solvent corresponding to the name
        
        Args:
            name (str): Name of solvent
        
        Returns:
            gaussian_name (str): Name of Gaussian solvent
        """
        # Get the inchikey from OPSIN.
        inchikey = self.inchikey_from_name(name)
        
        # Check for TMS and change to cyclopentane
        if inchikey == "CZDYPVPMEAXLPK-UHFFFAOYSA-N":
            warnings.warn("Changing TMS to cyclopentane.")
            inchikey = "RGSFGYAAUTVSQA-UHFFFAOYSA-N"

        # Check if solvent exists in Gaussian. Otherwise get nearest neighbor.
        gaussian_entry = self._gaussian_solvents.get(inchikey)
        if gaussian_entry:
            gaussian_name = gaussian_entry["name"]
        else:
            neighbor_inchikey = self._get_nearest_neighbor(inchikey)
            gaussian_entry = self._gaussian_solvents.get(neighbor_inchikey)
            gaussian_name = gaussian_entry["name"]
        
        self.gaussian_solvent = gaussian_name
        
        return gaussian_name
    
    def get_xtb_solvent(self, name):
        """Get the name of the xtb solvent corresponding to the name

        Args:
            name (str): Name of solvent
        
        Returns:
            xtb_name (str): Name of xtb solvent
        """
        # Get the inchikey. 
        inchikey = self.inchikey_from_name(name)

        # Check for TMS and change to cyclopentane        
        if inchikey == "CZDYPVPMEAXLPK-UHFFFAOYSA-N":
            warnings.warn("Changing TMS to cyclopentane.")
            inchikey = "RGSFGYAAUTVSQA-UHFFFAOYSA-N"

        # Check if solvent exists in Gaussian. Otherwise get nearest neighbor.
        gaussian_entry = self._gaussian_solvents.get(inchikey)
        if not gaussian_entry:
            neighbor_inchikey = self._get_nearest_neighbor(inchikey)
            gaussian_entry = self._gaussian_solvents.get(neighbor_inchikey)
        
        # Check if solvent exists in XTB. Otherwise get nearest neighbor in
        # terms of dielectric constant.
        xtb_entry = self._xtb_solvents.get(inchikey)
        if xtb_entry:
            xtb_name = xtb_entry["name"]
        else:
            gaussian_epsilon = gaussian_entry["epsilon"]
            xtb_epsilons = np.array([value["epsilon"] 
                                     for value in self._xtb_solvents.values()])
            xtb_names = [value["name"] for value in self._xtb_solvents.values()]

            closest_epsilon = (np.abs(xtb_epsilons - gaussian_epsilon)).argmin()
            xtb_name = xtb_names[closest_epsilon]
        self.xtb_solvent = xtb_name
        
        return xtb_name
    
    def get_standard_state(self, name):
        """Get the standard state of solvent corresponding to name

        Args:
            name (str): Name of solvent
        
        Returns:
            standard_state (float): Standard state (M)
        """
        # Get inchikey from OPSIN.
        inchikey = self.inchikey_from_name(name)

        # Handle cases of TMS and non-recognized solvents.
        if inchikey == "CZDYPVPMEAXLPK-UHFFFAOYSA-N":
            warnings.warn("Changing TMS to cyclopentane.")
            inchikey = "RGSFGYAAUTVSQA-UHFFFAOYSA-N"
        elif inchikey == "":
            warnings.warn("Solvent could not be identified. Defaulting to acetonitrile.")
            inchikey = "WEVYAHXRMPXWCK-UHFFFAOYSA-N"
        
        # Get the standard state from the dictionary.
        standard_state = self._standard_states.get(inchikey)
        if standard_state:
            standard_state = standard_state["standard_state"]
        else:
            warnings.warn("Couldn't find standard state. Setting standard state to 1.")
            standard_state = 1

        return standard_state

    def inchikey_from_name(self, name):
        """Convert name to InChIKey using OPSIN
        
        Args:
            name (str): Name to be converted
        
        Returns:
            inchikey (str): InChIKey string
        """
        # Check if name is a trivial name
        if self.solvent_trivial_names.get(name.strip().lower()):
            name = self.solvent_trivial_names[name.strip().lower()]
        
        # Run OPSIN to convert name to inchikey
        process = Popen(f'java -jar "{self._jar_path.absolute()}" -ostdinchikey',
                        shell=True, stdin=PIPE, stdout=PIPE, stderr=DEVNULL)      
        inchikey = process.communicate(input=str.encode(name))[0].strip().decode()
        
        return inchikey

    def smiles_from_name(self, name):
        """Convert name to Smiles using OPSIN
        
        Args:
            name (str): Name to be converted
        
        Returns:
            smiles (str): Smiles string
        """
        # Check if name is a trivial name
        if self.solvent_trivial_names.get(name.strip().lower()):
            name = self.solvent_trivial_names[name.strip().lower()]
        
        # Run OPSIN to convert name to inchikey
        process = Popen(f'java -jar "{self._jar_path.absolute()}" -osmi',
                        shell=True, stdin=PIPE, stdout=PIPE, stderr=DEVNULL)      
        smiles = process.communicate(input=str.encode(name))[0].strip().decode()
        
        return smiles
    
    def get_influential_solvent(self, string):
        """Take out the influential solvent from a text string.

        Args:
            string (str): Text string.

        Returns:
            solvent (str): Returns name of the influential solvent.
        """
        # Find components and weights. Assume volume weigts
        components = re.findall(r"(?![0-9]{2,})([a-zA-Z0-9,_-]{2,})", string)
        weights = [float(match) for match in re.findall(r"(?!<=[,\-])[\d]{1,3}(?![,\-])", string)]
        
        # If only one component, take it.
        if len(components) == 1:
            solvent = components[0]
        else:
            # Handle case where weights are not given for all solvents. Assume
            # that it was given for the first.
            len_diff = len(components) - len(weights)
            if len_diff:
                rest = 100 - sum(weights)
                weights = weights + [rest / len_diff] * len_diff
        
            # Get the molar fractions
            inchikeys = [self.inchikey_from_name(component) for component in components]
            molar_volumes = [self._standard_states[inchikey]["molar_volume"] for inchikey in inchikeys]
            mols = np.array(weights) / np.array(molar_volumes)
            mol_sum = np.sum(mols)
            mol_fractions = mols / mol_sum
            
            # Determine the strongest solvent in terms of sum of dielectric constant, h bond acidity and h bond basicity.
            # These have been normalized beforehand.
            epsilon_hs = [self._epsilon_h.get(inchikey) for inchikey in inchikeys]
            # Handle case where we have H bond acidity and basicity for all cases.
            if all(epsilon_hs):
                summed_values = [np.sum([value["epsilon"], value["h_acidity"], value["h_basicity"]]) for value in epsilon_hs]
                strong_solvent_indices = np.argsort(summed_values)[::-1]
            # In case of missing H bonding parameters, use only epsilon.
            else:
                epsilons = [self._epsilon.get(inchikey) for inchikey in inchikeys]
                if all(epsilons):
                    summed_values = [value["epsilon"] for value in epsilons]
                    strong_solvent_indices = np.argsort(summed_values)[::-1]
            
            # Take out strong solvent if it has at least 20% mol fraction
            for index in strong_solvent_indices:
                if mol_fractions[index] > 0.2:
                    solvent = components[index]
                    break
        return solvent
    
    def _get_nearest_neighbor(self, inchikey):
        # Get the values of epsilon, h_a and h_b from the file
        epsilon = self._epsilon.get(inchikey)
        epsilon_h = self._epsilon_h.get(inchikey)

        # Handle case where hydrogen-bonding parameters are present.
        if epsilon_h:
            gaussian_inchikey = list(self._gaussian_solvents.keys())
            h_values = np.array([[value["epsilon"], value["h_acidity"], value["h_basicity"]] for inchikey, value in self._epsilon_h.items() if inchikey in gaussian_inchikey])
            h_inchikey = [inchikey for inchikey in self._epsilon_h.keys() if inchikey in gaussian_inchikey]
            solvent_value = np.array([epsilon_h["epsilon"], epsilon_h["h_acidity"], epsilon_h["h_basicity"]]).reshape(1, -1)
            solvent_index = cdist(solvent_value, h_values).argmin()
            inchikey = h_inchikey[solvent_index]
        # Handle case without hydrogen-bonding parameters. Use epsilon instead.
        elif epsilon:
            gaussian_inchikey = list(self._gaussian_solvents.keys())
            epsilon_values = np.array([value["epsilon"] for inchikey, value in self._epsilon.items() if inchikey in gaussian_inchikey]).reshape(-1, 1)
            epsilon_inchikey = [inchikey for inchikey in self._epsilon.keys() if inchikey in gaussian_inchikey]
            solvent_value = np.array([epsilon["epsilon"]]).reshape(1, -1)
            solvent_index = cdist(solvent_value, epsilon_values).argmin()
            inchikey = epsilon_inchikey[solvent_index]
        # In the worst case, pick acetontrile.
        else:
            warnings.warn(f"Warning! Solvent with InChIKey {inchikey} could not be treated. Defaulting to acetonitrile.")
            inchikey = "WEVYAHXRMPXWCK-UHFFFAOYSA-N"

        return inchikey


class GoodVibes:
    """Performs and stores results of thermochemical data with Paton's GoodVibes code.

    Args:
        filename (str): Name of Gaussian log file
        quasi_harmonic (str): Choice of method: 'grimme' (default) or 'truhlar'
        scaling (float): Scaling factor for frequencies (default: 1.0)
        standard_state (float): Standard state (M, default: 1.0)
        temperature (float): Temperature (K, default: 298.15)
    
    Attributes:
        enthalpy (float): Enthalpy (a.u)
        free_energy (float): Free energy (a.u.)
        free_energy_qh (float): Free energy with quasi-harmonic corrections (a.u.)
        t_entropy (float): Temperature times entropy (a.u.)
        t_entropy_qh (float): Temperature times entropy with quasi-harmonic corrections (a.u.)
    """
    def __init__(self, filename, temperature=298.15, scaling=1.0, standard_state=1.0, quasi_harmonic="grimme", cutoff=100):
        # Set up and run Goodvibes
        self.filename = filename
        submit_string = f"python -m  goodvibes --qs {quasi_harmonic} -f {cutoff} -v {scaling} -t {temperature} -c {standard_state} {filename}"
        process = Popen(submit_string.split(), stdout=PIPE)
        output = process.communicate()[0].decode()
        
        # Read output
        lines = output.split("\n")
        lines = [line.strip() for line in lines]
        result_line = lines[-3].split()
        
        self.free_energy = float(result_line[7])
        self.free_energy_qh = float(result_line[8])
        
        self.enthalpy = float(result_line[4])
        self.t_entropy = float(result_line[5])
        self.t_entropy_qh = float(result_line[6])
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.filename!r})"


def get_basis(name):
    """Get the basis set as dictionary from its name.

    Args:
        name (str): Name of basis set

    Returns:
        basis (dict): Basis set
    """
    base_path = Path(__file__).parent
    basis_file = base_path / f"../data/basis_sets/{name.lower()}-basis.pickle"
    with open(basis_file, "rb") as file:
        basis = pickle.load(file)

    return basis


def get_ecp(name):
    """Get ECP as dictionary based from its name.

    Args:
        name (str): Name of basis set
    
    Returns:
        ecp (dict): ECP
    """
    base_path = Path(__file__).parent
    basis_file = base_path / f"../data/basis_sets/{name.lower()}-ecp.pickle"
    with open(basis_file, "rb") as file:
        ecp = pickle.load(file)

    return ecp