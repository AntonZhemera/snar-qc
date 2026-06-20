import logging
import os
import glob
import shutil
import subprocess
from typing import NamedTuple, Union
import textwrap

import ase
import ase.io
import cclib
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from predict_snar import config
from predict_snar.data import get_basis, get_ecp
from predict_snar.data import EV_TO_KCAL, EV_TO_HARTREE, HARTREE_TO_KCAL, ANGSTROM_TO_BOHR, KCAL_TO_HARTREE
from predict_snar.helpers import cd, single_point_job
from predict_snar.parsers import NBOParser, XTBParser, GICParser, CRESTParser

logger = logging.getLogger("predict_snar")

class TSScan:
    """Handles transition state scan of two-step reaction.

    Args:
        atoms (object): ASE atoms object
        xtb_options (dict): Options for the XTB calculations
        dft_options (dict): Options for the DFT calculations
        general_options (dict): General information needed for the scan.

    Attributes:
        atoms (object): ASE atoms object
        geometries (list): ASE atoms objects for optimized points along the scan.
        xtb_energies (list): XTB energies along the scan (kcal/mol).
        dft_energes (list): DFT energies along the scan.
        nbo_data (list): NBO parsers for each point along the scan.
        hl_gaps (list): HOMO-LUMO gaps for each point along the scan (eV).
        xtb (object): Parent xtb calculator.
        g16 (object): Parent G16 calculator.
        peaks (list): Peaks along the scan.
        central_atom (int): Index of central atom
        nu_atom (int): Index of nu atom.
        lg_atom (int): Index of leaving atom. 
    """
    def __init__(self, atoms, xtb_options, dft_options, general_options):
        # Set up attributes
        self.atoms = atoms
        self.geometries = []
        self.xtb_energies = []
        self.dft_energies = []
        self.nbo_data = []
        self.peaks = []
        self.hl_gaps = []
        self.central_atom = general_options["central_atom"]
        self.nu_atom = general_options["nu_atom"]
        self.lg_atom = general_options["lg_atom"]

        # Set up calculators
        self.xtb = XTBCalculator(atoms, options=xtb_options)
        self.g16 = G16Calculator(atoms, options=dft_options)
        
        # Set specific settings for the single points.
        self.g16.options["int_acc"] = "fine"
        self.g16.options["scf_acc"] = "sleazy"
        self.g16.options["nbo"] = True
        self.g16.options["chk"] = False

        # Set specific settings for the xtb scan.
        self.xtb.options["force_constant"] = 0.5
        self.xtb.options["fix_force_constant"] = 0.01
        self.xtb.options["verbose"] = True
        if config.general_info["azide_nucleophile"]:
            angle_atom_list = config.general_info["azide_angle"]
            angle_value = 180
            self.xtb.add_angle_constraint(angle_atom_list, angle_value)

    def check_electronic_temperature(self):
        """Check the minimum electron temperature along the scan
        
        Returns:
            min_temp (float): The minimum temperature along the scan.
        """
        min_temp = min(self.hl_gaps)

        return min_temp

    def find_peaks(self, prominence=0.01):
        """Finds the peaks along the scan.
        
        Args:
            prominence (float): Threshold for peak height (kcal/mol)
        """
        # Use DFT energies if available.
        if self.dft_energies:
            energies = self.dft_energies
        else:
            energies = self.xtb_energies
        
        # Find the peaks with scipy.
        energies = np.array(energies)
        maxima, properties = find_peaks(energies, prominence=prominence)
        minima, _ = find_peaks(-energies, prominence=prominence)
        left_bases = properties["left_bases"]
        right_bases = properties["right_bases"]
        prominences = properties["prominences"]

        # Select points half-way to the minima
        peak_list = []
        for maximum, left_base, right_base, prominence in zip(maxima, left_bases, right_bases, prominences):
            # Set left base
            minima_left = minima[minima < maximum]
            if any(minima_left):
                left_base = minima_left[-1]

            # Set right base
            minima_right = minima[minima > maximum]
            if any(minima_right):
                right_base = minima_right[0]

            # Make sure that there is at least one point distance between the peak and the refinement
            point_1 = left_base + int((maximum - left_base) / 2)
            if maximum - point_1 < 2:
                if maximum - 2 >= left_base:
                    point_1 = maximum - 2
            point_2 = right_base - int((right_base - maximum) / 2)
            if point_2 - maximum < 2:
                if maximum + 2 <= right_base:
                    point_2 = maximum + 2
            energy = self.dft_energies[maximum]

            # Add peaks to list
            peak = Peak(left_base, point_1, maximum, point_2, right_base, energy, prominence)
            peak_list.append(peak)

        self.peaks = peak_list

    def validate_peaks(self, intermediate=False, threshold=0.5):
        """Checks if bond order changes by more than threshold for TS.
        If not, peak is considered non-reactive and is ignored

        Args:
            intermediate (bool): Whether and intermediate exists or not.
            threshold (float): Threshold for considering bond changes.
        """
        remove_list = []
        for peak in self.peaks:
            # Check bond order criterion
            bo_nu_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.nu_atom)
            bo_nu_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.nu_atom)
            bo_nu_diff = abs(bo_nu_left - bo_nu_right)

            bo_lg_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.lg_atom)
            bo_lg_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.lg_atom)
            bo_lg_diff = abs(bo_lg_left - bo_lg_right)

            if (bo_nu_diff < 0.05) and (bo_lg_diff < 0.05):
                remove_list.append(peak)
                logger.info(f"Removed peak at {peak.maximum} due to bond order criterion")
            
            # Check intermediate + energy criterion
            test_energies = self.dft_energies[0:peak.left_base + 1]
            
            #  Test for strictly decreasing and no intermediate
            if peak.left_base != 0 and bo_lg_left > 0.5:
                if all(x>y for x, y in zip(test_energies, test_energies[1:])) and not intermediate:
                    remove_list.append(peak)
                    logger.info(f"Removed peak at {peak.maximum} due to no intermediate + energy strictly decreasing")

        for peak in remove_list:
            self.peaks.remove(peak)

    def make_plot(self):
        """Makes a plot of the peaks and their start and stop points for refinement"""
        
        # Draw the plot 
        x = range(1, len(self.dft_energies) + 1)
        plt.plot(x, self.dft_energies, '-o', label="DFT")
        plt.plot(x, self.xtb_energies, '--o', markerfacecolor="none", label="XTB")
        plt.legend()

        for peak in self.peaks:
            plt.plot(peak.maximum + 1, peak.energy, 'o', color='red', markersize=40, alpha=0.5)
        
        # Save the plot and clear the current figure
        plt.savefig("GSM.png")
        plt.clf()

    def constrain_bond(self, atom_1, atom_2, value):
        """Add bond constraint.
        
        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            value (float): Length of bond
        """
        self.xtb.add_constraint(atom_1, atom_2, value)

    def add_fragment(self, fragment):
        """Constrain fragment.

        Args:
            fragment (list): Atom indices of fragment.
        """
        self.xtb.add_fragment(fragment)

    def add_scan(self, atom_1, atom_2, value, start, stop, steps):
        """Add a scan to the calculator.
        
        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            value (int or str): Value of the constraint. 'auto' for xtb scan.
            start (float): Start value of scan
            stop (float): Stop value of scan
            steps (int): Number of steps in scan.
        """
        scan = BondScan(atom_1, atom_2, value, start, stop, steps)
        self.xtb.add_scan(scan)

    def add_fixed(self, atom_list):
        """Add fix atom constraints to scan.
        
        Args:
            atom_list (list): Indices of atoms to fix.
        """
        self.xtb.add_fixed(atom_list)

    def read_scan_output(self):
        """Read the GSM output and store the geometries and energies"""
        # Move file
        if os.path.isfile("xtbscan.log"):
            shutil.move("xtbscan.log", "scan.xyz")
        
        # Read geometries to ASE objects.
        self.geometries = [geometry for geometry in ase.io.iread("scan.xyz")]
        for geometry in self.geometries:
            geometry.info["charge"] = self.atoms.info["charge"]
        
        # Parse energier and convert to kcal/mol.
        parser = XTBParser("xtb.out")
        energies = parser.energy
        energies = [(energy - energies[0]) * HARTREE_TO_KCAL for energy in energies]
        self.xtb_energies = energies

        # Parse HOMO-LUMO gaps.
        hl_gaps = [lumo - homo for homo, lumo in zip(parser.homo, parser.lumo)] 
        self.hl_gaps = hl_gaps

    def read_sp_output(self):
        """Read DFT single point output and store the energies"""

        # Parse energies and NBO data. Convert energies to kcal/mol
        dft_energies = []
        nbo_data = []
        for i in range(1, len(self.geometries) + 1):
            nbo = NBOParser(f"sps/{i}.log")
            nbo_data.append(nbo)

            data = cclib.io.ccread(f"sps/{i}.log")
            energy = data.scfenergies[-1] * EV_TO_KCAL
            dft_energies.append(energy)

        # Reference energies to first point of the scan.
        normalized_energies = [energy - dft_energies[0] for energy in dft_energies]
        self.dft_energies = normalized_energies
        self.nbo_data = nbo_data

    def run_sps(self, n_procs, mem):
        """Run DFT single point calculations
        
        Args:
            n_procs (int): Number of processors
            mem (float): Memory (GB)
        """
        # Set up directory and jobs
        os.mkdir("sps")
        dft_options = self.g16.options
        g16_list = [G16Calculator(atoms, file=f"sps/{counter + 1}.gjf", options=dft_options) for counter, atoms in enumerate(self.geometries)]

        # Calculate resources for each job.
        n_calcs = len(g16_list)
        n_procs_job = max(n_procs // n_calcs, 1)
        n_jobs_simul = n_procs // n_procs_job
        mem_job = mem / n_procs * n_procs_job
  
        # Run jobs in parallel
        Parallel(n_jobs=n_jobs_simul, prefer="threads")(delayed(single_point_job)(g16, n_procs_job, mem_job) for g16 in g16_list)

    def run_scan(self, n_procs=1):
        """Run scan.

        Args: 
            n_procs (int): Number of processors

        Returns:
            process (object): Popen process object of the calculation.
        """
        process = self.xtb.opt(n_procs)
        return process

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.atoms.symbols}')"


class GICTSScan:
    """Run TS scan with GICs using Gaussian.

    Args:
        atoms (object): ASE atoms object
        xtb_options (dict): Options for the XTB calculations
        dft_options (dict): Options for the DFT calculations
        general_options (dict): General information needed for the scan.

    Attributes:
        atoms (object): ASE atoms object
        geometries (list): ASE atoms objects for optimized points along the scan.
        xtb_energies (list): XTB energies along the scan (kcal/mol).
        dft_energes (list): DFT energies along the scan (kcal/mol).
        nbo_data (list): NBO parsers for each point along the scan.
        hl_gaps (list): HOMO-LUMO gaps for all optimization steps (eV).
        xtb (object): Parent xtb calculator.
        g16 (object): Parent G16 calculator.
        peaks (list): Peaks along the scan.
        central_atom (int): Index of central atom
        nu_atom (int): Index of nu atom.
        lg_atom (int): Index of leaving atom. 
    """
    def __init__(self, atoms, xtb_options, dft_options, general_options):
        # Set up attributes
        self.atoms = atoms
        self.geometries = []
        self.xtb_energies = []
        self.dft_energies = []
        self.nbo_data = []
        self.hl_gaps = []
        self.peaks = []
        self.central_atom = general_options["central_atom"]
        self.nu_atom = general_options["nu_atom"]
        self.lg_atom = general_options["lg_atom"]        

        # Set up calculators
        self.xtb = XTBCalculator(atoms, options=xtb_options)
        self.g16 = G16Calculator(atoms, options=dft_options)

        # Set up scan-specific options for calculations
        self.g16.options["int_acc"] = "fine"
        self.g16.options["scf_acc"] = "sleazy"
        self.g16.options["nbo"] = True
        self.g16.options["chk"] = False
        self.g16_gic = G16Calculator(atoms, "scan.gjf")
        self.g16_gic.options["opt_acc"] = "loose"
        self.g16_gic.options["nosymm"] = True

    def check_electronic_temperature(self):
        """Get the minimum electronic temperature along the scan.

        Returns:
            min_temp (float): Minimum electronic temperature along the scan.
        """
        min_temp = min(self.hl_gaps)

        return min_temp

    def find_peaks(self, prominence=0.01):
        """Finds the peaks along the scan.

        Args:
            prominence (float): Threshold for peak height.
        """
        # Use DFT energies if possible
        if self.dft_energies:
            energies = self.dft_energies
        else:
            energies = self.xtb_energies

        # Find peaks with scipy.
        energies = np.array(energies)
        maxima, properties = find_peaks(energies, prominence=prominence)
        minima, _ = find_peaks(-energies, prominence=prominence)
        left_bases = properties["left_bases"]
        right_bases = properties["right_bases"]
        prominences = properties["prominences"]

        # Select points half-way to the minima
        peak_list = []
        for maximum, left_base, right_base, prominence in zip(maxima, left_bases, right_bases, prominences):
            # Set left base
            minima_left = minima[minima < maximum]
            if any(minima_left):
                left_base = minima_left[-1]

            # Set right base
            minima_right = minima[minima > maximum]
            if any(minima_right):
                right_base = minima_right[0]

            # Make sure that there is at least one point distance between the peak and the refinement
            point_1 = left_base + int((maximum - left_base) / 2)
            if maximum - point_1 < 2:
                if maximum - 2 >= left_base:
                    point_1 = maximum - 2
            point_2 = right_base - int((right_base - maximum) / 2)
            if point_2 - maximum < 2:
                if maximum + 2 <= right_base:
                    point_2 = maximum + 2
            energy = self.dft_energies[maximum]

            # Add peaks to list
            peak = Peak(left_base, point_1, maximum, point_2, right_base, energy, prominence)
            peak_list.append(peak)

        self.peaks = peak_list

    def validate_peaks(self, intermediate=False, threshold=0.5):
        """Checks if bond order changes by more than threshold for TS.
        If not, peak is considered non-reactive and is ignored

        Args:
            intermediate (bool): Whether an intermediate exists or not.
            threshold (float): Threshold for considering bond changes.
        """
        remove_list = []
        for peak in self.peaks:
            # Check bond order criterion
            bo_nu_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.nu_atom)
            bo_nu_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.nu_atom)
            bo_nu_diff = abs(bo_nu_left - bo_nu_right)

            bo_lg_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.lg_atom)
            bo_lg_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.lg_atom)
            bo_lg_diff = abs(bo_lg_left - bo_lg_right)

            if (bo_nu_diff < 0.05) and (bo_lg_diff < 0.05):
                remove_list.append(peak)
                logger.info(f"Peak at {peak.maximum} fails bond order criterion")
            
            # Check intermediate + energy criterion
            test_energies = self.dft_energies[0:peak.left_base + 1]
            
            #  Test for strictly decreasing and no intermediate
            if peak.left_base != 0 and bo_lg_left > 0.5:
                if all(x>y for x, y in zip(test_energies, test_energies[1:])) and not intermediate and peak.energy < -4.0:
                    remove_list.append(peak)
                    logger.info(f"Peak at {peak.maximum} fails criterion: no intermediate + energy strictly decreasing")

        for peak in remove_list:
            if peak in self.peaks:
                logger.info(f"Peak at {peak.maximum} removed.")
                self.peaks.remove(peak)

    def make_plot(self):
        """Makes a plot of the peaks and their start and stop points for refinement"""
        # Make the plot
        x = range(1, len(self.dft_energies) + 1)
        plt.plot(x, self.dft_energies, '-o', label="DFT")
        plt.plot(x, self.xtb_energies, '--o', markerfacecolor="none", label="XTB")
        plt.legend()

        # Add positions of peaks.
        for peak in self.peaks:
            plt.plot(peak.maximum + 1, peak.energy, 'o', color='red', markersize=40, alpha=0.5)
            plt.axvline(peak.left_base + 1, color="red", alpha=0.5)
            plt.axvline(peak.right_base + 1, color="red", alpha=0.5)
        
        # Save plot and clear figure.
        plt.savefig("GSM.png")
        plt.clf()

    def constrain_bond(self, name, atom_1, atom_2, value):
        """Add bond constraint to scan
        
        Args:
            name (str): Name of type of constraint for Gaussian 16.
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            value (str): Value of constraint, e.g., 'freeze'.
        """
        constraint = GICConstraint(name=name, atoms=(atom_1, atom_2), value=value)
        self.g16_gic.add_gic_constraint(constraint)

    def constrain_angle(self, name, atom_1, atom_2, atom_3, value):
        """Add angle constraint to scan
        
        Args:
            name (str): Gaussian 16 name of constraint
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            atom_3 (int): Index of atom 3 (1-indexed)
            value (str): Value of constraint, e.g., 'freeze'
        """
        constraint = GICConstraint(name=name, atoms=(atom_1, atom_2, atom_3), value=value)
        self.g16_gic.add_gic_constraint(constraint)
    
    def constrain_definition(self, name, definition, value):
        """Add GIC definition constraint to scan
        
        Args:
            name (str): Name of constraint
            definition (str): GIC definition of constraint.
            value (str): Value of constraint, e.g., 'freeze'
        """
        constraint = GICConstraint(name=name, definition=definition, value=value)
        self.g16_gic.add_gic_constraint(constraint)        

    def add_scan(self, name, n_steps, step_size, definition):
        """Add GIC scan.

        Args:
            name (str): Name of scan
            n_steps (int): Number of steps in scan
            step_size (float): Step size of scan
            definition (str): GIC definition of which internal coordinate to scan.
        """
        self.g16_gic.add_gic_scan(name, n_steps, step_size, definition)

    def read_scan_output(self):
        # Extract geometries along the scan.
        parser = GICParser("scan.log")
        positions = parser.geometries
        geometries = [ase.Atoms(symbols=self.atoms.get_chemical_symbols(), positions=position) for position in positions]
        for atoms in geometries:
            atoms.info["charge"] = self.atoms.info["charge"]
        self.geometries = geometries

        # Write xyz file with geometries
        ase.io.write("scan.xyz", geometries, plain=True)

        # Parse XTB energies and convert to kcal/mol.
        energies = parser.energies
        energies = [(energy - energies[0]) * HARTREE_TO_KCAL for energy in energies]
        self.xtb_energies = energies

        # Parse HOMO-LUMO gaps along the scan.
        hl_gaps = []
        for filename in glob.glob(".xtb_*.out"):
            parser = XTBParser(filename)
            hl_gap = parser.lumo - parser.homo
            hl_gaps.append(hl_gap)
        self.hl_gaps = hl_gaps

    def read_sp_output(self):
        """Read DFT single point output and store the energies"""
        # Parse the energies and the NBO data.
        dft_energies = []
        nbo_data = []
        for i in range(1, len(self.geometries) + 1):
            nbo = NBOParser(f"sps/{i}.log")
            nbo_data.append(nbo)

            data = cclib.io.ccread(f"sps/{i}.log")
            energy = data.scfenergies[-1] * EV_TO_KCAL
            dft_energies.append(energy)

        # Reference energies to first step of the scan
        normalized_energies = [energy - dft_energies[0] for energy in dft_energies]
        self.dft_energies = normalized_energies
        self.nbo_data = nbo_data

    def run_sps(self, n_procs, mem):
        """Run DFT single point calculations.
        
        Args:
            n_procs (int): Number of processors
            mem (float): Memory (GB)
        """
        # Create directory and set up calculations
        os.mkdir("sps")
        dft_options = self.g16.options
        g16_list = [G16Calculator(atoms, file=f"sps/{counter + 1}.gjf", options=dft_options) for counter, atoms in enumerate(self.geometries)]

        # Determine resources for each job.
        n_calcs = len(g16_list)
        n_procs_job = max(n_procs // n_calcs, 1)
        n_jobs_simul = n_procs // n_procs_job
        mem_job = mem / n_procs * n_procs_job
  
        # Run jobs in parallel
        Parallel(n_jobs=n_jobs_simul, prefer="threads")(delayed(single_point_job)(g16, n_procs_job, mem_job) for g16 in g16_list)

    def run_scan(self, n_procs, mem):
        """Run the GIC scan calculation.
        
        Args:
            n_procs (int): Number of processors
            mem (float): Memory (GB)
        
        Returns:
            process (object): Popen process object of the scan calculation.
        """
        xtb_string = self.xtb.get_submit_string()
        self.g16_gic.options["external"] = xtb_string
        process = self.g16_gic.opt(n_procs, mem)

        return process

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.atoms.symbols}')"


class Constraint(NamedTuple):
    """A constraint for use either with xtb or g16."""
    atom_1: int
    atom_2: int
    value: Union[float, str]


class GICConstraint(NamedTuple):
    """A GIC constraint for g16"""
    name: str
    value: str
    atoms: tuple = None
    definition: str = None


class GICScan(NamedTuple):
    """A GIC scan for g16"""
    name: str
    n_steps: int
    step_size: float
    definition: str


class AngleConstraint(NamedTuple):
    """A constraint for use either with xtb or g16."""
    atom_1: int
    atom_2: int
    atom_3: int
    value: float


class DihedralConstraint(NamedTuple):
    """A constraint for use either with xtb or g16."""
    atom_1: int
    atom_2: int
    atom_3: int
    atom_4: int
    value: float


class BondScan(NamedTuple):
    """A constraint for use either with xtb or g16."""
    atom_1: int
    atom_2: int
    value: float
    start: float
    stop: float
    steps: int


class Peak(NamedTuple):
    """A class to hold peak data for GSMs."""
    left_base: int
    start: int
    maximum: int
    stop: int
    right_base: int
    energy: float
    prominence: float


class Calculator:
    """Base calculator class.

    Args:
        atoms (object): ASE Atoms object
        file (str): Name of input file

    Attributes:
        atoms (object): ASE Atoms object
        angle_constraints (list): Optimization constraints for angles
        contraints (list): Optimization constraints for bonds
        file (str): File name
        options (dict): Calculation options
    """
    def __init__(self, atoms=None, file=None):
        # Set attributes.
        self.file = file
        self.atoms = atoms
        self.options = {}
        self.constraints = []
        self.angle_constraints = []

    def set_options(self, options):
        """Sets the options dictionary from options file.
        
        Args:
            options (dict): New options.
        """
        for key, value in options.items():
            self.options[key] = value

    def add_constraint(self, atom_1, atom_2, distance):
        """Sets bond constraint.

        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            distance (str or float): Constraint value.
        """
        constraint = Constraint(atom_1, atom_2, distance)
        self.constraints.append(constraint)

    def add_angle_constraint(self, atom_1, atom_2, atom_3, angle):
        """Sets angle constraint

        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            atom_3 (int): Index of atom 3 (1-indexed)
            angle (str or float): Constraint value.
        """
        constraint = AngleConstraint(atom_1, atom_2, atom_3, angle)
        self.angle_constraints.append(constraint)

    def single_point(self, *args, **kwargs):
        """Perform single point calculation
        
        Args:
            *args: Arguments passed to calculator
            **kwargs: Keyword arguments passed to calculator.
        
        Returns:
            process (object): POpen process object for running calculation.
        """
        # Set calculator options for single point.
        self.options["opt"] = False
        self.options["freq"] = False
        self.options["ts"] = False

        # Run calculation
        process = self.run_calc(*args, **kwargs)

        return process

    def opt(self, *args, **kwargs):
        """Perform geometry optimization.
        
        Args:
            *args: Arguments passed to calculator
            **kwargs: Keyword arguments passed to calculator.
        
        Returns:
            process (object): POpen process object for running calculation.
        """
        # Set calculator options for optimization.
        self.options["opt"] = True
        self.options["freq"] = False
        self.options["ts"] = False

        # Run calculation
        process = self.run_calc(*args, **kwargs)

        return process

    def opt_freq(self, *args, **kwargs):
        """Perform geometry optimization and subsequent frequency calculation.
        
        Args:
            *args: Arguments passed to calculator
            **kwargs: Keyword arguments passed to calculator.
        
        Returns:
            process (object): POpen process object for running calculation.
        """
        # Set calculator options for opt+freq calculation.
        self.options["opt"] = True
        self.options["freq"] = True
        self.options["ts"] = False

        # Run calculation
        process = self.run_calc(*args, **kwargs)

        return process

    def freq(self, *args, **kwargs):
        """Perform frequency calculation.
        
        Args:
            *args: Arguments passed to calculator
            **kwargs: Keyword arguments passed to calculator.
        
        Returns:
            process (object): POpen process object for running calculation.
        """
        # Set calculator options for frequency calculation
        self.options["opt"] = False
        self.options["freq"] = True
        self.options["ts"] = False

        # Run calculation
        process = self.run_calc(*args, **kwargs)

        return process

    def write_xyz_input(self, filename):
        """Write an xyz file as input for the xtb calculation.

        Args:
            filename (str): Name of xyz file to write.
        """
        ase.io.write(filename, self.atoms, plain=True)
        self.file = filename

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.atoms.symbols}')"


class G16Calculator(Calculator):
    """Calculator for performing Gaussian 16 calculations.

    Args:
        options (dict): Initital options
        atoms (object): ASE Atoms object
        file (str): Name of input file.

    Attributes:
        options (dict): Calculation options
        atoms (object): ASE Atoms object
        file (str): Name of input file.
        output (str): Name of output file.
        iops (list): Iops to use in the calculation.
        mod_connectivity (list): Strings used to modify the connectivity matrix.
        gic_constraints (list): GIC-type constraints
        gic_scans (list): GIC scans.        
    """
    def __init__(self, atoms=None, file=None, options=None):
        # Initialize from parent class
        super().__init__(atoms, file)

        # Set attributes
        self.options = {"solvent":          None,
                        "basis_set":        None,
                        "ecp":              None,
                        "charge":           None,
                        "functional":       None,
                        "dispersion":       None,
                        "opt_max_step":     None,
                        "int_acc":          None,
                        "opt_cycles":       None,
                        "opt_acc":          None,
                        "scf_acc":          None,
                        "solvation_model":  None,
                        "nbo":              None,
                        "hirshfeld":        None,
                        "calculation":      None,
                        "read_fc":          None,
                        "calc_fc":          None,
                        "read_wf":          None,
                        "opt":              None,
                        "freq":             None,
                        "ts":               None,
                        "epn":              None,
                        "wfx":              None,
                        "brinck":           None,
                        "recalc":           None,
                        "chk":              True,
                        "oldchk":           None,
                        "no_incore":        None,
                        "external":         None,
                        "xqc":              None,
                        "nosymm":           None,
                        }
        if options:
            self.set_options(options)
        self.options["charge"] = atoms.info["charge"]
        if self.file:
            prefix = os.path.splitext(self.file)[0]
            self.output = prefix + ".log"
        else:
            self.file = "g16.gjf"
            self.output = "g16.log"
        self.gic_constraints = []
        self.gic_scans = []
        self.mod_connectivity = []
        self.iops = []

    def add_gic_constraint(self, constraint):
        """Add constraint of GIC type.

        Args:
            constraint (object): GIC constraint object.
        """
        self.gic_constraints.append(constraint)
    
    def add_gic_scan(self, name, n_steps, step_size, definition):
        """Add GIC scan.

        Args:
            name (str): Name of constraint.
            n_steps (int): Number of steps
            step_size (float): Step size
            definition (str): Gaussian GIC definition string.
        """
        scan = GICScan(name, n_steps, step_size, definition)
        self.gic_scans.append(scan)

    def get_input_string(self):
        """Constructs the content of the G16 input file

        Returns:
            input_string (str)  :   Constructed G16 input as string
        """
        options = self.options
        ecp = False
        external_basis_set = False

        # Check for need of ECP
        basis_set = options['basis_set']
        if basis_set:
            gaussian_ecps = ["Def2SV", "Def2SVP", "Def2SVPP", "Def2TZV", "Def2TZVP", "Def2TZVPP", 
                             "Def2QZV", "Def2QZVP", "Def2QZVPP", "QZVP", "SDD", "SHC", "CEP-4G",
                             "CEP-31G", "CEP-121G", "LanL2MB", "LanL2DZ", "SDDAll"]
            external_basis = ["def2-svp", "def2-svpd", "def2-tzvp", "def2-tzvpd"]
            gaussian_ecps = [ecp.lower() for ecp in gaussian_ecps]
            
            
            if basis_set.lower() in external_basis:
                basis_dict = get_basis(options['basis_set'])
                ecp_dict = get_ecp(options['basis_set'])
                external_basis_set = True
            if basis_set.lower() not in gaussian_ecps:
                basis_dict = get_basis(options['ecp'])
                ecp_dict = get_ecp(options['ecp'])
            
            symbol_list = [atom.symbol for atom in self.atoms]
            ecp_list = set([symbol.capitalize() for symbol in symbol_list if symbol.capitalize() in ecp_dict.keys()])
            non_ecp_list = set([symbol.capitalize() for symbol in symbol_list if symbol.capitalize() not in ecp_dict.keys()])
            if any(ecp_list):
                ecp = True

        # Make the header
        input_string = "# "
        if options["external"]:
            input_string += f"external='{config.interface_script} {options['external']}' "
        else:
            if options["functional"]:
                input_string += f"{options['functional']} "
            if options["basis_set"] and not ecp and not external_basis_set:
                input_string += f"{options['basis_set']} "
            elif ecp:
                input_string += f"genecp "
            elif external_basis_set:
                input_string += f"gen "
            if options["dispersion"]:
                input_string += f"empiricaldispersion={options['dispersion']} "
            if options["read_wf"]:
                input_string += "guess=read "
            if options["nbo"]:
                input_string += "pop=(nboread, always) "
            if options["hirshfeld"]:
                input_string += "pop=(hirshfeld, always) "
            if options["solvent"]:
                input_string += f"scrf(solvent={options['solvent']}, "
                if options["solvation_model"]:
                    input_string += f"{options['solvation_model']}, "
                input_string = input_string.strip(", ")
                input_string += ") "
            if options["scf_acc"]:
                input_string += f"scf={options['scf_acc']} "
            if options["xqc"]:
                input_string += "scf=xqc "
            if options["int_acc"]:
                input_string += f"integral={options['int_acc']} "
            if options["epn"]:
                input_string += "prop=potential "
            if options["wfx"]:
                input_string += "output=wfx density=current "
            if options["brinck"]:
                input_string += "6d GFINPUT IOP(6/7=3) "
            if options["no_incore"]:
                input_string += "SCF=NoInCore "

        if options["opt"]:
            input_string += "opt("
            if options["ts"]:
                input_string += "ts, noeigentest, "
            if self.gic_constraints or self.gic_scans:
                input_string += "addgic, "
            if options["external"]:
                input_string += "nomicro, "
            if self.constraints:
                input_string += "modredundant, "
            if options["read_fc"]:
                input_string += "rcfc, "
            if options["calc_fc"]:
                input_string += "calcfc, "
            if options["recalc"]:
                input_string += f"recalc={options['recalc']}, "
            if options["opt_acc"]:
                input_string += f"{options['opt_acc']}, "
            if options["opt_cycles"]:
                input_string += f"maxcycles={options['opt_cycles']}, "
            if options["opt_max_step"]:
                input_string += f"maxstep={options['opt_max_step']}, "
            input_string = input_string.strip(", ")
            input_string += ") "
            
            if options["nosymm"]:
                input_string += "nosymm "

        if options["freq"]:
            input_string += "freq "
        
        if self.iops:
            for iop in self.iops:
                input_string += f"IOp({iop}) "
        
        if self.mod_connectivity:
            input_string += "geom=modconnectivity "

        input_string += "\n"

        # Make title charge and multiplicity
        input_string += "\n"
        input_string += "predict_snar calculation\n"
        input_string += "\n"

        # Set multiplicity to singlet for even electrons, doublet for odd
        n_electrons = sum(self.atoms.numbers) + options['charge']

        if n_electrons % 2 == 0:
            mult = 1
        else:
            mult = 2

        input_string += f"{options['charge']} {mult}\n"

        # Add coordinates
        for atom in self.atoms:
            input_string += f"{atom.symbol:10s}{atom.position[0]:12.6f}{atom.position[1]:12.6f}{atom.position[2]:12.6f}\n"
        input_string += "\n"

        # Add footer
        if self.mod_connectivity:
            for entry in self.mod_connectivity:
                input_string += f"{entry}\n"
            input_string += "\n"

        if self.gic_constraints or self.gic_scans:
            for constraint in self.gic_constraints:
                c_str = f"{constraint.name}"
                if constraint.value:
                    c_str += f"({constraint.value})"
                c_str += "="
                if constraint.atoms:
                    if len(constraint.atoms) == 2:
                        char = "R"
                    elif len(constraint.atoms) == 3:
                        char = "A"
                    elif len(constraint.atoms) == 4:
                        char = "D"           
                    c_str += f"{char}({','.join([str(atom) for atom in constraint.atoms])})\n"
                elif constraint.definition:
                    c_str += f"({constraint.definition})\n"
                input_string += c_str
            
            for scan in self.gic_scans:
                input_string += f"{scan.name}(NSteps={scan.n_steps},StepSize={scan.step_size})=({scan.definition})\n"
        elif self.constraints:
            for constraint in self.constraints:
                input_string += f"B {constraint.atom_1} {constraint.atom_2} {constraint.value}\n"
            input_string += "\n"

        if ecp or external_basis_set:
            # Write all atoms without ECPs
            if len(non_ecp_list) > 0:
                if external_basis_set:
                    for symbol in non_ecp_list:
                        input_string += "".join(basis_dict[symbol])
                        input_string += "****\n"
                else:
                    symbol_string = " ". join(set(non_ecp_list))
                    input_string += symbol_string + " 0\n"
                    input_string += f"{options['basis_set']}\n"
                    input_string += "****\n"

            # Write basis for atoms with ECPs
            if ecp:
                for symbol in ecp_list:
                    input_string += "".join(basis_dict[symbol])
                    input_string += "****\n"
            input_string += "\n"

            # Write ECPs for atoms with ECPs
            if ecp:
                for symbol in ecp_list:
                    input_string += "".join(ecp_dict[symbol])
                input_string += "\n"

        if options["nbo"]:
            input_string += "$nbo bndidx $end\n"
            input_string += "\n"
        if options["wfx"]:
            prefix = os.path.splitext(self.file)[0]
            input_string += f"{prefix}.wfx"
            input_string += "\n"

        return input_string

    def run_calc(self, n_procs, mem):
        """Runs the G16 calculation.

        Args:
            n_procs (int)   :   Number of processors
            mem (float)       :   Memory in GB
        
        Returns:
            process (object): POpen process object of running calculation.
        """
        # Set resources.
        self.n_procs = n_procs
        self.mem = mem

        # Write input file.
        input_string = self.get_input_string()
        prefix = os.path.splitext(self.file)[0]
        with open(self.file, 'w') as file:
            file.write(input_string)

        # Reduce memory somewhat so that g16 is stable
        mem = round(mem * 0.9 * 1024)

        # Set up command line string for the calculation.
        submit_string = f"g16 -p={n_procs} -m={mem}MB "
        if self.options["chk"]:
            submit_string += f"-y={prefix}.chk "
        if self.options["oldchk"]:
            submit_string += f"-ic={self.options['oldchk']} "
        submit_string += f"{self.file}"

        # Run the calculation.
        error_file = open(prefix + ".err", "w")
        process = subprocess.Popen(submit_string.split(), stdout=error_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        error_file.close()
        self.process = process

        return process

    def restart(self):
        """Restart a previous calculation with the same settings."""
        self.run_calc(self.n_procs, self.mem)

    def ts(self, *args, **kwargs):
        """Optimize a TS
        
        Args:
            *args: Arguments to the calculator
            **kwargs: Keyword arguments to the calculator.

        Returns:
            process (object): POpen process object of running calculation.
        """
        # Set options appropriate for TS calculation
        self.options["opt"] = True
        self.options["ts"] = True
        self.options["freq"] = False
        if not self.options["read_fc"]:
            self.options["calc_fc"] = True
        
        # Run calculation
        process = self.run_calc(*args, **kwargs)

        return process

    def ts_freq(self, *args, **kwargs):
        """Optimize a TS and do subsequent frequency calculation.
        
        Args:
            *args: Arguments to the calculator
            **kwargs: Keyword arguments to the calculator.

        Returns:
            process (object): POpen process object of running calculation.
        """
        # Set calculator options appropriate for TS+freq calculation.
        self.options["opt"] = True
        self.options["ts"] = True
        self.options["freq"] = True
        if not self.options["read_fc"]:
            self.options["calc_fc"] = True
        
        # Run calculation
        process = self.run_calc(*args, **kwargs)

        return process


class XTBCalculator(Calculator):
    """Calculator for performing xtb calculations with and without fragments.

    Args:
        file (str): Input filename
        options (dict): Initital options
        atoms (object): ASE atoms object

    Attributes:
        atoms (object): ASE Atoms object
        contraints (list): Optimization constraints for bonds
        constrained_atoms (set): Atoms to constrain during optimization
        fixed_atoms (set): Atoms to fix during optimization.
        fragments (set): Fragments for constrained optimization
        file (str): Name of input file
        options (dict): Calculation options
        output (str): Name of output file.
        xcontrol (bool): Toggles use of xcontrol with constraints
    """
    def __init__(self, atoms=None, file=None, options=None):
        # Initialize from parent
        super().__init__(atoms, file)
        
        # Set attributes
        self.scans = set()
        self.dihedral_constraint = None
        self.angle_constraint = None
        self.fragments = set()
        self.fixed_atoms = set()
        self.constrained_atoms = set()
        self.options = {"solvent":              None,
                        "el_temp":              300,
                        "charge":               None,
                        "gfn_version":          None,
                        "force_constant":       0.05,
                        "fragment_distance":    3.0,
                        "opt":                  None,
                        "maxdispl":             None,
                        "grad":                 None,
                        "freq":                 None,
                        "opt_cycles":           None,
                        "lmo":                  None,
                        "uhf":                  None,
                        "verbose":              True,
                        "temperature":          None,
                        "cma":                  None,
                        "center":               None,
                        "scan_mode":            "concerted",
                        }
        self.xcontrol = False
        self.output = "xtb.out"

        if options:
            self.set_options(options)
        self.options["charge"] = atoms.info["charge"]

    def add_fragment(self, fragment):
        """Set fragment constraint.
        
        Args:
            fragment (list): Indices of atoms in fragment.
        """
        self.fragments.add(tuple(fragment))

    def add_constraint(self, atom_1, atom_2, value):
        """Add bond constraint.

        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            value (float or str): Bond length (Å) or string, e.g. 'auto'
        """
        constraint = Constraint(atom_1, atom_2, value)
        self.constraints.append(constraint)

    def add_dihedral_constraint(self, atom_list, value):
        """Add dihedral constraint.

        Args:
            atom_list (list): Indices of atoms in dihedral (1-indexed)
            value (float, str): Dihedral angle (Å) or string, e.g., 'auto'
        """
        self.dihedral_constraint = DihedralConstraint(*atom_list, value)

    def add_angle_constraint(self, atom_list, value):
        """Add angle constraint.
        
        Args:
            atom_list (list): Atom indices in angle (1-indexed)
            value (float, str): Angle (Å) or string, e.g., 'auto'
        """
        self.angle_constraint = AngleConstraint(*atom_list, value)

    def add_scan(self, scan):
        """Add scan.

        Args:
            scan (object): Scan object
        """
        self.scans.add(scan)

    def add_fixed(self, atom_list):
        """Add fixed atoms.
        
        Args:
            atom_list (list): Atom indices to fix (1-indexed)
        """
        if type(atom_list) == list:
            self.fixed_atoms.update(atom_list)
        else:
            self.fixed_atoms.add(atom_list)

    def write_xcontrol(self):
        """Write xcontrol file with constraints etc."""
        with open('xcontrol', 'w') as file:
            # Write temperature
            if self.options["temperature"] and self.options["freq"]:
                file.write("$thermo\n")
                file.write(f"temp = {self.options['temperature']}\n")
            
            # Write optimization options
            file.write("$opt\n")           
            if self.options["opt_cycles"]:
                file.write(f"maxcycle={self.options['opt_cycles']}\n")
            if self.options["maxdispl"]:
                file.write(f"maxdispl={self.options['maxdispl']}\n")                
            if self.fixed_atoms:
                file.write("$fix\n")
                string = ", ".join([str(i) for i in self.fixed_atoms])
                wrapped_string = textwrap.wrap(string, break_long_words=False)
                for string in wrapped_string:                
                    file.write(f"atoms: {string}\n")
            
            # Write fragment information.
            if self.fragments:
                file.write("$split\n")
                for counter, fragment in enumerate(self.fragments, start=1):
                    fragment_string_list = []
                    fragment_string = ""
                    atom_counter = 0
                    for atom in fragment:
                        fragment_string += f"{atom}, "
                        atom_counter +=1
                        if atom_counter == 20:
                            fragment_string_list.append(fragment_string.strip(", "))
                            fragment_string = ""
                            atom_counter = 0
                    if fragment_string:
                        fragment_string_list.append(fragment_string.strip(", "))
                    for fragment_string in fragment_string_list:
                        file.write(f"fragment: {counter}, {fragment_string}\n")
            
            # Write constraints and scans.
            if self.constraints or self.fragments or self.dihedral_constraint or self.angle_constraint \
                    or self.options["cma"] or self.options["center"] or self.constrained_atoms:
                file.write("$constrain\n")
                if self.options["force_constant"]:
                    file.write(f"force constant={self.options['force_constant']:f}\n")
                if self.constraints:
                    for constraint in self.constraints:
                        file.write(f"distance: {constraint.atom_1}, {constraint.atom_2}, {constraint.value}\n")
                if self.angle_constraint:
                    file.write(f"angle: {self.angle_constraint.atom_1}, {self.angle_constraint.atom_2}, {self.angle_constraint.atom_3}, {self.angle_constraint.value}\n")
                if self.dihedral_constraint:
                    file.write(f"dihedral: {self.dihedral_constraint.atom_1}, {self.dihedral_constraint.atom_2}, {self.dihedral_constraint.atom_3}, {self.dihedral_constraint.atom_4}, {self.dihedral_constraint.value}\n")
                if self.fragments and not self.scans:
                    file.write(f"cma: {self.options['fragment_distance']}\n")
                if self.constrained_atoms:
                    string = ", ".join([str(i) for i in self.constrained_atoms])
                    wrapped_string = textwrap.wrap(string, break_long_words=False)
                    for string in wrapped_string:                
                        file.write(f"atoms: {string}\n")
                elif self.fragments and self.scans:
                    file.write("cma: auto\n")
                elif self.options["cma"]:
                    file.write(f"cma: {self.options['cma']:f}\n")
                elif self.options["center"]:
                    file.write(f"center: 0.000001, {self.options['center']}\n")
                if self.scans:
                    file.write("$scan\n")
                    if len(self.scans) > 1:
                        file.write(f"mode={self.options['scan_mode']}\n")
                    for scan in self.scans:
                        for counter, constraint in enumerate(self.constraints, start=1):
                            if scan.atom_1 == constraint.atom_1 and scan.atom_2 == constraint.atom_2:
                                file.write(f"{counter}: {scan.start}, {scan.stop}, {scan.steps}\n")

        self.xcontrol = True

    def run_calc(self, n_procs=1):
        """Submit xtb calculation
        
        Args:
            n_procs (int): Number of processors.

        Returns:
            process (object): POpen process object.
        """
        # Write xcontrol file.
        if any([self.constraints,
                self.fragments,
                self.options["temperature"],
                self.options["cma"],
                self.options["center"],
                ]
               ):
            self.write_xcontrol()
        
        # Set input file
        if not self.file:
            self.file ="xtb.xyz"

        # Get submission string.
        submit_string = self.get_submit_string(n_procs)
        submit_string = f"{config.xtb} {self.file} " + submit_string

        # Write input file 
        self.write_xyz_input(self.file)

        # Run calculation.
        out_file = open("xtb.out", "w")
        err_file = open("xtb.err", "w")
        process = subprocess.Popen(submit_string.split(), stdout=out_file, stderr=err_file, preexec_fn=os.setsid)
        out_file.close()
        err_file.close()

        return process

    def get_submit_string(self, n_procs=1):
        #TODO seems that n_procs is not used currently. But doesn't matter too much.
        """Construct the string to submit to the shell
        
        Args:
            n_procs (int): Number of processors 

        Returns:
            submit_string (str): Bash submit string for the calculation.
        """
        submit_string = ""

        # Set type of calculation.
        if self.options["opt"] and self.options["freq"]:
            calculation = "ohess"
        elif self.options["opt"]:
            calculation = "opt"
        elif self.options["freq"]:
            calculation = "hess"
        elif self.options["grad"]:
            calculation = "grad"
        else:
            calculation = "sp"

        submit_string += f"--{calculation} "

        # Set charge
        charge = self.options.get("charge")
        submit_string += f"--chrg {charge} "

        # Set electronic temperature
        el_temp = self.options.get("el_temp")
        if el_temp:
            submit_string += f"--etemp {el_temp} "

        # Set solvent
        solvent = self.options.get("solvent")
        if solvent:
            submit_string += f"--gbsa {solvent} "

        # Set GFN version
        gfn_version = self.options.get("gfn_version")
        if gfn_version:
            submit_string += f"--gfn {gfn_version} "

        # Set special
        if self.options["lmo"]:
            submit_string += "--lmo "

        if self.options["uhf"]:
            submit_string += f"--uhf {self.options['uhf']} "

        if self.xcontrol:
            submit_string += f"--input xcontrol "
        
        if self.options["verbose"]:
            submit_string += "--verbose "

        return submit_string


class CRESTCalculator:
    """CREST conformational sampling calculator.
    
    Args:
        atoms (object): ASE Atoms object
        xtb_options (dict): Options for XTB
        crest_options (dict): Options for CREST
    
    Attributes:
        options (dict): Calculator options.
        atoms (object): ASE Atoms object.
        constraints (list): Bond constraints.
        constrained_atoms (set): Constrained atoms
        conformers (list): Conformers as ASE atoms objects.
        weights (list): Boltzmann weights for each conformer.
        degeneracies (list): Degeneracies for each conformer.
        energies (list): Energies for each conformer (a.u.)
        g16 (object): G16 calculator object for DFT single point evaluations.
        scratch_dir (str): Path to scratch directory
        submit_dir (str): Path to submission directory.
    """
    def __init__(self, atoms=None, xtb_options=None, crest_options=None):
        # Set default options.
        self.options = {"solvent": None,
                        "charge": None,
                        "gfn_version": None,
                        "zsort": False,
                        "force_constant": 2.0,
                        "energy_window": None,
                        "xnam": "xtb",
                        "nci": False,
                        "speed": "normal",
                        "mrest": None,
                        "el_temp": None,
                        }

        # Remove the electronic temperature from the xtb options.
        xtb_options = xtb_options.copy()
        xtb_options.pop('el_temp', None)

        # Set options and attributes.
        if xtb_options:
            self.set_options(xtb_options)
        if crest_options:
            self.set_options(crest_options)            
        self.options["charge"] = atoms.info["charge"]
        self.atoms = atoms
        self.constraints = []
        self.constrained_atoms = set()
        self.conformers = []
        self.weights = []
        self.degeneracies = []
        self.energies = []
        self.dft_energies = []
        self.g16 = None
        self.scratch_dir = None
        self.submit_dir = None
    
    def run_calc(self, n_procs=1):
        """Submit CREST calculation.
        
        Args:
            n_procs (int): Number of processors.
        
        Returns:
            process (object): Popen process object for running calculation.
        """
        # Write the Turbomole coordinate file
        self.write_coord_file()

        # Write xcontrol file
        if self.constraints or self.constrained_atoms:
            self.write_xcontrol()
        
        # Write reference coordinate file.
        if self.constrained_atoms or self.constraints:
            shutil.copyfile("coord", "coord.original")
        
        # Get submission string.
        submit_string = self.get_submit_string(n_procs)
        submit_string = config.crest + " " + submit_string

        # Set up environment and run calculation.
        env = os.environ.copy()
        out_file = open(self.submit_dir / "crest.out", "w")
        err_file = open(self.submit_dir / "crest.err", "w")
        if self.scratch_dir:
            env["TMPDIR"] = self.scratch_dir.resolve().as_posix()
        process = subprocess.Popen(submit_string.split(), stdout=out_file, stderr=err_file, preexec_fn=os.setsid, env=env)
        out_file.close()
        err_file.close()

        return process

    def set_options(self, options):
        """Sets the options dictionary from options file.
        
        Args:
            options (dict): Calculator options.
        """
        for key, value in options.items():
            if key in self.options.keys():
                self.options[key] = value

    def add_constraint(self, atom_1, atom_2, value):
        """Add bond constraint.
        
        Args:
            atom_1 (int): Index of atom 1 (1-indexed)
            atom_2 (int): Index of atom 2 (1-indexed)
            value (float): Bond length (Å)
        """
        constraint = Constraint(atom_1, atom_2, value)
        self.constraints.append(constraint)
    
    def get_submit_string(self, n_procs=1):
        """Get the string to submit to the shell.
        
        Args: 
            n_procs (int): Number of processors.

        Returns:
            submit_string (str): Submission string.
        """
        options = self.options

        submit_string = ""
        if options["gfn_version"]:
            submit_string += f"-gfn{options['gfn_version']} "
        if options["xnam"]:
            submit_string += f"-xnam {options['xnam']} " 
        if options["charge"]:
            submit_string += f"-chrg {options['charge']} "
        if options["solvent"]:
            submit_string += f"-g {options['solvent']} "
        if not options["zsort"]:
            submit_string += "-nozs "
        if options["nci"]:
            submit_string += "-nci "
        if options["speed"] == "quick":
            submit_string += "-quick "
        if options["speed"] == "squick":
            submit_string += "-squick "
        if options["mrest"]:
            submit_string += f"-mrest {options['mrest']} "
        if options["energy_window"]:
            submit_string += f"-ewin {options['energy_window']} "
        submit_string += f"-T {n_procs} "
        #!TODO Can use the inbuilt capacity from CREST for using scractch directory when they have fixed bugs.
        #if self.scratch_dir:
        #    submit_string += f"-scratch {self.scratch_dir.resolve().as_posix()} "
        submit_string += "-cinp xcontrol "

        return submit_string
    
    def rank_conformers(self, initial_dft_options, n_procs=1, mem=1):
        """Rank conformers based on DFT single point calculations.
        
        Args:
            initial_dft_options (dict): Options for DFT calculator.
            n_procs (int): Number of processors.
            mem (float): Memory (GB)
        """
        # Set up Gaussian calculator 
        self.g16 = G16Calculator(self.atoms, options=initial_dft_options)

        # Set particular options for quick evaluation.
        self.g16.options["int_acc"] = "fine"
        self.g16.options["scf_acc"] = "sleazy"
        self.g16.options["chk"] = False

        # Set up DFT jobs
        os.mkdir("sps")
        with cd("sps"):
            # Set up calculators from parent
            dft_options = self.g16.options
            g16_list = [G16Calculator(atoms, file=f"{counter + 1}.gjf", options=dft_options) for counter, atoms in enumerate(self.conformers)]

            # Determine resources for each job.
            n_calcs = len(g16_list)
            n_procs_job = max(n_procs // n_calcs, 1)
            n_jobs_simul = n_procs // n_procs_job
            mem_job = mem / n_procs * n_procs_job
    
            # Run jobs in parallel
            Parallel(n_jobs=n_jobs_simul, prefer="threads")(delayed(single_point_job)(g16, n_procs_job, mem_job) for g16 in g16_list)
            
            # Read DFT single point output and store the energies"""
            dft_energies = []
            for i in range(1, len(self.conformers) + 1):
                data = cclib.io.ccread(f"{i}.log")
                energy = data.scfenergies[-1] * EV_TO_HARTREE
                dft_energies.append(energy)
    
            self.dft_energies = dft_energies

        # Write results file
        with open("conformer_ranking", "w") as file:
            rel_dft_energies = [(energy - min(self.dft_energies)) * HARTREE_TO_KCAL for energy in self.dft_energies]
            rel_xtb_energies = [(energy - min(self.energies)) * HARTREE_TO_KCAL for energy in self.energies]
            file.write(f"{'Conformer':>18s}{'XTB (kcal/mol)':>18s}{'DFT (kcal/mol)':>18s}{'Degeneracy':>18s}\n")
            for i, (xtb_energy, dft_energy, degeneracy) in enumerate(zip(rel_xtb_energies, rel_dft_energies, self.degeneracies), start=1):
                file.write(f"{i:18d}{xtb_energy:18.6f}{dft_energy:18.6f}{degeneracy:18d}\n")

        # Reorder conformers based on dft energies
        self.dft_energies, self.conformers, self.energies, self.weights, self.degeneracies = \
            zip(*sorted(zip(self.dft_energies, self.conformers, self.energies, self.weights, self.degeneracies), key=lambda x: x[0]))

    def parse_conformers(self):
        """Parse the conformers from the CREST output file"""
        # Read the output file
        parser = CRESTParser("crest.out")
        
        # Store data. Convert energies to hartree.
        self.weights = parser.weights
        self.degeneracies = parser.degeneracies
        self.energies = np.array(parser.energies) * KCAL_TO_HARTREE

        # Read the geometries as ASE atoms objects.
        self.conformers = [conformer for conformer in ase.io.iread("crest_conformers.xyz")]
        for geometry in self.conformers:
            geometry.info["charge"] = self.atoms.info["charge"]

    def write_xcontrol(self):
        """Write the xcontrol file"""
        with open('xcontrol', 'w') as file:
            file.write("$constrain\n")
            if self.options["force_constant"]:
                file.write(f"force constant={self.options['force_constant']}\n")
            for constraint in self.constraints:
                file.write(f"distance: {constraint.atom_1}, {constraint.atom_2}, {constraint.value}\n")
            if self.constrained_atoms or self.constraints:
                file.write("reference=coord.original\n")
                all_atoms = set(range(1, self.atoms.get_number_of_atoms() + 1))
                if self.constraints:
                    constraints_atoms = []
                    for constraint in self.constraints:
                        constraints_atoms.extend([constraint.atom_1, constraint.atom_2])
                    constraints_atoms = set(constraints_atoms)
                    all_constrained_atoms = self.constrained_atoms | constraints_atoms
                else:
                    all_constrained_atoms = self.constrained_atoms
                free_atoms = all_atoms.difference(all_constrained_atoms)
                string = ", ".join([str(i) for i in self.constrained_atoms])
                wrapped_string = textwrap.wrap(string, break_long_words=False)
                for string in wrapped_string:
                    file.write(f"atoms: {string}\n")
                file.write("$metadyn\n")
                string = ", ".join([str(i) for i in free_atoms])
                wrapped_string = textwrap.wrap(string, break_long_words=False)
                for string in wrapped_string:                
                    file.write(f"atoms: {string}\n")
            if self.options["el_temp"]:
                file.write("$scc\n")
                file.write(f"temp={self.options['el_temp']}\n")
    
    def write_coord_file(self):
        """Write Turbomole style coord file"""
        symbols = self.atoms.get_chemical_symbols()
        coordinates = self.atoms.get_positions()
        coordinates = coordinates * ANGSTROM_TO_BOHR
        with open("coord", 'w') as file:
            file.write("$coord\n")
            for symbol, coordinate in zip(symbols, coordinates):
                file.write(f"{coordinate[0]:24.10f}{coordinate[1]:24.10f}{coordinate[2]:24.10f}{symbol:>4s}\n")
            file.write("$end\n")