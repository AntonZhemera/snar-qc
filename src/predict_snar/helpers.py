import contextlib
try:
    import fcntl
except:
    print("fcntl not avaiable on Windows.")
import logging
from pathlib import Path
import math
import numpy as np
import os
try:
    import resource
except:
    print("resource not available on Windows.")
import shelve
import signal
import time
import uuid

import ase
from ase import Atoms
import cclib
from mendeleev import element
from rdkit import Chem
from rdkit.Chem import AllChem


from predict_snar import config, results
from predict_snar.data import GoodVibes
from predict_snar.data import EV_TO_HARTREE, GAS_CONSTANT, AMU, PLANCK, BOLTZMANN, ATM, HARTREE, STANDARD_STATE_UNIT_CORRECTION, AVOGADRO_CONSTANT, MOL
from predict_snar.parsers import NBOParser, XTBParser, GaussianParser

# Set logger
logger = logging.getLogger("predict_snar")

def single_point_job(calculator, n_procs, mem):
    """Helper function to run single-point jobs in parallel
    
    Args:
        calculator (object): G16 calculator object.
        n_procs (int): Number of processors
        mem (float): Memory in GB
    """
    # Do single point with calculation monitor
    calculator.single_point(n_procs=n_procs, mem=mem)
    calculation_monitor(calculator)


@contextlib.contextmanager
def cd(new_directory):
    """Emulates the cd command in bash
    
    Args:
        new_directory (str): Name of directory.
    """
    current_dir = os.getcwd()
    os.chdir(new_directory)
    try:
        yield
    finally:
        os.chdir(current_dir)


@contextlib.contextmanager
def lock(file):
    """Context manager for locking a file to prevent concurrent access.

    Args:
        file (str): The name of the file to be locked (with or without suffix)
    """
    # Lock file
    file_path = Path(file)
    lock_path = file_path.with_suffix(".lock")
    lock_file = open(lock_path, "w")
    fcntl.lockf(lock_file, fcntl.LOCK_EX)
    try:
        yield
    finally:
        # Unlock file
        fcntl.lockf(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def get_electronic_temp(hl_gap, neutral_scan=False, separated=False):
    """Get the suggested xtb electronic temperature for a certain HOMO-LUMO gap
    of nucleophile and electrophile.

    Args:
        hl_gap (float): HOMO-LUMO gap (eV)
        neutral_scan (bool): Whether the temperature should be adjusted for a
            neutral TS scan
        separated (bool): Whether the HOMO-LUMO gap refers to separated 
            eactants.

    Returns:
        el_temp (float): Electronic temperature
    """
    # Get intervals depending on keyword arguments.
    if neutral_scan and separated:
        raise Exception("Give either neutral scan or separated.")
    if neutral_scan:
        limits = {
            (-np.inf, 1.0): 2000,
            (1.0, 2.0): 4000,
            (2.0, np.inf): 7000,
        }
    elif separated:
        limits = {
            (-np.inf, -0.5): 4000,
            (-0.5, np.inf): 7000,
        }
    else:
        limits = {
            (-np.inf, 2.0): 4000,
            (2.0, np.inf): 7000,
        }

    # Select temperature based on intervals
    for (low_lim, high_lim), cand_temp in limits.items():
        if hl_gap > low_lim and hl_gap <= high_lim:
            el_temp = cand_temp
    
    # Return temperature
    return el_temp


def set_electronic_temp(new_el_temp):
    """Set electronic temperature in the config module
    
    Args:
        new_el_temp (float): Suggested new electronic temperature.

    Returns
        changed (bool): Whether electronic temperature was changed.
    """
    # Get current electronic temperature.
    old_el_temp = config.xtb_options.get("el_temp", 7000)

    # Use the default value of 7000.
    if not old_el_temp:
        old_el_temp = 7000

    # Change to new electronic temperature if it is lower than the old.
    if new_el_temp < old_el_temp:
        logger.info(f"Switch to electronic temperature of {new_el_temp}.")
        config.xtb_options["el_temp"] = new_el_temp
        changed = True
    else:
        changed = False
    
    return changed


def get_results(calculator, standard_state=1):
    """Get the results from a DFT calculation on a structure with single-point
    at the TZ level.

    Args:
        calculator (object): Calculator object
        standard_state (float): Standard state to use for the free-energy

    Returns:
        opt_atoms (object): ASE Atoms object of the optimized structure.
    """
    # Read the data and create a new Atoms object.
    data = cclib.io.ccread(calculator.output)
    opt_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
    opt_atoms.info["charge"] = calculator.atoms.info["charge"]

    # Get free energies and enthalpies with Goodvibes according to Grimme's correction
    temperature = config.general_options["temperature"]
    gv = GoodVibes(calculator.output, temperature=temperature, scaling=1.0,
                   standard_state=standard_state, quasi_harmonic="grimme", cutoff=100)

    electronic_energy_dz = data.scfenergies[-1] * EV_TO_HARTREE
    enthalpy_dz = gv.enthalpy
    free_energy_dz = gv.free_energy
    free_energy_qh_grimme_dz = gv.free_energy_qh
    enthalpy_corr_dz = enthalpy_dz - electronic_energy_dz
    entropy_corr_dz = gv.t_entropy
    entropy_corr_qh_grimme_dz = gv.t_entropy_qh
    free_energy_corr_dz = free_energy_dz - electronic_energy_dz
    free_energy_corr_qh_grimme_dz = free_energy_qh_grimme_dz - electronic_energy_dz

    # Get free energies and enthalpies with Goodvibes according to Truhlar's correction
    gv = GoodVibes(calculator.output, temperature=temperature, scaling=1.0,
                   standard_state=standard_state, quasi_harmonic="truhlar", cutoff=100)  

    free_energy_qh_truhlar_dz = gv.free_energy_qh
    free_energy_corr_qh_truhlar_dz = free_energy_qh_truhlar_dz - electronic_energy_dz
    entropy_corr_qh_truhlar_dz = gv.t_entropy_qh

    # Do DFT SP with TZ
    dh_functionals = ["b2plyp", "mpw2plyp", "b2plypd", "b2blypd3", "dsdpbep86",
                      "pbe0dh", "pbeqidh"]
    os.mkdir("sp-tz")
    with cd("sp-tz"):
        # Set the ultrafinegrid explicitly as the optimization might have used
        # the superfinegrid.
        calculator.options["int_acc"] = "UltraFineGrid"
        
        # Use checkpoint file if there is one.
        if not calculator.options["chk"]:
            calculator.options["read_wf"] = None
            calculator.options["read_fc"] = None
            calculator.options["oldchk"] = None
        else:
            prefix = os.path.splitext(calculator.file)[0]
            calculator.options["oldchk"] = f"../{prefix}.chk"
            calculator.options["read_wf"] = True
            calculator.options["read_fc"] = None

        # Set calculation options
        calculator.set_options(config.dft_sp_options)
        calculator.atoms = opt_atoms

        # Do single point
        calculator.single_point(n_procs=config.n_procs, mem=config.mem)
        calculation_monitor(calculator)

        # Parse the results. Use special parser for double hybrids where cclib
        # fails.
        if calculator.options["functional"] in dh_functionals:
            parser = GaussianParser(calculator.output)
            electronic_energy_tz = parser.dh_energy
        else:
            data_sp = cclib.io.ccread(calculator.output)
            electronic_energy_tz = data_sp.scfenergies[-1] * EV_TO_HARTREE

    # Calculate enthalpy and free energy based on the single point energies at
    # the TZ level.
    enthalpy_tz = electronic_energy_tz + enthalpy_corr_dz
    free_energy_tz = electronic_energy_tz + free_energy_corr_dz
    free_energy_qh_grimme_tz = electronic_energy_tz + free_energy_corr_qh_grimme_dz
    free_energy_qh_truhlar_tz = electronic_energy_tz + free_energy_corr_qh_truhlar_dz

    # Store all the results in the atoms object.
    opt_atoms.info["electronic_energy"] = electronic_energy_tz
    opt_atoms.info["enthalpy"] = enthalpy_tz
    opt_atoms.info["free_energy"] = free_energy_tz
    opt_atoms.info["free_energy_qh_grimme"] = free_energy_qh_grimme_tz
    opt_atoms.info["free_energy_qh_truhlar"] = free_energy_qh_truhlar_tz
    opt_atoms.info["enthalpy_corr"] = enthalpy_corr_dz
    opt_atoms.info["entropy_corr"] = entropy_corr_dz   
    opt_atoms.info["entropy_corr_qh_grimme"] = entropy_corr_qh_grimme_dz
    opt_atoms.info["entropy_corr_qh_truhlar"] = entropy_corr_qh_truhlar_dz

    # Return atoms object
    return opt_atoms


def calculation_monitor(calculator, errors=True, dissociate=False, gic=False, displace_imaginary=0):
    """Calculation monitor that corrects various G16 errors and errors during
    geometry optimizations.

    Args:
        calculator (object): G16 calculator object that has been started.
        errors (bool): Whether to correct for errors
        dissociate (bool): Whether to check for dissociation and then end the
            calculation.
        gic (bool): Whether GICs are in operation
        displace_imaginary (bool): Whether displacement of imaginary frequencies
            should be done during geometry optimization.
    
    Returns:
        (bool): Indicator whether the calculation is done.
    """
    # Counter for number of displacements done
    displace_counter = 0

    # Set up list with IOps to correct coordinate system errors.
    iop_list = ["1/59=10", "1/59=14", "1/59=4", "1/59=40", "1/59=44"]
    new_iop = None

    # Max step ladder to use with GIC optimization.
    max_step_ladder = [1, 5, 15, 30]

    # Monitoring loop
    while True:
        # Set up flags for errors
        error = False
        imaginary = False
        gic_step_error = False

        # Loop to check for dissocation.
        if dissociate:
            while True:
                # Check every 5 seconds
                time.sleep(5)

                # Run dissociation check
                if has_dissociated(calculator.output):
                    # Kill job group
                    os.killpg(os.getpgid(calculator.process.pid), signal.SIGTERM)
                    logger.info("Dissociated, aborting calculation.")

                    # Wait 10 seconds for all background processes to die.
                    time.sleep(10)

                    # Return 
                    return None
                
                # Break from loop if calculation is done.
                if calculator.process.poll() != None:
                    break
        
        # Wait until job is done.
        calculator.process.wait()

        # Check for errors
        if errors:
            # Make dictionary of added calculator options
            options = {}

            # Read the output of the calculator
            output_text = open(calculator.output).read()

            # Check for random errors which are not reproducible. Just restart.
            if "Internal input file was deleted!" in output_text:
                error = True
                logger.info("File error. Restarting.")

            if "Perhaps someone deleted the data files for this job while it was running." in output_text:
                error = True
                logger.info("File error. Restarting.")

            if "-- Gradient out of range." in output_text:
                error = True
                logger.info("Gradient out of range. Assumed file error. Restarting.")
            
            if "Inaccurate quadrature in CalDSu." in output_text:
                error = True
                logger.info("Inaccurate quadrature in CalDSu. Random error. Restarting.")

            if "Input densites are not normalized." in output_text:
                error = True
                logger.info("Input densites are not normalized. Random error. Restarting.")
            
            if "Operation on file out of range." in output_text:
                error = True
                logger.info("Operation on file out of range. Random error. Restarting.")
            
            if "PCMFIO: Object Nord is empty!" in output_text:
                error = True
                logger.info("PCMFIO: Object Nord is empty! Random error. Restarting.")
            
            if "PCMFIO: A=read W=Nord IRw=" in output_text:
                error = True
                logger.info("PCMFIO: A=read W=Nord IRw=... Random error. Restarting.")
            
            if "RdWrB1 read garbage pointers." in output_text:
                error = True
                logger.info("PRdWrB1 read garbage pointers. Random error. Restarting.")

            if "FxIJDg NoPBat error." in output_text:
                error = True
                logger.info("FxIJDg NoPBat error. Random error. Restarting.")

            if "Logic error #1 in SortMO" in output_text:
                error = True
                logger.info("Logic error #1 in SortMO. Random error. Restarting.")

            if "RWCPar cannot convert versions." in output_text:
                error = True
                logger.info("RWCPar cannot convert versions. Random error. Restarting.")

            # Fix PCM inversion problem by switching to iterative approach
            if "Inv3 failed in PCMMkU" in output_text:
                error = True
                calculator.iops.append("3/140=2")
                logger.info("Inv3 failed in PCMMkU. Switching to iterative procedure.")
           
            # Check if the problem was RegRaf. Force SCF=NoInCore as suggested by Gaussian support.
            if "Not enough memory in RegRaf" in output_text:
                error = True
                options["no_incore"] = True
                logger.info("RegRaf error. Using SCF=NoInCore.")
            
            # Check for error in frequency calculation due to PCM.
            if "OrtVc1 failed #1." in output_text:
                error = True
                calculator.iops.append("3/142=8")
                logger.info("OrtVc1 failed #1. Increasing accuracy of PCM iteration.")
            
            # Check for convergence error
            if "Convergence failure -- run terminated." in output_text:
                error = True
                options["xqc"] = True
                logger.info("Convergence failure -- run terminated. Adding scf=xqc.")

            # Check for RedCar type errors associated with the interal coordinate system.
            if "RedCar failed" in output_text or "Error imposing constraints" in output_text or "CrVal0: Division by zero detected!" in output_text:
                error = True
                iops = calculator.iops
                try:
                    if new_iop:
                        iops.remove(new_iop)                    
                    new_iop = iop_list.pop(0)
                    iops.append(new_iop)
                    logger.info(f"Problem with coordinate updates. Adjusting IOps. {new_iop}")
                except IndexError:
                    angle_constraints = calculator.angle_constraints
                    if len(angle_constraints) > 0:
                        logger.info("No new IOps to try. Try to remove angle constraints.")
                        calculator.angle_constraints = []
                except ValueError:
                    pass
                # A smaller max step might help in this case
                calculator.options["opt_max_step"] = 5
            
            # Set suggested fix options for the calculator
            calculator.set_options(options)

        # Check for imaginary frequencies and displace
        if displace_imaginary:
            # Read data file

            data = cclib.io.ccread(calculator.output)
            if getattr(data, "optdone", None):
                # Check if lowest frequency is imaginary
                lowest_fc = data.vibfreqs[0]
                if lowest_fc < 0:
                    # Only displace a certain number of times.
                    if displace_counter < displace_imaginary:
                        imaginary = True

                        # Compute displaced geometry from the imaginary frequency.
                        disps = data.vibdisps[0]
                        geometry = data.atomcoords[-1]
                        disp_geometry = geometry + 0.2 * disps

                        # Creat new Atoms object
                        atoms = Atoms(symbols=data.atomnos, positions=disp_geometry)
                        atoms.info["charge"] = data.charge

                        # Change the atoms of the calculator and create new file
                        calculator.atoms = atoms
                        prefix = os.path.splitext(calculator.file)[0]
                        calculator.file = prefix + "_displaced.gjf"
                        calculator.output = prefix +"_displaced.log"
                        
                        # Use special options for azides
                        if config.general_info["azide_nucleophile"]:
                            calculator.options["int_acc"] = "SuperFineGrid"
                            calculator.options["opt_max_step"] = 5
                        else:
                            calculator.options["opt_max_step"] = 10
                        
                        # Increment displacement counter
                        displace_counter += 1
                        logger.info(f"Imaginary frequency found. Displacing along normal mode and reoptimizing. Attempt {displace_counter} of {displace_imaginary}")
                    else:
                        logger.info(f"Not attempting to displace imaginary frequency. Counter: {displace_counter} Attempts: {displace_imaginary}")
        
        # Treat problem with GIC optimizations. Try to reduce max step.
        if gic:
            output_text = open(calculator.output).read()
            if "Number of steps exceeded" in output_text:
                max_step = calculator.options.get("opt_max_step")
                if not max_step:
                    max_step = 30
                i = np.searchsorted(max_step_ladder, max_step) - 1
                if i != -1:
                    calculator.options["opt_max_step"] = max_step_ladder[i]
                    gic_step_error = True
        
        # Restart calculator in case of errors. Otherwise return True.
        if error or imaginary or gic_step_error:
            calculator.restart()
        else:
            return True


def check_reaction_database(smiles, database_location):
    #TODO Rewrite this so that it takes care of temperature and solvent.
    """Check if reaction is already in the database.

    Args:
        smiles (str): Reaction smiles to check
        database_location (str): Path to database.

    Returns:
        in_database (bool): Whether the reaction is in the database or not.
    """
    # Read database
    data_base_path = Path(database_location) / "db"
    file_path = data_base_path.as_posix()
    in_database = False
    with lock(file_path):
        # Canonicalize SMILES
        canonical_smiles = AllChem.ReactionToSmiles(AllChem.ReactionFromSmarts(smiles, useSmiles=True))
        db = shelve.open(file_path)
        for value in db.values():
            reaction_smiles = value["smiles"]["reaction"]
            # Canonicalize SMILES in the database
            canonical_reaction_smiles = AllChem.ReactionToSmiles(AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True))
            if canonical_reaction_smiles == canonical_smiles:
                in_database = True
        db.close()

    return in_database


def write_reaction_database():
    """Write an entry into the reaction database"""
    # Create entry
    entry = {}
    entry["smiles"] = results.smiles
    entry["inchi"] = results.inchi
    entry["inchi_key"] = results.inchi_key

    # Enter energies and geometries. Set up dictionaries.
    electronic_energies = {}
    enthalpies = {}
    free_energies = {}
    free_energies_qh_grimme = {}
    free_energies_qh_truhlar = {}
    enthalpy_corrs = {}
    entropy_corrs = {}
    entropy_corrs_qh_grimme = {}
    entropy_corrs_qh_truhlar = {}
    geometries = {}

    # Set up jobs
    jobs = ["substrate",
            "product",
            "leaving_group",
            ]
    if not config.intramolecular:
        jobs.append("nucleophile")
    if results.dft_atoms["intermediate"]:
        jobs.append("intermediate")
    if config.agent:
        jobs.append("agent")
    
    # Enter information for jobs
    for name in jobs:
        atoms = results.dft_atoms[name]
        electronic_energies[name] = atoms.info["electronic_energy"]
        enthalpies[name] = atoms.info["enthalpy"]
        free_energies[name] = atoms.info["free_energy"]
        free_energies_qh_grimme[name] = atoms.info["free_energy_qh_grimme"]
        free_energies_qh_truhlar[name] = atoms.info["free_energy_qh_truhlar"]
        enthalpy_corrs[name] = atoms.info["enthalpy_corr"]
        entropy_corrs[name] = atoms.info["entropy_corr"]
        entropy_corrs_qh_grimme[name] = atoms.info["entropy_corr_qh_grimme"]
        entropy_corrs_qh_truhlar[name] = atoms.info["entropy_corr_qh_truhlar"]
        geometries[name] = atoms.get_positions().tolist()

    # Enter information from TS.
    if results.dft_atoms["ts"]:
        name = "ts"
        atoms = results.dft_atoms[name]
        electronic_energies[name] = [atoms.info["electronic_energy"] for atoms in atoms]
        enthalpies[name] = [atoms.info["enthalpy"] for atoms in atoms]
        free_energies[name] = [atoms.info["free_energy"] for atoms in atoms]
        free_energies_qh_grimme[name] = [atoms.info["free_energy_qh_grimme"] for atoms in atoms]
        free_energies_qh_truhlar[name] = [atoms.info["free_energy_qh_truhlar"] for atoms in atoms]
        enthalpy_corrs[name] = [atoms.info["enthalpy_corr"] for atoms in atoms]
        entropy_corrs[name] = [atoms.info["entropy_corr"] for atoms in atoms]
        entropy_corrs_qh_grimme[name] = [atoms.info["entropy_corr_qh_grimme"] for atoms in atoms]
        entropy_corrs_qh_truhlar[name] = [atoms.info["entropy_corr_qh_truhlar"] for atoms in atoms]
        geometries[name] = [atoms.get_positions().tolist() for atoms in atoms]

    # Enter dictionaries into main entry.           
    entry["electronic_energies"] = electronic_energies
    entry["enthalpies"] = enthalpies
    entry["free_energies"] = free_energies
    entry["free_energies_qh_grimme"] = free_energies_qh_grimme
    entry["free_energies_qh_truhlar"] = free_energies_qh_truhlar
    entry["enthalpy_corr"] = enthalpy_corrs
    entry["entropy_corr"] = entropy_corrs
    entry["entropy_corr_qh_grimme"] = entropy_corrs_qh_grimme
    entry["entropy_corr_qh_truhlar"] = entropy_corrs_qh_truhlar
    entry["geometries"] = geometries

    # Enter symbols
    entry["symbols"] = {"substrate": results.dft_atoms["substrate"].get_chemical_symbols(),
                        "product": results.dft_atoms["product"].get_chemical_symbols(),
                        "leaving_group": results.dft_atoms["leaving_group"].get_chemical_symbols(),
                        }
    if not config.intramolecular:
        entry["symbols"]["nucleophile"] = results.dft_atoms["nucleophile"].get_chemical_symbols()
    if config.agent:
        entry["symbols"]["agent"] = results.dft_atoms["agent"].get_chemical_symbols()
    if results.xtb_atoms.get("reaction_complex"):
        entry["symbols"]["complex"] = results.xtb_atoms["reaction_complex"].get_chemical_symbols()
    else:
        # TODO get the symbols from the molecular database of the TS
        pass
    
    # Enter reactive atoms
    entry["reactive_atoms"] = {"central_atom": config.reactive_atoms["central_atom"],
                               "nu_atom": config.reactive_atoms["nu_atom"],
                               "lg_atom": config.reactive_atoms["lg_atom"],
                               "central_atom_prod": config.reactive_atoms["central_atom_prod"],
                               "added_atom": config.reactive_atoms["added_atom"],
                               }
    if not config.intramolecular:
        entry["reactive_atoms"]["nu_atom_orig"] = config.reactive_atoms["nu_atom_orig"]
    else:
        entry["reactive_atoms"]["nu_atom_orig"] = None
    if config.agent:
        entry["reactive_atoms"]["agent_atom"] = config.reactive_atoms["agent_atom"]
        entry["reactive_atoms"]["agent_atom_orig"] = config.reactive_atoms["agent_atom_orig"]

    # Enter descriptors
    if config.descriptor_options["calculate_descriptors"]:
        entry["descriptors"] = results.descriptors

    # Enter misc. information
    entry["concerted"] = config.concerted
    entry["intramolecular"] = config.intramolecular
    entry["flat_PES"] = config.flat_PES
    entry["end_time"] = results.end_time
    entry["run_time"] = results.run_time
    entry["n_cpus"] = config.n_procs
    entry["temperature"] = config.general_options["temperature"]
    entry["agent"] = config.general_info["agent"]
    entry["clustering_energies"] = results.clustering_energies

    # Solvent
    entry["solvent"] = config.general_options["solvent_smiles"]
    mol = Chem.MolFromSmiles(config.general_options["solvent_smiles"])
    smiles = Chem.MolToSmiles(mol)
    inchi = Chem.MolToInchi(mol)
    inchi_key = Chem.MolToInchiKey(mol)
    entry["smiles"]["solvent"] = smiles 
    entry["inchi"]["solvent"] = inchi
    entry["inchi_key"]["solvent"] = inchi_key
    
    # Write to database.
    db_path = config.directories["database"]
    db_path = Path(db_path)
    db_file = db_path / "db"
    db_file_full = db_file.as_posix()
    logger.info(f"Writing to database {db_file}...")
    with lock(db_file_full):
        db = shelve.open(db_file_full)
        record_id = str(uuid.uuid4())
        db[record_id] = entry
        db.close()
    logger.info(f"Writing entry with identifier {record_id}")
    logger.info("...writing completed.")


def has_dissociated(output):
    """Checks if intermediate or TS has dissociated

    Uses NBO Wiberg bond orders (BOs) to check if the bonds between the central
    atom and the nucleophile/leaving group are completely broken (BO > 0.2)

    Args:
        output (str): Name of Gaussian output file with NBO Wiberg BOs

    Returns:
        has_dissociated (bool): True if it has dissociated, false if not.
    """
    # Read the bond orders
    parsed_output = NBOParser(output)
    central_atom = config.reactive_atoms["central_atom"]
    nu_atom = config.reactive_atoms["nu_atom"]
    lg_atom = config.reactive_atoms["lg_atom"]

    # Returns False if matrix is empty (no NBO)
    if parsed_output.bo_matrix.any() == False:
        return False

    # Get bond order between reactive atoms.
    bo_1 = parsed_output.get_bo(central_atom, nu_atom)
    bo_2 = parsed_output.get_bo(central_atom, lg_atom)

    # If bond order goes below 0.3, dissociation is considered to have occurred.
    if (bo_1 < 0.3) or (bo_2 < 0.3):
        has_dissociated = True
    else:
        has_dissociated = False
    
    return has_dissociated


def displace_coordinates(output, factor):
    """Displaces the geometry along the first normal mode based on a frequency calculation.

    Args:
        output (string) :   Name of Gaussian output file with frequency calculation
        factor (float)  :   Factor to multiply the displacement with

    Returns:
        atoms (object)  :   Atoms object of displaced geometry.
    """
    # Read the ouptut with cclib.
    data = cclib.io.ccread(output)

    # Get displacements, geometry and calculate new geometry.
    disps = data.vibdisps[0]
    geometry = data.atomcoords[-1]
    disp_geometry = geometry + factor * disps

    # Create and return new Atoms object.
    atoms = Atoms(symbols=data.atomnos, positions=disp_geometry)
    atoms.info["charge"] = data.charge

    return atoms


def set_config(options):
    """Set config module from dictionary.
    
    Args:
        options (dict): Dictionary of options.
    """
    config.general_options = options.general_options
    config.xtb_options = options.xtb_options
    config.dft_options = options.dft_options
    config.dft_sp_options = options.dft_sp_options
    config.crest_options = options.crest_options
    config.directories = options.directories
    config.descriptor_options = options.descriptor_options


def set_info(options):
    """Set config module from dictionary

    Args:
        options (dict): Dictionary of options.
    """
    config.charges = options.charges
    config.general_info = options.general_info
    config.clustering = options.clustering
    config.agent = options.general_info["agent"]
    config.reactive_atoms = options.reactive_atoms


def set_resources(n_procs=None, mem=None):
    """Set resources for the calculations.

    Args:
        n_procs (int): Number of processors.
        mem (float): Amount of memory in GB.
    """
    # In the first case, take the number from the Gaussian environment variable.
    if os.environ.get("GAUSS_PDEF"):
        config.n_procs = int(os.environ["GAUSS_PDEF"])
    elif n_procs:
        config.n_procs = n_procs
    else:
        config.n_procs = 1
    
    # Set the OMP number of threads to max 2 due to xtb efficiency considerations.
    os.environ["OMP_NUM_THREADS"] = str(np.clip(n_procs, 1, 2))
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    
    # Set according to xtb requirements.
    os.environ["OMP_STACKSIZE"] = "1000m"
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    # Set memory for Gaussian. Use the environmental variable as default.
    if os.environ.get("GAUSS_MDEF"):
        config.mem = float(os.environ["GAUSS_MDEF"].strip("MGB"))
    elif mem:
        config.mem = mem
    else:
        config.mem = 1
    
    # Set up executables
    config.xtb = str(Path(config.directories["xtb"]) / "bin" / "xtb")
    config.crest = str(Path(config.directories["crest"]) / "crest")
    config.chargemol = str(Path(config.directories["chargemol"]) / "chargemol_parallel")
    config.hs95_4 = str(Path(config.directories["hs95"]) / "hs95_4.run")
    config.hs95_1 = str(Path(config.directories["hs95"]) / "hs95.run")
    config.interface_script = config.directories["interface_script"]

    # Export path variables
    os.environ["XTBHOME"] = config.directories["xtb"]
    os.environ["XTBPATH"] = config.directories["xtb"] + ":" + str(Path.home())
    os.environ["PATH"] = os.environ["PATH"] + ":" + str(Path(config.directories["xtb"]) / "bin")

    # Create scratch directories
    gauss_scr_dir = Path(config.directories["gaussian_scratch"]) / os.environ["SLURM_JOB_ID"]
    if not gauss_scr_dir.is_dir():
        gauss_scr_dir.mkdir(parents=True)
    os.environ["GAUSS_SCRDIR"] = str(Path(config.directories["gaussian_scratch"]) / os.environ["SLURM_JOB_ID"])


def get_bond_length(symbol_1, symbol_2):
    """Get the bond length between two atom types from the covalent radii.
    
    Args:
        symbol_1 (str): Symbol of first atom.
        symbol_2 (str): Symbol of second atom.
    
    Returns:
        bond_length (float): Bond length (Å).
    """
    # Get the covalent radii from mendeleev.
    covalent_radius_1 = element(symbol_1).covalent_radius_pyykko
    covalent_radius_2 = element(symbol_2).covalent_radius_pyykko

    # Calculate the bond length as the sum of the covalent radii and convert
    # to Ånström
    bond_length = (covalent_radius_1 + covalent_radius_2) / 100

    return bond_length


def get_fragment_distance(symbol_1, symbol_2):
    """Get suggested distance between two fragments based on their vdW radii.
    Args:
        symbol_1 (str): Symbol of first atom.
        symbol_2 (str): Symbol of second atom.

    Returns:
        fragment_distance (float): Suggested distance between the two fragments.
    """
    # Get the vdW radii from mendeleev.
    vdw_radius_1 = element(symbol_1).vdw_radius
    vdw_radius_2 = element(symbol_2).vdw_radius

    # Get the fragment distance as the sum of the vdw_radii. Convert to Å.
    fragment_distance = (vdw_radius_1 + vdw_radius_2) / 100

    return fragment_distance 


def get_vdw_radius(symbol):
    """Get vdW radius of element in Å.
    
    Args:
        symbol (str): Symbol of atom

    Returns:
        vdw_radius (float): vdW radius (Å)
    """
    # Concert to Å
    vdw_radius = element(symbol).vdw_radius / 100

    return vdw_radius


def get_nucleophilic_centers(output):
    """Return nucleophilic centers which corresponds to a localized molecular
    orbital energy within 2 eV of the highest energy.

    Args:
        output (str): Path to xtb output file.

    Returns:
        nucleophilic_centers (list): Nucleophilic centers of the molecule.
    """
    # Read data and get the highest energy
    data = XTBParser(output)
    highest_energy = max(data.lmo_dict.values())

    # Parse through potential nucleophilic centers
    nucleophilic_centers = []
    for atom, energy in data.lmo_dict.items():
        if abs(energy - highest_energy) < 2:
            nucleophilic_centers.append(atom)

    return nucleophilic_centers


def add_azide_constraints(calculator):
    """Convenience function to add sets of constraints when calculating
    azides.
    
    Args:
        calculator (object): Calculator object
    """
    if config.general_info["azide_nucleophile"]:
        # Set dihedral constraints. 
        dihedral_atom_list = config.general_info["azide_dihedral"]
        dihedral_value = 180
        calculator.add_dihedral_constraint(dihedral_atom_list, dihedral_value)

        # Set angle constraints.
        angle_atom_list = config.general_info["azide_angle"]
        angle_value = 180
        calculator.add_angle_constraint(angle_atom_list, angle_value)


def thermal_analysis_atom(mass, reference="atm", temperature=298.15):
    """Do thermal analysis for an atom.

    Args:
        mass (float): Atomic mass
        reference (str): Reference state 'atm' for atmosphere and 'M' for molar.
        temperature (float): Temperature

    Returns:
        enthalpy (float): Enthalpy (Hartree)
        t_entropy (float): Temperature * entropy (Hartree)
    """
    # Calculate molar volume
    if reference == "atm":
        V = BOLTZMANN * temperature / ATM * AVOGADRO_CONSTANT
    if reference == "M":
        V = 1

    # Calculate enthalpy
    enthalpy = 5 / 2 * GAS_CONSTANT * temperature
    
    # Calculate entropy
    entropy = GAS_CONSTANT * (np.log((2 * np.pi * mass * AMU * BOLTZMANN * temperature / PLANCK ** 2) ** (3 / 2) * V / AVOGADRO_CONSTANT) + 5 / 2)
    t_entropy = temperature * entropy

    # Convert to atomic units
    enthalpy = enthalpy / HARTREE / MOL
    t_entropy = t_entropy / HARTREE / MOL
    
    return enthalpy, t_entropy


def standard_state_correction(concentration, reference="atm", temperature=298.15):
    """Calculate standard state correction for a certain concentration.

    Args:
        concentration (float): Concentration of species of interest
        reference (str): Reference state 'atm' for atmosphere and 'M' for molar.
        temperature (float): Temperature

    Returns:
        ss_corr (float): Standard state correction to the free energy.
    """
    # Calculate molar volume
    if reference == "atm":
        V = BOLTZMANN * temperature / ATM * AVOGADRO_CONSTANT
        reference_concentration = 1 / (V * 1000)
    if reference == "M":
        reference_concentration = 1
    
    # Calculate standard state correction
    ss_corr = temperature * GAS_CONSTANT * STANDARD_STATE_UNIT_CORRECTION * math.log(concentration / reference_concentration)

    return ss_corr


class TSValidator:
    """Validate if negative modes of TS corresponds to the right bonds formed
    or broken.

    Args:
        atom_pairs (list): List of atom pairs (tuples/lists)
        check_atoms (list): Atoms to check against bond vectors
        data (object): cclib data object
        threshold (float): Threshold of projected value of atom displacement
            along vector.

    Attributes:
        max_disp (float): Maximum projected displacement
        n_imag (int): Number of imaginary frequencies.
        proj_disps (list): Projected displacements
        validated (bool): Whether TS is valid or not
        vectors (list): Vectors derived from atom pairs.
    """
    def __init__(self, data, atom_pairs, check_atoms, threshold=0.13):
        # Calculate normalized vectors between each atom pair.
        vectors = []
        positions = data.atomcoords[-1]
        for atom_1, atom_2 in atom_pairs:
            vector = positions[atom_1 - 1] - positions[atom_2 - 1]
            vector /= np.linalg.norm(vector)
            vectors.append(vector)

        # Project displacement vectors of negative modes onto the atom pair vectors.
        displacements = data.vibdisps[data.vibfreqs < 0]
        proj_disps = []
        for disp in displacements:
            for atom in check_atoms:
                for vector in vectors:
                    proj_disps.append(np.abs(np.dot(disp[atom - 1], vector)))
        
        # Check maximum projected length against the threshold.
        max_disp = max(proj_disps)
        
        if max_disp > threshold:
            self.validated = True
        else:
            self.validated = False
        
        # Store attributes
        self.n_imag = np.sum(data.vibfreqs < 0)
        self.vectors = vectors
        self.proj_disps = proj_disps
        self.max_disp = max_disp