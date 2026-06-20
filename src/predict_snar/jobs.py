import itertools
import logging
import math
import os
from pathlib import Path
import shutil

from ase import Atoms
import ase.io
from ase.geometry.analysis import Analysis
import cclib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from predict_snar import config, results
from predict_snar.calculators import XTBCalculator, G16Calculator, CRESTCalculator, Constraint, GICTSScan, TSScan
from predict_snar.data import EV_TO_HARTREE, ANGSTROM_TO_BOHR, HARTREE_TO_KCAL, KCAL_TO_HARTREE, atomic_masses
from predict_snar.helpers import cd, thermal_analysis_atom
from predict_snar.helpers import calculation_monitor, get_results, set_electronic_temp, add_azide_constraints, standard_state_correction, get_electronic_temp
from predict_snar.helpers import TSValidator
from predict_snar.parsers import XTBParser

logger = logging.getLogger("predict_snar")

def search_conformers(atoms, constrained_atoms=None, constraints=None, dft_ranking=True, crest_options=None):
    """Search for conformers with CREST.

    Args:
        constrained_atoms (list): Indices of atoms to constrain (1-indexed).
        constraints (list): Bond constraints.
        dft_ranking (bool): Whether to rank the conformers with DFT.
        crest_options (dict): Options for the CREST calculator.

    Returns:
        atoms (object): The conformer with the lowest energy
    """
    if not constrained_atoms:
        constrained_atoms = set()
    if not constraints:
        constraints = []
    if not crest_options:
        crest_options = {}

    # Only do conformationl search if there are more than two atoms. 
    if atoms.get_number_of_atoms() > 2:
        # Set up scratch directory.
        job_id = os.environ["SLURM_JOB_ID"]
        scratch_dir = Path(config.directories["crest_scratch"]) / f"{job_id}" / "crest"
        submit_dir = Path.cwd()
        if not os.path.isdir(scratch_dir):
            os.makedirs(scratch_dir)
        
        # Run calculation in scratch directory.
        with cd(scratch_dir):
            # Set up calculation.
            crest = CRESTCalculator(atoms, config.xtb_options, config.crest_options)
            crest.set_options(crest_options)
            crest.constraints.extend(constraints)
            crest.constrained_atoms.update(constrained_atoms)
            crest.scratch_dir = scratch_dir
            crest.submit_dir = submit_dir

            # Run calculation.
            logger.info("Doing conformational search...")
            crest.run_calc(config.n_procs).wait()
            logger.info("...conformational search complete.")

            # Move key files back to submission directory.
            for file in ["coord", "xcontrol", "crest_conformers.xyz"]:
                if os.path.isfile(file):
                    shutil.move(file, submit_dir)
        
        # Remove scratch directory
        if os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir)
        
        # Parse conformers
        crest.parse_conformers()
        logger.info(f"Found {len(crest.conformers)} conformers.")
        
        # Rank conformers with DFT in the case that there are more than one.
        if len(crest.conformers) > 1 and dft_ranking:
            logger.info("Ranking conformers with DFT...")
            crest.rank_conformers(config.dft_options, n_procs=config.n_procs, mem=config.mem)
            logger.info("...ranking complete.")
        atoms = crest.conformers[0]

    return atoms


def optimize_intermediate():
    """Run optimization of intermediate"""
    # Get atoms file from the reaction complex.
    atoms = results.xtb_atoms["reaction_complex"]

    # Load reactive atoms from config
    is_agent = config.general_info["agent"]
    central_atom = config.reactive_atoms["central_atom"]
    nu_atom = config.reactive_atoms["nu_atom"]
    lg_atom = config.reactive_atoms["lg_atom"]
    nu_h_atoms = config.reactive_atoms["nu_h_atoms"]
    ring_atoms = config.reactive_atoms["ring_atoms"]    

    if is_agent:
        agent_atom = config.reactive_atoms["agent_atom"]
        coordinated_h_atom = config.reactive_atoms["nu_h_atoms"][0]
        agent_h_distance = atoms.get_distance(agent_atom - 1, coordinated_h_atom - 1)

    # Optimize the intermediate with GFN2-xTB
    os.mkdir("intermediate")
    os.mkdir("intermediate/xtb")
    with cd("intermediate/xtb"):
        # Get distances from config.
        nu_distance = config.general_options["intermediate_center_nu"]
        lg_distance = config.general_options["intermediate_center_lg"]
            
        # Set up the xtb calculator
        xtb = XTBCalculator(atoms, file="frozen_intermediate.xyz", options=config.xtb_options)

        #  Set up constraints
        xtb.options["force_constant"] = 0.05
        xtb.add_constraint(central_atom, nu_atom, nu_distance)
        xtb.add_constraint(central_atom, lg_atom, lg_distance)
        if is_agent:
            xtb.add_constraint(agent_atom, coordinated_h_atom, agent_h_distance)
        for h_atom, h_distance in zip(config.reactive_atoms["nu_h_atoms"], config.general_options["nu_h_distances"]):
            xtb.add_constraint(nu_atom, h_atom, h_distance)
        if config.general_info["azide_nucleophile"]:
            azide_atom_list = config.general_info["azide_dihedral"]
            dihedral_value = 180
            xtb.add_dihedral_constraint(azide_atom_list, dihedral_value)

            angle_atom_list = config.general_info["azide_angle"]
            angle_value = 180
            xtb.add_angle_constraint(angle_atom_list, angle_value)

        #  Run the xtb constrained optimization
        logger.info("Optimizing constrained intermediate with xtb...")
        xtb.opt().wait()
        pre_optimized = ase.io.read("xtbopt.xyz")
        pre_optimized.info["charge"] = atoms.info["charge"]

        # Harden constraints
        xtb.atoms = pre_optimized
        xtb.options["force_constant"] = 2.0
        xtb.opt().wait()

        # Parse the results and change electronic temperature from H-L gap
        parser = XTBParser("xtb.out")
        hl_gap = parser.lumo - parser.homo
        el_temp = get_electronic_temp(hl_gap)
        result = set_electronic_temp(el_temp)

        # Rerun if electronic temperature was changed.
        if result:
            logger.info("Re-optimizing intermediate...")
            xtb.options["el_temp"] = el_temp
            xtb.opt().wait()
            logger.info("...optimization completed.")
            parser = XTBParser("xtb.out")

        # Get the geometry from the frozen intermediate calculation.
        frozen_intermediate = ase.io.read("xtbopt.xyz")
        frozen_intermediate.info["charge"] = atoms.info["charge"]

        # Save parser for comparison
        parser_before = parser

        # Do conformational search with CREST and take lowest energy one
        if config.general_options["find_intermediate"]:
            os.mkdir("crest")
            with cd("crest"):
                # Add constraints.
                constrained_atoms = set([central_atom, nu_atom, lg_atom])
                # TODO These atoms cannot be constrained while getting a full conformational sampling.
                #if is_agent:
                #    constrained_atoms.update([agent_atom, coordinated_h_atom])
                constraints = []
                if nu_h_atoms:
                    for h_atom in nu_h_atoms:
                        distance = frozen_intermediate.get_distance(nu_atom - 1, h_atom - 1)
                        constraint = Constraint(nu_atom, h_atom, distance)
                        constraints.append(constraint)
                if ring_atoms:
                    constrained_atoms.update(ring_atoms)
                crest_options = {"el_temp": config.xtb_options["el_temp"]}

                # Run conformational search.
                frozen_intermediate = search_conformers(frozen_intermediate, constrained_atoms=constrained_atoms, constraints=constraints, crest_options=crest_options)

        # Optimize again with right electronic temperature outside CREST
        xtb.atoms = frozen_intermediate
        xtb.opt().wait()
        
        # Check if dissociation has happened
        parser = XTBParser("xtb.out")

        # Get bond order matrix and check if disscoiation has happened.
        # In that case, change the electronic temperature.
        bo_matrix_diff = np.abs(parser.bo_matrix - parser_before.bo_matrix)
        if np.max(bo_matrix_diff) > 0.5:
            logger.info("Dissociation happened during CREST run.")
            # Try to change electronic temperature
            if config.xtb_options["el_temp"] == 7000:
                config.xtb_options["el_temp"] = 4000
                logger.info("Decreasing electronic temperature to 4000 and trying again.")
            else:
                raise Exception("Dissociation during CREST run. Stopping.")
            # Optimize again
            with cd("crest"):
                frozen_intermediate = search_conformers(frozen_intermediate, constrained_atoms=constrained_atoms, constraints=constraints, crest_options=crest_options)
            xtb.atoms = frozen_intermediate
            xtb.opt().wait()

        # Get the geometry of the frozen intermediate. 
        frozen_intermediate = ase.io.read("xtbopt.xyz")
        frozen_intermediate.info["charge"] = atoms.info["charge"]
        results.xtb_atoms["intermediate"] = frozen_intermediate

        logger.info("...optimization completed")
    
    # Stop if DFT should not be done
    if not config.general_options["find_intermediate"]:
        results.dft_atoms["intermediate"] = None        
        return

    # Optimize the intermediate with DFT
    os.mkdir("intermediate/dft")
    with cd("intermediate/dft"):
        # Do calculation with frozen bond lenghts first. Set up calculation and
        # add contraints.
        g16 = G16Calculator(frozen_intermediate, "intermediate_frozen.gjf", config.dft_options)
        g16.options["opt_acc"] = "loose"
        g16.options["int_acc"] = "fine"
        g16.add_constraint(central_atom, nu_atom, "F")
        g16.add_constraint(central_atom, lg_atom, "F")

        # Run calculation.
        logger.info("Pre-optimizing intermediate with DFT...")
        g16.opt(n_procs=config.n_procs, mem=config.mem)
        calculation_monitor(g16, errors=True)
        logger.info("...optimization completed.")

        # Read results.
        data = cclib.io.ccread(g16.output)
        pre_opt_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
        pre_opt_atoms.info["charge"] = atoms.info["charge"]

        # Set up G16 calculator for real optimization.
        g16 = G16Calculator(pre_opt_atoms, file="intermediate.gjf", options=config.dft_options)
        g16.options["opt_max_step"] = 15
        g16.options["opt_cycles"] = 50
        g16.options["nbo"] = True

        # Run the G16 optimization. Monitor for dissociation and quit in that case.
        logger.info("Optimizing intermediate with DFT...")
        g16.opt_freq(n_procs=config.n_procs, mem=config.mem)
        result = calculation_monitor(g16, errors=True, dissociate=True, displace_imaginary=1)
        if not result:
            logger.info("Intermediate not stable. Aborting optimization.")
            return

        # If not optimized, decrease step size and increase integral grid size.
        data = cclib.io.ccread(g16.output)
        if not data.optdone:
            continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
            continue_atoms.info["charge"] = atoms.info["charge"]
            g16.atoms = continue_atoms
            prefix = os.path.splitext(g16.file)[0]
            g16.file = prefix + "_cont.gjf"
            g16.output = prefix +"_cont.log"
            g16.options["opt_max_step"] = 5
            g16.options["int_acc"] = "SuperFineGrid"
            g16.opt_freq(n_procs=config.n_procs, mem=config.mem)
            result = calculation_monitor(g16, displace_imaginary=1, dissociate=True)

        if not result:
            logger.info("Intermediate not stable. Aborting optimization.")
            return

        # If not optimized now, proceed with frequency calculation anyway.
        data = cclib.io.ccread(g16.output)
        if not data.optdone:
            continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
            continue_atoms.info["charge"] = atoms.info["charge"]
            g16.atoms = continue_atoms
            prefix = os.path.splitext(g16.file)[0]
            g16.file = prefix + "cont_freq.gjf"
            g16.output = prefix +"cont_freq.log"
            g16.freq(n_procs=config.n_procs, mem=config.mem)
            result = calculation_monitor(g16)

        # Return the optimized intermediate as an Atoms object
        data = cclib.io.ccread(g16.output)
        lowest_fc = data.vibfreqs[0]
        if lowest_fc < 0:
            logger.info("Negative eigenvalues for optimized structure. Assuming flat PES is the cause. Continuing.")
        opt_intermediate = get_results(g16)
        results.dft_atoms["intermediate"] = opt_intermediate

        logger.info("...optimization completed")

    return


def optimize_xtb(atoms, name, options=None):
    """Optimize a structure with xtb.

    Args:
        atoms (object): ASE Atoms object.
        name (str): Name of the calculation.
        options (dict): Options for the xtb calculation.

    Returns:
        atoms (object): Optimized structure as ASE atoms object.
    """
    # Set up options.
    if not options:
        options = {}

    # Create directory and enter it.
    os.mkdir(name)
    with cd(name):
        logger.info(f"Optimizing {name} with xtb...")
    
        # Do conformational search with CREST and take lowest energy conformer.
        os.mkdir("crest")
        with cd("crest"):
            atoms = search_conformers(atoms)

        # Redo calculation with the proper xtb options.
        xtb = XTBCalculator(atoms, f"{name}.xyz", config.xtb_options)
        xtb.set_options(options)
        xtb.opt().wait()

        # Read results.
        logger.info("...optimization completed")
        opt_atoms = ase.io.read("xtbopt.xyz")
        opt_atoms.info["charge"] = atoms.info["charge"]

    return atoms


def optimize_react_complex():
    """Optimize the reaction complex with XTB."""
    # Set up atoms object from MM optimizations.
    atoms = results.mm_atoms["reaction_complex"]

    # Get the relevant reactive atoms and distances.
    central_atom = config.reactive_atoms["central_atom"]
    ortho_carbons = config.reactive_atoms["ortho_carbons"]
    nu_atom = config.reactive_atoms["nu_atom"]
    distance = config.general_options["fragment_distance_reactant"]

    if config.agent:
        agent_atom = config.reactive_atoms["agent_atom"]
        coordinated_h_atom = config.reactive_atoms["nu_h_atoms"][0]
        agent_h_distance = atoms.get_distance(agent_atom - 1, coordinated_h_atom - 1)
        nu_h_distance = atoms.get_distance(nu_atom - 1, coordinated_h_atom - 1)
    
    # Calculate distance to ortho carbons through Pythagora's theorem
    distance_central_ortho_1 = atoms.get_distance(central_atom - 1, ortho_carbons[0] - 1)
    distance_central_ortho_2 = atoms.get_distance(central_atom - 1, ortho_carbons[1] - 1)
    distance_central_ortho = (distance_central_ortho_1 + distance_central_ortho_2) / 2
    distance_nu_ortho = (distance ** 2 + distance_central_ortho ** 2) ** (1 / 2)

    # Optimize the reactant complex
    os.mkdir("reactant_complex")
    with cd("reactant_complex"):
        # Do a first rough optimization with small force constant and few steps.
        os.mkdir("step_1")
        with cd("step_1"):
            # Set up calculator
            xtb = XTBCalculator(atoms, "reactant_complex.xyz", config.xtb_options)

            # Add constraints
            xtb.add_constraint(central_atom, nu_atom, distance)
            if config.agent:
                xtb.add_constraint(agent_atom, coordinated_h_atom, agent_h_distance)
                xtb.add_constraint(nu_atom, coordinated_h_atom, nu_h_distance)            
            for i in ortho_carbons:
                xtb.add_constraint(nu_atom, i, distance_nu_ortho)
            add_azide_constraints(xtb)
            xtb.options["force_constant"] = 0.05
            xtb.options["opt_cycles"] = 30

            # Do optimization
            xtb.opt().wait()

            # Read results
            opt_atoms = ase.io.read("xtbopt.xyz")
            opt_atoms.info["charge"] = atoms.info["charge"]
        
        # Do a second precise optimization.
        os.mkdir("step_2")
        with cd("step_2"):      
            # Set up calculator
            xtb = XTBCalculator(opt_atoms, "reactant_complex.xyz", config.xtb_options)

            # Add constraints
            xtb.add_constraint(central_atom, nu_atom, distance)
            if config.agent:
                xtb.add_constraint(agent_atom, coordinated_h_atom, agent_h_distance)
                xtb.add_constraint(nu_atom, coordinated_h_atom, nu_h_distance)
            for i in ortho_carbons:
                xtb.add_constraint(nu_atom, i, distance_nu_ortho)
            add_azide_constraints(xtb)
            xtb.options["force_constant"] = 2.0
    
            # Perform calculation
            logger.info("Optimizing reactant complex...")
            xtb.opt().wait()
            logger.info("...optimization completed.")
           
            # Check electronic temperature
            parser = XTBParser("xtb.out")
            hl_gap = parser.lumo - parser.homo
            el_temp = get_electronic_temp(hl_gap)
            result = set_electronic_temp(el_temp)
            
            # Reoptimize if necessary
            if result:
                xtb.options["el_temp"] = config.xtb_options["el_temp"]
                logger.info("Re-optimizing reactant complex...")
                xtb.opt().wait()
                logger.info("...optimization completed.")
            
            # Create atoms object
            opt_atoms = ase.io.read("xtbopt.xyz")
            opt_atoms.info["charge"] = atoms.info["charge"]

    # Store results.
    results.xtb_atoms["reaction_complex"] = opt_atoms


def optimize_prod_complex():
    """Optimize product complex with XTB."""
    # Get the intermediate from the best possible source.
    if results.dft_atoms["intermediate"]:
        atoms = results.dft_atoms["intermediate"]
    else:
        atoms = results.xtb_atoms["intermediate"]

    # Get the relevant options
    central_atom = config.reactive_atoms["central_atom"]
    ortho_carbons = config.reactive_atoms["ortho_carbons"]
    lg_atom = config.reactive_atoms["lg_atom"]
    distance = config.general_options["fragment_distance_product"]
    
    # Calculate distance to ortho carbons through Pythagora's theorem
    distance_central_ortho_1 = atoms.get_distance(central_atom - 1, ortho_carbons[0] - 1)
    distance_central_ortho_2 = atoms.get_distance(central_atom - 1, ortho_carbons[1] - 1)
    distance_central_ortho = (distance_central_ortho_1 + distance_central_ortho_2) / 2
    distance_lg_ortho = (distance ** 2 + distance_central_ortho ** 2) ** (1 / 2)    

    # Optimize the product complex
    os.mkdir("product_complex")
    with cd("product_complex"):
        logger.info("Optimizing product complex...")  

        # First optimize with a smaller force constant for the first steps to
        # avoid "shooting out" leaving group
        os.mkdir("step_1")
        with cd("step_1"):
            # Set up calculation.
            xtb = XTBCalculator(atoms, "product_complex.xyz", config.xtb_options)
            
            # Add constraints
            xtb.add_constraint(central_atom, lg_atom, distance)
            for i in ortho_carbons:
                xtb.add_constraint(lg_atom, i, distance_lg_ortho)
            add_azide_constraints(xtb)
            xtb.options["maxdispl"] = 0.1
            xtb.options["force_constant"] = 0.05
            xtb.options["opt_cycles"] = 5

            # Run calculation
            xtb.opt().wait()

            # Read results.
            opt_atoms = ase.io.read("xtbopt.xyz")
            opt_atoms.info["charge"] = atoms.info["charge"]

        # Proceed with the full optimization here with the right force constant.
        os.mkdir("step_2")
        with cd("step_2"):
            # Set up calculation
            xtb = XTBCalculator(opt_atoms, "product_complex.xyz", config.xtb_options)

            # Add constraints
            xtb.add_constraint(central_atom, lg_atom, distance)
            for i in ortho_carbons:
                xtb.add_constraint(lg_atom, i, distance_lg_ortho)            
            xtb.options["maxdispl"] = 0.1
            xtb.options["force_constant"] = 2.0

            # Run calculation
            xtb.opt().wait()

            # Read results.
            opt_atoms = ase.io.read("xtbopt.xyz")
            opt_atoms.info["charge"] = atoms.info["charge"]
            logger.info("...optimization completed.")

            # Check electronic temperature
            parser = XTBParser("xtb.out")
            hl_gap = parser.lumo - parser.homo
            el_temp = get_electronic_temp(hl_gap)
            result = set_electronic_temp(el_temp)

            # Redo calculation if electronic temperature is changed.
            if result:
                xtb.options["el_temp"] = config.xtb_options["el_temp"]
                logger.info("Re-optimizing product complex...")
                xtb.opt().wait()
                logger.info("...optimization completed.")

            # Get the resulting geometry.
            opt_atoms = ase.io.read("xtbopt.xyz")
            opt_atoms.info["charge"] = atoms.info["charge"]
    
    # Store the results.
    results.xtb_atoms["product_complex"] = opt_atoms


def optimize_ts(atoms, atom_pairs, check_atoms, extra_crest_constraints=None, extra_dft_constraints=None):
    """Optimize transition state

    Args:
        atom_pairs (list): Atom pairs (as tuples) to use when evaluating TS
            optimization success.
        check_atoms (list): Atoms that should move in the TS.
        extra_crest_constraints (list): Extra constraint for the CREST
            calculations.
        extra_dft_constraints (list): Extra constraints for the DFT
            calculations.

    Returns:
        opt_ts (object): Optimized transition state. Returns None if
            optimization failed.
    """
    # Set up constraint lists.
    if not extra_crest_constraints:
        extra_crest_constraints = []
    if not extra_dft_constraints:
        extra_dft_constraints = []

    # Load reactive atoms and distances from config file.
    is_agent = config.general_info["agent"]
    central_atom = config.reactive_atoms["central_atom"]
    nu_atom = config.reactive_atoms["nu_atom"]
    lg_atom = config.reactive_atoms["lg_atom"]
    nu_h_atoms = config.reactive_atoms["nu_h_atoms"]
    ring_atoms = config.reactive_atoms["ring_atoms"]       
    if is_agent:
        agent_atom = config.reactive_atoms["agent_atom"]
        coordinated_h_atom = config.reactive_atoms["nu_h_atoms"][0]
        agent_h_distance = atoms.get_distance(agent_atom - 1, coordinated_h_atom - 1)
        nu_h_distance = atoms.get_distance(nu_atom - 1, coordinated_h_atom - 1)
    ts_max_step = config.general_options["ts_max_step"]

    # Do conformational search with CREST and take lowest energy one
    os.mkdir("crest")
    with cd("crest"):
        # Set up constraints
        constrained_atoms = set([central_atom, nu_atom, lg_atom])
        constraints = []
        if is_agent:
            constrained_atoms.add(agent_atom)
        if nu_h_atoms:
            for h_atom in nu_h_atoms:
                distance = atoms.get_distance(nu_atom - 1, h_atom - 1)
                constraint = Constraint(nu_atom, h_atom, distance)
                constraints.append(constraint)
        constraints.extend(extra_crest_constraints)
        if ring_atoms:
            constrained_atoms.update(ring_atoms)   
        
        # Change electronic temperature to match the xtb one.
        crest_options = {"el_temp": config.xtb_options["el_temp"]}

        # Do conformational search.
        atoms = search_conformers(atoms,
                                  constrained_atoms=constrained_atoms,
                                  constraints=constraints,
                                  crest_options=crest_options)

    # Pre-optimize TS
    os.mkdir("frozen")
    with cd("frozen"):
        # Set up calculator
        g16 = G16Calculator(atoms, "ts_frozen.gjf", config.dft_options)
        g16.options["opt_acc"] = "loose"
        g16.options["opt_cycles"] = 30
        g16.options["opt_max_step"] = 15
        g16.options["int_acc"] = "fine"

        # Add constraints.
        g16.add_constraint(central_atom, nu_atom, "F")
        g16.add_constraint(central_atom, lg_atom, "F")
        if nu_h_atoms:
            for h_atom in nu_h_atoms:
                g16.add_constraint(nu_atom, h_atom, "F")
        g16.constraints.extend(extra_dft_constraints)

        # Run calculation
        logger.info("Pre-optimizing TS with DFT...")
        g16.opt(n_procs=config.n_procs, mem=config.mem)
        calculation_monitor(g16)
        logger.info("...optimization completed.")

        # Read results.
        data = cclib.io.ccread(g16.output)
        pre_opt_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
        pre_opt_atoms.info["charge"] = atoms.info["charge"]

        # Frequency calculation
        logger.info("Doing frequency calculation on frozen TS...")
        g16 = G16Calculator(pre_opt_atoms, "ts_frozen_freq.gjf", config.dft_options)
        g16.options["oldchk"] = "ts_frozen.chk"                
        g16.options["read_wf"] = True
        g16.freq(n_procs=config.n_procs, mem=config.mem)
        calculation_monitor(g16)
        logger.info("...frequency calculation completed.")

        # Test for negative vib. freq., return None if positive
        data = cclib.io.ccread(g16.output)
        lowest_fc = data.vibfreqs[0]
        if lowest_fc > 0:
            logger.info("No negative eigenvalues: TS optimization aborted.")
            opt_ts = None
            return opt_ts
        
        # Test that negative frequencies correspond to correct displacements
        ts_validator = TSValidator(data, atom_pairs, check_atoms)
        if not ts_validator.validated:
            logger.info("Displacements do not correspond to correct bond changes.")
            opt_ts = None
            return opt_ts

    # Full optimization
    os.mkdir("full")
    with cd("full"):
        # Set up calculator
        g16 = G16Calculator(pre_opt_atoms, "ts_1.gjf", config.dft_options)
        g16.options["oldchk"] = "../frozen/ts_frozen_freq.chk"
        g16.options["read_wf"] = True
        g16.options["read_fc"] = True
        g16.options["opt_cycles"] = 20
        g16.options["opt_max_step"] = ts_max_step

        # Run calculation
        logger.info("Fully optimizing TS with DFT...")
        g16.ts_freq(n_procs=config.n_procs, mem=config.mem)
        calculation_monitor(g16)

        # Get results
        data = cclib.io.ccread(g16.output)

        # Continue with smaller step size if not done.
        if not data.optdone:
            # Set up calculator
            continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
            continue_atoms.info["charge"] = atoms.info["charge"]
            g16 = G16Calculator(continue_atoms, "ts_2.gjf", config.dft_options)
            g16.options["oldchk"] = "ts_1.chk"        
            g16.options["read_wf"] = True
            g16.options["opt_cycles"] = 50
            g16.options["opt_max_step"] = max(ts_max_step - 5, 3)
            
            # Run calculation
            g16.ts_freq(n_procs=config.n_procs, mem=config.mem)
            calculation_monitor(g16)

        logger.info("...optimization completed.")

        # Read output and check for right number of negative frequencies and
        # right displacements.
        data = cclib.io.ccread(g16.output)
        if data.optdone:
            lowest_fc = data.vibfreqs[0]
            if lowest_fc > 0:
                logger.info("No negative eigenvalues: TS optimization aborted.")
                opt_ts = None
                return opt_ts
            ts_validator = TSValidator(data, atom_pairs, check_atoms)
            if not ts_validator.validated:
                logger.info("Displacements do not correspond to correct bond changes.")
                opt_ts = None
                return opt_ts

            # Get the results at the TZ level.
            opt_ts = get_results(g16)
        else:
            opt_ts = None
            return opt_ts
        return opt_ts


def optimize_dft(atoms, standard_state=1):
    """Optimize ground state structure with DFT.

    Args:
        atoms (object): ASE Atoms object.
        standard_state (float): Standard state of species (M)

    Returns:
        opt_atoms (object): ASE Atoms object of optimized structure.
    """
    # Set up calculator
    g16 = G16Calculator(atoms, file="g16.gjf", options=config.dft_options)
    g16.options["chk"] = False
    g16.options["opt_cycles"] = 50
    if atoms.get_number_of_atoms() < 4:
        g16.options["nosymm"] = False

    # Run calculation with displacement.
    g16.opt_freq(n_procs=config.n_procs, mem=config.mem)
    if atoms.get_number_of_atoms() > 2:
        calculation_monitor(g16, displace_imaginary=1)
    else:
        calculation_monitor(g16)

    # Read data
    data = cclib.io.ccread(g16.output)

    # Continue with smaller step size if calculation not done
    if not data.optdone:
        # Set up calculator
        continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
        continue_atoms.info["charge"] = atoms.info["charge"]
        g16 = G16Calculator(continue_atoms, "g16_2.gjf", config.dft_options)
        g16.options["opt_max_step"] = 5
        if continue_atoms.get_number_of_atoms() < 4:
            g16.options["nosymm"] = False

        # Run calculation with displacement
        g16.opt_freq(n_procs=config.n_procs, mem=config.mem)
        if atoms.get_number_of_atoms() > 2:
            calculation_monitor(g16, displace_imaginary=1)
        else:
            calculation_monitor(g16)

    # Get the results at the TZ level.
    opt_atoms = get_results(g16, standard_state=standard_state)

    return opt_atoms


def find_ts_GIC_scan():
    """Find candidate TS structure with GIC scan.
    
    Returns:
        scans (list): Scan objects.
    """
    # Set up atoms object
    rc_atoms = results.xtb_atoms["reaction_complex"]
    pc_atoms = results.xtb_atoms["product_complex"]
    
    # Read in reactive atoms
    central_atom = config.reactive_atoms["central_atom"]
    ortho_carbons = config.reactive_atoms["ortho_carbons"]
    nu_atom = config.reactive_atoms["nu_atom"]
    lg_atom = config.reactive_atoms["lg_atom"]

    # Create scan
    scan = GICTSScan(rc_atoms, config.xtb_options, config.dft_options, config.reactive_atoms)

    # Calculate scan steps
    distance_c_nu_rc = rc_atoms.get_distance(central_atom - 1, nu_atom - 1)
    distance_c_lg_rc = rc_atoms.get_distance(central_atom - 1, lg_atom - 1)
    distance_c_nu_pc = pc_atoms.get_distance(central_atom - 1, nu_atom - 1)
    distance_c_lg_pc = pc_atoms.get_distance(central_atom - 1, lg_atom - 1)
    max_value = distance_c_nu_rc - distance_c_lg_rc
    min_value = distance_c_nu_pc - distance_c_lg_pc
    scan_range = (max_value - min_value) * ANGSTROM_TO_BOHR
    scan_steps = 31
    step_size = -(scan_range / scan_steps)
    
    # Constrain active bonds
    scan.constrain_bond("B1", central_atom, lg_atom, "")
    scan.constrain_bond("B2", central_atom, nu_atom, "")
    for i, c_atom in enumerate(ortho_carbons, start=1):
        scan.constrain_bond(f"O{str(i)}", nu_atom, c_atom, "")
    scan.constrain_definition("Diff", "O1-O2", "Freeze")

    # Add azide constraints
    if config.general_info["azide_nucleophile"]:
        angle_atom_list = config.general_info["azide_angle"]
        scan.constrain_angle("Azide", *angle_atom_list, "Freeze")

    # Initialize scan
    scan.add_scan("Scan", scan_steps, step_size, "B2-B1")
    scan.g16_gic.mod_connectivity.append(f"{central_atom} {nu_atom} 0.5")
    
    # Run scan
    logger.info("Running scan.")
    scan.run_scan(n_procs=max(2, config.n_procs), mem=config.mem)
    calculation_monitor(scan.g16_gic, errors=True, gic=True)
    scan.read_scan_output()
    
    # Check the electronic temperature and re-run if needed
    hl_gap = scan.check_electronic_temperature()
    el_temp = get_electronic_temp(hl_gap)
    result = set_electronic_temp(el_temp)
    if result:
        scan.xtb.options["el_temp"] = config.xtb_options["el_temp"]
        scan.run_scan(n_procs=max(2, config.n_procs), mem=config.mem)
        calculation_monitor(scan.g16_gic, errors=True, gic=True)
        scan.read_scan_output()
    logger.info("...scan completed.")
    
    # Run DFT single-points and find peaks
    logger.info("Running DFT SPs on the scan.")
    scan.run_sps(n_procs=config.n_procs, mem=config.mem)
    scan.read_sp_output()
    scan.find_peaks()
    scan.validate_peaks(intermediate=False, threshold=0.5)
    scan.make_plot()
    logger.info("...single points completed.")

    scans = [scan]

    return scans


def find_ts_scan(step):
    """Find candidate TSs with regular scan.

    Args:
        step (str): Type of step: '1', '2'
    
    Returns:
        scans (list): Scan objects.
    """
    scans = []

    # Set up atoms object
    if results.dft_atoms["intermediate"]:
        intermediate = results.dft_atoms["intermediate"]
    else:
        intermediate = results.xtb_atoms["intermediate"]

    # Set up constraints depending on the step 
    if step == "1":
        constrained = config.reactive_atoms["lg_atom"]
        scanned = config.reactive_atoms["nu_atom"]
        reaction_complex = results.xtb_atoms["reaction_complex"]
    if step == "2":
        constrained = config.reactive_atoms["nu_atom"]
        scanned = config.reactive_atoms["lg_atom"]
        reaction_complex = results.xtb_atoms["product_complex"]
    
    # Read in reactive atoms
    central_atom = config.reactive_atoms["central_atom"]
    ortho_carbons = config.reactive_atoms["ortho_carbons"]
    
    # Set up scan
    # Handle negatively charged nucleophile with assumption of early TS and
    # constraints to prevent collapse to ortho carbons
    scan = TSScan(intermediate, config.xtb_options, config.dft_options, config.reactive_atoms)
    if step == "1" and config.charges["nu_atom"] < 0:
        scan.xtb.options["el_temp"] = 300 
        distance_c_constrained = reaction_complex.get_distance(central_atom - 1, constrained - 1)
        # Add constraint on bonds to keep the bond lengths reasonable for anionic nucleophiles which otherwise suffer from
        # contraction due to negative hyperconjugation.
        if not config.intramolecular:
            for i, i_orig in zip(config.reactive_atoms["nu_sp3_neighbors"], config.reactive_atoms["nu_sp3_neighbors_orig"]):
                dft_nu = results.dft_atoms["nucleophile"]
                nu_atom_orig = config.reactive_atoms["nu_atom_orig"]
                distance = dft_nu.get_distance(nu_atom_orig - 1, i_orig - 1)
                scan.constrain_bond(scanned, i, distance)
    # Handle other cases with assumption of intermediate-like TS where constrained
    # bond lenght is intermediate beteen 
    else:
        distance_c_constrained_1 = reaction_complex.get_distance(central_atom - 1, constrained - 1)
        distance_c_constrained_2 = intermediate.get_distance(central_atom - 1, constrained - 1)
        diff_c_constrained = distance_c_constrained_2 - distance_c_constrained_1
        distance_c_constrained = intermediate.get_distance(central_atom - 1, constrained - 1) - diff_c_constrained / 2
    
    # Set up ortho carbon constraints for step 1.
    if (step == "1" and config.charges["nu_atom"] < 0) or step == "2":
        # Calculate distance to ortho carbons through Pythagoras' theorem
        distance_ortho_1 = intermediate.get_distance(scanned - 1, ortho_carbons[0] - 1)
        distance_ortho_2 = intermediate.get_distance(scanned - 1, ortho_carbons[1] - 1)
        distance_ortho = (distance_ortho_1 + distance_ortho_2) / 2
        max_distance_ortho_1 = reaction_complex.get_distance(scanned - 1, ortho_carbons[0] - 1)
        max_distance_ortho_2 = reaction_complex.get_distance(scanned - 1, ortho_carbons[1] - 1)
        max_distance_ortho = (max_distance_ortho_1 + max_distance_ortho_2) / 2
        for i in ortho_carbons:
            scan.constrain_bond(scanned, i, "auto")
            scan.add_scan(scanned, i, "auto", distance_ortho, max_distance_ortho, 16)        
    
    # Set up bond constraints.
    scan.constrain_bond(central_atom, scanned, "auto")
    scan.constrain_bond(central_atom, constrained, distance_c_constrained)
    max_distance = reaction_complex.get_distance(central_atom - 1, scanned - 1)
    distance_c_scanned = intermediate.get_distance(central_atom - 1, scanned - 1)
    scan.add_scan(central_atom, scanned, "auto", distance_c_scanned, max_distance, 16)
    
    # Run scan    
    logger.info("Running scan.")
    scan.run_scan(n_procs=2).wait()
    scan.read_scan_output()
    
    # Check the electronic temperature and re-run if needed
    if config.charges["nu_atom"] == 0 and step == "1":
        hl_gap = scan.check_electronic_temperature()
        el_temp = get_electronic_temp(hl_gap, neutral_scan=True)
        if el_temp != config.xtb_options["el_temp"]:
            if el_temp > 2000:
                set_electronic_temp(el_temp)
                scan.xtb.options["el_temp"] = config.xtb_options["el_temp"]
            else:
                scan.xtb.options["el_temp"] = el_temp
            scan.run_scan(n_procs=2).wait()
            scan.read_scan_output()

    logger.info("...scan completed.")

    # Run DFT single-points and find peaks
    logger.info("Running DFT SPs on the scan.")
    scan.run_sps(n_procs=config.n_procs, mem=config.mem)
    scan.read_sp_output()
    scan.find_peaks()
    scan.validate_peaks(intermediate=True, threshold=0.5)
    scan.make_plot()
    logger.info("...single points completed.")

    scans.append(scan)

    # If step 2 and proton transfer necessary, do GIC with proton transfer
    if config.general_info["proton_transfer"] and step == "2":
        os.mkdir("gic_scan")
        with cd("gic_scan"):
            # Take out Nu-H atom closest to LG in the intermediate and calculate distances
            lg = results.xtb_atoms["leaving_group"]
            nu_h_atoms = config.reactive_atoms["nu_h_atoms"]
            lg_h_atoms = config.reactive_atoms["lg_h_atoms"]
            lg_atom_orig = config.reactive_atoms["lg_atom_orig"]
            distances = [intermediate.get_distance(scanned - 1, atom - 1) for atom in nu_h_atoms]
            h_atom = nu_h_atoms[np.argmin(distances)]
            other_h_atoms = [i for i in nu_h_atoms if i != h_atom]
            init_lg_h = np.min(distances)
            distances = [lg.get_distance(lg_atom_orig - 1, atom - 1) for atom in lg_h_atoms]
            final_lg_h = np.mean(distances)
    
            # Create scan
            scan = GICTSScan(intermediate, config.xtb_options, config.dft_options, config.reactive_atoms)
    
            # Calculate scan steps
            init_c_lg = distance_c_scanned
            final_c_lg = max_distance
            init_value = init_c_lg - init_lg_h
            final_value = final_c_lg - final_lg_h
            scan_range = (final_value - init_value) * ANGSTROM_TO_BOHR
            scan_steps = 15
            step_size = scan_range / scan_steps
            
            # Constrain active bonds
            scan.constrain_bond("B1", scanned, h_atom, "")
            scan.constrain_bond("B2", central_atom, scanned, "")
            scan.constrain_bond("B3", central_atom, constrained, "Freeze")
            scan.constrain_bond("B4", constrained, h_atom, "Kill")

            for i, atom in enumerate(other_h_atoms, start=1):
                scan.constrain_bond(f"H{i}", constrained, atom, "Freeze")
        
            # Initialize scan. Set eltemp to 300 to force proton transfer
            scan.add_scan("Scan", scan_steps, step_size, "B2-B1")
            scan.g16_gic.mod_connectivity.append(f"{scanned} {h_atom} 0.5")
            scan.xtb.options["el_temp"] = 300

            # Run scan
            logger.info("Running scan.")
            scan.run_scan(n_procs=max(2, config.n_procs), mem=config.mem)
            calculation_monitor(scan.g16_gic, errors=True, gic=True)
            scan.read_scan_output()
            
            # Run DFT single-points and find peaks
            logger.info("Running DFT SPs on the scan.")
            scan.run_sps(n_procs=config.n_procs, mem=config.mem)
            scan.read_sp_output()
            scan.find_peaks()
            scan.validate_peaks(intermediate=False, threshold=0.5)
            scan.make_plot()
            logger.info("...single points completed.")
    
            scans.append(scan)

    return scans


def find_ts(scans, step="1"):
    """Locate transition states from candidate peaks of scans.

    Args:
        scans (list): Scans with store candidate peaks.
        step (str): Type of step '1', '2' or 'concerted'
    """
    # Set up reactive atoms
    central_atom = config.reactive_atoms["central_atom"]
    nu_atom = config.reactive_atoms["nu_atom"]
    lg_atom = config.reactive_atoms["lg_atom"]

    # Sort out proton transfer GIC scans
    #!TODO this is a bit ugly way to check.
    gic_scan = None
    if step == "2":
        for scan in scans:
            if hasattr(scan, "g16_gic"):
                gic_scan = scan

    # Loop through scans to get trial TS structures.
    ts_list = []
    gic_ts_list = []
    for scan in scans:
        if len(scan.peaks) > 0:
            if len(scan.peaks) > 1:
                logger.info("Several peaks found. Take the one with the highest energy.")
            peak_heights = [peak.energy for peak in scan.peaks]
            sorted_peaks = [peak for height, peak in sorted(zip(peak_heights, scan.peaks), key=lambda x: x[0], reverse=True)]
            peak_geometries = [scan.geometries[peak.maximum] for peak in sorted_peaks]
            if scan is gic_scan:
                gic_ts_list.extend(peak_geometries)
            else:
                ts_list.extend(peak_geometries)
    # If no peaks are found, but we know that there is an intermediate,
    # choose a step one way away from the intermediate
    if len(ts_list) == 0:
            if not config.concerted:
                if step == "2":
                    if scans[0].dft_energies[0] > scans[0].dft_energies[-1]:
                        logger.info("No peaks found, but energy decreases and intermediate is found. Optimize TS at first point after intermediate.")
                        ts_list.append(scans[0].geometries[1])        
    # Optimize TSs
    opt_ts_list = []
    logger.info(f"Found {len(ts_list)} TSs to optimize.")
    for counter, ts in enumerate(ts_list, start=1):
        os.mkdir(f"ts_{counter}")
        with cd(f"ts_{counter}"):
            atom_pairs = [[central_atom, nu_atom], [central_atom, lg_atom]]
            if step == "1" or step == "concerted":
                check_atoms = [nu_atom]
            elif step == "2":
                check_atoms = [lg_atom]                
            logger.info(f"Optimizing TS {counter}")
            opt_ts = optimize_ts(ts, atom_pairs, check_atoms)
            logger.info(f"Finished optimizing TS {counter}")
            if opt_ts:
                logger.info("A TS found. Ignoring any other peaks.")
                opt_ts_list.append(opt_ts)
                ase.io.write(f"../../optimized_structures/step_{step}_ts_{counter}.xyz", opt_ts, plain=True)
                break
            else:
                logger.info("TS could not be optimized.")

        logger.info(f"Optimized {len(opt_ts_list)} TSs for step {step}")

    # If no TSs found, we try with the reserve GIC scans for proton transfers.
    if len(opt_ts_list) < 1:
        if step == "2":
            if len(gic_ts_list) > 0:
                logger.info("Attempting with TS from proton transfer GIC scan.")
                for counter, ts in enumerate(gic_ts_list, start=1):
                    # Set up active atoms for TS validation.
                    atom_pairs = [[central_atom, nu_atom]]
                    check_atoms = [lg_atom]
                    
                    # Add constraints
                    extra_crest_constraints = []
                    extra_dft_constraints = []
                    for constraint in gic_scan.g16_gic.gic_constraints:
                        if constraint.name == "B1":
                            atom_pairs.append([lg_atom, constraint.atoms[1]])
                            check_atoms.append(constraint.atoms[1])
                            extra_dft_constraints.append(Constraint(constraint.atoms[0], constraint.atoms[1], "F"))
                            distance = ts.get_distance(constraint.atoms[0] - 1, constraint.atoms[1] -1)
                            extra_crest_constraints.append(Constraint(constraint.atoms[0], constraint.atoms[1], distance))
                    
                    # Optimize TS structures.
                    os.mkdir(f"gic_ts_{counter}")
                    with cd(f"gic_ts_{counter}"):
                        logger.info(f"Optimizing GIC TS {counter}")
                        opt_ts = optimize_ts(ts, atom_pairs, check_atoms, extra_dft_constraints=extra_dft_constraints, extra_crest_constraints=extra_crest_constraints)
                        logger.info(f"Finished optimizing GIC TS {counter}")
                        if opt_ts:
                            logger.info("A GIC TS found. Ignoring any other peaks.")
                            opt_ts_list.append(opt_ts)
                            ase.io.write(f"../../optimized_structures/step_{step}_ts_{counter}.xyz", opt_ts, plain=True)
                            break
                        else:
                            logger.info("GIC TS could not be optimized.")                
            else:
                logger.info("No TS found for step 2 despite intermediate. Very flat PES. Assuming TS for step 1 is rate determining.")
                config.flat_PES = True
        else:
            logger.info("No TS found. Reaction could be barrierless.")

    # Store results.
    results.dft_atoms["ts"].extend(opt_ts_list)       


def do_clustering_xtb(cutoff=2, opt_dft=True, crest_dft_ranking=False):
    """Do explicit solvent clustering with XTB
    
    Args:
        cutoff (float): Cutoff above which to consider clustering (kcal/mol).
        opt_dft (bool): Optimize cluster with DFT.
        crest_dft_ranking (bool): Rank cluster conformers with DFT
    """
    # Check if solvent contains hydrogen bond donors. Otherwise exit
    solvent_smiles = config.general_options["solvent_smiles"]
    solvent = Chem.MolFromSmiles(solvent_smiles)
    smarts = "[!H0;#7,#8,#9]" # N, O, F with at least one hydrogen atom
    pattern = Chem.MolFromSmarts(smarts)
    has_h_donors = solvent.HasSubstructMatch(pattern)
    logging.info("No H bond donors in solvent. Don't do clustering.")
    if not has_h_donors:
        return
    
    # Set up jobs
    jobs = []
    candidates = ["leaving_group"]
    if not config.intramolecular:
        candidates.append("nucleophile")
    for name in candidates:
        if config.clustering[name] and config.general_options[f"cluster_{name}"]:
            jobs.append(name)
    if config.clustering["ts"] and config.general_options[f"cluster_ts"]:
        for i in range(1, len(results.dft_atoms["ts"]) + 1):
            jobs.append(f"ts_{i}")
    if config.agent:
        if config.clustering["agent"] and config.general_info["agent"]:
            jobs.append("agent")
    
    if not any(jobs):
        return

    # Save mols for OH- in water check
    hydroxide_mol = Chem.MolFromSmiles("[OH-]")
    water_mol = Chem.MolFromSmiles("O")
    solvent_mol = Chem.Mol(solvent)

    # Create solvent molecule
    solvent = Chem.AddHs(solvent)
    AllChem.EmbedMolecule(solvent, randomSeed=1)

    # Convert solvent to ASE atoms object
    elements = [atom.GetSymbol() for atom in solvent.GetAtoms()]
    coordinates = solvent.GetConformer().GetPositions()
    charge = Chem.GetFormalCharge(solvent)
    solvent = Atoms(elements, coordinates)
    solvent.info["charge"] = charge  

    opt_solvent_dft = None

    os.mkdir("clustering")
    with cd("clustering"):
        clustering_energies = {}
        for name in jobs:
            os.mkdir(name)
            with cd(name):
                logger.info(f"Starting clustering calculation for {name}.")
                # Perform preliminary clustering
                if "ts" in name:
                    number = int(name.split("_")[1]) - 1
                    solute = results.dft_atoms["ts"][number]
                else:
                    # TODO change this to DFT atoms if appropriate to get better
                    # guess
                    solute = results.xtb_atoms[name]

                # Detect if OH- is the solute and H2O the solvent
                hydroxide_in_water = False
                if name == "nucleophile" or name == "leaving_group":
                    solute_mol = Chem.MolFromSmiles(results.smiles[name])
                    crit_1_1 = hydroxide_mol.GetNumAtoms() == solute_mol.GetNumAtoms()
                    crit_1_2 = len(solute_mol.GetSubstructMatch(hydroxide_mol)) == solute_mol.GetNumAtoms()
                    crit_2_1 = water_mol.GetNumAtoms() == solute_mol.GetNumAtoms()
                    crit_2_2 = len(solvent_mol.GetSubstructMatch(water_mol)) == solvent_mol.GetNumAtoms()
    
                    if all([crit_1_1, crit_1_2, crit_2_1, crit_2_2]):
                        hydroxide_in_water = True                 

                #  Set up constraints if TS present
                constraints = []
                constrained_atoms = set()
                if "ts" in name:
                    is_agent = config.general_info["agent"]
                    central_atom = config.reactive_atoms["central_atom"]
                    nu_atom = config.reactive_atoms["nu_atom"]
                    lg_atom = config.reactive_atoms["lg_atom"]
                    ring_atoms = config.reactive_atoms["ring_atoms"]
                    nu_h_atoms = config.reactive_atoms["nu_h_atoms"]

                    constrained_atoms.update([central_atom, nu_atom, lg_atom])
                    #if is_agent:
                    #    agent_atom = config.reactive_atoms["agent_atom"]
                    #    coordinated_h_atom = config.reactive_atoms["nu_h_atoms"][0]
                        #agent_h_distance = solute.get_distance(agent_atom - 1, coordinated_h_atom - 1)
                    #    constrained_atoms.update([agent_atom, coordinated_h_atom])
                    if nu_h_atoms:
                        for h_atom in nu_h_atoms:
                            distance = solute.get_distance(nu_atom - 1, h_atom - 1)
                            constraint = Constraint(nu_atom, h_atom, distance)
                            constraints.append(constraint)
                    if ring_atoms:
                        constrained_atoms.update(ring_atoms)
                    #if is_agent:
                    #    constraints.append(Constraint(nu_atom, coordinated_h_atom, agent_h_distance))                   
                
                # Run cluster generation
                opt_cluster, opt_solvent, best_n, best_energy = cluster_molecule_xtb(solute, solvent, 
                    constrained_atoms=constrained_atoms, n_solvent=4, dft_sps=True, optimize=True, crest_dft_ranking=crest_dft_ranking,
                    hydroxide_in_water=hydroxide_in_water)
    
                # Test for unstable cluster
                if best_n == 0:
                    logger.info("Cluster not better than continuum solvent. Aborting.")
                    config.clustering[name] = False
                    clustering_energies[name] = {"xtb": {"clustering_energy": 0.0},
                                                "dft": {"clustering_energy": 0.0}}
                    continue
        
                # Test predicted clustering energy against cutoff
                if abs(best_energy) < cutoff:
                    logger.info(f"Predicted cluster solvation energy of {best_energy:.2f} lower than cutoff of {cutoff:.2f}. Aborting clustering procedure")
                    config.clustering[name] = False
                    clustering_energies[name] = {"xtb": {"clustering_energy": 0.0},
                                                "dft": {"clustering_energy": 0.0}}                  
                    continue
                
                # Refine clustering energy with DFT
                if opt_dft:
                    if hydroxide_in_water:
                        # Add bond constraints in case of hydroxide in water
                        analysis = Analysis(opt_cluster, skin=0.1)
                        for i, entry in enumerate(analysis.unique_bonds[0]):
                            for j in entry:
                                distance = opt_cluster.get_distance(i, j)
                                constraints.append(Constraint(i + 1, j + 1, distance))                         
                    logger.info("Optimizing best cluster with DFT.")
                    os.mkdir("dft")
                    with cd("dft"):
                        # Redo conformational sampling with DFT ranking
                        os.mkdir("crest")
                        with cd("crest"):
                            logger.info("Performing conformational search...")
                            crest_options = {"nci": True, "solvent": config.xtb_options["solvent_cluster"], "el_temp": config.xtb_options["el_temp"]}
                            opt_cluster = search_conformers(opt_cluster, constrained_atoms=constrained_atoms, constraints=constraints, crest_options=crest_options)
                            logger.info("...conformational search done.")
    
                        # Do initial relaxation
                        logger.info("Optimizing with DFT...")
                        g16 = G16Calculator(opt_cluster, "cluster_pre.gjf", config.dft_options)
                        if "ts" in name:
                            g16.options["opt_acc"] = "loose"
                            g16.options["opt_cycles"] = 30
                            g16.options["opt_max_step"] = 15
                            g16.options["int_acc"] = "fine"
                            g16.add_constraint(central_atom, nu_atom, "F")
                            g16.add_constraint(central_atom, lg_atom, "F")
                            g16.opt(n_procs=config.n_procs, mem=config.mem)
                        else:
                            g16.options["opt_cycles"] = 10
                            g16.options["opt_max_step"] = 30
                            g16.opt_freq(n_procs=config.n_procs, mem=config.mem)
                        calculation_monitor(g16)

                        data = cclib.io.ccread(g16.output)
    
                        # Decrease step size
                        if "ts" in name:
                            continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
                            continue_atoms.info["charge"] = opt_cluster.info["charge"]                            
                            g16 = G16Calculator(continue_atoms, "cluster.gjf", config.dft_options)
                            g16.options["oldchk"] = "cluster_pre.chk"
                            g16.options["read_wf"] = True
                            g16.options["opt_cycles"] = 20
                            g16.options["opt_max_step"] = config.general_options["ts_max_step"]
                            g16.ts_freq(n_procs=config.n_procs, mem=config.mem)
                            calculation_monitor(g16)
                            data = cclib.io.ccread(g16.output)
                        else:
                            if not data.optdone:
                                continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
                                continue_atoms.info["charge"] = opt_cluster.info["charge"]
                                g16 = G16Calculator(continue_atoms, "cluster.gjf", config.dft_options)
                                g16.options["oldchk"] = "cluster_pre.chk"        
                                g16.options["read_wf"] = True
                                g16.options["opt_cycles"] = 50
                                g16.options["opt_max_step"] = 5
                                g16.opt_freq(n_procs=config.n_procs, mem=config.mem)
                                calculation_monitor(g16)
                                data = cclib.io.ccread(g16.output)
                        
                        # Do new step and perform frequency calculation anyway if not converged.
                        if not data.optdone:
                            if "ts" in name:
                                charge = continue_atoms.info["charge"]
                                continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
                                continue_atoms.info["charge"] = charge
                                g16 = G16Calculator(continue_atoms, "cluster_cont.gjf", config.dft_options)
                                g16.options["oldchk"] = "cluster.chk"        
                                g16.options["read_wf"] = True
                                g16.options["opt_cycles"] = 50
                                g16.options["opt_max_step"] = max(config.general_options["ts_max_step"] - 5, 3)
                                g16.ts_freq(n_procs=config.n_procs, mem=config.mem)                               
                            else:
                                continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
                                continue_atoms.info["charge"] = opt_cluster.info["charge"]
                                g16 = G16Calculator(continue_atoms, "cluster_freq.gjf", config.dft_options)
                                g16.options["oldchk"] = "cluster.chk"        
                                g16.options["read_wf"] = True
                                g16.freq(n_procs=config.n_procs, mem=config.mem)
                            calculation_monitor(g16)
                            data = cclib.io.ccread(g16.output)

                        # Get the final results for the cluster at the TZ level.
                        if "ts" in name:
                            if not data.optdone:
                                continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
                                continue_atoms.info["charge"] = opt_cluster.info["charge"]
                                g16 = G16Calculator(continue_atoms, "cluster_freq.gjf", config.dft_options)
                                g16.options["oldchk"] = "cluster_cont.chk"        
                                g16.options["read_wf"] = True  

                        logger.info("...optimization completed.")
                        dft_cluster = get_results(g16)

            if opt_dft:
                # Optimize solvent
                if not opt_solvent_dft:
                    logger.info("Optimizing solvent.")
                    os.mkdir("solvent")
                    with cd("solvent"):
                        os.mkdir("crest")
                        with cd("crest"):
                            logger.info("Performing conformational search...")
                            opt_solvent = search_conformers(opt_solvent)
                            logger.info("...conformational search done.")                            
                        os.mkdir("dft")
                        with cd("dft"):
                            standard_state = config.general_options["standard_state"]
                            logger.info("Optimizing solvent with DFT...")
                            opt_solvent_dft = optimize_dft(opt_solvent, standard_state=standard_state)
                            logger.info("Optimization done...")
        
                # Compute clustering energy
                free_energy_cluster = dft_cluster.info["free_energy"]
                free_energy_cluster_qh_grimme = dft_cluster.info["free_energy_qh_grimme"]
                free_energy_cluster_qh_truhlar = dft_cluster.info["free_energy_qh_truhlar"]
                
                free_energy_solvent = opt_solvent_dft.info["free_energy"]
                free_energy_solvent_qh_grimme = opt_solvent_dft.info["free_energy_qh_grimme"]
                free_energy_solvent_qh_truhlar = opt_solvent_dft.info["free_energy_qh_truhlar"]
        
                if "ts" in name:
                    number = int(name.split("_")[1]) - 1
                    solute = results.dft_atoms["ts"][number]
                else:
                    #TODO change this to DFT atoms
                    solute = results.dft_atoms[name]
                free_energy_solute = solute.info["free_energy"]
                free_energy_solute_qh_grimme = solute.info["free_energy_qh_grimme"]
                free_energy_solute_qh_truhlar = solute.info["free_energy_qh_truhlar"]
        
                clustering_energy = free_energy_cluster - free_energy_solute - best_n * free_energy_solvent
                clustering_energy_qh_grimme = free_energy_cluster_qh_grimme - free_energy_solute_qh_grimme - best_n * free_energy_solvent_qh_grimme
                clustering_energy_qh_truhlar = free_energy_cluster_qh_truhlar - free_energy_solute_qh_truhlar - best_n * free_energy_solvent_qh_truhlar
                
                best_energy_ha = best_energy * KCAL_TO_HARTREE

                logger.info(f"Final clustering energy: {clustering_energy * HARTREE_TO_KCAL: .1f}")
    
                dft_energies = {
                    "clustering_energy": clustering_energy,
                    "clustering_energy_qh_grimme": clustering_energy_qh_grimme,
                    "clustering_energy_qh_truhlar": clustering_energy_qh_truhlar,
                    }
                xtb_energies = {
                    "clustering_energy": best_energy_ha
                    }
                clustering_energies[name] = {"xtb": xtb_energies, "dft": dft_energies}
            else:
                xtb_energies = {
                    "clustering_energy": best_energy_ha
                    }
                clustering_energies[name] = {"xtb": xtb_energies}
        
        if config.clustering["ts"] and config.general_options[f"cluster_ts"]:
            results_ts = []
            for i in range(1, len(results.dft_atoms["ts"]) + 1):
                results_ts.append(clustering_energies.pop(f"ts_{i}"))
            clustering_energies["ts"] = results_ts
            #!TODO Look to see if this allows for multiple clustering energies.
        results.clustering_energies = clustering_energies


def cluster_molecule_xtb(solute, solvent, n_solvent=4, constrained_atoms=None, constraints=None,
                         crest_dft_ranking=False, dft_sps=False, optimize=False, hydroxide_in_water=False):
    """Generate cluster with XTB.

    Args:  
        constrained_atoms (set): Atoms to constrain during the calculation.
        constraints (list): Bond constraints.
        crest_dft_ranking (bool): Whether to rank the cluster conformers
            with DFT.
        dft_sps (bool): Whether to get the electornic energies at the DFT level.
        hydroxide_in_water (bool): Whether to use special constraints for 
            hydroxide in water to avoid artifacts with xtb.
        n_solvent (int): Maximum number of solvent molecules.
        optimize (bool): Whether to optimize the number of solvent molecules
            or just go with 'n_solvent'
        solute (object): ASE Atoms object of solute
        solvent (object): ASE Atoms object of solvent

    Returns:
        best_cluster (object): ASE Atoms object of best cluster
        best_energy (object): Stabilization free energy of cluster
            vs. continuum.
        best_n (int): Number of solvent molecules in best cluster.
        opt_solvent (object): ASE Atoms object of optimized solvent
    """
    # Copy solute and solvent molecules
    solute = solute.copy()
    solvent = solvent.copy()

    # Set up constraint lists
    if not constrained_atoms:
        constrained_atoms = set()
    if not constraints:
        constraints = []

    # Set up dictionary for energies
    energies = {"xtb": {}, "dft": {}}

    # Do conformational search and optimize solvent
    logger.info("Optimizing solvent...")
    os.mkdir("solvent")
    with cd("solvent"):
        # Do conformational search for solvent.
        os.mkdir("crest")
        with cd("crest"):
            crest_options = {"solvent": config.xtb_options["solvent_cluster"]}
            solvent = search_conformers(solvent, dft_ranking=crest_dft_ranking, crest_options=crest_options)
        # Optimize solvent with xtb.
        os.mkdir("xtb")
        with cd("xtb"):
            # Set up calculation and do optimization
            xtb = XTBCalculator(solvent, "xtb.xyz", config.xtb_options)
            xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
            xtb.options["temperature"] = config.general_options["temperature"]
            xtb.options["el_temp"] = 300
            xtb.opt_freq().wait()

            # Read data and calculate free energy.
            data = XTBParser("xtb.out")
            opt_solvent = ase.io.read("xtbopt.xyz")
            opt_solvent.info["charge"] = solvent.info["charge"]
            free_energy_solvent_xtb = data.free_energy
            free_energy_corr_solvent_xtb = data.free_energy - data.energy
            ss_state = config.general_options["standard_state"]
            ss_correction = standard_state_correction(ss_state, reference="M", temperature=config.general_options["temperature"])
            free_energy_corr_solvent_xtb += ss_correction
            free_energy_solvent_xtb += ss_correction
            energies["xtb"]["solvent"] = free_energy_solvent_xtb
        logger.info("...optimization done.")
        
        # Correct energy with DFT
        if dft_sps:
            os.mkdir("sp")
            with cd("sp"):
                logger.info("Refining energy with DFT.")

                # Set up calculator
                g16 = G16Calculator(opt_solvent, file="solvent.gjf", options=config.dft_options)
                g16.options["int_acc"] = "fine"
                g16.options["scf_acc"] = "sleazy"
                g16.options["chk"] = False
                g16.options["solvation_model"] = config.dft_sp_options["solvation_model"]

                # Do calculation
                g16.single_point(n_procs=config.n_procs, mem=config.mem)
                calculation_monitor(g16, errors=True)

                # Read result and calculate free energy.
                data = cclib.io.ccread(g16.output)
                electronic_energy_solvent_dft = data.scfenergies[-1] * EV_TO_HARTREE
                free_energy_solvent_dft = electronic_energy_solvent_dft + free_energy_corr_solvent_xtb
                energies["dft"]["solvent"] = free_energy_solvent_dft

    # Do conformational search and optimize solute
    logger.info("Optimizing solute...")
 
    os.mkdir("solute")                
    with cd("solute"):
        # Do conformational sampling
        os.mkdir("crest")
        with cd("crest"):
            crest_options = {"solvent": config.xtb_options["solvent_cluster"]}
            solute = search_conformers(solute, constrained_atoms=constrained_atoms, constraints=constraints, dft_ranking=crest_dft_ranking, crest_options=crest_options)
        
        #Optimize with xtb
        os.mkdir("xtb")
        with cd("xtb"):
            # Set up caluclation
            xtb = XTBCalculator(solute, "xtb.xyz", config.xtb_options)
            xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
            xtb.options["temperature"] = config.general_options["temperature"]
            xtb.options["el_temp"] = 300
            # Do sinlge point for only one atom as frequency calculation
            # crashes.
            if solute.get_number_of_atoms() == 1:
                xtb.single_point().wait()
                opt_solute = solute
            else:
                # Add constraints.
                for constraint in constraints:
                    xtb.add_constraint(constraint.atom_1, constraint.atom_2, constraint.value)           
                xtb.constrained_atoms.update(constrained_atoms)
                xtb.options["force_constant"] = 2.0
                # Perform optimization
                if len(constrained_atoms) > 0 or len(constraints) > 0:
                    # Optimization
                    xtb.opt().wait()
                    opt_solute = ase.io.read("xtbopt.xyz")
                    opt_solute.info["charge"] = solute.info["charge"]

                    # Frequency calculation
                    xtb = XTBCalculator(solute, "xtb.xyz", config.xtb_options)
                    xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
                    xtb.options["temperature"] = config.general_options["temperature"]
                    xtb.options["el_temp"] = 300
                    xtb.freq().wait()
                else:
                    xtb.opt_freq().wait()

                # Get the resulting structure      
                opt_solute = ase.io.read("xtbopt.xyz")
                opt_solute.info["charge"] = solute.info["charge"]                
            
            # Read the data
            data = XTBParser("xtb.out")

            # Calculate free energy. Special treatment for atoms.
            if solute.get_number_of_atoms() == 1:
                atomic_number = solute.get_atomic_numbers()[0]
                mass = atomic_masses[atomic_number]
                enthalpy, t_entropy = thermal_analysis_atom(mass, reference="M", temperature=config.general_options["temperature"])
                free_energy_corr_solute_xtb = enthalpy - t_entropy
                free_energy_solute_xtb = data.energy + free_energy_corr_solute_xtb                
            else:
                free_energy_solute_xtb = data.free_energy
                free_energy_corr_solute_xtb = data.free_energy - data.energy
            energies["xtb"]["solute"] = free_energy_solute_xtb
        logger.info("...optimization done.")          

        # Refine energy with DFT.
        if dft_sps:
            os.mkdir("sp")
            with cd("sp"):
                logger.info("Refining energy with DFT.")
                
                # Set up calculation
                g16 = G16Calculator(opt_solute, file="solute.gjf", options=config.dft_options)
                g16.options["int_acc"] = "fine"
                g16.options["scf_acc"] = "sleazy"
                g16.options["chk"] = False
                g16.options["solvation_model"] = config.dft_sp_options["solvation_model"]

                # Perform calculation
                g16.single_point(n_procs=config.n_procs, mem=config.mem)
                calculation_monitor(g16, errors=True)

                # Read output and calculate free energy.
                data = cclib.io.ccread(g16.output)
                electronic_energy_solute_dft = data.scfenergies[-1] * EV_TO_HARTREE
                free_energy_solute_dft = electronic_energy_solute_dft + free_energy_corr_solute_xtb
                energies["dft"]["solute"] = free_energy_solute_dft

    # Setting up number of jobs
    if optimize:
        jobs = range(1, n_solvent + 1)
    else:
        jobs = [n_solvent]
    
    # Set up output file
    with open("cluster_energies", "w") as file:
        file.write(f"{'n':>5s}{'XTB':>10s}{'DFT':>10s}\n")
    
    results = {0: {"energy": 0.0, "cluster": opt_solute}}
    for n in jobs:
        os.mkdir(f"{n}")
        with cd(f"{n}"):
            logger.info(f"Optimizing cluster with {n} solvent molecules...")            
            # Get cluster at MM level
            cluster_molecule = ClusterMoleculeXTB(solute, solvent, n, hydroxide_in_water)
            cluster = cluster_molecule.cluster

            # Add constraints in case of hydroxide in water
            analysis = Analysis(cluster, skin=0.1)
            if hydroxide_in_water:
                for i, entry in enumerate(analysis.unique_bonds[0]):
                    for j in entry:
                        distance = cluster.get_distance(i, j)
                        constraints.append(Constraint(i + 1, j + 1, distance))
        
            # Search for most stable conformer with CREST
            os.mkdir("crest")
            with cd("crest"):
                crest_options = {"nci": True, "solvent": config.xtb_options["solvent_cluster"]}
                cluster = search_conformers(cluster, constrained_atoms=constrained_atoms, constraints=constraints, dft_ranking=crest_dft_ranking, crest_options=crest_options)
            
            # Optimize with XTB
            os.mkdir("xtb")
            with cd("xtb"):
                # Set up calculator
                xtb = XTBCalculator(cluster, "xtb.xyz", config.xtb_options)
                xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
                xtb.options["temperature"] = config.general_options["temperature"]
                xtb.options["force_constant"] = 2.0
                xtb.options["el_temp"] = 300

                # Add constraints
                xtb.constrained_atoms.update(constrained_atoms)
                for constraint in constraints:
                    xtb.add_constraint(constraint.atom_1, constraint.atom_2, constraint.value)
                
                # Perform optimization and frequency calculation.
                if len(constrained_atoms) > 0 or len(constraints) > 0:
                    # Optimization
                    xtb.opt().wait()
                    opt_cluster = ase.io.read("xtbopt.xyz")
                    opt_cluster.info["charge"] = solute.info["charge"]

                    # Frequency calculation
                    xtb = XTBCalculator(opt_cluster, "xtb.xyz", config.xtb_options)
                    xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
                    xtb.options["temperature"] = config.general_options["temperature"]
                    xtb.options["el_temp"] = 300
                    xtb.freq().wait()
                else:
                    xtb.opt_freq().wait()                              
                
                # Read the data and calculate free energy.
                opt_cluster = ase.io.read("xtbopt.xyz")
                opt_cluster.info["charge"] = cluster.info["charge"]
                data = XTBParser("xtb.out")
                free_energy_cluster_xtb = data.free_energy
                free_energy_corr_cluster_xtb = data.free_energy - data.energy

                # Calculate stabilization energy
                stabilization_energy_xtb = (free_energy_cluster_xtb - energies["xtb"]["solute"] - n * energies["xtb"]["solvent"]) * HARTREE_TO_KCAL 
            logger.info("...Optimization done.")               

            # Refine energy at the DFT level
            if dft_sps:
                os.mkdir("sp")
                with cd("sp"):
                    logger.info("Refining energy with DFT.")

                    # Set up calculator
                    g16 = G16Calculator(opt_cluster, file="solute.gjf", options=config.dft_options)
                    g16.options["int_acc"] = "fine"
                    g16.options["scf_acc"] = "sleazy"
                    g16.options["chk"] = False
                    g16.options["solvation_model"] = config.dft_sp_options["solvation_model"]

                    # Perform calculation
                    g16.single_point(n_procs=config.n_procs, mem=config.mem)
                    calculation_monitor(g16, errors=True)

                    # Read results and calculate free energy
                    data = cclib.io.ccread(g16.output)
                    electronic_energy_cluster_dft = data.scfenergies[-1] * EV_TO_HARTREE
                    free_energy_cluster_dft = electronic_energy_cluster_dft + free_energy_corr_cluster_xtb

                    # Calculate stabilization energy
                    stabilization_energy_dft = (free_energy_cluster_dft - energies["dft"]["solute"] - n * energies["dft"]["solvent"]) * HARTREE_TO_KCAL
            else:
                stabilization_energy_dft = None
            
            # Save results to log file.
            with open("../cluster_energies", "a") as file:
                file.write(f"{n:5}{stabilization_energy_xtb:10.2f}{stabilization_energy_dft:10.2f}\n")

            # Select the best stabilization energy.
            if dft_sps:
                stabilization_energy = stabilization_energy_dft
            else:
                stabilization_energy = stabilization_energy_xtb

            # Store the in results dictionary
            results[n] = {"energy": stabilization_energy, "cluster": opt_cluster}
            logger.info(f"Stabilization energy: {stabilization_energy}")
            
            # Compare to previous stabilization energy and decide if to continue
            previous_energy = results[n - 1]["energy"]
            if stabilization_energy > previous_energy:
                logger.info("Energy rises. Aborting iterative clustering.")
                best_n = n - 1
                break
            else:
                best_n = n
    
    # Get the best energy and cluster
    best_energy = results[best_n]["energy"]
    best_cluster = results[best_n]["cluster"]
    
    # Write best cluster to file
    ase.io.write(f"cluster.xyz", best_cluster, plain=True)
    
    logger.info(f"Stabilization energy for {best_n} solvent molecules: {best_energy}")

    return best_cluster, opt_solvent, best_n, best_energy


class ClusterMoleculeXTB:
    """Generate candidate cluster with the help of XTB
    
    Args:
        hydroxide_in_water (bool): Whether to include corrections for hydroxide
            in water to correct XTB problems.
        molecule (object): ASE Atoms object of solute
        n_solvent (int): Number of solvent molecules in cluster
        solvent (object): ASE Atoms object of solvent

    Attributes:
        cluster (object): Cluster candidate as ASE Atoms object.
        hydroxide_in_water (bool): Whether to include corrections for hydroxide
            in water to correct XTB problems.        
        molecule (object): ASE Atoms object of solute
        n_solvent (int): Number of solvent molecules in cluster
        solvent (object): ASE Atoms object of solvent
    """
    def __init__(self, molecule, solvent, n_solvent, hydroxide_in_water=False):
        # Make copies of molecule and solvent
        molecule = molecule.copy()
        solvent = solvent.copy()

        # Derive radius of solvent sphere
        r_s = []
        for mol in [molecule, solvent]:
            coordinates = mol.get_positions()
            com = np.sum(coordinates, axis=0)
            dists = np.linalg.norm(coordinates - com, axis=1)
            r = np.max(dists)
            r_s.append(r)
            
        r_sphere = sum(r_s) + 4            
        
        # Construct solvents equidistant on the sphere
        solvent_points = self._get_equidistant(r_sphere, n_solvent)

        # Center all molecules at their center of mass
        for mol in [molecule, solvent]:
            com_positions = mol.get_positions() - mol.get_center_of_mass()
            mol.set_positions(com_positions)
                                         
        cluster = molecule
        solute_indices = list(range(cluster.get_number_of_atoms()))
        solvent_indices = []
        for solvent_point in solvent_points:
            solvent_copy = solvent.copy()
            new_positions = solvent_copy.get_positions() + solvent_point
            solvent_copy.set_positions(new_positions)
            indices = list(range(cluster.get_number_of_atoms(), cluster.get_number_of_atoms() + solvent_copy.get_number_of_atoms()))
            cluster.extend(solvent_copy)
            solvent_indices.append(indices)
               
        # Optimize reasonable cluster
        os.mkdir("preopt")
        with cd("preopt"):
            # Shrink solvent sphere with weak constraints.
            # Set up calculation.
            xtb = XTBCalculator(cluster, "xtb.xyz", config.xtb_options)
            xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
            xtb.options["el_temp"] = 300
            xtb.fixed_atoms.update([i + 1 for i in solute_indices])
            xtb.options["force_constant"] = 0.0005

            # Constrain the atoms of the solvent.
            for indices in solvent_indices:
                pair_indices = list(itertools.product(solute_indices, indices))
                pair_indices = tuple(zip(*pair_indices))
                distance_matrix = cluster.get_all_distances()
                i = np.argmin(distance_matrix[pair_indices])
                atom_1 = pair_indices[0][i] + 1
                atom_2 = pair_indices[1][i] + 1
                xtb.add_constraint(atom_1, atom_2, 2.5)
            
            # Perform optimization
            xtb.opt().wait()
            opt_cluster = ase.io.read("xtbopt.xyz")
            opt_cluster.info["charge"] = cluster.info["charge"]

            # Add constraints in case of hydroxide in water
            constraints = []
            if hydroxide_in_water:
                analysis = Analysis(opt_cluster, skin=0.1)
                for i, entry in enumerate(analysis.unique_bonds[0]):
                    for j in entry:
                        distance = opt_cluster.get_distance(i, j)
                        constraints.append(Constraint(i + 1, j + 1, distance))   

            # Do quick CREST sampling to get good initial conformation
            os.mkdir("crest")
            with cd("crest"):
                crest_options = {"nci": True, "solvent": config.xtb_options["solvent_cluster"], "squick": True, "mrest": 1}
                constrained_atoms = [i + 1 for i in solute_indices]
                opt_cluster = search_conformers(opt_cluster, constrained_atoms=constrained_atoms, constraints=constraints, dft_ranking=False, crest_options=crest_options)
        
        self.molecule = molecule
        self.solvent = solvent
        self.n_solvent = n_solvent
        self.cluster = opt_cluster
        self.hydroxide_in_water = hydroxide_in_water
    
    @staticmethod
    def _get_equidistant(radius, n_points):
        """Put points maximally distant on sphere"""
        # Point
        if n_points == 1:
            points = np.array([[1, 0, 0]])
        
        # Line
        if n_points == 2:
            p_1 = np.array([1, 0, 0])
            p_2 = np.array([-1, 0, 0])
            points = np.vstack([p_1, p_2])
        
        # Equilaterial triangle
        if n_points == 3:
            side = 1 * math.sqrt(3)
            y = np.cos(np.pi / 6) * side
            z = np.sin(np.pi / 6) * side
            p_1 = np.array([0, 1, 0])
            p_2 = np.array([0, 1 - y, z])
            p_3 = np.array([0, 1 - y, -z])
            points = np.vstack([p_1, p_2, p_3])
        
        # Tetrahedron
        if n_points == 4:
            p_1 = np.array([math.sqrt(8 / 9), 0, -1 / 3])
            p_2 = np.array([-math.sqrt(2 / 9), math.sqrt(2 / 3), -1 / 3])
            p_3 = np.array([-math.sqrt(2 / 9), -math.sqrt(2 / 3), -1 / 3])
            p_4 = np.array([0, 0, 1])
            points = np.vstack([p_1, p_2, p_3, p_4])
        
        # Triogonal bipyramid
        if n_points == 5:
            side = 1 * math.sqrt(3)
            y = np.cos(np.pi / 6) * side
            z = np.sin(np.pi / 6) * side
            p_1 = np.array([0, 1, 0])
            p_2 = np.array([0, 1 - y, z])
            p_3 = np.array([0, 1 - y, -z])
            p_4 = np.array([1, 0, 0])
            p_5 = np.array([-1, 0, 0])
            points = np.vstack([p_1, p_2, p_3, p_4, p_5])        
        
        # Octahedron
        if n_points == 6:
            p_1 = np.array([1, 0, 0])
            p_2 = np.array([-1, 0, 0])
            p_3 = np.array([0, 1, 0])
            p_4 = np.array([0, -1, 0])
            p_5 = np.array([0, 0, 1])
            p_6 = np.array([0, 0, -1])
            points = np.vstack([p_1, p_2, p_3, p_4, p_5, p_6])
    
        points = points * radius
    
        return points              