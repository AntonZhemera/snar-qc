import ase.io
from predict_snar.parsers import ConfigParser
from predict_snar import config
from predict_snar.smiles import SmilesToXYZ
from predict_snar.helpers import set_config, set_resources, cd
from predict_snar.jobs import optimize_ts, optimize_gsm, optimize_intermediate, optimize_react_complex, optimize_prod_complex, optimize_reactant_product
from ase.units import eV, Hartree, mol, kcal
import os
import logging
import argparse
import signal
from pathlib import Path
import fcntl
import uuid
from configparser import ConfigParser as ConfigParserOriginal
import traceback
import sys
import shelve
from datetime import datetime

def handler(signum=None, frame=None):
    """Function to handle abnormal program termination"""
    logging.critical("PROGRAM EXITED ABNORMALLY.")
    traceback.print_exc()
    sys.exit()

def main():
    start_time = datetime.now()
    # Define handler if the program crashes
    signal.signal(signal.SIGTERM, handler)

    # Capture arguments
    parser = argparse.ArgumentParser()
# TODO    parser.add_argument("--xyz", help="Input xyz file", type=str)
    parser.add_argument("input", help="Smiles of reaction", type=str)
    parser.add_argument("-m", "--mem", help="Amount of memory in GB", type=int)
    parser.add_argument("-p", "--n_procs", help="Number of processors", type=int)
    args = parser.parse_args()

    # Define logging level
    logging.basicConfig(filename='predict_snar.log',
                        level=logging.INFO,
                        format='%(asctime)s (%(module)s)     %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )

    # Create input files from Smiles
    smiles = args.input

    logging.info("Processing reaction smiles...")
    smiles_to_xyz = SmilesToXYZ(smiles)

    smiles_to_xyz.write_config()
    conf = ConfigParserOriginal()
    conf.read("config")
    conf.read("config_smiles")
    with open("config", "w") as file:
        conf.write(file)
    os.remove("config_smiles")

    smiles_to_xyz.write_xyz()
    logging.info("...processing completed.")

    # Load the program options from the config file
    options = ConfigParser("config")
    set_config(options)

    # Set the number of processors and amount of memory
    set_resources(args.n_procs, args.mem)

    # Make dir for optimzed files
    os.mkdir("optimized_structures")

    # Optimize the reactant and products and choose the electronic temperature
    substrate = ase.io.read("xyz_from_smiles/substrate.xyz")
    substrate.info["charge"] = config.charges["substrate"]
    nucleophile = ase.io.read("xyz_from_smiles/nucleophile.xyz")
    nucleophile.info["charge"] = config.charges["nu"]
    product = ase.io.read("xyz_from_smiles/product.xyz")
    product.info["charge"] = config.charges["product"]
    leaving_group = ase.io.read("xyz_from_smiles/leaving_group.xyz")
    leaving_group.info["charge"] = config.charges["lg"]
    os.mkdir("xtb_structures")
    with cd("xtb_structures"):
        opt_substrate, opt_nucleophile, opt_product, opt_leaving_group, el_temp = optimize_reactant_product(substrate, nucleophile, product, leaving_group)

    # Set the electronic temperature if there is none
    if not config.xtb_options.get("el_temp"):
        logging.info(f"Setting electronic temperature to {el_temp} based on HOMO-LUMO gap between substrate and nucleophile.")
        config.xtb_options["el_temp"] = el_temp

    # Optimize reactant complex
    react_complex = ase.io.read("xyz_from_smiles/reaction_complex.xyz")
    react_complex.info["charge"] = config.charges["substrate"] + config.charges["nu"]
    with cd("xtb_structures"):
        opt_react_complex = optimize_react_complex(react_complex)

    # Optimize the intermediate with DFT
    find_intermediate = options.general_options["find_intermediate"]

    if find_intermediate:
        opt_intermediate, frozen_intermediate = optimize_intermediate(opt_react_complex)
        if opt_intermediate:
            concerted = False
            ase.io.write("optimized_structures/intermediate.xyz", opt_intermediate, plain=True)
            start_struct = opt_intermediate
        else:
            concerted = True
            start_struct = frozen_intermediate

    # Optimize product complexes
    with cd("xtb_structures"):
        if opt_intermediate:
            opt_prod_complex = optimize_prod_complex(opt_intermediate)
        else:
            opt_prod_complex = optimize_prod_complex(frozen_intermediate)

    calc_list = []
    if not concerted:
        if config.general_info["azide_nucleophile"]:
            calc_list.append((opt_react_complex, frozen_intermediate, "step_1"))
        else:
            calc_list.append((opt_react_complex, opt_intermediate, "step_1"))
        calc_list.append((opt_intermediate, opt_prod_complex, "step_2"))
        logging.info("Reaction stepwise. Do one GSM for each step.")
    else:
        calc_list.append((opt_react_complex, opt_prod_complex, "concerted"))
        logging.info("Reaction concerted. Do only one GSM.")

    #ase.io.write("optimized_structures/reactant_complex.xyz", opt_react_complex, plain=True)
    #ase.io.write("optimized_structures/product_complex.xyz", opt_prod_complex, plain=True)

    for react, prod, name in calc_list:
        os.mkdir(name)
        with cd(name):
            logging.info(f"Starting GSM for {name}.")
            if not concerted:
                if name == "step_1":
                    gsm = optimize_gsm(react, prod, constrain="lg")
                elif name == "step_2":
                    gsm = optimize_gsm(react, prod, constrain="nu")
            else:
                options.gsm_options["n_points"] = options.gsm_options["n_points_concerted"]
                gsm = optimize_gsm(react, prod)
            logging.info(f"Finished GSM for {name}.")
            logging.info(f"Found {len(gsm.peaks)} peaks for refining.")
            gsm.make_plot()
            ts_list = []
            if config.gsm_options["refine"]:
                for counter, peak in enumerate(gsm.peaks, start=1):
                    os.mkdir(f"refine_{counter}")
                    with cd(f"refine_{counter}"):
                        start = gsm.geometries[peak.start]
                        stop = gsm.geometries[peak.stop]
                        factor = config.gsm_options["refine_factor"]
                        n_refine = gsm.get_refine_points(peak_index=counter - 1, factor=factor)
                        logging.info(f"Refining peak {counter}")
                        if name == "step_1":
                            refined_gsm = optimize_gsm(start, stop, refine=True, n_refine=n_refine, constrain="lg")
                        elif name == "step_2":
                            refined_gsm = optimize_gsm(start, stop, refine=True, n_refine=n_refine, constrain="nu")
                        else:
                            refined_gsm = optimize_gsm(start, stop, refine=True, n_refine=n_refine)
                        logging.info(f"Finished refining peak {counter}")
                        refined_gsm.make_plot()
                        for peak in refined_gsm.peaks:
                            peak_geometry = refined_gsm.geometries[peak.maximum]
                            ts_list.append(peak_geometry)
            else:
                for peak in gsm.peaks:
                    peak_geometry = gsm.geometries[peak.maximum]
                    ts_list.append(peak_geometry)
            opt_ts_list = []
            logging.info(f"Found {len(ts_list)} TSs to optimize.")
            for counter, ts in enumerate(ts_list, start=1):
                os.mkdir(f"ts_{counter}")
                with cd(f"ts_{counter}"):
                    logging.info(f"Optimizing TS {counter}")
                    opt_ts = optimize_ts(ts)
                    logging.info(f"Finished optimizing TS {counter}")
                    if opt_ts:
                        opt_ts_list.append(opt_ts)
        logging.info(f"Optimized {len(opt_ts_list)} TSs for {name}")
        if len(opt_ts_list) < 1:
            logging.info("No TS found. Reaction could be barrierless.")
        for counter, ts in enumerate(opt_ts_list, start=1):
            ase.io.write(f"optimized_structures/{name}_ts_{counter}.xyz", ts.atoms, plain=True)

    end_time = datetime.now()

    time_delta = end_time - start_time
    run_time_seconds = time_delta.total_seconds()
    run_time_str = str(time_delta)
    # Write the data to the database if requested
    if config.database_options["database_location"]:
        # Create entry
        entry = {}
        entry["smiles"] = {"substrate": smiles_to_xyz.smiles["substrate"],
                           "nucleophile": smiles_to_xyz.smiles["nu"],
                           "leaving_group": smiles_to_xyz.smiles["lg"],
                           "product": smiles_to_xyz.smiles["product"],
                           "reaction": smiles_to_xyz.smiles["reaction"],
                           }

        energy_list = [ts.free_energy for ts in opt_ts_list]
        min_energy = min(energy_list)
        relative_energies = [(energy - min_energy) * Hartree / (kcal / mol) for energy in energy_list]
        entry["energies"] = {"ts": relative_energies}

        geometry_dict = {"substrate": opt_substrate.get_positions().tolist(),
                         "nucleophile": opt_nucleophile.get_positions().tolist(),
                         "product": opt_product.get_positions().tolist(),
                         "leaving_group": opt_leaving_group.get_positions().tolist(),
                         }
        if opt_intermediate:
            geometry_dict["intermediate"] = opt_intermediate.get_positions().tolist()
        if any(opt_ts_list):
            geometry_dict["ts"] = [ts.atoms.get_positions().tolist() for ts in opt_ts_list]
        entry["geometries"] = geometry_dict

        entry["symbols"] = {"substrate": substrate.get_chemical_symbols(),
                            "nucleophile": nucleophile.get_chemical_symbols(),
                            "product": product.get_chemical_symbols(),
                            "leaving_group": leaving_group.get_chemical_symbols(),
                            "complex": react_complex.get_chemical_symbols(),
                            }
        entry["reactive_atoms"] = {"central": config.reactive_atoms["central_atom"],
                                   "nucleophile": config.reactive_atoms["nu_atom"],
                                   "leaving_group": config.reactive_atoms["lg_atom"],
                                   }
        entry["concerted"] = concerted
        entry["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry["run_time"] = run_time_seconds

        # Enter into databse if possible
        db_path = config.database_options["database_location"]
        db_path = Path(db_path)
        if not db_path.is_dir():
            db_path.mkdir()
        db_file = db_path / "db"
        db_file_full = db_file.as_posix()
        lock_file = db_file.with_suffix(".lock")
        logging.info(f"Writing to database {args.database}...")
        with open(lock_file, "w") as file:
            fcntl.lockf(file, fcntl.LOCK_EX)
            db = shelve.open(db_file_full)
            id = str(uuid.uuid4())
            db[id] = entry
            db.close()
            fcntl.lockf(file, fcntl.LOCK_UN)
        logging.info("...writing completed.")

    # Terminate the program

    logging.info(f"Program terminated succesfully. Total runtime was {run_time_str}")

if __name__ == "__main__":
    try:
        main()
    except:
        handler()
