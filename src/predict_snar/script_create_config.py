import argparse
from configparser import ConfigParser, NoOptionError, NoSectionError
from pathlib import Path

def main():
    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dft_solvent", help="Solvent for DFT", type=str, default="")
    parser.add_argument("-x", "--xtb_solvent", help="Solvent for XTB", type=str, default="")
    parser.add_argument("-t", "--temperature", help="Temperature", type=float, default=298.15)
    parser.add_argument("-e", "--electronic_temperature", help="Electronic temperature for xtb calculation", type=str, default="")
    parser.add_argument("-b", "--database", help="Database location", type=str, default="")
    parser.add_argument("-c", "--config_file", help="Name of configuration file.", type=str, default="config", required=False)
    parser.add_argument("--create-default", help="Write default config file to ~/.ps_config", action="store_true")
    args = parser.parse_args()

    config_file = args.config_file
    create_default = args.create_default

    # Create a configuration file
    config = ConfigParser()

    config["GENERAL"] = {"find_intermediate"        :   "True",
                         "opt_reactants"            :   "True",
                         "opt_products"             :   "True",
                         "cluster_nucleophile"      :   "True",
                         "cluster_leaving_group"    :   "True",
                         "cluster_ts"               :   "True",
                         "cluster_agent"            :   "True",
                         "ts_max_step"              :   "10",
                         "temperature"              :   args.temperature,
                         }
    config["XTB"] = {"electronic_temperature"   :   args.electronic_temperature,
                     "solvent"                  :   args.xtb_solvent,
                     "gfn_version"              :   "2",
                     }
    config["CREST"] = {"energy_window": 3.0,
                       "speed": "normal"}
    config["DFT"] = {"functional"           :   "wb97xd",
                     "basis_set"            :   "6-31+G(d)",
                     "ecp"                  :   "def2-svpd",
                     "dispersion_model"     :   "",
                     "solvent"              :   args.dft_solvent,
                     "solvation_model"      :   "smd",
                     "sp_functional"        :   "",
                     "sp_basis_set"         :   "6-311+G(d,p)",
                     "sp_ecp"               :   "def2-tzvpd",
                     "sp_dispersion_model"  :   "",
                     "sp_solvation_model"   :   "",
                     "nosymm"               :   "True"
                     }
    config["DESCRIPTORS"] = {"calculate_descriptors"    :   "True",
                             }                             
    config["DIRECTORIES"] = {"xtb": "",
                             "crest": "",
                             "chargemol": "",
                             "hs95": "",
                             "interface_script": "",
                             "crest_scratch" : "",
                             "gaussian_scratch": "",
                             "database": args.database,
                             "atomic_densities": ""}

    # Read default config file if there is one and use those options in case
    # they are not provided in the config file.
    default_config = Path.home() / ".ps_config"
    if default_config.is_file():
        new_config = ConfigParser()
        new_config.read(default_config)
        for section in new_config.sections():
            keys = new_config[section].keys()
            for key in keys:
                new_value = new_config.get(section, key)
                try:
                    current_value = config.get(section, key)
                    if current_value == "":
                        config.set(section, key, new_value)
                except NoSectionError:
                    config.add_section(section)
                    config.set(section, key, new_value)
                except NoOptionError:
                    config.set(section, key, new_value)        

    # Create default file
    if create_default:
        with open(default_config, "w") as file:
            config.write(file)

    with open(config_file, "w") as file:
        config.write(file)

if __name__ == "__main__":
    main()
