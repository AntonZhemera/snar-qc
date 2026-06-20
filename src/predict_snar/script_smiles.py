from predict_snar.smiles import SmilesToXYZ
from configparser import ConfigParser
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--substrate", help="Smiles of substrate", type=str)
    parser.add_argument("-n", "--nucleophile", help="Smiles of nucleophile", type=str)
    args = parser.parse_args()

    smiles_substrate = args.substrate
    smiles_nu = args.nucleophile

    smiles_to_xyz = SmilesToXYZ(smiles_substrate, smiles_nu)

    smiles_to_xyz.write_config()
    config = ConfigParser()
    config.read("config")
    config.read("config_smiles")
    with open("config", "w") as file:
        config.write(file)
    os.remove("config_smiles")

    smiles_to_xyz.write_xyz()

if __name__ == "__main__":
    main()
