gsm_options = {}
xtb_options = {}
general_options = {}
general_info = {}
dft_options = {}
crest_options = {}
dft_sp_options = {}
charges = {}
clustering = {}
reactive_atoms = {}
database_options = {}
descriptor_options = {}
n_procs = None
mem = None
xtb_scratch = None
agent = None

flat_PES = False
intramolecular = False
concerted = False
nucleophilic_centers = []

#!TODO move this to the config file for user to change.
descriptor_dft_options = {"functional"           :   "b3lyp",
                          "basis_set"            :   "6-31+G(d)",
                          "dispersion_model"     :   "d3bj",
                          }