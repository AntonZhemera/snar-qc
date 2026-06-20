from skimage.graph import route_through_array
from skimage.filters.rank import gradient
from skimage.morphology import local_minima

class GSM:
    """GSM object for handling the input, running and results of the GSM program

    Args:
        reactant (object)       :  Atoms object for the reactants
        product (object)        :  Atoms object for the products
        general_options (dict)  : General options
        gsm_options (dict)      :  Options for the GSM program
        xtb_options (dict)      :  Options for the xtb program
        dft_options (dict)      :  Options for G16

    Attributes:
        reactant (object)   :   Atoms object for the reactants
        product (object)    :   Atoms object for the products
        geometries (list)   :   List with atoms objects of the GSM points
        xtb_energies (list) :   List of xtb energies
        dft_energies (list) :   List of DFT energies
        peaks (list)        :   List of peaks
        options (dict)      :   List of options
        nbo_data (list)     :   List of NBOParser objects with NBO data
    """
    def __init__(self, reactant, product, general_options, gsm_options, xtb_options, dft_options):
        self.reactant = reactant
        self.product = product
        self.central_atom = general_options["central_atom"]
        self.nu_atom = general_options["nu_atom"]
        self.lg_atom = general_options["lg_atom"]
        self.geometries = []
        self.xtb_energies = []
        self.dft_energies = []
        self.nbo_data = []
        self.peaks = []
        self.options = {"max_iters"     :   50,
                        "n_points"      :   15,
                        "conv_tol"      :   0.00001,
                        }
        self.set_options(gsm_options)
        self.xtb = XTBCalculator(reactant, options=xtb_options)
        self.xtb.options["grad"] = True
        self.g16 = G16Calculator(reactant, options=dft_options)
        self.g16.options["int_acc"] = "fine"
        self.g16.options["scf_acc"] = "sleazy"
        self.g16.options["nbo"] = True
        self.constraints = set()
        self.dihedral_constraint = None
        self.angle_constraint = None

    def set_options(self, options):
        """Sets the options dictionary from options file"""
        for key, value in options.items():
            self.options[key] = value

    def check_electronic_temperature(self):
        hl_gaps = []
        for i in range(1, len(self.geometries)):
            outfile = f"scratch/orcain0000.{i:02d}.in.xtbout"
            parser = XTBParser(outfile)
            hl_gap = parser.lumo - parser.homo
            hl_gaps.append(hl_gap)
        return min(hl_gaps)

    def constrain_nu(self):
        distance = self.reactant.get_distance(self.central_atom - 1, self.nu_atom - 1)
        constraint = Constraint(self.central_atom, self.nu_atom, distance)
        self.constraints.add(constraint)

    def constrain_lg(self):
        distance = self.product.get_distance(self.central_atom - 1, self.lg_atom - 1)
        constraint = Constraint(self.central_atom, self.lg_atom, distance)
        self.constraints.add(constraint)

    def constrain_dihedral(self, atom_list, angle):
        self.dihedral_constraint = DihedralConstraint(*atom_list, angle)

    def constrain_angle(self, atom_list, angle):
        self.angle_constraint = AngleConstraint(*atom_list, angle)

    def constrain_bond(self, atom_list, value):
        constraint = Constraint(*atom_list, value)
        self.constraints.add(constraint)

    def get_refine_points(self, peak_index, factor=15):
        """Returns the number of refinement points for a GSM
        Args:
            factor (int)        :   Factor to control n_points
            peak_index (int)    :   Index of GSM peak to refine
        """
        peak = self.peaks[peak_index]
        bo_nu_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.nu_atom)
        bo_nu_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.nu_atom)
        bo_nu_diff = abs(bo_nu_left - bo_nu_right)

        bo_lg_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.lg_atom)
        bo_lg_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.lg_atom)
        bo_lg_diff = abs(bo_lg_left - bo_lg_right)

        n_points = round(max(bo_nu_diff, bo_lg_diff) * factor)

        return n_points

    def make_plot(self):
        """Makes a plot of the peaks and their start and stop points for refinement"""
        x = range(1, len(self.dft_energies) + 1)
        plt.plot(x, self.dft_energies, '-o')

        for peak in self.peaks:
            plt.plot(peak.maximum + 1, peak.energy, 'o', color='red', markersize=40, alpha=0.5)
            plt.axvline(peak.start + 1, color="red", alpha=0.5)
            plt.axvline(peak.stop + 1, color="red", alpha=0.5)
        plt.savefig("GSM.png")
        plt.clf()

    def find_peaks(self, prominence=0.5):
        """Finds the peaks and adds them to self.peaks"""
        if self.dft_energies:
            energies = self.dft_energies
        else:
            energies = self.xtb_energies
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
                left_right = minima_right[0]

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

    def validate_peaks(self):
        """Checks if bond order changes by more than 0.5 for TS. If not, peak is considered non-reactive and is ignored

        Args:
            central_atom (int)  :   Number of the central atom
            nu_atom (int)       :   Number of the nucleophile atom
            lg_atom (int)       :   Number of the leaving group atom
        """
        remove_list = []
        for peak in self.peaks:
            bo_nu_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.nu_atom)
            bo_nu_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.nu_atom)
            bo_nu_diff = abs(bo_nu_left - bo_nu_right)

            bo_lg_left = self.nbo_data[peak.left_base].get_bo(self.central_atom, self.lg_atom)
            bo_lg_right = self.nbo_data[peak.right_base].get_bo(self.central_atom, self.lg_atom)
            bo_lg_diff = abs(bo_lg_left - bo_lg_right)

            logging.info(peak)
            logging.info(bo_nu_diff)
            logging.info(bo_lg_diff)
            logging.info(self.peaks)
            if (bo_nu_diff < 0.5) and (bo_lg_diff < 0.5):
                remove_list.append(peak)

        for peak in remove_list:
            self.peaks.remove(peak)
            logging.info("peak removed")
            logging.info(self.peaks)

    def read_gsm_output(self):
        """Read the GSM output and store the geometries and energies"""
        if os.path.isfile("stringfile.xyz0000"):
            shutil.move("stringfile.xyz0000", "stringfile.xyz")
        self.geometries = [geometry for geometry in ase.io.iread("stringfile.xyz")]
        for geometry in self.geometries:
            geometry.info["charge"] = self.reactant.info["charge"]
        self.xtb_energies = [float(list(geometry.info.keys())[0]) for geometry in self.geometries]

    def read_sp_output(self):
        """Read DFT single point output and store the energies"""
        dft_energies = []
        nbo_data = []
        for i in range(1, len(self.geometries) + 1):
            nbo = NBOParser(f"sps/{i}.log")
            nbo_data.append(nbo)

            data = cclib.io.ccread(f"sps/{i}.log")
            energy = data.scfenergies[-1] * eV / (kcal / mol)
            dft_energies.append(energy)

        normalized_energies = [energy - dft_energies[0] for energy in dft_energies]
        self.dft_energies = normalized_energies
        self.nbo_data = nbo_data

    def run_sps(self, n_procs, mem):
        """Run DFT single point calculations"""
        os.mkdir("sps")
        dft_options = self.g16.options
        g16_list = [G16Calculator(atoms, file=f"sps/{counter + 1}", options=dft_options) for counter, atoms in enumerate(self.geometries)]
        n_calcs = len(g16_list)
        n_procs = max(n_procs // n_calcs, 1)
        mem = mem / n_calcs

        process_list = []
        for g16 in g16_list:
            process = g16.single_point(n_procs=n_procs, mem=mem)
            process_list.append(process)
        while True:
            time.sleep(5)
            poll_list = [process.poll() is not None for process in process_list]
            if all(poll_list):
                break

    def run_gsm(self, n_procs=1):
        """Run GSM calculation

        Returns:
            process (obj)   :   Popen process object of the gsm calculation.
        """
        self.write_initial()
        self.write_inpfile()
        self.write_ograd(n_procs)

        submit_string = f"gsm.orca > gsm.out"
        process = subprocess.Popen(submit_string, shell=True)

        return process

    def write_initial(self):
        """Write the initial xyz file with reactant and product geometries"""
        if not os.path.isdir("scratch"):
            os.mkdir("scratch")
        self.reactant.write("scratch/react.xyz", format="xyz", plain=True)
        react_content = open("scratch/react.xyz").read()
        self.product.write("scratch/prod.xyz", format="xyz", plain=True)
        prod_content = open("scratch/prod.xyz").read()
        initial_content = react_content + prod_content
        with open("scratch/initial0000.xyz", 'w') as file:
            file.write(initial_content)

    def write_ograd(self, n_procs=1):
        """Write the ograd file for use with the GSM program"""
        xtb_string = self.xtb.get_submit_string(n_procs)
        write_string = dedent(
            f"""\
            #!/bin/bash
            #Change the options for xtb here
            opts='{xtb_string}'

            ofile=orcain$1.in
            molfile=structure$1
            basename=${{ofile%.*}}

            cd scratch

            ########## XTB/TM settings: #################
            wc -l < $molfile > $ofile.xyz
            echo 'Dummy for XTB/TM calculation' >> $ofile.xyz
            cat $molfile | awk '{{printf \"%-10s%10.5f%10.5f%10.5f\\n\", $1, $2, $3, $4}}' >> $ofile.xyz
            """)
        if self.constraints or self.dihedral_constraint or self.angle_constraint:
            if (not self.dihedral_constraint) and (not self.angle_constraint):
                fc = 0.5
            else:
                fc = 0.05
            write_string += dedent(
                f"""\
                cat << FILE >> $ofile.xyz
                \$constrain
                force constant = {fc}
                """)
        if self.constraints:
            for constraint in self.constraints:
                write_string += dedent(
                    f"""\
                    distance: {constraint.atom_1}, {constraint.atom_2}, {constraint.value}
                    """)
        if self.angle_constraint:
            write_string += dedent(
                f"""\
                angle: {self.angle_constraint.atom_1}, {self.angle_constraint.atom_2}, {self.angle_constraint.atom_3}, {self.angle_constraint.value}
                """)
        if self.dihedral_constraint:
            write_string += dedent(
                f"""\
                dihedral: {self.dihedral_constraint.atom_1}, {self.dihedral_constraint.atom_2}, {self.dihedral_constraint.atom_3}, {self.dihedral_constraint.atom_4}, {self.dihedral_constraint.value}
                """)
        if self.constraints or self.dihedral_constraint or self.angle_constraint:
            write_string += dedent(
                """\
                FILE
                """)
        write_string += dedent(
            """\
            xtb $ofile.xyz $opts > $ofile.xtbout 2> xtb.errors
            awk '/Cart. coordinates/ {{flag=1;next}} /Z AO/{{flag=0}} flag {{print $2, $6, $7, $8}}' $ofile.xtbout | head -n -1 > coordfile
            tm2orca.py $basename
            cd ..
            """)
        with open("ograd", 'w') as file:
            file.write(write_string)

        # Make file executable
        st = os.stat("ograd")
        os.chmod("ograd", st.st_mode | stat.S_IEXEC)

    def write_inpfile(self):
        """Write the input file for the GSM program"""
        write_string = f"""\
                        # FSM/GSM/SSM inpfileq

                        ------------- QCHEM Scratch Info ------------------------
                        $QCSCRATCH/    # path for scratch dir. end with "/"
                        GSM_go1q       # name of run
                        ---------------------------------------------------------

                        ------------ String Info --------------------------------
                        SM_TYPE                 GSM      # SSM, FSM or GSM
                        RESTART                 0        # read restart.xyz
                        MAX_OPT_ITERS           {self.options["max_iters"]}      # maximum iterations
                        STEP_OPT_ITERS          30       # for FSM/SSM
                        CONV_TOL                {self.options["conv_tol"]}  # perp grad
                        ADD_NODE_TOL            0.3      # for GSM
                        SCALING		           1.0      # for opt steps
                        SSM_DQMAX               0.8      # add step size
                        GROWTH_DIRECTION        0        # normal/react/prod: 0/1/2
                        INT_THRESH              2.0      # intermediate detection
                        MIN_SPACING             5.0      # node spacing SSM
                        BOND_FRAGMENTS          1        # make IC's for fragments
                        INITIAL_OPT             0        # opt steps first node
                        FINAL_OPT               150      # opt steps last SSM node
                        PRODUCT_LIMIT           100.0    # kcal/mol
                        TS_FINAL_TYPE           1        # any/delta bond: 0/1
                        NNODES		            {self.options["n_points"]}       # including endpoints
                        ---------------------------------------------------------
                        """
        with open("inpfileq", 'w') as file:
            file.write(dedent(write_string))

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.reactant.symbols}')"

def optimize_gsm(react_atoms, prod_atoms, refine=False, n_refine=11, constrain=None):
    # Initialize the gsm
    gsm = GSM(react_atoms, prod_atoms, general_options=config.general_options, gsm_options=config.gsm_options, xtb_options=config.xtb_options, dft_options=config.dft_options)
    if refine:
        gsm.set_options({"n_points": n_refine})
    if constrain == "nu":
        gsm.constrain_nu()
    elif constrain == "lg":
        gsm.constrain_lg()

    if config.general_info["azide_nucleophile"]:
        azide_atom_list = config.general_info["azide_dihedral"]
        dihedral_value = 180
        gsm.constrain_dihedral(azide_atom_list, dihedral_value)

        angle_atom_list = config.general_info["azide_angle"]
        angle_value = 180
        gsm.constrain_angle(angle_atom_list, angle_value)

        distance_1 = react_atoms.get_distance(angle_atom_list[0] - 1, angle_atom_list[1] - 1)
        distance_2 = react_atoms.get_distance(angle_atom_list[1] - 1, angle_atom_list[2] - 1)
        gsm.constrain_bond([angle_atom_list[0], angle_atom_list[1]], distance_1)
        gsm.constrain_bond([angle_atom_list[1], angle_atom_list[2]], distance_2)

    # Run the GSM
    logging.info("Optimizing GSM...")
    gsm.run_gsm(n_procs=config.n_procs).wait()
    logging.info("...optimization completed")
    gsm.read_gsm_output()

    hl_gap = gsm.check_electronic_temperature()
    if hl_gap < 2.0 and config.xtb_options["el_temp"] > 4000:
        logging.info("HOMO-LUMO gap below 2.0 eV for GSM. Switch to electronic temperature of 4000. Optimizing again...")
        config.xtb_options["el_temp"] = 4000
        gsm.xtb.options["el_temp"] = config.xtb_options["el_temp"]
        gsm.run_gsm(n_procs=config.n_procs).wait()
        logging.info("...optimization completed")
        gsm.read_gsm_output()
    logging.info("Running DFT SPs...")
    gsm.run_sps(n_procs=config.n_procs, mem=config.mem)
    logging.info("...DFT SPs completed")
    gsm.read_sp_output()

    # Find and validate the peaks
    gsm.find_peaks()
    gsm.validate_peaks()

    #  If refining, take only the top peak
    if refine:
        if gsm.peaks:
            top_peak = sorted(gsm.peaks, key=lambda x: x.energy, reverse=True)[0]
            gsm.peaks = [top_peak]

    logging.info(f"Found {len(gsm.peaks)} peaks.")

    return gsm

class Scan2D:
    def __init__(self, atoms, xtb_options, dft_options):
        self.atoms = atoms
        self.geometries = []
        self.xtb_energies = []
        self.dft_energies = []
        self.nbo_data = []
        self.xtb = XTBCalculator(atoms, options=xtb_options)
        self.g16 = G16Calculator(atoms, options=dft_options)
        self.g16.options["int_acc"] = "fine"
        self.g16.options["scf_acc"] = "sleazy"
        self.g16.options["nbo"] = True
        self.g16.options["chk"] = False
        self.xtb.options["force_constant"] = 10
        self.peaks = []

    def add_scan_distance(self, atom_1, atom_2, value, start, stop, steps):
        scan = BondScan(atom_1, atom_2, value, start, stop, steps)
        self.scan_distance = scan

    def add_loop_distance(self, atom_1, atom_2, value, start, stop, steps):
        scan = BondScan(atom_1, atom_2, value, start, stop, steps)
        self.loop_distance = scan

    def run_scan(self, n_procs=1):
        distances = np.linspace(self.loop_distance.start, self.loop_distance.stop, self.loop_distance.steps)
        geometries_2D = []
        xtb_energies_2D = []
        for distance in distances:
            path = str(round(distance, 2))
            os.mkdir(path)
            with cd(path):
                # Reset xtb calculator
                xtb = XTBCalculator(self.atoms, options=self.xtb.options)
                xtb.add_scan(self.scan_distance)
                xtb.add_constraint(self.loop_distance.atom_1, self.loop_distance.atom_2, distance)
                xtb.add_constraint(self.scan_distance.atom_1, self.scan_distance.atom_2, "auto")
                xtb.opt(n_procs=n_procs).wait()
                geometries, xtb_energies = self.read_scan_output()
                if self.scan_distance.start > self.scan_distance.stop:
                    geometries.reverse()
                    xtb_energies.reverse()
                geometries_2D.append(geometries)
                xtb_energies_2D.append(xtb_energies)
        if self.loop_distance.start > self.loop_distance.stop:
            geometries_2D.reverse()
            xtb_energies_2D.reverse()
        self.geometries_2D = geometries_2D
        self.xtb_energies_2D = xtb_energies_2D

        x = []
        y = []
        for row in self.geometries_2D:
            row_x = []
            row_y = []
            for atoms in row:
                x_dist = atoms.get_distance(self.loop_distance.atom_1 - 1, self.loop_distance.atom_2 - 1)
                y_dist = atoms.get_distance(self.scan_distance.atom_1 - 1, self.scan_distance.atom_2 - 1)
                row_x.append(x_dist)
                row_y.append(y_dist)
            x.append(row_x)
            y.append(row_y)
        self.x = x
        self.y = y

    def calculate_gradients(self, n_procs=1):
        geometries = self.geometries_2D
        grads = []
        os.mkdir("gradients")
        with cd("gradients"):
            for row in geometries:
                row_grads = []
                for atoms in row:
                    xtb = XTBCalculator(atoms, options=self.xtb.options)
                    xtb.options["grad"] = True
                    xtb.run_calc(n_procs=n_procs).wait()
                    results = XTBParser("xtb.out")
                    grad = results.grad
                    row_grads.append(grad)
                grads.append(row_grads)
        self.grads_2D = grads

    def read_scan_output(self):
        """Read the GSM output and store the geometries and energies"""
        if os.path.isfile("xtbscan.log"):
            shutil.move("xtbscan.log", "scan.xyz")
        geometries = [geometry for geometry in ase.io.iread("scan.xyz")]
        for geometry in geometries:
            geometry.info["charge"] = self.atoms.info["charge"]
        energies = [float(list(geometry.info.keys())[2]) for geometry in geometries]
        xtb_energies = energies

        return geometries, xtb_energies

    def find_path(self, n_points=16, plot=False, geometric=False):
        energies = np.array(self.xtb_energies_2D)
        grads = np.array(self.grads_2D)
        minima = local_minima(energies)
        indices = np.where(minima == 1)

        x = np.array(self.x)
        y = np.array(self.y)

        path, cost = route_through_array(grads, indices[0], indices[1], fully_connected=True, geometric=geometric)
        path = [(x[i, j], y[i, j]) for i, j in path]

        # Fit parametric spline to the points
        x_path = [x for x, y in path]
        y_path = [y for x, y in path]
        tck, u = interpolate.splprep([x_path, y_path], s=0)

        if plot:
            # Plot the whole parametric spline
            unew = np.arange(0, 1.01, 0.01)
            out = interpolate.splev(unew, tck)
            plt.contourf(x, y, energies, cmap="viridis", levels=50)
            plt.plot(out[0], out[1], c="r")
            for point in path:
                plt.scatter(point[0], point[1], c="r")
            plt.savefig("energies.png")
            plt.clf()
            plt.contourf(x, y, grads, cmap="viridis", levels=50)
            plt.plot(out[0], out[1], c="r")
            for point in path:
                plt.scatter(point[0], point[1], c="r")
            plt.savefig("gradients.png")
            plt.clf()

        unew = np.linspace(0, 1, 16)
        out = interpolate.splev(unew, tck)
        XY = np.column_stack(out)

        self.path_bond_lenghts = XY.tolist()

    def optimize_path(self, n_procs=1):
        path_geometries = []
        os.mkdir("path")
        with cd("path"):
            for i, entry in enumerate(self.path_bond_lenghts, start=1):
                os.mkdir(f"{i}")
                with cd(f"{i}"):
                    bond_length_loop = entry[0]
                    bond_length_scan = entry[1]
                    xtb = XTBCalculator(self.atoms, options=self.xtb.options)
                    xtb.add_constraint(self.loop_distance.atom_1, self.loop_distance.atom_2, bond_length_loop)
                    xtb.add_constraint(self.scan_distance.atom_1, self.scan_distance.atom_2, bond_length_scan)
                    xtb.opt(n_procs=n_procs).wait()
                    atoms = ase.io.read("xtbopt.xyz")
                    atoms.info["charge"] = self.atoms.info["charge"]
                    path_geometries.append(atoms)
        ase.io.write("scan.xyz", path_geometries, plain=True)
        self.path_geometries = path_geometries


    def read_sp_output(self):
        """Read DFT single point output and store the energies"""
        dft_energies = []
        nbo_data = []
        for i in range(1, len(self.path_geometries) + 1):
            nbo = NBOParser(f"sps/{i}.log")
            nbo_data.append(nbo)

            data = cclib.io.ccread(f"sps/{i}.log")
            energy = data.scfenergies[-1] * eV / (kcal / mol)
            dft_energies.append(energy)

        normalized_energies = [energy - dft_energies[0] for energy in dft_energies]
        self.path_dft_energies = normalized_energies
        self.path_nbo_data = nbo_data

    def run_sps(self, n_procs, mem):
        """Run DFT single point calculations"""
        os.mkdir("sps")
        dft_options = self.g16.options
        g16_list = [G16Calculator(atoms, file=f"sps/{counter + 1}.gjf", options=dft_options) for counter, atoms in enumerate(self.path_geometries)]
        n_calcs = len(g16_list)
        n_procs = max(n_procs // n_calcs, 1)
        mem = mem / n_calcs

        # Dummy function to run the jobs
        def job(calculator, n_procs, mem):
            calculator.single_point(n_procs=n_procs, mem=mem)
            calculation_monitor(calculator)

        process_list = []
        for g16 in g16_list:
            process = multiprocessing.Process(target=job, args=(g16, n_procs, mem))
            process_list.append(process)
            process.start()

        for process in process_list:
            process.join()

    def find_peaks(self, prominence=0.01):
        """Finds the peaks and adds them to self.peaks"""
        if self.path_dft_energies:
            energies = self.path_dft_energies
        else:
            energies = self.path_xtb_energies
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
                left_right = minima_right[0]

            # Make sure that there is at least one point distance between the peak and the refinement
            point_1 = left_base + int((maximum - left_base) / 2)
            if maximum - point_1 < 2:
                if maximum - 2 >= left_base:
                    point_1 = maximum - 2
            point_2 = right_base - int((right_base - maximum) / 2)
            if point_2 - maximum < 2:
                if maximum + 2 <= right_base:
                    point_2 = maximum + 2
            energy = self.path_dft_energies[maximum]

            # Add peaks to list
            peak = Peak(left_base, point_1, maximum, point_2, right_base, energy, prominence)
            peak_list.append(peak)

        peak_energies = [peak.energy for peak in peak_list]
        top_energy = max(peak_energies)
        top_peak = peak_list[peak_energies.index(top_energy)]
        self.peaks = peak_list

    def make_plot(self):
        """Makes a plot of the peaks and their start and stop points for refinement"""
        x = range(1, len(self.path_dft_energies) + 1)
        plt.plot(x, self.path_dft_energies, '-o', label="DFT")
        plt.legend()

        for peak in self.peaks:
            plt.plot(peak.maximum + 1, peak.energy, 'o', color='red', markersize=40, alpha=0.5)
        plt.savefig("GSM.png")
        plt.clf()

def cluster_molecule(molecule, solute_smiles, solvent_smiles, n_solvent=4, crest_dft_ranking=False, dft_sps=False, optimize=False):
    # Create solute molecule
    solute = Chem.MolFromSmiles(solute_smiles)
    solute = Chem.AddHs(solute)
    AllChem.EmbedMolecule(solute, randomSeed=1)

    # Create solvent molecule
    solvent = Chem.MolFromSmiles(solvent_smiles)
    solvent = Chem.AddHs(solvent)
    AllChem.EmbedMolecule(solvent, randomSeed=1)

    # Set up dictionary for energies
    energies = {"xtb": {}, "dft": {}}

    # Convert solvent to ASE atoms object
    elements = [atom.GetSymbol() for atom in solvent.GetAtoms()]
    coordinates = solvent.GetConformer().GetPositions()
    charge = Chem.GetFormalCharge(solvent)
    solvent_atoms = Atoms(elements, coordinates)
    solvent_atoms.info["charge"] = charge  

    # Do conformational search and optimize solvent
    logger.info("Optimizing solvent...")
    os.mkdir("solvent")
    with cd("solvent"):
        os.mkdir("crest")
        with cd("crest"):
            crest_options = {"nci": True, "solvent": config.xtb_options["solvent_cluster"]}
            solvent_atoms = search_conformers(solvent_atoms, dft_ranking=crest_dft_ranking, crest_options=crest_options)
        os.mkdir("xtb")
        with cd("xtb"):
            xtb = XTBCalculator(solvent_atoms, "xtb.xyz", config.xtb_options)
            xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
            xtb.options["temperature"] = config.general_options["temperature"]
            xtb.opt_freq().wait()
            data = XTBParser("xtb.out")
            opt_solvent = ase.io.read("xtbopt.xyz")
            opt_solvent.info["charge"] = solvent_atoms.info["charge"]
            free_energy_solvent_xtb = data.free_energy
            free_energy_corr_solvent_xtb = data.free_energy - data.energy
            ss_state = config.general_options["standard_state"]
            ss_correction = standard_state_correction(ss_state, reference="M", temperature=config.general_options["temperature"])
            free_energy_corr_solvent_xtb += ss_correction
            free_energy_solvent_xtb += ss_correction
            energies["xtb"]["solvent"] = free_energy_solvent_xtb
        logger.info("...optimization done.")

        if dft_sps:
            os.mkdir("sp")
            with cd("sp"):
                logger.info("Refining energy with DFT.")
                g16 = G16Calculator(opt_solvent, file="solvent.gjf", options=config.dft_options)
                g16.options["int_acc"] = "fine"
                g16.options["scf_acc"] = "sleazy"
                g16.options["chk"] = False
                g16.single_point(n_procs=config.n_procs, mem=config.mem)
                calculation_monitor(g16, errors=True)
                data = cclib.io.ccread(g16.output)
                electronic_energy_solvent_dft = data.scfenergies[-1] * EV_TO_HARTREE
                free_energy_solvent_dft = electronic_energy_solvent_dft + free_energy_corr_solvent_xtb
                energies["dft"]["solvent"] = free_energy_solvent_dft

    # Do conformational search and optimize solute
    logger.info("Optimizing solute...")
    os.mkdir("solute")                
    with cd("solute"):
        os.mkdir("crest")
        with cd("crest"):
            crest_options = {"nci": True, "solvent": config.xtb_options["solvent_cluster"]}
            molecule = search_conformers(molecule, dft_ranking=crest_dft_ranking, crest_options=crest_options)
        os.mkdir("xtb")
        with cd("xtb"):
            xtb = XTBCalculator(molecule, "xtb.xyz", config.xtb_options)
            xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
            xtb.options["temperature"] = config.general_options["temperature"]
            if molecule.get_number_of_atoms() == 1:
                xtb.single_point()
            else:
                xtb.opt_freq().wait()
            data = XTBParser("xtb.out")
            opt_solute = ase.io.read("xtbopt.xyz")
            opt_solute.info["charge"] = molecule.info["charge"]
            if molecule.get_number_of_atoms() == 1:
                atomic_number = molecule.get_atomic_numbers()[0]
                mass = atomic_masses[atomic_number]
                enthalpy, t_entropy = thermal_analysis_atom(mass, reference="M", temperature=config.general_options["temperature"])
                free_energy_corr_solute_xtb = enthalpy + t_entropy
                free_energy_solute_xtb = data.energy + free_energy_corr_solute_xtb                
            else:
                free_energy_solute_xtb = data.free_energy
                free_energy_corr_solute_xtb = data.free_energy - data.energy
            energies["xtb"]["solute"] = free_energy_solute_xtb
        logger.info("...optimization done.")          

        if dft_sps:
            os.mkdir("sp")
            with cd("sp"):
                logger.info("Refining energy with DFT.")
                g16 = G16Calculator(opt_solute, file="solvent.gjf", options=config.dft_options)
                g16.options["int_acc"] = "fine"
                g16.options["scf_acc"] = "sleazy"
                g16.options["chk"] = False
                g16.single_point(n_procs=config.n_procs, mem=config.mem)
                calculation_monitor(g16, errors=True)
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
            cluster_molecule = ClusterMolecule(solute, solvent, n, epsilon=1)
            cluster = cluster_molecule.cluster
        
            # Convert cluster to ASE atoms object
            elements = [atom.GetSymbol() for atom in cluster.GetAtoms()]
            coordinates = cluster.GetConformer().GetPositions()
            charge = Chem.GetFormalCharge(cluster)
            cluster_atoms = Atoms(elements, coordinates)
            cluster_atoms.info["charge"] = charge
        
            # Search for most stable conformer with CREST
            os.mkdir("crest")
            with cd("crest"):
                crest_options = {"nci": True, "solvent": config.xtb_options["solvent_cluster"]}
                cluster_atoms = search_conformers(cluster_atoms, dft_ranking=crest_dft_ranking, crest_options=crest_options)
            os.mkdir("xtb")
            with cd("xtb"):
                xtb = XTBCalculator(cluster_atoms, "xtb.xyz", config.xtb_options)
                xtb.options["solvent"] = config.xtb_options["solvent_cluster"]
                xtb.options["temperature"] = config.general_options["temperature"]
                xtb.opt_freq().wait()
                data = XTBParser("xtb.out")
                free_energy_cluster_xtb = data.free_energy
                free_energy_corr_cluster_xtb = data.free_energy - data.energy
                opt_cluster = ase.io.read("xtbopt.xyz")
                opt_cluster.info["charge"] = cluster_atoms.info["charge"]
                stabilization_energy_xtb = (free_energy_cluster_xtb - energies["xtb"]["solute"] - n * energies["xtb"]["solvent"]) * HARTREE_TO_KCAL 
            logger.info("...Optimization done.")                

            if dft_sps:
                os.mkdir("sp")
                with cd("sp"):
                    logger.info("Refining energy with DFT.")
                    g16 = G16Calculator(opt_cluster, file="solvent.gjf", options=config.dft_options)
                    g16.options["int_acc"] = "fine"
                    g16.options["scf_acc"] = "sleazy"
                    g16.options["chk"] = False
                    g16.single_point(n_procs=config.n_procs, mem=config.mem)
                    calculation_monitor(g16, errors=True)
                    data = cclib.io.ccread(g16.output)
                    electronic_energy_cluster_dft = data.scfenergies[-1] * EV_TO_HARTREE
                    free_energy_cluster_dft = electronic_energy_cluster_dft + free_energy_corr_cluster_xtb
                    stabilization_energy_dft = (free_energy_cluster_dft - energies["dft"]["solute"] - n * energies["dft"]["solvent"]) * HARTREE_TO_KCAL
            else:
                stabilization_energy_dft = None
            
            with open("../cluster_energies", "a") as file:
                file.write(f"{n:5}{stabilization_energy_xtb:10.2f}{stabilization_energy_dft:10.2f}\n")

            if dft_sps:
                stabilization_energy = stabilization_energy_dft
            else:
                stabilization_energy = stabilization_energy_xtb

            results[n] = {"energy": stabilization_energy, "cluster": opt_cluster}
            logger.info(f"Stabilization energy: {stabilization_energy}")
            
            previous_energy = results[n - 1]["energy"]
            if stabilization_energy > previous_energy:
                logger.info("Energy rises. Aborting iterative clustering.")
                best_n = n - 1
                break
            else:
                best_n = n
    best_energy = results[best_n]["energy"]
    best_cluster = results[best_n]["cluster"]
    ase.io.write(f"cluster.xyz", best_cluster, plain=True)
    
    logger.info(f"Stabilization energy for {best_n} solvent molecules: {best_energy}")

    return best_cluster, opt_solvent, best_n, best_energy

class ClusterMolecule:
    def __init__(self, molecule, solvent, n_solvent, epsilon=1):
        # Derive radius of solvent sphere
        r_s = []
        for mol in [molecule, solvent]:
            coordinates = mol.GetConformer().GetPositions()
            com = np.sum(coordinates, axis=0)
            dists = np.linalg.norm(coordinates - com, axis=1)
            r = np.max(dists)
            r_s.append(r)
            
        r_sphere = sum(r_s) + 4            
        
        # Construct solvents on the sphere
        solvent_points = self._get_equidistant(r_sphere, n_solvent)
                                         
        cluster = molecule
        molecule_indices = [atom.GetIdx() for atom in cluster.GetAtoms()]                                         
        solvent_indices = []
        for solvent_point in solvent_points:
            offset = Point3D(*solvent_point)
            indices = [atom.GetIdx() + cluster.GetNumAtoms() for atom in solvent.GetAtoms()] 
            cluster = Chem.CombineMols(cluster, solvent, offset=offset)
            solvent_indices.append(indices)
        Chem.SanitizeMol(cluster)            
        
        # Shrink solvent spheres with weak constraints. Set up force field
        mol_prop = AllChem.MMFFGetMoleculeProperties(cluster)
        mol_prop.SetMMFFDielectricConstant(epsilon)
        ff = AllChem.MMFFGetMoleculeForceField(cluster, mol_prop, ignoreInterfragInteractions=False)
        
        # Fix molecule positions
        for i in molecule_indices:
            ff.AddFixedPoint(i)
        
        # Add weak constraint to pull solvents to center of mass
        point_index = ff.AddExtraPoint(0.0, 0.0, 0.0) - 1
        for indices in solvent_indices:
            for i in indices:
                ff.MMFFAddDistanceConstraint(i, point_index, False, 0.0, 0.0 , 0.01)
        
        # Optimize 
        ff.Initialize()
        ff.Minimize()
        
        self.molecule = molecule
        self.solvent = solvent
        self.epsilon = epsilon
        self.n_solvent = n_solvent
        self.cluster = cluster
    
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
        
        # Triognal bipyramid
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

def do_clustering(cutoff=2, opt_dft=True, crest_dft_ranking=False):
    jobs = []
    candidates = ["leaving_group"]
    if not config.intramolecular:
        candidates.append("nucleophile")
    for name in candidates:
        if config.clustering[name] and config.general_options[f"cluster_{name}"]:
            jobs.append(name)
    if config.agent:
        if config.clustering["agent"] and config.general_info["agent"]:
            jobs.append("agent")
    
    if not any(jobs):
        return

    opt_solvent_dft = None

    os.mkdir("clustering")
    with cd("clustering"):
        clustering_energies = {}
        for name in jobs:
            os.mkdir(name)
            with cd(name):
                logger.info(f"Starting clustering calculation for {name}.")
                # Perform preliminary clustering
                molecule = results.mm_atoms[name]
                solvent_smiles = config.general_options["solvent_smiles"]
                solute_smiles = results.smiles[name]
                opt_cluster, opt_solvent, best_n, best_energy = cluster_molecule(molecule, solute_smiles, solvent_smiles, n_solvent=4, dft_sps=True, optimize=True, crest_dft_ranking=crest_dft_ranking)
    
                # Test for unstable cluster
                if best_n == 0:
                    logger.info("Cluster not better than continuum solvent. Aborting.")
                    config.clustering[name] = False
                    continue
        
                # Test predicted clustering energy against cutoff
                if abs(best_energy) < cutoff:
                    logger.info(f"Predicted cluster solvation energy of {best_energy:.2f} lower than cutoff of {cutoff:.2f}. Aborting clustering procedure")
                    config.clustering[name] = False
                    continue
                
                # Refine clustering energy with DFT
                if opt_dft:
                    logger.info("Optimizing best cluster with DFT.")
                    os.mkdir("dft")
                    with cd("dft"):
                        # Reperform conformational sampling with DFT ranking
                        os.mkdir("crest")
                        with cd("crest"):
                            logger.info("Performing conformational search...")
                            crest_options = {"nci": True, "solvent": config.xtb_options["solvent_cluster"]}
                            opt_cluster = search_conformers(opt_cluster, crest_options=crest_options)
                            logger.info("...conformational search done.")
    
                        # Do initial relaxation
                        g16 = G16Calculator(opt_cluster, "cluster_pre.gjf", config.dft_options)
                        g16.options["opt_cycles"] = 10
                        g16.options["opt_max_step"] = 30
                        logger.info("Optimizing with DFT...")
                        g16.opt(n_procs=config.n_procs, mem=config.mem)
                        calculation_monitor(g16)
                        data = cclib.io.ccread(g16.output)
    
                        # Decrease step size 
                        if not data.optdone:
                            continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
                            continue_atoms.info["charge"] = opt_cluster.info["charge"]
                            g16 = G16Calculator(continue_atoms, "cluster.gjf", config.dft_options)
                            g16.options["oldchk"] = "cluster_pre.chk"        
                            g16.options["read_wf"] = True
                            g16.options["opt_cycles"] = 30
                            g16.options["opt_max_step"] = 5
                            g16.opt_freq(n_procs=config.n_procs, mem=config.mem)
                            calculation_monitor(g16)
                            data = cclib.io.ccread(g16.output)
                        
                        # Perform frequency calculation anyway if not converged.
                        if not data.optdone:
                            continue_atoms = ase.Atoms(symbols=data.atomnos, positions=data.atomcoords[-1])
                            continue_atoms.info["charge"] = opt_cluster.info["charge"]
                            g16 = G16Calculator(continue_atoms, "cluster_freq.gjf", config.dft_options)
                            g16.options["oldchk"] = "cluster.chk"        
                            g16.options["read_wf"] = True
                            g16.freq(n_procs=config.n_procs, mem=config.mem)
                            calculation_monitor(g16)
                            data = cclib.io.ccread(g16.output)                        
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
        
                solute = results.dft_atoms[name]
                free_energy_solute = solute.info["free_energy"]
                free_energy_solute_qh_grimme = solute.info["free_energy_qh_grimme"]
                free_energy_solute_qh_truhlar = solute.info["free_energy_qh_truhlar"]
        
                clustering_energy = free_energy_cluster - free_energy_solute - best_n * free_energy_solvent
                clustering_energy_qh_grimme = free_energy_cluster_qh_grimme - free_energy_solute_qh_grimme - best_n * free_energy_solvent_qh_grimme
                clustering_energy_qh_truhlar = free_energy_cluster_qh_truhlar - free_energy_solute_qh_truhlar - best_n * free_energy_solvent_qh_truhlar
                
                best_energy_ha = best_energy * KCAL_TO_HARTREE
    
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
        results.clustering_energies = clustering_energies
