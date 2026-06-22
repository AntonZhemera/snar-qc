# Stage B — GPU gradient + geometry minimisation

**Goal:** `GPU4PySCFCalculator.opt` (minimisation) via GPU gradients, at geometry/energy
parity with the Psi4 Cartesian min-opt.

**Files**
- Extend `src/snar_qc/qc/gpu4pyscf_calculator.py` with the `opt` path.

**Approach**
- Gradient: `mf.nuc_grad_method()`.
- Minimise with geomeTRIC: `pyscf.geomopt.geometric_solver.optimize(mf, ...)`.
- **Carry the Psi4 lesson forward:** rigid planar aromatics with a near-linear substituent
  (aryl nitriles, e.g. the 5-ring reference) drove optking's internals degenerate, so the Psi4
  path forces Cartesian min-opt. Verify geomeTRIC converges these; if its internals show the
  same ill-conditioning, select Cartesian/TRIC coordinates accordingly.

**Tests**
- Slow (GPU): optimise the 5-ring reference aryl halide; final energy within tol of the Psi4
  min-opt; converged in a sane iteration count.
- Fast: option/coordinate-selection logic.
- Re-run the full suite.

**Done when:** min-opt parity on the hard aryl-nitrile case, suite green, uncommitted.
