"""snar_qc.qc -- quantum-chemistry engine wrappers for snar_qc.

This subpackage hosts snar_qc's own QC calculators, which build on the vendored
``predict_snar`` calculator framework (``predict_snar.calculators.Calculator`` and
its ``single_point`` / ``opt`` / ``opt_freq`` / ``freq`` dispatch) without editing
the vendored code.

The first engine is :class:`snar_qc.qc.psi4_calculator.Psi4Calculator`, a Psi4
(Python-API) drop-in for predict-snar's Gaussian-16 ``G16Calculator``. It is imported
explicitly (``from snar_qc.qc.psi4_calculator import Psi4Calculator``) rather than re-
exported here, so that importing this subpackage does not pull in Psi4 / predict_snar.
"""
