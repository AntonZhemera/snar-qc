"""snar_qc.ts -- transition-state search wrappers for snar_qc.

This subpackage adapts the vendored predict-snar relaxed-scan transition-state search
(:class:`predict_snar.calculators.TSScan`) onto snar_qc's Psi4 backend, without editing
the vendored code.

:class:`snar_qc.ts.psi4_tsscan.Psi4TSScan` keeps predict-snar's xTB relaxed scan
unchanged but runs the DFT single points along the scan with Psi4 (synchronously,
in-process) instead of Gaussian 16, and replaces the Gaussian NBO bond orders used by
the peak-validation criterion with Psi4 (Mayer) bond orders. It is imported explicitly
(``from snar_qc.ts.psi4_tsscan import Psi4TSScan``) so that importing this subpackage
does not pull in Psi4 / predict_snar.
"""
