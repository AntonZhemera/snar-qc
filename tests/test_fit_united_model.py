"""Tests for the united three-leaving-group ΔG‡ calibration.

The fit must (a) recover the slope and per-LG intercepts of synthetic
``comp = beta*exp + gamma[LG]`` data, (b) keep the partial F-test *non*-significant
when the data really do share a slope, yet flag it when one leaving group has a
divergent slope, and (c) backfill missing substrates from the fallback join.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_united_model as fum  # noqa: E402


def _frame(exp, comp, lg, model="smd"):
    return pd.DataFrame(
        {
            "lu_id": range(len(exp)),
            "leaving_group": lg,
            "exp_dg_kcal": exp,
            fum.COLUMN: comp,
            "source_model": model,
        }
    )


def test_shared_slope_recovered():
    rng = np.random.default_rng(0)
    exp = rng.uniform(15, 23, 60)
    lg = np.array(["F", "Cl", "Br"] * 20)
    gamma = {"F": -1.0, "Cl": -4.5, "Br": -5.0}
    comp = 1.5 * exp + np.array([gamma[g] for g in lg]) + rng.normal(0, 0.2, 60)
    stats = fum.fit_united(_frame(exp, comp, lg))

    ss = stats["united_shared_slope"]
    assert abs(ss["slope"] - 1.5) < 0.05
    for g in ("F", "Cl", "Br"):
        assert abs(ss["intercept_by_lg"][g] - gamma[g]) < 0.4
    # Genuinely shared slope -> per-LG slopes must NOT be flagged as worthwhile.
    assert stats["slope_choice_f_test"]["p_value"] >= 0.05
    assert stats["calibrated_prediction"]["spearman_r"] > 0.95


def test_divergent_slope_flagged():
    rng = np.random.default_rng(1)
    exp = rng.uniform(15, 23, 90)
    lg = np.array(["F", "Cl", "Br"] * 30)
    slope = {"F": 0.4, "Cl": 1.6, "Br": 1.6}  # F responds very differently
    comp = np.array([slope[g] for g in lg]) * exp + rng.normal(0, 0.1, 90)
    stats = fum.fit_united(_frame(exp, comp, lg))
    assert stats["slope_choice_f_test"]["p_value"] < 0.05
    assert (
        stats["per_lg_slope_model"]["r_squared"]
        > stats["united_shared_slope"]["r_squared"]
    )


def test_blended_backfills_missing_substrate():
    primary = _frame([15.0, 16.0], [20.0, 21.0], ["Cl", "Cl"], model="smd")
    fallback = _frame(
        [15.0, 16.0, 17.0], [19.5, 20.5, 22.0], ["Cl", "Cl", "F"], "iefpcm"
    )
    fallback["lu_id"] = [0, 1, 2]
    blended = fum.build_blended(primary, fallback)
    assert len(blended) == 3  # the F substrate (lu_id 2) backfilled from IEF-PCM
    assert blended.loc[blended["lu_id"] == 2, "source_model"].iloc[0] == "iefpcm"
    assert set(blended.loc[blended["lu_id"].isin([0, 1]), "source_model"]) == {"smd"}
