#!/usr/bin/env python
"""Fit the united three-leaving-group ΔG‡ calibration from per-model validation joins.

Each leaving group (F / Cl / Br) carries its **own systematic offset** between the
computed POC barrier (gas-phase neutral methylamine) and the experimental Lu_74
barrier (anionic alkoxide, solution) -- fluoride is over-penalised ~2x relative to
the Br/Cl cluster. ``validate_poc.py`` reports each model/LG separately; this script
folds the three leaving groups into a **single** calibration so one model serves all
of F/Cl/Br.

Two nested calibrations are fit (computed-on-experimental, the validation orientation
``comp = slope * exp + intercept``):

* **United (shared slope, per-LG intercept)** -- ``comp = beta * exp + gamma[LG]``.
  One slope across all leaving groups; the per-LG intercept ``gamma[LG]`` *is* the
  individual offset correction. This is the headline deliverable: the calibrated
  ranker a downstream consumer applies across F/Cl/Br.
* **Per-LG slope (full)** -- ``comp = beta[LG] * exp + gamma[LG]``, i.e. the three
  independent per-LG fits stacked. Used only as the comparison arm.

A partial F-test decides whether the per-LG slopes are worth their two extra
parameters; with the small per-LG counts (F n<=9) the shared-slope model is
preferred unless the F-test is clearly significant -- the aim is a *generalising*
calibration, not magnitude tuning to Lu_74.

The headline metrics are reported on the **calibrated experimental prediction**
``exp_hat = (comp - gamma[LG]) / beta``: pooled Spearman (ranking, the headline),
Pearson, MAE and RMSE in kcal/mol.

Input join (``--primary``, optionally backfilled by ``--fallback`` per missing
substrate) is a ``poc_validation_join.csv`` from ``validate_poc.py``; the standard
workflow is SMD primary, IEF-PCM fallback. Every row is tagged with its
``source_model`` so the provenance mix stays auditable.

Run in any env with pandas / numpy / scipy / matplotlib (e.g. ``snar-qc``)::

    python scripts/fit_united_model.py \\
        --primary  notes/assets/gpu_dmso_smd/poc_validation_join.csv     --primary-model smd \\
        --fallback notes/assets/gpu_dmso_iefpcm/poc_validation_join.csv  --fallback-model iefpcm \\
        --outdir   notes/assets/united_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

PALETTE = {"Cl": "#2c7fb8", "F": "#d95f0e", "Br": "#31a354"}
GROUPS = ("F", "Cl", "Br")
COLUMN = "delta_g_qh_kcal"


def load_completed(join_csv: Path, model: str) -> pd.DataFrame:
    """Confirmed-saddle rows of a validation join, tagged with their source model."""
    df = pd.read_csv(join_csv)
    ok = df[(df["status"] == "completed") & df[COLUMN].notna()].copy()
    ok["source_model"] = model
    return ok[["lu_id", "leaving_group", "exp_dg_kcal", COLUMN, "source_model"]]


def build_blended(
    primary: pd.DataFrame, fallback: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Primary join, backfilled per missing ``lu_id`` from the fallback join."""
    if fallback is None:
        return primary.reset_index(drop=True)
    missing = fallback[~fallback["lu_id"].isin(primary["lu_id"])]
    return pd.concat([primary, missing], ignore_index=True).sort_values("lu_id")


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares solve; returns (coefficients, residual sum of squares)."""
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    rss = float(np.sum((y - x @ coef) ** 2))
    return coef, rss


def fit_united(df: pd.DataFrame) -> dict:
    """Fit the shared-slope and per-LG-slope calibrations and compare them.

    Args:
        df: Blended frame with ``leaving_group``, ``exp_dg_kcal`` and the computed
            barrier column for every confirmed-saddle substrate.

    Returns:
        Stats dict: shared-slope coefficients (beta + per-LG intercepts), the
        calibrated-prediction metrics, the per-LG-slope comparison, and the
        partial F-test that justifies the slope choice.
    """
    groups = [g for g in GROUPS if (df["leaving_group"] == g).any()]
    exp = df["exp_dg_kcal"].to_numpy(float)
    comp = df[COLUMN].to_numpy(float)
    n = len(df)
    dummies = {g: (df["leaving_group"] == g).to_numpy(float) for g in groups}
    # Intuitive per-LG correction: the mean computed-minus-experimental offset
    # (the slope-1 correction, matching validate_poc's mean_offset_kcal).
    mean_offset = {
        g: round(float(np.mean(comp[dummies[g] == 1] - exp[dummies[g] == 1])), 3)
        for g in groups
    }

    # Shared slope, per-LG intercept: comp = beta*exp + sum_g gamma_g * 1[LG=g].
    x_shared = np.column_stack([exp] + [dummies[g] for g in groups])
    coef_s, rss_s = _fit(x_shared, comp)
    beta = float(coef_s[0])
    gamma = {g: float(coef_s[1 + i]) for i, g in enumerate(groups)}

    # Per-LG slope (full): comp = sum_g (beta_g*exp + gamma_g) 1[LG=g].
    x_full = np.column_stack(
        [exp * dummies[g] for g in groups] + [dummies[g] for g in groups]
    )
    coef_f, rss_f = _fit(x_full, comp)
    beta_g = {g: float(coef_f[i]) for i, g in enumerate(groups)}
    gamma_g = {g: float(coef_f[len(groups) + i]) for i, g in enumerate(groups)}

    tss = float(np.sum((comp - comp.mean()) ** 2))
    k_s, k_f = 1 + len(groups), 2 * len(groups)
    r2_s = 1.0 - rss_s / tss
    r2_f = 1.0 - rss_f / tss
    adj = lambda r2, k: 1.0 - (1.0 - r2) * (n - 1) / (n - k)  # noqa: E731

    # Partial F-test: do the per-LG slopes earn their two extra parameters?
    df_num, df_den = k_f - k_s, n - k_f
    f_stat = (
        ((rss_s - rss_f) / df_num) / (rss_f / df_den) if df_den > 0 else float("nan")
    )
    f_p = float(sps.f.sf(f_stat, df_num, df_den)) if df_den > 0 else float("nan")

    # Calibrated experimental prediction from the shared-slope model.
    gamma_arr = np.array([gamma[g] for g in df["leaving_group"]])
    exp_hat = (comp - gamma_arr) / beta
    resid = exp_hat - exp
    spear = float(sps.spearmanr(exp_hat, exp).statistic)
    pear = float(sps.pearsonr(exp_hat, exp)[0])

    return {
        "metric_column": COLUMN,
        "n": n,
        "leaving_groups": groups,
        "united_shared_slope": {
            "slope": round(beta, 4),
            "intercept_by_lg": {g: round(gamma[g], 4) for g in groups},
            "mean_offset_by_lg": mean_offset,
            "r_squared": round(r2_s, 4),
            "adj_r_squared": round(adj(r2_s, k_s), 4),
            "n_params": k_s,
        },
        "calibrated_prediction": {
            "spearman_r": round(spear, 4),
            "pearson_r": round(pear, 4),
            "mae_kcal": round(float(np.mean(np.abs(resid))), 3),
            "rmse_kcal": round(float(np.sqrt(np.mean(resid**2))), 3),
            "max_abs_err_kcal": round(float(np.max(np.abs(resid))), 3),
        },
        "per_lg_slope_model": {
            "slope_by_lg": {g: round(beta_g[g], 4) for g in groups},
            "intercept_by_lg": {g: round(gamma_g[g], 4) for g in groups},
            "r_squared": round(r2_f, 4),
            "adj_r_squared": round(adj(r2_f, k_f), 4),
            "n_params": k_f,
        },
        "slope_choice_f_test": {
            "f_stat": round(f_stat, 4),
            "p_value": round(f_p, 4),
            "df": [df_num, df_den],
            "verdict": (
                "shared slope sufficient (per-LG slopes not justified, p>=0.05)"
                if not (f_p < 0.05)
                else "per-LG slopes significantly improve fit (p<0.05)"
            ),
        },
        "source_model_counts": df["source_model"].value_counts().to_dict(),
    }


def make_plot(df: pd.DataFrame, stats: dict, out_png: Path) -> None:
    """Two panels: raw comp-vs-exp with the per-LG fit lines, and the collapsed fit."""
    groups = stats["leaving_groups"]
    beta = stats["united_shared_slope"]["slope"]
    gamma = stats["united_shared_slope"]["intercept_by_lg"]
    cp = stats["calibrated_prediction"]
    exp = df["exp_dg_kcal"].to_numpy(float)
    comp = df[COLUMN].to_numpy(float)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.0))

    # Left: raw computed vs experimental, three parallel shared-slope lines.
    xs = np.array([exp.min() - 0.5, exp.max() + 0.5])
    for g in groups:
        sub = df[df["leaving_group"] == g]
        axl.scatter(
            sub["exp_dg_kcal"],
            sub[COLUMN],
            c=PALETTE.get(g, "#666"),
            s=55,
            zorder=3,
            label=f"{g} (n={len(sub)})",
        )
        axl.plot(xs, beta * xs + gamma[g], "-", c=PALETTE.get(g, "#666"), lw=1.1)
    axl.set_title(
        f"United model: comp = {beta:.3f}·exp + γ(LG)\n"
        f"shared slope, per-LG intercept  (R²={stats['united_shared_slope']['r_squared']})",
        fontsize=9,
    )
    axl.set_xlabel("Experimental ΔG‡ (kcal/mol)")
    axl.set_ylabel(f"Computed {COLUMN} (kcal/mol)")
    axl.legend(fontsize=8, loc="upper left")

    # Right: calibrated experimental prediction vs experimental, all LGs collapsed.
    gamma_arr = np.array([gamma[g] for g in df["leaving_group"]])
    exp_hat = (comp - gamma_arr) / beta
    for g in groups:
        m = df["leaving_group"] == g
        axr.scatter(
            exp[m.to_numpy()],
            exp_hat[m.to_numpy()],
            c=PALETTE.get(g, "#666"),
            s=55,
            zorder=3,
            label=g,
        )
    lo, hi = min(exp.min(), exp_hat.min()) - 1, max(exp.max(), exp_hat.max()) + 1
    axr.plot([lo, hi], [lo, hi], "--", c="grey", lw=1, label="parity")
    axr.set_title(
        f"Calibrated prediction (per-LG offset removed)\n"
        f"ρ={cp['spearman_r']}  r={cp['pearson_r']}  MAE={cp['mae_kcal']} kcal/mol",
        fontsize=9,
    )
    axr.set_xlabel("Experimental ΔG‡ (kcal/mol)")
    axr.set_ylabel("Calibrated ΔG‡ prediction (kcal/mol)")
    axr.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="primary validation join CSV")
    parser.add_argument(
        "--primary-model", default="smd", help="provenance tag for --primary"
    )
    parser.add_argument(
        "--fallback", help="optional fallback join CSV (per-substrate backfill)"
    )
    parser.add_argument(
        "--fallback-model", default="iefpcm", help="provenance tag for --fallback"
    )
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    primary = load_completed(Path(args.primary), args.primary_model)
    fallback = (
        load_completed(Path(args.fallback), args.fallback_model)
        if args.fallback
        else None
    )
    df = build_blended(primary, fallback)

    stats = fit_united(df)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "united_model_join.csv").write_text(df.to_csv(index=False))
    (out / "united_model_stats.json").write_text(json.dumps(stats, indent=2))
    make_plot(df, stats, out / "united_model_scatter.png")

    print(json.dumps(stats, indent=2))
    print(f"\nOutdir: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
