#!/usr/bin/env python
"""Validate POC ΔG‡ against the Lu_74 reference: correlation, magnitude, scatter.

Joins the per-substrate sidecars written by ``run_poc.py`` to the experimental Lu_74
barriers and reports the POC's quality. Because the computed barriers use a neutral
model amine (methylamine) in the gas phase while Lu_74 was measured with an anionic
benzyl-alkoxide in solution, the headline metric is **ranking** (Spearman / Pearson),
which is invariant to the expected systematic offset; magnitude (MAE, and MAE after
removing the mean offset) is reported alongside but weighted less.

Outputs (into ``--outdir``):

* ``poc_validation_join.csv`` -- experimental vs computed, per substrate.
* ``poc_validation_stats.json`` -- Pearson / Spearman / MAE / offset-corrected MAE.
* ``poc_validation_scatter.png`` -- computed vs experimental ΔG‡ scatter.

Run inside the ``snar-qc`` conda env, e.g.::

    python scripts/validate_poc.py --slice data/external/lu74_poc_slice.csv \\
        --run data/processed/poc_run --outdir notes/assets
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
from scipy.stats import pearsonr, spearmanr  # noqa: E402

KJ_PER_KCAL = 4.184
# Leaving-group colours, shared by every scatter -- the LG-dependent offset is the headline.
PALETTE = {"Cl": "#2c7fb8", "F": "#d95f0e", "Br": "#31a354"}


def load_computed(run_dir: Path) -> pd.DataFrame:
    """Collect every ``result.json`` sidecar under a run directory into a frame."""
    rows = []
    for sidecar in sorted(run_dir.glob("*/result.json")):
        rows.append(json.loads(sidecar.read_text()))
    if not rows:
        raise SystemExit(f"No result.json sidecars found under {run_dir}.")
    return pd.DataFrame(rows)


def build_join(slice_csv: Path, run_dir: Path) -> pd.DataFrame:
    """Join the experimental slice to the computed sidecars on ``lu_id``."""
    exp = pd.read_csv(slice_csv)
    exp["exp_dg_kcal"] = exp["delta_g_kJmol"] / KJ_PER_KCAL
    comp = load_computed(run_dir)
    join = exp.merge(
        comp[
            [
                "lu_id",
                "status",
                "delta_g_qh_kcal",
                "delta_g_kcal",
                "delta_e_kcal",
                "n_imag_ts",
                "ts_imag_freq_cm",
            ]
        ],
        on="lu_id",
        how="left",
    )
    return join


def compute_stats(join: pd.DataFrame, column: str = "delta_g_qh_kcal") -> dict:
    """Correlation and magnitude metrics over the confirmed-saddle subset.

    Args:
        join: The experimental/computed join.
        column: Which computed barrier column to score.

    Returns:
        A metrics dict (counts, Pearson, Spearman, MAE, offset, offset-corrected MAE).
    """
    ok = join[(join["status"] == "completed") & join[column].notna()].copy()
    n = len(ok)
    stats: dict = {
        "metric_column": column,
        "n_substrates_total": int(len(join)),
        "n_completed": int(n),
        "completed_lu_ids": sorted(int(x) for x in ok["lu_id"].tolist()),
    }
    if n < 3:
        stats["note"] = f"Only {n} confirmed saddle(s); correlation needs >= 3."
        return stats

    exp = ok["exp_dg_kcal"].to_numpy()
    comp = ok[column].to_numpy()
    pear_r, pear_p = pearsonr(comp, exp)
    spear_r, spear_p = spearmanr(comp, exp)
    offset = float(np.mean(comp - exp))
    mae = float(np.mean(np.abs(comp - exp)))
    mae_oc = float(np.mean(np.abs((comp - offset) - exp)))
    stats.update(
        {
            "pearson_r": round(float(pear_r), 4),
            "pearson_p": round(float(pear_p), 4),
            "spearman_r": round(float(spear_r), 4),
            "spearman_p": round(float(spear_p), 4),
            "mae_kcal": round(mae, 3),
            "mean_offset_kcal": round(offset, 3),
            "mae_offset_corrected_kcal": round(mae_oc, 3),
            "computed_range_kcal": [
                round(float(comp.min()), 2),
                round(float(comp.max()), 2),
            ],
            "exp_range_kcal": [round(float(exp.min()), 2), round(float(exp.max()), 2)],
        }
    )

    # Per-leaving-group breakdown: the headline pooled correlation can be dragged
    # down by a leaving-group-dependent offset (e.g. gas-phase C-F cleavage is
    # over-penalised), so report each family separately too.
    by_group: dict = {}
    for group, sub in ok.groupby("leaving_group"):
        if len(sub) < 3:
            by_group[str(group)] = {"n": int(len(sub)), "note": "too few for r"}
            continue
        ge = sub["exp_dg_kcal"].to_numpy()
        gc = sub[column].to_numpy()
        goff = float(np.mean(gc - ge))
        # Per-LG calibration regression, computed-on-experimental
        # (comp = slope * exp + intercept). The slope answers the diagnostic
        # "does this leaving group respond more/less steeply than the others?";
        # an ideal calibration has slope ~1 and intercept ~mean_offset. R^2 is
        # direction-invariant (== pearson_r ** 2). The per-LG *correction* applied
        # downstream remains the mean offset (mean_offset_kcal).
        slope, intercept = (float(x) for x in np.polyfit(ge, gc, 1))
        pred = slope * ge + intercept
        ss_res = float(np.sum((gc - pred) ** 2))
        ss_tot = float(np.sum((gc - gc.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        by_group[str(group)] = {
            "n": int(len(sub)),
            "spearman_r": round(float(spearmanr(gc, ge).statistic), 4),
            "pearson_r": round(float(pearsonr(gc, ge)[0]), 4),
            "ols_slope": round(slope, 4),
            "ols_intercept": round(intercept, 4),
            "r_squared": round(r2, 4),
            "mean_offset_kcal": round(goff, 3),
            "mae_offset_corrected_kcal": round(
                float(np.mean(np.abs((gc - goff) - ge))), 3
            ),
        }
    stats["by_leaving_group"] = by_group
    return stats


def make_scatter(join: pd.DataFrame, stats: dict, out_png: Path, column: str) -> None:
    """Scatter of computed vs experimental ΔG‡ with a best-fit and parity line."""
    ok = join[(join["status"] == "completed") & join[column].notna()]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    exp = ok["exp_dg_kcal"].to_numpy()
    comp = ok[column].to_numpy()
    # Colour by leaving group -- the leaving-group-dependent offset is the headline.
    for group, sub in ok.groupby("leaving_group"):
        ax.scatter(
            sub["exp_dg_kcal"],
            sub[column],
            c=PALETTE.get(str(group), "#666666"),
            s=60,
            zorder=3,
            label=f"{group} (n={len(sub)})",
        )
    for _, row in ok.iterrows():
        ax.annotate(
            str(int(row["lu_id"])),
            (row["exp_dg_kcal"], row[column]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=8,
        )
    if len(ok) >= 2:
        lo = min(exp.min(), comp.min()) - 2
        hi = max(exp.max(), comp.max()) + 2
        # Best-fit line (rank/linear trend) and a parity guide shifted by the offset.
        slope, intercept = np.polyfit(exp, comp, 1)
        xs = np.array([exp.min() - 1, exp.max() + 1])
        ax.plot(xs, slope * xs + intercept, "-", c="black", lw=1.2, label="best fit")
        offset = stats.get("mean_offset_kcal", 0.0)
        ax.plot(
            [lo, hi],
            [lo + offset, hi + offset],
            "--",
            c="grey",
            lw=1,
            label="parity+offset",
        )
        ax.legend(fontsize=8, loc="upper left")
    title = "POC ΔG‡ (gas-phase methylamine) vs Lu_74 (soln, alkoxide)"
    if "spearman_r" in stats:
        title += (
            f"\nSpearman ρ={stats['spearman_r']}  Pearson r={stats['pearson_r']}  "
            f"MAE={stats['mae_kcal']} (oc {stats['mae_offset_corrected_kcal']}) kcal/mol"
        )
    ax.set_xlabel("Experimental ΔG‡ (kcal/mol)")
    ax.set_ylabel(f"Computed {column} (kcal/mol)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def make_per_lg_plot(
    join: pd.DataFrame, stats: dict, out_png: Path, column: str
) -> None:
    """One calibration panel per leaving group: points + that LG's own fit.

    Each panel shows computed vs experimental ΔG‡ for a single leaving group, the
    per-LG OLS trend (``comp = slope*exp + intercept``, matching ``ols_slope`` /
    ``ols_intercept`` in the stats) and a parity line shifted by that LG's mean
    offset (the individual correction). Visualises Step 2: each leaving group is
    calibrated by its own offset, and the slope shows whether one family responds
    more steeply than the others.
    """
    by_lg = stats.get("by_leaving_group", {})
    groups = [g for g in ("F", "Cl", "Br") if g in by_lg and "ols_slope" in by_lg[g]]
    if not groups:
        return
    ok = join[(join["status"] == "completed") & join[column].notna()]
    fig, axes = plt.subplots(
        1, len(groups), figsize=(4.4 * len(groups), 4.4), squeeze=False
    )
    for ax, group in zip(axes[0], groups):
        sub = ok[ok["leaving_group"] == group]
        gm = by_lg[group]
        exp = sub["exp_dg_kcal"].to_numpy()
        comp = sub[column].to_numpy()
        ax.scatter(exp, comp, c=PALETTE.get(group, "#666666"), s=55, zorder=3)
        xs = np.array([exp.min() - 0.5, exp.max() + 0.5])
        ax.plot(
            xs,
            gm["ols_slope"] * xs + gm["ols_intercept"],
            "-",
            c="black",
            lw=1.3,
            label=f"fit: slope={gm['ols_slope']}",
        )
        off = gm["mean_offset_kcal"]
        ax.plot(xs, xs + off, "--", c="grey", lw=1, label=f"parity+offset ({off:+.1f})")
        ax.set_title(
            f"{group} (n={gm['n']})  R²={gm.get('r_squared')}\n"
            f"ρ={gm['spearman_r']}  oc-MAE={gm['mae_offset_corrected_kcal']}",
            fontsize=9,
        )
        ax.set_xlabel("Experimental ΔG‡ (kcal/mol)")
        ax.set_ylabel(f"Computed {column} (kcal/mol)")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Per-leaving-group calibration (each LG by its own offset)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", required=True, help="Lu_74 slice CSV (experimental)")
    parser.add_argument("--run", required=True, help="run dir with */result.json")
    parser.add_argument("--outdir", default="notes/assets")
    parser.add_argument(
        "--column",
        default="delta_g_qh_kcal",
        help="computed barrier column to score (default quasi-harmonic ΔG‡)",
    )
    args = parser.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    join = build_join(Path(args.slice), Path(args.run))
    stats = compute_stats(join, column=args.column)

    join_out = out / "poc_validation_join.csv"
    join.to_csv(join_out, index=False)
    stats_out = out / "poc_validation_stats.json"
    stats_out.write_text(json.dumps(stats, indent=2))
    scatter_out = out / "poc_validation_scatter.png"
    per_lg_out = out / "poc_validation_per_lg.png"
    if stats.get("n_completed", 0) >= 2:
        make_scatter(join, stats, scatter_out, column=args.column)
        make_per_lg_plot(join, stats, per_lg_out, column=args.column)

    print(json.dumps(stats, indent=2))
    print(f"\nJoin:    {join_out}")
    print(f"Stats:   {stats_out}")
    if stats.get("n_completed", 0) >= 2:
        print(f"Scatter: {scatter_out}")
        if stats.get("by_leaving_group"):
            print(f"Per-LG:  {per_lg_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
