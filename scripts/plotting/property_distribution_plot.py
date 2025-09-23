#!/usr/bin/env python3
"""
Plot PRISM vs DiffSBDD distributions per pocket with p-values.

Input CSV should look like props_all_methods.csv produced by the collector:
columns include: method, pocket, file, mol_index, smiles,
                 qed, sa, logp, mw, hbd, hba, rotb, lipinski_count, lipinski_pass

Examples
--------
# Basic: QED across 6cm4, 6luq, 6v0u
python plot_props_violin.py \
  --csv /path/to/molecular_property_csvs/props_all_methods.csv \
  --property qed \
  --pockets 6cm4 6luq 6v0u

# Single pocket, e.g., 7e2z (auto pools 20, 30 nodes)
python plot_props_violin.py \
  --csv /path/to/molecular_property_csvs/props_all_methods.csv \
  --property qed \
  --pockets 7e2z
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import Mann–Whitney U, fall back to message if SciPy missing
try:
    from scipy.stats import mannwhitneyu
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------- Styling -----------
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.title_fontsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.alpha": 0.2,
    })


# ----------- Utils -----------
COLOR_PRISM = "orange"
COLOR_DIFF  = "steelblue"

STAR_THRESHOLDS = [
    (1e-4, "****"),
    (1e-3, "***"),
    (1e-2, "**"),
    (5e-2, "*"),
]

def p_to_stars(p: float) -> str:
    for thr, stars in STAR_THRESHOLDS:
        if p <= thr:
            return stars
    return "ns"

def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini–Hochberg FDR correction. Returns adjusted p-values."""
    if not pvals:
        return []
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n+1)

    adj = p * n / ranks
    # enforce monotonicity
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    adj_final = np.empty_like(adj_sorted)
    adj_final[order] = adj_sorted
    return np.clip(adj_final, 0.0, 1.0).tolist()

def draw_sig_bar(ax, x1, x2, y, text, line_height=0.015):
    """Draw a significance bar with text between x1 and x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], lw=1.0, c="k")
    ax.text((x1 + x2) / 2.0, y + line_height*1.1, text,
            ha="center", va="bottom", fontsize=11)


# ----------- Core -----------
def load_and_filter(csv_path: str, property_col: str, pockets: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalise columns
    required = {"method", "pocket", property_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # normalise method names, keep only PRISM and DiffSBDD
    df["method_norm"] = df["method"].astype(str).str.strip()
    df.loc[df["method_norm"].str.lower().str.contains("prism"), "method_norm"] = "PRISM"
    df.loc[df["method_norm"].str.lower().str.contains("diffsbbd"), "method_norm"] = "DiffSBDD"

    df = df[df["method_norm"].isin(["PRISM", "DiffSBDD"])].copy()

    if pockets:
        pockets_lower = {p.lower() for p in pockets}
        df = df[df["pocket"].astype(str).str.lower().isin(pockets_lower)].copy()

    # drop NaNs in the chosen property
    df = df[np.isfinite(df[property_col])].copy()
    return df


def prepare_data(df: pd.DataFrame, property_col: str, pockets: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns dict: pocket -> { "PRISM": values, "DiffSBDD": values }
    """
    data = {}
    # if pockets not provided, use sorted unique pockets present
    pockets_to_plot = pockets or sorted(df["pocket"].unique(), key=str)
    for p in pockets_to_plot:
        sub = df[df["pocket"].astype(str).str.lower() == str(p).lower()]
        data[p] = {
            "PRISM": sub[sub["method_norm"] == "PRISM"][property_col].to_numpy(),
            "DiffSBDD": sub[sub["method_norm"] == "DiffSBDD"][property_col].to_numpy(),
        }
    return data


def compute_stats(data: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Mann–Whitney U, two-sided, per pocket.
    Returns a dataframe with raw and BH-FDR corrected p-values.
    """
    if not _HAS_SCIPY:
        print("[WARN] SciPy not found, p-values will be set to NaN.")
        rows = []
        for pocket, d in data.items():
            n1, n2 = len(d["PRISM"]), len(d["DiffSBDD"])
            rows.append(dict(pocket=pocket, n_prism=n1, n_diff=n2, p_raw=np.nan, p_fdr=np.nan))
        return pd.DataFrame(rows)

    p_raw = []
    rows = []
    for pocket, d in data.items():
        a = d["PRISM"]
        b = d["DiffSBDD"]
        if len(a) == 0 or len(b) == 0:
            rows.append(dict(pocket=pocket, n_prism=len(a), n_diff=len(b), p_raw=np.nan))
            p_raw.append(np.nan)
        else:
            stat = mannwhitneyu(a, b, alternative="two-sided")
            rows.append(dict(pocket=pocket, n_prism=len(a), n_diff=len(b), p_raw=float(stat.pvalue)))
            p_raw.append(float(stat.pvalue))

    # FDR correction, keep NaNs untouched
    pvals = [x for x in p_raw if np.isfinite(x)]
    padj_map = {}
    if pvals:
        adj = bh_fdr(pvals)
        it = iter(adj)
        for pr in p_raw:
            if np.isfinite(pr):
                padj_map[pr] = next(it)

    out_rows = []
    for r in rows:
        pr = r.get("p_raw", np.nan)
        r["p_fdr"] = padj_map.get(pr, np.nan) if np.isfinite(pr) else np.nan
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def plot_violins(data: Dict[str, Dict[str, np.ndarray]],
                 stats: pd.DataFrame,
                 property_col: str,
                 save_dir: Path,
                 fname_stem: str,
                 ylim_padding: float = 0.05):
    set_pub_style()

    pockets = list(data.keys())
    n = len(pockets)
    if n == 0:
        print("No pockets to plot, aborting.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=True, dpi=300)

    # handle the case n == 1
    if n == 1:
        axes = [axes]

    y_max_overall = -np.inf
    y_min_overall = np.inf

    for ax, pocket in zip(axes, pockets):
        prism_vals = data[pocket]["PRISM"]
        diff_vals  = data[pocket]["DiffSBDD"]

        parts = ax.violinplot([prism_vals, diff_vals], showmeans=True, showmedians=True)
        # color bodies
        for pc, color in zip(parts['bodies'], [COLOR_PRISM, COLOR_DIFF]):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.8)

        # visible mean, median lines
        for k in ["cmeans", "cmedians", "cbars", "cmaxes", "cmins"]:
            if k in parts:
                parts[k].set_linewidth(1.0)
                parts[k].set_color("black")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["PRISM", "DiffSBDD"])
        ax.set_title(str(pocket).upper())

        ax.set_ylabel(property_col.upper())

        # collect y limits
        all_vals = np.concatenate([prism_vals, diff_vals]) if (len(prism_vals)+len(diff_vals)) else np.array([0.0])
        y_min_overall = min(y_min_overall, float(np.nanmin(all_vals)))
        y_max_overall = max(y_max_overall, float(np.nanmax(all_vals)))

        # annotate counts
        n_prism = len(prism_vals)
        n_diff  = len(diff_vals)
        ax.text(1, ax.get_ylim()[0], f"n={n_prism}", ha="center", va="bottom", fontsize=10)
        ax.text(2, ax.get_ylim()[0], f"n={n_diff}", ha="center", va="bottom", fontsize=10)

        # add significance
        srow = stats[stats["pocket"].astype(str).str.lower() == str(pocket).lower()]
        if len(srow) == 1:
            p_adj = srow["p_fdr"].values[0]
            label = p_to_stars(p_adj) if np.isfinite(p_adj) else "ns"
            # compute height slightly above current max
            y_top = float(np.nanmax(all_vals)) if np.isfinite(np.nanmax(all_vals)) else 0.0
            pad = (np.abs(y_top) + 1.0) * 0.05
            draw_sig_bar(ax, 1, 2, y_top + pad, label)

    # harmonise y limits with padding
    if np.isfinite(y_min_overall) and np.isfinite(y_max_overall):
        span = y_max_overall - y_min_overall
        pad = span * ylim_padding if span > 0 else (abs(y_max_overall) + 1.0) * ylim_padding
        for ax in axes:
            ax.set_ylim(y_min_overall - pad, y_max_overall + pad * 2.0)

    fig.tight_layout()

    png = save_dir / f"{fname_stem}.png"
    svg = save_dir / f"{fname_stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"[OK] Saved figures -> {png} , {svg}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Violin plots with PRISM vs DiffSBDD comparisons and FDR-corrected p-values."
    )
    ap.add_argument("--csv", required=True, type=str,
                    help="Path to props_all_methods.csv")
    ap.add_argument("--property", required=True, type=str,
                    choices=["qed", "sa", "logp", "mw", "hbd", "hba", "rotb"],
                    help="Property column to plot")
    ap.add_argument("--pockets", nargs="*", default=None,
                    help="Pocket codes to include, e.g., 6cm4 6luq 6v0u, if omitted, all pockets in CSV are used")
    ap.add_argument("--outname", type=str, default=None,
                    help="Output file stem, defaults to <property>_violins")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Figures go next to CSV in a 'plots' folder
    save_dir = csv_path.parent / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    prop = args.property.lower()
    df = load_and_filter(str(csv_path), prop, args.pockets)
    data = prepare_data(df, prop, args.pockets)

    # Stats and console table
    stats = compute_stats(data)
    stats = stats.sort_values("pocket")
    print("\nPer-pocket stats (Mann–Whitney U, two-sided, BH-FDR):")
    print(stats.to_string(index=False, justify="center"))

    # Plot
    fname_stem = args.outname or f"{prop}_violins"
    plot_violins(data, stats, prop, save_dir, fname_stem)


if __name__ == "__main__":
    main()
