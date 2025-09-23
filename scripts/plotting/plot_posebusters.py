import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy.stats import chi2_contingency
from statannotations.Annotator import Annotator

# ---------- Labels (single source of truth) ----------
PASS_LABEL = "PB-Valid"
FAIL_LABEL = "PB-Invalid"

# ---------- Styling ----------
def _set_pub_style():
    """Sets a clean, publication-ready style for the plot."""
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.title_fontsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })

def pvalue_to_asterisks(pvalue: float) -> str:
    """Converts a p-value to a standard significance string."""
    if pvalue <= 0.0001:
        return '****'
    if pvalue <= 0.001:
        return '***'
    elif pvalue <= 0.01:
        return '**'
    elif pvalue <= 0.05:
        return '*'
    else:
        return 'ns' # Not Significant

def plot_pass_fail_rates(input_data: list, output_file: str):
    """
    Builds a stacked percentage bar chart of PoseBusters results and adds
    chi-squared significance annotations.
    """
    _set_pub_style()

    # --- 1. Load and Preprocess Data ---
    all_dfs = []
    for method_name, csv_path in input_data:
        try:
            tmp = pd.read_csv(csv_path)
            tmp["Method"] = method_name
            all_dfs.append(tmp)
        except FileNotFoundError:
            print(f"Error: Input file not found: {csv_path}")
            return
        except Exception as e:
            print(f"Error reading or processing file '{csv_path}': {e}")
            return

    if not all_dfs:
        print("Error: No data could be loaded.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    if "passed_all_checks" not in df.columns:
        print("Error: CSV files must contain a 'passed_all_checks' column.")
        return

    # Coerce boolean column to handle various input formats (True, "True", 1, etc.)
    df["passed_all_checks"] = df["passed_all_checks"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).astype(bool)

    # --- 2. Aggregate Data and Calculate Percentages ---
    counts_df = df.groupby("Method")["passed_all_checks"].value_counts().unstack(fill_value=0)

    if True not in counts_df.columns: counts_df[True] = 0
    if False not in counts_df.columns: counts_df[False] = 0

    counts_df = counts_df.reindex(columns=[True, False])
    counts_df.columns = [PASS_LABEL, FAIL_LABEL]
    counts_df["Total"] = counts_df.sum(axis=1)

    # Calculate percentage columns for plotting
    counts_df["Valid (%)"] = counts_df[PASS_LABEL] / counts_df["Total"] * 100
    counts_df["Invalid (%)"] = counts_df[FAIL_LABEL] / counts_df["Total"] * 100

    # Order methods by pass rate for a more intuitive plot
    counts_df = counts_df.sort_values("Valid (%)", ascending=False)
    ordered_methods = counts_df.index.tolist()
    df["Method"] = pd.Categorical(df["Method"], categories=ordered_methods, ordered=True)

    print("--- Summary Statistics ---")
    print(counts_df)
    print("--------------------------\n")

    # --- 3. Create the Plot ---
    fig_width = max(8, 1.5 * len(ordered_methods))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bar_width = 0.6
    x = range(len(counts_df))

    # New lighter colors
    passed_color = "#99D5C9" # Light teal/green
    failed_color = "#F7B2B2" # Light red

    # Stacked bars using percentage values
    ax.bar(x, counts_df["Valid (%)"], width=bar_width, color=passed_color, label=PASS_LABEL)
    ax.bar(x, counts_df["Invalid (%)"], width=bar_width, bottom=counts_df["Valid (%)"], color=failed_color, label=FAIL_LABEL)

    # Add percentage labels inside the 'Valid' segment
    for i, method in enumerate(ordered_methods):
        valid_pct = counts_df.loc[method, "Valid (%)"]
        if valid_pct > 0:
            ax.text(i, valid_pct / 2, f"{valid_pct:.1f}%",
                    ha="center", va="center", color="black", fontsize=11, weight="bold")

    # --- 4. Perform Statistical Tests and Add Annotations ---
    if len(ordered_methods) > 1:
        # Create a dummy column for positioning annotations above the 100% mark
        df['y_pos'] = 100
        pairs_to_compare = list(combinations(ordered_methods, 2))
        p_values = []

        print("--- Statistical Analysis (Chi-Squared Test) ---")
        for m1, m2 in pairs_to_compare:
            contingency_table = [[counts_df.loc[m1, PASS_LABEL], counts_df.loc[m1, FAIL_LABEL]],
                                 [counts_df.loc[m2, PASS_LABEL], counts_df.loc[m2, FAIL_LABEL]]]
            _, p, _, _ = chi2_contingency(contingency_table, correction=True)
            p_values.append(p)
            print(f"Comparing '{m1}' vs '{m2}': p-value = {p:.4f} ({pvalue_to_asterisks(p)})")

        annotator = Annotator(ax, pairs_to_compare, data=df, x="Method", y="y_pos")
        annotator.configure(test=None, text_format="star", loc="outside", verbose=0)
        annotator.set_pvalues(p_values)
        annotator.annotate()

    # --- 5. Finalize and Save the Plot ---
    ax.set_ylabel("Percentage of predictions")
    ax.set_xlabel(None)
    ax.set_xticks(list(x))
    
    # Create new x-axis labels that include the total count (N)
    new_xticklabels = [
        f"{method}\n(N={counts_df.loc[method, 'Total']:,})" for method in ordered_methods
    ]
    ax.set_xticklabels(new_xticklabels, rotation=0, ha="center")
    
    # Format Y-axis to show percentage symbol
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 100 * 1.25) # Set Y-limit and add headroom for annotations

    ax.legend(title="Result", frameon=False, loc="upper right")
    ax.set_title("PB-Valid Predictions by Method", pad=20)
    
    plt.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(svg_path) # Also save a vector version for publications
    print(f"\n✅ Plot saved to: {output_path}")
    print(f"✅ Vector SVG saved to: {svg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a stacked percentage bar plot of PoseBusters results with chi-squared significance.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", action="append", nargs=2, metavar=("METHOD_NAME", "CSV_FILE"),
        required=True,
        help="Specify a method name and its CSV results file.\nUse multiple --input pairs to compare methods."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="PB_plots/pass_fail_comparison.png",
        help="Path to save the output PNG. An SVG will also be saved."
    )
    args = parser.parse_args()
    plot_pass_fail_rates(args.input, args.output)


if __name__ == "__main__":
    main()

