"""
Part A Statistical Analysis & Visualization (Runs on Mac).
Consumes data/results/part_a_raw.csv, computes Pearson/Spearman correlations,
p-values, and generates scatter/regression plots into analysis/plots/.

Usage:
    python analysis/part_a_stats.py
    python analysis/part_a_stats.py --csv path/to/custom.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path("analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_COLORS = {
    "constrained_factual": "#2196F3",
    "moderate_reasoning": "#FF9800",
    "open_ended": "#4CAF50",
    "adversarial_ambiguous": "#9C27B0",
}

def analyze_part_a_results(raw_csv_path: str = "data/results/part_a_raw.csv"):
    """Reads Part A CSV and computes Pearson/Spearman correlations per model, layer, and category."""
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Part A CSV not found at {raw_csv_path}. Run the GPU session first.")

    df = pd.read_csv(raw_csv_path)
    print(f"Loaded {len(df)} rows from {raw_csv_path}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Layers: {sorted(df['layer'].unique().tolist())}")
    print(f"Categories: {df['category'].unique().tolist()}")
    print()

    results = []

    for model in df["model"].unique():
        for layer in sorted(df["layer"].unique()):
            subset = df[(df["model"] == model) & (df["layer"] == layer)]
            if len(subset) < 10:
                print(f"  SKIP {model} L{layer}: too few samples (n={len(subset)})")
                continue

            x = subset["measured_entropy"].values
            y = subset["sae_sparsity_metric"].values

            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)

            result = {
                "model": model, "layer": layer, "scope": "all_categories",
                "n": len(subset), "pearson_r": round(float(r_p), 4), "pearson_p": round(float(p_p), 6),
                "spearman_r": round(float(r_s), 4), "spearman_p": round(float(p_s), 6),
                "significant_p05": bool(p_p < 0.05)
            }
            results.append(result)
            print(f"[{model} | Layer {layer} | ALL] n={len(subset)} "
                  f"Pearson r={r_p:.4f} (p={p_p:.4f}) | Spearman r={r_s:.4f} (p={p_s:.4f})")

        for layer in sorted(df["layer"].unique()):
            for category in df["category"].unique():
                subset = df[(df["model"] == model) & (df["layer"] == layer) & (df["category"] == category)]
                if len(subset) < 5:
                    continue

                x = subset["measured_entropy"].values
                y = subset["sae_sparsity_metric"].values
                r_p, p_p = stats.pearsonr(x, y)
                r_s, p_s = stats.spearmanr(x, y)

                result = {
                    "model": model, "layer": layer, "scope": category,
                    "n": len(subset), "pearson_r": round(float(r_p), 4), "pearson_p": round(float(p_p), 6),
                    "spearman_r": round(float(r_s), 4), "spearman_p": round(float(p_s), 6),
                    "significant_p05": bool(p_p < 0.05)
                }
                results.append(result)

    results_df = pd.DataFrame(results)
    out_csv = PLOTS_DIR / "part_a_correlations.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nCorrelation table saved to {out_csv}")

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        layers = sorted(model_df["layer"].unique())
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5), sharey=True)
        if len(layers) == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers):
            for category, color in CATEGORY_COLORS.items():
                sub = model_df[(model_df["layer"] == layer) & (model_df["category"] == category)]
                ax.scatter(sub["measured_entropy"], sub["sae_sparsity_metric"],
                           c=color, alpha=0.6, s=20, label=category)

            layer_sub = model_df[model_df["layer"] == layer]
            x = layer_sub["measured_entropy"].values
            y = layer_sub["sae_sparsity_metric"].values
            if len(x) > 1 and np.std(x) > 1e-6:
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, m * xline + b, "k--", linewidth=1.5, alpha=0.7)

            r_p, p_p = stats.pearsonr(x, y)
            ax.set_title(f"Layer {layer}\nr={r_p:.3f}, p={p_p:.4f}", fontsize=10)
            ax.set_xlabel("Measured Entropy (nats)")
            if ax == axes[0]:
                ax.set_ylabel("SAE Sparsity Metric")

        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=8, label=cat)
                   for cat, c in CATEGORY_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
        fig.suptitle(f"Part A: Entropy vs SAE Sparsity — {model}", fontsize=12, y=1.02)
        plt.tight_layout()
        safe_name = model.replace("/", "_").replace("-", "_")
        plot_path = PLOTS_DIR / f"part_a_scatter_{safe_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Scatter plot saved: {plot_path}")

    print("\nPart A analysis complete.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/results/part_a_raw.csv")
    args = parser.parse_args()
    analyze_part_a_results(args.csv)
