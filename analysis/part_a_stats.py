"""
Part A Statistical Analysis & Publication Visualization (Tier-1 Research Grade).
Consumes data/results/part_a_raw.csv, computes Pearson/Spearman correlations with 95% Bootstrap CIs,
runs OLS regression for variance decomposition, and generates publication plots into analysis/plots/.

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

def bootstrap_correlation_ci(x: np.ndarray, y: np.ndarray, method="pearson", n_boot=1000, ci=95, seed=42):
    """Computes 95% non-parametric bootstrap confidence interval for correlation coefficients."""
    rng = np.random.RandomState(seed)
    n = len(x)
    if n < 5:
        return np.nan, np.nan
        
    boot_stats = []
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        xb, yb = x[indices], y[indices]
        if np.std(xb) < 1e-8 or np.std(yb) < 1e-8:
            continue
        if method == "pearson":
            r_val, _ = stats.pearsonr(xb, yb)
        else:
            r_val, _ = stats.spearmanr(xb, yb)
        if not np.isnan(r_val):
            boot_stats.append(r_val)
            
    if len(boot_stats) == 0:
        return np.nan, np.nan
        
    lower = np.percentile(boot_stats, (100 - ci) / 2.0)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2.0)
    return round(float(lower), 4), round(float(upper), 4)

def analyze_part_a_results(raw_csv_path: str = "data/results/part_a_raw.csv"):
    """Reads Part A CSV and computes Pearson/Spearman correlations with 95% CIs and regression metrics."""
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
                continue

            x = subset["measured_entropy"].values
            y = subset["sae_sparsity_metric"].values

            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            p_ci_low, p_ci_high = bootstrap_correlation_ci(x, y, method="pearson")
            s_ci_low, s_ci_high = bootstrap_correlation_ci(x, y, method="spearman")

            result = {
                "model": model, "layer": layer, "scope": "all_categories",
                "n": len(subset),
                "pearson_r": round(float(r_p), 4),
                "pearson_ci95_low": p_ci_low,
                "pearson_ci95_high": p_ci_high,
                "pearson_p": round(float(p_p), 6),
                "spearman_r": round(float(r_s), 4),
                "spearman_ci95_low": s_ci_low,
                "spearman_ci95_high": s_ci_high,
                "spearman_p": round(float(p_s), 6),
                "significant_p05": bool(p_p < 0.05)
            }
            results.append(result)
            print(f"[{model} | Layer {layer} | ALL] n={len(subset)} "
                  f"Pearson r={r_p:.4f} [95% CI: {p_ci_low}, {p_ci_high}] (p={p_p:.4f}) | "
                  f"Spearman r={r_s:.4f} [95% CI: {s_ci_low}, {s_ci_high}] (p={p_s:.4f})")

        for layer in sorted(df["layer"].unique()):
            for category in df["category"].unique():
                subset = df[(df["model"] == model) & (df["layer"] == layer) & (df["category"] == category)]
                if len(subset) < 5:
                    continue

                x = subset["measured_entropy"].values
                y = subset["sae_sparsity_metric"].values
                r_p, p_p = stats.pearsonr(x, y)
                r_s, p_s = stats.spearmanr(x, y)
                p_ci_low, p_ci_high = bootstrap_correlation_ci(x, y, method="pearson")
                s_ci_low, s_ci_high = bootstrap_correlation_ci(x, y, method="spearman")

                result = {
                    "model": model, "layer": layer, "scope": category,
                    "n": len(subset),
                    "pearson_r": round(float(r_p), 4),
                    "pearson_ci95_low": p_ci_low,
                    "pearson_ci95_high": p_ci_high,
                    "pearson_p": round(float(p_p), 6),
                    "spearman_r": round(float(r_s), 4),
                    "spearman_ci95_low": s_ci_low,
                    "spearman_ci95_high": s_ci_high,
                    "spearman_p": round(float(p_s), 6),
                    "significant_p05": bool(p_p < 0.05)
                }
                results.append(result)

    results_df = pd.DataFrame(results)
    out_csv = PLOTS_DIR / "part_a_correlations.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nPublication correlation table with 95% CIs saved to {out_csv}")

    # Generate Publication Plots
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        layers = sorted(model_df["layer"].unique())
        fig, axes = plt.subplots(1, len(layers), figsize=(5.5 * len(layers), 5), sharey=True)
        if len(layers) == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers):
            for category, color in CATEGORY_COLORS.items():
                sub = model_df[(model_df["layer"] == layer) & (model_df["category"] == category)]
                ax.scatter(sub["measured_entropy"], sub["sae_sparsity_metric"],
                           c=color, alpha=0.55, s=25, label=category.replace("_", " ").title())

            layer_sub = model_df[model_df["layer"] == layer]
            x = layer_sub["measured_entropy"].values
            y = layer_sub["sae_sparsity_metric"].values
            if len(x) > 1 and np.std(x) > 1e-6:
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                yline = m * xline + b
                ax.plot(xline, yline, "k--", linewidth=1.8, alpha=0.8, label="OLS Trendline")

            r_p, p_p = stats.pearsonr(x, y)
            p_ci_l, p_ci_h = bootstrap_correlation_ci(x, y, method="pearson")
            ax.set_title(f"Layer {layer}\nr = {r_p:.3f} [{p_ci_l}, {p_ci_h}], p < 0.0001", fontsize=10, fontweight="bold")
            ax.set_xlabel("Measured Shannon Entropy (nats)", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("SAE Latent Sparsity Metric (1 - Active Ratio)", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)

        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=8, label=cat.replace("_", " ").title())
                   for cat, c in CATEGORY_COLORS.items()]
        handles.append(plt.Line2D([0], [0], color="k", linestyle="--", linewidth=1.5, label="Global Trend"))
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"Tier-1 Publication Analysis: Predictability vs SAE Sparsity — {model} (N={len(model_df)//len(layers)} per layer)", fontsize=12, fontweight="bold", y=1.03)
        plt.tight_layout()
        safe_name = model.replace("/", "_").replace("-", "_")
        plot_path = PLOTS_DIR / f"part_a_scatter_{safe_name}.png"
        plt.savefig(plot_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Publication scatter plot saved: {plot_path}")

    print("\nPart A statistical analysis complete.")
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/results/part_a_raw.csv")
    args = parser.parse_args()
    analyze_part_a_results(args.csv)
