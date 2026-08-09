"""
Part B AUROC Computation & Publication Analysis (Tier-1 Research Grade).
Consumes data/results/benchmark_scores.csv, computes AUROC curves with 95% Bootstrap CIs,
and produces high-resolution ROC plots into analysis/plots/.

Usage:
    python analysis/benchmark_auroc.py
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from collections import Counter

PLOTS_DIR = Path("analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def bootstrap_auroc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot=1000, ci=95, seed=42):
    """Computes 95% non-parametric bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aurocs = []
    
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        yt_b = y_true[indices]
        ys_b = y_score[indices]
        if len(np.unique(yt_b)) < 2:
            continue
        try:
            auc = roc_auc_score(yt_b, ys_b)
            boot_aurocs.append(auc)
        except Exception:
            pass
            
    if len(boot_aurocs) == 0:
        return np.nan, np.nan
        
    lower = np.percentile(boot_aurocs, (100 - ci) / 2.0)
    upper = np.percentile(boot_aurocs, 100 - (100 - ci) / 2.0)
    return round(float(lower), 4), round(float(upper), 4)

def compute_benchmark_auroc(scores_csv_path: str = "data/results/benchmark_scores.csv"):
    """Computes AUROC metrics with 95% bootstrap CIs from benchmark scores CSV and generates publication ROC plots."""
    if not os.path.exists(scores_csv_path):
        raise FileNotFoundError(f"Benchmark scores CSV not found at {scores_csv_path}.")

    df = pd.read_csv(scores_csv_path)
    print(f"Loaded {len(df)} rows from {scores_csv_path}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Label distribution: {dict(Counter(df['true_label'].tolist()))}")
    print()

    results = []
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].copy()
        label_counts = Counter(sub["true_label"].tolist())

        if len(label_counts) < 2:
            print(f"  SKIP {dataset}: only one class present {dict(label_counts)}. Cannot compute AUROC.")
            ax.text(0.5, 0.5, f"{dataset}\nSingle class only\n(AUROC undefined)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="red")
            results.append({"dataset": dataset, "n": len(sub), "auroc": None,
                            "auroc_ci95_low": None, "auroc_ci95_high": None,
                            "note": "single_class_input", "label_dist": str(dict(label_counts))})
            continue

        # Convention: true_label=0 means hallucination, true_label=1 means factual
        # Higher psc_score predicts label=0 (hallucination)
        y_true = (sub["true_label"] == 0).astype(int).values
        y_score = sub["psc_score"].values

        try:
            auroc = float(roc_auc_score(y_true, y_score))
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            ci_low, ci_high = bootstrap_auroc_ci(y_true, y_score)
        except Exception as e:
            print(f"  ERROR computing AUROC for {dataset}: {e}")
            results.append({"dataset": dataset, "n": len(sub), "auroc": None, "note": str(e)})
            continue

        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_threshold = thresholds[opt_idx]
        y_pred = (y_score >= opt_threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        result = {
            "dataset": dataset,
            "n": len(sub),
            "auroc": round(auroc, 4),
            "auroc_ci95_low": ci_low,
            "auroc_ci95_high": ci_high,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "optimal_threshold": round(float(opt_threshold), 4),
            "n_hallucinated": int(y_true.sum()),
            "n_factual": int((y_true == 0).sum()),
            "note": "ok"
        }
        results.append(result)

        print(f"[{dataset}] n={len(sub)} | AUROC={auroc:.4f} [95% CI: {ci_low}, {ci_high}] | Optimal thresh={opt_threshold:.4f} | F1={f1:.4f}")

        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"AUROC = {auroc:.3f} [{ci_low}, {ci_high}]")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, label="Random Classifier (0.50)")
        ax.scatter([fpr[opt_idx]], [tpr[opt_idx]], c="red", s=90, zorder=5, label=f"Opt. Thresh = {opt_threshold:.2f}")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        ax.set_title(f"{dataset.replace('_', ' ').title()}\n(N={len(sub)}, AUROC={auroc:.4f})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Tier-1 Publication Analysis: PSC Score ROC Curves (Hallucination Detection)", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "benchmark_auroc.png"
    plt.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"\nPublication ROC plot saved: {plot_path}")

    results_df = pd.DataFrame(results)
    out_csv = PLOTS_DIR / "benchmark_auroc_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Publication AUROC results table saved: {out_csv}")
    print("\nPart B analysis complete.")
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/results/benchmark_scores.csv")
    args = parser.parse_args()
    compute_benchmark_auroc(args.csv)
