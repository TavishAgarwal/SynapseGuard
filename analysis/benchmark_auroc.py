"""
Part B AUROC Computation (Runs on Mac).
Consumes data/results/benchmark_scores.csv, computes AUROC curves per dataset,
and produces classification report plots into analysis/plots/.

Usage:
    python analysis/benchmark_auroc.py
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from collections import Counter

PLOTS_DIR = Path("analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_benchmark_auroc(scores_csv_path: str = "data/results/benchmark_scores.csv"):
    """Computes AUROC metrics from benchmark scores CSV and generates ROC plots."""
    if not os.path.exists(scores_csv_path):
        raise FileNotFoundError(f"Benchmark scores CSV not found at {scores_csv_path}.")

    df = pd.read_csv(scores_csv_path)
    print(f"Loaded {len(df)} rows from {scores_csv_path}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Label distribution: {dict(Counter(df['true_label'].tolist()))}")
    print()

    results = []
    fig, axes = plt.subplots(1, len(df["dataset"].unique()), figsize=(5 * len(df["dataset"].unique()), 5))
    if len(df["dataset"].unique()) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].copy()
        label_counts = Counter(sub["true_label"].tolist())

        if len(label_counts) < 2:
            print(f"  SKIP {dataset}: only one class present {dict(label_counts)}. Cannot compute AUROC.")
            ax.text(0.5, 0.5, f"{dataset}\nSingle class only\n(AUROC undefined)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="red")
            results.append({"dataset": dataset, "n": len(sub), "auroc": None,
                            "note": "single_class_input", "label_dist": str(dict(label_counts))})
            continue

        # PSC score: higher = positive class (hallucination=0)
        # Our convention: true_label=0 means hallucination, true_label=1 means factual
        # We want higher psc_score to predict label=0 (hallucination)
        y_true = (sub["true_label"] == 0).astype(int).values
        y_score = sub["psc_score"].values

        try:
            auroc = float(roc_auc_score(y_true, y_score))
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
        except Exception as e:
            print(f"  ERROR computing AUROC for {dataset}: {e}")
            results.append({"dataset": dataset, "n": len(sub), "auroc": None, "note": str(e)})
            continue

        result = {
            "dataset": dataset, "n": len(sub), "auroc": round(auroc, 4),
            "n_hallucinated": int(y_true.sum()), "n_factual": int((y_true == 0).sum()),
            "note": "ok"
        }
        results.append(result)

        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_threshold = thresholds[opt_idx]
        y_pred = (y_score >= opt_threshold).astype(int)
        print(f"[{dataset}] n={len(sub)} | AUROC={auroc:.4f} | Optimal threshold={opt_threshold:.4f}")
        print(classification_report(y_true, y_pred, target_names=["Factual", "Hallucinated"], zero_division=0))

        ax.plot(fpr, tpr, linewidth=2, label=f"AUROC={auroc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.scatter([fpr[opt_idx]], [tpr[opt_idx]], c="red", s=80, zorder=5, label=f"Opt. thresh={opt_threshold:.2f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{dataset}\nAUROC = {auroc:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.suptitle("Part B: PSC Score ROC Curves (Hallucination Detection)", fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "benchmark_auroc.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nROC plot saved: {plot_path}")

    results_df = pd.DataFrame(results)
    out_csv = PLOTS_DIR / "benchmark_auroc_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"AUROC results saved: {out_csv}")
    print("\nPart B analysis complete.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/results/benchmark_scores.csv")
    args = parser.parse_args()
    compute_benchmark_auroc(args.csv)
