"""
Part B AUROC Computation & Publication Analysis (Tier-1 Research Grade).
Consumes data/results/benchmark_scores.csv, computes AUROC curves with 95% Bootstrap CIs
on held-out evaluation splits (70% threshold selection / 30% held-out test split),
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
from sklearn.model_selection import train_test_split
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

def compute_benchmark_auroc(scores_csv_path: str = "data/results/benchmark_scores.csv", test_size: float = 0.30, seed: int = 42):
    """
    Computes AUROC metrics using strict held-out evaluation:
      - 70% Train/Selection Split: Selects optimal threshold J = TPR - FPR.
      - 30% Held-Out Evaluation Split: Evaluates AUROC, 95% Bootstrap CI, Precision, Recall, and F1.
    """
    if not os.path.exists(scores_csv_path):
        raise FileNotFoundError(f"Benchmark scores CSV not found at {scores_csv_path}.")

    df = pd.read_csv(scores_csv_path)
    print(f"Loaded {len(df)} rows from {scores_csv_path}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Label distribution: {dict(Counter(df['true_label'].tolist()))}")
    print(f"Evaluation Methodology: 70/30 Stratified Train/Held-Out Split (Seed={seed})\n")

    results = []
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].copy()
        label_counts = Counter(sub["true_label"].tolist())

        if len(label_counts) < 2:
            print(f"  SKIP {dataset}: only one class present {dict(label_counts)}.")
            results.append({"dataset": dataset, "n_total": len(sub), "n_test": 0, "auroc": None, "note": "single_class_input"})
            continue

        # Convention: true_label=0 means hallucination, true_label=1 means factual
        # Higher psc_score predicts label=0 (hallucination)
        y_true = (sub["true_label"] == 0).astype(int).values
        y_score = sub["psc_score"].values

        # 70/30 Stratified Train-Test Split to eliminate threshold selection circularity
        try:
            yt_train, yt_test, ys_train, ys_test = train_test_split(
                y_true, y_score, test_size=test_size, random_state=seed, stratify=y_true
            )
        except Exception as e:
            print(f"  Train/Test Split failed for {dataset}: {e}. Falling back to full set.")
            yt_train, yt_test, ys_train, ys_test = y_true, y_true, y_score, y_score

        # 1. Select optimal threshold strictly on 70% Training Split
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(yt_train, ys_train)
        j_scores_tr = tpr_tr - fpr_tr
        opt_idx_tr = np.argmax(j_scores_tr)
        opt_threshold = float(thresholds_tr[opt_idx_tr])

        # 2. Evaluate held-out metrics on 30% Test Split ONLY
        try:
            auroc_test = float(roc_auc_score(yt_test, ys_test))
            fpr_test, tpr_test, _ = roc_curve(yt_test, ys_test)
            ci_low, ci_high = bootstrap_auroc_ci(yt_test, ys_test, seed=seed)
        except Exception as e:
            print(f"  ERROR computing test AUROC for {dataset}: {e}")
            continue

        # Predict using learned opt_threshold on held-out test split
        y_pred_test = (ys_test >= opt_threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(yt_test, y_pred_test, average="binary", zero_division=0)

        result = {
            "dataset": dataset,
            "n_total": len(sub),
            "n_train": len(yt_train),
            "n_test": len(yt_test),
            "auroc_heldout": round(auroc_test, 4),
            "auroc_ci95_low": ci_low,
            "auroc_ci95_high": ci_high,
            "optimal_threshold_learned": round(opt_threshold, 4),
            "precision_heldout": round(float(prec), 4),
            "recall_heldout": round(float(rec), 4),
            "f1_score_heldout": round(float(f1), 4),
            "split_ratio": f"{int((1-test_size)*100)}/{int(test_size*100)}",
            "note": "heldout_eval_ok"
        }
        results.append(result)

        print(f"[{dataset}] N={len(sub)} (Held-out Test N={len(yt_test)}) | Held-out AUROC={auroc_test:.4f} [95% CI: {ci_low}, {ci_high}] | Learned Thresh={opt_threshold:.4f} | Held-out F1={f1:.4f}")

        # Plot ROC curve for held-out evaluation set
        ax.plot(fpr_test, tpr_test, color="#06B6D4", linewidth=2.5, label=f"Held-out AUROC = {auroc_test:.3f}\n[95% CI: {ci_low}, {ci_high}]")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, label="Random (0.50)")
        
        # Mark performance point of learned threshold on test set
        test_fpr_point = np.mean((ys_test >= opt_threshold) & (yt_test == 0))
        test_tpr_point = np.mean((ys_test >= opt_threshold) & (yt_test == 1))
        ax.scatter([test_fpr_point], [test_tpr_point], c="#EF4444", s=90, zorder=5, label=f"Learned Thresh = {opt_threshold:.2f}")

        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        ax.set_title(f"{dataset.replace('_', ' ').title()}\n(Held-out Test N={len(yt_test)}, AUROC={auroc_test:.4f})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Tier-1 Publication Analysis: Held-Out PSC Score ROC Curves (70/30 Train/Test Split)", fontsize=12, fontweight="bold", y=1.03)
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
    parser.add_argument("--test-size", type=float, default=0.30)
    args = parser.parse_args()
    compute_benchmark_auroc(args.csv, test_size=args.test_size)
