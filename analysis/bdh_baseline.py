#!/usr/bin/env python3
"""
SynapseGuard — BDH Baseline Sparsity Analysis & Visualization

Instruments the local bdh.py toy model (~10M parameters) on CPU/MPS.
Measures sparsity behavior across simulated low-entropy (predictable) vs high-entropy (unpredictable)
sequences, and generates analysis/plots/bdh_baseline_sparsity.png.
"""

import os
import sys
import numpy as np
import torch

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bdh import BDHModel
import matplotlib.pyplot as plt

def run_bdh_baseline_experiment():
    print("Running BDH Baseline Sparsity Experiment...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Execution device for bdh.py: {device}")
    
    model = BDHModel(vocab_size=1000, d_model=128, d_sparse=2048, k_sparsity=64).to(device)
    model.eval()
    
    # 1. Generate Predictable (Repeated/Low-entropy) sequences vs Unpredictable (Random/High-entropy) sequences
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Predictable inputs: repeated pattern sequences
    predictable_seqs = torch.tensor([[10, 20, 30, 40] * 8 for _ in range(25)], device=device)
    # Unpredictable inputs: uniform random token sequences
    unpredictable_seqs = torch.randint(0, 1000, (25, 32), device=device)
    
    with torch.no_grad():
        _, pred_metrics, pred_latents = model(predictable_seqs)
        _, unpred_metrics, unpred_latents = model(unpredictable_seqs)
        
    # Extract L0 norm per token across batch
    pred_l0 = torch.sum((torch.abs(pred_latents) > 1e-5).float(), dim=-1).cpu().numpy().flatten()
    unpred_l0 = torch.sum((torch.abs(unpred_latents) > 1e-5).float(), dim=-1).cpu().numpy().flatten()
    
    # Normalize sparsity metric: fraction of dormant (zero) neurons
    pred_sparsity = 1.0 - (pred_l0 / 2048.0)
    unpred_sparsity = 1.0 - (unpred_l0 / 2048.0)
    
    mean_pred_sp = np.mean(pred_sparsity)
    mean_unpred_sp = np.mean(unpred_sparsity)
    
    print(f"Predictable Sequences Mean Sparsity: {mean_pred_sp:.4f}")
    print(f"Unpredictable Sequences Mean Sparsity: {mean_unpred_sp:.4f}")
    
    # Plot results
    os.makedirs("analysis/plots", exist_ok=True)
    plot_path = "analysis/plots/bdh_baseline_sparsity.png"
    
    plt.figure(figsize=(9, 5))
    plt.hist(pred_sparsity, bins=20, alpha=0.7, label="Predictable / Low-Entropy Inputs", color="#2ca02c")
    plt.hist(unpred_sparsity, bins=20, alpha=0.7, label="Unpredictable / High-Entropy Inputs", color="#d62728")
    plt.axvline(mean_pred_sp, color="#1b641b", linestyle="dashed", linewidth=2, label=f"Mean Pred ({mean_pred_sp:.3f})")
    plt.axvline(mean_unpred_sp, color="#8b1a1a", linestyle="dashed", linewidth=2, label=f"Mean Unpred ({mean_unpred_sp:.3f})")
    
    plt.title("BDH Baseline Sparsity Distribution (Toy ~10M Parameter Model)", fontsize=13, fontweight="bold")
    plt.xlabel("Latent Activation Sparsity (Fraction of Inactive Neurons)", fontsize=11)
    plt.ylabel("Frequency (Token Counts)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    return mean_pred_sp, mean_unpred_sp

if __name__ == "__main__":
    run_bdh_baseline_experiment()
