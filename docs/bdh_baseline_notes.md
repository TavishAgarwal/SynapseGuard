# BDH Baseline Instrumentation Notes

> **Claim Classification: [MEASURED] / [ESTABLISHED]**

## 1. Overview
The Biological Dense-Sparse Hybrid (BDH) baseline instrumentation (`bdh.py`) is implemented locally as a ~10M parameter PyTorch reference model. It models the core biological hypothesis — that activation sparsity in disentangled feature representations correlates with input predictability — on Apple Silicon (MPS/CPU).

## 2. Model Architecture & Parameters
- **Model File:** [`bdh.py`](file:///Users/tavishagarwal/Desktop/SynapseGuard/bdh.py)
- **Parameter Count:** ~10.8M parameters
- **Sparse Projection Layer:** `SparseMonosemanticLayer` (dimension expansion: 128 → 2048 latents with top-$k$ thresholding, $k=64$).
- **Execution Machine:** Mac M4 Air (MPS device).

## 3. Empirical Baseline Measurements
- **Predictable / Low-Entropy Inputs:** Mean activation sparsity = **0.9688** (fraction of inactive latents).
- **Unpredictable / High-Entropy Inputs:** Mean activation sparsity = **0.9688** (top-$k$ fixed-ratio constrained baseline).
- **Plot Artifact:** [`analysis/plots/bdh_baseline_sparsity.png`](file:///Users/tavishagarwal/Desktop/SynapseGuard/analysis/plots/bdh_baseline_sparsity.png)

## 4. Mandatory Disclosures & Claim Boundaries
Per `rules.md` (Research Integrity Rules):
1. **Model Scale:** This model is a **~10M parameter educational/baseline instrumentation** only. It does not possess full biological integrate-and-fire dynamics or scalable synaptic plasticity present in frontier BDH architectures.
2. **Evidence Boundary:** This result serves solely as baseline/measured grounding for toy inputs. It **must not** be cited as evidence of frontier BDH scaling or multi-billion parameter model behavior.
3. **Replication Target:** The main experiment (Part A) tests whether SAE-decomposed standard transformers (`Gemma-2-2B` 8-bit and `GPT-2` fp16) replicate the predictability-sparsity correlation under real-world LLM activations during the single GPU session.
