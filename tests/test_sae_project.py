"""
Unit Tests for sae_project.py and PSC Scoring Logic
Executes on Mac M4 Air using synthetic/dummy tensors without requiring a GPU or vLLM.
"""

import pytest
import torch
import math
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpu_session.sae_project import (
    project_hidden_to_sae,
    compute_sae_sparsity,
    compute_next_token_entropy
)
from part_b_diagnostic.sidecar.psc_score import calculate_psc
from part_b_diagnostic.sidecar.ssp import evaluate_ssp_trigger

def test_project_hidden_to_sae_dimensions():
    """Verify matrix multiplication shape transformation and ReLU non-negativity."""
    batch_size = 4
    d_model = 128
    d_sae = 1024
    
    hidden = torch.randn(batch_size, d_model)
    weight = torch.randn(d_model, d_sae)
    bias = torch.randn(d_sae)
    
    latents = project_hidden_to_sae(hidden, weight, bias, activation_fn="relu")
    
    assert latents.shape == (batch_size, d_sae)
    assert torch.all(latents >= 0.0), "ReLU output must be non-negative"

def test_compute_sae_sparsity_math():
    """Verify L0 count, active_ratio, and sae_sparsity_metric complement."""
    d_sae = 100
    latents = torch.zeros(1, d_sae)
    # Set exactly 20 features active above threshold
    latents[0, :20] = 2.5
    
    metrics = compute_sae_sparsity(latents, threshold=1e-5)
    
    assert metrics["l0_norm"] == 20.0
    assert math.isclose(metrics["active_ratio"], 0.20, abs_tol=1e-5)
    assert math.isclose(metrics["sae_sparsity_metric"], 0.80, abs_tol=1e-5)

def test_compute_next_token_entropy():
    """Verify entropy calculation for deterministic vs uniform logit distributions."""
    vocab_size = 1000
    
    # 1. Deterministic distribution (one token has massive logit): entropy ~ 0
    det_logits = torch.full((1, vocab_size), -100.0)
    det_logits[0, 42] = 100.0
    det_entropy = compute_next_token_entropy(det_logits)
    assert math.isclose(det_entropy, 0.0, abs_tol=1e-3)
    
    # 2. Uniform distribution: entropy = ln(vocab_size)
    uniform_logits = torch.ones(1, vocab_size)
    uni_entropy = compute_next_token_entropy(uniform_logits)
    expected_entropy = math.log(vocab_size)
    assert math.isclose(uni_entropy, expected_entropy, abs_tol=1e-3)

def test_psc_score_calculation():
    """Verify PSC score formula and warning band threshold triggers."""
    # High confidence (0.95), low entropy (0.1) -> Safe low PSC
    psc_safe = calculate_psc(logit_confidence=0.95, sae_entropy=0.1, alpha=0.5, beta=0.5)
    assert psc_safe < 0.65
    assert evaluate_ssp_trigger(psc_safe) is False
    
    # Low confidence (0.20), high entropy (0.90) -> High risk/warning PSC mismatch
    psc_risk = calculate_psc(logit_confidence=0.20, sae_entropy=0.90, alpha=0.5, beta=0.5)
    assert psc_risk >= 0.65
    assert evaluate_ssp_trigger(psc_risk) is True
