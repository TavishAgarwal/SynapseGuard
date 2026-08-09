"""
psc_score.py — Predictability-Sparsity Coherence (PSC) Score Engine

Canonical PSC scoring module shared across gpu_session/ and part_b_diagnostic/sidecar/.

Formula:
  PSC_Score = alpha * (1.0 - logit_confidence) + beta * sae_entropy

Threshold Bands:
  - SAFE:    PSC < 0.65
  - WARNING: 0.65 <= PSC < 0.85
  - HALT:    PSC >= 0.85
"""

from typing import Dict, Any, Tuple

WARNING_THRESHOLD = 0.65
HALT_THRESHOLD = 0.85

def calculate_psc(
    logit_confidence: float,
    sae_entropy: float,
    alpha: float = 0.5,
    beta: float = 0.5
) -> float:
    """
    Computes Predictability-Sparsity Coherence (PSC) score.
    
    Args:
        logit_confidence: Model output probability for top-1 token in [0.0, 1.0]
        sae_entropy: Entropy/sparsity score over SAE latents in [0.0, 1.0]
        alpha: Weight for logit confidence term
        beta: Weight for SAE latent entropy term
        
    Returns:
        psc_score: Normalized PSC score
    """
    confidence_deficit = 1.0 - max(0.0, min(1.0, logit_confidence))
    bounded_entropy = max(0.0, min(1.0, sae_entropy))
    
    score = (alpha * confidence_deficit) + (beta * bounded_entropy)
    return round(float(score), 4)

def classify_psc_status(psc_score: float) -> str:
    """
    Classifies PSC score into diagnostic status band: SAFE, WARNING, or HALT.
    """
    if psc_score >= HALT_THRESHOLD:
        return "HALT"
    elif psc_score >= WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "SAFE"

def evaluate_token_diagnostic(
    token_text: str,
    logit_confidence: float,
    sae_entropy: float,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates complete diagnostic metrics for a single generated token.
    """
    psc = calculate_psc(logit_confidence, sae_entropy, alpha, beta)
    status = classify_psc_status(psc)
    ssp_triggered = psc >= WARNING_THRESHOLD
    
    return {
        "token_text": token_text,
        "logit_confidence": logit_confidence,
        "sae_entropy": sae_entropy,
        "psc_score": psc,
        "status": status,
        "ssp_triggered": ssp_triggered
    }
