"""
ssp.py — Sample-Specific Prompting (SSP) Dynamic Thresholding Module

Per architecture.md and rules.md:
  - Uses pre-computed perturbation templates (avoids real-time per-sample optimization latency).
  - Triggers only when PSC crosses warning/halt thresholds.
"""

from typing import Dict, Any, Optional
from part_b_diagnostic.sidecar.psc_score import WARNING_THRESHOLD, HALT_THRESHOLD

DEFAULT_PERTURBATION_TEMPLATES = {
    "WARNING": "Notice: Verify factual grounds before proceeding. Question: {prompt}",
    "HALT": "High Hallucination Risk Interception. Grounded context requirement applied: {prompt}"
}

def evaluate_ssp_trigger(
    psc_score: float,
    threshold: float = WARNING_THRESHOLD
) -> bool:
    """Determines whether SSP perturbation is triggered."""
    return psc_score >= threshold

def get_ssp_perturbation(
    prompt: str,
    status: str,
    custom_template: Optional[str] = None
) -> Dict[str, Any]:
    """
    Applies pre-computed perturbation template if SSP is triggered.
    """
    if status not in ["WARNING", "HALT"]:
        return {
            "ssp_applied": False,
            "original_prompt": prompt,
            "perturbed_prompt": prompt,
            "template_type": None
        }
        
    template = custom_template or DEFAULT_PERTURBATION_TEMPLATES.get(status, DEFAULT_PERTURBATION_TEMPLATES["WARNING"])
    perturbed = template.format(prompt=prompt)
    
    return {
        "ssp_applied": True,
        "original_prompt": prompt,
        "perturbed_prompt": perturbed,
        "template_type": status
    }
