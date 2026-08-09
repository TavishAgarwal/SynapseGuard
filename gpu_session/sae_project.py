"""
sae_project.py — SAE Forward Pass & Sparsity/Entropy Metrics

Handles:
  1. SAE checkpoint loading via sae-lens or PyTorch tensors.
  2. SAE projection of model hidden states onto Sparse Autoencoder latent space.
  3. Sparsity/density metric computation (L0 norm, active latent fraction, sparsity score).
  4. Next-token Shannon entropy computation from model output logits.

Designed to run on CUDA during GPU session as well as CPU/MPS with synthetic tensors.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger("gpu_session.sae_project")

def load_sae_model(
    sae_release: str = "gemma-scope-2b-pt-res",
    sae_id: str = "layer_12/width_16k/canonical",
    device: str = "cuda",
    mock_mode: bool = False
) -> Dict[str, Any]:
    """
    Loads pretrained SAE checkpoint (Gemma Scope or SAE Lens).
    Returns an SAE handle dict containing the model instance or weight tensors.
    """
    if mock_mode:
        logger.info(f"[MOCK MODE] Returning mock SAE handle for {sae_release} ({sae_id}).")
        return {
            "release": sae_release,
            "sae_id": sae_id,
            "backend": "mock",
            "mock": True
        }

    try:
        from sae_lens import SAE
        logger.info(f"Loading pretrained SAE via sae_lens: release={sae_release}, sae_id={sae_id}...")
        res = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )
        if isinstance(res, tuple):
            sae = res[0]
            cfg_dict = res[1] if len(res) > 1 else {}
        else:
            sae = res
            cfg_dict = getattr(sae, "cfg", {})
            if hasattr(cfg_dict, "to_dict"):
                cfg_dict = cfg_dict.to_dict()
            elif not isinstance(cfg_dict, dict):
                cfg_dict = {"release": sae_release, "sae_id": sae_id}
                
        sae.eval()
        logger.info(f"SAE loaded successfully. Backend: sae_lens ({type(sae).__name__})")
        return {
            "release": sae_release,
            "sae_id": sae_id,
            "sae_instance": sae,
            "cfg_dict": cfg_dict,
            "backend": "sae_lens",
            "mock": False
        }
    except Exception as e:
        logger.error(
            f"FATAL: Failed to load SAE checkpoint via sae_lens. "
            f"Release='{sae_release}', SAE_ID='{sae_id}'. "
            f"Error: {e}. "
            f"The session cannot continue without a real SAE."
        )
        return {
            "release": sae_release,
            "sae_id": sae_id,
            "backend": "standalone",
            "mock": False
        }

def run_sae_encode(
    hidden_states: torch.Tensor,
    sae_handle: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Encodes dense model hidden states into high-dimensional SAE feature latents.
    Requires a real SAE checkpoint to be loaded. Raises RuntimeError if no valid
    SAE is available — never silently falls back to random projection.
    """
    if sae_handle is None:
        raise RuntimeError(
            "CRITICAL: run_sae_encode() called with sae_handle=None. "
            "No SAE checkpoint is loaded. Cannot produce valid sparsity metrics."
        )
    backend = sae_handle.get("backend", "unknown")
    if backend == "standalone":
        raise RuntimeError(
            f"CRITICAL: SAE backend is 'standalone' — sae_lens.SAE.from_pretrained() failed. "
            f"Release='{sae_handle.get('release')}', SAE_ID='{sae_handle.get('sae_id')}'. "
            f"Verify release name and internet connection. Aborting."
        )
    if backend == "mock":
        raise RuntimeError(
            "CRITICAL: run_sae_encode() called with mock SAE handle in production mode. Aborting."
        )

    if backend == "sae_lens" and "sae_instance" in sae_handle:
        sae = sae_handle["sae_instance"]
        if hasattr(sae, "encode"):
            return sae.encode(hidden_states)
        elif hasattr(sae, "W_enc"):
            b_enc = getattr(sae, "b_enc", None)
            return project_hidden_to_sae(hidden_states, sae.W_enc, b_enc)
        else:
            raise RuntimeError(
                f"CRITICAL: Loaded SAE object has neither .encode() nor .W_enc attribute. Type: {type(sae)}"
            )

    if "weight" in sae_handle:
        weight = sae_handle["weight"]
        bias = sae_handle.get("bias")
        return project_hidden_to_sae(hidden_states, weight, bias)

    raise RuntimeError(
        f"CRITICAL: sae_handle has backend='{backend}' but no usable SAE object or weights. Aborting."
    )

def project_hidden_to_sae(
    hidden_states: torch.Tensor,
    sae_weight: torch.Tensor,
    sae_bias: Optional[torch.Tensor] = None,
    activation_fn: str = "relu"
) -> torch.Tensor:
    """
    Projects dense model hidden states onto high-dimensional SAE feature space.
    
    Args:
        hidden_states: Tensor of shape [batch_size, seq_len, d_model] or [batch_size, d_model]
        sae_weight: Encoder weight tensor of shape [d_model, d_sae]
        sae_bias: Optional encoder bias tensor of shape [d_sae]
        activation_fn: Activation function ("relu" or "identity")
        
    Returns:
        sae_latents: Tensor of shape [..., d_sae]
    """
    latents = torch.matmul(hidden_states, sae_weight)
    if sae_bias is not None:
        latents = latents + sae_bias
        
    if activation_fn == "relu":
        latents = F.relu(latents)
    elif activation_fn == "identity":
        pass
    else:
        latents = F.relu(latents)
        
    return latents

def compute_sae_sparsity(
    sae_latents: torch.Tensor,
    threshold: float = 1e-5
) -> Dict[str, float]:
    """
    Computes L0 norm, active feature ratio, and normalized sparsity metric from SAE latents.
    
    Args:
        sae_latents: Tensor of shape [batch_size, d_sae] or [d_sae]
        threshold: Activation cutoff threshold
        
    Returns:
        dict containing:
            - l0_norm: Average count of active features per sample
            - active_ratio: Fraction of active features in [0, 1]
            - sae_sparsity_metric: Fraction of inactive features (1.0 - active_ratio)
    """
    if sae_latents.dim() == 1:
        sae_latents = sae_latents.unsqueeze(0)
        
    d_sae = sae_latents.size(-1)
    active_mask = (torch.abs(sae_latents) > threshold).float()
    
    l0_per_sample = torch.sum(active_mask, dim=-1)
    avg_l0 = float(torch.mean(l0_per_sample).item())
    active_ratio = float(avg_l0 / d_sae)
    sae_sparsity_metric = float(1.0 - active_ratio)
    
    return {
        "l0_norm": avg_l0,
        "active_ratio": active_ratio,
        "sae_sparsity_metric": sae_sparsity_metric
    }

def compute_next_token_entropy(logits: torch.Tensor) -> float:
    """
    Computes Shannon entropy (in nats) of the next-token probability distribution.
    
    Args:
        logits: Tensor of shape [vocab_size] or [batch_size, vocab_size]
        
    Returns:
        entropy: Average entropy value across batch
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    entropy_per_sample = -torch.sum(probs * log_probs, dim=-1)
    avg_entropy = float(torch.mean(entropy_per_sample).item())
    
    return avg_entropy
