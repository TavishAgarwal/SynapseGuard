"""
step1_load_models.py — Loads Base Models & SAE Checkpoints via vLLM with HF Fallback

Runs on RTX 3050 (6GB VRAM, WSL2 Ubuntu).
Sequential Model Management:
  1. Primary Model: Gemma-2-2B (8-bit quantized via bitsandbytes) + Gemma Scope SAE
  2. Cross-Val Model: GPT-2 small (fp16) + SAE Lens GPT-2 SAE

Includes HuggingFace Transformers 8-bit forward-hook fallback if vLLM OOMs on 6GB VRAM,
and explicit unload/memory-purge helper functions.
"""

import os
import gc
import yaml
import logging
from typing import Dict, Any, Tuple, Optional, List

from gpu_session.sae_project import load_sae_model

logger = logging.getLogger("gpu_session.step1")

def load_configs(config_dir: str = "configs") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads model_config.yaml and device_config.yaml."""
    model_cfg_path = os.path.join(config_dir, "model_config.yaml")
    device_cfg_path = os.path.join(config_dir, "device_config.yaml")
    
    with open(model_cfg_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(device_cfg_path, "r") as f:
        device_config = yaml.safe_load(f)
        
    return model_config, device_config

def get_model_commit_hash(model_obj: Any, default_hash: str = "hf-main-release") -> str:
    """Attempts to retrieve the exact HuggingFace commit hash or revision of a loaded model."""
    try:
        if hasattr(model_obj, "config") and hasattr(model_obj.config, "_commit_hash") and model_obj.config._commit_hash:
            return model_obj.config._commit_hash
        if hasattr(model_obj, "llm_engine") and hasattr(model_obj.llm_engine, "model_config"):
            return getattr(model_obj.llm_engine.model_config, "revision", default_hash)
    except Exception:
        pass
    return default_hash

def load_gemma_model(config_dir: str = "configs", mock_mode: bool = False) -> Dict[str, Any]:
    """
    Loads primary Gemma-2-2B model (8-bit int8) via vLLM, falling back to HF Transformers if OOM.
    Also loads official Gemma Scope SAE checkpoint.
    """
    model_cfg, device_cfg = load_configs(config_dir)
    primary_spec = model_cfg['primary_model']
    model_name = primary_spec['name']
    sae_release = primary_spec.get('sae_release', 'gemma-scope-2b-pt-res')
    sae_id = primary_spec.get('sae_id', 'layer_12/width_16k/canonical')
    target_layers = primary_spec.get('target_layers', [11, 12, 13, 14])
    
    logger.info(f"Loading Gemma-2-2B ({model_name}) int8 & SAE ({sae_release})...")
    
    if mock_mode:
        logger.info("[MOCK MODE] Returning mock handle for Gemma-2-2B and Gemma Scope SAE.")
        sae_handle = load_sae_model(sae_release, sae_id, mock_mode=True)
        return {
            "name": model_name,
            "status": "mock_loaded",
            "backend": "mock",
            "target_layers": target_layers,
            "commit_hash": "mock-commit-hash-gemma2b",
            "sae": sae_handle
        }

    device = device_cfg.get("device", "cuda")
    sae_handle = load_sae_model(sae_release, sae_id, device=device, mock_mode=False)

    import torch
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        logger.info(f"Available VRAM before Gemma load: {free_b / 1e9:.2f} GB free / {total_b / 1e9:.2f} GB total")
    logger.info("Using HuggingFace Transformers int8 path with forward hooks for real activation extraction.")
    handle = load_hf_fallback_model(model_name, quantization="int8", target_layers=target_layers)
    handle["sae"] = sae_handle
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        logger.info(f"Available VRAM after Gemma load: {free_b / 1e9:.2f} GB free / {total_b / 1e9:.2f} GB total")
    return handle

def load_gpt2_model(config_dir: str = "configs", mock_mode: bool = False) -> Dict[str, Any]:
    """
    Loads cross-validation GPT-2 small model (fp16) via HuggingFace Transformers.
    Also loads SAE Lens GPT-2 SAE checkpoint.
    """
    model_cfg, device_cfg = load_configs(config_dir)
    cross_spec = model_cfg['cross_val_model']
    model_name = cross_spec['name']
    sae_release = cross_spec.get('sae_release', 'sae-lens-gpt2-small')
    sae_id = cross_spec.get('sae_id', 'blocks.6.hook_resid_pre')
    target_layers = cross_spec.get('target_layers', [5, 6, 7, 8])
    
    logger.info(f"Loading GPT-2 small ({model_name}) fp16 & SAE ({sae_release})...")
    
    if mock_mode:
        logger.info("[MOCK MODE] Returning mock handle for GPT-2.")
        sae_handle = load_sae_model(sae_release, sae_id, mock_mode=True)
        return {
            "name": model_name,
            "status": "mock_loaded",
            "backend": "mock",
            "target_layers": target_layers,
            "commit_hash": "mock-commit-hash-gpt2",
            "sae": sae_handle
        }

    device = device_cfg.get("device", "cuda")
    sae_handle = load_sae_model(sae_release, sae_id, device=device, mock_mode=False)

    import torch
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        logger.info(f"Available VRAM before GPT-2 load: {free_b / 1e9:.2f} GB free / {total_b / 1e9:.2f} GB total")
    logger.info("Using HuggingFace Transformers fp16 path with forward hooks for real activation extraction.")
    handle = load_hf_fallback_model(model_name, quantization="fp16", target_layers=target_layers)
    handle["sae"] = sae_handle
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        logger.info(f"Available VRAM after GPT-2 load: {free_b / 1e9:.2f} GB free / {total_b / 1e9:.2f} GB total")
    return handle

def load_hf_fallback_model(model_name: str, quantization: str = "int8", target_layers: List[int] = None) -> Dict[str, Any]:
    """
    Fallback loader using HuggingFace AutoModelForCausalLM with bitsandbytes 8-bit quantization.
    Attaches PyTorch forward hooks for activation extraction on 6GB VRAM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading HF AutoModelForCausalLM: {model_name} (quantization={quantization})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if quantization == "int8":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        target_device = "cuda:0" if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=target_device
        )
    else:
        target_device = "cuda:0" if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=target_device
        )
        
    model.eval()
    commit_hash = get_model_commit_hash(model)
    return {
        "name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "backend": "hf_fallback",
        "target_layers": target_layers or [11, 12, 13, 14],
        "commit_hash": commit_hash
    }

def unload_model(model_handle: Optional[Dict[str, Any]]):
    """
    Unloads model weights and SAE from RAM/VRAM and purges PyTorch CUDA cache.
    Prevents simultaneous memory consumption on 6GB VRAM.
    """
    if not model_handle:
        return
        
    logger.info(f"Unloading model {model_handle.get('name', 'unknown')} and clearing CUDA memory...")
    
    if "model" in model_handle:
        del model_handle["model"]
    if "tokenizer" in model_handle:
        del model_handle["tokenizer"]
    if "sae" in model_handle:
        del model_handle["sae"]
        
    del model_handle
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("CUDA memory cache purged successfully.")
    except Exception:
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gemma = load_gemma_model(mock_mode=True)
    print("Gemma Load Result:", gemma)
    unload_model(gemma)
