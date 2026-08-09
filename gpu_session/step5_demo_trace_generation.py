"""
step5_demo_trace_generation.py — Demo Trace Generation Step with Real SAE-based PSC Scoring

Runs chosen demo prompts (mix of clean/factual and hallucination-prone prompts) token-by-token.
Records:
  - token text
  - logit confidence
  - PSC score (using real SAE latent sparsity)
  - SSP-triggered flag
  - status (SAFE / WARNING / HALT)

Output directory:
  data/results/demo_traces/
    - demo_01_factual.json
    - demo_02_hallucination_prone.json
    - demo_03_open_ended.json
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpu_session.sae_project import (
    run_sae_encode,
    compute_sae_sparsity,
    compute_next_token_entropy
)
from part_b_diagnostic.sidecar.psc_score import calculate_psc, WARNING_THRESHOLD, HALT_THRESHOLD
from part_b_diagnostic.sidecar.ssp import evaluate_ssp_trigger

logger = logging.getLogger("gpu_session.step5")

DEMO_PROMPTS = [
    {
        "filename": "demo_01_factual.json",
        "prompt": "What is the capital of France?",
        "mock_tokens": [
            ("The", 0.99, 0.05),
            (" capital", 0.98, 0.08),
            (" of", 0.99, 0.06),
            (" France", 0.99, 0.04),
            (" is", 0.99, 0.05),
            (" Paris", 0.99, 0.03),
            (".", 0.99, 0.02)
        ]
    },
    {
        "filename": "demo_02_hallucination_prone.json",
        "prompt": "Who was the first president of Mars in 1984?",
        "mock_tokens": [
            ("The", 0.95, 0.12),
            (" first", 0.91, 0.25),
            (" president", 0.88, 0.45),
            (" of", 0.89, 0.58),
            (" Mars", 0.94, 0.78),
            (" was", 0.92, 0.88)
        ]
    },
    {
        "filename": "demo_03_open_ended.json",
        "prompt": "Write a short poetic phrase about quantum computing.",
        "mock_tokens": [
            ("Superposition", 0.70, 0.20),
            (" dances", 0.65, 0.22),
            (" in", 0.85, 0.18),
            (" silence", 0.60, 0.25)
        ]
    }
]

def generate_demo_traces(
    model_handle: Dict[str, Any],
    output_dir: str = "data/results/demo_traces",
    mock_mode: bool = False
) -> List[str]:
    """Generates token-level JSON demo trace files for dashboard replay with real SAE PSC scoring."""
    logger.info(f"Starting Step 5: Demo Trace Generation. Output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    generated_files = []
    backend = model_handle.get("backend", "mock")
    sae_handle = model_handle.get("sae")
    
    for demo_spec in DEMO_PROMPTS:
        filename = demo_spec["filename"]
        prompt_text = demo_spec["prompt"]
        file_path = os.path.join(output_dir, filename)
        
        formatted_tokens = []
        
        if mock_mode or backend == "mock":
            # Mock token stream for dry-run verification
            for pos, (tok_str, conf, psc) in enumerate(demo_spec["mock_tokens"]):
                ssp_flag = evaluate_ssp_trigger(psc)
                status = "HALT" if psc >= HALT_THRESHOLD else ("WARNING" if psc >= WARNING_THRESHOLD else "SAFE")
                formatted_tokens.append({
                    "position": pos,
                    "token_text": tok_str,
                    "logit_confidence": conf,
                    "psc_score": psc,
                    "status": status,
                    "ssp_triggered": ssp_flag
                })
        else:
            # Real token-by-token generation via HF Transformers
            import torch
            model = model_handle["model"]
            tokenizer = model_handle["tokenizer"]
            sae_handle = model_handle.get("sae")

            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            max_new_tokens = 10

            with torch.no_grad():
                generated_ids = inputs["input_ids"].clone()
                for pos in range(max_new_tokens):
                    activations = {}
                    target_layers_list = model_handle.get("target_layers", [12])
                    primary_layer = target_layers_list[0]

                    def get_hook(module, inp, out):
                        activations["hidden"] = out[0] if isinstance(out, tuple) else out

                    if hasattr(model, "model") and hasattr(model.model, "layers"):
                        hook_h = model.model.layers[primary_layer].register_forward_hook(get_hook)
                    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                        hook_h = model.transformer.h[primary_layer].register_forward_hook(get_hook)
                    else:
                        hook_h = list(model.modules())[primary_layer].register_forward_hook(get_hook)

                    out = model(input_ids=generated_ids)
                    hook_h.remove()

                    logits_step = out.logits[0, -1, :] # [vocab_size]
                    probs_step = torch.softmax(logits_step, dim=-1)
                    conf = float(torch.max(probs_step).item())
                    next_token_id = int(torch.argmax(probs_step).item())
                    tok_text = tokenizer.decode([next_token_id])

                    # Real hidden state -> SAE -> sparsity
                    hidden = activations.get("hidden")
                    if hidden is None:
                        raise RuntimeError(f"Activation hook did not fire at layer {primary_layer}, pos {pos}")
                    last_hidden = hidden[0, -1, :]
                    sae_latents = run_sae_encode(last_hidden.unsqueeze(0), sae_handle)
                    sparsity = compute_sae_sparsity(sae_latents)
                    sae_entropy_val = 1.0 - sparsity["sae_sparsity_metric"]

                    psc = round(calculate_psc(logit_confidence=conf, sae_entropy=sae_entropy_val), 4)
                    ssp_flag = evaluate_ssp_trigger(psc)
                    status = "HALT" if psc >= HALT_THRESHOLD else ("WARNING" if psc >= WARNING_THRESHOLD else "SAFE")

                    formatted_tokens.append({
                        "position": pos,
                        "token_text": tok_text,
                        "logit_confidence": round(conf, 4),
                        "psc_score": psc,
                        "status": status,
                        "ssp_triggered": ssp_flag
                    })

                    generated_ids = torch.cat([
                        generated_ids,
                        torch.tensor([[next_token_id]], device=generated_ids.device)
                    ], dim=1)
                    if status == "HALT" or (hasattr(tokenizer, "eos_token_id") and next_token_id == tokenizer.eos_token_id):
                        break
                        
        trace_data = {
            "prompt": prompt_text,
            "model": model_handle.get("name", "gemma-2-2b-8bit"),
            "tokens": formatted_tokens,
            "generation_metadata": {
                "quantization": "int8",
                "layers_used": model_handle.get("target_layers", [11, 12, 13, 14]),
                "commit_hash": model_handle.get("commit_hash", "hf-main-release"),
                "session_timestamp": timestamp_str
            }
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)
            
        logger.info(f"Generated demo trace: {file_path} ({len(formatted_tokens)} tokens)")
        generated_files.append(file_path)
        
    return generated_files

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mock_m = {"name": "google/gemma-2-2b", "backend": "mock"}
    files = generate_demo_traces(model_handle=mock_m, mock_mode=True)
    print("Step 5 Dry Run Completed:", files)
