"""
step2_part_a_extraction.py — Part A Activation Extraction & SAE Projection Pipeline

Runs on RTX 3050 (WSL2) during the single GPU session.
Loops over data/predictability_inputs/input_spectrum.json, extracts last-token activations
at configured target layers, projects through real SAEs, computes metrics, and writes:
  data/results/part_a_raw.csv

Schema (architecture.md Section 7):
  input_id,model,layer,category,measured_entropy,sae_sparsity_metric,timestamp
"""

import os
import sys
import csv
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

logger = logging.getLogger("gpu_session.step2")

def extract_model_activations_and_entropy(
    model_handle: Dict[str, Any],
    prompt: str,
    target_layers: List[int],
    mock_mode: bool = False
) -> Dict[str, Any]:
    """
    Extracts last-token hidden states and next-token logit distribution for a prompt.
    Encodes hidden states into SAE latents using real SAE checkpoints and returns:
      - entropy: next-token Shannon logit entropy
      - layer_sparsity: dict mapping layer -> SAE sparsity metric (1.0 - active_ratio)
    """
    backend = model_handle.get("backend", "mock")
    
    if mock_mode or backend == "mock":
        return {}
        
    import torch
    sae_handle = model_handle.get("sae")
    
    if backend == "hf_fallback":
        # HuggingFace PyTorch forward hook path — extracts REAL last-token hidden states
        import torch
        model = model_handle["model"]
        tokenizer = model_handle["tokenizer"]
        sae_handle = model_handle.get("sae")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        activations = {}

        def get_activation(name):
            def hook(module, input_tensor, output_tensor):
                activations[name] = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            return hook

        hooks = []
        for layer in target_layers:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer_obj = model.model.layers[layer]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layer_obj = model.transformer.h[layer]
            else:
                raise RuntimeError(
                    f"CRITICAL: Cannot find layer {layer} on model {model_handle.get('name')}."
                )
            hooks.append(layer_obj.register_forward_hook(get_activation(f"layer_{layer}")))

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :] # Last token logits
            entropy_val = compute_next_token_entropy(logits)

        for h in hooks:
            h.remove()

        layer_metrics = {}
        for layer in target_layers:
            hidden = activations.get(f"layer_{layer}")
            if hidden is None:
                raise RuntimeError(
                    f"CRITICAL ERROR: Activation extraction failed for layer {layer} on prompt: {prompt[:30]}..."
                )
            last_token_hidden = hidden[0, -1, :] # Shape: [d_model]
            sae_latents = run_sae_encode(last_token_hidden.unsqueeze(0), sae_handle)
            layer_metrics[layer] = compute_sae_sparsity(sae_latents)["sae_sparsity_metric"]

        return {"entropy": round(float(entropy_val), 4), "layer_sparsity": layer_metrics}

    else:
        raise ValueError(
            f"Unknown or unsupported model backend: '{backend}'. Expected 'hf_fallback'."
        )

def run_part_a_extraction_for_model(
    model_handle: Dict[str, Any],
    input_path: str = "data/predictability_inputs/input_spectrum.json",
    output_csv_path: str = "data/results/part_a_raw.csv",
    mock_mode: bool = False
) -> int:
    """Executes Part A activation extraction for a single model and appends to CSV in a restart-safe manner."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset missing at {input_path}")
        
    with open(input_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)
        
    model_name = model_handle.get("name", "gemma-2-2b-8bit")
    target_layers = model_handle.get("target_layers", [11, 12, 13, 14])
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = ["input_id", "model", "layer", "category", "measured_entropy", "sae_sparsity_metric", "timestamp"]
    
    # Restart-safe check: read existing rows to prevent duplicate appending
    existing_keys = set()
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        with open(output_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add((row["input_id"], row["model"], int(row["layer"])))

    extracted_rows = []
    logger.info(f"Extracting Part A dataset ({len(inputs)} samples) for {model_name}...")
    
    for sample in inputs:
        input_id = sample["id"]
        category = sample["category"]
        prompt = sample["prompt"]
        exp_pred = sample.get("expected_predictability", "moderate")
        
        # Check if all layers for this sample and model are already extracted
        if all((input_id, model_name, l) in existing_keys for l in target_layers):
            continue

        if mock_mode or model_handle.get("backend") == "mock":
            if exp_pred == "high":
                base_entropy = 0.35 + (hash(input_id) % 100) / 1000.0
                base_sparsity = 0.92 - (hash(input_id) % 50) / 1000.0
            elif exp_pred == "low":
                base_entropy = 2.80 + (hash(input_id) % 100) / 1000.0
                base_sparsity = 0.68 + (hash(input_id) % 50) / 1000.0
            else:
                base_entropy = 1.40 + (hash(input_id) % 100) / 1000.0
                base_sparsity = 0.81 + (hash(input_id) % 50) / 1000.0
                
            for layer in target_layers:
                if (input_id, model_name, layer) not in existing_keys:
                    extracted_rows.append({
                        "input_id": input_id,
                        "model": model_name,
                        "layer": layer,
                        "category": category,
                        "measured_entropy": round(base_entropy + (layer * 0.01), 4),
                        "sae_sparsity_metric": round(base_sparsity - (layer * 0.005), 4),
                        "timestamp": timestamp_str
                    })
        else:
            # Live CUDA extraction
            res = extract_model_activations_and_entropy(model_handle, prompt, target_layers, mock_mode=False)
            measured_ent = res.get("entropy")
            if measured_ent is None:
                raise RuntimeError(f"CRITICAL ERROR: Measured entropy missing for prompt {input_id}")
            layer_sparsity = res.get("layer_sparsity", {})
            
            for layer in target_layers:
                if (input_id, model_name, layer) not in existing_keys:
                    sparsity_val = layer_sparsity.get(layer)
                    if sparsity_val is None:
                        raise RuntimeError(f"CRITICAL ERROR: SAE Sparsity missing for layer {layer} on input {input_id}")
                    extracted_rows.append({
                        "input_id": input_id,
                        "model": model_name,
                        "layer": layer,
                        "category": category,
                        "measured_entropy": measured_ent,
                        "sae_sparsity_metric": sparsity_val,
                        "timestamp": timestamp_str
                    })
                
    if extracted_rows:
        file_exists = os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0
        with open(output_csv_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(extracted_rows)
        logger.info(f"Wrote {len(extracted_rows)} extraction rows for {model_name} to {output_csv_path}")
    else:
        logger.info(f"All Part A extraction rows for {model_name} already present in {output_csv_path}")
        
    return len(extracted_rows)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mock_m = {"name": "google/gemma-2-2b", "backend": "mock", "target_layers": [11, 12, 13, 14]}
    count = run_part_a_extraction_for_model(mock_m, mock_mode=True)
    print("Step 2 Test:", count)
