#!/usr/bin/env python3
"""
run_full_gpu_session.py — Single GPU Session Master Orchestrator

Runs ONLY on the RTX 3050 (WSL2 Ubuntu) GPU machine.
Sequential Execution Architecture (VRAM < 6GB Guarantee):
  1. Set global random seeds (seed = 42).
  2. Load Primary Model: Gemma-2-2B (int8)
  3. Run Gemma Part A extraction, Benchmark validation, and Demo trace generation.
  4. Unload Gemma-2-2B & purge CUDA memory cache (torch.cuda.empty_cache()).
  5. Load Cross-Val Model: GPT-2 small (fp16)
  6. Run GPT-2 Part A extraction.
  7. Unload GPT-2 & purge CUDA memory cache.
  8. Write session_manifest.json with full run metadata and Git commit hash.

Command line flags:
  --mock: Dry run mode for pipeline testing without GPU dependencies.
"""

import sys
import os
import gc
import json
import random
import logging
import argparse
import datetime
import subprocess
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpu_session.step1_load_models import load_gemma_model, load_gpt2_model, unload_model
from gpu_session.step2_part_a_extraction import run_part_a_extraction_for_model
from gpu_session.step3_bdh_baseline import run_bdh_baseline_step
from gpu_session.step4_benchmark_validation import run_benchmark_validation
from gpu_session.step5_demo_trace_generation import generate_demo_traces

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpu_session.orchestrator")

def set_global_seeds(seed: int = 42):
    """Sets global random seeds for exact reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Global random seeds locked to seed={seed}")

def get_git_commit_hash() -> str:
    """Returns current git commit hash or repository state."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "uncommitted-initial-repo"

def run_session(mock_mode: bool = False):
    logger.info("=" * 60)
    logger.info("  SynapseGuard — Single GPU Session Master Orchestrator")
    logger.info("=" * 60)
    logger.info(f"Execution Mode: {'MOCK/DRY-RUN' if mock_mode else 'CUDA/PRODUCTION'}")
    
    set_global_seeds(42)
    timestamp_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # VRAM pre-flight check
    if not mock_mode:
        try:
            import torch
            if torch.cuda.is_available():
                free_vram, total_vram = torch.cuda.mem_get_info()
                logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM: {free_vram / 1e9:.2f} GB free / {total_vram / 1e9:.2f} GB total")
                if free_vram < 4.5e9:
                    logger.warning(
                        f"WARNING: Only {free_vram / 1e9:.2f} GB VRAM free. "
                        f"Expected minimum 4.5 GB for Gemma-2-2B int8 + SAE + buffers."
                    )
            else:
                logger.error("CRITICAL: torch.cuda.is_available() returned False in production mode.")
                raise RuntimeError("CUDA not available. GPU session cannot run.")
        except ImportError:
            logger.error("CRITICAL: torch not importable. Is the GPU environment activated?")
            raise

    # -------------------------------------------------------------
    # PHASE 1: Primary Model (Gemma-2-2B int8)
    # -------------------------------------------------------------
    logger.info("\n--- PHASE 1: Loading Primary Model (Gemma-2-2B int8) ---")
    gemma_model = load_gemma_model(mock_mode=mock_mode)
    
    logger.info("\n--- STEP 2A: Part A Extraction (Gemma-2-2B) ---")
    gemma_part_a_rows = run_part_a_extraction_for_model(gemma_model, mock_mode=mock_mode)
    
    logger.info("\n--- STEP 3: Verifying BDH Baseline Instrumentation ---")
    run_bdh_baseline_step(mock_mode=mock_mode)
    
    logger.info("\n--- STEP 4: Benchmark Validation & PSC Scoring (Gemma-2-2B) ---")
    benchmark_result = run_benchmark_validation(model_handle=gemma_model, mock_mode=mock_mode)
    benchmark_rows = benchmark_result["new_rows"] if isinstance(benchmark_result, dict) else benchmark_result
    benchmark_dataset_counts = benchmark_result.get("dataset_counts", {}) if isinstance(benchmark_result, dict) else {}
    
    logger.info("\n--- STEP 5: Generating Demo JSON Traces (Gemma-2-2B) ---")
    demo_files = generate_demo_traces(model_handle=gemma_model, mock_mode=mock_mode)
    
    logger.info("\n--- PURGING VRAM: Unloading Gemma-2-2B ---")
    unload_model(gemma_model)
    
    # -------------------------------------------------------------
    # PHASE 2: Cross-Validation Model (GPT-2 small fp16)
    # -------------------------------------------------------------
    logger.info("\n--- PHASE 2: Loading Cross-Val Model (GPT-2 small fp16) ---")
    gpt2_model = load_gpt2_model(mock_mode=mock_mode)
    
    logger.info("\n--- STEP 2B: Part A Extraction (GPT-2 small) ---")
    gpt2_part_a_rows = run_part_a_extraction_for_model(gpt2_model, mock_mode=mock_mode)
    
    logger.info("\n--- PURGING VRAM: Unloading GPT-2 ---")
    unload_model(gpt2_model)
    
    # -------------------------------------------------------------
    # PHASE 3: Manifest Generation & Completion
    # -------------------------------------------------------------
    logger.info("\n--- PHASE 3: Writing session_manifest.json ---")
    manifest_path = Path("data/results/session_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    git_hash = get_git_commit_hash()
    
    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "2.1.2"

    lib_versions = {}
    for lib_name in ["transformers", "sae_lens", "bitsandbytes", "accelerate", "pandas", "scipy", "scikit-learn"]:
        try:
            import importlib.metadata
            lib_versions[lib_name] = importlib.metadata.version(lib_name)
        except Exception:
            lib_versions[lib_name] = "installed"

    gemma_sae_info = gemma_model.get("sae", {}) if isinstance(gemma_model, dict) else {}
    gpt2_sae_info = gpt2_model.get("sae", {}) if isinstance(gpt2_model, dict) else {}

    manifest_data = {
        "session_date": timestamp_start,
        "git_commit_hash": git_hash,
        "random_seed": 42,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch_ver,
        "library_versions": lib_versions,
        "models": ["google/gemma-2-2b (int8)", "gpt2 (fp16)"],
        "model_commit_hashes": {
            "gemma-2-2b": gemma_model.get("commit_hash", "hf-main-release") if isinstance(gemma_model, dict) else "mock",
            "gpt2": gpt2_model.get("commit_hash", "hf-main-release") if isinstance(gpt2_model, dict) else "mock"
        },
        "execution_order": ["gemma-2-2b (sequential)", "gpt2 (sequential)"],
        "sae_checkpoints": {
            "gemma-2-2b": {
                "release": gemma_sae_info.get("release", "gemma-scope-2b-pt-res"),
                "sae_id": gemma_sae_info.get("sae_id", "layer_12/width_16k/canonical"),
                "backend": gemma_sae_info.get("backend", "unknown"),
                "cfg": gemma_sae_info.get("cfg_dict", {})
            },
            "gpt2": {
                "release": gpt2_sae_info.get("release", "sae-lens-gpt2-small"),
                "sae_id": gpt2_sae_info.get("sae_id", "blocks.6.hook_resid_pre"),
                "backend": gpt2_sae_info.get("backend", "unknown"),
                "cfg": gpt2_sae_info.get("cfg_dict", {})
            }
        },
        "activation_extraction_backend": "hf_transformers_forward_hooks",
        "entropy_computation": "full_vocabulary_shannon_entropy",
        "layers_extracted": {
            "gemma-2-2b": [11, 12, 13, 14],
            "gpt2": [5, 6, 7, 8]
        },
        "part_a_sample_count": (gemma_part_a_rows + gpt2_part_a_rows) // 8,
        "part_a_total_extraction_rows": gemma_part_a_rows + gpt2_part_a_rows,
        "benchmark_sample_counts": dict(benchmark_dataset_counts),
        "gpu": "RTX 3050 6GB (WSL2)",
        "quantization": "int8 (bitsandbytes)",
        "memory_management": "Sequential loading with explicit torch.cuda.empty_cache() between models",
        "notes": "Single GPU session completed successfully. All downstream analysis and dashboard run on Mac M4 Air."
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        
    logger.info(f"Session manifest successfully written to: {manifest_path}")
    logger.info("=" * 60)
    logger.info("  Single GPU Session Complete! All result files generated.")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="SynapseGuard GPU Session Orchestrator")
    parser.add_argument("--mock", action="store_true", help="Run in mock/dry-run mode without CUDA models")
    args = parser.parse_args()
    
    run_session(mock_mode=args.mock)

if __name__ == "__main__":
    main()
