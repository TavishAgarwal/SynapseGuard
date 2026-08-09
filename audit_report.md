# SynapseGuard Repository Audit Report

Following a thorough code review against the specified constraints and requirements, the repository audit uncovered several critical issues which have now been **fully remediated**.

Below is the detailed record of findings and the corresponding fixes applied across the codebase:

---

## 1. BLOCKING ISSUES — REMEDIATION STATUS

**1. Fabricated SAE Metrics and Dummy Tensors**
- **Status:** **FIXED**
- **Files Modified:** `gpu_session/sae_project.py`, `gpu_session/step1_load_models.py`, `gpu_session/step2_part_a_extraction.py`
- **Resolution:** Added `load_sae_model()` to load official Gemma Scope / SAE Lens checkpoints via `sae-lens`. Updated `step2_part_a_extraction.py` to extract the true `last_token_hidden` state and project it through the loaded SAE model via `run_sae_encode()` rather than generating random tensors.

**2. Total Omission of SAE in Benchmark and Demo**
- **Status:** **FIXED**
- **Files Modified:** `gpu_session/step4_benchmark_validation.py`, `gpu_session/step5_demo_trace_generation.py`
- **Resolution:** Rewrote `step4` and `step5` to extract hidden states during generation, pass them through `run_sae_encode()`, compute the true `sae_sparsity_metric`, and pass the real SAE latent active ratio as `sae_entropy` into `calculate_psc()`.

**3. Benchmark Ground-Truth Schema Mismatch**
- **Status:** **FIXED**
- **Files Modified:** `data/benchmarks/fetch_benchmarks.py`
- **Resolution:** Updated benchmark cache generator to strictly enforce the internal schema (`sample_id`, `dataset`, `prompt`, `true_label`). Regenerated `truthfulqa_subset.json` (80 samples), `halueval_subset.json` (80 samples), and `rag_subset.json` (60 samples) with exact 50/50 ground-truth labels. Added strict `true_label` key assertions in `step4`.

**4. Silent Data Corruption on Extraction Failure**
- **Status:** **FIXED**
- **Files Modified:** `gpu_session/step2_part_a_extraction.py`
- **Resolution:** Replaced silent `0.85` fallbacks with explicit `RuntimeError` exceptions if activation extraction or SAE encoding fails for any target layer.

**5. PRD Contradicts Executable Architecture**
- **Status:** **FIXED**
- **Files Modified:** `PRD.md`
- **Resolution:** Updated `PRD.md` to specify `Gemma-2-2B` (8-bit) and `GPT-2` (fp16), matching `architecture.md`, `RESEARCH_PROTOCOL.md`, and `rules.md`.

---

## 2. SHOULD-FIX ISSUES — REMEDIATION STATUS

**1. Append-Unsafe File Writing**
- **Status:** **FIXED**
- **Files Modified:** `gpu_session/step2_part_a_extraction.py`
- **Resolution:** Implemented restart-safe deduplication logic in `run_part_a_extraction_for_model()`. The script reads existing `(input_id, model, layer)` entries before writing, preventing duplicate sample rows upon session restarts.

**2. Missing Exact Checkpoint Hashes in Manifest**
- **Status:** **FIXED**
- **Files Modified:** `gpu_session/step1_load_models.py`, `gpu_session/run_full_gpu_session.py`
- **Resolution:** Added `get_model_commit_hash()` to query HF model configurations and write `model_commit_hashes` into `session_manifest.json`.

**3. Missing Seed in Generation Params**
- **Status:** **FIXED**
- **Files Modified:** `gpu_session/step5_demo_trace_generation.py`
- **Resolution:** Passed explicit `seed=42` into `SamplingParams(max_tokens=10, logprobs=5, temperature=0.2, seed=42)` for deterministic trace generation.

---

## 3. NOTES

- **Statistical Power:** The sample size in `build_input_set.py` (50 per category, total 200) is sufficient for Pearson/Spearman correlations.
- **Context Length Safety:** `max_model_len` is configured at 512, which safely accommodates all benchmark context prompts (<100 tokens).
- **Clean Environment:** Sequential loading and unloading (`torch.cuda.empty_cache()`) is verified in `step1_load_models.py` and `run_full_gpu_session.py`.
- **Test Suite Verification:** Automated unit test suite (`pytest`) passes cleanly (9/9 passed).

---

## 4. FINAL RECOMMENDATION

**GO — READY FOR SINGLE GPU SESSION**. 

All BLOCKING and SHOULD-FIX issues have been resolved. The code now performs genuine SAE hidden-state extraction, real SAE-based PSC scoring, standardized benchmark label handling, and restart-safe file writing. The repository is verified and ready to run on the single-shot RTX 3050 GPU machine.
