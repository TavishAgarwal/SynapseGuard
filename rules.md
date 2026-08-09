# Rules.md — What to Use and What to Avoid

## Research Integrity Rules (non-negotiable)
1. **Never claim `bdh.py` demonstrates full BDH capabilities.** It is a ~10M parameter educational implementation without confirmed integrate-and-fire dynamics or full synaptic plasticity. Always label it "toy/baseline instrumentation."
2. **Every claim in documentation must be tagged** ESTABLISHED, MEASURED, or EXPLORATORY. No untagged claims in the README's technical sections.
3. **Report null or partial results from Part A honestly.** Do not omit, downplay, or reframe a negative correlation result to fit the narrative. A null result is a valid, reportable outcome.
4. **Do not fabricate benchmark numbers.** Every AUROC, correlation coefficient, or accuracy figure must come from an actual run with logged outputs (raw result files must exist in `data/results/`).
5. **Do not silently substitute hardcoded values for computed ones** anywhere in the pipeline (a known past failure mode on other projects — treat this as a hard rule).
6. **Cite the specific BDH result being tested** (predictability–sparsity relationship) rather than general "monosemanticity" language, to keep the BDH-relevance claim precise and defensible.

## Technical Rules

### Use
- vLLM-Hook (IBM) for activation extraction — confirmed real, actively maintained.
- Off-the-shelf pre-trained SAEs matched to the exact base model checkpoint (mismatched SAE/model pairs invalidate results).
- `last_token` extraction mode for latency-sensitive Part B; full-sequence extraction is acceptable for offline Part A analysis if needed.
- Synthetic or public benchmark data only (TruthfulQA, HaluEval, RAGTruth or similar) — no scraped or private data requiring de-identification.
- Async/non-blocking transport (shared memory or async RPC) between vLLM and the FastAPI sidecar in Part B.

### Avoid
- Synchronous full-tensor RPC calls between the inference engine and sidecar (kills throughput — confirmed limitation in the original naive design).
- Training an SAE from scratch as the v1 default (too time-costly; only attempt as a stretch goal with time remaining).
- Optimizing SSP perturbations uniquely per-sample in real time — use pre-computed perturbation templates only, to avoid trivializing the metric and adding latency.
- Over-claiming cross-model generalization from a single base model — always attempt at least two models before claiming generality.
- Using closed/API-only models (GPT-4, Claude API, etc.) for the core experiment — hidden-state access requires open weights.
- Committing large raw activation tensors or model checkpoints to git — use `.gitignore` and document how to regenerate them.
- Adding unnecessary features/scope beyond what's needed to support the central claim (per the hackathon's "depth beats feature count" guidance).

## Documentation Rules
- Every major README claim must map to a file/script/result that a judge can actually run or inspect.
- Include exact commit hash and configuration if `bdh.py` is used.
- Include exact SAE checkpoint source/version and base model version in the "Technology or Research Anchor" section.
- State limitations explicitly — this increases trust and is directly rewarded in judging.

## Team Process Rules
- Keep `data/results/` populated with raw run outputs (not just final plots) so any claim can be re-verified.
- Any change to core metrics (PSC formula, sparsity definition) must be logged with a reason in a changelog, since consistency of the metric across Part A and Part B is central to the project's validity.

## Hardware Compatibility Rules

### Use
- 8-bit quantization (bitsandbytes) as the default for Gemma-2-2B on the 
  RTX 3050 — this is a VRAM necessity, not a stretch preference.
- `enforce_eager=True` in vLLM initialization on the RTX 3050 to avoid 
  CUDA graph compilation overhead consuming extra VRAM headroom.
- Conservative vLLM memory settings tuned for 6GB: 
  `gpu_memory_utilization=0.80–0.85`, `max_model_len=512` (sufficient for 
  short completion-style prompts; do not default to vLLM's larger context 
  presets).
- WSL2 + Ubuntu for all vLLM/CUDA work on the Windows machine — do not 
  attempt native Windows vLLM installation.
- GPT-2 small/medium as the cross-validation model specifically because it 
  needs no quantization and leaves large VRAM headroom, reducing risk of 
  OOM errors burning hackathon time.

### Avoid
- Any model above ~2–3B parameters even quantized, given 6GB VRAM — this 
  will not leave room for SAE latent expansion + KV cache simultaneously.
- Running vLLM and any other GPU-heavy process (e.g., a second model, 
  training a browser-based tool) simultaneously on the RTX 3050 — VRAM has 
  no slack for concurrent GPU workloads.
- fp16-only Gemma-2-2B on this card without quantization — will likely OOM 
  once SAE and vLLM's reserved KV cache are added.
- Writing any script that hardcodes `device="mps"` or `device="cuda"` — 
  always read from `device_config.yaml` so the repo stays portable.
- Testing the live demo's cross-machine connection for the first time 
  during the actual showcase — verify LAN connectivity between Mac and 
  Windows machine well in advance.

### Disclose in documentation
- The use of 8-bit quantization for the primary model must be stated 
  explicitly in the README's limitations section, since it introduces a 
  small, honestly-reportable deviation from the activations the SAE 
  checkpoint was originally trained on (fp16/bf16).