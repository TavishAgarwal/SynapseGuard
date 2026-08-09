# System Limitations & Disclosures — SynapseGuard

Per DragonForge guidelines and `rules.md`, all technical limitations are documented transparently:

1. **Quantization Shift [MEASURED]:** The primary model (`Gemma-2-2B`) runs with 8-bit quantization (`bitsandbytes`) to fit within the 6GB VRAM hardware constraint on the RTX 3050. This introduces a minor distributional shift relative to the fp16 activations the pre-trained Gemma Scope SAE was originally trained on.
2. **Single GPU Session & Replay Presentation [ESTABLISHED]:** All LLM inference, activation extraction, SAE projections, and benchmark scoring are executed during a single, orchestrated session on an RTX 3050. Dashboard presentations on Mac M4 Air replay these pre-computed, genuine activations with real-time-matched token pacing.
3. **Toy BDH Baseline [ESTABLISHED]:** The `bdh.py` implementation is a ~10M parameter educational baseline instrumentation and does not represent full BDH frontier capabilities.
4. **Benchmark Validation Scope [ESTABLISHED]:** The PSC benchmark validation (`step4_benchmark_validation.py`, `benchmark_scores.csv`) runs using Gemma-2-2B only. GPT-2 is used exclusively for Part A cross-validation. AUROC figures therefore reflect Gemma-2-2B's PSC scores against benchmark ground truth.
5. **Sample Size Constraint [ESTABLISHED]:** Part A uses 50 samples per category (200 total). At n=50, Pearson correlation has approximately 70% statistical power to detect a moderate effect (r=0.3) at α=0.05. Results with |r| < 0.3 are reported as exploratory.
6. **Activation Extraction via HF Forward Hooks [ESTABLISHED]:** Model hidden states are extracted using PyTorch `register_forward_hook()` on HuggingFace Transformers, which provides exact offline activation access within the 6GB VRAM budget.
