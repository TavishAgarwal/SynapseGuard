# Phases.md — Project Breakdown

## Phase 0: Setup & Prerequisites
- Environment setup, dependency installation, model/checkpoint access confirmed.
- Deliverable: working vLLM inference on the base model, confirmed vLLM-Hook extraction of a single activation tensor.
## Phase 0: Setup & Prerequisites
**Machine: both.**
- Mac: repo setup, virtual environment for non-GPU code, dashboard scaffolding.
- RTX 3050 (WSL2): vLLM + vLLM-Hook install, CUDA verification, Gemma-2-2B 
  8-bit load test, single-activation extraction test end to end.
## Phase 1: Baseline Grounding (bdh.py) + Input Set Construction
- Instrument public `bdh.py`, measure/visualize its own sparsity behavior on documented tasks.
- Build the predictability-controlled input dataset (factual/low-entropy → open-ended/high-entropy), with entropy pre-computed and logged per input.
- Deliverable: `docs/bdh_baseline_notes.md` + validated input dataset with entropy labels.
## Phase 1: Baseline Grounding + Input Set Construction
**Machine: Mac** (bdh.py is small enough to run locally; input dataset 
construction is pure Python/text, no GPU needed).
## Phase 2: Part A — Core Replication Test
- Run activation extraction across the full input set on the primary base model (LLaMA-3-8B).
- Project through SAE, compute sparsity/density metrics.
- Correlate against measured predictability; run statistical significance tests.
- Repeat on second model (Qwen2.5-7B) for cross-model validation.
- Deliverable: correlation report + plots, clearly stating whether the hypothesis held, partially held, or did not hold.
## Phase 2: Part A — Core Replication Test
**Machine: RTX 3050 (WSL2)** for extraction + SAE projection.
**Machine: Mac** for the correlation/statistics/plotting step, once raw 
metrics are pulled back via git.
## Phase 3: Decision Gate
- Explicit checkpoint: review Part A results.
  - If relationship replicates clearly → proceed to Phase 4 (Part B) as planned.
  - If null/weak result → pivot Part B scope to an honest "attempted application, results inconclusive, here's why" framing rather than forcing a detector on a shaky signal. This is still a valid, fundable submission.
## Phase 3: Decision Gate
**Machine: Mac** (review only, no compute).
## Phase 4: Part B — Applied Diagnostic Build
- Build PSC Score calculator.
- Build FastAPI sidecar, integrate vLLM-Hook for real-time extraction.
- Implement SSP dynamic thresholding with pre-computed perturbation templates.
- Deliverable: working real-time scoring endpoint, tested manually on a handful of prompts.
## Phase 4: Part B — Applied Diagnostic Build
**Machine: Mac** for FastAPI/PSC/SSP code logic (test against mock tensors).
**Machine: RTX 3050 (WSL2)** for wiring the sidecar to a live vLLM instance 
and end-to-end testing.
## Phase 5: Validation
- Run PSC scorer against TruthfulQA, HaluEval, RAG-grounding subsets.
- Compute and report AUROC per benchmark.
- Deliverable: validation report with honest metrics (including any benchmark where performance is weak).
## Phase 5: Validation
**Machine: RTX 3050 (WSL2)** for running the benchmark inference.
**Machine: Mac** for AUROC computation and result writeup, once results 
are pulled back.

## Phase 6: Dashboard & Demo
- Build live PSC gauge UI (React or terminal).
- Wire up end-to-end demo: user prompt → generation → live gauge → flag/halt on risk.
- Record the 20–30 second showcase clip.
## Phase 6: Dashboard & Demo
**Machine: Mac** for dashboard build (against mock data).
**Machine: both** for final wiring — dashboard on Mac connects to FastAPI 
sidecar running live on the RTX 3050 machine over LAN.
## Phase 7: Documentation & Submission
- Finalize README with all required DragonForge sections (claim, question, architecture diagram, how to run, proof, tech/research anchor, limitations, team contributions, demo link, "If We Had Larger BDH Access").
- Final review pass: verify every claim is tagged and every number is traceable to a result file.
- Push to public GitHub, verify clean setup from scratch on a fresh environment/machine if possible.

## Phase 7: Documentation & Submission
**Machine: Mac.**












