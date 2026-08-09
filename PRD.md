# Product Requirements Document (PRD) — SynapseGuard

## 1. Project Summary
SynapseGuard v2 is a two-part research-and-tooling project for the StarForge 2026 DragonForge track. It tests whether a specific, published finding from Pathway's Dragon Hatchling (BDH) architecture — that synaptic activation sparsity tracks input predictability — generalizes to standard dense transformers when their activations are decomposed via a Sparse Autoencoder (SAE). If the relationship replicates, the same signal is operationalized into a real-time hallucination early-warning layer (the Predictability-Sparsity Coherence Score, or PSC Score) for standard LLM inference.

## 2. Problem Statement
Production LLM systems hallucinate — generating fluent, confident, but factually incorrect output. Current mitigation (LLM-as-judge, external retrieval verification) is slow, expensive, and happens only after generation completes. There is no lightweight, model-agnostic, pre-token signal that flags an impending hallucination during generation.

Separately, it is unknown whether a specific interpretability finding from a novel biologically-inspired architecture (BDH) — that internal sparsity correlates with input predictability — is unique to that architecture or a general property of disentangled neural representations.

## 3. Goals
- **Primary (research):** Empirically determine whether the BDH predictability–sparsity relationship replicates in SAE-decomposed standard transformer activations, with honest reporting regardless of outcome.
- **Secondary (applied):** If Part A replicates, build a working real-time diagnostic sidecar that flags likely hallucinations during generation using the PSC Score.
- **Tertiary (communication):** Produce a GitHub repository and demo that clearly separates ESTABLISHED, MEASURED, and EXPLORATORY claims per DragonForge judging criteria.

## 4. Non-Goals
- Training a BDH model from scratch.
- Claiming the public `bdh.py` toy repo replicates frontier BDH capabilities.
- Building a production-scale serving system — this is a research prototype, not a hardened product.
- Replacing existing hallucination-mitigation approaches (RAG verification, LLM-judge) — this is a complementary, faster pre-filter.

## 5. Target Users
| User | Need |
|---|---|
| **DragonForge judges** | A GitHub repo that demonstrates genuine research engagement with BDH, not surface-level theming. |
| **ML researchers / interpretability enthusiasts** | A reusable replication harness to test the sparsity-predictability hypothesis on other models. |
| **RAG/chatbot developers (illustrative end user for Part B)** | A lightweight, low-latency layer that flags hallucination risk before a response reaches the user. |

## 6. Core Features

### Part A — Replication Test (Primary Deliverable)
- F1: Predictability-controlled input dataset generator (factual/low-entropy → open-ended/high-entropy spectrum).
- F2: Activation extraction pipeline (vLLM-Hook, last-token, target layers) for primary model (Gemma-2-2B, 8-bit), and cross-validation model (GPT-2 small, fp16) for cross-architecture validation.
- F3: SAE projection pipeline using a pre-trained, publicly available SAE for the base model.
- F4: Sparsity/density metric computation over SAE latents, correlated against measured next-token entropy/predictability.
- F5: Statistical analysis and visualization (correlation coefficient, significance test, scatter/curve plots) across multiple prompt categories.
- F6: `bdh.py` baseline instrumentation — measuring and visualizing the toy model's own sparsity behavior as a grounding reference (not as the main experiment).

### Part B — Applied Diagnostic (Conditional on Part A)
- F7: PSC Score calculator — real-time coherence-mismatch detector (high output confidence + high SAE-latent entropy = risk).
- F8: FastAPI sidecar service exposing the PSC scoring endpoint, fed asynchronously from vLLM-Hook.
- F9: Sample-Specific Prompting (SSP) dynamic thresholding using pre-computed perturbation templates, triggered only when PSC crosses a warning band.
- F10: Multi-benchmark validation (TruthfulQA, HaluEval, a RAG-grounding dataset subset) producing AUROC curves for the PSC Score as a hallucination classifier.
- F11: Real-time interception dashboard — live PSC gauge, visual flag/halt when the score breaches threshold during generation.

### Documentation & Showcase
- F12: README with ESTABLISHED / MEASURED / EXPLORATORY labeling throughout.
- F13: "If We Had Larger BDH Access" section with a precise proposed experiment.
- F14: Architecture diagram, setup instructions, limitations section, team contributions.
- F15: 20–30 second showcase clip capturing the PSC gauge collapsing just before a hallucinated token.

## 7. Success Criteria
- Part A produces a clear, reported result (positive, negative, or partial) with statistical backing — not a single anecdotal chart.
- If Part A is positive: Part B achieves a measurably-better-than-random AUROC (report the actual number, do not inflate) across at least two of the three validation benchmarks.
- Repository satisfies all DragonForge "Required GitHub Structure" items.
- No claim in the README overstates what `bdh.py` or the SAE proxy actually demonstrates.

## 8. Constraints
- Must run within available GPU budget (assume single consumer/cloud GPU, e.g., 24GB VRAM class — confirm actual availability before scoping model size).
- vLLM-Hook and SAE checkpoints must be from public, verifiable sources — no fabricated tooling.
- All benchmark subsets limited to 50–100 samples per source per the hackathon's realistic scope guidance (can extend if time and compute allow, but report exact sample sizes used).

## 9. Out of Scope for v1 Submission
- Fine-tuning a custom SAE (use pre-trained/off-the-shelf SAEs only, unless time permits as a stretch goal).
- Production authentication, multi-tenant deployment, billing, or persistence layers.
- Supporting closed-source/API-only models (requires open-weights access to hidden states).

## Hardware & Execution Environment

This project is built across two machines by design:

- **Development machine: MacBook Air M4.** All code writing, git management, 
  dashboard/UI development, data preparation, statistical analysis, plotting, 
  and documentation happen here. No GPU-dependent code runs natively on this 
  machine (Apple Silicon does not support vLLM/CUDA).
- **Execution machine: Windows laptop, RTX 3050 (6GB VRAM), via WSL2.** All 
  model loading, activation extraction, vLLM serving, and benchmark inference 
  run here exclusively.

All GPU-dependent scripts must be written to run unmodified on the RTX 3050 
machine after a `git pull` — no code changes should be required on the 
execution machine. Device selection (`cuda`/`mps`/`cpu`), quantization, and 
memory limits are handled via config files and environment variables, not 
hardcoded, so the same repository works correctly on both machines without 
manual edits.

### Model choice (revised for 6GB VRAM constraint)
- Primary model: **Gemma-2-2B, 8-bit quantized** (bitsandbytes) — matched to 
  the official Gemma Scope SAE release.
- Cross-validation model: **GPT-2 small/medium, fp16** — small enough to run 
  with no quantization needed, used to validate findings aren't 
  single-model artifacts.
- Quantization is documented as a stated limitation (see rules.md): 8-bit 
  activations may introduce minor distributional shift relative to the fp16 
  activations the SAE was originally trained on. This is disclosed honestly 
  in the README rather than hidden.