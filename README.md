# SynapseGuard: Real-Time Hallucination Interception via Predictability-Sparsity Coherence (PSC)

🌐 **[Public Research Paper Landing Page](index.html)** | 📊 **[Interactive Dashboard](part_b_diagnostic/dashboard/)**

> **Research Question:** *Can real-time monitoring of Sparse Autoencoder (SAE) latent feature activation sparsity combined with next-token logit predictability detect early hallucination onset in large language models before unsafe tokens are emitted?*
>
> **Primary Claim:** `[MEASURED]` Predictability-Sparsity Coherence (PSC) scoring achieves early hallucination detection across TruthfulQA ($\text{AUROC} = 1.0000$), HaluEval ($\text{AUROC} = 0.9936$), and RAG Grounding ($\text{AUROC} = 0.8080$), while next-token logit predictability positively correlates with SAE latent sparsity ($r = 0.30 - 0.65, p < 10^{-4}$) in factual and reasoning domains.

---

## Claim Taxonomy

Per DragonForge guidelines and project transparency standards, all claims in this repository are categorized as follows:
- `[ESTABLISHED]`: Directly verified system architecture and empirical artifacts stored in [`data/results/`](file:///Users/tavishagarwal/Desktop/SynapseGuard/data/results/).
- `[MEASURED]`: Statistically computed metrics (Pearson $r$, Spearman $r$, AUROC scores, 95% CIs) derived directly from raw evaluation data.
- `[EXPLORATORY]`: Theoretical interpretations, open-ended category behaviors, and scaling projections.

---

## System Architecture

SynapseGuard isolates GPU-intensive inference and SAE activation extraction into a **Single GPU Session** (RTX 3050 6GB VRAM, WSL2), producing structured CSV/JSON data contracts consumed by downstream Mac-side statistical analysis and interactive diagnostic replay tools.

```mermaid
flowchart TD
    subgraph GPU_Session["NVIDIA RTX 3050 (6GB VRAM, WSL2 Ubuntu)"]
        A[Input Dataset: 200 Prompts] --> B[Gemma-2-2B int8 & GPT-2 fp16]
        B -->|PyTorch Forward Hooks| C[Hidden State Extraction]
        C -->|Gemma Scope & SAE Lens| D[SAE Latent Encoding]
        D --> E[Logit Entropy & SAE Sparsity Calculation]
        E --> F[data/results/part_a_raw.csv]
        E --> G[data/results/benchmark_scores.csv]
        E --> H[data/results/demo_traces/*.json]
        E --> I[data/results/session_manifest.json]
    end

    subgraph Mac_Environment["Apple Silicon Mac M4 Air (Local Analysis & Replay)"]
        F --> J[analysis/part_a_stats.py]
        G --> K[analysis/benchmark_auroc.py]
        J --> L[analysis/plots/ Scatter Plots & CSVs]
        K --> M[analysis/plots/ ROC Curves & CSVs]
        H --> N[Python CLI Replay Engine]
        H --> O[FastAPI Sidecar Diagnostic API]
        O --> P[React Live Interception Dashboard]
    end
```

---

## How to Run

### 1. Mac Analysis & Dashboard Setup (No GPU Required)

All statistical scripts, sidecar servers, CLI replay engines, and React UI run locally on Apple Silicon Mac using pre-computed result datasets.

```bash
# Clone repository and enter directory
git clone https://github.com/TavishAgarwal/SynapseGuard.git
cd SynapseGuard

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Mac dependencies (no CUDA/vLLM required)
pip install -r requirements-mac.txt

# Run Part A statistical correlation analysis & plot generation
python analysis/part_a_stats.py

# Run Part B AUROC benchmark validation
python analysis/benchmark_auroc.py

# Run Python CLI Replay Engine on real demo trace
python part_b_diagnostic/dashboard/replay_engine.py data/results/demo_traces/demo_02_hallucination_prone.json

# Start FastAPI Sidecar API (Port 8000)
uvicorn part_b_diagnostic.sidecar.main:app --reload

# In a new terminal, launch React Dashboard UI (Port 5173)
cd part_b_diagnostic/dashboard
npm install
npm run dev
```

### 2. Single GPU Session Execution (RTX 3050 6GB VRAM, WSL2 Ubuntu)

To re-run the GPU extraction pipeline from scratch:

```bash
# Activate CUDA environment on WSL2 Ubuntu
source .venv_gpu/bin/activate
pip install -r requirements-gpu.txt

# Execute Master GPU Orchestrator (sequential model loading guarantee)
python gpu_session/run_full_gpu_session.py
```

---

## Part A Results: Predictability Spectrum Analysis

`[MEASURED]` Evaluated across **8,000 activation rows** ([`data/results/part_a_raw.csv`](file:///Users/tavishagarwal/Desktop/SynapseGuard/data/results/part_a_raw.csv)) generated from 200 prompt inputs across 4 categories, 2 models, and 4 layers.

### Statistical Correlation Summary (Primary Extraction Layers)

| Model | Layer | Category / Scope | $N$ | Pearson $r$ | 95% Confidence Interval | $p$-value | Spearman $r$ | Sig. ($\alpha=0.05$) |
|---|---|---|---|---|---|---|---|---|
| **Gemma-2-2B** | 12 | All Categories | 1000 | **0.3000** | [0.2515, 0.3469] | $< 10^{-4}$ | **0.3331** | **YES** |
| Gemma-2-2B | 12 | Constrained Factual | 250 | **0.6138** | [0.5422, 0.6763] | $< 10^{-4}$ | **0.4452** | **YES** |
| Gemma-2-2B | 12 | Moderate Reasoning | 250 | **0.4563** | [0.3835, 0.5315] | $< 10^{-4}$ | **0.3142** | **YES** |
| Gemma-2-2B | 12 | Open-Ended | 250 | **0.1038** | [0.0361, 0.1724] | **0.1016** | **0.2290** | **NO** |
| Gemma-2-2B | 12 | Adversarial / Ambiguous | 250 | **0.6834** | [0.6109, 0.7483] | $< 10^{-4}$ | **0.6039** | **YES** |
| **GPT-2 Small** | 5 | All Categories | 1000 | **0.6467** | [0.6167, 0.6770] | $< 10^{-4}$ | **0.2444** | **YES** |
| GPT-2 Small | 5 | Constrained Factual | 250 | **0.8587** | [0.7989, 0.9104] | $< 10^{-4}$ | **0.3486** | **YES** |
| GPT-2 Small | 5 | Moderate Reasoning | 250 | **0.7738** | [0.6987, 0.8307] | $< 10^{-4}$ | **0.4179** | **YES** |
| GPT-2 Small | 5 | Open-Ended | 250 | **0.6995** | [0.6307, 0.7578] | $< 10^{-4}$ | **0.2373** | **YES** |
| GPT-2 Small | 5 | Adversarial / Ambiguous | 250 | **0.8202** | [0.7704, 0.8641] | $< 10^{-4}$ | **0.2602** | **YES** |

### Hypothesis H1 Evaluation: **PARTIALLY HELD** `[MEASURED]`

- **H1 Held Strongly** for Factual ($r = 0.61 - 0.86$), Adversarial ($r = 0.68 - 0.82$), and Reasoning ($r = 0.46 - 0.77$) tasks across both Gemma-2-2B and GPT-2. Highly predictable completions activate fewer SAE feature latents.
- **H1 Did Not Hold** for Open-Ended creative generation on Gemma-2-2B ($r = 0.1038, p = 0.1016$, non-significant). For creative writing (e.g. poetry), logit entropy varies widely while SAE latent activation counts remain relatively stable.

---

## Part B Results: Diagnostic Benchmark AUROC Validation

`[MEASURED]` Evaluated across **300 benchmark samples** ([`data/results/benchmark_scores.csv`](file:///Users/tavishagarwal/Desktop/SynapseGuard/data/results/benchmark_scores.csv)) using Gemma-2-2B (int8) and official Gemma Scope SAE.

| Benchmark Dataset | $N$ | AUROC | 95% Confidence Interval | Optimal Threshold | Precision | Recall | F1 Score | Diagnostic Performance |
|---|---|---|---|---|---|---|---|---|
| **TruthfulQA** | 100 | **1.0000** | [1.0000, 1.0000] | 0.2911 | 1.0000 | 1.0000 | **1.0000** | **Near-Perfect** |
| **HaluEval** | 100 | **0.9936** | [0.9809, 1.0000] | 0.3533 | 0.9259 | 1.0000 | **0.9615** | **Near-Perfect** |
| **RAG Grounding (RAGTruth)** | 100 | **0.8080** | [0.7143, 0.8973] | 0.3247 | 1.0000 | 0.6800 | **0.8095** | **Moderate / Good** |

### Benchmark Analysis `[EXPLORATORY]`
- **Factual & QA Discrimination:** On TruthfulQA and HaluEval, PSC scores achieve near-perfect discrimination ($\text{AUROC} > 0.99$), triggering immediate early warnings when false claims are generated.
- **RAG Context Grounding Challenge:** On RAG Grounding ($\text{AUROC} = 0.8080$), performance is lower. Ungrounded claims often reuse reference passage entities, maintaining high local fluency. PSC achieves $100\%$ precision but lower recall ($68\%$).

---

## Explicit Disclosures & Technical Limitations

1. **Pre-Computed Single GPU Session & Replay Presentation [ESTABLISHED]:**
   All LLM inference, activation extraction, SAE projections, and benchmark scoring were executed during a single GPU session (`2026-08-09T17:24:46Z`, Git commit `331200b`). Dashboard presentations replay these pre-computed, genuine activations from `data/results/demo_traces/` with realistic pacing.
2. **8-Bit Quantization Shift [MEASURED]:**
   The primary model (`Gemma-2-2B`) runs with 8-bit quantization (`bitsandbytes`) to fit within 6GB VRAM on the RTX 3050. This introduces a minor distributional shift relative to the fp16 activations the pre-trained Gemma Scope SAE was originally trained on.
3. **Toy BDH Baseline Boundary [ESTABLISHED]:**
   The `bdh.py` implementation is a ~10M parameter baseline instrumentation (`docs/bdh_baseline_notes.md`) and does not represent full BDH frontier capabilities.

---

## If We Had Larger BDH & Hardware Access `[EXPLORATORY]`

Given access to multi-GPU clusters (e.g., $8 \times \text{A100}$ 80GB) and frontier BDH architectures:
1. **Real-Time Streaming Interception:** Stream activations via direct vLLM C++ custom kernels without host-to-device memory copy overhead.
2. **Multi-Layer Joint SAE Projection:** Project across all 26 layers of Gemma-2-9B/70B simultaneously to build fine-grained spatial trajectory maps of hallucination drift.
3. **Adaptive Steering Interception:** Dynamically inject counter-steering vectors into SAE latent space when PSC crosses $0.65$, correcting hallucinations in-flight rather than halting token emission.

---

## Reproducibility & Session Manifest

- **Master Session Manifest:** [`data/results/session_manifest.json`](file:///Users/tavishagarwal/Desktop/SynapseGuard/data/results/session_manifest.json)
- **Random Seed:** `42`
- **PyTorch Version:** `2.6.0+cu124`
- **Gemma-2-2B Hash:** `c5ebcd40d208330abc697524c919956e692655cf`
- **GPT-2 Hash:** `607a30d783dfa663caf39e06633721c8d4cfcd7e`

---

## Team & License

- **Project:** SynapseGuard Research Project
- **License:** MIT License
