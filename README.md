# SynapseGuard: Predictability-Sparsity Coherence (PSC) Score

> **Real-time Hallucination Early-Warning via Sparse Autoencoder Activation Dynamics**

SynapseGuard is a research and tooling system for StarForge 2026 (DragonForge track). It tests whether internal activation sparsity in standard dense transformers—decomposed via pre-trained Sparse Autoencoders (SAEs)—tracks input predictability, generalizing key findings from Pathway's Dragon Hatchling (BDH) architecture.

---

## 🏗️ Architecture & Single GPU Session Model

To operate reliably within laptop hardware constraints (RTX 3050 6GB VRAM), SynapseGuard is architected around a **Single GPU Session**:

1. **GPU Session (RTX 3050 WSL2):** Runs `gpu_session/run_full_gpu_session.py` once to extract hidden states (`vLLM-Hook`), compute SAE latent sparsity, execute benchmark evaluation, and write lightweight trace files (`data/results/`).
2. **Mac M4 Air (Downstream Analysis & Dashboard):** Consumes output CSV/JSON files to perform statistical analysis (`analysis/part_a_stats.py`), AUROC scoring (`analysis/benchmark_auroc.py`), and token-by-token UI replay (`part_b_diagnostic/dashboard/replay_engine.py`).

For full details, see [`architecture.md`](file:///Users/tavishagarwal/Desktop/SynapseGuard/architecture.md).

---

## 🏷️ Research Claim Classifications

- **[ESTABLISHED]** Standard LLMs exhibit variable output confidence across factual vs open-ended prompt distributions.
- **[MEASURED]** Gemma-2-2B (8-bit quantized) hidden layer activation sparsity mapped via Gemma Scope SAE latents across predictability-controlled inputs (`data/results/part_a_raw.csv`).
- **[EXPLORATORY]** Generalizability of the PSC Score to non-transformer biologically-inspired architectures at 100M+ scale.

---

## 📂 Repository Structure

```
synapseguard/
├── README.md
├── PRD.md
├── architecture.md
├── rules.md
├── phases.md
├── design.md
├── RESEARCH_PROTOCOL.md
├── requirements-gpu.txt    # RTX 3050 session dependencies
├── requirements-mac.txt    # Mac dependencies (no CUDA / vLLM)
├── .env.example
├── configs/
│   ├── model_config.yaml
│   ├── device_config.yaml
│   └── benchmark_config.yaml
├── data/
│   ├── predictability_inputs/
│   ├── benchmarks/
│   └── results/
│       ├── session_manifest.json
│       ├── part_a_raw.csv
│       ├── benchmark_scores.csv
│       └── demo_traces/
├── gpu_session/
│   ├── run_full_gpu_session.py
│   ├── step1_load_models.py
│   ├── step2_part_a_extraction.py
│   ├── step3_bdh_baseline.py
│   ├── step4_benchmark_validation.py
│   ├── step5_demo_trace_generation.py
│   └── sae_project.py
├── analysis/
│   ├── part_a_stats.py
│   ├── benchmark_auroc.py
│   └── plots/
├── part_b_diagnostic/
│   ├── sidecar/
│   └── dashboard/
├── notebooks/
├── docs/
└── demo/
```

---

## 🚀 Quickstart (Mac Setup)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Mac dependencies only (no CUDA / vLLM)
pip install -r requirements-mac.txt

# 3. Run sidecar diagnostic test / dashboard replay
python part_b_diagnostic/dashboard/replay_engine.py
```

---

## ⚠️ Stated Limitations & Disclosures

See [`docs/limitations.md`](file:///Users/tavishagarwal/Desktop/SynapseGuard/docs/limitations.md) for complete details on 8-bit quantization shift, single-session presentation rationale, and toy baseline boundaries.
