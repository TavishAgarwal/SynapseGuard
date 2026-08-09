# Architecture Document — SynapseGuard

## 1. Core Design Principle: Single GPU Session

This project is architected so the RTX 3050 (6GB VRAM) laptop is used
**exactly once**, in a single orchestrated session, to do all model-loading
and GPU-dependent computation. Everything downstream — statistics, 
benchmark scoring, the dashboard, and the "live" demo — runs entirely on 
the Mac M4 Air using only the lightweight output files produced by that 
one session.

This works because the only steps that genuinely require the base LLM 
(Gemma-2-2B / GPT-2) and a CUDA GPU are:
1. Running the model forward pass to get hidden states.
2. Running the model's own generation loop (for benchmark completions and 
   demo traces).

The SAE forward pass itself is small enough (a few hundred MB, simple 
matrix operations) to run on the GPU machine *during the same session*, so 
its output — sparsity/density metrics — can be computed and saved directly, 
rather than requiring the SAE to be loaded again later. The Mac never 
needs to load the LLM, the SAE, or CUDA at all.

## 2. High-Level System Flow

### The one-time GPU session (RTX 3050, via WSL2)
┌─────────────────────────────────────────────────────────────────────┐
│ SINGLE GPU SESSION — run_full_gpu_session.py │
│ (orchestrates all sub-steps below, in order, in one process) │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ Step 1: Load Gemma-2-2B (8-bit) + GPT-2 (fp16) via vLLM │
│ │
│ Step 2: PART A EXTRACTION │
│ Predictability-controlled inputs (from data/predictability_inputs)│
│ │ │
│ ▼ │
│ vLLM-Hook: last-token activation, target layers │
│ │ │
│ ▼ │
│ SAE projection (Gemma Scope / SAE Lens) — done here, same session│
│ │ │
│ ▼ │
│ Compute sparsity/density + entropy → data/results/part_a_raw.csv │
│ │
│ Step 3: BDH BASELINE (optional, can also run on Mac — see §6) │
│ │
│ Step 4: BENCHMARK VALIDATION │
│ TruthfulQA / HaluEval / RAG subset prompts │
│ │ │
│ ▼ │
│ Generate completions, extract activations, SAE-project, │
│ compute PSC Score per sample, attach ground-truth label │
│ │ │
│ ▼ │
│ → data/results/benchmark_scores.csv │
│ │
│ Step 5: DEMO TRACE GENERATION │
│ 3–5 chosen demo prompts (mix of clean + hallucination-prone) │
│ │ │
│ ▼ │
│ Generate token-by-token: token text, logit confidence, │
│ PSC score, SSP-triggered flag, status (SAFE/WARNING/HALT) │
│ │ │
│ ▼ │
│ → data/results/demo_traces/*.json (one file per demo prompt) │
│ │
│ Step 6: Write session_manifest.json (models, quantization, layers, │
│ checkpoint versions, sample counts, timestamps — for reproducibility)│
│ │
└─────────────────────────────────────────────────────────────────────┘
│
│ git push (small files only —
│ no raw tensors, no model weights)
▼
┌─────────────────────┐
│ GitHub repo │
└─────────────────────┘
│
│ git pull
▼

### Everything after that (Mac M4 Air, no GPU needed)
┌─────────────────────────────────────────────────────────────────────┐
│ MAC — all downstream work │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ data/results/part_a_raw.csv │
│ │ │
│ ▼ │
│ Statistical analysis (correlation, significance, plots) │
│ │ │
│ ▼ │
│ Part A report + charts │
│ │
│ data/results/benchmark_scores.csv │
│ │ │
│ ▼ │
│ AUROC computation (sklearn — no GPU needed) │
│ │ │
│ ▼ │
│ Part B validation report + charts │
│ │
│ data/results/demo_traces/*.json │
│ │ │
│ ▼ │
│ Dashboard (React or terminal UI) replays the trace token-by-token │
│ with realistic timing delays, animating the PSC gauge exactly as │
│ it would appear live — because the underlying scores are real, │
│ computed once from real model activations, not fabricated. │
│ │
│ README, architecture diagram, showcase clip recording │
│ │
└─────────────────────────────────────────────────────────────────────┘

## 3. Why the "Replay" Demo Is Legitimate, Not Fabricated

The DragonForge rubric rewards honest evidence, so this design choice must 
be documented transparently, not hidden:

- Every number in a demo trace (PSC score, token confidence, SAFE/WARNING/HALT 
  status) is computed from a **real forward pass on a real model**, during 
  the single GPU session. Nothing is invented or hand-tuned after the fact.
- The only thing that's "replayed" rather than live is the *display timing* 
  — the dashboard animates the stored trace with delays matching realistic 
  generation speed, so it looks and behaves like a live stream.
- The README must state this plainly: *"Demo traces are pre-computed from 
  real model activations in a single GPU session (hardware constraint: 
  6GB VRAM laptop) and replayed in the dashboard with real-time-matched 
  pacing. The scoring logic itself is fully real-time-capable and is 
  demonstrated as such in the architecture (vLLM-Hook async extraction, 
  FastAPI sidecar) — the demo replay is a hardware-driven presentation 
  choice, not a limitation of the method."*
- This is a defensible, honest framing that satisfies "Working Proof" 
  requirements: the proof is real, generated once, and reproducible by 
  anyone who reruns `run_full_gpu_session.py`.

## 4. Tech Stack

| Layer | Technology | Runs on |
|---|---|---|
| Inference engine | vLLM | RTX 3050 (WSL2), single session only |
| Activation extraction | vLLM-Hook (IBM) | RTX 3050 (WSL2), single session only |
| Quantization | bitsandbytes (int8) | RTX 3050 (WSL2), single session only |
| Interpretability layer | Gemma Scope SAE (Gemma-2-2B), SAE Lens (GPT-2) | RTX 3050 (WSL2), single session only |
| PSC/SSP scoring logic | Custom Python | Computed once on RTX 3050; logic also runnable standalone on Mac against stored data for verification |
| Statistical analysis | NumPy, SciPy, pandas | Mac M4 Air |
| AUROC / benchmark scoring | scikit-learn | Mac M4 Air |
| Plotting | matplotlib / plotly | Mac M4 Air |
| Dashboard | React (or terminal UI via `rich`/`textual`) | Mac M4 Air, reads local JSON trace files — no network call to RTX 3050 needed |
| Environment (GPU session) | Python 3.10+, WSL2 Ubuntu, CUDA 12.x | RTX 3050 laptop |
| Environment (everything else) | Python 3.10+, standard venv | Mac M4 Air |

**Note on the FastAPI sidecar:** it is still built and included in the 
repo as the production-path implementation (per the original architecture), 
and it is exercised during the single GPU session to prove it works 
end-to-end with a live vLLM instance. For the actual hackathon demo, 
however, the dashboard reads directly from the pre-computed trace files 
rather than calling a live sidecar — this avoids any cross-machine network 
dependency during the showcase itself, which is safer for a live 
presentation.

## 5. Folder Structure
synapseguard/
├── README.md
├── PRD.md
├── architecture.md
├── rules.md
├── phases.md
├── design.md
├── research_protocol.md
├── requirements-gpu.txt # deps needed ONLY on RTX 3050 session
├── requirements-mac.txt # deps needed on Mac (no torch-cuda, no vLLM)
├── .env.example
├── configs/
│ ├── model_config.yaml # base models, layers, SAE checkpoint paths
│ ├── device_config.yaml # device="cuda", quantization="int8",
│ │ # gpu_memory_utilization=0.85, max_model_len=512
│ └── benchmark_config.yaml # dataset paths, sample sizes
├── data/
│ ├── predictability_inputs/ # Part A input set (built on Mac, no GPU needed)
│ ├── benchmarks/ # cached benchmark subsets (built on Mac)
│ └── results/ # ALL GPU-session outputs — lightweight only
│ ├── session_manifest.json # models, quant level, layers, checkpoint
│ │ # versions, sample counts, timestamps
│ ├── part_a_raw.csv # per-sample: entropy, sparsity, category, model
│ ├── benchmark_scores.csv # per-sample: PSC score, true label, dataset
│ └── demo_traces/
│ ├── demo_01_factual.json
│ ├── demo_02_hallucination_prone.json
│ └── demo_03_open_ended.json
├── gpu_session/ # ONLY runs on RTX 3050 — self-contained
│ ├── run_full_gpu_session.py # single orchestrator script — entry point
│ ├── step1_load_models.py
│ ├── step2_part_a_extraction.py
│ ├── step3_bdh_baseline.py # optional here; can also run on Mac
│ ├── step4_benchmark_validation.py
│ ├── step5_demo_trace_generation.py
│ └── sae_project.py # SAE forward pass, sparsity computation
│ # (also called from Mac-side unit tests
│ # against mock tensors, if useful)
├── analysis/ # runs on Mac, consumes data/results/.csv
│ ├── part_a_stats.py # correlation, significance, plots
│ ├── benchmark_auroc.py # AUROC computation from benchmark_scores.csv
│ └── plots/
├── part_b_diagnostic/
│ ├── sidecar/
│ │ ├── main.py # FastAPI app — built/tested on Mac against
│ │ │ # mocks, exercised live only during the
│ │ │ # single GPU session
│ │ ├── psc_score.py # shared logic — identical file used in
│ │ │ # gpu_session/ and here, kept in sync
│ │ ├── ssp.py
│ │ └── hook_client.py
│ └── dashboard/
│ ├── (React app OR terminal UI)
│ └── replay_engine.js/py # reads demo_traces/.json, animates
│ # with realistic timing
├── notebooks/
│ └── exploratory/
├── docs/
│ ├── architecture_diagram.png
│ ├── if_we_had_larger_bdh.md
│ ├── limitations.md # must include the quantization +
│ │ # single-session/replay disclosures
│ └── bdh_baseline_notes.md
└── demo/
└── showcase_clip.mp4 (or link)

## 6. What Can Run on Either Machine (Flexibility Note)

The BDH baseline instrumentation (`bdh.py`, ~10M params) is light enough 
to run on CPU/MPS. It can be run on the Mac (recommended, avoids using 
GPU-session time on it) or folded into the single GPU session for 
convenience — either is acceptable. Document which one was actually used 
in `docs/bdh_baseline_notes.md`.

## 7. Data Contracts (Output Schemas)

**`part_a_raw.csv`** — one row per (input, model, layer):
input_id, model, layer, category, measured_entropy, sae_sparsity_metric, timestamp

**`benchmark_scores.csv`** — one row per benchmark sample:
sample_id, dataset, model, psc_score, ssp_triggered, predicted_label, true_label, timestamp

**`demo_traces/*.json`** — one file per demo prompt:
```json
{
  "prompt": "string",
  "model": "gemma-2-2b-8bit",
  "tokens": [
    {
      "position": 0,
      "token_text": "string",
      "logit_confidence": 0.94,
      "psc_score": 0.12,
      "status": "SAFE",
      "ssp_triggered": false
    }
  ],
  "generation_metadata": {
    "quantization": "int8",
    "layers_used": [11, 12, 13, 14],
    "session_timestamp": "ISO8601"
  }
}
```

**`session_manifest.json`** — written once at the end of the GPU session:
```json
{
  "session_date": "ISO8601",
  "models": ["gemma-2-2b (int8)", "gpt2-small (fp16)"],
  "sae_checkpoints": {"gemma-2-2b": "gemma-scope-...", "gpt2": "sae-lens-..."},
  "layers_extracted": [11, 12, 13, 14],
  "part_a_sample_count": 200,
  "benchmark_sample_counts": {"truthfulqa": 80, "halueval": 80, "rag_subset": 60},
  "gpu": "RTX 3050 6GB",
  "quantization": "int8 (bitsandbytes)",
  "notes": "Single GPU session, all downstream analysis run on Mac M4 Air."
}
```

## 8. Deployment Notes
- No hosting/deployment needed for submission — the dashboard runs locally 
  on the Mac, reading local files, for both development and the final demo.
- The FastAPI sidecar's live-serving capability is demonstrated *once*, 
  during the GPU session, as evidence it works — it does not need to be 
  running during the actual hackathon demo.
- Recommend recording the showcase clip directly from the Mac dashboard's 
  replay of `demo_02_hallucination_prone.json`, since that trace is 
  designed to contain the "PSC gauge collapses right before a hallucinated 
  token" moment.