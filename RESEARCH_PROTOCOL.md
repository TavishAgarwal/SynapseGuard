# Research Protocol — Part A Replication Test

## Hypothesis

**H1:** SAE-decomposed activation sparsity in a standard transformer 
negatively correlates with input predictability (measured as next-token 
entropy), mirroring the relationship BDH exhibits natively in its synaptic 
activity.

**H0 (null):** No significant correlation exists, or the relationship is 
inconsistent/model-specific, indicating the BDH finding does not 
generalize via this SAE-based proxy method.

## Execution Context

All extraction, SAE projection, and metric computation described in this 
protocol run **once**, inside the single orchestrated GPU session 
(`gpu_session/run_full_gpu_session.py`) on the RTX 3050 6GB laptop. The 
statistical analysis and interpretation of the resulting metrics run 
separately on the Mac, consuming only `data/results/part_a_raw.csv`. This 
separation does not affect the validity of the protocol — the science 
happens in the extraction and metric computation, which is fully captured 
in the CSV output.

## Models Used (revised for 6GB VRAM constraint)

| Model | Precision | Role | SAE source |
|---|---|---|---|
| Gemma-2-2B | 8-bit (bitsandbytes) | Primary model | Gemma Scope (official DeepMind release) |
| GPT-2 small/medium | fp16 | Cross-validation model | SAE Lens (community-maintained, well-established) |

**Quantization disclosure:** Gemma-2-2B runs in 8-bit due to the 6GB VRAM 
constraint. This is disclosed in `docs/limitations.md` as a potential 
source of minor distributional shift relative to the fp16/bf16 activations 
the Gemma Scope SAEs were originally trained on. If time permits, a small 
fp16 vs int8 sanity check (a handful of samples run at both precisions) 
can be added to `data/results/` to quantify this shift directly — 
treat this as a stretch goal, not a requirement.

## Input Set Design

- **Categories (minimum 3, target 4):**
  1. Highly constrained factual completions (e.g., "The capital of France 
     is ___") — expected low entropy / high predictability.
  2. Common-sense/moderate reasoning completions — expected mid entropy.
  3. Open-ended creative/subjective prompts — expected high entropy / low 
     predictability.
  4. (Optional) Adversarial/ambiguous prompts — tests edge behavior.
- **Sample size:** minimum 40–50 per category (200 total across 4 
  categories), chosen to keep the single GPU session's runtime well within 
  an hour even on 6GB VRAM. Document the exact N used in 
  `session_manifest.json`.
- **Entropy measurement:** computed from the base model's own next-token 
  logit distribution at the completion point, per model (i.e., Gemma-2-2B's 
  entropy values and GPT-2's entropy values are each measured against 
  their own predictions — do not cross-apply one model's entropy score to 
  the other model's activations).
- **Built on the Mac, no GPU needed** — this is pure text/data 
  construction and can happen well before the GPU session, so the GPU 
  session's single run can start immediately with a finalized input set.

## Extraction Protocol

- Fixed layer range per model, chosen based on available Gemma Scope / SAE 
  Lens checkpoint coverage (e.g., layers 11–14 equivalent for Gemma-2-2B; 
  document the exact GPT-2 layer mapping separately, since layer depth and 
  naming differ between architectures).
- Do not cherry-pick the best-looking layer after the fact — if multiple 
  layers are extracted, report results across all of them, not just the 
  strongest correlation.
- Last-token position only, extracted via `vLLM-Hook`'s `last_token` 
  config, consistent with the latency-sensitive design used in Part B.
- All extraction, for both models and all categories, happens within the 
  single GPU session — `step2_part_a_extraction.py` loops over the full 
  input set once per model.

## Metric Definition

- **Sparsity/density metric:** defined explicitly in `gpu_session/sae_project.py` 
  as [e.g., L0 norm of active SAE latent dimensions above a fixed 
  activation threshold, or normalized entropy over the latent activation 
  vector] — pick one definition and use it identically across Part A and 
  Part B without silent redefinition. Document the exact formula in this 
  file's docstring and reference it here once finalized.
- This metric is computed **during the GPU session**, immediately after 
  the SAE forward pass, and written directly to `part_a_raw.csv` — the raw 
  high-dimensional SAE latent vectors themselves are not saved by default 
  (too large, not needed for the Mac-side analysis), keeping the output 
  file small and git-friendly.

## Statistical Analysis (runs on Mac, from `part_a_raw.csv`)

- Pearson and Spearman correlation between `measured_entropy` and 
  `sae_sparsity_metric`, computed per model.
- Per-category breakdown (not just an aggregate correlation) — an 
  aggregate can mask category-specific effects.
- Compare Gemma-2-2B and GPT-2 results side by side — genuine cross-model, 
  cross-architecture validation, since these two models differ 
  significantly in scale and design.
- Report p-values and sample sizes alongside every correlation figure.

## Reporting Rules

- If H1 holds for both models: report effect sizes honestly, per model, 
  without rounding up borderline significance.
- If H1 holds for one model but not the other: report this explicitly as 
  a partial result — this is a legitimate, interesting finding (e.g., "the 
  relationship may depend on model scale or architecture family"), not a 
  failure to hide.
- If H0 holds for both: document this as the primary Part A finding, and 
  reframe Part B's applied claim accordingly — the PSC signal may still 
  have practical value as a hallucination detector even if it doesn't 
  mirror BDH's underlying mechanism exactly, but this must not be 
  described as validating the BDH analogy.
- All raw per-sample data (`part_a_raw.csv`) must remain in the repository 
  for independent verification, generated fresh by anyone who reruns 
  `run_full_gpu_session.py`.

## Reproducibility

Since the entire experiment is designed to run in one orchestrated 
session, reproducibility is straightforward to state in the README: 
*"Clone the repo, set up the GPU-session environment per requirements-gpu.txt 
on a CUDA-capable machine (tested on RTX 3050 6GB), run 
`python gpu_session/run_full_gpu_session.py`, and all files in 
data/results/ will be regenerated. All analysis and the dashboard then run 
identically on any machine using requirements-mac.txt, with no GPU 
required."* This is a stronger reproducibility story than a 
multi-session, ad hoc extraction process would offer — it's a genuine 
strength of this design.