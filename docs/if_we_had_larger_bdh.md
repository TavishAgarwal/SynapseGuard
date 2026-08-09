# Proposed Experiment: Scaling BDH Sparsity Analysis

**Claim Label: [EXPLORATORY]**

If full-scale BDH architecture checkpoints (100M+ parameters with active integrate-and-fire dynamics and biological synaptic plasticity) become available, the following experiment would directly compare native BDH sparsity dynamics against SAE-decomposed transformer representations:

1. **Sparsity Protocol Alignment:** Run identical predictability-controlled prompt spectrums across native BDH, Gemma-2-2B (SAE-decomposed), and GPT-2 (SAE-decomposed).
2. **Layer-wise Dynamic Mapping:** Compare dynamic activation sparsity in BDH synaptic matrices with feature activation density in SAE latents across varying context lengths.
3. **Intervention Testing:** Evaluate whether targeted suppression of high-entropy SAE features mirrors synaptic suppression in BDH architectures under high-uncertainty conditions.
