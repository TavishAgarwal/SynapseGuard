# Heavy Files & Cache Cleanup Guide — SynapseGuard

Per project architecture, all model weights and high-dimensional SAE checkpoints are downloaded to the local HuggingFace cache directory during execution. Once the single GPU session completes and all lightweight result files (`data/results/*.csv`, `data/results/*.json`) are generated, the large download files can be safely deleted to free up disk space.

---

## Heavy Cache Locations to Delete

### 1. HuggingFace Model & SAE Checkpoint Cache (~6.0 GB total)
Located at: `C:\Users\nikun_r4vgzi9\.cache\huggingface\hub\` (Windows) or `~/.cache/huggingface/hub/` (WSL)

Specific directories to delete:
- `models--google--gemma-2-2b` (~5.2 GB) — Gemma-2-2B base model weights.
- `models--google--gemma-scope-2b-pt-res` & `models--google--gemma-scope-2b-pt-res-canonical` (~200 MB) — Gemma Scope SAE weights.
- `models--gpt2` (~500 MB) — GPT-2 base model weights.

### 2. Cleanup Commands

#### PowerShell (Windows):
```powershell
Remove-Item -Recurse -Force "C:\Users\nikun_r4vgzi9\.cache\huggingface\hub\models--google--gemma-2-2b"
Remove-Item -Recurse -Force "C:\Users\nikun_r4vgzi9\.cache\huggingface\hub\models--google--gemma-scope-2b-pt-res*"
Remove-Item -Recurse -Force "C:\Users\nikun_r4vgzi9\.cache\huggingface\hub\models--gpt2"
```

#### Bash (WSL / Linux):
```bash
rm -rf ~/.cache/huggingface/hub/models--google--gemma-2-2b
rm -rf ~/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res*
rm -rf ~/.cache/huggingface/hub/models--gpt2
```

---

## What to Keep (Lightweight Results Only)
Do **NOT** delete `data/results/`. These files contain the permanent, reproducible experimental outputs (< 1 MB total):
- `data/results/part_a_raw.csv`
- `data/results/benchmark_scores.csv`
- `data/results/session_manifest.json`
- `data/results/demo_traces/*.json`
