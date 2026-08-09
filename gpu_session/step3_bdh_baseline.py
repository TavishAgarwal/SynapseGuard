"""
step3_bdh_baseline.py — BDH Baseline Step Wrapper

Runs during the GPU session orchestrator.
Note per architecture.md Section 6:
BDH baseline instrumentation (bdh.py, ~10M params) runs on CPU/MPS on the Mac.
This step verifies local baseline findings or acts as a clean wrapper during the GPU session.
"""

import os
import logging

logger = logging.getLogger("gpu_session.step3")

def run_bdh_baseline_step(mock_mode: bool = False):
    """BDH baseline wrapper step."""
    logger.info("Step 3: BDH Baseline step called.")
    baseline_notes_path = "docs/bdh_baseline_notes.md"
    plot_path = "analysis/plots/bdh_baseline_sparsity.png"
    
    if os.path.exists(baseline_notes_path) and os.path.exists(plot_path):
        logger.info(f"BDH baseline instrumentation verified on Mac: {baseline_notes_path} & {plot_path}")
        return {"status": "completed_on_mac", "notes": baseline_notes_path, "plot": plot_path}
    else:
        logger.info("Executing local BDH baseline measurement...")
        from analysis.bdh_baseline import run_bdh_baseline_experiment
        run_bdh_baseline_experiment()
        return {"status": "executed_locally"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_bdh_baseline_step(mock_mode=True)
    print("Step 3 Dry Run Result:", res)
