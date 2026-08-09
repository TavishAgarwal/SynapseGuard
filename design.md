# Design.md — UI Details

## Scope of UI
Two interfaces are needed. Both are secondary to the research/technical proof — keep them functional and legible over polished, per hackathon guidance ("do not build the largest project").

### 1. Part A — Research Output (static, not interactive)
- Not a UI in the traditional sense — output is a set of plots.
- **Chart 1:** Scatter/line plot, x-axis = measured predictability (next-token entropy, low→high), y-axis = SAE-latent sparsity/density. One series per model (LLaMA-3, Qwen).
- **Chart 2:** Correlation summary table (coefficient, p-value, sample size, per prompt category).
- Style: clean, publication-style (matplotlib/plotly), labeled axes, legend, no unnecessary decoration. This is a research artifact, not a marketing asset — clarity over flash.

### 2. Part B — Live Interception Dashboard
**Purpose:** Show, in real time, the PSC Score during generation, and visually flag risk before a hallucinated token is finalized.

**Layout (single screen, no navigation needed):**
┌─────────────────────────────────────────────┐
│ SynapseGuard — Live Diagnostic │
├─────────────────────────────────────────────┤
│ Prompt: [user input box] │
│ ┌─────────────────────────────────────────┐ │
│ │ Generated text streams here, token by │ │
│ │ token. Tokens flagged risky turn RED. │ │
│ └─────────────────────────────────────────┘ │
│ │
│ PSC Gauge: [ ▓▓▓▓▓▓░░░░ ] 0.62 WARNING │
│ (color: green = SAFE, amber = WARNING, │
│ red = HALT) │
│ │
│ [ SSP Triggered: Yes/No ] │
│ [ Status: Generation halted — risk detected]│
└─────────────────────────────────────────────┘

**States:**
- **SAFE** (green): PSC below warning threshold, generation proceeds normally.
- **WARNING** (amber): PSC entering risk band, SSP perturbation check triggered.
- **HALT** (red): PSC confirms high risk post-SSP, token generation visually intercepted (flash + red text), consistent with the "Instantaneous Interception" pitch.

**Interaction:**
- Minimal — a text input for the prompt, a "Generate" button, and the live gauge/stream. No settings panel needed for v1; hardcode thresholds in config, expose as a stretch goal only if time allows.
- If terminal-based instead of web: use a TUI library (e.g., Python `rich`/`textual`) with equivalent panels (prompt, streamed text with color-coded risk, gauge as a progress-bar-style widget).

## Visual Style
- Dark background (fits the "brain activity monitor" framing without being gimmicky).
- Use color sparingly and functionally (green/amber/red only for status — do not over-decorate).
- No BDH/dragon iconography needed — keep it looking like a legitimate diagnostic tool, not hackathon theming, to reinforce technical credibility for judges.

## What NOT to build
- No user accounts, no persistence/history UI, no multi-session support — out of scope for v1.
- No mobile-responsive design needed unless demo requires it.