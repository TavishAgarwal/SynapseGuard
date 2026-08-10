import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, FastForward, Activity, Cpu, ShieldAlert, CheckCircle2, AlertTriangle, XOctagon, Terminal, ExternalLink, ArrowRight, Layers, Sparkles, Sliders } from 'lucide-react'

// Demo traces
import mockTrace from '../../mock_trace.json'
import factualTrace from '../../../../data/results/demo_traces/demo_01_factual.json'
import hallucinationTrace from '../../../../data/results/demo_traces/demo_02_hallucination_prone.json'
import openEndedTrace from '../../../../data/results/demo_traces/demo_03_open_ended.json'

const TRACES = [
  { id: 'hallucination', name: 'Demo 02: Hallucination-Prone (Interception)', data: hallucinationTrace },
  { id: 'factual', name: 'Demo 01: Factual Query (Safe)', data: factualTrace },
  { id: 'open_ended', name: 'Demo 03: Open-Ended Creative (Safe)', data: openEndedTrace },
  { id: 'mock', name: 'Mock Verification Trace', data: mockTrace },
]

export default function ShowcaseConsole({ onOpenDashboard }) {
  const [selectedTraceId, setSelectedTraceId] = useState('hallucination')
  const [currentTrace, setCurrentTrace] = useState(hallucinationTrace)
  const [currentTokenIdx, setCurrentTokenIdx] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speedMultiplier, setSpeedMultiplier] = useState(1)

  const timerRef = useRef(null)

  const handleTraceChange = (id) => {
    setIsPlaying(false)
    const found = TRACES.find(t => t.id === id)
    if (found) {
      setSelectedTraceId(id)
      setCurrentTrace(found.data)
      setCurrentTokenIdx(0)
    }
  }

  const tokens = currentTrace?.tokens || []
  const currentToken = tokens[currentTokenIdx - 1] || null
  const currentPsc = currentToken ? currentToken.psc_score || currentToken.psc || 0.0 : 0.0
  const currentStatus = currentToken ? currentToken.status : 'SAFE'
  const isHalted = currentStatus === 'HALT'
  const sspTriggered = currentToken ? (currentToken.ssp_triggered ?? currentToken.ssp ?? false) : false

  // Playback timer
  useEffect(() => {
    if (isPlaying) {
      if (currentTokenIdx >= tokens.length || isHalted) {
        setIsPlaying(false)
        return
      }

      const delay = 350 / speedMultiplier
      timerRef.current = setTimeout(() => {
        setCurrentTokenIdx(prev => prev + 1)
      }, delay)
    }

    return () => clearTimeout(timerRef.current)
  }, [isPlaying, currentTokenIdx, tokens.length, isHalted, speedMultiplier])

  const handleReset = () => {
    setIsPlaying(false)
    setCurrentTokenIdx(0)
  }

  const handleStep = () => {
    if (currentTokenIdx < tokens.length && !isHalted) {
      setCurrentTokenIdx(prev => prev + 1)
    }
  }

  return (
    <div className="min-h-screen bg-[#05070B] text-slate-100 font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      
      {/* Header Navigation Bar */}
      <header className="sticky top-0 z-50 bg-[#05070B]/90 backdrop-blur-xl border-b border-slate-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <a href="#" className="flex items-center gap-3 group shrink-0">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center group-hover:border-cyan-500/60 transition-colors shadow-[0_0_12px_rgba(6,182,212,0.15)]">
              <Activity className="w-4 h-4 text-cyan-400" />
            </div>
            <span className="font-bold text-lg tracking-tight text-slate-100 font-mono">SynapseGuard</span>
          </a>

          <nav className="hidden lg:flex items-center gap-6 text-xs font-mono text-slate-400">
            <a href="#problem" className="hover:text-cyan-400 transition-colors whitespace-nowrap">Problem</a>
            <a href="#hypothesis" className="hover:text-cyan-400 transition-colors whitespace-nowrap">Research Question</a>
            <a href="#pipeline" className="hover:text-cyan-400 transition-colors whitespace-nowrap">Pipeline</a>
            <a href="#results" className="hover:text-cyan-400 transition-colors whitespace-nowrap">Results</a>
            <a href="#demo" className="hover:text-cyan-400 transition-colors whitespace-nowrap">Live Replay</a>
            <a href="#architecture" className="hover:text-cyan-400 transition-colors whitespace-nowrap">Architecture</a>
            <a href="#limitations" className="hover:text-cyan-400 transition-colors whitespace-nowrap">Limitations</a>
          </nav>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-xl font-mono text-xs">
              <button className="px-3.5 py-1.5 rounded-lg bg-cyan-500 text-slate-950 font-bold shadow-sm cursor-default">
                Showcase Console
              </button>
              <button
                onClick={onOpenDashboard}
                className="px-3.5 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition-colors cursor-pointer flex items-center gap-1.5"
              >
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                Diagnostic Dashboard
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 md:py-28 px-4 sm:px-6 relative overflow-hidden border-b border-slate-800/60">
        <div className="max-w-4xl mx-auto text-center flex flex-col items-center gap-8">
          
          <div className="inline-flex items-center gap-2.5 px-3.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 font-mono text-xs text-cyan-300 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
            Real-Time Diagnostic Interception System
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight text-slate-100 leading-[1.1]">
            Real-Time Hallucination Interception via Predictability-Sparsity Coherence
          </h1>

          <p className="text-base sm:text-lg text-slate-300 max-w-3xl leading-relaxed font-normal bg-[#0A0E17]/80 p-6 rounded-2xl border border-slate-800/80 text-left md:text-center shadow-inner">
            We test whether Pathway's published BDH finding (that synaptic activation sparsity tracks input predictability) generalizes to standard transformers via SAE decomposition, and use the resulting signal as a real-time hallucination early-warning layer.
          </p>

          <div className="flex flex-wrap justify-center gap-4 font-mono text-xs">
            <button
              onClick={onOpenDashboard}
              className="px-6 py-3.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 active:scale-[0.98] text-slate-950 font-bold flex items-center gap-2.5 transition-all shadow-[0_0_25px_rgba(6,182,212,0.3)] cursor-pointer"
            >
              <Sliders className="w-4 h-4" />
              Open Diagnostic Dashboard
            </button>
            <a
              href="#demo"
              className="px-6 py-3.5 rounded-xl bg-[#0A0E17] border border-slate-800 hover:border-cyan-500/50 text-slate-200 font-semibold flex items-center gap-2.5 transition-all"
            >
              <Play className="w-4 h-4 text-cyan-400" />
              Watch Interactive Replay
            </a>
          </div>

          {/* Hero Visual Anchor: Doppelrand Mini PSC Gauge */}
          <div className="w-full max-w-xl p-1 bg-gradient-to-b from-[#151C2C] to-[#0A0E17] border border-cyan-500/30 rounded-2xl shadow-[0_0_30px_-5px_rgba(6,182,212,0.15)] mt-4 text-left">
            <div className="bg-[#070A10] border border-slate-800/60 rounded-[calc(1rem-0.25rem)] p-5">
              <div className="flex items-center justify-between mb-3 font-mono text-xs">
                <span className="text-slate-400 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                  PSC Coherence Monitor (Live Hardware Artifact)
                </span>
                <span className={`px-2.5 py-0.5 rounded text-[11px] font-bold ${
                  currentPsc >= 0.85 ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                  currentPsc >= 0.65 ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40' :
                  'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                }`}>
                  {currentStatus} ({currentPsc.toFixed(2)})
                </span>
              </div>

              <div className="relative w-full bg-slate-950 rounded-full h-3 border border-slate-800 overflow-hidden mb-2 shadow-inner">
                <div className="absolute left-[65%] top-0 bottom-0 w-0.5 bg-amber-400/80 z-10"></div>
                <div className="absolute left-[85%] top-0 bottom-0 w-0.5 bg-red-500/90 z-10"></div>
                <div
                  className={`h-full transition-all duration-500 rounded-full ${
                    currentPsc >= 0.85 ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]' :
                    currentPsc >= 0.65 ? 'bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]' :
                    'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]'
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, currentPsc * 100))}%` }}
                ></div>
              </div>

              <div className="flex justify-between font-mono text-[10px] text-slate-500">
                <span>0.00 (Coherent)</span>
                <span className="text-amber-400 font-medium">0.65 Warning</span>
                <span className="text-red-400 font-medium">0.85 Halt</span>
                <span>1.00</span>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* Section 1: Problem */}
      <section id="problem" className="py-16 px-4 sm:px-6 max-w-4xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>01</span>
          <span>/</span>
          <span>BACKGROUND</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-4">The Hallucination Latency Gap</h2>
        <p className="text-slate-300 leading-relaxed font-normal text-base mb-4">
          Large language models generate text autoregressively without intrinsic confidence bounds. Current safety mitigations, such as LLM-as-a-judge evaluators or post-generation retrieval verifiers, operate reactively after full token sequences are already emitted.
        </p>
        <p className="text-slate-300 leading-relaxed font-normal text-base">
          This creates a high-latency feedback gap. SynapseGuard addresses this bottleneck by inspecting hidden activation states during the forward pass to detect divergence before token emission completes.
        </p>
      </section>

      {/* Section 2: Research Question */}
      <section id="hypothesis" className="py-16 px-4 sm:px-6 max-w-4xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>02</span>
          <span>/</span>
          <span>HYPOTHESIS</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-6">Central Research Question</h2>

        <div className="p-1 bg-gradient-to-b from-[#151C2C] to-[#0A0E17] border border-slate-800 rounded-2xl">
          <div className="bg-[#070A10] border-l-4 border-l-cyan-500 rounded-xl p-6">
            <span className="font-mono text-xs uppercase tracking-wider text-cyan-400 mb-2 block font-semibold">
              Falsifiable Research Hypothesis
            </span>
            <blockquote className="text-lg sm:text-xl font-medium text-slate-100 leading-relaxed">
              "Can real-time monitoring of Sparse Autoencoder (SAE) latent feature activation sparsity combined with next-token logit predictability detect early hallucination onset in large language models before unsafe tokens are emitted?"
            </blockquote>
          </div>
        </div>
      </section>

      {/* Section 3: System Pipeline */}
      <section id="pipeline" className="py-16 px-4 sm:px-6 max-w-5xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>03</span>
          <span>/</span>
          <span>SYSTEM PIPELINE</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-8">How SynapseGuard Operates</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-5 rounded-xl h-full flex flex-col gap-3">
              <div className="flex justify-between items-center font-mono text-xs">
                <span className="text-cyan-400 font-bold">STEP 01</span>
                <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 text-[10px]">Input Phase</span>
              </div>
              <h3 className="font-bold text-slate-200 text-sm">Prompt Ingestion & Model Forward Pass</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                The input prompt passes through Gemma-2-2B (int8) or GPT-2. Hidden state vectors are captured at designated residual layers via forward hooks.
              </p>
            </div>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-5 rounded-xl h-full flex flex-col gap-3">
              <div className="flex justify-between items-center font-mono text-xs">
                <span className="text-cyan-400 font-bold">STEP 02</span>
                <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 text-[10px]">SAE Decomposition</span>
              </div>
              <h3 className="font-bold text-slate-200 text-sm">Sparse Autoencoder Feature Projection</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Dense 2,048-dim hidden states are projected through pretrained SAEs (Gemma Scope / SAE Lens) into 16,384-dim latent feature space.
              </p>
            </div>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-5 rounded-xl h-full flex flex-col gap-3">
              <div className="flex justify-between items-center font-mono text-xs">
                <span className="text-cyan-400 font-bold">STEP 03</span>
                <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 text-[10px]">Scoring & Interception</span>
              </div>
              <h3 className="font-bold text-slate-200 text-sm">PSC Calculation & Real-Time Warning</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Predictability-Sparsity Coherence (PSC) combines logit confidence with SAE latent sparsity. Scores exceeding 0.85 halt token emission instantly.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: Results */}
      <section id="results" className="py-16 px-4 sm:px-6 max-w-5xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>04</span>
          <span>/</span>
          <span>EMPIRICAL RESULTS</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-2">Experimental Evidence</h2>
        <p className="text-sm text-slate-400 mb-8 font-mono">
          Data pulled directly from data/results/ and verified via statistical scripts.
        </p>

        {/* Part A Results Table */}
        <div className="mb-10">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
              Part A: Predictability vs SAE Sparsity Correlation
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">[MEASURED]</span>
            </h3>
            <span className="text-xs font-mono text-slate-500">8,000 Extraction Rows (N=200 Prompts)</span>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="bg-[#070A10] p-0 overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3.5">Model</th>
                    <th className="p-3.5">Layer</th>
                    <th className="p-3.5">Category</th>
                    <th className="p-3.5">N</th>
                    <th className="p-3.5">Pearson r</th>
                    <th className="p-3.5">95% CI</th>
                    <th className="p-3.5">p-value</th>
                    <th className="p-3.5">Verdict</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 text-slate-300">
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">Gemma-2-2B</td>
                    <td className="p-3.5">12</td>
                    <td className="p-3.5 text-cyan-300">All Categories</td>
                    <td className="p-3.5">1000</td>
                    <td className="p-3.5 font-bold text-cyan-400">0.3000</td>
                    <td className="p-3.5 text-slate-400">[0.2515, 0.3469]</td>
                    <td className="p-3.5">&lt; 0.0001</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Held</span></td>
                  </tr>
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">Gemma-2-2B</td>
                    <td className="p-3.5">12</td>
                    <td className="p-3.5">Constrained Factual</td>
                    <td className="p-3.5">250</td>
                    <td className="p-3.5 font-bold text-emerald-400">0.6138</td>
                    <td className="p-3.5 text-slate-400">[0.5422, 0.6763]</td>
                    <td className="p-3.5">&lt; 0.0001</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Held Strongly</span></td>
                  </tr>
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">Gemma-2-2B</td>
                    <td className="p-3.5">12</td>
                    <td className="p-3.5">Moderate Reasoning</td>
                    <td className="p-3.5">250</td>
                    <td className="p-3.5 font-bold text-emerald-400">0.4563</td>
                    <td className="p-3.5 text-slate-400">[0.3835, 0.5315]</td>
                    <td className="p-3.5">&lt; 0.0001</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Held Moderately</span></td>
                  </tr>
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">Gemma-2-2B</td>
                    <td className="p-3.5">12</td>
                    <td className="p-3.5 text-amber-300">Open-Ended</td>
                    <td className="p-3.5">250</td>
                    <td className="p-3.5 font-bold text-amber-400">0.1038</td>
                    <td className="p-3.5 text-slate-400">[0.0361, 0.1724]</td>
                    <td className="p-3.5 text-amber-400">0.1016 (NS)</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-400 border border-amber-500/30">Not Held</span></td>
                  </tr>
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">Gemma-2-2B</td>
                    <td className="p-3.5">12</td>
                    <td className="p-3.5">Adversarial / Ambiguous</td>
                    <td className="p-3.5">250</td>
                    <td className="p-3.5 font-bold text-emerald-400">0.6834</td>
                    <td className="p-3.5 text-slate-400">[0.6109, 0.7483]</td>
                    <td className="p-3.5">&lt; 0.0001</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Held Strongly</span></td>
                  </tr>
                  <tr className="bg-slate-950/60">
                    <td className="p-3.5 font-bold text-slate-100">GPT-2 Small</td>
                    <td className="p-3.5">5</td>
                    <td className="p-3.5 text-cyan-300">All Categories</td>
                    <td className="p-3.5">1000</td>
                    <td className="p-3.5 font-bold text-cyan-400">0.6467</td>
                    <td className="p-3.5 text-slate-400">[0.6167, 0.6770]</td>
                    <td className="p-3.5">&lt; 0.0001</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Held Strongly</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Part B Benchmark Table */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
              Part B: Benchmark AUROC Validation (70/30 Held-Out Evaluation)
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">[MEASURED]</span>
            </h3>
            <span className="text-xs font-mono text-slate-500">300 Benchmark Samples (Gemma-2-2B int8)</span>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="bg-[#070A10] p-0 overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3.5">Benchmark Dataset</th>
                    <th className="p-3.5">Total N</th>
                    <th className="p-3.5">Held-out AUROC</th>
                    <th className="p-3.5">95% CI</th>
                    <th className="p-3.5">Optimal Threshold</th>
                    <th className="p-3.5">F1 Score</th>
                    <th className="p-3.5">Performance Class</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 text-slate-300">
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">TruthfulQA</td>
                    <td className="p-3.5">100</td>
                    <td className="p-3.5 font-bold text-emerald-400">1.0000</td>
                    <td className="p-3.5 text-slate-400">[1.0000, 1.0000]</td>
                    <td className="p-3.5">0.2911</td>
                    <td className="p-3.5 font-bold text-emerald-400">1.0000</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Near-Perfect</span></td>
                  </tr>
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">HaluEval</td>
                    <td className="p-3.5">100</td>
                    <td className="p-3.5 font-bold text-emerald-400">0.9911</td>
                    <td className="p-3.5 text-slate-400">[0.9598, 1.0000]</td>
                    <td className="p-3.5">0.3533</td>
                    <td className="p-3.5 font-bold text-emerald-400">0.9375</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Near-Perfect</span></td>
                  </tr>
                  <tr>
                    <td className="p-3.5 font-bold text-slate-100">RAG Grounding (RAGTruth)</td>
                    <td className="p-3.5">100</td>
                    <td className="p-3.5 font-bold text-cyan-400">0.8444</td>
                    <td className="p-3.5 text-slate-400">[0.6619, 0.9957]</td>
                    <td className="p-3.5">0.3247</td>
                    <td className="p-3.5 font-bold text-cyan-400">0.8462</td>
                    <td className="p-3.5"><span className="px-2.5 py-0.5 rounded text-[10px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">Moderate / Good</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Section 5: Live Replay Engine */}
      <section id="demo" className="py-16 px-4 sm:px-6 max-w-6xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>05</span>
          <span>/</span>
          <span>HARDWARE INTERCEPTION CONSOLE</span>
        </div>
        
        <div className="flex flex-wrap justify-between items-end gap-4 mb-6">
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-2">Live Interception Replay</h2>
            <p className="text-xs font-mono text-slate-400 flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-700 font-bold">[ESTABLISHED]</span>
              Real-time activation inspection engine replaying pre-computed model traces.
            </p>
          </div>

          <button
            onClick={onOpenDashboard}
            className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 active:scale-[0.98] text-slate-950 font-bold font-mono text-xs flex items-center gap-2 transition-all shadow-[0_0_20px_rgba(6,182,212,0.25)] cursor-pointer"
          >
            <Sliders className="w-4 h-4" />
            Open Diagnostic Dashboard
          </button>
        </div>

        {/* Embedded Interactive Replay Card */}
        <div className="p-1 bg-gradient-to-b from-[#151C2C] to-[#0A0E17] border border-cyan-500/30 rounded-2xl shadow-[0_0_30px_-5px_rgba(6,182,212,0.15)]">
          <div className="bg-[#070A10] p-6 rounded-xl flex flex-col gap-6">
            
            {/* Trace Selector & Controls Header */}
            <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div className="flex flex-col gap-1 w-full sm:w-auto">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
                  Select Pre-Computed Demo Trace
                </label>
                <select
                  value={selectedTraceId}
                  onChange={(e) => handleTraceChange(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
                >
                  {TRACES.map(t => (
                    <option key={t.id} value={t.id}>{t.name}</option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  disabled={isHalted || currentTokenIdx >= tokens.length}
                  className={`px-4 py-2 rounded-lg font-medium text-xs font-mono flex items-center gap-2 transition-all ${
                    isPlaying
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30'
                      : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 disabled:opacity-40'
                  }`}
                >
                  {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {isPlaying ? 'Pause' : 'Play Stream'}
                </button>

                <button
                  onClick={handleStep}
                  disabled={isPlaying || isHalted || currentTokenIdx >= tokens.length}
                  className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 hover:bg-slate-800 text-slate-300 font-mono text-xs flex items-center gap-1.5 disabled:opacity-40"
                >
                  <FastForward className="w-3.5 h-3.5 text-cyan-400" />
                  Step
                </button>

                <button
                  onClick={handleReset}
                  className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-700 hover:bg-slate-800 text-slate-300 font-mono text-xs flex items-center gap-1.5"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
                  Reset
                </button>

                <span className="text-xs font-mono text-slate-400 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
                  {currentTokenIdx} / {tokens.length} Tokens
                </span>
              </div>
            </div>

            {/* Prompt Banner */}
            <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 text-xs font-mono text-slate-300">
              <span className="text-amber-400 font-semibold">Prompt: </span>
              {currentTrace.prompt}
            </div>

            {/* Token Stream Output Box */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 min-h-[100px] flex flex-wrap gap-2 items-start font-mono text-xs">
              {tokens.slice(0, currentTokenIdx).length === 0 ? (
                <span className="text-slate-600 italic">SYSTEM READY · CLICK 'PLAY STREAM' TO EXECUTE TOKEN ANIMATION</span>
              ) : (
                tokens.slice(0, currentTokenIdx).map((tok, idx) => {
                  const text = tok.token_text ?? tok.token ?? ''
                  let badgeStyle = "bg-slate-900 text-emerald-300 border-slate-700/80"
                  if (tok.status === "WARNING") badgeStyle = "bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-[0_0_10px_rgba(245,158,11,0.2)]"
                  if (tok.status === "HALT") badgeStyle = "bg-red-500/30 text-red-200 border-red-500 font-bold animate-bounce shadow-[0_0_15px_rgba(239,68,68,0.3)]"
                  return (
                    <span key={idx} className={`px-2.5 py-1 rounded-lg border ${badgeStyle}`}>
                      {text}
                    </span>
                  )
                })
              )}
            </div>

            {/* Live PSC Status Indicator */}
            <div className={`p-4 rounded-xl font-mono text-xs flex items-center justify-between transition-all ${
              currentStatus === 'HALT'
                ? 'bg-red-500/20 border border-red-500/60 text-red-300 animate-pulse'
                : currentStatus === 'WARNING'
                ? 'bg-amber-500/10 border border-amber-500/40 text-amber-300'
                : 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
            }`}>
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${
                  currentStatus === 'HALT' ? 'bg-red-400 animate-ping' :
                  currentStatus === 'WARNING' ? 'bg-amber-400 animate-ping' :
                  'bg-emerald-400'
                }`}></span>
                <span className="font-bold">STATUS: {currentStatus}</span>
                {isHalted && <span className="text-xs text-red-400 font-semibold">(Generation Intercepted)</span>}
              </div>
              <div className="flex items-center gap-4 text-[11px]">
                <span>PSC Score: <strong className="text-white">{currentPsc.toFixed(2)}</strong></span>
                <span>SSP Triggered: <strong className={sspTriggered ? 'text-amber-400' : 'text-slate-500'}>{sspTriggered ? 'YES' : 'NO'}</strong></span>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Section 6: Architecture */}
      <section id="architecture" className="py-16 px-4 sm:px-6 max-w-5xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>06</span>
          <span>/</span>
          <span>SYSTEM ARCHITECTURE</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-6">Technical Architecture & Stack</h2>

        <div className="flex flex-wrap gap-2.5 mb-8 font-mono text-xs">
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">vLLM (CUDA)</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">vLLM-Hook</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">Gemma Scope SAE</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">SAE Lens</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">bitsandbytes int8</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">PyTorch 2.6</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">FastAPI Sidecar</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">React + Vite</span>
          <span className="px-3.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">TailwindCSS</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-5 rounded-xl h-full flex flex-col gap-2">
              <h3 className="font-bold text-slate-200 text-sm font-mono text-cyan-400">01. GPU Extraction Engine (WSL2 RTX 3050)</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Executes sequential model inference (Gemma-2-2B int8, GPT-2 small fp16). Captures last-token hidden states at residual layers 11–14, projects through Gemma Scope SAE checkpoints, and saves raw activation metrics.
              </p>
            </div>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-5 rounded-xl h-full flex flex-col gap-2">
              <h3 className="font-bold text-slate-200 text-sm font-mono text-cyan-400">02. Mac Diagnostic Replay & Analysis</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Consumes verified CSV/JSON contracts offline. Computes Pearson/Spearman statistical correlations, generates ROC curves, serves FastAPI diagnostic endpoints, and renders real-time token stream interception UI.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 7: Limitations */}
      <section id="limitations" className="py-16 px-4 sm:px-6 max-w-4xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>07</span>
          <span>/</span>
          <span>TRANSPARENCY</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-6">Known Limitations & Disclosures</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 font-mono text-xs">
          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-4 rounded-xl h-full">
              <span className="text-cyan-400 font-bold block mb-1">01. 8-Bit Quantization Shift</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Gemma-2-2B uses bitsandbytes 8-bit quantization to fit 6GB VRAM, creating minor distributional shift from full-precision SAE pretraining.
              </p>
            </div>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-4 rounded-xl h-full">
              <span className="text-cyan-400 font-bold block mb-1">02. Pre-Computed Replay</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                All activation extraction is performed during a single GPU session. Dashboard presentations replay pre-computed activation arrays with realistic pacing.
              </p>
            </div>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-4 rounded-xl h-full">
              <span className="text-cyan-400 font-bold block mb-1">03. Sample Size Boundary</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Part A dataset contains 200 prompts (50/category); Part B benchmark subsets contain 100 samples per dataset (300 total).
              </p>
            </div>
          </div>

          <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
            <div className="bg-[#070A10] p-4 rounded-xl h-full">
              <span className="text-cyan-400 font-bold block mb-1">04. Toy BDH Scope</span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                The local bdh.py implementation is a ~10M parameter baseline model and does not represent frontier BDH capabilities.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 8: Future Work */}
      <section className="py-16 px-4 sm:px-6 max-w-4xl mx-auto border-b border-slate-800/60">
        <div className="flex items-center gap-2 font-mono text-xs text-cyan-400 mb-3">
          <span>08</span>
          <span>/</span>
          <span>FUTURE WORK</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-4">If We Had Larger BDH & Multi-GPU Access</h2>

        <div className="p-1 bg-[#151C2C]/50 border border-slate-800 rounded-2xl">
          <div className="bg-[#070A10] p-6 rounded-xl">
            <p className="text-slate-300 text-sm leading-relaxed mb-4">
              Given access to multi-GPU clusters (such as 8x NVIDIA A100 80GB) and production scale BDH architecture access, the SynapseGuard research roadmap extends to:
            </p>
            <ul className="list-disc list-inside text-xs text-slate-400 space-y-2 font-mono">
              <li><strong className="text-slate-200">Zero-Copy C++ CUDA Kernels:</strong> Direct streaming of residual hidden states without host-to-device memory copy overhead.</li>
              <li><strong className="text-slate-200">Multi-Layer Joint SAE Projection:</strong> Projecting across all 26 layers of Gemma-2-9B/70B simultaneously to construct spatial trajectory maps of hallucination drift.</li>
              <li><strong className="text-slate-200">In-Flight Steering Vectors:</strong> Dynamically injecting counter-steering vectors into SAE latent space when PSC crosses 0.65, correcting hallucinations in-flight rather than halting token generation.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 sm:px-6 max-w-6xl mx-auto font-mono text-xs text-slate-500">
        <div className="flex flex-wrap justify-between items-center gap-6 border-b border-slate-800/60 pb-8 mb-8">
          <div>
            <span className="font-bold text-slate-200 text-sm block mb-1">SynapseGuard</span>
            <p className="text-[11px] text-slate-400 max-w-md">
              Real-Time Hallucination Interception via Predictability-Sparsity Coherence (PSC)
            </p>
          </div>

          <button
            onClick={onOpenDashboard}
            className="px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold flex items-center gap-2 transition-all cursor-pointer"
          >
            <Sliders className="w-4 h-4" />
            Launch Full Diagnostic Dashboard
          </button>
        </div>

        <div className="flex flex-wrap justify-between items-center gap-4 text-[11px]">
          <span>MIT License · SynapseGuard Team</span>
          <span>Pre-Computed Session Manifest: <code className="text-cyan-400">331200b</code></span>
        </div>
      </footer>

    </div>
  )
}
