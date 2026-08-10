import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, FastForward, ShieldAlert, Cpu, CheckCircle2, AlertTriangle, XOctagon, Activity, ArrowLeft, Terminal } from 'lucide-react'

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

export default function DiagnosticDashboard({ onBackToShowcase }) {
  const [selectedTraceId, setSelectedTraceId] = useState('hallucination')
  const [currentTrace, setCurrentTrace] = useState(hallucinationTrace)
  const [currentTokenIdx, setCurrentTokenIdx] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speedMultiplier, setSpeedMultiplier] = useState(1)
  
  const timerRef = useRef(null)

  // Switch trace
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

  // Playback timer loop
  useEffect(() => {
    if (isPlaying) {
      if (currentTokenIdx >= tokens.length || isHalted) {
        setIsPlaying(false)
        return
      }

      const delay = (350 / speedMultiplier)
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
    <div className="min-h-screen bg-[#0A0C10] text-slate-100 p-4 md:p-6 lg:p-8 flex flex-col gap-6 max-w-[1600px] mx-auto font-sans">
      
      {/* 1. Header Bar */}
      <header className="glass-panel p-4 rounded-xl flex flex-wrap justify-between items-center gap-4 border border-slate-800">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-xl font-mono text-xs">
            <button
              onClick={onBackToShowcase}
              className="px-3.5 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition-colors cursor-pointer flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5 text-cyan-400" />
              Showcase Console
            </button>
            <button className="px-3.5 py-1.5 rounded-lg bg-cyan-500 text-slate-950 font-bold shadow-sm cursor-default">
              Diagnostic Dashboard
            </button>
          </div>

          <div className="flex items-center gap-3 border-l border-slate-800 pl-4">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
              <Activity className="w-5 h-5 text-emerald-400 animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-slate-100 via-slate-200 to-slate-400 bg-clip-text text-transparent font-mono">
                SynapseGuard
              </h1>
              <p className="text-xs text-slate-400 font-mono">
                Predictability-Sparsity Coherence (PSC) Diagnostic Interception Layer
              </p>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          <span className="px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            RTX 3050 (6GB VRAM)
          </span>
          <span className="px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-300">
            Gemma-2-2B (8-bit int8)
          </span>
          <span className="px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold">
            Single GPU Session Active
          </span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        
        {/* Left Column: Controls & Text Stream (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          
          {/* Prompt Selector & Controls Panel */}
          <div className="glass-panel p-5 rounded-xl border border-slate-800 flex flex-col gap-4">
            <div className="flex flex-wrap justify-between items-center gap-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
                Select Pre-Computed Demo Trace
              </label>
              <span className="text-xs text-slate-500 font-mono">
                {currentTokenIdx} / {tokens.length} Tokens
              </span>
            </div>

            <select
              value={selectedTraceId}
              onChange={(e) => handleTraceChange(e.target.value)}
              className="w-full bg-slate-900/90 border border-slate-700/80 rounded-lg px-4 py-2.5 text-sm font-mono text-slate-200 focus:outline-none focus:border-cyan-500 transition-colors"
            >
              {TRACES.map(t => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>

            <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 text-xs font-mono text-slate-300">
              <span className="text-amber-400 font-semibold">Prompt: </span>
              {currentTrace.prompt}
            </div>

            {/* Playback Button Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  disabled={isHalted || currentTokenIdx >= tokens.length}
                  className={`px-5 py-2 rounded-lg font-medium text-xs flex items-center gap-2 transition-all cursor-pointer ${
                    isPlaying
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30'
                      : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 disabled:opacity-40 disabled:cursor-not-allowed'
                  }`}
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isPlaying ? 'Pause' : 'Play Trace Stream'}
                </button>

                <button
                  onClick={handleStep}
                  disabled={isPlaying || isHalted || currentTokenIdx >= tokens.length}
                  className="px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-700 hover:bg-slate-800 text-slate-300 font-mono text-xs flex items-center gap-1.5 disabled:opacity-40 transition-colors cursor-pointer"
                >
                  <FastForward className="w-3.5 h-3.5 text-cyan-400" />
                  Step
                </button>

                <button
                  onClick={handleReset}
                  className="px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-700 hover:bg-slate-800 text-slate-300 font-mono text-xs flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
                  Reset
                </button>
              </div>

              {/* Speed Multiplier Select */}
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="text-slate-500">Speed:</span>
                {[1, 2, 4].map(s => (
                  <button
                    key={s}
                    onClick={() => setSpeedMultiplier(s)}
                    className={`px-2.5 py-1 rounded text-xs transition-colors ${
                      speedMultiplier === s
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 font-bold'
                        : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Token Stream Output Window */}
          <div className="glass-panel p-5 rounded-xl border border-slate-800 flex flex-col gap-3 min-h-[220px]">
            <div className="flex justify-between items-center font-mono text-xs">
              <span className="text-slate-400 uppercase tracking-wider font-semibold flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                Live Generated Stream
              </span>
              <span className="text-slate-500 text-[11px]">
                Token inspection at Layer 12
              </span>
            </div>

            <div className="bg-slate-950/90 border border-slate-800/90 rounded-lg p-4 font-mono text-sm leading-relaxed flex flex-wrap gap-1.5 items-start flex-1 min-h-[140px]">
              {/* Always display prompt tokens header */}
              <div className="w-full pb-2 mb-2 border-b border-slate-800/80 text-xs font-mono">
                <span className="text-amber-400 font-bold uppercase tracking-wider text-[11px] block mb-1">Input Prompt Sequence:</span>
                <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-200 inline-block font-semibold">
                  {currentTrace.prompt}
                </span>
              </div>

              {tokens.slice(0, currentTokenIdx).length === 0 ? (
                <span className="text-slate-600 italic text-xs w-full py-2">
                  SYSTEM READY · Click 'Play Trace Stream' to launch real-time token emission...
                </span>
              ) : (
                tokens.slice(0, currentTokenIdx).map((tok, idx) => {
                  const text = tok.token_text ?? tok.token ?? ''
                  const conf = tok.logit_confidence ?? tok.conf ?? 0.0
                  const psc = tok.psc_score ?? tok.psc ?? 0.0
                  let badgeStyle = "bg-slate-900/90 text-emerald-300 border-slate-700/80"
                  if (tok.status === "WARNING") badgeStyle = "bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-[0_0_10px_rgba(245,158,11,0.2)]"
                  if (tok.status === "HALT") badgeStyle = "bg-red-500/30 text-red-200 border-red-500 font-bold animate-bounce shadow-[0_0_15px_rgba(239,68,68,0.3)]"
                  return (
                    <span
                      key={idx}
                      className={`px-2.5 py-1 rounded-md border text-xs transition-all ${badgeStyle}`}
                      title={`Logit Conf: ${(conf * 100).toFixed(1)}% | PSC: ${psc.toFixed(2)}`}
                    >
                      {text}
                    </span>
                  )
                })
              )}
            </div>
          </div>

        </div>

        {/* Right Column: Gauges, Banners & Telemetry Log Table (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          
          {/* Status Gauge & Banner */}
          <div className="glass-panel p-5 rounded-xl border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between font-mono text-xs">
              <span className="text-slate-400 uppercase tracking-wider font-semibold">
                Predictability-Sparsity Coherence Gauge
              </span>
              <span className={`px-2.5 py-0.5 rounded text-xs font-bold font-mono ${
                currentPsc >= 0.85 ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                currentPsc >= 0.65 ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40' :
                'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              }`}>
                {currentPsc.toFixed(2)}
              </span>
            </div>

            {/* Gauge Progress Bar */}
            <div className="relative w-full bg-slate-950 rounded-full h-4 border border-slate-800 overflow-hidden shadow-inner">
              <div className="absolute left-[65%] top-0 bottom-0 w-0.5 bg-amber-400/80 z-10"></div>
              <div className="absolute left-[85%] top-0 bottom-0 w-0.5 bg-red-500/90 z-10"></div>
              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  currentPsc >= 0.85 ? 'bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.6)]' :
                  currentPsc >= 0.65 ? 'bg-amber-500 shadow-[0_0_12px_rgba(245,158,11,0.5)]' :
                  'bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.5)]'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, currentPsc * 100))}%` }}
              ></div>
            </div>

            <div className="flex justify-between font-mono text-[10px] text-slate-500">
              <span>0.00 (Coherent)</span>
              <span className="text-amber-400">0.65 Warning</span>
              <span className="text-red-400">0.85 Halt</span>
              <span>1.00</span>
            </div>

            {/* Status Banner */}
            <div className={`p-4 rounded-xl font-mono text-xs flex items-center justify-between transition-all ${
              currentStatus === 'HALT'
                ? 'bg-red-500/20 border border-red-500/60 text-red-300 animate-pulse shadow-[0_0_20px_rgba(239,68,68,0.2)]'
                : currentStatus === 'WARNING'
                ? 'bg-amber-500/10 border border-amber-500/40 text-amber-300'
                : 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
            }`}>
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${
                  currentStatus === 'HALT' ? 'bg-red-400 animate-ping' :
                  currentStatus === 'WARNING' ? 'bg-amber-400 animate-ping' :
                  'bg-emerald-400'
                }`}></span>
                <span className="font-bold">STATUS: {currentStatus}</span>
              </div>
              <span className="text-[11px]">
                SSP: <strong className={sspTriggered ? 'text-amber-400' : 'text-slate-500'}>{sspTriggered ? 'TRIGGERED' : 'INACTIVE'}</strong>
              </span>
            </div>
          </div>

          {/* Telemetry Log Table */}
          <div className="glass-panel p-5 rounded-xl border border-slate-800 flex flex-col gap-3 flex-1 overflow-hidden">
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
              Per-Token Telemetry Stream Log
            </h3>

            <div className="overflow-y-auto max-h-[320px] rounded-lg border border-slate-800/80 bg-slate-950/80">
              <table className="w-full text-left font-mono text-xs">
                <thead className="sticky top-0 bg-slate-950 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-2.5">#</th>
                    <th className="p-2.5">Token</th>
                    <th className="p-2.5">Conf</th>
                    <th className="p-2.5">PSC</th>
                    <th className="p-2.5">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {tokens.slice(0, currentTokenIdx).length === 0 ? (
                    <tr>
                      <td colSpan="5" className="p-4 text-center text-slate-600 italic">
                        No tokens emitted yet...
                      </td>
                    </tr>
                  ) : (
                    tokens.slice(0, currentTokenIdx).map((tok, idx) => {
                      const text = tok.token_text ?? tok.token ?? ''
                      const conf = tok.logit_confidence ?? tok.conf ?? 0.0
                      const psc = tok.psc_score ?? tok.psc ?? 0.0
                      return (
                        <tr key={idx} className={tok.status === 'HALT' ? 'bg-red-950/40' : (tok.status === 'WARNING' ? 'bg-amber-950/30' : '')}>
                          <td className="p-2.5 font-bold text-slate-500">{idx + 1}</td>
                          <td className="p-2.5 font-bold text-slate-100">{text}</td>
                          <td className="p-2.5 text-slate-300">{(conf * 100).toFixed(0)}%</td>
                          <td className={`p-2.5 font-bold ${psc >= 0.85 ? 'text-red-400' : (psc >= 0.65 ? 'text-amber-400' : 'text-emerald-400')}`}>
                            {psc.toFixed(2)}
                          </td>
                          <td className="p-2.5">
                            <span className={`px-2 py-0.5 rounded text-[10px] ${
                              tok.status === 'HALT' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                              tok.status === 'WARNING' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40' :
                              'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                            }`}>
                              {tok.status}
                            </span>
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

        </div>

      </div>

    </div>
  )
}
