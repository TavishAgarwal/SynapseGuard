import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, RotateCcw, FastForward, ShieldAlert, Cpu, CheckCircle2, AlertTriangle, XOctagon, Activity, Terminal } from 'lucide-react'

// Demo traces embedded / fallback data
import mockTrace from '../mock_trace.json'
import factualTrace from '../../../data/results/demo_traces/demo_01_factual.json'
import hallucinationTrace from '../../../data/results/demo_traces/demo_02_hallucination_prone.json'
import openEndedTrace from '../../../data/results/demo_traces/demo_03_open_ended.json'

const TRACES = [
  { id: 'mock', name: 'Mock Verification Trace', data: mockTrace },
  { id: 'factual', name: 'Demo 01: Factual Query (Safe)', data: factualTrace },
  { id: 'hallucination', name: 'Demo 02: Hallucination-Prone (Interception)', data: hallucinationTrace },
  { id: 'open_ended', name: 'Demo 03: Open-Ended Creative (Safe)', data: openEndedTrace },
]

export default function App() {
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
  const currentPsc = currentToken ? currentToken.psc_score : 0.0
  const currentStatus = currentToken ? currentToken.status : 'SAFE'
  const isHalted = currentStatus === 'HALT'
  const sspTriggered = currentToken ? currentToken.ssp_triggered : false

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
    <div className="min-h-screen bg-[#0A0C10] text-slate-100 p-4 md:p-6 lg:p-8 flex flex-col gap-6 max-w-[1600px] mx-auto">
      
      {/* 1. Header Bar */}
      <header className="glass-panel p-4 rounded-xl flex flex-wrap justify-between items-center gap-4 border border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
            <Activity className="w-6 h-6 text-emerald-400 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-slate-100 via-slate-200 to-slate-400 bg-clip-text text-transparent">
              SynapseGuard v2.0
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Predictability-Sparsity Coherence (PSC) Live Interception Layer
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          <span className="px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            Hardware: RTX 3050 (6GB VRAM)
          </span>
          <span className="px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-300">
            Model: Gemma-2-2B (8-bit int8)
          </span>
          <span className="px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold">
            Single GPU Session Manifest Active
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
                  className={`px-5 py-2 rounded-lg font-medium text-xs flex items-center gap-2 transition-all ${
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
                  className="px-3.5 py-2 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300 hover:bg-slate-700/80 text-xs font-medium disabled:opacity-40 transition-colors"
                >
                  Step Token
                </button>

                <button
                  onClick={handleReset}
                  className="px-3.5 py-2 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300 hover:bg-slate-700/80 text-xs font-medium flex items-center gap-1.5 transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Reset
                </button>
              </div>

              <div className="flex items-center gap-1.5 font-mono text-xs">
                <span className="text-slate-400 mr-1">Speed:</span>
                {[0.5, 1, 2, 5].map(spd => (
                  <button
                    key={spd}
                    onClick={() => setSpeedMultiplier(spd)}
                    className={`px-2.5 py-1 rounded text-xs transition-colors ${
                      speedMultiplier === spd
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 font-bold'
                        : 'bg-slate-900 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Streamed Output Monitor Box */}
          <div className={`glass-panel p-6 rounded-xl border flex-1 flex flex-col gap-3 min-h-[300px] transition-all duration-300 ${
            isHalted 
              ? 'glass-panel-glow-red border-red-500/60 bg-red-950/20' 
              : currentStatus === 'WARNING'
              ? 'glass-panel-glow-amber border-amber-500/40'
              : 'border-slate-800'
          }`}>
            <div className="flex justify-between items-center border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-semibold uppercase tracking-wider font-mono text-slate-300">
                  Live Token-by-Token Stream Monitor
                </span>
              </div>
              <span className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-full border ${
                currentStatus === 'HALT' ? 'bg-red-500/20 border-red-500 text-red-400 animate-pulse' :
                currentStatus === 'WARNING' ? 'bg-amber-500/20 border-amber-500 text-amber-400' :
                'bg-emerald-500/20 border-emerald-500 text-emerald-400'
              }`}>
                STATUS: {currentStatus}
              </span>
            </div>

            <div className="font-mono text-base leading-relaxed text-slate-200 flex-1 flex flex-wrap items-start content-start gap-1.5 p-2 overflow-y-auto">
              {tokens.slice(0, currentTokenIdx).map((tok, idx) => {
                let badgeStyle = 'bg-slate-800/80 text-emerald-300 border-slate-700'
                if (tok.status === 'WARNING') badgeStyle = 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                if (tok.status === 'HALT') badgeStyle = 'bg-red-500/30 text-red-200 border-red-500 animate-bounce font-bold shadow-lg shadow-red-500/20'

                return (
                  <span
                    key={idx}
                    className={`px-2 py-0.5 rounded border transition-all ${badgeStyle}`}
                    title={`Pos: ${tok.position} | Logit Conf: ${tok.logit_confidence} | PSC: ${tok.psc_score}`}
                  >
                    {tok.token_text}
                  </span>
                )
              })}

              {currentTokenIdx === 0 && (
                <span className="text-slate-500 italic text-sm">
                  Click 'Play Trace Stream' or 'Step Token' to begin real-time generation replay...
                </span>
              )}
            </div>

            {/* Hallucination Interception Flash Notice */}
            {isHalted && (
              <div className="bg-red-950/80 border border-red-500/80 rounded-lg p-3 text-red-200 text-xs font-mono flex items-center gap-3 animate-pulse">
                <ShieldAlert className="w-5 h-5 text-red-400 shrink-0" />
                <div>
                  <p className="font-bold">⚠️ INSTANTANEOUS INTERCEPTION TRIGGERED</p>
                  <p className="text-red-300/90 text-[11px]">
                    PSC score reached {currentPsc} (&ge; 0.85). Generation halted to prevent hallucinated completion.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: PSC Gauge & Diagnostics (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          
          {/* Radial/Bar Gauge Panel */}
          <div className="glass-panel p-6 rounded-xl border border-slate-800 flex flex-col gap-5">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold uppercase tracking-wider font-mono text-slate-400">
                PSC Coherence Gauge
              </span>
              <span className="text-xs font-mono text-slate-500">
                Threshold: Warning 0.65 | Halt 0.85
              </span>
            </div>

            {/* PSC Score Big Display */}
            <div className="text-center py-2">
              <div className={`text-5xl font-mono font-bold tracking-tight transition-colors ${
                currentPsc >= 0.85 ? 'text-red-400' :
                currentPsc >= 0.65 ? 'text-amber-400' : 'text-emerald-400'
              }`}>
                {currentPsc.toFixed(2)}
              </div>
              <span className="text-xs font-mono text-slate-400 uppercase tracking-widest mt-1 inline-block">
                Predictability-Sparsity Coherence
              </span>
            </div>

            {/* Gauge Progress Bar */}
            <div className="relative w-full bg-slate-900 rounded-full h-4 overflow-hidden border border-slate-800">
              {/* Warning marker line at 65% */}
              <div className="absolute left-[65%] top-0 bottom-0 w-0.5 bg-amber-400/70 z-10" />
              {/* Halt marker line at 85% */}
              <div className="absolute left-[85%] top-0 bottom-0 w-0.5 bg-red-500/80 z-10" />

              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  currentPsc >= 0.85 ? 'bg-gradient-to-r from-amber-500 to-red-500' :
                  currentPsc >= 0.65 ? 'bg-gradient-to-r from-emerald-500 to-amber-500' :
                  'bg-emerald-500'
                }`}
                style={{ width: `${Math.min(100, Math.max(5, currentPsc * 100))}%` }}
              />
            </div>

            <div className="flex justify-between text-[11px] font-mono text-slate-400 px-1">
              <span>0.00 (Safe)</span>
              <span className="text-amber-400 font-medium">0.65 (Warning)</span>
              <span className="text-red-400 font-medium">0.85 (Halt)</span>
              <span>1.00</span>
            </div>

            {/* Status Indicator Pill */}
            <div className={`p-4 rounded-xl border flex items-center justify-between font-mono text-xs transition-colors ${
              currentStatus === 'HALT' ? 'bg-red-500/10 border-red-500/40 text-red-300' :
              currentStatus === 'WARNING' ? 'bg-amber-500/10 border-amber-500/40 text-amber-300' :
              'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
            }`}>
              <div className="flex items-center gap-2">
                {currentStatus === 'HALT' && <XOctagon className="w-5 h-5 text-red-400" />}
                {currentStatus === 'WARNING' && <AlertTriangle className="w-5 h-5 text-amber-400" />}
                {currentStatus === 'SAFE' && <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
                <div>
                  <p className="font-bold">STATUS: {currentStatus}</p>
                  <p className="text-[11px] opacity-80">
                    {currentStatus === 'HALT' ? 'High hallucination mismatch' :
                     currentStatus === 'WARNING' ? 'Coherence anomaly detected' : 'Coherence normal'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* SSP Perturbation Status Banner */}
          <div className="glass-panel p-5 rounded-xl border border-slate-800 flex flex-col gap-3 font-mono text-xs">
            <div className="flex justify-between items-center">
              <span className="text-slate-400 uppercase tracking-wider font-semibold">
                SSP Perturbation Trigger
              </span>
              <span className={`px-2.5 py-0.5 rounded text-xs font-bold border ${
                sspTriggered 
                  ? 'bg-amber-500/20 border-amber-500 text-amber-300'
                  : 'bg-slate-900 border-slate-800 text-slate-500'
              }`}>
                {sspTriggered ? 'TRIGGERED: YES' : 'TRIGGERED: NO'}
              </span>
            </div>
            
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Sample-Specific Prompting uses pre-computed perturbation templates when PSC crosses 0.65 to test token stability without high latency.
            </p>
          </div>

          {/* Diagnostic Log Table */}
          <div className="glass-panel p-5 rounded-xl border border-slate-800 flex-1 flex flex-col gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider font-mono text-slate-400">
              Token Diagnostic Log
            </span>

            <div className="overflow-x-auto max-h-[220px] overflow-y-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-slate-900/90 text-slate-400 sticky top-0">
                  <tr>
                    <th className="p-2 border-b border-slate-800">Pos</th>
                    <th className="p-2 border-b border-slate-800">Token</th>
                    <th className="p-2 border-b border-slate-800">Logit Conf</th>
                    <th className="p-2 border-b border-slate-800">PSC Score</th>
                    <th className="p-2 border-b border-slate-800">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {tokens.slice(0, currentTokenIdx).map((t, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/50 transition-colors">
                      <td className="p-2 text-slate-500">{t.position}</td>
                      <td className="p-2 font-bold text-slate-200">{t.token_text}</td>
                      <td className="p-2 text-slate-300">{t.logit_confidence.toFixed(2)}</td>
                      <td className="p-2 font-bold text-cyan-400">{t.psc_score.toFixed(2)}</td>
                      <td className="p-2">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          t.status === 'HALT' ? 'bg-red-500/20 text-red-400' :
                          t.status === 'WARNING' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-emerald-500/20 text-emerald-400'
                        }`}>
                          {t.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>

      </div>

    </div>
  )
}
