#!/usr/bin/env python3
"""
replay_engine.py — SynapseGuard Python CLI Replay Engine

Reads a JSON trace file matching the schema in architecture.md Section 7
and animates token-by-token generation with realistic timing delays and colored gauge output.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def render_gauge_bar(score: float, width: int = 20) -> str:
    """Renders ASCII/Unicode progress bar for PSC score."""
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"

def run_cli_replay(trace_path: str, delay: float = 0.2):
    if not os.path.exists(trace_path):
        print(f"Error: Trace file not found at '{trace_path}'")
        sys.exit(1)
        
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    prompt = data.get("prompt", "")
    model = data.get("model", "gemma-2-2b-8bit")
    tokens = data.get("tokens", [])
    
    if RICH_AVAILABLE:
        console = Console()
        console.clear()
        console.print(Panel.fit(
            f"[bold cyan]SynapseGuard — Live Diagnostic Replay Engine[/bold cyan]\n"
            f"[dim]Model: {model} | File: {Path(trace_path).name}[/dim]",
            border_style="cyan"
        ))
        console.print(f"\n[bold yellow]Prompt:[/bold yellow] {prompt}\n")
        
        accumulated_text = Text()
        
        for tok in tokens:
            tok_text = tok["token_text"]
            conf = tok["logit_confidence"]
            psc = tok["psc_score"]
            status = tok["status"]
            ssp = tok.get("ssp_triggered", False)
            
            if status == "HALT":
                color_style = "bold white on red"
            elif status == "WARNING":
                color_style = "bold yellow"
            else:
                color_style = "green"
                
            accumulated_text.append(tok_text, style=color_style)
            
            gauge_str = render_gauge_bar(psc)
            ssp_str = "[bold red]YES[/bold red]" if ssp else "[dim]NO[/dim]"
            
            console.clear()
            console.print(Panel.fit(
                f"[bold cyan]SynapseGuard — Live Diagnostic Replay Engine[/bold cyan]\n"
                f"[dim]Model: {model} | Trace: {Path(trace_path).name}[/dim]",
                border_style="cyan"
            ))
            console.print(f"\n[bold yellow]Prompt:[/bold yellow] {prompt}\n")
            console.print(Panel(accumulated_text, title="[bold]Streamed Output[/bold]", border_style="dim"))
            console.print(f"\n[bold]PSC Gauge:[/bold] {gauge_str} | [bold]Status:[/bold] [{color_style}]{status}[/{color_style}] | [bold]SSP Triggered:[/bold] {ssp_str}")
            
            if status == "HALT":
                console.print("\n[bold red]⚠️ INTERCEPTION TRIGGERED: PSC gauge collapsed into HALT zone. Generation stopped.[/bold red]")
                break
                
            time.sleep(delay)
    else:
        print(f"=== SynapseGuard Replay Engine ({trace_path}) ===")
        print(f"Prompt: {prompt}\n")
        streamed = ""
        for tok in tokens:
            streamed += tok["token_text"]
            print(f"Token: {tok['token_text']:<12} | Conf: {tok['logit_confidence']:.2f} | PSC: {tok['psc_score']:.2f} | Status: {tok['status']}")
            if tok['status'] == "HALT":
                print("--> Generation Halted (HALT risk detected)")
                break
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="SynapseGuard Diagnostic Replay Engine")
    parser.add_argument("trace", nargs="?", default="part_b_diagnostic/dashboard/mock_trace.json", help="Path to JSON trace file")
    parser.add_argument("--delay", type=float, default=0.2, help="Per-token animation delay in seconds")
    args = parser.parse_args()
    
    run_cli_replay(args.trace, args.delay)

if __name__ == "__main__":
    main()
