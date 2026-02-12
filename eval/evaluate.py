"""Evaluate the long-form memory agent on the 1000-turn synthetic conversation."""

import json
import os
import sys
import time
import argparse
import numpy as np

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LongMemAgent


import logging

def evaluate():
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        filename="eval.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger("eval")
    
    console = Console()

    parser = argparse.ArgumentParser(description="Evaluate memory agent")
    parser.add_argument("--conversation", default=None, help="Path to conversation JSON (auto-detected based on --quick)")
    parser.add_argument("--scenarios", default=None, help="Path to scenarios JSON (auto-detected based on --quick)")
    parser.add_argument("--provider", default="groq", choices=["groq", "openai", "ollama"], help="LLM provider")
    parser.add_argument("--local", action="store_true", help="Run with local Ollama server (shorthand for --provider ollama)")
    parser.add_argument("--base-url", help="Base URL for the LLM API")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model name")
    parser.add_argument("--db", default="eval_memory.db", help="Path to database file")
    parser.add_argument("--limit", type=int, default=2048, help="Context limit")
    parser.add_argument("--flush", type=float, default=0.70, help="Flush threshold")
    parser.add_argument("--turns", type=int, default=None, help="Number of turns to evaluate (defaults to full conversation length)")
    parser.add_argument("--quick", action="store_true", help="Use quick scenario/conversation for faster evaluation")
    parser.add_argument("--export", help="Export detailed results to JSON file")
    args = parser.parse_args()
    
    # Auto-detect conversation/scenarios based on --quick flag
    eval_dir = os.path.dirname(__file__)
    if args.quick:
        if not args.conversation:
            args.conversation = os.path.join(eval_dir, "conversation_quick.json")
        if not args.scenarios:
            args.scenarios = os.path.join(eval_dir, "scenarios_quick.json")
    else:
        if not args.conversation:
            args.conversation = os.path.join(eval_dir, "conversation_1000.json")
        if not args.scenarios:
            args.scenarios = os.path.join(eval_dir, "scenarios.json")
    
    # Load conversation
    if not os.path.exists(args.conversation):
        console.print(f"[red]Error: {args.conversation} not found. Run generate.py first.[/red]")
        return

    with open(args.conversation) as f:
        conversation = json.load(f)
    
    # Load expected probes
    with open(args.scenarios) as f:
        scenarios = json.load(f)
    
    probes = {p["turn"]: p for p in scenarios["probes"]}
    
    # Default turns to full conversation length (so all probes are reached)
    if args.turns is None:
        args.turns = len(conversation)
    
    # Warn if no probes will be reached
    max_probe_turn = max(probes.keys()) if probes else 0
    if args.turns < max_probe_turn:
        console.print(f"[yellow]⚠ Warning: --turns={args.turns} but probes go up to turn {max_probe_turn}. "
                      f"Some probes won't be evaluated. Use --turns={max_probe_turn} or higher.[/yellow]\n")
    
    # Handle local shorthand
    if args.local:
        args.provider = "ollama"
        if not args.base_url:
            args.base_url = "http://localhost:11434/v1"
        if args.model == "llama-3.1-8b-instant":
             args.model = "llama3" # Default to common local model name if not specified
    
    # Initialize agent with fresh DB
    if os.path.exists(args.db):
        os.remove(args.db)
    
    agent = LongMemAgent(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        db_path=args.db, 
        context_limit=args.limit, 
        flush_threshold=args.flush
    )
    
    # Enhanced metrics tracking
    metrics = {
        "turn_latencies": [],
        "retrieval_latencies": [],
        "context_utilizations": [],
        "memory_counts": [],
        "flush_turns": [],
        "probe_results": [],
        "errors": [],
    }
    
    console.print(f"[bold]Running {args.turns}-turn evaluation...[/bold]\n")
    console.print(f"[dim]Provider: {args.provider} | Model: {args.model}[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Processing {args.turns} turns...", total=args.turns)
        
        for entry in conversation[:args.turns]:
            turn_id = entry["turn_id"]
            content = entry["content"]
            
            # Progress update
            progress.update(task, advance=1, description=f"[cyan]Turn {turn_id}/{args.turns}")
            
            # Chat with retry for rate limits
            start = time.time()
            max_retries = 5
            base_delay = 5.0
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Turn {turn_id} input: {content}")
                    result = agent.chat(content)
                    logger.info(f"Turn {turn_id} response: {result['response']}")
                    break
                except Exception as e:
                    # Catch 429/413 errors from Groq (which come as HTTPStatusError or APIStatusError)
                    err_str = str(e)
                    if "429" in err_str or "413" in err_str or "rate_limit" in err_str.lower():
                        delay = base_delay * (2 ** attempt)
                        progress.console.print(f"[yellow]Rate limit hit on turn {turn_id}. Retrying in {delay}s...[/yellow]")
                        time.sleep(delay)
                    else:
                        progress.console.print(f"[red]Error on turn {turn_id}: {e}[/red]")
                        metrics["errors"].append({"turn": turn_id, "error": str(e)})
                        logger.error(f"Turn {turn_id} error: {e}")
                        raise e
            else:
                progress.console.print(f"[red]Failed after {max_retries} retries on turn {turn_id}[/red]")
                return

            elapsed = (time.time() - start) * 1000
            
            # Track metrics
            metrics["turn_latencies"].append(elapsed)
            metrics["retrieval_latencies"].append(result["retrieval_ms"])
            metrics["context_utilizations"].append(float(result["context_utilization"].strip('%')) / 100.0)
            metrics["memory_counts"].append(result["total_memories"])
            
            if result["flush_triggered"]:
                metrics["flush_turns"].append(turn_id)
            
            # Evaluate probes
            if turn_id in probes:
                probe = probes[turn_id]
                retrieved_keys = set()
                for mem in result["active_memories"]:
                    # Extract the key part from "key: value"
                    key_part = mem["content"].split(":")[0].strip().lower()
                    retrieved_keys.add(key_part)
                
                # Check expected keys in retrieval
                expected_keys = probe.get("expected_keys", [])
                expected_keywords = probe.get("expected_keywords", [])
                
                # Simple check: Does response contain ANY of the expected keywords?
                response_lower = result["response"].lower()
                keyword_hit = False
                if expected_keywords:
                    keyword_hit = any(k.lower() in response_lower for k in expected_keywords)
                
                # Also track key recall separately
                key_hits = 0
                details = []
                for ek in expected_keys:
                    found = any(ek.lower() in rk for rk in retrieved_keys)
                    if found:
                        key_hits += 1
                    details.append({"key": ek, "retrieved": found})

                # For scoring, we'll use keyword hit as the primary success metric if keywords exist
                accuracy = 1.0 if keyword_hit else (0.0 if expected_keywords else (key_hits / len(expected_keys) if expected_keys else 1.0))
                
                probe_result = {
                    "turn": turn_id,
                    "description": probe["description"],
                    "expected": expected_keywords if expected_keywords else expected_keys,
                    "hits": 1 if keyword_hit else key_hits,
                    "total": 1 if expected_keywords else len(expected_keys),
                    "accuracy": accuracy,
                    "details": details,
                    "response_preview": result["response"][:150],
                    "response_full": result["response"],
                    "retrieved": [m["content"] for m in result["active_memories"]],
                    "retrieval_count": len(result["active_memories"]),
                }
                metrics["probe_results"].append(probe_result)
            
            # Rate limit for Groq free tier
            if args.provider == "groq" and elapsed < 2000:
                time.sleep((2000 - elapsed) / 1000.0)
    
    # Final snapshot
    agent.store.write_snapshot(args.turns)
    
    # Comprehensive report
    _print_comprehensive_report(console, metrics, args)
    
    # Export if requested
    if args.export:
        _export_results(metrics, args.export, args)
        console.print(f"\n[green]✓ Results exported to {args.export}[/green]")


def _print_comprehensive_report(console, metrics, args):
    """Print detailed evaluation report with comprehensive statistics."""
    
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]COMPREHENSIVE EVALUATION REPORT[/bold cyan]")
    console.print("=" * 80)
    
    # Configuration
    config_table = Table(title="Configuration", box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    config_table.add_row("Provider", args.provider)
    config_table.add_row("Model", args.model)
    config_table.add_row("Turns Evaluated", str(args.turns))
    config_table.add_row("Context Limit", str(args.limit))
    config_table.add_row("Flush Threshold", f"{args.flush:.0%}")
    console.print(config_table)
    
    # Probe results table
    if metrics["probe_results"]:
        console.print("\n")
        table = Table(title="Probe Results", box=box.SIMPLE_HEAVY)
        table.add_column("Turn", justify="right", width=6)
        table.add_column("Description", width=40)
        table.add_column("Retrieved", justify="center", width=10)
        table.add_column("Accuracy", justify="right", width=10)
        
        total_accuracy = 0
        
        for pr in metrics["probe_results"]:
            score_val = pr["accuracy"]
            total_accuracy += score_val
            
            score_str = f"{score_val:.0%}"
            style = "green" if score_val == 1.0 else ("yellow" if score_val > 0 else "red")
            
            table.add_row(
                str(pr["turn"]),
                pr["description"],
                f"{pr['retrieval_count']} mems",
                f"[{style}]{score_str}[/{style}]",
            )
        
        console.print(table)
        overall_accuracy = total_accuracy / len(metrics["probe_results"]) if metrics["probe_results"] else 0
    else:
        overall_accuracy = 0
    
    # Performance statistics
    console.print("\n")
    perf_table = Table(title="Performance Statistics", box=box.SIMPLE_HEAVY)
    perf_table.add_column("Metric", width=30)
    perf_table.add_column("Value", justify="right", width=20)
    perf_table.add_column("Details", justify="left", width=30)
    
    # Latency stats
    turn_latencies = np.array(metrics["turn_latencies"])
    retrieval_latencies = np.array(metrics["retrieval_latencies"])
    
    perf_table.add_row(
        "Overall Accuracy", 
        f"[bold]{overall_accuracy:.1%}[/bold]",
        f"{sum(pr['accuracy'] for pr in metrics['probe_results'])}/{len(metrics['probe_results'])} probes"
    )
    perf_table.add_row("", "", "")  # spacer
    
    perf_table.add_row(
        "Turn Latency (avg)", 
        f"{turn_latencies.mean():.0f}ms",
        f"p50:{np.percentile(turn_latencies, 50):.0f} p95:{np.percentile(turn_latencies, 95):.0f}"
    )
    perf_table.add_row(
        "Turn Latency (min/max)", 
        f"{turn_latencies.min():.0f}/{turn_latencies.max():.0f}ms",
        ""
    )
    perf_table.add_row(
        "Retrieval Latency (avg)", 
        f"{retrieval_latencies.mean():.1f}ms",
        f"p50:{np.percentile(retrieval_latencies, 50):.1f} p95:{np.percentile(retrieval_latencies, 95):.1f}"
    )
    perf_table.add_row("", "", "")  # spacer
    
    # Context stats
    ctx_utils = np.array(metrics["context_utilizations"])
    perf_table.add_row(
        "Context Utilization (avg)", 
        f"{ctx_utils.mean():.1%}",
        f"max: {ctx_utils.max():.1%}"
    )
    perf_table.add_row(
        "Context Flushes", 
        str(len(metrics["flush_turns"])),
        f"at turns: {', '.join(map(str, metrics['flush_turns'][:5]))}" if metrics["flush_turns"] else "none"
    )
    perf_table.add_row("", "", "")  # spacer
    
    # Memory stats
    mem_counts = np.array(metrics["memory_counts"])
    perf_table.add_row(
        "Active Memories (final)", 
        str(mem_counts[-1] if len(mem_counts) > 0 else 0),
        f"growth: {mem_counts[0] if len(mem_counts) > 0 else 0} → {mem_counts[-1] if len(mem_counts) > 0 else 0}"
    )
    perf_table.add_row(
        "Memory Growth Rate", 
        f"{(mem_counts[-1] - mem_counts[0]) / args.turns * 100:.2f}" if len(mem_counts) > 0 else "0.00",
        "memories per 100 turns"
    )
    
    console.print(perf_table)
    
    # Error summary
    if metrics["errors"]:
        console.print("\n")
        error_table = Table(title="Errors", box=box.SIMPLE)
        error_table.add_column("Turn", justify="right")
        error_table.add_column("Error")
        for err in metrics["errors"]:
            error_table.add_row(str(err["turn"]), err["error"][:60])
        console.print(error_table)
    
    # Detailed probe breakdown
    if metrics["probe_results"]:
        console.print("\n[bold]Detailed Probe Analysis:[/bold]")
        for pr in metrics["probe_results"]:
            style = "green" if pr["accuracy"] == 1.0 else ("yellow" if pr["accuracy"] > 0 else "red")
            console.print(f"\n  [{style}]Turn {pr['turn']}[/{style}]: {pr['description']}")
            console.print(f"    [dim]Expected:[/dim] {', '.join(pr['expected'])}")
            console.print(f"    [dim]Retrieved:[/dim] {pr['retrieval_count']} memories")
            if pr['retrieved']:
                for mem in pr['retrieved'][:3]:  # Show first 3
                    console.print(f"      • {mem}")
            console.print(f"    [dim]Response:[/dim] {pr['response_preview']}...")
            console.print(f"    [dim]Accuracy:[/dim] [{style}]{pr['accuracy']:.0%}[/{style}]")


def _export_results(metrics, filepath, args):
    """Export detailed results to JSON for further analysis."""
    export_data = {
        "config": {
            "provider": args.provider,
            "model": args.model,
            "turns": args.turns,
            "context_limit": args.limit,
            "flush_threshold": args.flush,
        },
        "summary": {
            "overall_accuracy": sum(pr["accuracy"] for pr in metrics["probe_results"]) / len(metrics["probe_results"]) if metrics["probe_results"] else 0,
            "total_probes": len(metrics["probe_results"]),
            "avg_turn_latency_ms": float(np.mean(metrics["turn_latencies"])) if metrics["turn_latencies"] else 0,
            "avg_retrieval_latency_ms": float(np.mean(metrics["retrieval_latencies"])) if metrics["retrieval_latencies"] else 0,
            "total_flushes": len(metrics["flush_turns"]),
            "final_memory_count": metrics["memory_counts"][-1] if metrics["memory_counts"] else 0,
        },
        "turn_latencies": metrics["turn_latencies"],
        "retrieval_latencies": metrics["retrieval_latencies"],
        "context_utilizations": metrics["context_utilizations"],
        "memory_counts": metrics["memory_counts"],
        "flush_turns": metrics["flush_turns"],
        "probe_results": metrics["probe_results"],
        "errors": metrics["errors"],
    }
    
    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)


if __name__ == "__main__":
    evaluate()
