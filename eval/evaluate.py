"""Evaluate the long-form memory agent on the 1000-turn synthetic conversation."""

import json
import os
import sys
import time
import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

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
    parser.add_argument("--conversation", default=os.path.join(os.path.dirname(__file__), "conversation_1000.json"), help="Path to conversation JSON")
    parser.add_argument("--scenarios", default=os.path.join(os.path.dirname(__file__), "scenarios.json"), help="Path to scenarios JSON")
    parser.add_argument("--provider", default="groq", choices=["groq", "openai", "ollama"], help="LLM provider")
    parser.add_argument("--local", action="store_true", help="Run with local Ollama server (shorthand for --provider ollama)")
    parser.add_argument("--base-url", help="Base URL for the LLM API")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model name")
    parser.add_argument("--db", default="eval_memory.db", help="Path to database file")
    parser.add_argument("--limit", type=int, default=2048, help="Context limit")
    parser.add_argument("--flush", type=float, default=0.70, help="Flush threshold")
    parser.add_argument("--turns", type=int, default=50, help="Number of turns to evaluate")
    args = parser.parse_args()
    
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
    
    # Metrics
    total_ms = 0.0
    retrieval_ms_total = 0.0
    flush_count = 0
    probe_results = []
    
    console.print(f"[bold]Running {args.turns}-turn evaluation...[/bold]\n")
    
    for entry in conversation[:args.turns]:
        turn_id = entry["turn_id"]
        content = entry["content"]
        
        # Progress
        if turn_id % 10 == 0:
            console.print(f"  Turn {turn_id}/1000 | Memories: {agent.store.active_count()} | Flushes: {agent.total_flushes}")
        
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
                    console.print(f"[yellow]Rate limit hit on turn {turn_id}. Retrying in {delay}s...[/yellow]")
                    time.sleep(delay)
                else:
                    console.print(f"[red]Error on turn {turn_id}: {e}[/red]")
                    raise e
        else:
             console.print(f"[red]Failed after {max_retries} retries on turn {turn_id}[/red]")
             return

        elapsed = (time.time() - start) * 1000
        total_ms += elapsed
        retrieval_ms_total += result["retrieval_ms"]
        
        if result["flush_triggered"]:
            flush_count += 1
        
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
            hits = 0
            details = []
            
            # 1. Did we retrieve the key?
            # 2. Did the response actually contain the right info (fuzzy match)?
            expected_keywords = probe.get("expected_keywords", [])
            
            # Simple check: Does response contain ANY of the expected keywords?
            response_lower = result["response"].lower()
            keyword_hit = False
            if expected_keywords:
                keyword_hit = any(k.lower() in response_lower for k in expected_keywords)
            
            # Also track key recall separately
            key_hits = 0
            for ek in expected_keys:
                found = any(ek.lower() in rk for rk in retrieved_keys)
                if found:
                    key_hits += 1
                details.append({"key": ek, "retrieved": found})

            # For scoring, we'll use keyword hit as the primary success metric if keywords exist
            accuracy = 1.0 if keyword_hit else (0.0 if expected_keywords else (key_hits / len(expected_keys) if expected_keys else 1.0))
            
            probe_results.append({
                "turn": turn_id,
                "description": probe["description"],
                "expected": expected_keywords if expected_keywords else expected_keys,
                "hits": 1 if keyword_hit else key_hits,
                "total": 1 if expected_keywords else len(expected_keys),
                "accuracy": accuracy,
                "details": details,
                "response_preview": result["response"][:150],
                "retrieved": [m["content"] for m in result["active_memories"]],
            })
        
        # Rate limit for Groq free tier
        if args.provider == "groq" and elapsed < 2000:
            time.sleep((2000 - elapsed) / 1000.0)
    
    # Final snapshot
    agent.store.write_snapshot(args.turns)
    
    # Report
    _print_report(console, probe_results, total_ms, retrieval_ms_total, 
                  flush_count, agent.store.active_count(), args.turns)


def _print_report(console, probe_results, total_ms, retrieval_ms, 
                  flush_count, final_memory_count, total_turns):
    
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]EVALUATION REPORT[/bold cyan]")
    console.print("=" * 60)
    
    # Probe results table
    table = Table(title="Probe Results", box=box.SIMPLE_HEAVY)
    table.add_column("Turn", justify="right", width=6)
    table.add_column("Description", width=40)
    table.add_column("Recall", justify="center", width=10)
    table.add_column("Score", justify="right", width=8)
    
    total_hits = 0
    total_expected = 0
    
    for pr in probe_results:
        # Simplification: hits/total for display
        # If using keyword matching, act as boolean
        is_keyword_match = isinstance(pr["expected"], list) and len(pr["expected"]) > 0 and isinstance(pr["expected"][0], str) 
        
        score_val = pr["accuracy"]
        total_hits += score_val # Sum accuracy for average
        total_expected += 1
        
        score_str = f"{score_val:.0%}"
        style = "green" if score_val == 1.0 else ("yellow" if score_val > 0 else "red")
        
        table.add_row(
            str(pr["turn"]),
            pr["description"],
            f"[{style}]{score_str}[/{style}]",
            f"{score_val:.2f}",
        )
    
    console.print(table)
    
    # Summary metrics
    overall_score = total_hits / total_expected if total_expected > 0 else 0
    avg_turn_ms = total_ms / total_turns
    avg_retrieval_ms = retrieval_ms / total_turns
    
    summary = Table(title="Summary Metrics", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", width=30)
    summary.add_column("Value", justify="right", width=20)
    
    summary.add_row("Overall Success Rate", f"[bold]{overall_score:.0%}[/bold]")
    summary.add_row("Avg Turn Latency", f"{avg_turn_ms:.0f}ms")
    summary.add_row("Avg Retrieval Latency", f"{avg_retrieval_ms:.1f}ms")
    summary.add_row("Total Context Flushes", str(flush_count))
    summary.add_row("Final Active Memories", str(final_memory_count))
    summary.add_row("Total Turns", str(total_turns))
    
    console.print(summary)
    
    # Detail view of each probe
    console.print("\n[bold]Probe Details:[/bold]")
    for pr in probe_results:
        style = "green" if pr["accuracy"] == 1.0 else "red"
        console.print(f"\n  Turn {pr['turn']}: [{style}]{pr['accuracy']:.0%}[/{style}] — {pr['description']}")
        console.print(f"    Expected: {pr['expected']}")
        console.print(f"    Retrieved: {pr['retrieved']}")
        console.print(f"    Response: {pr['response_preview']}...")


if __name__ == "__main__":
    evaluate()
