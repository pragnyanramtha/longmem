"""Interactive CLI for the long-form memory agent."""

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.agent import LongMemAgent


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the memory agent CLI")
    parser.add_argument("--provider", default="groq", choices=["groq", "openai", "ollama"], help="LLM provider")
    parser.add_argument("--local", action="store_true", help="Run with local Ollama server (shorthand for --provider ollama)")
    parser.add_argument("--base-url", help="Base URL for the LLM API (e.g. http://localhost:11434/v1 for Ollama)")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model name to use")
    parser.add_argument("--db", default="memory.db", help="Path to database file")
    args = parser.parse_args()
    
    # Handle local shorthand
    if args.local:
        args.provider = "ollama"
        if not args.base_url:
            args.base_url = "http://localhost:11434/v1"
        if args.model == "llama-3.1-8b-instant":
             args.model = "llama3"

    # Check API key only if using Groq default
    if args.provider == "groq" and not args.base_url:
        if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "gsk_your_key_here":
            print("Error: Set GROQ_API_KEY in .env file to use Groq, or specify another --provider")
            sys.exit(1)
    
    console = Console()
    
    console.print(Panel(
        "[bold cyan]Long-Form Memory Agent[/bold cyan]\n"
        "Type your messages. Info from turn 1 will be recalled at turn 1000.\n"
        "Commands: [dim]/memories[/dim] — show all  |  "
        "[dim]/distill[/dim] — extract memories now  |  "
        "[dim]/snapshot[/dim] — save snapshot  |  "
        "[dim]/quit[/dim] — exit",
        box=box.DOUBLE,
    ))
    
    agent = LongMemAgent(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        db_path=args.db
    )
    
    # Show continuation message if resuming
    if agent.turn_id > 0:
        console.print(f"[dim]Resuming conversation from turn {agent.turn_id}. "
                     f"Active memories: {agent.store.active_count()}[/dim]\n")
    
    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == "/quit":
            console.print("[dim]Goodbye.[/dim]")
            break
        
        if user_input.lower() == "/memories":
            _show_memories(console, agent)
            continue
        
        if user_input.lower() == "/distill":
            _distill_memories(console, agent)
            continue
        
        if user_input.lower() == "/snapshot":
            agent.store.write_snapshot(agent.turn_id)
            console.print(f"[dim]Snapshot saved to snapshots/turn_{agent.turn_id:05d}.md[/dim]")
            continue

      
        # Normal conversation
        result = agent.chat(user_input)
        
        # Display response
        response_text = result["response"]
        
        # Build metadata line
        meta_parts = [
            f"Turn {result['turn_id']}",
            f"Ctx: {result['context_utilization']}",
            f"{result['total_ms']:.0f}ms",
            f"Mems: {result['total_memories']}",
        ]
        if result["flush_triggered"]:
            meta_parts.append("⚡ FLUSH")
        
        meta_line = " │ ".join(meta_parts)
        
        # Show retrieved memories if any
        if result["active_memories"]:
            mem_text = "  🧠 "
            mem_parts = [
                f"[dim]{m['content']}[/dim] (t{m['origin_turn']})"
                for m in result["active_memories"]
            ]
            mem_text += " · ".join(mem_parts)
        else:
            mem_text = ""
        
        panel_content = f"{response_text}"
        if mem_text:
            panel_content += f"\n\n{mem_text}"
        
        console.print(Panel(
            panel_content,
            title=f"[bold blue]Assistant[/bold blue]",
            subtitle=f"[dim]{meta_line}[/dim]",
            box=box.ROUNDED,
            padding=(0, 1),
        ))


def _distill_memories(console: Console, agent: LongMemAgent):
    """Manually trigger memory distillation."""
    result = agent.manual_distill()
    
    if not result["success"]:
        console.print(f"[yellow]{result['message']}[/yellow]")
        return
    
    console.print(f"[green]✓ {result['message']}[/green]")
    console.print(f"[dim]Total active memories: {result['total_memories']}[/dim]")
    console.print(f"[dim]Snapshot: {result['snapshot_saved']}[/dim]")


def _show_memories(console: Console, agent: LongMemAgent):
    """Display all active memories in a table."""
    memories = agent.get_all_memories()
    
    if not memories:
        console.print("[dim]No memories stored yet.[/dim]")
        return
    
    table = Table(title="Active Memories", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="dim", width=12)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Key", style="green", width=24)
    table.add_column("Value", width=36)
    table.add_column("Turn", justify="right", width=6)
    table.add_column("Conf", justify="right", width=6)
    
    for m in memories:
        table.add_row(
            m["id"],
            m["type"],
            m["key"],
            m["value"],
            str(m["source_turn"]),
            f"{m['confidence']:.2f}",
        )
    
    console.print(table)
    
    profile = agent.store.get_profile()
    if profile:
        console.print("\n[bold]Profile:[/bold]")
        for k, v in profile.items():
            console.print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
