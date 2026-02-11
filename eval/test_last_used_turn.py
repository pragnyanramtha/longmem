#!/usr/bin/env python3
"""Quick test to verify last_used_turn tracking works correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LongMemAgent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

def test_last_used_turn():
    """Test that last_used_turn is tracked correctly."""
    
    # Create agent with test database
    agent = LongMemAgent(
        provider="groq",
        db_path="test_last_used.db",
        model="llama-3.1-8b-instant"
    )
    
    console.print(Panel("[bold cyan]Testing last_used_turn Tracking[/bold cyan]", box=box.DOUBLE))
    
    # Turn 1: Add some memories
    console.print("\n[bold]Turn 1:[/bold] Planting memories...")
    response1 = agent.chat("My name is Alice and I love Python programming.")
    console.print(f"  Response: {response1['response'][:80]}...")
    console.print(f"  Active memories: {response1['total_memories']}")
    
    # Manually distill to save memories
    console.print("\n[bold]Distilling memories...[/bold]")
    distill_result = agent.manual_distill()
    console.print(f"  {distill_result['message']}")
    
    # Turn 2: Query that should retrieve memories
    console.print("\n[bold]Turn 2:[/bold] Querying about name...")
    response2 = agent.chat("What's my name?")
    console.print(f"  Response: {response2['response']}")
    
    # Check active_memories in response
    if response2['active_memories']:
        console.print("\n[green]✓ Memories retrieved and tracked![/green]")
        
        table = Table(box=box.SIMPLE)
        table.add_column("Memory ID", style="dim")
        table.add_column("Content", style="cyan")
        table.add_column("Origin Turn", justify="right")
        table.add_column("Last Used", justify="right", style="green")
        
        for mem in response2['active_memories']:
            table.add_row(
                mem['memory_id'][:12],
                mem['content'][:40],
                str(mem['origin_turn']),
                str(mem['last_used_turn'])
            )
        
        console.print(table)
        
        # Verify last_used_turn is set to current turn
        expected_turn = response2['turn_id']
        actual_turn = response2['active_memories'][0]['last_used_turn']
        
        if actual_turn == expected_turn:
            console.print(f"\n[bold green]✓ TEST PASSED[/bold green]")
            console.print(f"  last_used_turn correctly set to {actual_turn}")
            return True
        else:
            console.print(f"\n[bold red]✗ TEST FAILED[/bold red]")
            console.print(f"  Expected last_used_turn={expected_turn}, got {actual_turn}")
            return False
    else:
        console.print("\n[yellow]⚠ No memories retrieved (possibly no relevant memories yet)[/yellow]")
        return None

if __name__ == "__main__":
    import os
    
    # Clean up test db if it exists
    if os.path.exists("test_last_used.db"):
        os.remove("test_last_used.db")
    
    try:
        result = test_last_used_turn()
        
        if result:
            console.print("\n[bold]last_used_turn tracking is working correctly! 🎉[/bold]")
            sys.exit(0)
        elif result is None:
            console.print("\n[yellow]Test inconclusive - run with more turns[/yellow]")
            sys.exit(1)
        else:
            console.print("\n[red]Test failed - check implementation[/red]")
            sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists("test_last_used.db"):
            os.remove("test_last_used.db")
