#!/bin/bash

# Atlas Long-Form Memory System - Automated Demo
# This script demonstrates the memory pipeline on a sample conversation

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}║     Atlas Long-Form Memory System - Demo Script          ║${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠  Warning: .env file not found${NC}"
    echo -e "${YELLOW}   Creating from .env.example...${NC}"
    cp .env.example .env
    echo ""
    echo -e "${YELLOW}   Please edit .env and add your API key, then run this script again.${NC}"
    echo -e "${YELLOW}   Example: GROQ_API_KEY=gsk_your_key_here${NC}"
    exit 1
fi

# Check if API key is set
source .env
if [ -z "$GROQ_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠  No API key found in .env${NC}"
    echo -e "${YELLOW}   Please set either GROQ_API_KEY or OPENAI_API_KEY${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment configured${NC}"
echo ""

# Clean up old demo database
if [ -f demo_memory.db ]; then
    echo -e "${YELLOW}Cleaning up previous demo database...${NC}"
    rm demo_memory.db
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Demo Scenario: Personal Assistant Conversation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Create a Python script for the demo
cat > /tmp/atlas_demo.py << 'DEMO_SCRIPT'
import os
import sys
import time
sys.path.insert(0, os.getcwd())

from src.agent import LongMemAgent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# Initialize agent with demo database
agent = LongMemAgent(
    db_path="demo_memory.db",
    model="llama-3.1-8b-instant"
)

def chat_and_display(message: str, turn_num: int):
    """Send a message and display results."""
    console.print(f"\n[bold cyan]Turn {turn_num}:[/bold cyan] [yellow]{message}[/yellow]")
    
    response = agent.chat(message)
    
    console.print(f"[bold green]Assistant:[/bold green] {response['response']}")
    
    # Show metadata
    console.print(f"  [dim]└─ Context: {response['context_utilization']} | "
                 f"Memories: {response['total_memories']} | "
                 f"Retrieval: {response['retrieval_ms']}ms[/dim]")
    
    # Show retrieved memories if any
    if response['active_memories']:
        console.print(f"  [dim]🧠 Retrieved {len(response['active_memories'])} memories:[/dim]")
        for mem in response['active_memories']:
            console.print(f"     [dim]• {mem['content']} (t{mem['origin_turn']} → t{mem['last_used_turn']})[/dim]")
    
    return response

# Demo conversation
console.print(Panel.fit(
    "[bold cyan]Atlas Memory Demo[/bold cyan]\n"
    "Watch how the system learns and recalls information across multiple turns",
    border_style="blue"
))

# Turn 1-3: Plant some memories
chat_and_display("Hi! My name is Alex and I'm a software engineer.", 1)
time.sleep(1)

chat_and_display("I'm allergic to shellfish and I'm vegetarian.", 2)
time.sleep(1)

chat_and_display("I have a meeting with Sarah every Monday at 2 PM.", 3)
time.sleep(1)

# Manually distill to save memories
console.print("\n[bold yellow]⚡ Triggering manual memory distillation...[/bold yellow]")
result = agent.manual_distill()
console.print(f"[green]✓ {result['message']}[/green]")
time.sleep(1)

# Show stored memories
console.print("\n[bold]📦 Stored Memories:[/bold]")
memories = agent.get_all_memories()
if memories:
    table = Table(box=box.SIMPLE)
    table.add_column("Type", style="cyan")
    table.add_column("Key", style="green")
    table.add_column("Value", style="white")
    table.add_column("Turn", justify="right", style="dim")
    
    for mem in memories[:5]:  # Show first 5
        table.add_row(
            mem['type'],
            mem['key'],
            mem['value'][:40] + "..." if len(mem['value']) > 40 else mem['value'],
            str(mem['source_turn'])
        )
    
    console.print(table)

time.sleep(2)

# Turn 4-7: Test recall
console.print("\n[bold blue]═══ Testing Memory Recall ═══[/bold blue]")
time.sleep(1)

chat_and_display("What's my name?", 4)
time.sleep(1)

chat_and_display("What foods should I avoid?", 5)
time.sleep(1)

chat_and_display("What do I have on Monday?", 6)
time.sleep(1)

chat_and_display("Tell me everything you know about me.", 7)
time.sleep(1)

# Final summary
console.print("\n" + "═" * 60)
console.print(Panel.fit(
    f"[bold green]✓ Demo Complete![/bold green]\n\n"
    f"Total turns: {agent.turn_id}\n"
    f"Active memories: {agent.store.active_count()}\n"
    f"Database: demo_memory.db\n\n"
    f"[dim]The agent successfully:\n"
    f"• Learned information from turns 1-3\n"
    f"• Distilled and stored structured memories\n"
    f"• Retrieved relevant memories on turns 4-7\n"
    f"• Tracked last_used_turn for each memory[/dim]",
    border_style="green"
))

console.print("\n[bold]🔍 Inspect the database:[/bold]")
console.print("  sqlite3 demo_memory.db \"SELECT key, value, source_turn, last_used_turn FROM memories\"")
console.print("\n[bold]🚀 Try it yourself:[/bold]")
console.print("  python main.py")
console.print()
DEMO_SCRIPT

# Run the demo
echo -e "${GREEN}Starting demo conversation...${NC}"
echo ""
python /tmp/atlas_demo.py

# Cleanup
rm /tmp/atlas_demo.py

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Demo completed successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Inspect demo database: ${GREEN}sqlite3 demo_memory.db${NC}"
echo -e "  2. Try interactive mode: ${GREEN}python main.py${NC}"
echo -e "  3. Run full evaluation: ${GREEN}python eval/evaluate.py${NC}"
echo -e "  4. View Jupyter demo: ${GREEN}jupyter notebook run_demo.ipynb${NC}"
echo ""
