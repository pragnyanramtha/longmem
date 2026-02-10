# Long-Form Memory System

A conversational AI agent with **persistent memory** that can recall information from turn 1 even at turn 1000.

## Features

✅ **Persistent Conversations**: Automatically resumes from your last session  
✅ **Hybrid Memory Search**: Combines semantic (vector) + keyword (FTS5) search with RRF  
✅ **Manual Memory Extraction**: `/distill` command to save memories on demand  
✅ **Token-Aware Context**: Auto-flushes at 70% context usage (configurable)  
✅ **Rich CLI**: Beautiful terminal interface with memory insights  

## Quick Start

### 1. Install Dependencies
```bash
uv add groq sentence-transformers sqlite-vec tiktoken rich python-dotenv
```

### 2. Configure API Key
Edit `.env`:
```bash
GROQ_API_KEY=gsk_your_actual_key_here
```

### 3. Run Interactive Chat
```bash
uv run python main.py
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/memories` | Show all active memories in a table |
| `/distill` | **Manually extract memories** from current conversation |
| `/snapshot` | Save current memory state to `snapshots/` |
| `/quit` | Exit (conversation state is saved automatically) |

## How It Works

### Memory Lifecycle
1. **Chat normally** - Your messages are in the context window
2. **Auto-flush at 70%** - When context is 70% full, memories are extracted automatically
3. **Manual extraction** - Use `/distill` anytime to save memories before 70%
4. **Persistent storage** - SQLite stores memories, vectors, and conversation history
5. **Next session** - Resume exactly where you left off with all memories intact

### Example Session

```bash
# First run
You: My name is Priya and I'm allergic to peanuts
Assistant: Nice to meet you, Priya! I'll remember that you're allergic to peanuts.

You: /distill
✓ Distillation complete. 2 memories extracted.
Total active memories: 2

# Close the terminal, restart the next day...

You: uv run python main.py
Resuming conversation from turn 1. Active memories: 2

You: What's my name?
Assistant: Your name is Priya.
  🧠 name: Priya (t1)
```

## Architecture

```
src/
├── models.py      → Data structures (Memory, DistilledMemory, etc.)
├── prompts.py     → LLM prompt templates
├── store.py       → SQLite + sqlite-vec + FTS5 persistence
├── context.py     → Token-aware context window manager
├── distiller.py   → LLM-based memory extraction (Groq API)
├── retriever.py   → Hybrid search (vector + keyword with RRF)
└── agent.py       → Main orchestration loop
```

## Why Manual `/distill` Matters

**Problem**: Not all conversations reach 70% context usage. Short sessions would lose memories.

**Solution**: The `/distill` command lets you:
- Extract memories after a few important messages
- Force a save before closing a short session
- Manually checkpoint before switching topics

## Evaluation

Run the 1000-turn stress test:
```bash
uv run python eval/generate.py
uv run python eval/evaluate.py
```

This generates a synthetic conversation with planted facts and tests recall accuracy across long-term memory.

## Persistence Details

All data is stored in `memory.db`:
- **memories** table: Active/inactive memories with metadata
- **profile** table: Key user preferences (auto-populated)
- **turns** table: Full conversation log
- **memories_vec** (sqlite-vec): 384-dim embeddings for semantic search
- **memories_fts** (FTS5): Full-text keyword index

On each run, the agent:
1. Loads the last turn ID from the database
2. Resumes counting from there
3. Loads existing memories for retrieval
4. Continues the conversation seamlessly

## License

MIT
