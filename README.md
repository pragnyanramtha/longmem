# Atlas - Long-Form Memory System

A conversational AI agent with **persistent, queryable memory** that can recall information from turn 1 even at turn 1000.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Spec Compliance](https://img.shields.io/badge/spec-100%25-brightgreen.svg)](eval/SPEC_COMPLIANCE_ANALYSIS.md)

---

## ✨ Features

✅ **Persistent Conversations** - Automatically resumes from your last session  
✅ **Hybrid Memory Search** - Semantic (vector) + keyword (FTS5) with RRF  
✅ **Manual Memory Extraction** - `/distill` command to save memories on demand  
✅ **Token-Aware Context** - Auto-flushes at 70% context usage  
✅ **last_used_turn Tracking** - Full specification compliance (100/100)  
✅ **Rich CLI** - Beautiful terminal interface with memory insights  
✅ **1000+ Turn Support** - Validated for long conversations  

---

## 🚀 Quick Start (5 minutes)

### Option 1: Using `uv` (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/pragnyanramtha/longmem.git
cd longmem

# 2. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Set up API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Run the demo
./run_demo.sh
```

### Option 2: Using pip/venv

```bash
# 1. Clone the repository
git clone https://github.com/pragnyanramtha/longmem.git
cd longmem

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the interactive CLI
python main.py
```

### Option 3: Using Conda

```bash
# 1. Clone the repository
git clone https://github.com/pragnyanramtha/longmem.git
cd longmem

# 2. Create environment
conda env create -f environment.yml
conda activate atlas

# 3. Set up API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Run the interactive CLI
python main.py
```

### Option 4: Using Docker

```bash
# 1. Clone the repository
git clone https://github.com/pragnyanramtha/longmem.git
cd longmem

# 2. Build the image
docker build -t atlas:latest .

# 3. Run with environment variables
docker run -it \
  -e GROQ_API_KEY=your_key_here \
  -v $(pwd)/memory.db:/app/memory.db \
  -v $(pwd)/snapshots:/app/snapshots \
  atlas:latest
```

---

## 📋 Prerequisites

- **Python 3.11+** (required)
- **API Key** from one of:
  - [Groq](https://console.groq.com) (recommended, fastest)
  - [OpenAI](https://platform.openai.com)
  - Local Ollama server (optional)

---

## 🎯 Step-by-Step Setup

### Step 1: Get an API Key

**Option A: Groq (Recommended - FREE)**
1. Visit https://console.groq.com
2. Sign up for a free account
3. Go to API Keys → Create API Key
4. Copy the key starting with `gsk_...`

**Option B: OpenAI**
1. Visit https://platform.openai.com
2. Create an account and add credits
3. Generate an API key starting with `sk-...`

**Option C: Local Model (Advanced)**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull mistral`
3. No API key needed

### Step 2: Configure Environment

Create `.env` file in the project root:

```bash
# For Groq (recommended)
GROQ_API_KEY=gsk_your_actual_key_here

# OR for OpenAI
# OPENAI_API_KEY=sk_your_actual_key_here

# OR for Ollama (local)
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_MODEL=mistral
```

### Step 3: Install Dependencies

Choose your preferred method:

**Using uv (fastest):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using conda:**
```bash
conda env create -f environment.yml
conda activate atlas
```

### Step 4: Run the Demo

**Interactive CLI:**
```bash
python main.py
```

**Automated Demo Script:**
```bash
./run_demo.sh
```

**Jupyter Notebook Demo:**
```bash
jupyter notebook run_demo.ipynb
```

---

## 📖 Usage Guide

### Interactive Commands

| Command | Description |
|---------|-------------|
| (normal text) | Chat with the agent |
| `/memories` | Show all active memories in a table |
| `/distill` | Manually extract memories from current conversation |
| `/snapshot` | Save current memory state to `snapshots/` |
| `/quit` | Exit (conversation state is saved automatically) |

### Example Session

```bash
$ python main.py

╔════════════════════════════════════════════════════════════╗
║         Long-Form Memory Agent - Interactive CLI          ║
╚════════════════════════════════════════════════════════════╝

Resuming conversation from turn 0. Active memories: 0

You: My name is Priya and I'm allergic to peanuts.
Assistant: Nice to meet you, Priya! I'll remember that you're allergic to peanuts.

You: /distill
✓ Distillation complete. 2 memories extracted.
Total active memories: 2

You: I like hiking on weekends.
Assistant: That's great! I'll remember you enjoy hiking on weekends.

You: /quit
Conversation saved. See you next time!

# --- Next day, restart the program ---

$ python main.py
Resuming conversation from turn 3. Active memories: 3

You: What's my name and what am I allergic to?
Assistant: Your name is Priya, and you're allergic to peanuts.
  🧠 name: Priya (t1)
  🧠 allergy: peanuts (t1)
```

---

## 🏗️ Architecture

```
atlas/
├── src/
│   ├── agent.py         # Main orchestration loop
│   ├── models.py        # Data structures (Memory, DistilledMemory)
│   ├── store.py         # SQLite + sqlite-vec + FTS5
│   ├── context.py       # Token-aware context window manager
│   ├── distiller.py     # LLM-based memory extraction
│   ├── retriever.py     # Hybrid search (vector + keyword)
│   └── prompts.py       # LLM prompt templates
├── eval/
│   ├── generate.py      # Generate synthetic 1000-turn conversation
│   ├── evaluate.py      # Run evaluation and calculate metrics
│   └── scenarios.json   # Test scenarios (planted memories + probes)
├── main.py              # Interactive CLI entry point
├── run_demo.sh          # Automated demo script
├── run_demo.ipynb       # Jupyter notebook demo
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment
└── Dockerfile           # Container image
```

### How It Works

1. **Chat normally** - Messages are in the context window
2. **Auto-flush at 70%** - When context is 70% full, memories are extracted
3. **Manual extraction** - Use `/distill` to save memories before 70%
4. **Persistent storage** - SQLite stores memories, vectors, and history
5. **Next session** - Resume exactly where you left off

### Memory Pipeline

```
User Input
    ↓
[1] Retrieve Relevant Memories (hybrid search)
    ↓
[2] Inject into System Prompt
    ↓
[3] LLM Generates Response
    ↓
[4] Track last_used_turn
    ↓
[5] Check Context Usage (70% threshold?)
    ↓
[6] If threshold hit: Distill & Save Memories
    ↓
Response + Metadata
```

---

## 🧪 Evaluation

Run the comprehensive 1000-turn evaluation:

```bash
# Generate synthetic conversation
python eval/generate.py

# Run evaluation (uses local model or API)
python eval/evaluate.py

# Or with specific model
python eval/evaluate.py --local --model mistral --turns 1000
```

**Expected Results:**
- ✅ Recall accuracy: >90%
- ✅ Memory persistence across 1000+ turns
- ✅ Query-specific retrieval working correctly
- ✅ last_used_turn tracking verified

See `eval/SPEC_COMPLIANCE_ANALYSIS.md` for detailed compliance report (**100/100 score**).

---

## 📊 Database Schema

All data stored in `memory.db`:

### memories
```sql
CREATE TABLE memories (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,      -- preference, fact, etc.
    category        TEXT NOT NULL,      -- language, schedule, etc.
    key             TEXT NOT NULL,      -- canonical identifier
    value           TEXT NOT NULL,      -- the actual information
    source_turn     INTEGER NOT NULL,   -- when it was created
    confidence      REAL DEFAULT 0.9,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    is_active       INTEGER DEFAULT 1,  -- soft delete flag
    last_used_turn  INTEGER DEFAULT 0   -- tracking retrieval usage
);
```

### Indexes
- **memories_vec** (sqlite-vec) - 384-dim embeddings for semantic search
- **memories_fts** (FTS5) - Full-text keyword index
- **profile** - User preferences auto-populated
- **turns** - Full conversation log

---

## 🛠️ Configuration

Edit `src/agent.py` constructor or pass parameters:

```python
agent = LongMemAgent(
    api_key="your_key",           # API key
    provider="groq",              # "groq", "openai", or "ollama"
    model="llama-3.1-8b-instant", # Model name
    db_path="memory.db",          # Database file
    context_limit=8192,           # Token limit
    flush_threshold=0.70,         # When to distill (70%)
)
```

### Supported Models

**Groq (recommended):**
- `llama-3.1-8b-instant` (fast, default)
- `llama-3.3-70b-versatile` (powerful)
- `mixtral-8x7b-32768` (large context)

**OpenAI:**
- `gpt-4o-mini` (cost-effective)
- `gpt-4o` (most capable)

**Ollama (local):**
- `mistral`
- `llama3.1`
- `qwen2.5`

---

## 🔬 Advanced Features

### Manual Memory Distillation

Force memory extraction without waiting for 70% threshold:

```python
from src.agent import LongMemAgent

agent = LongMemAgent()
result = agent.manual_distill()
print(result)  # {'success': True, 'memories_added': 5, ...}
```

### Programmatic Usage

```python
from src.agent import LongMemAgent

# Initialize agent
agent = LongMemAgent(provider="groq", model="llama-3.1-8b-instant")

# Single turn
response = agent.chat("My favorite color is blue")
print(response['response'])
print(response['active_memories'])  # Memories used in this turn

# Get all memories
all_memories = agent.get_all_memories()
for mem in all_memories:
    print(f"{mem['key']}: {mem['value']} (turn {mem['source_turn']})")
```

### Migration to New Schema

If upgrading from an older version:

```bash
python migrate_add_last_used_turn.py
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "sqlite3.OperationalError: no such module: vec0"
```bash
pip install --force-reinstall sqlite-vec
```

### "groq.APIError: Invalid API key"
```bash
# Check your .env file
cat .env
# Make sure GROQ_API_KEY is set correctly
```

### Memory not persisting between sessions
```bash
# Check if memory.db exists and has data
sqlite3 memory.db "SELECT COUNT(*) FROM memories"
# Should return > 0 after distillation
```

### Low recall in evaluation
```bash
# Increase retrieval top_k
# Edit src/agent.py line 103: top_k=10 instead of top_k=5
```

---

## 📚 Technical Details

### Hybrid Retrieval Strategy

Atlas uses **Reciprocal Rank Fusion (RRF)** to merge:
1. **Vector search** (semantic similarity via sentence-transformers)
2. **FTS5 search** (keyword matching)

This ensures both semantic understanding and exact term matching.

### Token Management

- Context window monitored per-turn
- Automatic flush at 70% utilization (configurable)
- Last 4 messages retained for continuity
- System prompt rebuilt with retrieved memories

### Memory Lifecycle

1. **Extraction** - LLM analyzes conversation segment
2. **Validation** - Structured format checked
3. **Storage** - Written to SQLite + vector + FTS indexes
4. **Retrieval** - Hybrid search on each turn
5. **Injection** - Added to system prompt
6. **Tracking** - last_used_turn updated
7. **Expiry** - Soft delete via is_active flag

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add streaming response support
- [ ] Implement memory decay based on last_used_turn
- [ ] Add multi-user support with session IDs
- [ ] Create web UI (FastAPI + React)
- [ ] Add more evaluation scenarios
- [ ] Implement memory conflict resolution
- [ ] Add export/import for memory backups

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- Built with [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search
- Powered by [Groq](https://groq.com) for fast LLM inference
- Embeddings via [sentence-transformers](https://www.sbert.net/)
- UI built with [rich](https://github.com/Textualize/rich)

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/pragnyanramtha/longmem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pragnyanramtha/longmem/discussions)
- **Documentation**: See `eval/SPEC_COMPLIANCE_ANALYSIS.md` for detailed architecture

---

## 🎓 Citation

If you use Atlas in your research, please cite:

```bibtex
@software{atlas_memory,
  title={Atlas: Long-Form Memory System for Conversational AI},
  author={Pragnyan Ramtha},
  year={2026},
  url={https://github.com/pragnyanramtha/longmem}
}
```

---

**Built with ❤️ for production-grade persistent memory in conversational AI**
