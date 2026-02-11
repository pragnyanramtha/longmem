# Atlas Project Structure

This document provides a detailed overview of the project organization.

## 📁 Directory Structure

```
atlas/
│
├── src/                       # Core source code
│   ├── agent.py              # Main orchestration loop & LongMemAgent class
│   ├── models.py             # Data structures (Memory, DistilledMemory, etc.)
│   ├── store.py              # SQLite persistence layer with vector & FTS
│   ├── context.py            # Token-aware context window manager
│   ├── distiller.py          # LLM-based memory extraction
│   ├── retriever.py          # Hybrid memory search (vector + keyword)
│   └── prompts.py            # LLM prompt templates
│
├── eval/                      # Evaluation & testing
│   ├── generate.py           # Generate synthetic 1000-turn conversations
│   ├── evaluate.py           # Run evaluation & calculate metrics
│   ├── scenarios.json        # Test scenarios (planted memories + probes)
│   ├── check_memory_injection.py  # Verify memory injection works
│   ├── test_last_used_turn.py     # Test last_used_turn tracking
│   ├── SPEC_COMPLIANCE_ANALYSIS.md # 100/100 compliance report
│   └── FIXES.md              # Implementation notes
│
├── main.py                   # Interactive CLI entry point
├── run_demo.sh               # Automated demo script (bash)
├── run_demo.ipynb            # Interactive Jupyter notebook demo
│
├── requirements.txt          # Python dependencies (pip)
├── environment.yml           # Conda environment spec
├── pyproject.toml            # Project metadata (uv)
├── Dockerfile                # Container image definition
│
├── migrate_add_last_used_turn.py  # Database migration script
│
├── README.md                 # Main documentation (setup & usage)
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
├── .gitignore                # Git ignore patterns
├── .env.example              # Example environment configuration
│
├── memory.db                 # Main SQLite database (gitignored)
├── snapshots/                # Memory snapshots (gitignored)
└── .venv/                    # Virtual environment (gitignored)
```

## 📦 Core Modules

### `src/agent.py`
**Purpose**: Main orchestration loop for the conversational agent

**Key Components**:
- `LongMemAgent` class - Main entry point
- `chat(user_message)` - Process a single turn
- `manual_distill()` - Force memory extraction
- `_flush()` - Context window cleanup & distillation
- `_rebuild_system_prompt()` - Inject memories into prompt

**Dependencies**: All other src modules

### `src/models.py`
**Purpose**: Data structures and type definitions

**Key Components**:
- `Memory` - Database representation of a memory
- `DistilledMemory` - LLM output before persistence
- `RetrievalResult` - Memory + relevance score
- `TurnRecord` - Conversation turn log
- Constants: `MEMORY_TYPES`, `MEMORY_ACTIONS`, `STOPWORDS`

**Dependencies**: None (pure data classes)

### `src/store.py`
**Purpose**: SQLite persistence with vector & full-text search

**Key Components**:
- `MemoryStore` class - Database operations
- `add_memory()` - Insert new memory
- `search_vector()` - Semantic similarity search
- `search_fts()` - Keyword search
- `touch_memory()` - Update last_used_turn
- `log_turn()` - Record conversation turn

**Technologies**:
- SQLite3 (main database)
- sqlite-vec (vector similarity)
- FTS5 (full-text search)
- sentence-transformers (embeddings)

### `src/context.py`
**Purpose**: Token-aware context window management

**Key Components**:
- `ContextManager` class - Tracks token usage
- `needs_flush()` - Check if threshold reached
- `add_message()` - Append message to context
- `reset()` - Clear context (keep last N messages)
- `get_messages_for_api()` - Format for LLM API

**Uses**: tiktoken for token counting

### `src/distiller.py`
**Purpose**: LLM-based memory extraction

**Key Components**:
- `MemoryDistiller` class - Extracts structured memories
- `distill()` - Analyze conversation segment
- `_parse_response()` - Parse JSON output from LLM

**Approach**: Prompts LLM to analyze conversation and output JSON with memory operations (add/update/keep/expire)

### `src/retriever.py`
**Purpose**: Hybrid memory search

**Key Components**:
- `MemoryRetriever` class - Find relevant memories
- `retrieve()` - Main search function
- RRF (Reciprocal Rank Fusion) - Merge vector + keyword results

**Strategy**:
1. Vector search (semantic similarity)
2. FTS search (keyword matching)
3. RRF merge both ranked lists
4. Return top-k results

### `src/prompts.py`
**Purpose**: LLM prompt templates

**Key Components**:
- `SYSTEM_PROMPT_TEMPLATE` - Main conversation prompt
- `DISTILL_PROMPT` - Memory extraction prompt
- `PROFILE_SECTION` - User profile template
- `MEMORIES_SECTION` - Retrieved memories template

## 🧪 Evaluation Suite

### `eval/generate.py`
Generates synthetic 1000-turn conversation by combining:
- Planted memories (from scenarios.json)
- Probe questions (test recall)
- Random filler messages

### `eval/evaluate.py`
Runs full evaluation pipeline:
1. Generate conversation (optional)
2. Run agent on all turns
3. Check if probes trigger correct memories
4. Calculate recall accuracy
5. Generate report

### `eval/scenarios.json`
Defines test cases:
- `plants`: Information to be learned (turn X: "My name is...")
- `probes`: Questions to test recall (turn Y: "What's my name?")
- `expected_keywords`: What should appear in responses

### `eval/check_memory_injection.py`
Demonstrates that memories are:
1. Stored in the database
2. Retrieved on relevant queries
3. Injected into the system prompt
4. Used to influence responses

## 🔧 Utility Scripts

### `main.py`
Interactive CLI with commands:
- Normal text → chat
- `/memories` → show all memories
- `/distill` → extract memories manually
- `/snapshot` → save to markdown
- `/quit` → exit

### `run_demo.sh`
Automated demo that:
1. Checks environment setup
2. Creates demo database
3. Plants memories via conversation
4. Triggers manual distillation
5. Tests recall with questions
6. Shows final statistics

### `run_demo.ipynb`
Jupyter notebook with:
- Step-by-step walkthrough
- Visual memory inspection
- Performance metrics
- Database queries
- Graphs and tables

### `migrate_add_last_used_turn.py`
Database migration to add `last_used_turn` column to existing databases. Safe to run multiple times (idempotent).

## 🗄️ Database Schema

### Tables

**memories**
```sql
id, type, category, key, value, source_turn, 
confidence, created_at, updated_at, is_active, last_used_turn
```

**profile**
```sql
key, value, updated_at, source_turn
```

**turns**
```sql
turn_id, role, content, timestamp, memories_retrieved
```

### Indexes

**memories_vec** (sqlite-vec)
- 384-dimensional embeddings
- L2 distance similarity

**memories_fts** (FTS5)
- key, value, category indexed
- Supports keyword search

## 📊 Data Flow

```
User Input
    ↓
[1] Check context usage
    ↓ (if > 70%)
[2] Distill memories → Store → Reset context
    ↓
[3] Retrieve relevant memories (hybrid search)
    ↓
[4] Rebuild system prompt with memories
    ↓
[5] LLM generates response
    ↓
[6] Track last_used_turn
    ↓
[7] Log turn to database
    ↓
Response + Metadata
```

## 🔌 Extension Points

### Custom LLM Providers
Edit `src/agent.py` constructor to add new providers:
```python
if provider == "anthropic":
    self.client = anthropic.Anthropic(api_key=api_key)
```

### Custom Embedding Models
Edit `src/store.py`:
```python
EMBEDDING_MODEL = "your-model-name"
EMBEDDING_DIM = 768  # adjust dimension
```

### Custom Memory Types
Edit `src/models.py`:
```python
MEMORY_TYPES = Literal[
    "preference", "fact", "commitment",
    "your_custom_type"  # Add here
]
```

### Custom Retrieval Strategy
Edit `src/retriever.py` `retrieve()` method to change:
- Search algorithms
- Ranking functions
- RRF parameters

## 📝 Configuration Files

### `.env`
Runtime configuration:
```bash
GROQ_API_KEY=your_key
OPENAI_API_KEY=alternative_key
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### `pyproject.toml`
Project metadata for uv:
- Dependencies
- Python version requirement
- Project description

### `requirements.txt`
Pip-compatible dependency list

### `environment.yml`
Conda environment specification

### `Dockerfile`
Container image with:
- Python 3.11
- All dependencies
- Source code
- Persistent volume mounts

## 🚀 Deployment Options

### Local Python
```bash
python main.py
```

### Docker
```bash
docker run -it \
  -e GROQ_API_KEY=xxx \
  -v $(pwd)/memory.db:/app/memory.db \
  atlas:latest
```

### Cloud (Future)
- FastAPI backend
- React frontend
- PostgreSQL (instead of SQLite)
- Redis (caching)
- Docker Compose orchestration

## 📚 Further Reading

- `README.md` - Setup and usage guide
- `CONTRIBUTING.md` - Development guidelines
- `eval/SPEC_COMPLIANCE_ANALYSIS.md` - Architecture deep dive
- `LAST_USED_TURN_IMPLEMENTATION.md` - Feature implementation notes

---

**Last Updated**: 2026-02-11
