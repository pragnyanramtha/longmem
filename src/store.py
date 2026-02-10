"""SQLite-based memory store with vector search (sqlite-vec) and full-text search (FTS5)."""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import time
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

from .models import Memory, DistilledMemory, STOPWORDS


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats into bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


class MemoryStore:
    """Persistent memory store backed by SQLite + sqlite-vec + FTS5."""

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        
        # Load sqlite-vec extension
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        
        self._init_tables()
        
        # Load embedding model (local, runs on CPU)
        self._embedder = None  # lazy load
    
    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.EMBEDDING_MODEL)
            print(f"[Store] Embedding model loaded on device: {self._embedder.device}")
        return self._embedder
    
    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                category    TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                source_turn INTEGER NOT NULL,
                confidence  REAL DEFAULT 0.9,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                is_active   INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS profile (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  REAL NOT NULL,
                source_turn INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                turn_id     INTEGER PRIMARY KEY,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                memories_retrieved TEXT DEFAULT '[]'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                key, value, category, content=''
            );
        """)
        
        # sqlite-vec virtual table — must check if exists differently
        # Using vec0 as per spec
        try:
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[384]
                );
            """)
        except sqlite3.OperationalError:
            pass  # already exists
        
        self.db.commit()

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text. Returns list of floats."""
        return self.embedder.encode(text).tolist()

    def add_memory(self, mem: DistilledMemory, turn_id: int) -> str:
        """Insert a new memory into all three stores. Returns memory ID."""
        mem_id = Memory.generate_id()
        now = time.time()
        
        # 1. Main table
        self.db.execute("""
            INSERT INTO memories (id, type, category, key, value, source_turn,
                                  confidence, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (mem_id, mem.type, mem.category, mem.key, mem.value,
              turn_id, mem.confidence, now, now))
        
        # 2. FTS index
        # FTS5 rowid is manually managed to match memories table rowid
        rowid = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.db.execute(
            "INSERT INTO memories_fts(rowid, key, value, category) VALUES (?, ?, ?, ?)",
            (rowid, mem.key, mem.value, mem.category)
        )
        
        # 3. Vector index
        embed_text = f"{mem.key}: {mem.value}"
        embedding = self.embed(embed_text)
        self.db.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            (mem_id, _serialize_f32(embedding))
        )
        
        # 4. Profile update for preferences/facts/constraints
        if mem.type in ("preference", "fact", "constraint"):
            self.db.execute(
                "INSERT OR REPLACE INTO profile (key, value, updated_at, source_turn) "
                "VALUES (?, ?, ?, ?)",
                (mem.key, mem.value, now, turn_id)
            )
        
        self.db.commit()
        return mem_id

    def deactivate_by_key(self, key: str):
        """Soft-delete all active memories with this key."""
        self.db.execute(
            "UPDATE memories SET is_active = 0, updated_at = ? WHERE key = ? AND is_active = 1",
            (time.time(), key)
        )
        self.db.commit()

    def get_active_memories(self) -> list[Memory]:
        """Get all active memories, ordered by confidence desc."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE is_active = 1 ORDER BY confidence DESC"
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_profile(self) -> dict[str, str]:
        """Get the current user profile as a dict."""
        rows = self.db.execute("SELECT key, value FROM profile").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def active_count(self) -> int:
        """Count of active memories."""
        return self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE is_active = 1"
        ).fetchone()[0]

    def search_vector(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Vector similarity search. Returns list of (memory_id, distance)."""
        query_emb = self.embed(query)
        # Using MATCH syntax with k=? as per gotchas
        rows = self.db.execute(
            "SELECT id, distance FROM memories_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_serialize_f32(query_emb), top_k)
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def search_fts(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Full-text search. Returns list of (rowid, rank)."""
        words = [w for w in query.lower().split() if w not in STOPWORDS and len(w) > 2]
        if not words:
            return []
        fts_query = " OR ".join(words[:10])
        try:
            rows = self.db.execute(
                "SELECT rowid, rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, top_k)
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        except sqlite3.OperationalError:
            return []

    def rowid_to_memory_id(self, rowid: int) -> str | None:
        """Map FTS rowid back to memory ID."""
        row = self.db.execute(
            "SELECT id FROM memories WHERE rowid = ?", (rowid,)
        ).fetchone()
        return row["id"] if row else None

    def get_memory_by_id(self, mem_id: str) -> Memory | None:
        """Fetch a single memory by ID."""
        row = self.db.execute(
            "SELECT * FROM memories WHERE id = ? AND is_active = 1", (mem_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def log_turn(self, turn_id: int, role: str, content: str, 
                 memories_retrieved: list[str] | None = None):
        """Log a conversation turn."""
        self.db.execute(
            "INSERT OR REPLACE INTO turns (turn_id, role, content, timestamp, memories_retrieved) "
            "VALUES (?, ?, ?, ?, ?)",
            (turn_id, role, content, time.time(), 
             json.dumps(memories_retrieved or []))
        )
        self.db.commit()

    def get_last_turn_id(self) -> int:
        """Get the last turn_id from the database. Returns 0 if no turns exist."""
        row = self.db.execute(
            "SELECT MAX(turn_id) as max_turn FROM turns"
        ).fetchone()
        return row["max_turn"] if row and row["max_turn"] is not None else 0

    def write_snapshot(self, turn_id: int, snapshot_dir: str = "snapshots"):
        """Write a human-readable markdown snapshot of all active memories."""
        os.makedirs(snapshot_dir, exist_ok=True)
        memories = self.get_active_memories()
        profile = self.get_profile()
        
        path = Path(snapshot_dir) / f"turn_{turn_id:05d}.md"
        with open(path, "w") as f:
            f.write(f"# Memory Snapshot — Turn {turn_id}\\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            if profile:
                f.write("## Profile\\n")
                for k, v in profile.items():
                    f.write(f"- **{k}**: {v}\\n")
                f.write("\\n")
            
            if memories:
                current_type = None
                for m in sorted(memories, key=lambda x: (x.type, x.key)):
                    if m.type != current_type:
                        current_type = m.type
                        f.write(f"## {current_type.title()}s\\n")
                    f.write(
                        f"- **{m.key}**: {m.value} "
                        f"(conf: {m.confidence:.2f}, turn: {m.source_turn})\\n"
                    )
                f.write("\\n")
            
            f.write(f"\\nTotal active: {len(memories)}\\n")

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            type=row["type"],
            category=row["category"],
            key=row["key"],
            value=row["value"],
            source_turn=row["source_turn"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
        )
