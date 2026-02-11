#!/usr/bin/env python3
"""
Migration: Add last_used_turn column to memories table.

This script safely adds the last_used_turn column to existing databases.
It's safe to run multiple times (idempotent).
"""

import sqlite3
import sys
from pathlib import Path

def migrate_database(db_path: str):
    """Add last_used_turn column if it doesn't exist."""
    print(f"Migrating database: {db_path}")
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if column already exists
    cursor.execute("PRAGMA table_info(memories)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "last_used_turn" in columns:
        print("✓ Column 'last_used_turn' already exists - skipping migration")
        conn.close()
        return True
    
    # Add the column
    try:
        cursor.execute("""
            ALTER TABLE memories 
            ADD COLUMN last_used_turn INTEGER DEFAULT 0
        """)
        conn.commit()
        print("✓ Added column 'last_used_turn' to memories table")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM memories")
        count = cursor.fetchone()[0]
        print(f"✓ Verified: {count} existing memories now have last_used_turn=0")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Migration failed: {e}")
        conn.close()
        return False

def main():
    """Migrate all known database files."""
    databases = ["memory.db", "eval_memory.db"]
    
    success_count = 0
    for db in databases:
        if Path(db).exists():
            if migrate_database(db):
                success_count += 1
        else:
            print(f"⊘ Skipping {db} (not found)")
    
    print(f"\n{'='*50}")
    if success_count > 0:
        print(f"✓ Migration complete: {success_count} database(s) updated")
        print("\nYou can now use the agent with last_used_turn tracking enabled.")
    else:
        print("⚠ No databases were migrated")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
