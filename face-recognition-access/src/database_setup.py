import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DB_PATH = BASE_DIR / "database" / "access.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DB_PATH) as connection:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            job_title TEXT NOT NULL,
            authorized INTEGER NOT NULL DEFAULT 0
                CHECK (authorized IN (0, 1)),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Commit the changes and close the connection
print(f"Database ready: {DB_PATH}")
print("Employee table created or already exists.")