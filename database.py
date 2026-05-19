import sqlite3
import os
from datetime import datetime

DB_PATH = "/app/data/querydoc.db"

def init_db():
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print(f"Initializing database at {DB_PATH}...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunks_count INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing database: {e}")
        raise

def add_document(doc_id: str, user_id: str, filename: str, chunks_count: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (id, user_id, filename, chunks_count, uploaded_at) VALUES (?, ?, ?, ?, ?)",
        (doc_id, user_id, filename, chunks_count, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_user_documents(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, chunks_count, uploaded_at FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": row[0], "filename": row[1], "chunks_count": row[2], "uploaded_at": row[3]}
        for row in rows
    ]

def delete_document(doc_id: str, user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    conn.commit()
    conn.close()