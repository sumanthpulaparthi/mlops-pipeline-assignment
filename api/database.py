import sqlite3
from datetime import datetime

DB_NAME = "logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input TEXT,
            prediction REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_db(input_data: list, prediction: float):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_logs (timestamp, input, prediction)
        VALUES (?, ?, ?)
    """, (datetime.now().isoformat(), str(input_data), prediction))
    conn.commit()
    conn.close()


def get_logs(limit):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT ?", (limit,))
    results = cursor.fetchall()
    conn.close()
    return results
