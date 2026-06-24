import sqlite3

conn = sqlite3.connect("radiology.db", check_same_thread=False)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    findings TEXT,
    impression TEXT,
    confidence INTEGER,
    recommendation TEXT
)
""")

conn.commit()