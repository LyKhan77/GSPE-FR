import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'db', 'attendance.db')
os.makedirs(os.path.join(BASE_DIR, 'db'), exist_ok=True)
conn = sqlite3.connect(DB_PATH)

c = conn.cursor()
for ddl in [
    "ALTER TABLE face_templates ADD COLUMN pose_label VARCHAR(16)",
    "ALTER TABLE face_templates ADD COLUMN quality_score FLOAT"
]:
    try: c.execute(ddl)
    except Exception as e: print(e)
conn.commit(); conn.close()