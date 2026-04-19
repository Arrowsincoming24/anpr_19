import sqlite3
import os

db_path = 'backend/anpr.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('ALTER TABLE detections ADD COLUMN city TEXT')
        conn.commit()
        print('Column city added successfully')
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Column 'city' already exists.")
        else:
            print(f"Operational error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
else:
    print("Database file not found. It will be created with the new schema on first run.")
