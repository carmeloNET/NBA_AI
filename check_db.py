"""Check SQLite database contents"""
import sqlite3

db_path = "C:/Users/AI_Agent/Documents/forks/NBA_AI/data/NBA_AI_current.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()
print("Tablas disponibles:")
for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
    count = cursor.fetchone()[0]
    print(f"  {t[0]}: {count} rows")

print("\n" + "="*50)
print("BOXSCORES:")
cursor.execute("SELECT * FROM Boxscores LIMIT 1")
cols = [d[0] for d in cursor.description]
print(f"Columns: {cols[:10]}...")

cursor.execute("SELECT COUNT(*) FROM Boxscores")
print(f"Total boxscores: {cursor.fetchone()[0]}")

print("\n" + "="*50)
print("SCHEDULE:")
cursor.execute("SELECT COUNT(*) FROM Schedule")
print(f"Total games in schedule: {cursor.fetchone()[0]}")

cursor.execute("SELECT date, COUNT(*) FROM Schedule GROUP BY date ORDER BY date DESC LIMIT 5")
print("Recent dates:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} games")

print("\n" + "="*50)
print("BETTING:")
cursor.execute("SELECT COUNT(*) FROM Betting")
print(f"Total betting rows: {cursor.fetchone()[0]}")

conn.close()