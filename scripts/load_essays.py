import sqlite3
import os

# Create SQLite database
conn = sqlite3.connect('../data/processed/essays.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS essays (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        content TEXT
    )
''')
conn.commit()

# Load .txt files
raw_dir = '../data/essays'
for filename in os.listdir(raw_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(raw_dir, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            title = filename.replace('.txt', '').replace('-', ' ').title()
            cursor.execute('INSERT INTO essays (title, content) VALUES (?, ?)',
                          (title, content))
            conn.commit()

conn.close()
print("Essays loaded into database!")