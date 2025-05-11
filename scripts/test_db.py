import sqlite3

conn = sqlite3.connect('../data/processed/essays.db')
cursor = conn.cursor()
cursor.execute('SELECT title, content FROM essays WHERE title LIKE ?', ['%Startup%'])
results = cursor.fetchall()
for title, content in results:
    print(f"Title: {title}\nContent (first 100 chars): {content[:100]}...\n")
conn.close()