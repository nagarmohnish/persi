import sqlite3
import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np

# Load essays from SQLite
conn = sqlite3.connect('../data/processed/essays.db')
cursor = conn.cursor()
cursor.execute('SELECT id, content FROM essays')
essays = cursor.fetchall()
conn.close()

# Load IndicBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', use_fast=False)
model = AutoModel.from_pretrained('ai4bharat/indic-bert')

# Function to get embeddings with mean pooling
def get_embedding(text, max_length=512):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling: Average the token embeddings (last hidden state)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Embed essays
essay_texts = [content for _, content in essays]
essay_ids = [id for id, _ in essays]
embeddings = []
for text in essay_texts:
    embedding = get_embedding(text)
    embeddings.append(embedding)
embeddings = np.array(embeddings, dtype='float32')

# Create FAISS index
dimension = embeddings.shape[1]  # 768 for IndicBERT
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and IDs
faiss.write_index(index, '../data/processed/essays_index.faiss')
with open('../data/processed/essay_ids.txt', 'w') as f:
    for id in essay_ids:
        f.write(f"{id}\n")

print("Embedding complete!")