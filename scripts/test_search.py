import sqlite3
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import numpy as np

# Load FAISS index and essay IDs
index = faiss.read_index('../data/processed/essays_index.faiss')
with open('../data/processed/essay_ids.txt', 'r') as f:
    essay_ids = [int(line.strip()) for line in f]

# Load IndicBERT model and tokenizer for embeddings
tokenizer_bert = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', use_fast=False)
model_bert = AutoModel.from_pretrained('ai4bharat/indic-bert')

# Load IndicTrans2 for translation (Indian languages to English)
tokenizer_trans = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
model_trans = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')

# Function to translate text
def translate_text(text, src_lang, tgt_lang):
    inputs = tokenizer_trans(text, return_tensors="pt", padding=True, truncation=True)
    translated = model_trans.generate(
        **inputs,
        forced_bos_token_id=tokenizer_trans.lang_code_to_id[tgt_lang],
        max_length=512
    )
    return tokenizer_trans.decode(translated[0], skip_special_tokens=True)

# Function to get embeddings (same as embed_essays.py)
def get_embedding(text, max_length=512):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Test queries in Indian languages
queries = [
    ("What does Paul Graham say about hiring?", "en"),  # English
    ("पॉल ग्राहम हायरिंग के बारे में क्या कहते हैं?", "hin"),  # Hindi
    ("பால் கிரகாம் பணியமர்த்தல் பற்றி என்ன சொல்கிறார்?", "tam"),  # Tamil
    ("పాల్ గ్రాహం హైరింగ్ గురించి ఏమి చెప్పారు?", "tel")  # Telugu
]

conn = sqlite3.connect('../data/processed/essays.db')
cursor = conn.cursor()

for query, lang in queries:
    print(f"\nQuery ({lang}): {query}")
    # Translate to English if not already in English
    if lang != "en":
        query_en = translate_text(query, f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu", "eng_Latn")
        print(f"Translated to English: {query_en}")
    else:
        query_en = query

    # Get embedding and search
    query_embedding = get_embedding(query_en)
    D, I = index.search(np.array([query_embedding], dtype='float32'), k=3)  # Top 3 results

    # Fetch results
    for idx in I[0]:
        cursor.execute('SELECT title, content FROM essays WHERE id = ?', (essay_ids[idx],))
        title, content = cursor.fetchone()
        print(f"Title: {title}\nContent (first 200 chars): {content[:200]}...\n")

conn.close()