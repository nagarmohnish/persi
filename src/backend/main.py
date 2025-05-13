from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import torch
import faiss
import numpy as np
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to frontend URL post-deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Startup Assistant Bot API. Use /query to ask questions."}

# File paths
DATA_DIR = "data/processed"
DB_PATH = os.path.join(DATA_DIR, "essays.db")
FAISS_PATH = os.path.join(DATA_DIR, "essays_index.faiss")
IDS_PATH = os.path.join(DATA_DIR, "essay_ids.txt")

# Load FAISS index and essay IDs
try:
    index = faiss.read_index(FAISS_PATH)
    with open(IDS_PATH, 'r') as f:
        essay_ids = [int(line.strip()) for line in f]
except Exception as e:
    raise Exception(f"Failed to load FAISS index or IDs: {str(e)}")

# Load models
try:
    tokenizer_bert = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', use_fast=False)
    model_bert = AutoModel.from_pretrained('ai4bharat/indic-bert')
    tokenizer_trans = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    model_trans = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
    generator = pipeline('text-generation', model='distilgpt2', tokenizer='distilgpt2', max_new_tokens=50)
except Exception as e:
    raise Exception(f"Failed to load models: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    lang: str

def translate_text(text, src_lang, tgt_lang):
    inputs = tokenizer_trans(text, return_tensors="pt", padding=True, truncation=True)
    translated = model_trans.generate(
        **inputs,
        forced_bos_token_id=tokenizer_trans.lang_code_to_id[tgt_lang],
        max_length=512
    )
    return tokenizer_trans.decode(translated[0], skip_special_tokens=True)

def get_embedding(text, max_length=512):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def retrieve_essays(query, k=3):
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    results = []
    for idx in I[0]:
        cursor.execute('SELECT title, content FROM essays WHERE id = ?', (essay_ids[idx],))
        result = cursor.fetchone()
        if result:
            title, content = result
            results.append(f"Title: {title}\nContent: {content[:500]}...")
    conn.close()
    return results

def generate_answer(context, question):
    prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {question}\nProvide a concise answer in 2-3 sentences."
    response = generator(prompt, num_return_sequences=1)
    answer = response[0]['generated_text'].split(prompt)[-1].strip()
    return answer

@app.post("/query")
async def process_query(request: QueryRequest):
    query, lang = request.query, request.lang
    if lang not in ["en", "hin", "tam", "tel"]:
        raise HTTPException(status_code=400, detail="Unsupported language")
    try:
        query_en = translate_text(query, f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu", "eng_Latn") if lang != "en" else query
        retrieved_essays = retrieve_essays(query_en)
        context = "\n\n".join(retrieved_essays)
        answer = generate_answer(context, query_en)
        if lang != "en":
            answer = translate_text(answer, "eng_Latn", f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu")
        return {"answer": answer, "retrieved_essays": retrieved_essays}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")