from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import numpy as np
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Startup Assistant Bot API. Use /query to ask questions."}

# Load FAISS index and essay IDs
index = faiss.read_index('../../data/processed/essays_index.faiss')
with open('../../data/processed/essay_ids.txt', 'r') as f:
    essay_ids = [int(line.strip()) for line in f]

# Load IndicBERT model and tokenizer for embeddings
tokenizer_bert = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', use_fast=False)
model_bert = AutoModel.from_pretrained('ai4bharat/indic-bert')

# Load IndicTrans2 (using NLLB for translation)
tokenizer_trans = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
model_trans = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')

# Load language model for generation
generator = pipeline('text-generation', model='distilgpt2', tokenizer='distilgpt2', max_new_tokens=50)

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str
    lang: str

# Translate text
def translate_text(text, src_lang, tgt_lang):
    inputs = tokenizer_trans(text, return_tensors="pt", padding=True, truncation=True)
    translated = model_trans.generate(
        **inputs,
        forced_bos_token_id=tokenizer_trans.lang_code_to_id[tgt_lang],
        max_length=512
    )
    return tokenizer_trans.decode(translated[0], skip_special_tokens=True)

# Get embeddings
def get_embedding(text, max_length=512):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Retrieve essays
def retrieve_essays(query, k=3):
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)
    conn = sqlite3.connect('../../data/processed/essays.db')
    cursor = conn.cursor()
    results = []
    for idx in I[0]:
        cursor.execute('SELECT title, content FROM essays WHERE id = ?', (essay_ids[idx],))
        title, content = cursor.fetchone()
        results.append(f"Title: {title}\nContent: {content[:500]}...")
    conn.close()
    return results

# Generate answer
def generate_answer(context, question):
    prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {question}\nProvide a concise answer in 2-3 sentences."
    response = generator(prompt, num_return_sequences=1)
    answer = response[0]['generated_text'].split(prompt)[-1].strip()  # Extract the generated part
    return answer

# API endpoint
@app.post("/query")
async def process_query(request: QueryRequest):
    query, lang = request.query, request.lang
    if lang not in ["en", "hin", "tam", "tel"]:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # Translate query to English
    if lang != "en":
        query_en = translate_text(query, f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu", "eng_Latn")
    else:
        query_en = query

    # Retrieve and generate answer
    retrieved_essays = retrieve_essays(query_en)
    context = "\n\n".join(retrieved_essays)
    answer = generate_answer(context, query_en)

    # Translate answer back
    if lang != "en":
        answer = translate_text(answer, "eng_Latn", f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu")

    return {"answer": answer, "retrieved_essays": retrieved_essays}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)