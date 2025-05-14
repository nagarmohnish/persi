from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import torch
import faiss
import numpy as np
import os
import logging
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Startup Assistant Bot API. Use /query to ask questions."}

@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        return {"status": "healthy", "message": "Database connection successful"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# File paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed"))
logger.info(f"DATA_DIR resolved to: {DATA_DIR}")
DB_PATH = os.getenv("DATABASE_PATH", os.path.join(DATA_DIR, "essays.db"))
FAISS_PATH = os.path.join(DATA_DIR, "essays_index.faiss")
IDS_PATH = os.path.join(DATA_DIR, "essay_ids.txt")

# Verify file existence
for path in [DB_PATH, FAISS_PATH, IDS_PATH]:
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

# Load FAISS index and essay IDs
try:
    logger.info(f"Loading FAISS index from {FAISS_PATH}")
    index = faiss.read_index(FAISS_PATH)
    logger.info(f"Loading essay IDs from {IDS_PATH}")
    with open(IDS_PATH, 'r') as f:
        essay_ids = [int(line.strip()) for line in f]
except Exception as e:
    logger.error(f"Failed to load FAISS index or IDs: {str(e)}")
    raise Exception(f"Failed to load FAISS index or IDs: {str(e)}")

# Load smaller models to reduce memory usage
try:
    logger.info("Loading DistilBERT model and tokenizer on CPU")
    tokenizer_bert = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model_bert = AutoModel.from_pretrained('distilbert-base-uncased').to('cpu')
    logger.info("Loading minimal NLLB model on CPU")
    tokenizer_trans = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    model_trans = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M').to('cpu')
    logger.info("Loading distilgpt2 for text generation on CPU")
    generator = pipeline('text-generation', model='distilgpt2', tokenizer='distilgpt2', max_new_tokens=50, device=-1)
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise Exception(f"Failed to load models: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    lang: str

def translate_text(text, src_lang, tgt_lang):
    try:
        logger.info(f"Translating text from {src_lang} to {tgt_lang}")
        inputs = tokenizer_trans(text, return_tensors="pt", padding=True, truncation=True).to('cpu')
        translated = model_trans.generate(
            **inputs,
            forced_bos_token_id=tokenizer_trans.lang_code_to_id[tgt_lang],
            max_length=512
        )
        return tokenizer_trans.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise

def get_embedding(text, max_length=512):
    try:
        logger.info("Generating embedding for text")
        inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to('cpu')
        with torch.no_grad():
            outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise

def retrieve_essays(query, k=3):
    try:
        logger.info(f"Retrieving essays for query: {query}")
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
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise

def generate_answer(context, question):
    try:
        logger.info("Generating answer")
        prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {question}\nProvide a concise answer in 2-3 sentences."
        response = generator(prompt, num_return_sequences=1)
        answer = response[0]['generated_text'].split(prompt)[-1].strip()
        return answer
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise

@app.post("/query")
async def process_query(request: QueryRequest):
    query, lang = request.query, request.lang
    if lang not in ["en", "hin", "tam", "tel"]:
        raise HTTPException(status_code=400, detail="Unsupported language")
    try:
        logger.info(f"Processing query: {query} in language {lang}")
        query_en = translate_text(query, f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu", "eng_Latn") if lang != "en" else query
        retrieved_essays = retrieve_essays(query_en)
        context = "\n\n".join(retrieved_essays)
        answer = generate_answer(context, query_en)
        if lang != "en":
            answer = translate_text(answer, "eng_Latn", f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu")
        logger.info("Query processed successfully")
        return {"answer": answer, "retrieved_essays": retrieved_essays}
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

















# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import sqlite3
# import torch
# import faiss
# import numpy as np
# import os
# import logging
# from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline
# from fastapi.middleware.cors import CORSMiddleware

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Startup Assistant Bot API. Use /query to ask questions."}

# @app.get("/health")
# async def health_check():
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         conn.close()
#         return {"status": "healthy", "message": "Database connection successful"}
#     except Exception as e:
#         logger.error(f"Health check failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# # File paths
# DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
# DB_PATH = os.getenv("DATABASE_PATH", os.path.join(DATA_DIR, "essays.db"))
# FAISS_PATH = os.path.join(DATA_DIR, "essays_index.faiss")
# IDS_PATH = os.path.join(DATA_DIR, "essay_ids.txt")

# # Verify file existence
# for path in [DB_PATH, FAISS_PATH, IDS_PATH]:
#     if not os.path.exists(path):
#         logger.error(f"File not found: {path}")
#         raise FileNotFoundError(f"File not found: {path}")

# # Load FAISS index and essay IDs
# try:
#     logger.info(f"Loading FAISS index from {FAISS_PATH}")
#     index = faiss.read_index(FAISS_PATH)
#     logger.info(f"Loading essay IDs from {IDS_PATH}")
#     with open(IDS_PATH, 'r') as f:
#         essay_ids = [int(line.strip()) for line in f]
# except Exception as e:
#     logger.error(f"Failed to load FAISS index or IDs: {str(e)}")
#     raise Exception(f"Failed to load FAISS index or IDs: {str(e)}")

# # Load models
# try:
#     logger.info("Loading IndicBERT model and tokenizer")
#     tokenizer_bert = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', use_fast=False)
#     model_bert = AutoModel.from_pretrained('ai4bharat/indic-bert')
#     logger.info("Loading NLLB model and tokenizer")
#     tokenizer_trans = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
#     model_trans = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
#     logger.info("Loading distilgpt2 for text generation")
#     generator = pipeline('text-generation', model='distilgpt2', tokenizer='distilgpt2', max_new_tokens=50)
# except Exception as e:
#     logger.error(f"Failed to load models: {str(e)}")
#     raise Exception(f"Failed to load models: {str(e)}")

# class QueryRequest(BaseModel):
#     query: str
#     lang: str

# def translate_text(text, src_lang, tgt_lang):
#     try:
#         logger.info(f"Translating text from {src_lang} to {tgt_lang}")
#         inputs = tokenizer_trans(text, return_tensors="pt", padding=True, truncation=True)
#         translated = model_trans.generate(
#             **inputs,
#             forced_bos_token_id=tokenizer_trans.lang_code_to_id[tgt_lang],
#             max_length=512
#         )
#         return tokenizer_trans.decode(translated[0], skip_special_tokens=True)
#     except Exception as e:
#         logger.error(f"Translation error: {str(e)}")
#         raise

# def get_embedding(text, max_length=512):
#     try:
#         logger.info("Generating embedding for text")
#         inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
#         with torch.no_grad():
#             outputs = model_bert(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#         return embeddings
#     except Exception as e:
#         logger.error(f"Embedding error: {str(e)}")
#         raise

# def retrieve_essays(query, k=3):
#     try:
#         logger.info(f"Retrieving essays for query: {query}")
#         query_embedding = get_embedding(query)
#         D, I = index.search(np.array([query_embedding], dtype='float32'), k)
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         results = []
#         for idx in I[0]:
#             cursor.execute('SELECT title, content FROM essays WHERE id = ?', (essay_ids[idx],))
#             result = cursor.fetchone()
#             if result:
#                 title, content = result
#                 results.append(f"Title: {title}\nContent: {content[:500]}...")
#         conn.close()
#         return results
#     except Exception as e:
#         logger.error(f"Retrieval error: {str(e)}")
#         raise

# def generate_answer(context, question):
#     try:
#         logger.info("Generating answer")
#         prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {question}\nProvide a concise answer in 2-3 sentences."
#         response = generator(prompt, num_return_sequences=1)
#         answer = response[0]['generated_text'].split(prompt)[-1].strip()
#         return answer
#     except Exception as e:
#         logger.error(f"Generation error: {str(e)}")
#         raise

# @app.post("/query")
# async def process_query(request: QueryRequest):
#     query, lang = request.query, request.lang
#     if lang not in ["en", "hin", "tam", "tel"]:
#         raise HTTPException(status_code=400, detail="Unsupported language")
#     try:
#         logger.info(f"Processing query: {query} in language {lang}")
#         query_en = translate_text(query, f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu", "eng_Latn") if lang != "en" else query
#         retrieved_essays = retrieve_essays(query_en)
#         context = "\n\n".join(retrieved_essays)
#         answer = generate_answer(context, query_en)
#         if lang != "en":
#             answer = translate_text(answer, "eng_Latn", f"{lang}_Deva" if lang == "hin" else f"{lang}_Taml" if lang == "tam" else f"{lang}_Telu")
#         logger.info("Query processed successfully")
#         return {"answer": answer, "retrieved_essays": retrieved_essays}
#     except Exception as e:
#         logger.error(f"Query processing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")