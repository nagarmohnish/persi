from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import logging
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
DB_PATH = os.path.join(DATA_DIR, "essays.db")

# Verify file existence
if not os.path.exists(DB_PATH):
    logger.error(f"File not found: {DB_PATH}")
    raise FileNotFoundError(f"File not found: {DB_PATH}")

class QueryRequest(BaseModel):
    query: str
    lang: str

def search_essays(query, k=3):
    try:
        logger.info(f"Searching essays for query: {query}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title, content FROM essays WHERE title LIKE ? OR content LIKE ? LIMIT ?",
            (f"%{query}%", f"%{query}%", k)
        )
        results = []
        for title, content in cursor.fetchall():
            results.append(f"Title: {title}\nContent: {content[:500]}...")
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise

@app.post("/query")
async def process_query(request: QueryRequest):
    query, lang = request.query, request.lang
    try:
        logger.info(f"Processing query: {query} in language {lang}")
        retrieved_essays = search_essays(query)
        answer = "\n\n".join(retrieved_essays) if retrieved_essays else "No relevant essays found."
        logger.info("Query processed successfully")
        return {"answer": answer, "retrieved_essays": retrieved_essays}
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.get("/query-test")
async def query_test_get(query: str, lang: str = "en"):
    request = QueryRequest(query=query, lang=lang)
    return await process_query(request)