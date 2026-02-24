from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    lang: str = "en"


@router.post("/query")
async def legacy_query(request: QueryRequest):
    """Legacy endpoint preserved for backward compatibility. Will be removed after Phase 3."""
    return {
        "answer": "This legacy endpoint is deprecated. Use the new chat system at /api/v1/conversations.",
        "retrieved_essays": [],
    }
