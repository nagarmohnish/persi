from sqlalchemy import Column, String, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from datetime import datetime, timezone
from app.models.user import Base


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)  # pg_essay, yc_library, framework, playbook
    source_title = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, default=0)
    stage_relevance = Column(ARRAY(String), default=list)
    topics = Column(ARRAY(String), default=list)
    # embedding column deferred to Phase 3 (pgvector)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
