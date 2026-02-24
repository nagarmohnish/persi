from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class NoteCreate(BaseModel):
    title: str
    content: Optional[str] = None
    content_html: Optional[str] = None
    note_type: str = "idea"
    linked_meeting_id: Optional[str] = None
    linked_conversation_id: Optional[str] = None
    tags: list = []
    startup_stage: Optional[str] = None


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    content_html: Optional[str] = None
    note_type: Optional[str] = None
    tags: Optional[list] = None
    startup_stage: Optional[str] = None


class NoteOut(BaseModel):
    id: str
    user_id: str
    title: str
    content: Optional[str] = None
    content_html: Optional[str] = None
    note_type: str
    linked_meeting_id: Optional[str] = None
    linked_conversation_id: Optional[str] = None
    tags: list = []
    is_pinned: bool
    startup_stage: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class NoteVersionOut(BaseModel):
    id: str
    note_id: str
    version_number: int
    content: Optional[str] = None
    change_summary: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
