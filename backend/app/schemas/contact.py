from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ContactCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    contact_type: str = "other"
    company: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    relationship_strength: str = "cold"
    tags: list = []
    notes: Optional[str] = None


class ContactUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    contact_type: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    relationship_strength: Optional[str] = None
    tags: Optional[list] = None
    notes: Optional[str] = None
    next_follow_up_at: Optional[datetime] = None


class ContactOut(BaseModel):
    id: str
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    contact_type: str
    company: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    relationship_strength: str
    tags: list = []
    notes: Optional[str] = None
    last_contacted_at: Optional[datetime] = None
    next_follow_up_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ContactInteractionCreate(BaseModel):
    interaction_type: str = "other"
    summary: Optional[str] = None
    linked_meeting_id: Optional[str] = None
    linked_note_id: Optional[str] = None
    occurred_at: Optional[datetime] = None


class ContactInteractionOut(BaseModel):
    id: str
    contact_id: str
    user_id: str
    interaction_type: str
    summary: Optional[str] = None
    linked_meeting_id: Optional[str] = None
    linked_note_id: Optional[str] = None
    occurred_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}
