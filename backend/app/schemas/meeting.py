from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class MeetingCreate(BaseModel):
    title: str
    description: Optional[str] = None
    meeting_type: str = "other"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    platform: Optional[str] = None
    is_recurring: bool = False
    recurrence_rule: Optional[str] = None


class MeetingUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    meeting_type: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    platform: Optional[str] = None


class MeetingOut(BaseModel):
    id: str
    user_id: str
    title: str
    description: Optional[str] = None
    meeting_type: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    platform: Optional[str] = None
    is_recurring: bool
    startup_stage: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MeetingParticipantCreate(BaseModel):
    name: str
    email: Optional[str] = None
    role: str = "attendee"


class MeetingParticipantOut(BaseModel):
    id: str
    meeting_id: str
    name: str
    email: Optional[str] = None
    role: str
    rsvp_status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MeetingSummaryOut(BaseModel):
    id: str
    meeting_id: str
    summary: Optional[str] = None
    key_decisions: list = []
    action_items: list = []
    follow_ups: list = []
    topics_discussed: list = []
    next_steps: Optional[str] = None
    generated_by: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MeetingTemplateCreate(BaseModel):
    name: str
    meeting_type: Optional[str] = None
    default_agenda: list = []
    default_duration_minutes: int = 30
    pre_meeting_prompts: list = []
    post_meeting_prompts: list = []


class MeetingTemplateOut(BaseModel):
    id: str
    user_id: str
    name: str
    meeting_type: Optional[str] = None
    default_agenda: list = []
    default_duration_minutes: int
    pre_meeting_prompts: list = []
    post_meeting_prompts: list = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
