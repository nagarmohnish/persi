from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CalendarEventCreate(BaseModel):
    title: str
    description: Optional[str] = None
    event_type: str = "meeting"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    all_day: bool = False
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    linked_meeting_id: Optional[str] = None
    recurrence_rule: Optional[str] = None
    reminders: list = []
    color: Optional[str] = None


class CalendarEventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    event_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    all_day: Optional[bool] = None
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    reminders: Optional[list] = None
    color: Optional[str] = None


class CalendarEventOut(BaseModel):
    id: str
    user_id: str
    title: str
    description: Optional[str] = None
    event_type: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    all_day: bool
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    linked_meeting_id: Optional[str] = None
    recurrence_rule: Optional[str] = None
    reminders: list = []
    color: Optional[str] = None
    external_source: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SchedulingPreferenceOut(BaseModel):
    id: str
    user_id: str
    working_hours: dict = {}
    timezone: str
    focus_blocks: list = []
    meeting_buffer_minutes: int
    max_meetings_per_day: Optional[int] = None
    preferred_meeting_durations: dict = {}
    scheduling_link: Optional[str] = None

    model_config = {"from_attributes": True}


class SchedulingPreferenceUpdate(BaseModel):
    working_hours: Optional[dict] = None
    timezone: Optional[str] = None
    focus_blocks: Optional[list] = None
    meeting_buffer_minutes: Optional[int] = None
    max_meetings_per_day: Optional[int] = None
    preferred_meeting_durations: Optional[dict] = None
    scheduling_link: Optional[str] = None
