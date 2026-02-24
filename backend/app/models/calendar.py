from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.user import Base


class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    event_type = Column(String, default="meeting")  # meeting, deadline, milestone, reminder, focus_time, travel, personal
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    all_day = Column(Boolean, default=False)
    location = Column(Text, nullable=True)
    meeting_link = Column(Text, nullable=True)
    linked_meeting_id = Column(String, ForeignKey("meetings.id"), nullable=True)
    recurrence_rule = Column(Text, nullable=True)
    reminders = Column(JSONB, default=list)  # [{minutes_before, method}]
    color = Column(String, nullable=True)
    external_calendar_id = Column(String, nullable=True)
    external_source = Column(String, nullable=True)  # google, outlook, apple, manual
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="calendar_events")


class CalendarIntegration(Base):
    __tablename__ = "calendar_integrations"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String, nullable=False)  # google, outlook, apple
    access_token_encrypted = Column(Text, nullable=True)
    refresh_token_encrypted = Column(Text, nullable=True)
    calendar_ids = Column(JSONB, default=list)
    sync_direction = Column(String, default="import")  # import, export, bidirectional
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="calendar_integrations")


class SchedulingPreference(Base):
    __tablename__ = "scheduling_preferences"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    working_hours = Column(JSONB, default=dict)  # {mon: {start: "09:00", end: "18:00"}, ...}
    timezone = Column(String, default="UTC")
    focus_blocks = Column(JSONB, default=list)  # [{day, start, end, label}]
    meeting_buffer_minutes = Column(Integer, default=15)
    max_meetings_per_day = Column(Integer, nullable=True)
    preferred_meeting_durations = Column(JSONB, default=dict)  # {investor: 60, standup: 15, ...}
    scheduling_link = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="scheduling_preference")
