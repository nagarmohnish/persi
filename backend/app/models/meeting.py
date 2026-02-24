from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, Integer, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.user import Base


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    meeting_type = Column(String, default="other")  # one_on_one, team, investor, customer, advisory, demo, standup, other
    status = Column(String, default="scheduled")  # scheduled, in_progress, completed, cancelled
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    location = Column(Text, nullable=True)
    meeting_link = Column(Text, nullable=True)
    platform = Column(String, nullable=True)  # zoom, google_meet, teams, in_person, other
    is_recurring = Column(Boolean, default=False)
    recurrence_rule = Column(Text, nullable=True)
    startup_stage = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="meetings")
    participants = relationship("MeetingParticipant", back_populates="meeting", cascade="all, delete-orphan")
    recordings = relationship("MeetingRecording", back_populates="meeting", cascade="all, delete-orphan")
    summaries = relationship("MeetingSummary", back_populates="meeting", cascade="all, delete-orphan")


class MeetingParticipant(Base):
    __tablename__ = "meeting_participants"

    id = Column(String, primary_key=True)
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=True)
    role = Column(String, default="attendee")  # organizer, attendee, optional, investor, advisor, customer
    rsvp_status = Column(String, default="pending")  # pending, accepted, declined, tentative
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    meeting = relationship("Meeting", back_populates="participants")


class MeetingRecording(Base):
    __tablename__ = "meeting_recordings"

    id = Column(String, primary_key=True)
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False, index=True)
    recording_url = Column(Text, nullable=True)
    recording_platform = Column(String, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    transcription_status = Column(String, default="pending")  # pending, processing, completed, failed
    transcription_text = Column(Text, nullable=True)
    transcription_segments = Column(JSONB, nullable=True)  # [{start, end, speaker, text}]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    meeting = relationship("Meeting", back_populates="recordings")
    summaries = relationship("MeetingSummary", back_populates="recording")


class MeetingSummary(Base):
    __tablename__ = "meeting_summaries"

    id = Column(String, primary_key=True)
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False, index=True)
    recording_id = Column(String, ForeignKey("meeting_recordings.id"), nullable=True)
    summary = Column(Text, nullable=True)
    key_decisions = Column(JSONB, default=list)  # [{decision, context, owner}]
    action_items = Column(JSONB, default=list)  # [{task, assignee, deadline, priority}]
    follow_ups = Column(JSONB, default=list)  # [{item, responsible, due_date}]
    sentiment_analysis = Column(JSONB, nullable=True)  # {overall, by_participant}
    topics_discussed = Column(JSONB, default=list)  # [{topic, duration, key_points}]
    next_steps = Column(Text, nullable=True)
    generated_by = Column(String, default="ai")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    meeting = relationship("Meeting", back_populates="summaries")
    recording = relationship("MeetingRecording", back_populates="summaries")


class MeetingTemplate(Base):
    __tablename__ = "meeting_templates"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    meeting_type = Column(String, nullable=True)
    default_agenda = Column(JSONB, default=list)  # [{item, duration_minutes, description}]
    default_duration_minutes = Column(Integer, default=30)
    pre_meeting_prompts = Column(JSONB, default=list)
    post_meeting_prompts = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="meeting_templates")
