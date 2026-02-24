from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, nullable=False, unique=True)
    full_name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    onboarding_completed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    startups = relationship("Startup", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    meetings = relationship("Meeting", back_populates="user")
    meeting_templates = relationship("MeetingTemplate", back_populates="user")
    notes = relationship("Note", back_populates="user")
    calendar_events = relationship("CalendarEvent", back_populates="user")
    calendar_integrations = relationship("CalendarIntegration", back_populates="user")
    scheduling_preference = relationship("SchedulingPreference", back_populates="user", uselist=False)
    trips = relationship("Trip", back_populates="user")
    contacts = relationship("Contact", back_populates="user")
    journey_stages = relationship("JourneyStage", back_populates="user")
    journey_milestones = relationship("JourneyMilestone", back_populates="user")
