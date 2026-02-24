from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.user import Base


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    contact_type = Column(String, default="other")  # investor, advisor, customer, partner, team_member, mentor, service_provider, other
    company = Column(String, nullable=True)
    title = Column(String, nullable=True)
    linkedin_url = Column(String, nullable=True)
    twitter_url = Column(String, nullable=True)
    relationship_strength = Column(String, default="cold")  # cold, warm, hot, close
    tags = Column(JSONB, default=list)
    notes = Column(Text, nullable=True)
    last_contacted_at = Column(DateTime(timezone=True), nullable=True)
    next_follow_up_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="contacts")
    interactions = relationship("ContactInteraction", back_populates="contact", cascade="all, delete-orphan", order_by="ContactInteraction.occurred_at.desc()")


class ContactInteraction(Base):
    __tablename__ = "contact_interactions"

    id = Column(String, primary_key=True)
    contact_id = Column(String, ForeignKey("contacts.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    interaction_type = Column(String, default="other")  # email, meeting, call, message, introduction, other
    summary = Column(Text, nullable=True)
    linked_meeting_id = Column(String, ForeignKey("meetings.id"), nullable=True)
    linked_note_id = Column(String, ForeignKey("notes.id"), nullable=True)
    occurred_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    contact = relationship("Contact", back_populates="interactions")
    user = relationship("User")
